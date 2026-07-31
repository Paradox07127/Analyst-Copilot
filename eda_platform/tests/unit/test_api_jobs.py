"""Job API + SSE end to end: real spawned worker processes on a tiny CSV
(offline LLM, no report) so each job completes within seconds."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from eda_platform.api.main import create_app
from eda_platform.application.ports import JobCommand
from eda_platform.application.services import job_service as job_service_module
from eda_platform.core.session_deletion import SessionDeletionCoordinator
from eda_platform.core.store import ArtifactStore
from eda_platform.infrastructure.job_backend import LocalProcessJobBackend
from eda_platform.infrastructure.job_lifecycle import JobLifecycleRepository
from eda_platform.schemas.sessions import TraceEvent

_JOB_TIMEOUT_SECONDS = 120.0


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    store = ArtifactStore(tmp_path)
    store.ensure_project("demo", name="Demo")
    seed = tmp_path / "seed" / "orders.csv"
    seed.parent.mkdir(parents=True, exist_ok=True)
    seed.write_text(
        "region,amount\n" + "\n".join(f"r{i % 3},{100 + i}" for i in range(40)) + "\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def client(workspace: Path) -> TestClient:
    return TestClient(create_app(workspace))


def _create_job(client: TestClient, session_id: str, **overrides: object) -> dict:
    body: dict[str, object] = {
        "kind": "auto_eda",
        "project_id": "demo",
        "datasets": ["seed/orders.csv"],
        "llm": "offline",
        "generate_report": False,
    }
    body.update(overrides)
    response = client.post(f"/api/v1/sessions/{session_id}/jobs", json=body)
    assert response.status_code == 201, response.text
    return response.json()


def _queue_durable_job(
    store: ArtifactStore, *, job_id: str, session_id: str, params_json: str
) -> dict:
    return JobLifecycleRepository(store).create_queued_job(
        job_id=job_id,
        session_id=session_id,
        project_id="demo",
        kind="auto_eda",
        params_json=params_json,
        idempotency_key=None,
        lane_key=session_id,
        request_digest=f"digest-{job_id}",
        request_scope=session_id,
    )


def _wait_terminal(client: TestClient, job_id: str) -> dict:
    deadline = time.monotonic() + _JOB_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        body = client.get(f"/api/v1/jobs/{job_id}").json()
        if body["status"] in {"completed", "failed", "cancelled"}:
            return body
        time.sleep(0.2)
    raise AssertionError(f"Job {job_id} did not reach a terminal status in time.")


def _read_frames(response: Any) -> list[dict]:
    frames: list[dict] = []
    current: dict = {}
    for line in response.iter_lines():
        if line == "":
            if current:
                frames.append(current)
                current = {}
        elif line.startswith(":"):
            continue
        elif line.startswith("id: "):
            current["id"] = int(line[len("id: ") :])
        elif line.startswith("event: "):
            current["event"] = line[len("event: ") :]
        elif line.startswith("data: "):
            current["data"] = json.loads(line[len("data: ") :])
    if current:
        frames.append(current)
    return frames


def test_job_runs_to_completion_and_streams_events(
    client: TestClient, workspace: Path
) -> None:
    created = _create_job(client, "run_job_demo")
    assert created["session_id"] == "run_job_demo"
    assert created["status"] == "queued"
    assert created["events_url"] == f"/api/v1/jobs/{created['job_id']}/events"

    final = _wait_terminal(client, created["job_id"])
    assert final["status"] == "completed", final
    assert final["started_at"] and final["finished_at"]

    store = ArtifactStore(workspace)
    assert store.get_session_status("run_job_demo") == "completed"
    artifacts, _warnings = store.list_artifacts_safe(project_id="demo", session_id="run_job_demo")
    assert artifacts

    with client.stream("GET", created["events_url"]) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        frames = _read_frames(response)
    types = [frame["event"] for frame in frames]
    assert types[0] == "job.queued"
    assert "job.started" in types
    assert types[-1] == "job.completed"
    ids = [frame["id"] for frame in frames]
    assert ids == sorted(ids)
    assert frames[0]["data"]["job_id"] == created["job_id"]
    with sqlite3.connect(ArtifactStore(workspace).db_path) as conn:
        correlations = conn.execute(
            """
            select event_type, job_id, job_generation
            from trace_events where project_id = ? and session_id = ?
            order by id
            """,
            ("demo", "run_job_demo"),
        ).fetchall()
    assert correlations
    assert all(row[1] == created["job_id"] for row in correlations)
    assert all(row[2] is not None for row in correlations)
    assert {row[2] for row in correlations} == {0, 1}

    # Last-Event-ID resumes strictly after the given id and still terminates.
    middle_id = ids[len(ids) // 2]
    with client.stream(
        "GET", created["events_url"], headers={"Last-Event-ID": str(middle_id)}
    ) as response:
        replay = _read_frames(response)
    assert replay
    assert all(frame["id"] > middle_id for frame in replay if "id" in frame)
    assert replay[-1]["event"] == "job.completed"


def test_cancel_marks_job_cancelled(client: TestClient) -> None:
    created = _create_job(client, "run_job_cancel")
    cancel = client.post(f"/api/v1/jobs/{created['job_id']}/cancel")
    assert cancel.status_code == 200
    assert cancel.json()["cancel_requested"] is True

    final = _wait_terminal(client, created["job_id"])
    assert final["status"] == "cancelled", final

    with client.stream("GET", created["events_url"]) as response:
        frames = _read_frames(response)
    assert frames[-1]["event"] == "job.cancelled"


def test_idempotency_key_returns_same_job(client: TestClient) -> None:
    headers = {"Idempotency-Key": "same-key"}
    body = {
        "kind": "auto_eda",
        "project_id": "demo",
        "datasets": ["seed/orders.csv"],
        "llm": "offline",
        "generate_report": False,
    }
    first = client.post("/api/v1/sessions/run_idem/jobs", json=body, headers=headers)
    second = client.post("/api/v1/sessions/run_idem/jobs", json=body, headers=headers)
    assert first.status_code == 201 and second.status_code == 201
    assert first.json()["job_id"] == second.json()["job_id"]
    _wait_terminal(client, first.json()["job_id"])


def test_idempotency_key_different_body_returns_typed_422(client: TestClient) -> None:
    headers = {"Idempotency-Key": "content-bound-key"}
    body = {
        "kind": "auto_eda",
        "project_id": "demo",
        "datasets": ["seed/orders.csv"],
        "business_context": "first request",
        "llm": "offline",
        "generate_report": False,
    }
    first = client.post(
        "/api/v1/sessions/run_content_bound/jobs",
        json=body,
        headers=headers,
    )
    assert first.status_code == 201, first.text

    changed = client.post(
        "/api/v1/sessions/run_content_bound/jobs",
        json={**body, "business_context": "different request"},
        headers=headers,
    )

    assert changed.status_code == 422, changed.text
    assert changed.json()["error"]["code"] == "idempotency_key_reused"


def test_job_create_rejects_run_reserved_for_deletion(
    client: TestClient, workspace: Path
) -> None:
    store = ArtifactStore(workspace)
    store.start_session("demo", "session_deleting")

    def crash_after_reserve(stage: str, _op_id: str, _ordinal: int | None) -> None:
        if stage == "after_reserve":
            raise RuntimeError("injected crash")

    with pytest.raises(RuntimeError, match="injected crash"):
        SessionDeletionCoordinator(store, fault_hook=crash_after_reserve).delete(
            "session_deleting"
        )

    response = client.post(
        "/api/v1/sessions/session_deleting/jobs",
        json={
            "kind": "auto_eda",
            "project_id": "demo",
            "datasets": ["seed/orders.csv"],
            "llm": "offline",
            "generate_report": False,
        },
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "session_deleting"
    assert ArtifactStore(workspace).list_active_jobs() == []


def test_backend_cancel_before_start_is_deterministic(workspace: Path) -> None:
    """A queued cancellation terminalizes without spawning any child."""
    store = ArtifactStore(workspace)
    _queue_durable_job(
        store,
        job_id="job_pre",
        session_id="run_pre",
        params_json=json.dumps(
            {"dataset_paths": ["seed/orders.csv"], "llm": "offline", "generate_report": False}
        ),
    )
    backend = LocalProcessJobBackend(workspace, store)
    backend.cancel("job_pre")
    assert backend.join("job_pre", timeout=0) is None
    job = store.get_job("job_pre")
    assert job is not None
    assert job["status"] == "cancelled"
    # No pipeline step ran, so no artifacts were written for the run.
    artifacts, _warnings = store.list_artifacts_safe(project_id="demo", session_id="run_pre")
    assert artifacts == []


def test_job_error_envelope_paths(client: TestClient) -> None:
    missing = client.get("/api/v1/jobs/job_missing")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "job_not_found"

    bad_ref = client.post(
        "/api/v1/sessions/run_x/jobs",
        json={"kind": "auto_eda", "project_id": "demo", "datasets": ["nope.csv"]},
    )
    assert bad_ref.status_code == 422
    assert bad_ref.json()["error"]["code"] == "job_invalid"

    bad_kind = client.post(
        "/api/v1/sessions/run_x/jobs",
        json={"kind": "other", "project_id": "demo", "datasets": ["seed/orders.csv"]},
    )
    assert bad_kind.status_code == 422

    missing_project = client.post(
        "/api/v1/sessions/run_x/jobs",
        json={"kind": "auto_eda", "project_id": "nope", "datasets": ["seed/orders.csv"]},
    )
    assert missing_project.status_code == 404
    assert missing_project.json()["error"]["code"] == "project_not_found"


def _emit(
    store: ArtifactStore,
    session_id: str,
    event_type: str,
    name: str,
    *,
    job_id: str,
) -> None:
    store.append_trace(
        "demo",
        TraceEvent(
            session_id=session_id,
            event_type=event_type,
            name=name,
            job_id=job_id,
            finished_at=datetime.now(UTC),
        ),
    )


# Review F3: a run with an active job rejects a second one with 409.
def test_second_job_on_active_run_returns_409(client: TestClient, workspace: Path) -> None:
    store = ArtifactStore(workspace)
    store.create_job(
        job_id="job_busy", session_id="run_conflict", project_id="demo", kind="auto_eda"
    )
    response = client.post(
        "/api/v1/sessions/run_conflict/jobs",
        json={
            "kind": "auto_eda",
            "project_id": "demo",
            "datasets": ["seed/orders.csv"],
            "llm": "offline",
            "generate_report": False,
        },
    )
    assert response.status_code == 409, response.text
    body = response.json()
    assert body["error"]["code"] == "job_conflict"
    assert "job_busy" in body["error"]["message"]


# Review F2: a full page must keep draining instead of closing synthetically.
def test_sse_replays_all_events_beyond_page_limit(
    client: TestClient, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(job_service_module, "EVENTS_PAGE_LIMIT", 5)
    store = ArtifactStore(workspace)
    store.create_job(job_id="job_big", session_id="run_big", project_id="demo", kind="auto_eda")
    _emit(store, "run_big", "job.queued", "job_big", job_id="job_big")
    step_count = 21  # >4x the patched page limit
    for i in range(step_count):
        _emit(store, "run_big", "profile", f"step_{i}", job_id="job_big")
    _emit(store, "run_big", "job.completed", "job_big", job_id="job_big")
    store.mark_job_status("job_big", "completed")

    with client.stream("GET", "/api/v1/jobs/job_big/events") as response:
        frames = _read_frames(response)
    assert len(frames) == step_count + 2  # every event arrived, none truncated away
    assert frames[0]["event"] == "job.queued"
    assert frames[-1]["event"] == "job.completed"
    assert "synthetic" not in frames[-1]["data"]["summary"]
    ids = [frame["id"] for frame in frames]
    assert ids == sorted(ids)


def test_old_job_reconnect_does_not_stream_later_job_progress(
    client: TestClient,
    workspace: Path,
) -> None:
    store = ArtifactStore(workspace)
    store.create_job(
        job_id="job_old",
        session_id="run_reconnect_scope",
        project_id="demo",
        kind="auto_eda",
        lane_key="lane_old",
    )
    _emit(
        store,
        "run_reconnect_scope",
        "job.queued",
        "job_old",
        job_id="job_old",
    )
    _emit(
        store,
        "run_reconnect_scope",
        "profile",
        "old_profile",
        job_id="job_old",
    )
    _emit(
        store,
        "run_reconnect_scope",
        "job.completed",
        "job_old",
        job_id="job_old",
    )
    store.mark_job_status("job_old", "completed")
    with sqlite3.connect(store.db_path) as conn:
        old_terminal_id = conn.execute(
            """
            select id from trace_events
            where job_id = 'job_old' and event_type = 'job.completed'
            """
        ).fetchone()[0]

    store.create_job(
        job_id="job_later",
        session_id="run_reconnect_scope",
        project_id="demo",
        kind="auto_eda",
        lane_key="lane_later",
    )
    _emit(
        store,
        "run_reconnect_scope",
        "insight",
        "later_job_insight",
        job_id="job_later",
    )

    response = client.get(
        "/api/v1/jobs/job_old/events",
        headers={"Last-Event-ID": str(old_terminal_id)},
    )
    assert response.status_code == 204
    assert "later_job_insight" not in response.text


# Review F4: orphaned non-terminal jobs are failed when the API boots.
def test_orphan_jobs_reaped_at_startup(workspace: Path) -> None:
    store = ArtifactStore(workspace)
    dead = subprocess.Popen([sys.executable, "-c", ""])
    dead.wait()
    store.create_job(job_id="job_orphan", session_id="run_o", project_id="demo", kind="auto_eda")
    store.mark_job_status("job_orphan", "running")
    store.set_job_pid("job_orphan", dead.pid)
    store.create_job(job_id="job_alive", session_id="run_l", project_id="demo", kind="auto_eda")
    store.mark_job_status("job_alive", "running")
    store.set_job_pid("job_alive", os.getpid())

    client = TestClient(create_app(workspace))
    orphan = client.get("/api/v1/jobs/job_orphan").json()
    assert orphan["status"] == "failed"
    assert orphan["error_code"] == "orphaned"
    alive = client.get("/api/v1/jobs/job_alive").json()
    assert alive["status"] == "running"

    # SSE for the reaped job closes on the real job.failed trace event.
    with client.stream("GET", "/api/v1/jobs/job_orphan/events") as response:
        frames = _read_frames(response)
    assert frames[-1]["event"] == "job.failed"


# Review F5: the worker is a detached session; the API process never joins it.
def test_worker_runs_in_its_own_session(workspace: Path) -> None:
    store = ArtifactStore(workspace)
    params_json = json.dumps(
        {"dataset_paths": ["seed/orders.csv"], "llm": "offline", "generate_report": False}
    )
    _queue_durable_job(
        store,
        job_id="job_sess",
        session_id="run_sess",
        params_json=params_json,
    )
    backend = LocalProcessJobBackend(workspace, store)
    ref = backend.enqueue(
        JobCommand(
            job_id="job_sess",
            session_id="run_sess",
            project_id="demo",
            kind="auto_eda",
            params_json=params_json,
        )
    )
    assert ref.pid is not None
    assert os.getsid(ref.pid) != os.getsid(0)
    job = store.get_job("job_sess")
    assert job is not None and job["pid"] == ref.pid
    backend.cancel("job_sess")
    assert backend.join("job_sess", timeout=60) == 0
