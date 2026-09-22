"""Job refresh projects the exact durable exploration attempt outcome."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from eda_platform.api.main import create_app
from eda_platform.core.store import ArtifactStore
from eda_platform.infrastructure.job_lifecycle import JobLifecycleRepository
from eda_platform.schemas.sessions import TraceEvent


def _completed_job(root: Path, *, kind: str = "exploration_run") -> tuple[ArtifactStore, int]:
    store = ArtifactStore(root)
    store.ensure_project("p", "Project")
    lifecycle = JobLifecycleRepository(store)
    lifecycle.create_queued_job(
        job_id="job_test", session_id="run", project_id="p", kind=kind,
        params_json="{}", idempotency_key=None, lane_key="run",
        request_digest="test", request_scope="run",
    )
    claim = lifecycle.claim_launch("job_test", owner="test")
    lifecycle.acknowledge_spawn(claim, pid=os.getpid(), birth_identity="unit-test")
    assert lifecycle.child_start(claim) is not None
    assert lifecycle.finish(claim, "completed")
    return store, claim.attempt


def _event(
    store: ArtifactStore, generation: int, summary: dict[str, Any], *, job_id: str = "job_test"
) -> None:
    store.append_trace("p", TraceEvent(
        session_id="run", job_id=job_id, job_generation=generation,
        event_type="exploration.attempt_finished", name=job_id, summary=summary,
    ))


@pytest.mark.parametrize(("status", "reason"), [
    ("paused", None), ("stopped", "completed"), ("stopped", "budget_exhausted"),
    ("stopped", "failed"),
])
def test_job_refresh_projects_latest_outcome_for_current_job_and_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str, reason: str | None,
) -> None:
    store, generation = _completed_job(tmp_path)
    older = {
        "exploration_id": "explore_1", "exploration_status": "stopped", "stop_reason": "cancelled"
    }
    _event(store, generation, older)
    _event(store, generation, {
        "exploration_id": "explore_1", "exploration_status": status, "stop_reason": reason
    })
    # Later events from another job or stale generation must not repaint this attempt.
    _event(store, generation, older, job_id="job_other")
    _event(store, generation - 1, older)

    def unbounded_read(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("Job refresh must not scan session traces")

    monkeypatch.setattr(ArtifactStore, "list_trace_events", unbounded_read)
    # A fresh application has no in-memory SSE events or workflow objects.
    response = TestClient(create_app(tmp_path)).get("/api/v1/jobs/job_test")
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "completed"
    assert response.json()["domain_outcome"] == {
        "status": status, "stop_reason": reason, "exploration_id": "explore_1"
    }


@pytest.mark.parametrize("summary", [
    {},
    {"exploration_id": "exp", "exploration_status": "unknown"},
    {"exploration_id": 17, "exploration_status": "paused"},
    {"exploration_id": "exp", "exploration_status": "stopped", "stop_reason": "unknown"},
    {"exploration_id": "exp", "exploration_status": "stopped"},
    {"exploration_id": "exp", "exploration_status": "paused", "stop_reason": "completed"},
])
def test_malformed_latest_domain_outcome_is_unavailable(
    tmp_path: Path, summary: dict[str, Any],
) -> None:
    store, generation = _completed_job(tmp_path)
    _event(store, generation, {
        "exploration_id": "exp", "exploration_status": "paused", "stop_reason": None
    })
    _event(store, generation, summary)
    response = TestClient(create_app(tmp_path)).get("/api/v1/jobs/job_test")
    assert response.status_code == 200, response.text
    assert response.json()["domain_outcome"] is None


@pytest.mark.parametrize("kind", ["auto_eda", "exploration_run"])
def test_job_without_domain_event_keeps_outcome_unavailable(tmp_path: Path, kind: str) -> None:
    _completed_job(tmp_path, kind=kind)
    response = TestClient(create_app(tmp_path)).get("/api/v1/jobs/job_test")
    assert response.status_code == 200
    assert response.json()["domain_outcome"] is None
