"""Cleaning API vertical slice: preview registers a server-side approval,
apply consumes it once and forks a real offline auto_eda job on the cleaned
version. Uses spawned worker processes like test_api_jobs (offline, tiny CSV,
no report) so the whole chain stays fast and free."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest
from data_operation_helpers import await_data_operation, operation_result_response
from fastapi.testclient import TestClient
from httpx2 import Response

from eda_platform.api.main import create_app
from eda_platform.core.store import ArtifactStore

_JOB_TIMEOUT_SECONDS = 120.0

# Trailing whitespace + one exact duplicate row so trim and dedupe both bite.
_SEED_CSV = (
    "region,amount\n"
    "north ,100\n"
    "north ,100\n"
    " south,200\n"
    "east,300\n"
    "west,400\n"
    "north,500\n"
)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    store = ArtifactStore(tmp_path)
    store.ensure_project("demo", name="Demo")
    seed = tmp_path / "seed" / "orders.csv"
    seed.parent.mkdir(parents=True, exist_ok=True)
    seed.write_text(_SEED_CSV, encoding="utf-8")
    return tmp_path


@pytest.fixture
def client(workspace: Path) -> TestClient:
    class DataOperationClient(TestClient):
        def post(self, url: str, *args: object, **kwargs: object) -> Response:
            started = super().post(url, *args, **kwargs)
            if started.status_code != 202 or "/cleaning/" not in url:
                return started
            result_path = (
                "cleaning-preview-result"
                if url.endswith("/preview")
                else "cleaning-apply-result"
            )
            response = operation_result_response(
                *await_data_operation(self, started, result_path)
            )
            if response.status_code == 200 and url.endswith("/apply"):
                return Response(status_code=201, json=response.json())
            return response

    return DataOperationClient(create_app(workspace))


def _wait_terminal(client: TestClient, job_id: str) -> dict:
    deadline = time.monotonic() + _JOB_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        body = client.get(f"/api/v1/jobs/{job_id}").json()
        if body["status"] in {"completed", "failed", "cancelled"}:
            return body
        time.sleep(0.2)
    raise AssertionError(f"Job {job_id} did not reach a terminal status in time.")


@pytest.fixture
def source_run(client: TestClient) -> str:
    """A completed offline run whose dataset the cleaning slice operates on."""
    session_id = "run_clean_src"
    response = client.post(
        f"/api/v1/sessions/{session_id}/jobs",
        json={
            "kind": "auto_eda",
            "project_id": "demo",
            "datasets": ["seed/orders.csv"],
            "llm": "offline",
            "generate_report": False,
        },
    )
    assert response.status_code == 201, response.text
    final = _wait_terminal(client, response.json()["job_id"])
    assert final["status"] == "completed", final
    return session_id


def _dataset_id(client: TestClient, session_id: str) -> str:
    handles = client.get(f"/api/v1/sessions/{session_id}/datasets").json()
    assert handles, "source run has no datasets"
    return handles[0]["dataset_id"]


def _preview(client: TestClient, session_id: str, dataset_id: str, **options: object) -> dict:
    body: dict[str, object] = {"dataset_id": dataset_id}
    body.update(options)
    response = client.post(f"/api/v1/sessions/{session_id}/cleaning/preview", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def _apply_body(preview_result: dict, **extra: object) -> dict:
    body: dict[str, object] = {
        "action_hash": preview_result["action_hash"],
        "approval_token": preview_result["approval_token"],
        "llm": "offline",
    }
    body.update(extra)
    return body


def test_preview_returns_diff_and_pending_hash(client: TestClient, source_run: str) -> None:
    dataset_id = _dataset_id(client, source_run)
    result = _preview(client, source_run, dataset_id)

    assert result["session_id"] == source_run
    assert result["dataset_id"] == dataset_id
    assert len(result["action_hash"]) == 64
    assert len(result["approval_token"]) == 32
    ops = {op["transform_id"]: op for op in result["operations"]}
    assert "dedupe" in ops and ops["dedupe"]["lossy"] is True
    assert any(key.startswith("trim_") for key in ops)

    preview = result["preview"]
    assert preview["row_count_before"] == 6
    assert preview["row_count_after"] == 5  # exact duplicate dropped after trim
    assert preview["rows_dropped"] == 1
    assert preview["rows_edited"] >= 2  # whitespace-trimmed rows
    assert preview["target_version"] == 2
    assert any(change["column"] == "region" for change in preview["column_changes"])


def test_preview_is_idempotent_for_same_options(client: TestClient, source_run: str) -> None:
    dataset_id = _dataset_id(client, source_run)
    first = _preview(client, source_run, dataset_id)
    second = _preview(client, source_run, dataset_id)
    assert first["action_hash"] == second["action_hash"]
    # C1: the hash is stable, but every preview mints a fresh one-time token.
    assert first["approval_token"] != second["approval_token"]
    # Different options bind to a different hash.
    other = _preview(client, source_run, dataset_id, drop_missing_rows=True)
    assert other["action_hash"] != first["action_hash"]


def test_apply_full_chain_offline_then_replay_conflicts(
    client: TestClient, workspace: Path, source_run: str
) -> None:
    dataset_id = _dataset_id(client, source_run)
    result = _preview(client, source_run, dataset_id)

    applied = client.post(
        f"/api/v1/sessions/{source_run}/cleaning/apply",
        json=_apply_body(result),
    )
    assert applied.status_code == 201, applied.text
    body = applied.json()
    assert body["session_id"] == source_run
    assert body["new_session_id"] != source_run
    assert body["dataset_id"] == dataset_id
    assert body["target_version"] == 2
    assert body["job"]["session_id"] == body["new_session_id"]

    # The cleaned version exists on disk under the project's cleaned/ tree.
    cleaned = list((workspace / "projects" / "demo" / "cleaned").rglob("*.csv"))
    assert len(cleaned) == 1
    assert f"{dataset_id}/v2" in str(cleaned[0].as_posix())

    # The fork job completes offline and produces artifacts for the new run.
    final = _wait_terminal(client, body["job"]["job_id"])
    assert final["status"] == "completed", final
    store = ArtifactStore(workspace)
    artifacts, _warnings = store.list_artifacts_safe(
        project_id="demo", session_id=body["new_session_id"]
    )
    assert artifacts

    # The public start endpoint is asynchronous, so the durable job row—not a
    # Python exception class name—must carry the stable approval contract that
    # React uses for its guided recovery state.
    raw_client = TestClient(client.app)
    replay_started = raw_client.post(
        f"/api/v1/sessions/{source_run}/cleaning/apply",
        json=_apply_body(result),
    )
    assert replay_started.status_code == 202, replay_started.text
    replay_job = _wait_terminal(raw_client, replay_started.json()["job"]["job_id"])
    assert replay_job["status"] == "failed"
    assert replay_job["error_code"] == "approval_consumed"

    # The test adapter reconstructs the matching synchronous error envelope.
    replay = client.post(
        f"/api/v1/sessions/{source_run}/cleaning/apply",
        json=_apply_body(result),
    )
    assert replay.status_code == 409, replay.text
    assert replay.json()["error"]["code"] == "approval_consumed"
    assert len(list((workspace / "projects" / "demo" / "cleaned").rglob("*.csv"))) == 1


def test_apply_with_tampered_hash_is_404(client: TestClient, source_run: str) -> None:
    dataset_id = _dataset_id(client, source_run)
    result = _preview(client, source_run, dataset_id)
    response = client.post(
        f"/api/v1/sessions/{source_run}/cleaning/apply",
        json=_apply_body(result, action_hash="0" * 64),
    )
    assert response.status_code == 404, response.text
    assert response.json()["error"]["code"] == "approval_not_found"


def test_apply_with_stale_token_after_repreview_is_404(
    client: TestClient, source_run: str
) -> None:
    """C1: a re-preview rotates the one-time token, so the token from the
    older preview must 404 — a consumed/superseded approval never revives."""
    dataset_id = _dataset_id(client, source_run)
    first = _preview(client, source_run, dataset_id)
    second = _preview(client, source_run, dataset_id)
    assert first["action_hash"] == second["action_hash"]

    stale = client.post(
        f"/api/v1/sessions/{source_run}/cleaning/apply",
        json=_apply_body(first),
    )
    assert stale.status_code == 404, stale.text
    assert stale.json()["error"]["code"] == "approval_not_found"

    # The current token still applies fine.
    fresh = client.post(
        f"/api/v1/sessions/{source_run}/cleaning/apply",
        json=_apply_body(second),
    )
    assert fresh.status_code == 201, fresh.text
    _wait_terminal(client, fresh.json()["job"]["job_id"])


def test_apply_after_source_csv_changed_is_409(
    client: TestClient, workspace: Path, source_run: str
) -> None:
    """C4: the source CSV was rewritten between preview and apply — the stale
    approval must not clean the new content."""
    dataset_id = _dataset_id(client, source_run)
    result = _preview(client, source_run, dataset_id)
    source_uri = client.get(f"/api/v1/sessions/{source_run}/datasets").json()[0][
        "original_uri"
    ]
    source_path = workspace / source_uri
    source_path.write_text(_SEED_CSV + "tampered,999\n", encoding="utf-8")

    response = client.post(
        f"/api/v1/sessions/{source_run}/cleaning/apply",
        json=_apply_body(result),
    )
    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "cleaning_source_changed"
    # Nothing was written for the refused apply.
    assert not list((workspace / "projects" / "demo" / "cleaned").rglob("*.csv"))


def test_apply_on_wrong_run_is_404(client: TestClient, source_run: str) -> None:
    dataset_id = _dataset_id(client, source_run)
    result = _preview(client, source_run, dataset_id)
    response = client.post(
        "/api/v1/sessions/run_other/cleaning/apply",
        json=_apply_body(result),
    )
    assert response.status_code == 404, response.text
    assert response.json()["error"]["code"] == "session_not_found"


def test_apply_expired_hash_is_410(
    client: TestClient, workspace: Path, source_run: str
) -> None:
    dataset_id = _dataset_id(client, source_run)
    result = _preview(client, source_run, dataset_id)
    with sqlite3.connect(workspace / "state.sqlite") as conn:
        conn.execute(
            "update pending_actions set expires_at = ? where action_hash = ?",
            ("2000-01-01T00:00:00+00:00", result["action_hash"]),
        )
    response = client.post(
        f"/api/v1/sessions/{source_run}/cleaning/apply",
        json=_apply_body(result),
    )
    assert response.status_code == 410, response.text
    assert response.json()["error"]["code"] == "approval_expired"


def test_apply_job_create_failure_compensates_and_same_token_retries(
    workspace: Path,
) -> None:
    """C6: when the fork job cannot be created after the approval was consumed
    and the version written, the pending row is re-armed and the version dir
    removed, so the very same token retries successfully."""
    app = create_app(workspace)
    client = TestClient(app, raise_server_exceptions=False)
    session_id = "run_clean_src"
    created = client.post(
        f"/api/v1/sessions/{session_id}/jobs",
        json={
            "kind": "auto_eda",
            "project_id": "demo",
            "datasets": ["seed/orders.csv"],
            "llm": "offline",
            "generate_report": False,
        },
    )
    assert created.status_code == 201, created.text
    assert _wait_terminal(client, created.json()["job_id"])["status"] == "completed"

    service = app.state.cleaning_service
    result = service.preview(session_id, dataset_id=_dataset_id(client, session_id))
    real_jobs = service._jobs

    class ExplodingJobs:
        def create_job(self, *args: object, **kwargs: object) -> object:
            raise RuntimeError("job backend down")

    service._jobs = ExplodingJobs()
    try:
        with pytest.raises(RuntimeError, match="job backend down"):
            service.apply(
                session_id,
                action_hash=result.action_hash,
                approval_token=result.approval_token,
                llm="offline",
            )
    finally:
        service._jobs = real_jobs

    # Compensation: approval re-armed with the same token, version dir removed.
    store = ArtifactStore(workspace)
    row = store.get_pending_action(result.action_hash, session_id=session_id)
    assert row is not None and row["status"] == "pending"
    assert not list((workspace / "projects" / "demo" / "cleaned").rglob("*.csv"))

    retried = service.apply(
        session_id,
        action_hash=result.action_hash,
        approval_token=result.approval_token,
        llm="offline",
    )
    assert retried.target_version == 2
    assert len(list((workspace / "projects" / "demo" / "cleaned").rglob("*.csv"))) == 1
    _wait_terminal(client, retried.job.job_id)


def test_preview_reports_next_free_version(
    client: TestClient, workspace: Path, source_run: str
) -> None:
    """Slice-E F3a: with v2 already on disk the next apply writes v3, and the
    preview must say v3 — not a blind source_version + 1."""
    dataset_id = _dataset_id(client, source_run)
    handles = client.get(f"/api/v1/sessions/{source_run}/datasets").json()
    name = Path(handles[0]["original_uri"]).name
    occupied = workspace / "projects" / "demo" / "cleaned" / dataset_id / "v2" / name
    occupied.parent.mkdir(parents=True, exist_ok=True)
    occupied.write_text("region,amount\nstale,1\n", encoding="utf-8")

    result = _preview(client, source_run, dataset_id)
    assert result["preview"]["target_version"] == 3


def test_apply_idempotency_fast_path_rejects_foreign_key(
    client: TestClient, workspace: Path, source_run: str
) -> None:
    """A key owned by another canonical request fails with typed 422."""
    dataset_id = _dataset_id(client, source_run)
    result = _preview(client, source_run, dataset_id)
    store = ArtifactStore(workspace)
    store.ensure_project("other", name="Other")
    store.create_job(
        job_id="job_foreign",
        session_id="run_foreign",
        project_id="other",
        kind="auto_eda",
        idempotency_key="stolen-key",
    )

    response = client.post(
        f"/api/v1/sessions/{source_run}/cleaning/apply",
        json=_apply_body(result),
        headers={"Idempotency-Key": "stolen-key"},
    )
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "idempotency_key_reused"
    row = store.get_pending_action(result["action_hash"], session_id=source_run)
    assert row is not None and row["status"] == "pending"


def test_apply_idempotency_fast_path_rejects_unconsumed_hash(
    client: TestClient, workspace: Path, source_run: str
) -> None:
    """A fabricated approval hash cannot reach the idempotency replay path."""
    store = ArtifactStore(workspace)
    store.create_job(
        job_id="job_prior",
        session_id="run_prior",
        project_id="demo",
        kind="auto_eda",
        idempotency_key="reused-key",
    )

    response = client.post(
        f"/api/v1/sessions/{source_run}/cleaning/apply",
        json={
            "action_hash": "f" * 64,
            "approval_token": "f" * 32,
            "llm": "offline",
        },
        headers={"Idempotency-Key": "reused-key"},
    )
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "idempotency_key_reused"


def test_cross_project_same_content_applies_independently(
    client: TestClient, workspace: Path, source_run: str
) -> None:
    """Slice-E F2: the same CSV content in two projects collides on
    action_hash by construction; each run must still preview and apply its own
    approval without stealing the other's pending row."""
    store = ArtifactStore(workspace)
    store.ensure_project("other", name="Other")
    seed2 = workspace / "seed2" / "orders.csv"
    seed2.parent.mkdir(parents=True, exist_ok=True)
    seed2.write_text(_SEED_CSV, encoding="utf-8")
    run2 = "run_clean_src2"
    created = client.post(
        f"/api/v1/sessions/{run2}/jobs",
        json={
            "kind": "auto_eda",
            "project_id": "other",
            "datasets": ["seed2/orders.csv"],
            "llm": "offline",
            "generate_report": False,
        },
    )
    assert created.status_code == 201, created.text
    assert _wait_terminal(client, created.json()["job_id"])["status"] == "completed"

    first = _preview(client, source_run, _dataset_id(client, source_run))
    second = _preview(client, run2, _dataset_id(client, run2))
    assert first["action_hash"] == second["action_hash"]

    applied1 = client.post(
        f"/api/v1/sessions/{source_run}/cleaning/apply",
        json=_apply_body(first),
    )
    assert applied1.status_code == 201, applied1.text
    applied2 = client.post(
        f"/api/v1/sessions/{run2}/cleaning/apply",
        json=_apply_body(second),
    )
    assert applied2.status_code == 201, applied2.text
    assert list((workspace / "projects" / "demo" / "cleaned").rglob("*.csv"))
    assert list((workspace / "projects" / "other" / "cleaned").rglob("*.csv"))
    _wait_terminal(client, applied1.json()["job"]["job_id"])
    _wait_terminal(client, applied2.json()["job"]["job_id"])


def test_apply_idempotency_key_replays_same_job(
    client: TestClient, source_run: str
) -> None:
    dataset_id = _dataset_id(client, source_run)
    result = _preview(client, source_run, dataset_id)
    headers = {"Idempotency-Key": "clean-apply-key"}
    first = client.post(
        f"/api/v1/sessions/{source_run}/cleaning/apply",
        json=_apply_body(result),
        headers=headers,
    )
    assert first.status_code == 201, first.text
    second = client.post(
        f"/api/v1/sessions/{source_run}/cleaning/apply",
        json=_apply_body(result),
        headers=headers,
    )
    assert second.status_code == 201, second.text
    assert second.json()["job"]["job_id"] == first.json()["job"]["job_id"]
    assert second.json()["new_session_id"] == first.json()["new_session_id"]
    _wait_terminal(client, first.json()["job"]["job_id"])


def test_preview_error_paths(client: TestClient, source_run: str) -> None:
    unknown_run = client.post(
        "/api/v1/sessions/run_missing/cleaning/preview", json={"dataset_id": "ds_x"}
    )
    assert unknown_run.status_code == 404
    assert unknown_run.json()["error"]["code"] == "session_not_found"

    unknown_dataset = client.post(
        f"/api/v1/sessions/{source_run}/cleaning/preview", json={"dataset_id": "ds_missing"}
    )
    assert unknown_dataset.status_code == 404
    assert unknown_dataset.json()["error"]["code"] == "dataset_not_found"

    nothing_selected = client.post(
        f"/api/v1/sessions/{source_run}/cleaning/preview",
        json={
            "dataset_id": _dataset_id(client, source_run),
            "trim_whitespace": False,
            "drop_duplicate_rows": False,
        },
    )
    assert nothing_selected.status_code == 422
    assert nothing_selected.json()["error"]["code"] == "cleaning_invalid"
