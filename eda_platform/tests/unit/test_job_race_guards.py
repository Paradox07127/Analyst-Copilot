"""Review round 4 (codex-D): cancel/completion race, terminal reconnect 204,
early-exit run status, upstream traceback scrubbing."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from eda_platform.api.main import create_app
from eda_platform.core.store import ArtifactStore
from eda_platform.worker.runner import _sanitize_error


def _seed_job(store: ArtifactStore, job_id: str, status: str) -> None:
    store.ensure_project("demo", name="Demo")
    store.create_job(
        job_id=job_id,
        session_id="run_x",
        project_id="demo",
        kind="auto_eda",
        idempotency_key=None,
    )
    if status != "queued":
        store.mark_job_status(job_id, status)


def test_cancel_flag_never_lands_on_terminal_row(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    _seed_job(store, "job_done", "completed")
    assert store.request_cancel("job_done") is False
    job = store.get_job("job_done")
    assert job is not None and not job["cancel_requested"]


def test_reconnect_on_finished_job_returns_204(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    _seed_job(store, "job_fin", "completed")
    client = TestClient(create_app(tmp_path))
    # Fresh connect (no Last-Event-ID): stream replays and closes normally.
    with client.stream("GET", "/api/v1/jobs/job_fin/events") as fresh:
        assert fresh.status_code == 200
    # Reconnect with a cursor at/past the end: 204 so EventSource stops.
    response = client.get(
        "/api/v1/jobs/job_fin/events", headers={"Last-Event-ID": "999999"}
    )
    assert response.status_code == 204


def test_sanitize_strips_upstream_traceback(tmp_path: Path) -> None:
    raw = (
        "LLM call failed: 500 Internal Server Error\n"
        'Traceback (most recent call last):\n  File "/srv/llm/app.py", line 3\n'
    )
    cleaned = _sanitize_error(raw, str(tmp_path))
    assert "Traceback" not in cleaned
    assert "/srv/llm" not in cleaned
    assert "upstream traceback removed" in cleaned
