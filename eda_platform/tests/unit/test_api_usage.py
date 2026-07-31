"""Workspace usage rollup (GET /usage): the one cross-run aggregate in the API.

Everything else in the trace surface is per-run, so the home dashboard had no
way to answer "what has this workspace spent" without fanning out one request
per run. These pin the two things that make the rollup trustworthy: it counts
exactly the runs the session lists show, and it never folds an unpriced run
into a cost total silently.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from eda_platform.api.main import create_app
from eda_platform.core.ids import INTERNAL_SESSION_MARKER
from eda_platform.core.llm_ledger import LLM_USAGE_EVENT
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.sessions import TraceEvent

PROJECT = "demo"
OTHER = "scratch"

NOW = datetime.now(UTC)


def _usage_event(session_id: str, *, when: datetime, tokens: int, cost: float) -> TraceEvent:
    return TraceEvent(
        session_id=session_id,
        event_type=LLM_USAGE_EVENT,
        name="profile",
        started_at=when,
        finished_at=when,
        summary={
            "total_tokens": tokens,
            "prompt_tokens": tokens,
            "cached_tokens": 0,
            "estimated_cost_usd": cost,
        },
    )


def _start_run(store: ArtifactStore, project_id: str, session_id: str, *, created_at: datetime) -> None:
    """A run as the pipeline leaves it: started, with a manifest carrying the
    created_at the session lists and the activity window read."""
    store.start_session(project_id, session_id)
    manifest = store.session_dir(project_id, session_id) / "manifest.json"
    manifest.write_text(json.dumps({"created_at": created_at.isoformat()}), encoding="utf-8")
    store.refresh_session_index(project_id, session_id)


def _price(store: ArtifactStore, project_id: str, session_id: str, *, tokens: int, cost: float) -> None:
    """The SessionMetrics artifact a finished run persists. The rollup reads only
    these — see TraceService.workspace_usage."""
    store.save_artifact(
        Artifact(
            id=f"run_metrics_{session_id}",
            type=ArtifactType.SESSION_METRICS,
            project_id=project_id,
            session_id=session_id,
            payload={
                "session_id": session_id,
                "llm_calls": 1,
                "total_tokens": tokens,
                "est_cost_usd": cost,
            },
        )
    )


def _profile(
    store: ArtifactStore,
    project_id: str,
    session_id: str,
    *,
    dataset_id: str,
    rows: int,
) -> None:
    store.save_artifact(
        Artifact(
            id=f"profile_{session_id}_{dataset_id}",
            type=ArtifactType.DATASET_PROFILE,
            project_id=project_id,
            session_id=session_id,
            payload={"dataset_id": dataset_id, "name": f"{dataset_id}.csv", "rows": rows},
        )
    )


def _set_updated_at(store: ArtifactStore, session_id: str, updated_at: datetime) -> None:
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "update sessions set updated_at = ? where session_id = ?",
            (updated_at.isoformat(), session_id),
        )


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    store = ArtifactStore(tmp_path)
    store.ensure_project(PROJECT, name="Demo")
    store.ensure_project(OTHER, name="Scratch")
    # Two priced runs in one project, one unpriced run in another, plus an
    # internal run that must not appear in any figure.
    _start_run(store, PROJECT, "run_a", created_at=NOW)
    _start_run(store, PROJECT, "run_b", created_at=NOW)
    _start_run(store, OTHER, "run_c", created_at=NOW)
    _start_run(store, PROJECT, f"audit{INTERNAL_SESSION_MARKER}", created_at=NOW)
    store.append_trace(PROJECT, _usage_event("run_a", when=NOW, tokens=1000, cost=0.01))
    store.append_trace(PROJECT, _usage_event("run_b", when=NOW, tokens=500, cost=0.005))
    _price(store, PROJECT, "run_a", tokens=1000, cost=0.01)
    _price(store, PROJECT, "run_b", tokens=500, cost=0.005)
    return store


@pytest.fixture
def client(store: ArtifactStore) -> TestClient:
    return TestClient(create_app(store.root))


def test_usage_counts_only_the_runs_the_session_lists_show(client: TestClient) -> None:
    body = client.get("/api/v1/usage").json()

    assert body["window_days"] == 180
    assert body["project_count"] == 2
    # run_a, run_b, run_c — the internal audit run is excluded.
    assert body["session_count"] == 3


def test_usage_separates_priced_from_unpriced_sessions(client: TestClient) -> None:
    body = client.get("/api/v1/usage").json()

    assert body["total_tokens"] == 1500
    assert body["est_cost_usd"] == pytest.approx(0.015)
    assert body["priced_sessions"] == 2
    # run_c produced no LLM usage, so it contributes nothing and says so rather
    # than being averaged into the total.
    assert body["unpriced_sessions"] == 1


def test_a_metrics_artifact_without_a_cost_is_not_a_priced_session(
    client: TestClient, store: ArtifactStore
) -> None:
    """A run can record tokens and still fail to price them (unknown model).
    Counting it as priced would add $0 to a total that claims to cover it."""
    _start_run(store, PROJECT, "run_unpriceable", created_at=NOW)
    store.save_artifact(
        Artifact(
            id="run_metrics_run_unpriceable",
            type=ArtifactType.SESSION_METRICS,
            project_id=PROJECT,
            session_id="run_unpriceable",
            payload={"session_id": "run_unpriceable", "total_tokens": 900, "llm_calls": 2},
        )
    )

    body = client.get("/api/v1/usage").json()

    assert body["priced_sessions"] == 2
    assert body["unpriced_sessions"] == 2
    # Its tokens are still real and still counted; only its cost is unknown.
    assert body["total_tokens"] == 2400
    assert body["est_cost_usd"] == pytest.approx(0.015)


def test_usage_agrees_with_the_run_metrics_endpoint_after_a_later_call(
    client: TestClient, store: ArtifactStore
) -> None:
    """Usage recorded after the metrics artifact was written must land in both
    the per-run figure and the workspace total, or two screens disagree."""
    later = NOW + timedelta(minutes=5)
    store.append_trace(PROJECT, _usage_event("run_a", when=later, tokens=250, cost=0.004))

    per_run = client.get("/api/v1/sessions/run_a/metrics").json()
    workspace = client.get("/api/v1/usage").json()

    assert per_run["total_tokens"] == 1250
    assert workspace["total_tokens"] == 1750


def test_usage_buckets_sessions_by_utc_day_within_the_window(client: TestClient) -> None:
    body = client.get("/api/v1/usage?days=7").json()

    assert len(body["daily"]) == 7
    assert [day["date"] for day in body["daily"]] == sorted(
        day["date"] for day in body["daily"]
    )
    today = NOW.date().isoformat()
    by_date = {day["date"]: day["sessions"] for day in body["daily"]}
    assert by_date[today] == 3


def test_usage_recency_and_activity_follow_last_update(
    client: TestClient, store: ArtifactStore
) -> None:
    old = NOW - timedelta(days=40)
    recent = NOW + timedelta(minutes=10)
    _start_run(store, PROJECT, "run_revisited", created_at=old)
    _set_updated_at(store, "run_revisited", recent)

    body = client.get("/api/v1/usage?days=7").json()

    assert body["recent"][0]["session_id"] == "run_revisited"
    assert body["recent"][0]["created_at"].startswith(old.date().isoformat())
    assert body["recent"][0]["updated_at"].startswith(recent.date().isoformat())
    by_date = {day["date"]: day["sessions"] for day in body["daily"]}
    assert by_date[recent.date().isoformat()] == 4


def test_usage_window_is_bounded(client: TestClient) -> None:
    assert client.get("/api/v1/usage?days=0").status_code == 422
    assert client.get("/api/v1/usage?days=400").status_code == 422


def test_usage_reports_status_breakdown(client: TestClient) -> None:
    body = client.get("/api/v1/usage").json()

    assert sum(body["status_counts"].values()) == body["session_count"]


def test_usage_reports_profiled_rows_and_current_data_size(
    client: TestClient, store: ArtifactStore
) -> None:
    _profile(store, PROJECT, "run_a", dataset_id="orders", rows=1200)
    _profile(store, PROJECT, "run_b", dataset_id="customers", rows=300)
    with sqlite3.connect(store.db_path) as conn:
        conn.executemany(
            "insert into upload_usage(project_id, dataset_id, byte_size) values(?, ?, ?)",
            [(PROJECT, "orders", 2048), (PROJECT, "customers", 1024)],
        )

    body = client.get("/api/v1/usage").json()

    assert body["dataset_count"] == 2
    assert body["profiled_rows"] == 1500
    assert body["data_bytes"] == 3072


def test_usage_on_an_empty_workspace_is_zeroed_not_absent(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "empty")
    client = TestClient(create_app(store.root))

    body = client.get("/api/v1/usage").json()

    assert body["project_count"] == 0
    assert body["session_count"] == 0
    assert body["est_cost_usd"] == 0.0
    assert body["unpriced_sessions"] == 0
    assert body["daily"] != []


def test_usage_counts_sessions_that_belong_to_no_project(
    client: TestClient, store: ArtifactStore
) -> None:
    """A standalone session spends real money, so it has to reach these totals.
    Its storage bucket is still not a project and must not raise project_count."""
    store.ensure_project("unfiled-sessions", name="Unfiled sessions")
    _start_run(store, "unfiled-sessions", "run_unfiled", created_at=NOW)

    body = client.get("/api/v1/usage").json()

    assert body["project_count"] == 2
    assert body["session_count"] == 4
    assert "run_unfiled" in {item["session_id"] for item in body["recent"]}


def test_usage_window_excludes_older_sessions_from_period_totals(
    client: TestClient, store: ArtifactStore
) -> None:
    old = NOW - timedelta(days=40)
    _start_run(store, PROJECT, "run_old", created_at=old)
    _set_updated_at(store, "run_old", old)

    body = client.get("/api/v1/usage?days=7").json()

    # It remains eligible for Recent work, but not the selected-period figures.
    assert body["session_count"] == 3
    assert sum(day["sessions"] for day in body["daily"]) == 3
    assert sum(body["status_counts"].values()) == 3
