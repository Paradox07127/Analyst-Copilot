"""Trace & cost endpoints (§10.1 Trace): metrics source preference, event
pagination/filtering straight from SQL, and path relativization."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from eda_platform.api.main import create_app
from eda_platform.application.dto import ClientFailureRequest, SessionMetricsView
from eda_platform.application.services.trace_service import (
    ClientFailureRateLimitError,
    TraceService,
)
from eda_platform.core.ids import INTERNAL_SESSION_MARKER
from eda_platform.core.llm_ledger import LLM_USAGE_EVENT
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.session_metrics import SessionMetrics
from eda_platform.schemas.sessions import TraceEvent

PROJECT = "demo"
RUN = "run_1"
BARE_RUN = "run_2"

START = datetime(2026, 7, 25, 12, 0, tzinfo=UTC)


def test_metrics_api_exposes_every_typed_kernel_metric() -> None:
    assert set(SessionMetrics.model_fields) <= set(SessionMetricsView.model_fields)


def _client_failure(key: str | None = None) -> dict[str, str]:
    return {
        "error_code": "server_error",
        "operation": "mutation",
        "dedupe_key": key or str(uuid4()),
    }


def _event(
    event_type: str,
    name: str,
    *,
    offset: float,
    duration: float | None = None,
    summary: dict | None = None,
) -> TraceEvent:
    started = START + timedelta(seconds=offset)
    return TraceEvent(
        session_id=RUN,
        event_type=event_type,
        name=name,
        started_at=started,
        finished_at=None if duration is None else started + timedelta(seconds=duration),
        summary=summary or {},
    )


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    store = ArtifactStore(tmp_path)
    store.ensure_project(PROJECT, name="Demo")
    store.start_session(PROJECT, RUN)
    store.start_session(PROJECT, BARE_RUN)
    events = [
        _event("step_started", "profile_dataset", offset=0),
        _event(
            LLM_USAGE_EVENT,
            "profile",
            offset=1,
            summary={
                "total_tokens": 1000,
                "prompt_tokens": 800,
                "cached_tokens": 400,
                "estimated_cost_usd": 0.002,
            },
        ),
        _event("step_completed", "profile_dataset", offset=0, duration=4.0),
        _event("step_started", "export_agentic_report", offset=5),
        _event(
            LLM_USAGE_EVENT,
            "report",
            offset=6,
            summary={
                "total_tokens": 3000,
                "prompt_tokens": 2200,
                "cached_tokens": 1100,
                "estimated_cost_usd": 0.008,
            },
        ),
        # Narrative per-driver event: SessionMetrics bills the run from the ledger
        # but attributes per-step tokens from these, so both are present here.
        _event("llm_call", "report", offset=6, summary={"total_tokens": 3000}),
        _event("step_completed", "export_agentic_report", offset=5, duration=20.0),
        _event("failure_recorded", "report", offset=26, summary={"where": "exporter"}),
    ]
    for event in events:
        store.append_trace(PROJECT, event)
    return store


@pytest.fixture
def client(store: ArtifactStore) -> TestClient:
    return TestClient(create_app(store.root))


def test_metrics_aggregate_from_trace_events(client: TestClient) -> None:
    body = client.get(f"/api/v1/sessions/{RUN}/metrics").json()
    assert body["source"] == "aggregated"
    assert body["llm_calls"] == 2
    assert body["total_tokens"] == 4000
    assert body["prompt_tokens"] == 3000
    assert body["cached_tokens"] == 1500
    assert body["cache_hit_rate"] == 0.5
    assert body["est_cost_usd"] == 0.01
    assert body["failures_count"] == 1
    assert body["event_count"] == 8
    steps = {step["step_name"]: step for step in body["steps"]}
    assert steps["profile_dataset"]["duration_seconds"] == 4.0
    assert steps["export_agentic_report"]["duration_seconds"] == 20.0
    assert steps["export_agentic_report"]["tokens"] == 3000


def test_client_failure_is_typed_deduplicated_and_visible_in_trace(
    client: TestClient,
    store: ArtifactStore,
) -> None:
    payload = _client_failure()
    first = client.post(f"/api/v1/sessions/{RUN}/client-failures", json=payload)
    duplicate = client.post(f"/api/v1/sessions/{RUN}/client-failures", json=payload)

    assert first.status_code == 201
    assert first.json() == {"event_type": "failure_recorded", "recorded": True}
    assert duplicate.status_code == 201
    assert duplicate.json() == {"event_type": "failure_recorded", "recorded": False}

    trace = client.get(
        f"/api/v1/sessions/{RUN}/trace",
        params={"type": "failure_recorded", "limit": 100},
    ).json()
    client_rows = [row for row in trace["items"] if row["summary"].get("source") == "react"]
    assert len(client_rows) == 1
    assert client_rows[0]["name"] == "mutation"
    assert client_rows[0]["summary"] == {
        "source": "react",
        "error_code": "server_error",
        "operation": "mutation",
    }
    assert payload["dedupe_key"] not in str(client_rows[0])
    trace_lines = (
        store.session_dir(PROJECT, RUN) / "trace.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert sum('"source":"react"' in line for line in trace_lines) == 1
    assert payload["dedupe_key"] not in "\n".join(trace_lines)
    with store._connect() as conn:
        stored = conn.execute(
            "select event_key, payload from trace_events where event_key like ?",
            ("client-failure:%",),
        ).fetchall()
    assert stored
    assert payload["dedupe_key"] not in str(stored)


@pytest.mark.parametrize(
    "extra",
    [
        {"message": "secret customer@example.com"},
        {"stack": "at /Users/private/project.ts:10"},
        {"path": "/api/v1/private/raw/path"},
        {"body": {"password": "do-not-store"}},
        {"request_body": {"password": "do-not-store"}},
        {"error_code": "arbitrary_backend_exception"},
        {"operation": "/api/v1/private/raw/path"},
    ],
)
def test_client_failure_rejects_sensitive_or_unallowlisted_fields(
    client: TestClient,
    extra: dict,
) -> None:
    response = client.post(
        f"/api/v1/sessions/{RUN}/client-failures",
        json={**_client_failure(), **extra},
    )
    assert response.status_code == 422
    trace = client.get(
        f"/api/v1/sessions/{RUN}/trace",
        params={"type": "failure_recorded", "limit": 100},
    ).json()
    assert all(row["summary"].get("source") != "react" for row in trace["items"])


def test_client_failure_enforces_body_and_rate_boundaries(
    client: TestClient,
    store: ArtifactStore,
) -> None:
    too_large = client.post(
        f"/api/v1/sessions/{RUN}/client-failures",
        json=_client_failure(),
        headers={"Content-Length": "513"},
    )
    assert too_large.status_code == 413
    assert too_large.json()["error"]["code"] == "client_failure_too_large"

    # A backend failure uses the same event type but must not consume the
    # browser-specific rate budget.
    store.append_trace(
        PROJECT,
        TraceEvent(
            session_id=RUN,
            event_type="failure_recorded",
            name="backend",
            finished_at=datetime.now(UTC),
            summary={"source": "worker"},
        ),
    )
    for _ in range(20):
        response = client.post(
            f"/api/v1/sessions/{RUN}/client-failures",
            json=_client_failure(),
        )
        assert response.status_code == 201
    limited = client.post(
        f"/api/v1/sessions/{RUN}/client-failures",
        json=_client_failure(),
    )
    assert limited.status_code == 429
    assert limited.headers["retry-after"] == "60"
    assert limited.json()["error"]["code"] == "client_failure_rate_limited"


def test_client_failure_dedupe_and_rate_limit_are_one_durable_transaction(
    store: ArtifactStore,
) -> None:
    service = TraceService(store)
    shared = ClientFailureRequest.model_validate(_client_failure())

    with ThreadPoolExecutor(max_workers=8) as pool:
        duplicate_results = list(
            pool.map(
                lambda _: service.record_client_failure(BARE_RUN, shared).recorded,
                range(12),
            )
        )
    assert duplicate_results.count(True) == 1
    assert duplicate_results.count(False) == 11

    def record_distinct(_: int) -> str:
        try:
            recorded = service.record_client_failure(
                RUN, ClientFailureRequest.model_validate(_client_failure())
            )
            return "recorded" if recorded.recorded else "duplicate"
        except ClientFailureRateLimitError:
            return "limited"

    with ThreadPoolExecutor(max_workers=12) as pool:
        outcomes = list(pool.map(record_distinct, range(30)))
    assert outcomes.count("recorded") == 20
    assert outcomes.count("limited") == 10
    assert outcomes.count("duplicate") == 0
    rows = [
        event
        for event in store.list_trace_events(project_id=PROJECT, session_id=RUN)
        if event.event_type == "failure_recorded"
        and event.summary.get("source") == "react"
    ]
    assert len(rows) == 20


def test_metrics_prefer_the_persisted_run_metrics_artifact(store: ArtifactStore) -> None:
    """A run that already summarized itself is served from that artifact, not
    re-aggregated — the two differ here so the source is unambiguous."""
    store.save_artifact(
        Artifact(
            id="run_metrics_1",
            type=ArtifactType.SESSION_METRICS,
            project_id=PROJECT,
            session_id=RUN,
            payload={"session_id": RUN, "llm_calls": 99, "total_tokens": 12345},
        )
    )
    body = TestClient(create_app(store.root)).get(f"/api/v1/sessions/{RUN}/metrics").json()
    assert body["source"] == "artifact"
    assert (body["llm_calls"], body["total_tokens"]) == (99, 12345)
    assert body["generated_at"] is not None
    # The event count is always live, never a stale copy inside the artifact.
    assert body["event_count"] == 8


def test_metrics_expose_typed_quality_and_degradation_rollup(
    store: ArtifactStore,
) -> None:
    store.save_artifact(
        Artifact(
            id="run_metrics_quality",
            type=ArtifactType.SESSION_METRICS,
            project_id=PROJECT,
            session_id=RUN,
            payload={
                "session_id": RUN,
                "question_llm_skipped": True,
                "question_proposals_dropped": 2,
                "question_answered": 4,
                "question_abstained": 1,
                "result_contract_failures": {"invalid_payload": 3},
                "interpretation_fallbacks": 2,
                "semantic_degraded_claims": 1,
                "numeric_unverified_claims": 2,
                "coverage_limited": True,
                "publication_blocked": True,
                "publication_freshness": "unverifiable",
            },
        )
    )

    body = TestClient(create_app(store.root)).get(f"/api/v1/sessions/{RUN}/metrics").json()

    assert body["question_llm_skipped"] is True
    assert body["question_proposals_dropped"] == 2
    assert body["question_answered"] == 4
    assert body["question_abstained"] == 1
    assert body["result_contract_failures"] == {"invalid_payload": 3}
    assert body["interpretation_fallbacks"] == 2
    assert body["semantic_degraded_claims"] == 1
    assert body["numeric_unverified_claims"] == 2
    assert body["coverage_limited"] is True
    assert body["publication_blocked"] is True
    assert body["publication_freshness"] == "unverifiable"


def test_incremental_metrics_include_call_that_straddles_artifact_creation(
    store: ArtifactStore,
) -> None:
    started = datetime.now(UTC) - timedelta(minutes=5)
    store.save_artifact(
        Artifact(
            id="run_metrics_before_straddled_call",
            type=ArtifactType.SESSION_METRICS,
            project_id=PROJECT,
            session_id=RUN,
            payload={"session_id": RUN},
        )
    )
    store.append_trace(
        PROJECT,
        TraceEvent(
            session_id=RUN,
            event_type=LLM_USAGE_EVENT,
            name="straddled_chat_call",
            call_id="chat-call-1",
            started_at=started,
            finished_at=datetime.now(UTC),
            summary={
                "prompt_tokens": 10,
                "cached_tokens": 2,
                "total_tokens": 15,
                "estimated_cost_usd": 0.001,
            },
        ),
    )

    body = TestClient(create_app(store.root)).get(f"/api/v1/sessions/{RUN}/metrics").json()

    assert body["source"] == "artifact+incremental"
    assert body["llm_calls"] == 1
    assert body["total_tokens"] == 15


def test_metrics_empty_run(client: TestClient) -> None:
    body = client.get(f"/api/v1/sessions/{BARE_RUN}/metrics").json()
    assert (body["llm_calls"], body["total_tokens"], body["event_count"]) == (0, 0, 0)
    assert body["est_cost_usd"] is None


def test_trace_paginates_and_reports_event_types(client: TestClient) -> None:
    first = client.get(f"/api/v1/sessions/{RUN}/trace", params={"limit": 3}).json()
    assert [item["event_type"] for item in first["items"]] == [
        "step_started",
        LLM_USAGE_EVENT,
        "step_completed",
    ]
    assert first["total"] == 8
    assert first["event_types"][LLM_USAGE_EVENT] == 2
    assert first["next_cursor"]

    second = client.get(
        f"/api/v1/sessions/{RUN}/trace",
        params={"limit": 3, "cursor": first["next_cursor"]},
    ).json()
    assert [item["event_id"] for item in second["items"]] == [
        item["event_id"] + 3 for item in first["items"]
    ]
    third = client.get(
        f"/api/v1/sessions/{RUN}/trace",
        params={"limit": 3, "cursor": second["next_cursor"]},
    ).json()
    assert len(third["items"]) == 2
    assert third["next_cursor"] is None


def test_trace_filters_by_event_type(client: TestClient) -> None:
    body = client.get(
        f"/api/v1/sessions/{RUN}/trace", params={"type": "step_completed"}
    ).json()
    assert [item["name"] for item in body["items"]] == [
        "profile_dataset",
        "export_agentic_report",
    ]
    assert body["total"] == 2
    assert body["items"][1]["duration_seconds"] == 20.0
    # The histogram still describes the whole run, so the filter stays usable.
    assert body["event_types"][LLM_USAGE_EVENT] == 2


def test_trace_cursor_is_bound_to_its_filter(client: TestClient) -> None:
    cursor = client.get(
        f"/api/v1/sessions/{RUN}/trace", params={"limit": 1, "type": "step_started"}
    ).json()["next_cursor"]
    assert cursor
    replay = client.get(f"/api/v1/sessions/{RUN}/trace", params={"limit": 1, "cursor": cursor})
    assert replay.status_code == 400
    assert replay.json()["error"]["code"] == "invalid_cursor"


def test_trace_bad_cursor_400(client: TestClient) -> None:
    response = client.get(f"/api/v1/sessions/{RUN}/trace", params={"cursor": "%%%"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_cursor"


def test_trace_reads_sqlite_not_trace_jsonl(store: ArtifactStore) -> None:
    """The feed is pure SQL: deleting trace.jsonl must not change the response."""
    trace_path = store.session_dir(PROJECT, RUN) / "trace.jsonl"
    assert trace_path.is_file()
    trace_path.unlink()
    body = TestClient(create_app(store.root)).get(f"/api/v1/sessions/{RUN}/trace").json()
    assert len(body["items"]) == 8


def test_trace_summary_paths_are_relativized(store: ArtifactStore) -> None:
    absolute = str((store.root / "projects" / PROJECT / "uploads" / "orders.csv").resolve())
    store.append_trace(
        PROJECT, _event("dataset_loaded", "orders", offset=30, summary={"source": absolute})
    )
    response = TestClient(create_app(store.root)).get(
        f"/api/v1/sessions/{RUN}/trace", params={"type": "dataset_loaded"}
    )
    assert response.json()["items"][0]["summary"]["source"] == (
        f"projects/{PROJECT}/uploads/orders.csv"
    )
    assert str(store.root) not in response.text


def test_trace_and_metrics_unknown_run_404(client: TestClient) -> None:
    for path in ("/api/v1/sessions/missing/trace", "/api/v1/sessions/missing/metrics"):
        response = client.get(path)
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "session_not_found"


def test_trace_internal_run_404(client: TestClient) -> None:
    response = client.get(f"/api/v1/sessions/{RUN}{INTERNAL_SESSION_MARKER}_probe/trace")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "session_not_found"
