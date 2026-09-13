"""T3: per-session exploration listing and the LLM spend rollup surfaces."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from eda_platform.api.main import create_app
from eda_platform.application.ports import JobCommand, JobRef
from eda_platform.application.services.approval_service import ApprovalService
from eda_platform.application.services.exploration_service import (
    ExplorationService,
    ExplorationSourceSnapshot,
)
from eda_platform.application.services.job_service import JobService
from eda_platform.application.services.trace_service import (
    EXPLORATION_COST_EVENT,
    TraceService,
)
from eda_platform.core.exploration_shadow_store import shadow_run_root
from eda_platform.core.llm_ledger import BUDGET_SETTLED_EVENT, LLM_USAGE_EVENT
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.sessions import TraceEvent
from eda_platform.worker.runner import exploration_spend_rollup

SOURCE = "run_t3_source"
PROJECT = "demo"
DATASET = "ds_orders"
WITNESS = "dsw1_" + "c" * 32


class _RecordingBackend:
    def enqueue(self, command: JobCommand) -> JobRef:
        return JobRef(job_id=command.job_id)

    def cancel(self, job_id: str) -> None:  # pragma: no cover - unused
        raise NotImplementedError

    def status(self, job_id: str) -> str:
        return "queued"


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    store = ArtifactStore(tmp_path)
    store.ensure_project(PROJECT, name="Demo")
    store.start_session(PROJECT, SOURCE)
    return store


@pytest.fixture
def service(store: ArtifactStore) -> ExplorationService:
    def source(
        session_id: str, dataset_ids: tuple[str, ...]
    ) -> ExplorationSourceSnapshot:
        return ExplorationSourceSnapshot(PROJECT, dataset_ids, WITNESS)

    return ExplorationService(
        store,
        ApprovalService(store),
        JobService(store, _RecordingBackend()),
        source_snapshot_resolver=source,
    )


def _start(service: ExplorationService, *, goal: str | None = None) -> str:
    prepared = service.prepare(
        SOURCE,
        mode="open" if goal is None else "goal_directed",
        goal=goal,
        dataset_ids=(DATASET,),
        thinking_level="quick",
        provider="openai",
    )
    started = service.start(
        SOURCE,
        action_hash=prepared.action_hash,
        approval_token=prepared.approval_token,
        provider="openai",
        payload_policy=None,
        llm_env=None,
        idempotency_key=None,
    )
    return started.exploration.exploration_id


def test_lists_a_sessions_explorations_with_goal_and_status(
    service: ExplorationService,
) -> None:
    assert service.list_for_session(SOURCE) == ()

    exploration_id = _start(service, goal="Why did June conversion drop?")

    items = service.list_for_session(SOURCE)
    assert [item.exploration_id for item in items] == [exploration_id]
    item = items[0]
    assert item.session_id == SOURCE
    assert item.goal == "Why did June conversion drop?"
    assert item.mode == "goal_directed"
    assert item.thinking_level == "quick"
    assert item.status == "running"
    assert item.stop_reason is None
    assert item.report_available is False
    assert item.created_at is not None

    # The listing is scoped to the session, not the workspace.
    assert service.list_for_session("run_other") == ()


def test_open_mode_listing_reads_back_the_default_goal(
    service: ExplorationService,
) -> None:
    _start(service, goal=None)
    (item,) = service.list_for_session(SOURCE)
    assert item.mode == "open"
    assert item.goal == "Explore freely"


def test_openapi_registers_the_session_exploration_listing(tmp_path: Path) -> None:
    schema = create_app(tmp_path).openapi()
    path = "/api/v1/sessions/{session_id}/explorations"
    assert "get" in schema["paths"][path]


def _ledger_event(event_type: str, call_id: str, summary: dict) -> str:
    return TraceEvent(
        session_id="explsess_x",
        event_type=event_type,
        name="probe",
        call_id=call_id,
        started_at=datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
        summary=summary,
    ).model_dump_json()


def test_spend_rollup_sums_the_shadow_ledgers_billed_usage(
    store: ArtifactStore,
) -> None:
    exploration_id = "expl_" + "a" * 32
    root = shadow_run_root(store.root, exploration_id)
    root.mkdir(parents=True)
    lines = [
        _ledger_event(
            LLM_USAGE_EVENT,
            "call_1",
            {
                "prompt_tokens": 800,
                "completion_tokens": 150,
                "cached_tokens": 200,
                "total_tokens": 950,
                "estimated_cost_usd": 0.004,
            },
        ),
        _ledger_event(
            LLM_USAGE_EVENT,
            "call_2",
            {
                "prompt_tokens": 400,
                "completion_tokens": 100,
                "total_tokens": 500,
                "estimated_cost_usd": 0.002,
            },
        ),
        _ledger_event(BUDGET_SETTLED_EVENT, "call_2", {"status": "settled"}),
    ]
    (root / "llm-budget.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    rollup = exploration_spend_rollup(store, exploration_id)
    assert rollup == {
        "llm_calls": 2,
        "prompt_tokens": 1200,
        "completion_tokens": 250,
        "cached_tokens": 200,
        "total_tokens": 1450,
        "est_cost_usd": 0.006,
    }


def test_spend_rollup_is_none_without_billed_usage(store: ArtifactStore) -> None:
    assert exploration_spend_rollup(store, "expl_" + "b" * 32) is None


def test_metrics_surface_the_latest_exploration_cost_per_run(
    store: ArtifactStore,
) -> None:
    def cost_event(name: str, seq: int, summary: dict) -> TraceEvent:
        return TraceEvent(
            session_id=SOURCE,
            event_type=EXPLORATION_COST_EVENT,
            name=name,
            event_key=f"exploration_cost:{name}:{seq}",
            finished_at=datetime.now(UTC),
            summary=summary,
        )

    # Two rollups for the same run are cumulative snapshots: only the latest
    # counts, otherwise a resumed run would double-charge the page.
    store.append_trace(
        PROJECT,
        cost_event(
            "expl_1", 5, {"llm_calls": 2, "total_tokens": 900, "est_cost_usd": 0.004}
        ),
    )
    store.append_trace(
        PROJECT,
        cost_event(
            "expl_1", 9, {"llm_calls": 5, "total_tokens": 2000, "est_cost_usd": 0.01}
        ),
    )
    store.append_trace(
        PROJECT,
        cost_event(
            "expl_2", 3, {"llm_calls": 1, "total_tokens": 100, "est_cost_usd": 0.001}
        ),
    )

    metrics = TraceService(store).get_metrics(SOURCE)
    assert metrics.exploration_runs == 2
    assert metrics.exploration_llm_calls == 6
    assert metrics.exploration_total_tokens == 2100
    assert metrics.exploration_est_cost_usd == pytest.approx(0.011)


def test_metrics_report_no_exploration_spend_by_default(
    store: ArtifactStore,
) -> None:
    metrics = TraceService(store).get_metrics(SOURCE)
    assert metrics.exploration_runs == 0
    assert metrics.exploration_llm_calls == 0
    assert metrics.exploration_total_tokens == 0
    assert metrics.exploration_est_cost_usd is None
