"""The spend ledger is the single accounting seam for LLM cost.

Regression cover for the 2026-07-22 audit finding: per-driver ``llm_call``
events missed whole task families (m3_build_plan, di4_l1_interpretation), so
SessionMetrics under-counted calls and the UI header — reading the report-only
SessionSummary — showed 45-60% of real spend.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from threading import Lock
from typing import Any, cast

import pytest
from pydantic import BaseModel

from eda_platform.application.workbench import run_cost_summary
from eda_platform.core.budget import (
    BudgetUsageUncertain,
    SessionBudgetExceeded,
    SessionBudgetPolicy,
    SessionBudgetState,
)
from eda_platform.core.dev_log import InstrumentedLLMClient, read_llm_debug
from eda_platform.core.llm import (
    CancellableLLMClient,
    LLMProvider,
    LLMResultMetadata,
    LLMSettings,
    LLMUsage,
    OfflineLLMClient,
    OpenAICompatibleLLMClient,
    is_offline_client,
)
from eda_platform.core.llm_ledger import (
    LLM_USAGE_EVENT,
    LedgerLLMClient,
    meter_llm_client,
    restore_run_budget_state,
)
from eda_platform.core.session_metrics import summarize_session
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.sessions import TraceEvent


class _Answer(BaseModel):
    value: str = "ok"


class _FakeLLM:
    """Meters every call; raises on the task named ``boom``."""

    def __init__(self) -> None:
        self._last: LLMResultMetadata | None = None
        self.calls = 0

    def _meter(self) -> None:
        self.calls += 1
        self._last = LLMResultMetadata(
            provider="fake",
            model="fake-1",
            usage=LLMUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            estimated_cost_usd=0.001,
        )

    def structured(self, *, task: str, schema: type[BaseModel], payload: dict) -> Any:
        if task == "boom":
            raise RuntimeError("provider exploded")
        self._meter()
        return schema()

    def text(self, *, task: str, payload: dict) -> str:
        self._meter()
        return "text"

    def last_usage(self) -> LLMResultMetadata | None:
        return self._last


class _CopyingUsageLLM(_FakeLLM):
    """Returns an equal copy, as adapters that deserialize usage commonly do."""

    def last_usage(self) -> LLMResultMetadata | None:
        return deepcopy(self._last)


class _MissingUsageLLM(_FakeLLM):
    """A provider that answers but reports no usage block."""

    def text(self, *, task: str, payload: dict) -> str:
        self.calls += 1
        self._last = LLMResultMetadata(
            provider="fake", model="fake-1", usage_reported=False
        )
        return "text"


class _PassthroughCancellation:
    def checkpoint(self) -> None:
        return None

    @contextmanager
    def interrupt_on_cancel(self, _abort: Any) -> Any:
        yield


class _ConcurrentUsageLLM(_FakeLLM):
    def __init__(self) -> None:
        super().__init__()
        self._guard = Lock()
        self.active = 0
        self.max_active = 0

    def text(self, *, task: str, payload: dict) -> str:
        with self._guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.02)
        tokens = 10 if task == "ten" else 20
        self._last = LLMResultMetadata(
            provider="fake",
            model="fake-1",
            usage=LLMUsage(
                prompt_tokens=tokens,
                completion_tokens=0,
                total_tokens=tokens,
            ),
            estimated_cost_usd=0.0,
        )
        with self._guard:
            self.active -= 1
        return task


def _ledger(events: list[TraceEvent]) -> LedgerLLMClient:
    return LedgerLLMClient(cast(Any, _FakeLLM()), session_id="run_x", emit=events.append)


def test_every_call_lands_in_the_ledger_even_without_a_driver_event() -> None:
    events: list[TraceEvent] = []
    client = _ledger(events)

    client.structured(task="m3_build_plan", schema=_Answer, payload={})
    client.text(task="di4_l1_interpretation", payload={})

    ledger = [e for e in events if e.event_type == LLM_USAGE_EVENT]
    assert [e.name for e in ledger] == ["m3_build_plan", "di4_l1_interpretation"]
    assert all(e.summary["total_tokens"] == 150 for e in ledger)


def test_meter_writes_the_debug_capture_production_reads_back(tmp_path: Path) -> None:
    """Nothing in production built an InstrumentedLLMClient, so the file behind
    GET /sessions/{id}/llm-debug was always empty outside tests."""
    events: list[TraceEvent] = []
    store = ArtifactStore(tmp_path)
    store.ensure_project("p", name="P")
    store.start_session("p", "session_x")
    session_dir = store.session_dir("p", "session_x")
    client = meter_llm_client(
        cast(Any, _FakeLLM()),
        session_id="session_x",
        emit=events.append,
        session_dir=session_dir,
    )

    client.text(task="captured", payload={"value": 1})

    assert [r["task"] for r in read_llm_debug(session_dir)] == ["captured"]


def test_meter_does_not_stack_a_second_debug_wrapper(tmp_path: Path) -> None:
    events: list[TraceEvent] = []
    store = ArtifactStore(tmp_path)
    store.ensure_project("p", name="P")
    store.start_session("p", "session_x")
    session_dir = store.session_dir("p", "session_x")
    already = InstrumentedLLMClient(cast(Any, _FakeLLM()), session_dir=session_dir)
    client = meter_llm_client(
        cast(Any, already),
        session_id="session_x",
        emit=events.append,
        session_dir=session_dir,
    )

    client.text(task="captured", payload={})

    assert len(read_llm_debug(session_dir)) == 1


def test_cancellation_seam_keeps_provider_pricing_reachable() -> None:
    """Every worker job goes through CancellableLLMClient; when it swallowed
    .settings the ledger priced a metered call as cost-unknown."""
    settings = LLMSettings(
        provider=LLMProvider.OPENAI,
        model="gpt-4o-mini",
        api_key="k",
    )
    inner = OpenAICompatibleLLMClient(settings)
    seam = CancellableLLMClient(cast(Any, inner), cast(Any, _PassthroughCancellation()))

    client = LedgerLLMClient(cast(Any, seam), session_id="session_x", emit=lambda _e: None)

    assert client.settings is settings


def test_success_without_provider_usage_is_unknown_not_zero_cost() -> None:
    """A provider that returns no usage block used to be recorded as a real
    call with zero tokens, which reads as a free call rather than an unknown."""
    events: list[TraceEvent] = []
    client = LedgerLLMClient(
        cast(Any, _MissingUsageLLM()),
        session_id="session_x",
        emit=events.append,
    )

    client.text(task="missing_usage", payload={})

    event = events[-1]
    assert event.summary["status"] == "success"
    assert event.summary["usage_known"] is False
    assert event.summary["estimated_cost_usd"] is None


def test_failed_call_is_counted_but_not_billed_the_previous_call_tokens() -> None:
    events: list[TraceEvent] = []
    client = _ledger(events)

    client.structured(task="m2_report_claim_plan", schema=_Answer, payload={})
    with pytest.raises(RuntimeError):
        client.structured(task="boom", schema=_Answer, payload={})

    ledger = [e for e in events if e.event_type == LLM_USAGE_EVENT]
    assert len(ledger) == 2  # the failure still consumed a call slot
    assert ledger[1].summary["status"] == "RuntimeError"
    assert ledger[1].summary["total_tokens"] == 0  # not the previous call's 150


def test_failed_call_rejects_equal_but_nonidentical_stale_usage() -> None:
    events: list[TraceEvent] = []
    client = LedgerLLMClient(
        cast(Any, _CopyingUsageLLM()),
        session_id="run_x",
        emit=events.append,
    )

    client.text(task="first", payload={})
    with pytest.raises(RuntimeError):
        client.structured(task="boom", schema=_Answer, payload={})

    assert events[-1].summary["status"] == "RuntimeError"
    assert events[-1].summary["total_tokens"] == 0


def test_run_budget_rejects_before_a_second_provider_call() -> None:
    events: list[TraceEvent] = []
    inner = _FakeLLM()
    budget = SessionBudgetState(SessionBudgetPolicy(max_requests=1))
    client = LedgerLLMClient(
        cast(Any, inner),
        session_id="run_x",
        emit=events.append,
        budget=budget,
    )

    client.text(task="first", payload={})
    with pytest.raises(SessionBudgetExceeded) as raised:
        client.text(task="second", payload={})

    assert raised.value.dimension == "requests"
    assert inner.calls == 1
    assert budget.requests_used == 1
    assert [event.event_type for event in events] == [
        "budget_reserved",
        "llm_usage",
        "budget_settled",
        "budget_rejected",
    ]
    assert events[0].call_id
    assert events[1].call_id == events[0].call_id == events[2].call_id
    assert events[3].call_id != events[0].call_id


def test_ledger_serializes_mutable_last_usage_across_concurrent_calls() -> None:
    events: list[TraceEvent] = []
    inner = _ConcurrentUsageLLM()
    client = LedgerLLMClient(
        cast(Any, inner),
        session_id="run_x",
        emit=events.append,
        budget=SessionBudgetState(SessionBudgetPolicy(max_requests=2)),
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda task: client.text(task=task, payload={}),
                ("ten", "twenty"),
            )
        )

    assert results == ["ten", "twenty"]
    assert inner.max_active == 1
    usage = [event for event in events if event.event_type == LLM_USAGE_EVENT]
    assert sorted(event.summary["total_tokens"] for event in usage) == [10, 20]


def test_reservation_trace_failure_refuses_provider_call() -> None:
    inner = _FakeLLM()
    budget = SessionBudgetState(SessionBudgetPolicy(max_requests=1))

    def fail_emit(_event: TraceEvent) -> None:
        raise OSError("disk full")

    client = LedgerLLMClient(
        cast(Any, inner),
        session_id="run_x",
        emit=fail_emit,
        budget=budget,
    )

    with pytest.raises(BudgetUsageUncertain) as raised:
        client.text(task="never_called", payload={})

    assert raised.value.stage == "reservation"
    assert inner.calls == 0
    assert budget.active_reservations == ()


def test_unknown_failed_call_consumes_reservation_and_reconciles_ledger() -> None:
    events: list[TraceEvent] = []
    budget = SessionBudgetState(SessionBudgetPolicy(max_requests=2))
    client = LedgerLLMClient(
        cast(Any, _FakeLLM()),
        session_id="run_x",
        emit=events.append,
        budget=budget,
    )

    with pytest.raises(RuntimeError, match="provider exploded"):
        client.structured(task="boom", schema=_Answer, payload={"value": "x"})

    assert budget.requests_used == 1
    assert budget.total_tokens_used > 0
    usage_event = next(event for event in events if event.event_type == "llm_usage")
    settled_event = next(event for event in events if event.event_type == "budget_settled")
    assert usage_event.summary["usage_known"] is False
    assert usage_event.summary["total_tokens"] == budget.total_tokens_used
    assert settled_event.summary["status"] == "uncertain"


def test_cost_limit_fails_closed_when_provider_pricing_is_unknown() -> None:
    events: list[TraceEvent] = []
    inner = _FakeLLM()
    budget = SessionBudgetState(SessionBudgetPolicy(max_cost_usd=0.01))
    client = LedgerLLMClient(
        cast(Any, inner),
        session_id="run_x",
        emit=events.append,
        budget=budget,
    )

    with pytest.raises(BudgetUsageUncertain):
        client.text(task="priced_call", payload={})

    assert inner.calls == 0
    assert [event.event_type for event in events] == ["budget_rejected"]


def test_output_limit_fails_closed_when_adapter_has_no_request_cap() -> None:
    events: list[TraceEvent] = []
    inner = _FakeLLM()
    budget = SessionBudgetState(SessionBudgetPolicy(max_output_tokens=10))
    client = LedgerLLMClient(
        cast(Any, inner),
        session_id="run_x",
        emit=events.append,
        budget=budget,
    )

    with pytest.raises(BudgetUsageUncertain):
        client.text(task="uncapped_adapter", payload={})

    assert inner.calls == 0
    assert events[0].event_type == "budget_rejected"


@pytest.mark.parametrize(
    "policy",
    [
        SessionBudgetPolicy(max_total_tokens=10),
        SessionBudgetPolicy(max_cost_usd=0.01),
    ],
)
def test_total_or_cost_limit_requires_a_bounded_provider_output(
    policy: SessionBudgetPolicy,
) -> None:
    events: list[TraceEvent] = []
    inner = _FakeLLM()
    client = LedgerLLMClient(
        cast(Any, inner),
        session_id="run_x",
        emit=events.append,
        budget=SessionBudgetState(policy),
    )

    with pytest.raises(BudgetUsageUncertain) as raised:
        client.text(task="uncapped_adapter", payload={})

    assert raised.value.stage == "reservation"
    assert raised.value.missing == ("output_tokens",)
    assert inner.calls == 0


def test_persisted_budget_prevents_restart_from_resetting_run_limit() -> None:
    events: list[TraceEvent] = []
    policy = SessionBudgetPolicy(max_requests=1)
    first_budget = SessionBudgetState(policy)
    first = LedgerLLMClient(
        cast(Any, _FakeLLM()),
        session_id="run_x",
        emit=events.append,
        budget=first_budget,
    )
    first.text(task="first", payload={})

    restored = restore_run_budget_state(policy, events)
    second_inner = _FakeLLM()
    second = LedgerLLMClient(
        cast(Any, second_inner),
        session_id="run_x",
        emit=events.append,
        budget=restored,
    )
    with pytest.raises(SessionBudgetExceeded):
        second.text(task="second", payload={})

    assert restored.requests_used == 1
    assert second_inner.calls == 0


def test_restore_consumes_dangling_reservation_as_uncertain() -> None:
    policy = SessionBudgetPolicy(max_requests=1)
    events = [
        TraceEvent(
            session_id="run_x",
            event_type="budget_reserved",
            name="call",
            call_id="call_1",
            summary={
                "status": "reserved",
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "estimated_cost_usd": 0.0,
                "protected": False,
                "usage_known": None,
            },
        )
    ]

    restored = restore_run_budget_state(policy, events)

    assert restored.requests_used == 1
    reservation = restored.reservation("call_1")
    assert reservation is not None and reservation.status == "uncertain"


def test_restore_rejects_budget_policy_drift() -> None:
    events: list[TraceEvent] = []
    original = SessionBudgetPolicy(max_requests=1)
    client = LedgerLLMClient(
        cast(Any, _FakeLLM()),
        session_id="run_x",
        emit=events.append,
        budget=SessionBudgetState(original),
    )
    client.text(task="first", payload={})

    with pytest.raises(BudgetUsageUncertain, match="matching_budget_policy"):
        restore_run_budget_state(SessionBudgetPolicy(max_requests=2), events)


def test_restore_rejects_duplicate_budget_events() -> None:
    events: list[TraceEvent] = []
    policy = SessionBudgetPolicy(max_requests=2)
    client = LedgerLLMClient(
        cast(Any, _FakeLLM()),
        session_id="run_x",
        emit=events.append,
        budget=SessionBudgetState(policy),
    )
    client.text(task="first", payload={})
    reserved = next(event for event in events if event.event_type == "budget_reserved")

    with pytest.raises(BudgetUsageUncertain, match="unique_budget_reserved"):
        restore_run_budget_state(policy, [*events, reserved.model_copy()])


def test_metering_helper_does_not_stack_ledgers_for_the_same_run() -> None:
    first_events: list[TraceEvent] = []
    second_events: list[TraceEvent] = []
    first = meter_llm_client(
        cast(Any, _FakeLLM()),
        session_id="run_x",
        emit=first_events.append,
    )

    same = meter_llm_client(first, session_id="run_x", emit=second_events.append)
    same.text(task="one", payload={})

    assert same is first
    assert len(first_events) == 1
    assert second_events == []


def test_metering_helper_rebinds_existing_ledger_to_a_new_run() -> None:
    old_events: list[TraceEvent] = []
    new_events: list[TraceEvent] = []
    old = meter_llm_client(
        cast(Any, _FakeLLM()),
        session_id="run_old",
        emit=old_events.append,
    )

    new = meter_llm_client(old, session_id="run_new", emit=new_events.append)
    new.text(task="one", payload={})

    assert new.session_id == "run_new"
    assert old_events == []
    assert len(new_events) == 1


def test_run_metrics_prefers_the_ledger_over_narrative_llm_call_events(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "workspace")
    store.ensure_project("p", name="P")
    store.start_session("p", "r")
    usage = {"total_tokens": 150, "prompt_tokens": 100, "estimated_cost_usd": 0.001}
    # Two real calls; only one driver bothered to emit its narrative event.
    for task in ("m4_question_discovery", "m3_build_plan"):
        store.append_trace(
            "p", TraceEvent(session_id="r", event_type=LLM_USAGE_EVENT, name=task, summary=usage)
        )
    store.append_trace(
        "p",
        TraceEvent(
            session_id="r", event_type="llm_call", name="m4_question_discovery", summary=usage
        ),
    )

    metrics = summarize_session(store, "p", "r")

    assert metrics.llm_calls == 2  # ledger, not the single narrative event
    assert metrics.total_tokens == 300


def test_run_metrics_reconciles_budget_settlement_with_ledger(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "workspace")
    store.ensure_project("p", name="P")
    store.start_session("p", "r")
    budget = SessionBudgetState(SessionBudgetPolicy(max_requests=2))

    def emit(event: TraceEvent) -> None:
        store.append_trace("p", event)

    client = LedgerLLMClient(
        cast(Any, _FakeLLM()),
        session_id="r",
        emit=emit,
        budget=budget,
    )

    client.text(task="one", payload={})
    metrics = summarize_session(store, "p", "r")

    assert metrics.llm_calls == 1
    assert metrics.budget_reserved_calls == 1
    assert metrics.budget_settled_calls == 1
    assert metrics.budget_total_tokens == metrics.total_tokens == 150
    assert metrics.budget_est_cost_usd == metrics.est_cost_usd == 0.001
    assert metrics.budget_reconciliation == "verified"


def test_run_metrics_falls_back_to_llm_call_for_pre_ledger_runs(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "workspace")
    store.ensure_project("p", name="P")
    store.start_session("p", "r")
    store.append_trace(
        "p",
        TraceEvent(
            session_id="r",
            event_type="llm_call",
            name="m2_report_claim_plan",
            summary={"total_tokens": 150, "estimated_cost_usd": 0.001},
        ),
    )

    metrics = summarize_session(store, "p", "r")

    assert metrics.llm_calls == 1
    assert metrics.total_tokens == 150


def test_ui_cost_header_reports_whole_run_not_just_the_report_step() -> None:
    artifacts = [
        Artifact(
            id="run_metrics_1",
            type=ArtifactType.SESSION_METRICS,
            project_id="p",
            session_id="r",
            payload={"llm_calls": 22, "total_tokens": 103_905, "est_cost_usd": 0.0196},
        ),
        Artifact(
            id="runsummary_1",
            type=ArtifactType.SESSION_SUMMARY,
            project_id="p",
            session_id="r",
            payload={"llm_call_count": 3, "total_tokens": 63_442, "estimated_cost_usd": 0.0126},
        ),
    ]

    cost = run_cost_summary(artifacts)

    assert cost is not None
    assert cost["scope"] == "session"
    assert cost["llm_call_count"] == 22
    assert cost["estimated_cost_usd"] == 0.0196


def test_ui_cost_header_marks_pre_ledger_runs_as_report_only() -> None:
    artifacts = [
        Artifact(
            id="runsummary_1",
            type=ArtifactType.SESSION_SUMMARY,
            project_id="p",
            session_id="r",
            payload={"llm_call_count": 3, "total_tokens": 63_442, "estimated_cost_usd": 0.0126},
        )
    ]

    cost = run_cost_summary(artifacts)

    assert cost is not None and cost["scope"] == "report_only"


def test_offline_detection_survives_wrapping() -> None:
    """Capability checks must unwrap decorators, or offline runs go live."""
    offline = OfflineLLMClient()

    assert is_offline_client(offline)
    assert is_offline_client(LedgerLLMClient(offline, session_id="r", emit=lambda _e: None))
    assert is_offline_client(InstrumentedLLMClient(offline))
    assert is_offline_client(
        LedgerLLMClient(
            cast(Any, InstrumentedLLMClient(offline)), session_id="r", emit=lambda _e: None
        )
    )
    assert not is_offline_client(cast(Any, _FakeLLM()))
