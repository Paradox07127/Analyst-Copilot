"""E2-B gates: tool-call ledger latches, amendment chain monotonicity, reserve."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from eda_platform.core.budget import SessionBudgetExceeded, SessionBudgetState
from eda_platform.core.exploration_budget import (
    BudgetAmendmentError,
    ToolBudgetExceeded,
    ToolCallLedger,
    ToolCallProjection,
    amendment_chain_digest,
    effective_policy,
    finalization_view,
    policy_fingerprint,
)
from eda_platform.schemas.exploration_budget import (
    BudgetAmendment,
    BudgetCapIncrease,
    ExplorationBudgetPolicy,
    SessionBudgetPolicyModel,
)


def _policy(**overrides: Any) -> ExplorationBudgetPolicy:
    fields: dict[str, Any] = {
        "llm": SessionBudgetPolicyModel(),
        "max_successful_tool_calls": 10,
        "max_tool_calls_by_kind": {"run_open_analysis": 5, "profile_slice": 5},
        "max_rows_scanned": None,
        "max_result_cells": None,
        "idle_timeout_seconds": 60.0,
        "max_rounds": 8,
    }
    fields.update(overrides)
    return ExplorationBudgetPolicy(**fields)


def _amendment(amendment_id: str, previous: str, **increase: Any) -> BudgetAmendment:
    return BudgetAmendment(
        amendment_id=amendment_id,
        previous_effective_fingerprint=previous,
        increase=BudgetCapIncrease(**increase),
        reason="need more budget",
        approved_by="tester",
        created_at="2026-08-02T00:00:00Z",
    )


def _probe(kind: str = "run_open_analysis", **kwargs: int) -> list[ToolCallProjection]:
    return [ToolCallProjection(kind=kind, **kwargs)]


# --- ToolCallLedger: hard caps and latches ---


def test_total_success_cap_latches_and_rejects_new_requests() -> None:
    ledger = ToolCallLedger(_policy(max_successful_tool_calls=2))
    ledger.check_batch(_probe())
    ledger.record_success("run_open_analysis")
    ledger.check_batch(_probe())
    ledger.record_success("profile_slice")

    with pytest.raises(ToolBudgetExceeded) as exc:
        ledger.check_batch(_probe())
    assert exc.value.dimension == "successful_tool_calls"
    assert exc.value.latched is True
    assert "successful_tool_calls" in ledger.exhausted_dimensions()


def test_failed_calls_do_not_consume_the_success_budget() -> None:
    ledger = ToolCallLedger(_policy(max_successful_tool_calls=2))
    for _ in range(5):  # five failed executions: check passes, nothing recorded
        ledger.check_batch(_probe())
    assert ledger.snapshot()["successful_tool_calls"] == 0

    ledger.record_success("run_open_analysis")
    ledger.record_success("run_open_analysis")
    assert ledger.remaining()["successful_tool_calls"] == 0
    with pytest.raises(ToolBudgetExceeded):
        ledger.check_batch(_probe())


def test_per_kind_cap_latches_only_that_kind() -> None:
    ledger = ToolCallLedger(
        _policy(max_tool_calls_by_kind={"run_open_analysis": 1, "profile_slice": 2})
    )
    ledger.record_success("run_open_analysis")

    with pytest.raises(ToolBudgetExceeded) as exc:
        ledger.check_batch(_probe("run_open_analysis"))
    assert exc.value.dimension == "tool_calls_by_kind"
    assert exc.value.kind == "run_open_analysis"
    assert exc.value.latched is True

    ledger.check_batch(_probe("profile_slice"))
    assert "tool_calls_by_kind:run_open_analysis" in ledger.exhausted_dimensions()


def test_unlisted_kind_is_rejected_fail_closed() -> None:
    ledger = ToolCallLedger(_policy())
    with pytest.raises(ToolBudgetExceeded) as exc:
        ledger.check_batch(_probe("sneaky_tool"))
    assert exc.value.kind == "sneaky_tool"
    assert exc.value.limit == 0

    with pytest.raises(ToolBudgetExceeded) as settled:
        ledger.record_success("sneaky_tool")
    assert settled.value.stage == "settlement"


def test_rows_and_cells_caps_accumulate_and_latch() -> None:
    ledger = ToolCallLedger(_policy(max_rows_scanned=100, max_result_cells=50))
    ledger.record_success("run_open_analysis", rows_scanned=60, result_cells=10)

    with pytest.raises(ToolBudgetExceeded) as exc:
        ledger.check_batch(_probe(rows_scanned=50))
    assert exc.value.dimension == "rows_scanned"
    assert exc.value.latched is False  # cap not yet hit, only this batch too big

    ledger.record_success("run_open_analysis", rows_scanned=40)
    with pytest.raises(ToolBudgetExceeded) as latched:
        ledger.check_batch(_probe(rows_scanned=1))
    assert latched.value.latched is True
    assert "rows_scanned" in ledger.exhausted_dimensions()

    ledger.check_batch(_probe(result_cells=40))  # rows latched, rows=0 call still fine
    with pytest.raises(ToolBudgetExceeded) as cells:
        ledger.check_batch(_probe(result_cells=41))
    assert cells.value.dimension == "result_cells"


def test_check_batch_rejects_the_whole_batch_and_commits_nothing() -> None:
    ledger = ToolCallLedger(_policy(max_rows_scanned=100))
    batch = [
        ToolCallProjection(kind="run_open_analysis", rows_scanned=60),
        ToolCallProjection(kind="profile_slice", rows_scanned=50),
    ]
    with pytest.raises(ToolBudgetExceeded):
        ledger.check_batch(batch)
    assert ledger.snapshot() == {
        "successful_tool_calls": 0,
        "calls_by_kind": {},
        "rows_scanned": 0,
        "result_cells": 0,
    }

    ledger.check_batch(batch[:1])  # passing pre-check alone must not commit either
    assert ledger.snapshot()["successful_tool_calls"] == 0
    ledger.record_success("run_open_analysis", rows_scanned=60)
    assert ledger.snapshot()["rows_scanned"] == 60


def test_record_success_records_actuals_then_raises_like_settlement() -> None:
    ledger = ToolCallLedger(_policy(max_rows_scanned=100))
    with pytest.raises(ToolBudgetExceeded) as exc:
        ledger.record_success("run_open_analysis", rows_scanned=150)
    assert exc.value.stage == "settlement"
    assert ledger.snapshot()["rows_scanned"] == 150
    assert "rows_scanned" in ledger.exhausted_dimensions()


# --- Amendment chain ---


def _amend_base() -> ExplorationBudgetPolicy:
    return _policy(
        llm=SessionBudgetPolicyModel(max_total_tokens=2000),
        max_successful_tool_calls=3,
        max_rows_scanned=200,
    )


def test_amendment_chain_raises_caps_monotonically() -> None:
    base = _amend_base()
    fp0 = policy_fingerprint(base)
    a1 = _amendment(
        "a1",
        fp0,
        max_successful_tool_calls=2,
        max_tool_calls_by_kind={"run_open_analysis": 1, "fit_baseline": 3},
        max_rows_scanned=100,
        max_total_tokens=1000,
    )
    p1 = effective_policy(base, [a1])
    assert p1.max_successful_tool_calls == 5
    assert p1.max_tool_calls_by_kind == {
        "run_open_analysis": 6,
        "profile_slice": 5,
        "fit_baseline": 3,
    }
    assert p1.max_rows_scanned == 300
    assert p1.llm.max_total_tokens == 3000

    fp1 = policy_fingerprint(p1)
    assert fp1 != fp0
    a2 = _amendment("a2", fp1, max_rounds=2)
    p2 = effective_policy(base, [a1, a2])
    assert p2.max_rounds == 10

    assert policy_fingerprint(effective_policy(base, [a1])) == fp1  # deterministic
    assert amendment_chain_digest([a1, a2]) != amendment_chain_digest([a2, a1])
    assert amendment_chain_digest([a1, a2]) == amendment_chain_digest([a1, a2])


def test_wrong_previous_fingerprint_is_rejected() -> None:
    base = _amend_base()
    bad = _amendment("a1", "fp_of_nothing", max_rounds=1)
    with pytest.raises(BudgetAmendmentError, match="fingerprint"):
        effective_policy(base, [bad])


def test_out_of_order_chain_is_rejected() -> None:
    base = _amend_base()
    a1 = _amendment("a1", policy_fingerprint(base), max_rounds=1)
    fp1 = policy_fingerprint(effective_policy(base, [a1]))
    a2 = _amendment("a2", fp1, max_rounds=1)
    with pytest.raises(BudgetAmendmentError, match="fingerprint"):
        effective_policy(base, [a2, a1])


def test_replayed_amendment_id_is_rejected() -> None:
    base = _amend_base()
    a1 = _amendment("a1", policy_fingerprint(base), max_rounds=1)
    fp1 = policy_fingerprint(effective_policy(base, [a1]))
    replay = _amendment("a1", fp1, max_rounds=1)
    with pytest.raises(BudgetAmendmentError, match="a1"):
        effective_policy(base, [a1, replay])


def test_forged_cap_decrease_is_rejected() -> None:
    base = _amend_base()
    forged_increase = BudgetCapIncrease.model_construct(max_successful_tool_calls=-2)
    forged = BudgetAmendment.model_construct(
        amendment_id="forged",
        previous_effective_fingerprint=policy_fingerprint(base),
        increase=forged_increase,
        reason="lower it",
        approved_by="attacker",
        created_at="2026-08-02T00:00:00Z",
    )
    with pytest.raises(BudgetAmendmentError):
        effective_policy(base, [forged])


def test_update_policy_guard_has_teeth() -> None:
    ledger = ToolCallLedger(_amend_base())
    with pytest.raises(BudgetAmendmentError, match="max_successful_tool_calls"):
        ledger.update_policy(_amend_base().model_copy(update={"max_successful_tool_calls": 1}))
    with pytest.raises(BudgetAmendmentError, match="max_result_cells"):
        # unlimited (None) may not become a finite cap
        ledger.update_policy(_amend_base().model_copy(update={"max_result_cells": 100}))
    with pytest.raises(BudgetAmendmentError, match="profile_slice"):
        ledger.update_policy(
            _amend_base().model_copy(update={"max_tool_calls_by_kind": {"run_open_analysis": 9}})
        )
    with pytest.raises(BudgetAmendmentError, match="idle_timeout_seconds"):
        ledger.update_policy(_amend_base().model_copy(update={"idle_timeout_seconds": 30.0}))


def test_unlimited_caps_stay_unlimited_after_amendment() -> None:
    base = _policy(llm=SessionBudgetPolicyModel())  # every LLM cap None, rows/cells None
    a1 = _amendment(
        "a1", policy_fingerprint(base), max_rows_scanned=50, max_requests=5, max_rounds=1
    )
    amended = effective_policy(base, [a1])
    assert amended.max_rows_scanned is None
    assert amended.llm.max_requests is None
    assert amended.max_rounds == 9


def test_amended_policy_unlatches_a_hit_dimension() -> None:
    base = _policy(max_successful_tool_calls=1)
    ledger = ToolCallLedger(base)
    ledger.record_success("run_open_analysis")
    with pytest.raises(ToolBudgetExceeded):
        ledger.check_batch(_probe())

    a1 = _amendment("a1", policy_fingerprint(base), max_successful_tool_calls=1)
    ledger.update_policy(effective_policy(base, [a1]))
    assert "successful_tool_calls" not in ledger.exhausted_dimensions()
    ledger.check_batch(_probe())
    ledger.record_success("run_open_analysis")
    with pytest.raises(ToolBudgetExceeded):
        ledger.check_batch(_probe())


# --- Snapshot / restore ---


def test_snapshot_restore_round_trip_continues_the_account() -> None:
    policy = _policy(max_successful_tool_calls=3, max_rows_scanned=100)
    ledger = ToolCallLedger(policy)
    ledger.record_success("run_open_analysis", rows_scanned=30, result_cells=4)
    ledger.record_success("profile_slice", rows_scanned=20)

    snapshot = ledger.snapshot()
    restored = ToolCallLedger.restore(policy, snapshot)
    assert restored.snapshot() == snapshot
    assert restored.remaining() == ledger.remaining()

    restored.record_success("run_open_analysis", rows_scanned=50)
    assert restored.remaining()["successful_tool_calls"] == 0
    with pytest.raises(ToolBudgetExceeded):
        restored.check_batch(_probe())


def test_restore_rejects_corrupt_snapshots() -> None:
    policy = _policy()
    good = ToolCallLedger(policy).snapshot()

    with pytest.raises(ValueError, match="keys"):
        ToolCallLedger.restore(policy, {**good, "extra": 1})
    with pytest.raises(ValueError, match="sum"):
        ToolCallLedger.restore(policy, {**good, "successful_tool_calls": 2})
    with pytest.raises(ValueError, match="negative"):
        ToolCallLedger.restore(policy, {**good, "rows_scanned": -1})


# --- Finalization reserve (LLM dimensions via core.budget, zero new enforcement code) ---


def _reserve_policy() -> ExplorationBudgetPolicy:
    return _policy(
        llm=SessionBudgetPolicyModel(
            max_requests=3,
            max_total_tokens=1000,
            protected_requests=1,
            protected_total_tokens=200,
        )
    )


def test_protected_reserve_survives_main_budget_exhaustion() -> None:
    policy = _reserve_policy()
    state = SessionBudgetState(policy.llm.to_policy())

    state.reserve("probe_1", total_tokens=800)
    state.settle("probe_1", input_tokens=0, output_tokens=0, total_tokens=800)

    with pytest.raises(SessionBudgetExceeded) as exc:  # main budget latched
        state.reserve("probe_2", total_tokens=1)
    assert exc.value.dimension == "total_tokens"

    finalization = state.reserve("finalize", total_tokens=200, protected=True)
    assert finalization.protected is True
    state.settle("finalize", input_tokens=0, output_tokens=0, total_tokens=200)
    assert state.remaining("total_tokens", protected=True) == 0


def test_finalization_view_reports_reserved_and_remaining() -> None:
    policy = _reserve_policy()
    static = finalization_view(policy)
    assert static["total_tokens"] == {"reserved": 200, "remaining": None}
    assert static["requests"] == {"reserved": 1, "remaining": None}

    state = SessionBudgetState(policy.llm.to_policy())
    state.reserve("probe_1", total_tokens=800)
    state.settle("probe_1", input_tokens=0, output_tokens=0, total_tokens=800)
    live = finalization_view(policy, state)
    assert live["total_tokens"] == {"reserved": 200, "remaining": 200}
    assert live["requests"]["remaining"] == 2

    state.reserve("finalize", total_tokens=200, protected=True)
    state.settle("finalize", input_tokens=0, output_tokens=0, total_tokens=200)
    assert finalization_view(policy, state)["total_tokens"]["remaining"] == 0


def test_a_zero_increase_amendment_is_rejected() -> None:
    """An amendment that raises nothing would let a caller mint chain links
    (and burn amendment ids) without ever granting capacity."""
    with pytest.raises(ValidationError):
        BudgetCapIncrease()
    with pytest.raises(ValidationError):
        BudgetCapIncrease(max_tool_calls_by_kind={})


def test_protected_capacity_cannot_exceed_or_float_without_its_cap() -> None:
    with pytest.raises(ValidationError):
        SessionBudgetPolicyModel(protected_total_tokens=10)
    with pytest.raises(ValidationError):
        SessionBudgetPolicyModel(max_total_tokens=100, protected_total_tokens=101)


def test_per_kind_caps_must_be_positive_and_named() -> None:
    with pytest.raises(ValidationError):
        _policy(max_tool_calls_by_kind={"run_open_analysis": 0})
    with pytest.raises(ValidationError):
        _policy(max_tool_calls_by_kind={"": 3})
