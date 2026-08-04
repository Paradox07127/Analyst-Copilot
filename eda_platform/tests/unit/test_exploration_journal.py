"""Exploration journal: policy fingerprint, event model, reducer, and recovery.

Covers the journal half of the E2 gate: torn-tail recovery, epoch fencing,
four crash-injection positions, pause as a resumable (non-stop) status,
terminal stop semantics, and the six-family snapshot against the Eval-0
checkers.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import get_args

import pytest

from eda_platform.core.event_journal import (
    EventJournalCorruptionError,
    EventTransitionError,
)
from eda_platform.core.exploration_budget import ToolCallLedger
from eda_platform.core.exploration_journal import (
    ExplorationPolicyIntegrityError,
    ExplorationResumeIncompatibleError,
    JsonlExplorationJournal,
    RecoveredToolCommit,
    amended_policy_fingerprint,
    compute_policy_fingerprint,
    rebuild_exploration_state,
    sealed_policy,
)
from eda_platform.schemas.exploration import (
    BranchConstraint,
    ExplorationPolicy,
    ExplorationStateUnavailableError,
    ExplorationStopReason,
    InsightFamily,
    LlmCallStartedEvent,
    RoundStartedEvent,
)
from eda_platform.schemas.exploration_budget import (
    BudgetCapIncrease,
    ExplorationBranchPolicy,
    ExplorationBudgetPolicy,
    SessionBudgetPolicyModel,
)

CHECKERS_PATH = (
    Path(__file__).resolve().parents[1] / "evals" / "exploration_baseline" / "checkers.py"
)


def _budget(**overrides: object) -> ExplorationBudgetPolicy:
    fields: dict[str, object] = {
        "llm": SessionBudgetPolicyModel(max_requests=4),
        "max_successful_tool_calls": 3,
        "max_tool_calls_by_kind": {"run_open_analysis": 3},
        "idle_timeout_seconds": 30.0,
        "max_rounds": 3,
    }
    fields.update(overrides)
    return ExplorationBudgetPolicy.model_validate(fields)


def _policy_fields(**overrides: object) -> dict[str, object]:
    fields: dict[str, object] = {
        "mode": "open",
        "goal": "baseline goal",
        "dataset_scope": ("ds_a",),
        "thinking_level": "standard",
        "coverage_targets": (InsightFamily.DESCRIPTIVE, InsightFamily.DIAGNOSTIC),
        "budget": _budget(),
        "scoring_policy_version": "score-v1",
        "statistical_policy_version": "stats-v1",
        "tool_capability_digest": "tools-v1",
    }
    fields.update(overrides)
    return fields


def _policy(**overrides: object) -> ExplorationPolicy:
    return sealed_policy(ExplorationPolicy.model_validate(_policy_fields(**overrides)))


def _journal(
    tmp_path: Path, *, policy: ExplorationPolicy | None = None
) -> JsonlExplorationJournal:
    journal = JsonlExplorationJournal(
        tmp_path / "exploration" / "exploration.journal.jsonl"
    )
    journal.initialize(
        exploration_id="xpl_test",
        policy=policy or _policy(),
        code_fingerprint="code-v1",
        data_state_witness="dsw1_test",
    )
    return journal


def _run_one_tool_step(
    journal: JsonlExplorationJournal, step: str, receipt: str
) -> None:
    journal.append_new(
        "tool_call_started", logical_step_id=step, input_fingerprint=f"fp-{step}"
    )
    journal.append_new("receipt_prepared", logical_step_id=step, receipt_id=receipt)
    journal.append_new("receipt_committed", logical_step_id=step, receipt_id=receipt)


# ---------------------------------------------------------------- schema layer


def test_insight_family_matches_eval_checker_families_verbatim() -> None:
    spec = importlib.util.spec_from_file_location("eval0_checkers_snapshot", CHECKERS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert tuple(family.value for family in InsightFamily) == module.SIX_INSIGHT_FAMILIES


def test_stop_reasons_exclude_user_paused() -> None:
    assert set(get_args(ExplorationStopReason)) == {
        "completed",
        "budget_exhausted",
        "cancelled",
        "failed",
        "state_witness_changed",
        "no_new_information",
    }


def test_goal_directed_mode_requires_a_goal() -> None:
    with pytest.raises(ValueError, match="goal"):
        ExplorationPolicy.model_validate(
            _policy_fields(mode="goal_directed", goal=None)
        )


def test_fingerprint_covers_every_execution_affecting_field() -> None:
    base = ExplorationPolicy.model_validate(_policy_fields())
    baseline = compute_policy_fingerprint(base)

    mutations: dict[str, object] = {
        "mode": "goal_directed",
        "goal": "a different goal",
        "dataset_scope": ("ds_a", "ds_b"),
        "thinking_level": "deep",
        "coverage_targets": (InsightFamily.PREDICTIVE,),
        "budget": _budget(max_rounds=9),
        "scoring_policy_version": "score-v2",
        "statistical_policy_version": "stats-v2",
        "tool_capability_digest": "tools-v2",
    }
    # Every execution-affecting field must have a covering mutation; adding a
    # field to the policy without extending this map fails here by design.
    assert set(mutations) == set(ExplorationPolicy.model_fields) - {"policy_fingerprint"}

    for name, value in mutations.items():
        mutated = ExplorationPolicy.model_validate(_policy_fields(**{name: value}))
        assert compute_policy_fingerprint(mutated) != baseline, name


def test_sealed_policy_round_trip_and_tamper_rejection(tmp_path: Path) -> None:
    policy = _policy()
    assert policy.policy_fingerprint == compute_policy_fingerprint(policy)

    tampered = policy.model_copy(update={"policy_fingerprint": "xplcy_forged"})
    journal = JsonlExplorationJournal(tmp_path / "exploration.journal.jsonl")
    with pytest.raises(ExplorationPolicyIntegrityError):
        journal.initialize(
            exploration_id="xpl_test",
            policy=tampered,
            code_fingerprint="code-v1",
            data_state_witness="dsw1_test",
        )
    unsealed = ExplorationPolicy.model_validate(_policy_fields())
    with pytest.raises(ExplorationPolicyIntegrityError):
        journal.initialize(
            exploration_id="xpl_test",
            policy=unsealed,
            code_fingerprint="code-v1",
            data_state_witness="dsw1_test",
        )


# ------------------------------------------------------------------ lifecycle


def test_full_round_lifecycle_and_prefix_rebuild(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.claim_attempt()
    journal.append_new("round_started", round_index=0)
    call_id = "call-0"
    journal.append_new("llm_call_started", call_id=call_id)
    journal.append_new(
        "llm_call_completed",
        call_id=call_id,
        step_id="step-llm-0",
        response_digest="resp-0",
    )
    _run_one_tool_step(journal, "step-tool-0", "rcpt-0")
    journal.append_new("gate_verdict", claim_bundle_id="claim-0", verdict="passed")
    journal.append_new(
        "reduction_committed", frontier_digest="frontier-1", ledger_digest="ledger-1"
    )
    journal.append_new("round_settled", round_index=0, progress=True)
    final = journal.append_new(
        "exploration_stopped", stop_reason="completed", final_report_ref="report-1"
    )

    events = journal.events()
    for size in range(1, len(events) + 1):
        assert rebuild_exploration_state(events[:size]) is not None
    assert journal.rebuild() == final

    assert final.status == "stopped"
    assert final.require_stop_reason() == "completed"
    assert final.llm_calls_settled == 1
    assert final.remaining_llm_call_budget == 3
    assert final.tool_calls_committed == 1
    assert final.remaining_tool_call_budget == 2
    assert final.rounds_settled == 1
    assert final.remaining_round_budget == 2
    assert final.step_receipt_refs == {"step-tool-0": "rcpt-0"}
    assert final.gate_verdicts == {"claim-0": "passed"}
    assert final.require_frontier_digest() == "frontier-1"
    assert final.require_ledger_digest() == "ledger-1"


def test_round_settlement_durably_records_terminal_decision_for_crash_recovery(
    tmp_path: Path,
) -> None:
    journal = _journal(tmp_path)
    journal.claim_attempt()
    journal.append_new("round_started", round_index=0)
    journal.append_new(
        "reduction_committed", frontier_digest="frontier-1", ledger_digest="ledger-1"
    )

    settled = journal.append_new(
        "round_settled",
        round_index=0,
        progress=True,
        terminal_reason="completed",
        terminal_has_reduction=True,
    )

    assert settled.current_round_index is None
    assert settled.pending_terminal_reason == "completed"
    assert settled.pending_terminal_has_reduction is True
    with pytest.raises(EventTransitionError, match="terminal decision"):
        journal.append_new("round_started", round_index=1)
    with pytest.raises(EventTransitionError, match="does not match"):
        journal.append_new(
            "exploration_stopped",
            stop_reason="budget_exhausted",
            final_report_ref=None,
        )
    stopped = journal.append_new(
        "exploration_stopped", stop_reason="completed", final_report_ref="report-1"
    )
    assert stopped.pending_terminal_reason is None
    assert stopped.pending_terminal_has_reduction is False


def test_unrestored_fields_raise_instead_of_defaulting(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    state = journal.rebuild()
    assert state is not None
    with pytest.raises(ExplorationStateUnavailableError):
        state.require_stop_reason()
    with pytest.raises(ExplorationStateUnavailableError):
        state.require_frontier_digest()
    with pytest.raises(ExplorationStateUnavailableError):
        state.require_ledger_digest()
    with pytest.raises(ExplorationStateUnavailableError):
        state.require_current_round_index()


def test_pending_operations_block_round_transitions_but_not_new_slots(
    tmp_path: Path,
) -> None:
    """Multi-slot machine: concurrent starts are legal, duplicates are not,
    and round settlement still requires full quiescence."""
    journal = _journal(tmp_path)
    journal.append_new("round_started", round_index=0)
    journal.append_new("llm_call_started", call_id="call-0")
    state = journal.append_new(
        "tool_call_started", logical_step_id="step-1", input_fingerprint="fp-1"
    )
    assert state.pending_call_ids == ("call-0",)
    assert set(state.pending_tool_steps) == {"step-1"}
    with pytest.raises(EventTransitionError, match="already pending"):
        journal.append_new("llm_call_started", call_id="call-0")
    with pytest.raises(EventTransitionError, match="already pending"):
        journal.append_new(
            "tool_call_started", logical_step_id="step-1", input_fingerprint="fp-1"
        )
    with pytest.raises(EventTransitionError, match="pending"):
        journal.append_new("round_settled", round_index=0, progress=False)


def test_round_settled_tracks_the_consecutive_empty_frontier_streak(
    tmp_path: Path,
) -> None:
    """An empty frontier is its own signal, counted separately from
    no-progress rounds, and any non-empty round clears the streak."""
    journal = _journal(tmp_path)
    journal.append_new("round_started", round_index=0)
    state = journal.append_new(
        "round_settled", round_index=0, progress=False, frontier_empty=True
    )
    assert state.consecutive_empty_frontier == 1
    assert state.consecutive_no_progress == 1

    journal.append_new("round_started", round_index=1)
    state = journal.append_new(
        "round_settled", round_index=1, progress=False, frontier_empty=True
    )
    assert state.consecutive_empty_frontier == 2

    journal.append_new("round_started", round_index=2)
    state = journal.append_new("round_settled", round_index=2, progress=False)
    assert state.consecutive_empty_frontier == 0
    assert state.consecutive_no_progress == 3


def test_round_settled_tracks_the_no_adjudication_streak(tmp_path: Path) -> None:
    """Plan-B soft stop: the streak counts rounds with zero adjudicated
    transitions, resets on any adjudication, and treats legacy events (no
    field) as movement so resumed old journals never soft-stop retroactively."""
    journal = _journal(tmp_path, policy=_policy(budget=_budget(max_rounds=9)))
    journal.append_new("round_started", round_index=0)
    state = journal.append_new(
        "round_settled", round_index=0, progress=True, adjudicated_transitions=0
    )
    assert state.consecutive_no_adjudication == 1

    # Legacy event without the field: counted as movement, streak resets.
    journal.append_new("round_started", round_index=1)
    state = journal.append_new("round_settled", round_index=1, progress=True)
    assert state.consecutive_no_adjudication == 0

    journal.append_new("round_started", round_index=2)
    state = journal.append_new(
        "round_settled", round_index=2, progress=True, adjudicated_transitions=0
    )
    assert state.consecutive_no_adjudication == 1

    journal.append_new("round_started", round_index=3)
    state = journal.append_new(
        "round_settled", round_index=3, progress=True, adjudicated_transitions=2
    )
    assert state.consecutive_no_adjudication == 0

    # And a resume rebuilds the same streak from disk.
    assert journal.rebuild().consecutive_no_adjudication == 0


def test_round_settled_carries_the_productivity_observations(tmp_path: Path) -> None:
    """Recorded, replayable, and optional: pre-existing journals have None and
    must still rebuild."""
    journal = _journal(tmp_path)
    journal.append_new("round_started", round_index=0)
    journal.append_new(
        "round_settled",
        round_index=0,
        progress=True,
        adjudicated_transitions=3,
        supported_transitions=2,
        llm_calls_at_settle=11,
        tool_calls_at_settle=7,
    )
    settled = [
        event for event in journal.events() if event.event_type == "round_settled"
    ]
    assert [
        (
            event.supported_transitions,
            event.llm_calls_at_settle,
            event.tool_calls_at_settle,
        )
        for event in settled
    ] == [(2, 11, 7)]

    # A legacy settle carries none of them and still replays.
    journal.append_new("round_started", round_index=1)
    journal.append_new("round_settled", round_index=1, progress=True)
    legacy = journal.events()[-1]
    assert legacy.supported_transitions is None
    assert legacy.llm_calls_at_settle is None
    assert journal.rebuild() is not None


def test_budget_counters_decrement_and_reject_at_zero(tmp_path: Path) -> None:
    policy = _policy(
        budget=_budget(
            llm=SessionBudgetPolicyModel(max_requests=1),
            max_successful_tool_calls=1,
            max_rounds=1,
        )
    )
    journal = _journal(tmp_path, policy=policy)
    journal.append_new("round_started", round_index=0)
    journal.append_new("llm_call_started", call_id="call-0")
    state = journal.append_new(
        "llm_call_completed", call_id="call-0", step_id="s-0", response_digest="r-0"
    )
    assert state.remaining_llm_call_budget == 0
    with pytest.raises(EventTransitionError, match="llm request cap"):
        journal.append_new("llm_call_started", call_id="call-1")

    _run_one_tool_step(journal, "step-0", "rcpt-0")
    with pytest.raises(EventTransitionError, match="max_successful_tool_calls"):
        journal.append_new(
            "tool_call_started", logical_step_id="step-1", input_fingerprint="fp-1"
        )

    journal.append_new("round_settled", round_index=0, progress=True)
    with pytest.raises(EventTransitionError, match="max_rounds"):
        journal.append_new("round_started", round_index=1)


def test_provider_rejections_consume_the_request_cap(tmp_path: Path) -> None:
    policy = _policy(
        budget=_budget(llm=SessionBudgetPolicyModel(max_requests=1))
    )
    journal = _journal(tmp_path, policy=policy)
    journal.append_new("round_started", round_index=0)
    journal.append_new("llm_call_started", call_id="call-rejected")
    state = journal.append_new(
        "llm_call_rejected", call_id="call-rejected", error="provider HTTP 429"
    )

    assert state.llm_calls_settled == 1
    assert state.remaining_llm_call_budget == 0
    with pytest.raises(EventTransitionError, match="llm request cap"):
        journal.append_new("llm_call_started", call_id="call-bypass")


def test_failed_tool_calls_do_not_consume_the_success_budget(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.append_new("round_started", round_index=0)
    journal.append_new(
        "tool_call_started", logical_step_id="step-0", input_fingerprint="fp-0"
    )
    state = journal.append_new(
        "tool_call_failed", logical_step_id="step-0", error="query timeout"
    )
    assert state.tool_calls_committed == 0
    assert state.remaining_tool_call_budget == 3
    assert state.pending_logical_step_id is None
    assert state.failure_history == ["query timeout"]


def test_tool_kind_and_resource_usage_are_journal_durable_and_restore_ledger(
    tmp_path: Path,
) -> None:
    policy = _policy()
    journal = _journal(tmp_path, policy=policy)
    journal.append_new("round_started", round_index=0)
    journal.append_new(
        "tool_call_started",
        logical_step_id="step-usage",
        input_fingerprint="probe-fp",
        tool_kind="run_open_analysis",
        projected_rows_scanned=100,
        projected_result_cells=20,
    )
    journal.append_new(
        "receipt_prepared",
        logical_step_id="step-usage",
        receipt_id="receipt-usage",
    )
    state = journal.append_new(
        "receipt_committed",
        logical_step_id="step-usage",
        receipt_id="receipt-usage",
        rows_scanned=80,
        result_cells=12,
    )

    assert state.tool_calls_by_kind == {"run_open_analysis": 1}
    assert state.rows_scanned == 80
    assert state.result_cells == 12
    assert state.completed_probe_fingerprints == ["probe-fp"]
    restored = ToolCallLedger.restore_from_journal_state(policy.budget, state)
    assert restored.snapshot() == {
        "successful_tool_calls": 1,
        "calls_by_kind": {"run_open_analysis": 1},
        "rows_scanned": 80,
        "result_cells": 12,
    }


def test_crashed_tool_charges_projection_or_adopts_durable_body(tmp_path: Path) -> None:
    charged = _journal(tmp_path / "charged")
    charged.append_new("round_started", round_index=0)
    charged.append_new(
        "tool_call_started",
        logical_step_id="step-crash",
        input_fingerprint="probe-crash",
        tool_kind="run_open_analysis",
        projected_rows_scanned=120,
        projected_result_cells=30,
    )
    charged_state = JsonlExplorationJournal(charged.path).claim_recovery()
    assert charged_state.rows_scanned == 120
    assert charged_state.result_cells == 30
    assert charged_state.tool_calls_committed == 0

    adopted = _journal(tmp_path / "adopted")
    adopted.append_new("round_started", round_index=0)
    adopted.append_new(
        "tool_call_started",
        logical_step_id="step-adopt",
        input_fingerprint="probe-adopt",
        tool_kind="run_open_analysis",
        projected_rows_scanned=120,
        projected_result_cells=30,
    )
    adopted_state = JsonlExplorationJournal(adopted.path).claim_recovery(
        completed_tool_result=lambda step_id: (
            RecoveredToolCommit(
                "receipt-adopt",
                result_digest="tool-result-digest",
                rows_scanned=70,
                result_cells=10,
            )
            if step_id == "step-adopt"
            else None
        )
    )
    assert adopted_state.step_receipt_refs == {"step-adopt": "receipt-adopt"}
    assert adopted_state.rows_scanned == 70
    assert adopted_state.result_cells == 10
    assert adopted_state.completed_probe_fingerprints == ["probe-adopt"]


# ---------------------------------------------------------- crash / recovery


def test_torn_tail_recovery_on_the_exploration_journal(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.append_new("round_started", round_index=0)
    before = journal.rebuild()

    with journal.path.open("ab") as handle:
        handle.write(b'{"seq":2,"exploration_id"')

    recovered = JsonlExplorationJournal(journal.path)
    assert recovered.rebuild() == before
    after = recovered.append_new("round_settled", round_index=0, progress=False)
    assert after.rounds_settled == 1

    lines = recovered.path.read_bytes().splitlines(keepends=True)
    recovered.path.write_bytes(lines[0] + b"{broken}\n" + b"".join(lines[1:]))
    with pytest.raises(EventJournalCorruptionError, match="Invalid committed record"):
        recovered.rebuild()


def test_two_executors_are_mutually_fenced_by_attempt_epoch(tmp_path: Path) -> None:
    first = _journal(tmp_path)
    second = JsonlExplorationJournal(first.path)
    assert first.claim_attempt().attempt_epoch == 1
    assert second.claim_attempt().attempt_epoch == 2

    with pytest.raises(EventTransitionError, match="stale exploration executor"):
        first.append_new("round_started", round_index=0)
    with pytest.raises(EventTransitionError, match="stale exploration executor"):
        with first.fenced_side_effect():
            raise AssertionError("stale executor entered the commit section")

    state = second.append_new("round_started", round_index=0)
    assert state.require_current_round_index() == 0


def test_crash_before_llm_return_marks_uncertain_and_consumes_reservation(
    tmp_path: Path,
) -> None:
    journal = _journal(tmp_path)
    journal.append_new("round_started", round_index=0)
    journal.append_new("llm_call_started", call_id="call-0")

    recovered = JsonlExplorationJournal(journal.path)
    state = recovered.claim_recovery()
    assert state.pending_call_id is None
    assert state.llm_calls_uncertain == 1
    assert state.llm_calls_settled == 0
    assert state.remaining_llm_call_budget == 3  # reservation fully consumed
    assert state.status == "running"
    state = recovered.append_new("round_settled", round_index=0, progress=False)
    assert state.rounds_settled == 1


def test_crash_after_response_body_adopts_bound_step_before_marking_uncertain(
    tmp_path: Path,
) -> None:
    journal = _journal(tmp_path)
    journal.append_new("round_started", round_index=0)
    journal.append_new(
        "llm_call_started",
        call_id="call-0",
        step_id="step-llm-0",
    )

    recovered = JsonlExplorationJournal(journal.path)
    state = recovered.claim_recovery(
        completed_response_digest=lambda step_id: (
            "response-digest-0" if step_id == "step-llm-0" else None
        )
    )

    assert state.pending_call_id is None
    assert state.pending_call_step_id is None
    assert state.llm_calls_settled == 1
    assert state.llm_calls_uncertain == 0
    assert "step-llm-0" in state.completed_step_ids
    assert [event.event_type for event in recovered.events()][-2:] == [
        "attempt_started",
        "llm_call_completed",
    ]


def test_bound_pending_call_rejects_completion_for_another_step(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.append_new("round_started", round_index=0)
    journal.append_new(
        "llm_call_started",
        call_id="call-0",
        step_id="step-llm-0",
    )
    with pytest.raises(EventTransitionError, match="pending call"):
        journal.append_new(
            "llm_call_completed",
            call_id="call-0",
            step_id="step-other",
            response_digest="response-other",
        )


def test_crash_after_llm_return_replays_without_resending(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.append_new("round_started", round_index=0)
    journal.append_new("llm_call_started", call_id="call-0")
    journal.append_new(
        "llm_call_completed", call_id="call-0", step_id="step-llm-0", response_digest="r-0"
    )

    recovered = JsonlExplorationJournal(journal.path)
    state = recovered.claim_recovery()
    assert state.llm_calls_uncertain == 0
    assert "step-llm-0" in state.completed_step_ids  # executor injects, not resends
    recovered.append_new("llm_call_started", call_id="call-1")
    with pytest.raises(EventTransitionError, match="already recorded"):
        recovered.append_new(
            "llm_call_completed",
            call_id="call-1",
            step_id="step-llm-0",
            response_digest="r-1",
        )


def test_crash_before_receipt_commit_allows_logical_rerun_exactly_once(
    tmp_path: Path,
) -> None:
    journal = _journal(tmp_path)
    journal.append_new("round_started", round_index=0)
    journal.append_new(
        "tool_call_started", logical_step_id="step-corr", input_fingerprint="fp-1"
    )

    recovered = JsonlExplorationJournal(journal.path)
    state = recovered.claim_recovery()
    assert state.pending_logical_step_id is None
    assert state.failure_history[-1].startswith("tool outcome unknown after crash")
    recovered.append_new(
        "tool_call_started", logical_step_id="step-corr", input_fingerprint="fp-1"
    )
    recovered.append_new(
        "receipt_prepared", logical_step_id="step-corr", receipt_id="rcpt-1"
    )
    state = recovered.append_new(
        "receipt_committed", logical_step_id="step-corr", receipt_id="rcpt-1"
    )
    assert state.step_receipt_refs == {"step-corr": "rcpt-1"}
    assert state.tool_calls_committed == 1


def test_crash_after_receipt_commit_adopts_the_committed_receipt(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.append_new("round_started", round_index=0)
    _run_one_tool_step(journal, "step-corr", "rcpt-1")

    recovered = JsonlExplorationJournal(journal.path)
    state = recovered.claim_recovery()
    assert state.step_receipt_refs == {"step-corr": "rcpt-1"}
    with pytest.raises(EventTransitionError, match="already committed"):
        recovered.append_new(
            "tool_call_started", logical_step_id="step-corr", input_fingerprint="fp-1"
        )


def test_prepared_receipt_cannot_be_replaced_or_mismatched(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.append_new("round_started", round_index=0)
    journal.append_new(
        "tool_call_started", logical_step_id="step-0", input_fingerprint="fp-0"
    )
    journal.append_new("receipt_prepared", logical_step_id="step-0", receipt_id="rcpt-a")
    # Idempotent re-prepare of the same receipt is allowed (crash between the
    # prepared event and the outbox write).
    journal.append_new("receipt_prepared", logical_step_id="step-0", receipt_id="rcpt-a")
    with pytest.raises(EventTransitionError, match="cannot be replaced"):
        journal.append_new(
            "receipt_prepared", logical_step_id="step-0", receipt_id="rcpt-b"
        )
    with pytest.raises(EventTransitionError, match="does not match the prepared receipt"):
        journal.append_new(
            "receipt_committed", logical_step_id="step-0", receipt_id="rcpt-b"
        )


def test_receipt_commit_requires_a_prepared_receipt(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.append_new("round_started", round_index=0)
    journal.append_new(
        "tool_call_started", logical_step_id="step-0", input_fingerprint="fp-0"
    )
    with pytest.raises(EventTransitionError, match="prepared"):
        journal.append_new(
            "receipt_committed", logical_step_id="step-0", receipt_id="rcpt-a"
        )


# -------------------------------------------------------------- pause / resume


def test_pause_resume_cycle_never_emits_a_stop_event(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.append_new("round_started", round_index=0)
    state = journal.append_new("pause_requested")
    assert state.status == "pause_requested"
    state = journal.append_new("paused")
    assert state.status == "paused"
    state = journal.append_new("resumed")
    assert state.status == "running"

    assert "exploration_stopped" not in {event.event_type for event in journal.events()}
    state = journal.append_new("round_settled", round_index=0, progress=True)
    assert state.rounds_settled == 1


def test_pause_requested_drains_in_flight_work_but_blocks_new_work(
    tmp_path: Path,
) -> None:
    journal = _journal(tmp_path)
    journal.append_new("round_started", round_index=0)
    journal.append_new("llm_call_started", call_id="call-0")
    journal.append_new("pause_requested")

    with pytest.raises(EventTransitionError, match="pause"):
        journal.append_new("llm_call_started", call_id="call-1")
    # In-flight settlement still lands while draining.
    state = journal.append_new(
        "llm_call_completed", call_id="call-0", step_id="s-0", response_digest="r-0"
    )
    assert state.llm_calls_settled == 1

    with pytest.raises(EventTransitionError, match="pause"):
        journal.append_new(
            "tool_call_started", logical_step_id="step-0", input_fingerprint="fp-0"
        )


def test_paused_requires_quiescence_and_blocks_new_work(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.append_new("round_started", round_index=0)
    journal.append_new("llm_call_started", call_id="call-0")
    journal.append_new("pause_requested")
    with pytest.raises(EventTransitionError, match="pending"):
        journal.append_new("paused")
    journal.append_new(
        "llm_call_completed", call_id="call-0", step_id="s-0", response_digest="r-0"
    )
    journal.append_new("paused")
    with pytest.raises(EventTransitionError):
        journal.append_new("round_started", round_index=1)
    with pytest.raises(EventTransitionError):
        journal.append_new("llm_call_started", call_id="call-1")


def test_resume_is_only_valid_from_paused(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    with pytest.raises(EventTransitionError):
        journal.append_new("resumed")
    journal.append_new("pause_requested")
    with pytest.raises(EventTransitionError):
        journal.append_new("resumed")


def test_cancel_is_allowed_while_paused_but_completed_is_not(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.append_new("pause_requested")
    journal.append_new("paused")
    with pytest.raises(EventTransitionError):
        journal.append_new("exploration_stopped", stop_reason="completed")
    state = journal.append_new("exploration_stopped", stop_reason="cancelled")
    assert state.status == "stopped"
    assert state.require_stop_reason() == "cancelled"


# ------------------------------------------------------------------- terminal


def test_stopped_is_terminal_for_every_event_type(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.append_new("exploration_stopped", stop_reason="cancelled")
    for attempt in (
        lambda: journal.append_new("round_started", round_index=0),
        lambda: journal.append_new("pause_requested"),
        lambda: journal.append_new("resumed"),
        lambda: journal.claim_attempt(),
        lambda: journal.append_new(
            "budget_amended",
            amendment_id="amend-1",
            effective_policy_fingerprint="xplcy_next",
            increase=BudgetCapIncrease(max_rounds=1),
        ),
    ):
        with pytest.raises(EventTransitionError, match="stopped"):
            attempt()


def test_natural_endings_require_quiescence_but_aborts_do_not(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.append_new("round_started", round_index=0)
    journal.append_new("llm_call_started", call_id="call-0")
    with pytest.raises(EventTransitionError, match="pending"):
        journal.append_new("exploration_stopped", stop_reason="completed")
    state = journal.append_new(
        "exploration_stopped", stop_reason="state_witness_changed"
    )
    assert state.status == "stopped"
    assert state.require_stop_reason() == "state_witness_changed"
    assert state.llm_calls_uncertain == 1
    assert state.uncertain_call_ids == ["call-0"]
    assert state.remaining_llm_call_budget == 3


def test_gate_and_reduction_cannot_commit_around_pending_work(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.append_new("round_started", round_index=0)
    journal.append_new(
        "tool_call_started",
        logical_step_id="pending-tool",
        input_fingerprint="pending-fp",
    )
    with pytest.raises(EventTransitionError, match="pending"):
        journal.append_new(
            "gate_verdict", claim_bundle_id="claim-1", verdict="passed"
        )
    with pytest.raises(EventTransitionError, match="pending"):
        journal.append_new(
            "reduction_committed",
            frontier_digest="frontier-1",
            ledger_digest="ledger-1",
        )


def test_forged_seq_and_epoch_rollback_are_rejected(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.claim_attempt()
    state = journal.rebuild()
    assert state is not None

    with pytest.raises(EventTransitionError, match="seq must be"):
        journal.append(
            RoundStartedEvent(
                seq=99, exploration_id="xpl_test", attempt_epoch=1, round_index=0
            )
        )
    with pytest.raises(EventTransitionError, match="seq must be"):
        journal.append(
            RoundStartedEvent(
                seq=state.last_seq, exploration_id="xpl_test", attempt_epoch=1, round_index=0
            )
        )
    with pytest.raises(EventTransitionError, match="attempt_epoch must be"):
        journal.append(
            LlmCallStartedEvent(
                seq=state.last_seq + 1,
                exploration_id="xpl_test",
                attempt_epoch=0,
                call_id="call-forged",
            )
        )


# ------------------------------------------------------------------ amendments


def test_budget_amendment_extends_caps_from_the_journal(tmp_path: Path) -> None:
    policy = _policy(budget=_budget(max_rounds=1))
    journal = _journal(tmp_path, policy=policy)
    journal.append_new("round_started", round_index=0)
    journal.append_new("round_settled", round_index=0, progress=True)
    with pytest.raises(EventTransitionError, match="max_rounds"):
        journal.append_new("round_started", round_index=1)

    increase = BudgetCapIncrease(max_rounds=2, max_successful_tool_calls=1)
    state = journal.amend_budget(
        amendment_id="amend-1",
        increase=increase,
    )
    assert state.max_rounds == 3
    assert state.remaining_round_budget == 2
    assert state.max_successful_tool_calls == 4
    assert state.effective_policy_fingerprint == amended_policy_fingerprint(
        policy.policy_fingerprint, "amend-1", increase
    )
    assert state.policy_fingerprint == policy.policy_fingerprint  # base is frozen
    state = journal.append_new("round_started", round_index=1)
    assert state.require_current_round_index() == 1

    with pytest.raises(EventTransitionError, match="already applied"):
        journal.append_new(
            "budget_amended",
            amendment_id="amend-1",
            effective_policy_fingerprint="xplcy_amended-2",
            increase=BudgetCapIncrease(max_rounds=1),
        )


def test_budget_amendment_is_allowed_while_paused(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.append_new("pause_requested")
    journal.append_new("paused")
    state = journal.amend_budget(
        amendment_id="amend-1",
        increase=BudgetCapIncrease(max_requests=2),
    )
    assert state.status == "paused"
    assert state.max_llm_requests == 6
    assert state.remaining_llm_call_budget == 6


def test_amendments_cannot_lower_caps_by_construction() -> None:
    with pytest.raises(ValueError):
        BudgetCapIncrease(max_rounds=-1)
    with pytest.raises(ValueError, match="at least one cap"):
        BudgetCapIncrease()


def test_budget_amendment_rejects_a_caller_forged_fingerprint(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    with pytest.raises(EventTransitionError, match="effective_policy_fingerprint"):
        journal.append_new(
            "budget_amended",
            amendment_id="amend-forged",
            effective_policy_fingerprint="xplcy_caller_chosen",
            increase=BudgetCapIncrease(max_rounds=1),
        )


# -------------------------------------------------------------------- resume


def test_initialize_is_idempotent_and_fails_closed_on_identity_drift(
    tmp_path: Path,
) -> None:
    policy = _policy()
    journal = _journal(tmp_path, policy=policy)
    journal.append_new("round_started", round_index=0)

    resumed = JsonlExplorationJournal(journal.path)
    state = resumed.initialize(
        exploration_id="xpl_test",
        policy=policy,
        code_fingerprint="code-v1",
        data_state_witness="dsw1_test",
    )
    assert state.require_current_round_index() == 0

    with pytest.raises(ExplorationResumeIncompatibleError, match="data_state_witness"):
        resumed.initialize(
            exploration_id="xpl_test",
            policy=policy,
            code_fingerprint="code-v1",
            data_state_witness="dsw1_other",
        )
    with pytest.raises(ExplorationResumeIncompatibleError, match="policy_fingerprint"):
        resumed.initialize(
            exploration_id="xpl_test",
            policy=_policy(thinking_level="deep"),
            code_fingerprint="code-v1",
            data_state_witness="dsw1_test",
        )


# ------------------------------------------------------------- E6 branching


def _branch_budget(**overrides: object) -> ExplorationBudgetPolicy:
    fields: dict[str, object] = {
        "branching": ExplorationBranchPolicy(
            trigger_stagnant_rounds=2, max_branches=2
        ),
        "max_rounds": 8,
    }
    fields.update(overrides)
    return _budget(**fields)


def _branch_journal(
    tmp_path: Path, **budget_overrides: object
) -> JsonlExplorationJournal:
    return _journal(
        tmp_path, policy=_policy(budget=_branch_budget(**budget_overrides))
    )


def _settle_no_progress_round(
    journal: JsonlExplorationJournal, round_index: int, *, branch_id: str | None = None
) -> None:
    journal.append_new("round_started", round_index=round_index, branch_id=branch_id)
    journal.append_new("round_settled", round_index=round_index, progress=False)


def test_branch_abandoned_requires_the_system_stagnation_signal(
    tmp_path: Path,
) -> None:
    """A provider (or buggy driver) cannot declare branch mode: the reducer
    rejects abandonment unless consecutive no-progress rounds reached the
    policy trigger."""
    journal = _branch_journal(tmp_path)
    _settle_no_progress_round(journal, 0)
    with pytest.raises(EventTransitionError, match="stagnation"):
        journal.append_new(
            "branch_abandoned", branch_id="main", round_index=0, constraints=()
        )


def test_branch_abandonment_switches_line_and_resets_stagnation(
    tmp_path: Path,
) -> None:
    journal = _branch_journal(tmp_path)
    _settle_no_progress_round(journal, 0)
    _settle_no_progress_round(journal, 1)
    state = journal.append_new(
        "branch_abandoned", branch_id="main", round_index=1, constraints=()
    )
    assert state.consecutive_no_progress == 0
    assert state.abandoned_line_ids == ["main"]

    # The abandoned trunk cannot be continued; the next round must open br_1.
    with pytest.raises(EventTransitionError, match="branch"):
        journal.append_new("round_started", round_index=2)
    with pytest.raises(EventTransitionError, match="br_1"):
        journal.append_new("round_started", round_index=2, branch_id="br_2")
    state = journal.append_new("round_started", round_index=2, branch_id="br_1")
    assert state.active_branch_id == "br_1"
    journal.append_new("round_settled", round_index=2, progress=False)
    # An open branch keeps its id on every subsequent round.
    with pytest.raises(EventTransitionError, match="active branch"):
        journal.append_new("round_started", round_index=3)


def test_branch_events_are_rejected_when_branching_is_disabled(
    tmp_path: Path,
) -> None:
    journal = _journal(tmp_path)
    _settle_no_progress_round(journal, 0)
    _settle_no_progress_round(journal, 1)
    with pytest.raises(EventTransitionError, match="disabled"):
        journal.append_new(
            "branch_abandoned", branch_id="main", round_index=1, constraints=()
        )
    with pytest.raises(EventTransitionError, match="disabled"):
        journal.append_new("round_started", round_index=2, branch_id="br_1")


def test_no_new_information_requires_branch_exhaustion(tmp_path: Path) -> None:
    journal = _branch_journal(tmp_path, branching=ExplorationBranchPolicy(
        trigger_stagnant_rounds=2, max_branches=1
    ))
    _settle_no_progress_round(journal, 0)
    journal.append_new("round_started", round_index=1)
    with pytest.raises(EventTransitionError, match="exhaust"):
        journal.append_new(
            "round_settled",
            round_index=1,
            progress=False,
            terminal_reason="no_new_information",
        )
    journal.append_new("round_settled", round_index=1, progress=False)
    journal.append_new(
        "branch_abandoned", branch_id="main", round_index=1, constraints=()
    )
    _settle_no_progress_round(journal, 2, branch_id="br_1")
    journal.append_new("round_started", round_index=3, branch_id="br_1")
    # br_1 is the last allowed branch: terminating is now legal.
    state = journal.append_new(
        "round_settled",
        round_index=3,
        progress=False,
        terminal_reason="no_new_information",
    )
    assert state.pending_terminal_reason == "no_new_information"


def test_branch_budget_and_line_identity_fail_closed(tmp_path: Path) -> None:
    journal = _branch_journal(tmp_path, branching=ExplorationBranchPolicy(
        trigger_stagnant_rounds=2, max_branches=1
    ))
    _settle_no_progress_round(journal, 0)
    _settle_no_progress_round(journal, 1)
    with pytest.raises(EventTransitionError, match="round_index"):
        journal.append_new(
            "branch_abandoned", branch_id="main", round_index=0, constraints=()
        )
    with pytest.raises(EventTransitionError, match="line"):
        journal.append_new(
            "branch_abandoned", branch_id="br_1", round_index=1, constraints=()
        )
    journal.append_new(
        "branch_abandoned", branch_id="main", round_index=1, constraints=()
    )
    with pytest.raises(EventTransitionError, match="abandoned"):
        journal.append_new(
            "branch_abandoned", branch_id="main", round_index=1, constraints=()
        )
    _settle_no_progress_round(journal, 2, branch_id="br_1")
    _settle_no_progress_round(journal, 3, branch_id="br_1")
    # max_branches=1: abandoning br_1 would leave no successor to branch into.
    with pytest.raises(EventTransitionError, match="budget"):
        journal.append_new(
            "branch_abandoned", branch_id="br_1", round_index=3, constraints=()
        )


def test_branch_id_schema_caps_depth_at_two(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        RoundStartedEvent(
            seq=1, exploration_id="xpl_test", round_index=0, branch_id="br_1.1.1"
        )
    with pytest.raises(ValueError):
        RoundStartedEvent(
            seq=1, exploration_id="xpl_test", round_index=0, branch_id="main"
        )
    event = RoundStartedEvent(
        seq=1, exploration_id="xpl_test", round_index=0, branch_id="br_1.1"
    )
    assert event.branch_id == "br_1.1"


def test_branch_constraints_accumulate_in_state(tmp_path: Path) -> None:
    journal = _branch_journal(tmp_path)
    _settle_no_progress_round(journal, 0)
    _settle_no_progress_round(journal, 1)
    constraint = BranchConstraint(
        hypothesis_fingerprint="a" * 16,
        coverage_key="cov_a",
        family=InsightFamily.DIAGNOSTIC,
        reason="refuted",
        detail_code="rcpt_1",
    )
    state = journal.append_new(
        "branch_abandoned", branch_id="main", round_index=1, constraints=(constraint,)
    )
    assert state.abandoned_constraints == [constraint]
    rebuilt = rebuild_exploration_state(journal.events())
    assert rebuilt is not None
    assert rebuilt.abandoned_constraints == [constraint]
