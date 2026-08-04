"""Multi-slot pending state: interleaved tool/LLM sessions, in-flight budget
admission, crash recovery over every slot, and legacy snapshot compatibility."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eda_platform.core.event_journal import EventTransitionError
from eda_platform.core.exploration_journal import (
    JsonlExplorationJournal,
    RecoveredToolCommit,
    sealed_policy,
)
from eda_platform.schemas.exploration import (
    ExplorationLoopState,
    ExplorationPolicy,
    InsightFamily,
)
from eda_platform.schemas.exploration_budget import (
    ExplorationBudgetPolicy,
    SessionBudgetPolicyModel,
)

LEGACY_SCALAR_FIELDS = (
    "pending_call_id",
    "pending_call_step_id",
    "pending_logical_step_id",
    "prepared_receipt_id",
    "prepared_result_digest",
    "pending_tool_kind",
    "pending_tool_input_fingerprint",
)
MULTI_SLOT_FIELDS = ("pending_tool_steps", "pending_call_ids", "pending_call_steps")


def _budget(**overrides: object) -> ExplorationBudgetPolicy:
    fields: dict[str, object] = {
        "llm": SessionBudgetPolicyModel(max_requests=6),
        "max_successful_tool_calls": 4,
        "max_tool_calls_by_kind": {"run_open_analysis": 4},
        "idle_timeout_seconds": 30.0,
        "max_rounds": 3,
    }
    fields.update(overrides)
    return ExplorationBudgetPolicy.model_validate(fields)


def _policy(**budget_overrides: object) -> ExplorationPolicy:
    return sealed_policy(
        ExplorationPolicy.model_validate(
            {
                "mode": "open",
                "goal": "multislot goal",
                "dataset_scope": ("ds_a",),
                "thinking_level": "standard",
                "coverage_targets": (InsightFamily.DESCRIPTIVE,),
                "budget": _budget(**budget_overrides),
                "scoring_policy_version": "score-v1",
                "statistical_policy_version": "stats-v1",
                "tool_capability_digest": "tools-v1",
            }
        )
    )


def _journal(tmp_path: Path, **budget_overrides: object) -> JsonlExplorationJournal:
    journal = JsonlExplorationJournal(tmp_path / "exploration" / "journal.jsonl")
    journal.initialize(
        exploration_id="xpl_multislot",
        policy=_policy(**budget_overrides),
        code_fingerprint="code-v1",
        data_state_witness="dsw1_test",
    )
    return journal


def _start_tool(journal: JsonlExplorationJournal, step: str) -> ExplorationLoopState:
    return journal.append_new(
        "tool_call_started",
        logical_step_id=step,
        input_fingerprint=f"fp-{step}",
        tool_kind="run_open_analysis",
        projected_rows_scanned=10,
        projected_result_cells=5,
    )


# ------------------------------------------------------------- interleaving


def test_two_tool_sessions_interleave_and_both_commit(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.append_new("round_started", round_index=0)
    _start_tool(journal, "step-a")
    _start_tool(journal, "step-b")
    journal.append_new("receipt_prepared", logical_step_id="step-b", receipt_id="rcpt-b")
    journal.append_new("receipt_committed", logical_step_id="step-b", receipt_id="rcpt-b")
    journal.append_new("receipt_prepared", logical_step_id="step-a", receipt_id="rcpt-a")
    state = journal.append_new(
        "receipt_committed", logical_step_id="step-a", receipt_id="rcpt-a"
    )

    assert state.step_receipt_refs == {"step-a": "rcpt-a", "step-b": "rcpt-b"}
    assert state.tool_calls_committed == 2
    assert sorted(state.completed_probe_fingerprints) == ["fp-step-a", "fp-step-b"]
    # A clean replay from disk reproduces the same state.
    assert JsonlExplorationJournal(journal.path).rebuild() == state


def test_two_llm_calls_interleave_and_both_settle(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.append_new("round_started", round_index=0)
    journal.append_new("llm_call_started", call_id="call-a", step_id="step-llm-a")
    journal.append_new("llm_call_started", call_id="call-b", step_id="step-llm-b")
    journal.append_new(
        "llm_call_completed",
        call_id="call-b",
        step_id="step-llm-b",
        response_digest="resp-b",
    )
    state = journal.append_new(
        "llm_call_completed",
        call_id="call-a",
        step_id="step-llm-a",
        response_digest="resp-a",
    )

    assert state.llm_calls_settled == 2
    assert state.pending_call_ids == ()
    assert {"step-llm-a", "step-llm-b"} <= set(state.completed_step_ids)
    assert JsonlExplorationJournal(journal.path).rebuild() == state


def test_tool_and_llm_slots_may_be_pending_at_the_same_time(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.append_new("round_started", round_index=0)
    journal.append_new("llm_call_started", call_id="call-a", step_id="step-llm-a")
    state = _start_tool(journal, "step-tool-a")
    assert state.pending_call_ids == ("call-a",)
    assert set(state.pending_tool_steps) == {"step-tool-a"}


def test_settled_slot_lookup_is_per_step_not_last_started(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.append_new("round_started", round_index=0)
    _start_tool(journal, "step-a")
    _start_tool(journal, "step-b")
    journal.append_new("receipt_prepared", logical_step_id="step-a", receipt_id="rcpt-a")
    with pytest.raises(EventTransitionError, match="does not match the prepared"):
        journal.append_new(
            "receipt_committed", logical_step_id="step-a", receipt_id="rcpt-b"
        )
    # step-b never prepared: committing it must fail independently of step-a.
    with pytest.raises(EventTransitionError, match="prepared"):
        journal.append_new(
            "receipt_committed", logical_step_id="step-b", receipt_id="rcpt-b"
        )


def test_duplicate_pending_starts_are_rejected(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.append_new("round_started", round_index=0)
    journal.append_new("llm_call_started", call_id="call-a")
    with pytest.raises(EventTransitionError, match="already pending"):
        journal.append_new("llm_call_started", call_id="call-a")
    _start_tool(journal, "step-a")
    with pytest.raises(EventTransitionError, match="already pending"):
        _start_tool(journal, "step-a")


def test_settling_an_unknown_slot_is_rejected(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.append_new("round_started", round_index=0)
    journal.append_new("llm_call_started", call_id="call-a")
    _start_tool(journal, "step-a")
    with pytest.raises(EventTransitionError, match="pending call"):
        journal.append_new(
            "llm_call_completed",
            call_id="call-other",
            step_id="s-x",
            response_digest="r-x",
        )
    with pytest.raises(EventTransitionError, match="pending tool step"):
        journal.append_new(
            "receipt_prepared", logical_step_id="step-other", receipt_id="rcpt-x"
        )
    with pytest.raises(EventTransitionError, match="pending tool step"):
        journal.append_new(
            "tool_call_failed", logical_step_id="step-other", error="boom"
        )


# ------------------------------------------------------- in-flight admission


def test_in_flight_tool_slots_count_against_the_success_budget(tmp_path: Path) -> None:
    journal = _journal(
        tmp_path,
        max_successful_tool_calls=2,
        max_tool_calls_by_kind={"run_open_analysis": 2},
    )
    journal.append_new("round_started", round_index=0)
    _start_tool(journal, "step-a")
    _start_tool(journal, "step-b")
    # Budget 2 with two slots in flight: a third start would over-admit.
    with pytest.raises(EventTransitionError, match="max_successful_tool_calls"):
        _start_tool(journal, "step-c")


def test_in_flight_llm_calls_count_against_the_request_cap(tmp_path: Path) -> None:
    journal = _journal(tmp_path, llm=SessionBudgetPolicyModel(max_requests=2))
    journal.append_new("round_started", round_index=0)
    journal.append_new("llm_call_started", call_id="call-a")
    journal.append_new("llm_call_started", call_id="call-b")
    with pytest.raises(EventTransitionError, match="llm request cap"):
        journal.append_new("llm_call_started", call_id="call-c")


def test_round_settle_requires_every_slot_drained(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.append_new("round_started", round_index=0)
    journal.append_new("llm_call_started", call_id="call-a")
    _start_tool(journal, "step-a")
    _start_tool(journal, "step-b")
    journal.append_new("receipt_prepared", logical_step_id="step-a", receipt_id="rcpt-a")
    journal.append_new("receipt_committed", logical_step_id="step-a", receipt_id="rcpt-a")
    # One tool slot and one call still pending.
    with pytest.raises(EventTransitionError, match="pending"):
        journal.append_new("round_settled", round_index=0, progress=True)
    journal.append_new(
        "llm_call_completed", call_id="call-a", step_id="s-a", response_digest="r-a"
    )
    with pytest.raises(EventTransitionError, match="pending"):
        journal.append_new("round_settled", round_index=0, progress=True)
    journal.append_new("tool_call_failed", logical_step_id="step-b", error="boom")
    state = journal.append_new("round_settled", round_index=0, progress=True)
    assert state.rounds_settled == 1


# --------------------------------------------------------------- recovery


def test_crash_with_two_in_flight_llm_calls_marks_both_uncertain(
    tmp_path: Path,
) -> None:
    journal = _journal(tmp_path)
    journal.append_new("round_started", round_index=0)
    journal.append_new("llm_call_started", call_id="call-a")
    journal.append_new("llm_call_started", call_id="call-b")

    state = JsonlExplorationJournal(journal.path).claim_recovery()
    assert state.pending_call_ids == ()
    assert state.llm_calls_uncertain == 2
    assert sorted(state.uncertain_call_ids) == ["call-a", "call-b"]
    assert state.remaining_llm_call_budget == 4


def test_crash_recovery_settles_every_pending_tool_slot(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.append_new("round_started", round_index=0)
    _start_tool(journal, "step-a")
    _start_tool(journal, "step-b")

    state = JsonlExplorationJournal(journal.path).claim_recovery(
        completed_tool_result=lambda step_id: (
            RecoveredToolCommit(
                "rcpt-a", result_digest="digest-a", rows_scanned=7, result_cells=3
            )
            if step_id == "step-a"
            else None
        )
    )
    assert state.pending_tool_steps == {}
    # step-a adopted its durable body; step-b charged its projection and failed.
    assert state.step_receipt_refs == {"step-a": "rcpt-a"}
    assert state.rows_scanned == 7 + 10
    assert state.result_cells == 3 + 5
    assert state.failure_history[-1].startswith("tool outcome unknown after crash")


def test_crash_recovery_adopts_each_bound_llm_step_individually(
    tmp_path: Path,
) -> None:
    journal = _journal(tmp_path)
    journal.append_new("round_started", round_index=0)
    journal.append_new("llm_call_started", call_id="call-a", step_id="step-llm-a")
    journal.append_new("llm_call_started", call_id="call-b", step_id="step-llm-b")

    state = JsonlExplorationJournal(journal.path).claim_recovery(
        completed_response_digest=lambda step_id: (
            "digest-a" if step_id == "step-llm-a" else None
        )
    )
    assert state.llm_calls_settled == 1
    assert state.llm_calls_uncertain == 1
    assert "step-llm-a" in state.completed_step_ids
    assert state.uncertain_call_ids == ["call-b"]


def test_abort_stop_marks_all_pending_calls_uncertain_and_charges_all_slots(
    tmp_path: Path,
) -> None:
    journal = _journal(tmp_path)
    journal.append_new("round_started", round_index=0)
    journal.append_new("llm_call_started", call_id="call-a")
    journal.append_new("llm_call_started", call_id="call-b")
    _start_tool(journal, "step-a")
    _start_tool(journal, "step-b")

    state = journal.append_new(
        "exploration_stopped", stop_reason="state_witness_changed"
    )
    assert state.llm_calls_uncertain == 2
    assert sorted(state.uncertain_call_ids) == ["call-a", "call-b"]
    assert state.remaining_llm_call_budget == 4
    assert state.rows_scanned == 20  # both projections charged
    assert state.result_cells == 10
    assert state.pending_tool_steps == {}
    assert state.pending_call_ids == ()


# ----------------------------------------------------- legacy compatibility


def test_new_reducer_always_keeps_legacy_scalar_fields_none(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.append_new("round_started", round_index=0)
    journal.append_new("llm_call_started", call_id="call-a")
    state = _start_tool(journal, "step-a")
    for name in LEGACY_SCALAR_FIELDS:
        assert getattr(state, name) is None, name
    assert state.pending_projected_rows_scanned == 0
    assert state.pending_projected_result_cells == 0


def test_old_root_snapshot_without_multislot_fields_equals_fresh_replay(
    tmp_path: Path,
) -> None:
    """An old (single-slot) reducer's terminal snapshot has legacy scalars at
    None and no multi-slot keys at all. Reloading it must compare equal to a
    fresh rebuild by the new reducer."""
    journal = _journal(tmp_path)
    journal.append_new("round_started", round_index=0)
    journal.append_new("llm_call_started", call_id="call-a")
    journal.append_new(
        "llm_call_completed", call_id="call-a", step_id="s-a", response_digest="r-a"
    )
    _start_tool(journal, "step-a")
    journal.append_new("receipt_prepared", logical_step_id="step-a", receipt_id="rcpt-a")
    journal.append_new("receipt_committed", logical_step_id="step-a", receipt_id="rcpt-a")
    journal.append_new("round_settled", round_index=0, progress=True)
    journal.append_new(
        "exploration_stopped", stop_reason="cancelled", final_report_ref=None
    )
    journal.write_snapshot()

    raw = json.loads(journal.snapshot_path.read_text(encoding="utf-8"))
    for key in MULTI_SLOT_FIELDS:
        raw.pop(key, None)  # an old snapshot never wrote these keys
    for key in LEGACY_SCALAR_FIELDS:
        assert key in raw and raw[key] is None  # and always wrote these as null
    journal.snapshot_path.write_text(json.dumps(raw), encoding="utf-8")

    reloaded = journal.read_snapshot()
    assert reloaded is not None
    assert reloaded == journal.rebuild()


def test_legacy_mid_run_snapshot_migrates_scalars_into_slots(tmp_path: Path) -> None:
    """An old snapshot taken with a pending call/tool step maps its scalar
    fields into one slot each (plan §P2: single value reads as a 1-tuple)."""
    journal = _journal(tmp_path)
    journal.append_new("round_started", round_index=0)
    journal.append_new("llm_call_started", call_id="call-a", step_id="step-llm-a")
    modern = _start_tool(journal, "step-a")

    raw = modern.model_dump(mode="json")
    for key in MULTI_SLOT_FIELDS:
        raw.pop(key, None)
    raw.update(
        {
            "pending_call_id": "call-a",
            "pending_call_step_id": "step-llm-a",
            "pending_logical_step_id": "step-a",
            "pending_tool_kind": "run_open_analysis",
            "pending_tool_input_fingerprint": "fp-step-a",
            "pending_projected_rows_scanned": 10,
            "pending_projected_result_cells": 5,
            "prepared_receipt_id": None,
            "prepared_result_digest": None,
        }
    )

    migrated = ExplorationLoopState.model_validate(raw)
    assert migrated == modern
    for name in LEGACY_SCALAR_FIELDS:
        assert getattr(migrated, name) is None, name
