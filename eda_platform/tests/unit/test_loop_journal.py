from __future__ import annotations

from pathlib import Path

import pytest

from eda_platform.core.loop_journal import (
    JsonlLoopJournal,
    LoopJournalCorruptionError,
    LoopResumeIncompatibleError,
    LoopTransitionError,
    assert_resume_compatible,
    make_investigation_id,
    make_loop_call_id,
    make_loop_probe_id,
    make_loop_step_id,
    rebuild_loop_state,
)
from eda_platform.schemas.deep_investigation import InvestigationLoopEvent


def _journal(tmp_path: Path, *, max_steps: int = 2, llm_call_cap: int = 3) -> JsonlLoopJournal:
    journal = JsonlLoopJournal(tmp_path / "investigation" / "loop.journal.jsonl")
    journal.initialize(
        investigation_id="inv_test",
        source_session_id="run_test",
        question_id="question_test",
        plan_fingerprint="plan-v1",
        policy_fingerprint="policy-v1",
        code_fingerprint="code-v1",
        max_steps=max_steps,
        llm_call_cap=llm_call_cap,
    )
    return journal


def _complete_iteration(journal: JsonlLoopJournal, iteration: int) -> None:
    call_id = make_loop_call_id("inv_test", iteration)
    journal.append_new(
        "decision_call_started",
        iteration=iteration,
        call_id=call_id,
    )
    journal.append_new(
        "decision_call_completed",
        iteration=iteration,
        call_id=call_id,
        step_id=make_loop_step_id("decision", call_id),
        response_hash=f"response-{iteration}",
        typed_decision={"action": "probe", "purpose": "validate segment"},
    )
    probe_fingerprint = f"probe-fingerprint-{iteration}"
    probe_id = make_loop_probe_id("inv_test", iteration, probe_fingerprint)
    journal.append_new(
        "probe_started",
        iteration=iteration,
        probe_id=probe_id,
        probe_fingerprint=probe_fingerprint,
    )
    journal.append_new(
        "artifact_committed",
        iteration=iteration,
        probe_id=probe_id,
        artifact_ref=f"artifact-{iteration}",
    )
    journal.append_new(
        "probe_completed",
        iteration=iteration,
        probe_id=probe_id,
        step_id=make_loop_step_id("probe", probe_id),
    )


def test_journal_rebuilds_identical_state_from_every_complete_prefix(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    claimed = journal.claim_attempt()
    assert claimed.attempt_epoch == 1
    _complete_iteration(journal, 0)
    final = journal.append_new("loop_concluded", final_draft_ref="draft-1")

    events = journal.events()
    for size in range(1, len(events) + 1):
        prefix_state = rebuild_loop_state(events[:size])
        sequential_state = None
        for event in events[:size]:
            sequential_state = rebuild_loop_state([*events[: event.seq], event])
        assert prefix_state == sequential_state

    rebuilt = journal.rebuild()
    assert rebuilt == final
    assert rebuilt is not None
    assert rebuilt.status == "concluded"
    assert rebuilt.probes_completed == 1
    assert rebuilt.llm_calls_settled == 1
    assert rebuilt.remaining_probe_budget == 1
    assert rebuilt.remaining_call_budget == 2
    assert rebuilt.next_iteration == 1
    assert len(rebuilt.completed_step_ids) == 2
    assert rebuilt.step_artifact_refs


def test_append_rejects_sequence_gaps_and_stale_attempt_epoch(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.claim_attempt()
    line_count = len(journal.events())

    with pytest.raises(LoopTransitionError, match="seq must be"):
        journal.append(
            InvestigationLoopEvent(
                seq=10,
                investigation_id="inv_test",
                event_type="decision_call_started",
                attempt_epoch=1,
                iteration=0,
                call_id="call-0",
            )
        )

    current = journal.rebuild()
    assert current is not None
    with pytest.raises(LoopTransitionError, match="attempt_epoch"):
        journal.append(
            InvestigationLoopEvent(
                seq=current.last_seq + 1,
                investigation_id="inv_test",
                event_type="decision_call_started",
                attempt_epoch=0,
                iteration=0,
                call_id="call-0",
            )
        )

    assert len(journal.events()) == line_count


def test_old_executor_is_fenced_after_a_new_owner_claims(tmp_path: Path) -> None:
    first = _journal(tmp_path)
    second = JsonlLoopJournal(first.path)
    assert first.claim_attempt().attempt_epoch == 1
    assert second.claim_attempt().attempt_epoch == 2

    with pytest.raises(LoopTransitionError, match="stale loop executor"):
        first.append_new(
            "decision_call_started",
            iteration=0,
            call_id="stale-owner-call",
        )

    state = second.append_new(
        "decision_call_started",
        iteration=0,
        call_id="current-owner-call",
    )
    assert state.pending_call_id == "current-owner-call"


def test_completed_step_ids_are_unique_and_counters_cannot_exceed_caps(
    tmp_path: Path,
) -> None:
    journal = _journal(tmp_path, max_steps=1, llm_call_cap=2)
    _complete_iteration(journal, 0)

    with pytest.raises(LoopTransitionError, match="max_steps"):
        journal.append_new(
            "probe_started",
            iteration=1,
            probe_id="probe-next",
            probe_fingerprint="next",
        )

    call_id = make_loop_call_id("inv_test", 1)
    journal.append_new("decision_call_started", iteration=1, call_id=call_id)
    first_step_id = make_loop_step_id("decision", make_loop_call_id("inv_test", 0))
    with pytest.raises(LoopTransitionError, match="already recorded"):
        journal.append_new(
            "decision_call_completed",
            iteration=1,
            call_id=call_id,
            step_id=first_step_id,
            response_hash="response-1",
            typed_decision={"action": "conclude"},
        )

    state = journal.rebuild()
    assert state is not None
    assert state.probes_completed == 1
    assert state.llm_calls_settled == 1
    assert state.pending_call_id == call_id


def test_unterminated_invalid_tail_is_ignored_but_middle_corruption_fails_closed(
    tmp_path: Path,
) -> None:
    journal = _journal(tmp_path)
    journal.claim_attempt()
    before = journal.rebuild()

    with journal.path.open("ab") as handle:
        handle.write(b'{"schema_version":1,"seq"')

    assert journal.rebuild() == before

    valid_lines = journal.path.read_bytes().splitlines(keepends=True)[:2]
    journal.path.write_bytes(valid_lines[0] + b"{broken-json}\n" + valid_lines[1])
    with pytest.raises(LoopJournalCorruptionError, match="Invalid committed record"):
        journal.rebuild()


def test_valid_unterminated_tail_is_a_complete_event(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    body = journal.path.read_bytes()
    journal.path.write_bytes(body.removesuffix(b"\n"))

    state = journal.rebuild()
    assert state is not None
    assert state.last_seq == 0


def test_snapshot_is_optional_and_never_hides_journal_corruption(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    _complete_iteration(journal, 0)
    state = journal.rebuild()
    snapshot_path = journal.write_snapshot(state)

    assert snapshot_path.exists()
    assert journal.read_snapshot() == state

    lines = journal.path.read_bytes().splitlines(keepends=True)
    journal.path.write_bytes(lines[0] + b"{corrupt}\n" + b"".join(lines[2:]))
    with pytest.raises(LoopJournalCorruptionError):
        journal.rebuild()
    assert journal.read_snapshot() == state


def test_uncertain_call_is_visible_and_not_settled(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    call_id = make_loop_call_id("inv_test", 0)
    journal.append_new(
        "decision_call_started",
        iteration=0,
        call_id=call_id,
    )
    state = journal.append_new(
        "loop_call_uncertain",
        iteration=0,
        call_id=call_id,
        error="provider response may have been consumed",
    )

    assert state.status == "uncertain"
    assert state.pending_call_id == call_id
    assert state.llm_calls_settled == 0
    assert state.remaining_call_budget == 3
    assert state.failure_history == ["provider response may have been consumed"]
    with pytest.raises(LoopTransitionError, match="terminal"):
        journal.append_new("attempt_started", attempt_epoch=1)


def test_preflight_budget_rejection_clears_pending_call_before_terminal_exit(
    tmp_path: Path,
) -> None:
    journal = _journal(tmp_path)
    call_id = make_loop_call_id("inv_test", 0)
    journal.append_new("decision_call_started", iteration=0, call_id=call_id)
    rejected = journal.append_new(
        "decision_call_rejected",
        iteration=0,
        call_id=call_id,
        error="Run requests budget exhausted.",
    )
    terminal = journal.append_new("loop_budget_exhausted")

    assert rejected.pending_call_id is None
    assert rejected.llm_calls_settled == 0
    assert terminal.status == "budget_exhausted"


def test_append_truncates_an_unterminated_crash_tail(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    with journal.path.open("ab") as handle:
        handle.write(b'{"seq":')

    resumed = JsonlLoopJournal(journal.path)
    before = resumed.rebuild()
    assert before is not None and before.last_seq == 0

    resumed.append_new("attempt_started", attempt_epoch=1)

    after = resumed.rebuild()
    assert after is not None
    assert after.last_seq == 1
    assert after.attempt_epoch == 1


def test_stale_attempt_cannot_commit_an_external_side_effect(tmp_path: Path) -> None:
    first = _journal(tmp_path)
    first.claim_attempt()
    second = JsonlLoopJournal(first.path)
    second.claim_attempt()

    with pytest.raises(LoopTransitionError, match="stale"):
        with first.fenced_side_effect():
            raise AssertionError("stale executor entered the commit section")


def test_fingerprints_fail_closed_on_initialize_resume_and_event(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    state = journal.rebuild()
    assert state is not None

    with pytest.raises(LoopResumeIncompatibleError, match="policy_fingerprint"):
        assert_resume_compatible(
            state,
            plan_fingerprint="plan-v1",
            policy_fingerprint="policy-v2",
            code_fingerprint="code-v1",
        )

    with pytest.raises(LoopResumeIncompatibleError, match="persisted policy_fingerprint"):
        journal.append_new(
            "attempt_started",
            attempt_epoch=1,
            policy_fingerprint="policy-v2",
        )

    with pytest.raises(LoopResumeIncompatibleError, match="changed policy_fingerprint"):
        journal.initialize(
            investigation_id="inv_test",
            source_session_id="run_test",
            question_id="question_test",
            plan_fingerprint="plan-v1",
            policy_fingerprint="policy-v2",
            code_fingerprint="code-v1",
            max_steps=2,
            llm_call_cap=3,
        )


def test_stable_identifiers_change_only_with_identity_inputs() -> None:
    investigation_id = make_investigation_id("run-1", "question-1", "plan-1")
    assert investigation_id == make_investigation_id("run-1", "question-1", "plan-1")
    assert investigation_id != make_investigation_id("run-1", "question-2", "plan-1")

    call_id = make_loop_call_id(investigation_id, 0)
    assert call_id == make_loop_call_id(investigation_id, 0)
    assert call_id != make_loop_call_id(investigation_id, 1)

    probe_id = make_loop_probe_id(investigation_id, 0, "probe-plan")
    assert probe_id == make_loop_probe_id(investigation_id, 0, "probe-plan")
    assert probe_id != make_loop_probe_id(investigation_id, 0, "other-probe")
    assert make_loop_step_id("probe", probe_id) == make_loop_step_id("probe", probe_id)
