"""E4a supervisor state, recovery, stop, and progress contracts."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest

from eda_platform.agents.exploration.supervisor import (
    BudgetPort,
    CandidateBatch,
    CompletedStepRecoveryPort,
    ExplorationSupervisor,
    FinalizationOutcome,
    FrontierItem,
    PhaseContext,
    PhaseTransition,
    ProbeOutcome,
    ProbeSelection,
    ReductionOutcome,
    ScoredFrontier,
    SupervisorBudgetExhausted,
    SupervisorConfig,
    SupervisorJournalState,
    SupervisorPhase,
    SupervisorRunResult,
    ValidationOutcome,
    phase_step_id,
    reduction_outcome_digest,
    render_soft_countdown,
)


class FakeJournal:
    def __init__(self, state: SupervisorJournalState | None = None) -> None:
        self.state = state or _state()
        self.started: list[int] = []
        self.settled: list[tuple[int, bool]] = []
        self.reductions: list[tuple[str, str]] = []
        self.stops: list[str] = []

    def snapshot(self) -> SupervisorJournalState:
        return self.state

    def start_round(self, round_index: int) -> SupervisorJournalState:
        assert self.state.current_round_index is None
        assert round_index == self.state.rounds_started
        self.started.append(round_index)
        self.state = replace(
            self.state,
            rounds_started=self.state.rounds_started + 1,
            current_round_index=round_index,
            remaining_round_budget=self.state.remaining_round_budget - 1,
            current_round_receipt_ids=frozenset(),
            current_round_reduction_committed=False,
        )
        return self.state

    def commit_reduction(
        self, *, frontier_digest: str, ledger_digest: str, reduction_digest: str
    ) -> SupervisorJournalState:
        self.reductions.append((frontier_digest, ledger_digest))
        self.state = replace(
            self.state,
            current_round_reduction_committed=True,
            frontier_digest=frontier_digest,
            ledger_digest=ledger_digest,
            reduction_digest=reduction_digest,
        )
        return self.state

    def settle_round(
        self,
        round_index: int,
        *,
        progress: bool,
        terminal_reason: str | None,
    ) -> SupervisorJournalState:
        assert round_index == self.state.current_round_index
        self.settled.append((round_index, progress))
        terminal_has_reduction = bool(
            terminal_reason and self.state.current_round_reduction_committed
        )
        self.state = replace(
            self.state,
            rounds_settled=self.state.rounds_settled + 1,
            current_round_index=None,
            consecutive_no_progress=(
                0 if progress else self.state.consecutive_no_progress + 1
            ),
            current_round_receipt_ids=frozenset(),
            current_round_reduction_committed=False,
            pending_terminal_reason=cast(object, terminal_reason),
            pending_terminal_has_reduction=terminal_has_reduction,
        )
        return self.state

    def mark_paused(self) -> SupervisorJournalState:
        assert self.state.status == "pause_requested"
        self.state = replace(self.state, status="paused")
        return self.state

    def stop(self, reason: str, *, report_ref: str | None) -> SupervisorJournalState:
        self.stops.append(reason)
        self.state = replace(
            self.state,
            status="stopped",
            stop_reason=cast(object, reason),
            final_report_ref=report_ref,
            pending_terminal_reason=None,
            pending_terminal_has_reduction=False,
        )
        return self.state

    def complete(self, step_id: str) -> None:
        self.state = replace(
            self.state,
            completed_step_ids=self.state.completed_step_ids | {step_id},
        )

    def receipt(self, receipt_id: str) -> None:
        self.state = replace(
            self.state,
            current_round_receipt_ids=self.state.current_round_receipt_ids
            | {receipt_id},
        )


class FakeRecovery:
    def __init__(self, values: dict[str, object] | None = None) -> None:
        self.values = values or {}
        self.loads: list[str] = []

    def load_required(self, logical_step_id: str) -> object:
        self.loads.append(logical_step_id)
        if logical_step_id not in self.values:
            raise KeyError(logical_step_id)
        return self.values[logical_step_id]

    def remember(self, logical_step_id: str, result: object) -> None:
        self.values[logical_step_id] = result


class FakeWitness:
    def __init__(self, verdicts: list[bool] | None = None) -> None:
        self.verdicts = verdicts or [True]
        self.calls: list[str] = []

    def recheck(self, expected_witness: str) -> bool:
        self.calls.append(expected_witness)
        index = min(len(self.calls) - 1, len(self.verdicts) - 1)
        return self.verdicts[index]


class FakeBudget:
    def __init__(self, *, exhaust: bool = False) -> None:
        self.exhaust = exhaust
        self.contexts: list[SupervisorJournalState] = []

    def check_start_round(self, state: SupervisorJournalState) -> None:
        if self.exhaust:
            raise SupervisorBudgetExhausted("wall time exhausted")

    def remaining(self, state: SupervisorJournalState) -> dict[str, object]:
        self.contexts.append(state)
        return {
            "requests": state.remaining_llm_call_budget,
            "rounds": state.remaining_round_budget,
            "tools": state.remaining_tool_call_budget,
        }


class FakeControl:
    def __init__(self, cancel_target: SupervisorPhase | None = None) -> None:
        self.transitions: list[PhaseTransition] = []
        self.cancel_target = cancel_target

    def checkpoint(self, transition: PhaseTransition) -> None:
        self.transitions.append(transition)
        if transition.target is self.cancel_target:
            from eda_platform.agents.exploration.supervisor import SupervisorCancelled

            raise SupervisorCancelled


class FakeGenerator:
    def __init__(self, journal: FakeJournal) -> None:
        self.journal = journal
        self.calls: list[PhaseContext] = []

    def generate(self, context: PhaseContext, *, logical_step_id: str) -> CandidateBatch:
        self.calls.append(context)
        self.journal.complete(logical_step_id)
        return CandidateBatch(({"candidate": context.round_index},))


class FakeScheduler:
    def __init__(self, *, priority: float = 0.8, empty: bool = False) -> None:
        self.priority = priority
        self.empty = empty

    def admit_and_score(
        self, context: PhaseContext, candidates: CandidateBatch
    ) -> ScoredFrontier:
        items = () if self.empty else (FrontierItem("hyp-1", self.priority),)
        return ScoredFrontier(items=items, digest=f"frontier-{context.round_index}")

    def select(
        self, context: PhaseContext, frontier: ScoredFrontier
    ) -> ProbeSelection | None:
        return ProbeSelection((frontier.items[0],)) if frontier.items else None


class FakeExecutor:
    def __init__(self, journal: FakeJournal, *, commit_receipt: bool) -> None:
        self.journal = journal
        self.commit_receipt = commit_receipt
        self.calls: list[PhaseContext] = []

    def execute(self, context: PhaseContext, selection: ProbeSelection) -> ProbeOutcome:
        self.calls.append(context)
        if self.commit_receipt:
            self.journal.receipt(f"rcpt-{context.round_index}")
        return ProbeOutcome({"selected": selection.items[0].hypothesis_id})


class FakeValidator:
    def validate(self, context: PhaseContext, probes: ProbeOutcome) -> ValidationOutcome:
        return ValidationOutcome(probes.payload)


class FakeReducer:
    def __init__(
        self,
        *,
        transitions: tuple[str, ...] = ("new",),
        goal_satisfied: bool = True,
        priority: float = 0.8,
        fail: bool = False,
    ) -> None:
        self.transitions = transitions
        self.goal_satisfied = goal_satisfied
        self.priority = priority
        self.fail = fail
        self.calls = 0

    def reduce(
        self,
        context: PhaseContext,
        validated: ValidationOutcome,
        frontier: ScoredFrontier,
        *,
        logical_step_id: str,
    ) -> ReductionOutcome:
        self.calls += 1
        if self.fail:
            raise ValueError("bad reduction")
        return _reduction(
            round_index=context.round_index,
            transitions=cast(tuple, self.transitions),
            goal_satisfied=self.goal_satisfied,
            priority=self.priority,
        )


class FakeFinalizer:
    def __init__(self, journal: FakeJournal) -> None:
        self.journal = journal
        self.synth_calls = 0
        self.render_calls = 0

    def synthesize(
        self,
        context: PhaseContext,
        reduction: ReductionOutcome,
        *,
        logical_step_id: str,
    ) -> FinalizationOutcome:
        self.synth_calls += 1
        self.journal.complete(logical_step_id)
        return FinalizationOutcome("report-1")

    def render_deterministic(
        self, context: PhaseContext, reduction: ReductionOutcome
    ) -> FinalizationOutcome:
        self.render_calls += 1
        return FinalizationOutcome("fallback-1")


def _state(**overrides: object) -> SupervisorJournalState:
    fields: dict[str, object] = {
        "exploration_id": "xpl-1",
        "data_state_witness": "witness-1",
        "status": "running",
        "rounds_started": 0,
        "rounds_settled": 0,
        "remaining_round_budget": 4,
        "remaining_llm_call_budget": 10,
        "remaining_tool_call_budget": 10,
    }
    fields.update(overrides)
    return SupervisorJournalState(**fields)  # type: ignore[arg-type]


def _reduction(
    *,
    round_index: int,
    transitions: tuple[str, ...] = ("new",),
    goal_satisfied: bool = False,
    priority: float = 0.8,
) -> ReductionOutcome:
    return ReductionOutcome(
        transitions=cast(tuple, transitions),
        frontier=ScoredFrontier(
            (FrontierItem("hyp-1", priority),), f"reduced-{round_index}"
        ),
        ledger_digest=f"ledger-{round_index}",
        goal_satisfied=goal_satisfied,
    )


def _supervisor(
    journal: FakeJournal,
    *,
    witness: FakeWitness | None = None,
    budget: FakeBudget | None = None,
    control: FakeControl | None = None,
    generator: FakeGenerator | None = None,
    scheduler: FakeScheduler | None = None,
    executor: FakeExecutor | None = None,
    reducer: FakeReducer | None = None,
    recovery: FakeRecovery | None = None,
) -> tuple[ExplorationSupervisor, FakeControl, FakeGenerator, FakeExecutor, FakeReducer]:
    actual_control = control or FakeControl()
    actual_generator = generator or FakeGenerator(journal)
    actual_executor = executor or FakeExecutor(journal, commit_receipt=True)
    actual_reducer = reducer or FakeReducer()
    supervisor = ExplorationSupervisor(
        config=SupervisorConfig(admission_score_threshold=0.5),
        journal=journal,
        witness=witness or FakeWitness(),
        budget=cast(BudgetPort, budget or FakeBudget()),
        control=actual_control,
        generator=actual_generator,
        scheduler=scheduler or FakeScheduler(),
        executor=actual_executor,
        validator=FakeValidator(),
        reducer=actual_reducer,
        finalizer=FakeFinalizer(journal),
        recovery=cast(CompletedStepRecoveryPort, recovery or FakeRecovery()),
    )
    return supervisor, actual_control, actual_generator, actual_executor, actual_reducer


def _targets(result: SupervisorRunResult) -> list[SupervisorPhase]:
    return [transition.target for transition in result.transitions]


def test_happy_path_is_explicit_and_every_edge_has_a_cancel_checkpoint() -> None:
    journal = FakeJournal()
    supervisor, control, generator, executor, reducer = _supervisor(journal)

    result = supervisor.run()

    assert result.stop_reason == "completed"
    assert result.report_ref == "report-1"
    assert journal.settled == [(0, True)]
    assert generator.calls[0].soft_countdown_context.startswith(
        "[exploration_soft_countdown]"
    )
    assert executor.calls[0].soft_countdown_context == generator.calls[0].soft_countdown_context
    assert reducer.calls == 1
    assert _targets(result) == [
        SupervisorPhase.ORIENT,
        SupervisorPhase.GENERATE,
        SupervisorPhase.ADMIT_AND_SCORE,
        SupervisorPhase.SELECT,
        SupervisorPhase.EXECUTE_PROBES,
        SupervisorPhase.VALIDATE,
        SupervisorPhase.REDUCE,
        SupervisorPhase.SYNTHESIZE,
        SupervisorPhase.STOP,
    ]
    assert tuple(control.transitions) == result.transitions


@pytest.mark.parametrize(
    ("commit_receipt", "transitions"),
    [(True, ("inconclusive",)), (False, ("new",))],
)
def test_progress_requires_both_committed_receipt_and_material_transition(
    commit_receipt: bool, transitions: tuple[str, ...]
) -> None:
    journal = FakeJournal(_state(remaining_round_budget=2))
    reducer = FakeReducer(
        transitions=transitions,
        goal_satisfied=False,
        priority=0.2,
    )
    supervisor, _, _, _, _ = _supervisor(
        journal,
        executor=FakeExecutor(journal, commit_receipt=commit_receipt),
        reducer=reducer,
        scheduler=FakeScheduler(priority=0.2),
    )

    result = supervisor.run()

    assert result.stop_reason == "no_new_information"
    assert journal.settled == [(0, False), (1, False)]
    assert reducer.calls == 2


def test_priority_equal_to_admission_line_does_not_trigger_no_information() -> None:
    journal = FakeJournal(
        _state(remaining_round_budget=1, consecutive_no_progress=1)
    )
    supervisor, _, _, _, _ = _supervisor(
        journal,
        executor=FakeExecutor(journal, commit_receipt=False),
        reducer=FakeReducer(
            transitions=(), goal_satisfied=False, priority=0.5
        ),
        scheduler=FakeScheduler(priority=0.5),
    )
    result = supervisor.run()
    assert result.stop_reason == "budget_exhausted"


def test_witness_is_rechecked_each_round_and_mismatch_stops_before_side_effects() -> None:
    journal = FakeJournal(_state(remaining_round_budget=3))
    witness = FakeWitness([True, False])
    generator = FakeGenerator(journal)
    supervisor, _, _, _, _ = _supervisor(
        journal,
        witness=witness,
        generator=generator,
        executor=FakeExecutor(journal, commit_receipt=False),
        reducer=FakeReducer(transitions=(), goal_satisfied=False, priority=0.8),
    )

    result = supervisor.run()

    assert result.stop_reason == "state_witness_changed"
    assert witness.calls == ["witness-1", "witness-1"]
    assert journal.started == [0]
    assert len(generator.calls) == 1


def test_resume_adopts_completed_generate_step_without_resending() -> None:
    step_id = phase_step_id("xpl-1", 0, SupervisorPhase.GENERATE)
    journal = FakeJournal(
        _state(
            rounds_started=1,
            rounds_settled=0,
            current_round_index=0,
            remaining_round_budget=3,
            completed_step_ids=frozenset({step_id}),
        )
    )
    recovery = FakeRecovery({step_id: CandidateBatch(({"recovered": True},))})
    generator = FakeGenerator(journal)
    supervisor, _, _, _, _ = _supervisor(
        journal, generator=generator, recovery=recovery
    )

    result = supervisor.run()

    assert result.stop_reason == "completed"
    assert generator.calls == []
    assert recovery.loads[0] == step_id
    assert journal.started == []


def test_resume_adopts_committed_reduction_and_preserves_round_progress() -> None:
    step_id = phase_step_id("xpl-1", 0, SupervisorPhase.REDUCE)
    outcome = _reduction(round_index=0, transitions=("refuted",), goal_satisfied=True)
    journal = FakeJournal(
        _state(
            rounds_started=1,
            rounds_settled=0,
            current_round_index=0,
            remaining_round_budget=3,
            current_round_receipt_ids=frozenset({"rcpt-before-crash"}),
            current_round_reduction_committed=True,
            frontier_digest=outcome.frontier.digest,
            ledger_digest=outcome.ledger_digest,
            reduction_digest=reduction_outcome_digest(outcome),
        )
    )
    recovery = FakeRecovery({step_id: outcome})
    generator = FakeGenerator(journal)
    executor = FakeExecutor(journal, commit_receipt=True)
    reducer = FakeReducer(fail=True)
    supervisor, _, _, _, _ = _supervisor(
        journal,
        generator=generator,
        executor=executor,
        reducer=reducer,
        recovery=recovery,
    )

    result = supervisor.run()

    assert result.stop_reason == "completed"
    assert journal.settled == [(0, True)]
    assert generator.calls == []
    assert executor.calls == []
    assert reducer.calls == 0
    assert SupervisorPhase.REDUCE in _targets(result)


def test_resume_rejects_reduction_body_tampering_even_when_legacy_digests_match() -> None:
    step_id = phase_step_id("xpl-1", 0, SupervisorPhase.REDUCE)
    committed = _reduction(round_index=0, transitions=("new",), goal_satisfied=False)
    tampered = replace(committed, goal_satisfied=True)
    journal = FakeJournal(
        _state(
            rounds_started=1,
            rounds_settled=0,
            current_round_index=0,
            remaining_round_budget=3,
            current_round_reduction_committed=True,
            frontier_digest=committed.frontier.digest,
            ledger_digest=committed.ledger_digest,
            reduction_digest=reduction_outcome_digest(committed),
        )
    )
    supervisor, _, generator, executor, reducer = _supervisor(
        journal,
        recovery=FakeRecovery({step_id: tampered}),
    )

    result = supervisor.run()

    assert result.stop_reason == "failed"
    assert result.error is not None
    assert "recovered reduction digests do not match" in result.error
    assert generator.calls == []
    assert executor.calls == []
    assert reducer.calls == 0


def test_resume_after_settlement_adopts_terminal_reduction_without_new_round() -> None:
    step_id = phase_step_id("xpl-1", 0, SupervisorPhase.REDUCE)
    outcome = _reduction(round_index=0, transitions=("new",), goal_satisfied=True)
    journal = FakeJournal(
        _state(
            rounds_started=1,
            rounds_settled=1,
            remaining_round_budget=3,
            pending_terminal_reason="completed",
            pending_terminal_has_reduction=True,
            frontier_digest=outcome.frontier.digest,
            ledger_digest=outcome.ledger_digest,
            reduction_digest=reduction_outcome_digest(outcome),
        )
    )
    recovery = FakeRecovery({step_id: outcome})
    generator = FakeGenerator(journal)
    reducer = FakeReducer(fail=True)
    supervisor, _, _, _, _ = _supervisor(
        journal,
        generator=generator,
        reducer=reducer,
        recovery=recovery,
    )

    result = supervisor.run()

    assert result.stop_reason == "completed"
    assert result.report_ref == "report-1"
    assert journal.started == []
    assert journal.settled == []
    assert generator.calls == []
    assert reducer.calls == 0
    assert recovery.loads[0] == step_id


def test_resume_after_terminal_settlement_rechecks_witness_before_synthesis() -> None:
    step_id = phase_step_id("xpl-1", 0, SupervisorPhase.REDUCE)
    journal = FakeJournal(
        _state(
            rounds_started=1,
            rounds_settled=1,
            remaining_round_budget=3,
            pending_terminal_reason="completed",
            pending_terminal_has_reduction=True,
        )
    )
    supervisor, _, generator, _, _ = _supervisor(
        journal,
        witness=FakeWitness([False]),
        recovery=FakeRecovery({step_id: _reduction(round_index=0)}),
    )

    result = supervisor.run()

    assert result.stop_reason == "state_witness_changed"
    assert generator.calls == []
    assert journal.started == []


def test_empty_frontier_settles_each_round_once_and_recovers_terminal_without_reduction() -> None:
    journal = FakeJournal(_state(remaining_round_budget=2))
    supervisor, _, generator, executor, reducer = _supervisor(
        journal,
        scheduler=FakeScheduler(empty=True),
    )

    result = supervisor.run()

    assert result.stop_reason == "no_new_information"
    assert journal.settled == [(0, False), (1, False)]
    assert len(generator.calls) == 2
    assert executor.calls == []
    assert reducer.calls == 0

    recovered = FakeJournal(
        _state(
            rounds_started=2,
            rounds_settled=2,
            remaining_round_budget=0,
            consecutive_no_progress=2,
            pending_terminal_reason="no_new_information",
            pending_terminal_has_reduction=False,
        )
    )
    resumed, _, resumed_generator, _, _ = _supervisor(recovered)
    resumed_result = resumed.run()
    assert resumed_result.stop_reason == "no_new_information"
    assert resumed_generator.calls == []
    assert recovered.started == []


@pytest.mark.parametrize("status", ["pause_requested", "paused"])
def test_pause_is_resumable_and_never_writes_a_stop(status: str) -> None:
    journal = FakeJournal(_state(status=status))  # type: ignore[arg-type]
    supervisor, _, generator, _, _ = _supervisor(journal)
    result = supervisor.run()
    assert result.status == "paused"
    assert result.stop_reason is None
    assert result.phase is SupervisorPhase.PAUSED
    assert journal.stops == []
    assert generator.calls == []


def test_cancel_checkpoint_prevents_target_phase_and_stops_cancelled() -> None:
    journal = FakeJournal()
    control = FakeControl(cancel_target=SupervisorPhase.EXECUTE_PROBES)
    supervisor, _, _, executor, reducer = _supervisor(journal, control=control)

    result = supervisor.run()

    assert result.stop_reason == "cancelled"
    assert executor.calls == []
    assert reducer.calls == 0
    assert journal.stops == ["cancelled"]


def test_budget_and_max_rounds_stop_before_new_paid_work() -> None:
    budget_journal = FakeJournal()
    supervisor, _, generator, _, _ = _supervisor(
        budget_journal, budget=FakeBudget(exhaust=True)
    )
    budget_result = supervisor.run()
    assert budget_result.stop_reason == "budget_exhausted"
    assert generator.calls == []
    assert budget_journal.started == []

    capped_journal = FakeJournal(_state(remaining_round_budget=0))
    supervisor, _, generator, _, _ = _supervisor(capped_journal)
    capped_result = supervisor.run()
    assert capped_result.stop_reason == "budget_exhausted"
    assert generator.calls == []
    assert capped_journal.started == []


def test_missing_recovery_data_and_component_faults_fail_closed() -> None:
    step_id = phase_step_id("xpl-1", 0, SupervisorPhase.GENERATE)
    journal = FakeJournal(
        _state(
            rounds_started=1,
            current_round_index=0,
            remaining_round_budget=3,
            completed_step_ids=frozenset({step_id}),
        )
    )
    supervisor, _, generator, _, _ = _supervisor(journal, recovery=FakeRecovery())
    result = supervisor.run()
    assert result.stop_reason == "failed"
    assert result.error and "KeyError" in result.error
    assert generator.calls == []

    failing_journal = FakeJournal()
    supervisor, _, _, _, _ = _supervisor(
        failing_journal, reducer=FakeReducer(fail=True)
    )
    result = supervisor.run()
    assert result.stop_reason == "failed"
    assert result.error == "ValueError: bad reduction"


def test_soft_countdown_is_canonical_and_rejects_domain_objects() -> None:
    left = render_soft_countdown({"tools": 2, "requests": 3})
    right = render_soft_countdown({"requests": 3, "tools": 2})
    assert left == right
    assert 'remaining={"requests":3,"tools":2}' in left
    with pytest.raises(ValueError, match="serializable"):
        render_soft_countdown({"bad": object()})


def test_journal_projection_rejects_inconsistent_resume_state() -> None:
    with pytest.raises(ValueError, match="round counters"):
        _state(rounds_started=2, rounds_settled=0, current_round_index=0)
    with pytest.raises(ValueError, match="round-local"):
        _state(current_round_receipt_ids=frozenset({"rcpt-orphan"}))
