"""E4a supervisor state, recovery, stop, and progress contracts."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest

from eda_platform.agents.exploration.supervisor import (
    BranchConstraintPort,
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
        self.round_branch_ids: list[str | None] = []
        self.abandonments: list[tuple[str, int, tuple[object, ...]]] = []
        self.settled: list[tuple[int, bool]] = []
        self.settled_frontier_empty: list[bool] = []
        self.reductions: list[tuple[str, str]] = []
        self.stops: list[str] = []

    def snapshot(self) -> SupervisorJournalState:
        return self.state

    def start_round(
        self, round_index: int, *, branch_id: str | None = None
    ) -> SupervisorJournalState:
        assert self.state.current_round_index is None
        assert round_index == self.state.rounds_started
        self.started.append(round_index)
        self.round_branch_ids.append(branch_id)
        branch_updates: dict[str, object] = {}
        if branch_id is not None and branch_id != self.state.active_branch_id:
            assert self.state.current_line_abandoned
            assert branch_id == f"br_{self.state.branches_started + 1}"
            branch_updates = {
                "active_branch_id": branch_id,
                "branches_started": self.state.branches_started + 1,
                "current_line_abandoned": False,
            }
        else:
            assert not self.state.current_line_abandoned
            assert branch_id == self.state.active_branch_id
        self.state = replace(
            self.state,
            rounds_started=self.state.rounds_started + 1,
            current_round_index=round_index,
            remaining_round_budget=self.state.remaining_round_budget - 1,
            current_round_receipt_ids=frozenset(),
            current_round_reduction_committed=False,
            **branch_updates,  # type: ignore[arg-type]
        )
        return self.state

    def abandon_branch(
        self,
        *,
        branch_id: str,
        round_index: int,
        constraints: tuple[object, ...],
    ) -> SupervisorJournalState:
        assert self.state.current_round_index is None
        assert not self.state.current_line_abandoned
        assert branch_id == (self.state.active_branch_id or "main")
        self.abandonments.append((branch_id, round_index, constraints))
        self.state = replace(
            self.state,
            current_line_abandoned=True,
            consecutive_no_progress=0,
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
        frontier_empty: bool = False,
        adjudicated_transitions: int = 0,
    ) -> SupervisorJournalState:
        assert round_index == self.state.current_round_index
        self.settled.append((round_index, progress))
        self.settled_frontier_empty.append(frontier_empty)
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
            consecutive_no_adjudication=(
                0
                if adjudicated_transitions > 0
                else self.state.consecutive_no_adjudication + 1
            ),
            consecutive_empty_frontier=(
                self.state.consecutive_empty_frontier + 1 if frontier_empty else 0
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
        admitted_bundle_count: int = 0,
    ) -> None:
        self.transitions = transitions
        self.goal_satisfied = goal_satisfied
        self.priority = priority
        self.fail = fail
        self.admitted_bundle_count = admitted_bundle_count
        self.calls = 0
        self.probe_free_calls = 0

    def reduce_without_probes(
        self,
        context: PhaseContext,
        frontier: ScoredFrontier,
        *,
        logical_step_id: str,
    ) -> ReductionOutcome:
        self.probe_free_calls += 1
        return ReductionOutcome(
            transitions=(),
            frontier=frontier,
            ledger_digest=f"ledger-empty-{context.round_index}",
        )

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
            admitted_bundle_count=self.admitted_bundle_count,
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
    admitted_bundle_count: int = 0,
) -> ReductionOutcome:
    return ReductionOutcome(
        transitions=cast(tuple, transitions),
        frontier=ScoredFrontier(
            (FrontierItem("hyp-1", priority),), f"reduced-{round_index}"
        ),
        ledger_digest=f"ledger-{round_index}",
        goal_satisfied=goal_satisfied,
        admitted_bundle_count=admitted_bundle_count,
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
    config: SupervisorConfig | None = None,
    branch_deriver: object | None = None,
) -> tuple[ExplorationSupervisor, FakeControl, FakeGenerator, FakeExecutor, FakeReducer]:
    actual_control = control or FakeControl()
    actual_generator = generator or FakeGenerator(journal)
    actual_executor = executor or FakeExecutor(journal, commit_receipt=True)
    actual_reducer = reducer or FakeReducer()
    supervisor = ExplorationSupervisor(
        config=config or SupervisorConfig(admission_score_threshold=0.5),
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
        branch_deriver=cast("BranchConstraintPort | None", branch_deriver),
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


def test_bundles_admitted_through_the_gate_count_as_progress() -> None:
    """A round that passed claims through the evidence gate did work, even when
    no insight transition was material."""
    journal = FakeJournal(_state(remaining_round_budget=1))
    supervisor, _, _, _, _ = _supervisor(
        journal,
        reducer=FakeReducer(
            transitions=("inconclusive",),
            goal_satisfied=False,
            priority=0.2,
            admitted_bundle_count=2,
        ),
        scheduler=FakeScheduler(priority=0.2),
    )

    result = supervisor.run()

    assert journal.settled == [(0, True)]
    assert result.stop_reason == "budget_exhausted"


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


def test_empty_frontier_commits_a_probe_free_reduction_each_round() -> None:
    """The round still scored candidates, and the reduction's frontier digest
    is the only thing binding those decisions to the journal."""
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
    assert reducer.probe_free_calls == 2
    assert journal.reductions == [
        ("frontier-0", "ledger-empty-0"),
        ("frontier-1", "ledger-empty-1"),
    ]
    # The terminal edge still runs the finalizer, so a report ref exists.
    assert result.report_ref == "report-1"


class EmptyOnRoundsScheduler(FakeScheduler):
    """Empty frontier on the listed rounds, a normal one everywhere else."""

    def __init__(self, empty_rounds: frozenset[int], *, priority: float = 0.8) -> None:
        super().__init__(priority=priority)
        self.empty_rounds = empty_rounds

    def admit_and_score(
        self, context: PhaseContext, candidates: CandidateBatch
    ) -> ScoredFrontier:
        items = (
            ()
            if context.round_index in self.empty_rounds
            else (FrontierItem("hyp-1", self.priority),)
        )
        return ScoredFrontier(items=items, digest=f"frontier-{context.round_index}")


def test_one_empty_frontier_round_is_not_exhaustion() -> None:
    """"Nothing to try this round" is a scheduling gap; only a repeated empty
    frontier is evidence the line is actually exhausted."""
    journal = FakeJournal(
        _state(remaining_round_budget=2, consecutive_no_progress=1)
    )
    supervisor, _, generator, _, reducer = _supervisor(
        journal,
        scheduler=EmptyOnRoundsScheduler(frozenset({0}), priority=0.8),
        executor=FakeExecutor(journal, commit_receipt=False),
        reducer=FakeReducer(transitions=(), goal_satisfied=False, priority=0.8),
    )

    result = supervisor.run()

    # Round 1 ran a real probe: the empty round 0 did not stop the run.
    assert journal.settled == [(0, False), (1, False)]
    assert journal.settled_frontier_empty == [True, False]
    assert reducer.probe_free_calls == 1
    assert reducer.calls == 1
    assert len(generator.calls) == 2
    # Plan-B soft stop: after the probe round also adjudicated nothing, two
    # zero-adjudication rounds end the run instead of coasting to the budget.
    assert result.stop_reason == "no_new_information"


def test_zero_adjudication_streak_stops_softly() -> None:
    """Plan-B soft stop (user decision 2026-08-03): gate admissions keep
    counting as progress, but two consecutive rounds with zero adjudicated
    transitions must end the run instead of burning to the hard budget cap
    (seed-6 failure mode)."""
    journal = FakeJournal(_state(remaining_round_budget=6))
    supervisor, _, _, _, reducer = _supervisor(
        journal,
        reducer=FakeReducer(
            transitions=("inconclusive",),
            goal_satisfied=False,
            admitted_bundle_count=1,
        ),
    )

    result = supervisor.run()

    assert result.stop_reason == "no_new_information"
    # Both rounds made "progress" (admitted bundles) yet adjudicated nothing.
    assert journal.settled == [(0, True), (1, True)]
    assert reducer.calls == 2


def test_adjudication_resets_the_soft_stop_streak() -> None:
    class AlternatingReducer(FakeReducer):
        def reduce(self, context, validated, frontier, *, logical_step_id):
            self.transitions = (
                ("new",) if context.round_index == 0 else ("inconclusive",)
            )
            return super().reduce(
                context, validated, frontier, logical_step_id=logical_step_id
            )

    journal = FakeJournal(_state(remaining_round_budget=6))
    supervisor, _, _, _, reducer = _supervisor(
        journal,
        reducer=AlternatingReducer(
            goal_satisfied=False, admitted_bundle_count=1
        ),
    )

    result = supervisor.run()

    # Round 0 adjudicated ("new"); the streak restarts, so rounds 1 and 2
    # form the stopping pair and the run settles three rounds in total.
    assert result.stop_reason == "no_new_information"
    assert reducer.calls == 3


def test_soft_stop_defers_to_branching_when_enabled() -> None:
    journal = FakeJournal(_state(remaining_round_budget=3))
    supervisor, _, _, _, reducer = _supervisor(
        journal,
        config=SupervisorConfig(
            admission_score_threshold=0.5,
            branch_trigger_stagnant_rounds=2,
            max_branches=1,
        ),
        reducer=FakeReducer(
            transitions=("inconclusive",),
            goal_satisfied=False,
            admitted_bundle_count=1,
        ),
        branch_deriver=FakeBranchDeriver((_branch_constraint(),)),
    )

    result = supervisor.run()

    # Branch-enabled runs keep the E6 stagnation machinery authoritative:
    # progress stays true here, so no branch and no soft stop — budget ends it.
    assert result.stop_reason == "budget_exhausted"
    assert reducer.calls == 3


def test_two_consecutive_empty_frontiers_do_terminate() -> None:
    journal = FakeJournal(
        _state(remaining_round_budget=4, consecutive_no_progress=1)
    )
    supervisor, _, _, _, reducer = _supervisor(
        journal, scheduler=FakeScheduler(empty=True)
    )

    result = supervisor.run()

    assert result.stop_reason == "no_new_information"
    assert journal.settled == [(0, False), (1, False)]
    assert journal.settled_frontier_empty == [True, True]
    assert reducer.probe_free_calls == 2


def test_terminal_resume_without_a_reduction_stays_supported() -> None:
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


def test_budget_death_reports_the_last_committed_reduction() -> None:
    """budget_exhausted must mean one contract, not two: the max_rounds path
    already renders a report, so a hard budget latch must too."""
    step_id = phase_step_id("xpl-1", 0, SupervisorPhase.REDUCE)
    outcome = _reduction(round_index=0, transitions=("new",))
    journal = FakeJournal(
        _state(
            rounds_started=1,
            rounds_settled=1,
            remaining_round_budget=3,
            frontier_digest=outcome.frontier.digest,
            ledger_digest=outcome.ledger_digest,
            reduction_digest=reduction_outcome_digest(outcome),
        )
    )
    supervisor, _, generator, _, _ = _supervisor(
        journal,
        budget=FakeBudget(exhaust=True),
        recovery=FakeRecovery({step_id: outcome}),
    )

    result = supervisor.run()

    assert result.stop_reason == "budget_exhausted"
    # The deterministic renderer, not the paid synthesizer.
    assert result.report_ref == "fallback-1"
    assert generator.calls == []
    assert journal.started == []
    assert journal.settled == []


def test_budget_death_without_a_recoverable_reduction_stays_bare() -> None:
    no_rounds = FakeJournal()
    supervisor, _, _, _, _ = _supervisor(no_rounds, budget=FakeBudget(exhaust=True))
    result = supervisor.run()
    assert result.stop_reason == "budget_exhausted"
    assert result.report_ref is None

    # A settled round whose reduction body cannot be recovered must not be
    # upgraded to "failed" either.
    lost = FakeJournal(
        _state(rounds_started=1, rounds_settled=1, remaining_round_budget=3)
    )
    supervisor, _, _, _, _ = _supervisor(
        lost, budget=FakeBudget(exhaust=True), recovery=FakeRecovery()
    )
    result = supervisor.run()
    assert result.stop_reason == "budget_exhausted"
    assert result.report_ref is None


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


# ------------------------------------------------------------- E6 branch mode


class FakeBranchDeriver:
    def __init__(self, constraints: tuple[object, ...] = ()) -> None:
        self.constraints = constraints
        self.calls: list[PhaseContext] = []

    def derive(self, context: PhaseContext) -> tuple[object, ...]:
        self.calls.append(context)
        return self.constraints


def _branch_constraint() -> object:
    from eda_platform.schemas.exploration import BranchConstraint, InsightFamily

    return BranchConstraint(
        hypothesis_fingerprint="f" * 16,
        coverage_key="cov-1",
        family=InsightFamily.DIAGNOSTIC,
        reason="refuted",
        detail_code="rcpt-1",
    )


def test_stagnation_switches_to_a_branch_before_terminating() -> None:
    journal = FakeJournal(_state(remaining_round_budget=8))
    deriver = FakeBranchDeriver((_branch_constraint(),))
    supervisor, _control, _generator, _executor, _reducer = _supervisor(
        journal,
        executor=FakeExecutor(journal, commit_receipt=False),
        reducer=FakeReducer(goal_satisfied=False, priority=0.2),
        config=SupervisorConfig(
            admission_score_threshold=0.5,
            branch_trigger_stagnant_rounds=2,
            max_branches=1,
        ),
        branch_deriver=deriver,
    )

    result = supervisor.run()

    assert result.stop_reason == "no_new_information"
    # Two stagnant rounds on main, then a single branch, then exhaustion.
    assert journal.round_branch_ids == [None, None, "br_1", "br_1"]
    assert [
        (branch_id, round_index) for branch_id, round_index, _ in journal.abandonments
    ] == [("main", 1)]
    assert journal.abandonments[0][2] == deriver.constraints
    assert len(deriver.calls) == 1


def test_branch_mode_requires_a_constraint_deriver() -> None:
    journal = FakeJournal()
    with pytest.raises(ValueError, match="deriver"):
        _supervisor(
            journal,
            config=SupervisorConfig(
                admission_score_threshold=0.5,
                branch_trigger_stagnant_rounds=2,
                max_branches=1,
            ),
        )


def test_branch_config_requires_both_trigger_and_budget() -> None:
    with pytest.raises(ValueError, match="branch"):
        SupervisorConfig(
            admission_score_threshold=0.5, branch_trigger_stagnant_rounds=2
        )
    with pytest.raises(ValueError, match="branch"):
        SupervisorConfig(admission_score_threshold=0.5, max_branches=1)


def test_branching_disabled_keeps_the_existing_termination() -> None:
    journal = FakeJournal(_state(remaining_round_budget=8))
    supervisor, _control, _generator, _executor, _reducer = _supervisor(
        journal,
        executor=FakeExecutor(journal, commit_receipt=False),
        reducer=FakeReducer(goal_satisfied=False, priority=0.2),
    )

    result = supervisor.run()

    assert result.stop_reason == "no_new_information"
    assert journal.abandonments == []
    assert journal.round_branch_ids == [None, None]
