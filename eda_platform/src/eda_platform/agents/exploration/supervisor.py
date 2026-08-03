"""Deterministic E4a exploration supervisor.

The supervisor owns control flow, not analysis semantics.  Candidate generation,
scheduling, probe execution, validation, reduction, and final rendering are narrow
ports so E4a components can evolve independently.  The journal remains the recovery
authority: a paid step recorded as completed is recovered and is never sent again.

Normal rounds follow the explicit state sequence from the authoritative plan::

    ORIENT -> GENERATE -> ADMIT_AND_SCORE -> SELECT -> EXECUTE_PROBES
           -> VALIDATE -> REDUCE -> (ORIENT | SYNTHESIZE -> STOP)

Every state edge passes through ``ControlPort.checkpoint``.  Resume may jump from
ORIENT to REDUCE when the journal proves that round's reduction was committed; this
is an adoption path, not a second execution of the skipped steps.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, Protocol

from eda_platform.core.ids import stable_hash
from eda_platform.schemas.exploration import (
    MAIN_LINE_ID,
    BranchConstraint,
    ExplorationGracefulStopReason,
    ExplorationStopReason,
)

JournalStatus = Literal["running", "pause_requested", "paused", "stopped"]
InsightTransition = Literal["new", "reinforced", "refuted", "inconclusive"]

_PROGRESS_TRANSITIONS = frozenset({"new", "reinforced", "refuted"})


class SupervisorPhase(StrEnum):
    """Typed outer-loop states.  PAUSED is resumable and never a stop reason."""

    ORIENT = "ORIENT"
    GENERATE = "GENERATE"
    ADMIT_AND_SCORE = "ADMIT_AND_SCORE"
    SELECT = "SELECT"
    EXECUTE_PROBES = "EXECUTE_PROBES"
    VALIDATE = "VALIDATE"
    REDUCE = "REDUCE"
    SYNTHESIZE = "SYNTHESIZE"
    PAUSED = "PAUSED"
    STOP = "STOP"


@dataclass(frozen=True, slots=True)
class PhaseTransition:
    source: SupervisorPhase | None
    target: SupervisorPhase
    round_index: int | None


@dataclass(frozen=True, slots=True)
class SupervisorJournalState:
    """Small projection a driver builds from the append-only exploration journal.

    ``current_round_receipt_ids`` must be derived from events since the open
    ``round_started`` event.  A global receipt count is insufficient on resume:
    receipts committed before a crash still count as progress for that open round.
    ``current_round_reduction_committed`` is likewise derived from the event suffix.
    """

    exploration_id: str
    data_state_witness: str
    status: JournalStatus
    stop_reason: ExplorationStopReason | None = None
    final_report_ref: str | None = None
    rounds_started: int = 0
    rounds_settled: int = 0
    current_round_index: int | None = None
    remaining_round_budget: int = 0
    remaining_llm_call_budget: int | None = None
    remaining_tool_call_budget: int = 0
    consecutive_no_progress: int = 0
    consecutive_empty_frontier: int = 0
    consecutive_no_adjudication: int = 0
    completed_step_ids: frozenset[str] = field(default_factory=frozenset)
    completed_probe_fingerprints: frozenset[str] = field(default_factory=frozenset)
    uncertain_call_ids: frozenset[str] = field(default_factory=frozenset)
    step_receipt_refs: Mapping[str, str] = field(default_factory=dict)
    current_round_receipt_ids: frozenset[str] = field(default_factory=frozenset)
    current_round_reduction_committed: bool = False
    pending_terminal_reason: ExplorationGracefulStopReason | None = None
    pending_terminal_has_reduction: bool = False
    frontier_digest: str | None = None
    ledger_digest: str | None = None
    reduction_digest: str | None = None
    active_branch_id: str | None = None
    current_line_abandoned: bool = False
    branches_started: int = 0

    def __post_init__(self) -> None:
        if not self.exploration_id or not self.data_state_witness:
            raise ValueError("journal projection requires exploration_id and witness.")
        counts = {
            "rounds_started": self.rounds_started,
            "rounds_settled": self.rounds_settled,
            "remaining_round_budget": self.remaining_round_budget,
            "remaining_tool_call_budget": self.remaining_tool_call_budget,
            "consecutive_no_progress": self.consecutive_no_progress,
            "consecutive_no_adjudication": self.consecutive_no_adjudication,
            "consecutive_empty_frontier": self.consecutive_empty_frontier,
            "branches_started": self.branches_started,
        }
        if self.remaining_llm_call_budget is not None:
            counts["remaining_llm_call_budget"] = self.remaining_llm_call_budget
        for name, value in counts.items():
            if isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        has_open_round = self.current_round_index is not None
        if self.rounds_started != self.rounds_settled + int(has_open_round):
            raise ValueError("round counters do not match the open-round projection.")
        if has_open_round and self.current_round_index != self.rounds_settled:
            raise ValueError("current_round_index must identify the unsettled round.")
        if not has_open_round and (
            self.current_round_receipt_ids or self.current_round_reduction_committed
        ):
            raise ValueError("round-local recovery fields require an open round.")
        if self.pending_terminal_reason is not None and has_open_round:
            raise ValueError("a pending terminal decision cannot have an open round.")
        if self.pending_terminal_has_reduction and self.pending_terminal_reason is None:
            raise ValueError("pending terminal reduction requires a terminal decision.")
        if (self.status == "stopped") != (self.stop_reason is not None):
            raise ValueError("stop_reason must be present exactly for stopped state.")


@dataclass(frozen=True, slots=True)
class SupervisorConfig:
    admission_score_threshold: float
    synthesize_on_graceful_stop: bool = True
    # E6: stagnation-triggered branching. Both fields set = enabled; both unset
    # = the pre-E6 termination behavior, unchanged.
    branch_trigger_stagnant_rounds: int | None = None
    max_branches: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(self.admission_score_threshold):
            raise ValueError("admission_score_threshold must be finite.")
        if (self.branch_trigger_stagnant_rounds is None) != (self.max_branches == 0):
            raise ValueError(
                "branch mode requires both branch_trigger_stagnant_rounds "
                "and max_branches (or neither)."
            )
        if self.branch_trigger_stagnant_rounds is not None and (
            self.branch_trigger_stagnant_rounds < 1 or self.max_branches < 1
        ):
            raise ValueError("branch trigger and budget must be positive.")


@dataclass(frozen=True, slots=True)
class PhaseContext:
    exploration_id: str
    round_index: int
    phase: SupervisorPhase
    data_state_witness: str
    soft_countdown_context: str
    completed_step_ids: frozenset[str]
    # Set only on terminal renders where the stop latched mid-round, so no
    # settled-round event carries the reason yet (seed-6 regression).
    terminal_reason: str | None = None

    def for_phase(self, phase: SupervisorPhase) -> PhaseContext:
        return PhaseContext(
            exploration_id=self.exploration_id,
            round_index=self.round_index,
            phase=phase,
            data_state_witness=self.data_state_witness,
            soft_countdown_context=self.soft_countdown_context,
            completed_step_ids=self.completed_step_ids,
            terminal_reason=self.terminal_reason,
        )


@dataclass(frozen=True, slots=True)
class CandidateBatch:
    candidates: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class FrontierItem:
    hypothesis_id: str
    priority: float
    payload: object | None = None

    def __post_init__(self) -> None:
        if not self.hypothesis_id or not math.isfinite(self.priority):
            raise ValueError("frontier items require an id and finite priority.")


@dataclass(frozen=True, slots=True)
class ScoredFrontier:
    items: tuple[FrontierItem, ...]
    digest: str

    def __post_init__(self) -> None:
        if not self.digest:
            raise ValueError("frontier digest cannot be empty.")
        ids = [item.hypothesis_id for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("frontier hypothesis ids must be unique.")

    @property
    def highest_priority(self) -> float | None:
        return max((item.priority for item in self.items), default=None)


@dataclass(frozen=True, slots=True)
class ProbeSelection:
    items: tuple[FrontierItem, ...]

    def __post_init__(self) -> None:
        if not self.items:
            raise ValueError("a probe selection cannot be empty.")


@dataclass(frozen=True, slots=True)
class ProbeOutcome:
    payload: object | None = None


@dataclass(frozen=True, slots=True)
class ValidationOutcome:
    payload: object | None = None
    validator_exhausted: bool = False


@dataclass(frozen=True, slots=True)
class ReductionOutcome:
    transitions: tuple[InsightTransition, ...]
    frontier: ScoredFrontier
    ledger_digest: str
    goal_satisfied: bool = False
    coverage_target_met: bool = False
    # Claim bundles that newly passed the evidence gate this round.  Admissions
    # are progress on their own: a round can pass gates and commit receipts
    # without any insight in the ledger changing state.
    admitted_bundle_count: int = 0

    def __post_init__(self) -> None:
        if not self.ledger_digest:
            raise ValueError("ledger_digest cannot be empty.")
        if isinstance(self.admitted_bundle_count, bool) or self.admitted_bundle_count < 0:
            raise ValueError("admitted_bundle_count must be a non-negative integer.")


def reduction_outcome_digest(value: ReductionOutcome) -> str:
    return "reduction_" + stable_hash(
        {
            "transitions": value.transitions,
            "frontier": {
                "digest": value.frontier.digest,
                "items": [
                    {
                        "hypothesis_id": item.hypothesis_id,
                        "priority": item.priority,
                    }
                    for item in value.frontier.items
                ],
            },
            "ledger_digest": value.ledger_digest,
            "goal_satisfied": value.goal_satisfied,
            "coverage_target_met": value.coverage_target_met,
            "admitted_bundle_count": value.admitted_bundle_count,
        },
        length=32,
    )


@dataclass(frozen=True, slots=True)
class FinalizationOutcome:
    report_ref: str | None


@dataclass(frozen=True, slots=True)
class SupervisorRunResult:
    status: Literal["paused", "stopped"]
    phase: SupervisorPhase
    stop_reason: ExplorationStopReason | None
    report_ref: str | None
    transitions: tuple[PhaseTransition, ...]
    rounds_started: int
    rounds_settled: int
    error: str | None = None


class SupervisorJournalPort(Protocol):
    """Append-only journal adapter used by the control plane."""

    def snapshot(self) -> SupervisorJournalState: ...

    def start_round(
        self, round_index: int, *, branch_id: str | None = None
    ) -> SupervisorJournalState: ...

    def abandon_branch(
        self,
        *,
        branch_id: str,
        round_index: int,
        constraints: tuple[BranchConstraint, ...],
    ) -> SupervisorJournalState: ...

    def commit_reduction(
        self, *, frontier_digest: str, ledger_digest: str, reduction_digest: str
    ) -> SupervisorJournalState: ...

    def settle_round(
        self,
        round_index: int,
        *,
        progress: bool,
        terminal_reason: ExplorationGracefulStopReason | None,
        frontier_empty: bool = False,
        adjudicated_transitions: int = 0,
    ) -> SupervisorJournalState: ...

    def mark_paused(self) -> SupervisorJournalState: ...

    def stop(
        self, reason: ExplorationStopReason, *, report_ref: str | None
    ) -> SupervisorJournalState: ...


class WitnessPort(Protocol):
    def recheck(self, expected_witness: str) -> bool: ...


class BudgetPort(Protocol):
    """Combines wall/idle/LLM/tool checks without coupling their concrete ledgers."""

    def check_start_round(self, state: SupervisorJournalState) -> None: ...

    def remaining(self, state: SupervisorJournalState) -> Mapping[str, object]: ...


class ControlPort(Protocol):
    def checkpoint(self, transition: PhaseTransition) -> None: ...


class BranchConstraintPort(Protocol):
    """Deterministic "tried + why it failed" derivation for one abandonment."""

    def derive(self, context: PhaseContext) -> tuple[BranchConstraint, ...]: ...


class CandidateGeneratorPort(Protocol):
    """Must durably complete ``logical_step_id`` before returning."""

    def generate(self, context: PhaseContext, *, logical_step_id: str) -> CandidateBatch: ...


class SchedulerPort(Protocol):
    def admit_and_score(
        self, context: PhaseContext, candidates: CandidateBatch
    ) -> ScoredFrontier: ...

    def select(
        self, context: PhaseContext, frontier: ScoredFrontier
    ) -> ProbeSelection | None: ...


class ProbeExecutorPort(Protocol):
    """Resumes from ``context.completed_step_ids``; completed probes are adopted."""

    def execute(self, context: PhaseContext, selection: ProbeSelection) -> ProbeOutcome: ...


class ValidatorPort(Protocol):
    def validate(self, context: PhaseContext, probes: ProbeOutcome) -> ValidationOutcome: ...


class ReducerPort(Protocol):
    """May use a model proposal, but must adopt a completed logical step."""

    def reduce(
        self,
        context: PhaseContext,
        validated: ValidationOutcome,
        frontier: ScoredFrontier,
        *,
        logical_step_id: str,
    ) -> ReductionOutcome: ...

    def reduce_without_probes(
        self,
        context: PhaseContext,
        frontier: ScoredFrontier,
        *,
        logical_step_id: str,
    ) -> ReductionOutcome: ...


class FinalizerPort(Protocol):
    """Synthesis is optional polish; deterministic rendering is the safe fallback."""

    def synthesize(
        self,
        context: PhaseContext,
        reduction: ReductionOutcome,
        *,
        logical_step_id: str,
    ) -> FinalizationOutcome: ...

    def render_deterministic(
        self, context: PhaseContext, reduction: ReductionOutcome
    ) -> FinalizationOutcome: ...


class CompletedStepRecoveryPort(Protocol):
    """Durable phase-result storage keyed by stable logical step id."""

    def load_required(self, logical_step_id: str) -> object: ...

    def remember(self, logical_step_id: str, result: object) -> None: ...


class SupervisorCancelled(RuntimeError):
    """Raised by a cancel checkpoint or a bounded component."""


class SupervisorPauseRequested(RuntimeError):
    """Raised only after pause_requested has been journaled externally."""


class SupervisorBudgetExhausted(RuntimeError):
    """Raised by BudgetPort or a bounded component when a hard cap latches."""


class SynthesisUnavailable(RuntimeError):
    """Safe signal: use the deterministic renderer instead of failing the run."""


class SupervisorInvariantError(RuntimeError):
    """A dependency violated its durable/fail-closed contract."""


@dataclass(slots=True)
class _Cursor:
    phase: SupervisorPhase
    transitions: list[PhaseTransition]
    context: PhaseContext | None = None
    reduction: ReductionOutcome | None = None


def phase_step_id(
    exploration_id: str, round_index: int, phase: SupervisorPhase
) -> str:
    """Stable id used to adopt paid/reduction steps after a crash."""
    return f"{exploration_id}:round:{round_index}:{phase.value.lower()}"


def render_soft_countdown(remaining: Mapping[str, object]) -> str:
    """Fixed, deterministic system context injected at ORIENT for the full round."""
    if not all(isinstance(key, str) and key for key in remaining):
        raise ValueError("budget countdown keys must be non-empty strings.")
    try:
        encoded = json.dumps(
            dict(remaining),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            default=_json_scalar,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("remaining budget must be deterministically serializable.") from exc
    return (
        "[exploration_soft_countdown] remaining="
        f"{encoded}. Prefer the highest-value admissible probe and preserve enough "
        "budget to finish gracefully."
    )


def _json_scalar(value: object) -> str:
    # Decimal and other scalar wrappers have deterministic string forms.  Containers
    # and arbitrary domain objects must not silently enter governance context.
    if value.__class__.__module__ in {"decimal", "enum"}:
        return str(value)
    raise TypeError(f"unsupported countdown value {type(value).__name__}")


class ExplorationSupervisor:
    """Single-authority E4a state machine with fail-closed recovery semantics."""

    def __init__(
        self,
        *,
        config: SupervisorConfig,
        journal: SupervisorJournalPort,
        witness: WitnessPort,
        budget: BudgetPort,
        control: ControlPort,
        generator: CandidateGeneratorPort,
        scheduler: SchedulerPort,
        executor: ProbeExecutorPort,
        validator: ValidatorPort,
        reducer: ReducerPort,
        finalizer: FinalizerPort,
        recovery: CompletedStepRecoveryPort,
        branch_deriver: BranchConstraintPort | None = None,
    ) -> None:
        if config.branch_trigger_stagnant_rounds is not None and branch_deriver is None:
            raise ValueError("branch mode requires a branch constraint deriver.")
        self._config = config
        self._journal = journal
        self._witness = witness
        self._budget = budget
        self._control = control
        self._generator = generator
        self._scheduler = scheduler
        self._executor = executor
        self._validator = validator
        self._reducer = reducer
        self._finalizer = finalizer
        self._recovery = recovery
        self._branch_deriver = branch_deriver

    def run(self) -> SupervisorRunResult:
        """Run until a durable terminal event or a resumable pause is reached."""
        initial = self._journal.snapshot()
        if initial.status == "stopped":
            return self._result_from_state(initial, (), error=None)
        if initial.status in {"pause_requested", "paused"}:
            return self._pause_result(initial, ())

        cursor = _Cursor(phase=SupervisorPhase.ORIENT, transitions=[])
        try:
            self._enter(cursor, SupervisorPhase.ORIENT, source=None, round_index=None)
            return self._drive(cursor)
        except SupervisorPauseRequested:
            return self._pause_result(self._journal.snapshot(), cursor.transitions)
        except SupervisorCancelled:
            return self._abort(cursor, "cancelled", error=None)
        except _WitnessChanged:
            return self._terminal(cursor, "state_witness_changed", error=None)
        except SupervisorBudgetExhausted as exc:
            return self._budget_exhausted_terminal(cursor, error=str(exc) or None)
        except Exception as exc:
            # Any unclassified dependency/recovery fault is terminal.  No partial
            # result is presented as a successful/no-information exploration.
            return self._terminal(
                cursor,
                "failed",
                error=f"{type(exc).__name__}: {exc}",
            )

    def _drive(self, cursor: _Cursor) -> SupervisorRunResult:
        while True:
            settled = self._journal.snapshot()
            if settled.pending_terminal_reason is not None:
                return self._resume_settled_terminal(cursor, settled)
            state, context = self._orient(cursor)
            cursor.context = context

            if state.current_round_reduction_committed:
                self._move(cursor, SupervisorPhase.REDUCE)
                reduction = self._recover_reduction(context)
                cursor.reduction = reduction
                result = self._settle_reduction(cursor, reduction)
                if result is not None:
                    return result
                continue

            self._move(cursor, SupervisorPhase.GENERATE)
            candidates = self._generate(context.for_phase(SupervisorPhase.GENERATE))

            self._move(cursor, SupervisorPhase.ADMIT_AND_SCORE)
            frontier = self._scheduler.admit_and_score(
                context.for_phase(SupervisorPhase.ADMIT_AND_SCORE), candidates
            )
            self._require_type(frontier, ScoredFrontier, "scheduler frontier")
            if not frontier.items:
                result = self._settle_empty_frontier(cursor, frontier)
                if result is not None:
                    return result
                continue

            self._move(cursor, SupervisorPhase.SELECT)
            selection = self._scheduler.select(
                context.for_phase(SupervisorPhase.SELECT), frontier
            )
            if selection is None:
                raise SupervisorInvariantError(
                    "scheduler returned no selection for a non-empty frontier."
                )
            self._require_type(selection, ProbeSelection, "scheduler selection")

            self._move(cursor, SupervisorPhase.EXECUTE_PROBES)
            probes = self._executor.execute(
                self._fresh_context(context, SupervisorPhase.EXECUTE_PROBES), selection
            )
            self._require_type(probes, ProbeOutcome, "probe outcome")

            self._move(cursor, SupervisorPhase.VALIDATE)
            validated = self._validator.validate(
                self._fresh_context(context, SupervisorPhase.VALIDATE), probes
            )
            self._require_type(validated, ValidationOutcome, "validation outcome")
            if validated.validator_exhausted:
                raise SupervisorInvariantError("validator retry budget exhausted.")

            self._move(cursor, SupervisorPhase.REDUCE)
            reduction = self._reduce(
                self._fresh_context(context, SupervisorPhase.REDUCE),
                validated,
                frontier,
            )
            cursor.reduction = reduction
            result = self._settle_reduction(cursor, reduction)
            if result is not None:
                return result

    def _orient(
        self, cursor: _Cursor
    ) -> tuple[SupervisorJournalState, PhaseContext]:
        state = self._journal.snapshot()
        self._honor_journal_control(state)
        if state.remaining_round_budget <= 0 and state.current_round_index is None:
            raise SupervisorBudgetExhausted("max_rounds exhausted")
        self._budget.check_start_round(state)

        # Recheck on every ORIENT, including adoption of an already-open round after
        # resume.  A mismatch occurs before round_started or any component side effect.
        if not self._witness.recheck(state.data_state_witness):
            return self._raise_witness_changed()

        if state.current_round_index is None:
            if state.current_line_abandoned:
                branch_id = f"br_{state.branches_started + 1}"
            else:
                branch_id = state.active_branch_id
            state = self._journal.start_round(state.rounds_started, branch_id=branch_id)
        round_index = state.current_round_index
        if round_index is None:
            raise SupervisorInvariantError("journal did not open the requested round.")
        reminder = render_soft_countdown(self._budget.remaining(state))
        return state, PhaseContext(
            exploration_id=state.exploration_id,
            round_index=round_index,
            phase=SupervisorPhase.ORIENT,
            data_state_witness=state.data_state_witness,
            soft_countdown_context=reminder,
            completed_step_ids=state.completed_step_ids,
        )

    def _generate(self, context: PhaseContext) -> CandidateBatch:
        step_id = phase_step_id(
            context.exploration_id, context.round_index, SupervisorPhase.GENERATE
        )
        state = self._journal.snapshot()
        if step_id in state.completed_step_ids:
            recovered = self._recovery.load_required(step_id)
            return self._require_type(
                recovered, CandidateBatch, "recovered candidate batch"
            )
        result = self._generator.generate(context, logical_step_id=step_id)
        self._require_type(result, CandidateBatch, "candidate batch")
        if step_id not in self._journal.snapshot().completed_step_ids:
            raise SupervisorInvariantError(
                "generator returned before its logical step was durably completed."
            )
        self._recovery.remember(step_id, result)
        return result

    def _reduce(
        self,
        context: PhaseContext,
        validated: ValidationOutcome,
        frontier: ScoredFrontier,
    ) -> ReductionOutcome:
        step_id = phase_step_id(
            context.exploration_id, context.round_index, SupervisorPhase.REDUCE
        )
        result = self._reducer.reduce(
            context,
            validated,
            frontier,
            logical_step_id=step_id,
        )
        self._require_type(result, ReductionOutcome, "reduction outcome")
        # Persist the reconstructable outcome before the journal commit.  If the
        # process dies before commit, deterministic reduction is rerun; after commit,
        # the saved outcome is adopted and the reducer is not called again.
        self._recovery.remember(step_id, result)
        state = self._journal.commit_reduction(
            frontier_digest=result.frontier.digest,
            ledger_digest=result.ledger_digest,
            reduction_digest=reduction_outcome_digest(result),
        )
        if not state.current_round_reduction_committed:
            raise SupervisorInvariantError("journal did not commit reduction.")
        return result

    def _reduce_without_probes(
        self, context: PhaseContext, frontier: ScoredFrontier
    ) -> ReductionOutcome:
        step_id = phase_step_id(
            context.exploration_id, context.round_index, SupervisorPhase.REDUCE
        )
        result = self._reducer.reduce_without_probes(
            context, frontier, logical_step_id=step_id
        )
        self._require_type(result, ReductionOutcome, "probe-free reduction outcome")
        if result.transitions:
            raise SupervisorInvariantError(
                "a probe-free reduction cannot report insight transitions."
            )
        if result.admitted_bundle_count:
            raise SupervisorInvariantError(
                "a probe-free reduction cannot admit claim bundles."
            )
        if result.frontier is not frontier:
            raise SupervisorInvariantError(
                "a probe-free reduction must carry the scheduler's frontier."
            )
        self._recovery.remember(step_id, result)
        state = self._journal.commit_reduction(
            frontier_digest=result.frontier.digest,
            ledger_digest=result.ledger_digest,
            reduction_digest=reduction_outcome_digest(result),
        )
        if not state.current_round_reduction_committed:
            raise SupervisorInvariantError("journal did not commit reduction.")
        return result

    def _recover_reduction(self, context: PhaseContext) -> ReductionOutcome:
        step_id = phase_step_id(
            context.exploration_id, context.round_index, SupervisorPhase.REDUCE
        )
        result = self._recovery.load_required(step_id)
        recovered = self._require_type(
            result, ReductionOutcome, "recovered reduction outcome"
        )
        state = self._journal.snapshot()
        if (
            state.frontier_digest != recovered.frontier.digest
            or state.ledger_digest != recovered.ledger_digest
            or state.reduction_digest != reduction_outcome_digest(recovered)
        ):
            raise SupervisorInvariantError(
                "recovered reduction digests do not match the journal commit."
            )
        return recovered
    def _settle_reduction(
        self, cursor: _Cursor, reduction: ReductionOutcome
    ) -> SupervisorRunResult | None:
        context = self._require_context(cursor)
        before_settle = self._journal.snapshot()
        adjudicated_transitions = sum(
            1
            for transition in reduction.transitions
            if transition in _PROGRESS_TRANSITIONS
        )
        progress = bool(
            before_settle.current_round_receipt_ids
            and (adjudicated_transitions > 0 or reduction.admitted_bundle_count > 0)
        )
        terminal_reason, should_branch = self._settle_decision(
            before_settle,
            progress=progress,
            adjudicated_transitions=adjudicated_transitions,
            frontier=reduction.frontier,
            goal_satisfied=(
                reduction.goal_satisfied or reduction.coverage_target_met
            ),
        )
        self._journal.settle_round(
            context.round_index,
            progress=progress,
            terminal_reason=terminal_reason,
            frontier_empty=not reduction.frontier.items,
            adjudicated_transitions=adjudicated_transitions,
        )

        if terminal_reason is not None:
            return self._graceful_terminal(cursor, terminal_reason, reduction)
        if should_branch:
            self._abandon_current_line(cursor)
        self._move(cursor, SupervisorPhase.ORIENT)
        cursor.context = None
        cursor.reduction = None
        return None

    def _settle_empty_frontier(
        self, cursor: _Cursor, frontier: ScoredFrontier
    ) -> SupervisorRunResult | None:
        """Commit a probe-free reduction, then settle like any other round.

        The round still produced scheduling decisions, and ``frontier_digest``
        on the reduction event is the only thing that binds them to the journal;
        settling without one leaves them unverifiable (the E4a issuer rejects
        such a root). Routing through the normal settle path also keeps the
        finalizer — and therefore the report artifact — on the terminal edge.
        """
        context = self._require_context(cursor)
        self._move(cursor, SupervisorPhase.REDUCE)
        reduction = self._reduce_without_probes(
            self._fresh_context(context, SupervisorPhase.REDUCE), frontier
        )
        cursor.reduction = reduction
        return self._settle_reduction(cursor, reduction)

    def _settle_decision(
        self,
        state: SupervisorJournalState,
        *,
        progress: bool,
        adjudicated_transitions: int,
        frontier: ScoredFrontier,
        goal_satisfied: bool,
    ) -> tuple[ExplorationGracefulStopReason | None, bool]:
        """Terminal reason plus whether to switch to a new branch instead.

        Branch mode is only ever entered from the system stagnation signal; no
        component or provider input reaches this decision (plan E6 gate 1).
        """
        if goal_satisfied:
            return "completed", False
        # Predicted exactly as the journal will record it, so the decision and
        # the durable counters can never disagree by one round.
        consecutive_no_progress = 0 if progress else state.consecutive_no_progress + 1
        consecutive_empty_frontier = (
            state.consecutive_empty_frontier + 1 if not frontier.items else 0
        )
        highest = frontier.highest_priority
        # "No candidate" is not "every candidate is worthless": a single empty
        # frontier is a scheduling gap, and only a repeated one is exhaustion.
        below_admission = (
            highest is not None and highest < self._config.admission_score_threshold
        )
        line_exhausted = below_admission or consecutive_empty_frontier >= 2
        trigger = self._config.branch_trigger_stagnant_rounds
        # Plan-B soft stop (user decision 2026-08-03): gate admissions still
        # count as progress, but two straight rounds without one adjudicated
        # (new/reinforced/refuted) transition end the run — the seed-6 mode of
        # burning to the hard cap on admissions alone. Branch-enabled runs
        # keep the E6 stagnation machinery authoritative instead.
        consecutive_no_adjudication = (
            0
            if adjudicated_transitions > 0
            else state.consecutive_no_adjudication + 1
        )
        if trigger is None and consecutive_no_adjudication >= 2:
            return "no_new_information", False
        if trigger is None:
            if consecutive_no_progress >= 2 and line_exhausted:
                return "no_new_information", False
        elif consecutive_no_progress >= trigger and line_exhausted:
            branches_remain = state.branches_started < self._config.max_branches
            if branches_remain and state.remaining_round_budget > 0:
                return None, True
            if not branches_remain:
                return "no_new_information", False
        if state.remaining_round_budget <= 0:
            return "budget_exhausted", False
        return None, False

    def _abandon_current_line(self, cursor: _Cursor) -> None:
        context = self._require_context(cursor)
        deriver = self._branch_deriver
        if deriver is None:
            raise SupervisorInvariantError(
                "branch mode requires a branch constraint deriver."
            )
        state = self._journal.snapshot()
        line_id = state.active_branch_id or MAIN_LINE_ID
        constraints = deriver.derive(context)
        if not isinstance(constraints, tuple) or not all(
            isinstance(item, BranchConstraint) for item in constraints
        ):
            raise SupervisorInvariantError(
                "branch constraints must be a tuple of BranchConstraint."
            )
        after = self._journal.abandon_branch(
            branch_id=line_id,
            round_index=context.round_index,
            constraints=constraints,
        )
        if not after.current_line_abandoned:
            raise SupervisorInvariantError("journal did not record the abandonment.")

    def _resume_settled_terminal(
        self,
        cursor: _Cursor,
        state: SupervisorJournalState,
    ) -> SupervisorRunResult:
        reason = state.pending_terminal_reason
        if reason is None or state.rounds_settled <= 0:
            raise SupervisorInvariantError("invalid pending terminal projection.")
        if not self._witness.recheck(state.data_state_witness):
            raise _WitnessChanged
        round_index = state.rounds_settled - 1
        context = PhaseContext(
            exploration_id=state.exploration_id,
            round_index=round_index,
            phase=SupervisorPhase.ORIENT,
            data_state_witness=state.data_state_witness,
            soft_countdown_context=render_soft_countdown(self._budget.remaining(state)),
            completed_step_ids=state.completed_step_ids,
        )
        cursor.context = context
        if not state.pending_terminal_has_reduction:
            return self._terminal(cursor, reason, error=None)
        self._move(cursor, SupervisorPhase.REDUCE)
        reduction = self._recover_reduction(context)
        cursor.reduction = reduction
        return self._graceful_terminal(cursor, reason, reduction)

    def _budget_exhausted_terminal(
        self,
        cursor: _Cursor,
        *,
        error: str | None,
    ) -> SupervisorRunResult:
        """Budget death keeps the report the max_rounds path would have produced.

        Synthesis is skipped on purpose — the budget that just latched is the
        same one synthesis would spend — so the deterministic renderer runs
        instead.  A reduction that cannot be recovered falls back to the bare
        terminal rather than turning a budget stop into a failure.
        """
        state = self._journal.snapshot()
        if state.rounds_settled > 0:
            try:
                context = PhaseContext(
                    exploration_id=state.exploration_id,
                    round_index=state.rounds_settled - 1,
                    phase=SupervisorPhase.ORIENT,
                    data_state_witness=state.data_state_witness,
                    soft_countdown_context=render_soft_countdown(
                        self._budget.remaining(state)
                    ),
                    completed_step_ids=state.completed_step_ids,
                    # A mid-round latch leaves no settled terminal reason in
                    # the journal; the renderer accepts this one instead.
                    terminal_reason="budget_exhausted",
                )
                reduction = self._recover_reduction(context)
                cursor.context = context
                cursor.reduction = reduction
                return self._graceful_terminal(
                    cursor,
                    "budget_exhausted",
                    reduction,
                    deterministic_only=True,
                    error=error,
                )
            except Exception:
                cursor.context = None
                cursor.reduction = None
        return self._terminal(cursor, "budget_exhausted", error=error)

    def _graceful_terminal(
        self,
        cursor: _Cursor,
        reason: ExplorationStopReason,
        reduction: ReductionOutcome,
        *,
        deterministic_only: bool = False,
        error: str | None = None,
    ) -> SupervisorRunResult:
        if not (self._config.synthesize_on_graceful_stop or deterministic_only):
            return self._terminal(cursor, reason, error=error)
        context = self._require_context(cursor)
        self._move(cursor, SupervisorPhase.SYNTHESIZE)
        synth_context = self._fresh_context(context, SupervisorPhase.SYNTHESIZE)
        if deterministic_only:
            outcome = self._require_type(
                self._finalizer.render_deterministic(synth_context, reduction),
                FinalizationOutcome,
                "deterministic finalization outcome",
            )
            return self._terminal(
                cursor, reason, error=error, report_ref=outcome.report_ref
            )
        step_id = phase_step_id(
            context.exploration_id, context.round_index, SupervisorPhase.SYNTHESIZE
        )
        state = self._journal.snapshot()
        try:
            if step_id in state.completed_step_ids:
                outcome = self._require_type(
                    self._recovery.load_required(step_id),
                    FinalizationOutcome,
                    "recovered finalization outcome",
                )
            else:
                outcome = self._finalizer.synthesize(
                    synth_context,
                    reduction,
                    logical_step_id=step_id,
                )
                self._require_type(outcome, FinalizationOutcome, "finalization outcome")
                if step_id not in self._journal.snapshot().completed_step_ids:
                    raise SupervisorInvariantError(
                        "synthesizer returned before its logical step was completed."
                    )
                self._recovery.remember(step_id, outcome)
        except SynthesisUnavailable:
            outcome = self._finalizer.render_deterministic(synth_context, reduction)
            self._require_type(
                outcome, FinalizationOutcome, "deterministic finalization outcome"
            )
        return self._terminal(
            cursor,
            reason,
            error=error,
            report_ref=outcome.report_ref,
        )

    def _terminal(
        self,
        cursor: _Cursor,
        reason: ExplorationStopReason,
        *,
        error: str | None,
        report_ref: str | None = None,
    ) -> SupervisorRunResult:
        self._move(cursor, SupervisorPhase.STOP)
        state = self._journal.stop(reason, report_ref=report_ref)
        return self._result_from_state(state, cursor.transitions, error=error)

    def _abort(
        self,
        cursor: _Cursor,
        reason: Literal["cancelled"],
        *,
        error: str | None,
    ) -> SupervisorRunResult:
        # Cancellation was itself observed at a checkpoint (or propagated by a
        # bounded component), so do not call a second checkpoint that can mask it.
        if cursor.phase is not SupervisorPhase.STOP:
            cursor.transitions.append(
                PhaseTransition(
                    source=cursor.phase,
                    target=SupervisorPhase.STOP,
                    round_index=(
                        None if cursor.context is None else cursor.context.round_index
                    ),
                )
            )
            cursor.phase = SupervisorPhase.STOP
        state = self._journal.stop(reason, report_ref=None)
        return self._result_from_state(state, cursor.transitions, error=error)

    def _pause_result(
        self,
        state: SupervisorJournalState,
        transitions: list[PhaseTransition] | tuple[PhaseTransition, ...],
    ) -> SupervisorRunResult:
        if state.status == "pause_requested":
            state = self._journal.mark_paused()
        if state.status != "paused":
            raise SupervisorInvariantError(
                "pause signal requires journal status pause_requested or paused."
            )
        return SupervisorRunResult(
            status="paused",
            phase=SupervisorPhase.PAUSED,
            stop_reason=None,
            report_ref=None,
            transitions=tuple(transitions),
            rounds_started=state.rounds_started,
            rounds_settled=state.rounds_settled,
        )

    def _move(self, cursor: _Cursor, target: SupervisorPhase) -> None:
        context = cursor.context
        self._enter(
            cursor,
            target,
            source=cursor.phase,
            round_index=None if context is None else context.round_index,
        )

    def _enter(
        self,
        cursor: _Cursor,
        target: SupervisorPhase,
        *,
        source: SupervisorPhase | None,
        round_index: int | None,
    ) -> None:
        transition = PhaseTransition(source=source, target=target, round_index=round_index)
        self._control.checkpoint(transition)
        self._honor_journal_control(self._journal.snapshot())
        cursor.transitions.append(transition)
        cursor.phase = target

    @staticmethod
    def _raise_witness_changed() -> tuple[SupervisorJournalState, PhaseContext]:
        raise _WitnessChanged

    def _honor_journal_control(self, state: SupervisorJournalState) -> None:
        if state.status in {"pause_requested", "paused"}:
            raise SupervisorPauseRequested
        if state.status == "stopped":
            if state.stop_reason == "cancelled":
                raise SupervisorCancelled
            raise SupervisorInvariantError(
                f"journal stopped unexpectedly with {state.stop_reason!r}."
            )

    def _fresh_context(
        self, base: PhaseContext, phase: SupervisorPhase
    ) -> PhaseContext:
        state = self._journal.snapshot()
        return PhaseContext(
            exploration_id=base.exploration_id,
            round_index=base.round_index,
            phase=phase,
            data_state_witness=base.data_state_witness,
            soft_countdown_context=base.soft_countdown_context,
            completed_step_ids=state.completed_step_ids,
            terminal_reason=base.terminal_reason,
        )

    @staticmethod
    def _require_context(cursor: _Cursor) -> PhaseContext:
        if cursor.context is None:
            raise SupervisorInvariantError("supervisor context is unavailable.")
        return cursor.context

    @staticmethod
    def _require_type[T](value: object, expected: type[T], label: str) -> T:
        if not isinstance(value, expected):
            raise SupervisorInvariantError(
                f"{label} must be {expected.__name__}, got {type(value).__name__}."
            )
        return value

    @staticmethod
    def _result_from_state(
        state: SupervisorJournalState,
        transitions: list[PhaseTransition] | tuple[PhaseTransition, ...],
        *,
        error: str | None,
    ) -> SupervisorRunResult:
        if state.status != "stopped" or state.stop_reason is None:
            raise SupervisorInvariantError("terminal result requires stopped journal state.")
        return SupervisorRunResult(
            status="stopped",
            phase=SupervisorPhase.STOP,
            stop_reason=state.stop_reason,
            report_ref=state.final_report_ref,
            transitions=tuple(transitions),
            rounds_started=state.rounds_started,
            rounds_settled=state.rounds_settled,
            error=error,
        )


class _WitnessChanged(RuntimeError):
    pass
