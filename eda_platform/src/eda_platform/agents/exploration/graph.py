"""Checkpointed exploration phases with native LangGraph pause/resume."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt

from eda_platform.agents.exploration.supervisor import (
    CandidateBatch,
    ExplorationRunChannels,
    ExplorationSupervisor,
    FrontierItem,
    PhaseContext,
    PhaseTransition,
    ProbeOutcome,
    ProbeSelection,
    ReductionOutcome,
    ScoredFrontier,
    SupervisorBudgetExhausted,
    SupervisorCancelled,
    SupervisorInvariantError,
    SupervisorJournalState,
    SupervisorPauseRequested,
    SupervisorPhase,
    SupervisorRunResult,
    ValidationOutcome,
    _WitnessChanged,
)
from eda_platform.core.cancellation import CancellationError
from eda_platform.core.graph_execution import graph_execution
from eda_platform.core.kernel import SessionCancelled


class ExplorationState(ExplorationRunChannels):
    candidates: CandidateBatch | None
    frontier: ScoredFrontier | None
    selection: ProbeSelection | None
    probes: ProbeOutcome | None
    validated: ValidationOutcome | None
    round_state: SupervisorJournalState | None
    next_node: str
    resume_node: str
    reason: str | None
    result: SupervisorRunResult | None


@dataclass
class ExplorationServices:
    owner: ExplorationSupervisor


def _run_channels(state: ExplorationRunChannels) -> ExplorationRunChannels:
    return {
        "phase": state["phase"],
        "transitions": state["transitions"],
        "context": state["context"],
        "reduction": state["reduction"],
    }


def _owner(runtime: Runtime[ExplorationServices]) -> ExplorationSupervisor:
    return runtime.context.owner


def _orient(state: ExplorationState, runtime: Runtime[ExplorationServices]) -> dict[str, Any]:
    owner = _owner(runtime)
    run_state = state
    journal = owner._journal.snapshot()
    if journal.pending_terminal_reason is not None:
        result = owner._resume_settled_terminal(run_state, journal)
        return {"result": result, **_run_channels(run_state), "next_node": END}
    if not run_state["transitions"]:
        owner._record_transition(run_state, SupervisorPhase.ORIENT, source=None, round_index=None)
    journal, context = owner._open_round()
    run_state["context"] = context
    return {
        **_run_channels(run_state),
        "next_node": "reduce" if journal.current_round_reduction_committed else "generate",
        "candidates": None,
        "frontier": None,
        "selection": None,
        "probes": None,
        "validated": None,
        "result": None,
        "reason": None,
    }


def _generate(state: ExplorationState, runtime: Runtime[ExplorationServices]) -> dict[str, Any]:
    owner, run_state = _owner(runtime), state
    owner._record_phase(run_state, SupervisorPhase.GENERATE)
    context = owner._require_context(run_state).for_phase(SupervisorPhase.GENERATE)
    return {
        "candidates": owner._generate(context), **_run_channels(run_state), "next_node": "schedule"
    }


def _schedule(state: ExplorationState, runtime: Runtime[ExplorationServices]) -> dict[str, Any]:
    owner, run_state = _owner(runtime), state
    owner._record_phase(run_state, SupervisorPhase.ADMIT_AND_SCORE)
    context = owner._require_context(run_state).for_phase(SupervisorPhase.ADMIT_AND_SCORE)
    frontier = owner._scheduler.admit_and_score(context, cast(CandidateBatch, state["candidates"]))
    owner._require_type(frontier, ScoredFrontier, "scheduler frontier")
    return {
        "frontier": frontier,
        "candidates": None,
        **_run_channels(run_state),
        "next_node": "select" if frontier.items else "reduce",
    }


def _select(state: ExplorationState, runtime: Runtime[ExplorationServices]) -> dict[str, Any]:
    owner, run_state = _owner(runtime), state
    owner._record_phase(run_state, SupervisorPhase.SELECT)
    selection = owner._scheduler.select(
        owner._require_context(run_state).for_phase(SupervisorPhase.SELECT),
        cast(ScoredFrontier, state["frontier"]),
    )
    if selection is None:
        raise SupervisorInvariantError("scheduler returned no selection for a non-empty frontier.")
    owner._require_type(selection, ProbeSelection, "scheduler selection")
    return {"selection": selection, **_run_channels(run_state), "next_node": "execute_probes"}


def _execute(state: ExplorationState, runtime: Runtime[ExplorationServices]) -> dict[str, Any]:
    owner, run_state = _owner(runtime), state
    owner._record_phase(run_state, SupervisorPhase.EXECUTE_PROBES)
    probes = owner._executor.execute(
        owner._fresh_context(owner._require_context(run_state), SupervisorPhase.EXECUTE_PROBES),
        cast(ProbeSelection, state["selection"]),
    )
    owner._require_type(probes, ProbeOutcome, "probe outcome")
    return {
        "probes": probes, "selection": None, **_run_channels(run_state), "next_node": "validate",
        "reason": "budget_exhausted" if probes.budget_exhausted else None,
    }


def _validate(state: ExplorationState, runtime: Runtime[ExplorationServices]) -> dict[str, Any]:
    owner, run_state = _owner(runtime), state
    owner._record_phase(run_state, SupervisorPhase.VALIDATE)
    validated = owner._validator.validate(
        owner._fresh_context(owner._require_context(run_state), SupervisorPhase.VALIDATE),
        cast(ProbeOutcome, state["probes"]),
    )
    owner._require_type(validated, ValidationOutcome, "validation outcome")
    if validated.validator_exhausted:
        raise SupervisorInvariantError("validator retry budget exhausted.")
    return {
        "validated": validated, "probes": None, **_run_channels(run_state), "next_node": "reduce"
    }


def _reduce(state: ExplorationState, runtime: Runtime[ExplorationServices]) -> dict[str, Any]:
    owner, run_state = _owner(runtime), state
    owner._record_phase(run_state, SupervisorPhase.REDUCE)
    context = owner._fresh_context(owner._require_context(run_state), SupervisorPhase.REDUCE)
    journal = owner._journal.snapshot()
    if journal.current_round_reduction_committed:
        reduction = owner._recover_reduction(context)
    elif not cast(ScoredFrontier, state["frontier"]).items:
        reduction = owner._reduce_without_probes(context, cast(ScoredFrontier, state["frontier"]))
    else:
        reduction = owner._reduce(
            context,
            cast(ValidationOutcome, state["validated"]),
            cast(ScoredFrontier, state["frontier"]),
        )
    run_state["reduction"] = reduction
    return {
        **_run_channels(run_state),
        "round_state": owner._journal.snapshot(),
        "next_node": "settle",
        "probes": None,
        "validated": None,
        "selection": None,
        "candidates": None,
    }


def _settle(state: ExplorationState, runtime: Runtime[ExplorationServices]) -> dict[str, Any]:
    owner, run_state = _owner(runtime), state
    context = owner._require_context(run_state)
    reduction = cast(ReductionOutcome, run_state["reduction"])
    before = cast(SupervisorJournalState, state["round_state"])
    adjudicated = sum(t in {"new", "reinforced", "refuted"} for t in reduction.transitions)
    supported = sum(t in {"new", "reinforced"} for t in reduction.transitions)
    progress = bool(
        before.current_round_receipt_ids
        and (adjudicated > 0 or reduction.admitted_bundle_count > 0)
    )
    if state.get("reason") == "budget_exhausted":
        reason, branch = "budget_exhausted", False
    else:
        reason, branch = owner._settle_decision(
            before,
            progress=progress,
            adjudicated_transitions=adjudicated,
            frontier=reduction.frontier,
            goal_satisfied=reduction.goal_satisfied or reduction.coverage_target_met,
        )
    current = owner._journal.snapshot()
    if current.rounds_settled <= context.round_index:
        owner._journal.settle_round(
            context.round_index,
            progress=progress,
            terminal_reason=reason,
            frontier_empty=not reduction.frontier.items,
            adjudicated_transitions=adjudicated,
            supported_transitions=supported,
            llm_calls_at_settle=before.llm_calls_settled,
            tool_calls_at_settle=before.tool_calls_committed,
        )
    if reason is not None:
        return {"reason": reason, **_run_channels(run_state), "next_node": "synthesize"}
    if branch and not owner._journal.snapshot().current_line_abandoned:
        owner._abandon_current_line(run_state)
    owner._record_phase(run_state, SupervisorPhase.ORIENT)
    run_state["context"], run_state["reduction"] = None, None
    return {**_run_channels(run_state), "next_node": "orient", "round_state": None}


def _synthesize(state: ExplorationState, runtime: Runtime[ExplorationServices]) -> dict[str, Any]:
    owner, run_state = _owner(runtime), state
    result = owner._graceful_terminal(
        run_state, cast(Any, state["reason"]), cast(ReductionOutcome, run_state["reduction"]),
        deterministic_only=state["reason"] == "budget_exhausted",
    )
    return {"result": result, **_run_channels(run_state), "next_node": END}


def _pause(state: ExplorationState) -> dict[str, Any]:
    interrupt({"kind": "exploration_paused", "phase": state["resume_node"]})
    return {"result": None, "next_node": state["resume_node"]}


def _guard(name: str, action: Any) -> Any:
    def node(state: ExplorationState, runtime: Runtime[ExplorationServices]) -> dict[str, Any]:
        working = cast(ExplorationState, {**state, **deepcopy(_run_channels(state))})
        owner, run_state = _owner(runtime), working
        try:
            try:
                return action(working, runtime)
            except (
                SupervisorPauseRequested, SupervisorCancelled, SessionCancelled, CancellationError
            ):
                raise
            except _WitnessChanged:
                result = owner._terminal(run_state, "state_witness_changed", error=None)
            except SupervisorBudgetExhausted as exc:
                result = owner._budget_exhausted_terminal(run_state, error=str(exc) or None)
            except Exception as exc:
                # A control request can arrive inside a phase, including between
                # a provider response and journal admission of its next tool.
                # The journal blocks new work; translate that rejection before
                # treating it as a defect. Terminal handlers can observe the
                # same race, so their signals are handled by the outer guard.
                owner._honor_journal_control(owner._journal.snapshot())
                result = owner._terminal(run_state, "failed", error=f"{type(exc).__name__}: {exc}")
        except SupervisorPauseRequested:
            paused = owner._pause_result(owner._journal.snapshot(), run_state["transitions"])
            return {"result": paused, "next_node": "pause", "resume_node": name}
        except (SupervisorCancelled, SessionCancelled, CancellationError):
            result = owner._abort(run_state, "cancelled", error=None)
        return {"result": result, **_run_channels(run_state), "next_node": END}

    return node


def build_exploration_graph() -> StateGraph[ExplorationState, ExplorationServices]:
    builder = StateGraph(ExplorationState, context_schema=ExplorationServices)
    actions = {
        "orient": _orient,
        "generate": _generate,
        "schedule": _schedule,
        "select": _select,
        "execute_probes": _execute,
        "validate": _validate,
        "reduce": _reduce,
        "settle": _settle,
        "synthesize": _synthesize,
    }
    for name, action in actions.items():
        builder.add_node(name, _guard(name, action))
    builder.add_node("pause", _pause)
    for name in (*actions, "pause"):
        builder.add_conditional_edges(name, lambda state: state["next_node"])
    builder.add_edge(START, "orient")
    return builder


def _checkpoint_types() -> tuple[type, ...]:
    from eda_platform.agents.exploration.candidates import CandidateSeed
    from eda_platform.agents.exploration.executor import ProbeExecutionResult
    from eda_platform.agents.exploration.workflow import ExecutedProbeBatch
    from eda_platform.schemas.artifacts import Artifact, ArtifactType
    from eda_platform.schemas.exploration import InsightFamily
    from eda_platform.schemas.hypotheses import HypothesisProposal
    from eda_platform.schemas.receipts import EvidenceReceipt

    return (
        SupervisorPhase,
        PhaseContext,
        PhaseTransition,
        CandidateBatch,
        FrontierItem,
        ScoredFrontier,
        ProbeSelection,
        ProbeOutcome,
        ValidationOutcome,
        ReductionOutcome,
        SupervisorRunResult,
        SupervisorJournalState,
        CandidateSeed,
        ProbeExecutionResult,
        ExecutedProbeBatch,
        Artifact,
        ArtifactType,
        InsightFamily,
        HypothesisProposal,
        EvidenceReceipt,
    )


def run_exploration_graph(owner: ExplorationSupervisor) -> SupervisorRunResult:
    journal = owner._journal.snapshot()
    with graph_execution(
        owner._persistence,
        definition="exploration-v4",
        checkpoint_types=_checkpoint_types(),
        inputs={"config": asdict(owner._config), "witness": journal.data_state_witness},
        recursion_limit=max(
            1000, (journal.rounds_started + journal.remaining_round_budget + 1) * 15
        ),
    ) as execution:
        saver = execution.saver or InMemorySaver()
        config: RunnableConfig = execution.config
        if execution.saver is None:
            config = {**config, "configurable": {"thread_id": journal.exploration_id}}
        graph = build_exploration_graph().compile(checkpointer=saver)
        snapshot = graph.get_state(config)
        if snapshot.values and not owner._witness.recheck(journal.data_state_witness):
            run_state = cast(ExplorationState, snapshot.values)
            return owner._terminal(run_state, "state_witness_changed", error=None)
        if snapshot.values and not snapshot.next:
            return snapshot.values["result"]
        if snapshot.interrupts:
            value: Any = Command(resume=True)
        elif snapshot.values:
            value = None
        else:
            value = {
                "phase": SupervisorPhase.ORIENT,
                "transitions": [],
                "context": None,
                "reduction": None,
                "next_node": "orient",
                "result": None,
            }
        final = graph.invoke(value, config, context=ExplorationServices(owner), durability="sync")
        result = final.get("result")
        if not isinstance(result, SupervisorRunResult):
            raise SupervisorInvariantError("Exploration graph exited without a typed result.")
        return result
