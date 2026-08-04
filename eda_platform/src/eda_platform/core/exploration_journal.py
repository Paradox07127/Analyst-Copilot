"""Exploration event journal: policy fingerprint, reducer, and JSONL shell.

Reducer semantics are isomorphic to the investigation loop journal (seq is
monotonic, attempt epochs fence executors, pending operations are exclusive,
budget counters decrease with events), with two exploration-specific rules:
pause is a resumable status rather than a stop, and an uncertain LLM call
consumes its reservation without terminating the run (plan R3.1/R3.3).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import TypeAdapter

from eda_platform.core.event_journal import (
    EventJournalError,
    EventTransitionError,
    JsonlEventJournal,
)
from eda_platform.core.ids import stable_hash
from eda_platform.schemas.exploration import (
    EXPLORATION_JOURNAL_SCHEMA_VERSION,
    MAIN_LINE_ID,
    AttemptStartedEvent,
    BranchAbandonedEvent,
    BudgetAmendedEvent,
    ExplorationLoopEvent,
    ExplorationLoopState,
    ExplorationPolicy,
    ExplorationStartedEvent,
    ExplorationStoppedEvent,
    GateVerdictEvent,
    LlmCallCompletedEvent,
    LlmCallRejectedEvent,
    LlmCallStartedEvent,
    LlmCallUncertainEvent,
    PausedEvent,
    PauseRequestedEvent,
    ReceiptCommittedEvent,
    ReceiptPreparedEvent,
    ReductionCommittedEvent,
    ResumedEvent,
    RoundSettledEvent,
    RoundStartedEvent,
    ToolCallFailedEvent,
    ToolCallStartedEvent,
)
from eda_platform.schemas.exploration_budget import BudgetCapIncrease

_POLICY_FINGERPRINT_PREFIX = "xplcy_"
_NATURAL_STOP_REASONS = frozenset({"completed", "budget_exhausted", "no_new_information"})


@dataclass(frozen=True, slots=True)
class RecoveredToolCommit:
    receipt_id: str
    result_digest: str
    rows_scanned: int = 0
    result_cells: int = 0

    def __post_init__(self) -> None:
        if not self.receipt_id:
            raise ValueError("recovered tool commit requires receipt_id.")
        if not self.result_digest:
            raise ValueError("recovered tool commit requires result_digest.")
        if self.rows_scanned < 0 or self.result_cells < 0:
            raise ValueError("recovered tool usage cannot be negative.")


class ExplorationResumeIncompatibleError(EventJournalError):
    """Raised when identity or fingerprints do not match a persisted exploration."""


class ExplorationPolicyIntegrityError(EventJournalError):
    """Raised when a policy fingerprint is missing or does not match its fields."""


def compute_policy_fingerprint(policy: ExplorationPolicy) -> str:
    """Hash every execution-affecting policy field (all fields but the fingerprint)."""
    payload = policy.model_dump(mode="json", exclude={"policy_fingerprint"})
    return _POLICY_FINGERPRINT_PREFIX + stable_hash(payload, length=24)


def sealed_policy(policy: ExplorationPolicy) -> ExplorationPolicy:
    """Return the policy with its fingerprint attached."""
    return policy.model_copy(
        update={"policy_fingerprint": compute_policy_fingerprint(policy)}
    )


def assert_policy_sealed(policy: ExplorationPolicy) -> None:
    if policy.policy_fingerprint != compute_policy_fingerprint(policy):
        raise ExplorationPolicyIntegrityError(
            "policy_fingerprint is missing or does not match the policy fields."
        )


def amended_policy_fingerprint(
    previous_fingerprint: str,
    amendment_id: str,
    increase: BudgetCapIncrease,
) -> str:
    """System-derived, order-sensitive identity for one effective policy link."""
    return _POLICY_FINGERPRINT_PREFIX + stable_hash(
        {
            "previous_effective_policy_fingerprint": previous_fingerprint,
            "amendment_id": amendment_id,
            "increase": increase.model_dump(mode="json"),
        },
        length=24,
    )


def reduce_exploration_event(
    state: ExplorationLoopState | None,
    event: ExplorationLoopEvent,
) -> ExplorationLoopState:
    """Apply one event without performing I/O; every rule fails closed."""
    if event.schema_version != EXPLORATION_JOURNAL_SCHEMA_VERSION:
        raise EventTransitionError(
            f"Unsupported exploration journal schema version: {event.schema_version}."
        )
    if state is None:
        return _start_exploration(event)

    if state.schema_version != EXPLORATION_JOURNAL_SCHEMA_VERSION:
        raise EventTransitionError(
            f"Unsupported exploration state schema version: {state.schema_version}."
        )
    if isinstance(event, ExplorationStartedEvent):
        raise EventTransitionError("exploration_started may only be the first event.")
    if event.exploration_id != state.exploration_id:
        raise EventTransitionError("event exploration_id does not match journal state.")
    if event.seq != state.last_seq + 1:
        raise EventTransitionError(
            f"event seq must be {state.last_seq + 1}, got {event.seq}."
        )
    _check_attempt_epoch(state, event)
    if state.status == "stopped":
        raise EventTransitionError(
            f"cannot append to a stopped exploration (stop_reason={state.stop_reason!r})."
        )

    values = state.model_dump()
    values["last_seq"] = event.seq

    if isinstance(event, AttemptStartedEvent):
        values["attempt_epoch"] = event.attempt_epoch
    elif isinstance(event, RoundStartedEvent):
        _require_running(state, "round_started")
        _require_no_pending(state)
        if state.pending_terminal_reason is not None:
            raise EventTransitionError(
                "cannot start a round while a terminal decision is pending."
            )
        if state.current_round_index is not None:
            raise EventTransitionError("the previous round is still open.")
        if state.remaining_round_budget <= 0:
            raise EventTransitionError("round would exceed max_rounds.")
        if event.round_index != state.rounds_started:
            raise EventTransitionError(
                f"round_index must be {state.rounds_started}, got {event.round_index}."
            )
        _apply_round_branch_rules(values, state, event.branch_id)
        values["current_round_index"] = event.round_index
        values["rounds_started"] = state.rounds_started + 1
        values["remaining_round_budget"] = state.remaining_round_budget - 1
        values["current_round_reduction_committed"] = False
    elif isinstance(event, LlmCallStartedEvent):
        _require_running(state, "llm_call_started")
        if event.call_id in state.pending_call_ids:
            raise EventTransitionError(
                f"llm call {event.call_id!r} is already pending."
            )
        # In-flight calls hold a slot of the request cap so concurrent starts
        # cannot over-admit before any of them settles.
        if (
            state.remaining_llm_call_budget is not None
            and state.remaining_llm_call_budget - len(state.pending_call_ids) <= 0
        ):
            raise EventTransitionError(
                "llm call would exceed the llm request cap (max_requests)."
            )
        values["pending_call_ids"] = (*state.pending_call_ids, event.call_id)
        if event.step_id is not None:
            values["pending_call_steps"] = {
                **state.pending_call_steps,
                event.call_id: event.step_id,
            }
    elif isinstance(event, LlmCallCompletedEvent):
        _settle_pending_call(values, state, event.call_id, "completed")
        bound_step = state.pending_call_steps.get(event.call_id)
        if bound_step is not None and event.step_id != bound_step:
            raise EventTransitionError("completed step does not match the pending call.")
        _append_completed_step(values, state, event.step_id)
        values["llm_calls_settled"] = state.llm_calls_settled + 1
        if state.remaining_llm_call_budget is not None:
            values["remaining_llm_call_budget"] = state.remaining_llm_call_budget - 1
    elif isinstance(event, LlmCallRejectedEvent):
        _settle_pending_call(values, state, event.call_id, "rejected")
        # A known provider rejection is still one attempted request. Treat it
        # as settled so repeated 4xx/5xx responses cannot bypass max_requests.
        values["llm_calls_settled"] = state.llm_calls_settled + 1
        if state.remaining_llm_call_budget is not None:
            values["remaining_llm_call_budget"] = state.remaining_llm_call_budget - 1
        values["failure_history"] = [*state.failure_history, event.error]
    elif isinstance(event, LlmCallUncertainEvent):
        # Fail closed: the provider outcome is unknown, so the reservation is
        # fully consumed, but the run itself stays resumable (R3.3).
        _settle_pending_call(values, state, event.call_id, "uncertain")
        values["llm_calls_uncertain"] = state.llm_calls_uncertain + 1
        values["uncertain_call_ids"] = [*state.uncertain_call_ids, event.call_id]
        if state.remaining_llm_call_budget is not None:
            values["remaining_llm_call_budget"] = state.remaining_llm_call_budget - 1
        values["failure_history"] = [*state.failure_history, event.error]
    elif isinstance(event, ToolCallStartedEvent):
        _require_running(state, "tool_call_started")
        if event.logical_step_id in state.pending_tool_steps:
            raise EventTransitionError(
                f"tool step {event.logical_step_id!r} is already pending."
            )
        if event.logical_step_id in state.completed_step_ids:
            raise EventTransitionError(
                f"logical step {event.logical_step_id!r} is already committed; "
                "adopt its receipt instead of re-running."
            )
        # In-flight slots reserve success budget so concurrent sessions cannot
        # over-admit before any receipt commits.
        if state.remaining_tool_call_budget - len(state.pending_tool_steps) <= 0:
            raise EventTransitionError(
                "tool call would exceed max_successful_tool_calls."
            )
        values["pending_tool_steps"] = {
            **_dumped_tool_slots(state),
            event.logical_step_id: {
                "tool_kind": event.tool_kind,
                "input_fingerprint": event.input_fingerprint,
                "projected_rows_scanned": event.projected_rows_scanned,
                "projected_result_cells": event.projected_result_cells,
                "prepared_receipt_id": None,
                "prepared_result_digest": None,
            },
        }
    elif isinstance(event, ReceiptPreparedEvent):
        slot = state.pending_tool_steps.get(event.logical_step_id)
        if slot is None:
            raise EventTransitionError("receipt does not match a pending tool step.")
        if slot.prepared_receipt_id is not None and (
            slot.prepared_receipt_id != event.receipt_id
        ):
            raise EventTransitionError("prepared receipt cannot be replaced.")
        if (
            slot.prepared_result_digest is not None
            and slot.prepared_result_digest != event.result_digest
        ):
            raise EventTransitionError("prepared tool result digest cannot be replaced.")
        values["pending_tool_steps"] = {
            **_dumped_tool_slots(state),
            event.logical_step_id: {
                **slot.model_dump(),
                "prepared_receipt_id": event.receipt_id,
                "prepared_result_digest": event.result_digest,
            },
        }
    elif isinstance(event, ReceiptCommittedEvent):
        slot = state.pending_tool_steps.get(event.logical_step_id)
        if slot is None:
            raise EventTransitionError("receipt does not match a pending tool step.")
        if slot.prepared_receipt_id is None:
            raise EventTransitionError("receipt must be prepared before it is committed.")
        if slot.prepared_receipt_id != event.receipt_id:
            raise EventTransitionError(
                "committed receipt does not match the prepared receipt."
            )
        if slot.prepared_result_digest != event.result_digest:
            raise EventTransitionError(
                "committed tool result digest does not match the prepared body."
            )
        _append_completed_step(values, state, event.logical_step_id)
        refs = dict(state.step_receipt_refs)
        refs[event.logical_step_id] = event.receipt_id
        values["step_receipt_refs"] = refs
        if event.result_digest is not None:
            result_digests = dict(state.step_result_digests)
            result_digests[event.logical_step_id] = event.result_digest
            values["step_result_digests"] = result_digests
        tool_kind = slot.tool_kind or "legacy_unknown"
        values["tool_calls_by_kind"] = {
            **state.tool_calls_by_kind,
            tool_kind: state.tool_calls_by_kind.get(tool_kind, 0) + 1,
        }
        values["rows_scanned"] = state.rows_scanned + event.rows_scanned
        values["result_cells"] = state.result_cells + event.result_cells
        if slot.input_fingerprint is not None:
            values["completed_probe_fingerprints"] = [
                *state.completed_probe_fingerprints,
                slot.input_fingerprint,
            ]
        _remove_tool_slot(values, state, event.logical_step_id)
        values["tool_calls_committed"] = state.tool_calls_committed + 1
        values["remaining_tool_call_budget"] = state.remaining_tool_call_budget - 1
    elif isinstance(event, ToolCallFailedEvent):
        # Success-counted budget: a failed call consumes nothing (plan §4.2).
        if event.logical_step_id not in state.pending_tool_steps:
            raise EventTransitionError("failed call does not match a pending tool step.")
        values["rows_scanned"] = state.rows_scanned + event.rows_scanned
        values["result_cells"] = state.result_cells + event.result_cells
        _remove_tool_slot(values, state, event.logical_step_id)
        values["failure_history"] = [*state.failure_history, event.error]
    elif isinstance(event, GateVerdictEvent):
        _require_running(state, "gate_verdict")
        _require_no_pending(state)
        if state.current_round_index is None:
            raise EventTransitionError("gate verdict requires an open round.")
        prior_verdict = state.gate_verdicts.get(event.claim_bundle_id)
        if prior_verdict is not None and prior_verdict != event.verdict:
            raise EventTransitionError("a gate verdict cannot be overwritten.")
        values["gate_verdicts"] = {
            **state.gate_verdicts,
            event.claim_bundle_id: event.verdict,
        }
    elif isinstance(event, ReductionCommittedEvent):
        _require_running(state, "reduction_committed")
        _require_no_pending(state)
        if state.current_round_index is None:
            raise EventTransitionError("reduction requires an open round.")
        values["frontier_digest"] = event.frontier_digest
        values["ledger_digest"] = event.ledger_digest
        values["reduction_digest"] = event.reduction_digest
        values["current_round_reduction_committed"] = True
    elif isinstance(event, RoundSettledEvent):
        _require_no_pending(state)
        if state.current_round_index is None:
            raise EventTransitionError("no round is open to settle.")
        if event.round_index != state.current_round_index:
            raise EventTransitionError(
                f"round_settled round_index must be {state.current_round_index}, "
                f"got {event.round_index}."
            )
        if (
            event.terminal_reason is not None
            and event.terminal_has_reduction
            != state.current_round_reduction_committed
        ):
            raise EventTransitionError(
                "terminal reduction metadata does not match the settled round."
            )
        if (
            event.terminal_reason == "no_new_information"
            and state.branch_trigger_stagnant_rounds is not None
            and len(state.started_branch_ids) < state.max_branches
        ):
            raise EventTransitionError(
                "branches must be exhausted before no_new_information (plan E6)."
            )
        values["current_round_index"] = None
        values["current_round_reduction_committed"] = False
        values["rounds_settled"] = state.rounds_settled + 1
        values["consecutive_no_progress"] = (
            0 if event.progress else state.consecutive_no_progress + 1
        )
        values["consecutive_empty_frontier"] = (
            state.consecutive_empty_frontier + 1 if event.frontier_empty else 0
        )
        # None = pre-plan-B event: counted as movement so a resumed legacy
        # journal can never soft-stop on rounds it never measured.
        values["consecutive_no_adjudication"] = (
            0
            if event.adjudicated_transitions is None
            or event.adjudicated_transitions > 0
            else state.consecutive_no_adjudication + 1
        )
        values["pending_terminal_reason"] = event.terminal_reason
        values["pending_terminal_has_reduction"] = event.terminal_has_reduction
    elif isinstance(event, BranchAbandonedEvent):
        _require_running(state, "branch_abandoned")
        _require_no_pending(state)
        if state.branch_trigger_stagnant_rounds is None:
            raise EventTransitionError("branching is disabled for this run.")
        if state.current_round_index is not None:
            raise EventTransitionError("branch abandonment requires no open round.")
        if state.pending_terminal_reason is not None:
            raise EventTransitionError(
                "cannot abandon a line after a terminal decision."
            )
        if state.current_line_abandoned:
            raise EventTransitionError("the current line is already abandoned.")
        if state.rounds_settled == 0 or event.round_index != state.rounds_settled - 1:
            raise EventTransitionError(
                "branch_abandoned round_index must identify the last settled round."
            )
        current_line = state.active_branch_id or MAIN_LINE_ID
        if event.branch_id != current_line:
            raise EventTransitionError(
                f"only the current line {current_line!r} can be abandoned."
            )
        if state.consecutive_no_progress < state.branch_trigger_stagnant_rounds:
            raise EventTransitionError(
                "branch abandonment requires the system stagnation signal "
                f"({state.consecutive_no_progress} < "
                f"{state.branch_trigger_stagnant_rounds} stagnant rounds)."
            )
        if len(state.started_branch_ids) >= state.max_branches:
            raise EventTransitionError(
                "branch budget is exhausted; no successor branch can start."
            )
        values["abandoned_line_ids"] = [*state.abandoned_line_ids, event.branch_id]
        values["current_line_abandoned"] = True
        values["consecutive_no_progress"] = 0
        values["consecutive_no_adjudication"] = 0
        values["abandoned_constraints"] = [
            *state.abandoned_constraints,
            *event.constraints,
        ]
    elif isinstance(event, PauseRequestedEvent):
        if state.status != "running":
            raise EventTransitionError("pause can only be requested while running.")
        values["status"] = "pause_requested"
    elif isinstance(event, PausedEvent):
        if state.status != "pause_requested":
            raise EventTransitionError("paused requires a prior pause_requested.")
        _require_no_pending(state)
        values["status"] = "paused"
    elif isinstance(event, ResumedEvent):
        if state.status != "paused":
            raise EventTransitionError("resumed requires a paused exploration.")
        values["status"] = "running"
    elif isinstance(event, BudgetAmendedEvent):
        if event.amendment_id in state.amendment_ids:
            raise EventTransitionError(
                f"budget amendment {event.amendment_id!r} is already applied."
            )
        increase = event.increase
        expected_fingerprint = amended_policy_fingerprint(
            state.effective_policy_fingerprint,
            event.amendment_id,
            increase,
        )
        if event.effective_policy_fingerprint != expected_fingerprint:
            raise EventTransitionError(
                "budget amendment effective_policy_fingerprint does not match "
                "the prior policy and requested increase."
            )
        if state.max_llm_requests is not None and increase.max_requests:
            values["max_llm_requests"] = state.max_llm_requests + increase.max_requests
            remaining = state.remaining_llm_call_budget
            if remaining is not None:
                values["remaining_llm_call_budget"] = remaining + increase.max_requests
        values["max_successful_tool_calls"] = (
            state.max_successful_tool_calls + increase.max_successful_tool_calls
        )
        max_by_kind = dict(state.max_tool_calls_by_kind)
        for kind, delta in increase.max_tool_calls_by_kind.items():
            max_by_kind[kind] = max_by_kind.get(kind, 0) + delta
        values["max_tool_calls_by_kind"] = max_by_kind
        if state.max_rows_scanned is not None:
            values["max_rows_scanned"] = (
                state.max_rows_scanned + increase.max_rows_scanned
            )
        if state.max_result_cells is not None:
            values["max_result_cells"] = (
                state.max_result_cells + increase.max_result_cells
            )
        values["remaining_tool_call_budget"] = (
            state.remaining_tool_call_budget + increase.max_successful_tool_calls
        )
        values["max_rounds"] = state.max_rounds + increase.max_rounds
        values["remaining_round_budget"] = (
            state.remaining_round_budget + increase.max_rounds
        )
        values["effective_policy_fingerprint"] = event.effective_policy_fingerprint
        values["amendment_ids"] = [*state.amendment_ids, event.amendment_id]
    elif isinstance(event, ExplorationStoppedEvent):
        if event.stop_reason in _NATURAL_STOP_REASONS:
            if state.status != "running":
                raise EventTransitionError(
                    f"stop_reason {event.stop_reason!r} requires a running exploration."
                )
            _require_no_pending(state)
            if (
                state.pending_terminal_reason is not None
                and event.stop_reason != state.pending_terminal_reason
            ):
                raise EventTransitionError(
                    "natural stop reason does not match the durable terminal decision."
                )
        values["status"] = "stopped"
        values["stop_reason"] = event.stop_reason
        values["final_report_ref"] = event.final_report_ref
        if state.pending_call_ids:
            values["llm_calls_uncertain"] = state.llm_calls_uncertain + len(
                state.pending_call_ids
            )
            values["uncertain_call_ids"] = [
                *state.uncertain_call_ids,
                *state.pending_call_ids,
            ]
            if state.remaining_llm_call_budget is not None:
                values["remaining_llm_call_budget"] = state.remaining_llm_call_budget - len(
                    state.pending_call_ids
                )
        values["pending_call_ids"] = ()
        values["pending_call_steps"] = {}
        if state.pending_tool_steps:
            values["rows_scanned"] = state.rows_scanned + sum(
                slot.projected_rows_scanned
                for slot in state.pending_tool_steps.values()
            )
            values["result_cells"] = state.result_cells + sum(
                slot.projected_result_cells
                for slot in state.pending_tool_steps.values()
            )
        values["pending_tool_steps"] = {}
        values["pending_terminal_reason"] = None
        values["pending_terminal_has_reduction"] = False
    else:  # pragma: no cover - the discriminated union is exhaustive
        raise EventTransitionError(f"unsupported transition: {event.event_type}.")

    return ExplorationLoopState.model_validate(values)


def rebuild_exploration_state(
    events: Sequence[ExplorationLoopEvent],
) -> ExplorationLoopState | None:
    """Rebuild the same state from any complete event prefix."""
    state: ExplorationLoopState | None = None
    for event in events:
        state = reduce_exploration_event(state, event)
    return state


_EVENT_ADAPTER: TypeAdapter[ExplorationLoopEvent] = TypeAdapter(ExplorationLoopEvent)
_STATE_ADAPTER: TypeAdapter[ExplorationLoopState] = TypeAdapter(ExplorationLoopState)


class JsonlExplorationJournal(
    JsonlEventJournal[ExplorationLoopEvent, ExplorationLoopState]
):
    """A flushed, fsynced JSONL journal for one exploration run."""

    def __init__(self, path: Path | str) -> None:
        super().__init__(
            path,
            event_adapter=_EVENT_ADAPTER,
            state_adapter=_STATE_ADAPTER,
            reducer=reduce_exploration_event,
            id_field="exploration_id",
            label="exploration",
            executor_lock_prefix="exploration-executor",
        )

    def initialize(
        self,
        *,
        exploration_id: str,
        policy: ExplorationPolicy,
        code_fingerprint: str,
        data_state_witness: str,
    ) -> ExplorationLoopState:
        """Atomically create the first event, or validate an existing identity."""
        assert_policy_sealed(policy)
        event = ExplorationStartedEvent(
            seq=0,
            exploration_id=exploration_id,
            policy_fingerprint=policy.policy_fingerprint,
            code_fingerprint=code_fingerprint,
            data_state_witness=data_state_witness,
            budget=policy.budget,
        )
        with self._locked():
            state = self._rebuild_unlocked()
            if state is not None:
                expected = {
                    "exploration_id": exploration_id,
                    "policy_fingerprint": policy.policy_fingerprint,
                    "code_fingerprint": code_fingerprint,
                    "data_state_witness": data_state_witness,
                }
                mismatches = [
                    name
                    for name, value in expected.items()
                    if getattr(state, name) != value
                ]
                if mismatches:
                    raise ExplorationResumeIncompatibleError(
                        "Cannot resume exploration with changed "
                        + ", ".join(mismatches)
                        + "."
                    )
                return state
            state = reduce_exploration_event(None, event)
            self._append_unlocked(event)
            return state

    def claim_recovery(
        self,
        *,
        completed_response_digest: Callable[[str], str | None] | None = None,
        completed_tool_result: Callable[[str], RecoveredToolCommit | None] | None = None,
        settle_pending_tool: bool = True,
        uncertain_error: str = (
            "provider call outcome unknown after crash; "
            "reservation consumed (fail closed)."
        ),
    ) -> ExplorationLoopState:
        """Fence out older executors, then settle every in-flight LLM call as
        uncertain — the provider may have consumed each request, so none is
        resent. Every pending tool slot is settled the same way one was before:
        a durable body is adopted, otherwise the projection is charged before a
        safe logical rerun."""
        state = self.claim_attempt()
        for call_id in state.pending_call_ids:
            bound_step = state.pending_call_steps.get(call_id)
            digest = None
            if bound_step is not None and completed_response_digest is not None:
                digest = completed_response_digest(bound_step)
            if digest:
                state = self.append_new(
                    "llm_call_completed",
                    call_id=call_id,
                    step_id=bound_step,
                    response_digest=digest,
                )
            else:
                state = self.append_new(
                    "llm_call_uncertain",
                    call_id=call_id,
                    error=uncertain_error,
                )
        if settle_pending_tool:
            for logical_step_id in list(state.pending_tool_steps):
                slot = state.pending_tool_steps[logical_step_id]
                recovered_tool = (
                    completed_tool_result(logical_step_id)
                    if completed_tool_result is not None
                    else None
                )
                if recovered_tool is not None:
                    if slot.prepared_receipt_id not in {None, recovered_tool.receipt_id}:
                        raise EventTransitionError(
                            "durable tool body does not match the prepared receipt."
                        )
                    if slot.prepared_receipt_id is not None:
                        if slot.prepared_result_digest is None:
                            raise EventTransitionError(
                                "prepared receipt has no durable tool-result digest; "
                                "refusing crash adoption."
                            )
                        if slot.prepared_result_digest != recovered_tool.result_digest:
                            raise EventTransitionError(
                                "durable tool body digest does not match the prepared receipt."
                            )
                    if slot.prepared_receipt_id is None:
                        state = self.append_new(
                            "receipt_prepared",
                            logical_step_id=logical_step_id,
                            receipt_id=recovered_tool.receipt_id,
                            result_digest=recovered_tool.result_digest,
                        )
                    state = self.append_new(
                        "receipt_committed",
                        logical_step_id=logical_step_id,
                        receipt_id=recovered_tool.receipt_id,
                        rows_scanned=recovered_tool.rows_scanned,
                        result_cells=recovered_tool.result_cells,
                        result_digest=recovered_tool.result_digest,
                    )
                else:
                    state = self.append_new(
                        "tool_call_failed",
                        logical_step_id=logical_step_id,
                        error=(
                            "tool outcome unknown after crash; projected resource usage "
                            "charged before safe logical rerun."
                        ),
                        rows_scanned=slot.projected_rows_scanned,
                        result_cells=slot.projected_result_cells,
                    )
        return state

    def amend_budget(
        self,
        *,
        amendment_id: str,
        increase: BudgetCapIncrease,
    ) -> ExplorationLoopState:
        """Append an amendment with its fingerprint derived under the writer lock."""
        with self._locked():
            state = self._rebuild_unlocked()
            if state is None:
                raise EventTransitionError(
                    "initialize the journal before amending its budget."
                )
            fingerprint = amended_policy_fingerprint(
                state.effective_policy_fingerprint,
                amendment_id,
                increase,
            )
            return self.append_new(
                "budget_amended",
                amendment_id=amendment_id,
                effective_policy_fingerprint=fingerprint,
                increase=increase,
            )


def _start_exploration(event: ExplorationLoopEvent) -> ExplorationLoopState:
    if not isinstance(event, ExplorationStartedEvent) or event.seq != 0:
        raise EventTransitionError(
            "the first event must be exploration_started with seq 0."
        )
    max_llm_requests = event.budget.llm.max_requests
    branching = event.budget.branching
    return ExplorationLoopState(
        branch_trigger_stagnant_rounds=(
            None if branching is None else branching.trigger_stagnant_rounds
        ),
        max_branches=0 if branching is None else branching.max_branches,
        exploration_id=event.exploration_id,
        policy_fingerprint=event.policy_fingerprint,
        effective_policy_fingerprint=event.policy_fingerprint,
        code_fingerprint=event.code_fingerprint,
        data_state_witness=event.data_state_witness,
        attempt_epoch=event.attempt_epoch,
        max_llm_requests=max_llm_requests,
        max_successful_tool_calls=event.budget.max_successful_tool_calls,
        max_tool_calls_by_kind=event.budget.max_tool_calls_by_kind,
        max_rows_scanned=event.budget.max_rows_scanned,
        max_result_cells=event.budget.max_result_cells,
        max_rounds=event.budget.max_rounds,
        remaining_llm_call_budget=max_llm_requests,
        remaining_tool_call_budget=event.budget.max_successful_tool_calls,
        remaining_round_budget=event.budget.max_rounds,
        last_seq=event.seq,
    )


def _check_attempt_epoch(
    state: ExplorationLoopState,
    event: ExplorationLoopEvent,
) -> None:
    expected = (
        state.attempt_epoch + 1
        if isinstance(event, AttemptStartedEvent)
        else state.attempt_epoch
    )
    if event.attempt_epoch != expected:
        raise EventTransitionError(
            f"event attempt_epoch must be {expected}, got {event.attempt_epoch}."
        )


def _apply_round_branch_rules(
    values: dict[str, object],
    state: ExplorationLoopState,
    branch_id: str | None,
) -> None:
    if branch_id is None:
        if state.current_line_abandoned:
            raise EventTransitionError(
                "this line was abandoned; the next round must open a new branch."
            )
        if state.active_branch_id is not None:
            raise EventTransitionError(
                "an active branch requires its branch id on round_started."
            )
        return
    if state.branch_trigger_stagnant_rounds is None:
        raise EventTransitionError("branching is disabled for this run.")
    if state.current_line_abandoned:
        expected = f"br_{len(state.started_branch_ids) + 1}"
        if branch_id != expected:
            raise EventTransitionError(
                f"the next branch must be {expected!r}, got {branch_id!r}."
            )
        if len(state.started_branch_ids) >= state.max_branches:
            raise EventTransitionError("branch budget is exhausted.")
        values["active_branch_id"] = branch_id
        values["started_branch_ids"] = [*state.started_branch_ids, branch_id]
        values["current_line_abandoned"] = False
        return
    if branch_id != state.active_branch_id:
        raise EventTransitionError(
            "a new branch may only start after the current line is abandoned."
        )


def _require_running(state: ExplorationLoopState, event_label: str) -> None:
    if state.status != "running":
        raise EventTransitionError(
            f"{event_label} requires a running exploration; status is "
            f"{state.status!r} (pause blocks new work)."
        )


def _require_no_pending(state: ExplorationLoopState) -> None:
    if state.pending_call_ids or state.pending_tool_steps:
        raise EventTransitionError(
            "another exploration operation is already pending."
        )


def _settle_pending_call(
    values: dict[str, object],
    state: ExplorationLoopState,
    call_id: str,
    event_label: str,
) -> None:
    if call_id not in state.pending_call_ids:
        raise EventTransitionError(
            f"{event_label} call does not match a pending call."
        )
    values["pending_call_ids"] = tuple(
        pending for pending in state.pending_call_ids if pending != call_id
    )
    values["pending_call_steps"] = {
        pending: step
        for pending, step in state.pending_call_steps.items()
        if pending != call_id
    }


def _dumped_tool_slots(state: ExplorationLoopState) -> dict[str, dict[str, object]]:
    return {step: slot.model_dump() for step, slot in state.pending_tool_steps.items()}


def _remove_tool_slot(
    values: dict[str, object],
    state: ExplorationLoopState,
    logical_step_id: str,
) -> None:
    values["pending_tool_steps"] = {
        step: slot.model_dump()
        for step, slot in state.pending_tool_steps.items()
        if step != logical_step_id
    }


def _append_completed_step(
    values: dict[str, object],
    state: ExplorationLoopState,
    step_id: str,
) -> None:
    if step_id in state.completed_step_ids:
        raise EventTransitionError(f"completed step {step_id!r} is already recorded.")
    values["completed_step_ids"] = [*state.completed_step_ids, step_id]
