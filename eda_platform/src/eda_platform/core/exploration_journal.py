"""Exploration event journal: policy fingerprint, reducer, and JSONL shell.

Reducer semantics are isomorphic to the investigation loop journal (seq is
monotonic, attempt epochs fence executors, pending operations are exclusive,
budget counters decrease with events), with two exploration-specific rules:
pause is a resumable status rather than a stop, and an uncertain LLM call
consumes its reservation without terminating the run (plan R3.1/R3.3).
"""

from __future__ import annotations

from collections.abc import Sequence
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
    AttemptStartedEvent,
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

_POLICY_FINGERPRINT_PREFIX = "xplcy_"
_NATURAL_STOP_REASONS = frozenset({"completed", "budget_exhausted", "no_new_information"})


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
        if state.current_round_index is not None:
            raise EventTransitionError("the previous round is still open.")
        if state.remaining_round_budget <= 0:
            raise EventTransitionError("round would exceed max_rounds.")
        if event.round_index != state.rounds_started:
            raise EventTransitionError(
                f"round_index must be {state.rounds_started}, got {event.round_index}."
            )
        values["current_round_index"] = event.round_index
        values["rounds_started"] = state.rounds_started + 1
        values["remaining_round_budget"] = state.remaining_round_budget - 1
    elif isinstance(event, LlmCallStartedEvent):
        _require_running(state, "llm_call_started")
        _require_no_pending(state)
        if (
            state.remaining_llm_call_budget is not None
            and state.remaining_llm_call_budget <= 0
        ):
            raise EventTransitionError(
                "llm call would exceed the llm request cap (max_requests)."
            )
        values["pending_call_id"] = event.call_id
    elif isinstance(event, LlmCallCompletedEvent):
        if event.call_id != state.pending_call_id:
            raise EventTransitionError("completed call does not match the pending call.")
        _append_completed_step(values, state, event.step_id)
        values["pending_call_id"] = None
        values["llm_calls_settled"] = state.llm_calls_settled + 1
        if state.remaining_llm_call_budget is not None:
            values["remaining_llm_call_budget"] = state.remaining_llm_call_budget - 1
    elif isinstance(event, LlmCallRejectedEvent):
        if event.call_id != state.pending_call_id:
            raise EventTransitionError("rejected call does not match the pending call.")
        values["pending_call_id"] = None
        values["failure_history"] = [*state.failure_history, event.error]
    elif isinstance(event, LlmCallUncertainEvent):
        # Fail closed: the provider outcome is unknown, so the reservation is
        # fully consumed, but the run itself stays resumable (R3.3).
        if event.call_id != state.pending_call_id:
            raise EventTransitionError("uncertain call does not match the pending call.")
        values["pending_call_id"] = None
        values["llm_calls_uncertain"] = state.llm_calls_uncertain + 1
        if state.remaining_llm_call_budget is not None:
            values["remaining_llm_call_budget"] = state.remaining_llm_call_budget - 1
        values["failure_history"] = [*state.failure_history, event.error]
    elif isinstance(event, ToolCallStartedEvent):
        _require_running(state, "tool_call_started")
        _require_no_pending(state)
        if state.remaining_tool_call_budget <= 0:
            raise EventTransitionError(
                "tool call would exceed max_successful_tool_calls."
            )
        if event.logical_step_id in state.completed_step_ids:
            raise EventTransitionError(
                f"logical step {event.logical_step_id!r} is already committed; "
                "adopt its receipt instead of re-running."
            )
        values["pending_logical_step_id"] = event.logical_step_id
    elif isinstance(event, ReceiptPreparedEvent):
        if event.logical_step_id != state.pending_logical_step_id:
            raise EventTransitionError("receipt does not match the pending tool step.")
        prior = state.prepared_receipt_id
        if prior is not None and prior != event.receipt_id:
            raise EventTransitionError("prepared receipt cannot be replaced.")
        values["prepared_receipt_id"] = event.receipt_id
    elif isinstance(event, ReceiptCommittedEvent):
        if event.logical_step_id != state.pending_logical_step_id:
            raise EventTransitionError("receipt does not match the pending tool step.")
        if state.prepared_receipt_id is None:
            raise EventTransitionError("receipt must be prepared before it is committed.")
        if state.prepared_receipt_id != event.receipt_id:
            raise EventTransitionError(
                "committed receipt does not match the prepared receipt."
            )
        _append_completed_step(values, state, event.logical_step_id)
        refs = dict(state.step_receipt_refs)
        refs[event.logical_step_id] = event.receipt_id
        values["step_receipt_refs"] = refs
        values["pending_logical_step_id"] = None
        values["prepared_receipt_id"] = None
        values["tool_calls_committed"] = state.tool_calls_committed + 1
        values["remaining_tool_call_budget"] = state.remaining_tool_call_budget - 1
    elif isinstance(event, ToolCallFailedEvent):
        # Success-counted budget: a failed call consumes nothing (plan §4.2).
        if event.logical_step_id != state.pending_logical_step_id:
            raise EventTransitionError("failed call does not match the pending tool step.")
        values["pending_logical_step_id"] = None
        values["prepared_receipt_id"] = None
        values["failure_history"] = [*state.failure_history, event.error]
    elif isinstance(event, GateVerdictEvent):
        values["gate_verdicts"] = {
            **state.gate_verdicts,
            event.claim_bundle_id: event.verdict,
        }
    elif isinstance(event, ReductionCommittedEvent):
        values["frontier_digest"] = event.frontier_digest
        values["ledger_digest"] = event.ledger_digest
    elif isinstance(event, RoundSettledEvent):
        _require_no_pending(state)
        if state.current_round_index is None:
            raise EventTransitionError("no round is open to settle.")
        if event.round_index != state.current_round_index:
            raise EventTransitionError(
                f"round_settled round_index must be {state.current_round_index}, "
                f"got {event.round_index}."
            )
        values["current_round_index"] = None
        values["rounds_settled"] = state.rounds_settled + 1
        values["consecutive_no_progress"] = (
            0 if event.progress else state.consecutive_no_progress + 1
        )
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
        if state.max_llm_requests is not None and increase.max_requests:
            values["max_llm_requests"] = state.max_llm_requests + increase.max_requests
            remaining = state.remaining_llm_call_budget
            if remaining is not None:
                values["remaining_llm_call_budget"] = remaining + increase.max_requests
        values["max_successful_tool_calls"] = (
            state.max_successful_tool_calls + increase.max_successful_tool_calls
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
        values["status"] = "stopped"
        values["stop_reason"] = event.stop_reason
        values["final_report_ref"] = event.final_report_ref
        values["pending_call_id"] = None
        values["pending_logical_step_id"] = None
        values["prepared_receipt_id"] = None
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
        uncertain_error: str = (
            "provider call outcome unknown after crash; "
            "reservation consumed (fail closed)."
        ),
    ) -> ExplorationLoopState:
        """Fence out older executors, then settle an in-flight LLM call as
        uncertain — the provider may have consumed the request, so it is never
        resent. Pending tool steps stay pending: local re-runs are safe and the
        receipt outbox guarantees the single logical commit."""
        state = self.claim_attempt()
        if state.pending_call_id is not None:
            state = self.append_new(
                "llm_call_uncertain",
                call_id=state.pending_call_id,
                error=uncertain_error,
            )
        return state


def _start_exploration(event: ExplorationLoopEvent) -> ExplorationLoopState:
    if not isinstance(event, ExplorationStartedEvent) or event.seq != 0:
        raise EventTransitionError(
            "the first event must be exploration_started with seq 0."
        )
    max_llm_requests = event.budget.llm.max_requests
    return ExplorationLoopState(
        exploration_id=event.exploration_id,
        policy_fingerprint=event.policy_fingerprint,
        effective_policy_fingerprint=event.policy_fingerprint,
        code_fingerprint=event.code_fingerprint,
        data_state_witness=event.data_state_witness,
        attempt_epoch=event.attempt_epoch,
        max_llm_requests=max_llm_requests,
        max_successful_tool_calls=event.budget.max_successful_tool_calls,
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


def _require_running(state: ExplorationLoopState, event_label: str) -> None:
    if state.status != "running":
        raise EventTransitionError(
            f"{event_label} requires a running exploration; status is "
            f"{state.status!r} (pause blocks new work)."
        )


def _require_no_pending(state: ExplorationLoopState) -> None:
    if state.pending_call_id is not None or state.pending_logical_step_id is not None:
        raise EventTransitionError(
            "another exploration operation is already pending."
        )


def _append_completed_step(
    values: dict[str, object],
    state: ExplorationLoopState,
    step_id: str,
) -> None:
    if step_id in state.completed_step_ids:
        raise EventTransitionError(f"completed step {step_id!r} is already recorded.")
    values["completed_step_ids"] = [*state.completed_step_ids, step_id]
