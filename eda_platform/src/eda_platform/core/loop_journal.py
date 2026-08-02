"""Append-only durable state for the bounded investigation loop.

The journal is the source of truth. Snapshots are optional caches and are
never used by :meth:`JsonlLoopJournal.rebuild`, so corruption in an older
event cannot be hidden by a newer snapshot. File mechanics (locking, fsync,
tail truncation, epoch fencing) live in the generic
:class:`~eda_platform.core.event_journal.JsonlEventJournal`; this module owns
only the investigation event/state semantics.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

from pydantic import TypeAdapter

from eda_platform.core.event_journal import EventJournalError, JsonlEventJournal
from eda_platform.core.ids import stable_hash
from eda_platform.schemas.deep_investigation import (
    LOOP_JOURNAL_SCHEMA_VERSION,
    InvestigationLoopEvent,
    InvestigationLoopState,
)


class LoopJournalError(EventJournalError):
    """Base error for journal persistence or transition failures."""


class LoopJournalCorruptionError(LoopJournalError):
    """Raised when any committed (non-tail) journal record is unreadable."""


class LoopTransitionError(LoopJournalError):
    """Raised when an event is invalid for the current reconstructed state."""


class LoopResumeIncompatibleError(LoopJournalError):
    """Raised when policy or code fingerprints do not match a persisted loop."""


def make_investigation_id(
    source_session_id: str,
    question_id: str,
    plan_fingerprint: str,
) -> str:
    """Return a stable investigation identity for the same approved plan."""
    return "inv_" + stable_hash(
        {
            "source_session_id": source_session_id,
            "question_id": question_id,
            "plan_fingerprint": plan_fingerprint,
        },
        length=24,
    )


def make_loop_call_id(investigation_id: str, iteration: int) -> str:
    """Return the stable provider call reference for one loop iteration."""
    return "call_" + stable_hash(
        {"investigation_id": investigation_id, "iteration": iteration},
        length=24,
    )


def make_loop_probe_id(
    investigation_id: str,
    iteration: int,
    probe_fingerprint: str,
) -> str:
    """Return the stable probe reference used for idempotent artifacts."""
    return "probe_" + stable_hash(
        {
            "investigation_id": investigation_id,
            "iteration": iteration,
            "probe_fingerprint": probe_fingerprint,
        },
        length=24,
    )


def make_loop_step_id(kind: str, stable_ref: str) -> str:
    """Return a stable completed-step identity."""
    return "step_" + stable_hash({"kind": kind, "ref": stable_ref}, length=24)


def reduce_loop_event(
    state: InvestigationLoopState | None,
    event: InvestigationLoopEvent,
) -> InvestigationLoopState:
    """Apply one event without performing I/O.

    This is deliberately strict: a caller cannot skip sequence numbers,
    decrease an attempt epoch, reuse a completed step, exceed a hard cap, or
    mutate the policy/code identity of an in-flight investigation.
    """
    if event.schema_version != LOOP_JOURNAL_SCHEMA_VERSION:
        raise LoopTransitionError(
            f"Unsupported loop journal schema version: {event.schema_version}."
        )
    if state is None:
        return _start_loop(event)

    if state.schema_version != LOOP_JOURNAL_SCHEMA_VERSION:
        raise LoopTransitionError(
            f"Unsupported loop state schema version: {state.schema_version}."
        )
    if event.event_type == "loop_started":
        raise LoopTransitionError("loop_started may only be the first event.")
    if event.investigation_id != state.investigation_id:
        raise LoopTransitionError("event investigation_id does not match journal state.")
    if event.seq != state.last_seq + 1:
        raise LoopTransitionError(
            f"event seq must be {state.last_seq + 1}, got {event.seq}."
        )
    _check_fingerprints(state, event)
    _check_attempt_epoch(state, event)
    if state.status != "running":
        raise LoopTransitionError(f"cannot append to terminal loop status {state.status!r}.")

    values = state.model_dump()
    values["last_seq"] = event.seq

    if event.event_type == "attempt_started":
        values["attempt_epoch"] = event.attempt_epoch
    elif event.event_type == "decision_call_started":
        _require(event, "call_id", "iteration")
        if state.pending_call_id is not None or state.pending_probe_id is not None:
            raise LoopTransitionError("another loop operation is already pending.")
        if state.remaining_call_budget <= 0:
            raise LoopTransitionError("decision call would exceed llm_call_cap.")
        values["pending_call_id"] = event.call_id
    elif event.event_type == "decision_call_completed":
        _require(event, "call_id", "step_id", "response_hash", "typed_decision")
        if event.call_id != state.pending_call_id:
            raise LoopTransitionError("completed decision does not match pending call.")
        _append_completed_step(values, state, event)
        values["pending_call_id"] = None
        values["llm_calls_settled"] = state.llm_calls_settled + 1
        values["remaining_call_budget"] = state.remaining_call_budget - 1
    elif event.event_type == "decision_call_rejected":
        _require(event, "call_id", "error")
        if event.call_id != state.pending_call_id:
            raise LoopTransitionError("rejected decision does not match pending call.")
        values["pending_call_id"] = None
        values["failure_history"] = [*state.failure_history, event.error]
    elif event.event_type == "probe_started":
        _require(event, "probe_id", "probe_fingerprint", "iteration")
        if state.pending_call_id is not None or state.pending_probe_id is not None:
            raise LoopTransitionError("another loop operation is already pending.")
        if state.remaining_probe_budget <= 0:
            raise LoopTransitionError("probe would exceed max_steps.")
        values["pending_probe_id"] = event.probe_id
    elif event.event_type == "artifact_committed":
        _require(event, "probe_id", "artifact_ref")
        if event.probe_id != state.pending_probe_id:
            raise LoopTransitionError("artifact does not match pending probe.")
        probe_id = cast(str, event.probe_id)
        artifact_ref = cast(str, event.artifact_ref)
        refs = dict(state.step_artifact_refs)
        prior = refs.get(probe_id)
        if prior is not None and prior != artifact_ref:
            raise LoopTransitionError("probe artifact reference cannot be replaced.")
        refs[probe_id] = artifact_ref
        values["step_artifact_refs"] = refs
    elif event.event_type == "probe_completed":
        _require(event, "probe_id", "step_id", "iteration")
        if event.probe_id != state.pending_probe_id:
            raise LoopTransitionError("completed probe does not match pending probe.")
        _append_completed_step(values, state, event)
        values["pending_probe_id"] = None
        values["probes_completed"] = state.probes_completed + 1
        values["remaining_probe_budget"] = state.remaining_probe_budget - 1
        values["next_iteration"] = max(
            state.next_iteration,
            cast(int, event.iteration) + 1,
        )
    elif event.event_type == "loop_concluded":
        _require_no_pending(state)
        values["status"] = "concluded"
        values["final_draft_ref"] = event.final_draft_ref
    elif event.event_type == "loop_budget_exhausted":
        _require_no_pending(state)
        values["status"] = "budget_exhausted"
        values["final_draft_ref"] = event.final_draft_ref
    elif event.event_type == "loop_failed":
        _require(event, "error")
        values["status"] = "failed"
        values["failure_history"] = [*state.failure_history, event.error]
    elif event.event_type == "loop_call_uncertain":
        _require(event, "call_id", "error")
        if event.call_id != state.pending_call_id:
            raise LoopTransitionError("uncertain call does not match pending call.")
        values["status"] = "uncertain"
        values["failure_history"] = [*state.failure_history, event.error]
    else:
        raise LoopTransitionError(f"unsupported transition: {event.event_type}.")

    return InvestigationLoopState.model_validate(values)


def rebuild_loop_state(
    events: Sequence[InvestigationLoopEvent],
) -> InvestigationLoopState | None:
    """Rebuild the same state from any complete event prefix."""
    state: InvestigationLoopState | None = None
    for event in events:
        state = reduce_loop_event(state, event)
    return state


def assert_resume_compatible(
    state: InvestigationLoopState,
    *,
    plan_fingerprint: str,
    policy_fingerprint: str,
    code_fingerprint: str,
) -> None:
    """Fail closed before old state is resumed under incompatible behavior."""
    expected = {
        "plan_fingerprint": plan_fingerprint,
        "policy_fingerprint": policy_fingerprint,
        "code_fingerprint": code_fingerprint,
    }
    mismatches = [
        name for name, value in expected.items() if getattr(state, name) != value
    ]
    if mismatches:
        raise LoopResumeIncompatibleError(
            "Cannot resume investigation with changed " + ", ".join(mismatches) + "."
        )


_LOOP_EVENT_ADAPTER: TypeAdapter[InvestigationLoopEvent] = TypeAdapter(
    InvestigationLoopEvent
)
_LOOP_STATE_ADAPTER: TypeAdapter[InvestigationLoopState] = TypeAdapter(
    InvestigationLoopState
)


class JsonlLoopJournal(JsonlEventJournal[InvestigationLoopEvent, InvestigationLoopState]):
    """A flushed, fsynced JSONL journal with an investigation-level file lock."""

    def __init__(self, path: Path | str) -> None:
        super().__init__(
            path,
            event_adapter=_LOOP_EVENT_ADAPTER,
            state_adapter=_LOOP_STATE_ADAPTER,
            reducer=reduce_loop_event,
            id_field="investigation_id",
            label="loop",
            executor_lock_prefix="loop-executor",
            corruption_error=LoopJournalCorruptionError,
            transition_error=LoopTransitionError,
        )

    def initialize(
        self,
        *,
        investigation_id: str,
        source_session_id: str,
        question_id: str,
        plan_fingerprint: str,
        policy_fingerprint: str,
        code_fingerprint: str,
        max_steps: int,
        llm_call_cap: int,
    ) -> InvestigationLoopState:
        """Atomically create the first event, or validate an existing identity."""
        event = InvestigationLoopEvent(
            seq=0,
            investigation_id=investigation_id,
            event_type="loop_started",
            source_session_id=source_session_id,
            question_id=question_id,
            plan_fingerprint=plan_fingerprint,
            policy_fingerprint=policy_fingerprint,
            code_fingerprint=code_fingerprint,
            max_steps=max_steps,
            llm_call_cap=llm_call_cap,
        )
        with self._locked():
            state = self._rebuild_unlocked()
            if state is not None:
                assert_resume_compatible(
                    state,
                    plan_fingerprint=plan_fingerprint,
                    policy_fingerprint=policy_fingerprint,
                    code_fingerprint=code_fingerprint,
                )
                if (
                    state.investigation_id != investigation_id
                    or state.source_session_id != source_session_id
                    or state.question_id != question_id
                    or state.max_steps != max_steps
                    or state.llm_call_cap != llm_call_cap
                ):
                    raise LoopResumeIncompatibleError(
                        "Existing journal identity or hard caps do not match initialization."
                    )
                return state
            state = reduce_loop_event(None, event)
            self._append_unlocked(event)
            return state


def _start_loop(event: InvestigationLoopEvent) -> InvestigationLoopState:
    if event.event_type != "loop_started" or event.seq != 0:
        raise LoopTransitionError("the first event must be loop_started with seq 0.")
    _require(
        event,
        "source_session_id",
        "question_id",
        "plan_fingerprint",
        "policy_fingerprint",
        "code_fingerprint",
        "max_steps",
        "llm_call_cap",
    )
    return InvestigationLoopState(
        investigation_id=event.investigation_id,
        source_session_id=cast(str, event.source_session_id),
        question_id=cast(str, event.question_id),
        plan_fingerprint=cast(str, event.plan_fingerprint),
        policy_fingerprint=cast(str, event.policy_fingerprint),
        code_fingerprint=cast(str, event.code_fingerprint),
        attempt_epoch=event.attempt_epoch,
        max_steps=cast(int, event.max_steps),
        llm_call_cap=cast(int, event.llm_call_cap),
        remaining_probe_budget=cast(int, event.max_steps),
        remaining_call_budget=cast(int, event.llm_call_cap),
        last_seq=event.seq,
    )


def _check_fingerprints(
    state: InvestigationLoopState,
    event: InvestigationLoopEvent,
) -> None:
    for name in ("plan_fingerprint", "policy_fingerprint", "code_fingerprint"):
        value = getattr(event, name)
        if value is not None and value != getattr(state, name):
            raise LoopResumeIncompatibleError(f"event changes persisted {name}.")


def _check_attempt_epoch(
    state: InvestigationLoopState,
    event: InvestigationLoopEvent,
) -> None:
    expected = (
        state.attempt_epoch + 1
        if event.event_type == "attempt_started"
        else state.attempt_epoch
    )
    if event.attempt_epoch != expected:
        raise LoopTransitionError(
            f"event attempt_epoch must be {expected}, got {event.attempt_epoch}."
        )


def _append_completed_step(
    values: dict[str, object],
    state: InvestigationLoopState,
    event: InvestigationLoopEvent,
) -> None:
    if event.step_id in state.completed_step_ids:
        raise LoopTransitionError(f"completed step {event.step_id!r} is already recorded.")
    values["completed_step_ids"] = [*state.completed_step_ids, event.step_id]


def _require(event: InvestigationLoopEvent, *fields: str) -> None:
    missing = [field for field in fields if getattr(event, field) is None]
    if missing:
        raise LoopTransitionError(
            f"{event.event_type} requires fields: {', '.join(missing)}."
        )


def _require_no_pending(state: InvestigationLoopState) -> None:
    if state.pending_call_id is not None or state.pending_probe_id is not None:
        raise LoopTransitionError("cannot terminate while a loop operation is pending.")
