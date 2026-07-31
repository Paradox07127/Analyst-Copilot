"""Append-only durable state for the bounded investigation loop.

The journal is the source of truth. Snapshots are optional caches and are
never used by :meth:`JsonlLoopJournal.rebuild`, so corruption in an older
event cannot be hidden by a newer snapshot.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol, cast

from eda_platform.core.file_lock import lock_exclusive, unlock
from eda_platform.core.ids import stable_hash
from eda_platform.schemas.deep_investigation import (
    LOOP_JOURNAL_SCHEMA_VERSION,
    InvestigationLoopEvent,
    InvestigationLoopState,
    LoopJournalEventType,
)


class LoopJournalError(RuntimeError):
    """Base error for journal persistence or transition failures."""


class LoopJournalCorruptionError(LoopJournalError):
    """Raised when any committed (non-tail) journal record is unreadable."""


class LoopTransitionError(LoopJournalError):
    """Raised when an event is invalid for the current reconstructed state."""


class LoopResumeIncompatibleError(LoopJournalError):
    """Raised when policy or code fingerprints do not match a persisted loop."""


class LoopJournal(Protocol):
    """Persistence contract consumed by a future loop executor."""

    def events(self) -> list[InvestigationLoopEvent]: ...

    def rebuild(self) -> InvestigationLoopState | None: ...

    def append(self, event: InvestigationLoopEvent) -> InvestigationLoopState: ...

    def write_snapshot(self, state: InvestigationLoopState | None = None) -> Path: ...


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


class JsonlLoopJournal:
    """A flushed, fsynced JSONL journal with an investigation-level file lock."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.snapshot_path = self.path.with_suffix(".snapshot.json")
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.executor_lock_path = self.path.with_suffix(self.path.suffix + ".executor.lock")
        self._claimed_attempt_epoch: int | None = None

    def events(self) -> list[InvestigationLoopEvent]:
        if not self.path.exists():
            return []
        return list(_read_events(self.path))

    def rebuild(self) -> InvestigationLoopState | None:
        return rebuild_loop_state(self.events())

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
            state = (
                rebuild_loop_state(list(_read_events(self.path)))
                if self.path.exists()
                else None
            )
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

    def append(self, event: InvestigationLoopEvent) -> InvestigationLoopState:
        with self._locked():
            current = (
                rebuild_loop_state(list(_read_events(self.path)))
                if self.path.exists()
                else None
            )
            if (
                current is not None
                and self._claimed_attempt_epoch is not None
                and current.attempt_epoch != self._claimed_attempt_epoch
            ):
                raise LoopTransitionError(
                    "stale loop executor attempt epoch; another owner claimed the journal."
                )
            updated = reduce_loop_event(current, event)
            self._append_unlocked(event)
            return updated

    def append_new(
        self,
        event_type: LoopJournalEventType,
        **fields: object,
    ) -> InvestigationLoopState:
        """Build and append the next event while holding the single-writer lock."""
        with self._locked():
            current = (
                rebuild_loop_state(list(_read_events(self.path)))
                if self.path.exists()
                else None
            )
            if current is None:
                raise LoopTransitionError("initialize the journal before appending events.")
            if (
                event_type != "attempt_started"
                and self._claimed_attempt_epoch is not None
                and current.attempt_epoch != self._claimed_attempt_epoch
            ):
                raise LoopTransitionError(
                    "stale loop executor attempt epoch; another owner claimed the journal."
                )
            event_epoch = (
                self._claimed_attempt_epoch
                if self._claimed_attempt_epoch is not None
                else current.attempt_epoch
            )
            event = InvestigationLoopEvent.model_validate(
                {
                    **fields,
                    "seq": current.last_seq + 1,
                    "investigation_id": current.investigation_id,
                    "event_type": event_type,
                    "attempt_epoch": fields.get("attempt_epoch", event_epoch),
                }
            )
            updated = reduce_loop_event(current, event)
            self._append_unlocked(event)
            return updated

    def claim_attempt(self) -> InvestigationLoopState:
        """Advance the fencing epoch for a new executor owner."""
        current = self.rebuild()
        if current is None:
            raise LoopTransitionError("initialize the journal before claiming an attempt.")
        claimed = self.append_new(
            "attempt_started",
            attempt_epoch=current.attempt_epoch + 1,
        )
        self._claimed_attempt_epoch = claimed.attempt_epoch
        return claimed

    @contextmanager
    def execution_lock(self) -> Iterator[None]:
        """Serialize one investigation executor for the whole primary+loop lifecycle."""
        session_dir = next(
            (parent for parent in self.path.parents if parent.parent.name == "sessions"),
            None,
        )
        lock_path = self.executor_lock_path
        if session_dir is not None:
            workspace = session_dir.parent.parent.parent.parent
            lock_path = (
                workspace
                / ".storage-operations"
                / "locks"
                / f"loop-executor-{stable_hash(str(self.path.resolve(strict=False)))}.lock"
            )
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as handle:
            lock_exclusive(handle.fileno())
            try:
                yield
            finally:
                unlock(handle.fileno())

    @contextmanager
    def fenced_side_effect(self) -> Iterator[int]:
        """Hold the journal lock while committing an epoch-fenced external result."""
        with self._locked():
            current = (
                rebuild_loop_state(list(_read_events(self.path)))
                if self.path.exists()
                else None
            )
            if current is None or self._claimed_attempt_epoch is None:
                raise LoopTransitionError(
                    "claim an executor attempt before committing loop side effects."
                )
            if current.attempt_epoch != self._claimed_attempt_epoch:
                raise LoopTransitionError(
                    "stale loop executor attempt epoch; refusing side-effect commit."
                )
            yield current.attempt_epoch

    def write_snapshot(self, state: InvestigationLoopState | None = None) -> Path:
        """Atomically write an optional state cache; the journal remains authoritative."""
        with self._locked():
            rebuilt = rebuild_loop_state(list(_read_events(self.path)))
            if rebuilt is None:
                raise LoopTransitionError("cannot snapshot an empty journal.")
            if state is not None and state != rebuilt:
                raise LoopTransitionError("snapshot state does not match the journal.")
            self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.snapshot_path.with_suffix(self.snapshot_path.suffix + ".tmp")
            body = rebuilt.model_dump_json(indent=2).encode("utf-8")
            with temporary.open("wb") as handle:
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.snapshot_path)
            return self.snapshot_path

    def read_snapshot(self) -> InvestigationLoopState | None:
        if not self.snapshot_path.exists():
            return None
        try:
            return InvestigationLoopState.model_validate_json(
                self.snapshot_path.read_bytes()
            )
        except (OSError, ValueError) as exc:
            raise LoopJournalCorruptionError(
                f"Invalid loop snapshot {self.snapshot_path}: {exc}"
            ) from exc

    def _append_unlocked(self, event: InvestigationLoopEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._truncate_unterminated_tail_unlocked()
        encoded = event.model_dump_json().encode("utf-8") + b"\n"
        with self.path.open("ab") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())

    def _truncate_unterminated_tail_unlocked(self) -> None:
        """Discard an uncommitted crash tail before appending the next event."""
        if not self.path.exists():
            return
        raw = self.path.read_bytes()
        if not raw or raw.endswith(b"\n"):
            return
        committed_end = raw.rfind(b"\n") + 1
        with self.path.open("r+b") as handle:
            handle.truncate(committed_end)
            handle.flush()
            os.fsync(handle.fileno())

    @contextmanager
    def _locked(self) -> Iterator[None]:
        from eda_platform.core.store import ArtifactStore

        session_dir = next(
            (
                parent
                for parent in self.path.parents
                if parent.parent.name == "sessions"
            ),
            None,
        )
        if session_dir is None:
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            with self.lock_path.open("a+b") as handle:
                lock_exclusive(handle.fileno())
                try:
                    yield
                finally:
                    unlock(handle.fileno())
            return
        project_dir = session_dir.parent.parent
        workspace = project_dir.parent.parent
        store = ArtifactStore(workspace, init_db=False)
        with store.session_write_guard(project_dir.name, session_dir.name):
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            with self.lock_path.open("a+b") as handle:
                lock_exclusive(handle.fileno())
                try:
                    yield
                finally:
                    unlock(handle.fileno())


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


def _read_events(path: Path) -> Iterator[InvestigationLoopEvent]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise LoopJournalCorruptionError(f"Cannot read loop journal {path}: {exc}") from exc
    if not raw:
        return

    chunks = raw.split(b"\n")
    has_trailing_newline = raw.endswith(b"\n")
    last_record_index = len(chunks) - (2 if has_trailing_newline else 1)
    for index, line in enumerate(chunks):
        if has_trailing_newline and index == len(chunks) - 1:
            continue
        is_unterminated_tail = not has_trailing_newline and index == last_record_index
        if not line:
            if is_unterminated_tail:
                continue
            raise LoopJournalCorruptionError(
                f"Blank committed record at line {index + 1} in {path}."
            )
        try:
            yield InvestigationLoopEvent.model_validate_json(line)
        except (ValueError, UnicodeDecodeError) as exc:
            if is_unterminated_tail:
                return
            raise LoopJournalCorruptionError(
                f"Invalid committed record at line {index + 1} in {path}: {exc}"
            ) from exc
