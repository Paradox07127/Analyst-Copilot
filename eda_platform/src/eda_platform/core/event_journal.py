"""Generic append-only JSONL event journal (extracted from loop_journal).

The journal is the source of truth. Snapshots are optional caches and are
never used by :meth:`JsonlEventJournal.rebuild`, so corruption in an older
event cannot be hidden by a newer snapshot. Event/state types and the reducer
are injected; every domain transition rule lives in the reducer.
"""

from __future__ import annotations

import os
import tempfile
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Protocol, cast

from pydantic import TypeAdapter

from eda_platform.core.file_lock import lock_exclusive, unlock
from eda_platform.core.fs import BINARY_FLAG
from eda_platform.core.ids import stable_hash

_writer_locks = threading.local()


def _held_writer_locks() -> set[str]:
    """Lock files this thread already holds. flock is per-fd, so re-acquiring on
    a fresh fd blocks the holder against itself; keyed by path rather than by
    journal instance because two journal objects can share one file."""
    held: set[str] | None = getattr(_writer_locks, "held", None)
    if held is None:
        held = set()
        _writer_locks.held = held
    return held


class EventJournalError(RuntimeError):
    """Base error for journal persistence or transition failures."""


class EventJournalCorruptionError(EventJournalError):
    """Raised when any committed (non-tail) journal record is unreadable."""


class EventTransitionError(EventJournalError):
    """Raised when an event is invalid for the current reconstructed state."""


class JournalEventLike(Protocol):
    """An immutable event; must also carry seq/event_type/attempt_epoch and the
    journal's id field, which the injected reducer and adapter enforce."""

    def model_dump_json(self) -> str: ...


class JournalStateLike(Protocol):
    """State rebuilt by the reducer; the journal only reads fencing bookkeeping."""

    attempt_epoch: int
    last_seq: int

    def model_dump_json(self, *, indent: int | None = None) -> str: ...


class JsonlEventJournal[EventT: JournalEventLike, StateT: JournalStateLike]:
    """A flushed, fsynced JSONL journal with a single-writer file lock."""

    def __init__(
        self,
        path: Path | str,
        *,
        event_adapter: TypeAdapter[EventT],
        state_adapter: TypeAdapter[StateT],
        reducer: Callable[[StateT | None, EventT], StateT],
        id_field: str,
        label: str,
        executor_lock_prefix: str,
        corruption_error: type[EventJournalError] = EventJournalCorruptionError,
        transition_error: type[EventJournalError] = EventTransitionError,
        epoch_claim_event_type: str = "attempt_started",
    ) -> None:
        self.path = Path(path)
        self.snapshot_path = self.path.with_suffix(".snapshot.json")
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.executor_lock_path = self.path.with_suffix(self.path.suffix + ".executor.lock")
        self._claimed_attempt_epoch: int | None = None
        self._event_adapter = event_adapter
        self._state_adapter = state_adapter
        self._reducer = reducer
        self._id_field = id_field
        self._label = label
        self._executor_lock_prefix = executor_lock_prefix
        self._corruption_error = corruption_error
        self._transition_error = transition_error
        self._epoch_claim_event_type = epoch_claim_event_type

    def events(self) -> list[EventT]:
        if not self.path.exists():
            return []
        return list(self._read_events())

    def rebuild(self) -> StateT | None:
        return self._rebuild_unlocked()

    def append(self, event: EventT) -> StateT:
        with self._locked():
            current = self._rebuild_unlocked()
            if (
                current is not None
                and self._claimed_attempt_epoch is not None
                and current.attempt_epoch != self._claimed_attempt_epoch
            ):
                raise self._transition_error(
                    f"stale {self._label} executor attempt epoch; "
                    "another owner claimed the journal."
                )
            updated = self._reducer(current, event)
            self._append_unlocked(event)
            return updated

    def append_new(self, event_type: str, **fields: object) -> StateT:
        """Build and append the next event while holding the single-writer lock."""
        with self._locked():
            current = self._rebuild_unlocked()
            if current is None:
                raise self._transition_error(
                    "initialize the journal before appending events."
                )
            if (
                event_type != self._epoch_claim_event_type
                and self._claimed_attempt_epoch is not None
                and current.attempt_epoch != self._claimed_attempt_epoch
            ):
                raise self._transition_error(
                    f"stale {self._label} executor attempt epoch; "
                    "another owner claimed the journal."
                )
            if event_type == self._epoch_claim_event_type:
                event_epoch = current.attempt_epoch + 1
            elif self._claimed_attempt_epoch is not None:
                event_epoch = self._claimed_attempt_epoch
            else:
                event_epoch = current.attempt_epoch
            event = self._event_adapter.validate_python(
                {
                    **fields,
                    "seq": current.last_seq + 1,
                    self._id_field: getattr(current, self._id_field),
                    "event_type": event_type,
                    "attempt_epoch": fields.get("attempt_epoch", event_epoch),
                }
            )
            updated = self._reducer(current, event)
            self._append_unlocked(event)
            return updated

    def claim_attempt(self) -> StateT:
        """Advance the fencing epoch for a new executor owner.

        The next epoch is derived inside append_new's lock. Reading it from an
        unlocked rebuild made concurrent recovery lose the race and report
        "attempt_epoch must be N" — a corruption-shaped error for plain
        contention.
        """
        claimed = self.append_new(self._epoch_claim_event_type)
        self._claimed_attempt_epoch = claimed.attempt_epoch
        return claimed

    @contextmanager
    def execution_lock(self) -> Iterator[None]:
        """Serialize one journal executor for its whole lifecycle."""
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
                / (
                    f"{self._executor_lock_prefix}-"
                    f"{stable_hash(str(self.path.resolve(strict=False)))}.lock"
                )
            )
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with _open_append_no_follow(lock_path) as handle:
            lock_exclusive(handle.fileno())
            try:
                yield
            finally:
                unlock(handle.fileno())

    @contextmanager
    def fenced_side_effect(self) -> Iterator[int]:
        """Hold the journal lock while committing an epoch-fenced external result."""
        with self._locked():
            current = self._rebuild_unlocked()
            if current is None or self._claimed_attempt_epoch is None:
                raise self._transition_error(
                    f"claim an executor attempt before committing {self._label} "
                    "side effects."
                )
            if current.attempt_epoch != self._claimed_attempt_epoch:
                raise self._transition_error(
                    f"stale {self._label} executor attempt epoch; "
                    "refusing side-effect commit."
                )
            yield current.attempt_epoch

    def write_snapshot(self, state: StateT | None = None) -> Path:
        """Atomically write an optional state cache; the journal remains authoritative."""
        with self._locked():
            rebuilt = self._rebuild_unlocked()
            if rebuilt is None:
                raise self._transition_error("cannot snapshot an empty journal.")
            if state is not None and state != rebuilt:
                raise self._transition_error("snapshot state does not match the journal.")
            self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            body = rebuilt.model_dump_json(indent=2).encode("utf-8")
            descriptor, temporary_name = tempfile.mkstemp(
                dir=self.snapshot_path.parent,
                prefix=f".{self.snapshot_path.stem}-",
                suffix=".tmp",
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(body)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self.snapshot_path)
                _fsync_directory(self.snapshot_path.parent)
                return self.snapshot_path
            finally:
                temporary.unlink(missing_ok=True)

    def read_snapshot(self) -> StateT | None:
        if not self.snapshot_path.exists():
            return None
        try:
            return self._state_adapter.validate_json(
                _read_bytes_no_follow(self.snapshot_path)
            )
        except (OSError, ValueError) as exc:
            raise self._corruption_error(
                f"Invalid {self._label} snapshot {self.snapshot_path}: {exc}"
            ) from exc

    def _rebuild_unlocked(self) -> StateT | None:
        if not self.path.exists():
            return None
        state: StateT | None = None
        for event in self._read_events():
            state = self._reducer(state, event)
        return state

    def _append_unlocked(self, event: EventT) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._truncate_unterminated_tail_unlocked()
        encoded = event.model_dump_json().encode("utf-8") + b"\n"
        with _open_append_no_follow(self.path) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(self.path.parent)

    def _truncate_unterminated_tail_unlocked(self) -> None:
        """Discard an uncommitted crash tail before appending the next event."""
        if not self.path.exists():
            return
        raw = _read_bytes_no_follow(self.path)
        if not raw or raw.endswith(b"\n"):
            return
        committed_end = raw.rfind(b"\n") + 1
        flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | BINARY_FLAG
        descriptor = os.open(self.path, flags)
        with os.fdopen(descriptor, "r+b") as handle:
            handle.truncate(committed_end)
            handle.flush()
            os.fsync(handle.fileno())

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """Hold the single-writer lock, re-entrantly within one thread.

        Re-entrancy is what lets a fenced side effect record itself: the outer
        holder already provides the exclusion, so the nested block must not try
        to take the lock again.
        """
        held = _held_writer_locks()
        key = str(self.lock_path.resolve(strict=False))
        if key in held:
            yield
            return
        held.add(key)
        try:
            with self._acquire_writer_lock():
                yield
        finally:
            held.discard(key)

    @contextmanager
    def _acquire_writer_lock(self) -> Iterator[None]:
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
            with _open_append_no_follow(self.lock_path) as handle:
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
            with _open_append_no_follow(self.lock_path) as handle:
                lock_exclusive(handle.fileno())
                try:
                    yield
                finally:
                    unlock(handle.fileno())

    def _read_events(self) -> Iterator[EventT]:
        try:
            raw = _read_bytes_no_follow(self.path)
        except OSError as exc:
            raise self._corruption_error(
                f"Cannot read {self._label} journal {self.path}: {exc}"
            ) from exc
        if not raw:
            return

        # "\n" is the commit marker: fsync only returns after the whole line,
        # newline included, so the final chunk of a split is either empty (the
        # file ended cleanly) or an uncommitted crash tail. Neither is an event,
        # and counting a tail that happens to parse would contradict
        # _truncate_unterminated_tail_unlocked, which deletes it — that
        # disagreement left a permanent seq gap and bricked the journal.
        chunks = raw.split(b"\n")
        for index, line in enumerate(chunks[:-1]):
            if not line:
                raise self._corruption_error(
                    f"Blank committed record at line {index + 1} in {self.path}."
                )
            try:
                yield cast(EventT, self._event_adapter.validate_json(line))
            except (ValueError, UnicodeDecodeError) as exc:
                raise self._corruption_error(
                    f"Invalid committed record at line {index + 1} in {self.path}: {exc}"
                ) from exc


def _fsync_directory(directory: Path) -> None:
    """Persist a newly created/replaced directory entry on POSIX filesystems."""
    if os.name == "nt":  # opening directories with os.open is not supported
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_bytes_no_follow(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as handle:
        return handle.read()


@contextmanager
def _open_append_no_follow(path: Path) -> Iterator[BinaryIO]:
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_APPEND
        | getattr(os, "O_NOFOLLOW", 0)
        | BINARY_FLAG
    )
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "a+b") as handle:
        yield handle
