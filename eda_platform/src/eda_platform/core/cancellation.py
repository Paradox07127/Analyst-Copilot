"""Thread-safe cooperative cancellation and hard-kill fencing primitives.

The context never sends a signal.  Its kill-fence state only says whether a
cancelled operation has reached a point where an executor may *consider*
termination.  Executors must additionally perform an authoritative process
identity check immediately before signalling.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from enum import StrEnum
from threading import Event, RLock, Thread
from typing import Protocol, runtime_checkable


class CancellationCause(StrEnum):
    ACTIVE = "active"
    CANCEL_REQUESTED = "cancel_requested"
    DEADLINE_EXCEEDED = "deadline_exceeded"


class KillFenceState(StrEnum):
    """Cancellation half of a kill fence.

    ``ELIGIBLE`` is deliberately not named "allowed": PID start identity and
    executor ownership still need to be verified before any signal is sent.
    """

    BLOCKED_ACTIVE = "blocked_active"
    BLOCKED_SHIELDED = "blocked_shielded"
    ELIGIBLE = "eligible"


@dataclass(frozen=True, slots=True)
class CancellationSnapshot:
    cause: CancellationCause
    reason: str | None
    deadline: float | None
    shield_depth: int
    kill_fence_state: KillFenceState

    @property
    def cancellation_requested(self) -> bool:
        return self.cause is not CancellationCause.ACTIVE


@runtime_checkable
class KillFence(Protocol):
    """Minimal interface consumed by a future process executor/backend."""

    def kill_fence_state(self) -> KillFenceState:
        """Return the current cancellation fence state."""
        ...


@runtime_checkable
class CancellationToken(KillFence, Protocol):
    """Stable execution-seam contract used by queries, providers and sandboxes."""

    def checkpoint(self) -> None: ...

    @contextmanager
    def shield(self) -> Iterator[None]: ...

    @contextmanager
    def interrupt_on_cancel(self, callback: Callable[[], None]) -> Iterator[None]:
        """Register an active-operation interrupt for the scope."""
        ...


class CancellationError(RuntimeError):
    def __init__(self, snapshot: CancellationSnapshot) -> None:
        self.snapshot = snapshot
        message = snapshot.reason or snapshot.cause.value
        super().__init__(message)


class CancellationRequested(CancellationError):
    pass


class DeadlineExceeded(CancellationError):
    pass


class CancellationOwnershipLost(CancellationError):
    """The durable job generation/owner no longer belongs to this executor."""


class ShieldRejectedError(RuntimeError):
    """An outermost shield cannot begin after cancellation became due."""


class CancellationContext:
    """A small, storage-independent cooperative cancellation context."""

    def __init__(
        self,
        *,
        deadline: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._clock = clock
        self._deadline = deadline
        self._lock = RLock()
        self._cancel_reason: str | None = None
        self._cancel_requested = False
        self._deadline_exceeded = False
        self._shield_depth = 0
        self._interrupts: dict[int, Callable[[], None]] = {}
        self._next_interrupt_id = 0

    @classmethod
    def with_timeout(
        cls,
        timeout_seconds: float,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> CancellationContext:
        if timeout_seconds < 0:
            raise ValueError("timeout_seconds must be non-negative")
        return cls(deadline=clock() + timeout_seconds, clock=clock)

    @property
    def deadline(self) -> float | None:
        return self._deadline

    def request_cancel(self, reason: str | None = None) -> bool:
        """Request cancellation once; the first request owns the reason."""

        with self._lock:
            if self._cancel_requested or self._deadline_due_locked():
                return False
            self._cancel_requested = True
            self._cancel_reason = reason
            callbacks = self._interrupt_callbacks_locked()
        self._invoke_interrupts(callbacks)
        return True

    def snapshot(self) -> CancellationSnapshot:
        with self._lock:
            return self._snapshot_locked()

    def checkpoint(self) -> None:
        """Raise when cancellation is due and the context is not shielded."""

        with self._lock:
            snapshot = self._snapshot_locked()
            if snapshot.shield_depth:
                return
        if snapshot.cause is CancellationCause.CANCEL_REQUESTED:
            raise CancellationRequested(snapshot)
        if snapshot.cause is CancellationCause.DEADLINE_EXCEEDED:
            raise DeadlineExceeded(snapshot)

    def kill_fence_state(self) -> KillFenceState:
        """Return cancellation eligibility; never grants signal authority."""

        return self.snapshot().kill_fence_state

    @contextmanager
    def interrupt_on_cancel(self, callback: Callable[[], None]) -> Iterator[None]:
        """Interrupt active blocking work when cancellation becomes eligible.

        Registration is scoped so a later cancellation can never interrupt a
        reused DuckDB connection or provider handle from an already-finished
        operation.
        """

        with self._lock:
            interrupt_id = self._next_interrupt_id
            self._next_interrupt_id += 1
            self._interrupts[interrupt_id] = callback
            invoke_now = (
                self._snapshot_locked().kill_fence_state is KillFenceState.ELIGIBLE
            )
        if invoke_now:
            self._invoke_interrupts((callback,))
        try:
            yield
        finally:
            with self._lock:
                self._interrupts.pop(interrupt_id, None)

    @contextmanager
    def shield(self) -> Iterator[None]:
        """Defer checkpoints and hard termination across a critical section.

        Nested shields are supported.  A new outermost shield is rejected once
        cancellation or the deadline is already due, preventing new critical
        work from starting after shutdown began.
        """

        with self._lock:
            snapshot = self._snapshot_locked()
            if self._shield_depth == 0 and snapshot.cancellation_requested:
                raise ShieldRejectedError(
                    "cannot enter a cancellation shield after cancellation is due"
                )
            self._shield_depth += 1
        try:
            yield
        finally:
            with self._lock:
                self._shield_depth -= 1
                if self._shield_depth < 0:  # defensive invariant
                    self._shield_depth = 0
                    raise RuntimeError("cancellation shield depth underflow")
                callbacks = self._interrupt_callbacks_locked()
            self._invoke_interrupts(callbacks)

    def _interrupt_callbacks_locked(self) -> tuple[Callable[[], None], ...]:
        if self._snapshot_locked().kill_fence_state is not KillFenceState.ELIGIBLE:
            return ()
        return tuple(self._interrupts.values())

    @staticmethod
    def _invoke_interrupts(callbacks: tuple[Callable[[], None], ...]) -> None:
        for callback in callbacks:
            try:
                callback()
            except Exception:
                # Cancellation must remain latched even when an advisory
                # provider/connection interrupt is unavailable.
                continue

    def _deadline_due_locked(self) -> bool:
        if (
            not self._deadline_exceeded
            and self._deadline is not None
            and self._clock() >= self._deadline
        ):
            self._deadline_exceeded = True
        return self._deadline_exceeded

    def _snapshot_locked(self) -> CancellationSnapshot:
        deadline_due = self._deadline_due_locked()
        if self._cancel_requested:
            cause = CancellationCause.CANCEL_REQUESTED
            reason = self._cancel_reason
        elif deadline_due:
            cause = CancellationCause.DEADLINE_EXCEEDED
            reason = "operation deadline exceeded"
        else:
            cause = CancellationCause.ACTIVE
            reason = None
        if cause is CancellationCause.ACTIVE:
            fence = KillFenceState.BLOCKED_ACTIVE
        elif self._shield_depth:
            fence = KillFenceState.BLOCKED_SHIELDED
        else:
            fence = KillFenceState.ELIGIBLE
        return CancellationSnapshot(
            cause=cause,
            reason=reason,
            deadline=self._deadline,
            shield_depth=self._shield_depth,
            kill_fence_state=fence,
        )


@dataclass(frozen=True, slots=True)
class DurableCancellationRecord:
    """Storage row projection consumed without coupling core to one repository."""

    job_id: str
    generation: int
    owner: str
    cancel_requested: bool
    reason: str | None = None


DurableCancellationReader = Callable[[str], DurableCancellationRecord | None]

_CURRENT_CANCELLATION_TOKEN: ContextVar[CancellationToken | None] = ContextVar(
    "eda_platform_current_cancellation_token",
    default=None,
)


def current_cancellation_token() -> CancellationToken | None:
    """Return the cancellation token scoped to the current worker operation."""

    return _CURRENT_CANCELLATION_TOKEN.get()


@contextmanager
def cancellation_scope(token: CancellationToken) -> Iterator[None]:
    """Install ``token`` for implicit query/provider/sandbox cancellation.

    ContextVar reset tokens make nested scopes safe and guarantee that a
    worker operation never leaks its durable ownership into later work.
    """

    reset: Token[CancellationToken | None] = _CURRENT_CANCELLATION_TOKEN.set(token)
    try:
        yield
    finally:
        _CURRENT_CANCELLATION_TOKEN.reset(reset)


class StorageBackedCancellationToken:
    """Cooperative token guarded by durable job ownership and generation.

    The worker/store integration supplies ``reader``. Every checkpoint reloads
    the row; a missing row or ownership/generation change fails closed. Drivers
    may call :meth:`poll` while a provider is in flight, and registered query or
    sandbox interrupts fire as soon as durable cancellation is observed.
    """

    def __init__(
        self,
        *,
        job_id: str,
        generation: int,
        owner: str,
        reader: DurableCancellationReader,
        deadline: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        enter_outer_shield: Callable[[], bool] | None = None,
        exit_outer_shield: Callable[[], bool] | None = None,
    ) -> None:
        if generation < 0:
            raise ValueError("generation must be non-negative")
        if not job_id or not owner:
            raise ValueError("job_id and owner must be non-empty")
        self.job_id = job_id
        self.generation = generation
        self.owner = owner
        self._reader = reader
        self._context = CancellationContext(deadline=deadline, clock=clock)
        self._ownership_lost = False
        self._enter_outer_shield = enter_outer_shield
        self._exit_outer_shield = exit_outer_shield
        self._shield_lock = RLock()
        self._shield_depth = 0

    def poll(self) -> CancellationSnapshot:
        try:
            record = self._reader(self.job_id)
        except Exception:
            record = None
        if (
            record is None
            or record.job_id != self.job_id
            or record.generation != self.generation
            or record.owner != self.owner
        ):
            self._ownership_lost = True
            self._context.request_cancel("cancellation ownership or generation changed")
        elif record.cancel_requested:
            self._context.request_cancel(record.reason or "durable cancellation requested")
        return self._context.snapshot()

    @contextmanager
    def watch(self, *, poll_interval_seconds: float = 0.05) -> Iterator[None]:
        """Poll durable state while a blocking provider/query call is active."""

        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        stopped = Event()

        def poll_until_stopped() -> None:
            while not stopped.wait(poll_interval_seconds):
                if self.poll().cancellation_requested:
                    return

        watcher = Thread(
            target=poll_until_stopped,
            name=f"cancel-watch-{self.job_id}",
            daemon=True,
        )
        self.poll()
        watcher.start()
        try:
            yield
        finally:
            stopped.set()
            watcher.join(timeout=max(1.0, poll_interval_seconds * 2))

    def checkpoint(self) -> None:
        snapshot = self.poll()
        if snapshot.shield_depth:
            return
        if self._ownership_lost:
            raise CancellationOwnershipLost(snapshot)
        self._context.checkpoint()

    def kill_fence_state(self) -> KillFenceState:
        return self.poll().kill_fence_state

    @contextmanager
    def shield(self) -> Iterator[None]:
        outermost = False
        with self._shield_lock:
            outermost = self._shield_depth == 0
            if (
                outermost
                and self._enter_outer_shield is not None
                and not self._enter_outer_shield()
            ):
                self._fail_ownership("durable cancellation shield was rejected")
            self._shield_depth += 1
        try:
            with self._context.shield():
                yield
        finally:
            with self._shield_lock:
                self._shield_depth -= 1
                if self._shield_depth < 0:
                    self._shield_depth = 0
                    raise RuntimeError("storage-backed shield depth underflow")
                leaving_outermost = self._shield_depth == 0
            if (
                leaving_outermost
                and self._exit_outer_shield is not None
                and not self._exit_outer_shield()
            ):
                self._fail_ownership("durable cancellation shield exit lost ownership")

    def _fail_ownership(self, reason: str) -> None:
        self._ownership_lost = True
        self._context.request_cancel(reason)
        raise CancellationOwnershipLost(self._context.snapshot())

    @contextmanager
    def interrupt_on_cancel(self, callback: Callable[[], None]) -> Iterator[None]:
        # Active blocking work cannot call checkpoint itself. Keep the durable
        # row hot-polled for exactly this scope so a storage flag can fire the
        # registered DuckDB/provider interrupt mid-flight.
        with self._context.interrupt_on_cancel(callback), self.watch():
            yield
