from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event

import pytest

from eda_platform.core.cancellation import (
    CancellationCause,
    CancellationContext,
    CancellationOwnershipLost,
    CancellationRequested,
    DeadlineExceeded,
    DurableCancellationRecord,
    KillFence,
    KillFenceState,
    ShieldRejectedError,
    StorageBackedCancellationToken,
    cancellation_scope,
    current_cancellation_token,
)


class _Clock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_cancel_is_cooperative_and_exposes_non_authorizing_kill_fence() -> None:
    context = CancellationContext()
    assert isinstance(context, KillFence)
    assert context.kill_fence_state() is KillFenceState.BLOCKED_ACTIVE
    context.checkpoint()

    assert context.request_cancel("user requested stop")
    assert not context.request_cancel("second reason")
    assert context.kill_fence_state() is KillFenceState.ELIGIBLE
    with pytest.raises(CancellationRequested) as caught:
        context.checkpoint()
    assert caught.value.snapshot.reason == "user requested stop"


def test_deadline_latches_and_raises_at_checkpoint() -> None:
    clock = _Clock()
    context = CancellationContext.with_timeout(5, clock=clock)
    assert context.deadline == 105
    clock.now = 104.999
    context.checkpoint()
    clock.now = 105

    with pytest.raises(DeadlineExceeded):
        context.checkpoint()
    assert context.snapshot().cause is CancellationCause.DEADLINE_EXCEEDED
    assert context.kill_fence_state() is KillFenceState.ELIGIBLE

    # Once observed, expiry cannot be undone even by a faulty test clock.
    clock.now = 1
    with pytest.raises(DeadlineExceeded):
        context.checkpoint()


def test_nested_shield_defers_checkpoint_and_blocks_kill() -> None:
    context = CancellationContext()
    with context.shield():
        assert context.request_cancel("stop after commit")
        assert context.kill_fence_state() is KillFenceState.BLOCKED_SHIELDED
        context.checkpoint()
        with context.shield():
            assert context.snapshot().shield_depth == 2
            context.checkpoint()
        assert context.snapshot().shield_depth == 1
    assert context.kill_fence_state() is KillFenceState.ELIGIBLE
    with pytest.raises(CancellationRequested):
        context.checkpoint()


def test_new_outer_shield_is_rejected_after_cancel_or_deadline() -> None:
    cancelled = CancellationContext()
    cancelled.request_cancel()
    with pytest.raises(ShieldRejectedError):
        with cancelled.shield():
            pass

    clock = _Clock()
    expired = CancellationContext(deadline=clock.now, clock=clock)
    with pytest.raises(ShieldRejectedError):
        with expired.shield():
            pass


def test_cancel_transition_is_thread_safe_and_first_request_wins() -> None:
    context = CancellationContext()
    workers = 16
    barrier = Barrier(workers)

    def cancel(index: int) -> bool:
        barrier.wait()
        return context.request_cancel(f"worker-{index}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        transitions = list(pool.map(cancel, range(workers)))

    assert transitions.count(True) == 1
    snapshot = context.snapshot()
    assert snapshot.cause is CancellationCause.CANCEL_REQUESTED
    assert snapshot.reason is not None and snapshot.reason.startswith("worker-")
    assert snapshot.kill_fence_state is KillFenceState.ELIGIBLE


def test_interrupt_registration_is_scoped_and_deferred_by_shield() -> None:
    context = CancellationContext()
    calls: list[str] = []
    with context.interrupt_on_cancel(lambda: calls.append("interrupt")):
        with context.shield():
            context.request_cancel("stop")
            assert calls == []
        assert calls == ["interrupt"]
    context.request_cancel("again")
    assert calls == ["interrupt"]


def test_storage_token_polls_durable_cancel_and_guards_ownership_generation() -> None:
    current = DurableCancellationRecord(
        job_id="job_1",
        generation=7,
        owner="worker-a",
        cancel_requested=False,
    )

    def read(_job_id: str) -> DurableCancellationRecord:
        return current

    token = StorageBackedCancellationToken(
        job_id="job_1",
        generation=7,
        owner="worker-a",
        reader=read,
    )
    token.checkpoint()

    current = DurableCancellationRecord(
        job_id="job_1",
        generation=7,
        owner="worker-a",
        cancel_requested=True,
        reason="user cancelled",
    )
    with pytest.raises(CancellationRequested, match="user cancelled"):
        token.checkpoint()

    current = DurableCancellationRecord(
        job_id="job_1",
        generation=8,
        owner="worker-b",
        cancel_requested=False,
    )
    replacement = StorageBackedCancellationToken(
        job_id="job_1",
        generation=7,
        owner="worker-a",
        reader=read,
    )
    with pytest.raises(CancellationOwnershipLost):
        replacement.checkpoint()


def test_storage_token_watcher_interrupts_blocking_operation() -> None:
    cancelled = False

    def read(_job_id: str) -> DurableCancellationRecord:
        return DurableCancellationRecord(
            job_id="job_1",
            generation=2,
            owner="worker-a",
            cancel_requested=cancelled,
        )

    token = StorageBackedCancellationToken(
        job_id="job_1",
        generation=2,
        owner="worker-a",
        reader=read,
    )
    interrupted = Event()
    with token.interrupt_on_cancel(interrupted.set), token.watch(
        poll_interval_seconds=0.01
    ):
        cancelled = True
        assert interrupted.wait(timeout=1)


def test_cancellation_scope_is_nested_and_does_not_leak() -> None:
    outer = CancellationContext()
    inner = CancellationContext()
    assert current_cancellation_token() is None
    with cancellation_scope(outer):
        assert current_cancellation_token() is outer
        with cancellation_scope(inner):
            assert current_cancellation_token() is inner
        assert current_cancellation_token() is outer
    assert current_cancellation_token() is None


def test_storage_shield_rejected_by_durable_owner_never_yields_body() -> None:
    entered_body = False
    token = StorageBackedCancellationToken(
        job_id="job_shield",
        generation=1,
        owner="owner",
        reader=lambda _job_id: DurableCancellationRecord(
            job_id="job_shield",
            generation=1,
            owner="owner",
            cancel_requested=False,
        ),
        enter_outer_shield=lambda: False,
        exit_outer_shield=lambda: True,
    )

    with pytest.raises(CancellationOwnershipLost, match="shield was rejected"):
        with token.shield():
            entered_body = True

    assert entered_body is False


def test_storage_shield_exit_failure_is_fail_closed() -> None:
    token = StorageBackedCancellationToken(
        job_id="job_shield",
        generation=1,
        owner="owner",
        reader=lambda _job_id: DurableCancellationRecord(
            job_id="job_shield",
            generation=1,
            owner="owner",
            cancel_requested=False,
        ),
        enter_outer_shield=lambda: True,
        exit_outer_shield=lambda: False,
    )

    with pytest.raises(CancellationOwnershipLost, match="exit lost ownership"):
        with token.shield():
            pass
    with pytest.raises(CancellationOwnershipLost):
        token.checkpoint()
