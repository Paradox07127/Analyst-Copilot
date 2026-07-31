"""Durable critical-section and post-kill publish-fence probes."""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from eda_platform.application.services.job_service import recover_job_lifecycle
from eda_platform.core.process_identity import (
    ProcessIdentityComparison,
    read_process_identity,
)
from eda_platform.core.store import (
    ArtifactStore,
    SessionPublishFencedError,
)
from eda_platform.infrastructure.job_backend import LocalProcessJobBackend
from eda_platform.infrastructure.job_lifecycle import (
    JobLifecycleRepository,
    LaunchClaim,
    serialize_process_identity,
)


@pytest.fixture
def running_job(
    tmp_path: Path,
) -> tuple[ArtifactStore, JobLifecycleRepository, LaunchClaim]:
    store = ArtifactStore(tmp_path)
    store.ensure_project("demo", "Demo")
    lifecycle = JobLifecycleRepository(store)
    lifecycle.create_queued_job(
        job_id="job_fence",
        session_id="session_fence",
        project_id="demo",
        kind="auto_eda",
        params_json="{}",
        idempotency_key=None,
        lane_key="session_fence",
        request_digest="digest",
        request_scope="session_fence",
    )
    claim = lifecycle.claim_launch("job_fence", owner="test")
    identity = read_process_identity(os.getpid())
    if identity is None:
        pytest.skip("authoritative process identity unavailable")
    lifecycle.acknowledge_spawn(
        claim,
        pid=os.getpid(),
        birth_identity=serialize_process_identity(identity),
    )
    assert lifecycle.child_start(claim) is not None
    return store, lifecycle, claim


def test_cross_repository_shield_blocks_signal_until_exit(
    running_job: tuple[ArtifactStore, JobLifecycleRepository, LaunchClaim],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, owner, claim = running_job
    observer = JobLifecycleRepository(ArtifactStore(store.root))
    assert owner.enter_critical(claim)
    assert owner.enter_critical(claim)
    observer.request_cancel(claim.job_id, grace_seconds=0)
    signals: list[tuple[int, str]] = []
    monkeypatch.setattr(
        "eda_platform.infrastructure.job_lifecycle.check_process_identity",
        lambda _identity: ProcessIdentityComparison.MATCH,
    )
    monkeypatch.setattr(
        "eda_platform.infrastructure.job_lifecycle.request_stop",
        lambda pid: signals.append((pid, "stop")),
    )
    monkeypatch.setattr(
        "eda_platform.infrastructure.job_lifecycle.force_kill",
        lambda pid: signals.append((pid, "kill")),
    )

    assert observer.terminate_identity_safe(claim.job_id, claim=claim) is False
    assert signals == []
    assert owner.exit_critical(claim)
    assert observer.terminate_identity_safe(claim.job_id, claim=claim) is False
    assert signals == []
    assert owner.exit_critical(claim)

    exits = iter((False, True))
    monkeypatch.setattr(
        "eda_platform.infrastructure.job_lifecycle._wait_identity_exit",
        lambda _identity, _timeout: next(exits),
    )
    assert observer.terminate_identity_safe(
        claim.job_id, claim=claim, grace_seconds=0
    )
    assert signals == [
        (os.getpid(), "stop"),
        (os.getpid(), "kill"),
    ]


def test_stale_generation_cannot_unshield_current_owner(
    running_job: tuple[ArtifactStore, JobLifecycleRepository, LaunchClaim],
) -> None:
    store, lifecycle, stale = running_job
    assert lifecycle.enter_critical(stale)
    current = LaunchClaim(stale.job_id, "replacement-token", stale.attempt + 1)
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            """
            update jobs set launch_token = ?, launch_attempt = ?,
                state_version = state_version + 1
            where job_id = ?
            """,
            (current.token, current.attempt, current.job_id),
        )

    assert lifecycle.exit_critical(stale) is False
    assert lifecycle.exit_critical(current) is False
    job = store.get_job(stale.job_id)
    assert job is not None
    assert job["critical_depth"] == 1
    assert job["critical_owner_generation"] == stale.attempt

    lifecycle.recover_startup()
    job = store.get_job(stale.job_id)
    assert job is not None
    assert job["critical_depth"] == 0
    assert job["critical_owner_generation"] is None


def test_committed_kill_fence_rejects_late_run_writer(
    running_job: tuple[ArtifactStore, JobLifecycleRepository, LaunchClaim],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, lifecycle, claim = running_job
    lifecycle.request_cancel(claim.job_id, grace_seconds=0)
    monkeypatch.setattr(
        "eda_platform.infrastructure.job_lifecycle.check_process_identity",
        lambda _identity: ProcessIdentityComparison.MATCH,
    )

    assert lifecycle.authorize_signal(claim.job_id, claim=claim) is not None
    with pytest.raises(SessionPublishFencedError, match="publish fence is committed"):
        store.write_session_text("demo", "session_fence", "late.txt", "must-not-publish")
    assert not (store.session_dir("demo", "session_fence") / "late.txt").exists()
    with sqlite3.connect(store.db_path) as conn, pytest.raises(
        sqlite3.IntegrityError, match="publish fence is committed"
    ):
        conn.execute(
            """
            insert into trace_events(session_id, project_id, event_type, name, payload)
            values('session_fence', 'demo', 'late.write', 'late', '{}')
            """
        )


def _wait_status(store: ArtifactStore, job_id: str, status: str) -> dict:
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        job = store.get_job(job_id)
        if job is not None and str(job["status"]) == status:
            return job
        time.sleep(0.02)
    raise AssertionError(f"{job_id} did not reach {status}")


def _running_external_process(
    tmp_path: Path, *, suffix: str
) -> tuple[
    ArtifactStore,
    JobLifecycleRepository,
    LaunchClaim,
    subprocess.Popen[bytes],
]:
    store = ArtifactStore(tmp_path)
    store.ensure_project("demo", "Demo")
    lifecycle = JobLifecycleRepository(store)
    job_id = f"job_{suffix}"
    session_id = f"run_{suffix}"
    lifecycle.create_queued_job(
        job_id=job_id,
        session_id=session_id,
        project_id="demo",
        kind="auto_eda",
        params_json="{}",
        idempotency_key=None,
        lane_key=session_id,
        request_digest=f"digest-{suffix}",
        request_scope=session_id,
    )
    claim = lifecycle.claim_launch(job_id, owner="test")
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    identity = read_process_identity(process.pid)
    if identity is None:
        process.terminate()
        process.wait(timeout=5)
        pytest.skip("authoritative child identity unavailable")
    lifecycle.acknowledge_spawn(
        claim,
        pid=process.pid,
        birth_identity=serialize_process_identity(identity),
    )
    assert lifecycle.child_start(claim) is not None
    return store, lifecycle, claim, process


def test_backend_timer_retries_shield_then_cancels_automatically(
    tmp_path: Path,
) -> None:
    store, lifecycle, claim, process = _running_external_process(
        tmp_path, suffix="timer"
    )
    backend = LocalProcessJobBackend(store.root, store)
    backend._processes[claim.job_id] = process
    try:
        assert lifecycle.enter_critical(claim)
        lifecycle.request_cancel(claim.job_id, grace_seconds=0)
        backend.resume_cancel(claim.job_id)
        time.sleep(0.15)
        assert process.poll() is None
        assert store.get_job(claim.job_id)["status"] == "cancelling"  # type: ignore[index]

        assert lifecycle.exit_critical(claim)
        terminal = _wait_status(store, claim.job_id, "cancelled")
        assert terminal["kill_fence_state"] == "committed"
        assert process.wait(timeout=5) != 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_long_critical_wait_is_rate_limited_and_never_signals(
    running_job: tuple[ArtifactStore, JobLifecycleRepository, LaunchClaim],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, lifecycle, claim = running_job
    backend = LocalProcessJobBackend(store.root, store)
    assert lifecycle.enter_critical(claim)
    lifecycle.request_cancel(claim.job_id, grace_seconds=0)
    polls = 0
    original = backend._lifecycle.cancellation_blocked_by_critical

    def counted(current: LaunchClaim) -> bool:
        nonlocal polls
        polls += 1
        return original(current)

    signals: list[tuple[int, str]] = []
    monkeypatch.setattr(
        backend._lifecycle, "cancellation_blocked_by_critical", counted
    )
    monkeypatch.setattr(
        "eda_platform.infrastructure.job_lifecycle.request_stop",
        lambda pid: signals.append((pid, "stop")),
    )
    monkeypatch.setattr(
        "eda_platform.infrastructure.job_lifecycle.force_kill",
        lambda pid: signals.append((pid, "kill")),
    )
    backend.resume_cancel(claim.job_id)
    time.sleep(0.22)

    assert 2 <= polls <= 8
    assert signals == []
    assert lifecycle.fail_active(
        claim.job_id,
        error_code="test_stop",
        error_message="stop retry loop",
    )
    time.sleep(0.08)
    assert signals == []


def test_startup_resumes_due_shielded_cancellation(
    tmp_path: Path,
) -> None:
    store, lifecycle, claim, process = _running_external_process(
        tmp_path, suffix="restart"
    )
    try:
        assert lifecycle.enter_critical(claim)
        lifecycle.request_cancel(claim.job_id, grace_seconds=0)
        restarted = LocalProcessJobBackend(store.root, store)
        threading.Thread(target=process.wait, daemon=True).start()
        assert recover_job_lifecycle(store, restarted) == 0
        time.sleep(0.12)
        assert process.poll() is None

        assert lifecycle.exit_critical(claim)
        _wait_status(store, claim.job_id, "cancelled")
        assert process.wait(timeout=5) != 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
