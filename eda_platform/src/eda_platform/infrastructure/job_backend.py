"""Local subprocess backend with a durable launch claim and start gate."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from eda_platform.application.ports import JobCommand, JobRef
from eda_platform.core.config import require_absolute_workspace
from eda_platform.core.process_identity import read_process_identity
from eda_platform.core.store import ArtifactStore
from eda_platform.infrastructure.job_lifecycle import (
    JobLifecycleRepository,
    serialize_process_identity,
)
from eda_platform.infrastructure.launch_gate import (
    START_ACK_TIMEOUT_SECONDS,
    open_parent_gate,
)

CANCEL_GRACE_SECONDS = 2.0
CANCEL_SHIELD_POLL_SECONDS = 0.05

# Detach the worker from the parent's job control so a terminal signal aimed at
# the API does not also reach a running job.
_CREATE_NEW_PROCESS_GROUP = 0x00000200
_DETACHED_PROCESS = 0x00000008


@dataclass(frozen=True, slots=True)
class _SpawnOptions:
    start_new_session: bool
    pass_fds: tuple[int, ...]
    creationflags: int


def _detached_spawn_options(inheritable: tuple[int, ...]) -> _SpawnOptions:
    """Popen arguments for a detached child, per platform.

    ``pass_fds`` and ``start_new_session`` are POSIX-only: Windows asserts on a
    non-empty ``pass_fds`` and silently ignores ``start_new_session``, so
    detaching there has to go through creation flags instead. ``creationflags``
    is the mirror image and must stay zero off Windows.
    """
    if sys.platform == "win32":  # pragma: no cover - platform branch
        return _SpawnOptions(
            start_new_session=False,
            pass_fds=(),
            creationflags=_CREATE_NEW_PROCESS_GROUP | _DETACHED_PROCESS,
        )
    return _SpawnOptions(
        start_new_session=True, pass_fds=inheritable, creationflags=0
    )


class LocalProcessJobBackend:
    def __init__(self, workspace: Path | str, store: ArtifactStore | None = None) -> None:
        self._workspace = require_absolute_workspace(workspace)
        self._store = store if store is not None else ArtifactStore(self._workspace)
        self._lifecycle = JobLifecycleRepository(self._store)
        self._owner = f"api-{os.getpid()}-{uuid4().hex[:8]}"
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._cancel_escalations: set[str] = set()
        self._cancel_lock = threading.Lock()

    def enqueue(self, command: JobCommand) -> JobRef:
        self._prune_processes()
        claim = self._lifecycle.claim_launch(command.job_id, owner=self._owner)
        process: subprocess.Popen[bytes] | None = None
        with open_parent_gate(claim.token) as gate:
            options = _detached_spawn_options(gate.inheritable_descriptors())
            try:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "eda_platform.worker.runner",
                        str(self._workspace),
                        command.job_id,
                        claim.token,
                        str(claim.attempt),
                        gate.child_argument(),
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env={**os.environ, **command.env} if command.env else None,
                    start_new_session=options.start_new_session,
                    pass_fds=options.pass_fds,
                    creationflags=options.creationflags,
                )
                gate.wait_for_acknowledgement(START_ACK_TIMEOUT_SECONDS)
                identity = read_process_identity(process.pid)
                if identity is None:
                    raise RuntimeError("Worker PID birth identity is unavailable.")
                self._lifecycle.acknowledge_spawn(
                    claim,
                    pid=process.pid,
                    birth_identity=serialize_process_identity(identity),
                )
                gate.release()
                self._processes[command.job_id] = process
                return JobRef(job_id=command.job_id, pid=process.pid)
            except Exception as exc:
                if process is not None:
                    self._terminate_spawned_process(process)
                self._lifecycle.fail_active(
                    command.job_id,
                    error_code="launch_failed",
                    error_message=str(exc),
                    clear_idempotency=True,
                )
                raise

    def cancel(self, job_id: str) -> None:
        job = self._lifecycle.request_cancel(job_id)
        if str(job["status"]) != "cancelling":
            return
        self.resume_cancel(job_id)

    def resume_cancel(self, job_id: str) -> None:
        """Rebuild a future cancellation timer without appending a new event."""
        job = self._store.get_job(job_id)
        if job is None or str(job["status"]) != "cancelling":
            return
        deadline = _parse_deadline(job.get("cancel_deadline_at"))
        delay = max(0.0, (deadline - datetime.now(UTC)).total_seconds())
        with self._cancel_lock:
            if job_id in self._cancel_escalations:
                return
            self._cancel_escalations.add(job_id)
        thread = threading.Thread(
            target=self._run_cancel_escalation,
            args=(job_id, delay),
            daemon=True,
            name=f"job-cancel-{job_id}",
        )
        thread.start()

    def _run_cancel_escalation(self, job_id: str, delay: float) -> None:
        try:
            self._escalate_cancel(job_id, delay)
        finally:
            with self._cancel_lock:
                self._cancel_escalations.discard(job_id)

    def _escalate_cancel(self, job_id: str, delay: float) -> None:
        if delay > 0:
            time.sleep(delay)
        while True:
            claim = self._lifecycle.cancellation_claim_due(job_id)
            if claim is None:
                return
            process = self._processes.get(job_id)
            if process is not None and process.poll() is not None:
                with suppress(Exception):
                    process.wait(timeout=0)
                if self._lifecycle.finish(
                    claim,
                    "cancelled",
                    error_code="cancelled_before_start",
                    error_message="Worker exited after cancellation before escalation.",
                ):
                    with suppress(Exception):
                        self._lifecycle.materialize_trace(job_id)
                return
            if self._lifecycle.cancellation_blocked_by_critical(claim):
                time.sleep(CANCEL_SHIELD_POLL_SECONDS)
                continue
            if process is not None:
                # Reap promptly after TERM/KILL so a dead child cannot remain
                # a zombie whose PID birth identity still appears live.
                threading.Thread(
                    target=process.wait,
                    daemon=True,
                    name=f"job-reap-{job_id}",
                ).start()
            if not self._lifecycle.terminate_identity_safe(
                job_id,
                grace_seconds=CANCEL_GRACE_SECONDS,
                claim=claim,
            ):
                return
            if process is not None:
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=1)
            changed = self._lifecycle.finish(
                claim,
                "cancelled",
                error_code="cancelled_by_api",
                error_message="Worker was stopped after cancellation was requested.",
            )
            if changed:
                with suppress(Exception):
                    self._lifecycle.materialize_trace(job_id)
            return

    def status(self, job_id: str) -> str:
        job = self._store.get_job(job_id)
        return "unknown" if job is None else str(job["status"])

    def join(self, job_id: str, timeout: float | None = None) -> int | None:
        process = self._processes.get(job_id)
        if process is None:
            return None
        try:
            return process.wait(timeout)
        except subprocess.TimeoutExpired:
            return None

    def _prune_processes(self) -> None:
        for job_id, process in list(self._processes.items()):
            if process.poll() is not None:
                del self._processes[job_id]

    @staticmethod
    def _terminate_spawned_process(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        with suppress(ProcessLookupError):
            process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                process.kill()
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=1)


def _parse_deadline(value: object) -> datetime:
    if isinstance(value, str):
        with suppress(ValueError):
            parsed = datetime.fromisoformat(value)
            return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    return datetime.now(UTC)
