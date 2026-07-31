from __future__ import annotations

import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import pytest

from eda_platform.core.cancellation import (
    CancellationContext,
    DurableCancellationRecord,
    StorageBackedCancellationToken,
    cancellation_scope,
)
from eda_platform.core.sandbox import (
    PortableSubprocessBackend,
    SandboxLimits,
    _terminate_process,
)
from eda_platform.core.sandbox_docker import _cancel_container_run


def test_portable_sandbox_cancellation_terminates_long_operation(tmp_path: Path) -> None:
    cancellation = CancellationContext()
    backend = PortableSubprocessBackend(work_root=tmp_path)

    def execute_in_scope():
        with cancellation_scope(cancellation):
            return backend.run_python(
                "while True:\n    pass\n",
                limits=SandboxLimits(timeout_seconds=30),
            )

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(execute_in_scope)
        time.sleep(0.15)
        cancellation.request_cancel("user stopped code")
        artifact = future.result(timeout=3)

    assert artifact.status == "blocked"
    assert artifact.error == "Execution cancelled: user stopped code"
    assert artifact.exit_code is not None


def test_portable_sandbox_polls_durable_cancel_during_execution(tmp_path: Path) -> None:
    durable_cancelled = False

    def read(_job_id: str) -> DurableCancellationRecord:
        return DurableCancellationRecord(
            job_id="job_code",
            generation=9,
            owner="worker-a",
            cancel_requested=durable_cancelled,
            reason="durable code stop",
        )

    cancellation = StorageBackedCancellationToken(
        job_id="job_code",
        generation=9,
        owner="worker-a",
        reader=read,
    )
    backend = PortableSubprocessBackend(work_root=tmp_path)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            backend.run_python,
            "while True:\n    pass\n",
            limits=SandboxLimits(timeout_seconds=30),
            cancellation=cancellation,
        )
        time.sleep(0.15)
        durable_cancelled = True
        artifact = future.result(timeout=3)

    assert artifact.status == "blocked"
    assert artifact.error == "Execution cancelled: durable code stop"


def test_process_termination_escalates_term_to_kill_and_reaps() -> None:
    calls: list[str] = []

    class Process:
        returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            calls.append("terminate")

        def wait(self, timeout: float) -> int:
            calls.append(f"wait:{timeout}")
            if calls.count(f"wait:{timeout}") == 1:
                raise subprocess.TimeoutExpired("python", timeout)
            self.returncode = -9
            return self.returncode

        def kill(self) -> None:
            calls.append("kill")

    _terminate_process(cast(Any, Process()), grace_seconds=0.25)

    assert calls == ["terminate", "wait:0.25", "kill", "wait:0.25"]


def test_docker_cancellation_terminates_before_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "eda_platform.core.sandbox_docker._terminate_docker_cli",
        lambda _proc: calls.append("terminate"),
    )
    monkeypatch.setattr(
        "eda_platform.core.sandbox_docker._cleanup_container",
        lambda _name, _env: calls.append("cleanup"),
    )
    monkeypatch.setattr("eda_platform.core.sandbox_docker.time.sleep", lambda _delay: None)

    _cancel_container_run(cast(Any, object()), "container", {})

    assert calls == ["terminate", "cleanup", "cleanup"]
