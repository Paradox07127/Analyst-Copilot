"""F-022 durable local-process lifecycle fault and recovery probes."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path

import pytest

from eda_platform.application.ports import JobCommand, JobRef
from eda_platform.application.services.job_service import (
    JobService,
    recover_job_lifecycle,
)
from eda_platform.core.process_identity import (
    ProcessIdentity,
    ProcessIdentityComparison,
    read_process_identity,
)
from eda_platform.core.store import ArtifactStore
from eda_platform.infrastructure.job_backend import LocalProcessJobBackend
from eda_platform.infrastructure.job_lifecycle import (
    JobLifecycleRepository,
    serialize_process_identity,
)
from eda_platform.infrastructure.launch_gate import acknowledge_and_wait
from eda_platform.worker.runner import run_job


class _RecordingBackend:
    def __init__(self) -> None:
        self.commands: list[JobCommand] = []

    def enqueue(self, command: JobCommand) -> JobRef:
        self.commands.append(command)
        return JobRef(job_id=command.job_id)

    def cancel(self, job_id: str) -> None:
        return None

    def status(self, job_id: str) -> str:
        return "queued"


@pytest.fixture
def lifecycle(tmp_path: Path) -> tuple[ArtifactStore, JobLifecycleRepository]:
    store = ArtifactStore(tmp_path)
    store.ensure_project("demo", name="Demo")
    return store, JobLifecycleRepository(store)


def _queue(
    lifecycle: JobLifecycleRepository,
    *,
    job_id: str = "job_f022",
    session_id: str = "run_f022",
    params: dict[str, object] | None = None,
) -> dict:
    return lifecycle.create_queued_job(
        job_id=job_id,
        session_id=session_id,
        project_id="demo",
        kind="auto_eda",
        params_json=json.dumps(params or {}),
        idempotency_key=None,
        lane_key=session_id,
        request_digest=f"digest-{job_id}",
        request_scope=session_id,
    )


def _event_types(store: ArtifactStore, job_id: str) -> list[str]:
    with sqlite3.connect(store.db_path) as conn:
        rows = conn.execute(
            "select event_type from trace_events where job_id = ? order by id",
            (job_id,),
        ).fetchall()
    return [str(row[0]) for row in rows]


def test_only_one_local_process_backend_class_definition() -> None:
    source_root = Path(__file__).parents[2] / "src" / "eda_platform"
    definitions = [
        path
        for path in source_root.rglob("*.py")
        if "class LocalProcessJobBackend" in path.read_text(encoding="utf-8")
    ]
    assert definitions == [source_root / "infrastructure" / "job_backend.py"]


def test_queued_row_and_trace_roll_back_together(
    lifecycle: tuple[ArtifactStore, JobLifecycleRepository],
) -> None:
    store, repository = lifecycle
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            """
            create trigger f022_reject_queued_trace
            before insert on trace_events
            when new.event_type = 'job.queued'
            begin
                select raise(abort, 'injected queued trace failure');
            end
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="queued trace failure"):
        _queue(repository)

    assert store.get_job("job_f022") is None
    assert store.get_session_index_row("run_f022") is None


def test_launch_ack_failure_never_releases_child_to_business(
    lifecycle: tuple[ArtifactStore, JobLifecycleRepository],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, repository = lifecycle
    job = _queue(repository)
    backend = LocalProcessJobBackend(store.root, store)

    def fail_ack(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected ack persistence failure")

    monkeypatch.setattr(backend._lifecycle, "acknowledge_spawn", fail_ack)
    with pytest.raises(RuntimeError, match="ack persistence failure"):
        backend.enqueue(
            JobCommand(
                job_id=str(job["job_id"]),
                session_id=str(job["session_id"]),
                project_id=str(job["project_id"]),
                kind=str(job["kind"]),
                params_json="{}",
            )
        )

    failed = store.get_job(str(job["job_id"]))
    assert failed is not None
    assert failed["status"] == "failed"
    assert _event_types(store, str(job["job_id"])) == ["job.queued", "job.failed"]


def test_backend_argv_has_no_payload_and_runner_uses_db_params(
    lifecycle: tuple[ArtifactStore, JobLifecycleRepository],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, repository = lifecycle
    job = _queue(repository, params={"business_context": "db-truth"})
    captured_argv: list[str] = []

    class FakePopen:
        pid = os.getpid()

        def __init__(self, argv: list[str], **kwargs: object) -> None:
            captured_argv.extend(argv)
            # Drive the real child half of the gate so the argv contract and
            # the handshake stay tested together.
            token = argv[-3]
            # An in-process fake shares the parent's descriptor table, so it
            # must take its own copies before the parent drops the child side.
            gate_fd, ready_fd = (int(part) for part in argv[-1].split(":")[1:])
            child_argument = f"fd:{os.dup(gate_fd)}:{os.dup(ready_fd)}"
            threading.Thread(
                target=lambda: acknowledge_and_wait(child_argument, token, 5.0),
                daemon=True,
            ).start()

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            return None

        def kill(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            return 0

    monkeypatch.setattr(
        "eda_platform.infrastructure.job_backend.subprocess.Popen", FakePopen
    )
    backend = LocalProcessJobBackend(store.root, store)
    backend.enqueue(
        JobCommand(
            job_id=str(job["job_id"]),
            session_id=str(job["session_id"]),
            project_id=str(job["project_id"]),
            kind=str(job["kind"]),
            params_json='{"business_context":"argv-tamper"}',
        )
    )
    assert not any("argv-tamper" in argument for argument in captured_argv)
    assert len(captured_argv) == 8

    launched = store.get_job(str(job["job_id"]))
    assert launched is not None
    observed: list[dict[str, object]] = []
    monkeypatch.setattr(
        "eda_platform.worker.runner._run_auto_eda_job",
        lambda _store, _workspace, _job, params, **_kwargs: observed.append(params),
    )
    run_job(
        str(store.root),
        str(job["job_id"]),
        launch_token=str(launched["launch_token"]),
        launch_attempt=int(launched["launch_attempt"]),
    )
    assert observed == [{"business_context": "db-truth"}]


def test_terminal_trace_fault_rolls_back_job_and_run(
    lifecycle: tuple[ArtifactStore, JobLifecycleRepository],
) -> None:
    store, repository = lifecycle
    job = _queue(repository)
    claim = repository.claim_launch(str(job["job_id"]), owner="test")
    repository.acknowledge_spawn(claim, pid=os.getpid(), birth_identity="test")
    assert repository.child_start(claim) is not None
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            """
            create trigger f022_reject_terminal_trace
            before insert on trace_events
            when new.event_type = 'job.completed'
            begin
                select raise(abort, 'injected terminal trace failure');
            end
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="terminal trace failure"):
        repository.finish(claim, "completed")

    current = store.get_job(str(job["job_id"]))
    run = store.get_session_index_row(str(job["session_id"]))
    with sqlite3.connect(store.db_path) as conn:
        active_job_id = conn.execute(
            "select active_job_id from sessions where session_id = ?",
            (str(job["session_id"]),),
        ).fetchone()[0]
    assert current is not None and current["status"] == "running"
    assert run is not None
    assert run["status"] == "running"
    assert active_job_id == job["job_id"]
    assert _event_types(store, str(job["job_id"])) == ["job.queued", "job.started"]


def test_terminal_row_cannot_be_revived_by_stale_runner(
    lifecycle: tuple[ArtifactStore, JobLifecycleRepository],
) -> None:
    store, repository = lifecycle
    job = _queue(repository)
    claim = repository.claim_launch(str(job["job_id"]), owner="test")
    repository.acknowledge_spawn(claim, pid=os.getpid(), birth_identity="test")
    assert repository.child_start(claim) is not None
    assert repository.finish(claim, "completed") is True

    run_job(
        str(store.root),
        str(job["job_id"]),
        launch_token=claim.token,
        launch_attempt=claim.attempt,
    )

    current = store.get_job(str(job["job_id"]))
    assert current is not None and current["status"] == "completed"
    assert _event_types(store, str(job["job_id"])) == [
        "job.queued",
        "job.started",
        "job.completed",
    ]
    with pytest.raises(SystemExit, match="legacy ungated"):
        from eda_platform.worker.runner import main

        main([str(store.root), str(job["job_id"]), "{}"])
    assert store.get_job(str(job["job_id"]))["status"] == "completed"  # type: ignore[index]


def test_startup_relaunches_durable_queued_params(
    lifecycle: tuple[ArtifactStore, JobLifecycleRepository],
) -> None:
    store, repository = lifecycle
    _queue(repository, params={"business_context": "durable"})
    backend = _RecordingBackend()

    assert recover_job_lifecycle(store, backend) == 1
    assert len(backend.commands) == 1
    assert json.loads(backend.commands[0].params_json) == {
        "business_context": "durable"
    }


def test_filesystem_setup_fault_compensates_before_dispatch(
    lifecycle: tuple[ArtifactStore, JobLifecycleRepository],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _repository = lifecycle
    backend = _RecordingBackend()
    service = JobService(store, backend)

    def fail_start_run(*args: object, **kwargs: object) -> None:
        raise OSError("injected run filesystem failure")

    monkeypatch.setattr(store, "start_session", fail_start_run)
    with pytest.raises(OSError, match="filesystem failure"):
        service._create_and_enqueue(
            "run_setup_fault",
            kind="auto_eda",
            project_id="demo",
            idempotency_key=None,
            build_params=lambda _project_id: {},
            request_scope="run_setup_fault",
        )

    assert backend.commands == []
    failed = store.latest_job_for_lane("run_setup_fault")
    assert failed is not None and failed["status"] == "failed"
    assert store.find_active_job_for_lane("run_setup_fault") is None
    assert _event_types(store, str(failed["job_id"])) == [
        "job.queued",
        "job.failed",
    ]


def test_pid_identity_mismatch_never_signals(
    lifecycle: tuple[ArtifactStore, JobLifecycleRepository],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, repository = lifecycle
    job = _queue(repository)
    claim = repository.claim_launch(str(job["job_id"]), owner="test")
    observed = read_process_identity(os.getpid())
    if observed is None:
        pytest.skip("authoritative process identity unavailable")
    mismatched = ProcessIdentity(
        pid=observed.pid,
        source=observed.source,
        start_token=f"{observed.start_token}-mismatch",
    )
    repository.acknowledge_spawn(
        claim,
        pid=os.getpid(),
        birth_identity=serialize_process_identity(mismatched),
    )
    signals: list[tuple[int, str]] = []
    monkeypatch.setattr(
        "eda_platform.infrastructure.job_lifecycle.request_stop",
        lambda pid: signals.append((pid, "stop")),
    )
    monkeypatch.setattr(
        "eda_platform.infrastructure.job_lifecycle.force_kill",
        lambda pid: signals.append((pid, "kill")),
    )

    assert repository.terminate_identity_safe(str(job["job_id"])) is True
    assert signals == []
    current = store.get_job(str(job["job_id"]))
    assert current is not None and current["kill_fence_state"] == "open"


def test_cooperative_cancel_before_deadline_schedules_without_signal(
    lifecycle: tuple[ArtifactStore, JobLifecycleRepository],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, repository = lifecycle
    job = _queue(repository)
    claim = repository.claim_launch(str(job["job_id"]), owner="test")
    repository.acknowledge_spawn(claim, pid=os.getpid(), birth_identity="test")
    assert repository.child_start(claim) is not None
    backend = LocalProcessJobBackend(store.root, store)
    scheduled: list[tuple[str, float]] = []

    class FakeThread:
        def __init__(
            self,
            *,
            target: object,
            args: tuple[str, float],
            daemon: bool,
            name: str,
        ) -> None:
            scheduled.append(args)

        def start(self) -> None:
            return None

    signals: list[tuple[int, str]] = []
    monkeypatch.setattr(
        "eda_platform.infrastructure.job_backend.threading.Thread", FakeThread
    )
    monkeypatch.setattr(
        "eda_platform.infrastructure.job_lifecycle.request_stop",
        lambda pid: signals.append((pid, "stop")),
    )
    monkeypatch.setattr(
        "eda_platform.infrastructure.job_lifecycle.force_kill",
        lambda pid: signals.append((pid, "kill")),
    )
    backend.cancel(str(job["job_id"]))

    assert scheduled and scheduled[0][0] == job["job_id"]
    assert scheduled[0][1] > 0
    assert signals == []
    current = store.get_job(str(job["job_id"]))
    assert current is not None and current["status"] == "cancelling"


def test_startup_rebuilds_future_cancel_deadline_timer(
    lifecycle: tuple[ArtifactStore, JobLifecycleRepository],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, repository = lifecycle
    job = _queue(repository)
    claim = repository.claim_launch(str(job["job_id"]), owner="test")
    repository.acknowledge_spawn(claim, pid=os.getpid(), birth_identity="test")
    assert repository.child_start(claim) is not None
    repository.request_cancel(str(job["job_id"]), grace_seconds=30)
    scheduled: list[tuple[str, float]] = []

    class FakeThread:
        def __init__(
            self,
            *,
            target: object,
            args: tuple[str, float],
            daemon: bool,
            name: str,
        ) -> None:
            scheduled.append(args)

        def start(self) -> None:
            return None

    monkeypatch.setattr(
        "eda_platform.infrastructure.job_backend.threading.Thread", FakeThread
    )
    backend = LocalProcessJobBackend(store.root, store)

    assert recover_job_lifecycle(store, backend) == 0
    assert scheduled and scheduled[0][0] == job["job_id"]
    assert scheduled[0][1] > 0
    assert _event_types(store, str(job["job_id"])) == [
        "job.queued",
        "job.started",
        "job.cancel_requested",
    ]


def test_stubborn_cancel_after_deadline_uses_term_then_kill(
    lifecycle: tuple[ArtifactStore, JobLifecycleRepository],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, repository = lifecycle
    job = _queue(repository)
    claim = repository.claim_launch(str(job["job_id"]), owner="test")
    observed = read_process_identity(os.getpid())
    if observed is None:
        pytest.skip("authoritative process identity unavailable")
    repository.acknowledge_spawn(
        claim,
        pid=os.getpid(),
        birth_identity=serialize_process_identity(observed),
    )
    assert repository.child_start(claim) is not None
    repository.request_cancel(str(job["job_id"]), grace_seconds=0)
    signals: list[tuple[int, str]] = []
    exits = iter((False, True))
    monkeypatch.setattr(
        "eda_platform.infrastructure.job_lifecycle.check_process_identity",
        lambda _identity: ProcessIdentityComparison.MATCH,
    )
    monkeypatch.setattr(
        "eda_platform.infrastructure.job_lifecycle._wait_identity_exit",
        lambda _identity, _timeout: next(exits),
    )
    monkeypatch.setattr(
        "eda_platform.infrastructure.job_lifecycle.request_stop",
        lambda pid: signals.append((pid, "stop")),
    )
    monkeypatch.setattr(
        "eda_platform.infrastructure.job_lifecycle.force_kill",
        lambda pid: signals.append((pid, "kill")),
    )

    due = repository.cancellation_claim_due(str(job["job_id"]))
    assert due == claim
    assert due is not None
    assert repository.terminate_identity_safe(
        str(job["job_id"]), claim=due, grace_seconds=0
    )
    assert signals == [(os.getpid(), "stop"), (os.getpid(), "kill")]
    assert repository.finish(due, "cancelled") is True
