from __future__ import annotations

import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event, Lock

import pytest

from eda_platform.application.ports import JobCommand, JobRef
from eda_platform.application.services.job_service import JobConflictError, JobService
from eda_platform.application.services.report_generation_service import (
    ReportGenerationService,
)
from eda_platform.core.session_deletion import SessionDeletionCoordinator
from eda_platform.core.store import ArtifactStore, SessionStorageDeletingError
from eda_platform.infrastructure.job_lifecycle import JobLifecycleRepository
from eda_platform.schemas.artifacts import Artifact, ArtifactType


class _RecordingBackend:
    def __init__(self) -> None:
        self.commands: list[JobCommand] = []
        self._lock = Lock()

    def enqueue(self, command: JobCommand) -> JobRef:
        with self._lock:
            self.commands.append(command)
        return JobRef(job_id=command.job_id)

    def cancel(self, job_id: str) -> None:
        del job_id

    def status(self, job_id: str) -> str:
        del job_id
        return "queued"


def test_two_connections_cannot_reserve_the_same_active_run_lane(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    ArtifactStore(workspace).ensure_project("demo", "Demo")
    barrier = Barrier(2)

    def reserve(job_id: str) -> str:
        store = ArtifactStore(workspace)
        barrier.wait()
        try:
            store.create_job(
                job_id=job_id,
                session_id="shared_run",
                project_id="demo",
                kind="auto_eda",
            )
        except sqlite3.IntegrityError:
            return "conflict"
        return "created"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = sorted(pool.map(reserve, ("job_a", "job_b")))

    assert outcomes == ["conflict", "created"]
    active = ArtifactStore(workspace).list_active_jobs()
    assert [(item["session_id"], item["status"]) for item in active] == [
        ("shared_run", "queued")
    ]


def test_delete_reservation_fences_late_job_on_a_second_connection(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    store = ArtifactStore(workspace)
    store.ensure_project("demo", "Demo")
    store.start_session("demo", "run_delete")
    reserved = Event()
    release = Event()

    def pause_after_reserve(stage: str, _op_id: str, _ordinal: int | None) -> None:
        if stage == "after_reserve":
            reserved.set()
            assert release.wait(timeout=2)

    coordinator = SessionDeletionCoordinator(store, fault_hook=pause_after_reserve)
    with ThreadPoolExecutor(max_workers=2) as pool:
        deleting = pool.submit(coordinator.delete, "run_delete")
        assert reserved.wait(timeout=2)
        late_job = pool.submit(
            ArtifactStore(workspace).create_job,
            job_id="job_late",
            session_id="derived_run",
            project_id="demo",
            kind="report_generate",
            lane_key="run_delete",
        )
        assert not late_job.done()
        release.set()
        assert deleting.result(timeout=2).deleted
        with pytest.raises(SessionStorageDeletingError) as caught:
            late_job.result(timeout=2)
        assert caught.value.session_id == "run_delete"

    assert ArtifactStore(workspace).get_job("job_late") is None


def test_concurrent_identical_idempotent_requests_create_one_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    store = ArtifactStore(workspace)
    store.ensure_project("demo", "Demo")
    seed = workspace / "seed" / "orders.csv"
    seed.parent.mkdir(parents=True)
    seed.write_text("amount\n1\n", encoding="utf-8")
    backend = _RecordingBackend()
    barrier = Barrier(2)
    original_create_job = JobLifecycleRepository.create_queued_job

    def synchronized_create_job(
        self: JobLifecycleRepository, **kwargs: object
    ) -> dict:
        barrier.wait()
        return original_create_job(self, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        JobLifecycleRepository, "create_queued_job", synchronized_create_job
    )
    services = [
        JobService(ArtifactStore(workspace), backend),
        JobService(ArtifactStore(workspace), backend),
    ]

    def submit(index: int) -> str:
        return services[index].create_job(
            "same_run",
            kind="auto_eda",
            project_id="demo",
            datasets=["seed/orders.csv"],
            business_context="same body",
            idempotency_key="same-key",
        ).job_id

    with ThreadPoolExecutor(max_workers=2) as pool:
        job_ids = list(pool.map(submit, range(2)))

    assert job_ids[0] == job_ids[1]
    assert len(backend.commands) == 1
    jobs = ArtifactStore(workspace).list_active_jobs()
    assert [job["job_id"] for job in jobs] == [job_ids[0]]


def test_real_derived_producer_concurrency_replays_one_execution_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    store = ArtifactStore(workspace)
    store.ensure_project("demo", "Demo")
    store.start_session("demo", "source_run")
    store.save_artifact(
        Artifact(
            id="profile_1",
            type=ArtifactType.DATASET_PROFILE,
            project_id="demo",
            session_id="source_run",
            payload={"dataset_id": "dataset_1", "name": "orders"},
        )
    )
    store.mark_session_status("demo", "source_run", "completed")
    backend = _RecordingBackend()
    barrier = Barrier(2)
    original_create_job = JobLifecycleRepository.create_queued_job

    def synchronized_create_job(
        self: JobLifecycleRepository, **kwargs: object
    ) -> dict:
        barrier.wait()
        return original_create_job(self, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        JobLifecycleRepository, "create_queued_job", synchronized_create_job
    )
    services = [
        ReportGenerationService(
            ArtifactStore(workspace),
            JobService(ArtifactStore(workspace), backend),
        )
        for _ in range(2)
    ]

    def submit(index: int) -> tuple[str, str]:
        started = services[index].generate(
            "source_run",
            llm="offline",
            idempotency_key="same-report-key",
        )
        return started.job.job_id, started.execution_session_id

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(submit, range(2)))

    assert results[0] == results[1]
    assert len(backend.commands) == 1
    job = ArtifactStore(workspace).find_by_idempotency_key("same-report-key")
    assert job is not None
    assert job["job_id"] == results[0][0]
    assert job["session_id"] == results[0][1]


def test_derived_jobs_share_their_source_lane_and_retry_stays_idempotent(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "workspace")
    store.ensure_project("demo", "Demo")
    store.start_session("demo", "source_run")
    backend = _RecordingBackend()
    service = JobService(store, backend)

    first = service.create_run_fork_job(
        "fksess_first",
        project_id="demo",
        source_session_id="source_run",
        decision_kind="ml_target",
        idempotency_key="fork-key",
    )
    replay = service.create_run_fork_job(
        "fksess_first",
        project_id="demo",
        source_session_id="source_run",
        decision_kind="ml_target",
        idempotency_key="fork-key",
    )

    assert replay.job_id == first.job_id
    with pytest.raises(JobConflictError, match="active job"):
        service.create_run_fork_job(
            "fksess_second",
            project_id="demo",
            source_session_id="source_run",
            decision_kind="dataset",
        )
    assert len(backend.commands) == 1


def test_lane_race_stays_typed_if_the_winner_settles_before_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ArtifactStore(tmp_path / "workspace")
    store.ensure_project("demo", "Demo")
    store.start_session("demo", "source_run")
    winner = store.create_job(
        job_id="job_winner",
        session_id="fksess_winner",
        project_id="demo",
        kind="session_fork",
        lane_key="source_run",
    )
    store.mark_job_status(str(winner["job_id"]), "completed")

    def lose_reservation(**_kwargs: object) -> dict:
        raise sqlite3.IntegrityError("UNIQUE constraint failed: jobs.lane_key")

    service = JobService(store, _RecordingBackend())
    monkeypatch.setattr(service._lifecycle, "create_queued_job", lose_reservation)

    with pytest.raises(JobConflictError) as raised:
        service.create_run_fork_job(
            "fksess_loser",
            project_id="demo",
            source_session_id="source_run",
            decision_kind="ml_target",
        )
    assert raised.value.job_id == "job_winner"


def test_legacy_duplicate_active_lanes_are_reconciled_before_index(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    store = ArtifactStore(workspace)
    store.ensure_project("demo", "Demo")
    with sqlite3.connect(store.db_path) as conn:
        conn.execute("drop index idx_jobs_active_lane")
        # Simulate a pre-foundation schema with no lane immutability trigger.
        conn.execute("drop trigger trg_jobs_request_identity_immutable")
        conn.execute("drop trigger trg_jobs_lane_key_present_insert")
        conn.execute("drop trigger trg_jobs_lane_key_present_update")
        conn.execute("drop trigger trg_jobs_reject_deleting_run")
        conn.execute("drop trigger trg_jobs_reject_deleting_run_update")
        conn.execute("alter table jobs drop column lane_key")
        conn.execute("delete from jobs")
        conn.executemany(
            """
            insert into jobs(
                job_id, session_id, project_id, kind, status, created_at,
                idempotency_key
            )
            values(?, 'shared_run', 'demo', 'auto_eda', 'queued', ?, ?)
            """,
            (
                ("job_oldest", "2026-07-26T10:00:00+00:00", "oldest-key"),
                ("job_duplicate", "2026-07-26T10:00:01+00:00", "duplicate-key"),
            ),
        )

    migrated = ArtifactStore(workspace)

    active = migrated.find_active_job_for_lane("shared_run")
    assert active is not None
    assert active["job_id"] == "job_oldest"
    duplicate = migrated.get_job("job_duplicate")
    assert duplicate is not None
    assert duplicate["status"] == "failed"
    assert duplicate["error_code"] == "lane_migration_conflict"
    assert duplicate["idempotency_key"] is None
    with pytest.raises(sqlite3.IntegrityError):
        migrated.create_job(
            job_id="job_after_migration",
            session_id="shared_run",
            project_id="demo",
            kind="auto_eda",
        )


def test_legacy_unscoped_derived_job_without_live_worker_is_terminalized(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    store = ArtifactStore(workspace)
    store.ensure_project("demo", "Demo")
    with sqlite3.connect(store.db_path) as conn:
        conn.execute("drop index idx_jobs_active_lane")
        # Simulate a pre-foundation schema with no lane immutability trigger.
        conn.execute("drop trigger trg_jobs_request_identity_immutable")
        conn.execute("drop trigger trg_jobs_lane_key_present_insert")
        conn.execute("drop trigger trg_jobs_lane_key_present_update")
        conn.execute("drop trigger trg_jobs_reject_deleting_run")
        conn.execute("drop trigger trg_jobs_reject_deleting_run_update")
        conn.execute("alter table jobs drop column lane_key")
        conn.execute(
            """
            insert into jobs(
                job_id, session_id, project_id, kind, status, created_at,
                idempotency_key, pid
            )
            values(
                'job_legacy_report', 'rpsess_legacy', 'demo',
                'report_generate', 'queued',
                '2026-07-26T10:00:00+00:00', 'legacy-key', null
            )
            """
        )

    migrated = ArtifactStore(workspace)

    job = migrated.get_job("job_legacy_report")
    assert job is not None
    assert job["status"] == "failed"
    assert job["error_code"] == "lane_migration_unscoped"
    assert job["idempotency_key"] is None
    assert migrated.find_active_job_for_session("rpsess_legacy") is None


def test_legacy_unscoped_derived_job_with_live_worker_blocks_upgrade(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    store = ArtifactStore(workspace)
    store.ensure_project("demo", "Demo")
    with sqlite3.connect(store.db_path) as conn:
        conn.execute("drop index idx_jobs_active_lane")
        # Simulate a pre-foundation schema with no lane immutability trigger.
        conn.execute("drop trigger trg_jobs_request_identity_immutable")
        conn.execute("drop trigger trg_jobs_lane_key_present_insert")
        conn.execute("drop trigger trg_jobs_lane_key_present_update")
        conn.execute("drop trigger trg_jobs_reject_deleting_run")
        conn.execute("drop trigger trg_jobs_reject_deleting_run_update")
        conn.execute("alter table jobs drop column lane_key")
        conn.execute(
            """
            insert into jobs(
                job_id, session_id, project_id, kind, status, created_at, pid
            )
            values(
                'job_live_legacy_fork', 'fksess_legacy', 'demo',
                'run_fork', 'running',
                '2026-07-26T10:00:00+00:00', ?
            )
            """,
            (os.getpid(),),
        )

    with pytest.raises(RuntimeError, match="active legacy derived jobs"):
        ArtifactStore(workspace)


def test_legacy_duplicate_lane_with_multiple_live_workers_blocks_upgrade(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    store = ArtifactStore(workspace)
    store.ensure_project("demo", "Demo")
    with sqlite3.connect(store.db_path) as conn:
        conn.execute("drop index idx_jobs_active_lane")
        # Simulate a pre-foundation schema with no lane immutability trigger.
        conn.execute("drop trigger trg_jobs_request_identity_immutable")
        conn.execute("drop trigger trg_jobs_lane_key_present_insert")
        conn.execute("drop trigger trg_jobs_lane_key_present_update")
        conn.execute("drop trigger trg_jobs_reject_deleting_run")
        conn.execute("drop trigger trg_jobs_reject_deleting_run_update")
        conn.execute("alter table jobs drop column lane_key")
        conn.executemany(
            """
            insert into jobs(
                job_id, session_id, project_id, kind, status, created_at, pid
            )
            values(?, 'shared_run', 'demo', 'auto_eda', 'running', ?, ?)
            """,
            (
                ("job_live_a", "2026-07-26T10:00:00+00:00", os.getpid()),
                ("job_live_b", "2026-07-26T10:00:01+00:00", os.getpid()),
            ),
        )

    with pytest.raises(RuntimeError, match="multiple live workers"):
        ArtifactStore(workspace)
