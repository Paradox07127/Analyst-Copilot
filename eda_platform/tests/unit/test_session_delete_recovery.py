from __future__ import annotations

import json
import shutil
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event

import pytest

from eda_platform.application.job_results import (
    JobResultNotReadyError,
    read_job_result,
    write_job_result,
)
from eda_platform.core import session_deletion
from eda_platform.core.bounded_pagination import (
    JsonlPageIndex,
    ResourcePageIndex,
    run_resource_scopes,
)
from eda_platform.core.session_deletion import (
    AUDIT_SESSION_ID,
    SessionDeletionBlockedError,
    SessionDeletionBusyError,
    SessionDeletionCoordinator,
    SessionDeletionNotFoundError,
    SessionDeletionRetryableError,
)
from eda_platform.core.store import ArtifactStore, SessionStorageDeletingError
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.sessions import SessionManifest, TraceEvent

PROJECT_ID = "demo"
RUN_ID = "run_delete_me"


class InjectedCrash(RuntimeError):
    pass


class _CrashOnce:
    def __init__(self, stage: str, *, ordinal: int | None = None) -> None:
        self.stage = stage
        self.ordinal = ordinal
        self.op_id: str | None = None
        self.fired = False

    def __call__(self, stage: str, op_id: str, ordinal: int | None) -> None:
        if (
            not self.fired
            and stage == self.stage
            and (self.ordinal is None or ordinal == self.ordinal)
        ):
            self.fired = True
            self.op_id = op_id
            raise InjectedCrash(stage)


class _PauseThenCrash:
    def __init__(self, stage: str) -> None:
        self.stage = stage
        self.reached = Event()
        self.release = Event()

    def __call__(self, stage: str, _op_id: str, _ordinal: int | None) -> None:
        if stage == self.stage:
            self.reached.set()
            assert self.release.wait(timeout=3)
            raise InjectedCrash(stage)


def _seed_run(workspace: Path) -> ArtifactStore:
    store = ArtifactStore(workspace)
    store.ensure_project(PROJECT_ID, "Demo")
    store.start_session(PROJECT_ID, RUN_ID)
    store.save_artifact(
        Artifact(
            id="profile_delete",
            type=ArtifactType.DATASET_PROFILE,
            project_id=PROJECT_ID,
            session_id=RUN_ID,
            payload={"dataset_id": "dataset_delete", "name": "orders.csv"},
        )
    )
    store.append_trace(
        PROJECT_ID,
        TraceEvent(
            session_id=RUN_ID,
            event_type="step_completed",
            name="profile",
        ),
    )
    chat = store.project_dir(PROJECT_ID) / "chat" / f"{RUN_ID}.jsonl"
    chat.parent.mkdir(parents=True, exist_ok=True)
    chat.write_text('{"role":"user","content":"hello"}\n', encoding="utf-8")
    now = datetime.now(UTC)
    store.create_pending_action(
        action_hash="a" * 64,
        session_id=RUN_ID,
        project_id=PROJECT_ID,
        kind="cleaning_apply",
        payload_json="{}",
        created_at=now.isoformat(),
        expires_at=(now + timedelta(minutes=30)).isoformat(),
        generation="generation",
        payload_digest="b" * 64,
    )
    direct = store.create_job(
        job_id="job_terminal_direct",
        session_id=RUN_ID,
        project_id=PROJECT_ID,
        kind="auto_eda",
        idempotency_key="direct-key",
    )
    store.mark_job_status(str(direct["job_id"]), "completed")
    derived = store.create_job(
        job_id="job_terminal_derived",
        session_id="rpsess_terminal",
        project_id=PROJECT_ID,
        kind="report_generate",
        lane_key=RUN_ID,
        idempotency_key="derived-key",
    )
    store.mark_job_status(str(derived["job_id"]), "completed")
    return store


def _operation_state(workspace: Path, op_id: str) -> str:
    with sqlite3.connect(workspace / "state.sqlite") as conn:
        row = conn.execute(
            "select state from storage_operations where op_id = ?", (op_id,)
        ).fetchone()
    assert row is not None
    return str(row[0])


def _audit_rows(workspace: Path, op_id: str) -> list[str]:
    with sqlite3.connect(workspace / "state.sqlite") as conn:
        return [
            str(row[0])
            for row in conn.execute(
                "select payload from trace_events where event_key = ?",
                (f"run-delete:{op_id}",),
            ).fetchall()
        ]


def _assert_complete(workspace: Path, op_id: str) -> None:
    store = ArtifactStore(workspace)
    assert store.get_session_index_row(RUN_ID) is None
    assert not store.session_dir(PROJECT_ID, RUN_ID).exists()
    assert not (store.project_dir(PROJECT_ID) / "chat" / f"{RUN_ID}.jsonl").exists()
    with sqlite3.connect(store.db_path) as conn:
        assert conn.execute(
            "select count(*) from artifacts where session_id = ?", (RUN_ID,)
        ).fetchone() == (0,)
        assert conn.execute(
            "select count(*) from trace_events where session_id = ?", (RUN_ID,)
        ).fetchone() == (0,)
        assert conn.execute(
            "select count(*) from pending_actions where session_id = ?", (RUN_ID,)
        ).fetchone() == (0,)
        keys = conn.execute(
            """
            select idempotency_key from jobs
            where job_id in ('job_terminal_direct', 'job_terminal_derived')
            order by job_id
            """
        ).fetchall()
        assert keys == [(None,), (None,)]
        work_paths = [
            str(row[0])
            for row in conn.execute(
                """
                select work_relpath from storage_operation_items
                where op_id = ? order by ordinal
                """,
                (op_id,),
            ).fetchall()
        ]
    assert _operation_state(workspace, op_id) == "done"
    assert len(_audit_rows(workspace, op_id)) == 1
    event = TraceEvent.model_validate_json(_audit_rows(workspace, op_id)[0])
    assert event.event_type == "session.deleted"
    assert event.name == RUN_ID
    assert event.summary["storage_operation_id"] == op_id
    trace = store.project_dir(PROJECT_ID) / "sessions" / AUDIT_SESSION_ID / "trace.jsonl"
    mirrored = [
        TraceEvent.model_validate_json(line)
        for line in trace.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert sum(event.summary.get("event_key") == f"run-delete:{op_id}" for event in mirrored) == 1
    assert all(not (workspace / relative).exists() for relative in work_paths)
    assert not (workspace / ".storage-operations" / "quarantine" / op_id).exists()


def test_delete_happy_path_commits_audit_and_purges_quarantine(
    tmp_path: Path,
) -> None:
    store = _seed_run(tmp_path)

    result = SessionDeletionCoordinator(store).delete(RUN_ID)

    assert result.session_id == RUN_ID
    assert result.project_id == PROJECT_ID
    assert result.deleted
    _assert_complete(tmp_path, result.op_id)
    with pytest.raises(SessionDeletionNotFoundError):
        SessionDeletionCoordinator(ArtifactStore(tmp_path)).delete(RUN_ID)


@pytest.mark.parametrize(
    ("stage", "ordinal"),
    [
        ("after_reserve", None),
        ("after_quarantine_item", 0),
        ("after_quarantine_before_state", None),
        ("before_db_commit", None),
        ("after_db_changes_before_commit", None),
        ("after_db_commit", None),
        ("after_audit_mirror", None),
        ("after_purge_item", 0),
        ("after_done", None),
    ],
)
def test_every_fault_cut_point_recovers_with_a_new_store(
    tmp_path: Path,
    stage: str,
    ordinal: int | None,
) -> None:
    store = _seed_run(tmp_path)
    crash = _CrashOnce(stage, ordinal=ordinal)

    with pytest.raises(InjectedCrash):
        SessionDeletionCoordinator(store, fault_hook=crash).delete(RUN_ID)

    assert crash.op_id is not None
    result = SessionDeletionCoordinator(ArtifactStore(tmp_path)).recover(crash.op_id)
    assert result.op_id == crash.op_id
    _assert_complete(tmp_path, crash.op_id)


def test_final_database_transaction_rolls_back_every_change_on_fault(
    tmp_path: Path,
) -> None:
    store = _seed_run(tmp_path)
    crash = _CrashOnce("after_db_changes_before_commit")

    with pytest.raises(InjectedCrash):
        SessionDeletionCoordinator(store, fault_hook=crash).delete(RUN_ID)

    assert crash.op_id is not None
    with sqlite3.connect(store.db_path) as conn:
        run = conn.execute(
            "select storage_state, delete_op_id from sessions where session_id = ?",
            (RUN_ID,),
        ).fetchone()
        assert run == ("deleting", crash.op_id)
        assert conn.execute(
            "select count(*) from artifacts where session_id = ?", (RUN_ID,)
        ).fetchone() == (1,)
        assert conn.execute(
            "select count(*) from trace_events where event_key = ?",
            (f"run-delete:{crash.op_id}",),
        ).fetchone() == (0,)
    assert _operation_state(tmp_path, crash.op_id) == "fs_applied"

    SessionDeletionCoordinator(ArtifactStore(tmp_path)).recover(crash.op_id)
    _assert_complete(tmp_path, crash.op_id)


def test_source_mutation_before_database_commit_blocks_and_rolls_back(
    tmp_path: Path,
) -> None:
    store = _seed_run(tmp_path)

    def recreate_source(stage: str, _op_id: str, _ordinal: int | None) -> None:
        if stage == "after_db_changes_before_commit":
            store.session_dir(PROJECT_ID, RUN_ID).mkdir(parents=True)

    with pytest.raises(SessionDeletionBlockedError) as caught:
        SessionDeletionCoordinator(store, fault_hook=recreate_source).delete(RUN_ID)

    assert _operation_state(tmp_path, caught.value.op_id) == "blocked"
    with sqlite3.connect(store.db_path) as conn:
        assert conn.execute(
            "select storage_state from sessions where session_id = ?", (RUN_ID,)
        ).fetchone() == ("deleting",)
        assert conn.execute(
            "select count(*) from artifacts where session_id = ?", (RUN_ID,)
        ).fetchone() == (1,)


def test_source_mutation_after_database_commit_prevents_done(
    tmp_path: Path,
) -> None:
    store = _seed_run(tmp_path)

    def recreate_source(stage: str, _op_id: str, _ordinal: int | None) -> None:
        if stage == "after_db_commit":
            store.session_dir(PROJECT_ID, RUN_ID).mkdir(parents=True)

    with pytest.raises(SessionDeletionBlockedError) as caught:
        SessionDeletionCoordinator(store, fault_hook=recreate_source).delete(RUN_ID)

    assert _operation_state(tmp_path, caught.value.op_id) == "db_committed"
    assert store.session_dir(PROJECT_ID, RUN_ID).is_dir()


def test_active_job_created_after_quarantine_blocks_final_database_delete(
    tmp_path: Path,
) -> None:
    store = _seed_run(tmp_path)
    crash = _CrashOnce("before_db_commit")
    with pytest.raises(InjectedCrash):
        SessionDeletionCoordinator(store, fault_hook=crash).delete(RUN_ID)

    assert crash.op_id is not None
    assert _operation_state(tmp_path, crash.op_id) == "fs_applied"
    with sqlite3.connect(store.db_path) as conn:
        # Simulate a pre-fence/legacy database to exercise the coordinator's
        # independent final-commit late-job check.
        conn.execute("drop trigger trg_jobs_reject_deleting_run")
        conn.execute(
            """
            insert into jobs(
                job_id, session_id, project_id, kind, status, created_at, lane_key
            ) values(
                'job_late_lane', 'rpsess_late', ?, 'report_generate', 'queued', ?, ?
            )
            """,
            (PROJECT_ID, datetime.now(UTC).isoformat(), RUN_ID),
        )

    with pytest.raises(SessionDeletionBusyError) as caught:
        SessionDeletionCoordinator(ArtifactStore(tmp_path)).recover(crash.op_id)

    assert caught.value.job_id == "job_late_lane"
    assert _operation_state(tmp_path, crash.op_id) == "fs_applied"
    assert ArtifactStore(tmp_path).get_session_index_row(RUN_ID) is None
    with sqlite3.connect(store.db_path) as conn:
        assert conn.execute(
            "select storage_state from sessions where session_id = ?", (RUN_ID,)
        ).fetchone() == ("deleting",)
    assert _audit_rows(tmp_path, crash.op_id) == []

    # Even a terminal raw/status API update is fenced once deletion owns the
    # run. This row was injected through a deliberately triggerless legacy
    # path, so simulate that legacy owner withdrawing the impossible row.
    with pytest.raises(sqlite3.IntegrityError, match="job target run is deleting"):
        store.mark_job_status("job_late_lane", "completed")
    with sqlite3.connect(store.db_path) as conn:
        conn.execute("delete from jobs where job_id = 'job_late_lane'")
    SessionDeletionCoordinator(ArtifactStore(tmp_path)).recover(crash.op_id)
    _assert_complete(tmp_path, crash.op_id)


def test_db_committed_operation_is_found_by_run_id_after_runs_row_is_gone(
    tmp_path: Path,
) -> None:
    store = _seed_run(tmp_path)
    crash = _CrashOnce("after_db_commit")
    with pytest.raises(InjectedCrash):
        SessionDeletionCoordinator(store, fault_hook=crash).delete(RUN_ID)

    assert crash.op_id is not None
    assert ArtifactStore(tmp_path).get_session_index_row(RUN_ID) is None
    result = SessionDeletionCoordinator(ArtifactStore(tmp_path)).delete(RUN_ID)
    assert result.op_id == crash.op_id
    _assert_complete(tmp_path, crash.op_id)


@pytest.mark.parametrize(
    "writer_kind",
    ["artifact", "manifest", "trace", "chat", "pending", "report"],
)
def test_late_run_writer_waits_for_delete_lock_and_cannot_recreate_source(
    tmp_path: Path,
    writer_kind: str,
) -> None:
    store = _seed_run(tmp_path)
    pause = _PauseThenCrash("after_db_commit")
    coordinator = SessionDeletionCoordinator(store, fault_hook=pause)

    def write_late() -> None:
        late_store = ArtifactStore(tmp_path)
        if writer_kind == "artifact":
            late_store.save_artifact(
                Artifact(
                    id="late_artifact",
                    type=ArtifactType.DATASET_PROFILE,
                    project_id=PROJECT_ID,
                    session_id=RUN_ID,
                    payload={"dataset_id": "late", "name": "late.csv"},
                )
            )
        elif writer_kind == "manifest":
            late_store.write_manifest(
                SessionManifest(
                    session_id=RUN_ID,
                    project_id=PROJECT_ID,
                    input_hashes={},
                    code_version="late",
                )
            )
        elif writer_kind == "trace":
            late_store.append_trace(
                PROJECT_ID,
                TraceEvent(session_id=RUN_ID, event_type="late", name="late"),
            )
        elif writer_kind == "chat":
            late_store.append_chat_line(PROJECT_ID, RUN_ID, '{"role":"user"}')
        elif writer_kind == "pending":
            now = datetime.now(UTC)
            late_store.create_pending_action(
                action_hash="c" * 64,
                session_id=RUN_ID,
                project_id=PROJECT_ID,
                kind="cleaning_apply",
                payload_json="{}",
                created_at=now.isoformat(),
                expires_at=(now + timedelta(minutes=30)).isoformat(),
                generation="late",
                payload_digest="d" * 64,
            )
        else:
            late_store.write_session_text(
                PROJECT_ID,
                RUN_ID,
                "report/report.md",
                "late report",
            )

    with ThreadPoolExecutor(max_workers=2) as pool:
        deleting = pool.submit(coordinator.delete, RUN_ID)
        assert pause.reached.wait(timeout=3)
        late = pool.submit(write_late)
        time.sleep(0.05)
        assert not late.done()
        pause.release.set()
        with pytest.raises(InjectedCrash):
            deleting.result(timeout=3)
        with pytest.raises(SessionStorageDeletingError):
            late.result(timeout=3)

    assert not store.session_dir(PROJECT_ID, RUN_ID).exists()
    assert not (store.project_dir(PROJECT_ID) / "chat" / f"{RUN_ID}.jsonl").exists()


def test_audit_jsonl_is_exactly_once_after_mirror_crash_and_recovery(
    tmp_path: Path,
) -> None:
    store = _seed_run(tmp_path)
    crash = _CrashOnce("after_audit_mirror")
    with pytest.raises(InjectedCrash):
        SessionDeletionCoordinator(store, fault_hook=crash).delete(RUN_ID)

    assert crash.op_id is not None
    coordinator = SessionDeletionCoordinator(ArtifactStore(tmp_path))
    coordinator.recover(crash.op_id)
    coordinator.recover(crash.op_id)
    _assert_complete(tmp_path, crash.op_id)


@pytest.mark.parametrize("existing_kind", ["wrong", "duplicate"])
def test_audit_jsonl_matching_key_requires_one_identical_event(
    tmp_path: Path,
    existing_kind: str,
) -> None:
    store = _seed_run(tmp_path)
    crash = _CrashOnce("after_db_commit")
    with pytest.raises(InjectedCrash):
        SessionDeletionCoordinator(store, fault_hook=crash).delete(RUN_ID)

    assert crash.op_id is not None
    sqlite_payload = _audit_rows(tmp_path, crash.op_id)[0]
    if existing_kind == "wrong":
        wrong = json.loads(sqlite_payload)
        wrong["summary"]["artifact_count"] = 999
        lines = [json.dumps(wrong)]
    else:
        lines = [sqlite_payload, sqlite_payload]
    trace = store.project_dir(PROJECT_ID) / "sessions" / AUDIT_SESSION_ID / "trace.jsonl"
    trace.parent.mkdir(parents=True, exist_ok=True)
    trace.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(SessionDeletionBlockedError):
        SessionDeletionCoordinator(ArtifactStore(tmp_path)).recover(crash.op_id)

    # The frozen schema forbids db_committed -> blocked; the coordinator keeps
    # the irreversible state and raises SessionDeletionBlockedError on every retry.
    assert _operation_state(tmp_path, crash.op_id) == "db_committed"
    assert trace.read_text(encoding="utf-8") == "\n".join(lines) + "\n"


def test_source_symlink_blocks_without_touching_its_target(tmp_path: Path) -> None:
    store = _seed_run(tmp_path)
    session_dir = store.session_dir(PROJECT_ID, RUN_ID)
    shutil.rmtree(session_dir)
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    session_dir.symlink_to(outside, target_is_directory=True)

    with pytest.raises(SessionDeletionBlockedError) as caught:
        SessionDeletionCoordinator(store).delete(RUN_ID)

    assert marker.read_text(encoding="utf-8") == "keep"
    assert session_dir.is_symlink()
    assert _operation_state(tmp_path, caught.value.op_id) == "blocked"
    assert ArtifactStore(tmp_path).get_session_index_row(RUN_ID) is None
    with sqlite3.connect(store.db_path) as conn:
        assert conn.execute(
            "select storage_state from sessions where session_id = ?", (RUN_ID,)
        ).fetchone() == ("deleting",)


def test_source_and_quarantine_both_existing_blocks_recovery(
    tmp_path: Path,
) -> None:
    store = _seed_run(tmp_path)
    crash = _CrashOnce("after_reserve")
    with pytest.raises(InjectedCrash):
        SessionDeletionCoordinator(store, fault_hook=crash).delete(RUN_ID)
    assert crash.op_id is not None
    with sqlite3.connect(store.db_path) as conn:
        work = conn.execute(
            """
            select work_relpath from storage_operation_items
            where op_id = ? and ordinal = 0
            """,
            (crash.op_id,),
        ).fetchone()
    assert work is not None
    work_path = tmp_path / str(work[0])
    work_path.mkdir(parents=True)

    with pytest.raises(SessionDeletionBlockedError):
        SessionDeletionCoordinator(ArtifactStore(tmp_path)).recover(crash.op_id)
    assert _operation_state(tmp_path, crash.op_id) == "blocked"
    assert store.session_dir(PROJECT_ID, RUN_ID).is_dir()


def test_required_run_directory_missing_from_source_and_quarantine_blocks(
    tmp_path: Path,
) -> None:
    store = _seed_run(tmp_path)
    crash = _CrashOnce("after_reserve")
    with pytest.raises(InjectedCrash):
        SessionDeletionCoordinator(store, fault_hook=crash).delete(RUN_ID)

    assert crash.op_id is not None
    shutil.rmtree(store.session_dir(PROJECT_ID, RUN_ID))

    with pytest.raises(SessionDeletionBlockedError):
        SessionDeletionCoordinator(ArtifactStore(tmp_path)).recover(crash.op_id)

    assert _operation_state(tmp_path, crash.op_id) == "blocked"
    assert ArtifactStore(tmp_path).get_session_index_row(RUN_ID) is None
    with sqlite3.connect(store.db_path) as conn:
        assert conn.execute(
            "select storage_state from sessions where session_id = ?", (RUN_ID,)
        ).fetchone() == ("deleting",)
    assert _audit_rows(tmp_path, crash.op_id) == []


def test_rename_oserror_is_retryable_and_forward_recovery_finishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _seed_run(tmp_path)
    original_replace = session_deletion.os.replace
    failed = False

    def fail_first_replace(source: Path, target: Path) -> None:
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("injected rename failure")
        original_replace(source, target)

    monkeypatch.setattr(session_deletion.os, "replace", fail_first_replace)
    with pytest.raises(SessionDeletionRetryableError) as caught:
        SessionDeletionCoordinator(store).delete(RUN_ID)
    assert "injected rename failure" in caught.value.reason
    assert _operation_state(tmp_path, caught.value.op_id) == "prepared"

    monkeypatch.setattr(session_deletion.os, "replace", original_replace)
    SessionDeletionCoordinator(ArtifactStore(tmp_path)).recover(caught.value.op_id)
    _assert_complete(tmp_path, caught.value.op_id)


def test_active_derived_lane_wins_real_sqlite_race_and_delete_is_busy(
    tmp_path: Path,
) -> None:
    store = _seed_run(tmp_path)
    coordinator = SessionDeletionCoordinator(store)
    writer = sqlite3.connect(store.db_path, timeout=5, isolation_level=None)
    writer.execute("pragma busy_timeout = 5000")
    writer.execute("begin immediate")
    writer.execute(
        """
        insert into jobs(
            job_id, session_id, project_id, kind, status, created_at, lane_key
        ) values(
            'job_live_lane', 'rpsess_live', ?, 'report_generate', 'queued', ?, ?
        )
        """,
        (PROJECT_ID, datetime.now(UTC).isoformat(), RUN_ID),
    )

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(coordinator.delete, RUN_ID)
        time.sleep(0.05)
        assert not future.done()
        writer.commit()
        with pytest.raises(SessionDeletionBusyError) as caught:
            future.result(timeout=2)
    writer.close()

    assert caught.value.job_id == "job_live_lane"
    row = ArtifactStore(tmp_path).get_session_index_row(RUN_ID)
    assert row is not None
    with sqlite3.connect(store.db_path) as conn:
        assert conn.execute(
            "select storage_state from sessions where session_id = ?", (RUN_ID,)
        ).fetchone() == ("live",)
        assert conn.execute(
            "select count(*) from storage_operations where resource_key = ?",
            (RUN_ID,),
        ).fetchone() == (0,)


def test_concurrent_same_run_deletes_join_one_operation_and_one_audit(
    tmp_path: Path,
) -> None:
    _seed_run(tmp_path)
    reserved = Event()
    release = Event()

    def pause_after_reserve(stage: str, _op_id: str, _ordinal: int | None) -> None:
        if stage == "after_reserve":
            reserved.set()
            assert release.wait(timeout=2)

    first = SessionDeletionCoordinator(ArtifactStore(tmp_path), fault_hook=pause_after_reserve)
    second = SessionDeletionCoordinator(ArtifactStore(tmp_path))
    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(first.delete, RUN_ID)
        assert reserved.wait(timeout=2)
        second_future = pool.submit(second.delete, RUN_ID)
        time.sleep(0.05)
        assert not second_future.done()
        release.set()
        first_result = first_future.result(timeout=2)
        second_result = second_future.result(timeout=2)

    assert first_result.op_id == second_result.op_id
    _assert_complete(tmp_path, first_result.op_id)
    with sqlite3.connect(tmp_path / "state.sqlite") as conn:
        assert conn.execute(
            """
            select count(*) from storage_operations
            where op_kind = 'delete_session' and resource_key = ?
            """,
            (RUN_ID,),
        ).fetchone() == (1,)


def test_delete_purges_the_run_page_indexes(tmp_path: Path) -> None:
    """Page-index rows are keyed by run, so deleting a run must drop them or
    state.sqlite keeps a full projection of every run ever created."""
    store = _seed_run(tmp_path)
    chat = store.project_dir(PROJECT_ID) / "chat" / f"{RUN_ID}.jsonl"
    JsonlPageIndex(store.db_path, store.root).ensure(chat, accept=lambda _line: True)
    resources = ResourcePageIndex(store.db_path)
    for scope in run_resource_scopes(PROJECT_ID, RUN_ID):
        resources.replace(scope, "version-1", {"items": ['{"x": 1}']})
    with sqlite3.connect(store.db_path) as conn:
        assert conn.execute("select count(*) from jsonl_page_entries").fetchone()[0] > 0
        assert conn.execute(
            "select count(*) from resource_page_entries"
        ).fetchone()[0] > 0

    SessionDeletionCoordinator(store).delete(RUN_ID)

    with sqlite3.connect(store.db_path) as conn:
        counts = {
            table: conn.execute(f"select count(*) from {table}").fetchone()[0]
            for table in (
                "jsonl_page_entries",
                "jsonl_page_sources",
                "resource_page_entries",
                "resource_page_sources",
            )
        }
    assert counts == dict.fromkeys(counts, 0)


def test_delete_removes_job_results_and_sandbox_scratch(tmp_path: Path) -> None:
    """Both trees are run-owned but live outside the run directory, so nothing
    else ever reclaims them once the run row and its media are gone."""
    store = _seed_run(tmp_path)
    store.start_session(PROJECT_ID, "run_keep")
    write_job_result(store.root, PROJECT_ID, RUN_ID, "job_terminal_direct", '{"ok":1}')
    write_job_result(store.root, PROJECT_ID, "run_keep", "job_keep", '{"ok":2}')
    scratch = store.root / "_sandbox" / "code_agent" / PROJECT_ID / RUN_ID
    scratch.mkdir(parents=True)
    (scratch / "analysis.py").write_text("print(1)\n", encoding="utf-8")
    kept_scratch = store.root / "_sandbox" / "code_agent" / PROJECT_ID / "run_keep"
    kept_scratch.mkdir(parents=True)

    SessionDeletionCoordinator(store).delete(RUN_ID)

    with pytest.raises(JobResultNotReadyError):
        read_job_result(store.root, PROJECT_ID, RUN_ID, "job_terminal_direct")
    assert not scratch.exists()
    assert read_job_result(store.root, PROJECT_ID, "run_keep", "job_keep") == '{"ok":2}'
    assert kept_scratch.exists()


def test_delete_is_busy_while_a_data_operation_job_runs_for_the_run(
    tmp_path: Path,
) -> None:
    """A data-operation job carries a ``dop_`` run id and its own lane, so only
    request_scope ties it back to the run whose media it is still writing."""
    store = _seed_run(tmp_path)
    store.create_job(
        job_id="job_dop_active",
        session_id="dop_active",
        project_id=PROJECT_ID,
        kind="cleaning_apply",
        lane_key="dop_lane_active",
        request_scope=RUN_ID,
    )

    with pytest.raises(SessionDeletionBusyError):
        SessionDeletionCoordinator(store).delete(RUN_ID)
