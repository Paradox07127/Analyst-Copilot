"""Unified SQLite storage/lifecycle schema and additive migration invariants."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.sessions import TraceEvent

FOUNDATION_MIGRATION_NAME = "unified_storage_foundation_v1"


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {
        str(row[1]) for row in conn.execute(f"pragma table_info({table})").fetchall()
    }


def _primary_key(conn: sqlite3.Connection, table: str) -> tuple[str, ...]:
    rows = conn.execute(f"pragma table_info({table})").fetchall()
    return tuple(str(row[1]) for row in sorted(rows, key=lambda row: int(row[5])) if row[5])


def _schema_sql(conn: sqlite3.Connection, kind: str, name: str) -> str:
    row = conn.execute(
        "select sql from sqlite_master where type = ? and name = ?",
        (kind, name),
    ).fetchone()
    assert row is not None and row[0] is not None
    return " ".join(str(row[0]).lower().split())


def test_fresh_workspace_has_unified_storage_foundation(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    with sqlite3.connect(store.db_path) as conn:
        assert _primary_key(conn, "schema_migrations") == ("version",)
        assert _primary_key(conn, "resource_heads") == (
            "resource_kind",
            "project_id",
            "resource_key",
        )
        assert _primary_key(conn, "storage_operations") == ("op_id",)
        assert _primary_key(conn, "storage_operation_items") == (
            "op_id",
            "ordinal",
        )
        assert _columns(conn, "resource_heads") == {
            "resource_kind",
            "project_id",
            "resource_key",
            "relative_path",
            "version",
            "content_digest",
            "updated_at",
        }
        assert _columns(conn, "storage_operations") == {
            "op_id",
            "op_kind",
            "resource_kind",
            "project_id",
            "resource_key",
            "expected_version",
            "target_version",
            "base_digest",
            "target_digest",
            "request_key",
            "state",
            "error_code",
            "error_message",
            "created_at",
            "updated_at",
        }
        assert _columns(conn, "storage_operation_items") == {
            "op_id",
            "ordinal",
            "mode",
            "source_relpath",
            "work_relpath",
            "base_digest",
            "target_digest",
            "payload",
            "required",
        }
        foreign_keys = conn.execute(
            "pragma foreign_key_list(storage_operation_items)"
        ).fetchall()
        assert [(row[2], row[3], row[4], row[6]) for row in foreign_keys] == [
            ("storage_operations", "op_id", "op_id", "CASCADE")
        ]
        assert {
            "state_version",
            "active_job_id",
            "storage_state",
            "delete_op_id",
        } <= _columns(conn, "sessions")
        assert {
            "state_version",
            "launch_attempt",
            "launch_token",
            "lease_owner",
            "lease_expires_at",
            "heartbeat_at",
            "pid_start_identity",
            "cancel_requested_at",
            "cancel_deadline_at",
            "kill_fence_state",
        } <= _columns(conn, "jobs")
        assert {"job_id", "job_generation", "event_key"} <= _columns(
            conn, "trace_events"
        )
        migrations = conn.execute(
            "select version, name from schema_migrations order by version"
        ).fetchall()
        assert migrations == [(1, FOUNDATION_MIGRATION_NAME)]


def test_foundation_partial_indexes_are_database_owned(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    with sqlite3.connect(store.db_path) as conn:
        active_target = _schema_sql(
            conn, "index", "idx_storage_operations_active_target"
        )
        assert "unique index" in active_target
        assert "(resource_kind, project_id, resource_key)" in active_target
        assert (
            "where state in ('prepared', 'fs_applied', 'db_committed', 'blocked')"
            in active_target
        )
        request_key = _schema_sql(
            conn, "index", "idx_storage_operations_request"
        )
        assert "unique index" in request_key
        assert "(op_kind, request_key)" in request_key
        assert "where request_key is not null" in request_key
        live_runs = _schema_sql(conn, "index", "idx_runs_live_listing")
        assert "where storage_state = 'live'" in live_runs
        trace_jobs = _schema_sql(conn, "index", "idx_trace_events_job")
        assert "(job_id, job_generation, id)" in trace_jobs
        assert "where job_id is not null" in trace_jobs
        event_key = _schema_sql(conn, "index", "idx_trace_events_event_key")
        assert "unique index" in event_key
        assert "where event_key is not null" in event_key
        # F-013/F-020 invariants remain present through this migration.
        assert "request_scope" in _columns(conn, "jobs")
        assert "request_digest" in _columns(conn, "jobs")
        active_lane = _schema_sql(conn, "index", "idx_jobs_active_lane")
        assert "unique index" in active_lane


@pytest.mark.parametrize(
    ("session_id", "lane_key"),
    [("session_deleting", "other_lane"), ("derived_run", "session_deleting")],
)
def test_raw_job_insert_trigger_rejects_deleting_run_or_lane(
    tmp_path: Path,
    session_id: str,
    lane_key: str,
) -> None:
    store = ArtifactStore(tmp_path)
    store.ensure_project("demo", "Demo")
    store.start_session("demo", "session_deleting")
    now = "2026-07-27T00:00:00+00:00"
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            """
            insert into storage_operations(
                op_id, op_kind, resource_kind, project_id, resource_key,
                request_key, state, created_at, updated_at
            ) values(
                'del_trigger', 'delete_session', 'session', 'demo', 'session_deleting',
                'del_trigger', 'prepared', ?, ?
            )
            """,
            (now, now),
        )
        conn.execute(
            """
            update sessions
            set storage_state = 'deleting', delete_op_id = 'del_trigger'
            where session_id = 'session_deleting'
            """
        )
        with pytest.raises(sqlite3.IntegrityError, match="target run is deleting"):
            conn.execute(
                """
                insert into jobs(
                    job_id, session_id, project_id, kind, status, created_at, lane_key
                ) values('job_late', ?, 'demo', 'auto_eda', 'queued', ?, ?)
                """,
                (session_id, now, lane_key),
            )


def test_raw_job_insert_trigger_rejects_active_delete_op_after_run_row_is_gone(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path)
    store.ensure_project("demo", "Demo")
    now = "2026-07-27T00:00:00+00:00"
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            """
            insert into storage_operations(
                op_id, op_kind, resource_kind, project_id, resource_key,
                request_key, state, created_at, updated_at
            ) values(
                'del_committed', 'delete_session', 'session', 'demo', 'run_gone',
                'del_committed', 'db_committed', ?, ?
            )
            """,
            (now, now),
        )
        with pytest.raises(sqlite3.IntegrityError, match="target run is deleting"):
            conn.execute(
                """
                insert into jobs(
                    job_id, session_id, project_id, kind, status, created_at, lane_key
                ) values(
                    'job_after_commit', 'run_gone', 'demo', 'auto_eda',
                    'queued', ?, 'run_gone'
                )
                """,
                (now,),
            )


def test_raw_run_scoped_mutations_are_fenced_while_deleting(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path)
    store.ensure_project("demo", "Demo")
    store.start_session("demo", "session_fenced")
    store.start_session("demo", "run_other")
    store.save_artifact(
        Artifact(
            id="artifact_existing",
            type=ArtifactType.DATASET_PROFILE,
            project_id="demo",
            session_id="session_fenced",
            payload={"dataset_id": "dataset", "name": "orders.csv"},
        )
    )
    store.append_trace(
        "demo",
        TraceEvent(session_id="session_fenced", event_type="existing", name="existing"),
    )
    store.create_pending_action(
        action_hash="a" * 64,
        session_id="session_fenced",
        project_id="demo",
        kind="cleaning_apply",
        payload_json="{}",
        created_at="2026-07-27T00:00:00+00:00",
        expires_at="2026-07-27T01:00:00+00:00",
        generation="generation",
        payload_digest="b" * 64,
    )
    store.create_job(
        job_id="job_existing",
        session_id="session_fenced",
        project_id="demo",
        kind="auto_eda",
    )
    store.create_job(
        job_id="job_retarget",
        session_id="run_other",
        project_id="demo",
        kind="auto_eda",
    )
    now = "2026-07-27T00:00:00+00:00"
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            """
            insert into storage_operations(
                op_id, op_kind, resource_kind, project_id, resource_key,
                request_key, state, created_at, updated_at
            ) values(
                'del_fenced', 'delete_session', 'session', 'demo', 'session_fenced',
                'del_fenced', 'prepared', ?, ?
            )
            """,
            (now, now),
        )
        conn.execute(
            """
            update sessions set storage_state = 'deleting', delete_op_id = 'del_fenced'
            where session_id = 'session_fenced'
            """
        )
        conn.commit()
        mutations = [
            (
                "update artifacts set path = path where artifact_id = 'artifact_existing'",
                "artifact run is not live",
            ),
            (
                """
                insert into artifacts(
                    artifact_id, artifact_type, project_id, session_id, path
                ) values('artifact_late', 'DatasetProfile', 'demo', 'session_fenced', 'late')
                """,
                "artifact run is not live",
            ),
            (
                "update trace_events set name = 'late' where session_id = 'session_fenced'",
                "trace run is not live",
            ),
            (
                """
                insert into trace_events(session_id, project_id, event_type, name, payload)
                values('session_fenced', 'demo', 'late', 'late', '{}')
                """,
                "trace run is not live",
            ),
            (
                """
                update pending_actions set payload_json = '{}'
                where session_id = 'session_fenced'
                """,
                "pending action run is not live",
            ),
            (
                """
                insert into pending_actions(
                    action_hash, session_id, project_id, kind, payload_json,
                    created_at, expires_at, status, generation, payload_digest
                ) values(
                    'cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc',
                    'session_fenced', 'demo', 'cleaning_apply', '{}', ?, ?,
                    'pending', 'late', ?
                )
                """,
                "pending action run is not live",
            ),
            (
                """
                update jobs set status = 'running', state_version = state_version + 1
                where job_id = 'job_existing'
                """,
                "job target run is deleting",
            ),
            (
                """
                update jobs set status = 'completed', state_version = state_version + 1
                where job_id = 'job_existing'
                """,
                "job target run is deleting",
            ),
        ]
        for sql, message in mutations:
            with pytest.raises(sqlite3.IntegrityError, match=message):
                conn.execute(sql, (now, now, "d" * 64) if "cccc" in sql else ())
            conn.rollback()

        # Exercise the deleting-target trigger itself without the independent
        # immutable-identity trigger masking raw run/lane retargets.
        conn.execute("drop trigger trg_jobs_request_identity_immutable")
        for column in ("session_id", "lane_key"):
            with pytest.raises(
                sqlite3.IntegrityError, match="job target run is deleting"
            ):
                conn.execute(
                    f"update jobs set {column} = 'session_fenced' "
                    "where job_id = 'job_retarget'"
                )
            conn.rollback()


def test_job_and_run_state_triggers_fail_closed(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    store.ensure_project("demo", "Demo")
    store.start_session("demo", "run_1")
    job = store.create_job(
        job_id="job_1",
        session_id="run_1",
        project_id="demo",
        kind="auto_eda",
        lane_key="run_1",
        request_digest="a" * 64,
        request_scope="run_1",
    )
    assert job["job_id"] == "job_1"
    assert job["kill_fence_state"] == "open"

    with sqlite3.connect(store.db_path) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="job status"):
            conn.execute("update jobs set status = 'unknown' where job_id = 'job_1'")
        conn.rollback()
        conn.execute(
            "update jobs set state_version = 1, status = 'completed' "
            "where job_id = 'job_1'"
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError, match="terminal job status"):
            conn.execute("update jobs set status = 'running' where job_id = 'job_1'")
        conn.rollback()
        with pytest.raises(sqlite3.IntegrityError, match="job state_version"):
            conn.execute("update jobs set state_version = 0 where job_id = 'job_1'")
        conn.rollback()
        for column, value in (
            ("request_digest", "b" * 64),
            ("request_scope", "other_run"),
            ("lane_key", "other_lane"),
            ("session_id", "other_run"),
        ):
            with pytest.raises(sqlite3.IntegrityError, match="job request identity"):
                conn.execute(
                    f"update jobs set {column} = ? where job_id = 'job_1'",
                    (value,),
                )
            conn.rollback()
        with pytest.raises(sqlite3.IntegrityError, match="kill fence"):
            conn.execute(
                "update jobs set kill_fence_state = 'unsafe' where job_id = 'job_1'"
            )
        conn.rollback()
        conn.execute(
            "update jobs set kill_fence_state = 'committed' where job_id = 'job_1'"
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError, match="kill fence"):
            conn.execute(
                "update jobs set kill_fence_state = 'shielded' where job_id = 'job_1'"
            )
        conn.rollback()
        conn.execute("update sessions set state_version = 2 where session_id = 'run_1'")
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError, match="run state_version"):
            conn.execute("update sessions set state_version = 1 where session_id = 'run_1'")
        conn.rollback()
        with pytest.raises(sqlite3.IntegrityError, match="run storage_state"):
            conn.execute("update sessions set storage_state = 'unknown' where session_id = 'run_1'")


def test_job_lane_and_lifecycle_transition_matrix(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    with sqlite3.connect(store.db_path) as conn:
        for suffix, lane_key in (("null", None), ("empty", "")):
            with pytest.raises(sqlite3.IntegrityError, match="lane_key"):
                conn.execute(
                    """
                    insert into jobs(
                        job_id, session_id, project_id, kind, status,
                        created_at, lane_key
                    ) values(?, ?, 'demo', 'auto_eda', 'queued', ?, ?)
                    """,
                    (
                        f"job_lane_{suffix}",
                        f"run_lane_{suffix}",
                        "2026-07-27T00:00:00+00:00",
                        lane_key,
                    ),
                )
            conn.rollback()

        legal = {
            "queued": ("launching", "running", "completed", "failed", "cancelled"),
            "launching": ("running", "cancelling", "failed", "cancelled"),
            "running": ("cancelling", "completed", "failed", "cancelled"),
            "cancelling": ("cancelled", "failed"),
        }
        ordinal = 0
        for old, targets in legal.items():
            for new in targets:
                ordinal += 1
                job_id = f"job_legal_{ordinal}"
                conn.execute(
                    """
                    insert into jobs(
                        job_id, session_id, project_id, kind, status,
                        created_at, lane_key
                    ) values(?, ?, 'demo', 'auto_eda', ?, ?, ?)
                    """,
                    (
                        job_id,
                        f"run_legal_{ordinal}",
                        old,
                        "2026-07-27T00:00:00+00:00",
                        f"lane_legal_{ordinal}",
                    ),
                )
                conn.execute(
                    "update jobs set status = ?, state_version = 1 where job_id = ?",
                    (new, job_id),
                )
        conn.commit()

        conn.execute(
            """
            insert into jobs(
                job_id, session_id, project_id, kind, status, created_at, lane_key
            ) values(
                'job_invalid_transition', 'run_invalid_transition', 'demo',
                'auto_eda', 'queued', '2026-07-27T00:00:00+00:00',
                'lane_invalid_transition'
            )
            """
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError, match="status transition"):
            conn.execute(
                """
                update jobs set status = 'cancelling', state_version = 1
                where job_id = 'job_invalid_transition'
                """
            )
        conn.rollback()
        with pytest.raises(sqlite3.IntegrityError, match="state_version"):
            conn.execute(
                """
                update jobs set status = 'running'
                where job_id = 'job_invalid_transition'
                """
            )
        conn.rollback()
        with pytest.raises(sqlite3.IntegrityError, match="lane_key|request identity"):
            conn.execute(
                """
                update jobs set lane_key = ''
                where job_id = 'job_invalid_transition'
                """
            )
        conn.rollback()

    store.mark_job_status("job_invalid_transition", "running")
    running = store.get_job("job_invalid_transition")
    assert running is not None
    assert running["status"] == "running"
    assert running["state_version"] == 1
    store.mark_job_status("job_invalid_transition", "running")
    same = store.get_job("job_invalid_transition")
    assert same is not None and same["state_version"] == 1
    store.mark_job_status("job_invalid_transition", "completed")
    completed = store.get_job("job_invalid_transition")
    assert completed is not None and completed["state_version"] == 2

    shielded = store.create_job(
        job_id="job_shielded",
        session_id="run_shielded",
        project_id="demo",
        kind="auto_eda",
    )
    assert shielded["kill_fence_state"] == "open"
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "update jobs set kill_fence_state = 'shielded' "
            "where job_id = 'job_shielded'"
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError, match="kill fence transition"):
            conn.execute(
                "update jobs set kill_fence_state = 'committed' "
                "where job_id = 'job_shielded'"
            )
        conn.rollback()
        conn.execute(
            "update jobs set kill_fence_state = 'open' "
            "where job_id = 'job_shielded'"
        )


def test_resource_head_content_changes_require_newer_version(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            """
            insert into resource_heads(
                resource_kind, project_id, resource_key, relative_path,
                version, content_digest, updated_at
            ) values(
                'board', 'demo', 'main', 'projects/demo/boards/main.json',
                1, 'digest-1', '2026-07-27T00:00:00+00:00'
            )
            """
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError, match="requires a newer version"):
            conn.execute(
                """
                update resource_heads
                set content_digest = 'digest-2'
                where resource_kind = 'board'
                  and project_id = 'demo' and resource_key = 'main'
                """
            )
        conn.rollback()
        conn.execute(
            """
            update resource_heads
            set version = 2, content_digest = 'digest-2'
            where resource_kind = 'board'
              and project_id = 'demo' and resource_key = 'main'
            """
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError, match="resource version"):
            conn.execute(
                """
                update resource_heads set version = 1
                where resource_kind = 'board'
                  and project_id = 'demo' and resource_key = 'main'
                """
            )


def test_storage_operation_uniqueness_and_state_machine_are_atomic(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path)
    with sqlite3.connect(store.db_path) as conn:
        params = (
            "delete",
            "run",
            "demo",
            "run_1",
            "request-1",
            "prepared",
            "2026-07-27T00:00:00+00:00",
            "2026-07-27T00:00:00+00:00",
        )
        conn.execute(
            """
            insert into storage_operations(
                op_id, op_kind, resource_kind, project_id, resource_key,
                request_key, state, created_at, updated_at
            ) values('op_1', ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            params,
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                insert into storage_operations(
                    op_id, op_kind, resource_kind, project_id, resource_key,
                    request_key, state, created_at, updated_at
                ) values(
                    'op_2', 'delete', 'run', 'demo', 'run_1',
                    'request-2', 'blocked',
                    '2026-07-27T00:00:00+00:00',
                    '2026-07-27T00:00:00+00:00'
                )
                """
            )
        conn.rollback()
        conn.execute(
            "update storage_operations set state = 'fs_applied' where op_id = 'op_1'"
        )
        conn.execute(
            "update storage_operations set state = 'done' where op_id = 'op_1'"
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError, match="storage operation state"):
            conn.execute(
                "update storage_operations set state = 'prepared' where op_id = 'op_1'"
            )
        conn.rollback()
        conn.execute(
            """
            insert into storage_operations(
                op_id, op_kind, resource_kind, project_id, resource_key,
                request_key, state, created_at, updated_at
            ) values(
                'op_2', 'delete', 'run', 'demo', 'run_1',
                'request-2', 'prepared',
                '2026-07-27T00:00:00+00:00',
                '2026-07-27T00:00:00+00:00'
            )
            """
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                insert into storage_operations(
                    op_id, op_kind, resource_kind, project_id, resource_key,
                    request_key, state, created_at, updated_at
                ) values(
                    'op_request_duplicate', 'delete', 'run', 'demo', 'run_2',
                    'request-2', 'prepared',
                    '2026-07-27T00:00:00+00:00',
                    '2026-07-27T00:00:00+00:00'
                )
                """
            )
        conn.rollback()
        with pytest.raises(sqlite3.IntegrityError, match="storage operation state"):
            conn.execute(
                """
                insert into storage_operations(
                    op_id, op_kind, resource_kind, project_id, resource_key,
                    state, created_at, updated_at
                ) values(
                    'op_bad', 'delete', 'run', 'demo', 'run_bad',
                    'unknown', '2026-07-27T00:00:00+00:00',
                    '2026-07-27T00:00:00+00:00'
                )
                """
            )


def test_storage_operation_transition_matrix(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    legal = {
        "prepared": ("fs_applied", "blocked", "aborted"),
        "fs_applied": ("db_committed", "done", "blocked"),
        "db_committed": ("done",),
        "blocked": ("prepared", "aborted"),
    }
    with sqlite3.connect(store.db_path) as conn:
        ordinal = 0
        for old, targets in legal.items():
            for new in targets:
                ordinal += 1
                op_id = f"op_legal_{ordinal}"
                conn.execute(
                    """
                    insert into storage_operations(
                        op_id, op_kind, resource_kind, project_id, resource_key,
                        state, created_at, updated_at
                    ) values(
                        ?, 'replace_resource', 'semantic', 'demo', ?,
                        ?, '2026-07-27T00:00:00+00:00',
                        '2026-07-27T00:00:00+00:00'
                    )
                    """,
                    (op_id, f"resource_{ordinal}", old),
                )
                conn.execute(
                    "update storage_operations set state = ? where op_id = ?",
                    (new, op_id),
                )
        conn.commit()

        for ordinal, (old, new) in enumerate(
            (
                ("prepared", "done"),
                ("fs_applied", "aborted"),
                ("db_committed", "blocked"),
                ("blocked", "done"),
            ),
            start=1,
        ):
            op_id = f"op_invalid_{ordinal}"
            conn.execute(
                """
                insert into storage_operations(
                    op_id, op_kind, resource_kind, project_id, resource_key,
                    state, created_at, updated_at
                ) values(
                    ?, 'replace_resource', 'semantic', 'demo', ?,
                    ?, '2026-07-27T00:00:00+00:00',
                    '2026-07-27T00:00:00+00:00'
                )
                """,
                (op_id, f"invalid_resource_{ordinal}", old),
            )
            conn.commit()
            with pytest.raises(sqlite3.IntegrityError, match="state transition"):
                conn.execute(
                    "update storage_operations set state = ? where op_id = ?",
                    (new, op_id),
                )
            conn.rollback()

        conn.execute(
            """
            insert into storage_operations(
                op_id, op_kind, resource_kind, project_id, resource_key,
                state, created_at, updated_at
            ) values(
                'op_same', 'replace_resource', 'semantic', 'demo', 'same',
                'blocked', '2026-07-27T00:00:00+00:00',
                '2026-07-27T00:00:00+00:00'
            )
            """
        )
        conn.execute(
            "update storage_operations set state = 'blocked' where op_id = 'op_same'"
        )


def test_additive_legacy_migration_preserves_jobs_and_is_repeatable(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "state.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            create table sessions (
                session_id text primary key,
                project_id text not null,
                path text not null,
                status text not null default 'running'
            );
            create table jobs (
                job_id text primary key,
                session_id text not null,
                project_id text not null,
                kind text not null,
                status text not null default 'queued',
                cancel_requested integer not null default 0,
                created_at text not null,
                started_at text,
                finished_at text,
                error_code text,
                error_message text,
                idempotency_key text,
                pid integer,
                lane_key text,
                request_digest text,
                request_scope text
            );
            create index idx_jobs_legacy_probe on jobs(kind);
            insert into sessions(session_id, project_id, path, status)
            values('run_legacy', 'demo', 'projects/demo/sessions/run_legacy', 'completed');
            insert into jobs(
                job_id, session_id, project_id, kind, status, cancel_requested,
                created_at, finished_at, idempotency_key, pid, lane_key,
                request_digest, request_scope
            ) values(
                'job_legacy', 'run_legacy', 'demo', 'report_generate',
                'completed', 0, '2026-07-26T00:00:00+00:00',
                '2026-07-26T00:01:00+00:00', 'legacy-key', 123,
                'source_run', 'dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd',
                'source_run'
            );
            """
        )

    first = ArtifactStore(tmp_path)
    second = ArtifactStore(tmp_path)
    assert first.db_path == second.db_path
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            select lane_key, request_digest, request_scope, pid,
                   state_version, launch_attempt, kill_fence_state
            from jobs where job_id = 'job_legacy'
            """
        ).fetchone()
        assert row == (
            "source_run",
            "d" * 64,
            "source_run",
            123,
            0,
            0,
            "shielded",
        )
        assert conn.execute(
            "select count(*) from schema_migrations where version = 1"
        ).fetchone() == (1,)
        assert conn.execute(
            "select 1 from sqlite_master "
            "where type = 'index' and name = 'idx_jobs_legacy_probe'"
        ).fetchone() == (1,)
