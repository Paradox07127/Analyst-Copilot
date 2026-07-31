"""backfill_sessions_index: fills legacy rows, inserts missing rows, idempotent."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from eda_platform.application.backfill import backfill_sessions_index
from eda_platform.application.services.session_service import SessionService
from eda_platform.core.session_deletion import SessionDeletionCoordinator
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.sessions import SessionManifest


def _seed_workspace(tmp_path: Path) -> ArtifactStore:
    store = ArtifactStore(tmp_path)
    store.ensure_project("demo", name="Demo")
    for day in (1, 2):
        session_id = f"run_{day}"
        store.start_session("demo", session_id)
        store.write_manifest(
            SessionManifest(
                session_id=session_id,
                project_id="demo",
                input_hashes={"orders.csv": "abc"},
                code_version="v1",
                created_at=datetime(2026, 7, day, tzinfo=UTC),
                title=f"Run {day}",
            )
        )
        store.mark_session_status("demo", session_id, "completed")
    return store


def _index_snapshot(db_path: Path) -> list[tuple]:
    with sqlite3.connect(db_path) as conn:
        return conn.execute(
            "select session_id, project_id, status, title, created_at,"
            " dataset_names_json, artifact_count, report_status, chat_message_count"
            " from sessions order by session_id"
        ).fetchall()


def test_backfill_restores_legacy_rows(tmp_path: Path) -> None:
    store = _seed_workspace(tmp_path)
    db_path = tmp_path / "state.sqlite"
    populated = _index_snapshot(db_path)
    # Simulate the pre-migration DB: 4-column rows, one run missing entirely,
    # and the project row absent.
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "update sessions set title=null, created_at=null, updated_at=null,"
            " dataset_names_json=null, artifact_count=0, report_status=null,"
            " chat_message_count=0"
        )
        conn.execute("delete from sessions where session_id = 'run_2'")
        conn.execute("delete from projects")
    assert not store.project_exists("demo")

    count = backfill_sessions_index(tmp_path)

    assert count == 2
    # run_2 lost its DB row entirely, so its SQLite-sourced status resets to
    # "unknown"; every derived column is rebuilt from disk.
    expected = [
        row if row[0] != "run_2" else (row[0], row[1], "unknown", *row[3:])
        for row in populated
    ]
    assert _index_snapshot(db_path) == expected
    service = SessionService(ArtifactStore(tmp_path))
    assert [p.project_id for p in service.list_projects()] == ["demo"]
    assert [r.session_id for r in service.list_sessions("demo").items] == ["run_2", "run_1"]


def test_backfill_is_idempotent(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    db_path = tmp_path / "state.sqlite"
    first_count = backfill_sessions_index(tmp_path)
    first = _index_snapshot(db_path)
    second_count = backfill_sessions_index(tmp_path)
    assert (first_count, second_count) == (2, 2)
    assert _index_snapshot(db_path) == first


def test_backfill_never_adopts_reappeared_run_with_delete_tombstone(
    tmp_path: Path,
) -> None:
    store = _seed_workspace(tmp_path)
    stale_manifest = (store.session_dir("demo", "run_1") / "manifest.json").read_bytes()
    assert SessionDeletionCoordinator(store).delete("run_1").deleted
    stale_dir = store.session_dir("demo", "run_1")
    stale_dir.mkdir(parents=True)
    (stale_dir / "manifest.json").write_bytes(stale_manifest)

    backfill_sessions_index(tmp_path)

    with sqlite3.connect(store.db_path) as conn:
        assert conn.execute(
            "select 1 from sessions where session_id = 'run_1'"
        ).fetchone() is None


def test_backfill_empty_workspace(tmp_path: Path) -> None:
    assert backfill_sessions_index(tmp_path / "fresh") == 0


def test_backfill_reindexes_artifacts_missing_from_db(tmp_path: Path) -> None:
    """Pre-index-era runs have artifact files but no DB rows; backfill restores
    them so the datasets/artifacts APIs stop returning empty for old sessions."""
    store = _seed_workspace(tmp_path)
    store.save_artifact(
        Artifact(
            id="prof_backfill_1",
            type=ArtifactType.DATASET_PROFILE,
            project_id="demo",
            session_id="run_1",
            payload={"dataset_id": "ds_x", "name": "orders.csv"},
        )
    )
    with sqlite3.connect(tmp_path / "state.sqlite") as conn:
        conn.execute("delete from artifacts")

    backfill_sessions_index(tmp_path)

    with sqlite3.connect(tmp_path / "state.sqlite") as conn:
        rows = conn.execute(
            "select artifact_id, artifact_type from artifacts where session_id='run_1'"
        ).fetchall()
    assert ("prof_backfill_1", "DatasetProfile") in rows


def test_backfill_prunes_rows_for_vanished_run_dirs(tmp_path: Path) -> None:
    """A DB row whose run directory is gone must not surface in the API list —
    the legacy directory-scan listing never showed it either."""
    store = _seed_workspace(tmp_path)
    import shutil

    shutil.rmtree(store.session_dir("demo", "run_1"))

    count = backfill_sessions_index(tmp_path)

    assert count == 1
    assert [row[0] for row in _index_snapshot(tmp_path / "state.sqlite")] == ["run_2"]
    service = SessionService(ArtifactStore(tmp_path))
    assert [r.session_id for r in service.list_sessions("demo").items] == ["run_2"]


def test_backfill_restores_rows_stolen_by_legacy_upsert(tmp_path: Path) -> None:
    """Index rows the legacy single-PK upsert moved to another partition come
    back: insert-or-ignore now keys on (artifact_id, project_id, session_id)."""
    store = ArtifactStore(tmp_path)
    for project, run in (("proj_a", "run_a"), ("proj_b", "run_b")):
        store.ensure_project(project, name=project)
        store.start_session(project, run)
        store.write_manifest(
            SessionManifest(
                session_id=run, project_id=project, input_hashes={}, code_version="v1"
            )
        )
        store.save_artifact(
            Artifact(
                id="prof_shared01",
                type=ArtifactType.DATASET_PROFILE,
                project_id=project,
                session_id=run,
                payload={"dataset_id": "ds_orders", "name": "orders.csv", "rows": 1},
            )
        )
    # Simulate the legacy steal: proj_a's index row was upserted away.
    with sqlite3.connect(tmp_path / "state.sqlite") as conn:
        conn.execute("delete from artifacts where project_id = 'proj_a'")

    backfill_sessions_index(tmp_path)

    with sqlite3.connect(tmp_path / "state.sqlite") as conn:
        rows = conn.execute(
            "select project_id, session_id from artifacts where artifact_id = 'prof_shared01'"
            " order by project_id"
        ).fetchall()
    assert rows == [("proj_a", "run_a"), ("proj_b", "run_b")]
