"""Write-path guards for the runs index (review round 2: F1/F2/F3/F4)."""

from __future__ import annotations

import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from eda_platform.application.backfill import backfill_sessions_index
from eda_platform.application.services.session_service import (
    SessionNotFoundError,
    SessionService,
)
from eda_platform.core.session_deletion import SessionDeletionCoordinator
from eda_platform.core.store import ArtifactStore, SessionStorageDeletingError
from eda_platform.schemas.sessions import SessionManifest


def _make_run(store: ArtifactStore, project_id: str, session_id: str, *, day: int = 1) -> None:
    store.start_session(project_id, session_id)
    store.write_manifest(
        SessionManifest(
            session_id=session_id,
            project_id=project_id,
            input_hashes={"orders.csv": "abc"},
            code_version="v1",
            created_at=datetime(2026, 7, day, tzinfo=UTC),
            title=f"Run {session_id}",
        )
    )
    store.mark_session_status(project_id, session_id, "completed")


def _row(db_path: Path, session_id: str) -> tuple | None:
    with sqlite3.connect(db_path) as conn:
        return conn.execute(
            "select status, title, created_at from sessions where session_id = ?", (session_id,)
        ).fetchone()


def test_mark_status_on_never_started_run_creates_no_row(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    store.ensure_project("demo", name="Demo")
    store.mark_session_status("demo", "never_started", "completed")
    assert _row(tmp_path / "state.sqlite", "never_started") is None


def test_mark_status_does_not_resurrect_deleted_run(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    store.ensure_project("demo", name="Demo")
    _make_run(store, "demo", "doomed")
    assert SessionDeletionCoordinator(store).delete("doomed").deleted
    with pytest.raises(SessionStorageDeletingError):
        store.mark_session_status("demo", "doomed", "completed")
    with pytest.raises(SessionStorageDeletingError):
        store.start_session("demo", "doomed")
    assert _row(tmp_path / "state.sqlite", "doomed") is None


def test_deleting_run_is_hidden_and_cannot_be_revived_by_writers_or_backfill(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path)
    store.ensure_project("demo", name="Demo")
    _make_run(store, "demo", "doomed")

    def crash_after_reserve(stage: str, _op_id: str, _ordinal: int | None) -> None:
        if stage == "after_reserve":
            raise RuntimeError("injected crash")

    with pytest.raises(RuntimeError, match="injected crash"):
        SessionDeletionCoordinator(store, fault_hook=crash_after_reserve).delete("doomed")

    service = SessionService(store)
    assert service.list_sessions("demo").items == []
    assert service.list_projects()[0].session_count == 0
    assert store.list_sessions("demo") == []
    with pytest.raises(SessionNotFoundError):
        service.get_session_detail("doomed")
    with pytest.raises(SessionStorageDeletingError):
        store.start_session("demo", "doomed")

    with pytest.raises(SessionStorageDeletingError):
        store.mark_session_status("demo", "doomed", "completed")
    store.refresh_session_index("demo", "doomed")
    backfill_sessions_index(tmp_path)
    with sqlite3.connect(store.db_path) as conn:
        row = conn.execute(
            """
            select storage_state, status from sessions where session_id = 'doomed'
            """
        ).fetchone()
    assert row == ("deleting", "completed")


def test_torn_manifest_does_not_wipe_known_good_index_values(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    store.ensure_project("demo", name="Demo")
    _make_run(store, "demo", "run_a")
    before = _row(tmp_path / "state.sqlite", "run_a")
    assert before is not None and before[1] == "Run run_a" and before[2] is not None

    manifest_path = store.session_dir("demo", "run_a") / "manifest.json"
    manifest_path.write_text('{"session_id": "run_a", "trunc', encoding="utf-8")
    store.refresh_session_index("demo", "run_a")

    after = _row(tmp_path / "state.sqlite", "run_a")
    assert after is not None
    assert after[0] == "completed"  # status not reset to unknown
    assert after[1] == "Run run_a"  # title survives the torn read
    assert after[2] == before[2]  # created_at survives the torn read


def test_backfill_skips_prune_when_runs_dir_is_missing(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    store.ensure_project("demo", name="Demo")
    _make_run(store, "demo", "run_a")
    shutil.rmtree(store.project_dir("demo") / "sessions")

    backfill_sessions_index(tmp_path)

    assert _row(tmp_path / "state.sqlite", "run_a") is not None


def test_backfill_keeps_dangling_symlink_run_and_other_tables(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    store.ensure_project("demo", name="Demo")
    _make_run(store, "demo", "run_a")
    _make_run(store, "demo", "run_b", day=2)
    run_a_dir = store.session_dir("demo", "run_a")
    shutil.rmtree(run_a_dir)
    run_a_dir.symlink_to(tmp_path / "nowhere")  # dangling

    backfill_sessions_index(tmp_path)

    # Dangling symlink is not treated as "vanished": row survives.
    assert _row(tmp_path / "state.sqlite", "run_a") is not None


def test_backfill_prune_never_touches_artifacts_or_trace_events(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    store.ensure_project("demo", name="Demo")
    _make_run(store, "demo", "run_gone")
    with sqlite3.connect(tmp_path / "state.sqlite") as conn:
        conn.execute(
            "insert into trace_events(session_id, project_id, event_type, name, payload)"
            " values('run_gone','demo','step_started','profile','{}')"
        )
    shutil.rmtree(store.session_dir("demo", "run_gone"))

    backfill_sessions_index(tmp_path)

    with sqlite3.connect(tmp_path / "state.sqlite") as conn:
        runs = conn.execute("select count(*) from sessions where session_id='run_gone'").fetchone()[0]
        traces = conn.execute(
            "select count(*) from trace_events where session_id='run_gone'"
        ).fetchone()[0]
    assert runs == 0  # index row pruned
    assert traces == 1  # evidence tables untouched


def test_internal_run_detail_is_hidden(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    store.ensure_project("demo", name="Demo")
    _make_run(store, "demo", "plan_r1__internal_followup")
    service = SessionService(ArtifactStore(tmp_path))
    with pytest.raises(SessionNotFoundError):
        service.get_session_detail("plan_r1__internal_followup")


def test_null_updated_at_runs_sort_last_across_pages(tmp_path: Path) -> None:
    """Docstring contract with teeth: the page is ordered by updated_at, and a
    run that has never been touched (NULL updated_at) sorts last rather than
    first. Cursor pagination must walk that NULL boundary without losing or
    repeating rows.

    `start_session` alone leaves updated_at NULL — `refresh_session_index` is what
    stamps it — so the untouched runs here deliberately skip the refresh.
    """
    store = ArtifactStore(tmp_path)
    store.ensure_project("demo", name="Demo")
    # _make_run stamps updated_at via mark_session_status, newest last written.
    for day, session_id in ((1, "run_old"), (2, "run_new")):
        _make_run(store, "demo", session_id, day=day)
    for session_id in ("run_null_b", "run_null_a"):
        store.start_session("demo", session_id)

    service = SessionService(ArtifactStore(tmp_path))
    walked: list[str] = []
    cursor: str | None = None
    while True:
        page = service.list_sessions("demo", limit=1, cursor=cursor)
        walked.extend(run.session_id for run in page.items)
        if page.next_cursor is None:
            break
        cursor = page.next_cursor
    assert walked == ["run_new", "run_old", "run_null_b", "run_null_a"]
