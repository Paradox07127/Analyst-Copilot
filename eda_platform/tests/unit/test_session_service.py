"""SessionService: DB-only session listing, cursor pagination, internal-run filter."""

from __future__ import annotations

import pathlib
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from eda_platform.application.services.session_service import (
    InvalidCursorError,
    ProjectNotFoundError,
    SessionNotFoundError,
    SessionService,
)
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.sessions import SessionManifest


def _make_run(
    store: ArtifactStore,
    project_id: str,
    session_id: str,
    *,
    created_at: datetime,
    title: str | None = None,
    status: str = "completed",
) -> None:
    store.start_session(project_id, session_id)
    store.write_manifest(
        SessionManifest(
            session_id=session_id,
            project_id=project_id,
            input_hashes={"orders.csv": "abc"},
            code_version="v1",
            created_at=created_at,
            title=title,
        )
    )
    store.mark_session_status(project_id, session_id, status)


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    store = ArtifactStore(tmp_path)
    store.ensure_project("demo", name="Demo")
    return store


def test_empty_workspace_lists_nothing(tmp_path: Path) -> None:
    service = SessionService(ArtifactStore(tmp_path / "fresh"))
    assert service.list_projects() == []
    with pytest.raises(ProjectNotFoundError):
        service.list_sessions("missing")


def test_unfiled_sessions_bucket_is_not_a_user_project(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "unfiled")
    store.ensure_project("unfiled-sessions", name="Unfiled sessions")
    store.ensure_project("client-work", name="Client work")

    projects = SessionService(store).list_projects()

    assert [(project.project_id, project.name) for project in projects] == [
        ("client-work", "Client work")
    ]


def test_list_projects_counts_non_internal_runs(store: ArtifactStore) -> None:
    _make_run(store, "demo", "run_a", created_at=datetime(2026, 7, 1, tzinfo=UTC))
    store.start_session("demo", "run_b__internal_x")
    store.mark_session_status("demo", "run_b__internal_x", "completed")
    projects = SessionService(store).list_projects()
    assert [(p.project_id, p.name, p.session_count) for p in projects] == [("demo", "Demo", 1)]


def test_pagination_walks_all_runs_without_duplicates(store: ArtifactStore) -> None:
    for day in range(1, 6):
        _make_run(
            store,
            "demo",
            f"run_{day:02d}",
            created_at=datetime(2026, 7, day, tzinfo=UTC),
        )
    service = SessionService(store)
    seen: list[str] = []
    cursor: str | None = None
    pages = 0
    while True:
        page = service.list_sessions("demo", limit=2, cursor=cursor)
        seen.extend(run.session_id for run in page.items)
        pages += 1
        if page.next_cursor is None:
            break
        cursor = page.next_cursor
    assert seen == ["run_05", "run_04", "run_03", "run_02", "run_01"]
    assert pages == 3


def test_pagination_tie_breaks_on_session_id(store: ArtifactStore) -> None:
    same_moment = datetime(2026, 7, 4, 12, tzinfo=UTC)
    for suffix in ("aa", "bb", "cc"):
        _make_run(store, "demo", f"run_{suffix}", created_at=same_moment)
    service = SessionService(store)
    first = service.list_sessions("demo", limit=2)
    second = service.list_sessions("demo", limit=2, cursor=first.next_cursor)
    assert [r.session_id for r in first.items] == ["run_cc", "run_bb"]
    assert [r.session_id for r in second.items] == ["run_aa"]
    assert second.next_cursor is None


def test_cursor_past_last_row_returns_empty_page(store: ArtifactStore) -> None:
    _make_run(store, "demo", "run_a", created_at=datetime(2026, 7, 1, tzinfo=UTC))
    service = SessionService(store)
    page = service.list_sessions("demo", limit=1)
    assert page.next_cursor is None
    # A full page with no successor still yields next_cursor=None on follow-up.
    full = service.list_sessions("demo", limit=30)
    assert [r.session_id for r in full.items] == ["run_a"]


def test_invalid_cursor_raises(store: ArtifactStore) -> None:
    service = SessionService(store)
    for bad in ("@@not-base64@@", "aGVsbG8="):  # garbage; valid base64 of non-JSON
        with pytest.raises(InvalidCursorError):
            service.list_sessions("demo", cursor=bad)


def test_internal_runs_are_filtered_from_list(store: ArtifactStore) -> None:
    _make_run(store, "demo", "run_visible", created_at=datetime(2026, 7, 1, tzinfo=UTC))
    _make_run(
        store,
        "demo",
        "investigation_x__internal_1",
        created_at=datetime(2026, 7, 2, tzinfo=UTC),
    )
    page = SessionService(store).list_sessions("demo")
    assert [r.session_id for r in page.items] == ["run_visible"]


def test_run_summary_fields_come_from_index(store: ArtifactStore) -> None:
    _make_run(
        store,
        "demo",
        "run_a",
        created_at=datetime(2026, 7, 1, tzinfo=UTC),
        title="Orders EDA",
    )
    store.save_artifact(
        Artifact(
            id="prof_1",
            type=ArtifactType.DATASET_PROFILE,
            project_id="demo",
            session_id="run_a",
            payload={"name": "orders"},
        )
    )
    store.mark_session_status("demo", "run_a", "completed")
    run = SessionService(store).list_sessions("demo").items[0]
    assert run.title == "Orders EDA"
    assert run.status == "completed"
    assert run.created_at == datetime(2026, 7, 1, tzinfo=UTC)
    assert run.dataset_names == ["orders.csv"]
    assert run.artifact_count == 1
    assert run.chat_message_count == 0


def test_get_run_detail_and_missing_run(store: ArtifactStore) -> None:
    _make_run(store, "demo", "run_a", created_at=datetime(2026, 7, 1, tzinfo=UTC))
    store.save_artifact(
        Artifact(
            id="prof_1",
            type=ArtifactType.DATASET_PROFILE,
            project_id="demo",
            session_id="run_a",
            payload={"name": "orders"},
        )
    )
    store.mark_session_status("demo", "run_a", "completed")
    service = SessionService(store)
    detail = service.get_session_detail("run_a")
    assert detail.project_id == "demo"
    assert detail.code_version == "v1"
    assert detail.seed == 42
    assert detail.artifact_type_counts == {"DatasetProfile": 1}
    assert detail.warnings == []
    with pytest.raises(SessionNotFoundError):
        service.get_session_detail("run_missing")


def test_detail_without_manifest_reports_warning(store: ArtifactStore) -> None:
    store.start_session("demo", "run_bare")
    store.mark_session_status("demo", "run_bare", "failed")
    detail = SessionService(store).get_session_detail("run_bare")
    assert detail.warnings == ["manifest missing"]
    assert detail.code_version is None


def test_list_path_never_touches_the_filesystem(
    store: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Session list must be a pure SQL query (§8.1): no dir enumeration,
    no artifact globs, no manifest/chat reads."""
    for day in (1, 2, 3):
        _make_run(store, "demo", f"run_{day}", created_at=datetime(2026, 7, day, tzinfo=UTC))
    service = SessionService(store)

    def _forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("filesystem access during list_sessions")

    monkeypatch.setattr(pathlib.Path, "iterdir", _forbidden)
    monkeypatch.setattr(pathlib.Path, "glob", _forbidden)
    monkeypatch.setattr(pathlib.Path, "read_text", _forbidden)
    monkeypatch.setattr(pathlib.Path, "open", _forbidden)
    monkeypatch.setattr(ArtifactStore, "list_sessions", _forbidden)
    monkeypatch.setattr(ArtifactStore, "_build_session_info", _forbidden)

    projects = service.list_projects()
    page = service.list_sessions("demo", limit=2)
    rest = service.list_sessions("demo", limit=2, cursor=page.next_cursor)
    assert projects[0].session_count == 3
    assert [r.session_id for r in page.items] + [r.session_id for r in rest.items] == [
        "run_3",
        "run_2",
        "run_1",
    ]


def test_write_hooks_keep_index_columns_fresh(store: ArtifactStore, tmp_path: Path) -> None:
    _make_run(
        store,
        "demo",
        "run_a",
        created_at=datetime(2026, 7, 1, tzinfo=UTC),
        title="Fresh",
        status="running",
    )
    store.mark_session_status("demo", "run_a", "completed")
    with sqlite3.connect(tmp_path / "state.sqlite") as conn:
        row = conn.execute(
            "select title, status, created_at, dataset_names_json, artifact_count"
            " from sessions where session_id = 'run_a'"
        ).fetchone()
    assert row[0] == "Fresh"
    assert row[1] == "completed"
    assert row[2] == "2026-07-01T00:00:00+00:00"
    assert row[3] == '["orders.csv"]'
    assert row[4] == 0


def test_the_no_project_bucket_cannot_be_deleted_as_a_project(tmp_path: Path) -> None:
    """One DELETE on the bucket used to take every standalone session with it.
    It is not in list_projects, so it must not be addressable here either."""
    store = ArtifactStore(tmp_path / "ws")
    store.ensure_project("unfiled-sessions", name="Unfiled sessions")
    store.start_session("unfiled-sessions", "sess_solo")

    with pytest.raises(ProjectNotFoundError):
        SessionService(store).delete_project("unfiled-sessions")

    assert (store.project_dir("unfiled-sessions") / "sessions" / "sess_solo").is_dir()
