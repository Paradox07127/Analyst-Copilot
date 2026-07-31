"""One-shot backfill of the sessions-table index for historical sessions."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from eda_platform.core.store import ArtifactStore, SessionStorageDeletingError


def backfill_sessions_index(workspace: Path | str) -> int:
    """Populate the session-index columns from on-disk session directories.

    Idempotent: rerunning simply rewrites the same derived values. Disk is the
    source of truth: index rows whose session directory no longer exists are pruned
    (DB rows only — the legacy directory-scan listing never showed them either).
    Returns the number of sessions indexed.
    """
    store = ArtifactStore(workspace)
    projects_root = store.root / "projects"
    if projects_root.is_symlink() or not projects_root.is_dir():
        return 0
    resolved_projects_root = projects_root.resolve()
    refreshed = 0
    for project_dir in sorted(
        p
        for p in projects_root.iterdir()
        if not p.is_symlink()
        and p.is_dir()
        and p.resolve().parent == resolved_projects_root
    ):
        project_id = project_dir.name
        if not store.project_exists(project_id):
            store.ensure_project(project_id, name=project_id)
        sessions_dir = project_dir / "sessions"
        if sessions_dir.is_symlink() or not sessions_dir.is_dir():
            # No sessions/ dir at all: suspicious (unmounted volume, partial copy).
            # Refuse to prune on an untrustworthy disk view (review F1).
            continue
        resolved_sessions_dir = sessions_dir.resolve()
        for session_dir in sorted(
            p
            for p in sessions_dir.iterdir()
            if not p.is_symlink()
            and p.is_dir()
            and p.resolve().parent == resolved_sessions_dir
        ):
            try:
                store.adopt_legacy_session(project_id, session_dir.name)
            except (SessionStorageDeletingError, ValueError):
                # A deletion tombstone always wins over stale/reappeared disk
                # content; unsafe legacy paths are never adopted.
                continue
            _reindex_artifacts(store, project_id, session_dir.name)
            store.refresh_session_index(project_id, session_dir.name)
            refreshed += 1
        _prune_missing_sessions(store, project_id, sessions_dir)
    return refreshed


def _reindex_artifacts(store: ArtifactStore, project_id: str, session_id: str) -> None:
    """Insert missing artifacts-table rows for on-disk artifact JSONs.

    Runs recorded before the artifacts index existed have files but no rows,
    which leaves the datasets/artifacts APIs empty for them. Insert-or-ignore
    keys on the composite PK (artifact_id, project_id, session_id), so rows stolen
    by the legacy single-column-PK upsert are restored per partition while
    rows written by save_artifact stay authoritative."""
    artifacts, _warnings = store.list_artifacts_safe(project_id=project_id, session_id=session_id)
    if not artifacts:
        return
    with closing(sqlite3.connect(store.db_path, timeout=5.0)) as conn, conn:
        for artifact in artifacts:
            path = store.artifact_path(project_id, session_id, artifact.id)
            try:
                rel = str(path.resolve().relative_to(store.root.resolve()))
            except ValueError:
                continue
            conn.execute(
                """
                insert or ignore into artifacts(
                    artifact_id, artifact_type, project_id, session_id, path)
                values(?, ?, ?, ?, ?)
                """,
                (artifact.id, artifact.type.value, project_id, session_id, rel),
            )


def _prune_missing_sessions(
    store: ArtifactStore, project_id: str, sessions_dir: Path
) -> None:
    """Drop session-index rows whose session directory is gone. Index rows only —
    artifacts/trace_events deletion stays the province of delete_session. Each
    candidate is re-checked directly (not via the enumeration) so a dangling
    symlink or transient stat error never causes a deletion (review F1)."""
    with closing(sqlite3.connect(store.db_path, timeout=5.0)) as conn, conn:
        rows = conn.execute(
            """
            select session_id from sessions
            where project_id = ? and storage_state = 'live'
            """,
            (project_id,),
        ).fetchall()
        for (session_id,) in rows:
            candidate = sessions_dir / session_id
            if candidate.exists() or candidate.is_symlink():
                continue
            conn.execute(
                """
                delete from sessions
                where session_id = ? and project_id = ? and storage_state = 'live'
                """,
                (session_id, project_id),
            )
