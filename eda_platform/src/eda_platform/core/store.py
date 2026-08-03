from __future__ import annotations

import json
import os
import shutil
import sqlite3
import time
import unicodedata
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, closing, contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from eda_platform.core.config import require_absolute_workspace
from eda_platform.core.debug_log import mirror_event_to_debug_log
from eda_platform.core.ids import AUDIT_SESSION_ID, validate_session_id
from eda_platform.core.observability import mirror_trace_event
from eda_platform.core.process_control import pid_is_alive
from eda_platform.core.provenance import env_digest
from eda_platform.core.session_fence import session_key_lock
from eda_platform.core.trace_correlation import current_trace_job
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.sessions import SessionInfo, SessionManifest, TraceEvent

# A lineage node with more children than this is malformed, not merely large.
MAX_SESSION_CHILDREN = 1024


class ProjectOrderConflictError(RuntimeError):
    """A reorder named a project that no longer exists when the fence opened."""

    def __init__(self, project_ids: list[str]) -> None:
        super().__init__(f"Projects no longer exist: {', '.join(project_ids)}")
        self.project_ids = project_ids


class SessionStorageDeletingError(RuntimeError):
    """A write targeted a run already owned by a delete operation."""

    def __init__(self, session_id: str) -> None:
        super().__init__(f"Session is being deleted: {session_id}")
        self.session_id = session_id


class SessionPublishFencedError(RuntimeError):
    """A worker attempted to publish after its hard-kill fence committed."""

    def __init__(self, session_id: str, job_id: str) -> None:
        super().__init__(f"Session publish fence is committed: {session_id} ({job_id})")
        self.session_id = session_id
        self.job_id = job_id


JOB_RESULTS_DIRNAME = "_job_results"


def session_results_relative_path(project_id: str, session_id: str) -> str:
    """Workspace-relative tree holding one session's async job results.

    Results are grouped by session rather than left flat under ``_job_results`` so
    session deletion can derive the tree from ``(project_id, session_id)`` alone and
    reclaim it with the same item protocol as the session's other media.
    """
    validate_session_id(session_id)
    _validate_project_id_segment(project_id)
    return str(PurePosixPath(JOB_RESULTS_DIRNAME) / project_id / session_id)


def session_dir_path(workspace: Path | str, project_id: str, session_id: str) -> Path:
    """Return a contained on-disk directory for one safe session-id segment."""
    validate_session_id(session_id)
    _validate_project_id_segment(project_id)
    projects_root = _safe_projects_root(workspace)
    project_dir = projects_root / project_id
    if not _is_inside(project_dir, projects_root):
        raise ValueError("Project directory must stay inside the workspace projects root.")
    sessions_root = project_dir / "sessions"
    resolved_project = project_dir.resolve()
    if not _is_inside(sessions_root, resolved_project):
        raise ValueError("Sessions directory must stay inside its project directory.")
    session_dir = sessions_root / session_id
    if not _is_inside(session_dir, sessions_root.resolve()):
        raise ValueError("Session directory must stay inside its project sessions directory.")
    return session_dir


def _active_project_job(conn: sqlite3.Connection, project_id: str) -> tuple[object, ...] | None:
    return conn.execute(
        """
        select job_id from jobs
        where project_id = ?
          and status not in ('completed', 'failed', 'cancelled')
        order by created_at, job_id limit 1
        """,
        (project_id,),
    ).fetchone()


class ArtifactStore:
    def __init__(self, root: Path | str, *, init_db: bool = True) -> None:
        """`init_db=False` skips schema setup for a short-lived store that wraps
        a workspace another store already opened (the chat turn tracer)."""
        self.root = require_absolute_workspace(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "state.sqlite"
        if init_db:
            self._init_db()

    def ensure_project(self, project_id: str, name: str) -> None:
        projects_root = _safe_projects_root(self.root)
        projects_root.mkdir(parents=True, exist_ok=True)
        project_dir = self.project_dir(project_id)
        if project_dir.is_symlink():
            raise ValueError(f"Project directory cannot be a symbolic link: {project_id}")
        project_dir.mkdir(exist_ok=True)
        if project_dir.is_symlink() or not project_dir.is_dir():
            raise ValueError(f"Project directory is not a safe directory: {project_id}")
        if not _is_inside(project_dir, projects_root.resolve()):
            raise ValueError("Project directory must stay inside the workspace projects root.")
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                insert into projects(project_id, name, path, created_at, sort_order)
                values(
                    ?, ?, ?, ?,
                    (select coalesce(max(sort_order), -1) + 1 from projects)
                )
                on conflict(project_id) do update set
                    name=excluded.name,
                    path=excluded.path,
                    created_at=coalesce(projects.created_at, excluded.created_at)
                """,
                (project_id, name, self._rel(project_dir), datetime.now(UTC).isoformat()),
            )

    def rename_project(self, project_id: str, name: str) -> bool:
        """Update only the human-facing project name; its stable id/path stays put."""
        with closing(self._connect()) as conn, conn:
            result = conn.execute(
                "update projects set name = ? where project_id = ?",
                (name, project_id),
            )
        return result.rowcount == 1

    def reorder_projects(self, project_ids: Sequence[str]) -> None:
        """Persist one complete user-defined project order atomically.

        ``project_ids`` is deliberately the whole visible list rather than a
        single "move before" instruction.  That makes a retry converge to the
        same order and leaves no ambiguous gaps when a project is moved more
        than once before the next refresh.

        Existence is re-checked inside the fence, not by the caller beforehand:
        the caller's precondition read runs on its own connection, so a project
        deleted in between would otherwise be silently skipped by the update
        while still appearing in the response built from the stale read.
        """
        ordered = list(project_ids)
        if not ordered:
            return
        with closing(self._connect()) as conn, conn:
            conn.execute("begin immediate")
            try:
                placeholders = ",".join("?" * len(ordered))
                live = {
                    row[0]
                    for row in conn.execute(
                        f"select project_id from projects where project_id in ({placeholders})",
                        tuple(ordered),
                    )
                }
                missing = [project_id for project_id in ordered if project_id not in live]
                if missing:
                    raise ProjectOrderConflictError(missing)
                conn.executemany(
                    "update projects set sort_order = ? where project_id = ?",
                    enumerate(ordered),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def start_session(self, project_id: str, session_id: str) -> Path:
        session_dir = self.session_dir(project_id, session_id)
        with session_key_lock(self.root, session_id), closing(self._connect()) as conn:
            conn.execute("begin immediate")
            try:
                deleting_target = self._deleting_target(conn, (session_id,))
                if deleting_target is not None:
                    raise SessionStorageDeletingError(deleting_target)
                existing = conn.execute(
                    "select project_id, storage_state from sessions where session_id = ?",
                    (session_id,),
                ).fetchone()
                if existing is None:
                    deleted = conn.execute(
                        """
                        select 1
                        from storage_operations
                        where resource_key = ? and state != 'aborted'
                        limit 1
                        """,
                        (session_id,),
                    ).fetchone()
                    if deleted is not None:
                        raise SessionStorageDeletingError(session_id)
                if existing is not None and (
                    str(existing[0]) != project_id or str(existing[1]) != "live"
                ):
                    raise SessionStorageDeletingError(session_id)
                self._assert_publish_allowed(conn, project_id, session_id)
                (session_dir / "artifacts").mkdir(parents=True, exist_ok=True)
                (session_dir / "charts").mkdir(parents=True, exist_ok=True)
                (session_dir / "report").mkdir(parents=True, exist_ok=True)
                conn.execute(
                    """
                    insert into sessions(session_id, project_id, path, status)
                    values(?, ?, ?, 'running')
                    on conflict(session_id) do update set
                        path=excluded.path, status='running'
                    where sessions.storage_state = 'live'
                      and sessions.project_id = excluded.project_id
                    """,
                    (session_id, project_id, self._rel(session_dir)),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return session_dir

    def mark_session_status(self, project_id: str, session_id: str, status: str) -> None:
        with session_key_lock(self.root, session_id), closing(self._connect()) as conn:
            conn.execute("begin immediate")
            try:
                if self._deleting_target(conn, (session_id,)) is not None:
                    raise SessionStorageDeletingError(session_id)
                row = conn.execute(
                    "select 1 from sessions where session_id = ? and project_id = ?",
                    (session_id, project_id),
                ).fetchone()
                if row is None:
                    deleted = conn.execute(
                        """
                        select 1 from storage_operations
                        where op_kind = 'delete_session' and resource_kind = 'session'
                          and resource_key = ? and state != 'aborted'
                        limit 1
                        """,
                        (session_id,),
                    ).fetchone()
                    if deleted is not None:
                        raise SessionStorageDeletingError(session_id)
                self._assert_publish_allowed(conn, project_id, session_id)
                conn.execute(
                    """
                    update sessions set status = ?
                    where session_id = ? and project_id = ? and storage_state = 'live'
                    """,
                    (status, session_id, project_id),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        self.refresh_session_index(project_id, session_id)

    def get_session_status(self, session_id: str) -> str | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "select status from sessions where session_id = ? and storage_state = 'live'",
                (session_id,),
            ).fetchone()
        return None if row is None else row[0]

    def save_artifact(self, artifact: Artifact) -> Path:
        # Stamp provenance centrally while preserving an explicitly supplied digest.
        if artifact.env_digest is None:
            artifact.env_digest = env_digest()
        artifact_path = self.artifact_path(artifact.project_id, artifact.session_id, artifact.id)
        with self._session_write_transaction(artifact.project_id, artifact.session_id) as conn:
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = artifact_path.with_suffix(".json.tmp")
            temporary.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
            os.replace(temporary, artifact_path)
            conn.execute(
                """
                insert into artifacts(artifact_id, artifact_type, project_id, session_id, path)
                values(?, ?, ?, ?, ?)
                on conflict(artifact_id, project_id, session_id) do update set
                    artifact_type=excluded.artifact_type,
                    path=excluded.path
                """,
                (
                    artifact.id,
                    artifact.type.value,
                    artifact.project_id,
                    artifact.session_id,
                    self._rel(artifact_path),
                ),
            )
            # Keep the session index live: artifacts saved after a terminal
            # status (SessionMetrics, on-demand reports) must still show up in
            # list counts. Update-only — never fabricates a runs row.
            conn.execute(
                """
                update sessions set
                    artifact_count = (
                        select count(*) from artifacts
                        where project_id = ? and session_id = ?
                    ),
                    updated_at = ?
                where session_id = ? and project_id = ? and storage_state = 'live'
                """,
                (
                    artifact.project_id,
                    artifact.session_id,
                    datetime.now(UTC).isoformat(),
                    artifact.session_id,
                    artifact.project_id,
                ),
            )
        if artifact.type in (ArtifactType.SESSION_SUMMARY, ArtifactType.REPORT_BUNDLE):
            # These carry report_status; refresh the derived column too.
            self.refresh_session_index(artifact.project_id, artifact.session_id)
        return artifact_path

    def mutate_artifact(
        self,
        *,
        project_id: str,
        session_id: str,
        artifact_id: str,
        mutate: Callable[[Artifact], Artifact],
    ) -> Artifact:
        """Cross-process read-modify-write of one existing artifact.

        The callback executes while the run fence and SQLite write transaction
        are held, so it may enforce a payload version without a stale writer
        slipping between the check and atomic file replacement.
        """
        artifact_path = self.artifact_path(project_id, session_id, artifact_id)
        with self._session_write_transaction(project_id, session_id):
            current = Artifact.model_validate_json(artifact_path.read_text(encoding="utf-8"))
            updated = mutate(current)
            if (
                updated.id != artifact_id
                or updated.project_id != project_id
                or updated.session_id != session_id
            ):
                raise ValueError("Artifact mutation cannot change resource identity.")
            temporary = artifact_path.with_suffix(".json.tmp")
            temporary.write_text(updated.model_dump_json(indent=2), encoding="utf-8")
            os.replace(temporary, artifact_path)
        return updated

    def get_artifact(
        self,
        artifact_id: str,
        *,
        project_id: str | None = None,
        session_id: str | None = None,
    ) -> Artifact:
        """Read an artifact payload by id.

        artifact_id is content-derived, so the same id can be indexed under
        several (project_id, session_id) partitions. Incomplete scope is accepted
        only when it identifies exactly one row; ambiguous identity fails
        closed instead of selecting whichever partition was indexed last.
        """
        sql = "select path from artifacts where artifact_id = ?"
        params: list[object] = [artifact_id]
        if project_id is not None:
            sql += " and project_id = ?"
            params.append(project_id)
        if session_id is not None:
            sql += " and session_id = ?"
            params.append(session_id)
        sql += " order by rowid desc limit 2"
        with closing(self._connect()) as conn:
            rows = conn.execute(sql, params).fetchall()
        if not rows:
            raise KeyError(f"Artifact not found: {artifact_id}")
        if len(rows) > 1:
            raise ValueError(
                f"ambiguous artifact identity: {artifact_id!r} requires project_id and session_id"
            )
        return Artifact.model_validate_json(self._abs(rows[0][0]).read_text(encoding="utf-8"))

    def list_artifacts(self, *, project_id: str, session_id: str) -> list[Artifact]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                select path from artifacts
                where project_id = ? and session_id = ?
                order by rowid
                """,
                (project_id, session_id),
            ).fetchall()
        return [
            Artifact.model_validate_json(self._abs(row[0]).read_text(encoding="utf-8"))
            for row in rows
        ]

    def reset_session_outputs(self, *, project_id: str, session_id: str) -> None:
        """Discard generated outputs before reusing a run with new inputs.

        Source uploads remain at project scope. Only session-derived artifacts,
        reports, charts, checkpoints, traces, and pending HITL actions are
        removed, so downstream consumers cannot observe stale evidence.
        """
        session_dir = self.session_dir(project_id, session_id)
        output_dirs = ("artifacts", "charts", "checkpoints", "report")
        output_files = ("trace.jsonl", "debug.jsonl", "loop.journal.jsonl")
        with self._session_write_transaction(project_id, session_id) as conn:
            for dirname in output_dirs:
                path = session_dir / dirname
                if path.exists():
                    shutil.rmtree(path)
            for filename in output_files:
                (session_dir / filename).unlink(missing_ok=True)
            conn.execute(
                "delete from artifacts where project_id = ? and session_id = ?",
                (project_id, session_id),
            )
            conn.execute(
                "delete from trace_events where project_id = ? and session_id = ?",
                (project_id, session_id),
            )
            conn.execute(
                "delete from pending_actions where project_id = ? and session_id = ?",
                (project_id, session_id),
            )
            conn.execute(
                """
                update sessions set artifact_count = 0, report_status = null,
                    updated_at = ?
                where project_id = ? and session_id = ? and storage_state = 'live'
                """,
                (datetime.now(UTC).isoformat(), project_id, session_id),
            )

    def list_artifacts_safe(
        self, *, project_id: str, session_id: str
    ) -> tuple[list[Artifact], list[str]]:
        """Read a run's artifacts straight from ``artifacts/*.json`` on disk."""
        artifacts_dir = self.session_dir(project_id, session_id) / "artifacts"
        if not artifacts_dir.is_dir():
            return [], []
        artifacts: list[Artifact] = []
        warnings: list[str] = []
        for path in sorted(artifacts_dir.glob("*.json")):
            try:
                artifacts.append(Artifact.model_validate_json(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                warnings.append(f"skipped unreadable artifact: {path.name}")
        return artifacts, warnings

    def list_sessions(self, project_id: str) -> list[SessionInfo]:
        """Enumerate a project's runs as defensive :class:`SessionInfo` summaries."""
        runs_dir = self.project_dir(project_id) / "sessions"
        if not runs_dir.is_dir():
            return []
        with closing(self._connect()) as conn:
            hidden_session_ids = {
                str(row[0])
                for row in conn.execute(
                    """
                    select session_id from sessions
                    where project_id = ? and storage_state != 'live'
                    """,
                    (project_id,),
                ).fetchall()
            }
        infos: list[SessionInfo] = []
        for run_path in sorted(
            p
            for p in runs_dir.iterdir()
            if p.is_dir() and p.name not in hidden_session_ids and p.name != AUDIT_SESSION_ID
        ):
            infos.append(self._build_session_info(project_id, run_path.name))
        infos.sort(
            key=lambda info: (
                info.created_at is not None,
                info.created_at or datetime.min,
                info.session_id,
            ),
            reverse=True,
        )
        return infos

    def _build_session_info(self, project_id: str, session_id: str) -> SessionInfo:
        info = SessionInfo(session_id=session_id)
        # SQLite status is authoritative when present; fall back to "unknown".
        with suppress(sqlite3.Error):
            status = self.get_session_status(session_id)
            if status:
                info.status = status
        # Manifest: created_at + code_version (defensive json load).
        manifest = self.session_dir(project_id, session_id) / "manifest.json"
        manifest_data: object = None
        with suppress(OSError, ValueError):
            manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
        if isinstance(manifest_data, dict):
            info.manifest_read = True
            code_version = manifest_data.get("code_version")
            info.code_version = code_version if isinstance(code_version, str) else None
            seed = manifest_data.get("seed")
            info.seed = seed if isinstance(seed, int) and not isinstance(seed, bool) else None
            source_session_id = manifest_data.get("source_session_id")
            info.source_session_id = (
                source_session_id if isinstance(source_session_id, str) else None
            )
            input_hashes = manifest_data.get("input_hashes")
            if isinstance(input_hashes, dict) and all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in input_hashes.items()
            ):
                info.input_hashes = dict(input_hashes)
            model_versions = manifest_data.get("model_versions")
            if isinstance(model_versions, dict) and all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in model_versions.items()
            ):
                info.model_versions = dict(model_versions)
            title = manifest_data.get("title")
            if isinstance(title, str) and title.strip():
                info.title = title.strip()
            created_raw = manifest_data.get("created_at")
            if isinstance(created_raw, str):
                with suppress(ValueError):
                    info.created_at = datetime.fromisoformat(created_raw)
        # Session history is loaded often by the workbench. Keep this path
        # metadata-only: counting directory entries and reading the few indexed
        # summary artifacts avoids deserializing every artifact for every run.
        artifacts_dir = self.session_dir(project_id, session_id) / "artifacts"
        info.artifact_count = (
            sum(1 for path in artifacts_dir.glob("*.json") if path.is_file())
            if artifacts_dir.is_dir()
            else 0
        )
        if isinstance(manifest_data, dict):
            hashes = manifest_data.get("input_hashes")
            if isinstance(hashes, dict):
                info.dataset_names = [str(key) for key in hashes]
        if not info.dataset_names:
            info.dataset_names = self._indexed_dataset_names(project_id, session_id)
        info.report_status = self._indexed_report_status(project_id, session_id)
        if (
            info.report_status is None
            and (self.session_dir(project_id, session_id) / "report" / "report.md").is_file()
        ):
            info.report_status = "generated"
        # Chat transcript: one line per persisted message (0 when absent).
        info.chat_message_count = self._count_chat_messages(project_id, session_id)
        return info

    def _indexed_dataset_names(self, project_id: str, session_id: str) -> list[str]:
        artifacts = self._indexed_artifacts(
            project_id,
            session_id,
            artifact_types=(ArtifactType.DATASET_PROFILE,),
        )
        return _dataset_names(artifacts)

    def _indexed_report_status(self, project_id: str, session_id: str) -> str | None:
        artifacts = self._indexed_artifacts(
            project_id,
            session_id,
            artifact_types=(ArtifactType.SESSION_SUMMARY, ArtifactType.REPORT_BUNDLE),
        )
        return _report_status(artifacts)

    def list_indexed_artifacts(
        self,
        *,
        project_id: str,
        session_id: str,
        artifact_types: tuple[ArtifactType, ...],
    ) -> list[Artifact]:
        """Type-filtered artifacts via the SQLite index (API/services read path)."""
        return self._indexed_artifacts(project_id, session_id, artifact_types=artifact_types)

    def list_artifacts_of_types(
        self,
        *,
        project_id: str,
        session_id: str,
        artifact_types: Sequence[ArtifactType],
    ) -> tuple[list[Artifact], list[str]]:
        """Load only the requested artifact types for one session.

        Prefer the SQLite index so project-wide readers (findings library,
        coverage, freshness) do not deserialize every chart/profile payload.
        Fall back to a filtered disk scan only for legacy sessions that have
        on-disk artifact JSON but no index rows yet.
        """
        types = tuple(artifact_types)
        if not types:
            return [], []
        try:
            indexed = self._indexed_artifacts(project_id, session_id, artifact_types=types)
            if self._session_has_artifact_index(project_id, session_id):
                return indexed, []
            if not self._session_has_disk_artifacts(project_id, session_id):
                # Empty modern session: index is authoritative, no disk walk.
                return [], []
        except sqlite3.Error:
            pass
        artifacts, warnings = self.list_artifacts_safe(project_id=project_id, session_id=session_id)
        wanted = frozenset(types)
        return [artifact for artifact in artifacts if artifact.type in wanted], warnings

    def _session_has_artifact_index(self, project_id: str, session_id: str) -> bool:
        try:
            with closing(self._connect()) as conn:
                row = conn.execute(
                    """
                    select 1 from artifacts
                    where project_id = ? and session_id = ?
                    limit 1
                    """,
                    (project_id, session_id),
                ).fetchone()
        except sqlite3.Error:
            return False
        return row is not None

    def _session_has_disk_artifacts(self, project_id: str, session_id: str) -> bool:
        artifacts_dir = self.session_dir(project_id, session_id) / "artifacts"
        if not artifacts_dir.is_dir():
            return False
        return any(path.suffix == ".json" and path.is_file() for path in artifacts_dir.iterdir())

    def _indexed_artifacts(
        self,
        project_id: str,
        session_id: str,
        *,
        artifact_types: tuple[ArtifactType, ...],
    ) -> list[Artifact]:
        if not artifact_types:
            return []
        placeholders = ",".join("?" for _ in artifact_types)
        try:
            with closing(self._connect()) as conn:
                rows = conn.execute(
                    f"""
                    select path from artifacts
                    where project_id = ? and session_id = ?
                      and artifact_type in ({placeholders})
                    order by rowid
                    """,
                    (project_id, session_id, *(item.value for item in artifact_types)),
                ).fetchall()
        except sqlite3.Error:
            return []
        artifacts: list[Artifact] = []
        for row in rows:
            with suppress(OSError, ValueError):
                artifacts.append(
                    Artifact.model_validate_json(self._abs(row[0]).read_text(encoding="utf-8"))
                )
        return artifacts

    def query_artifact_index_rows(
        self,
        project_id: str,
        session_id: str,
        *,
        artifact_type: str | None = None,
        limit: int,
        after_rowid: int | None = None,
    ) -> list[dict]:
        """Cursor page of artifact index rows ordered by rowid — no payload reads."""
        sql = (
            "select rowid, artifact_id, artifact_type, path from artifacts"
            " where project_id = ? and session_id = ?"
        )
        params: list[object] = [project_id, session_id]
        if artifact_type is not None:
            sql += " and artifact_type = ?"
            params.append(artifact_type)
        if after_rowid is not None:
            sql += " and rowid > ?"
            params.append(after_rowid)
        sql += " order by rowid limit ?"
        params.append(limit)
        with closing(self._connect()) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            {
                "rowid": row[0],
                "artifact_id": row[1],
                "artifact_type": row[2],
                "path": self._abs(row[3]),
            }
            for row in rows
        ]

    def latest_artifact_index_rows(
        self, project_id: str, session_id: str, artifact_type: str
    ) -> list[dict]:
        """Index rows of one type, newest (highest rowid) first — no payload reads."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                select rowid, artifact_id from artifacts
                where project_id = ? and session_id = ? and artifact_type = ?
                order by rowid desc
                """,
                (project_id, session_id, artifact_type),
            ).fetchall()
        return [{"rowid": row[0], "artifact_id": row[1]} for row in rows]

    def latest_artifact_index_row(
        self, project_id: str, session_id: str, artifact_type: str
    ) -> dict | None:
        """Newest row of one type without materializing its full history."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                select rowid, artifact_id from artifacts
                where project_id = ? and session_id = ? and artifact_type = ?
                order by rowid desc limit 1
                """,
                (project_id, session_id, artifact_type),
            ).fetchone()
        return None if row is None else {"rowid": row[0], "artifact_id": row[1]}

    def artifact_index_row(
        self,
        artifact_id: str,
        *,
        project_id: str | None = None,
        session_id: str | None = None,
    ) -> dict | None:
        """Artifact index row with its absolute on-disk path (no file read).

        Same multi-partition semantics as :meth:`get_artifact`: incomplete
        scope is allowed only when it identifies exactly one row.
        """
        sql = (
            "select artifact_id, artifact_type, project_id, session_id, path"
            " from artifacts where artifact_id = ?"
        )
        params: list[object] = [artifact_id]
        if project_id is not None:
            sql += " and project_id = ?"
            params.append(project_id)
        if session_id is not None:
            sql += " and session_id = ?"
            params.append(session_id)
        sql += " order by rowid desc limit 2"
        with closing(self._connect()) as conn:
            rows = conn.execute(sql, params).fetchall()
        if not rows:
            return None
        if len(rows) > 1:
            raise ValueError(
                f"ambiguous artifact identity: {artifact_id!r} requires project_id and session_id"
            )
        row = rows[0]
        return {
            "artifact_id": row[0],
            "artifact_type": row[1],
            "project_id": row[2],
            "session_id": row[3],
            "path": self._abs(row[4]),
        }

    def _count_chat_messages(self, project_id: str, session_id: str) -> int:
        chat_path = self.project_dir(project_id) / "chat" / f"{session_id}.jsonl"
        if not chat_path.is_file():
            return 0
        try:
            with chat_path.open("r", encoding="utf-8") as handle:
                return sum(1 for line in handle if line.strip())
        except OSError:
            return 0

    def write_manifest(self, manifest: SessionManifest) -> Path:
        path = self.session_dir(manifest.project_id, manifest.session_id) / "manifest.json"
        with self._session_write_transaction(manifest.project_id, manifest.session_id):
            path.parent.mkdir(parents=True, exist_ok=True)
            # Atomic replace: a crash mid-write must never leave a torn manifest,
            # which would degrade the run index (created_at/title read as missing).
            tmp_path = path.with_suffix(".json.tmp")
            tmp_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
            os.replace(tmp_path, path)
        self.refresh_session_index(manifest.project_id, manifest.session_id)
        return path

    def adopt_legacy_session(self, project_id: str, session_id: str) -> None:
        """Index an existing pre-index run without reviving a deleted identity."""
        session_dir = self.session_dir(project_id, session_id)
        if session_dir.is_symlink() or not session_dir.is_dir():
            raise ValueError("Legacy session adoption requires a safe on-disk session directory.")
        with session_key_lock(self.root, session_id), closing(self._connect()) as conn:
            conn.execute("begin immediate")
            try:
                row = conn.execute(
                    "select project_id, storage_state from sessions where session_id = ?",
                    (session_id,),
                ).fetchone()
                if row is not None:
                    if str(row[0]) != project_id or str(row[1]) != "live":
                        raise SessionStorageDeletingError(session_id)
                    conn.commit()
                    return
                tombstone = conn.execute(
                    """
                    select 1 from storage_operations
                    where op_kind = 'delete_session' and resource_kind = 'session'
                      and resource_key = ? and state != 'aborted'
                    limit 1
                    """,
                    (session_id,),
                ).fetchone()
                if tombstone is not None:
                    raise SessionStorageDeletingError(session_id)
                conn.execute(
                    """
                    insert into sessions(session_id, project_id, path, status, storage_state)
                    values(?, ?, ?, 'unknown', 'live')
                    """,
                    (session_id, project_id, self._rel(session_dir)),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def refresh_session_index(self, project_id: str, session_id: str) -> None:
        """Sync the runs-table summary columns so listing never rescans disk.

        Guards (review F2/F3): a run with no on-disk directory is never
        INSERTed — otherwise mark_session_status on a never-started/deleted run
        would fabricate/resurrect index rows the legacy directory listing
        never showed. And a degraded read (torn manifest, transient IO error)
        must not overwrite known-good title/created_at/status with NULL/unknown.
        """
        session_dir = self.session_dir(project_id, session_id)
        with session_key_lock(self.root, session_id):
            if not session_dir.is_dir():
                return
            info = self._build_session_info(project_id, session_id)
            now = datetime.now(UTC).isoformat()
            # Normalize to UTC so the TEXT column sorts chronologically even when a
            # manifest carries a non-UTC offset; naive values are treated as UTC.
            created = None
            if info.created_at is not None:
                aware = (
                    info.created_at
                    if info.created_at.tzinfo is not None
                    else info.created_at.replace(tzinfo=UTC)
                )
                created = aware.astimezone(UTC).isoformat()
            with closing(self._connect()) as conn:
                conn.execute("begin immediate")
                try:
                    if self._deleting_target(conn, (session_id,)) is not None:
                        conn.rollback()
                        return
                    row = conn.execute(
                        """
                        select 1 from sessions
                        where session_id = ? and project_id = ? and storage_state = 'live'
                        """,
                        (session_id, project_id),
                    ).fetchone()
                    if row is None:
                        conn.rollback()
                        return
                    self._assert_publish_allowed(conn, project_id, session_id)
                    conn.execute(
                        """
                        update sessions set
                            status=case
                                when ? = 'unknown' then status
                                else ?
                            end,
                            title=coalesce(?, title),
                            created_at=coalesce(?, created_at),
                            updated_at=?,
                            dataset_names_json=coalesce(?, dataset_names_json),
                            artifact_count=?,
                            report_status=coalesce(?, report_status),
                            chat_message_count=?,
                            -- The manifest is authoritative for these, so a
                            -- correction that REMOVES a value has to clear the
                            -- column. `coalesce` could only ever add: a run
                            -- re-pointed at a root manifest kept its old
                            -- source_session_id and stayed in the wrong family
                            -- forever. Guarded by manifest_read so a torn or
                            -- missing file still cannot wipe good data.
                            source_session_id=case when ?
                                then ? else source_session_id end,
                            code_version=case when ? then ? else code_version end,
                            seed=case when ? then ? else seed end,
                            input_hashes_json=case when ?
                                then ? else input_hashes_json end,
                            model_versions_json=case when ?
                                then ? else model_versions_json end
                        where session_id = ? and project_id = ? and storage_state = 'live'
                        """,
                        (
                            info.status,
                            info.status,
                            info.title,
                            created,
                            now,
                            json.dumps(info.dataset_names, ensure_ascii=False),
                            info.artifact_count,
                            info.report_status,
                            info.chat_message_count,
                            info.manifest_read,
                            info.source_session_id,
                            info.manifest_read,
                            info.code_version,
                            info.manifest_read,
                            info.seed,
                            info.manifest_read,
                            (
                                json.dumps(info.input_hashes, sort_keys=True)
                                if info.input_hashes is not None
                                else None
                            ),
                            info.manifest_read,
                            (
                                json.dumps(info.model_versions, sort_keys=True)
                                if info.model_versions is not None
                                else None
                            ),
                            session_id,
                            project_id,
                        ),
                    )
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise

    def project_index_rows(
        self,
        *,
        exclude_session_id_containing: str | None = None,
        exclude_session_id_prefixes: Sequence[str] = (),
    ) -> list[dict]:
        """Projects with an indexed run count, straight from SQLite.

        The exclusions must match :meth:`query_session_index_rows`, or a project
        advertises more runs than its own run list will ever show.
        """
        marker = exclude_session_id_containing or ""
        conditions = ""
        params: list[object] = [marker, marker]
        for prefix in exclude_session_id_prefixes:
            conditions += " and substr(r.session_id, 1, ?) <> ?"
            params.extend([len(prefix), prefix])
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"""
                select p.project_id, p.name,
                       (select count(*) from sessions r
                         where r.project_id = p.project_id
                           and r.storage_state = 'live'
                           and (? = '' or instr(r.session_id, ?) = 0){conditions}) as session_count
                from projects p
                order by p.sort_order asc, p.created_at asc, p.rowid asc
                """,
                params,
            ).fetchall()
        return [{"project_id": r[0], "name": r[1], "session_count": r[2]} for r in rows]

    def project_exists(self, project_id: str) -> bool:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "select 1 from projects where project_id = ?", (project_id,)
            ).fetchone()
        return row is not None

    def workspace_upload_totals(self) -> tuple[int, int]:
        """Current uploaded datasets and bytes across the whole workspace."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                "select count(*), coalesce(sum(byte_size), 0) from upload_usage"
            ).fetchone()
        if row is None:
            return 0, 0
        return int(row[0]), int(row[1])

    def delete_project(self, project_id: str) -> None:
        """Remove one project and every project-scoped row from the index.

        The caller removes the corresponding directory first.  Keeping the
        database operation here makes the project boundary explicit and means
        future project-scoped tables are removed automatically rather than
        becoming orphaned state.
        """
        with closing(self._connect()) as conn:
            conn.execute("begin immediate")
            try:
                exists = conn.execute(
                    "select 1 from projects where project_id = ?", (project_id,)
                ).fetchone()
                if exists is None:
                    raise KeyError(project_id)
                active = _active_project_job(conn, project_id)
                if active is not None:
                    raise RuntimeError(str(active[0]))

                tables = conn.execute(
                    "select name from sqlite_master where type = 'table'"
                ).fetchall()
                for (table_name,) in tables:
                    table = str(table_name)
                    if table.startswith("sqlite_") or table == "projects":
                        continue
                    columns = {str(row[1]) for row in conn.execute(f"pragma table_info({table})")}
                    if "project_id" in columns:
                        conn.execute(f' delete from "{table}" where project_id = ?', (project_id,))
                conn.execute("delete from projects where project_id = ?", (project_id,))
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def active_project_job(self, project_id: str) -> str | None:
        with closing(self._connect()) as conn:
            row = _active_project_job(conn, project_id)
        return None if row is None else str(row[0])

    _SESSION_INDEX_COLUMNS = (
        "session_id, project_id, status, title, created_at, updated_at, "
        "dataset_names_json, artifact_count, report_status, chat_message_count, "
        "source_session_id, code_version, seed, input_hashes_json, model_versions_json"
    )

    def query_session_index_rows(
        self,
        project_id: str,
        *,
        limit: int,
        before: tuple[str, str] | None = None,
        exclude_session_id_containing: str | None = None,
        exclude_session_id_prefixes: Sequence[str] = (),
        search: str | None = None,
    ) -> list[dict]:
        """Cursor page of runs ordered by (updated_at desc, session_id desc), SQL only.

        ``before`` is the previous page's last (updated_at, session_id); NULL
        updated_at coalesces to '' so unknown-age runs sort last.

        ``search`` matches title, dataset names or run id with a LIKE contains
        pattern; the term is escaped so a literal % or _ never widens it.
        """
        marker = exclude_session_id_containing or ""
        sql = (
            f"select {self._SESSION_INDEX_COLUMNS} from sessions "
            "where project_id = ? and storage_state = 'live' "
            "and (? = '' or instr(session_id, ?) = 0)"
        )
        params: list[object] = [project_id, marker, marker]
        for prefix in exclude_session_id_prefixes:
            # substr rather than LIKE: no wildcard escaping to get wrong.
            sql += " and substr(session_id, 1, ?) <> ?"
            params.extend([len(prefix), prefix])
        if search:
            pattern = f"%{_escape_like(search)}%"
            sql += (
                " and (coalesce(title, '') like ? escape '\\'"
                " or coalesce(dataset_names_json, '') like ? escape '\\'"
                " or session_id like ? escape '\\')"
            )
            params.extend([pattern, pattern, pattern])
        if before is not None:
            sql += " and (coalesce(updated_at, ''), session_id) < (?, ?)"
            params.extend(before)
        sql += " order by coalesce(updated_at, '') desc, session_id desc limit ?"
        params.append(limit)
        with closing(self._connect()) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._session_index_row_to_dict(row) for row in rows]

    def get_session_index_row(self, session_id: str) -> dict | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                f"""
                select {self._SESSION_INDEX_COLUMNS} from sessions
                where session_id = ? and storage_state = 'live'
                """,
                (session_id,),
            ).fetchone()
        return None if row is None else self._session_index_row_to_dict(row)

    def query_session_children(
        self,
        project_id: str,
        source_session_id: str,
        *,
        limit: int = MAX_SESSION_CHILDREN,
    ) -> list[dict]:
        """Direct lineage children, without the user-list visibility filters.

        The limit is in SQL rather than applied by the caller: a malformed
        lineage chain would otherwise materialise every matching row before
        anything got the chance to truncate it.
        """
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"""
                select {self._SESSION_INDEX_COLUMNS} from sessions
                where project_id = ? and source_session_id = ? and storage_state = 'live'
                order by session_id
                limit ?
                """,
                (project_id, source_session_id, limit),
            ).fetchall()
        return [self._session_index_row_to_dict(row) for row in rows]

    @staticmethod
    def _session_index_row_to_dict(row: tuple) -> dict:
        return {
            "session_id": row[0],
            "project_id": row[1],
            "status": row[2],
            "title": row[3],
            "created_at": row[4],
            "updated_at": row[5],
            "dataset_names_json": row[6],
            "artifact_count": row[7],
            "report_status": row[8],
            "chat_message_count": row[9],
            "source_session_id": row[10],
            "code_version": row[11],
            "seed": row[12],
            "input_hashes_json": row[13],
            "model_versions_json": row[14],
        }

    def artifact_type_counts(self, project_id: str, session_id: str) -> dict[str, int]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                select artifact_type, count(*) from artifacts
                where project_id = ? and session_id = ?
                group by artifact_type order by artifact_type
                """,
                (project_id, session_id),
            ).fetchall()
        return {row[0]: row[1] for row in rows}

    def read_manifest(self, project_id: str, session_id: str) -> SessionManifest | None:
        path = self.session_dir(project_id, session_id) / "manifest.json"
        if not path.exists():
            return None
        return SessionManifest.model_validate_json(path.read_text(encoding="utf-8"))

    def append_trace(self, project_id: str, event: TraceEvent) -> Path:
        correlation = current_trace_job()
        if correlation is not None:
            if event.job_id is not None and event.job_id != correlation.job_id:
                raise ValueError("Trace event job_id conflicts with the active job scope.")
            if event.job_generation is not None and event.job_generation != correlation.generation:
                raise ValueError("Trace event job_generation conflicts with the active job scope.")
            event = event.model_copy(
                update={
                    "job_id": correlation.job_id,
                    "job_generation": correlation.generation,
                }
            )
        session_dir = self.session_dir(project_id, event.session_id)
        with self._session_write_transaction(project_id, event.session_id) as conn:
            session_dir.mkdir(parents=True, exist_ok=True)
            trace_path = session_dir / "trace.jsonl"
            if event.event_key is not None:
                existing = conn.execute(
                    "select 1 from trace_events where event_key = ? limit 1",
                    (event.event_key,),
                ).fetchone()
                if existing is not None:
                    return trace_path
            line = event.model_dump_json()
            with trace_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{line}\n")
            conn.execute(
                """
                insert into trace_events(
                    session_id, project_id, event_type, name, payload,
                    job_id, job_generation, event_key
                )
                values(?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.session_id,
                    project_id,
                    event.event_type,
                    event.name,
                    line,
                    event.job_id,
                    event.job_generation,
                    event.event_key,
                ),
            )
            # jsonl + sqlite above are the source of truth; mirroring to an
            # OpenInference span exporter is an optional, off-by-default side effect
            # that no-ops (and never raises) unless EDA_OBSERVABILITY is set.
            mirror_trace_event(event)
            # Same contract for the structured debug JSONL: off by default, gated by
            # EDA_DEBUG_LOG=1, and any write failure is swallowed inside the mirror.
            mirror_event_to_debug_log(session_dir, event)
        return trace_path

    def append_trace_bounded(
        self,
        project_id: str,
        event: TraceEvent,
        *,
        max_events: int,
        since_iso: str,
        summary_source: str,
    ) -> str:
        """Atomically deduplicate, rate-check, and append a trace event.

        Returning a small outcome string keeps policy in the caller while the
        durable decisions remain in the same SQLite/run-file transaction.
        """
        if event.event_key is None:
            raise ValueError("A bounded trace event requires an event_key.")
        session_dir = self.session_dir(project_id, event.session_id)
        with self._session_write_transaction(project_id, event.session_id) as conn:
            session_dir.mkdir(parents=True, exist_ok=True)
            trace_path = session_dir / "trace.jsonl"
            existing = conn.execute(
                "select 1 from trace_events where event_key = ? limit 1",
                (event.event_key,),
            ).fetchone()
            if existing is not None:
                return "duplicate"
            row = conn.execute(
                """
                select count(*) from trace_events
                where project_id = ? and session_id = ? and event_type = ?
                  and julianday(coalesce(
                    json_extract(payload, '$.finished_at'),
                    json_extract(payload, '$.started_at')
                  )) >= julianday(?)
                  and json_extract(payload, '$.summary.source') = ?
                """,
                (
                    project_id,
                    event.session_id,
                    event.event_type,
                    since_iso,
                    summary_source,
                ),
            ).fetchone()
            if row is not None and int(row[0]) >= max_events:
                return "rate_limited"
            line = event.model_dump_json()
            with trace_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{line}\n")
            conn.execute(
                """
                insert into trace_events(
                    session_id, project_id, event_type, name, payload,
                    job_id, job_generation, event_key
                )
                values(?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.session_id,
                    project_id,
                    event.event_type,
                    event.name,
                    line,
                    event.job_id,
                    event.job_generation,
                    event.event_key,
                ),
            )
            mirror_trace_event(event)
            mirror_event_to_debug_log(session_dir, event)
        return "inserted"

    def count_recent_trace_events(
        self,
        *,
        project_id: str,
        session_id: str,
        event_type: str,
        since_iso: str,
        summary_source: str | None = None,
    ) -> int:
        """Count a typed event in a bounded time window without loading payloads."""
        source_clause = (
            " and json_extract(payload, '$.summary.source') = ?"
            if summary_source is not None
            else ""
        )
        params: list[object] = [project_id, session_id, event_type, since_iso]
        if summary_source is not None:
            params.append(summary_source)
        with closing(self._connect()) as conn:
            row = conn.execute(
                f"""
                select count(*) from trace_events
                where project_id = ? and session_id = ? and event_type = ?
                  and julianday(coalesce(
                    json_extract(payload, '$.finished_at'),
                    json_extract(payload, '$.started_at')
                  )) >= julianday(?)
                  {source_clause}
                """,  # noqa: S608 - source_clause is a fixed internal fragment
                params,
            ).fetchone()
        return int(row[0]) if row is not None else 0

    def trace_event_key_exists(self, event_key: str) -> bool:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "select 1 from trace_events where event_key = ? limit 1",
                (event_key,),
            ).fetchone()
        return row is not None

    def list_trace_events(
        self,
        *,
        project_id: str,
        session_id: str,
        event_types: Sequence[str] | None = None,
    ) -> list[TraceEvent]:
        """Whole-run events, or only the given types when the caller needs a slice."""
        sql = "select payload from trace_events where project_id = ? and session_id = ?"
        params: list[object] = [project_id, session_id]
        if event_types is not None:
            sql += f" and event_type in ({','.join('?' for _ in event_types)})"
            params.extend(event_types)
        sql += " order by id"
        with closing(self._connect()) as conn:
            rows = conn.execute(sql, params).fetchall()  # noqa: S608 - bound placeholders only
        return [TraceEvent.model_validate_json(row[0]) for row in rows]

    def earliest_trace_started_at(self, *, project_id: str, session_id: str) -> str | None:
        """Run start as recorded in the trace, without loading any payload."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                select min(json_extract(payload, '$.started_at')) from trace_events
                where project_id = ? and session_id = ?
                """,
                (project_id, session_id),
            ).fetchone()
        return None if row is None or row[0] is None else str(row[0])

    def list_trace_rows_after(
        self, *, project_id: str, session_id: str, after_id: int, limit: int = 500
    ) -> list[tuple[int, str]]:
        """Run-wide trace rows past ``after_id`` for forensic callers."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                select id, payload from trace_events
                where project_id = ? and session_id = ? and id > ?
                order by id limit ?
                """,
                (project_id, session_id, after_id, limit),
            ).fetchall()
        return [(row[0], row[1]) for row in rows]

    def list_job_trace_rows_after(
        self, *, job_id: str, after_id: int, limit: int = 500
    ) -> list[tuple[int, str]]:
        """Exact job-correlated SSE rows; unowned legacy rows fail closed."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                select id, payload from trace_events
                where job_id = ? and id > ?
                order by id limit ?
                """,
                (job_id, after_id, limit),
            ).fetchall()
        return [(int(row[0]), str(row[1])) for row in rows]

    def persist_step_failure_fallback(
        self,
        project_id: str,
        event: TraceEvent,
    ) -> None:
        """Minimal DB-only failure record when normal reporting is broken.

        This deliberately bypasses ``mark_session_status`` and ``append_trace``:
        checkpoint/filesystem failures must not leave a run active merely
        because the ordinary JSONL mirror or a trace callback also failed.
        """
        if event.event_type != "step_failed" or not event.event_key:
            raise ValueError("A keyed step_failed event is required.")
        payload = event.model_dump_json()
        with closing(self._connect()) as conn:
            conn.execute("begin immediate")
            try:
                cursor = conn.execute(
                    """
                    update sessions set status = ?
                    where session_id = ? and project_id = ? and storage_state = 'live'
                    """,
                    ("failed", event.session_id, project_id),
                )
                if cursor.rowcount != 1:
                    raise KeyError(f"Session not found: {event.session_id}")
                conn.execute(
                    """
                    insert into trace_events(
                        session_id, project_id, event_type, name, payload,
                        job_id, job_generation, event_key
                    )
                    values(?, ?, ?, ?, ?, ?, ?, ?)
                    on conflict(event_key) where event_key is not null do nothing
                    """,
                    (
                        event.session_id,
                        project_id,
                        event.event_type,
                        event.name,
                        payload,
                        event.job_id,
                        event.job_generation,
                        event.event_key,
                    ),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def query_trace_rows(
        self,
        *,
        project_id: str,
        session_id: str,
        event_type: str | None = None,
        after_id: int | None = None,
        limit: int,
    ) -> list[tuple[int, str]]:
        """(id, payload) trace rows for the read API: cursor page, optional type filter."""
        sql = "select id, payload from trace_events where project_id = ? and session_id = ?"
        params: list[object] = [project_id, session_id]
        if event_type is not None:
            sql += " and event_type = ?"
            params.append(event_type)
        if after_id is not None:
            sql += " and id > ?"
            params.append(after_id)
        sql += " order by id limit ?"
        params.append(limit)
        with closing(self._connect()) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [(row[0], row[1]) for row in rows]

    def count_session_artifacts(self, *, project_id: str, session_id: str) -> int:
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                select count(*) from artifacts
                where project_id = ? and session_id = ?
                """,
                (project_id, session_id),
            ).fetchone()
        return int(row[0]) if row is not None else 0

    def query_artifact_debug_rows(
        self,
        *,
        project_id: str,
        session_id: str,
        after_rowid: int,
        limit: int,
    ) -> list[tuple[int, str, str, Path]]:
        """Bounded artifact-index page for the developer inspector."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                select rowid, artifact_id, artifact_type, path
                from artifacts
                where project_id = ? and session_id = ? and rowid > ?
                order by rowid limit ?
                """,
                (project_id, session_id, after_rowid, limit),
            ).fetchall()
        return [(int(row[0]), str(row[1]), str(row[2]), self._abs(str(row[3]))) for row in rows]

    def trace_event_type_counts(self, *, project_id: str, session_id: str) -> dict[str, int]:
        """Event-type histogram for a run — powers the trace filter without payload reads."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                select event_type, count(*) from trace_events
                where project_id = ? and session_id = ?
                group by event_type order by event_type
                """,
                (project_id, session_id),
            ).fetchall()
        return {row[0]: row[1] for row in rows}

    def project_artifact_index_rows(self, project_id: str, artifact_type: str) -> list[dict]:
        """Index rows of one type across every run of a project — no payload reads."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                select artifact_id, session_id, path from artifacts
                where project_id = ? and artifact_type = ?
                order by rowid
                """,
                (project_id, artifact_type),
            ).fetchall()
        return [
            {"artifact_id": row[0], "session_id": row[1], "path": self._abs(row[2])} for row in rows
        ]

    _JOB_COLUMNS = (
        "job_id, session_id, project_id, kind, status, cancel_requested, created_at, "
        "started_at, finished_at, error_code, error_message, idempotency_key, pid, lane_key, "
        "request_digest, request_scope, state_version, launch_attempt, launch_token, "
        "lease_owner, lease_expires_at, heartbeat_at, pid_start_identity, "
        "cancel_requested_at, cancel_deadline_at, kill_fence_state, "
        "critical_depth, critical_owner_generation"
    )

    def create_job(
        self,
        *,
        job_id: str,
        session_id: str,
        project_id: str,
        kind: str,
        idempotency_key: str | None = None,
        lane_key: str | None = None,
        request_digest: str | None = None,
        request_scope: str | None = None,
    ) -> dict:
        effective_lane = lane_key or session_id
        with ExitStack() as stack:
            for target in sorted({session_id, effective_lane}):
                stack.enter_context(session_key_lock(self.root, target))
            conn = stack.enter_context(closing(self._connect()))
            conn.execute("begin immediate")
            try:
                deleting_target = self._deleting_target(conn, (session_id, effective_lane))
                if deleting_target is not None:
                    raise SessionStorageDeletingError(deleting_target)
                targets = (
                    (session_id,) if effective_lane == session_id else (session_id, effective_lane)
                )
                for target in targets:
                    row = conn.execute(
                        "select storage_state from sessions where session_id = ?",
                        (target,),
                    ).fetchone()
                    deleted = conn.execute(
                        """
                        select 1 from storage_operations
                        where op_kind = 'delete_session' and resource_kind = 'session'
                          and resource_key = ? and state != 'aborted'
                        limit 1
                        """,
                        (target,),
                    ).fetchone()
                    if deleted is not None or (
                        target == effective_lane
                        and effective_lane != session_id
                        and row is not None
                        and str(row[0]) != "live"
                    ):
                        raise SessionStorageDeletingError(target)
                conn.execute(
                    """
                    insert into sessions(session_id, project_id, path, status)
                    values(?, ?, ?, 'running')
                    on conflict(session_id) do nothing
                    """,
                    (
                        session_id,
                        project_id,
                        self._rel(self.session_dir(project_id, session_id)),
                    ),
                )
                conn.execute(
                    """
                    insert into jobs(job_id, session_id, project_id, kind, status,
                                     created_at, idempotency_key, lane_key, request_digest,
                                     request_scope)
                    values(?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        session_id,
                        project_id,
                        kind,
                        datetime.now(UTC).isoformat(),
                        idempotency_key,
                        effective_lane,
                        request_digest,
                        request_scope,
                    ),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        job = self.get_job(job_id)
        assert job is not None
        return job

    def list_active_session_deletion_op_ids(self) -> list[str]:
        """Active delete operations in deterministic recovery order."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                select op_id from storage_operations
                where op_kind = 'delete_session' and resource_kind = 'session'
                  and state in ('prepared', 'fs_applied', 'db_committed', 'blocked')
                order by created_at, op_id
                """
            ).fetchall()
        return [str(row[0]) for row in rows]

    def is_session_deleting(self, session_id: str) -> bool:
        with closing(self._connect()) as conn:
            return self._deleting_target(conn, (session_id,)) is not None

    @contextmanager
    def _session_write_transaction(
        self,
        project_id: str,
        session_id: str,
        *,
        create_if_missing: bool = False,
    ) -> Iterator[sqlite3.Connection]:
        """Lock one run, then verify its live DB ownership in one write txn."""
        with session_key_lock(self.root, session_id), closing(self._connect()) as conn:
            conn.execute("begin immediate")
            try:
                if self._deleting_target(conn, (session_id,)) is not None:
                    raise SessionStorageDeletingError(session_id)
                row = conn.execute(
                    """
                    select project_id, storage_state from sessions where session_id = ?
                    """,
                    (session_id,),
                ).fetchone()
                if row is None and create_if_missing:
                    deleted = conn.execute(
                        """
                        select 1 from storage_operations
                        where op_kind = 'delete_session' and resource_kind = 'session'
                          and resource_key = ? and state != 'aborted'
                        limit 1
                        """,
                        (session_id,),
                    ).fetchone()
                    if deleted is not None:
                        raise SessionStorageDeletingError(session_id)
                    conn.execute(
                        """
                        insert into sessions(session_id, project_id, path, status)
                        values(?, ?, ?, 'running')
                        """,
                        (
                            session_id,
                            project_id,
                            self._rel(self.session_dir(project_id, session_id)),
                        ),
                    )
                    row = (project_id, "live")
                if row is None or str(row[0]) != project_id or str(row[1]) != "live":
                    raise SessionStorageDeletingError(session_id)
                self._assert_publish_allowed(conn, project_id, session_id)
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    @staticmethod
    def _assert_publish_allowed(conn: sqlite3.Connection, project_id: str, session_id: str) -> None:
        row = conn.execute(
            """
            select jobs.job_id
            from sessions join jobs on jobs.job_id = sessions.active_job_id
            where sessions.session_id = ? and sessions.project_id = ?
              and sessions.storage_state = 'live'
              and jobs.kill_fence_state = 'committed'
            """,
            (session_id, project_id),
        ).fetchone()
        if row is not None:
            raise SessionPublishFencedError(session_id, str(row[0]))

    @staticmethod
    def _deleting_target(conn: sqlite3.Connection, targets: Sequence[str]) -> str | None:
        unique_targets = tuple(dict.fromkeys(targets))
        placeholders = ",".join("?" for _ in unique_targets)
        row = conn.execute(
            f"""
            select target from (
                select session_id as target, 0 as priority
                from sessions
                where session_id in ({placeholders}) and storage_state = 'deleting'
                union all
                select resource_key as target, 1 as priority
                from storage_operations
                where op_kind = 'delete_session' and resource_kind = 'session'
                  and resource_key in ({placeholders})
                  and state in ('prepared', 'fs_applied', 'db_committed', 'blocked')
            )
            order by priority, target limit 1
            """,
            (*unique_targets, *unique_targets),
        ).fetchone()
        return None if row is None else str(row[0])

    def get_job(self, job_id: str) -> dict | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                f"select {self._JOB_COLUMNS} from jobs where job_id = ?",
                (job_id,),
            ).fetchone()
        return None if row is None else self._job_row_to_dict(row)

    def find_active_job_for_session(self, session_id: str) -> dict | None:
        """First non-terminal job executing on this concrete run."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                f"""
                select {self._JOB_COLUMNS} from jobs
                where session_id = ? and status not in ('completed', 'failed', 'cancelled')
                order by created_at limit 1
                """,
                (session_id,),
            ).fetchone()
        return None if row is None else self._job_row_to_dict(row)

    def find_active_job_for_lane(self, lane_key: str) -> dict | None:
        """First non-terminal job owning a logical source/plan lane."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                f"""
                select {self._JOB_COLUMNS} from jobs
                where lane_key = ? and status not in ('completed', 'failed', 'cancelled')
                order by created_at limit 1
                """,
                (lane_key,),
            ).fetchone()
        return None if row is None else self._job_row_to_dict(row)

    def latest_job_for_lane(self, lane_key: str) -> dict | None:
        """Newest owner of a logical lane, including terminal jobs."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                f"""
                select {self._JOB_COLUMNS} from jobs
                where lane_key = ?
                order by created_at desc limit 1
                """,
                (lane_key,),
            ).fetchone()
        return None if row is None else self._job_row_to_dict(row)

    def latest_job_for_session(self, session_id: str) -> dict | None:
        """Newest lifecycle attempt attached to one concrete derived run."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                f"""
                select {self._JOB_COLUMNS} from jobs
                where session_id = ?
                order by created_at desc limit 1
                """,
                (session_id,),
            ).fetchone()
        return None if row is None else self._job_row_to_dict(row)

    def list_active_jobs(self) -> list[dict]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"""
                select {self._JOB_COLUMNS} from jobs
                where status not in ('completed', 'failed', 'cancelled')
                order by created_at
                """
            ).fetchall()
        return [self._job_row_to_dict(row) for row in rows]

    def set_job_pid(self, job_id: str, pid: int) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute("update jobs set pid = ? where job_id = ?", (pid, job_id))

    def find_by_idempotency_key(self, idempotency_key: str) -> dict | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                f"select {self._JOB_COLUMNS} from jobs where idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
        return None if row is None else self._job_row_to_dict(row)

    def mark_job_status(
        self,
        job_id: str,
        status: str,
        *,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        started_at = now if status == "running" else None
        finished_at = now if status in ("completed", "failed", "cancelled") else None
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                update jobs set
                    state_version = case
                        when status = ? then state_version
                        else state_version + 1
                    end,
                    status = ?,
                    started_at = coalesce(?, started_at),
                    finished_at = coalesce(?, finished_at),
                    error_code = coalesce(?, error_code),
                    error_message = coalesce(?, error_message)
                where job_id = ?
                """,
                (
                    status,
                    status,
                    started_at,
                    finished_at,
                    error_code,
                    error_message,
                    job_id,
                ),
            )

    def clear_job_idempotency_key(self, job_id: str) -> None:
        """Release a failed job's key so a same-key retry runs fresh instead of
        replaying a row nothing will ever pick up (questions review D)."""
        with closing(self._connect()) as conn, conn:
            conn.execute("update jobs set idempotency_key = null where job_id = ?", (job_id,))

    def request_cancel(self, job_id: str) -> bool:
        """Set the cooperative-cancel flag; returns whether a row was updated.

        Guarded against the cancel/completion race: a job that reached a
        terminal state must never gain the flag afterwards (review codex-D #3).
        """
        with closing(self._connect()) as conn, conn:
            cursor = conn.execute(
                """
                update jobs set cancel_requested = 1
                where job_id = ?
                  and status not in ('completed', 'failed', 'cancelled')
                """,
                (job_id,),
            )
        return cursor.rowcount > 0

    _PENDING_ACTION_COLUMNS = (
        "action_hash, session_id, project_id, kind, payload_json, created_at, expires_at, "
        "status, generation, payload_digest, consumed_idempotency_key"
    )

    def create_pending_action(
        self,
        *,
        action_hash: str,
        session_id: str,
        project_id: str,
        kind: str,
        payload_json: str,
        created_at: str,
        expires_at: str,
        generation: str,
        payload_digest: str,
    ) -> None:
        """Insert or re-arm a pending action: a fresh preview always restarts
        the lifecycle (with a fresh one-time generation token), so a consumed/
        expired hash can be approved again only after the user has seen a new
        preview — and any token from an older preview goes stale (C1)."""
        with self._session_write_transaction(
            project_id,
            session_id,
            create_if_missing=True,
        ) as conn:
            conn.execute(
                """
                insert into pending_actions(action_hash, session_id, project_id, kind,
                                            payload_json, created_at, expires_at, status,
                                            generation, payload_digest,
                                            consumed_idempotency_key)
                values(?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, null)
                on conflict(action_hash, session_id) do update set
                    project_id = excluded.project_id,
                    kind = excluded.kind,
                    payload_json = excluded.payload_json,
                    created_at = excluded.created_at,
                    expires_at = excluded.expires_at,
                    status = 'pending',
                    generation = excluded.generation,
                    payload_digest = excluded.payload_digest,
                    consumed_idempotency_key = null
                """,
                (
                    action_hash,
                    session_id,
                    project_id,
                    kind,
                    payload_json,
                    created_at,
                    expires_at,
                    generation,
                    payload_digest,
                ),
            )

    def append_chat_line(self, project_id: str, session_id: str, line: str) -> Path:
        """Append one already-serialized chat line under the shared run fence."""
        with self._session_write_transaction(project_id, session_id):
            path = self.project_dir(project_id) / "chat" / f"{session_id}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(f"{line}\n")
            return path

    def write_session_text(
        self,
        project_id: str,
        session_id: str,
        relative_path: str,
        content: str,
    ) -> Path:
        """Atomically replace a text file contained by one live run."""
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
            raise ValueError("Session output path must be a contained relative path.")
        with self._session_write_transaction(project_id, session_id):
            session_dir = self.session_dir(project_id, session_id)
            path = session_dir / relative
            if not path.resolve(strict=False).is_relative_to(session_dir.resolve()):
                raise ValueError("Session output path must stay inside the session directory.")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.tmp")
            temporary.write_text(content, encoding="utf-8")
            os.replace(temporary, path)
            return path

    def append_session_line(
        self,
        project_id: str,
        session_id: str,
        relative_path: str,
        line: str,
    ) -> Path:
        """Append one line to a contained live-run file under the shared fence."""
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
            raise ValueError("Session output path must be a contained relative path.")
        with self._session_write_transaction(project_id, session_id):
            session_dir = self.session_dir(project_id, session_id)
            path = session_dir / relative
            if not path.resolve(strict=False).is_relative_to(session_dir.resolve()):
                raise ValueError("Session output path must stay inside the session directory.")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(f"{line}\n")
            return path

    @contextmanager
    def session_write_guard(self, project_id: str, session_id: str) -> Iterator[None]:
        """Public guard for specialized run writers that own their file format."""
        with self._session_write_transaction(project_id, session_id):
            yield

    def get_pending_action(self, action_hash: str, *, session_id: str) -> dict | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                f"select {self._PENDING_ACTION_COLUMNS} from pending_actions "
                "where action_hash = ? and session_id = ?",
                (action_hash, session_id),
            ).fetchone()
        return None if row is None else self._pending_action_row_to_dict(row)

    def list_pending_actions(self, *, session_id: str, kind: str) -> list[dict]:
        """Still-pending rows for one run and kind, oldest first."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"select {self._PENDING_ACTION_COLUMNS} from pending_actions "
                "where session_id = ? and kind = ? and status = 'pending' "
                "order by created_at",
                (session_id, kind),
            ).fetchall()
        return [self._pending_action_row_to_dict(row) for row in rows]

    @staticmethod
    def _pending_action_row_to_dict(row: tuple) -> dict:
        return {
            "action_hash": row[0],
            "session_id": row[1],
            "project_id": row[2],
            "kind": row[3],
            "payload_json": row[4],
            "created_at": row[5],
            "expires_at": row[6],
            "status": row[7],
            "generation": row[8],
            "payload_digest": row[9],
            "consumed_idempotency_key": row[10],
        }

    def consume_pending_action(
        self,
        action_hash: str,
        *,
        session_id: str,
        generation: str,
        now: str,
        idempotency_key: str | None = None,
    ) -> bool:
        """Atomic pending→consumed flip; rowcount 0 means the hash was already
        consumed/expired or the generation token belongs to a superseded
        preview — the caller must not execute (replay guard, C1)."""
        with closing(self._connect()) as conn, conn:
            cursor = conn.execute(
                """
                update pending_actions
                set status = 'consumed', consumed_idempotency_key = ?
                where action_hash = ? and session_id = ? and generation = ?
                  and status = 'pending' and expires_at > ?
                """,
                (idempotency_key, action_hash, session_id, generation, now),
            )
        return cursor.rowcount > 0

    def restore_pending_action(self, action_hash: str, *, session_id: str) -> bool:
        """Compensation for a failed post-consume step (C6): flip the consumed
        row back to pending with its generation untouched, so the same token
        can retry."""
        with closing(self._connect()) as conn, conn:
            cursor = conn.execute(
                """
                update pending_actions
                set status = 'pending', consumed_idempotency_key = null
                where action_hash = ? and session_id = ? and status = 'consumed'
                """,
                (action_hash, session_id),
            )
        return cursor.rowcount > 0

    def expire_pending_action(self, action_hash: str, *, session_id: str, now: str) -> bool:
        with closing(self._connect()) as conn, conn:
            cursor = conn.execute(
                """
                update pending_actions set status = 'expired'
                where action_hash = ? and session_id = ? and status = 'pending' and expires_at <= ?
                """,
                (action_hash, session_id, now),
            )
        return cursor.rowcount > 0

    @staticmethod
    def _job_row_to_dict(row: tuple) -> dict:
        return {
            "job_id": row[0],
            "session_id": row[1],
            "project_id": row[2],
            "kind": row[3],
            "status": row[4],
            "cancel_requested": bool(row[5]),
            "created_at": row[6],
            "started_at": row[7],
            "finished_at": row[8],
            "error_code": row[9],
            "error_message": row[10],
            "idempotency_key": row[11],
            "pid": row[12],
            "lane_key": row[13],
            "request_digest": row[14],
            "request_scope": row[15],
            "state_version": row[16],
            "launch_attempt": row[17],
            "launch_token": row[18],
            "lease_owner": row[19],
            "lease_expires_at": row[20],
            "heartbeat_at": row[21],
            "pid_start_identity": row[22],
            "cancel_requested_at": row[23],
            "cancel_deadline_at": row[24],
            "kill_fence_state": row[25],
            "critical_depth": row[26],
            "critical_owner_generation": row[27],
        }

    def project_dir(self, project_id: str) -> Path:
        _validate_project_id_segment(project_id)
        projects_root = _safe_projects_root(self.root)
        candidate = projects_root / project_id
        if not _is_inside(candidate, projects_root):
            raise ValueError("Project id must be one safe path segment.")
        return candidate

    def reconcile_upload_quota(self, *, reservation_ttl_seconds: int = 24 * 3600) -> None:
        """Rebuild durable completed usage from canonical files at startup.

        A restart has no live requests, so stale reservations can be discarded.
        The filesystem remains authoritative for legacy uploads created before
        the quota schema existed.
        """
        # The scan deliberately runs inside `begin immediate`: this rebuild is a
        # delete-then-reinsert, so a completion that lands mid-scan would be
        # erased by stale rows. See test_reconcile_cannot_erase_a_concurrent_completion.
        with closing(self._connect()) as conn:
            conn.execute("begin immediate")
            try:
                projects = [
                    str(row[0])
                    for row in conn.execute("select project_id from projects").fetchall()
                ]
                rows: list[tuple[str, str, int]] = []
                for project_id in projects:
                    uploads = self.project_dir(project_id) / "uploads"
                    if uploads.is_symlink() or not uploads.is_dir():
                        continue
                    for dataset_dir in uploads.iterdir():
                        version_dir = dataset_dir / "v1"
                        if (
                            dataset_dir.is_symlink()
                            or version_dir.is_symlink()
                            or not version_dir.is_dir()
                        ):
                            continue
                        byte_size = 0
                        for path in version_dir.iterdir():
                            if path.is_symlink() or not path.is_file():
                                continue
                            with suppress(OSError):
                                byte_size += path.stat().st_size
                        if byte_size:
                            rows.append((project_id, dataset_dir.name, byte_size))
                # Preserve recent reservations: another worker may already be
                # streaming. Only crash leftovers older than the staging TTL are
                # safe to reclaim during a rolling/multi-worker startup.
                conn.execute(
                    "delete from upload_reservations where created_at <= ?",
                    (time.time() - reservation_ttl_seconds,),
                )
                conn.execute("delete from upload_usage")
                conn.executemany(
                    """
                    insert into upload_usage(project_id, dataset_id, byte_size)
                    values(?, ?, ?)
                    """,
                    rows,
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def reserve_upload(
        self,
        upload_id: str,
        project_id: str,
        *,
        file_quota: int,
        concurrent_quota: int,
        now: float,
    ) -> str | None:
        """Atomically reserve one file/concurrency slot; return rejection kind."""
        with closing(self._connect()) as conn:
            conn.execute("begin immediate")
            try:
                files = int(
                    conn.execute(
                        "select count(*) from upload_usage where project_id = ?",
                        (project_id,),
                    ).fetchone()[0]
                )
                active = int(
                    conn.execute(
                        "select count(*) from upload_reservations where project_id = ?",
                        (project_id,),
                    ).fetchone()[0]
                )
                if active >= concurrent_quota:
                    conn.rollback()
                    return "concurrent"
                if files + active >= file_quota:
                    conn.rollback()
                    return "files"
                conn.execute(
                    """
                    insert into upload_reservations(
                        upload_id, project_id, reserved_bytes, created_at, updated_at
                    ) values(?, ?, 0, ?, ?)
                    """,
                    (upload_id, project_id, now, now),
                )
                conn.commit()
                return None
            except Exception:
                conn.rollback()
                raise

    def update_upload_reservation(
        self,
        upload_id: str,
        *,
        byte_size: int,
        byte_quota: int,
        now: float,
    ) -> bool:
        """Atomically grow a reservation while checking project aggregate bytes."""
        with closing(self._connect()) as conn:
            conn.execute("begin immediate")
            try:
                row = conn.execute(
                    "select project_id from upload_reservations where upload_id = ?",
                    (upload_id,),
                ).fetchone()
                if row is None:
                    conn.rollback()
                    raise RuntimeError(f"Upload reservation disappeared: {upload_id}")
                project_id = str(row[0])
                used = int(
                    conn.execute(
                        "select coalesce(sum(byte_size), 0) from upload_usage where project_id = ?",
                        (project_id,),
                    ).fetchone()[0]
                )
                other = int(
                    conn.execute(
                        """
                        select coalesce(sum(reserved_bytes), 0)
                        from upload_reservations
                        where project_id = ? and upload_id != ?
                        """,
                        (project_id, upload_id),
                    ).fetchone()[0]
                )
                if used + other + byte_size > byte_quota:
                    conn.rollback()
                    return False
                conn.execute(
                    """
                    update upload_reservations
                    set reserved_bytes = ?, updated_at = ?
                    where upload_id = ?
                    """,
                    (byte_size, now, upload_id),
                )
                conn.commit()
                return True
            except Exception:
                conn.rollback()
                raise

    def complete_upload_reservation(self, upload_id: str, dataset_id: str, byte_size: int) -> None:
        with closing(self._connect()) as conn:
            conn.execute("begin immediate")
            try:
                row = conn.execute(
                    "select project_id from upload_reservations where upload_id = ?",
                    (upload_id,),
                ).fetchone()
                if row is None:
                    raise RuntimeError(f"Upload reservation disappeared: {upload_id}")
                conn.execute(
                    """
                    insert into upload_usage(project_id, dataset_id, byte_size)
                    values(?, ?, ?)
                    on conflict(project_id, dataset_id)
                    do update set byte_size = excluded.byte_size
                    """,
                    (str(row[0]), dataset_id, byte_size),
                )
                conn.execute("delete from upload_reservations where upload_id = ?", (upload_id,))
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def release_upload_reservation(self, upload_id: str) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute("delete from upload_reservations where upload_id = ?", (upload_id,))

    def forget_upload_usage(self, project_id: str, dataset_id: str) -> bool:
        """Drop one upload's row from the durable quota ledger.

        Returns whether a row existed. The caller removes the files first, so a
        crash between the two leaves an orphan directory that
        :meth:`reconcile_upload_quota` re-counts at startup — quota over-counts,
        which is the safe direction.
        """
        with closing(self._connect()) as conn, conn:
            cursor = conn.execute(
                "delete from upload_usage where project_id = ? and dataset_id = ?",
                (project_id, dataset_id),
            )
        return cursor.rowcount > 0

    def check_upload_rate_limit(
        self,
        client_id: str,
        *,
        now: float,
        window_seconds: int,
        limit: int,
    ) -> tuple[bool, int]:
        """Persist a fixed-window-equivalent sliding log for remote uploads."""
        cutoff = now - window_seconds
        with closing(self._connect()) as conn:
            conn.execute("begin immediate")
            try:
                conn.execute("delete from upload_rate_events where occurred_at <= ?", (cutoff,))
                events = [
                    float(row[0])
                    for row in conn.execute(
                        """
                        select occurred_at from upload_rate_events
                        where client_id = ? order by occurred_at
                        """,
                        (client_id,),
                    ).fetchall()
                ]
                if len(events) >= limit:
                    retry_after = max(1, int(events[0] + window_seconds - now) + 1)
                    conn.commit()
                    return False, retry_after
                conn.execute(
                    "insert into upload_rate_events(client_id, occurred_at) values(?, ?)",
                    (client_id, now),
                )
                conn.commit()
                return True, 0
            except Exception:
                conn.rollback()
                raise

    def session_dir(self, project_id: str, session_id: str) -> Path:
        return session_dir_path(self.root, project_id, session_id)

    def artifact_path(self, project_id: str, session_id: str, artifact_id: str) -> Path:
        return self.session_dir(project_id, session_id) / "artifacts" / f"{artifact_id}.json"

    def _rel(self, path: Path) -> str:
        """Store paths relative to the workspace root so it stays relocatable."""
        try:
            return str(path.resolve().relative_to(self.root.resolve()))
        except ValueError:
            return str(path)

    def _abs(self, stored: str) -> Path:
        candidate = Path(stored)
        return candidate if candidate.is_absolute() else self.root / candidate

    def _connect(self) -> sqlite3.Connection:
        # WAL (persistent, set once in _init_db) + a 5s busy timeout so API
        # readers and the run writer can share the DB without "database is
        # locked" errors surfacing on first contention.
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.execute("pragma busy_timeout = 5000")
        conn.execute("pragma foreign_keys = on")
        return conn

    def _init_db(self) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute("pragma journal_mode = wal")
            conn.executescript(
                """
                create table if not exists projects (
                    project_id text primary key,
                    name text not null,
                    path text not null,
                    created_at text,
                    sort_order integer
                );

                create table if not exists sessions (
                    session_id text primary key,
                    project_id text not null,
                    path text not null,
                    status text not null default 'running'
                );

                create table if not exists artifacts (
                    artifact_id text not null,
                    artifact_type text not null,
                    project_id text not null,
                    session_id text not null,
                    path text not null,
                    primary key (artifact_id, project_id, session_id)
                );

                create table if not exists trace_events (
                    id integer primary key autoincrement,
                    session_id text not null,
                    project_id text not null,
                    event_type text not null,
                    name text not null,
                    payload text not null
                );

                create table if not exists upload_usage (
                    project_id text not null,
                    dataset_id text not null,
                    byte_size integer not null check(byte_size >= 0),
                    primary key (project_id, dataset_id)
                );

                create table if not exists upload_reservations (
                    upload_id text primary key,
                    project_id text not null,
                    reserved_bytes integer not null default 0 check(reserved_bytes >= 0),
                    created_at real not null,
                    updated_at real not null
                );

                create table if not exists upload_rate_events (
                    id integer primary key autoincrement,
                    client_id text not null,
                    occurred_at real not null
                );

                create table if not exists jobs (
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
                    lane_key text not null,
                    request_digest text,
                    request_scope text
                );

                create index if not exists idx_runs_project
                    on sessions(project_id);
                create index if not exists idx_jobs_run
                    on jobs(session_id);
                create index if not exists idx_jobs_status
                    on jobs(status);
                create unique index if not exists idx_jobs_idempotency
                    on jobs(idempotency_key) where idempotency_key is not null;
                create index if not exists idx_artifacts_run_type
                    on artifacts(project_id, session_id, artifact_type);
                create index if not exists idx_artifacts_run_order
                    on artifacts(project_id, session_id);
                create index if not exists idx_trace_events_run
                    on trace_events(project_id, session_id, id);
                create index if not exists idx_upload_usage_project
                    on upload_usage(project_id);
                create index if not exists idx_upload_reservations_project
                    on upload_reservations(project_id);
                create index if not exists idx_upload_rate_client_time
                    on upload_rate_events(client_id, occurred_at);
                """
            )
            # Key-space rule: artifact_id is content-derived, so the same
            # dataset indexed in two projects/runs yields the same id. The old
            # single-column PK let save_artifact's upsert steal an index row
            # across partitions (project A's dataset list went empty after
            # project B saved the same id). Rebuild with the composite PK,
            # keeping the existing rows — unlike pending_actions these are
            # durable index data, and backfill later restores the stolen rows.
            legacy_artifact_pk_cols = sum(
                1 for row in conn.execute("pragma table_info(artifacts)") if row[5] > 0
            )
            if legacy_artifact_pk_cols == 1:
                conn.execute("alter table artifacts rename to artifacts_legacy")
                conn.execute(
                    """
                    create table artifacts (
                        artifact_id text not null,
                        artifact_type text not null,
                        project_id text not null,
                        session_id text not null,
                        path text not null,
                        primary key (artifact_id, project_id, session_id)
                    )
                    """
                )
                conn.execute(
                    """
                    insert into artifacts(artifact_id, artifact_type, project_id, session_id, path)
                    select artifact_id, artifact_type, project_id, session_id, path
                    from artifacts_legacy order by rowid
                    """
                )
                # Dropping the legacy table drops its (renamed-along) index;
                # recreate it against the rebuilt table.
                conn.execute("drop table artifacts_legacy")
                conn.execute(
                    "create index if not exists idx_artifacts_run_type"
                    " on artifacts(project_id, session_id, artifact_type)"
                )
                conn.execute(
                    "create index if not exists idx_artifacts_run_order"
                    " on artifacts(project_id, session_id)"
                )
            # Slice-E F2: the same CSV in two projects yields the same
            # action_hash, so the old single-column PK let a second preview
            # steal the first run's pending row. Composite PK (action_hash,
            # session_id) keeps the rows independent; a legacy single-PK table is
            # dropped, not migrated — pending rows are 30-minute-TTL scratch.
            legacy_pk_cols = sum(
                1 for row in conn.execute("pragma table_info(pending_actions)") if row[5] > 0
            )
            if legacy_pk_cols == 1:
                conn.execute("drop table pending_actions")
            conn.execute(
                """
                create table if not exists pending_actions (
                    action_hash text not null,
                    session_id text not null,
                    project_id text not null,
                    kind text not null,
                    payload_json text not null,
                    created_at text not null,
                    expires_at text not null,
                    status text not null default 'pending',
                    generation text not null default '',
                    payload_digest text not null default '',
                    consumed_idempotency_key text,
                    primary key (action_hash, session_id)
                )
                """
            )
            # Pre-C1 composite-PK tables lack the token columns; '' never
            # matches a client-supplied token, so legacy rows just re-preview.
            self._ensure_column(conn, "pending_actions", "generation", "text not null default ''")
            self._ensure_column(
                conn, "pending_actions", "payload_digest", "text not null default ''"
            )
            self._ensure_column(conn, "pending_actions", "consumed_idempotency_key", "text")
            self._ensure_column(conn, "sessions", "status", "text not null default 'running'")
            self._ensure_column(conn, "projects", "created_at", "text")
            self._ensure_column(conn, "projects", "sort_order", "integer")
            # Preserve the legacy creation order before introducing manual
            # ordering.  New projects append after the highest persisted rank.
            conn.execute("update projects set sort_order = rowid where sort_order is null")
            # Session-index columns (§8.1): the list API reads these instead of
            # scanning run directories; refresh_session_index/backfill maintain them.
            self._ensure_column(conn, "sessions", "title", "text")
            self._ensure_column(conn, "sessions", "created_at", "text")
            self._ensure_column(conn, "sessions", "updated_at", "text")
            self._ensure_column(conn, "sessions", "dataset_names_json", "text")
            self._ensure_column(conn, "sessions", "artifact_count", "integer not null default 0")
            self._ensure_column(conn, "sessions", "report_status", "text")
            self._ensure_column(
                conn, "sessions", "chat_message_count", "integer not null default 0"
            )
            self._ensure_column(conn, "sessions", "source_session_id", "text")
            self._ensure_column(conn, "sessions", "code_version", "text")
            self._ensure_column(conn, "sessions", "seed", "integer")
            self._ensure_column(conn, "sessions", "input_hashes_json", "text")
            self._ensure_column(conn, "sessions", "model_versions_json", "text")
            conn.execute(
                """
                create index if not exists idx_sessions_project_source
                on sessions(project_id, source_session_id)
                """
            )
            # Worker process id for orphan detection at startup (review F4).
            self._ensure_column(conn, "jobs", "pid", "integer")
            # F-013: NULL intentionally marks a legacy row whose original
            # request cannot be reconstructed. Reusing its key must fail closed.
            self._ensure_column(conn, "jobs", "request_digest", "text")
            # Logical request identity is stable across freshly generated
            # lifecycle run ids. NULL marks a pre-F-013-v2 row and replays
            # fail closed because its original stable scope is unknowable.
            self._ensure_column(conn, "jobs", "request_scope", "text")
            # A lifecycle job may run under a derived run id while reading or
            # writing a source/plan run. Persist that logical lane so SQLite,
            # rather than a process-local check-then-insert, owns exclusivity.
            job_columns = {
                str(row[1]) for row in conn.execute("pragma table_info(jobs)").fetchall()
            }
            job_state_version_present = "state_version" in job_columns
            lane_key_was_missing = "lane_key" not in job_columns
            self._ensure_column(conn, "jobs", "lane_key", "text")
            if lane_key_was_missing:
                # Legacy derived jobs do not persist their source/plan id, so
                # their correct lane cannot be reconstructed. Dead/unqueued
                # rows are safely terminalized; a live detached worker blocks
                # the upgrade until it settles, preventing the new API from
                # admitting a conflicting source-lane job.
                unscoped_legacy = conn.execute(
                    """
                    select job_id, pid
                    from jobs
                    where status not in ('completed', 'failed', 'cancelled')
                      and kind not in ('auto_eda', 'question_exec', 'skill_replay')
                    order by created_at, job_id
                    """
                ).fetchall()
                live = [str(row[0]) for row in unscoped_legacy if _stored_pid_alive(row[1])]
                if live:
                    raise RuntimeError(
                        "Cannot migrate active legacy derived jobs; wait for "
                        "their detached workers to settle, then restart."
                    )
                if unscoped_legacy:
                    now = datetime.now(UTC).isoformat()
                    state_version_update = (
                        ", state_version = state_version + 1" if job_state_version_present else ""
                    )
                    conn.executemany(
                        f"""
                        update jobs
                        set status = 'failed',
                            finished_at = ?,
                            error_code = 'lane_migration_unscoped',
                            error_message = ?,
                            idempotency_key = null
                            {state_version_update}
                        where job_id = ?
                        """,
                        [
                            (
                                now,
                                "Legacy derived job had no persisted source "
                                "lane and no live worker; it was terminalized "
                                "during lane migration.",
                                str(row[0]),
                            )
                            for row in unscoped_legacy
                        ],
                    )
            conn.execute("update jobs set lane_key = session_id where lane_key is null")
            # The old process-local guard could already have admitted multiple
            # active rows for one run. Preserve one deterministic owner
            # (prefer a running worker, otherwise the oldest queued row) and
            # terminalize the duplicates before installing the invariant.
            active_by_lane: dict[str, list[tuple[object, ...]]] = {}
            for row in conn.execute(
                """
                select job_id, lane_key, status, pid, started_at, created_at
                from jobs
                where status not in ('completed', 'failed', 'cancelled')
                order by lane_key, created_at, job_id
                """
            ).fetchall():
                active_by_lane.setdefault(str(row[1]), []).append(row)
            duplicate_job_ids: list[str] = []
            for lane_rows in active_by_lane.values():
                if len(lane_rows) < 2:
                    continue
                live_rows = [row for row in lane_rows if _stored_pid_alive(row[3])]
                if len(live_rows) > 1:
                    raise RuntimeError(
                        "Cannot migrate a legacy lane with multiple live "
                        "workers; wait for them to settle, then restart."
                    )
                owner = (
                    live_rows[0]
                    if live_rows
                    else min(
                        lane_rows,
                        key=lambda row: (
                            0 if str(row[2]) == "running" else 1,
                            str(row[4] or row[5]),
                            str(row[5]),
                            str(row[0]),
                        ),
                    )
                )
                duplicate_job_ids.extend(str(row[0]) for row in lane_rows if row[0] != owner[0])
            if duplicate_job_ids:
                now = datetime.now(UTC).isoformat()
                state_version_update = (
                    ", state_version = state_version + 1" if job_state_version_present else ""
                )
                conn.executemany(
                    f"""
                    update jobs
                    set status = 'failed',
                        finished_at = ?,
                        error_code = 'lane_migration_conflict',
                        error_message = ?,
                        idempotency_key = null
                        {state_version_update}
                    where job_id = ?
                    """,
                    [
                        (
                            now,
                            "Legacy workspace contained more than one active "
                            "job for this lane; this duplicate was terminalized "
                            "during lane migration.",
                            job_id,
                        )
                        for job_id in duplicate_job_ids
                    ],
                )
            conn.execute(
                """
                create unique index if not exists idx_jobs_active_lane
                on jobs(lane_key)
                where status not in ('completed', 'failed', 'cancelled')
                """
            )
            conn.execute(
                "create index if not exists idx_runs_project_created"
                " on sessions(project_id, created_at desc, session_id)"
            )
            self._apply_storage_foundation(conn)

    def _apply_storage_foundation(self, conn: sqlite3.Connection) -> None:
        """Install the additive Phase-R storage/lifecycle schema.

        This intentionally runs after the older ad-hoc migrations. In
        particular, jobs is never rebuilt: F-013 request identity and F-020
        lane ownership survive via ALTER TABLE additions.
        """
        conn.executescript(
            """
            create table if not exists schema_migrations (
                version integer primary key,
                name text not null unique,
                applied_at text not null
            );

            create table if not exists resource_heads (
                resource_kind text not null,
                project_id text not null,
                resource_key text not null,
                relative_path text not null,
                version integer not null default 0,
                content_digest text not null,
                updated_at text not null,
                primary key (resource_kind, project_id, resource_key)
            );

            create table if not exists storage_operations (
                op_id text primary key,
                op_kind text not null,
                resource_kind text not null,
                project_id text not null,
                resource_key text not null,
                expected_version integer,
                target_version integer,
                base_digest text,
                target_digest text,
                request_key text,
                state text not null default 'prepared',
                error_code text,
                error_message text,
                created_at text not null,
                updated_at text not null
            );

            create table if not exists storage_operation_items (
                op_id text not null,
                ordinal integer not null,
                mode text not null,
                source_relpath text not null,
                work_relpath text not null,
                base_digest text,
                target_digest text,
                payload blob,
                required integer not null default 1,
                primary key (op_id, ordinal),
                foreign key (op_id) references storage_operations(op_id)
                    on delete cascade
            );
            """
        )

        self._ensure_column(conn, "sessions", "state_version", "integer not null default 0")
        self._ensure_column(conn, "sessions", "active_job_id", "text")
        self._ensure_column(conn, "sessions", "storage_state", "text not null default 'live'")
        self._ensure_column(conn, "sessions", "delete_op_id", "text")

        job_columns = {str(row[1]) for row in conn.execute("pragma table_info(jobs)").fetchall()}
        kill_fence_was_missing = "kill_fence_state" not in job_columns
        self._ensure_column(conn, "jobs", "state_version", "integer not null default 0")
        self._ensure_column(conn, "jobs", "launch_attempt", "integer not null default 0")
        self._ensure_column(conn, "jobs", "launch_token", "text")
        self._ensure_column(conn, "jobs", "lease_owner", "text")
        self._ensure_column(conn, "jobs", "lease_expires_at", "text")
        self._ensure_column(conn, "jobs", "heartbeat_at", "text")
        self._ensure_column(conn, "jobs", "pid_start_identity", "text")
        self._ensure_column(conn, "jobs", "cancel_requested_at", "text")
        self._ensure_column(conn, "jobs", "cancel_deadline_at", "text")
        self._ensure_column(
            conn,
            "jobs",
            "kill_fence_state",
            "text not null default 'open'",
        )
        self._ensure_column(conn, "jobs", "critical_depth", "integer not null default 0")
        self._ensure_column(conn, "jobs", "critical_owner_generation", "integer")
        if kill_fence_was_missing:
            # Existing jobs predate PID birth identities and must fail closed.
            # The column default remains open for jobs inserted after migration.
            conn.execute("update jobs set kill_fence_state = 'shielded'")

        self._ensure_column(conn, "trace_events", "job_id", "text")
        self._ensure_column(conn, "trace_events", "job_generation", "integer")
        self._ensure_column(conn, "trace_events", "event_key", "text")

        conn.executescript(
            """
            create index if not exists idx_resource_heads_path
                on resource_heads(project_id, relative_path);
            create unique index if not exists idx_storage_operations_active_target
                on storage_operations(resource_kind, project_id, resource_key)
                where state in ('prepared', 'fs_applied', 'db_committed', 'blocked');
            create unique index if not exists idx_storage_operations_request
                on storage_operations(op_kind, request_key)
                where request_key is not null;
            create index if not exists idx_storage_operation_items_op
                on storage_operation_items(op_id, ordinal);

            create index if not exists idx_runs_live_listing
                on sessions(project_id, created_at desc, session_id)
                where storage_state = 'live';
            create unique index if not exists idx_runs_active_job
                on sessions(active_job_id)
                where active_job_id is not null and storage_state = 'live';

            create unique index if not exists idx_jobs_launch_token
                on jobs(launch_token) where launch_token is not null;
            create index if not exists idx_jobs_active_lease
                on jobs(lease_expires_at, job_id)
                where status in ('queued', 'launching', 'running', 'cancelling')
                  and lease_expires_at is not null;
            create index if not exists idx_jobs_active_heartbeat
                on jobs(heartbeat_at, job_id)
                where status in ('launching', 'running', 'cancelling')
                  and heartbeat_at is not null;

            create index if not exists idx_trace_events_job
                on trace_events(job_id, job_generation, id)
                where job_id is not null;
            create unique index if not exists idx_trace_events_event_key
                on trace_events(event_key) where event_key is not null;

            create trigger if not exists trg_jobs_status_valid_insert
            before insert on jobs
            when new.status not in (
                'queued', 'launching', 'running', 'cancelling',
                'completed', 'failed', 'cancelled'
            )
            begin
                select raise(abort, 'invalid job status');
            end;

            create trigger if not exists trg_jobs_status_valid_update
            before update of status on jobs
            when new.status not in (
                'queued', 'launching', 'running', 'cancelling',
                'completed', 'failed', 'cancelled'
            )
            begin
                select raise(abort, 'invalid job status');
            end;

            create trigger if not exists trg_jobs_status_transition_legal
            before update of status on jobs
            when new.status is not old.status
             and not (
                (old.status = 'queued' and new.status in (
                    'launching', 'running', 'completed', 'failed', 'cancelled'
                ))
                or (old.status = 'launching' and new.status in (
                    'running', 'cancelling', 'failed', 'cancelled'
                ))
                or (old.status = 'running' and new.status in (
                    'cancelling', 'completed', 'failed', 'cancelled'
                ))
                or (old.status = 'cancelling' and new.status in (
                    'cancelled', 'failed'
                ))
             )
            begin
                select raise(abort, 'invalid job status transition');
            end;

            create trigger if not exists trg_jobs_status_transition_versioned
            before update of status on jobs
            when new.status is not old.status
             and new.state_version != old.state_version + 1
            begin
                select raise(abort, 'job status transition must increment state_version');
            end;

            create trigger if not exists trg_jobs_terminal_absorbing
            before update of status on jobs
            when old.status in ('completed', 'failed', 'cancelled')
             and new.status is not old.status
            begin
                select raise(abort, 'terminal job status is absorbing');
            end;

            create trigger if not exists trg_jobs_lane_key_present_insert
            before insert on jobs
            when new.lane_key is null or trim(new.lane_key) = ''
            begin
                select raise(abort, 'job lane_key must be non-empty');
            end;

            create trigger if not exists trg_jobs_reject_deleting_run
            before insert on jobs
            when exists (
                select 1 from sessions
                where session_id in (new.session_id, new.lane_key)
                  and storage_state = 'deleting'
            ) or exists (
                select 1 from storage_operations
                where op_kind = 'delete_session' and resource_kind = 'session'
                  and resource_key in (new.session_id, new.lane_key)
                  and state in ('prepared', 'fs_applied', 'db_committed', 'blocked')
            )
            begin
                select raise(abort, 'job target run is deleting');
            end;

            drop trigger if exists trg_jobs_reject_deleting_run_update;
            create trigger trg_jobs_reject_deleting_run_update
            before update of session_id, lane_key, status on jobs
            when (
                exists (
                    select 1 from sessions
                    where session_id in (new.session_id, new.lane_key)
                      and storage_state != 'live'
                )
                or exists (
                    select 1 from storage_operations
                    where op_kind = 'delete_session' and resource_kind = 'session'
                      and resource_key in (new.session_id, new.lane_key)
                      and state in ('prepared', 'fs_applied', 'db_committed', 'blocked')
                )
             )
            begin
                select raise(abort, 'job target run is deleting');
            end;

            create trigger if not exists trg_artifacts_require_live_run_insert
            before insert on artifacts
            when not exists (
                select 1 from sessions
                where session_id = new.session_id and project_id = new.project_id
                  and storage_state = 'live'
            ) or exists (
                select 1 from storage_operations
                where op_kind = 'delete_session' and resource_kind = 'session'
                  and resource_key = new.session_id
                  and state in ('prepared', 'fs_applied', 'db_committed', 'blocked')
            )
            begin
                select raise(abort, 'artifact run is not live');
            end;

            create trigger if not exists trg_artifacts_require_live_run_update
            before update on artifacts
            when not exists (
                select 1 from sessions
                where session_id = new.session_id and project_id = new.project_id
                  and storage_state = 'live'
            ) or exists (
                select 1 from storage_operations
                where op_kind = 'delete_session' and resource_kind = 'session'
                  and resource_key = new.session_id
                  and state in ('prepared', 'fs_applied', 'db_committed', 'blocked')
            )
            begin
                select raise(abort, 'artifact run is not live');
            end;

            create trigger if not exists trg_trace_require_live_run_insert
            before insert on trace_events
            when instr(new.session_id, '__internal') = 0
             and (
                not exists (
                    select 1 from sessions
                    where session_id = new.session_id and project_id = new.project_id
                      and storage_state = 'live'
                )
                or exists (
                    select 1 from storage_operations
                    where op_kind = 'delete_session' and resource_kind = 'session'
                      and resource_key = new.session_id
                      and state in ('prepared', 'fs_applied', 'db_committed', 'blocked')
                )
             )
            begin
                select raise(abort, 'trace run is not live');
            end;

            create trigger if not exists trg_trace_require_live_run_update
            before update on trace_events
            when instr(new.session_id, '__internal') = 0
             and (
                not exists (
                    select 1 from sessions
                    where session_id = new.session_id and project_id = new.project_id
                      and storage_state = 'live'
                )
                or exists (
                    select 1 from storage_operations
                    where op_kind = 'delete_session' and resource_kind = 'session'
                      and resource_key = new.session_id
                      and state in ('prepared', 'fs_applied', 'db_committed', 'blocked')
                )
             )
            begin
                select raise(abort, 'trace run is not live');
            end;

            create trigger if not exists trg_pending_actions_require_live_run_insert
            before insert on pending_actions
            when not exists (
                select 1 from sessions
                where session_id = new.session_id and project_id = new.project_id
                  and storage_state = 'live'
            ) or exists (
                select 1 from storage_operations
                where op_kind = 'delete_session' and resource_kind = 'session'
                  and resource_key = new.session_id
                  and state in ('prepared', 'fs_applied', 'db_committed', 'blocked')
            )
            begin
                select raise(abort, 'pending action run is not live');
            end;

            create trigger if not exists trg_pending_actions_require_live_run_update
            before update on pending_actions
            when not exists (
                select 1 from sessions
                where session_id = new.session_id and project_id = new.project_id
                  and storage_state = 'live'
            ) or exists (
                select 1 from storage_operations
                where op_kind = 'delete_session' and resource_kind = 'session'
                  and resource_key = new.session_id
                  and state in ('prepared', 'fs_applied', 'db_committed', 'blocked')
            )
            begin
                select raise(abort, 'pending action run is not live');
            end;

            create trigger if not exists trg_jobs_lane_key_present_update
            before update of lane_key on jobs
            when new.lane_key is null or trim(new.lane_key) = ''
            begin
                select raise(abort, 'job lane_key must be non-empty');
            end;

            drop trigger if exists trg_jobs_request_identity_immutable;
            create trigger trg_jobs_request_identity_immutable
            before update of request_digest, request_scope, lane_key, session_id on jobs
            when (old.request_digest is not null
                  and new.request_digest is not old.request_digest)
              or (old.request_scope is not null
                  and new.request_scope is not old.request_scope)
              or (old.lane_key is not null and new.lane_key is not old.lane_key)
              or new.session_id is not old.session_id
            begin
                select raise(abort, 'job request identity is immutable');
            end;

            create trigger if not exists trg_jobs_state_version_monotonic
            before update of state_version on jobs
            when new.state_version < old.state_version
            begin
                select raise(abort, 'job state_version cannot decrease');
            end;

            create trigger if not exists trg_jobs_kill_fence_valid_insert
            before insert on jobs
            when new.kill_fence_state not in ('open', 'shielded', 'committed')
            begin
                select raise(abort, 'invalid kill fence state');
            end;

            create trigger if not exists trg_jobs_kill_fence_valid_update
            before update of kill_fence_state on jobs
            when new.kill_fence_state not in ('open', 'shielded', 'committed')
            begin
                select raise(abort, 'invalid kill fence state');
            end;

            create trigger if not exists trg_jobs_kill_fence_committed_absorbing
            before update of kill_fence_state on jobs
            when old.kill_fence_state = 'committed'
             and new.kill_fence_state is not old.kill_fence_state
            begin
                select raise(abort, 'committed kill fence is absorbing');
            end;

            create trigger if not exists trg_jobs_kill_fence_transition_legal
            before update of kill_fence_state on jobs
            when not (
                new.kill_fence_state is old.kill_fence_state
                or (
                    old.kill_fence_state = 'open'
                    and new.kill_fence_state in ('shielded', 'committed')
                )
                or (
                    old.kill_fence_state = 'shielded'
                    and new.kill_fence_state = 'open'
                )
            )
            begin
                select raise(abort, 'invalid kill fence transition');
            end;

            create trigger if not exists trg_jobs_critical_depth_valid_insert
            before insert on jobs
            when new.critical_depth < 0
              or (new.critical_depth = 0 and new.critical_owner_generation is not null)
              or (new.critical_depth > 0 and new.critical_owner_generation is null)
            begin
                select raise(abort, 'invalid job critical section state');
            end;

            create trigger if not exists trg_jobs_critical_depth_valid_update
            before update of critical_depth, critical_owner_generation on jobs
            when new.critical_depth < 0
              or (new.critical_depth = 0 and new.critical_owner_generation is not null)
              or (new.critical_depth > 0 and new.critical_owner_generation is null)
            begin
                select raise(abort, 'invalid job critical section state');
            end;

            create trigger if not exists trg_artifacts_reject_committed_publish_insert
            before insert on artifacts
            when exists (
                select 1 from sessions join jobs
                  on jobs.job_id = sessions.active_job_id
                where sessions.session_id = new.session_id
                  and sessions.project_id = new.project_id
                  and jobs.kill_fence_state = 'committed'
            )
            begin
                select raise(abort, 'run publish fence is committed');
            end;

            create trigger if not exists trg_artifacts_reject_committed_publish_update
            before update on artifacts
            when exists (
                select 1 from sessions join jobs
                  on jobs.job_id = sessions.active_job_id
                where sessions.session_id = new.session_id
                  and sessions.project_id = new.project_id
                  and jobs.kill_fence_state = 'committed'
            )
            begin
                select raise(abort, 'run publish fence is committed');
            end;

            create trigger if not exists trg_trace_reject_committed_publish_insert
            before insert on trace_events
            when exists (
                select 1 from sessions join jobs
                  on jobs.job_id = sessions.active_job_id
                where sessions.session_id = new.session_id
                  and sessions.project_id = new.project_id
                  and jobs.kill_fence_state = 'committed'
            )
            begin
                select raise(abort, 'run publish fence is committed');
            end;

            create trigger if not exists trg_trace_reject_committed_publish_update
            before update on trace_events
            when exists (
                select 1 from sessions join jobs
                  on jobs.job_id = sessions.active_job_id
                where sessions.session_id = new.session_id
                  and sessions.project_id = new.project_id
                  and jobs.kill_fence_state = 'committed'
            )
            begin
                select raise(abort, 'run publish fence is committed');
            end;

            create trigger if not exists trg_pending_reject_committed_publish_insert
            before insert on pending_actions
            when exists (
                select 1 from sessions join jobs
                  on jobs.job_id = sessions.active_job_id
                where sessions.session_id = new.session_id
                  and sessions.project_id = new.project_id
                  and jobs.kill_fence_state = 'committed'
            )
            begin
                select raise(abort, 'run publish fence is committed');
            end;

            create trigger if not exists trg_pending_reject_committed_publish_update
            before update on pending_actions
            when exists (
                select 1 from sessions join jobs
                  on jobs.job_id = sessions.active_job_id
                where sessions.session_id = new.session_id
                  and sessions.project_id = new.project_id
                  and jobs.kill_fence_state = 'committed'
            )
            begin
                select raise(abort, 'run publish fence is committed');
            end;

            create trigger if not exists trg_runs_reject_committed_publish_update
            before update on sessions
            when old.active_job_id is not null
              and new.active_job_id is old.active_job_id
              and exists (
                  select 1 from jobs
                  where jobs.job_id = old.active_job_id
                    and jobs.kill_fence_state = 'committed'
              )
            begin
                select raise(abort, 'run publish fence is committed');
            end;

            create trigger if not exists trg_runs_storage_state_valid_insert
            before insert on sessions
            when new.storage_state not in ('live', 'deleting')
            begin
                select raise(abort, 'invalid run storage_state');
            end;

            create trigger if not exists trg_runs_storage_state_valid_update
            before update of storage_state on sessions
            when new.storage_state not in ('live', 'deleting')
            begin
                select raise(abort, 'invalid run storage_state');
            end;

            create trigger if not exists trg_runs_state_version_monotonic
            before update of state_version on sessions
            when new.state_version < old.state_version
            begin
                select raise(abort, 'run state_version cannot decrease');
            end;

            create trigger if not exists trg_resource_heads_version_monotonic
            before update of version on resource_heads
            when new.version < old.version
            begin
                select raise(abort, 'resource version cannot decrease');
            end;

            create trigger if not exists trg_resource_heads_content_versioned
            before update of relative_path, content_digest, version on resource_heads
            when (
                new.relative_path is not old.relative_path
                or new.content_digest is not old.content_digest
            ) and new.version <= old.version
            begin
                select raise(abort, 'resource content change requires a newer version');
            end;

            create trigger if not exists trg_storage_operations_state_valid_insert
            before insert on storage_operations
            when new.state not in (
                'prepared', 'fs_applied', 'db_committed', 'blocked',
                'done', 'aborted'
            )
            begin
                select raise(abort, 'invalid storage operation state');
            end;

            create trigger if not exists trg_storage_operations_state_valid_update
            before update of state on storage_operations
            when new.state not in (
                'prepared', 'fs_applied', 'db_committed', 'blocked',
                'done', 'aborted'
            )
            begin
                select raise(abort, 'invalid storage operation state');
            end;

            create trigger if not exists trg_storage_operations_terminal_absorbing
            before update of state on storage_operations
            when old.state in ('done', 'aborted')
             and new.state is not old.state
            begin
                select raise(abort, 'terminal storage operation state is absorbing');
            end;

            create trigger if not exists trg_storage_operations_state_transition_legal
            before update of state on storage_operations
            when new.state is not old.state
             and not (
                (old.state = 'prepared' and new.state in (
                    'fs_applied', 'blocked', 'aborted'
                ))
                or (old.state = 'fs_applied' and new.state in (
                    'db_committed', 'done', 'blocked'
                ))
                or (old.state = 'db_committed' and new.state = 'done')
                or (old.state = 'blocked' and new.state in (
                    'prepared', 'aborted'
                ))
             )
            begin
                select raise(abort, 'invalid storage operation state transition');
            end;
            """
        )
        conn.execute(
            """
            insert into schema_migrations(version, name, applied_at)
            values(1, 'unified_storage_foundation_v1', ?)
            on conflict(version) do nothing
            """,
            (datetime.now(UTC).isoformat(),),
        )
        migration = conn.execute("select name from schema_migrations where version = 1").fetchone()
        if migration is None or str(migration[0]) != "unified_storage_foundation_v1":
            raise RuntimeError("Storage foundation migration version 1 is incompatible.")

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
        existing = {row[1] for row in conn.execute(f"pragma table_info({table})").fetchall()}
        if column not in existing:
            conn.execute(f"alter table {table} add column {column} {ddl}")


def _escape_like(term: str) -> str:
    """Neutralize LIKE wildcards so a search for "50%" is not a match-anything."""
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _stored_pid_alive(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and pid_is_alive(value)


def _is_inside(path: Path, root: Path) -> bool:
    """Return whether ``path`` resolves strictly inside ``root``."""
    try:
        resolved = path.resolve()
    except (OSError, ValueError):
        return False
    return resolved != root and resolved.is_relative_to(root)


def _safe_projects_root(workspace: Path | str) -> Path:
    """Return the canonical projects root only when it is a real child dir."""
    workspace_root = require_absolute_workspace(workspace)
    projects_path = workspace_root / "projects"
    if projects_path.is_symlink():
        raise ValueError("Workspace projects root cannot be a symbolic link.")
    projects_root = projects_path.resolve()
    if projects_root.parent != workspace_root:
        raise ValueError("Workspace projects root escaped the workspace.")
    return projects_root


def _validate_project_id_segment(project_id: str) -> None:
    """Require one canonical POSIX-safe segment without narrowing valid names."""

    if (
        not isinstance(project_id, str)
        or not project_id
        or project_id in {".", ".."}
        or "\x00" in project_id
        or "/" in project_id
        or "\\" in project_id
        or unicodedata.normalize("NFC", project_id) != project_id
    ):
        raise ValueError("Project id must be one canonical non-empty path segment.")


def _dataset_names(artifacts: list[Artifact]) -> list[str]:
    """Ordered, de-duplicated dataset names from DatasetProfile payloads."""
    names: list[str] = []
    for artifact in artifacts:
        if artifact.type is not ArtifactType.DATASET_PROFILE:
            continue
        name = artifact.payload.get("name")
        if isinstance(name, str) and name not in names:
            names.append(name)
    return names


def _report_status(artifacts: list[Artifact]) -> str | None:
    """Release status from the SessionSummary/ReportBundle artifact if present."""
    for artifact in artifacts:
        if artifact.type is ArtifactType.SESSION_SUMMARY:
            status = artifact.payload.get("report_status")
            if isinstance(status, str):
                return status
    for artifact in artifacts:
        if artifact.type is ArtifactType.REPORT_BUNDLE:
            status = artifact.payload.get("status")
            if isinstance(status, str):
                return status
    return None
