"""Project/Run use cases backed by the SQLite runs index.

The list path is DB-only by design (§8.1): no run-directory enumeration, no
artifact globbing, no chat JSONL scans. Those costs are paid once at write time
(`ArtifactStore.refresh_session_index`) or during backfill.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

from eda_platform.application.dto import (
    Page,
    ProjectDeleted,
    ProjectSummary,
    SessionDeleted,
    SessionDetail,
    SessionSummary,
)
from eda_platform.core.bounded_pagination import InvalidCursorError
from eda_platform.core.fs import fsync_directory, remove_tree
from eda_platform.core.ids import (
    DERIVED_SESSION_PREFIXES,
    INTERNAL_SESSION_MARKER,
    is_internal_project_id,
)
from eda_platform.core.session_deletion import (
    SessionDeletionBlockedError,
    SessionDeletionBusyError,
    SessionDeletionCoordinator,
    SessionDeletionNotFoundError,
    SessionDeletionRetryableError,
)
from eda_platform.core.store import ArtifactStore, ProjectOrderConflictError
from eda_platform.schemas.sessions import clip_run_title

DEFAULT_PAGE_LIMIT = 30
MAX_PAGE_LIMIT = 100
MAX_SEARCH_LENGTH = 200

MAX_PROJECT_ID_LENGTH = 64
MAX_PROJECT_NAME_LENGTH = 200
# Spaces are allowed: real workspaces already hold ids like "Brazilian E-Commerce".
_PROJECT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.-]*$")

_STALE_PROJECT_ORDER = (
    "Project order must contain each current project exactly once. "
    "Refresh and try again."
)


class SessionServiceError(Exception):
    pass


class ProjectNotFoundError(SessionServiceError):
    def __init__(self, project_id: str) -> None:
        super().__init__(f"Project not found: {project_id}")
        self.project_id = project_id


class SessionNotFoundError(SessionServiceError):
    error_code = "session_not_found"

    def __init__(self, session_id: str) -> None:
        super().__init__(f"Session not found: {session_id}")
        self.session_id = session_id


class SessionBusyError(SessionServiceError):
    def __init__(self, session_id: str, job_id: str) -> None:
        super().__init__(
            f"Run {session_id} has an active job ({job_id}); cancel it before deleting the run."
        )
        self.session_id = session_id
        self.job_id = job_id


class SessionDeleteRetryableError(SessionServiceError):
    def __init__(self, op_id: str, reason: str) -> None:
        super().__init__(f"Run deletion {op_id} can be retried: {reason}")
        self.op_id = op_id
        self.reason = reason


class SessionDeleteBlockedError(SessionServiceError):
    def __init__(self, op_id: str, reason: str) -> None:
        super().__init__(f"Run deletion {op_id} is blocked: {reason}")
        self.op_id = op_id
        self.reason = reason


class ProjectValidationError(SessionServiceError):
    pass


class ProjectConflictError(SessionServiceError):
    def __init__(self, project_id: str, existing_project_id: str) -> None:
        super().__init__(
            f"Project id {project_id!r} only differs in case from the existing "
            f"project {existing_project_id!r}. On a case-insensitive filesystem both "
            "would share one directory; open the existing project or pick another id."
        )
        self.project_id = project_id
        self.existing_project_id = existing_project_id


class ProjectBusyError(SessionServiceError):
    def __init__(self, project_id: str, job_id: str) -> None:
        super().__init__(
            f"Project {project_id} has an active job ({job_id}); "
            "cancel it before deleting the project."
        )
        self.project_id = project_id
        self.job_id = job_id


class SessionService:
    def __init__(
        self,
        store: ArtifactStore,
        deletion_coordinator: SessionDeletionCoordinator | None = None,
    ) -> None:
        self._store = store
        self._deletion = deletion_coordinator or SessionDeletionCoordinator(store)

    def list_projects(self) -> list[ProjectSummary]:
        rows = self._store.project_index_rows(
            exclude_session_id_containing=INTERNAL_SESSION_MARKER,
            exclude_session_id_prefixes=DERIVED_SESSION_PREFIXES,
        )
        # Global New session is stored in an internal bucket so the existing
        # project-scoped filesystem, quotas and APIs stay intact. It is never a
        # user project and must not appear in project navigation or management.
        return [
            ProjectSummary.model_validate(row)
            for row in rows
            if not is_internal_project_id(row["project_id"])
        ]

    def create_project(self, project_id: str, name: str = "") -> tuple[ProjectSummary, bool]:
        """Register a project so a fresh workspace can be used from the API alone.

        Returns `(summary, created)`; `created` is False when the project already
        existed, which the route reports as 200 instead of 201.
        """
        project_id = _validated_project_id(project_id)
        display_name = name.strip()[:MAX_PROJECT_NAME_LENGTH] or project_id

        existing = self.list_projects()
        by_id = {row.project_id: row for row in existing}
        if project_id in by_id:
            return by_id[project_id], False
        # macOS/Windows resolve "Sales" and "sales" to the same directory, so a
        # near-miss id would silently write into the other project's workspace.
        folded = project_id.casefold()
        for row in existing:
            if row.project_id.casefold() == folded:
                raise ProjectConflictError(project_id, row.project_id)

        try:
            self._store.ensure_project(project_id, name=display_name)
        except ValueError as exc:
            raise ProjectValidationError(str(exc)) from exc
        return ProjectSummary(project_id=project_id, name=display_name, session_count=0), True

    def rename_project(self, project_id: str, name: str) -> ProjectSummary:
        """Rename a project's display label without changing its id or files."""
        project_id = _validated_project_id(project_id)
        if is_internal_project_id(project_id) or not self._store.project_exists(project_id):
            raise ProjectNotFoundError(project_id)
        display_name = name.strip()[:MAX_PROJECT_NAME_LENGTH]
        if not display_name:
            raise ProjectValidationError("Project name cannot be blank.")
        if not self._store.rename_project(project_id, display_name):
            raise ProjectNotFoundError(project_id)
        count = next(
            (row.session_count for row in self.list_projects() if row.project_id == project_id),
            0,
        )
        return ProjectSummary(project_id=project_id, name=display_name, session_count=count)

    def reorder_projects(self, project_ids: list[str]) -> list[ProjectSummary]:
        """Store the user's complete project order.

        Internal buckets are intentionally excluded: they are implementation
        detail, never draggable navigation entries.  Requiring an exact set
        also prevents an old browser tab from accidentally hiding a project
        created in another tab.
        """
        current_ids = {project.project_id for project in self.list_projects()}
        if len(project_ids) != len(set(project_ids)) or set(project_ids) != current_ids:
            raise ProjectValidationError(_STALE_PROJECT_ORDER)
        try:
            self._store.reorder_projects(project_ids)
        except ProjectOrderConflictError as exc:
            # The check above ran on its own connection; the store re-checks
            # inside the fence and lands here when a project was deleted in
            # between. Returning the pre-read snapshot would have shown it.
            raise ProjectValidationError(_STALE_PROJECT_ORDER) from exc
        return self.list_projects()

    def rename_session(self, session_id: str, name: str) -> SessionSummary:
        """Change a run's display title while keeping the id, URL and files stable."""
        row = self._store.get_session_index_row(session_id)
        if row is None or INTERNAL_SESSION_MARKER in session_id:
            raise SessionNotFoundError(session_id)
        title = clip_run_title(name)
        if not title:
            raise ProjectValidationError("Session name cannot be blank.")
        manifest = self._store.read_manifest(row["project_id"], session_id)
        if manifest is None:
            raise SessionNotFoundError(session_id)
        self._store.write_manifest(manifest.model_copy(update={"title": title}))
        updated = self._store.get_session_index_row(session_id)
        if updated is None:
            raise SessionNotFoundError(session_id)
        return _row_to_summary(updated)

    def delete_project(self, project_id: str) -> ProjectDeleted:
        """Irreversibly delete a project's runs, uploads and project metadata.

        File removal happens before the index transaction: if it fails, the
        project remains visible and the user can retry instead of receiving a
        false success for data that is still on disk.
        """
        project_id = _validated_project_id(project_id)
        # The bucket holding sessions that belong to no project is storage, not
        # a project: list_projects already hides it, so deleting it is not a
        # thing a caller can mean — and one such call takes every standalone
        # session with it. 404 keeps that consistent with the listing.
        if is_internal_project_id(project_id):
            raise ProjectNotFoundError(project_id)
        if not self._store.project_exists(project_id):
            raise ProjectNotFoundError(project_id)
        active_job = self._store.active_project_job(project_id)
        if active_job is not None:
            raise ProjectBusyError(project_id, active_job)
        project_dir = self._store.project_dir(project_id)
        if project_dir.is_symlink():
            raise ProjectValidationError("Project directory cannot be a symbolic link.")
        try:
            if project_dir.exists():
                remove_tree(project_dir)
                fsync_directory(project_dir.parent)
            self._store.delete_project(project_id)
        except RuntimeError as exc:
            raise ProjectBusyError(project_id, str(exc)) from exc
        return ProjectDeleted(project_id=project_id)

    def list_sessions(
        self,
        project_id: str,
        *,
        limit: int = DEFAULT_PAGE_LIMIT,
        cursor: str | None = None,
        include_derived: bool = False,
        q: str | None = None,
    ) -> Page[SessionSummary]:
        """Top-level runs of a project. Runs a user action derived from another
        run (question batches, skill replays) are noise here, so they are
        excluded unless asked for; they stay reachable by direct id.

        ``q`` filters on title, dataset names and run id in SQL, so search keeps
        the DB-only listing contract (§8.1)."""
        limit = max(1, min(limit, MAX_PAGE_LIMIT))
        if not self._store.project_exists(project_id):
            raise ProjectNotFoundError(project_id)
        normalized_query = _normalized_query(q)
        fingerprint = _cursor_fingerprint(
            project_id=project_id,
            query=normalized_query,
            include_derived=include_derived,
        )
        before = _decode_cursor(cursor, fingerprint) if cursor else None
        rows = self._store.query_session_index_rows(
            project_id,
            limit=limit + 1,
            before=before,
            exclude_session_id_containing=INTERNAL_SESSION_MARKER,
            exclude_session_id_prefixes=() if include_derived else DERIVED_SESSION_PREFIXES,
            search=normalized_query,
        )
        has_more = len(rows) > limit
        rows = rows[:limit]
        next_cursor = None
        if has_more and rows:
            last = rows[-1]
            next_cursor = _encode_cursor(
                last["updated_at"] or "",
                last["session_id"],
                fingerprint,
            )
        return Page[SessionSummary](
            items=[_row_to_summary(row) for row in rows],
            next_cursor=next_cursor,
        )

    def get_session_detail(self, session_id: str) -> SessionDetail:
        # Machinery-owned derived runs are hidden from lists; hide them from
        # direct lookup too so the API never exposes them.
        if INTERNAL_SESSION_MARKER in session_id:
            raise SessionNotFoundError(session_id)
        row = self._store.get_session_index_row(session_id)
        if row is None:
            raise SessionNotFoundError(session_id)
        summary = _row_to_summary(row)
        warnings: list[str] = []
        code_version: str | None = None
        seed: int | None = None
        source_session_id: str | None = None
        try:
            manifest = self._store.read_manifest(summary.project_id, session_id)
        except (OSError, ValueError):
            manifest = None
            warnings.append("manifest unreadable")
        if manifest is None:
            if "manifest unreadable" not in warnings:
                warnings.append("manifest missing")
        else:
            code_version = manifest.code_version
            seed = manifest.seed
            source_session_id = manifest.source_session_id
        return SessionDetail(
            **summary.model_dump(),
            code_version=code_version,
            seed=seed,
            source_session_id=source_session_id,
            artifact_type_counts=self._store.artifact_type_counts(summary.project_id, session_id),
            warnings=warnings,
        )

    def delete_session(self, session_id: str) -> SessionDeleted:
        """Delete a run through the recoverable cross-media coordinator."""
        if INTERNAL_SESSION_MARKER in session_id:
            raise SessionNotFoundError(session_id)
        try:
            result = self._deletion.delete(session_id)
        except SessionDeletionNotFoundError as exc:
            raise SessionNotFoundError(session_id) from exc
        except SessionDeletionBusyError as exc:
            raise SessionBusyError(exc.session_id, exc.job_id) from exc
        except SessionDeletionRetryableError as exc:
            raise SessionDeleteRetryableError(exc.op_id, exc.reason) from exc
        except SessionDeletionBlockedError as exc:
            raise SessionDeleteBlockedError(exc.op_id, exc.reason) from exc
        return SessionDeleted(
            session_id=result.session_id,
            project_id=result.project_id,
            deleted=result.deleted,
        )


def _normalized_query(q: str | None) -> str | None:
    if q is None:
        return None
    trimmed = q.strip()[:MAX_SEARCH_LENGTH]
    return trimmed or None


def _validated_project_id(raw: str) -> str:
    project_id = raw.strip()
    if not project_id:
        raise ProjectValidationError("project_id must not be empty.")
    if len(project_id) > MAX_PROJECT_ID_LENGTH:
        raise ProjectValidationError(
            f"project_id must be at most {MAX_PROJECT_ID_LENGTH} characters."
        )
    # The id becomes a directory name under the workspace, so anything the OS
    # would resolve elsewhere must be rejected before it reaches the filesystem.
    if project_id in {".", ".."} or Path(project_id).name != project_id:
        raise ProjectValidationError("project_id must be a single path segment.")
    if not _PROJECT_ID_RE.fullmatch(project_id):
        raise ProjectValidationError(
            "project_id must start with a letter or digit and may contain only "
            "letters, digits, spaces, '_', '.' and '-'."
        )
    return project_id


def _row_to_summary(row: dict) -> SessionSummary:
    dataset_names: list[str] = []
    raw_names = row.get("dataset_names_json")
    if isinstance(raw_names, str):
        with suppress(ValueError):
            parsed = json.loads(raw_names)
            if isinstance(parsed, list):
                dataset_names = [str(name) for name in parsed]
    return SessionSummary(
        session_id=row["session_id"],
        project_id=row["project_id"],
        title=row.get("title"),
        status=row.get("status") or "unknown",
        created_at=_parse_datetime(row.get("created_at")),
        updated_at=_parse_datetime(row.get("updated_at")),
        dataset_names=dataset_names,
        artifact_count=int(row.get("artifact_count") or 0),
        report_status=row.get("report_status"),
        chat_message_count=int(row.get("chat_message_count") or 0),
    )


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    # The API contract is RFC 3339: a naive legacy value is declared UTC rather
    # than serialized without an offset (strict clients reject that).
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _cursor_fingerprint(
    *, project_id: str, query: str | None, include_derived: bool
) -> str:
    canonical = json.dumps(
        {
            "include_derived": include_derived,
            "project_id": project_id,
            "q": query or "",
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


def _encode_cursor(created_at: str, session_id: str, fingerprint: str) -> str:
    raw = json.dumps(
        {"v": 1, "c": created_at, "r": session_id, "f": fingerprint},
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def _decode_cursor(cursor: str, expected_fingerprint: str) -> tuple[str, str]:
    try:
        decoded = json.loads(base64.urlsafe_b64decode(cursor.encode("ascii")))
    except (ValueError, UnicodeDecodeError):
        raise InvalidCursorError() from None
    if not isinstance(decoded, dict):
        raise InvalidCursorError()
    created_at, session_id, fingerprint = (
        decoded.get("c"),
        decoded.get("r"),
        decoded.get("f"),
    )
    if (
        decoded.get("v") != 1
        or not isinstance(created_at, str)
        or not isinstance(session_id, str)
        or not isinstance(fingerprint, str)
        or not hmac.compare_digest(fingerprint, expected_fingerprint)
    ):
        raise InvalidCursorError()
    return created_at, session_id
