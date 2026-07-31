"""Artifact browsing use cases (§7.4/§13.2): summaries are index-only rows
(never payloads); the detail read resolves the indexed path and refuses
anything outside the workspace before deserializing."""

from __future__ import annotations

import base64
import json
from pathlib import Path

from eda_platform.application.dto import ArtifactDetail, ArtifactSummary, Page
from eda_platform.application.services.session_service import InvalidCursorError, SessionNotFoundError
from eda_platform.application.workspace_paths import (
    relativize_warnings,
    relativize_workspace_paths,
)
from eda_platform.core.ids import INTERNAL_SESSION_MARKER
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact

DEFAULT_ARTIFACT_LIMIT = 50
MAX_ARTIFACT_LIMIT = 100
MAX_ARTIFACT_PAYLOAD_BYTES = 5 * 1024 * 1024


class ArtifactServiceError(Exception):
    pass


class ArtifactNotFoundError(ArtifactServiceError):
    def __init__(self, artifact_id: str) -> None:
        super().__init__(f"Artifact not found: {artifact_id}")
        self.artifact_id = artifact_id


class ArtifactTooLargeError(ArtifactServiceError):
    def __init__(self, artifact_id: str) -> None:
        super().__init__(f"Artifact too large to serve: {artifact_id}")
        self.artifact_id = artifact_id


class ArtifactService:
    def __init__(self, store: ArtifactStore) -> None:
        self._store = store

    def list_artifacts(
        self,
        session_id: str,
        *,
        artifact_type: str | None = None,
        limit: int = DEFAULT_ARTIFACT_LIMIT,
        cursor: str | None = None,
    ) -> Page[ArtifactSummary]:
        limit = max(1, min(limit, MAX_ARTIFACT_LIMIT))
        project_id = self._project_for_run(session_id)
        after_rowid = _decode_cursor(cursor, artifact_type, session_id) if cursor else None
        rows = self._store.query_artifact_index_rows(
            project_id,
            session_id,
            artifact_type=artifact_type,
            limit=limit + 1,
            after_rowid=after_rowid,
        )
        has_more = len(rows) > limit
        rows = rows[:limit]
        next_cursor = (
            _encode_cursor(rows[-1]["rowid"], artifact_type, session_id) if has_more and rows else None
        )
        return Page[ArtifactSummary](
            items=[
                ArtifactSummary(artifact_id=row["artifact_id"], type=row["artifact_type"])
                for row in rows
            ],
            next_cursor=next_cursor,
        )

    def get_artifact(self, session_id: str, artifact_id: str) -> ArtifactDetail:
        """Read one artifact from the partition named by the detail URL."""
        project_id = self._project_for_run(session_id)
        row = self._store.artifact_index_row(
            artifact_id, project_id=project_id, session_id=session_id
        )
        if row is None:
            raise ArtifactNotFoundError(artifact_id)
        # Internal derived runs are hidden from every API surface; their
        # artifacts must not leak through direct id lookup.
        if INTERNAL_SESSION_MARKER in str(row["session_id"]):
            raise ArtifactNotFoundError(artifact_id)
        path: Path = row["path"]
        # Containment: the indexed path is data, not trusted — refuse reads
        # that resolve outside the workspace (store._abs trust is a known defer).
        try:
            if not path.resolve().is_relative_to(self._store.root.resolve()):
                raise ArtifactNotFoundError(artifact_id)
        except OSError:
            raise ArtifactNotFoundError(artifact_id) from None
        try:
            size = path.stat().st_size
        except OSError:
            raise ArtifactNotFoundError(artifact_id) from None
        if size > MAX_ARTIFACT_PAYLOAD_BYTES:
            raise ArtifactTooLargeError(artifact_id)
        try:
            artifact = Artifact.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise ArtifactNotFoundError(artifact_id) from None
        # Identity check: the payload on disk must be the artifact the index
        # points at — a swapped or aliased file must not serve under this id.
        if (
            artifact.id != artifact_id
            or artifact.session_id != str(row["session_id"])
            or artifact.project_id != str(row["project_id"])
        ):
            raise ArtifactNotFoundError(artifact_id)
        root = self._store.root
        return ArtifactDetail(
            artifact_id=artifact.id,
            type=artifact.type.value,
            project_id=artifact.project_id,
            session_id=artifact.session_id,
            created_at=artifact.created_at,
            payload=relativize_workspace_paths(artifact.payload, root),
            warnings=relativize_warnings(list(artifact.warnings), root),
        )

    def _project_for_run(self, session_id: str) -> str:
        if INTERNAL_SESSION_MARKER in session_id:
            raise SessionNotFoundError(session_id)
        row = self._store.get_session_index_row(session_id)
        if row is None:
            raise SessionNotFoundError(session_id)
        return str(row["project_id"])


def _encode_cursor(rowid: int, artifact_type: str | None, session_id: str) -> str:
    raw = json.dumps({"i": rowid, "t": artifact_type or "", "r": session_id}).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def _decode_cursor(cursor: str, artifact_type: str | None, session_id: str) -> int:
    try:
        decoded = json.loads(base64.urlsafe_b64decode(cursor.encode("ascii")))
    except (ValueError, UnicodeDecodeError):
        raise InvalidCursorError() from None
    if not isinstance(decoded, dict) or not isinstance(decoded.get("i"), int):
        raise InvalidCursorError()
    # A cursor is bound to the type filter and run it was minted under;
    # replaying it elsewhere would silently skip or repeat rows.
    if decoded.get("t") != (artifact_type or "") or decoded.get("r") != session_id:
        raise InvalidCursorError()
    return decoded["i"]
