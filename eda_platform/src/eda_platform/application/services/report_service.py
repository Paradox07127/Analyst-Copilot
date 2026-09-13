"""Report reader use case: same source preference as core.session_loader
(_report_markdown) — the MarkdownReport artifact payload first, then the
report/report.md file. A missing report is a normal ReportView(status="none")."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from eda_platform.application.dto import ReportView
from eda_platform.application.services.session_service import SessionNotFoundError
from eda_platform.core.ids import INTERNAL_SESSION_MARKER
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType

MAX_REPORT_READ_BYTES = 5 * 1024 * 1024


class ReportTooLargeError(Exception):
    def __init__(self, max_bytes: int = MAX_REPORT_READ_BYTES) -> None:
        super().__init__(f"Report exceeds the {max_bytes} byte read limit.")
        self.max_bytes = max_bytes


class ReportService:
    def __init__(self, store: ArtifactStore) -> None:
        self._store = store

    def get_report(self, session_id: str) -> ReportView:
        if INTERNAL_SESSION_MARKER in session_id:
            raise SessionNotFoundError(session_id)
        row = self._store.get_session_index_row(session_id)
        if row is None:
            raise SessionNotFoundError(session_id)
        project_id = str(row["project_id"])
        status = row.get("report_status")

        degraded_reason = self._fallback_note(project_id, session_id)

        # Newest MarkdownReport wins (a regenerated report supersedes older ones);
        # every candidate path is containment-checked before it is read.
        for candidate in self._store.latest_artifact_index_rows(
            project_id, session_id, ArtifactType.MARKDOWN_REPORT.value
        ):
            artifact = self._read_contained_artifact(
                str(candidate["artifact_id"]), project_id=project_id, session_id=session_id
            )
            if artifact is None:
                continue
            markdown = artifact.payload.get("markdown")
            if isinstance(markdown, str):
                return ReportView(
                    session_id=session_id,
                    status=status if isinstance(status, str) else "generated",
                    markdown=markdown,
                    generated_at=artifact.created_at,
                    degraded=degraded_reason is not None,
                    degraded_reason=degraded_reason,
                )

        report_file = self._store.session_dir(project_id, session_id) / "report" / "report.md"
        # Containment: session_id/project_id come from our own index, but a report
        # read is still refused unless the resolved path stays in the workspace.
        try:
            resolved = report_file.resolve()
            contained = resolved.is_relative_to(self._store.root.resolve())
        except OSError:
            contained = False
        if contained and report_file.is_file():
            try:
                markdown = _read_report_text(report_file)
                generated_at = datetime.fromtimestamp(report_file.stat().st_mtime, tz=UTC)
            except (OSError, UnicodeDecodeError):
                markdown = None
                generated_at = None
            if markdown is not None:
                return ReportView(
                    session_id=session_id,
                    status=status if isinstance(status, str) else "generated",
                    markdown=markdown,
                    generated_at=generated_at,
                    degraded=degraded_reason is not None,
                    degraded_reason=degraded_reason,
                )

        return ReportView(session_id=session_id, status="none", markdown="")

    def _fallback_note(self, project_id: str, session_id: str) -> str | None:
        """The newest report audit's deterministic-fallback note, if any.

        The audit validates the exact bundle the newest report came from, so
        its semantic note is the authoritative "this report was written
        without the model" signal.
        """
        for candidate in self._store.latest_artifact_index_rows(
            project_id, session_id, ArtifactType.REPORT_AUDIT.value
        ):
            artifact = self._read_contained_artifact(
                str(candidate["artifact_id"]), project_id=project_id, session_id=session_id
            )
            if artifact is None:
                continue
            notes = artifact.payload.get("semantic_notes")
            if not isinstance(notes, list):
                return None
            for note in notes:
                if isinstance(note, str) and note.startswith("Deterministic fallback"):
                    return note
            return None
        return None

    def _read_contained_artifact(
        self, artifact_id: str, *, project_id: str, session_id: str
    ) -> Artifact | None:
        """Read an indexed artifact only if its resolved path stays in the workspace."""
        row = self._store.artifact_index_row(
            artifact_id, project_id=project_id, session_id=session_id
        )
        if row is None:
            return None
        path = row["path"]
        try:
            if not path.resolve().is_relative_to(self._store.root.resolve()):
                return None
        except OSError:
            return None
        try:
            return Artifact.model_validate_json(_read_report_text(path))
        except (OSError, ValueError):
            return None


def _read_report_text(path: Path) -> str:
    with path.open("rb") as handle:
        payload = handle.read(MAX_REPORT_READ_BYTES + 1)
    if len(payload) > MAX_REPORT_READ_BYTES:
        raise ReportTooLargeError(MAX_REPORT_READ_BYTES)
    return payload.decode("utf-8")
