"""Report download use case: the HTML/PDF/decision-Markdown exports the
Report page download formats, behind one endpoint.

Everything is served from memory. Persisted HTML reports run 12-80 KB and the
PDFs WeasyPrint renders from them a few hundred KB, so a temp file with an
expiry sweeper would add a lifecycle to manage for no benefit;
``MAX_EXPORT_SOURCE_BYTES`` is the ceiling that keeps that true.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from pydantic import ValidationError

from eda_platform.application.services.decision_report_service import DecisionReportService
from eda_platform.application.services.session_service import SessionNotFoundError
from eda_platform.core.ids import INTERNAL_SESSION_MARKER
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.decision_report import DecisionReport

ReportExportFormat = Literal["html", "pdf", "md"]

# A persisted HTML report is self-contained but still bounded: refuse anything
# large enough to make an in-memory export a memory-pressure problem.
MAX_EXPORT_SOURCE_BYTES = 25 * 1024 * 1024

PDF_INSTALL_HINT = (
    "PDF export needs WeasyPrint and its pango libraries. Install them with "
    "`uv sync --extra pdf` and, on macOS, `brew install pango`."
)

_FILENAME_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")
_MAX_FILENAME_STEM = 64


class ReportNotExportableError(Exception):
    """The run has no artifact this format could be rendered from."""


class ReportExportUnavailableError(Exception):
    """The format is supported but its renderer is not installed on this host."""


@dataclass(frozen=True)
class ReportDownload:
    filename: str
    media_type: str
    content: bytes


def export_filename(stem: str, suffix: str) -> str:
    """Header-safe download name.

    Content-Disposition is a header, so any CR/LF or quote in a stem that came
    from a run or report id has to be gone before it gets there.
    """
    safe = _FILENAME_UNSAFE.sub("_", stem).strip("._")[:_MAX_FILENAME_STEM]
    return f"{safe or 'report'}.{suffix}"


class ReportExportService:
    def __init__(
        self, store: ArtifactStore, decision_reports: DecisionReportService
    ) -> None:
        self._store = store
        self._decision_reports = decision_reports

    def download(self, session_id: str, export_format: ReportExportFormat) -> ReportDownload:
        if export_format == "md":
            return self._decision_report_markdown(session_id)
        html = self._html_report(session_id)
        if export_format == "html":
            return ReportDownload(
                filename=export_filename(session_id, "html"),
                media_type="text/html; charset=utf-8",
                content=html.encode("utf-8"),
            )
        return ReportDownload(
            filename=export_filename(session_id, "pdf"),
            media_type="application/pdf",
            content=_render_pdf(html),
        )

    def _html_report(self, session_id: str) -> str:
        project_id = self._project_for_run(session_id)
        for candidate in self._store.latest_artifact_index_rows(
            project_id, session_id, ArtifactType.HTML_REPORT.value
        ):
            artifact = self._read_contained(
                str(candidate["artifact_id"]), project_id=project_id, session_id=session_id
            )
            if artifact is None:
                continue
            html = artifact.payload.get("html")
            if isinstance(html, str) and html.strip():
                return html
        raise ReportNotExportableError(
            "This run has no HTML report. Re-run the analysis with reporting enabled."
        )

    def _decision_report_markdown(self, session_id: str) -> ReportDownload:
        from eda_platform.tools.exporter import decision_report_to_markdown

        view = self._decision_reports.get_decision_report(session_id)
        if view.status != "available" or not view.artifact_id:
            raise ReportNotExportableError("This project has no decision report yet.")
        # Freshness gates the *button*, not the endpoint — same layer the legacy
        # synthesis page put it at (`export_available` on the view).
        project_id = self._project_for_run(session_id)
        artifact = self._read_contained(
            view.artifact_id, project_id=project_id, session_id=view.report_session_id
        )
        if artifact is None or artifact.type is not ArtifactType.DECISION_REPORT:
            raise ReportNotExportableError("The decision report artifact is unreadable.")
        try:
            report = DecisionReport.model_validate(artifact.payload)
        except ValidationError as exc:
            raise ReportNotExportableError(
                "The decision report artifact is not a valid report."
            ) from exc
        return ReportDownload(
            filename=export_filename(report.report_id, "md"),
            media_type="text/markdown; charset=utf-8",
            content=decision_report_to_markdown(report).encode("utf-8"),
        )

    def _read_contained(
        self, artifact_id: str, *, project_id: str, session_id: str | None
    ) -> Artifact | None:
        """Read an indexed artifact only if its resolved path stays in the
        workspace and its size stays under the export ceiling."""
        row = self._store.artifact_index_row(
            artifact_id, project_id=project_id, session_id=session_id
        )
        if row is None:
            return None
        path: Path = row["path"]
        try:
            if not path.resolve().is_relative_to(self._store.root.resolve()):
                return None
            if path.stat().st_size > MAX_EXPORT_SOURCE_BYTES:
                return None
            return Artifact.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _project_for_run(self, session_id: str) -> str:
        if INTERNAL_SESSION_MARKER in session_id:
            raise SessionNotFoundError(session_id)
        row = self._store.get_session_index_row(session_id)
        if row is None:
            raise SessionNotFoundError(session_id)
        return str(row["project_id"])


def _render_pdf(html: str) -> bytes:
    from eda_platform.tools.pdf_exporter import export_pdf, is_pdf_available

    if not is_pdf_available():
        raise ReportExportUnavailableError(PDF_INSTALL_HINT)
    with TemporaryDirectory(prefix="eda_report_pdf_") as temp_dir:
        temp_path = Path(temp_dir)
        html_path = temp_path / "report.html"
        html_path.write_text(html, encoding="utf-8")
        try:
            return export_pdf(html_path, temp_path / "report.pdf").read_bytes()
        except RuntimeError as exc:
            raise ReportExportUnavailableError(PDF_INSTALL_HINT) from exc
