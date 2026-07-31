"""Report download endpoint: format routing, typed refusals, and the path /
header guards (§7.5 export slice)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import eda_platform.application.services.report_export_service as export_module
from eda_platform.api.main import create_app
from eda_platform.application.services.report_export_service import export_filename
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType

PROJECT = "demo"
RUN = "run_1"
BARE_RUN = "run_2"

HTML_BODY = "<!doctype html>\n<html lang=\"en\"><body><h1>EDA Agent Report</h1></body></html>\n"

DECISION_PAYLOAD = {
    "report_id": "dreport_1",
    "brief_id": "brief_1",
    "project_id": PROJECT,
    "title": "Churn drivers",
    "scqa": {
        "situation": "Churn rose in Q3.",
        "complication": "The cause is unclear.",
        "question": "Which segment drives it?",
        "answer": "Enterprise renewals.",
    },
    "sections": [{"title": "Evidence", "body": "Renewals fell 12%.", "finding_artifact_ids": []}],
    "limitations": ["Single quarter only."],
    "investigation_gaps": [],
    "report_readiness": "eligible",
    "source_finding_artifact_ids": ["finding_1"],
}


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    store = ArtifactStore(tmp_path)
    store.ensure_project(PROJECT, name="Demo")
    store.start_session(PROJECT, RUN)
    store.start_session(PROJECT, BARE_RUN)
    store.save_artifact(
        Artifact(
            id="html_1",
            type=ArtifactType.HTML_REPORT,
            project_id=PROJECT,
            session_id=RUN,
            payload={"html": HTML_BODY},
        )
    )
    return tmp_path


@pytest.fixture
def client(workspace: Path) -> TestClient:
    return TestClient(create_app(workspace))


def test_download_html_streams_the_persisted_report(client: TestClient) -> None:
    response = client.get(f"/api/v1/sessions/{RUN}/report/download", params={"format": "html"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["content-disposition"] == 'attachment; filename="run_1.html"'
    assert response.content.decode("utf-8").startswith("<!doctype html>")


def test_download_defaults_to_html(client: TestClient) -> None:
    response = client.get(f"/api/v1/sessions/{RUN}/report/download")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


def test_download_html_without_a_report_is_404(client: TestClient) -> None:
    response = client.get(f"/api/v1/sessions/{BARE_RUN}/report/download", params={"format": "html"})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "report_not_exportable"


def test_download_pdf_renders_when_weasyprint_is_present(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_exporter = pytest.importorskip("eda_platform.tools.pdf_exporter")
    if not pdf_exporter.is_pdf_available():
        pytest.skip("WeasyPrint/pango not installed on this host")
    response = client.get(f"/api/v1/sessions/{RUN}/report/download", params={"format": "pdf"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["content-disposition"] == 'attachment; filename="run_1.pdf"'
    assert response.content.startswith(b"%PDF-")


def test_download_pdf_without_weasyprint_is_503_not_500(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _unavailable(html: str) -> bytes:
        raise export_module.ReportExportUnavailableError(export_module.PDF_INSTALL_HINT)

    monkeypatch.setattr(export_module, "_render_pdf", _unavailable)
    response = client.get(f"/api/v1/sessions/{RUN}/report/download", params={"format": "pdf"})
    assert response.status_code == 503
    body = response.json()["error"]
    assert body["code"] == "report_export_unavailable"
    assert "uv sync --extra pdf" in body["message"]


def test_download_markdown_renders_the_decision_report(
    workspace: Path, client: TestClient
) -> None:
    store = ArtifactStore(workspace)
    store.save_artifact(
        Artifact(
            id="dreport_1",
            type=ArtifactType.DECISION_REPORT,
            project_id=PROJECT,
            session_id=RUN,
            payload=DECISION_PAYLOAD,
        )
    )
    response = client.get(f"/api/v1/sessions/{RUN}/report/download", params={"format": "md"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert response.headers["content-disposition"] == 'attachment; filename="dreport_1.md"'
    text = response.content.decode("utf-8")
    assert text.startswith("# Churn drivers")
    assert "## Situation" in text


def test_download_markdown_without_a_decision_report_is_404(client: TestClient) -> None:
    response = client.get(f"/api/v1/sessions/{RUN}/report/download", params={"format": "md"})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "report_not_exportable"


def test_download_unknown_run_is_404(client: TestClient) -> None:
    response = client.get("/api/v1/sessions/missing/report/download")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "session_not_found"


@pytest.mark.parametrize(
    "session_id",
    [
        "../../../etc/passwd",
        "..%2F..%2Fetc%2Fpasswd",
        "%2e%2e%2f%2e%2e%2fstate.sqlite",
        "..",
        "....",
        f"{PROJECT}__internal_probe",
    ],
)
def test_download_refuses_traversal_session_ids(client: TestClient, session_id: str) -> None:
    """A traversal-shaped run id never resolves to a file.

    Ids carrying a separator do not even match the route (404 http_error); the
    rest reach the handler and miss the runs index (404 run_not_found). Either
    way nothing off the workspace is read.
    """
    response = client.get(f"/api/v1/sessions/{session_id}/report/download", params={"format": "html"})
    assert response.status_code == 404
    assert response.json()["error"]["code"] in {"session_not_found", "http_error"}
    assert b"root:" not in response.content
    assert b"SQLite format" not in response.content


def test_download_internal_run_marker_is_run_not_found(client: TestClient) -> None:
    response = client.get(f"/api/v1/sessions/{PROJECT}__internal/report/download")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "session_not_found"


@pytest.mark.parametrize("bad_format", ["../../etc/passwd", "docx", "", "HTML"])
def test_download_rejects_unknown_formats(client: TestClient, bad_format: str) -> None:
    response = client.get(
        f"/api/v1/sessions/{RUN}/report/download", params={"format": bad_format}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_download_refuses_an_html_artifact_outside_the_workspace(
    workspace: Path, client: TestClient, tmp_path: Path
) -> None:
    """Containment has teeth: repoint the index at a file outside the root."""
    outside = tmp_path.parent / "outside_report.json"
    outside.write_text(
        Artifact(
            id="html_1",
            type=ArtifactType.HTML_REPORT,
            project_id=PROJECT,
            session_id=RUN,
            payload={"html": "<html>leaked</html>"},
        ).model_dump_json(),
        encoding="utf-8",
    )
    import sqlite3

    with sqlite3.connect(workspace / "state.sqlite") as conn:
        conn.execute(
            "update artifacts set path = ? where artifact_id = ?", (str(outside), "html_1")
        )
    response = client.get(f"/api/v1/sessions/{RUN}/report/download", params={"format": "html"})
    assert response.status_code == 404
    assert b"leaked" not in response.content


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("run_1", "run_1.html"),
        ('run"\r\nX-Injected: 1', "run___X-Injected__1.html"),
        ("../../etc/passwd", "etc_passwd.html"),
        ("...", "report.html"),
        ("", "report.html"),
        ("a" * 200, "a" * 64 + ".html"),
    ],
)
def test_export_filename_is_header_safe(stem: str, expected: str) -> None:
    name = export_filename(stem, "html")
    assert name == expected
    assert "\r" not in name and "\n" not in name and '"' not in name
