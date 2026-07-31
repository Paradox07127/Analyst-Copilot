"""ReportService: artifact-first source preference, file fallback, none status,
workspace containment."""

from __future__ import annotations

from pathlib import Path

import pytest

from eda_platform.application.services.report_service import ReportService
from eda_platform.application.services.session_service import SessionNotFoundError
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType

PROJECT = "demo"
RUN = "run_1"


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    store = ArtifactStore(tmp_path)
    store.ensure_project(PROJECT, name="Demo")
    store.start_session(PROJECT, RUN)
    return store


@pytest.fixture
def service(store: ArtifactStore) -> ReportService:
    return ReportService(store)


def _save_markdown_report(store: ArtifactStore, markdown: str) -> Artifact:
    artifact = Artifact(
        id="md_report_1",
        type=ArtifactType.MARKDOWN_REPORT,
        project_id=PROJECT,
        session_id=RUN,
        payload={"markdown": markdown},
    )
    store.save_artifact(artifact)
    return artifact


def test_prefers_markdown_report_artifact(store: ArtifactStore, service: ReportService) -> None:
    (store.session_dir(PROJECT, RUN) / "report" / "report.md").write_text(
        "# stale file", encoding="utf-8"
    )
    artifact = _save_markdown_report(store, "# From artifact\n\ncontent")

    view = service.get_report(RUN)

    assert view.markdown == "# From artifact\n\ncontent"
    assert view.status != "none"
    assert view.generated_at == artifact.created_at


def test_newest_markdown_report_wins(store: ArtifactStore, service: ReportService) -> None:
    store.save_artifact(
        Artifact(
            id="md_report_old",
            type=ArtifactType.MARKDOWN_REPORT,
            project_id=PROJECT,
            session_id=RUN,
            payload={"markdown": "# Old report"},
        )
    )
    store.save_artifact(
        Artifact(
            id="md_report_new",
            type=ArtifactType.MARKDOWN_REPORT,
            project_id=PROJECT,
            session_id=RUN,
            payload={"markdown": "# New report"},
        )
    )

    assert service.get_report(RUN).markdown == "# New report"


def test_report_artifact_symlink_escaping_workspace_is_refused(
    tmp_path: Path, store: ArtifactStore, service: ReportService
) -> None:
    artifact = _save_markdown_report(store, "# Secret outside")
    payload_path = store.artifact_path(PROJECT, RUN, artifact.id)
    outside = tmp_path.parent / "outside_artifact.json"
    outside.write_text(payload_path.read_text(encoding="utf-8"), encoding="utf-8")
    payload_path.unlink()
    payload_path.symlink_to(outside)

    view = service.get_report(RUN)

    assert view.status == "none"
    assert view.markdown == ""


def test_status_comes_from_run_index(store: ArtifactStore, service: ReportService) -> None:
    _save_markdown_report(store, "# Report")
    store.save_artifact(
        Artifact(
            id="runsummary_1",
            type=ArtifactType.SESSION_SUMMARY,
            project_id=PROJECT,
            session_id=RUN,
            payload={"report_status": "validated"},
        )
    )

    assert service.get_report(RUN).status == "validated"


def test_falls_back_to_report_file(store: ArtifactStore, service: ReportService) -> None:
    (store.session_dir(PROJECT, RUN) / "report" / "report.md").write_text(
        "# From file", encoding="utf-8"
    )

    view = service.get_report(RUN)

    assert view.markdown == "# From file"
    assert view.status == "generated"
    assert view.generated_at is not None


def test_no_report_returns_none_status(service: ReportService) -> None:
    view = service.get_report(RUN)
    assert view.status == "none"
    assert view.markdown == ""
    assert view.generated_at is None


def test_unknown_and_internal_runs_raise(service: ReportService) -> None:
    with pytest.raises(SessionNotFoundError):
        service.get_report("missing_run")
    with pytest.raises(SessionNotFoundError):
        service.get_report("qsess__internal_1")


def test_report_symlink_escaping_workspace_is_refused(
    tmp_path: Path, store: ArtifactStore, service: ReportService
) -> None:
    outside = tmp_path.parent / "outside_report.md"
    outside.write_text("# secret", encoding="utf-8")
    report_file = store.session_dir(PROJECT, RUN) / "report" / "report.md"
    report_file.symlink_to(outside)

    view = service.get_report(RUN)

    assert view.status == "none"
    assert view.markdown == ""
