from __future__ import annotations

from pathlib import Path

from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.auto_eda import AutoEDAResult, run_auto_eda
from eda_platform.drivers.investigation_library import load_investigation_library
from eda_platform.drivers.investigation_orchestrator import (
    approve_plan,
    create_investigation_plans,
    execute_investigation_plans,
)
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.investigations import InvestigationRecord
from eda_platform.schemas.questions import QuestionCandidate, QuestionCandidateSet

GOLDEN_DATA = Path(__file__).parents[1] / "golden" / "data"


def _validated_source(tmp_path: Path) -> tuple[AutoEDAResult, str, QuestionCandidate]:
    source = run_auto_eda(
        [GOLDEN_DATA / "ecommerce_orders.csv"],
        workspace=tmp_path / "workspace",
        project_id="project_demo",
        session_id="source_run",
    )
    candidate_artifact = next(
        item
        for item in source.artifacts
        if item.type is ArtifactType.QUESTION_CANDIDATE_SET
    )
    candidates = QuestionCandidateSet.model_validate(candidate_artifact.payload)
    candidate = next(
        item
        for item in candidates.candidates
        if item.origin == "template" and item.sql_template is not None
    )
    planned = create_investigation_plans(
        project_id=source.project_id,
        source_session_id=source.session_id,
        question_ids=[candidate.question_id],
        workspace=source.workspace,
        session_id="plan_run",
    )
    plan_artifact = next(
        item
        for item in planned.artifacts
        if item.type is ArtifactType.INVESTIGATION_PLAN
    )
    approve_plan(
        project_id=source.project_id,
        plan_session_id=planned.session_id,
        plan_id=plan_artifact.id,
        workspace=source.workspace,
    )
    execute_investigation_plans(
        project_id=source.project_id,
        plan_session_id=planned.session_id,
        plan_ids=[plan_artifact.id],
        workspace=source.workspace,
    )
    return source, planned.session_id, candidate


def test_library_uses_only_validated_findings_and_records(tmp_path: Path) -> None:
    source, plan_session_id, candidate = _validated_source(tmp_path)

    library = load_investigation_library(
        workspace=str(source.workspace),
        project_id=source.project_id,
    )
    assert len(library.findings) == 1
    assert library.findings[0].session_id == plan_session_id
    assert library.findings[0].finding.question_id == candidate.question_id
    assert library.findings[0].finding.claim_class == "observed"
    assert len(library.records) == 1
    assert library.records[0].record.status == "validated"
    assert library.records[0].record.finding_artifact_id == library.findings[0].artifact_id
    assert library.records[0].question == candidate.question_en


def test_orphan_finding_without_a_matching_record_is_excluded(tmp_path: Path) -> None:
    source, _, _ = _validated_source(tmp_path)
    library = load_investigation_library(
        workspace=str(source.workspace),
        project_id=source.project_id,
    )
    store = ArtifactStore(source.workspace)
    orphan = Artifact(
        id="finding_orphan",
        type=ArtifactType.VALIDATED_FINDING,
        project_id=source.project_id,
        session_id=source.session_id,
        payload=library.findings[0].finding.model_dump(mode="json"),
    )
    store.save_artifact(orphan)
    refreshed = load_investigation_library(
        workspace=str(source.workspace),
        project_id=source.project_id,
    )
    assert [item.artifact_id for item in refreshed.findings] == [
        library.findings[0].artifact_id
    ]
    assert any(
        "excluding unverified ValidatedFinding artifact finding_orphan" in warning
        for warning in refreshed.warnings
    )


def test_cross_run_record_cannot_authenticate_a_finding(tmp_path: Path) -> None:
    """M1: a validated record from another run must not vouch for a finding."""
    source, _, _ = _validated_source(tmp_path)
    library = load_investigation_library(
        workspace=str(source.workspace),
        project_id=source.project_id,
    )
    finding_item = library.findings[0]
    store = ArtifactStore(source.workspace)

    # Copy the real finding into a fresh run and forge a validated record that
    # names it, but only inside that different run.
    store.start_session(source.project_id, "forged_run")
    forged_finding = Artifact(
        id="finding_forged",
        type=ArtifactType.VALIDATED_FINDING,
        project_id=source.project_id,
        session_id="forged_run",
        payload=finding_item.finding.model_dump(mode="json"),
    )
    store.save_artifact(forged_finding)
    forged_record = InvestigationRecord(
        record_id="irec_forged",
        investigation_id=finding_item.finding.investigation_id,
        question_id=finding_item.finding.question_id,
        status="validated",
        reason_code="finding_validated",
        reason="Forged cross-run authentication attempt.",
        next_action="This record lives in a different run than the finding.",
        finding_artifact_id="finding_forged",
    )
    store.save_artifact(
        Artifact(
            id="irecord_forged",
            type=ArtifactType.INVESTIGATION_RECORD,
            project_id=source.project_id,
            session_id=source.session_id,  # different run than the forged finding
            payload=forged_record.model_dump(mode="json"),
        )
    )
    refreshed = load_investigation_library(
        workspace=str(source.workspace),
        project_id=source.project_id,
    )
    assert "finding_forged" not in {item.artifact_id for item in refreshed.findings}
    assert any(
        "excluding unverified ValidatedFinding artifact finding_forged" in warning
        for warning in refreshed.warnings
    )
