from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.investigation_library import load_investigation_library
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.investigations import (
    InvestigationPlan,
    InvestigationRecord,
    ValidatedFinding,
)
from eda_platform.schemas.questions import QuestionFinding

PROJECT = "project_demo"
SOURCE_RUN = "source_run"
PLAN_RUN = "plan_run"
QUESTION_ID = "q_orders"
QUESTION_TEXT = "Which product category drives revenue?"


def _validated_source(tmp_path: Path) -> tuple[SimpleNamespace, str, SimpleNamespace]:
    """A validated finding, its record, and its plan, built directly in the store."""
    store = ArtifactStore(tmp_path / "workspace")
    store.ensure_project(PROJECT, PROJECT)
    store.start_session(PROJECT, SOURCE_RUN)
    store.start_session(PROJECT, PLAN_RUN)
    plan = InvestigationPlan(
        investigation_id="inv_orders",
        source_session_id=SOURCE_RUN,
        question_id=QUESTION_ID,
        card_version=1,
        candidate_fingerprint="fingerprint_orders",
        question=QUESTION_TEXT,
        target_datasets=["orders.csv"],
        method_family="descriptive",
        method_recipe="compare values",
        allowed_tools=["sql"],
        feasibility="ready",
        status="planned",
        status_reason="Ready.",
    )
    finding = ValidatedFinding(
        finding_id="finding_orders",
        investigation_id="inv_orders",
        question_id=QUESTION_ID,
        question=QUESTION_TEXT,
        claim_class="observed",
        findings=[QuestionFinding(text="Electronics leads revenue across the year.")],
        evidence_support="high",
        analytical_reliability="high",
        decision_readiness="medium",
        report_eligible=True,
        report_readiness="eligible",
        report_readiness_reason="The deterministic test fixture is eligible.",
    )
    record = InvestigationRecord(
        record_id="irec_orders",
        investigation_id="inv_orders",
        question_id=QUESTION_ID,
        status="validated",
        reason_code="finding_validated",
        reason="Validated.",
        next_action="none",
        finding_artifact_id="finding_1",
    )
    for artifact_id, artifact_type, payload in (
        ("plan_1", ArtifactType.INVESTIGATION_PLAN, plan.model_dump(mode="json")),
        ("finding_1", ArtifactType.VALIDATED_FINDING, finding.model_dump(mode="json")),
        ("record_1", ArtifactType.INVESTIGATION_RECORD, record.model_dump(mode="json")),
    ):
        store.save_artifact(
            Artifact(
                id=artifact_id,
                type=artifact_type,
                project_id=PROJECT,
                session_id=PLAN_RUN,
                payload=payload,
            )
        )
    source = SimpleNamespace(
        workspace=store.root, project_id=PROJECT, session_id=SOURCE_RUN
    )
    candidate = SimpleNamespace(question_id=QUESTION_ID, question_en=QUESTION_TEXT)
    return source, PLAN_RUN, candidate


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
