from __future__ import annotations

from pathlib import Path

import pytest

from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.decision_report import create_decision_report
from eda_platform.drivers.investigation_orchestrator import reject_plan
from eda_platform.drivers.knowledge_promotion import build_promotion_candidate
from eda_platform.drivers.synthesis_orchestrator import (
    _source_columns,
    create_synthesis_brief,
)
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.investigations import (
    InvestigationPlan,
    InvestigationRecord,
    ValidatedFinding,
)
from eda_platform.schemas.questions import QuestionFinding
from eda_platform.schemas.synthesis import SynthesisBrief, SynthesisStoryBeat


def _store(tmp_path: Path) -> ArtifactStore:
    store = ArtifactStore(tmp_path / "workspace")
    for project_id in ("project_a", "project_b"):
        store.ensure_project(project_id, project_id)
    return store


def _save(
    store: ArtifactStore,
    *,
    artifact_id: str,
    artifact_type: ArtifactType,
    project_id: str,
    session_id: str,
    payload: dict[str, object],
) -> Artifact:
    store.start_session(project_id, session_id)
    artifact = Artifact(
        id=artifact_id,
        type=artifact_type,
        project_id=project_id,
        session_id=session_id,
        payload=payload,
    )
    store.save_artifact(artifact)
    return artifact


def _finding(question: str) -> ValidatedFinding:
    return ValidatedFinding(
        finding_id=f"finding_{question}",
        investigation_id=f"inv_{question}",
        question_id=f"q_{question}",
        question=question,
        claim_class="observed",
        findings=[QuestionFinding(text=f"{question} is supported.")],
        evidence_support="high",
        analytical_reliability="high",
        decision_readiness="medium",
        report_eligible=True,
        report_readiness="eligible",
        report_readiness_reason="The deterministic test fixture is eligible.",
    )


def _plan(investigation_id: str, question_id: str) -> InvestigationPlan:
    return InvestigationPlan(
        investigation_id=investigation_id,
        source_session_id="source_run",
        question_id=question_id,
        card_version=1,
        candidate_fingerprint=f"fingerprint_{investigation_id}",
        question=f"Question for {investigation_id}",
        target_datasets=["orders.csv"],
        method_family="descriptive",
        method_recipe="compare values",
        allowed_tools=["sql"],
        feasibility="ready",
        status="planned",
        status_reason="Ready.",
    )


def test_promotion_candidate_uses_exact_finding_partition(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _save(
        store,
        artifact_id="shared_finding",
        artifact_type=ArtifactType.VALIDATED_FINDING,
        project_id="project_a",
        session_id="finding_run_a",
        payload=_finding("Project A question").model_dump(mode="json"),
    )
    _save(
        store,
        artifact_id="shared_finding",
        artifact_type=ArtifactType.TABLE,
        project_id="project_a",
        session_id="finding_run_b",
        payload={"column": "wrong_b"},
    )

    candidate = build_promotion_candidate(
        store,
        "project_a",
        "shared_finding",
        session_id="finding_run_a",
    )

    assert candidate.question == "Project A question"


def test_store_rejects_ambiguous_incomplete_artifact_scope(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for project_id, session_id in (
        ("project_a", "run_a"),
        ("project_b", "run_b"),
    ):
        _save(
            store,
            artifact_id="shared_table",
            artifact_type=ArtifactType.TABLE,
            project_id=project_id,
            session_id=session_id,
            payload={"column": project_id},
        )

    with pytest.raises(ValueError, match="ambiguous artifact identity"):
        store.get_artifact("shared_table")
    with pytest.raises(ValueError, match="ambiguous artifact identity"):
        store.artifact_index_row("shared_table")

    exact = store.get_artifact(
        "shared_table",
        project_id="project_a",
        session_id="run_a",
    )
    assert (exact.project_id, exact.session_id) == ("project_a", "run_a")


def test_synthesis_source_columns_use_exact_source_partition(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _save(
        store,
        artifact_id="shared_source",
        artifact_type=ArtifactType.TABLE,
        project_id="project_a",
        session_id="finding_run_a",
        payload={"column": "correct_a"},
    )
    _save(
        store,
        artifact_id="shared_source",
        artifact_type=ArtifactType.TABLE,
        project_id="project_a",
        session_id="finding_run_b",
        payload={"column": "wrong_b"},
    )

    columns = _source_columns(
        store,
        ["shared_source"],
        project_id="project_a",
        artifact_session_ids={"shared_source": "finding_run_a"},
    )

    assert columns == ["correct_a"]


def test_synthesis_selection_uses_exact_finding_run(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for session_id, question in (
        ("finding_run_old", "Old run"),
        ("finding_run_new", "New run"),
    ):
        finding = _finding(question)
        _save(
            store,
            artifact_id="shared_finding",
            artifact_type=ArtifactType.VALIDATED_FINDING,
            project_id="project_a",
            session_id=session_id,
            payload=finding.model_dump(mode="json"),
        )
        record = InvestigationRecord(
            record_id=f"record_{session_id}",
            investigation_id=finding.investigation_id,
            question_id=finding.question_id,
            status="validated",
            reason_code="ok",
            reason="Validated.",
            next_action="none",
            finding_artifact_id="shared_finding",
        )
        _save(
            store,
            artifact_id=f"record_{session_id}",
            artifact_type=ArtifactType.INVESTIGATION_RECORD,
            project_id="project_a",
            session_id=session_id,
            payload=record.model_dump(mode="json"),
        )

    result = create_synthesis_brief(
        project_id="project_a",
        finding_artifact_ids=["shared_finding"],
        finding_session_ids={"shared_finding": "finding_run_new"},
        workspace=store.root,
        session_id="synthesis_run",
    )
    brief = SynthesisBrief.model_validate(result.artifact.payload)

    assert brief.selected_finding_session_ids == {
        "shared_finding": "finding_run_new"
    }
    assert "New run is supported." in brief.headline


def test_investigation_rejection_uses_exact_plan_partition(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _save(
        store,
        artifact_id="shared_plan",
        artifact_type=ArtifactType.INVESTIGATION_PLAN,
        project_id="project_a",
        session_id="plan_run_a",
        payload=_plan("inv_a", "question_a").model_dump(mode="json"),
    )
    _save(
        store,
        artifact_id="shared_plan",
        artifact_type=ArtifactType.INVESTIGATION_PLAN,
        project_id="project_a",
        session_id="plan_run_b",
        payload=_plan("inv_b", "question_b").model_dump(mode="json"),
    )

    artifacts = reject_plan(
        project_id="project_a",
        plan_session_id="plan_run_a",
        plan_id="shared_plan",
        workspace=store.root,
        reason="Reject only project A's plan.",
    )
    record = next(
        InvestigationRecord.model_validate(artifact.payload)
        for artifact in artifacts
        if artifact.type is ArtifactType.INVESTIGATION_RECORD
    )

    assert record.investigation_id == "inv_a"
    assert record.question_id == "question_a"


def test_decision_report_uses_exact_brief_partition(tmp_path: Path) -> None:
    store = _store(tmp_path)
    finding = _save(
        store,
        artifact_id="finding_a",
        artifact_type=ArtifactType.VALIDATED_FINDING,
        project_id="project_a",
        session_id="finding_run_a",
        payload=_finding("Report question").model_dump(mode="json"),
    )
    brief = SynthesisBrief(
        brief_id="brief_a",
        project_id="project_a",
        selected_finding_artifact_ids=[finding.id],
        selected_finding_session_ids={finding.id: finding.session_id},
        decision_context="Choose the supported option.",
        headline="A supported observation is available.",
        storyline=[
            SynthesisStoryBeat(
                title="Evidence",
                body="The finding is ready for review.",
                finding_artifact_ids=[finding.id],
            )
        ],
        report_eligible=True,
        report_readiness="eligible",
    )
    _save(
        store,
        artifact_id="shared_brief",
        artifact_type=ArtifactType.SYNTHESIS_BRIEF,
        project_id="project_a",
        session_id="brief_run_a",
        payload=brief.model_dump(mode="json"),
    )
    _save(
        store,
        artifact_id=finding.id,
        artifact_type=ArtifactType.TABLE,
        project_id="project_a",
        session_id="finding_run_b",
        payload={"column": "wrong_b"},
    )
    _save(
        store,
        artifact_id="shared_brief",
        artifact_type=ArtifactType.TABLE,
        project_id="project_a",
        session_id="brief_run_b",
        payload={"column": "wrong_b"},
    )

    report_id = create_decision_report(
        store,
        project_id="project_a",
        brief_artifact_id="shared_brief",
        brief_session_id="brief_run_a",
    )
    report = store.get_artifact(
        report_id,
        project_id="project_a",
        session_id="brief_run_a",
    )

    assert report.project_id == "project_a"
    assert report.session_id == "brief_run_a"
