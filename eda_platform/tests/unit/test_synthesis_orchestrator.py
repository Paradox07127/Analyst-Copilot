from __future__ import annotations

from pathlib import Path

import pytest

from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.synthesis_orchestrator import (
    create_synthesis_brief,
    load_synthesis_briefs,
)
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.investigations import InvestigationRecord, ValidatedFinding
from eda_platform.schemas.questions import QuestionFinding
from eda_platform.schemas.synthesis import SynthesisBrief

PROJECT = "project_demo"
FINDING_RUN = "finding_run"


def _source_with_finding(tmp_path: Path) -> tuple[ArtifactStore, str]:
    """A report-eligible ValidatedFinding plus its record, built directly in the store."""
    store = ArtifactStore(tmp_path / "workspace")
    store.ensure_project(PROJECT, PROJECT)
    store.start_session(PROJECT, FINDING_RUN)
    finding = ValidatedFinding(
        finding_id="finding_trend",
        investigation_id="inv_trend",
        question_id="q_trend",
        question="How do monthly order amounts trend over the year?",
        claim_class="observed",
        findings=[
            QuestionFinding(text="Monthly order amounts rise steadily across the year.")
        ],
        evidence_support="high",
        analytical_reliability="high",
        decision_readiness="medium",
        report_eligible=True,
        report_readiness="eligible",
        report_readiness_reason="The deterministic test fixture is eligible.",
    )
    store.save_artifact(
        Artifact(
            id="finding_trend",
            type=ArtifactType.VALIDATED_FINDING,
            project_id=PROJECT,
            session_id=FINDING_RUN,
            payload=finding.model_dump(mode="json"),
        )
    )
    record = InvestigationRecord(
        record_id="record_trend",
        investigation_id="inv_trend",
        question_id="q_trend",
        status="validated",
        reason_code="ok",
        reason="Validated.",
        next_action="none",
        finding_artifact_id="finding_trend",
    )
    store.save_artifact(
        Artifact(
            id="record_trend",
            type=ArtifactType.INVESTIGATION_RECORD,
            project_id=PROJECT,
            session_id=FINDING_RUN,
            payload=record.model_dump(mode="json"),
        )
    )
    return store, "finding_trend"


def test_synthesis_builds_a_story_bound_to_selected_findings(tmp_path: Path) -> None:
    store, finding_artifact_id = _source_with_finding(tmp_path)
    synthesis = create_synthesis_brief(
        project_id=PROJECT,
        finding_artifact_ids=[finding_artifact_id],
        workspace=store.root,
        business_context="Prioritize operations decisions that reduce avoidable loss.",
        session_id="synthesis_run",
    )
    brief = SynthesisBrief.model_validate(synthesis.artifact.payload)
    assert brief.selected_finding_artifact_ids == [finding_artifact_id]
    assert brief.report_eligible is True
    assert brief.report_readiness in {"eligible", "eligible_with_limitations"}
    assert [beat.title for beat in brief.storyline] == [
        "Decision context",
        "Validated evidence",
        "Decision implication",
        "Limits and next validation",
    ]
    assert all(beat.finding_artifact_ids == [finding_artifact_id] for beat in brief.storyline)

    briefs, warnings = load_synthesis_briefs(
        project_id=PROJECT,
        workspace=store.root,
    )
    assert not warnings
    assert [item.artifact_id for item in briefs] == [synthesis.artifact.id]


def test_synthesis_rejects_a_finding_that_is_not_report_eligible(tmp_path: Path) -> None:
    store, finding_artifact_id = _source_with_finding(tmp_path)
    finding_artifact = store.get_artifact(finding_artifact_id)
    finding_artifact.payload["report_eligible"] = False
    finding_artifact.payload["report_readiness"] = "not_eligible"
    store.save_artifact(finding_artifact)
    with pytest.raises(ValueError, match="report-eligible"):
        create_synthesis_brief(
            project_id=PROJECT,
            finding_artifact_ids=[finding_artifact_id],
            workspace=store.root,
        )


def test_business_context_text_never_enters_a_beat_body(tmp_path: Path) -> None:
    store, finding_artifact_id = _source_with_finding(tmp_path)
    sentinel = "ZZZ_UNVERIFIED_BUSINESS_FRAMING_SENTINEL_44771"
    synthesis = create_synthesis_brief(
        project_id=PROJECT,
        finding_artifact_ids=[finding_artifact_id],
        workspace=store.root,
        business_context=f"Our strategy note says {sentinel} matters most.",
        session_id="synthesis_run",
    )
    brief = SynthesisBrief.model_validate(synthesis.artifact.payload)
    assert sentinel in brief.business_context
    for beat in brief.storyline:
        assert sentinel not in beat.body
    assert sentinel not in brief.decision_context


def test_numbers_in_value_hypothesis_never_enter_a_beat_body(tmp_path: Path) -> None:
    store, finding_artifact_id = _source_with_finding(tmp_path)
    finding_artifact = store.get_artifact(finding_artifact_id)
    # A tampered / LLM-authored value hypothesis carrying an unsupported number
    # must never be rendered as a claim in the decision story.
    finding_artifact.payload["value_hypothesis"] = "This could lift revenue by 987654 dollars."
    store.save_artifact(finding_artifact)
    synthesis = create_synthesis_brief(
        project_id=PROJECT,
        finding_artifact_ids=[finding_artifact_id],
        workspace=store.root,
        session_id="synthesis_run",
    )
    brief = SynthesisBrief.model_validate(synthesis.artifact.payload)
    for beat in brief.storyline:
        assert "987654" not in beat.body
