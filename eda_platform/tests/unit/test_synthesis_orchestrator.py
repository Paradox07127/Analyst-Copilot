from __future__ import annotations

from pathlib import Path

import pytest

from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.auto_eda import AutoEDAResult, run_auto_eda
from eda_platform.drivers.investigation_library import load_investigation_library
from eda_platform.drivers.investigation_orchestrator import (
    approve_plan,
    create_investigation_plans,
    execute_investigation_plans,
)
from eda_platform.drivers.synthesis_orchestrator import (
    create_synthesis_brief,
    load_synthesis_briefs,
)
from eda_platform.schemas.artifacts import ArtifactType
from eda_platform.schemas.questions import QuestionCandidateSet
from eda_platform.schemas.synthesis import SynthesisBrief


def _clean_time_series_csv(tmp_path: Path) -> Path:
    path = tmp_path / "orders.csv"
    rows = ["order_id,order_date,amount"]
    rows.extend(
        f"O{index:03d},2026-{index:02d}-01,{100 + index * 10}"
        for index in range(1, 13)
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _source_with_finding(tmp_path: Path) -> tuple[AutoEDAResult, str]:
    source = run_auto_eda(
        [_clean_time_series_csv(tmp_path)],
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
        if item.template_id == "trend" and item.score.quality_risk < 0.3
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
    library = load_investigation_library(
        workspace=str(source.workspace),
        project_id=source.project_id,
    )
    finding = library.findings[0]
    assert finding.finding.report_eligible is True
    return source, finding.artifact_id


def test_synthesis_builds_a_story_bound_to_selected_findings(tmp_path: Path) -> None:
    source, finding_artifact_id = _source_with_finding(tmp_path)
    synthesis = create_synthesis_brief(
        project_id=source.project_id,
        finding_artifact_ids=[finding_artifact_id],
        workspace=source.workspace,
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
        project_id=source.project_id,
        workspace=source.workspace,
    )
    assert not warnings
    assert [item.artifact_id for item in briefs] == [synthesis.artifact.id]


def test_synthesis_rejects_a_finding_that_is_not_report_eligible(tmp_path: Path) -> None:
    source, finding_artifact_id = _source_with_finding(tmp_path)
    store = ArtifactStore(source.workspace)
    finding_artifact = store.get_artifact(finding_artifact_id)
    finding_artifact.payload["report_eligible"] = False
    finding_artifact.payload["report_readiness"] = "not_eligible"
    store.save_artifact(finding_artifact)
    with pytest.raises(ValueError, match="report-eligible"):
        create_synthesis_brief(
            project_id=source.project_id,
            finding_artifact_ids=[finding_artifact_id],
            workspace=source.workspace,
        )


def test_business_context_text_never_enters_a_beat_body(tmp_path: Path) -> None:
    source, finding_artifact_id = _source_with_finding(tmp_path)
    sentinel = "ZZZ_UNVERIFIED_BUSINESS_FRAMING_SENTINEL_44771"
    synthesis = create_synthesis_brief(
        project_id=source.project_id,
        finding_artifact_ids=[finding_artifact_id],
        workspace=source.workspace,
        business_context=f"Our strategy note says {sentinel} matters most.",
        session_id="synthesis_run",
    )
    brief = SynthesisBrief.model_validate(synthesis.artifact.payload)
    assert sentinel in brief.business_context
    for beat in brief.storyline:
        assert sentinel not in beat.body
    assert sentinel not in brief.decision_context


def test_numbers_in_value_hypothesis_never_enter_a_beat_body(tmp_path: Path) -> None:
    source, finding_artifact_id = _source_with_finding(tmp_path)
    store = ArtifactStore(source.workspace)
    finding_artifact = store.get_artifact(finding_artifact_id)
    # A tampered / LLM-authored value hypothesis carrying an unsupported number
    # must never be rendered as a claim in the decision story.
    finding_artifact.payload["value_hypothesis"] = "This could lift revenue by 987654 dollars."
    store.save_artifact(finding_artifact)
    synthesis = create_synthesis_brief(
        project_id=source.project_id,
        finding_artifact_ids=[finding_artifact_id],
        workspace=source.workspace,
        session_id="synthesis_run",
    )
    brief = SynthesisBrief.model_validate(synthesis.artifact.payload)
    for beat in brief.storyline:
        assert "987654" not in beat.body
