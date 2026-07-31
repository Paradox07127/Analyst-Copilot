from __future__ import annotations

from eda_platform.core.publication_state import derive_publication_state
from eda_platform.schemas.artifacts import Artifact, ArtifactType


def _artifact(artifact_id: str, artifact_type: ArtifactType, payload: dict) -> Artifact:
    return Artifact(
        id=artifact_id,
        type=artifact_type,
        project_id="project_demo",
        session_id="run_demo",
        payload=payload,
    )


def _answered_qexec(*, exploratory: bool = False, with_finding: bool = True) -> Artifact:
    return _artifact(
        "qexec_answered",
        ArtifactType.QUESTION_EXECUTION_RESULT,
        {
            "question_id": "q1",
            "question": "What is revenue?",
            "origin": "llm" if exploratory else "template",
            "status": "succeeded",
            "findings": (
                [{"text": "Revenue is 10.", "exploratory": exploratory}] if with_finding else []
            ),
            "exploratory": exploratory,
        },
    )


def _validated_finding(*, eligible: bool) -> Artifact:
    return _artifact(
        "finding_1",
        ArtifactType.VALIDATED_FINDING,
        {
            "finding_id": "finding_1",
            "investigation_id": "inv_1",
            "question_id": "q1",
            "question": "What is revenue?",
            "claim_class": "observed",
            "findings": [{"text": "Revenue is 10."}],
            "evidence_support": "high",
            "analytical_reliability": "high",
            "decision_readiness": "high",
            "report_eligible": eligible,
            "report_readiness": "eligible" if eligible else "not_eligible",
            "report_readiness_reason": "Evidence passed." if eligible else "Needs context.",
        },
    )


def _decision_report() -> Artifact:
    return _artifact(
        "decision",
        ArtifactType.DECISION_REPORT,
        {
            "report_id": "report_1",
            "brief_id": "brief_1",
            "project_id": "project_demo",
            "title": "Decision Report",
            "scqa": {
                "situation": "Revenue was reviewed.",
                "complication": "Evidence is bounded.",
                "question": "What should change?",
                "answer": "Review the validated result.",
            },
            "sections": [{"title": "Finding", "body": "Revenue is 10."}],
            "report_readiness": "eligible",
            "source_finding_artifact_ids": ["finding_1"],
        },
    )


def _condition(state, condition_type: str):  # noqa: ANN001 - compact test helper
    return next(item for item in state.conditions if item.type == condition_type)


def test_automated_answer_is_analysis_not_validated_investigation() -> None:
    state = derive_publication_state([_answered_qexec(exploratory=True)])

    assert state.readiness == "analysis_available"
    assert state.answered_questions == 1
    assert state.automated_findings == 1
    assert state.exploratory_answers == 1
    assert state.validated_findings == 0
    assert _condition(state, "answerability").status == "true"
    assert _condition(state, "investigation").status == "false"


def test_answer_without_generated_finding_still_exposes_analysis() -> None:
    state = derive_publication_state([_answered_qexec(with_finding=False)])

    assert state.readiness == "analysis_available"
    assert state.answered_questions == 1
    assert state.automated_findings == 0
    assert _condition(state, "answerability").status == "true"
    assert _condition(state, "investigation").status == "false"


def test_technical_report_validation_does_not_imply_decision_readiness() -> None:
    report = _artifact("bundle", ArtifactType.REPORT_BUNDLE, {"status": "validated"})

    state = derive_publication_state([_answered_qexec(), report])

    assert state.technical_report_status == "validated"
    assert _condition(state, "report_gate").status == "true"
    assert state.readiness == "analysis_available"


def test_eligible_investigation_advances_to_publication_ready() -> None:
    state = derive_publication_state([_answered_qexec(), _validated_finding(eligible=True)])

    assert state.readiness == "publication_ready"
    assert state.validated_findings == 1
    assert state.report_eligible_findings == 1
    assert _condition(state, "investigation").status == "true"
    assert _condition(state, "publication").status == "false"


def test_decision_report_is_published_without_erasing_conditions() -> None:
    decision_report = _decision_report()
    state = derive_publication_state(
        [_answered_qexec(), _validated_finding(eligible=True), decision_report]
    )

    assert state.readiness == "published"
    assert _condition(state, "publication").status == "true"
    assert _condition(state, "investigation").status == "true"
    assert _condition(state, "freshness").status == "unknown"


def test_stale_report_remains_published_history_but_is_not_currently_reusable() -> None:
    decision_report = _decision_report()
    state = derive_publication_state(
        [_answered_qexec(), _validated_finding(eligible=True), decision_report],
        decision_report_freshness={decision_report.id: "stale"},
    )

    assert state.readiness == "published"
    assert state.publication_freshness == "stale"
    assert _condition(state, "publication").status == "true"
    assert _condition(state, "freshness").status == "false"
    assert "stale" in state.reasons[0]


def test_unverifiable_report_freshness_is_unknown_not_false() -> None:
    decision_report = _decision_report()
    state = derive_publication_state(
        [decision_report],
        decision_report_freshness={decision_report.id: "unverifiable"},
    )

    assert state.publication_freshness == "unverifiable"
    assert _condition(state, "freshness").status == "unknown"


def test_invalid_decision_report_does_not_advance_publication() -> None:
    invalid = _artifact("invalid", ArtifactType.DECISION_REPORT, {})

    state = derive_publication_state([invalid])

    assert state.readiness == "draft"
    assert state.decision_reports == 0
