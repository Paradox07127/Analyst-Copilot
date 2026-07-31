"""Derive one publication state from persisted artifacts, without model calls."""

from __future__ import annotations

from collections.abc import Mapping

from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.decision_report import DecisionReport
from eda_platform.schemas.investigations import ValidatedFinding
from eda_platform.schemas.publication import (
    PublicationCondition,
    PublicationFreshness,
    PublicationReadiness,
    PublicationState,
)
from eda_platform.schemas.questions import QuestionExecutionResult


def derive_publication_state(
    artifacts: list[Artifact],
    *,
    decision_report_freshness: Mapping[str, PublicationFreshness] | None = None,
) -> PublicationState:
    """Project execution/investigation/report artifacts -> consistent readiness."""
    answered = abstained = failed = automated_findings = exploratory = 0
    validated = eligible = 0
    decision_report_artifacts: list[Artifact] = []
    technical_status: str | None = None

    for artifact in artifacts:
        if artifact.type is ArtifactType.QUESTION_EXECUTION_RESULT:
            try:
                result = QuestionExecutionResult.model_validate(artifact.payload)
            except ValueError:
                continue
            outcome = result.outcome or ("answered" if result.status == "succeeded" else "failed")
            if outcome == "answered":
                answered += 1
                automated_findings += len(result.findings)
                exploratory += int(result.exploratory)
            elif outcome == "abstained":
                abstained += 1
            elif outcome == "failed":
                failed += 1
        elif artifact.type is ArtifactType.VALIDATED_FINDING:
            try:
                finding = ValidatedFinding.model_validate(artifact.payload)
            except ValueError:
                continue
            validated += 1
            eligible += int(finding.report_eligible)
        elif artifact.type is ArtifactType.DECISION_REPORT:
            try:
                DecisionReport.model_validate(artifact.payload)
            except ValueError:
                continue
            decision_report_artifacts.append(artifact)
        elif artifact.type is ArtifactType.REPORT_BUNDLE:
            status = artifact.payload.get("status")
            if isinstance(status, str):
                technical_status = status

    decision_reports = len(decision_report_artifacts)
    publication_freshness = _publication_freshness(
        decision_report_artifacts, decision_report_freshness
    )
    readiness = _readiness(
        answered_questions=answered,
        validated_findings=validated,
        eligible_findings=eligible,
        decision_reports=decision_reports,
    )
    return PublicationState(
        readiness=readiness,
        answered_questions=answered,
        abstained_questions=abstained,
        failed_questions=failed,
        automated_findings=automated_findings,
        exploratory_answers=exploratory,
        validated_findings=validated,
        report_eligible_findings=eligible,
        decision_reports=decision_reports,
        technical_report_status=technical_status,
        publication_freshness=publication_freshness,
        conditions=_conditions(
            qexec_count=answered + abstained + failed,
            answered_questions=answered,
            automated_findings=automated_findings,
            validated_findings=validated,
            eligible_findings=eligible,
            decision_reports=decision_reports,
            technical_status=technical_status,
            publication_freshness=publication_freshness,
        ),
        reasons=_reasons(
            readiness,
            answered_questions=answered,
            automated_findings=automated_findings,
            validated_findings=validated,
            eligible_findings=eligible,
            publication_freshness=publication_freshness,
        ),
    )


def _readiness(
    *,
    answered_questions: int,
    validated_findings: int,
    eligible_findings: int,
    decision_reports: int,
) -> PublicationReadiness:
    if decision_reports:
        return "published"
    if eligible_findings:
        return "publication_ready"
    if validated_findings:
        return "investigation_validated"
    if answered_questions:
        return "analysis_available"
    return "draft"


def _reasons(
    readiness: PublicationReadiness,
    *,
    answered_questions: int,
    automated_findings: int,
    validated_findings: int,
    eligible_findings: int,
    publication_freshness: PublicationFreshness,
) -> list[str]:
    if publication_freshness == "stale":
        return [
            "A decision report exists, but at least one source finding is stale; "
            "re-run the affected investigation before reuse."
        ]
    if publication_freshness == "unverifiable":
        return [
            "A decision report exists, but its source freshness cannot be verified; "
            "treat it as blocked for current reuse."
        ]
    if readiness == "analysis_available":
        return [
            f"{answered_questions} automated analysis answer(s) and {automated_findings} "
            "finding(s) are available, but none passed the approved investigation gates yet."
        ]
    if readiness == "investigation_validated":
        return [
            f"{validated_findings} finding(s) passed investigation gates, but none is "
            "report-eligible."
        ]
    if readiness == "publication_ready":
        return [f"{eligible_findings} validated finding(s) are ready for decision synthesis."]
    if readiness == "published":
        return ["A decision report has been assembled from validated findings."]
    return ["No answered analysis with evidence is available yet."]


def _conditions(
    *,
    qexec_count: int,
    answered_questions: int,
    automated_findings: int,
    validated_findings: int,
    eligible_findings: int,
    decision_reports: int,
    technical_status: str | None,
    publication_freshness: PublicationFreshness,
) -> list[PublicationCondition]:
    answerability = PublicationCondition(
        type="answerability",
        status="true" if answered_questions else "false" if qexec_count else "unknown",
        reason=(
            "answers_with_evidence"
            if answered_questions
            else "no_answered_findings"
            if qexec_count
            else "not_executed"
        ),
        message=(
            f"{answered_questions} answer(s) passed result contracts and produced "
            f"{automated_findings} finding(s)."
        ),
    )
    investigation = PublicationCondition(
        type="investigation",
        status=("true" if validated_findings else "false" if answered_questions else "unknown"),
        reason=(
            "validated_findings_present"
            if validated_findings
            else "approved_investigation_required"
            if answered_questions
            else "no_analysis_to_investigate"
        ),
        message=f"{validated_findings} finding(s) passed investigation gates.",
    )
    report_gate = PublicationCondition(
        type="report_gate",
        status=(
            "true"
            if technical_status == "validated"
            else "false"
            if technical_status in {"needs_revision", "blocked_for_review"}
            else "unknown"
        ),
        reason=technical_status or "no_report_audit",
        message="Technical report validation is separate from decision readiness.",
    )
    publication = PublicationCondition(
        type="publication",
        status="true" if decision_reports else "false" if eligible_findings else "unknown",
        reason=(
            "decision_report_present"
            if decision_reports
            else "synthesis_required"
            if eligible_findings
            else "no_report_eligible_findings"
        ),
        message=f"{decision_reports} decision report(s) exist.",
    )
    freshness = PublicationCondition(
        type="freshness",
        status=(
            "true"
            if publication_freshness == "fresh"
            else "false"
            if publication_freshness == "stale"
            else "unknown"
        ),
        reason=publication_freshness,
        message="Published output is reusable only while every source finding is fresh.",
    )
    return [answerability, investigation, report_gate, publication, freshness]


def _publication_freshness(
    reports: list[Artifact],
    statuses: Mapping[str, PublicationFreshness] | None,
) -> PublicationFreshness:
    if not reports:
        return "not_applicable"
    if statuses is None:
        return "unknown"
    latest = max(reports, key=lambda artifact: artifact.created_at)
    return statuses.get(latest.id, "unknown")
