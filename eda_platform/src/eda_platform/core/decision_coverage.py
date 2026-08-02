"""Deterministic, read-only assessment of project decision coverage."""

from __future__ import annotations

from typing import TYPE_CHECKING

from eda_platform.core.ids import is_internal_session_id, make_artifact_id
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.decision_coverage import DecisionCoverage
from eda_platform.schemas.investigations import InvestigationRecord, ValidatedFinding
from eda_platform.schemas.questions import QuestionCandidate, QuestionCandidateSet

if TYPE_CHECKING:
    from eda_platform.core.store import ArtifactStore

_TERMINAL_ARTIFACT_TYPES = frozenset(
    {ArtifactType.VALIDATED_FINDING, ArtifactType.INVESTIGATION_RECORD}
)
_COVERAGE_ARTIFACT_TYPES = (
    ArtifactType.QUESTION_CANDIDATE_SET,
    ArtifactType.VALIDATED_FINDING,
    ArtifactType.INVESTIGATION_RECORD,
)
_TOP_CARD_LIMIT = 5


def assess_decision_coverage(store: ArtifactStore, project_id: str) -> DecisionCoverage:
    """Assess coverage from artifacts already persisted across project sessions."""
    candidates_by_id: dict[str, QuestionCandidate] = {}
    terminal_question_ids: set[str] = set()
    validated_findings: list[ValidatedFinding] = []

    for run in store.list_sessions(project_id):
        if is_internal_session_id(run.session_id):
            continue  # macro-loop helper runs are machinery, not project coverage
        artifacts, _warnings = store.list_artifacts_of_types(
            project_id=project_id,
            session_id=run.session_id,
            artifact_types=_COVERAGE_ARTIFACT_TYPES,
        )
        for artifact in artifacts:
            if artifact.type is ArtifactType.QUESTION_CANDIDATE_SET:
                _collect_candidates(artifact, candidates_by_id)
            elif artifact.type in _TERMINAL_ARTIFACT_TYPES:
                terminal = _parse_terminal(artifact)
                if terminal is not None:
                    terminal_question_ids.add(terminal.question_id)
                    if isinstance(terminal, ValidatedFinding):
                        validated_findings.append(terminal)

    top_cards = sorted(
        candidates_by_id.values(),
        key=lambda candidate: (-candidate.score.deterministic_score, candidate.question_id),
    )[:_TOP_CARD_LIMIT]
    uninvestigated = [
        candidate.question_en
        for candidate in top_cards
        if candidate.question_id not in terminal_question_ids
    ]
    top_cards_total = len(top_cards)
    top_cards_terminal = top_cards_total - len(uninvestigated)
    validated_count = len(validated_findings)
    coverage_ready = (
        top_cards_total > 0 and top_cards_terminal == top_cards_total and validated_count >= 1
    )

    return DecisionCoverage(
        project_id=project_id,
        top_cards_total=top_cards_total,
        top_cards_terminal=top_cards_terminal,
        uninvestigated_high_value=uninvestigated[:_TOP_CARD_LIMIT],
        findings_not_eligible=sum(
            1 for finding in validated_findings if not finding.report_eligible
        ),
        validated_findings=validated_count,
        coverage_ready=coverage_ready,
        gaps=_coverage_gaps(
            top_cards_total=top_cards_total,
            uninvestigated_count=len(uninvestigated),
            validated_findings=validated_count,
            coverage_ready=coverage_ready,
        ),
    )


def persist_decision_coverage(store: ArtifactStore, project_id: str, session_id: str) -> str:
    """Assess and persist one idempotent DECISION_COVERAGE artifact per run."""
    coverage = assess_decision_coverage(store, project_id)
    artifact = Artifact(
        id=make_artifact_id(
            "decision_coverage", {"project_id": project_id, "session_id": session_id}
        ),
        type=ArtifactType.DECISION_COVERAGE,
        project_id=project_id,
        session_id=session_id,
        payload=coverage.model_dump(mode="json"),
    )
    store.save_artifact(artifact)
    return artifact.id


def _collect_candidates(artifact: Artifact, candidates_by_id: dict[str, QuestionCandidate]) -> None:
    try:
        candidate_set = QuestionCandidateSet.model_validate(artifact.payload)
    except ValueError:
        return
    for candidate in candidate_set.candidates:
        if candidate.feasibility is None or candidate.feasibility.status not in {
            "ready",
            "constrained",
        }:
            continue
        current = candidates_by_id.get(candidate.question_id)
        if current is None or _candidate_preference(candidate) < _candidate_preference(current):
            candidates_by_id[candidate.question_id] = candidate


def _candidate_preference(candidate: QuestionCandidate) -> tuple[float, str]:
    """Prefer the best score, then stable display text for equal-score duplicates."""
    return (-candidate.score.deterministic_score, candidate.question_en)


def _parse_terminal(
    artifact: Artifact,
) -> ValidatedFinding | InvestigationRecord | None:
    try:
        if artifact.type is ArtifactType.VALIDATED_FINDING:
            return ValidatedFinding.model_validate(artifact.payload)
        return InvestigationRecord.model_validate(artifact.payload)
    except ValueError:
        return None


def _coverage_gaps(
    *,
    top_cards_total: int,
    uninvestigated_count: int,
    validated_findings: int,
    coverage_ready: bool,
) -> list[str]:
    if coverage_ready:
        return []
    gaps: list[str] = []
    if top_cards_total == 0:
        gaps.append("No feasible high-value cards are available to assess yet.")
    elif uninvestigated_count:
        verb = "has" if uninvestigated_count == 1 else "have"
        gaps.append(
            f"{uninvestigated_count} of the {top_cards_total} highest-value cards "
            f"{verb} not been investigated yet."
        )
    if validated_findings == 0:
        gaps.append("At least one validated finding is required before coverage is ready.")
    return gaps[:_TOP_CARD_LIMIT]
