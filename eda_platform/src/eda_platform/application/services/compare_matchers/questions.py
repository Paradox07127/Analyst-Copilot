from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .canonical import canonical_json, normalize_text, normalize_values, versioned_key
from .generic import (
    MatchConfidence,
    MatchResult,
    MatchStatus,
    RuleMatch,
    deterministic_one_to_one_match,
)

QUESTION_MATCHER_VERSION = "question-v1"


@dataclass(frozen=True)
class QuestionComparable:
    record_id: str
    question_text: str
    target_datasets: tuple[str, ...] = ()
    lineage_identity: str = ""
    template_id: str = ""
    metric_id: str = ""
    candidate_fingerprint: str = ""
    comparison_payload: dict[str, Any] = field(default_factory=dict)


def match_questions(
    left: list[QuestionComparable],
    right: list[QuestionComparable],
) -> list[MatchResult[QuestionComparable]]:
    return deterministic_one_to_one_match(
        left,
        right,
        identity=lambda question: question.record_id,
        candidate=_question_candidate,
        equivalent=lambda first, second: canonical_json(_content(first))
        == canonical_json(_content(second)),
        version=QUESTION_MATCHER_VERSION,
    )


def _question_candidate(
    left: QuestionComparable,
    right: QuestionComparable,
) -> RuleMatch | None:
    if left.lineage_identity and left.lineage_identity == right.lineage_identity:
        return _rule(
            priority=0,
            namespace="question-lineage",
            value=left.lineage_identity,
            reason="shared lineage question identity",
            confidence=MatchConfidence.EXACT,
            status=MatchStatus.EXACT,
        )

    datasets_match = normalize_values(left.target_datasets) == normalize_values(
        right.target_datasets
    )
    if (
        left.template_id
        and left.metric_id
        and normalize_text(left.template_id) == normalize_text(right.template_id)
        and normalize_text(left.metric_id) == normalize_text(right.metric_id)
        and datasets_match
    ):
        return _rule(
            priority=1,
            namespace="question-template",
            value=(
                normalize_text(left.template_id),
                normalize_text(left.metric_id),
                normalize_values(left.target_datasets),
            ),
            reason="same template, metric, and target datasets",
            confidence=MatchConfidence.HIGH,
            status=MatchStatus.STRONG,
        )

    if (
        left.candidate_fingerprint
        and left.candidate_fingerprint == right.candidate_fingerprint
    ):
        return _rule(
            priority=2,
            namespace="question-fingerprint",
            value=left.candidate_fingerprint,
            reason="same candidate fingerprint",
            confidence=MatchConfidence.HIGH,
            status=MatchStatus.STRONG,
        )

    normalized_question = normalize_text(left.question_text)
    if (
        normalized_question
        and normalized_question == normalize_text(right.question_text)
        and datasets_match
    ):
        return _rule(
            priority=3,
            namespace="question-text",
            value=(normalized_question, normalize_values(left.target_datasets)),
            reason="same normalized question text and target datasets",
            confidence=MatchConfidence.MEDIUM,
            status=MatchStatus.PROBABLE,
        )
    return None


def _rule(
    *,
    priority: int,
    namespace: str,
    value: object,
    reason: str,
    confidence: MatchConfidence,
    status: MatchStatus,
) -> RuleMatch:
    return RuleMatch(
        priority=priority,
        score=1.0,
        match_key=versioned_key(namespace, value),
        reason=reason,
        confidence=confidence,
        status=status,
    )


def _content(question: QuestionComparable) -> object:
    return {
        "question_text": question.question_text,
        "target_datasets": normalize_values(question.target_datasets),
        "template_id": question.template_id,
        "metric_id": question.metric_id,
        "candidate_fingerprint": question.candidate_fingerprint,
        "payload": question.comparison_payload,
    }
