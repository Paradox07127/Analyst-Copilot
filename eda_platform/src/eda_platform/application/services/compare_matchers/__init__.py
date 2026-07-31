from .artifacts import ArtifactComparable, match_artifacts
from .canonical import (
    artifact_logical_key,
    artifact_parent_logical_key,
    execution_logical_span_key,
    normalize_text,
)
from .execution import ExecutionComparable, match_execution_records
from .generic import (
    Change,
    MatchConfidence,
    MatchResult,
    MatchStatus,
    RuleMatch,
    deterministic_one_to_one_match,
)
from .questions import QuestionComparable, match_questions
from .reports import (
    ReportClaimComparable,
    ReportSectionComparable,
    match_report_claims,
    match_report_sections,
)

__all__ = [
    "ArtifactComparable",
    "Change",
    "ExecutionComparable",
    "MatchConfidence",
    "MatchResult",
    "MatchStatus",
    "QuestionComparable",
    "ReportClaimComparable",
    "ReportSectionComparable",
    "RuleMatch",
    "artifact_logical_key",
    "artifact_parent_logical_key",
    "deterministic_one_to_one_match",
    "execution_logical_span_key",
    "match_artifacts",
    "match_execution_records",
    "match_questions",
    "match_report_claims",
    "match_report_sections",
    "normalize_text",
]
