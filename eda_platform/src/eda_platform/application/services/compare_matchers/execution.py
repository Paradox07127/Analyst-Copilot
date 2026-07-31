from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .canonical import canonical_json, execution_logical_span_key
from .generic import (
    MatchConfidence,
    MatchResult,
    MatchStatus,
    RuleMatch,
    deterministic_one_to_one_match,
)

EXECUTION_MATCHER_VERSION = "execution-v1"


@dataclass(frozen=True)
class ExecutionComparable:
    record_id: str
    parent_match_key: str
    span_kind: str
    operation_name: str
    originating_key: str = ""
    tool_name: str = ""
    occurrence_index: int = 0
    model: str = ""
    model_config: dict[str, Any] = field(default_factory=dict)
    comparison_payload: dict[str, Any] = field(default_factory=dict)

    @property
    def logical_span_key(self) -> str:
        return execution_logical_span_key(
            parent_match_key=self.parent_match_key,
            span_kind=self.span_kind,
            operation_name=self.operation_name,
            originating_key=self.originating_key,
            tool_name=self.tool_name,
            occurrence_index=self.occurrence_index,
        )


def match_execution_records(
    left: list[ExecutionComparable],
    right: list[ExecutionComparable],
) -> list[MatchResult[ExecutionComparable]]:
    return deterministic_one_to_one_match(
        left,
        right,
        identity=lambda record: record.record_id,
        candidate=_execution_candidate,
        equivalent=lambda first, second: canonical_json(_content(first))
        == canonical_json(_content(second)),
        version=EXECUTION_MATCHER_VERSION,
    )


def _execution_candidate(
    left: ExecutionComparable,
    right: ExecutionComparable,
) -> RuleMatch | None:
    if left.logical_span_key != right.logical_span_key:
        return None
    return RuleMatch(
        priority=0,
        score=1.0,
        match_key=left.logical_span_key,
        reason="same logical execution span",
        confidence=MatchConfidence.EXACT,
        status=MatchStatus.EXACT,
    )


def _content(record: ExecutionComparable) -> object:
    return {
        "logical_span_key": record.logical_span_key,
        "model": record.model,
        "model_config": record.model_config,
        "payload": record.comparison_payload,
    }
