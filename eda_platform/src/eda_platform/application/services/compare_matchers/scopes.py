from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .canonical import canonical_json, versioned_key
from .generic import (
    MatchConfidence,
    MatchResult,
    MatchStatus,
    RuleMatch,
    deterministic_one_to_one_match,
)

SCOPE_RECORD_MATCHER_VERSION = "scope-record-v1"


@dataclass(frozen=True)
class ScopeComparable:
    """A bounded semantic record used by Analysis and Findings.

    ``stable_key`` is assembled by the scope extractor from domain fields; raw
    artifact/session ids are intentionally excluded so independent runs can
    match. ``comparison_payload`` contains only allowlisted bounded values.
    """

    record_id: str
    stable_key: str
    comparison_payload: dict[str, Any] = field(default_factory=dict)


def match_scope_records(
    left: list[ScopeComparable],
    right: list[ScopeComparable],
) -> list[MatchResult[ScopeComparable]]:
    return deterministic_one_to_one_match(
        left,
        right,
        identity=lambda record: record.record_id,
        candidate=_candidate,
        equivalent=lambda first, second: (
            canonical_json(first.comparison_payload) == canonical_json(second.comparison_payload)
        ),
        version=SCOPE_RECORD_MATCHER_VERSION,
    )


def _candidate(left: ScopeComparable, right: ScopeComparable) -> RuleMatch | None:
    if not left.stable_key or left.stable_key != right.stable_key:
        return None
    return RuleMatch(
        priority=0,
        score=1.0,
        match_key=versioned_key("scope-record", left.stable_key),
        reason="same deterministic scope identity",
        confidence=MatchConfidence.HIGH,
        status=MatchStatus.STRONG,
    )
