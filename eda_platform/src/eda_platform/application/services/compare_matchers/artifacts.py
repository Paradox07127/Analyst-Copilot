from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .canonical import artifact_logical_key, canonical_json
from .generic import (
    MatchConfidence,
    MatchResult,
    MatchStatus,
    RuleMatch,
    deterministic_one_to_one_match,
)

ARTIFACT_MATCHER_VERSION = "artifact-v1"


@dataclass(frozen=True)
class ArtifactComparable:
    record_id: str
    artifact_type: str
    stable_identity: str
    parent_logical_keys: tuple[str, ...] = ()
    raw_parent_ids: tuple[str, ...] = ()
    """Deliberately never read. Carrying it lets
    `test_artifact_matching_uses_logical_parents_not_raw_parent_ids` feed two
    sides conflicting raw ids and prove matching ignores them; content ids are
    not unique across sessions, so matching on them would cross-contaminate."""
    occurrence_index: int = 0
    comparison_payload: dict[str, Any] = field(default_factory=dict)

    @property
    def logical_key(self) -> str:
        return artifact_logical_key(
            artifact_type=self.artifact_type,
            stable_identity=self.stable_identity,
            parent_match_keys=self.parent_logical_keys,
            occurrence_index=self.occurrence_index,
        )


def match_artifacts(
    left: list[ArtifactComparable],
    right: list[ArtifactComparable],
) -> list[MatchResult[ArtifactComparable]]:
    return deterministic_one_to_one_match(
        left,
        right,
        identity=lambda artifact: artifact.record_id,
        candidate=_artifact_candidate,
        equivalent=lambda first, second: canonical_json(_content(first))
        == canonical_json(_content(second)),
        version=ARTIFACT_MATCHER_VERSION,
    )


def _artifact_candidate(
    left: ArtifactComparable,
    right: ArtifactComparable,
) -> RuleMatch | None:
    if left.logical_key != right.logical_key:
        return None
    return RuleMatch(
        priority=0,
        score=1.0,
        match_key=left.logical_key,
        reason="same artifact identity and matched logical parentage",
        confidence=MatchConfidence.HIGH,
        status=MatchStatus.STRONG,
    )


def _content(artifact: ArtifactComparable) -> object:
    return {
        "logical_key": artifact.logical_key,
        "payload": artifact.comparison_payload,
    }
