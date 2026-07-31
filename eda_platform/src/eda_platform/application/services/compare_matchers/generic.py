from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from .canonical import CANONICAL_KEY_VERSION, versioned_key

MATCHER_VERSION = "deterministic-v1"


class Change(StrEnum):
    ADDED = "added"
    REMOVED = "removed"
    CHANGED = "changed"
    SAME = "same"


class MatchStatus(StrEnum):
    EXACT = "exact"
    STRONG = "strong"
    PROBABLE = "probable"
    UNMATCHED = "unmatched"


class MatchConfidence(StrEnum):
    EXACT = "exact"
    HIGH = "high"
    MEDIUM = "medium"
    NONE = "none"


@dataclass(frozen=True)
class RuleMatch:
    priority: int
    score: float
    match_key: str
    reason: str
    confidence: MatchConfidence
    status: MatchStatus


@dataclass(frozen=True)
class MatchResult[T]:
    match_key: str
    version: str
    reason: str
    confidence: MatchConfidence
    match_status: MatchStatus
    change: Change
    left: T | None
    right: T | None
    canonicalizer_version: str = CANONICAL_KEY_VERSION
    """Kept separate from `version` on purpose (plan section 17.2): `version`
    is the matching algorithm, this is the key-canonicalization contract. A
    stored key stays interpretable when only one of the two changes."""


@dataclass(frozen=True)
class _Edge[T]:
    left_key: str
    right_key: str
    left: T
    right: T
    rule: RuleMatch

    @property
    def rank(self) -> tuple[int, float]:
        return (self.rule.priority, -self.rule.score)


def deterministic_one_to_one_match[T](
    left_items: Sequence[T],
    right_items: Sequence[T],
    *,
    identity: Callable[[T], str],
    candidate: Callable[[T, T], RuleMatch | None],
    equivalent: Callable[[T, T], bool],
    version: str = MATCHER_VERSION,
) -> list[MatchResult[T]]:
    """Match only candidates that are the unique best choice on both sides."""
    left = _index_items(left_items, identity, side="left")
    right = _index_items(right_items, identity, side="right")
    edges = _candidate_edges(left, right, candidate)
    remaining_left = set(left)
    remaining_right = set(right)
    matched: list[MatchResult[T]] = []

    while True:
        active = [
            edge
            for edge in edges
            if edge.left_key in remaining_left and edge.right_key in remaining_right
        ]
        unique_left = _unique_best(active, key=lambda edge: edge.left_key)
        unique_right = _unique_best(active, key=lambda edge: edge.right_key)
        mutual = [
            edge
            for edge in unique_left.values()
            if unique_right.get(edge.right_key) == edge
        ]
        if not mutual:
            break
        for edge in sorted(mutual, key=_edge_sort_key):
            remaining_left.remove(edge.left_key)
            remaining_right.remove(edge.right_key)
            matched.append(
                MatchResult(
                    match_key=edge.rule.match_key,
                    version=version,
                    reason=edge.rule.reason,
                    confidence=edge.rule.confidence,
                    match_status=edge.rule.status,
                    change=(
                        Change.SAME if equivalent(edge.left, edge.right) else Change.CHANGED
                    ),
                    left=edge.left,
                    right=edge.right,
                )
            )

    results = matched
    for left_key in sorted(remaining_left):
        results.append(
            MatchResult(
                match_key=versioned_key("unmatched-left", left_key),
                version=version,
                reason=_unmatched_reason(edges, left_key=left_key),
                confidence=MatchConfidence.NONE,
                match_status=MatchStatus.UNMATCHED,
                change=Change.REMOVED,
                left=left[left_key],
                right=None,
            )
        )
    for right_key in sorted(remaining_right):
        results.append(
            MatchResult(
                match_key=versioned_key("unmatched-right", right_key),
                version=version,
                reason=_unmatched_reason(edges, right_key=right_key),
                confidence=MatchConfidence.NONE,
                match_status=MatchStatus.UNMATCHED,
                change=Change.ADDED,
                left=None,
                right=right[right_key],
            )
        )
    return sorted(results, key=lambda result: _result_sort_key(result, identity))


def _index_items[T](
    items: Sequence[T],
    identity: Callable[[T], str],
    *,
    side: str,
) -> dict[str, T]:
    indexed: dict[str, T] = {}
    for item in items:
        key = identity(item)
        if not key:
            raise ValueError(f"{side} comparison identity must not be empty")
        if key in indexed:
            raise ValueError(f"duplicate {side} comparison identity: {key}")
        indexed[key] = item
    return indexed


def _candidate_edges[T](
    left: dict[str, T],
    right: dict[str, T],
    candidate: Callable[[T, T], RuleMatch | None],
) -> list[_Edge[T]]:
    edges: list[_Edge[T]] = []
    for left_key in sorted(left):
        for right_key in sorted(right):
            rule = candidate(left[left_key], right[right_key])
            if rule is None:
                continue
            if rule.priority < 0:
                raise ValueError("match priority must be non-negative")
            if not math.isfinite(rule.score):
                raise ValueError("match score must be finite")
            edges.append(
                _Edge(
                    left_key=left_key,
                    right_key=right_key,
                    left=left[left_key],
                    right=right[right_key],
                    rule=rule,
                )
            )
    return edges


def _unique_best[T](
    edges: Iterable[_Edge[T]],
    *,
    key: Callable[[_Edge[T]], str],
) -> dict[str, _Edge[T]]:
    grouped: dict[str, list[_Edge[T]]] = {}
    for edge in edges:
        grouped.setdefault(key(edge), []).append(edge)
    unique: dict[str, _Edge[T]] = {}
    for item_key, item_edges in grouped.items():
        best_rank = min(edge.rank for edge in item_edges)
        best = [edge for edge in item_edges if edge.rank == best_rank]
        if len(best) == 1:
            unique[item_key] = best[0]
    return unique


def _edge_sort_key[T](edge: _Edge[T]) -> tuple[int, float, str, str, str]:
    return (
        edge.rule.priority,
        -edge.rule.score,
        edge.rule.match_key,
        edge.left_key,
        edge.right_key,
    )


def _unmatched_reason[T](
    edges: Iterable[_Edge[T]],
    *,
    left_key: str | None = None,
    right_key: str | None = None,
) -> str:
    has_candidate = any(
        (left_key is not None and edge.left_key == left_key)
        or (right_key is not None and edge.right_key == right_key)
        for edge in edges
    )
    if has_candidate:
        return "ambiguous or non-mutual deterministic candidates"
    return "no deterministic candidate"


def _result_sort_key[T](
    result: MatchResult[T],
    identity: Callable[[T], str],
) -> tuple[str, str, str]:
    left_key = identity(result.left) if result.left is not None else ""
    right_key = identity(result.right) if result.right is not None else ""
    return (result.match_key, left_key, right_key)
