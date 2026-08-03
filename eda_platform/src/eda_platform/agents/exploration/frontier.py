"""Deterministic HypothesisCard frontier state machine (E4a §7.1)."""

from __future__ import annotations

from dataclasses import dataclass, replace

from eda_platform.agents.exploration.candidates import CandidateSeed, HypothesisStatus

_ALLOWED_TRANSITIONS: dict[HypothesisStatus, frozenset[HypothesisStatus]] = {
    "proposed": frozenset(
        {
            "admitted",
            "rejected_duplicate",
            "rejected_infeasible",
            "rejected_policy",
        }
    ),
    "admitted": frozenset(
        {"running", "rejected_duplicate", "rejected_infeasible", "rejected_policy"}
    ),
    "running": frozenset({"supported", "refuted", "inconclusive"}),
    "supported": frozenset(),
    "refuted": frozenset(),
    "inconclusive": frozenset(),
    "rejected_duplicate": frozenset(),
    "rejected_infeasible": frozenset(),
    "rejected_policy": frozenset(),
}


class FrontierTransitionError(ValueError):
    """Raised when a transition would violate the documented state machine."""


@dataclass(frozen=True)
class FrontierSnapshot:
    candidates: tuple[CandidateSeed, ...]
    hypothesis_fingerprints: tuple[str, ...]


class Frontier:
    """In-memory state with snapshots suitable for content-addressed artifacts."""

    def __init__(self, candidates: tuple[CandidateSeed, ...] = ()) -> None:
        self._by_id: dict[str, CandidateSeed] = {}
        self._by_fingerprint: dict[str, str] = {}
        for candidate in candidates:
            self.add(candidate)

    def add(self, candidate: CandidateSeed) -> CandidateSeed:
        """Add a proposed candidate or return a deterministic duplicate rejection."""
        existing_id = self._by_fingerprint.get(candidate.hypothesis_fingerprint)
        if existing_id is not None:
            return replace(candidate, status="rejected_duplicate")
        if candidate.hypothesis_id in self._by_id:
            raise FrontierTransitionError(
                f"hypothesis id {candidate.hypothesis_id!r} already exists "
                "with another fingerprint."
            )
        self._by_id[candidate.hypothesis_id] = candidate
        self._by_fingerprint[candidate.hypothesis_fingerprint] = candidate.hypothesis_id
        return candidate

    def get(self, hypothesis_id: str) -> CandidateSeed:
        try:
            return self._by_id[hypothesis_id]
        except KeyError as exc:
            raise FrontierTransitionError(
                f"unknown hypothesis id {hypothesis_id!r}."
            ) from exc

    def transition(
        self, hypothesis_id: str, status: HypothesisStatus
    ) -> CandidateSeed:
        current = self.get(hypothesis_id)
        if status not in _ALLOWED_TRANSITIONS[current.status]:
            raise FrontierTransitionError(
                f"invalid hypothesis transition {current.status!r} -> {status!r}."
            )
        updated = replace(current, status=status)
        self._by_id[hypothesis_id] = updated
        return updated

    def update_priority(self, hypothesis_id: str, priority: float) -> CandidateSeed:
        current = self.get(hypothesis_id)
        updated = replace(current, priority=priority)
        self._by_id[hypothesis_id] = updated
        return updated

    def snapshot(self) -> FrontierSnapshot:
        ordered = tuple(
            sorted(
                self._by_id.values(),
                key=lambda item: (item.sequence_index, item.hypothesis_fingerprint),
            )
        )
        return FrontierSnapshot(
            candidates=ordered,
            hypothesis_fingerprints=tuple(
                sorted(self._by_fingerprint)
            ),
        )

    def highest_priority(
        self, *, statuses: frozenset[HypothesisStatus] = frozenset({"admitted"})
    ) -> float | None:
        priorities = [
            candidate.priority
            for candidate in self._by_id.values()
            if candidate.status in statuses
        ]
        return max(priorities, default=None)
