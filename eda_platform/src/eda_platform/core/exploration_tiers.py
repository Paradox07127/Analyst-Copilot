"""Single analysis-depth vocabulary for product exploration tiers."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Literal

ExplorationTier = Literal["quick", "standard", "deep"]

# Settings historically accepted four integer values. E5 exposes only three
# honest product tiers, so the two highest legacy values share the same bounded
# deep profile rather than creating a fourth, differently named budget regime.
ANALYSIS_DEPTH_TO_EXPLORATION_TIER: Final[Mapping[int, ExplorationTier]] = (
    MappingProxyType(
        {
            0: "quick",
            1: "standard",
            2: "deep",
            3: "deep",
        }
    )
)


def exploration_tier_for_analysis_depth(analysis_depth: int) -> ExplorationTier:
    """Map a persisted settings depth to its one bounded exploration profile."""
    if not isinstance(analysis_depth, int) or isinstance(analysis_depth, bool):
        raise ValueError("analysis_depth must be an integer between 0 and 3.")
    try:
        return ANALYSIS_DEPTH_TO_EXPLORATION_TIER[analysis_depth]
    except KeyError as exc:
        allowed = sorted(ANALYSIS_DEPTH_TO_EXPLORATION_TIER)
        raise ValueError(
            f"analysis_depth must be between {allowed[0]} and {allowed[-1]}."
        ) from exc
