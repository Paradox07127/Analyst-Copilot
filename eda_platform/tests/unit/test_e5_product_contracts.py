from __future__ import annotations

import pytest

from eda_platform.core.exploration_profiles import (
    EXPLORATION_READ_ONLY_TOOL_NAMES,
    build_read_only_exploration_toolset,
    exploration_budget_profile,
)
from eda_platform.core.exploration_tiers import (
    ANALYSIS_DEPTH_TO_EXPLORATION_TIER,
    exploration_tier_for_analysis_depth,
)


def test_analysis_depth_has_one_exploration_tier_mapping() -> None:
    assert dict(ANALYSIS_DEPTH_TO_EXPLORATION_TIER) == {
        0: "quick",
        1: "standard",
        2: "deep",
        3: "deep",
    }
    assert tuple(
        exploration_tier_for_analysis_depth(depth) for depth in range(4)
    ) == ("quick", "standard", "deep", "deep")


@pytest.mark.parametrize("analysis_depth", (-1, 4, 99))
def test_unknown_analysis_depth_does_not_silently_downgrade(
    analysis_depth: int,
) -> None:
    with pytest.raises(ValueError, match="analysis_depth"):
        exploration_tier_for_analysis_depth(analysis_depth)


def test_exploration_toolset_is_constructed_only_from_the_read_only_allowlist() -> None:
    registered = {
        "inspect_data_catalog": "catalog",
        "run_sql": "sql",
        "apply_cleaning": "mutation",
        "write_artifact": "mutation",
        "delete_dataset": "mutation",
    }

    selected = build_read_only_exploration_toolset(registered)

    assert selected == ("catalog", "sql")
    assert set(exploration_budget_profile("standard").max_tool_calls_by_kind) == set(
        EXPLORATION_READ_ONLY_TOOL_NAMES
    )
    assert {
        "apply_cleaning",
        "write_artifact",
        "delete_dataset",
    }.isdisjoint(EXPLORATION_READ_ONLY_TOOL_NAMES)
