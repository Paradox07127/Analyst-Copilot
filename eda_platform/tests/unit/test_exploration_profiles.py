from __future__ import annotations

import pytest

from eda_platform.core.exploration_journal import assert_policy_sealed
from eda_platform.core.exploration_profiles import (
    build_exploration_policy,
    exploration_budget_profile,
)
from eda_platform.schemas.exploration import InsightFamily


@pytest.mark.parametrize("tier", ("quick", "standard", "deep"))
def test_every_exploration_tier_has_non_empty_hard_caps(tier: str) -> None:
    policy = exploration_budget_profile(tier)  # type: ignore[arg-type]
    assert policy.llm.max_requests
    assert policy.llm.max_total_tokens
    assert policy.llm.max_cost_usd
    assert policy.llm.max_wall_seconds
    assert policy.max_successful_tool_calls
    assert policy.max_tool_calls_by_kind["run_open_analysis"]
    assert policy.max_rows_scanned
    assert policy.max_result_cells
    assert policy.idle_timeout_seconds
    assert policy.max_rounds


def test_profiles_increase_monotonically_with_thinking_level() -> None:
    quick = exploration_budget_profile("quick")
    standard = exploration_budget_profile("standard")
    deep = exploration_budget_profile("deep")
    assert quick.llm.max_requests < standard.llm.max_requests < deep.llm.max_requests  # type: ignore[operator]
    assert quick.max_successful_tool_calls < standard.max_successful_tool_calls
    assert standard.max_successful_tool_calls < deep.max_successful_tool_calls
    assert quick.max_rounds < standard.max_rounds < deep.max_rounds


def test_quick_has_narrow_coverage_but_standard_and_deep_have_all_families() -> None:
    quick = build_exploration_policy(
        tier="quick", dataset_scope=("ds_1",), tool_capability_digest="cap_1"
    )
    standard = build_exploration_policy(
        tier="standard", dataset_scope=("ds_1",), tool_capability_digest="cap_1"
    )
    deep = build_exploration_policy(
        tier="deep", dataset_scope=("ds_1",), tool_capability_digest="cap_1"
    )
    assert quick.coverage_targets == (
        InsightFamily.DESCRIPTIVE,
        InsightFamily.DIAGNOSTIC,
        InsightFamily.EXPLORATORY,
    )
    assert set(standard.coverage_targets) == set(InsightFamily)
    assert set(deep.coverage_targets) == set(InsightFamily)
    assert_policy_sealed(quick)
    assert_policy_sealed(standard)
    assert_policy_sealed(deep)


def test_returned_profiles_do_not_share_mutable_tool_cap_maps() -> None:
    first = exploration_budget_profile("quick")
    second = exploration_budget_profile("quick")
    first.max_tool_calls_by_kind["run_sql"] = 999
    assert second.max_tool_calls_by_kind["run_sql"] != 999
