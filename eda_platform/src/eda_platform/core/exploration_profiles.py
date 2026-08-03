"""Versioned, non-empty quick/standard/deep exploration policy profiles."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import Literal

from eda_platform.core.exploration_journal import sealed_policy
from eda_platform.core.exploration_tiers import ExplorationTier
from eda_platform.schemas.exploration import ExplorationPolicy, InsightFamily
from eda_platform.schemas.exploration_budget import (
    ExplorationBudgetPolicy,
    SessionBudgetPolicyModel,
)

EXPLORATION_PROFILE_VERSION = "e4a-experimental-v1"
EXPLORATION_STATISTICAL_POLICY_VERSION = "claim-gates-v1"

# Every entry must be able to emit an EvidenceReceipt: the probe executor
# refuses a successful tool call without one, so a receipt-less capability here
# kills any run whose model calls it. inspect_data_catalog / list_artifacts /
# read_artifact / run_sql / list_saved_skills / run_saved_skill were removed on
# 2026-08-03 for exactly that reason (a real gpt-5.6-terra trial called
# inspect_data_catalog and died). Guarded by
# test_every_certified_exploration_tool_can_produce_an_evidence_receipt.
EXPLORATION_READ_ONLY_TOOL_NAMES = (
    "assess_join_keys",
    "screen_anomalies",
    "run_domain_metrics",
    "recommend_cleaning",
    "run_stat_test",
    "correlate_columns",
    "profile_slice",
    "analyze_time_series",
    "diagnose_missingness",
    "run_baseline_model",
    "run_open_analysis",
)

_ALL_FAMILIES = tuple(InsightFamily)
_QUICK_FAMILIES = (
    InsightFamily.DESCRIPTIVE,
    InsightFamily.DIAGNOSTIC,
    InsightFamily.EXPLORATORY,
)


def build_read_only_exploration_toolset[ToolValue](
    registered_tools: Mapping[str, ToolValue],
) -> tuple[ToolValue, ...]:
    """Construct the exploration inventory from an allowlist, never a denylist.

    Mutating tools are absent from the returned object even when the broader
    application registry contains them. E4b can pass this tuple directly to
    the existing read-only executor once its service composition lands.
    """
    return tuple(
        registered_tools[name]
        for name in EXPLORATION_READ_ONLY_TOOL_NAMES
        if name in registered_tools
    )


def exploration_budget_profile(tier: ExplorationTier) -> ExplorationBudgetPolicy:
    """Return a fresh policy object so nested per-tool caps cannot alias callers."""
    if tier == "quick":
        # Sized from the 2026-08-03 deepseek-v4-flash calibration: reasoning
        # models spend up to ~2.5k completion tokens per call (reasoning is
        # 60%+ of it), probe prompts grow to ~7.5k. 12 requests cover the
        # 3-round x (1 generate + 2-3 probe calls) shape that 3 mandatory
        # probes at batch size 1 actually need; 8 could never finish round 3.
        return _budget(
            requests=12,
            input_tokens=96_000,
            output_tokens=36_000,
            total_tokens=132_000,
            cost="1.50",
            wall_seconds=300,
            protected_requests=1,
            protected_tokens=8_000,
            tool_calls=10,
            per_kind=3,
            open_analysis=1,
            rows=2_000_000,
            cells=100_000,
            idle=60,
            rounds=3,
        )
    if tier == "standard":
        return _budget(
            requests=18,
            input_tokens=144_000,
            output_tokens=54_000,
            total_tokens=198_000,
            cost="5.00",
            wall_seconds=600,
            protected_requests=2,
            protected_tokens=20_000,
            tool_calls=30,
            per_kind=8,
            open_analysis=2,
            rows=10_000_000,
            cells=500_000,
            idle=120,
            rounds=8,
        )
    if tier == "deep":
        return _budget(
            requests=36,
            input_tokens=300_000,
            output_tokens=108_000,
            total_tokens=408_000,
            cost="15.00",
            wall_seconds=1_800,
            protected_requests=3,
            protected_tokens=40_000,
            tool_calls=72,
            per_kind=18,
            open_analysis=4,
            rows=30_000_000,
            cells=1_500_000,
            idle=180,
            rounds=16,
        )
    raise ValueError(f"unknown exploration tier: {tier!r}")


def build_exploration_policy(
    *,
    tier: ExplorationTier,
    dataset_scope: tuple[str, ...],
    tool_capability_digest: str,
    mode: Literal["open", "goal_directed"] = "open",
    goal: str | None = None,
) -> ExplorationPolicy:
    """Build and seal the immutable policy snapshot used by approval and resume."""
    return sealed_policy(
        ExplorationPolicy(
            mode=mode,
            goal=goal,
            dataset_scope=dataset_scope,
            thinking_level=tier,
            coverage_targets=_QUICK_FAMILIES if tier == "quick" else _ALL_FAMILIES,
            budget=exploration_budget_profile(tier),
            scoring_policy_version=EXPLORATION_PROFILE_VERSION,
            statistical_policy_version=EXPLORATION_STATISTICAL_POLICY_VERSION,
            tool_capability_digest=tool_capability_digest,
        )
    )


def _budget(
    *,
    requests: int,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    cost: str,
    wall_seconds: float,
    protected_requests: int,
    protected_tokens: int,
    tool_calls: int,
    per_kind: int,
    open_analysis: int,
    rows: int,
    cells: int,
    idle: float,
    rounds: int,
) -> ExplorationBudgetPolicy:
    tool_caps = {name: per_kind for name in EXPLORATION_READ_ONLY_TOOL_NAMES}
    tool_caps["run_open_analysis"] = open_analysis
    return ExplorationBudgetPolicy(
        llm=SessionBudgetPolicyModel(
            max_requests=requests,
            max_input_tokens=input_tokens,
            max_output_tokens=output_tokens,
            max_total_tokens=total_tokens,
            max_cost_usd=Decimal(cost),
            max_wall_seconds=wall_seconds,
            protected_requests=protected_requests,
            protected_total_tokens=protected_tokens,
            protected_cost_usd=Decimal(cost) / Decimal("10"),
        ),
        max_successful_tool_calls=tool_calls,
        max_tool_calls_by_kind=tool_caps,
        max_rows_scanned=rows,
        max_result_cells=cells,
        idle_timeout_seconds=idle,
        max_rounds=rounds,
    )
