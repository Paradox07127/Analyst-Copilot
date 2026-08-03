"""Checker v2: two-pass matching, origin attribution, and harness seeding.

Pass 1 is unchanged predicate identity; pass 2 credits semantic matches
(tool + columns + typed support + proof edge, gated by alternate_operators
and required_fact_values). Older checker versions must recompute exactly as
before — the version string is the switch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.exploration_baseline.run_e4a_trials import (  # noqa: E402
    DATASET_ID,
    load_planted_bundle,
    seed_profile_artifacts,
)

from eda_platform.agents.data_tools import (  # noqa: E402
    DataToolContext,
    RecommendCleaningArguments,
    RunDomainMetricsArguments,
    build_data_tools,
)
from eda_platform.agents.exploration.candidates import (  # noqa: E402
    CandidateSeed,
    candidate_seed,
)
from eda_platform.agents.exploration.workflow import (  # noqa: E402
    ExplorationWorkflowState,
)
from eda_platform.agents.receipts import build_receipt  # noqa: E402
from eda_platform.drivers.exploration_evidence_issuer import (  # noqa: E402
    E4A_CHECKER_VERSION_V2,
    E4aExpectedStructure,
    E4aGroundTruthFixture,
    _recompute_checker,
)
from eda_platform.schemas.exploration import (  # noqa: E402
    ExplorationLoopEvent,
    ExplorationLoopState,
    InsightFamily,
    RoundSettledEvent,
    RoundStartedEvent,
)
from eda_platform.schemas.hypotheses import (  # noqa: E402
    HypothesisPredicate,
    HypothesisProposal,
)
from eda_platform.schemas.insights import InsightProof, InsightRecord  # noqa: E402
from eda_platform.schemas.receipts import (  # noqa: E402
    EvidenceReceipt,
    ReceiptFact,
    ReceiptMethod,
    ReceiptScope,
    ReceiptStatistics,
)
from eda_platform.tools.sql_runner import build_catalog  # noqa: E402

_XPL = "xpl-checker-v2"
_WITNESS = "witness-checker-v2"

_REGION_PREDICATE = HypothesisPredicate(
    metric="revenue", operator="differs", left_operand="region", right_operand="groups"
)
_TREND_PREDICATE = HypothesisPredicate(
    metric="revenue", operator="greater_than", right_operand="order_date"
)

_REGION_STRUCTURE = E4aExpectedStructure(
    structure_id="region_difference",
    target_metric="region_difference_recall",
    tool_names=("run_stat_test",),
    required_columns=("region", "revenue"),
    predicate=_REGION_PREDICATE,
)
_TREND_STRUCTURE = E4aExpectedStructure(
    structure_id="planted_trend_revenue",
    target_metric="trend_recall",
    tool_names=("analyze_time_series",),
    required_columns=("order_date", "revenue"),
    predicate=_TREND_PREDICATE,
    alternate_operators=("has_spike",),
    required_fact_values=(("trend_direction", "increasing"),),
)
_FIXTURE = E4aGroundTruthFixture(
    item_id="planted_retail_v1",
    expected_structures=(_REGION_STRUCTURE, _TREND_STRUCTURE),
)


def _candidate(
    predicate: HypothesisPredicate,
    *,
    columns: tuple[str, ...],
    method_family: str,
    probe_kind: str,
    origin: str,
) -> CandidateSeed:
    proposal = HypothesisProposal(
        statement=f"{probe_kind} via {predicate.operator}",
        rationale="Checker fixture.",
        expected_evidence="A typed supporting receipt.",
        falsification_conditions=("No detectable structure.",),
        family=InsightFamily.DIAGNOSTIC,
        method_family=method_family,
        dataset_ids=("ds-1",),
        columns=columns,
        probe_kind=probe_kind,
        predicate=predicate,
    )
    return candidate_seed(proposal, sequence_index=1, origin=origin)  # type: ignore[arg-type]


def _supporting_receipt(
    name: str,
    *,
    tool_name: str,
    columns: tuple[str, ...],
    hypothesis_id: str,
    extra_facts: tuple[ReceiptFact, ...] = (),
) -> EvidenceReceipt:
    return build_receipt(
        tool_call_id=f"call-{name}",
        tool_name=tool_name,
        tool_version="1",
        arguments={"call": name},
        raw_output={"rows": []},
        artifact_ids=(),
        result_count=1,
        scope=ReceiptScope(
            dataset_ids=("ds-1",), columns=columns, scope_resolution="explicit"
        ),
        facts=(
            ReceiptFact(
                fact_id="effect_size",
                name="effect_size",
                value=0.5,
                value_type="number",
                support_type="direct",
            ),
            *extra_facts,
        ),
        method=ReceiptMethod(family=tool_name),
        statistics=ReceiptStatistics(
            hypothesis_id=hypothesis_id,
            hypothesis_outcome="supports",
            test_name="welch_anova",
            test_statistic=10.0,
            p_value=0.001,
            effect_size=0.5,
            ci_low=0.1,
            ci_high=0.9,
            sample_size=100,
            sequence_index=1,
        ),
        data_state_witness=_WITNESS,
        created_at="2026-08-03T00:00:00Z",
    )


def _insight(
    insight_id: str,
    receipt: EvidenceReceipt,
    *,
    created_round: int,
    status: str = "new",
) -> InsightRecord:
    assert receipt.statistics is not None and receipt.statistics.hypothesis_id
    supports = status in {"new", "reinforced"}
    return InsightRecord(
        insight_id=insight_id,
        hypothesis_id=receipt.statistics.hypothesis_id,
        family=InsightFamily.DIAGNOSTIC,
        status=status,  # type: ignore[arg-type]
        trust_level="supported" if supports else "refuted",
        claim_bundle_id=f"cb_{insight_id}",
        supporting_receipt_ids=(receipt.receipt_id,) if supports else (),
        contradicting_receipt_ids=() if supports else (receipt.receipt_id,),
        proof=(
            InsightProof(
                receipt_id=receipt.receipt_id,
                fact_ids=("effect_size",),
                comparison="supports" if supports else "contradicts",
            ),
        ),
        created_round=created_round,
        last_updated_round=created_round,
    )


def _terminal(receipts: tuple[EvidenceReceipt, ...]) -> ExplorationLoopState:
    return ExplorationLoopState(
        exploration_id=_XPL,
        policy_fingerprint="policy-v1",
        effective_policy_fingerprint="policy-v1",
        code_fingerprint="code-v1",
        data_state_witness=_WITNESS,
        attempt_epoch=0,
        max_successful_tool_calls=10,
        max_rounds=5,
        tool_calls_committed=len(receipts),
        tool_calls_by_kind={"probe": len(receipts)} if receipts else {},
        remaining_tool_call_budget=10 - len(receipts),
        rounds_started=2,
        rounds_settled=2,
        remaining_round_budget=3,
        last_seq=50,
        step_receipt_refs={
            f"step-{receipt.receipt_id}": receipt.receipt_id for receipt in receipts
        },
    )


def _events() -> list[ExplorationLoopEvent]:
    return [
        RoundStartedEvent(seq=1, exploration_id=_XPL, round_index=0),
        RoundSettledEvent(seq=2, exploration_id=_XPL, round_index=0, progress=True),
        RoundStartedEvent(seq=3, exploration_id=_XPL, round_index=1),
        RoundSettledEvent(seq=4, exploration_id=_XPL, round_index=1, progress=True),
    ]


def _scenario(
    *,
    agent_trend_operator: str = "has_spike",
    trend_direction: str = "increasing",
) -> tuple[ExplorationWorkflowState, ExplorationLoopState, dict[str, CandidateSeed]]:
    """Mandatory region insight (pass 1), agent region replication and agent
    trend discovery (pass 2), plus one refuted insight."""
    mandatory_region = _candidate(
        _REGION_PREDICATE,
        columns=("region", "revenue"),
        method_family="compare_groups",
        probe_kind="region_difference",
        origin="mandatory",
    )
    agent_region = _candidate(
        HypothesisPredicate(
            metric="revenue", operator="associated_with", left_operand="region"
        ),
        columns=("region", "revenue"),
        method_family="run_stat_test",
        probe_kind="region_association",
        origin="agent",
    )
    agent_trend = _candidate(
        HypothesisPredicate(
            metric="daily_revenue",
            operator=agent_trend_operator,  # type: ignore[arg-type]
            left_operand="order_date",
        ),
        columns=("order_date", "revenue"),
        method_family="analyze_time_series",
        probe_kind="revenue_trend",
        origin="agent",
    )
    refuted = _candidate(
        HypothesisPredicate(metric="units", operator="differs", left_operand="channel"),
        columns=("channel", "units"),
        method_family="run_stat_test",
        probe_kind="channel_units",
        origin="agent",
    )
    receipt_mandatory = _supporting_receipt(
        "mandatory-region",
        tool_name="run_stat_test",
        columns=("region", "revenue"),
        hypothesis_id=mandatory_region.hypothesis_id,
    )
    receipt_agent_region = _supporting_receipt(
        "agent-region",
        tool_name="run_stat_test",
        columns=("region", "revenue"),
        hypothesis_id=agent_region.hypothesis_id,
    )
    receipt_agent_trend = _supporting_receipt(
        "agent-trend",
        tool_name="analyze_time_series",
        columns=("order_date", "revenue"),
        hypothesis_id=agent_trend.hypothesis_id,
        extra_facts=(
            ReceiptFact(
                fact_id="trend_direction",
                name="trend_direction",
                value=trend_direction,
                value_type="string",
                support_type="direct",
            ),
        ),
    )
    receipt_refuted = _supporting_receipt(
        "refuted-units",
        tool_name="run_stat_test",
        columns=("channel", "units"),
        hypothesis_id=refuted.hypothesis_id,
    )
    receipts = (
        receipt_mandatory,
        receipt_agent_region,
        receipt_agent_trend,
        receipt_refuted,
    )
    workflow = ExplorationWorkflowState()
    for receipt in receipts:
        workflow.committed_receipts[receipt.receipt_id] = receipt
    for insight in (
        _insight("ins_mandatory_region", receipt_mandatory, created_round=0),
        _insight("ins_agent_region", receipt_agent_region, created_round=0),
        _insight("ins_agent_trend", receipt_agent_trend, created_round=1),
        _insight("ins_refuted", receipt_refuted, created_round=1, status="refuted"),
    ):
        workflow.insights[insight.insight_id] = insight
    candidates = {
        seed.hypothesis_id: seed
        for seed in (mandatory_region, agent_region, agent_trend, refuted)
    }
    return workflow, _terminal(receipts), candidates


def _recompute(workflow, terminal, candidates, checker_version=E4A_CHECKER_VERSION_V2):
    return _recompute_checker(
        workflow=workflow,
        events=_events(),
        terminal=terminal,
        candidates=candidates,
        stop_reason="budget_exhausted",
        fixture=_FIXTURE,
        checker_version=checker_version,
    )


def test_v2_two_pass_matching_and_origin_metrics() -> None:
    workflow, terminal, candidates = _scenario()
    checker = _recompute(workflow, terminal, candidates)
    assert set(checker.matched_structure_ids) == {
        "region_difference",
        "planted_trend_revenue",
    }
    assert checker.unmatched_insight_ids == ()
    scores = checker.scores
    assert scores["recall"] == 1.0
    assert scores["precision"] == 1.0
    assert scores["mandatory_probe_recall"] == 0.5
    assert scores["agent_discovery_count"] == 2.0
    assert scores["independent_replication_count"] == 1.0
    assert scores["agent_novel_supported_count"] == 0.0
    assert scores["refuted_insight_count"] == 1.0
    assert scores["agent_first_discovery_round"] == 0.0
    assert scores["trend_recall"] == 1.0
    assert scores["region_difference_recall"] == 1.0
    # Structure-level dynamics: region at round 0, trend at round 1, over 2
    # rounds -> (0.5 + 1.0) / 2.
    assert scores["auc_over_steps"] == 0.75
    assert scores["first_improvement_step"] == 0.0


def test_alternate_operators_gate_the_semantic_pass() -> None:
    workflow, terminal, candidates = _scenario(agent_trend_operator="associated_with")
    checker = _recompute(workflow, terminal, candidates)
    assert "planted_trend_revenue" not in checker.matched_structure_ids
    scores = checker.scores
    assert scores["recall"] == 0.5
    assert scores["trend_recall"] == 0.0
    assert scores["agent_novel_supported_count"] == 1.0


def test_required_fact_values_gate_the_semantic_pass() -> None:
    workflow, terminal, candidates = _scenario(trend_direction="decreasing")
    checker = _recompute(workflow, terminal, candidates)
    assert "planted_trend_revenue" not in checker.matched_structure_ids
    assert checker.scores["trend_recall"] == 0.0


def test_legacy_checker_version_keeps_v1_semantics() -> None:
    workflow, terminal, candidates = _scenario()
    checker = _recompute(workflow, terminal, candidates, checker_version="e4a-checker-v1")
    assert checker.matched_structure_ids == ("region_difference",)
    scores = checker.scores
    assert scores["recall"] == 0.5
    assert scores["precision"] == round(1 / 3, 6)
    assert "mandatory_probe_recall" not in scores
    assert "trend_recall" not in scores
    assert "agent_discovery_count" not in scores


def test_seeded_context_satisfies_domain_metrics_preconditions() -> None:
    bundle = load_planted_bundle()
    context = DataToolContext(
        datasets=[bundle.dataset],
        catalog=build_catalog([bundle.dataset]),
        project_id="e4a-trials",
        session_id="seeding-test",
        store=None,
        payload_policy="schema+aggregates",
    )
    registered = {tool.name: tool for tool in build_data_tools(context)}
    with pytest.raises(ValueError, match="needs dataset profile artifacts"):
        registered["run_domain_metrics"].execute(RunDomainMetricsArguments())
    with pytest.raises(ValueError, match="profile artifact first"):
        registered["recommend_cleaning"].execute(
            RecommendCleaningArguments(dataset_id=DATASET_ID)
        )
    seed_profile_artifacts(context)
    assert registered["run_domain_metrics"].execute(RunDomainMetricsArguments())
    assert registered["recommend_cleaning"].execute(
        RecommendCleaningArguments(dataset_id=DATASET_ID)
    )
