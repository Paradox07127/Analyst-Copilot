"""Typed predicate adjudication at the receipt boundary.

These tests intentionally use the real run_stat_test adapter for statistical
receipts: its effect is a group-comparison quantity, while an absolute
greater_than/less_than threshold is expressed in the metric's own units.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import pytest

from eda_platform.agents.data_tools import (
    AnalyzeTimeSeriesArguments,
    DataToolContext,
    DiagnoseMissingnessArguments,
    RunStatTestArguments,
    build_data_tools,
)
from eda_platform.agents.exploration.reducer import reduce_insight
from eda_platform.agents.receipts import (
    adjudicate_receipt_hypothesis,
    build_receipt,
)
from eda_platform.agents.tool_context import (
    HypothesisExecutionBinding,
    ToolExecutionContext,
    tool_execution_scope,
)
from eda_platform.schemas.claims import Claim, ClaimBundle
from eda_platform.schemas.datasets import DatasetRecord
from eda_platform.schemas.exploration import InsightFamily
from eda_platform.schemas.hypotheses import HypothesisPredicate
from eda_platform.schemas.insights import TransitionProposal
from eda_platform.schemas.receipts import (
    EvidenceReceipt,
    ReceiptFact,
    ReceiptMethod,
    ReceiptScope,
    verify_receipt_digest,
)
from eda_platform.tools.loader import LoadedDataset
from eda_platform.tools.sql_runner import build_catalog

PredicateOperator = Literal["greater_than", "less_than"]


def _binding(
    operator: PredicateOperator | Literal["differs"],
    *,
    threshold: float | None,
    method_family: str = "compare_groups",
) -> HypothesisExecutionBinding:
    return HypothesisExecutionBinding(
        hypothesis_id=f"hyp_{operator}",
        predicate=HypothesisPredicate(
            metric="revenue",
            operator=operator,
            left_operand="segment",
            threshold=threshold,
        ),
        method_family=method_family,
        dataset_ids=("ds_sales",),
        columns=("segment", "revenue"),
    )


def _stat_receipt(
    *,
    high_group_first: bool,
    operator: PredicateOperator | Literal["differs"],
    threshold: float | None,
) -> EvidenceReceipt:
    low = [101.0 + (index % 5) * 0.1 for index in range(40)]
    high = [103.0 + (index % 5) * 0.1 for index in range(40)]
    first, second = (high, low) if high_group_first else (low, high)
    dataset = LoadedDataset(
        record=DatasetRecord(
            dataset_id="ds_sales",
            name="sales.csv",
            path=Path("/data/sales.csv"),
            content_hash="hash-sales",
        ),
        frame=pd.DataFrame(
            {
                "segment": ["A"] * 40 + ["B"] * 40,
                "revenue": first + second,
            }
        ),
    )
    context = DataToolContext(
        datasets=[dataset],
        catalog=build_catalog([dataset]),
        project_id="project_t",
        session_id=f"run_{operator}_{'high' if high_group_first else 'low'}",
        store=None,
        payload_policy="schema+aggregates",
    )
    tool = next(item for item in build_data_tools(context) if item.name == "run_stat_test")
    execution = ToolExecutionContext(
        run_id=context.session_id,
        provider_call_id="provider_call_1",
        logical_step_id=f"step_{operator}_{'high' if high_group_first else 'low'}",
        sequence_index=1,
        hypothesis=_binding(operator, threshold=threshold),
    )
    with tool_execution_scope(execution):
        result = tool.execute(
            RunStatTestArguments(
                dataset_id="ds_sales",
                test_type="independent_t_test",
                group_column="segment",
                value_column="revenue",
            )
        )
    assert result.receipt_artifact is not None
    return EvidenceReceipt.model_validate(result.receipt_artifact.payload)


@pytest.mark.parametrize(
    ("high_group_first", "operator", "threshold", "effect_sign"),
    (
        (True, "greater_than", 1_000.0, 1),
        (False, "less_than", 0.0, -1),
    ),
)
def test_statistical_effect_direction_cannot_answer_an_absolute_metric_threshold(
    high_group_first: bool,
    operator: PredicateOperator,
    threshold: float,
    effect_sign: int,
) -> None:
    receipt = _stat_receipt(
        high_group_first=high_group_first,
        operator=operator,
        threshold=threshold,
    )

    assert verify_receipt_digest(receipt)
    assert receipt.tool_name == "run_stat_test"
    assert receipt.statistics is not None
    assert receipt.statistics.p_value is not None
    assert receipt.statistics.p_value <= 0.05
    assert receipt.statistics.effect_size is not None
    assert receipt.statistics.effect_size * effect_sign > 0
    assert {fact.fact_id for fact in receipt.facts} == {
        "p_value",
        "statistic",
        "effect_size",
        "sample_size",
    }
    assert receipt.statistics.hypothesis_outcome is None


def test_compare_groups_alias_still_adjudicates_a_real_statistical_difference() -> None:
    receipt = _stat_receipt(
        high_group_first=True,
        operator="differs",
        threshold=None,
    )

    assert receipt.tool_name == "run_stat_test"
    assert receipt.statistics is not None
    assert receipt.statistics.hypothesis_outcome == "supports"


def _direct_metric_outcome(
    operator: PredicateOperator,
    threshold: float,
) -> Literal["supports", "contradicts"] | None:
    receipt = build_receipt(
        tool_call_id=f"call-{operator}-{threshold}",
        tool_name="summarize_metric",
        tool_version="1",
        arguments={"metric": "revenue"},
        raw_output={"revenue": 102.0},
        artifact_ids=(),
        result_count=1,
        scope=ReceiptScope(
            dataset_ids=("ds_sales",),
            columns=("segment", "revenue"),
            scope_resolution="explicit",
        ),
        facts=(
            ReceiptFact(
                fact_id="revenue",
                name="revenue",
                value=102.0,
                value_type="number",
            ),
        ),
        method=ReceiptMethod(family="summarize_metric"),
        data_state_witness="witness-1",
        created_at="2026-08-02T00:00:00+00:00",
    )
    adjudicated = adjudicate_receipt_hypothesis(
        receipt,
        _binding(
            operator,
            threshold=threshold,
            method_family="summarize_metric",
        ),
    )
    assert verify_receipt_digest(adjudicated)
    assert adjudicated.statistics is not None
    return adjudicated.statistics.hypothesis_outcome


@pytest.mark.parametrize(
    ("operator", "threshold", "expected"),
    (
        ("greater_than", 100.0, "supports"),
        ("greater_than", 1_000.0, "contradicts"),
        ("less_than", 1_000.0, "supports"),
        ("less_than", 100.0, "contradicts"),
    ),
)
def test_absolute_threshold_uses_a_direct_fact_in_the_metrics_own_units(
    operator: PredicateOperator,
    threshold: float,
    expected: Literal["supports", "contradicts"],
) -> None:
    assert _direct_metric_outcome(operator, threshold) == expected


def _real_tool_receipt(
    *,
    frame: pd.DataFrame,
    binding: HypothesisExecutionBinding,
    tool_name: str,
    arguments: AnalyzeTimeSeriesArguments | DiagnoseMissingnessArguments,
) -> EvidenceReceipt:
    dataset = LoadedDataset(
        record=DatasetRecord(
            dataset_id="ds_observations",
            name="observations.csv",
            path=Path("/data/observations.csv"),
            content_hash="hash-observations",
        ),
        frame=frame,
    )
    context = DataToolContext(
        datasets=[dataset],
        catalog=build_catalog([dataset]),
        project_id="project_t",
        session_id=f"run_{tool_name}",
        store=None,
        payload_policy="schema+aggregates",
    )
    tool = next(item for item in build_data_tools(context) if item.name == tool_name)
    execution = ToolExecutionContext(
        run_id=context.session_id,
        provider_call_id=f"provider_{tool_name}",
        logical_step_id=f"step_{tool_name}",
        sequence_index=1,
        hypothesis=binding,
    )
    with tool_execution_scope(execution):
        result = tool.execute(arguments)
    assert result.receipt_artifact is not None
    return EvidenceReceipt.model_validate(result.receipt_artifact.payload)


def _reduce_typed_receipt(
    receipt: EvidenceReceipt,
    *,
    hypothesis_id: str,
    comparison: Literal["supports", "contradicts"] = "supports",
) -> tuple[str, str]:
    bundle = ClaimBundle(
        claim_bundle_id=f"bundle_{hypothesis_id}",
        hypothesis_id=hypothesis_id,
        evidence_lane="exploratory",
        claims=(
            Claim(
                claim_id=f"claim_{hypothesis_id}",
                claim_type="comparison",
                claim_text="The typed receipt supports the hypothesis.",
                support_type="direct",
                evidence_fact_ids=(
                    f"{receipt.receipt_id}:{receipt.facts[0].fact_id}",
                ),
            ),
        ),
    )
    proposal = TransitionProposal(
        hypothesis_id=hypothesis_id,
        insight_id=f"insight_{hypothesis_id}",
        family=InsightFamily.DIAGNOSTIC,
        claim_bundle_id=bundle.claim_bundle_id,
        supporting_receipt_ids=(receipt.receipt_id,) if comparison == "supports" else (),
        contradicting_receipt_ids=(
            (receipt.receipt_id,) if comparison == "contradicts" else ()
        ),
        proposed_status="new",
        limitations=receipt.method.warnings,
    )
    record = reduce_insight(
        proposal,
        prior=None,
        committed_receipts={receipt.receipt_id: receipt},
        admitted_claim_bundles={bundle.claim_bundle_id: bundle},
        expected_witness=receipt.data_state_witness,
        round_index=0,
        require_typed_hypothesis_outcome=True,
    )
    return record.status, record.trust_level


def test_real_warned_missingness_support_reduces_to_a_supported_new_insight() -> None:
    hypothesis_id = "hyp_missingness"
    binding = HypothesisExecutionBinding(
        hypothesis_id=hypothesis_id,
        predicate=HypothesisPredicate(
            metric="score",
            operator="associated_with",
            right_operand="channel",
            threshold=5.0,
        ),
        method_family="diagnose_missingness",
        dataset_ids=("ds_observations",),
        columns=("score", "channel"),
    )
    receipt = _real_tool_receipt(
        frame=pd.DataFrame(
            {
                "channel": ["phone"] * 30 + ["web"] * 30,
                "score": [None] * 24
                + [float(index) for index in range(6)]
                + [None] * 3
                + [float(index) for index in range(27)],
            }
        ),
        binding=binding,
        tool_name="diagnose_missingness",
        arguments=DiagnoseMissingnessArguments(
            dataset_id="ds_observations",
            group_columns=["channel"],
        ),
    )

    assert verify_receipt_digest(receipt)
    assert receipt.method.warnings
    assert receipt.statistics is not None
    assert receipt.statistics.hypothesis_outcome == "supports"
    assert _reduce_typed_receipt(receipt, hypothesis_id=hypothesis_id) == (
        "new",
        "supported",
    )


def test_real_warned_spike_support_reduces_to_a_supported_new_insight() -> None:
    hypothesis_id = "hyp_spike"
    binding = HypothesisExecutionBinding(
        hypothesis_id=hypothesis_id,
        predicate=HypothesisPredicate(
            metric="revenue",
            operator="has_spike",
            threshold=3.5,
        ),
        method_family="analyze_time_series",
        dataset_ids=("ds_observations",),
        columns=("date", "revenue"),
    )
    receipt = _real_tool_receipt(
        frame=pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=35, freq="D"),
                "revenue": [10.0] * 17 + [1_000.0] + [10.0] * 17,
            }
        ),
        binding=binding,
        tool_name="analyze_time_series",
        arguments=AnalyzeTimeSeriesArguments(
            dataset_id="ds_observations",
            time_column="date",
            value_column="revenue",
            freq="D",
            agg="sum",
        ),
    )

    assert verify_receipt_digest(receipt)
    assert receipt.method.warnings
    assert receipt.statistics is not None
    assert receipt.statistics.hypothesis_outcome == "supports"
    assert _reduce_typed_receipt(receipt, hypothesis_id=hypothesis_id) == (
        "new",
        "supported",
    )


@pytest.mark.parametrize("comparison", ("supports", "contradicts"))
def test_explicit_method_invalidation_reduces_inconclusive_on_either_side(
    comparison: Literal["supports", "contradicts"],
) -> None:
    hypothesis_id = "hyp_invalid_method"
    binding = HypothesisExecutionBinding(
        hypothesis_id=hypothesis_id,
        predicate=HypothesisPredicate(
            metric="revenue",
            operator="greater_than",
            threshold=100.0,
        ),
        method_family="summarize_metric",
        dataset_ids=("ds_observations",),
        columns=("revenue",),
    )
    receipt = build_receipt(
        tool_call_id="call-invalid-method",
        tool_name="summarize_metric",
        tool_version="1",
        arguments={"metric": "revenue"},
        raw_output={"revenue": 102.0},
        artifact_ids=(),
        result_count=1,
        scope=ReceiptScope(
            dataset_ids=("ds_observations",),
            columns=("revenue",),
            scope_resolution="explicit",
        ),
        facts=(
            ReceiptFact(
                fact_id="revenue",
                name="revenue",
                value=102.0,
                value_type="number",
            ),
        ),
        method=ReceiptMethod(
            family="summarize_metric",
            parameters={"hypothesis_evidence_valid": False},
            warnings=("The method precondition was not met.",),
        ),
        data_state_witness="witness-invalid-method",
        created_at="2026-08-02T00:00:00+00:00",
    )
    adjudicated = adjudicate_receipt_hypothesis(receipt, binding)

    assert verify_receipt_digest(adjudicated)
    assert adjudicated.statistics is not None
    assert adjudicated.statistics.hypothesis_outcome is None
    assert _reduce_typed_receipt(
        adjudicated,
        hypothesis_id=hypothesis_id,
        comparison=comparison,
    ) == (
        "inconclusive",
        "unsupported",
    )
