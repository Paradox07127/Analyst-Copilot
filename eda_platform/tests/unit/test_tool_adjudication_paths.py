"""Typed adjudication paths for correlate_columns, screen_anomalies and
run_baseline_model receipts."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import pytest
from pydantic import BaseModel

from eda_platform.agents.data_tools import (
    CorrelateColumnsArguments,
    DataToolContext,
    RunBaselineModelArguments,
    ScreenAnomaliesArguments,
    build_data_tools,
)
from eda_platform.agents.receipts import (
    adjudicate_receipt_hypothesis,
    build_receipt,
)
from eda_platform.agents.tool_context import (
    HypothesisExecutionBinding,
    ToolExecutionContext,
    tool_execution_scope,
)
from eda_platform.schemas.datasets import DatasetRecord
from eda_platform.schemas.hypotheses import HypothesisPredicate
from eda_platform.schemas.receipts import (
    EvidenceReceipt,
    ReceiptFact,
    ReceiptMethod,
    ReceiptScope,
    verify_receipt_digest,
)
from eda_platform.tools.loader import LoadedDataset
from eda_platform.tools.sql_runner import build_catalog

Outcome = Literal["supports", "contradicts"] | None


def _real_tool_receipt(
    *,
    frame: pd.DataFrame,
    binding: HypothesisExecutionBinding,
    tool_name: str,
    arguments: BaseModel,
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
        session_id=f"run_{tool_name}_{binding.hypothesis_id}",
        store=None,
        payload_policy="schema+aggregates",
    )
    tool = next(item for item in build_data_tools(context) if item.name == tool_name)
    execution = ToolExecutionContext(
        run_id=context.session_id,
        provider_call_id=f"provider_{tool_name}",
        logical_step_id=f"step_{tool_name}_{binding.hypothesis_id}",
        sequence_index=1,
        hypothesis=binding,
    )
    with tool_execution_scope(execution):
        result = tool.execute(arguments)
    assert result.receipt_artifact is not None
    return EvidenceReceipt.model_validate(result.receipt_artifact.payload)


# --- correlate_columns -------------------------------------------------------


def _correlation_binding(
    *,
    metric: str = "x",
    right_operand: str | None = "y",
    threshold: float | None = None,
) -> HypothesisExecutionBinding:
    return HypothesisExecutionBinding(
        hypothesis_id="hyp_correlation",
        predicate=HypothesisPredicate(
            metric=metric,
            operator="associated_with",
            right_operand=right_operand,
            threshold=threshold,
        ),
        method_family="correlate_columns",
        dataset_ids=("ds_observations",),
        columns=("x", "y"),
    )


def _correlation_receipt(
    *,
    coefficient: float | None,
    adjusted_p: float | None,
    pair: str = "x~y",
) -> EvidenceReceipt:
    return build_receipt(
        tool_call_id=f"call-corr-{coefficient}-{adjusted_p}-{pair}",
        tool_name="correlate_columns",
        tool_version="1",
        arguments={"dataset_id": "ds_observations"},
        raw_output={"pair": pair},
        artifact_ids=(),
        result_count=1,
        scope=ReceiptScope(
            dataset_ids=("ds_observations",),
            columns=("x", "y", "z"),
            scope_resolution="resolved",
        ),
        facts=(
            ReceiptFact(fact_id="pairs_tested", name="pairs_tested", value=1, value_type="count"),
            ReceiptFact(
                fact_id="pair0.coefficient",
                name="pair0.coefficient",
                value=coefficient,
                value_type="number",
            ),
            ReceiptFact(
                fact_id="pair0.adjusted_p",
                name="pair0.adjusted_p",
                value=adjusted_p,
                value_type="number",
            ),
            ReceiptFact(
                fact_id="pair0.columns", name="pair0.columns", value=pair, value_type="string"
            ),
        ),
        method=ReceiptMethod(family="pearson_correlation_screen"),
        data_state_witness="witness-corr",
        created_at="2026-08-03T00:00:00+00:00",
    )


def test_correlate_columns_real_significant_pair_supports() -> None:
    values = [float(index) for index in range(60)]
    frame = pd.DataFrame(
        {
            "x": values,
            "y": [2.0 * value + (0.3 if index % 2 else -0.3) for index, value in enumerate(values)],
        }
    )
    receipt = _real_tool_receipt(
        frame=frame,
        binding=_correlation_binding(threshold=0.5),
        tool_name="correlate_columns",
        arguments=CorrelateColumnsArguments(
            dataset_id="ds_observations", columns=["x", "y"]
        ),
    )

    assert verify_receipt_digest(receipt)
    assert receipt.statistics is not None
    assert receipt.statistics.hypothesis_outcome == "supports"


@pytest.mark.parametrize(
    ("coefficient", "adjusted_p", "pair", "threshold", "expected"),
    (
        # Significant and above materiality.
        (0.9, 0.01, "x~y", 0.5, "supports"),
        # Non-significant screen directly negates the association.
        (0.05, 0.5, "x~y", None, "contradicts"),
        # Significant but below the required materiality.
        (0.3, 0.01, "x~y", 0.5, "contradicts"),
        # Significant, no materiality required.
        (0.2, 0.01, "x~y", None, "supports"),
        # The predicate pair was never tested in this receipt.
        (0.9, 0.01, "x~z", None, None),
        # The pair row exists but carries no p-value.
        (0.9, None, "x~y", None, None),
    ),
)
def test_correlate_columns_adjudication_matrix(
    coefficient: float | None,
    adjusted_p: float | None,
    pair: str,
    threshold: float | None,
    expected: Outcome,
) -> None:
    receipt = _correlation_receipt(coefficient=coefficient, adjusted_p=adjusted_p, pair=pair)
    adjudicated = adjudicate_receipt_hypothesis(
        receipt, _correlation_binding(threshold=threshold)
    )
    assert verify_receipt_digest(adjudicated)
    assert adjudicated.statistics is not None
    assert adjudicated.statistics.hypothesis_outcome == expected


def test_correlate_columns_without_right_operand_stays_unadjudicated() -> None:
    receipt = _correlation_receipt(coefficient=0.9, adjusted_p=0.01)
    adjudicated = adjudicate_receipt_hypothesis(
        receipt, _correlation_binding(right_operand=None)
    )
    assert adjudicated.statistics is not None
    assert adjudicated.statistics.hypothesis_outcome is None


# --- screen_anomalies --------------------------------------------------------


def _spike_binding(*, metric: str = "value") -> HypothesisExecutionBinding:
    return HypothesisExecutionBinding(
        hypothesis_id="hyp_outliers",
        predicate=HypothesisPredicate(metric=metric, operator="has_spike"),
        method_family="screen_anomalies",
        dataset_ids=("ds_observations",),
        columns=(metric,),
    )


def test_screen_anomalies_real_outliers_support_has_spike() -> None:
    frame = pd.DataFrame({"value": [10.0 + (index % 5) * 0.1 for index in range(40)] + [500.0]})
    receipt = _real_tool_receipt(
        frame=frame,
        binding=_spike_binding(),
        tool_name="screen_anomalies",
        arguments=ScreenAnomaliesArguments(dataset_id="ds_observations", column="value"),
    )

    assert verify_receipt_digest(receipt)
    assert receipt.statistics is not None
    assert receipt.statistics.hypothesis_outcome == "supports"


def test_screen_anomalies_real_clean_scan_contradicts() -> None:
    frame = pd.DataFrame({"value": [10.0 + (index % 5) * 0.1 for index in range(40)]})
    receipt = _real_tool_receipt(
        frame=frame,
        binding=_spike_binding(),
        tool_name="screen_anomalies",
        arguments=ScreenAnomaliesArguments(dataset_id="ds_observations", column="value"),
    )

    assert receipt.statistics is not None
    assert receipt.statistics.hypothesis_outcome == "contradicts"


def test_screen_anomalies_scan_of_another_column_stays_unadjudicated() -> None:
    frame = pd.DataFrame(
        {
            "value": [10.0 + (index % 5) * 0.1 for index in range(40)] + [500.0],
            "other": [1.0] * 41,
        }
    )
    binding = HypothesisExecutionBinding(
        hypothesis_id="hyp_outliers_elsewhere",
        # The predicate targets `other`, but only `value` was scanned.
        predicate=HypothesisPredicate(metric="other", operator="has_spike"),
        method_family="screen_anomalies",
        dataset_ids=("ds_observations",),
        columns=(),
    )
    receipt = _real_tool_receipt(
        frame=frame,
        binding=binding,
        tool_name="screen_anomalies",
        arguments=ScreenAnomaliesArguments(dataset_id="ds_observations", column="value"),
    )

    assert receipt.statistics is not None
    assert receipt.statistics.hypothesis_outcome is None


# --- run_baseline_model ------------------------------------------------------


def _baseline_binding(
    *,
    metric: str = "churn",
    right_operand: str | None = None,
    threshold: float | None = None,
    columns: tuple[str, ...] = ("churn",),
) -> HypothesisExecutionBinding:
    return HypothesisExecutionBinding(
        hypothesis_id="hyp_predictable",
        predicate=HypothesisPredicate(
            metric=metric,
            operator="associated_with",
            right_operand=right_operand,
            threshold=threshold,
        ),
        method_family="run_baseline_model",
        dataset_ids=("ds_observations",),
        columns=columns,
    )


def _baseline_receipt(
    *,
    task_type: str,
    target_column: str = "churn",
    accuracy: float | None = None,
    baseline_accuracy: float | None = None,
    r2: float | None = None,
    cv_accuracy_std: float | None = None,
) -> EvidenceReceipt:
    facts = [
        ReceiptFact(fact_id="task_type", name="task_type", value=task_type, value_type="string"),
        ReceiptFact(
            fact_id="target_column",
            name="target_column",
            value=target_column,
            value_type="string",
        ),
    ]
    if accuracy is not None:
        facts.append(
            ReceiptFact(
                fact_id="metric.accuracy",
                name="metric.accuracy",
                value=accuracy,
                value_type="number",
            )
        )
    if baseline_accuracy is not None:
        facts.append(
            ReceiptFact(
                fact_id="baseline_accuracy",
                name="baseline_accuracy",
                value=baseline_accuracy,
                value_type="number",
            )
        )
    if r2 is not None:
        facts.append(
            ReceiptFact(fact_id="metric.r2", name="metric.r2", value=r2, value_type="number")
        )
    if cv_accuracy_std is not None:
        facts.append(
            ReceiptFact(
                fact_id="metric.cv_accuracy_std",
                name="metric.cv_accuracy_std",
                value=cv_accuracy_std,
                value_type="number",
            )
        )
    return build_receipt(
        tool_call_id=f"call-baseline-{task_type}-{accuracy}-{baseline_accuracy}-{r2}",
        tool_name="run_baseline_model",
        tool_version="1",
        arguments={"dataset_id": "ds_observations", "target_column": target_column},
        raw_output={"task_type": task_type},
        artifact_ids=(),
        result_count=1,
        scope=ReceiptScope(
            dataset_ids=("ds_observations",),
            columns=("churn", "x"),
            scope_resolution="resolved",
        ),
        facts=tuple(facts),
        method=ReceiptMethod(family="ml_baseline"),
        data_state_witness="witness-baseline",
        created_at="2026-08-03T00:00:00+00:00",
    )


def test_run_baseline_model_real_classification_supports() -> None:
    # ~20% deterministic label flips spread across every x group: predictable
    # well above the majority baseline without tripping the leakage exclusion.
    rows = 100
    x = [float(index % 10) for index in range(rows)]
    labels = [
        ("low" if value >= 5.0 else "high")
        if (index // 10) % 5 == 0
        else ("high" if value >= 5.0 else "low")
        for index, value in enumerate(x)
    ]
    frame = pd.DataFrame(
        {
            "x": x,
            "noise": [float((index * 37) % 11) for index in range(rows)],
            "churn": labels,
        }
    )
    receipt = _real_tool_receipt(
        frame=frame,
        binding=_baseline_binding(),
        tool_name="run_baseline_model",
        arguments=RunBaselineModelArguments(
            dataset_id="ds_observations", target_column="churn"
        ),
    )

    assert verify_receipt_digest(receipt)
    assert receipt.statistics is not None
    assert receipt.statistics.hypothesis_outcome == "supports"


@pytest.mark.parametrize(
    ("kwargs", "threshold", "expected"),
    (
        # Above the majority baseline.
        (
            {"task_type": "classification", "accuracy": 0.9, "baseline_accuracy": 0.55},
            None,
            "supports",
        ),
        # Below the majority baseline.
        (
            {"task_type": "classification", "accuracy": 0.6, "baseline_accuracy": 0.7},
            None,
            "contradicts",
        ),
        # Above baseline but below the required materiality margin.
        (
            {"task_type": "classification", "accuracy": 0.72, "baseline_accuracy": 0.7},
            0.05,
            "contradicts",
        ),
        # The majority baseline is missing: nothing to compare against.
        ({"task_type": "classification", "accuracy": 0.9}, None, None),
        # R^2 is skill over the mean-predictor baseline.
        ({"task_type": "regression", "r2": 0.4}, None, "supports"),
        ({"task_type": "regression", "r2": -0.2}, None, "contradicts"),
    ),
)
def test_run_baseline_model_adjudication_matrix(
    kwargs: dict[str, object],
    threshold: float | None,
    expected: Outcome,
) -> None:
    receipt = _baseline_receipt(**kwargs)  # type: ignore[arg-type]
    adjudicated = adjudicate_receipt_hypothesis(
        receipt, _baseline_binding(threshold=threshold)
    )
    assert verify_receipt_digest(adjudicated)
    assert adjudicated.statistics is not None
    assert adjudicated.statistics.hypothesis_outcome == expected


def test_run_baseline_model_predicate_about_another_target_stays_unadjudicated() -> None:
    receipt = _baseline_receipt(
        task_type="classification", accuracy=0.9, baseline_accuracy=0.55
    )
    adjudicated = adjudicate_receipt_hypothesis(
        receipt, _baseline_binding(metric="revenue")
    )
    assert adjudicated.statistics is not None
    assert adjudicated.statistics.hypothesis_outcome is None


# --- metric-name resolution (seed-7 predicate vocabulary) --------------------
#
# The model's natural predicates are named derived metrics under numeric
# comparison ("pearson_correlation > 0.6", "r2 > 0.2"), while facts are
# namespaced by the tool ("pair0.coefficient", "metric.r2"). Without a
# resolution layer every one of these adjudicates to None (seed-7: 30/30).


def _numeric_binding(
    *,
    metric: str,
    operator: str,
    threshold: float,
    method_family: str,
    right_operand: str | None = None,
    left_operand: str = "x",
    columns: tuple[str, ...] = ("x", "y"),
) -> HypothesisExecutionBinding:
    return HypothesisExecutionBinding(
        hypothesis_id="hyp_named_metric",
        predicate=HypothesisPredicate(
            metric=metric,
            operator=operator,  # type: ignore[arg-type]
            left_operand=left_operand,
            right_operand=right_operand,
            threshold=threshold,
        ),
        method_family=method_family,
        dataset_ids=("ds_observations",),
        columns=columns,
    )


@pytest.mark.parametrize(
    ("metric", "coefficient", "threshold", "expected"),
    (
        ("pearson_correlation", 0.9, 0.6, "supports"),
        ("pearson_correlation", 0.3, 0.6, "contradicts"),
        ("correlation_coefficient", -0.8, -0.9, "supports"),
        ("spearman_correlation", 0.7, 0.6, "supports"),
    ),
)
def test_correlation_coefficient_names_resolve_to_the_pair_fact(
    metric: str, coefficient: float, threshold: float, expected: Outcome
) -> None:
    receipt = _correlation_receipt(coefficient=coefficient, adjusted_p=0.01)
    adjudicated = adjudicate_receipt_hypothesis(
        receipt,
        _numeric_binding(
            metric=metric,
            operator="greater_than",
            threshold=threshold,
            method_family="correlate_columns",
            right_operand="y",
        ),
    )
    assert adjudicated.statistics is not None
    assert adjudicated.statistics.hypothesis_outcome == expected


def test_correlation_name_without_pair_row_stays_unadjudicated() -> None:
    receipt = _correlation_receipt(coefficient=0.9, adjusted_p=0.01, pair="x~z")
    adjudicated = adjudicate_receipt_hypothesis(
        receipt,
        _numeric_binding(
            metric="pearson_correlation",
            operator="greater_than",
            threshold=0.5,
            method_family="correlate_columns",
            right_operand="y",
        ),
    )
    assert adjudicated.statistics is not None
    assert adjudicated.statistics.hypothesis_outcome is None


@pytest.mark.parametrize(
    ("metric", "r2", "threshold", "expected"),
    (
        ("r2", 0.45, 0.2, "supports"),
        ("r2", 0.10, 0.2, "contradicts"),
        ("baseline_r_squared", 0.45, 0.05, "supports"),
        ("baseline_r2", 0.30, 0.5, "contradicts"),
        ("r_squared", 0.45, 0.2, "supports"),
    ),
)
def test_r2_aliases_resolve_to_the_metric_namespace(
    metric: str, r2: float, threshold: float, expected: Outcome
) -> None:
    receipt = _baseline_receipt(task_type="regression", r2=r2)
    adjudicated = adjudicate_receipt_hypothesis(
        receipt,
        _numeric_binding(
            metric=metric,
            operator="greater_than",
            threshold=threshold,
            method_family="run_baseline_model",
            left_operand="churn",
            columns=("churn",),
        ),
    )
    assert adjudicated.statistics is not None
    assert adjudicated.statistics.hypothesis_outcome == expected


def test_metric_namespace_fallback_is_tool_agnostic() -> None:
    receipt = _baseline_receipt(
        task_type="classification", accuracy=0.9, baseline_accuracy=0.5
    )
    adjudicated = adjudicate_receipt_hypothesis(
        receipt,
        _numeric_binding(
            metric="accuracy",
            operator="greater_than",
            threshold=0.8,
            method_family="run_baseline_model",
            left_operand="churn",
            columns=("churn",),
        ),
    )
    assert adjudicated.statistics is not None
    assert adjudicated.statistics.hypothesis_outcome == "supports"


@pytest.mark.parametrize(
    ("metric", "method_family"),
    (
        # Requires dividing two facts: computation, not lookup.
        ("model_vs_baseline_rmse_ratio", "run_baseline_model"),
        # analyze_time_series emits no slope fact at all.
        ("trend_slope", "analyze_time_series"),
    ),
)
def test_unresolvable_named_metrics_stay_honestly_unadjudicated(
    metric: str, method_family: str
) -> None:
    receipt = _baseline_receipt(task_type="regression", r2=0.4)
    adjudicated = adjudicate_receipt_hypothesis(
        receipt,
        _numeric_binding(
            metric=metric,
            operator="less_than",
            threshold=0.9,
            method_family=method_family,
            left_operand="churn",
            columns=("churn",),
        ),
    )
    assert adjudicated.statistics is not None
    assert adjudicated.statistics.hypothesis_outcome is None


# --- significance and materiality floors (deepseek seed 9 regression) --------
#
# Two false "supports" reached a certified report with the real numbers below.
# Both came from this session's own changes: metric-name resolution let a
# statistical estimate be compared with a bare ">", and an absent materiality
# threshold was read as "any nonzero effect counts".


def test_a_named_correlation_below_significance_does_not_support() -> None:
    """seed 9: customer_age~units, r=+0.0246, adjusted_p=0.417 — noise, but
    "0.0246 > 0.0" is literally true."""
    receipt = _correlation_receipt(coefficient=0.0246, adjusted_p=0.41713555)
    adjudicated = adjudicate_receipt_hypothesis(
        receipt,
        _numeric_binding(
            metric="correlation",
            operator="greater_than",
            threshold=0.0,
            method_family="correlate_columns",
            right_operand="y",
        ),
    )
    assert adjudicated.statistics is not None
    assert adjudicated.statistics.hypothesis_outcome == "contradicts"


def test_a_named_correlation_that_is_significant_still_supports() -> None:
    """Control: the resolution layer must keep working for real associations."""
    receipt = _correlation_receipt(coefficient=0.62, adjusted_p=0.0001)
    adjudicated = adjudicate_receipt_hypothesis(
        receipt,
        _numeric_binding(
            metric="correlation",
            operator="greater_than",
            threshold=0.0,
            method_family="correlate_columns",
            right_operand="y",
        ),
    )
    assert adjudicated.statistics is not None
    assert adjudicated.statistics.hypothesis_outcome == "supports"


def test_baseline_skill_inside_its_own_cross_validation_noise_does_not_support() -> None:
    """seed 9: accuracy 0.2209 vs majority baseline 0.2191 — a 0.0018 gain the
    card's own cv_accuracy_std of 0.0171 cannot distinguish from noise."""
    receipt = _baseline_receipt(
        task_type="classification",
        accuracy=0.2209,
        baseline_accuracy=0.219136,
        cv_accuracy_std=0.0171,
    )
    adjudicated = adjudicate_receipt_hypothesis(
        receipt,
        _baseline_binding(),
    )
    assert adjudicated.statistics is not None
    assert adjudicated.statistics.hypothesis_outcome == "contradicts"


def test_baseline_skill_beyond_its_noise_still_supports() -> None:
    receipt = _baseline_receipt(
        task_type="classification",
        accuracy=0.62,
        baseline_accuracy=0.22,
        cv_accuracy_std=0.0171,
    )
    adjudicated = adjudicate_receipt_hypothesis(
        receipt,
        _baseline_binding(),
    )
    assert adjudicated.statistics is not None
    assert adjudicated.statistics.hypothesis_outcome == "supports"
