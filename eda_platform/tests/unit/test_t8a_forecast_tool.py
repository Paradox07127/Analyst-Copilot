"""run_forecast agent tool: fpp3 baseline backtest selection, projection band, receipt contract.

The backtest numbers are recomputed by hand in the tests (independent naive
rolling-origin loop) so the tool's metrics are verified, not trusted.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
from pydantic import ValidationError

from eda_platform.agents.data_tools import (
    DataToolContext,
    RunForecastArguments,
    build_data_tools,
)
from eda_platform.core.claim_language import (
    asserts_model_capability,
    implies_causation,
)
from eda_platform.core.methods import METHOD_REGISTRY
from eda_platform.drivers.question_exec import _method_contract_failure
from eda_platform.schemas.artifacts import AnalysisTable, Artifact, ArtifactType
from eda_platform.schemas.datasets import DatasetRecord
from eda_platform.schemas.questions import QuestionCandidate, QuestionScore
from eda_platform.schemas.receipts import EvidenceReceipt, verify_receipt_digest
from eda_platform.tools.forecast import FORECAST_LIMITATION
from eda_platform.tools.loader import LoadedDataset
from eda_platform.tools.sql_runner import build_catalog


def _dataset(name: str, frame: pd.DataFrame, dataset_id: str) -> LoadedDataset:
    return LoadedDataset(
        record=DatasetRecord(
            dataset_id=dataset_id,
            name=name,
            path=Path(f"/data/{name}"),
            content_hash="hash_" + dataset_id,
        ),
        frame=frame,
    )


def _context(frame: pd.DataFrame) -> DataToolContext:
    datasets = [_dataset("series.csv", frame, "ds_series")]
    return DataToolContext(
        datasets=datasets,
        catalog=build_catalog(datasets),
        project_id="project_t8a",
        session_id="run_t8a",
        store=None,
        payload_policy="schema+aggregates",
        artifacts=[],
    )


def _tool(context: DataToolContext) -> Any:
    return next(tool for tool in build_data_tools(context) if tool.name == "run_forecast")


def _receipt(context: DataToolContext) -> EvidenceReceipt:
    artifact = [
        a for a in context.artifacts if a.type is ArtifactType.EVIDENCE_RECEIPT
    ][-1]
    return EvidenceReceipt.model_validate(artifact.payload)


def _fact(receipt: EvidenceReceipt, fact_id: str) -> Any:
    return next(fact for fact in receipt.facts if fact.fact_id == fact_id).value


def _linear_frame(n: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "day": pd.date_range("2024-01-01", periods=n, freq="D"),
            "amount": [float(value) for value in range(1, n + 1)],
        }
    )


def _rows(context: DataToolContext) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    table = next(a for a in context.artifacts if a.type is ArtifactType.TABLE)
    rows = AnalysisTable.model_validate(table.payload).rows
    baselines = [row for row in rows if "baseline" in row]
    projections = [row for row in rows if "period" in row]
    return baselines, projections


def _hand_naive_backtest(
    values: list[float], *, initial: int, horizon: int
) -> tuple[float, float, int]:
    """Independent rolling-origin naive backtest: MAE, RMSE, origin count."""
    errors: list[float] = []
    origins = 0
    for origin in range(initial, len(values)):
        origins += 1
        steps = min(horizon, len(values) - origin)
        last = values[origin - 1]
        errors.extend(values[origin + step] - last for step in range(steps))
    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    return mae, rmse, origins


def test_linear_series_backtest_is_hand_recomputable_and_drift_wins() -> None:
    context = _context(_linear_frame())
    tool = _tool(context)

    result = tool.execute(
        RunForecastArguments(
            dataset_id="ds_series", time_column="day", value_column="amount"
        )
    )

    content = cast(dict[str, Any], result.content)
    assert content["chosen_baseline"] == "drift"
    baselines, projections = _rows(context)
    by_name = {row["baseline"]: row for row in baselines}

    # Daily data, default period 7: initial train window max(2*7, 8) = 14.
    mae, rmse, origins = _hand_naive_backtest(
        [float(value) for value in range(1, 21)], initial=14, horizon=6
    )
    assert origins == 6 and content["backtest_origins"] == 6
    assert by_name["naive"]["backtest_mae"] == pytest.approx(mae, abs=1e-6)
    assert by_name["naive"]["backtest_rmse"] == pytest.approx(rmse, abs=1e-6)
    # Naive scale mean(|diff|) is exactly 1, so naive MASE equals its MAE.
    assert by_name["naive"]["mase"] == pytest.approx(mae, abs=1e-6)
    # A perfectly linear series makes drift error-free; it must be chosen.
    assert by_name["drift"]["backtest_mae"] == 0.0
    assert {"naive", "seasonal_naive", "drift", "mean"} == set(by_name)

    # Drift projection continues the line; zero residuals give a zero-width band.
    assert [row["projected_value"] for row in projections] == [
        pytest.approx(value) for value in (21.0, 22.0, 23.0, 24.0, 25.0, 26.0)
    ]
    assert all(
        row["band_low"] == pytest.approx(row["projected_value"])
        and row["band_high"] == pytest.approx(row["projected_value"])
        for row in projections
    )
    assert content["projection_start"] == "2024-01-21T00:00:00"
    assert content["projection_end"] == "2024-01-26T00:00:00"


def test_receipt_carries_traceable_projection_facts_and_no_p_value() -> None:
    context = _context(_linear_frame())
    _tool(context).execute(
        RunForecastArguments(
            dataset_id="ds_series", time_column="day", value_column="amount"
        )
    )

    receipt = _receipt(context)
    assert verify_receipt_digest(receipt)
    assert receipt.tool_name == "run_forecast"
    assert _fact(receipt, "chosen_baseline") == "drift"
    assert _fact(receipt, "horizon") == 6
    assert _fact(receipt, "backtest_origins") == 6
    assert _fact(receipt, "backtest_mae") == 0.0
    assert _fact(receipt, "projection0.value") == pytest.approx(21.0)
    assert _fact(receipt, "projection5.value") == pytest.approx(26.0)
    # Descriptive projection: no hypothesis test, so no p-value on the ledger.
    assert receipt.statistics is None or receipt.statistics.p_value is None
    assert FORECAST_LIMITATION in receipt.method.assumptions
    assert receipt.method.family == "forecast_baseline"
    assert receipt.method.parameters["backtest_scheme"] == "rolling_origin"
    manifest = receipt.fact_manifest
    assert manifest is not None and manifest.total_rows == 6
    assert [entry.fact_id for entry in manifest.entries] == [
        f"projection{index}" for index in range(6)
    ]


def test_seasonal_series_selects_seasonal_naive() -> None:
    pattern = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
    frame = pd.DataFrame(
        {
            "day": pd.date_range("2024-01-01", periods=35, freq="D"),
            "amount": pattern * 5,
        }
    )
    context = _context(frame)

    result = _tool(context).execute(
        RunForecastArguments(
            dataset_id="ds_series", time_column="day", value_column="amount"
        )
    )

    content = cast(dict[str, Any], result.content)
    assert content["chosen_baseline"] == "seasonal_naive"
    baselines, projections = _rows(context)
    by_name = {row["baseline"]: row for row in baselines}
    assert by_name["seasonal_naive"]["backtest_mae"] == 0.0
    assert by_name["naive"]["backtest_mae"] > 0.0
    # The next cycle repeats the pattern exactly.
    assert [row["projected_value"] for row in projections] == [
        pytest.approx(value) for value in pattern[:6]
    ]


def test_short_history_fails_closed_without_projection() -> None:
    context = _context(_linear_frame(n=10))

    with pytest.raises(ValueError, match="history too short for a backtested baseline"):
        _tool(context).execute(
            RunForecastArguments(
                dataset_id="ds_series", time_column="day", value_column="amount"
            )
        )

    assert not [a for a in context.artifacts if a.type is ArtifactType.TABLE]


def test_rejects_bad_arguments() -> None:
    with pytest.raises(ValidationError):
        RunForecastArguments(
            dataset_id="ds", time_column="day", value_column="amount", horizon=0
        )
    with pytest.raises(ValidationError):
        RunForecastArguments(
            dataset_id="ds",
            time_column="day",
            value_column="amount",
            extra_field="nope",  # type: ignore[call-arg]
        )


def test_forecast_answer_contract_is_satisfied_by_the_tool_output() -> None:
    context = _context(_linear_frame())
    result = _tool(context).execute(
        RunForecastArguments(
            dataset_id="ds_series", time_column="day", value_column="amount"
        )
    )

    candidate = QuestionCandidate(
        question_id="q_forecast",
        question_en="If recent patterns continue, what is next month's volume?",
        origin="llm",
        analysis_mode="forecast",
        score=QuestionScore(
            data_availability=1.0,
            statistical_signal=0.5,
            quality_risk=0.1,
            join_risk=0.0,
            deterministic_score=0.8,
        ),
    )
    failure = _method_contract_failure(
        candidate,
        evidence_artifacts=cast(list[Artifact], result.artifacts),
        tool_names=["run_forecast"],
    )
    assert failure is None


def test_language_stays_descriptive_not_predictive() -> None:
    context = _context(_linear_frame())
    _tool(context).execute(
        RunForecastArguments(
            dataset_id="ds_series", time_column="day", value_column="amount"
        )
    )

    table = next(a for a in context.artifacts if a.type is ArtifactType.TABLE)
    description = AnalysisTable.model_validate(table.payload).description
    assert FORECAST_LIMITATION in description
    tool_description = _tool(context).description
    for text in (FORECAST_LIMITATION, description, tool_description):
        assert not implies_causation(text)
        assert not asserts_model_capability(text)
        assert "will reach" not in text.lower() and "predicts" not in text.lower()

    # The exact gate-ok wording is pinned in test_di_method_registry.
    assert METHOD_REGISTRY["forecast"].supported is True
