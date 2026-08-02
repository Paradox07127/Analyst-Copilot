"""profile_slice and analyze_time_series agent tools.

Both tools must validate arguments through closed Pydantic models, stay
read-only, persist an EvidenceReceipt whose digest verifies, and refuse (not
warn) when their preconditions fail: unsafe WHERE bodies, oversized slices,
fewer than two complete seasonal cycles, unparseable time columns.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pydantic import ValidationError

from eda_platform.agents.data_tools import (
    AnalyzeTimeSeriesArguments,
    DataToolContext,
    ProfileSliceArguments,
    build_data_tools,
)
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.datasets import DatasetRecord
from eda_platform.schemas.receipts import EvidenceReceipt, verify_receipt_digest
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


def _context(datasets: list[LoadedDataset]) -> DataToolContext:
    return DataToolContext(
        datasets=datasets,
        catalog=build_catalog(datasets),
        project_id="project_t",
        session_id="run_t",
        store=None,
        payload_policy="schema+aggregates",
        artifacts=[],
    )


def _tool(context: DataToolContext, name: str) -> Any:
    return next(tool for tool in build_data_tools(context) if tool.name == name)


def _last_receipt(context: DataToolContext) -> tuple[Artifact, EvidenceReceipt]:
    artifact = [
        a for a in context.artifacts if a.type is ArtifactType.EVIDENCE_RECEIPT
    ][-1]
    return artifact, EvidenceReceipt.model_validate(artifact.payload)


def _fact(receipt: EvidenceReceipt, fact_id: str) -> Any:
    return next(fact for fact in receipt.facts if fact.fact_id == fact_id)


# ---------------------------------------------------------------------------
# profile_slice
# ---------------------------------------------------------------------------


def _slice_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region": ["east"] * 50 + ["west"] * 50,
            "amount": [float(i) for i in range(100)],
            "flag": [None] * 100,
        }
    )


def _slice_context() -> DataToolContext:
    return _context([_dataset("sales.csv", _slice_frame(), "ds_sales")])


def test_profile_slice_produces_verified_receipt_and_table() -> None:
    context = _slice_context()
    tool = _tool(context, "profile_slice")
    result = tool.execute(
        ProfileSliceArguments(dataset_id="ds_sales", where_sql="region = 'east'")
    )

    table_artifact = next(a for a in context.artifacts if a.type is ArtifactType.TABLE)
    rows = {row["column"]: row for row in table_artifact.payload["rows"]}
    assert set(rows) == {"region", "amount", "flag"}
    assert rows["amount"]["mean"] == pytest.approx(24.5)
    assert rows["amount"]["median"] == pytest.approx(24.5)
    assert {"p1", "q1", "q3", "p95", "p99", "skew", "kurtosis"} <= set(rows["amount"])
    assert rows["region"]["unique_count"] == 1
    assert rows["region"]["distribution_kind"] == "constant"

    receipt_artifact, receipt = _last_receipt(context)
    assert verify_receipt_digest(receipt)
    assert receipt.tool_name == "profile_slice"
    assert receipt.artifact_ids == (table_artifact.id,)
    assert receipt_artifact.parents == [table_artifact.id]
    assert receipt.scope.filters == "region = 'east'"
    assert receipt.result_count == 50
    assert _fact(receipt, "rows_in_slice").value == 50
    assert _fact(receipt, "rows_total").value == 100
    assert _fact(receipt, "slice_share_percent").value == pytest.approx(50.0)
    assert _fact(receipt, "amount.mean").value == pytest.approx(24.5)
    assert isinstance(result.content, dict)
    assert result.content["rows_in_slice"] == 50


def test_profile_slice_without_where_covers_the_full_table() -> None:
    context = _slice_context()
    tool = _tool(context, "profile_slice")
    tool.execute(ProfileSliceArguments(dataset_id="ds_sales", columns=["amount"]))
    table_artifact = next(a for a in context.artifacts if a.type is ArtifactType.TABLE)
    assert [row["column"] for row in table_artifact.payload["rows"]] == ["amount"]
    _, receipt = _last_receipt(context)
    assert verify_receipt_digest(receipt)
    assert _fact(receipt, "slice_share_percent").value == pytest.approx(100.0)


def test_profile_slice_rejects_bad_arguments() -> None:
    with pytest.raises(ValidationError):
        ProfileSliceArguments(dataset_id="")
    with pytest.raises(ValidationError):
        ProfileSliceArguments(dataset_id="ds", columns=[])
    with pytest.raises(ValidationError):
        ProfileSliceArguments(dataset_id="ds", columns=[f"c{i}" for i in range(41)])
    with pytest.raises(ValidationError):
        ProfileSliceArguments(dataset_id="ds", nope=1)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "where_sql",
    [
        "amount > 1; select 1",
        "amount > 1 -- comment",
        "amount > 1 /* comment */",
        "amount in (select amount from slice_src)",
        "delete from slice_src",
    ],
)
def test_profile_slice_rejects_unsafe_where_sql(where_sql: str) -> None:
    context = _slice_context()
    tool = _tool(context, "profile_slice")
    with pytest.raises(ValueError):
        tool.execute(ProfileSliceArguments(dataset_id="ds_sales", where_sql=where_sql))
    assert not any(
        a.type is ArtifactType.EVIDENCE_RECEIPT for a in context.artifacts
    ), "a rejected slice must not leave a receipt behind"


def test_profile_slice_unknown_column_is_rejected() -> None:
    context = _slice_context()
    tool = _tool(context, "profile_slice")
    with pytest.raises(ValueError, match="not_a_column"):
        tool.execute(
            ProfileSliceArguments(dataset_id="ds_sales", columns=["not_a_column"])
        )


def test_profile_slice_empty_slice_returns_absence_receipt() -> None:
    context = _slice_context()
    tool = _tool(context, "profile_slice")
    result = tool.execute(
        ProfileSliceArguments(dataset_id="ds_sales", where_sql="amount > 1000000")
    )
    _, receipt = _last_receipt(context)
    assert verify_receipt_digest(receipt)
    assert receipt.result_count == 0
    assert _fact(receipt, "rows_in_slice").value == 0
    assert _fact(receipt, "empty_slice").support_type == "absence"
    assert not any(a.type is ArtifactType.TABLE for a in context.artifacts)
    assert isinstance(result.content, dict)
    assert result.content["rows_in_slice"] == 0


def test_profile_slice_all_nan_column_is_profiled_gracefully() -> None:
    context = _slice_context()
    tool = _tool(context, "profile_slice")
    tool.execute(
        ProfileSliceArguments(dataset_id="ds_sales", where_sql="region = 'west'")
    )
    table_artifact = next(a for a in context.artifacts if a.type is ArtifactType.TABLE)
    flag_row = next(r for r in table_artifact.payload["rows"] if r["column"] == "flag")
    assert flag_row["missing_percent"] == pytest.approx(100.0)
    assert flag_row["unique_count"] == 0
    assert flag_row["distribution_kind"] == "empty"
    assert "mean" not in flag_row


def test_profile_slice_row_cap_exceeded_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from eda_platform.tools import slice_profile

    monkeypatch.setattr(slice_profile, "_MAX_SLICE_ROWS", 10)
    context = _slice_context()
    tool = _tool(context, "profile_slice")
    with pytest.raises(ValueError, match="[Aa]ggregate"):
        tool.execute(
            ProfileSliceArguments(dataset_id="ds_sales", where_sql="region = 'east'")
        )


# ---------------------------------------------------------------------------
# analyze_time_series
# ---------------------------------------------------------------------------


def _ts_frame(periods: int = 56) -> pd.DataFrame:
    days = pd.date_range("2024-01-01", periods=periods, freq="D")
    values = [
        20.0 + 0.5 * i + 6.0 * math.sin(2.0 * math.pi * i / 7.0)
        for i in range(periods)
    ]
    return pd.DataFrame({"day": days.astype(str), "sales": values})


def _ts_context(frame: pd.DataFrame) -> DataToolContext:
    return _context([_dataset("daily.csv", frame, "ds_daily")])


def test_analyze_time_series_receipt_and_diagnostics() -> None:
    context = _ts_context(_ts_frame())
    tool = _tool(context, "analyze_time_series")
    result = tool.execute(
        AnalyzeTimeSeriesArguments(
            dataset_id="ds_daily",
            time_column="day",
            value_column="sales",
            freq="D",
            period=7,
        )
    )

    table_artifact = next(a for a in context.artifacts if a.type is ArtifactType.TABLE)
    receipt_artifact, receipt = _last_receipt(context)
    assert verify_receipt_digest(receipt)
    assert receipt.tool_name == "analyze_time_series"
    assert receipt.artifact_ids == (table_artifact.id,)
    assert receipt_artifact.parents == [table_artifact.id]

    assert _fact(receipt, "n_periods").value == 56
    assert _fact(receipt, "gap_count").value == 0
    assert _fact(receipt, "trend_direction").value == "increasing"
    assert _fact(receipt, "seasonal_strength").value == pytest.approx(1.0, abs=0.05)
    assert _fact(receipt, "ljung_box_p").value is not None
    assert _fact(receipt, "adf_p").value is not None
    assert _fact(receipt, "kpss_p").value is not None
    assert _fact(receipt, "stationarity_verdict").value in {
        "stationary",
        "non_stationary",
        "trend_stationary",
        "difference_stationary",
    }
    assert receipt.method.parameters["decomposition_performed"] is True
    assert receipt.statistics is not None
    assert receipt.statistics.test_name == "ljung_box"
    assert receipt.statistics.p_value == _fact(receipt, "ljung_box_p").value
    assert receipt.scope.time_range is not None
    assert isinstance(result.content, dict)
    assert result.content["stationarity_verdict"] == _fact(
        receipt, "stationarity_verdict"
    ).value


def test_analyze_time_series_rejects_bad_arguments() -> None:
    with pytest.raises(ValidationError):
        AnalyzeTimeSeriesArguments(
            dataset_id="ds", time_column="t", value_column="v", agg="max"  # type: ignore[arg-type]
        )
    with pytest.raises(ValidationError):
        AnalyzeTimeSeriesArguments(
            dataset_id="ds", time_column="t", value_column="v", period=1
        )
    with pytest.raises(ValidationError):
        AnalyzeTimeSeriesArguments(
            dataset_id="ds", time_column="t", value_column="v", nope=1  # type: ignore[call-arg]
        )


def test_analyze_time_series_refuses_decomposition_under_two_cycles() -> None:
    context = _ts_context(_ts_frame(periods=10))
    tool = _tool(context, "analyze_time_series")
    tool.execute(
        AnalyzeTimeSeriesArguments(
            dataset_id="ds_daily",
            time_column="day",
            value_column="sales",
            freq="D",
            period=7,
        )
    )
    _, receipt = _last_receipt(context)
    assert verify_receipt_digest(receipt)
    assert receipt.method.parameters["decomposition_performed"] is False
    assert any("2 complete cycles" in w for w in receipt.method.warnings)
    assert _fact(receipt, "seasonal_strength").value is None
    # The descriptive fallback must still name a trend.
    assert _fact(receipt, "trend_direction").value == "increasing"


def test_analyze_time_series_unparseable_time_column_is_clear_error() -> None:
    frame = pd.DataFrame(
        {"day": [f"not-a-date-{i}" for i in range(20)], "sales": [1.0] * 20}
    )
    context = _ts_context(frame)
    tool = _tool(context, "analyze_time_series")
    with pytest.raises(ValueError, match="parse"):
        tool.execute(
            AnalyzeTimeSeriesArguments(
                dataset_id="ds_daily", time_column="day", value_column="sales"
            )
        )


def test_analyze_time_series_all_nan_value_column_is_clear_error() -> None:
    frame = _ts_frame(periods=20)
    frame["sales"] = None
    context = _ts_context(frame)
    tool = _tool(context, "analyze_time_series")
    with pytest.raises(ValueError, match="numeric"):
        tool.execute(
            AnalyzeTimeSeriesArguments(
                dataset_id="ds_daily", time_column="day", value_column="sales"
            )
        )


def test_analyze_time_series_constant_series_is_indeterminate_not_a_crash() -> None:
    days = pd.date_range("2024-01-01", periods=30, freq="D")
    frame = pd.DataFrame({"day": days.astype(str), "sales": [5.0] * 30})
    context = _ts_context(frame)
    tool = _tool(context, "analyze_time_series")
    tool.execute(
        AnalyzeTimeSeriesArguments(
            dataset_id="ds_daily",
            time_column="day",
            value_column="sales",
            freq="D",
            period=7,
        )
    )
    _, receipt = _last_receipt(context)
    assert verify_receipt_digest(receipt)
    assert _fact(receipt, "stationarity_verdict").value == "indeterminate"
    assert _fact(receipt, "trend_direction").value == "flat"
    assert receipt.method.warnings, "the degenerate inputs must be disclosed"


def test_analyze_time_series_counts_gaps_and_infers_frequency() -> None:
    frame = _ts_frame(periods=21)
    frame = frame.drop(index=10).reset_index(drop=True)  # one missing day
    context = _ts_context(frame)
    tool = _tool(context, "analyze_time_series")
    tool.execute(
        AnalyzeTimeSeriesArguments(
            dataset_id="ds_daily", time_column="day", value_column="sales", period=7
        )
    )
    _, receipt = _last_receipt(context)
    assert verify_receipt_digest(receipt)
    assert _fact(receipt, "gap_count").value == 1
    assert _fact(receipt, "regular_frequency").value == "D"
    assert _fact(receipt, "n_periods").value == 21


def test_analyze_time_series_captures_kpss_interpolation_warning() -> None:
    from statsmodels.tools.sm_exceptions import InterpolationWarning

    days = pd.date_range("2024-01-01", periods=60, freq="D")
    frame = pd.DataFrame(
        {"day": days.astype(str), "sales": [float(i) for i in range(60)]}
    )
    context = _ts_context(frame)
    tool = _tool(context, "analyze_time_series")
    with warnings.catch_warnings():
        warnings.simplefilter("error", InterpolationWarning)
        tool.execute(
            AnalyzeTimeSeriesArguments(
                dataset_id="ds_daily",
                time_column="day",
                value_column="sales",
                freq="D",
                period=7,
            )
        )
    _, receipt = _last_receipt(context)
    assert verify_receipt_digest(receipt)
    assert _fact(receipt, "kpss_p").value is not None
    assert any("lookup table" in w for w in receipt.method.warnings)
