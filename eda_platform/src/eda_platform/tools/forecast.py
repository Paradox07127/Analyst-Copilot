"""Descriptive baseline projection: fpp3 simple baselines picked by rolling-origin backtest.

Candidates are the fpp3 §5.2 baselines (naive, seasonal naive, drift, mean);
selection is rolling-forecasting-origin backtest MAE (fpp3 §5.10), and the
projection band is the empirical 10%/90% quantile of backtest residuals — a
historical error band, never a parametric prediction interval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from eda_platform.schemas.artifacts import AnalysisTable
from eda_platform.tools.time_series import (
    TimeSeriesAgg,
    prepare_regular_series,
)

FORECAST_LIMITATION = (
    "This is a descriptive baseline projection (naive/seasonal-naive/drift/mean), "
    "not a forecasting model. It assumes recent patterns continue and carries no "
    "confidence guarantee."
)

# Fixed candidate order also breaks backtest-MAE ties deterministically.
_CANDIDATE_ORDER = ("naive", "seasonal_naive", "drift", "mean")
_MIN_BACKTEST_ORIGINS = 5
_BAND_QUANTILES = (0.1, 0.9)


@dataclass(slots=True)
class BaselineBacktest:
    baseline: str
    mae: float
    rmse: float
    mase: float | None
    origins: int


@dataclass(slots=True)
class ProjectionPeriod:
    period: str
    value: float
    band_low: float
    band_high: float


@dataclass(slots=True)
class ForecastBaselineResult:
    chosen_baseline: str
    horizon: int
    n_periods: int
    gap_count: int
    regular_frequency: str
    period: int | None
    backtest_origins: int
    backtests: list[BaselineBacktest]
    projections: list[ProjectionPeriod]
    projection_start: str
    projection_end: str
    time_range: str
    warnings: list[str]
    table: AnalysisTable | None = None


def run_forecast_baselines(
    frame: pd.DataFrame,
    *,
    dataset_id: str,
    dataset_name: str,
    time_column: str,
    value_column: str,
    freq: str | None = None,
    period: int | None = None,
    agg: TimeSeriesAgg = "sum",
    horizon: int = 6,
) -> ForecastBaselineResult:
    """Backtest the fpp3 baselines on one aggregated series and project the winner."""
    prepared = prepare_regular_series(
        frame,
        dataset_name=dataset_name,
        time_column=time_column,
        value_column=value_column,
        freq=freq,
        period=period,
        agg=agg,
    )
    series = prepared.series
    values = series.to_numpy(dtype="float64")
    n = int(values.size)
    resolved_period = prepared.period

    candidates = ["naive", "drift", "mean"]
    seasonal_usable = (
        resolved_period is not None and resolved_period >= 2 and n >= 2 * resolved_period
    )
    if seasonal_usable:
        candidates.insert(1, "seasonal_naive")
    # The seasonal window only constrains the training start when seasonal
    # naive actually competes; otherwise short histories still get the
    # non-seasonal baselines.
    initial = max(2 * cast(int, resolved_period), 8) if seasonal_usable else 8
    origins = n - initial
    if origins < _MIN_BACKTEST_ORIGINS:
        raise ValueError(
            f"Series has {n} period(s) after aggregation, history too short for a "
            f"backtested baseline (needs at least {initial + _MIN_BACKTEST_ORIGINS} "
            f"periods at frequency `{prepared.freq}`)."
        )

    errors: dict[str, list[float]] = {name: [] for name in candidates}
    residuals_by_step: dict[str, dict[int, list[float]]] = {name: {} for name in candidates}
    for origin in range(initial, n):
        steps = min(horizon, n - origin)
        train = values[:origin]
        actual = values[origin : origin + steps]
        for name in candidates:
            projected = _baseline_forecast(name, train, steps, resolved_period)
            step_errors = actual - projected
            errors[name].extend(float(error) for error in step_errors)
            per_step = residuals_by_step[name]
            for step, error in enumerate(step_errors, start=1):
                per_step.setdefault(step, []).append(float(error))

    # fpp3 §5.8 MASE scale: in-sample one-step naive MAE over the full history.
    naive_scale = float(np.mean(np.abs(np.diff(values)))) if n > 1 else 0.0
    backtests: list[BaselineBacktest] = []
    for name in _CANDIDATE_ORDER:
        if name not in candidates:
            continue
        pooled = np.asarray(errors[name], dtype="float64")
        mae = float(np.mean(np.abs(pooled)))
        rmse = float(np.sqrt(np.mean(pooled**2)))
        mase = round(mae / naive_scale, 6) if naive_scale > 0 else None
        backtests.append(
            BaselineBacktest(
                baseline=name,
                mae=round(mae, 6),
                rmse=round(rmse, 6),
                mase=mase,
                origins=origins,
            )
        )
    chosen = min(backtests, key=lambda backtest: backtest.mae)

    projected = _baseline_forecast(chosen.baseline, values, horizon, resolved_period)
    future_index = pd.date_range(
        cast(pd.Timestamp, series.index[-1]), periods=horizon + 1, freq=prepared.freq
    )[1:]
    chosen_steps = residuals_by_step[chosen.baseline]
    max_step = max(chosen_steps)
    projections: list[ProjectionPeriod] = []
    for index, (timestamp, value) in enumerate(zip(future_index, projected, strict=True)):
        # Steps past the deepest backtested horizon reuse its residual pool.
        residuals = np.asarray(chosen_steps[min(index + 1, max_step)], dtype="float64")
        low_q, high_q = np.quantile(residuals, _BAND_QUANTILES)
        projections.append(
            ProjectionPeriod(
                period=cast(pd.Timestamp, timestamp).isoformat(),
                value=round(float(value), 6),
                band_low=round(float(value + low_q), 6),
                band_high=round(float(value + high_q), 6),
            )
        )

    start = cast(pd.Timestamp, series.index.min())
    end = cast(pd.Timestamp, series.index.max())
    result = ForecastBaselineResult(
        chosen_baseline=chosen.baseline,
        horizon=horizon,
        n_periods=prepared.n_periods,
        gap_count=prepared.gap_count,
        regular_frequency=prepared.freq,
        period=resolved_period,
        backtest_origins=origins,
        backtests=backtests,
        projections=projections,
        projection_start=projections[0].period,
        projection_end=projections[-1].period,
        time_range=f"{start.isoformat()}/{end.isoformat()}",
        warnings=list(prepared.notes),
    )
    result.table = _forecast_table(
        result, dataset_id=dataset_id, dataset_name=dataset_name, agg=agg
    )
    return result


def _baseline_forecast(
    name: str, train: np.ndarray, steps: int, period: int | None
) -> np.ndarray:
    if name == "mean":
        return np.full(steps, float(train.mean()))
    if name == "naive":
        return np.full(steps, float(train[-1]))
    if name == "drift":
        slope = (float(train[-1]) - float(train[0])) / (train.size - 1)
        return float(train[-1]) + slope * np.arange(1, steps + 1, dtype="float64")
    if name == "seasonal_naive":
        cycle = cast(int, period)
        indexes = [train.size - cycle + ((step - 1) % cycle) for step in range(1, steps + 1)]
        return train[indexes].astype("float64")
    raise ValueError(f"unknown baseline {name!r}")


def _forecast_table(
    result: ForecastBaselineResult,
    *,
    dataset_id: str,
    dataset_name: str,
    agg: TimeSeriesAgg,
) -> AnalysisTable:
    rows: list[dict[str, object]] = [
        {
            "baseline": backtest.baseline,
            "backtest_mae": backtest.mae,
            "backtest_rmse": backtest.rmse,
            "mase": backtest.mase,
            "origins": backtest.origins,
        }
        for backtest in result.backtests
    ]
    rows.extend(
        {
            "period": projection.period,
            "projected_value": projection.value,
            "band_low": projection.band_low,
            "band_high": projection.band_high,
        }
        for projection in result.projections
    )
    return AnalysisTable(
        dataset_id=dataset_id,
        title=f"{dataset_name} - Baseline backtest and projection",
        kind="numeric_summary",
        description=(
            f"Baseline `{result.chosen_baseline}` had the lowest rolling-origin backtest "
            f"MAE over {result.backtest_origins} origins ({agg} at frequency "
            f"{result.regular_frequency}); it is projected {result.horizon} period(s) "
            "ahead with an empirical 10%-90% band from historical backtest residuals. "
            f"{FORECAST_LIMITATION}"
        ),
        rows=rows,
    )
