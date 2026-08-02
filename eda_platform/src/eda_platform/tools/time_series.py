"""Time-series diagnostics: regular aggregation, decomposition, autocorrelation, stationarity."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from eda_platform.schemas.artifacts import AnalysisTable

TimeSeriesAgg = Literal["sum", "mean", "count"]
StationarityVerdict = Literal[
    "stationary",
    "non_stationary",
    "trend_stationary",
    "difference_stationary",
    "indeterminate",
]

_ALPHA = 0.05
_TIME_PARSE_MIN_SUCCESS_PERCENT = 80.0
_FLAT_RELATIVE_CHANGE = 0.05
_MAX_LJUNG_BOX_LAG = 10
# Level-dependent spread this strong reads as multiplicative seasonality;
# the additive decomposition then runs on log values (fpp3 §3.6).
_LOG_SPREAD_CORRELATION = 0.6

# Default seasonal period per pandas offset alias family; None = no meaningful
# sub-cycle at that granularity, so decomposition needs an explicit period.
_PERIOD_BY_FREQ_BASE = {
    "D": 7,
    "B": 5,
    "W": 52,
    "M": 12,
    "MS": 12,
    "ME": 12,
    "Q": 4,
    "QS": 4,
    "QE": 4,
    "H": 24,
    "MIN": 60,
    "T": 60,
}


@dataclass(slots=True)
class TimeSeriesDiagnostics:
    n_periods: int
    gap_count: int
    regular_frequency: str
    period: int | None
    decomposition_performed: bool
    log_transformed: bool
    trend_direction: Literal["increasing", "decreasing", "flat"]
    seasonal_strength: float | None
    ljung_box_p: float | None
    ljung_box_lag: int | None
    adf_p: float | None
    kpss_p: float | None
    stationarity_verdict: StationarityVerdict
    time_range: str
    warnings: list[str] = field(default_factory=list)
    table: AnalysisTable | None = None


def analyze_series(
    frame: pd.DataFrame,
    *,
    dataset_id: str,
    dataset_name: str,
    time_column: str,
    value_column: str,
    freq: str | None = None,
    period: int | None = None,
    agg: TimeSeriesAgg = "sum",
) -> TimeSeriesDiagnostics:
    """Aggregate one time/value pair into a regular series and diagnose it."""
    for column in (time_column, value_column):
        if column not in frame.columns:
            raise ValueError(f"Column `{column}` is not in dataset `{dataset_name}`.")
    notes: list[str] = []
    parsed_time = _parse_time(cast(pd.Series, frame[time_column]), time_column)
    values = cast(pd.Series, pd.to_numeric(frame[value_column], errors="coerce"))
    if agg != "count" and int(values.notna().sum()) == 0:
        raise ValueError(
            f"Value column `{value_column}` has no numeric values to aggregate."
        )
    data = (
        pd.DataFrame({"t": parsed_time, "v": values})
        .dropna(subset=["t"])
        .sort_values("t")
        .set_index("t")
    )
    if freq is None:
        freq = _infer_frequency(data.index)
        notes.append(f"Frequency was not supplied; inferred `{freq}` from the timestamps.")
    sizes = data.resample(freq).size()
    gap_count = int((sizes == 0).sum())
    if agg == "count":
        series = cast(pd.Series, sizes.astype("float64"))
    elif agg == "sum":
        series = cast(pd.Series, cast(pd.Series, data["v"]).resample(freq).sum(min_count=1))
    else:
        series = cast(pd.Series, cast(pd.Series, data["v"]).resample(freq).mean())
    n_periods = int(len(series))
    if bool(series.isna().any()):
        notes.append(
            f"{int(series.isna().sum())} empty/NaN period(s) were linearly "
            "interpolated for the diagnostics below."
        )
        series = series.interpolate(limit_direction="both")
    series = series.dropna()
    resolved_period = period if period is not None else _default_period(freq)
    if period is None and resolved_period is not None:
        notes.append(
            f"Seasonal period was not supplied; defaulted to {resolved_period} "
            f"for frequency `{freq}`."
        )

    values_now = series.to_numpy(dtype="float64")
    decomposed = None
    log_transformed = False
    if resolved_period is None:
        notes.append(
            f"No seasonal period is known for frequency `{freq}`; decomposition "
            "skipped, descriptive trend only."
        )
    elif len(series) < 2 * resolved_period:
        # statsmodels seasonal_decompose precondition, verbatim from its docs:
        # "x must contain 2 complete cycles."
        notes.append(
            f"Series has {len(series)} periods, fewer than 2 complete cycles of "
            f"period {resolved_period}; decomposition refused, descriptive trend only."
        )
    else:
        from statsmodels.tsa.seasonal import seasonal_decompose

        log_transformed = _prefers_log(series, resolved_period)
        target = cast(pd.Series, np.log(series)) if log_transformed else series
        if log_transformed:
            notes.append(
                "Values are strictly positive with level-dependent spread; the "
                "additive decomposition was run on log values."
            )
        decomposed = seasonal_decompose(target, model="additive", period=resolved_period)

    if decomposed is not None:
        trend_values = np.asarray(decomposed.trend, dtype="float64")
        trend_direction = _trend_direction(trend_values[np.isfinite(trend_values)])
        seasonal_strength = _seasonal_strength(decomposed)
    else:
        trend_direction = _trend_direction(values_now)
        seasonal_strength = None
    if decomposed is not None and seasonal_strength is None:
        notes.append("Seasonal strength is undefined (zero seasonal+residual variance).")

    ljung_box_p, ljung_box_lag = _ljung_box(values_now, notes)
    adf_p = _adf_p(values_now, notes)
    kpss_p = _kpss_p(values_now, notes)
    verdict = _stationarity_verdict(adf_p, kpss_p)
    if verdict == "indeterminate":
        notes.append(
            "ADF and/or KPSS could not be computed; stationarity is indeterminate."
        )
    start = cast(pd.Timestamp, series.index.min())
    end = cast(pd.Timestamp, series.index.max())
    time_range = f"{start.isoformat()}/{end.isoformat()}"

    result = TimeSeriesDiagnostics(
        n_periods=n_periods,
        gap_count=gap_count,
        regular_frequency=str(freq),
        period=resolved_period,
        decomposition_performed=decomposed is not None,
        log_transformed=log_transformed,
        trend_direction=trend_direction,
        seasonal_strength=seasonal_strength,
        ljung_box_p=ljung_box_p,
        ljung_box_lag=ljung_box_lag,
        adf_p=adf_p,
        kpss_p=kpss_p,
        stationarity_verdict=verdict,
        time_range=time_range,
        warnings=notes,
    )
    result.table = _diagnostics_table(
        result, dataset_id=dataset_id, dataset_name=dataset_name, agg=agg
    )
    return result


def _parse_time(series: pd.Series, time_column: str) -> pd.Series:
    non_null = series.dropna()
    if non_null.empty:
        raise ValueError(f"Time column `{time_column}` has no values.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.Series(
            pd.to_datetime(series, errors="coerce"), index=series.index
        )
    success_percent = float(parsed.loc[non_null.index].notna().mean()) * 100
    if success_percent < _TIME_PARSE_MIN_SUCCESS_PERCENT:
        raise ValueError(
            f"Time column `{time_column}` cannot be parsed as datetime "
            f"({success_percent:.1f}% of values parse)."
        )
    return parsed


def _infer_frequency(index: pd.Index) -> str:
    timestamps = pd.DatetimeIndex(index).drop_duplicates().sort_values()
    if len(timestamps) >= 3:
        inferred = pd.infer_freq(timestamps)
        if inferred is not None:
            return inferred
    if len(timestamps) < 2:
        raise ValueError("At least two distinct timestamps are needed to infer a frequency.")
    median_seconds = float(
        pd.Series(timestamps).diff().dropna().dt.total_seconds().median()
    )
    for threshold, alias in (
        (90.0, "min"),
        (5_400.0, "h"),
        (1.5 * 86_400.0, "D"),
        (10.0 * 86_400.0, "W"),
        (45.0 * 86_400.0, "MS"),
        (120.0 * 86_400.0, "QS"),
    ):
        if median_seconds <= threshold:
            return alias
    return "YS"


def _default_period(freq: str) -> int | None:
    base = freq.split("-")[0].upper()
    return _PERIOD_BY_FREQ_BASE.get(base)


def _prefers_log(series: pd.Series, period: int) -> bool:
    values = series.to_numpy(dtype="float64")
    if values.size < 2 * period or float(values.min()) <= 0:
        return False
    rolling = pd.Series(values).rolling(window=period)
    paired = pd.DataFrame({"mean": rolling.mean(), "std": rolling.std()}).dropna()
    spreads = cast(pd.Series, paired["std"])
    if len(paired) < 3 or float(spreads.max()) <= 0:
        return False
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = float(cast(pd.Series, paired["mean"]).corr(spreads))
    return np.isfinite(correlation) and correlation >= _LOG_SPREAD_CORRELATION


def _trend_direction(values: np.ndarray) -> Literal["increasing", "decreasing", "flat"]:
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return "flat"
    x = np.arange(finite.size, dtype="float64")
    slope = float(np.polyfit(x, finite, 1)[0])
    change = slope * (finite.size - 1)
    scale = max(abs(float(finite.mean())), float(finite.std()), 1e-9)
    if abs(change) < _FLAT_RELATIVE_CHANGE * scale:
        return "flat"
    return "increasing" if change > 0 else "decreasing"


def _seasonal_strength(decomposed: Any) -> float | None:
    """fpp3's variance-ratio strength: 1 - Var(remainder)/Var(seasonal+remainder)."""
    seasonal = np.asarray(decomposed.seasonal, dtype="float64")
    resid = np.asarray(decomposed.resid, dtype="float64")
    mask = np.isfinite(seasonal) & np.isfinite(resid)
    if int(mask.sum()) < 3:
        return None
    combined_variance = float(np.var(seasonal[mask] + resid[mask]))
    if combined_variance <= 0:
        return None
    strength = 1.0 - float(np.var(resid[mask])) / combined_variance
    return round(max(0.0, min(1.0, strength)), 6)


def _ljung_box(values: np.ndarray, notes: list[str]) -> tuple[float | None, int | None]:
    lag = min(_MAX_LJUNG_BOX_LAG, values.size // 5)
    if lag < 1 or float(np.nanstd(values)) <= 0:
        notes.append("Ljung-Box was skipped (series too short or constant).")
        return None, None
    from statsmodels.stats.diagnostic import acorr_ljungbox

    table = acorr_ljungbox(values, lags=[lag])
    p_value = float(table["lb_pvalue"].iloc[0])
    if not np.isfinite(p_value):
        notes.append("Ljung-Box produced a non-finite p-value on this series.")
        return None, lag
    return p_value, lag


def _adf_p(values: np.ndarray, notes: list[str]) -> float | None:
    from statsmodels.tsa.stattools import adfuller

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            p_value = float(adfuller(values)[1])
    except Exception as exc:  # constant/short series raise inside statsmodels
        notes.append(f"ADF test unavailable: {exc}")
        return None
    return p_value if np.isfinite(p_value) else None


def _kpss_p(values: np.ndarray, notes: list[str]) -> float | None:
    from statsmodels.tools.sm_exceptions import InterpolationWarning
    from statsmodels.tsa.stattools import kpss

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            p_value = float(kpss(values, regression="c", nlags="auto")[1])
        if any(issubclass(item.category, InterpolationWarning) for item in caught):
            notes.append(
                "KPSS p-value is outside the lookup table and was clipped to its boundary."
            )
    except Exception as exc:
        notes.append(f"KPSS test unavailable: {exc}")
        return None
    return p_value if np.isfinite(p_value) else None


def _stationarity_verdict(
    adf_p: float | None, kpss_p: float | None
) -> StationarityVerdict:
    """ADF+KPSS joint reading per the statsmodels stationarity notebook.

    ADF's null is a unit root; KPSS's null is stationarity. Both concluding
    stationary -> stationary; both non-stationary -> non_stationary; only KPSS
    stationary -> trend_stationary (detrend); only ADF stationary ->
    difference_stationary (difference).
    """
    if adf_p is None or kpss_p is None:
        return "indeterminate"
    adf_stationary = adf_p < _ALPHA
    kpss_stationary = kpss_p >= _ALPHA
    if adf_stationary and kpss_stationary:
        return "stationary"
    if not adf_stationary and not kpss_stationary:
        return "non_stationary"
    if kpss_stationary:
        return "trend_stationary"
    return "difference_stationary"


def _diagnostics_table(
    result: TimeSeriesDiagnostics,
    *,
    dataset_id: str,
    dataset_name: str,
    agg: TimeSeriesAgg,
) -> AnalysisTable:
    rows = [
        {"metric": name, "value": value}
        for name, value in (
            ("n_periods", result.n_periods),
            ("gap_count", result.gap_count),
            ("regular_frequency", result.regular_frequency),
            ("period", result.period),
            ("decomposition_performed", result.decomposition_performed),
            ("log_transformed", result.log_transformed),
            ("trend_direction", result.trend_direction),
            ("seasonal_strength", result.seasonal_strength),
            ("ljung_box_p", result.ljung_box_p),
            ("ljung_box_lag", result.ljung_box_lag),
            ("adf_p", result.adf_p),
            ("kpss_p", result.kpss_p),
            ("stationarity_verdict", result.stationarity_verdict),
            ("time_range", result.time_range),
        )
    ]
    return AnalysisTable(
        dataset_id=dataset_id,
        title=f"{dataset_name} - Time series diagnostics",
        kind="numeric_summary",
        description=(
            f"Diagnostics for {dataset_name} aggregated by {agg} at frequency "
            f"{result.regular_frequency}: trend {result.trend_direction}, "
            f"stationarity verdict {result.stationarity_verdict}."
        ),
        rows=rows,
    )
