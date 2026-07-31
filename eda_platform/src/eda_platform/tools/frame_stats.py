from __future__ import annotations

import math
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

# Shared frame-level statistics helpers.

DistributionKind = Literal["empty", "constant", "binary", "discrete", "continuous"]

# A numeric column with few distinct integral values is a category wearing an
# integer dtype (rating 1-5, weekday 0-6). Binning it into ranges produces the
# classic "two tall end bars, eight empty" chart.
_MAX_DISCRETE_LEVELS = 12
_MIN_HISTOGRAM_BINS = 5
_MAX_HISTOGRAM_BINS = 60
_LOG_SCALE_MIN_SKEW = 2.0


def distribution_kind(series: pd.Series) -> DistributionKind:
    """Classify what a column's values actually are, ignoring the dtype.

    Chart form, bin counts, and downstream statistics all key off this rather
    than off ``is_numeric_dtype``, which cannot tell 0/1 flags from measures.
    """
    non_null = series.dropna()
    if non_null.empty:
        return "empty"
    distinct = int(pd.Series(non_null.to_numpy()).nunique())
    if distinct <= 1:
        return "constant"
    if distinct == 2:
        return "binary"
    # Text is never binned into ranges, however many levels it has; how many
    # of those levels to display is a separate, cardinality-based decision.
    if not pd.api.types.is_numeric_dtype(series):
        return "discrete"
    numeric = cast(pd.Series, pd.to_numeric(non_null, errors="coerce")).dropna()
    if numeric.empty:
        return "discrete"
    values = numeric.to_numpy(dtype="float64")
    integral = bool(np.all(np.isfinite(values))) and bool(
        np.all(values == np.round(values))
    )
    if integral and distinct <= _MAX_DISCRETE_LEVELS:
        return "discrete"
    return "continuous"


def histogram_bin_count(series: pd.Series) -> int:
    """Freedman-Diaconis bin count, clamped to a legible range.

    A fixed bin count under-resolves large samples and over-resolves small
    ones; FD adapts to both spread and sample size. Sturges covers the case
    where the IQR is zero because the mass sits on one value.
    """
    numeric = cast(pd.Series, pd.to_numeric(series, errors="coerce")).dropna()
    values = numeric.to_numpy(dtype="float64")
    values = values[np.isfinite(values)]
    size = int(values.size)
    if size == 0:
        return 1
    spread = float(values.max() - values.min())
    if spread <= 0:
        return 1
    q1, q3 = np.percentile(values, [25, 75])
    iqr = float(q3 - q1)
    if iqr > 0:
        width = 2.0 * iqr / (size ** (1.0 / 3.0))
        bins = math.ceil(spread / width) if width > 0 else _MIN_HISTOGRAM_BINS
    else:
        bins = math.ceil(math.log2(size) + 1) if size > 1 else 1
    return max(_MIN_HISTOGRAM_BINS, min(_MAX_HISTOGRAM_BINS, int(bins)))


def prefers_log_scale(series: pd.Series) -> bool:
    """True when a strictly positive column is skewed enough that equal-width
    bins would collapse it into a single bar."""
    numeric = cast(pd.Series, pd.to_numeric(series, errors="coerce")).dropna()
    values = numeric.to_numpy(dtype="float64")
    values = values[np.isfinite(values)]
    if values.size < 8 or float(values.min()) <= 0:
        return False
    if float(values.max()) / float(values.min()) < 100.0:
        return False
    skew = float(cast(Any, pd.Series(values).skew()))
    return math.isfinite(skew) and abs(skew) >= _LOG_SCALE_MIN_SKEW


def missing_percent_by_column(frame: pd.DataFrame) -> dict[Any, float]:
    """Percent of missing values per column, keyed by the raw column label."""
    row_count = int(len(frame))
    if row_count == 0:
        return {column: 0.0 for column in frame.columns}
    return {
        column: round(int(cast(pd.Series, frame[column]).isna().sum()) / row_count * 100, 2)
        for column in frame.columns
    }


def drop_iqr_outlier_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that fall outside the 1.5*IQR fence of any numeric column."""
    keep_mask = pd.Series(True, index=frame.index)
    for column in frame.select_dtypes(include="number").columns:
        numeric = cast(pd.Series, pd.to_numeric(frame[column], errors="coerce"))
        bounds = iqr_bounds(numeric)
        if bounds is None:
            continue
        lower, upper = bounds
        keep_mask &= numeric.isna() | numeric.between(lower, upper)
    return frame.loc[keep_mask]


def iqr_bounds(series: pd.Series) -> tuple[float, float] | None:
    """Return the (lower, upper) 1.5*IQR fence for a numeric series."""
    numeric = cast(pd.Series, pd.to_numeric(series, errors="coerce"))
    finite = numeric.dropna()
    if len(finite) < 4:
        return None
    q1 = float(finite.quantile(0.25))
    q3 = float(finite.quantile(0.75))
    iqr = q3 - q1
    if iqr <= 0:
        return None
    return (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
