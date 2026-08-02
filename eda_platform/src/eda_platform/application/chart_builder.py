"""Custom-chart frame shaping and Vega-Lite spec assembly for the API."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd

ROW_COUNT_Y = "Row count"
CUSTOM_CHART_ROW_LIMIT = 5_000
# An aggregated chart accumulates one state per group while it streams, so a
# high-cardinality X (or X x color) is the unbounded-memory path. A plot with
# this many marks is unreadable anyway, so the request is refused rather than
# silently sampled down.
CUSTOM_CHART_GROUP_LIMIT = 1_000
# Median is the one aggregate that cannot be folded incrementally; it keeps
# every value. Bound the total retained across all groups.
CUSTOM_CHART_MEDIAN_VALUE_LIMIT = 1_000_000


def default_custom_agg(frame: pd.DataFrame, y_column: str) -> str:
    """Choose the default aggregation for a custom chart's Y column."""
    if y_column == ROW_COUNT_Y or y_column not in frame.columns:
        return "count"
    if pd.api.types.is_numeric_dtype(frame[y_column]):
        return "sum"
    return "count"


def select_chart_columns(
    frame: pd.DataFrame, selected_columns: list[str], *, drop_missing: bool
) -> pd.DataFrame:
    """The row-wise half of the shaping: safe to apply one chunk at a time."""
    # Guard against duplicate labels reaching .loc/.dropna (would yield a
    # DataFrame per label and crash); callers de-duplicate, this is defensive.
    unique_columns = list(dict.fromkeys(selected_columns))
    working = frame.loc[:, unique_columns].copy()
    if drop_missing:
        working = working.dropna(subset=unique_columns)
    return working


def apply_outlier_bounds(
    frame: pd.DataFrame, y_column: str, bounds: tuple[float, float]
) -> pd.DataFrame:
    """The whole-column half: ``bounds`` must come from the complete Y column,
    never from the slice being filtered."""
    lower, upper = bounds
    numeric = cast("pd.Series", pd.to_numeric(frame[y_column], errors="coerce"))
    return frame.loc[numeric.isna() | numeric.between(lower, upper)]


def custom_chart_spec(
    *,
    chart_type: str,
    x_column: str,
    y_column: str | None,
    color_column: str | None,
    aggregate: str,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    mark = "bar" if chart_type == "histogram" else chart_type
    spec: dict[str, Any] = {
        "mark": mark,
        "encoding": {
            "x": custom_x_encoding(frame, x_column, chart_type=chart_type),
        },
    }
    if chart_type == "histogram":
        spec["encoding"]["y"] = {"aggregate": "count", "type": "quantitative"}
    elif y_column is None or aggregate == "count":
        spec["encoding"]["y"] = {"aggregate": "count", "type": "quantitative"}
    else:
        y_encoding: dict[str, Any] = {
            "field": y_column,
            "type": vegalite_type(cast("pd.Series", frame[y_column])),
        }
        if aggregate != "none":
            y_encoding["aggregate"] = aggregate
            y_encoding["type"] = "quantitative"
        spec["encoding"]["y"] = y_encoding
    if color_column is not None:
        spec["encoding"]["color"] = {
            "field": color_column,
            "type": vegalite_type(cast("pd.Series", frame[color_column])),
        }
    return spec


def custom_x_encoding(
    frame: pd.DataFrame,
    column: str,
    *,
    chart_type: str,
) -> dict[str, Any]:
    encoding: dict[str, Any] = {
        "field": column,
        "type": vegalite_type(cast("pd.Series", frame[column])),
    }
    if chart_type == "histogram":
        encoding["bin"] = True
        encoding["type"] = "quantitative"
    return encoding


def vegalite_type(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "quantitative"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "temporal"
    parsed = pd.to_datetime(series.dropna().head(50), errors="coerce", format="mixed")
    if len(parsed) > 0 and float(parsed.notna().mean()) >= 0.8:
        return "temporal"
    return "nominal"
