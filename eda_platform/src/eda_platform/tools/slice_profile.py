"""Re-profile a WHERE-filtered slice of one dataset with bounded recomputation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd
from pandas.api.types import is_numeric_dtype

from eda_platform.core.query import DuckDBQueryEngine, UnsafeQueryError, validate_select_statement
from eda_platform.schemas.artifacts import AnalysisTable
from eda_platform.tools.frame_stats import distribution_kind, missing_percent_by_column

_MAX_SLICE_ROWS = 500_000
_SLICE_RELATION = "slice_src"

# The WHERE body is only ever a condition. Statement separators, comments and
# nested SELECTs are rejected outright (even inside string literals — a
# conservative gate) before the composed statement reaches
# ``validate_select_statement`` and the read-only DuckDB connection.
_WHERE_FORBIDDEN_SUBSTRINGS = (";", "--", "/*", "*/")
_WHERE_FORBIDDEN_WORDS = re.compile(r"\b(select|with)\b", re.IGNORECASE)


@dataclass(slots=True)
class SliceProfile:
    rows_total: int
    rows_in_slice: int
    slice_share_percent: float
    table: AnalysisTable | None


def validate_where_clause(where_sql: str) -> str:
    condition = where_sql.strip()
    if not condition:
        raise UnsafeQueryError("where_sql must be a non-empty WHERE clause body.")
    for token in _WHERE_FORBIDDEN_SUBSTRINGS:
        if token in condition:
            raise UnsafeQueryError(
                f"where_sql must be a bare WHERE condition; `{token}` is not allowed."
            )
    if _WHERE_FORBIDDEN_WORDS.search(condition):
        raise UnsafeQueryError(
            "where_sql must be a bare WHERE condition; subqueries are not allowed."
        )
    return condition


def compute_slice_profile(
    frame: pd.DataFrame,
    *,
    dataset_id: str,
    dataset_name: str,
    where_sql: str | None,
    columns: list[str] | None,
) -> SliceProfile:
    """Filter ``frame`` through a read-only DuckDB WHERE and re-profile the slice."""
    if columns is not None:
        known = {str(column) for column in frame.columns}
        missing = [column for column in columns if column not in known]
        if missing:
            raise ValueError(f"Columns not found in the dataset: {missing}.")
    condition = None if where_sql is None else validate_where_clause(where_sql)
    select_list = "*" if columns is None else ", ".join(
        '"' + column.replace('"', '""') + '"' for column in columns
    )
    where_clause = "" if condition is None else f" where ({condition})"
    engine = DuckDBQueryEngine(max_rows=_MAX_SLICE_ROWS)
    engine.register_frame(_SLICE_RELATION, frame)
    count_frame = engine.execute_select(
        f"select count(*) as n from {_SLICE_RELATION}{where_clause}"
    )
    rows_in_slice = int(count_frame.iloc[0]["n"])
    if rows_in_slice > _MAX_SLICE_ROWS:
        raise ValueError(
            f"The slice matches {rows_in_slice} rows, above the {_MAX_SLICE_ROWS} "
            "row limit; aggregate with run_sql first or narrow the WHERE condition."
        )
    rows_total = int(len(frame))
    share = round(rows_in_slice / rows_total * 100, 4) if rows_total else 0.0
    if rows_in_slice == 0:
        return SliceProfile(
            rows_total=rows_total,
            rows_in_slice=0,
            slice_share_percent=share,
            table=None,
        )
    statement = validate_select_statement(
        f"select {select_list} from {_SLICE_RELATION}{where_clause}"
    )
    slice_frame = engine.execute_select(statement)
    scope_note = "the full table" if condition is None else f"WHERE {condition}"
    table = AnalysisTable(
        dataset_id=dataset_id,
        title=f"{dataset_name} - Slice profile",
        kind="numeric_summary",
        description=(
            f"Per-column profile of the {rows_in_slice}-row slice of {dataset_name} "
            f"({scope_note}); {share}% of {rows_total} total rows."
        ),
        rows=_column_summaries(slice_frame),
    )
    return SliceProfile(
        rows_total=rows_total,
        rows_in_slice=rows_in_slice,
        slice_share_percent=share,
        table=table,
    )


def _column_summaries(slice_frame: pd.DataFrame) -> list[dict[str, Any]]:
    missing = missing_percent_by_column(slice_frame)
    rows: list[dict[str, Any]] = []
    for column in slice_frame.columns:
        series = cast(pd.Series, slice_frame[column])
        row: dict[str, Any] = {
            "column": str(column),
            "missing_percent": float(missing[column]),
            "unique_count": int(series.dropna().nunique()),
            "distribution_kind": distribution_kind(series),
        }
        if is_numeric_dtype(series):
            non_null = cast(pd.Series, pd.to_numeric(series, errors="coerce")).dropna()
            if not non_null.empty:
                p1, q1, median, q3, p95, p99 = (
                    float(value)
                    for value in non_null.quantile([0.01, 0.25, 0.5, 0.75, 0.95, 0.99])
                )
                row.update(
                    {
                        "mean": _round(non_null.mean()),
                        "median": _round(median),
                        "p1": _round(p1),
                        "q1": _round(q1),
                        "q3": _round(q3),
                        "p95": _round(p95),
                        "p99": _round(p99),
                        "skew": _round(non_null.skew()),
                        "kurtosis": _round(non_null.kurtosis()),
                    }
                )
        rows.append(row)
    return rows


def _round(value: Any, *, digits: int = 4) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), digits)
