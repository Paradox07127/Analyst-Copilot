"""Re-profile a WHERE-filtered slice of one dataset with bounded recomputation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, cast

import duckdb
import pandas as pd
from pandas.api.types import is_numeric_dtype

from eda_platform.core.query import DuckDBQueryEngine, UnsafeQueryError, validate_select_statement
from eda_platform.schemas.artifacts import AnalysisTable
from eda_platform.tools.frame_stats import distribution_kind, missing_percent_by_column

_MAX_SLICE_ROWS = 500_000
# Post-resolution scope bounds: the column cap applies to the ACTUAL scanned
# columns (columns=None must not bypass it), and the cell cap bounds the
# rows x columns projection before the slice is materialized.
_MAX_SLICE_COLUMNS = 40
_MAX_SLICE_CELLS = 5_000_000
_SLICE_RELATION = "slice_src"

# Cheap pre-filter only: it fails fast with a readable message on the obvious
# shapes (statement separators, comments, subqueries), but it is not the safety
# boundary — ``_reject_unless_plain_filter`` below is. Matches inside string
# literals too, which is conservative and acceptable for a WHERE body.
_WHERE_FORBIDDEN_SUBSTRINGS = (";", "--", "/*", "*/")
_WHERE_FORBIDDEN_WORDS = re.compile(r"\b(select|with)\b", re.IGNORECASE)

# Clauses that a SELECT node may carry which would change *which rows* the
# profile is computed over while leaving the separately-counted `rows_in_slice`
# fact untouched.
_WHERE_NODE_EMPTY_FIELDS = ("modifiers", "group_expressions", "group_sets")
_WHERE_NODE_NULL_FIELDS = ("having", "qualify", "sample")


@dataclass(slots=True)
class SliceProfile:
    rows_total: int
    rows_in_slice: int
    slice_share_percent: float
    table: AnalysisTable | None
    # The columns actually scanned (post columns=None resolution); receipt
    # scopes must record this, never an empty tuple standing in for "all".
    resolved_columns: list[str] = field(default_factory=list)


def validate_where_clause(where_sql: str) -> str:
    """Accept ``where_sql`` only if DuckDB parses the composed statement as a plain filter.

    Keyword blacklists cannot see FROM-first set operations (``1=1) union all
    (from generate_series(1,5)``), so DuckDB's own parser is the authority.
    """
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
    _reject_unless_plain_filter(_parse_where_probe(condition))
    return condition


def _unsafe_where(reason: str) -> UnsafeQueryError:
    return UnsafeQueryError(f"where_sql must be a bare WHERE condition; {reason}.")


def _parse_where_probe(condition: str) -> dict[str, Any]:
    """Return the parsed query node of ``select 1 from <relation> where (condition)``."""
    probe = f"select 1 from {_SLICE_RELATION} where ({condition})"
    connection = duckdb.connect(config={"enable_external_access": False})
    try:
        row = connection.execute("select json_serialize_sql(?)", [probe]).fetchone()
    except duckdb.Error as exc:
        raise _unsafe_where(f"DuckDB could not parse it ({exc})") from exc
    finally:
        connection.close()
    payload = json.loads(row[0]) if row and row[0] else {}
    if payload.get("error"):
        raise _unsafe_where("DuckDB could not parse it")
    statements = payload.get("statements") or []
    if len(statements) != 1:
        raise _unsafe_where("it must compose into exactly one statement")
    node = statements[0].get("node")
    if not isinstance(node, dict):
        raise _unsafe_where("DuckDB could not parse it")
    return node


def _reject_unless_plain_filter(node: dict[str, Any]) -> None:
    if node.get("type") != "SELECT_NODE":
        raise _unsafe_where("it turns the query into a set operation")
    if node.get("cte_map", {}).get("map"):
        raise _unsafe_where("common table expressions are not allowed")
    from_table = node.get("from_table") or {}
    if (
        from_table.get("type") != "BASE_TABLE"
        or from_table.get("table_name") != _SLICE_RELATION
        or from_table.get("sample") is not None
    ):
        raise _unsafe_where("it must not change what the query reads from")
    if node.get("where_clause") is None:
        raise _unsafe_where("it must parse as the WHERE condition itself")
    for name in _WHERE_NODE_EMPTY_FIELDS:
        if node.get(name):
            raise _unsafe_where("it must not add clauses beyond the WHERE condition")
    for name in _WHERE_NODE_NULL_FIELDS:
        if node.get(name) is not None:
            raise _unsafe_where("it must not add clauses beyond the WHERE condition")
    if node.get("aggregate_handling") != "STANDARD_HANDLING":
        raise _unsafe_where("it must not aggregate")
    if _contains_subquery(node["where_clause"]):
        raise _unsafe_where("subqueries are not allowed")


def _contains_subquery(expression: Any) -> bool:
    if isinstance(expression, dict):
        if expression.get("class") == "SUBQUERY":
            return True
        return any(_contains_subquery(value) for value in expression.values())
    if isinstance(expression, list):
        return any(_contains_subquery(item) for item in expression)
    return False


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
    resolved_columns = (
        [str(column) for column in frame.columns] if columns is None else list(columns)
    )
    if len(resolved_columns) > _MAX_SLICE_COLUMNS:
        raise ValueError(
            f"The slice would scan {len(resolved_columns)} columns, above the "
            f"{_MAX_SLICE_COLUMNS} column limit; pass an explicit `columns` list of at "
            f"most {_MAX_SLICE_COLUMNS} columns."
        )
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
    projected_cells = rows_in_slice * len(resolved_columns)
    if projected_cells > _MAX_SLICE_CELLS:
        max_columns = max(1, _MAX_SLICE_CELLS // max(rows_in_slice, 1))
        max_rows = max(1, _MAX_SLICE_CELLS // len(resolved_columns))
        raise ValueError(
            f"The slice would materialize {projected_cells} cells "
            f"({rows_in_slice} rows x {len(resolved_columns)} columns), above the "
            f"{_MAX_SLICE_CELLS} cell limit. Narrow `columns` to at most {max_columns} "
            f"column(s) for this row count, or tighten the WHERE condition to at most "
            f"{max_rows} rows for these columns."
        )
    rows_total = int(len(frame))
    share = round(rows_in_slice / rows_total * 100, 4) if rows_total else 0.0
    if rows_in_slice == 0:
        return SliceProfile(
            rows_total=rows_total,
            rows_in_slice=0,
            slice_share_percent=share,
            table=None,
            resolved_columns=resolved_columns,
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
        resolved_columns=resolved_columns,
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
