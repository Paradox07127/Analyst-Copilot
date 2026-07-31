from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from eda_platform.core.ids import make_artifact_id
from eda_platform.core.query import DuckDBQueryEngine, QueryTimeout, validate_select_statement
from eda_platform.schemas.artifacts import Artifact, ArtifactType, SqlResult
from eda_platform.tools.loader import LoadedDataset

# A run of word characters, i.e. exactly what ``\bname\b`` would match.
_WORD_RUN = re.compile(r"[A-Za-z0-9_]+")


@dataclass(frozen=True)
class SqlCatalog:
    engine: DuckDBQueryEngine
    relations: dict[str, str]


def build_catalog(
    datasets: Sequence[LoadedDataset],
    *,
    max_rows: int = 10_000,
) -> SqlCatalog:
    engine = DuckDBQueryEngine(max_rows=max_rows)
    relations: dict[str, str] = {}
    used_relation_names: set[str] = set()
    for dataset in datasets:
        relation_name = _unique_relation_name(dataset.record.name, used_relation_names)
        engine.register_frame(relation_name, dataset.frame)
        relations.setdefault(dataset.record.name, relation_name)
        relations[dataset.record.dataset_id] = relation_name
    return SqlCatalog(engine=engine, relations=relations)


def relation_names_for(dataset_names: Sequence[str]) -> list[str]:
    """Relation names ``build_catalog`` would assign, without loading frames.

    Lets the API preview the SQL a skill replay will run (§10.3) while the
    frames stay out of the API process.
    """
    used: set[str] = set()
    return [_unique_relation_name(name, used) for name in dataset_names]


def rewrite_relation_names(sql: str, mapping: Mapping[str, str]) -> str:
    """Rebind relation names in a statement without touching string literals.

    A whole-word regex over the raw text also rewrites inside quotes: a skill
    holding ``SELECT 'orders' AS source ... FROM orders`` replayed on sales.csv
    silently returned rows labelled ``'sales'``, and ``WHERE channel = 'orders'``
    changed which rows were selected (review J1). Scan instead, and substitute
    only in code — plus in double-quoted identifiers, which are relation
    references, not data.
    """
    renames = {src: dst for src, dst in mapping.items() if src and src != dst}
    if not renames:
        return sql
    out: list[str] = []
    index = 0
    length = len(sql)
    while index < length:
        char = sql[index]
        if char == "'":
            end = _end_of_quoted(sql, index, "'")
            out.append(sql[index:end])
        elif char == '"':
            end = _end_of_quoted(sql, index, '"')
            inner = sql[index + 1 : end - 1]
            out.append(f'"{renames[inner]}"' if inner in renames else sql[index:end])
        elif sql.startswith("--", index):
            end = sql.find("\n", index)
            end = length if end == -1 else end
            out.append(sql[index:end])
        elif sql.startswith("/*", index):
            end = sql.find("*/", index + 2)
            end = length if end == -1 else end + 2
            out.append(sql[index:end])
        elif char.isalnum() or char == "_":
            match = _WORD_RUN.match(sql, index)
            assert match is not None  # the current char is already a word char
            end = match.end()
            out.append(renames.get(match.group(0), match.group(0)))
        else:
            end = index + 1
            out.append(char)
        index = end
    return "".join(out)


def run_sql(
    catalog: SqlCatalog,
    sql: str,
    *,
    project_id: str,
    session_id: str,
    preview_rows: int = 50,
    timeout_seconds: float = 10.0,
    output_units: Mapping[str, str] | None = None,
) -> Artifact:
    if timeout_seconds <= 0:
        raise QueryTimeout("Query exceeded the configured wall-clock timeout.")
    statement = validate_select_statement(sql)
    row_count = _execute_with_timeout(
        catalog.engine,
        f"select count(*) as __row_count from ({statement}) as q",
        timeout_seconds=timeout_seconds,
    )
    total_rows = int(row_count.iloc[0]["__row_count"])
    preview = _execute_with_timeout(
        catalog.engine,
        f"select * from ({statement}) as q limit {preview_rows + 1}",
        timeout_seconds=timeout_seconds,
    )
    truncated = total_rows > preview_rows or len(preview) > preview_rows
    preview = preview.head(preview_rows)
    result = SqlResult(
        sql=statement,
        columns=list(preview.columns),
        dtypes={column: str(dtype) for column, dtype in preview.dtypes.items()},
        units=dict(output_units or {}),
        rows_preview=_records(preview),
        row_count=total_rows,
        truncated=truncated,
    )
    payload = result.model_dump(mode="json")
    return Artifact(
        id=make_artifact_id("sql", {"session_id": session_id, "sql": statement}),
        type=ArtifactType.SQL_RESULT,
        project_id=project_id,
        session_id=session_id,
        payload=payload,
    )


def _execute_with_timeout(
    engine: DuckDBQueryEngine,
    sql: str,
    *,
    timeout_seconds: float,
) -> pd.DataFrame:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(engine.execute_select, sql)
        try:
            return future.result(timeout=timeout_seconds)
        except FutureTimeout as exc:
            engine.interrupt()
            raise QueryTimeout("Query exceeded the configured wall-clock timeout.") from exc


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(key): _json_value(value) for key, value in record.items()}
        for record in frame.to_dict("records")
    ]


def _json_value(value: Any) -> Any:
    # Arrays first: DuckDB LIST/ARRAY cells arrive as ndarray/list, and
    # ``pd.isna`` on those returns an element-wise mask that raises when used
    # as a bool — a plain ``list(x)`` aggregate crashed the job (review J3).
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _end_of_quoted(sql: str, start: int, quote: str) -> int:
    """Index just past the closing quote, honouring SQL's doubled-quote escape."""
    index = start + 1
    length = len(sql)
    while index < length:
        if sql[index] == quote:
            if index + 1 < length and sql[index + 1] == quote:
                index += 2
                continue
            return index + 1
        index += 1
    return length


def _unique_relation_name(filename: str, used: set[str]) -> str:
    base = filename.rsplit(".", maxsplit=1)[0]
    cleaned = "".join(char if char.isalnum() or char == "_" else "_" for char in base.strip())
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"t_{cleaned}"
    candidate = cleaned
    suffix = 2
    while candidate in used:
        candidate = f"{cleaned}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate
