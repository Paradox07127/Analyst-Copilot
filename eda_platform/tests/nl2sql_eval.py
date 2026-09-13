from __future__ import annotations

import math
from collections import Counter
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from eda_platform.core.llm import StructuredLLM
from eda_platform.drivers.chat import run_chat_turn
from eda_platform.schemas.artifacts import ArtifactType, SqlResult
from eda_platform.tools.loader import LoadedDataset


@dataclass(frozen=True)
class NL2SQLEvalCase:
    name: str
    question: str
    expected_rows_preview: list[dict[str, Any]]
    expected_validation_status: Literal["pass", "warn", "fail"] = "pass"


@dataclass(frozen=True)
class NL2SQLEvalOutcome:
    name: str
    passed: bool
    validation_status: str
    actual_sql: str
    actual_rows_preview: list[dict[str, Any]]
    expected_rows_preview: list[dict[str, Any]]
    message: str


def run_nl2sql_eval_case(
    case: NL2SQLEvalCase,
    *,
    datasets: Sequence[LoadedDataset],
    llm: StructuredLLM,
    project_id: str = "eval_project",
    session_id: str = "eval_run",
    preview_rows: int = 50,
) -> NL2SQLEvalOutcome:
    chat_result = run_chat_turn(
        case.question,
        datasets=datasets,
        project_id=project_id,
        session_id=session_id,
        llm=llm,
        preview_rows=preview_rows,
    )
    sql_artifact = next(
        (
            artifact
            for artifact in chat_result.artifacts
            if artifact.type is ArtifactType.SQL_RESULT
        ),
        None,
    )
    actual_rows: list[dict[str, Any]] = []
    if sql_artifact is not None:
        actual_rows = SqlResult.model_validate(sql_artifact.payload).rows_preview
    validation_status = (
        chat_result.validation.status if chat_result.validation is not None else "not_run"
    )
    passed = (
        validation_status == case.expected_validation_status
        and _canonical_row_multiset(actual_rows)
        == _canonical_row_multiset(case.expected_rows_preview)
    )
    return NL2SQLEvalOutcome(
        name=case.name,
        passed=passed,
        validation_status=validation_status,
        actual_sql=chat_result.sql or "",
        actual_rows_preview=actual_rows,
        expected_rows_preview=case.expected_rows_preview,
        message=chat_result.message,
    )


def _canonical_row_multiset(
    rows: list[dict[str, Any]],
) -> Counter[tuple[tuple[str, Hashable], ...]]:
    return Counter(_canonical_row(row) for row in rows)


def _canonical_row(row: dict[str, Any]) -> tuple[tuple[str, Hashable], ...]:
    return tuple(
        sorted((str(column), _normalize_value(value)) for column, value in row.items())
    )


def _normalize_value(value: Any) -> Hashable:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int | float):
        numeric = float(value)
        if math.isnan(numeric):
            return None
        return round(numeric, 6)
    if isinstance(value, dict):
        return tuple(
            sorted((str(column), _normalize_value(nested)) for column, nested in value.items())
        )
    if isinstance(value, list):
        return tuple(_normalize_value(item) for item in value)
    return value if isinstance(value, Hashable) else str(value)
