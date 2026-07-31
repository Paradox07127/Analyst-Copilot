from __future__ import annotations

from collections.abc import Callable
from typing import Any

from eda_platform.core.llm import StructuredLLM
from eda_platform.core.query import DuckDBQueryEngine, validate_select_statement
from eda_platform.core.tool_guard import (
    GuardViolation,
    ToolGuardError,
    check_enum,
    check_non_empty,
    raise_for_violations,
)
from eda_platform.schemas.plans import AnalysisPlan


def build_plan(
    message: str,
    *,
    llm: StructuredLLM,
    catalog_columns: dict[str, set[str]],
    value_context: dict[str, list[str]] | None = None,
    semantic_seeds: list[dict[str, str]] | None = None,
    engine: DuckDBQueryEngine | None = None,
    previous_error: str | None = None,
    on_guard_rejected: Callable[[ToolGuardError], None] | None = None,
) -> AnalysisPlan:
    error_context = previous_error
    for attempt in range(2):
        plan = _request_plan(
            message,
            llm=llm,
            catalog_columns=catalog_columns,
            value_context=value_context,
            semantic_seeds=semantic_seeds,
            previous_error=error_context,
        )
        try:
            _validate_plan(plan, catalog_columns, engine=engine)
        except ToolGuardError as exc:
            if on_guard_rejected is not None:
                on_guard_rejected(exc)
            if attempt == 0:
                error_context = exc.to_model_feedback()
                continue
            raise ValueError(f"Planner produced invalid SQL after retry: {exc}") from exc
        except ValueError as exc:
            if attempt == 0:
                error_context = str(exc)
                continue
            raise ValueError(f"Planner produced invalid SQL after retry: {exc}") from exc
        return plan
    raise AssertionError("unreachable")


def _request_plan(
    message: str,
    *,
    llm: StructuredLLM,
    catalog_columns: dict[str, set[str]],
    value_context: dict[str, list[str]] | None,
    semantic_seeds: list[dict[str, str]] | None,
    previous_error: str | None,
) -> AnalysisPlan:
    payload: dict[str, Any] = {
        "message": message,
        "catalog": {
            dataset: sorted(columns) for dataset, columns in sorted(catalog_columns.items())
        },
        "value_context": value_context or {},
        "semantic_seeds": semantic_seeds or [],
        "instructions": (
            "Create one read-only DuckDB SQL plan. Use only listed datasets and columns. "
            "Prefer a single-table aggregate unless the user explicitly requests otherwise. "
            "Set needs_approval=true for high-cost or cross-table work."
        ),
    }
    if previous_error is not None:
        payload["previous_error"] = previous_error
    return llm.structured(
        task="m3_build_plan",
        schema=AnalysisPlan,
        payload=payload,
    )


def _validate_plan(
    plan: AnalysisPlan,
    catalog_columns: dict[str, set[str]],
    *,
    engine: DuckDBQueryEngine | None,
) -> None:
    validate_plan_references(plan, catalog_columns)
    plan.sql = validate_select_statement(plan.sql)
    if engine is not None:
        engine.dry_run(plan.sql)


def validate_plan_references(plan: AnalysisPlan, catalog_columns: dict[str, set[str]]) -> None:
    guard_plan_references(plan, catalog_columns)


def guard_plan_references(plan: AnalysisPlan, catalog_columns: dict[str, set[str]]) -> None:
    scan_values = ("small", "medium", "large", "unknown")
    violations: list[GuardViolation | None] = [
        check_non_empty("dataset_names", plan.dataset_names),
        check_non_empty("columns", plan.columns),
        check_enum("estimated_scan", plan.estimated_scan, scan_values),
    ]
    if not catalog_columns:
        violations.append(
            GuardViolation(
                field="catalog",
                got=catalog_columns,
                allowed="at least one dataset with columns",
                fix_hint="Use a catalog that includes the datasets and columns before planning.",
                problem="Catalog is empty.",
            )
        )
        raise_for_violations("m3_build_plan", violations)

    datasets = plan.dataset_names or list(catalog_columns)
    unknown_datasets = [name for name in datasets if name not in catalog_columns]
    if unknown_datasets:
        violations.append(
            GuardViolation(
                field="dataset_names",
                got=sorted(unknown_datasets),
                allowed=", ".join(sorted(catalog_columns)),
                fix_hint="Use only dataset names from the provided catalog.",
                problem=f"Unknown dataset: {', '.join(sorted(unknown_datasets))}.",
            )
        )

    available_columns: set[str] = set()
    for dataset in datasets:
        if dataset in catalog_columns:
            available_columns.update(catalog_columns[dataset])

    unknown_columns = [column for column in plan.columns if column not in available_columns]
    if unknown_columns:
        violations.append(
            GuardViolation(
                field="columns",
                got=sorted(unknown_columns),
                allowed=", ".join(sorted(available_columns)),
                fix_hint="Replace hallucinated columns with exact names from the catalog.",
                problem=f"Unknown column: {', '.join(sorted(unknown_columns))}.",
            )
        )
    raise_for_violations("m3_build_plan", violations)
