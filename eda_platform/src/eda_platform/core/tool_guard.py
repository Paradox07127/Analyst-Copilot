from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal, cast

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_string_dtype,
)

ColumnSemanticType = Literal["numeric", "categorical", "datetime"]


@dataclass(frozen=True)
class GuardViolation:
    field: str
    got: Any
    allowed: str
    fix_hint: str
    problem: str

    def to_trace_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "got": _format_value(self.got),
            "allowed": self.allowed,
            "fix_hint": self.fix_hint,
            "problem": self.problem,
        }


class ToolGuardError(ValueError):
    def __init__(self, tool_name: str, violations: Sequence[GuardViolation]) -> None:
        self.tool_name = tool_name
        self.violations = tuple(violations)
        super().__init__(self.to_model_feedback())

    def to_model_feedback(self) -> str:
        wrong = [
            f"- `{violation.field}` got {_format_value(violation.got)}: {violation.problem}"
            for violation in self.violations
        ]
        allowed = [f"- `{violation.field}`: {violation.allowed}" for violation in self.violations]
        fixes = [f"- `{violation.field}`: {violation.fix_hint}" for violation in self.violations]
        return "\n".join(
            [
                f"Tool guard rejected parameters for `{self.tool_name}`.",
                "What was wrong:",
                *wrong,
                "Allowed:",
                *allowed,
                "How to fix:",
                *fixes,
                (
                    "Return corrected parameters that satisfy these constraints. "
                    "Do not compute tool results yourself."
                ),
            ]
        )

    def to_trace_summary(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "violation_count": len(self.violations),
            "violations": [violation.to_trace_dict() for violation in self.violations],
            "feedback": self.to_model_feedback(),
        }


def raise_for_violations(
    tool_name: str,
    violations: Iterable[GuardViolation | None],
) -> None:
    found = [violation for violation in violations if violation is not None]
    if found:
        raise ToolGuardError(tool_name, found)


def check_range(
    field: str,
    value: Any,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    fix_hint: str | None = None,
) -> GuardViolation | None:
    allowed = _range_allowed(minimum=minimum, maximum=maximum)
    hint = fix_hint or f"Set `{field}` to {allowed}."
    if isinstance(value, bool) or not isinstance(value, Real):
        return GuardViolation(
            field=field,
            got=value,
            allowed=allowed,
            fix_hint=hint,
            problem="expected a numeric value.",
        )
    numeric = float(value)
    if not math.isfinite(numeric):
        return GuardViolation(
            field=field,
            got=value,
            allowed=allowed,
            fix_hint=hint,
            problem="expected a finite numeric value.",
        )
    if minimum is not None and numeric < minimum:
        return GuardViolation(
            field=field,
            got=value,
            allowed=allowed,
            fix_hint=hint,
            problem=f"value is below the minimum {minimum}.",
        )
    if maximum is not None and numeric > maximum:
        return GuardViolation(
            field=field,
            got=value,
            allowed=allowed,
            fix_hint=hint,
            problem=f"value is above the maximum {maximum}.",
        )
    return None


def check_enum(
    field: str,
    value: Any,
    allowed_values: Sequence[str],
    *,
    fix_hint: str | None = None,
) -> GuardViolation | None:
    allowed = ", ".join(allowed_values)
    if value in allowed_values:
        return None
    return GuardViolation(
        field=field,
        got=value,
        allowed=allowed,
        fix_hint=fix_hint or f"Use one of: {allowed}.",
        problem="value is not one of the allowed enum options.",
    )


def check_non_empty(
    field: str,
    value: Any,
    *,
    fix_hint: str | None = None,
) -> GuardViolation | None:
    if value is None:
        empty = True
    elif isinstance(value, str):
        empty = not value.strip()
    else:
        try:
            empty = len(value) == 0
        except TypeError:
            empty = False
    if not empty:
        return None
    return GuardViolation(
        field=field,
        got=value,
        allowed="a non-empty value; lists must contain at least one item",
        fix_hint=fix_hint or f"Provide `{field}` with at least one item or character.",
        problem="value is empty.",
    )


def check_column_exists(
    field: str,
    column: Any,
    available_columns: Iterable[str],
    *,
    fix_hint: str | None = None,
) -> GuardViolation | None:
    columns = sorted({str(column_name) for column_name in available_columns})
    if isinstance(column, str) and column in columns:
        return None
    allowed = ", ".join(columns) if columns else "(no columns available)"
    return GuardViolation(
        field=field,
        got=column,
        allowed=allowed,
        fix_hint=fix_hint or f"Set `{field}` to exactly one available column name.",
        problem="column does not exist in the dataset schema.",
    )


_SQL_JOIN_RE = re.compile(r"\bjoin\b", re.IGNORECASE)


def check_sql_joins_declared(
    field: str,
    sql: Any,
    *,
    required_relations: Sequence[str],
    confirmed_joins: Iterable[str],
) -> list[GuardViolation]:
    """Reject SQL joins that are not covered by declared relations."""
    confirmed = {str(label) for label in confirmed_joins}
    allowed = ", ".join(sorted(confirmed)) or "(no confirmed joins in the whitelist)"
    violations: list[GuardViolation] = []
    sql_text = sql if isinstance(sql, str) else ""
    if _SQL_JOIN_RE.search(sql_text) and not required_relations:
        violations.append(
            GuardViolation(
                field=field,
                got="SQL containing a JOIN with no declared required_relations",
                allowed=(
                    "JOIN SQL only for questions declaring confirmed whitelist "
                    f"relations. Confirmed joins: {allowed}"
                ),
                fix_hint=(
                    "Declare the join in required_relations using a confirmed "
                    "whitelist label, or rewrite the SQL without a JOIN."
                ),
                problem="SQL joins tables but the question declares no required_relations.",
            )
        )
    for index, label in enumerate(required_relations):
        if label in confirmed:
            continue
        violations.append(
            GuardViolation(
                field=f"{field}.required_relations[{index}]",
                got=label,
                allowed=allowed,
                fix_hint=(
                    f"Use only confirmed join whitelist labels: {allowed}. "
                    "Ask the user to confirm the join on the Knowledge page, "
                    "or drop the relation and ask a single-table question."
                ),
                problem="required relation is not a confirmed join in the whitelist.",
            )
        )
    return violations


def check_column_semantic_type(
    field: str,
    column: Any,
    frame: pd.DataFrame,
    *,
    allowed_semantic_types: Sequence[ColumnSemanticType],
    fix_hint: str | None = None,
) -> GuardViolation | None:
    exists = check_column_exists(field, column, [str(name) for name in frame.columns])
    if exists is not None:
        return exists
    column_name = cast(str, column)
    semantic_type = infer_column_semantic_type(cast(pd.Series, frame[column_name]))
    if semantic_type in allowed_semantic_types:
        return None
    allowed = ", ".join(allowed_semantic_types)
    return GuardViolation(
        field=field,
        got=f"{column_name} ({semantic_type})",
        allowed=allowed,
        fix_hint=(fix_hint or f"Choose a column whose semantic type is one of: {allowed}."),
        problem=f"column semantic type is `{semantic_type}`.",
    )


def infer_column_semantic_type(series: pd.Series) -> ColumnSemanticType:
    if is_datetime64_any_dtype(series):
        return "datetime"
    if is_numeric_dtype(series) and not is_bool_dtype(series):
        return "numeric"
    if _looks_datetime_like(series):
        return "datetime"
    return "categorical"


def _looks_datetime_like(series: pd.Series) -> bool:
    if not (is_string_dtype(series) or series.dtype == object):
        return False
    non_null = series.dropna()
    if non_null.empty:
        return False
    parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
    return bool(float(parsed.notna().mean()) >= 0.8)


def _range_allowed(*, minimum: float | None, maximum: float | None) -> str:
    if minimum is not None and maximum is not None:
        return f"a number between {minimum} and {maximum}, inclusive"
    if minimum is not None:
        return f"a number greater than or equal to {minimum}"
    if maximum is not None:
        return f"a number less than or equal to {maximum}"
    return "a numeric value"


def _format_value(value: Any) -> str:
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, list | tuple | set):
        return repr(list(value))
    return repr(value)
