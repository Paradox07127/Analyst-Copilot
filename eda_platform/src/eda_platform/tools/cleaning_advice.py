"""Cleaning operations the scanned quality issues actually call for.

The cleaning form offers a fixed set of options that has no connection to what
the profiler found, so a column reported as "12 values failed numeric parsing"
had no reachable operation that would fix it. This maps findings to the
operations the engine already supports, and says why each one is proposed.
"""

from __future__ import annotations

from typing import Any

from eda_platform.schemas.artifacts import DatasetProfile, QualityIssueSet
from eda_platform.schemas.cleaning import LOSSY_TYPES

# One quality code to the operation that addresses it. Codes with no safe
# mechanical fix (id_missing, non_finite_numeric, empty_dataset) are absent on
# purpose: they need a decision, not a transform.
_CODE_TO_OPERATION: dict[str, str] = {
    "surrounding_whitespace": "trim_whitespace",
    "numeric_parse_failure": "parse_numeric",
    "duplicate_rows": "drop_duplicate_rows",
    "empty_column": "drop_column",
    "high_missing": "drop_column",
}


def recommended_cleaning_operations(
    profile: DatasetProfile,
    issue_set: QualityIssueSet,
) -> list[dict[str, Any]]:
    """Deterministic operation proposals derived from this dataset's issues.

    Proposals only: nothing here applies a transform, and every lossy entry is
    marked so the caller keeps the existing preview-and-approve gate.
    """
    known_columns = {column.name for column in profile.columns_detail}
    seen: set[tuple[str, str | None]] = set()
    proposals: list[dict[str, Any]] = []
    for issue in issue_set.issues:
        operation = _CODE_TO_OPERATION.get(issue.code)
        if operation is None:
            continue
        if issue.column is not None and issue.column not in known_columns:
            continue
        key = (operation, issue.column)
        if key in seen:
            continue
        seen.add(key)
        proposals.append(
            {
                "operation": operation,
                "column": issue.column,
                "reason": issue.message,
                "severity": issue.severity,
                "affected_count": issue.affected_count,
                "lossy": operation in LOSSY_TYPES,
            }
        )
    # Severity first so the destructive proposals a user must think about are
    # not buried under whitespace trims.
    order = {"critical": 0, "warn": 1, "info": 2}
    proposals.sort(key=lambda item: (order.get(str(item["severity"]), 3), str(item["operation"])))
    return proposals
