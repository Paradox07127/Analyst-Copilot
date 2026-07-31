"""Detect partial boundary periods in trend evidence."""

from __future__ import annotations

import re
from statistics import median

from pydantic import BaseModel, Field

from eda_platform.schemas.artifacts import SqlResult

DEFAULT_PARTIAL_RATIO = 0.5
DEFAULT_MIN_BUCKETS = 4

_PERIOD_RE = re.compile(
    r"^\s*(?P<year>\d{4})(?:[-/](?P<month>\d{1,2})(?:[-/](?P<day>\d{1,2}))?)?\s*$"
)
_COUNT_EXACT_NAMES = frozenset(
    {
        "count",
        "cnt",
        "n",
        "rows",
        "row_count",
        "record_count",
        "records",
        "freq",
        "frequency",
        "num",
        "size",
    }
)
_COUNT_AFFIX_RE = re.compile(r"(?:^(?:count|num|n)_|_(?:count|cnt|rows|n)$)")


class TimeBucket(BaseModel):
    """One period bucket of a time series with its record volume."""

    label: str
    sort_key: tuple[int, int, int] = (0, 0, 0)
    record_count: float | None = None


class TimeBoundaryAssessment(BaseModel):
    """Deterministic verdict over one time-bucketed series."""

    time_column: str = ""
    count_column: str | None = None
    bucket_labels: list[str] = Field(default_factory=list)
    partial_edge_labels: list[str] = Field(default_factory=list)
    interior_median_count: float | None = None
    partial_ratio_threshold: float = DEFAULT_PARTIAL_RATIO

    @property
    def flagged(self) -> bool:
        return bool(self.partial_edge_labels)

    @property
    def complete_labels(self) -> list[str]:
        partial = set(self.partial_edge_labels)
        return [label for label in self.bucket_labels if label not in partial]


def parse_period_label(value: object) -> tuple[int, int, int] | None:
    """Parse ``YYYY[-MM[-DD]]`` (``/`` also accepted) into a sortable key."""
    if not isinstance(value, str):
        return None
    match = _PERIOD_RE.fullmatch(value)
    if match is None:
        return None
    year = int(match.group("year"))
    if not 1900 <= year <= 2200:
        return None
    month = int(match.group("month") or 0)
    day = int(match.group("day") or 0)
    if month > 12 or day > 31:
        return None
    return (year, month, day)


def assess_buckets(
    buckets: list[TimeBucket],
    *,
    partial_ratio_threshold: float = DEFAULT_PARTIAL_RATIO,
    min_buckets: int = DEFAULT_MIN_BUCKETS,
) -> TimeBoundaryAssessment:
    """Flag partial first/last buckets against the interior median count."""
    ordered = sorted(buckets, key=lambda bucket: bucket.sort_key)
    labels = [bucket.label for bucket in ordered]
    assessment = TimeBoundaryAssessment(
        bucket_labels=labels,
        partial_ratio_threshold=partial_ratio_threshold,
    )
    if len(ordered) < min_buckets:
        return assessment
    interior_counts = [
        bucket.record_count
        for bucket in ordered[1:-1]
        if bucket.record_count is not None
    ]
    if not interior_counts:
        return assessment
    interior_median = float(median(interior_counts))
    assessment.interior_median_count = interior_median
    if interior_median <= 0:
        return assessment
    threshold = partial_ratio_threshold * interior_median
    for edge in (ordered[0], ordered[-1]):
        if edge.record_count is not None and edge.record_count < threshold:
            assessment.partial_edge_labels.append(edge.label)
    return assessment


def buckets_from_sql_result(sql_result: SqlResult) -> tuple[str, list[TimeBucket]] | None:
    """Extract ``(time_column, buckets)`` from a SQL preview, or None."""
    rows = sql_result.rows_preview
    if not rows:
        return None
    time_column: str | None = None
    for column in sql_result.columns:
        values = [row.get(column) for row in rows if row.get(column) is not None]
        if len(values) < 3:
            continue
        keys = [parse_period_label(value) for value in values]
        if all(key is not None for key in keys) and len(set(keys)) >= 3:
            time_column = column
            break
    if time_column is None:
        return None
    count_column = _count_column(sql_result.columns, exclude=time_column)
    buckets: list[TimeBucket] = []
    for row in rows:
        raw_label = row.get(time_column)
        key = parse_period_label(raw_label)
        if key is None:
            continue
        count = _as_count(row.get(count_column)) if count_column else None
        buckets.append(
            TimeBucket(label=str(raw_label).strip(), sort_key=key, record_count=count)
        )
    return time_column, buckets


def assess_sql_result(
    sql_result: SqlResult,
    *,
    partial_ratio_threshold: float = DEFAULT_PARTIAL_RATIO,
    min_buckets: int = DEFAULT_MIN_BUCKETS,
) -> TimeBoundaryAssessment | None:
    """One-call assessment: None when the result is not a time-bucketed series."""
    extracted = buckets_from_sql_result(sql_result)
    if extracted is None:
        return None
    time_column, buckets = extracted
    count_column = _count_column(sql_result.columns, exclude=time_column)
    assessment = assess_buckets(
        buckets,
        partial_ratio_threshold=partial_ratio_threshold,
        min_buckets=min_buckets,
    )
    assessment.time_column = time_column
    assessment.count_column = count_column
    return assessment


def split_edge_values(
    sql_result: SqlResult,
    assessment: TimeBoundaryAssessment,
) -> tuple[set[float], set[float]]:
    """Split preview cell numbers into (partial-bucket values, complete values)."""
    partial_labels = set(assessment.partial_edge_labels)
    partial_values: set[float] = set()
    complete_values: set[float] = set()
    for row in sql_result.rows_preview:
        raw_label = row.get(assessment.time_column)
        label = str(raw_label).strip() if raw_label is not None else ""
        target = partial_values if label in partial_labels else complete_values
        for value in row.values():
            if isinstance(value, bool):
                continue
            if isinstance(value, int | float):
                target.add(float(value))
    return partial_values, complete_values


def _count_column(columns: list[str], *, exclude: str) -> str | None:
    for column in columns:
        if column == exclude:
            continue
        lowered = column.strip().lower()
        if lowered in _COUNT_EXACT_NAMES or _COUNT_AFFIX_RE.search(lowered):
            return column
    return None


def _as_count(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
