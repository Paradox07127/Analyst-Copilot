"""Build deterministic EDA quality context without inferring business causes."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd
from pandas.api.types import is_numeric_dtype

from eda_platform.core.ids import make_artifact_id, stable_hash
from eda_platform.schemas.artifacts import (
    Artifact,
    ArtifactType,
    ColumnProfile,
    DatasetProfile,
    QualityIssue,
    QualityIssueSet,
)
from eda_platform.schemas.quality_context import QualityContext, QualityContextSet
from eda_platform.tools.frame_stats import iqr_bounds
from eda_platform.tools.loader import LoadedDataset
from eda_platform.tools.value_parsing import parse_numeric_like

_SKIPPED_CODES = {"no_high_missing"}


def build_quality_context(
    loaded: LoadedDataset,
    profile_artifact: Artifact,
    quality_artifact: Artifact,
    *,
    project_id: str,
    session_id: str,
) -> Artifact:
    """Create a traceable quality-context artifact for one dataset."""
    profile = DatasetProfile.model_validate(profile_artifact.payload)
    issue_set = QualityIssueSet.model_validate(quality_artifact.payload)
    if profile.dataset_id != issue_set.dataset_id:
        raise ValueError("Profile and quality artifacts must describe the same dataset.")
    contexts = [
        _context_for_issue(
            issue,
            loaded=loaded,
            profile=profile,
            profile_artifact_id=profile_artifact.id,
            quality_artifact_id=quality_artifact.id,
        )
        for issue in issue_set.issues
        if issue.code not in _SKIPPED_CODES
    ]
    context_set = QualityContextSet(
        dataset_id=profile.dataset_id,
        dataset_name=profile.name,
        contexts=contexts,
    )
    payload = context_set.model_dump(mode="json")
    return Artifact(
        id=make_artifact_id("qualityctx", payload),
        type=ArtifactType.QUALITY_CONTEXT_SET,
        project_id=project_id,
        session_id=session_id,
        parents=[profile_artifact.id, quality_artifact.id],
        payload=payload,
        plain_language=(
            "EDA-grounded data conditions, their potential analytical impact, and "
            "follow-up questions. Causes remain unconfirmed until investigated."
        ),
    )


def _context_for_issue(
    issue: QualityIssue,
    *,
    loaded: LoadedDataset,
    profile: DatasetProfile,
    profile_artifact_id: str,
    quality_artifact_id: str,
) -> QualityContext:
    column_label = issue.column or "this dataset"
    return QualityContext(
        context_id="qctx_"
        + stable_hash(
            {
                "dataset_id": profile.dataset_id,
                "issue_code": issue.code,
                "column": issue.column,
                "message": issue.message,
            }
        ),
        dataset_id=profile.dataset_id,
        dataset_name=profile.name,
        issue_code=issue.code,
        severity=issue.severity,
        column=issue.column,
        observation=issue.message,
        pattern_facts=_pattern_facts(issue, loaded.frame, profile),
        analysis_impacts=_analysis_impacts(issue, column_label),
        open_questions=_open_questions(issue, column_label),
        validation_steps=_validation_steps(issue, column_label),
        report_limitation=_report_limitation(issue, column_label),
        requires_data=issue.code == "empty_column",
        source_artifact_ids=[profile_artifact_id, quality_artifact_id],
    )


def _pattern_facts(
    issue: QualityIssue,
    frame: pd.DataFrame,
    profile: DatasetProfile,
) -> list[str]:
    column = issue.column
    if issue.code in {"high_missing", "empty_column"} and column in frame:
        column_name = cast(str, column)
        mask = _frame_series(frame, column_name).isna()
        facts = [f"{int(cast(Any, mask.sum()))} of {len(frame)} rows are missing {column_name}."]
        return [*facts, *_group_pattern_facts(mask, frame, profile, exclude=column_name)]
    if issue.code == "outlier_detected" and column in frame:
        column_name = cast(str, column)
        mask = _iqr_outlier_mask(_frame_series(frame, column_name), column_name)
        facts = [
            f"{int(cast(Any, mask.sum()))} of {len(frame)} values are IQR outliers "
            f"in {column_name}."
        ]
        return [*facts, *_group_pattern_facts(mask, frame, profile, exclude=column_name)]
    if issue.code == "duplicate_rows":
        return [_duplicate_fact(profile)]
    if issue.code == "date_parse_failure" and column in frame:
        detail = _column_profile(profile, cast(str, column))
        if detail is None:
            return []
        return [
            f"{detail.parse_failure_count} non-empty values in {column} "
            "fail datetime parsing."
        ]
    if issue.code == "high_cardinality_category" and column in frame:
        unique = int(cast(Any, _frame_series(frame, cast(str, column)).nunique(dropna=True)))
        return [f"{column} has {unique} distinct non-null values in {len(frame)} rows."]
    if issue.code == "constant_column" and column in frame:
        unique = int(cast(Any, _frame_series(frame, cast(str, column)).nunique(dropna=True)))
        return [f"{column} has {unique} non-null value."]
    return []


def _group_pattern_facts(
    mask: pd.Series,
    frame: pd.DataFrame,
    profile: DatasetProfile,
    *,
    exclude: str,
) -> list[str]:
    categorical = next(
        (
            column.name
            for column in profile.columns_detail
            if column.name != exclude
            and column.semantic_type in {"categorical", "boolean"}
            and 2 <= column.unique_count <= 20
            and column.name in frame
        ),
        None,
    )
    if categorical is None:
        return []
    groups = _frame_series(frame, categorical).astype("string").fillna("[missing]")
    rates = cast(
        pd.Series,
        cast(Any, mask.groupby(groups, dropna=False).mean()).sort_values(ascending=False),
    )
    if rates.empty:
        return []
    group = str(rates.index[0])
    return [f"The highest affected rate is {rates.iloc[0] * 100:.1f}% for {categorical}={group}."]


def _duplicate_fact(profile: DatasetProfile) -> str:
    """Restate the profiler's duplicate count. Recomputing it here with a whole-row
    `frame.duplicated()` reported 0 whenever a surrogate key made every row
    unique, contradicting the issue that opened the context."""
    scope = profile.duplicate_scope_columns
    if profile.exact_duplicate_rows == profile.duplicate_rows or not scope:
        return f"{profile.duplicate_rows} rows are exact duplicates."
    return (
        f"{profile.duplicate_rows} rows repeat across {len(scope)} non-identifier "
        f"field(s): {', '.join(scope)}."
    )


def _column_profile(profile: DatasetProfile, column: str) -> ColumnProfile | None:
    return next((item for item in profile.columns_detail if item.name == column), None)


def _iqr_outlier_mask(series: pd.Series, column_name: str) -> pd.Series:
    """profiler._iqr_outlier_count as a mask, so the group breakdown below and the
    quality issue agree. Plain `to_numeric` turned parsed columns ("80,824") into
    all-NaN and reported zero outliers under an issue that found some."""
    if is_numeric_dtype(series):
        numeric = cast(pd.Series, pd.to_numeric(series, errors="coerce"))
    else:
        numeric = pd.Series(
            [parse_numeric_like(value, column_name=column_name) for value in series],
            index=series.index,
            dtype="float64",
        )
    bounds = iqr_bounds(cast(pd.Series, numeric.dropna()))
    if bounds is None:
        return pd.Series(False, index=series.index)
    lower, upper = bounds
    return cast(pd.Series, (numeric < lower) | (numeric > upper))


def _frame_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return cast(pd.Series, frame[column])


def _analysis_impacts(issue: QualityIssue, column: str) -> list[str]:
    if issue.code == "high_missing":
        return [f"Analyses using {column} may exclude a non-random subset of rows."]
    if issue.code == "empty_column":
        return [f"{column} cannot support an analysis because every value is missing."]
    if issue.code == "outlier_detected":
        return [f"Means, sums, and trend magnitudes using {column} may be sensitive to extremes."]
    if issue.code == "duplicate_rows":
        return ["Counts and sums may be inflated if duplicate rows represent repeat ingestion."]
    if issue.code == "mixed_type_string":
        return [f"Numeric or grouped analysis of {column} may change after normalization."]
    if issue.code == "date_parse_failure":
        return [f"Time-based analysis using {column} may omit or misplace unparsed values."]
    if issue.code == "high_cardinality_category":
        return [f"Group comparisons using {column} may be sparse or difficult to interpret."]
    if issue.code == "constant_column":
        return [f"{column} cannot explain differences because it does not vary."]
    if issue.code == "likely_id_column":
        return [f"{column} should be treated as an identifier rather than a numeric measure."]
    return [issue.recommendation]


def _open_questions(issue: QualityIssue, column: str) -> list[str]:
    if issue.code == "high_missing":
        return [f"Does missingness in {column} differ by time, group, or data-collection path?"]
    if issue.code == "outlier_detected":
        return [f"Do extreme {column} values represent valid rare events or data-quality problems?"]
    if issue.code == "duplicate_rows":
        return ["Do duplicate rows represent repeat ingestion, repeated events, or valid records?"]
    if issue.code == "date_parse_failure":
        return [f"Which source formats cause {column} parsing failures?"]
    return []


def _validation_steps(issue: QualityIssue, column: str) -> list[str]:
    if issue.code == "high_missing":
        return [
            f"Compare missing and non-missing {column} rows across available groups.",
            f"State the missingness rate whenever reporting a result that uses {column}.",
        ]
    if issue.code == "outlier_detected":
        return [
            f"Compare mean and median results for {column}.",
            "Compare the original result with a documented robust or trimmed sensitivity check.",
        ]
    if issue.code == "duplicate_rows":
        return [
            "Compare totals before and after a documented duplicate-handling sensitivity check."
        ]
    if issue.code == "date_parse_failure":
        return [f"Quantify how many {column} values are excluded from time-based analysis."]
    if issue.code == "mixed_type_string":
        return [
            f"Document the normalization rule before treating {column} as numeric or categorical."
        ]
    if issue.code == "empty_column":
        return [f"Collect or supply a populated source before using {column} in analysis."]
    return [issue.recommendation]


def _report_limitation(issue: QualityIssue, column: str) -> str:
    if issue.code == "empty_column":
        return f"{column} is entirely missing and cannot support this analysis."
    return (
        f"Interpretation involving {column} should account for the observed "
        f"{issue.code} condition; its business cause remains unconfirmed."
    )
