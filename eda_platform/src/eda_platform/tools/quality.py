from __future__ import annotations

from eda_platform.core.ids import make_artifact_id
from eda_platform.schemas.artifacts import (
    Artifact,
    ArtifactType,
    DatasetProfile,
    QualityIssue,
    QualityIssueSet,
)


def scan_quality(
    profile_artifact: Artifact,
    *,
    project_id: str,
    session_id: str,
    high_missing_threshold: float = 20.0,
) -> Artifact:
    profile = DatasetProfile.model_validate(profile_artifact.payload)
    issues: list[QualityIssue] = []
    if profile.rows == 0:
        issues.append(
            QualityIssue(
                severity="critical",
                code="empty_dataset",
                message="Dataset contains a header but no data rows.",
                recommendation="Provide observed rows before using this dataset for analysis.",
                affected_count=0,
            )
        )
    for column, percent in profile.missing_percent.items():
        if percent > high_missing_threshold:
            issues.append(
                QualityIssue(
                    severity="warn",
                    code="high_missing",
                    column=column,
                    message=f"Column {column} has {percent:.2f}% missing values.",
                    recommendation="Review missingness before using this column in analysis.",
                    metric_value=percent,
                    metric_unit="percent",
                )
            )
    if profile.duplicate_rows > 0:
        duplicate_scope = profile.duplicate_scope_columns
        exact = (
            profile.exact_duplicate_rows == profile.duplicate_rows
            and profile.exact_duplicate_rows > 0
        )
        scope_note = (
            ""
            if not duplicate_scope or exact
            else f" across {len(duplicate_scope)} non-identifier field(s)"
        )
        issues.append(
            QualityIssue(
                severity="warn",
                code="duplicate_rows",
                message=(
                    f"Dataset contains {profile.duplicate_rows} "
                    f"{'exact ' if exact else 'potential '}duplicate row(s){scope_note}."
                ),
                recommendation=(
                    "Confirm the record grain and identifier semantics before deduplicating "
                    "or aggregating counts and sums."
                ),
                affected_count=profile.duplicate_rows,
            )
        )
    for column in profile.columns_detail:
        if column.missing_percent >= 100.0 and profile.rows > 0:
            issues.append(
                QualityIssue(
                    severity="critical",
                    code="empty_column",
                    column=column.name,
                    message=f"Column {column.name} is entirely empty (100% missing).",
                    recommendation="Drop this column or supply a data source that populates it.",
                    metric_value=column.missing_percent,
                    metric_unit="percent",
                )
            )
        if column.outlier_count > 0:
            issues.append(
                QualityIssue(
                    severity="warn",
                    code="outlier_detected",
                    column=column.name,
                    message=(
                        f"Column {column.name} has {column.outlier_count} IQR outlier value(s)."
                    ),
                    recommendation=(
                        "Inspect outliers before averaging or modeling; they may be errors "
                        "or genuine extremes."
                    ),
                    affected_count=column.outlier_count,
                )
            )
        if column.unique_count == 1 and profile.rows > 1:
            issues.append(
                QualityIssue(
                    severity="info",
                    code="constant_column",
                    column=column.name,
                    message=f"Column {column.name} has the same value in every non-null row.",
                    recommendation=(
                        "Exclude constant columns from modeling and correlation analysis."
                    ),
                )
            )
        if column.semantic_type == "id":
            issues.append(
                QualityIssue(
                    severity="info",
                    code="likely_id_column",
                    column=column.name,
                    message=f"Column {column.name} looks like an identifier.",
                    recommendation="Use identifier columns for keys, not as numeric measures.",
                )
            )
            if column.missing_count > 0:
                issues.append(
                    QualityIssue(
                        severity="warn",
                        code="id_missing",
                        column=column.name,
                        message=(
                            f"Identifier {column.name} is missing in "
                            f"{column.missing_count} row(s)."
                        ),
                        recommendation=(
                            "Resolve missing identifiers before joins, deduplication, or "
                            "entity-level aggregation."
                        ),
                        affected_count=column.missing_count,
                    )
                )
            observed_count = max(profile.rows - column.missing_count, 0)
            duplicate_identifier_count = max(observed_count - column.unique_count, 0)
            if duplicate_identifier_count > 0:
                issues.append(
                    QualityIssue(
                        severity="warn",
                        code="id_not_unique",
                        column=column.name,
                        message=(
                            f"Identifier {column.name} has at least "
                            f"{duplicate_identifier_count} repeated observed value(s)."
                        ),
                        recommendation=(
                            "Verify whether this field is a primary key, a foreign key, or "
                            "an event-level identifier before joining datasets."
                        ),
                        affected_count=duplicate_identifier_count,
                    )
                )
        if column.semantic_type == "categorical" and column.unique_percent >= 80:
            issues.append(
                QualityIssue(
                    severity="warn",
                    code="high_cardinality_category",
                    column=column.name,
                    message=(
                        f"Column {column.name} has high category cardinality "
                        f"({column.unique_percent:.2f}% unique)."
                    ),
                    recommendation="Group rare categories before using this field in charts.",
                    metric_value=column.unique_percent,
                    metric_unit="percent",
                )
            )
        if "date_parse_failure" in column.warnings:
            issues.append(
                QualityIssue(
                    severity="warn",
                    code="date_parse_failure",
                    column=column.name,
                    message=f"Column {column.name} has values that failed datetime parsing.",
                    recommendation="Standardize date formats before time-series analysis.",
                    affected_count=column.parse_failure_count or None,
                )
            )
        if (
            column.semantic_type == "numeric"
            and column.parse_failure_count > 0
        ):
            issues.append(
                QualityIssue(
                    severity="warn",
                    code="numeric_parse_failure",
                    column=column.name,
                    message=(
                        f"Column {column.name} has {column.parse_failure_count} non-empty "
                        "value(s) that failed numeric parsing."
                    ),
                    recommendation=(
                        "Inspect and normalize the failed values before numeric analysis."
                    ),
                    affected_count=column.parse_failure_count,
                )
            )
        if column.non_finite_count > 0:
            issues.append(
                QualityIssue(
                    severity="critical",
                    code="non_finite_numeric",
                    column=column.name,
                    message=(
                        f"Column {column.name} contains {column.non_finite_count} "
                        "infinite numeric value(s)."
                    ),
                    recommendation=(
                        "Replace or exclude infinities with an explicit, documented rule "
                        "before statistics, charts, or modeling."
                    ),
                    affected_count=column.non_finite_count,
                )
            )
        if column.whitespace_count > 0:
            issues.append(
                QualityIssue(
                    severity="info",
                    code="surrounding_whitespace",
                    column=column.name,
                    message=(
                        f"Column {column.name} contains {column.whitespace_count} value(s) "
                        "with leading or trailing whitespace."
                    ),
                    recommendation=(
                        "Trim surrounding whitespace before grouping, matching, or joining."
                    ),
                    affected_count=column.whitespace_count,
                )
            )
        if "mixed_type_string" in column.warnings:
            issues.append(
                QualityIssue(
                    severity="warn",
                    code="mixed_type_string",
                    column=column.name,
                    message=f"Column {column.name} mixes numeric-looking and text values.",
                    recommendation="Normalize this field before numeric analysis or grouping.",
                )
            )
    if not issues:
        issues.append(
            QualityIssue(
                severity="info",
                code="no_high_missing",
                message="No columns exceed the high-missing threshold.",
                recommendation="Continue with profiling and exploratory analysis.",
            )
        )
    issue_set = QualityIssueSet(dataset_id=profile.dataset_id, issues=issues)
    payload = issue_set.model_dump(mode="json")
    return Artifact(
        id=make_artifact_id("quality", payload),
        type=ArtifactType.QUALITY_ISSUE_SET,
        project_id=project_id,
        session_id=session_id,
        parents=[profile_artifact.id],
        payload=payload,
    )
