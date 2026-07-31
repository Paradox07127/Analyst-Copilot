from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from eda_platform.core.ids import make_dataset_id, stable_hash
from eda_platform.schemas.cleaning import (
    CleaningGuardrail,
    CleaningLineage,
    CleaningRecipe,
    CleaningTransform,
)
from eda_platform.tools.frame_stats import drop_iqr_outlier_rows, missing_percent_by_column
from eda_platform.tools.loader import load_csv

# Directory (relative to each uploaded file) where cleaned versions are written.
PRECLEANED_DIRNAME = "_precleaned"


@dataclass(frozen=True)
class PrecleanFrameResult:
    frame: pd.DataFrame
    dropped_missing_columns: list[str] = field(default_factory=list)
    dropped_missing_rows: int = 0
    dropped_outlier_rows: int = 0
    skipped_missing_column_drop: bool = False
    skipped_missing_row_drop: bool = False
    skipped_outlier_row_drop: bool = False

    @property
    def changed(self) -> bool:
        return bool(
            self.dropped_missing_columns
            or self.dropped_missing_rows
            or self.dropped_outlier_rows
        )

    @property
    def guard_triggered(self) -> bool:
        return bool(
            self.skipped_missing_column_drop
            or self.skipped_missing_row_drop
            or self.skipped_outlier_row_drop
        )


@dataclass(frozen=True)
class PrecleanReport:
    dataset: str
    dropped_missing_columns: list[str] = field(default_factory=list)
    dropped_missing_rows: int = 0
    dropped_outlier_rows: int = 0
    remaining_rows: int = 0
    remaining_columns: int = 0
    skipped_missing_column_drop: bool = False
    skipped_missing_row_drop: bool = False
    skipped_outlier_row_drop: bool = False

    @property
    def changed(self) -> bool:
        return bool(
            self.dropped_missing_columns
            or self.dropped_missing_rows
            or self.dropped_outlier_rows
        )

    @property
    def guard_triggered(self) -> bool:
        return bool(
            self.skipped_missing_column_drop
            or self.skipped_missing_row_drop
            or self.skipped_outlier_row_drop
        )


@dataclass(frozen=True)
class PrecleanBatch:
    reports: list[PrecleanReport] = field(default_factory=list)
    # Path the pipeline should ingest for each input (a cleaned new version
    # when anything changed, otherwise the untouched original).
    dataset_paths: list[Path] = field(default_factory=list)
    # New files written by pre-cleaning, so callers can clean them up.
    created_paths: list[Path] = field(default_factory=list)
    # CleaningRecipe recording each input's drops (columns/rows/outliers + thresholds +
    # raw lineage), aligned with ``dataset_paths``.
    recipes: list[CleaningRecipe | None] = field(default_factory=list)


def preclean_csv_files(
    file_paths: list[Path],
    *,
    clean_missing_values: bool,
    missing_threshold_percent: float,
    min_rows_keep_percent: float,
    drop_iqr_outliers: bool,
) -> PrecleanBatch:
    """Pre-clean each CSV non-destructively."""
    reports: list[PrecleanReport] = []
    dataset_paths: list[Path] = []
    created_paths: list[Path] = []
    recipes: list[CleaningRecipe | None] = []
    for path in file_paths:
        loaded = load_csv(path)
        frame = loaded.frame
        rows_before = int(len(frame))
        columns_before = int(len(frame.columns))
        result = preclean_frame(
            frame,
            clean_missing_values=clean_missing_values,
            missing_threshold_percent=missing_threshold_percent,
            min_rows_keep_percent=min_rows_keep_percent,
            drop_iqr_outliers=drop_iqr_outliers,
        )
        if result.changed:
            cleaned_path = path.parent / PRECLEANED_DIRNAME / path.name
            cleaned_path.parent.mkdir(parents=True, exist_ok=True)
            result.frame.to_csv(cleaned_path, index=False)
            dataset_paths.append(cleaned_path)
            created_paths.append(cleaned_path)
        else:
            dataset_paths.append(path)
        recipes.append(
            build_cleaning_recipe(
                result,
                source_dataset_id=make_dataset_id(path.name, loaded.record.content_hash),
                source_name=path.name,
                source_content_hash=loaded.record.content_hash,
                rows_before=rows_before,
                columns_before=columns_before,
                clean_missing_values=clean_missing_values,
                missing_threshold_percent=missing_threshold_percent,
                min_rows_keep_percent=min_rows_keep_percent,
                drop_iqr_outliers=drop_iqr_outliers,
            )
        )
        reports.append(
            PrecleanReport(
                dataset=path.name,
                dropped_missing_columns=result.dropped_missing_columns,
                dropped_missing_rows=result.dropped_missing_rows,
                dropped_outlier_rows=result.dropped_outlier_rows,
                remaining_rows=int(len(result.frame)),
                remaining_columns=int(len(result.frame.columns)),
                skipped_missing_column_drop=result.skipped_missing_column_drop,
                skipped_missing_row_drop=result.skipped_missing_row_drop,
                skipped_outlier_row_drop=result.skipped_outlier_row_drop,
            )
        )
    return PrecleanBatch(
        reports=reports,
        dataset_paths=dataset_paths,
        created_paths=created_paths,
        recipes=recipes,
    )


def preclean_frame(
    frame: pd.DataFrame,
    *,
    clean_missing_values: bool,
    missing_threshold_percent: float,
    min_rows_keep_percent: float,
    drop_iqr_outliers: bool,
) -> PrecleanFrameResult:
    working = frame.copy()
    original_rows = int(len(working))
    minimum_rows = _minimum_rows_to_keep(original_rows, min_rows_keep_percent)
    dropped_missing_columns: list[str] = []
    dropped_missing_rows = 0
    dropped_outlier_rows = 0
    skipped_missing_column_drop = False
    skipped_missing_row_drop = False
    skipped_outlier_row_drop = False

    if clean_missing_values and len(working.columns) > 0:
        threshold = _bounded_percent(missing_threshold_percent)
        missing_percent = missing_percent_by_column(working)
        missing_columns = [
            column for column in working.columns if missing_percent.get(column, 0.0) > threshold
        ]
        if missing_columns:
            if len(missing_columns) == len(working.columns):
                skipped_missing_column_drop = True
            else:
                dropped_missing_columns = [str(column) for column in missing_columns]
                working = working.drop(columns=missing_columns)

        # Recompute missingness after dropping columns.
        remaining_missing = missing_percent_by_column(working)
        row_drop_columns = [
            column for column in working.columns if remaining_missing.get(column, 0.0) > 0.0
        ]
        if row_drop_columns:
            row_cleaned = working.dropna(subset=row_drop_columns)
            removed_rows = int(len(working) - len(row_cleaned))
            if removed_rows > 0:
                if int(len(row_cleaned)) < minimum_rows:
                    skipped_missing_row_drop = True
                else:
                    working = row_cleaned
                    dropped_missing_rows = removed_rows

    if drop_iqr_outliers and len(working.columns) > 0:
        outlier_cleaned = drop_iqr_outlier_rows(working)
        removed_rows = int(len(working) - len(outlier_cleaned))
        if removed_rows > 0:
            if int(len(outlier_cleaned)) < minimum_rows:
                skipped_outlier_row_drop = True
            else:
                working = outlier_cleaned
                dropped_outlier_rows = removed_rows

    return PrecleanFrameResult(
        frame=working,
        dropped_missing_columns=dropped_missing_columns,
        dropped_missing_rows=dropped_missing_rows,
        dropped_outlier_rows=dropped_outlier_rows,
        skipped_missing_column_drop=skipped_missing_column_drop,
        skipped_missing_row_drop=skipped_missing_row_drop,
        skipped_outlier_row_drop=skipped_outlier_row_drop,
    )


def build_cleaning_recipe(
    result: PrecleanFrameResult,
    *,
    source_dataset_id: str,
    source_name: str,
    source_content_hash: str | None,
    rows_before: int,
    columns_before: int,
    clean_missing_values: bool,
    missing_threshold_percent: float,
    min_rows_keep_percent: float,
    drop_iqr_outliers: bool,
) -> CleaningRecipe | None:
    """Turn an applied pre-clean into a typed ``CleaningRecipe``."""
    threshold = _bounded_percent(missing_threshold_percent)
    min_rows_keep = _bounded_percent(min_rows_keep_percent)
    transforms: list[CleaningTransform] = []
    guardrails = _preclean_guardrails(result, threshold, min_rows_keep)
    if not result.changed and not guardrails:
        return None

    for column in result.dropped_missing_columns:
        transforms.append(
            CleaningTransform(
                transform_id=_preclean_transform_id("drop_column", column, threshold),
                type="drop_column",
                target_column=column,
                params={
                    "reason": "missing_over_threshold",
                    "missing_threshold_percent": threshold,
                },
                safety="lossy",
                reversible=False,
                description=(
                    f"Drop column '{column}' (over {threshold:.0f}% missing)."
                ),
            )
        )

    if result.dropped_missing_rows:
        transforms.append(
            CleaningTransform(
                transform_id=_preclean_transform_id("drop_missing_rows", "", threshold),
                type="drop_missing_rows",
                params={
                    "missing_threshold_percent": threshold,
                    "min_rows_keep_percent": min_rows_keep,
                },
                safety="lossy",
                reversible=False,
                expected_impact_rows=result.dropped_missing_rows,
                description=(
                    f"Drop {result.dropped_missing_rows} row(s) with missing values."
                ),
            )
        )

    if result.dropped_outlier_rows:
        transforms.append(
            CleaningTransform(
                transform_id=_preclean_transform_id("drop_outlier_rows", "", min_rows_keep),
                type="drop_outlier_rows",
                params={"method": "iqr", "min_rows_keep_percent": min_rows_keep},
                safety="lossy",
                reversible=False,
                expected_impact_rows=result.dropped_outlier_rows,
                description=(
                    f"Drop {result.dropped_outlier_rows} IQR-outlier row(s)."
                ),
            )
        )

    lineage = CleaningLineage(
        source_dataset_id=source_dataset_id,
        source_name=source_name,
        source_content_hash=source_content_hash,
        rows_before=rows_before,
        rows_after=int(len(result.frame)),
        columns_before=columns_before,
        columns_after=int(len(result.frame.columns)),
    )
    return CleaningRecipe(
        recipe_id=_preclean_recipe_id(
            source_dataset_id,
            source_content_hash,
            transforms,
            guardrails,
        ),
        dataset_id=source_dataset_id,
        source_version=1,
        transforms=transforms,
        guardrails=guardrails,
        created_by="precleaning",
        lineage=lineage,
    )


def _preclean_guardrails(
    result: PrecleanFrameResult,
    missing_threshold_percent: float,
    min_rows_keep_percent: float,
) -> list[CleaningGuardrail]:
    guardrails: list[CleaningGuardrail] = []
    if result.skipped_missing_column_drop:
        guardrails.append(
            CleaningGuardrail(
                code="missing_column_drop_would_remove_all_columns",
                message=(
                    "Skipped high-missing column drops because every column would "
                    "have been removed."
                ),
                params={"missing_threshold_percent": missing_threshold_percent},
            )
        )
    if result.skipped_missing_row_drop:
        guardrails.append(
            CleaningGuardrail(
                code="missing_row_drop_below_min_rows",
                message=(
                    "Skipped missing-value row drops because fewer than the "
                    "minimum retained rows threshold would remain."
                ),
                params={
                    "missing_threshold_percent": missing_threshold_percent,
                    "min_rows_keep_percent": min_rows_keep_percent,
                },
            )
        )
    if result.skipped_outlier_row_drop:
        guardrails.append(
            CleaningGuardrail(
                code="outlier_row_drop_below_min_rows",
                message=(
                    "Skipped IQR outlier row drops because fewer than the minimum "
                    "retained rows threshold would remain."
                ),
                params={"method": "iqr", "min_rows_keep_percent": min_rows_keep_percent},
            )
        )
    return guardrails


def _preclean_transform_id(kind: str, key: str, threshold: float) -> str:
    return f"clean_{stable_hash({'kind': kind, 'key': key, 'threshold': threshold})}"


def _preclean_recipe_id(
    source_dataset_id: str,
    source_content_hash: str | None,
    transforms: list[CleaningTransform],
    guardrails: list[CleaningGuardrail],
) -> str:
    # Deterministic: the same raw upload + same drops always yields the same id,
    # so re-running a run reuses the checkpoint instead of minting a new artifact.
    return "recipe_" + stable_hash(
        {
            "source": source_dataset_id,
            "hash": source_content_hash,
            "transforms": [transform.transform_id for transform in transforms],
            "guardrails": [guardrail.model_dump(mode="json") for guardrail in guardrails],
        }
    )


def _bounded_percent(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def _minimum_rows_to_keep(row_count: int, min_rows_keep_percent: float) -> int:
    if row_count <= 0:
        return 0
    fraction = _bounded_percent(min_rows_keep_percent) / 100.0
    return max(1, int(row_count * fraction + 0.999999))
