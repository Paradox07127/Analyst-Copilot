from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

CleaningTransformType = Literal[
    "trim_whitespace",
    "parse_numeric",
    "drop_duplicate_rows",
    "drop_rows",
    "drop_missing_rows",
    "drop_outlier_rows",
    "fill_missing",
    "drop_column",
    "clip_outliers",
    "flag_constant_column",
]

CleaningSafety = Literal["safe", "lossy"]
CleaningGuardrailCode = Literal[
    "missing_column_drop_would_remove_all_columns",
    "missing_row_drop_below_min_rows",
    "outlier_row_drop_below_min_rows",
]

# Lossy operation types are classified server-side, independent of payload labels.
LOSSY_TYPES: frozenset[str] = frozenset(
    {
        "parse_numeric",
        "drop_duplicate_rows",
        "drop_rows",
        "drop_missing_rows",
        "drop_outlier_rows",
        "fill_missing",
        "drop_column",
        "clip_outliers",
    }
)


def transform_is_lossy(transform: CleaningTransform) -> bool:
    """Return whether the operation type or declared tier marks a transform lossy."""
    return transform.type in LOSSY_TYPES or transform.safety == "lossy"


class CleaningTransform(BaseModel):
    transform_id: str = Field(default_factory=lambda: f"clean_{uuid4().hex[:12]}")
    type: CleaningTransformType
    target_column: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    safety: CleaningSafety = "safe"
    reversible: bool = True
    expected_impact_rows: int | None = None
    description: str = ""

    @field_validator("expected_impact_rows")
    @classmethod
    def _expected_impact_is_non_negative(cls, value: int | None) -> int | None:
        if value is None or value >= 0:
            return value
        raise ValueError("expected_impact_rows must be non-negative.")


class CleaningLineage(BaseModel):
    """Raw-dataset lineage for a recipe that produced a cleaned version."""

    source_dataset_id: str
    source_name: str
    source_content_hash: str | None = None
    rows_before: int
    rows_after: int
    columns_before: int
    columns_after: int

    @field_validator("source_dataset_id", "source_name")
    @classmethod
    def _required_strings_are_non_empty(cls, value: str) -> str:
        if value.strip():
            return value
        raise ValueError("field must be non-empty.")

    @field_validator("rows_before", "rows_after", "columns_before", "columns_after")
    @classmethod
    def _counts_are_non_negative(cls, value: int) -> int:
        if value >= 0:
            return value
        raise ValueError("count fields must be non-negative.")


class CleaningGuardrail(BaseModel):
    code: CleaningGuardrailCode
    message: str
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("message")
    @classmethod
    def _message_is_non_empty(cls, value: str) -> str:
        if value.strip():
            return value
        raise ValueError("message must be non-empty.")


class CleaningRecipe(BaseModel):
    dataset_id: str
    source_version: int = 1
    recipe_id: str = Field(default_factory=lambda: f"recipe_{uuid4().hex[:12]}")
    transforms: list[CleaningTransform] = Field(default_factory=list)
    guardrails: list[CleaningGuardrail] = Field(default_factory=list)
    created_by: Literal["deterministic", "llm", "user", "precleaning"] = "deterministic"
    # Set only after the recipe produces a new dataset version.
    lineage: CleaningLineage | None = None

    @field_validator("dataset_id")
    @classmethod
    def _dataset_id_is_non_empty(cls, value: str) -> str:
        if value.strip():
            return value
        raise ValueError("dataset_id must be non-empty.")

    @field_validator("source_version")
    @classmethod
    def _source_version_is_positive(cls, value: int) -> int:
        if value >= 1:
            return value
        raise ValueError("source_version must be at least 1.")

    @property
    def requires_approval(self) -> bool:
        return any(transform_is_lossy(transform) for transform in self.transforms)


class CleaningValueExample(BaseModel):
    before: str | int | float | bool | None
    after: str | int | float | bool | None


class CleaningColumnDiff(BaseModel):
    column: str
    before_dtype: str
    after_dtype: str
    before_missing: int
    after_missing: int
    changed_rows: int
    examples: list[CleaningValueExample] = Field(default_factory=list)

    @field_validator("before_missing", "after_missing", "changed_rows")
    @classmethod
    def _counts_are_non_negative(cls, value: int) -> int:
        if value >= 0:
            return value
        raise ValueError("diff counts must be non-negative.")


class CleaningPreview(BaseModel):
    dataset_id: str
    recipe_id: str
    source_version: int
    target_version: int
    row_count_before: int
    row_count_after: int
    # Detailed counters distinguish deletions from edits; affected_rows is legacy.
    affected_rows: int
    rows_dropped: int = 0
    rows_edited: int = 0
    cells_changed: int = 0
    column_diffs: list[CleaningColumnDiff] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @field_validator(
        "source_version",
        "target_version",
        "row_count_before",
        "row_count_after",
        "affected_rows",
        "rows_dropped",
        "rows_edited",
        "cells_changed",
    )
    @classmethod
    def _counts_are_non_negative(cls, value: int) -> int:
        if value >= 0:
            return value
        raise ValueError("preview counts must be non-negative.")


class CleaningApplyResult(BaseModel):
    dataset_id: str
    recipe_id: str
    source_version: int
    target_version: int
    output_path: Path
    row_count_before: int
    row_count_after: int
    applied_transform_ids: list[str] = Field(default_factory=list)

    @field_validator(
        "source_version",
        "target_version",
        "row_count_before",
        "row_count_after",
    )
    @classmethod
    def _counts_are_non_negative(cls, value: int) -> int:
        if value >= 0:
            return value
        raise ValueError("apply result counts must be non-negative.")
