"""Deterministic anomaly-screening result (method family: anomaly_detection).

Payloads stay bounded (top outliers capped) per the artifact granularity
policy; every number a claim cites must resolve to a locator in this payload.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

AnomalyMethod = Literal["robust_zscore", "iqr"]


class AnomalyOutlier(BaseModel):
    """One flagged observation, addressable for evidence locators."""

    row_index: int = Field(ge=0, description="Positional index in the screened frame")
    value: float
    score: float = Field(description="Robust z-score or IQR distance multiple")


class AnomalyScreenResult(BaseModel):
    """Outcome of screening one numeric column for robust statistical outliers."""

    schema_version: int = 1
    dataset_name: str
    column: str
    method: AnomalyMethod
    threshold: float = Field(gt=0)
    total_rows: int = Field(ge=0)
    non_null_rows: int = Field(ge=0)
    outlier_count: int = Field(ge=0)
    outlier_percent: float = Field(ge=0.0, le=100.0)
    median: float
    mad: float = Field(ge=0.0, description="Median absolute deviation")
    q1: float
    q3: float
    top_outliers: list[AnomalyOutlier] = Field(
        default_factory=list,
        max_length=10,
        description="Highest-score outliers only; payload stays bounded",
    )
    notes: list[str] = Field(default_factory=list)

    @field_validator("dataset_name", "column")
    @classmethod
    def _required_strings_are_non_empty(cls, value: str) -> str:
        if value.strip():
            return value
        raise ValueError("field must be non-empty.")
