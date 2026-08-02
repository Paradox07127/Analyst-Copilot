from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field, field_validator

TaskType = Literal["classification", "regression"]
SplitStrategy = Literal["random", "random_stratified", "time_ordered", "group"]


class LeakageCheck(BaseModel):
    code: str
    severity: Literal["info", "warn", "critical"]
    column: str | None = None
    action: Literal["passed", "excluded", "warned"]
    message: str


class FeatureImportance(BaseModel):
    feature: str
    # Backward-compatible non-negative display score. The signed estimate and
    # repeat dispersion below carry the analytical meaning.
    importance: float
    signed_importance: float | None = None
    importance_std: float | None = None

    @field_validator("importance")
    @classmethod
    def _importance_is_non_negative(cls, value: float) -> float:
        if math.isfinite(value) and value >= 0.0:
            return value
        raise ValueError("feature importance must be finite and non-negative.")

    @field_validator("signed_importance")
    @classmethod
    def _signed_importance_is_finite(cls, value: float | None) -> float | None:
        if value is None or math.isfinite(value):
            return value
        raise ValueError("signed feature importance must be finite.")

    @field_validator("importance_std")
    @classmethod
    def _importance_std_is_non_negative(cls, value: float | None) -> float | None:
        if value is None or (math.isfinite(value) and value >= 0.0):
            return value
        raise ValueError("feature importance std must be finite and non-negative.")


class ModelCard(BaseModel):
    dataset_id: str
    task_type: TaskType
    target_column: str
    feature_columns: list[str]
    excluded_features: list[str] = Field(default_factory=list)
    split_strategy: SplitStrategy
    train_rows: int
    test_rows: int
    model_type: str
    metrics: dict[str, float] = Field(default_factory=dict)
    # Train-set majority-class frequency; None for regression and legacy payloads.
    baseline_accuracy: float | None = None
    leakage_checks: list[LeakageCheck] = Field(default_factory=list)
    feature_importance: list[FeatureImportance] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @field_validator("target_column", "model_type")
    @classmethod
    def _required_strings_are_non_empty(cls, value: str) -> str:
        if value.strip():
            return value
        raise ValueError("field must be non-empty.")

    @field_validator("feature_columns")
    @classmethod
    def _feature_columns_are_non_empty(cls, value: list[str]) -> list[str]:
        if value and all(column.strip() for column in value):
            return value
        raise ValueError("feature_columns must contain at least one non-empty column.")

    @field_validator("train_rows", "test_rows")
    @classmethod
    def _row_counts_are_non_negative(cls, value: int) -> int:
        if value >= 0:
            return value
        raise ValueError("row counts must be non-negative.")

    @field_validator("baseline_accuracy")
    @classmethod
    def _baseline_accuracy_is_a_rate(cls, value: float | None) -> float | None:
        if value is None or (math.isfinite(value) and 0.0 <= value <= 1.0):
            return value
        raise ValueError("baseline_accuracy must be in [0.0, 1.0].")

    @field_validator("metrics")
    @classmethod
    def _metrics_are_in_valid_ranges(cls, value: dict[str, float]) -> dict[str, float]:
        for name, metric in value.items():
            if not math.isfinite(metric):
                raise ValueError(f"metric `{name}` must be finite.")
            if name in {"accuracy", "f1_weighted"} and not 0.0 <= metric <= 1.0:
                raise ValueError(f"metric `{name}` must be in [0.0, 1.0].")
            if name == "r2" and metric > 1.0:
                raise ValueError("metric `r2` must be <= 1.0.")
            if name in {"mae", "rmse"} and metric < 0.0:
                raise ValueError(f"metric `{name}` must be non-negative.")
        return value
