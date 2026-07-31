from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

StatTestType = Literal[
    "independent_t_test",
    "paired_t_test",
    "chi_square_independence",
    "one_way_anova",
    "welch_anova",
    "mann_whitney_u",
    "kruskal_wallis",
]


class StatAssumptionCheck(BaseModel):
    name: str
    status: Literal["passed", "warn", "not_applicable"]
    statistic: float | None = None
    p_value: float | None = None
    message: str

    @field_validator("p_value")
    @classmethod
    def _p_value_is_probability(cls, value: float | None) -> float | None:
        if value is None or 0.0 <= value <= 1.0:
            return value
        raise ValueError("p_value must be in [0.0, 1.0].")


class StatWarning(BaseModel):
    code: str
    severity: Literal["info", "warn", "critical"] = "warn"
    message: str


class StatTestResult(BaseModel):
    dataset_id: str
    test_type: StatTestType
    group_column: str | None = None
    value_column: str | None = None
    category_column: str | None = None
    pair_column: str | None = None
    # A test that cannot be computed on the given data (constant column, empty
    # group -> non-finite scipy result) is skipped upstream rather than emitted;
    # these stay optional so a transient non-computable result never crashes
    # construction before the skip gate in run_stat_test.
    statistic: float | None = None
    p_value: float | None = None
    adjusted_p_value: float | None = None
    correction_method: Literal["bonferroni"] | None = None
    effect_size: float | None = None
    degrees_of_freedom: int | None = None
    sample_size: int
    groups: dict[str, int] = Field(default_factory=dict)
    assumptions: list[StatAssumptionCheck] = Field(default_factory=list)
    warnings: list[StatWarning] = Field(default_factory=list)

    @field_validator("p_value", "adjusted_p_value")
    @classmethod
    def _p_value_is_probability(cls, value: float | None) -> float | None:
        if value is None or 0.0 <= value <= 1.0:
            return value
        raise ValueError("p_value must be in [0.0, 1.0].")

    @field_validator("degrees_of_freedom", "sample_size")
    @classmethod
    def _counts_are_non_negative(cls, value: int | None) -> int | None:
        if value is None or value >= 0:
            return value
        raise ValueError("count fields must be non-negative.")

    @field_validator("groups")
    @classmethod
    def _group_counts_are_non_negative(cls, value: dict[str, int]) -> dict[str, int]:
        if all(count >= 0 for count in value.values()):
            return value
        raise ValueError("group counts must be non-negative.")
