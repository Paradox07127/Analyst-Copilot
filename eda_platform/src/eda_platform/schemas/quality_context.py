"""EDA-grounded data conditions for Question Cards and later reporting.

Quality context records *observed conditions* (missingness, duplicates,
parsing failures, mixed types) as evidence to carry through an investigation
and disclose in the report. It is never a pass/fail score and never asserts a
business cause (m7-role-contract §4/§6).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

QualityContextSeverity = Literal["critical", "warn", "info"]


class QualityContext(BaseModel):
    """One observed data condition, without asserting its business cause."""

    context_id: str
    dataset_id: str
    dataset_name: str
    issue_code: str
    severity: QualityContextSeverity
    column: str | None = None
    observation: str
    pattern_facts: list[str] = Field(default_factory=list)
    analysis_impacts: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    validation_steps: list[str] = Field(default_factory=list)
    report_limitation: str
    requires_data: bool = False
    source_artifact_ids: list[str] = Field(default_factory=list)

    @field_validator(
        "context_id",
        "dataset_id",
        "dataset_name",
        "issue_code",
        "observation",
        "report_limitation",
    )
    @classmethod
    def _required_strings_are_non_empty(cls, value: str) -> str:
        if value.strip():
            return value
        raise ValueError("field must be non-empty.")


class QualityContextSet(BaseModel):
    """All EDA-derived quality context for one dataset version."""

    schema_version: int = 1
    dataset_id: str
    dataset_name: str
    contexts: list[QualityContext] = Field(default_factory=list)

    @field_validator("dataset_id", "dataset_name")
    @classmethod
    def _required_strings_are_non_empty(cls, value: str) -> str:
        if value.strip():
            return value
        raise ValueError("field must be non-empty.")
