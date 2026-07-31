"""Deterministic project-level decision coverage summary."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class DecisionCoverage(BaseModel):
    """Whether the project's highest-value feasible questions reached an outcome."""

    schema_version: int = 1
    project_id: str
    top_cards_total: int = Field(ge=0)
    top_cards_terminal: int = Field(ge=0)
    uninvestigated_high_value: list[str] = Field(default_factory=list, max_length=5)
    findings_not_eligible: int = Field(ge=0)
    validated_findings: int = Field(ge=0)
    coverage_ready: bool
    gaps: list[str] = Field(default_factory=list, max_length=5)

    @field_validator("project_id")
    @classmethod
    def _project_id_is_non_empty(cls, value: str) -> str:
        if value.strip():
            return value
        raise ValueError("project_id must be non-empty.")
