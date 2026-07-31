from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

IntentKind = Literal[
    "ask_from_artifacts",
    "new_analysis",
    "open_analysis",
    "meta_help",
    "out_of_scope",
    "refine_analysis",
]


class Intent(BaseModel):
    kind: IntentKind
    params: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0
    raw_message: str

    @field_validator("confidence")
    @classmethod
    def _confidence_is_probability(cls, value: float) -> float:
        if 0.0 <= value <= 1.0:
            return value
        raise ValueError("confidence must be in [0.0, 1.0].")

    @field_validator("raw_message")
    @classmethod
    def _raw_message_is_non_empty(cls, value: str) -> str:
        if value.strip():
            return value
        raise ValueError("raw_message must be non-empty.")


class AnalysisPlan(BaseModel):
    question: str
    dataset_names: list[str] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    filters: list[str] = Field(default_factory=list)
    sql: str
    method: str
    rationale: str
    needs_approval: bool = False
    estimated_scan: Literal["small", "medium", "large", "unknown"] = "unknown"

    @field_validator("question", "sql", "method", "rationale")
    @classmethod
    def _required_strings_are_non_empty(cls, value: str) -> str:
        if value.strip():
            return value
        raise ValueError("field must be non-empty.")

    @field_validator("dataset_names", "columns")
    @classmethod
    def _required_lists_are_non_empty(cls, value: list[str]) -> list[str]:
        if value and all(item.strip() for item in value):
            return value
        raise ValueError("field must contain at least one non-empty item.")
