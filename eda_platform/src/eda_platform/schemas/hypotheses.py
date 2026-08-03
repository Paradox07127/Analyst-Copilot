"""Agent-owned hypothesis proposals for the E4a exploration policy surface.

Only semantic content the model may propose lives here. Fingerprints, priority
features, admission outcomes and mandatory coverage flags are control-plane
data and are attached by :mod:`eda_platform.agents.exploration.candidates` and
the deterministic scheduler.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from eda_platform.schemas.exploration import InsightFamily


class HypothesisPredicate(BaseModel):
    """Machine-stable proposition identity; prose is display-only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: str = Field(min_length=1)
    operator: Literal[
        "differs",
        "associated_with",
        "greater_than",
        "less_than",
        "equal_to",
        "not_equal_to",
        "has_spike",
        "exists",
        "absent",
    ]
    left_operand: str | None = None
    right_operand: str | None = None
    threshold: float | None = Field(default=None, allow_inf_nan=False)

    @field_validator("metric", "left_operand", "right_operand")
    @classmethod
    def _predicate_strings_are_normalized(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("predicate strings must be non-blank.")
        return normalized


class HypothesisProposal(BaseModel):
    """Strict structured output accepted from bootstrap, agent or follow-up paths."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    statement: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    expected_evidence: str = Field(min_length=1)
    falsification_conditions: tuple[str, ...] = Field(min_length=1)
    family: InsightFamily
    method_family: str = Field(min_length=1)
    dataset_ids: tuple[str, ...] = Field(min_length=1)
    columns: tuple[str, ...] = ()
    segment: str | None = None
    time_scope: str | None = None
    probe_kind: str = Field(min_length=1)
    predicate: HypothesisPredicate
    parent_hypothesis_id: str | None = None

    @field_validator(
        "statement",
        "rationale",
        "expected_evidence",
        "method_family",
        "probe_kind",
    )
    @classmethod
    def _non_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value cannot be blank.")
        return normalized

    @field_validator("dataset_ids", "columns", "falsification_conditions")
    @classmethod
    def _unique_non_blank_tuple(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if any(not item for item in normalized):
            raise ValueError("items cannot be blank.")
        if len(normalized) != len(set(normalized)):
            raise ValueError("items must be unique.")
        return normalized

    @field_validator("segment", "time_scope", "parent_hypothesis_id")
    @classmethod
    def _optional_non_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("optional strings must be omitted rather than blank.")
        return normalized


class HypothesisProposalBatch(BaseModel):
    """Strict LLM boundary for one small generate step."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    concluded: bool = False
    conclusion_reason: str | None = None
    proposals: tuple[HypothesisProposal, ...] = Field(default=(), max_length=12)

    @model_validator(mode="after")
    def _conclusion_matches_payload(self) -> HypothesisProposalBatch:
        if self.concluded:
            if not (self.conclusion_reason or "").strip():
                raise ValueError("a concluded batch requires conclusion_reason.")
            if self.proposals:
                raise ValueError("a concluded batch cannot also contain proposals.")
        elif not self.proposals:
            raise ValueError("a non-concluded batch requires at least one proposal.")
        return self
