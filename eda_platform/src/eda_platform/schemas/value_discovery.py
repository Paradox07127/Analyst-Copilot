"""Role 1 (Value Discovery) handoff schemas.

A ``ValueMap`` is an interpretation-and-hypothesis artifact, never a
conclusion (m7-role-contract §4.1). LLM or heuristic code may author the
*text* fields; ``feasibility`` on an opportunity is a preliminary hint that
is ALWAYS recomputed by the deterministic method registry
(``core/methods.evaluate_feasibility``) before it reaches a Question Card.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from eda_platform.schemas.questions import (
    AnalysisMode,
    FeasibilityStatus,
    ValueCategory,
)


class DatasetValueProfile(BaseModel):
    """The decision-relevant capabilities observed in one dataset."""

    dataset_name: str
    dataset_display_name: str
    capabilities: list[str] = Field(default_factory=list)
    source_artifact_ids: list[str] = Field(default_factory=list)
    quality_issue_count: int = Field(default=0, ge=0)
    quality_context_count: int = Field(default=0, ge=0)


class ValueOpportunity(BaseModel):
    """A bounded, evidence-led route from data to a possible decision."""

    opportunity_id: str
    value_category: ValueCategory
    target_datasets: list[str] = Field(default_factory=list)
    data_signal: str
    value_hypothesis: str
    decision_action: str
    analysis_mode: AnalysisMode = Field(
        description="Closed analysis-mode vocabulary; free-text directions are "
        "not accepted (they defeat deterministic method dispatch)",
    )
    feasibility: FeasibilityStatus
    feasibility_reasons: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    source_artifact_ids: list[str] = Field(default_factory=list)


class KnowledgeSummary(BaseModel):
    """Counts and labels from user-confirmed semantic context."""

    field_meanings: list[str] = Field(default_factory=list)
    metric_definitions: list[str] = Field(default_factory=list)
    entity_notes: list[str] = Field(default_factory=list)
    verified_relations: list[str] = Field(default_factory=list)


class ValueMap(BaseModel):
    """Role 1 handoff: potential value paths, not asserted business results."""

    schema_version: int = 1
    version: int = 1
    business_context_provided: bool = False
    datasets: list[DatasetValueProfile] = Field(default_factory=list)
    opportunities: list[ValueOpportunity] = Field(default_factory=list)
    knowledge: KnowledgeSummary = Field(default_factory=KnowledgeSummary)
