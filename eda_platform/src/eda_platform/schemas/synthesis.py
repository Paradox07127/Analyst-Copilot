"""Typed, evidence-bounded handoff from validated findings to a decision story."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

from eda_platform.schemas.investigations import ReportReadiness


class SynthesisStoryBeat(BaseModel):
    """A bounded part of a decision story with explicit finding provenance."""

    title: str
    body: str
    finding_artifact_ids: list[str] = Field(default_factory=list)

    @field_validator("title", "body")
    @classmethod
    def _required_strings_are_non_empty(cls, value: str) -> str:
        if value.strip():
            return value
        raise ValueError("field must be non-empty.")


class SynthesisBrief(BaseModel):
    """User-selected, evidence-bounded input for a later final report."""

    schema_version: int = 1
    brief_id: str
    project_id: str
    business_context: str = Field(
        default="",
        description="Unverified user framing; labeled in the UI, never a claim or number source",
    )
    selected_finding_artifact_ids: list[str] = Field(min_length=1)
    selected_finding_session_ids: dict[str, str] = Field(default_factory=dict)
    source_record_artifact_ids: list[str] = Field(default_factory=list)
    decision_context: str
    headline: str
    storyline: list[SynthesisStoryBeat] = Field(min_length=1)
    limitations: list[str] = Field(default_factory=list)
    investigation_gaps: list[str] = Field(default_factory=list)
    report_eligible: bool = False
    report_readiness: ReportReadiness = "not_eligible"

    @field_validator("brief_id", "project_id", "decision_context", "headline")
    @classmethod
    def _required_strings_are_non_empty(cls, value: str) -> str:
        if value.strip():
            return value
        raise ValueError("field must be non-empty.")

    @field_validator("selected_finding_artifact_ids")
    @classmethod
    def _finding_ids_are_non_empty(cls, value: list[str]) -> list[str]:
        if value and all(item.strip() for item in value):
            return value
        raise ValueError("selected_finding_artifact_ids must contain non-empty ids.")

    @field_validator("selected_finding_session_ids")
    @classmethod
    def _finding_session_ids_are_non_empty(cls, value: dict[str, str]) -> dict[str, str]:
        if all(
            artifact_id.strip() and session_id.strip() for artifact_id, session_id in value.items()
        ):
            return value
        raise ValueError("selected_finding_session_ids must contain non-empty ids.")

    @model_validator(mode="after")
    def _eligibility_states_are_consistent(self) -> SynthesisBrief:
        """Contradictory states are producer bugs or tampering — reject them."""
        unexpected_session_ids = set(self.selected_finding_session_ids).difference(
            self.selected_finding_artifact_ids
        )
        if unexpected_session_ids:
            raise ValueError(
                "selected_finding_session_ids contains artifacts that were not selected."
            )
        if self.report_eligible and self.report_readiness == "not_eligible":
            raise ValueError("report_eligible=True contradicts report_readiness='not_eligible'.")
        if not self.report_eligible and self.report_readiness != "not_eligible":
            raise ValueError("report_eligible=False requires report_readiness='not_eligible'.")
        return self
