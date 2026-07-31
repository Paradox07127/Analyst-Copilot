"""Final evidence-validated decision story assembled from an approved brief."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from eda_platform.schemas.investigations import ReportReadiness


class SCQAFrame(BaseModel):
    """The pyramid-principle framing of the decision story."""

    situation: str
    complication: str
    question: str
    answer: str

    @field_validator("situation", "complication", "question", "answer")
    @classmethod
    def _required_strings_are_non_empty(cls, value: str) -> str:
        if value.strip():
            return value
        raise ValueError("field must be non-empty.")


class MetaInsightSkeleton(BaseModel):
    """Deterministic commonality-and-exception grouping of top findings."""

    commonality_statements: list[str] = Field(default_factory=list)
    commonality_finding_artifact_ids: list[str] = Field(default_factory=list)
    exception_statements: list[str] = Field(default_factory=list)
    exception_finding_artifact_ids: list[str] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.commonality_statements or self.exception_statements)


class DecisionReportSection(BaseModel):
    """One finding-backed section with explicit provenance."""

    title: str
    body: str
    finding_artifact_ids: list[str] = Field(default_factory=list)

    @field_validator("title", "body")
    @classmethod
    def _required_strings_are_non_empty(cls, value: str) -> str:
        if value.strip():
            return value
        raise ValueError("field must be non-empty.")


class DecisionReport(BaseModel):
    """Publishable decision story; artifact type ``DECISION_REPORT``."""

    schema_version: int = 1
    report_id: str
    brief_id: str
    project_id: str
    title: str
    scqa: SCQAFrame
    sections: list[DecisionReportSection] = Field(min_length=1)
    limitations: list[str] = Field(default_factory=list)
    investigation_gaps: list[str] = Field(default_factory=list)
    report_readiness: ReportReadiness
    narrative_status: Literal["deterministic", "llm_refined"] = "deterministic"
    # Why an attempted LLM refinement was not adopted; empty when none was tried
    # or the rewrite was accepted. Falling back must never be silent.
    narrative_fallback_reason: str = ""
    source_finding_artifact_ids: list[str] = Field(min_length=1)
    source_finding_session_ids: dict[str, str] = Field(default_factory=dict)
    # Optional publication-integrity fields preserve legacy readability.
    publication_input_fingerprint: str | None = None
    report_policy_version: str | None = None
    # Optional for previously persisted reports.
    meta_insight: MetaInsightSkeleton | None = None
    # Provenance for bounded write-time evidence reads.
    interleave_transcript_artifact_id: str | None = None
    granted_evidence_artifact_ids: list[str] = Field(default_factory=list)

    @field_validator("report_id", "brief_id", "project_id", "title")
    @classmethod
    def _required_strings_are_non_empty(cls, value: str) -> str:
        if value.strip():
            return value
        raise ValueError("field must be non-empty.")
