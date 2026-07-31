"""Unified publication-readiness projection for UI and run observability."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

PublicationReadiness = Literal[
    "draft",
    "analysis_available",
    "investigation_validated",
    "publication_ready",
    "published",
]
PublicationFreshness = Literal["not_applicable", "unknown", "fresh", "stale", "unverifiable"]


class PublicationCondition(BaseModel):
    """One orthogonal condition; readiness is only a summary over conditions."""

    type: Literal["answerability", "investigation", "report_gate", "publication", "freshness"]
    status: Literal["true", "false", "unknown"]
    reason: str
    message: str = ""


class PublicationState(BaseModel):
    """One deterministic read model over execution, investigation and report artifacts."""

    schema_version: int = 1
    readiness: PublicationReadiness
    answered_questions: int = 0
    abstained_questions: int = 0
    failed_questions: int = 0
    automated_findings: int = 0
    exploratory_answers: int = 0
    validated_findings: int = 0
    report_eligible_findings: int = 0
    decision_reports: int = 0
    technical_report_status: str | None = None
    publication_freshness: PublicationFreshness = "not_applicable"
    conditions: list[PublicationCondition] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
