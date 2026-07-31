"""Typed contracts for approved investigation plans, decisions, and results."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from eda_platform.schemas.artifacts import EvidenceRef
from eda_platform.schemas.quality_context import QualityContext
from eda_platform.schemas.questions import FeasibilityStatus, QuestionFinding

InvestigationPlanStatus = Literal["planned", "needs_data"]
InvestigationOutcome = Literal[
    "validated",
    "inconclusive",
    "failed",
    "needs_data",
    "rejected",
]
GateName = Literal["scope", "feasibility", "execution", "method", "claim"]
GateStatus = Literal["passed", "warning", "failed"]
ClaimClass = Literal["observed", "predictive", "causal_supported", "inconclusive"]
ReliabilityRating = Literal["high", "medium", "low"]
ReportReadiness = Literal["eligible", "eligible_with_limitations", "not_eligible"]
ApprovalDecision = Literal["approved", "rejected"]


class InvestigationGate(BaseModel):
    name: GateName
    status: GateStatus
    reason: str

    @field_validator("reason")
    @classmethod
    def _reason_is_non_empty(cls, value: str) -> str:
        if value.strip():
            return value
        raise ValueError("reason must be non-empty.")


class InvestigationPlan(BaseModel):
    """Immutable, execution-ready handoff for one approved Question Card."""

    schema_version: int = 1
    investigation_id: str
    source_session_id: str
    question_id: str
    card_version: int = Field(ge=1)
    candidate_fingerprint: str = Field(
        description="SHA-256 over the canonical JSON of the source candidate; "
        "execution fails closed unless the live candidate still matches",
    )
    question: str
    target_datasets: list[str] = Field(default_factory=list)
    source_artifact_ids: list[str] = Field(default_factory=list)
    source_artifact_session_ids: dict[str, str] = Field(default_factory=dict)
    allowed_relationship_references: list[str] = Field(default_factory=list)
    method_family: str
    method_recipe: str
    allowed_tools: list[str] = Field(default_factory=list)
    method_requirements: list[str] = Field(default_factory=list)
    execution_ready: bool = False
    quality_context: list[QualityContext] = Field(default_factory=list)
    quality_context_artifact_ids: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    validation_gates: list[InvestigationGate] = Field(default_factory=list)
    feasibility: FeasibilityStatus
    status: InvestigationPlanStatus
    status_reason: str
    user_approval_required: bool = True

    @field_validator(
        "investigation_id",
        "source_session_id",
        "question_id",
        "candidate_fingerprint",
        "question",
        "method_family",
        "method_recipe",
        "status_reason",
    )
    @classmethod
    def _required_strings_are_non_empty(cls, value: str) -> str:
        if value.strip():
            return value
        raise ValueError("field must be non-empty.")

    @field_validator("investigation_id")
    @classmethod
    def _investigation_id_is_a_safe_slug(cls, value: str) -> str:
        if len(value) > 128 or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for character in value
        ):
            raise ValueError(
                "investigation_id must be a path-safe slug containing only letters, "
                "digits, '_' or '-'."
            )
        return value

    @field_validator("target_datasets", "allowed_tools")
    @classmethod
    def _required_lists_are_non_empty(cls, value: list[str]) -> list[str]:
        if value and all(item.strip() for item in value):
            return value
        raise ValueError("field must contain at least one non-empty item.")


class InvestigationApproval(BaseModel):
    """Persisted user decision bound to the exact approved plan content."""

    schema_version: int = 1
    approval_id: str
    investigation_id: str
    plan_fingerprint: str = Field(
        description="SHA-256 over the canonical JSON of the approved plan",
    )
    decision: ApprovalDecision
    reason: str = ""
    decided_at: str = Field(description="ISO-8601 timestamp of the user decision")

    @field_validator("approval_id", "investigation_id", "plan_fingerprint", "decided_at")
    @classmethod
    def _required_strings_are_non_empty(cls, value: str) -> str:
        if value.strip():
            return value
        raise ValueError("field must be non-empty.")


class ValidatedFinding(BaseModel):
    """A validated finding with explicit evidence and hypothesis context."""

    schema_version: int = 1
    finding_id: str
    investigation_id: str
    question_id: str
    question: str
    value_hypothesis: str = Field(
        default="",
        description="LLM/template hypothesis context; never a claim source",
    )
    decision_action: str = Field(
        default="",
        description="LLM/template hypothesis context; never a claim source",
    )
    quality_context: list[QualityContext] = Field(default_factory=list)
    claim_class: ClaimClass
    findings: list[QuestionFinding] = Field(default_factory=list)
    evidence_support: ReliabilityRating
    analytical_reliability: ReliabilityRating
    decision_readiness: ReliabilityRating
    limitations: list[str] = Field(default_factory=list)
    report_eligible: bool = False
    report_readiness: ReportReadiness = "not_eligible"
    report_readiness_reason: str
    source_artifact_ids: list[str] = Field(default_factory=list)
    source_artifact_session_ids: dict[str, str] = Field(default_factory=dict)
    # Model interpretation admitted only after numeric and causal validation.
    interpretation: str = Field(
        default="",
        description="Validator-gated LLM interpretation; empty when absent",
    )
    interpretation_status: Literal["validated", "fallback", "absent"] = "absent"

    @field_validator(
        "finding_id",
        "investigation_id",
        "question_id",
        "question",
        "report_readiness_reason",
    )
    @classmethod
    def _required_strings_are_non_empty(cls, value: str) -> str:
        if value.strip():
            return value
        raise ValueError("field must be non-empty.")

    @model_validator(mode="after")
    def _eligibility_states_are_consistent(self) -> ValidatedFinding:
        """Contradictory states are producer bugs or tampering — reject them."""
        if self.report_eligible and self.report_readiness == "not_eligible":
            raise ValueError(
                "report_eligible=True contradicts report_readiness='not_eligible'."
            )
        if not self.report_eligible and self.report_readiness != "not_eligible":
            raise ValueError(
                "report_eligible=False requires report_readiness='not_eligible'."
            )
        return self


class InvestigationRecord(BaseModel):
    """Research memory for every completed, blocked, or rejected attempt."""

    schema_version: int = 1
    record_id: str
    investigation_id: str
    question_id: str
    status: InvestigationOutcome
    reason_code: str
    reason: str
    next_action: str
    validation_gates: list[InvestigationGate] = Field(default_factory=list)
    source_artifact_ids: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    finding_artifact_id: str | None = None

    @field_validator(
        "record_id",
        "investigation_id",
        "question_id",
        "reason_code",
        "reason",
        "next_action",
    )
    @classmethod
    def _required_strings_are_non_empty(cls, value: str) -> str:
        if value.strip():
            return value
        raise ValueError("field must be non-empty.")
