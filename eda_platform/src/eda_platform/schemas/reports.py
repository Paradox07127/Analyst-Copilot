from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field

from eda_platform.schemas.artifacts import EvidenceRef

CLAIMS_SECTION_BODY = "Validated evidence-backed findings are listed below."
FOCUS_SECTION_BODY = "Analysis focus questions from this run are listed below."
EMPTY_SECTION_BODY = "No validated conclusion is available for this section."


class ReportStatus(StrEnum):
    DRAFT = "draft"
    VALIDATED = "validated"
    NEEDS_REVISION = "needs_revision"
    BLOCKED_FOR_REVIEW = "blocked_for_review"


# Hard validation rejects claims; the semantic gate only grades survivors.
GateVerdict = Literal["pass", "degraded", "rejected"]
# F6 evidence-strength tiers; "verified" / "low_relevance" are legacy values
# kept only so pre-F6 persisted bundles load (new code writes the first three).
ConfidenceLabel = Literal[
    "strong", "indicative", "exploratory", "verified", "low_relevance"
]


class ReportSeverity(StrEnum):
    CRITICAL = "critical"
    WARN = "warn"
    INFO = "info"


class NumericSourceValue(BaseModel):
    value: float
    unit: str = "raw"
    # F2 match policy, derived by the validator (exact | rounded); default
    # keeps pre-F2 persisted data loadable.
    policy: str = "rounded"


# Claim rollup priority: failed > unverified > number_verified (see
# NumericTokenStatus). "not_evaluated" marks bundles that never went through
# validate_report_bundle; the validator writes "no_numbers" explicitly.
NumericRollup = Literal[
    "not_evaluated", "no_numbers", "number_verified", "unverified", "failed"
]


class NumericTokenStatus(BaseModel):
    """Verification state of one number token in claim text."""

    number: float
    is_percent: bool = False
    status: Literal["number_verified", "unverified", "failed"]


class NumericEvidenceSource(BaseModel):
    """One evidence ref that contributed (or failed to contribute) numeric values."""

    artifact_id: str | None = None
    locator: str = ""
    kind: str = ""
    # True: values came from persisted payload; False: nothing contributed
    # (inline evidence.value never enters the pool).
    resolved: bool = False
    value_count: int = 0
    # Contributed values with units, capped; value_count keeps the full size.
    values: list[NumericSourceValue] = Field(default_factory=list)


class NumericMismatchDetail(BaseModel):
    """One claim-text number the numeric gate could not support."""

    number: float
    is_percent: bool = False
    reason: Literal["no_evidence_values", "outside_tolerance"]
    # Sorted unique values from the pool checked for this number, capped at 20.
    evidence_values: list[float] = Field(default_factory=list)
    # Uncapped pool size.
    evidence_value_count: int = 0
    sources: list[NumericEvidenceSource] = Field(default_factory=list)


class ReportValidationFinding(BaseModel):
    severity: ReportSeverity
    code: str
    message: str
    section_title: str | None = None
    claim_id: str | None = None
    repair_mode: Literal["deterministic", "llm", "prune"] = "prune"
    numeric_details: list[NumericMismatchDetail] = Field(default_factory=list)


class ReportAudit(BaseModel):
    status: ReportStatus
    findings: list[ReportValidationFinding] = Field(default_factory=list)
    semantic_notes: list[str] = Field(default_factory=list)
    # Independent semantic-gate rollup for disclosure.
    gate_verdict: GateVerdict = "pass"
    degraded_claim_count: int = 0
    time_boundary_truncations: int = 0
    # Claims whose numeric rollup is "unverified" (not findings, not rewritten).
    numeric_unverified_claim_count: int = 0
    # Quantitative-section claims with zero verified numbers (F3 disclosure
    # metric; not findings, not rewritten).
    quantitative_coverage_gap_count: int = 0

    @property
    def has_critical_findings(self) -> bool:
        return any(finding.severity is ReportSeverity.CRITICAL for finding in self.findings)


class ReportClaim(BaseModel):
    id: str = ""
    text: str
    evidence: list[EvidenceRef] = Field(default_factory=list)
    referenced_datasets: list[str] = Field(default_factory=list)
    referenced_columns: list[str] = Field(default_factory=list)
    quality_issue_refs: list[str] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "medium"
    # F6 evidence-strength tier written by apply_semantic_gate; the "verified"
    # default is the legacy value bundles carry until the gate stamps them.
    confidence_label: ConfidenceLabel = "verified"
    gate_verdict: Literal["pass", "degraded"] = "pass"
    gate_flags: list[str] = Field(
        default_factory=list,
        description="Structured semantic-gate reasons, e.g. "
        "'low_impact_column:order_item_id' or 'partial_periods:2018-09'",
    )
    time_boundary_flag: str = Field(
        default="",
        description="Non-empty when a trend claim rests on partial edge "
        "periods, e.g. 'partial_periods:2018-09'",
    )
    # Per-token numeric verification written by validate_report_bundle.
    numeric_statuses: list[NumericTokenStatus] = Field(default_factory=list)
    numeric_rollup: NumericRollup = "not_evaluated"
    # True for platform-generated fallback claims (set at the generation site);
    # exempts them from the F3 coverage-gap metric. Default keeps old bundles
    # loadable.
    deterministic_source: bool = False
    # F3: quantitative-section claim published with zero verified numbers
    # (written by validate_report_bundle).
    quantitative_coverage_gap: bool = False


class ReportPlanClaim(ReportClaim):
    section_title: str


# Writers may request bounded, read-only evidence from persisted artifacts.

InterleaveRejectionCode = Literal[
    "unknown_artifact",
    "unsupported_type",
    "unresolvable_locator",
    "invalid_payload",
    "budget_exhausted",
]


class EvidenceRequest(BaseModel):
    """A writer-side typed request for evidence from a persisted artifact."""

    artifact_id: str
    locator: str = ""
    # Section used for per-section request accounting.
    section: str = ""
    reason: str = ""


class GrantValue(BaseModel):
    value: float
    unit: Literal["raw", "percent", "currency"] = "raw"
    unit_label: str | None = None
    unit_reference: str | None = None


class EvidenceGrant(BaseModel):
    """Deterministically resolved evidence returned to the writer."""

    artifact_id: str
    artifact_type: str
    locator: str = ""
    values: list[GrantValue] = Field(default_factory=list)
    texts: list[str] = Field(default_factory=list)


class EvidenceRejection(BaseModel):
    """Typed refusal with a teaching-style message and the usable catalog."""

    artifact_id: str
    locator: str = ""
    reason_code: InterleaveRejectionCode
    message: str
    available_artifacts: list[str] = Field(default_factory=list)


class InterleaveExchange(BaseModel):
    """One request/response pair in the bounded write-time loop."""

    section: str = ""
    request: EvidenceRequest
    grant: EvidenceGrant | None = None
    rejection: EvidenceRejection | None = None


class InterleaveTranscript(BaseModel):
    """Full request/grant/rejection record; artifact type
    ``EVIDENCE_INTERLEAVE_TRANSCRIPT``."""

    schema_version: int = 1
    exchanges: list[InterleaveExchange] = Field(default_factory=list)
    per_section_limit: int = 2
    total_limit: int = 8
    granted_count: int = 0
    rejected_count: int = 0


class ReportPlanDraft(BaseModel):
    """Compact model-authored plan containing evidence-backed section claims."""

    claims: list[ReportPlanClaim] = Field(default_factory=list)
    # Fine-grained reads requested before finalizing the plan.
    evidence_requests: list[EvidenceRequest] = Field(default_factory=list)


class ReportFocusItem(BaseModel):
    """One executed analysis question (F4): structured focus entry, not a claim —
    no evidence chain, no ledger row, no numeric gate."""

    question: str
    outcome: str
    question_id: str = ""
    # Why a non-answered question produced nothing; empty when it answered.
    reason: str = ""


class ReportSection(BaseModel):
    title: str
    body: str = ""
    # Connective prose over this section's claims, written after they passed
    # every gate and held to their figures. Empty when narration was off,
    # unavailable, or rejected.
    narrative: str = ""
    claims: list[ReportClaim] = Field(default_factory=list)
    # Default keeps pre-F4 persisted bundles loadable.
    focus_items: list[ReportFocusItem] = Field(default_factory=list)

    def structural_body(self) -> str:
        """The only prose allowed outside evidence-checked claims."""

        if self.claims:
            return CLAIMS_SECTION_BODY
        if self.focus_items:
            return FOCUS_SECTION_BODY
        return EMPTY_SECTION_BODY


class ReportBundle(BaseModel):
    project_id: str
    # Accept pre-session-migration bundles while always serializing the new name.
    session_id: str = Field(validation_alias=AliasChoices("session_id", "run_id"))
    status: ReportStatus = ReportStatus.DRAFT
    sections: list[ReportSection]
    audit: ReportAudit | None = None

    @classmethod
    def empty(
        cls,
        *,
        project_id: str,
        session_id: str,
    ) -> ReportBundle:
        return cls(
            project_id=project_id,
            session_id=session_id,
            sections=[ReportSection(title=title) for title in required_report_sections()],
        )


def merge_duplicate_sections(sections: list[ReportSection]) -> list[ReportSection]:
    """Merge same-title sections in first-seen order and deduplicate claims by ID."""
    merged: dict[str, ReportSection] = {}
    order: list[str] = []
    for section in sections:
        existing = merged.get(section.title)
        if existing is None:
            existing = ReportSection(
                title=section.title,
                body=section.body,
                narrative=section.narrative,
                claims=[],
            )
            merged[section.title] = existing
            order.append(section.title)
        elif not existing.narrative:
            existing.narrative = section.narrative
        seen_ids = {claim.id for claim in existing.claims if claim.id}
        for claim in section.claims:
            if claim.id and claim.id in seen_ids:
                continue
            if claim.id:
                seen_ids.add(claim.id)
            existing.claims.append(claim)
        seen_focus = {item.question_id or item.question for item in existing.focus_items}
        for item in section.focus_items:
            key = item.question_id or item.question
            if key in seen_focus:
                continue
            seen_focus.add(key)
            existing.focus_items.append(item)
    return [merged[title] for title in order]


def required_report_sections() -> list[str]:
    return [
        "Executive Summary",
        "Dataset Overview",
        "File-by-File EDA Summary",
        "Data Quality Findings",
        "Key EDA Insights",
        "Selected Analysis Focus",
        "Agent-Performed Analysis",
        "Business Findings",
        "Business Recommendations",
        "Limitations and Risks",
        "Appendix: Charts and Technical Summary",
    ]
