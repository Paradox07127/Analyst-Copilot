"""Public E4b exploration control-plane schemas.

These models intentionally expose journal projections, not job state.  The job
is an execution mechanism; the append-only exploration journal is the recovery
and API status authority.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from eda_platform.schemas.exploration import (
    ExplorationPolicy,
    ExplorationRunStatus,
    ExplorationStopReason,
)
from eda_platform.schemas.exploration_budget import (
    BudgetAmendment,
    BudgetCapIncrease,
    ExplorationBudgetPolicy,
)
from eda_platform.schemas.insights import InsightProof, InsightStatus, InsightTrustLevel


class ExplorationPrepareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["open", "goal_directed"] = "open"
    goal: str | None = Field(default=None, max_length=4_000)
    dataset_ids: tuple[str, ...] = Field(min_length=1, max_length=20)
    thinking_level: Literal["quick", "standard", "deep"] = "standard"

    @field_validator("dataset_ids")
    @classmethod
    def _dataset_ids_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("dataset ids cannot be blank")
        if len(set(value)) != len(value):
            raise ValueError("dataset ids must be unique")
        return value


class ExplorationStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_hash: str = Field(min_length=8)
    approval_token: str = Field(min_length=8)


class ExplorationControlRequest(BaseModel):
    """Empty on purpose: policy identities are always reconstructed server-side."""

    model_config = ConfigDict(extra="forbid")


class ExplorationExtendBudgetRequest(BaseModel):
    """Only additive caps and a human reason; no client-provided fingerprints."""

    model_config = ConfigDict(extra="forbid")

    increase: BudgetCapIncrease
    reason: str = Field(min_length=1, max_length=2_000)


class ExplorationCostRange(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_usd: Decimal = Field(ge=0)
    maximum_usd: Decimal = Field(gt=0)
    basis: Literal["policy_hard_cap"] = "policy_hard_cap"
    exact: Literal[False] = False


class ExplorationPrepared(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    exploration_id: str
    session_id: str
    project_id: str
    policy: ExplorationPolicy
    data_state_witness: str
    cost_range: ExplorationCostRange
    action_hash: str
    approval_token: str
    expires_at: datetime
    release_certificate_digest: str


class ExplorationJobView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    job_id: str
    execution_session_id: str
    status: str


class ExplorationBudgetView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    base: ExplorationBudgetPolicy
    max_llm_requests: int | None
    remaining_llm_requests: int | None
    max_successful_tool_calls: int
    remaining_successful_tool_calls: int
    max_rows_scanned: int | None
    rows_scanned: int
    max_result_cells: int | None
    result_cells: int
    max_rounds: int
    remaining_rounds: int
    max_cost_usd: Decimal | None
    cost_usd: Decimal
    remaining_cost_usd: Decimal | None
    llm_requests_used: int
    successful_tool_calls_used: int
    rounds_used: int
    amendments: tuple[BudgetAmendment, ...] = ()


class ExplorationHypothesisView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    hypothesis_id: str
    statement: str
    why_selected: str
    status: str
    priority: float


class ExplorationEvidenceView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    receipt_id: str
    tool_name: str
    summary: str
    fact_ids: tuple[str, ...] = ()


class ExplorationInsightView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    insight_id: str
    hypothesis_id: str
    statement: str
    family: str
    status: InsightStatus
    trust_level: InsightTrustLevel
    evidence_lane: Literal["exploratory", "confirmatory"]
    proof: tuple[InsightProof, ...]
    limitations: tuple[str, ...] = ()


class ExplorationReportView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    available: bool
    artifact_ref: str | None = None


class ExplorationView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    exploration_id: str
    session_id: str
    project_id: str
    goal: str
    thinking_level: Literal["quick", "standard", "deep"]
    status: ExplorationRunStatus
    stop_reason: ExplorationStopReason | None = None
    last_seq: int = Field(ge=0)
    policy_fingerprint: str
    effective_policy_fingerprint: str
    data_state_witness: str
    amendment_ids: tuple[str, ...] = ()
    current_hypothesis: ExplorationHypothesisView | None = None
    current_evidence: tuple[ExplorationEvidenceView, ...] = ()
    insights: tuple[ExplorationInsightView, ...] = ()
    limitations: tuple[str, ...] = ()
    coverage_targets: tuple[str, ...] = ()
    coverage_completed: tuple[str, ...] = ()
    coverage_unexplored: tuple[str, ...] = ()
    report: ExplorationReportView
    budget: ExplorationBudgetView
    job: ExplorationJobView | None = None
    events_url: str


class ExplorationStarted(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    exploration: ExplorationView
    job: ExplorationJobView


class ExplorationBudgetExtended(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    exploration: ExplorationView
    amendment: BudgetAmendment
    effective_policy_fingerprint: str


class ExplorationEventView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str
    exploration_id: str
    seq: int = Field(ge=0)
    type: str
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    data: dict[str, object] = Field(default_factory=dict)


class ExplorationRunMetadata(BaseModel):
    """Immutable resource identity stored beside, but never instead of, the journal."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    exploration_id: str
    source_session_id: str
    project_id: str
    policy: ExplorationPolicy
    data_state_witness: str
    release_certificate_digest: str
    approval_action_hash: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
