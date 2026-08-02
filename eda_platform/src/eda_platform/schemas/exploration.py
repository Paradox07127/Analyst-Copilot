"""Exploration run schemas: policy, insight families, journal events and state.

Layering: schemas never import core. The policy fingerprint is computed in
core.exploration_journal and stored here as an opaque field; pause is a
resumable status and deliberately absent from the stop reasons (plan R3.1).
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from eda_platform.schemas.exploration_budget import (
    BudgetCapIncrease,
    ExplorationBudgetPolicy,
)

EXPLORATION_JOURNAL_SCHEMA_VERSION = 1


class InsightFamily(StrEnum):
    """Six-family taxonomy; values stay literally identical to InsightEval's
    ``data_type`` (snapshot-tested against the Eval-0 checkers, doc §10.1)."""

    DESCRIPTIVE = "Descriptive"
    DIAGNOSTIC = "Diagnostic"
    PREDICTIVE = "Predictive"
    PRESCRIPTIVE = "Prescriptive"
    EVALUATIVE = "Evaluative"
    EXPLORATORY = "Exploratory"


ExplorationStopReason = Literal[
    "completed",
    "budget_exhausted",
    "cancelled",
    "failed",
    "state_witness_changed",
    "no_new_information",
]

ExplorationRunStatus = Literal["running", "pause_requested", "paused", "stopped"]

GateVerdictValue = Literal["passed", "rejected"]


class ExplorationStateUnavailableError(RuntimeError):
    """Raised when a not-yet-restored state field is read (no silent defaults)."""


class ExplorationPolicy(BaseModel):
    """Immutable per-run policy; the fingerprint covers every field but itself."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["open", "goal_directed"]
    goal: str | None = None
    dataset_scope: tuple[str, ...] = Field(min_length=1)
    thinking_level: Literal["quick", "standard", "deep"]
    coverage_targets: tuple[InsightFamily, ...]
    budget: ExplorationBudgetPolicy
    scoring_policy_version: str = Field(min_length=1)
    statistical_policy_version: str = Field(min_length=1)
    tool_capability_digest: str = Field(min_length=1)
    policy_fingerprint: str = ""  # sealed by core.exploration_journal.sealed_policy

    @model_validator(mode="after")
    def _goal_matches_mode(self) -> ExplorationPolicy:
        if self.mode == "goal_directed" and not (self.goal or "").strip():
            raise ValueError("goal_directed mode requires a non-empty goal.")
        return self


class ExplorationEventBase(BaseModel):
    """Common envelope; every transition rule lives in the exploration reducer."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = EXPLORATION_JOURNAL_SCHEMA_VERSION
    seq: int = Field(ge=0)
    exploration_id: str = Field(min_length=1)
    attempt_epoch: int = Field(ge=0, default=0)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ExplorationStartedEvent(ExplorationEventBase):
    event_type: Literal["exploration_started"] = "exploration_started"
    policy_fingerprint: str = Field(min_length=1)
    code_fingerprint: str = Field(min_length=1)
    data_state_witness: str = Field(min_length=1)
    budget: ExplorationBudgetPolicy


class AttemptStartedEvent(ExplorationEventBase):
    event_type: Literal["attempt_started"] = "attempt_started"


class RoundStartedEvent(ExplorationEventBase):
    event_type: Literal["round_started"] = "round_started"
    round_index: int = Field(ge=0)


class LlmCallStartedEvent(ExplorationEventBase):
    event_type: Literal["llm_call_started"] = "llm_call_started"
    call_id: str = Field(min_length=1)


class LlmCallCompletedEvent(ExplorationEventBase):
    event_type: Literal["llm_call_completed"] = "llm_call_completed"
    call_id: str = Field(min_length=1)
    step_id: str = Field(min_length=1)
    response_digest: str = Field(min_length=1)


class LlmCallRejectedEvent(ExplorationEventBase):
    event_type: Literal["llm_call_rejected"] = "llm_call_rejected"
    call_id: str = Field(min_length=1)
    error: str = Field(min_length=1)


class LlmCallUncertainEvent(ExplorationEventBase):
    event_type: Literal["llm_call_uncertain"] = "llm_call_uncertain"
    call_id: str = Field(min_length=1)
    error: str = Field(min_length=1)


class ToolCallStartedEvent(ExplorationEventBase):
    event_type: Literal["tool_call_started"] = "tool_call_started"
    logical_step_id: str = Field(min_length=1)
    input_fingerprint: str = Field(min_length=1)


class ReceiptPreparedEvent(ExplorationEventBase):
    event_type: Literal["receipt_prepared"] = "receipt_prepared"
    logical_step_id: str = Field(min_length=1)
    receipt_id: str = Field(min_length=1)


class ReceiptCommittedEvent(ExplorationEventBase):
    event_type: Literal["receipt_committed"] = "receipt_committed"
    logical_step_id: str = Field(min_length=1)
    receipt_id: str = Field(min_length=1)


class ToolCallFailedEvent(ExplorationEventBase):
    event_type: Literal["tool_call_failed"] = "tool_call_failed"
    logical_step_id: str = Field(min_length=1)
    error: str = Field(min_length=1)


class GateVerdictEvent(ExplorationEventBase):
    event_type: Literal["gate_verdict"] = "gate_verdict"
    claim_bundle_id: str = Field(min_length=1)
    verdict: GateVerdictValue


class ReductionCommittedEvent(ExplorationEventBase):
    event_type: Literal["reduction_committed"] = "reduction_committed"
    frontier_digest: str = Field(min_length=1)
    ledger_digest: str = Field(min_length=1)


class RoundSettledEvent(ExplorationEventBase):
    event_type: Literal["round_settled"] = "round_settled"
    round_index: int = Field(ge=0)
    progress: bool


class PauseRequestedEvent(ExplorationEventBase):
    event_type: Literal["pause_requested"] = "pause_requested"
    reason: str | None = None


class PausedEvent(ExplorationEventBase):
    event_type: Literal["paused"] = "paused"


class ResumedEvent(ExplorationEventBase):
    event_type: Literal["resumed"] = "resumed"


class BudgetAmendedEvent(ExplorationEventBase):
    event_type: Literal["budget_amended"] = "budget_amended"
    amendment_id: str = Field(min_length=1)
    effective_policy_fingerprint: str = Field(min_length=1)
    increase: BudgetCapIncrease


class ExplorationStoppedEvent(ExplorationEventBase):
    event_type: Literal["exploration_stopped"] = "exploration_stopped"
    stop_reason: ExplorationStopReason
    final_report_ref: str | None = None


ExplorationLoopEvent = Annotated[
    ExplorationStartedEvent
    | AttemptStartedEvent
    | RoundStartedEvent
    | LlmCallStartedEvent
    | LlmCallCompletedEvent
    | LlmCallRejectedEvent
    | LlmCallUncertainEvent
    | ToolCallStartedEvent
    | ReceiptPreparedEvent
    | ReceiptCommittedEvent
    | ToolCallFailedEvent
    | GateVerdictEvent
    | ReductionCommittedEvent
    | RoundSettledEvent
    | PauseRequestedEvent
    | PausedEvent
    | ResumedEvent
    | BudgetAmendedEvent
    | ExplorationStoppedEvent,
    Field(discriminator="event_type"),
]


class ExplorationLoopState(BaseModel):
    """Rebuildable state derived exclusively from a complete journal prefix."""

    schema_version: int = EXPLORATION_JOURNAL_SCHEMA_VERSION
    exploration_id: str = Field(min_length=1)
    policy_fingerprint: str = Field(min_length=1)
    effective_policy_fingerprint: str = Field(min_length=1)
    code_fingerprint: str = Field(min_length=1)
    data_state_witness: str = Field(min_length=1)
    attempt_epoch: int = Field(ge=0)
    status: ExplorationRunStatus = "running"
    stop_reason: ExplorationStopReason | None = None
    final_report_ref: str | None = None

    max_llm_requests: int | None = Field(default=None, ge=1)
    max_successful_tool_calls: int = Field(ge=1)
    max_rounds: int = Field(ge=1)

    llm_calls_settled: int = Field(ge=0, default=0)
    llm_calls_uncertain: int = Field(ge=0, default=0)
    tool_calls_committed: int = Field(ge=0, default=0)
    rounds_started: int = Field(ge=0, default=0)
    rounds_settled: int = Field(ge=0, default=0)
    consecutive_no_progress: int = Field(ge=0, default=0)
    current_round_index: int | None = Field(default=None, ge=0)

    remaining_llm_call_budget: int | None = Field(default=None, ge=0)
    remaining_tool_call_budget: int = Field(ge=0)
    remaining_round_budget: int = Field(ge=0)

    pending_call_id: str | None = None
    pending_logical_step_id: str | None = None
    prepared_receipt_id: str | None = None

    completed_step_ids: list[str] = Field(default_factory=list)
    step_receipt_refs: dict[str, str] = Field(default_factory=dict)
    failure_history: list[str] = Field(default_factory=list)
    amendment_ids: list[str] = Field(default_factory=list)
    gate_verdicts: dict[str, GateVerdictValue] = Field(default_factory=dict)

    frontier_digest: str | None = None
    ledger_digest: str | None = None

    last_seq: int = Field(ge=0)

    @model_validator(mode="after")
    def _derived_values_are_consistent(self) -> ExplorationLoopState:
        if len(self.completed_step_ids) != len(set(self.completed_step_ids)):
            raise ValueError("completed_step_ids must be unique.")
        if self.pending_call_id is not None and self.pending_logical_step_id is not None:
            raise ValueError("a call and a tool step cannot both be pending.")
        if self.prepared_receipt_id is not None and self.pending_logical_step_id is None:
            raise ValueError("a prepared receipt requires a pending tool step.")
        if self.remaining_tool_call_budget != (
            self.max_successful_tool_calls - self.tool_calls_committed
        ):
            raise ValueError("remaining_tool_call_budget is inconsistent.")
        if self.remaining_round_budget != self.max_rounds - self.rounds_started:
            raise ValueError("remaining_round_budget is inconsistent.")
        if self.max_llm_requests is None:
            if self.remaining_llm_call_budget is not None:
                raise ValueError("remaining_llm_call_budget requires max_llm_requests.")
        elif self.remaining_llm_call_budget != (
            self.max_llm_requests - self.llm_calls_settled - self.llm_calls_uncertain
        ):
            raise ValueError("remaining_llm_call_budget is inconsistent.")
        if self.rounds_started != self.rounds_settled + (
            1 if self.current_round_index is not None else 0
        ):
            raise ValueError("round counters are inconsistent with the open round.")
        if (self.stop_reason is not None) != (self.status == "stopped"):
            raise ValueError("stop_reason must be set exactly when status is stopped.")
        return self

    def require_stop_reason(self) -> ExplorationStopReason:
        if self.stop_reason is None:
            raise ExplorationStateUnavailableError(
                "stop_reason is not restored: the exploration has not stopped."
            )
        return self.stop_reason

    def require_frontier_digest(self) -> str:
        if self.frontier_digest is None:
            raise ExplorationStateUnavailableError(
                "frontier_digest is not restored: no reduction has been committed."
            )
        return self.frontier_digest

    def require_ledger_digest(self) -> str:
        if self.ledger_digest is None:
            raise ExplorationStateUnavailableError(
                "ledger_digest is not restored: no reduction has been committed."
            )
        return self.ledger_digest

    def require_current_round_index(self) -> int:
        if self.current_round_index is None:
            raise ExplorationStateUnavailableError(
                "current_round_index is not restored: no round is open."
            )
        return self.current_round_index
