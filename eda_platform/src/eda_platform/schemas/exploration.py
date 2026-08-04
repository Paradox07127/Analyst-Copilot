"""Exploration run schemas: policy, insight families, journal events and state.

Layering: schemas never import core. The policy fingerprint is computed in
core.exploration_journal and stored here as an opaque field; pause is a
resumable status and deliberately absent from the stop reasons (plan R3.1).
"""

from __future__ import annotations

import re
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

ExplorationGracefulStopReason = Literal[
    "completed",
    "budget_exhausted",
    "no_new_information",
]

ExplorationRunStatus = Literal["running", "pause_requested", "paused", "stopped"]

GateVerdictValue = Literal["passed", "rejected"]

MAIN_LINE_ID = "main"
_BRANCH_ID_PATTERN = r"^br_[1-9][0-9]*(\.[1-9][0-9]*)?$"  # depth <= 2 by shape

BranchConstraintReason = Literal["refuted", "gate_rejected", "inconclusive"]


class BranchConstraint(BaseModel):
    """One structured "tried and why it failed" fact from an abandoned line.

    Content must be deterministically recomputable from committed receipts and
    gate reports; the issuer re-derives it and rejects tampering (plan E6 #4).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    hypothesis_fingerprint: str = Field(min_length=1)
    coverage_key: str = Field(min_length=1)
    family: InsightFamily
    reason: BranchConstraintReason
    detail_code: str = Field(min_length=1)


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
    # None = the main line. Branch rounds carry their line id; the journal
    # stays linear and round non-overlap rules are unchanged (plan E6 #5).
    branch_id: str | None = Field(default=None, pattern=_BRANCH_ID_PATTERN)


class LlmCallStartedEvent(ExplorationEventBase):
    event_type: Literal["llm_call_started"] = "llm_call_started"
    call_id: str = Field(min_length=1)
    step_id: str | None = Field(default=None, min_length=1)


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
    tool_kind: str = Field(default="legacy_unknown", min_length=1)
    projected_rows_scanned: int = Field(default=0, ge=0)
    projected_result_cells: int = Field(default=0, ge=0)


class ReceiptPreparedEvent(ExplorationEventBase):
    event_type: Literal["receipt_prepared"] = "receipt_prepared"
    logical_step_id: str = Field(min_length=1)
    receipt_id: str = Field(min_length=1)
    result_digest: str | None = Field(default=None, min_length=1)


class ReceiptCommittedEvent(ExplorationEventBase):
    event_type: Literal["receipt_committed"] = "receipt_committed"
    logical_step_id: str = Field(min_length=1)
    receipt_id: str = Field(min_length=1)
    rows_scanned: int = Field(default=0, ge=0)
    result_cells: int = Field(default=0, ge=0)
    result_digest: str | None = Field(default=None, min_length=1)


class ToolCallFailedEvent(ExplorationEventBase):
    event_type: Literal["tool_call_failed"] = "tool_call_failed"
    logical_step_id: str = Field(min_length=1)
    error: str = Field(min_length=1)
    rows_scanned: int = Field(default=0, ge=0)
    result_cells: int = Field(default=0, ge=0)


class GateVerdictEvent(ExplorationEventBase):
    event_type: Literal["gate_verdict"] = "gate_verdict"
    claim_bundle_id: str = Field(min_length=1)
    verdict: GateVerdictValue


class ReductionCommittedEvent(ExplorationEventBase):
    event_type: Literal["reduction_committed"] = "reduction_committed"
    frontier_digest: str = Field(min_length=1)
    ledger_digest: str = Field(min_length=1)
    reduction_digest: str | None = Field(default=None, min_length=1)


class RoundSettledEvent(ExplorationEventBase):
    event_type: Literal["round_settled"] = "round_settled"
    round_index: int = Field(ge=0)
    progress: bool
    # Distinct from "no progress": the scheduler admitted nothing at all this
    # round. One such round is a gap; a streak of them is exhaustion.
    frontier_empty: bool = False
    # Adjudicated (new/reinforced/refuted) insight transitions this round.
    # None on pre-plan-B journals; treated as movement so legacy runs never
    # soft-stop retroactively.
    adjudicated_transitions: int | None = Field(default=None, ge=0)
    terminal_reason: ExplorationGracefulStopReason | None = None
    terminal_has_reduction: bool = False

    @model_validator(mode="after")
    def _terminal_metadata_is_consistent(self) -> RoundSettledEvent:
        if self.terminal_has_reduction and self.terminal_reason is None:
            raise ValueError("terminal_has_reduction requires terminal_reason.")
        return self


class BranchAbandonedEvent(ExplorationEventBase):
    """The current line is abandoned; its negatives become admission constraints."""

    event_type: Literal["branch_abandoned"] = "branch_abandoned"
    branch_id: str = Field(min_length=1)
    round_index: int = Field(ge=0)
    constraints: tuple[BranchConstraint, ...] = ()

    @model_validator(mode="after")
    def _line_id_is_valid(self) -> BranchAbandonedEvent:
        if self.branch_id != MAIN_LINE_ID and not re.fullmatch(
            _BRANCH_ID_PATTERN, self.branch_id
        ):
            raise ValueError("branch_id must be 'main' or a valid branch id.")
        return self


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
    | BranchAbandonedEvent
    | PauseRequestedEvent
    | PausedEvent
    | ResumedEvent
    | BudgetAmendedEvent
    | ExplorationStoppedEvent,
    Field(discriminator="event_type"),
]


class PendingToolStep(BaseModel):
    """One in-flight tool slot, keyed by its logical_step_id in the loop state."""

    model_config = ConfigDict(extra="forbid")

    tool_kind: str | None = None
    input_fingerprint: str | None = None
    projected_rows_scanned: int = Field(ge=0, default=0)
    projected_result_cells: int = Field(ge=0, default=0)
    prepared_receipt_id: str | None = None
    prepared_result_digest: str | None = None

    @model_validator(mode="after")
    def _prepared_fields_are_consistent(self) -> PendingToolStep:
        if self.prepared_result_digest is not None and self.prepared_receipt_id is None:
            raise ValueError("a prepared result digest requires a prepared receipt.")
        return self


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
    max_tool_calls_by_kind: dict[str, int] = Field(default_factory=dict)
    max_rows_scanned: int | None = Field(default=None, ge=1)
    max_result_cells: int | None = Field(default=None, ge=1)
    max_rounds: int = Field(ge=1)

    llm_calls_settled: int = Field(ge=0, default=0)
    llm_calls_uncertain: int = Field(ge=0, default=0)
    tool_calls_committed: int = Field(ge=0, default=0)
    tool_calls_by_kind: dict[str, int] = Field(default_factory=dict)
    rows_scanned: int = Field(ge=0, default=0)
    result_cells: int = Field(ge=0, default=0)
    rounds_started: int = Field(ge=0, default=0)
    rounds_settled: int = Field(ge=0, default=0)
    consecutive_no_progress: int = Field(ge=0, default=0)
    consecutive_empty_frontier: int = Field(ge=0, default=0)
    consecutive_no_adjudication: int = Field(ge=0, default=0)
    branch_trigger_stagnant_rounds: int | None = Field(default=None, ge=1)
    max_branches: int = Field(ge=0, default=0)
    active_branch_id: str | None = Field(default=None, pattern=_BRANCH_ID_PATTERN)
    current_line_abandoned: bool = False
    started_branch_ids: list[str] = Field(default_factory=list)
    abandoned_line_ids: list[str] = Field(default_factory=list)
    abandoned_constraints: list[BranchConstraint] = Field(default_factory=list)
    current_round_index: int | None = Field(default=None, ge=0)
    current_round_reduction_committed: bool = False
    pending_terminal_reason: ExplorationGracefulStopReason | None = None
    pending_terminal_has_reduction: bool = False

    remaining_llm_call_budget: int | None = Field(default=None, ge=0)
    remaining_tool_call_budget: int = Field(ge=0)
    remaining_round_budget: int = Field(ge=0)

    # Multi-slot pending state: any number of tool steps and LLM calls may be
    # in flight at once (parallel probe sessions, plan §3 P1/P2).
    pending_tool_steps: dict[str, PendingToolStep] = Field(default_factory=dict)
    pending_call_ids: tuple[str, ...] = ()
    # call_id -> bound logical step (only calls that declared a step_id).
    pending_call_steps: dict[str, str] = Field(default_factory=dict)

    # Legacy single-slot fields. They stay in the schema so old snapshots load,
    # but the reducer keeps them at their defaults forever; a non-default value
    # is migrated into the multi-slot fields by _migrate_legacy_pending.
    pending_call_id: str | None = None
    pending_call_step_id: str | None = None
    pending_logical_step_id: str | None = None
    prepared_receipt_id: str | None = None
    prepared_result_digest: str | None = None
    pending_tool_kind: str | None = None
    pending_tool_input_fingerprint: str | None = None
    pending_projected_rows_scanned: int = Field(ge=0, default=0)
    pending_projected_result_cells: int = Field(ge=0, default=0)

    completed_step_ids: list[str] = Field(default_factory=list)
    completed_probe_fingerprints: list[str] = Field(default_factory=list)
    uncertain_call_ids: list[str] = Field(default_factory=list)
    step_receipt_refs: dict[str, str] = Field(default_factory=dict)
    step_result_digests: dict[str, str] = Field(default_factory=dict)
    failure_history: list[str] = Field(default_factory=list)
    amendment_ids: list[str] = Field(default_factory=list)
    gate_verdicts: dict[str, GateVerdictValue] = Field(default_factory=dict)

    frontier_digest: str | None = None
    ledger_digest: str | None = None
    reduction_digest: str | None = None

    last_seq: int = Field(ge=0)

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_pending(cls, data: object) -> object:
        """Map an old single-slot snapshot (scalar pending fields) into one
        multi-slot entry each; new states always carry the scalars as None."""
        if not isinstance(data, dict):
            return data
        legacy_call = data.get("pending_call_id")
        legacy_step = data.get("pending_logical_step_id")
        if legacy_call is None and legacy_step is None:
            return data
        data = dict(data)
        if legacy_call is not None:
            call_ids = tuple(data.get("pending_call_ids") or ())
            if legacy_call not in call_ids:
                data["pending_call_ids"] = (*call_ids, legacy_call)
                bound_step = data.get("pending_call_step_id")
                if bound_step is not None:
                    call_steps = dict(data.get("pending_call_steps") or {})
                    call_steps[legacy_call] = bound_step
                    data["pending_call_steps"] = call_steps
            data["pending_call_id"] = None
            data["pending_call_step_id"] = None
        if legacy_step is not None:
            slots = dict(data.get("pending_tool_steps") or {})
            if legacy_step not in slots:
                slots[legacy_step] = {
                    "tool_kind": data.get("pending_tool_kind"),
                    "input_fingerprint": data.get("pending_tool_input_fingerprint"),
                    "projected_rows_scanned": data.get("pending_projected_rows_scanned")
                    or 0,
                    "projected_result_cells": data.get("pending_projected_result_cells")
                    or 0,
                    "prepared_receipt_id": data.get("prepared_receipt_id"),
                    "prepared_result_digest": data.get("prepared_result_digest"),
                }
                data["pending_tool_steps"] = slots
            data["pending_logical_step_id"] = None
            data["pending_tool_kind"] = None
            data["pending_tool_input_fingerprint"] = None
            data["pending_projected_rows_scanned"] = 0
            data["pending_projected_result_cells"] = 0
            data["prepared_receipt_id"] = None
            data["prepared_result_digest"] = None
        return data

    @model_validator(mode="after")
    def _derived_values_are_consistent(self) -> ExplorationLoopState:
        if len(self.completed_step_ids) != len(set(self.completed_step_ids)):
            raise ValueError("completed_step_ids must be unique.")
        if len(self.pending_call_ids) != len(set(self.pending_call_ids)):
            raise ValueError("pending_call_ids must be unique.")
        if not set(self.pending_call_steps) <= set(self.pending_call_ids):
            raise ValueError("pending_call_steps keys must be pending call ids.")
        if set(self.pending_tool_steps) & set(self.completed_step_ids):
            raise ValueError("a pending tool step cannot already be completed.")
        if self.pending_call_step_id is not None and self.pending_call_id is None:
            raise ValueError("a pending call step requires a pending call.")
        if len(self.completed_probe_fingerprints) != len(
            set(self.completed_probe_fingerprints)
        ):
            raise ValueError("completed_probe_fingerprints must be unique.")
        if len(self.uncertain_call_ids) != len(set(self.uncertain_call_ids)):
            raise ValueError("uncertain_call_ids must be unique.")
        if any(not kind or count < 0 for kind, count in self.tool_calls_by_kind.items()):
            raise ValueError("tool_calls_by_kind must contain non-negative counts.")
        if any(
            not kind or cap < 1 for kind, cap in self.max_tool_calls_by_kind.items()
        ):
            raise ValueError("max_tool_calls_by_kind must contain positive caps.")
        if sum(self.tool_calls_by_kind.values()) != self.tool_calls_committed:
            raise ValueError("tool_calls_by_kind must sum to tool_calls_committed.")
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
        if self.current_round_reduction_committed and self.current_round_index is None:
            raise ValueError("a current-round reduction requires an open round.")
        if self.pending_terminal_reason is not None and self.current_round_index is not None:
            raise ValueError("a pending terminal decision cannot have an open round.")
        if self.pending_terminal_has_reduction and self.pending_terminal_reason is None:
            raise ValueError("pending terminal reduction requires a terminal decision.")
        if (self.stop_reason is not None) != (self.status == "stopped"):
            raise ValueError("stop_reason must be set exactly when status is stopped.")
        if self.branch_trigger_stagnant_rounds is None:
            if (
                self.max_branches
                or self.active_branch_id is not None
                or self.current_line_abandoned
                or self.started_branch_ids
                or self.abandoned_line_ids
                or self.abandoned_constraints
            ):
                raise ValueError("branch state requires branching to be enabled.")
        else:
            if self.max_branches < 1:
                raise ValueError("enabled branching requires max_branches >= 1.")
            if len(self.started_branch_ids) != len(set(self.started_branch_ids)):
                raise ValueError("started_branch_ids must be unique.")
            if len(self.abandoned_line_ids) != len(set(self.abandoned_line_ids)):
                raise ValueError("abandoned_line_ids must be unique.")
            if len(self.started_branch_ids) > self.max_branches:
                raise ValueError("started branches cannot exceed max_branches.")
            if self.active_branch_id is not None and (
                self.active_branch_id not in self.started_branch_ids
            ):
                raise ValueError("active_branch_id must be a started branch.")
            if self.current_line_abandoned and (
                (self.active_branch_id or MAIN_LINE_ID) not in self.abandoned_line_ids
            ):
                raise ValueError("an abandoned current line must be recorded.")
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
