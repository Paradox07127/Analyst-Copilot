"""Level-2 bounded investigation loop artifacts (roadmap phase C).

The loop lets the LLM propose typed follow-up probes for ONE approved plan,
executed deterministically within the plan's scope, and MUST terminate via a
typed exit: a Conclude action, the hard step cap, or the LLM-call cap
(framework research 2026-07-17: `Union[Response, Plan]` pattern + hard
numeric ceilings — the only termination strategy practitioners actually
ship). Claims still come only from deterministic reducers; the LLM plans
probes and never authors numbers.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from eda_platform.schemas.questions import QuestionFinding

LoopExitReason = Literal[
    "concluded",
    "step_cap_reached",
    "llm_cap_reached",
    "probe_error_cap_reached",
    "repeated_action",
    "offline",
    "budget_exhausted",
]

LoopJournalStatus = Literal[
    "running",
    "concluded",
    "budget_exhausted",
    "failed",
    "uncertain",
    "resume_incompatible",
]

LoopJournalEventType = Literal[
    "loop_started",
    "attempt_started",
    "decision_call_started",
    "decision_call_completed",
    "decision_call_rejected",
    "probe_started",
    "artifact_committed",
    "probe_completed",
    "loop_concluded",
    "loop_budget_exhausted",
    "loop_failed",
    "loop_call_uncertain",
]

LOOP_JOURNAL_SCHEMA_VERSION = 1


class LoopStepRecord(BaseModel):
    """One executed probe (or the concluding step) of a bounded loop."""

    step_index: int = Field(ge=0)
    action: Literal["probe", "conclude"]
    purpose: str
    sql: str = ""
    result_artifact_id: str | None = None
    findings: list[QuestionFinding] = Field(default_factory=list)
    status: Literal["succeeded", "failed", "skipped"]
    error: str = ""

    @field_validator("purpose")
    @classmethod
    def _purpose_is_non_empty(cls, value: str) -> str:
        if value.strip():
            return value
        raise ValueError("purpose must be non-empty.")


class DeepInvestigationResult(BaseModel):
    """Transcript + typed exit of one bounded investigation loop."""

    schema_version: int = 1
    investigation_id: str
    question_id: str
    max_steps: int = Field(ge=1, le=10)
    llm_call_cap: int = Field(ge=1, le=30)
    steps: list[LoopStepRecord] = Field(default_factory=list)
    exit_reason: LoopExitReason
    llm_calls_used: int = Field(ge=0)
    probe_errors: int = Field(ge=0, default=0)
    conclusion_note: str = Field(
        default="",
        description="LLM concluding rationale; validator-gated like an L1 "
        "interpretation — never a number source",
    )

    @field_validator("investigation_id", "question_id")
    @classmethod
    def _required_strings_are_non_empty(cls, value: str) -> str:
        if value.strip():
            return value
        raise ValueError("field must be non-empty.")


class InvestigationLoopEvent(BaseModel):
    """One immutable transition in an investigation loop journal.

    Optional transition-specific fields are checked by the pure reducer. Keeping
    the envelope stable lets older journals be inspected before an explicit
    schema migration is applied.
    """

    schema_version: int = LOOP_JOURNAL_SCHEMA_VERSION
    seq: int = Field(ge=0)
    investigation_id: str
    event_type: LoopJournalEventType
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    attempt_epoch: int = Field(ge=0, default=0)

    source_session_id: str | None = None
    question_id: str | None = None
    plan_fingerprint: str | None = None
    policy_fingerprint: str | None = None
    code_fingerprint: str | None = None
    max_steps: int | None = Field(default=None, ge=1, le=10)
    llm_call_cap: int | None = Field(default=None, ge=1, le=30)

    iteration: int | None = Field(default=None, ge=0)
    step_id: str | None = None
    call_id: str | None = None
    probe_id: str | None = None
    probe_fingerprint: str | None = None
    response_hash: str | None = None
    typed_decision: dict[str, object] | None = None
    artifact_ref: str | None = None
    final_draft_ref: str | None = None
    error: str | None = None

    @field_validator(
        "investigation_id",
        "source_session_id",
        "question_id",
        "plan_fingerprint",
        "policy_fingerprint",
        "code_fingerprint",
        "step_id",
        "call_id",
        "probe_id",
        "probe_fingerprint",
        "response_hash",
        "artifact_ref",
        "final_draft_ref",
    )
    @classmethod
    def _present_identifiers_are_non_empty(cls, value: str | None) -> str | None:
        if value is None or value.strip():
            return value
        raise ValueError("identifier/reference fields must be non-empty when present.")


class InvestigationLoopState(BaseModel):
    """Rebuildable state derived exclusively from a complete journal prefix."""

    schema_version: int = LOOP_JOURNAL_SCHEMA_VERSION
    investigation_id: str
    source_session_id: str
    question_id: str
    plan_fingerprint: str
    policy_fingerprint: str
    code_fingerprint: str
    attempt_epoch: int = Field(ge=0)
    status: LoopJournalStatus = "running"
    max_steps: int = Field(ge=1, le=10)
    llm_call_cap: int = Field(ge=1, le=30)
    next_iteration: int = Field(ge=0, default=0)
    probes_completed: int = Field(ge=0, default=0)
    llm_calls_settled: int = Field(ge=0, default=0)
    remaining_probe_budget: int = Field(ge=0)
    remaining_call_budget: int = Field(ge=0)
    failure_history: list[str] = Field(default_factory=list)
    completed_step_ids: list[str] = Field(default_factory=list)
    pending_call_id: str | None = None
    pending_probe_id: str | None = None
    step_artifact_refs: dict[str, str] = Field(default_factory=dict)
    final_draft_ref: str | None = None
    last_seq: int = Field(ge=0)

    @field_validator(
        "investigation_id",
        "source_session_id",
        "question_id",
        "plan_fingerprint",
        "policy_fingerprint",
        "code_fingerprint",
    )
    @classmethod
    def _state_identity_is_non_empty(cls, value: str) -> str:
        if value.strip():
            return value
        raise ValueError("state identity/fingerprint fields must be non-empty.")

    @model_validator(mode="after")
    def _derived_values_are_consistent(self) -> InvestigationLoopState:
        if len(self.completed_step_ids) != len(set(self.completed_step_ids)):
            raise ValueError("completed_step_ids must be unique.")
        if self.probes_completed > self.max_steps:
            raise ValueError("probes_completed exceeds max_steps.")
        if self.llm_calls_settled > self.llm_call_cap:
            raise ValueError("llm_calls_settled exceeds llm_call_cap.")
        if self.remaining_probe_budget != self.max_steps - self.probes_completed:
            raise ValueError("remaining_probe_budget is inconsistent.")
        if self.remaining_call_budget != self.llm_call_cap - self.llm_calls_settled:
            raise ValueError("remaining_call_budget is inconsistent.")
        if self.pending_call_id is not None and self.pending_probe_id is not None:
            raise ValueError("a call and a probe cannot both be pending.")
        return self
