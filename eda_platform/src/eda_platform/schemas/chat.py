from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from eda_platform.schemas.artifacts import Artifact
from eda_platform.schemas.plans import AnalysisPlan, Intent

ChatTurnStatus = Literal["answer", "awaiting_approval", "refused", "error"]


class SqlResultValidation(BaseModel):
    status: Literal["pass", "warn", "fail"]
    findings: list[str] = Field(default_factory=list)


class ChatTurnResult(BaseModel):
    intent: Intent
    message: str
    status: ChatTurnStatus = "answer"
    plan: AnalysisPlan | None = None
    artifacts: list[Artifact] = Field(default_factory=list)
    validation: SqlResultValidation | None = None
    sql: str | None = None
    pending_action: dict[str, Any] | None = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    status: str = "answer"
    sql: str | None = None
    artifact_refs: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # Set only on an `awaiting_approval` line so a reloaded transcript can find
    # the plan again. The approval token is deliberately absent: it is re-issued
    # from the pending_actions row, never read back off disk.
    plan_id: str | None = None
    action_hash: str | None = None
    expires_at: datetime | None = None
