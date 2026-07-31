"""Questions endpoints (§7.5): list candidates with execution status, prepare
a server-side pending approval for one question, and execute it by consuming
the approval into a `question_exec` job on a derived batch run. The write side
adds card editing (inline, deterministic) and free-text drafting (approved,
then a `question_draft` job)."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Header, Request
from pydantic import BaseModel, Field

from eda_platform.api.errors import ApiErrorEnvelope
from eda_platform.api.routers.settings import SESSION_HEADER, session_id_from_header
from eda_platform.application.dto import (
    QuestionDraftPrepared,
    QuestionDraftStarted,
    QuestionExecutionPrepared,
    QuestionExecutionStarted,
    QuestionSummary,
    QuestionsView,
)
from eda_platform.application.services.question_service import (
    QuestionService,
    disclosure_settings_fingerprint,
)

_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    404: {"model": ApiErrorEnvelope},
    409: {"model": ApiErrorEnvelope},
    410: {"model": ApiErrorEnvelope},
    422: {"model": ApiErrorEnvelope},
    500: {"model": ApiErrorEnvelope},
}

router = APIRouter(tags=["questions"], responses=_ERROR_RESPONSES)


class QuestionPrepareRequest(BaseModel):
    llm: Literal["env", "offline"] = "env"


class QuestionExecuteRequest(BaseModel):
    """No `llm` field on purpose: execute runs the mode bound into the
    approval at prepare time; a client-sent value would bypass the review."""

    action_hash: str = Field(min_length=8)
    approval_token: str = Field(min_length=8)


def _service(request: Request) -> QuestionService:
    return request.app.state.question_service


@router.get("/sessions/{session_id}/questions", response_model=QuestionsView)
def list_questions(session_id: str, request: Request) -> QuestionsView:
    return _service(request).list_questions(session_id)


@router.post(
    "/sessions/{session_id}/questions/{question_id}/prepare",
    response_model=QuestionExecutionPrepared,
)
def prepare_question_execution(
    session_id: str,
    question_id: str,
    request: Request,
    body: QuestionPrepareRequest | None = None,
) -> QuestionExecutionPrepared:
    llm = body.llm if body is not None else "env"
    return _service(request).prepare_execution(session_id, question_id, llm=llm)


@router.post(
    "/sessions/{session_id}/questions/{question_id}/execute",
    status_code=201,
    response_model=QuestionExecutionStarted,
)
def execute_question(
    session_id: str,
    question_id: str,
    body: QuestionExecuteRequest,
    request: Request,
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
) -> QuestionExecutionStarted:
    return _service(request).execute(
        session_id,
        question_id=question_id,
        action_hash=body.action_hash,
        approval_token=body.approval_token,
        idempotency_key=idempotency_key,
    )


class QuestionCardEditRequest(BaseModel):
    """Framing fields only. SQL, target datasets and analysis mode are refused
    by the driver: changing them would alter what a prior approval covered."""

    expected_version: int = Field(ge=0)
    question_en: str | None = None
    business_decision: str | None = None
    value_hypothesis: str | None = None
    success_criterion: str | None = None
    data_signal: str | None = None
    priority_rationale: str | None = None
    risks: list[str] | None = None
    data_requirements: list[str] | None = None


class QuestionDraftPrepareRequest(BaseModel):
    question: str = Field(min_length=1)
    llm: Literal["env", "offline"] = "env"


class QuestionDraftRequest(BaseModel):
    """No question field: drafting runs the text bound into the approval."""

    action_hash: str = Field(min_length=8)
    approval_token: str = Field(min_length=8)


@router.patch(
    "/sessions/{session_id}/questions/{question_id}",
    response_model=QuestionSummary,
)
def edit_question_card(
    session_id: str,
    question_id: str,
    body: QuestionCardEditRequest,
    request: Request,
) -> QuestionSummary:
    """Rewrite a card's framing text; the driver bumps card_version and
    re-evaluates feasibility on the edited card."""
    return _service(request).edit_card(
        session_id,
        question_id,
        body.model_dump(
            exclude={"expected_version"}, exclude_unset=True, exclude_none=True
        ),
        expected_version=body.expected_version,
    )


@router.post(
    "/sessions/{session_id}/questions/prepare-draft",
    response_model=QuestionDraftPrepared,
)
def prepare_question_draft(
    session_id: str,
    body: QuestionDraftPrepareRequest,
    request: Request,
    x_eda_session: str | None = Header(None, alias=SESSION_HEADER),
) -> QuestionDraftPrepared:
    effective = request.app.state.settings_service.resolve(session_id_from_header(x_eda_session))
    return _service(request).prepare_draft(
        session_id,
        body.question,
        llm=body.llm,
        payload_policy=effective.payload_policy,
        disclosure_fingerprint=disclosure_settings_fingerprint(
            payload_policy=effective.payload_policy,
            provider=effective.llm.provider.value,
            model=effective.llm.model,
            base_url=effective.llm.base_url,
        ),
    )


@router.post(
    "/sessions/{session_id}/questions",
    status_code=201,
    response_model=QuestionDraftStarted,
)
def draft_question_card(
    session_id: str,
    body: QuestionDraftRequest,
    request: Request,
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    x_eda_session: str | None = Header(None, alias=SESSION_HEADER),
) -> QuestionDraftStarted:
    """Draft a full question card from the approved free-text question."""
    effective = request.app.state.settings_service.resolve(session_id_from_header(x_eda_session))
    return _service(request).draft(
        session_id,
        action_hash=body.action_hash,
        approval_token=body.approval_token,
        payload_policy=effective.payload_policy,
        disclosure_fingerprint=disclosure_settings_fingerprint(
            payload_policy=effective.payload_policy,
            provider=effective.llm.provider.value,
            model=effective.llm.model,
            base_url=effective.llm.base_url,
        ),
        llm_env=effective.env_overlay,
        idempotency_key=idempotency_key,
    )
