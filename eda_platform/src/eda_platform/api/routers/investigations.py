"""Investigation governance endpoints (Questions page plan
lifecycle): list plans with their decision and outcome, build plans for
selected questions, approve or reject one plan, execute the approved set, and
run the Ultra macro loop. Sync def: served from the worker threadpool."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Header, Request
from pydantic import BaseModel, Field

from eda_platform.api.errors import ApiErrorEnvelope
from eda_platform.api.routers.settings import SESSION_HEADER, session_id_from_header
from eda_platform.application.dto import (
    InvestigationDecisionPrepared,
    InvestigationDecisionRecorded,
    InvestigationExecutionPrepared,
    InvestigationExecutionStarted,
    InvestigationPlanBuildStarted,
    InvestigationsView,
    MacroLoopPrepared,
    MacroLoopStarted,
)
from eda_platform.application.services.investigation_service import InvestigationService

_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    404: {"model": ApiErrorEnvelope},
    409: {"model": ApiErrorEnvelope},
    410: {"model": ApiErrorEnvelope},
    422: {"model": ApiErrorEnvelope},
    500: {"model": ApiErrorEnvelope},
}

router = APIRouter(tags=["investigations"], responses=_ERROR_RESPONSES)


def _service(request: Request) -> InvestigationService:
    return request.app.state.investigation_service


def _effective(request: Request, session: str | None):
    return request.app.state.settings_service.resolve(session_id_from_header(session))


class InvestigationPlanRequest(BaseModel):
    question_ids: list[str] = Field(min_length=1)
    deep: bool = False
    """Deep probes are also enabled by a thinking level of Deep or higher."""


class InvestigationDecisionPrepareRequest(BaseModel):
    decision: Literal["approved", "rejected"]
    reason: str = Field(default="", max_length=1000)


class InvestigationDecisionRequest(BaseModel):
    """No decision or reason field: the decision runs what the approval bound."""

    action_hash: str = Field(min_length=8)
    approval_token: str = Field(min_length=8)


class InvestigationExecutePrepareRequest(BaseModel):
    plan_ids: list[str] = Field(min_length=1)
    llm: Literal["env", "offline"] = "env"


class InvestigationExecuteRequest(BaseModel):
    """No plan ids: execution runs exactly the set the approval froze."""

    action_hash: str = Field(min_length=8)
    approval_token: str = Field(min_length=8)


class MacroLoopPrepareRequest(BaseModel):
    plan_session_id: str = Field(min_length=1)
    llm: Literal["env", "offline"] = "env"


class MacroLoopRequest(BaseModel):
    action_hash: str = Field(min_length=8)
    approval_token: str = Field(min_length=8)


@router.get("/sessions/{session_id}/investigations", response_model=InvestigationsView)
def list_investigations(
    session_id: str,
    request: Request,
    x_eda_session: str | None = Header(None, alias=SESSION_HEADER),
) -> InvestigationsView:
    effective = _effective(request, x_eda_session)
    return _service(request).list_investigations(
        session_id, analysis_depth=effective.analysis_depth
    )


@router.post(
    "/sessions/{session_id}/investigations/plan",
    status_code=201,
    response_model=InvestigationPlanBuildStarted,
)
def build_investigation_plans(
    session_id: str,
    body: InvestigationPlanRequest,
    request: Request,
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    x_eda_session: str | None = Header(None, alias=SESSION_HEADER),
) -> InvestigationPlanBuildStarted:
    """Plan building is deterministic and spends no model budget, so it needs
    no approval — only the job system's idempotency key."""
    effective = _effective(request, x_eda_session)
    return _service(request).build_plans(
        session_id,
        question_ids=body.question_ids,
        deep=body.deep,
        analysis_depth=effective.analysis_depth,
        idempotency_key=idempotency_key,
    )


@router.post(
    "/sessions/{session_id}/investigations/{plan_id}/prepare-decision",
    response_model=InvestigationDecisionPrepared,
)
def prepare_investigation_decision(
    session_id: str,
    plan_id: str,
    body: InvestigationDecisionPrepareRequest,
    request: Request,
) -> InvestigationDecisionPrepared:
    return _service(request).prepare_decision(
        session_id, plan_id, decision=body.decision, reason=body.reason
    )


@router.post(
    "/sessions/{session_id}/investigations/{plan_id}/approve",
    response_model=InvestigationDecisionRecorded,
)
def approve_investigation_plan(
    session_id: str,
    plan_id: str,
    body: InvestigationDecisionRequest,
    request: Request,
) -> InvestigationDecisionRecorded:
    return _service(request).decide(
        session_id,
        plan_id,
        decision="approved",
        action_hash=body.action_hash,
        approval_token=body.approval_token,
    )


@router.post(
    "/sessions/{session_id}/investigations/{plan_id}/reject",
    response_model=InvestigationDecisionRecorded,
)
def reject_investigation_plan(
    session_id: str,
    plan_id: str,
    body: InvestigationDecisionRequest,
    request: Request,
) -> InvestigationDecisionRecorded:
    return _service(request).decide(
        session_id,
        plan_id,
        decision="rejected",
        action_hash=body.action_hash,
        approval_token=body.approval_token,
    )


@router.post(
    "/sessions/{session_id}/investigations/prepare-execute",
    response_model=InvestigationExecutionPrepared,
)
def prepare_investigation_execution(
    session_id: str,
    body: InvestigationExecutePrepareRequest,
    request: Request,
) -> InvestigationExecutionPrepared:
    return _service(request).prepare_execution(
        session_id, plan_ids=body.plan_ids, llm=body.llm
    )


@router.post(
    "/sessions/{session_id}/investigations/execute",
    status_code=201,
    response_model=InvestigationExecutionStarted,
)
def execute_investigation_plans(
    session_id: str,
    body: InvestigationExecuteRequest,
    request: Request,
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    x_eda_session: str | None = Header(None, alias=SESSION_HEADER),
) -> InvestigationExecutionStarted:
    effective = _effective(request, x_eda_session)
    return _service(request).execute(
        session_id,
        action_hash=body.action_hash,
        approval_token=body.approval_token,
        payload_policy=effective.payload_policy,
        llm_env=effective.env_overlay,
        idempotency_key=idempotency_key,
    )


@router.post(
    "/sessions/{session_id}/investigations/prepare-macro-loop",
    response_model=MacroLoopPrepared,
)
def prepare_macro_loop(
    session_id: str,
    body: MacroLoopPrepareRequest,
    request: Request,
    x_eda_session: str | None = Header(None, alias=SESSION_HEADER),
) -> MacroLoopPrepared:
    """The response states how many follow-up rounds approving this authorizes;
    the depth comes from the session's thinking level, not from the request."""
    effective = _effective(request, x_eda_session)
    return _service(request).prepare_macro_loop(
        session_id,
        plan_session_id=body.plan_session_id,
        analysis_depth=effective.analysis_depth,
        llm=body.llm,
    )


@router.post(
    "/sessions/{session_id}/investigations/macro-loop",
    status_code=201,
    response_model=MacroLoopStarted,
)
def start_macro_loop(
    session_id: str,
    body: MacroLoopRequest,
    request: Request,
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    x_eda_session: str | None = Header(None, alias=SESSION_HEADER),
) -> MacroLoopStarted:
    effective = _effective(request, x_eda_session)
    return _service(request).start_macro_loop(
        session_id,
        action_hash=body.action_hash,
        approval_token=body.approval_token,
        payload_policy=effective.payload_policy,
        llm_env=effective.env_overlay,
        idempotency_key=idempotency_key,
    )
