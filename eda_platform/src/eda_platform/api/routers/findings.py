"""Findings endpoints (§7.5): the project-level findings library viewed from
one run, with evidence artifact references and freshness, plus promotion of a
fresh finding into the project's semantic knowledge (prepare → promote)."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from eda_platform.api.errors import ApiErrorEnvelope
from eda_platform.application.dto import (
    DecisionCoverageView,
    FindingsView,
    KnowledgePromoted,
    KnowledgePromotionPrepared,
)
from eda_platform.application.services.finding_service import FindingService
from eda_platform.application.services.promotion_service import PromotionService

_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    404: {"model": ApiErrorEnvelope},
    409: {"model": ApiErrorEnvelope},
    410: {"model": ApiErrorEnvelope},
    422: {"model": ApiErrorEnvelope},
    500: {"model": ApiErrorEnvelope},
}

router = APIRouter(tags=["findings"], responses=_ERROR_RESPONSES)


class FindingPromoteRequest(BaseModel):
    """No answer text on purpose: promote writes exactly what the approval
    froze at prepare time, so a client cannot edit the knowledge in flight."""

    action_hash: str = Field(min_length=8)
    approval_token: str = Field(min_length=8)


@router.get("/sessions/{session_id}/findings", response_model=FindingsView)
def list_findings(session_id: str, request: Request) -> FindingsView:
    service: FindingService = request.app.state.finding_service
    return service.list_findings(session_id)


@router.get("/sessions/{session_id}/decision-coverage", response_model=DecisionCoverageView)
def get_decision_coverage(session_id: str, request: Request) -> DecisionCoverageView:
    service: FindingService = request.app.state.finding_service
    return service.get_decision_coverage(session_id)


def _promotion(request: Request) -> PromotionService:
    return request.app.state.promotion_service


@router.post(
    "/sessions/{session_id}/findings/{finding_id}/prepare-promote",
    response_model=KnowledgePromotionPrepared,
)
def prepare_promotion(
    session_id: str, finding_id: str, request: Request
) -> KnowledgePromotionPrepared:
    return _promotion(request).prepare(session_id, finding_id)


@router.post(
    "/sessions/{session_id}/findings/{finding_id}/promote",
    status_code=201,
    response_model=KnowledgePromoted,
)
def promote_finding(
    session_id: str,
    finding_id: str,
    body: FindingPromoteRequest,
    request: Request,
) -> KnowledgePromoted:
    return _promotion(request).promote(
        session_id,
        finding_id=finding_id,
        action_hash=body.action_hash,
        approval_token=body.approval_token,
    )
