"""Read-only deep-analysis endpoint (§10.2 P1). Sync def: served from the
worker threadpool."""

from __future__ import annotations

from fastapi import APIRouter, Request

from eda_platform.api.errors import ApiErrorEnvelope
from eda_platform.application.dto import AnalysisView
from eda_platform.application.services.analysis_service import AnalysisService

_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    404: {"model": ApiErrorEnvelope},
    422: {"model": ApiErrorEnvelope},
    500: {"model": ApiErrorEnvelope},
}

router = APIRouter(tags=["analysis"], responses=_ERROR_RESPONSES)


@router.get("/sessions/{session_id}/analysis", response_model=AnalysisView)
def get_analysis(session_id: str, request: Request) -> AnalysisView:
    service: AnalysisService = request.app.state.analysis_service
    return service.get_analysis(session_id)
