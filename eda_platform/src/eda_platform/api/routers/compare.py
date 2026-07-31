"""Compare endpoint (§10.3): two runs of one project side by side, read-only."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request

from eda_platform.api.errors import ApiErrorEnvelope
from eda_platform.application.dto import CompareView
from eda_platform.application.services.compare_service import CompareService

_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    404: {"model": ApiErrorEnvelope},
    422: {"model": ApiErrorEnvelope},
    500: {"model": ApiErrorEnvelope},
}

router = APIRouter(tags=["compare"], responses=_ERROR_RESPONSES)


@router.get("/compare", response_model=CompareView)
def compare_runs(
    request: Request,
    left: str = Query(min_length=1),
    right: str = Query(min_length=1),
) -> CompareView:
    service: CompareService = request.app.state.compare_service
    return service.compare_runs(left, right)
