"""Compare endpoint (§10.3): two runs of one project side by side, read-only."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Query, Request

from eda_platform.api.errors import ApiErrorEnvelope
from eda_platform.application.dto import CompareScopeName, CompareScopeView, CompareView
from eda_platform.application.services.compare_scope_service import CompareScopeService
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


@router.get("/compare/{scope}", response_model=CompareScopeView)
def compare_scope(
    scope: CompareScopeName,
    request: Request,
    left: str = Query(min_length=1),
    right: str = Query(min_length=1),
    filter_mode: Literal["all", "differences"] = Query(default="all", alias="filter"),
    limit: int = Query(default=50, ge=1, le=100),
    cursor: str | None = Query(default=None, max_length=2048),
) -> CompareScopeView:
    service: CompareScopeService = request.app.state.compare_scope_service
    return service.compare_scope(
        scope,
        left,
        right,
        filter_mode=filter_mode,
        limit=limit,
        cursor=cursor,
    )
