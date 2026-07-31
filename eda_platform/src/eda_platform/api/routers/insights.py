"""Read-only quality/profile/chart endpoints (§10.2 P1). Sync def: served
from the worker threadpool."""

from __future__ import annotations

from fastapi import APIRouter, Header, Query, Request

from eda_platform.api.errors import ApiErrorEnvelope
from eda_platform.application.dto import (
    ChartSummary,
    ChartView,
    CustomChartRequest,
    CustomChartView,
    DataOperationStarted,
    Page,
    ProfilesView,
    QualityView,
)
from eda_platform.application.services.data_operation_service import DataOperationService
from eda_platform.application.services.insight_service import (
    DEFAULT_CHART_LIMIT,
    MAX_CHART_LIMIT,
    InsightService,
)

_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    400: {"model": ApiErrorEnvelope},
    404: {"model": ApiErrorEnvelope},
    422: {"model": ApiErrorEnvelope},
    500: {"model": ApiErrorEnvelope},
}

router = APIRouter(tags=["insights"], responses=_ERROR_RESPONSES)


def _service(request: Request) -> InsightService:
    return request.app.state.insight_service


def _operations(request: Request) -> DataOperationService:
    return request.app.state.data_operation_service


@router.get("/sessions/{session_id}/quality", response_model=QualityView)
def get_quality(session_id: str, request: Request) -> QualityView:
    return _service(request).get_quality(session_id)


@router.get("/sessions/{session_id}/profiles", response_model=ProfilesView)
def get_profiles(session_id: str, request: Request) -> ProfilesView:
    return _service(request).get_profiles(session_id)


@router.get("/sessions/{session_id}/charts", response_model=Page[ChartSummary])
def list_charts(
    session_id: str,
    request: Request,
    limit: int = Query(DEFAULT_CHART_LIMIT, ge=1, le=MAX_CHART_LIMIT),
    cursor: str | None = Query(None),
) -> Page[ChartSummary]:
    return _service(request).list_charts(session_id, limit=limit, cursor=cursor)


@router.post(
    "/sessions/{session_id}/charts/custom",
    status_code=202,
    response_model=DataOperationStarted,
)
def build_custom_chart(
    session_id: str,
    body: CustomChartRequest,
    request: Request,
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
) -> DataOperationStarted:
    return _operations(request).start(
        session_id,
        kind="custom_chart",
        params={"chart": body.model_dump(mode="json")},
        idempotency_key=idempotency_key,
    )


@router.get(
    "/jobs/{job_id}/custom-chart-result",
    response_model=CustomChartView,
)
def get_custom_chart_result(
    job_id: str,
    request: Request,
) -> CustomChartView:
    return _operations(request).result(
        job_id,
        expected_kind="custom_chart",
        model=CustomChartView,
    )


@router.get("/sessions/{session_id}/charts/{chart_id}", response_model=ChartView)
def get_chart(session_id: str, chart_id: str, request: Request) -> ChartView:
    return _service(request).get_chart(session_id, chart_id)
