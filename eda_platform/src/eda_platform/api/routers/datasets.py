"""Read-only dataset endpoints (§7.2): metadata, schema, preview — lazy paths
only, no CSV materialisation. Sync def: served from the worker threadpool."""

from __future__ import annotations

from fastapi import APIRouter, Header, Query, Request

from eda_platform.api.errors import ApiErrorEnvelope
from eda_platform.application.dto import (
    ColumnDistributionsView,
    DataOperationStarted,
    DatasetHandle,
    DatasetPreview,
    DatasetSchema,
)
from eda_platform.application.services.data_operation_service import DataOperationService
from eda_platform.application.services.dataset_service import (
    DEFAULT_PREVIEW_LIMIT,
    MAX_PREVIEW_LIMIT,
    DatasetService,
)

_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    400: {"model": ApiErrorEnvelope},
    404: {"model": ApiErrorEnvelope},
    422: {"model": ApiErrorEnvelope},
    500: {"model": ApiErrorEnvelope},
}

router = APIRouter(tags=["datasets"], responses=_ERROR_RESPONSES)


def _service(request: Request) -> DatasetService:
    return request.app.state.dataset_service


def _operations(request: Request) -> DataOperationService:
    return request.app.state.data_operation_service


@router.get("/sessions/{session_id}/datasets", response_model=list[DatasetHandle])
def list_datasets(session_id: str, request: Request) -> list[DatasetHandle]:
    return _service(request).list_datasets(session_id)


@router.get("/sessions/{session_id}/datasets/{dataset_id}/schema", response_model=DatasetSchema)
def get_dataset_schema(session_id: str, dataset_id: str, request: Request) -> DatasetSchema:
    return _service(request).get_schema(dataset_id, session_id)


@router.get("/sessions/{session_id}/datasets/{dataset_id}/preview", response_model=DatasetPreview)
def get_dataset_preview(
    session_id: str,
    dataset_id: str,
    request: Request,
    limit: int = Query(DEFAULT_PREVIEW_LIMIT, ge=1, le=MAX_PREVIEW_LIMIT),
    offset: int = Query(0, ge=0),
) -> DatasetPreview:
    return _service(request).get_preview(dataset_id, session_id, limit=limit, offset=offset)


@router.post(
    "/sessions/{session_id}/datasets/{dataset_id}/distributions",
    status_code=202,
    response_model=DataOperationStarted,
)
def start_dataset_distributions(
    session_id: str,
    dataset_id: str,
    request: Request,
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
) -> DataOperationStarted:
    return _operations(request).start(
        session_id,
        kind="dataset_distributions",
        params={"dataset_id": dataset_id},
        idempotency_key=idempotency_key,
    )


@router.get(
    "/jobs/{job_id}/dataset-distributions-result",
    response_model=ColumnDistributionsView,
)
def get_dataset_distributions_result(
    job_id: str,
    request: Request,
) -> ColumnDistributionsView:
    return _operations(request).result(
        job_id,
        expected_kind="dataset_distributions",
        model=ColumnDistributionsView,
    )
