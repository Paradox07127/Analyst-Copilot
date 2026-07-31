"""Cleaning endpoints (§7.5): preview computes the recipe diff and registers a
server-side pending approval; apply consumes the approval hash and forks an
auto_eda job onto the cleaned dataset version. The two GETs are read-only
transparency views over what pre-cleaning recorded."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Header, Request
from pydantic import BaseModel, Field

from eda_platform.api.errors import ApiErrorEnvelope
from eda_platform.api.routers.settings import SESSION_HEADER, session_id_from_header
from eda_platform.application.dto import (
    CleaningApplied,
    CleaningLogView,
    CleaningPreviewResult,
    CleaningRawView,
    DataOperationStarted,
)
from eda_platform.application.services.cleaning_service import CleaningService
from eda_platform.application.services.data_operation_service import DataOperationService

_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    404: {"model": ApiErrorEnvelope},
    409: {"model": ApiErrorEnvelope},
    410: {"model": ApiErrorEnvelope},
    422: {"model": ApiErrorEnvelope},
    500: {"model": ApiErrorEnvelope},
}

router = APIRouter(tags=["cleaning"], responses=_ERROR_RESPONSES)


class CleaningPreviewRequest(BaseModel):
    dataset_id: str = Field(min_length=1)
    trim_whitespace: bool = True
    drop_duplicate_rows: bool = True
    drop_missing_rows: bool = False
    drop_outlier_rows: bool = False


class CleaningApplyRequest(BaseModel):
    action_hash: str = Field(min_length=8)
    approval_token: str = Field(min_length=8)
    llm: Literal["env", "offline"] = "env"


def _service(request: Request) -> CleaningService:
    return request.app.state.cleaning_service


def _operations(request: Request) -> DataOperationService:
    return request.app.state.data_operation_service


@router.get("/sessions/{session_id}/cleaning/log", response_model=CleaningLogView)
def get_cleaning_log(session_id: str, request: Request) -> CleaningLogView:
    return _service(request).get_log(session_id)


@router.get("/sessions/{session_id}/cleaning/raw", response_model=CleaningRawView)
def get_cleaning_raw(session_id: str, request: Request) -> CleaningRawView:
    return _service(request).get_raw(session_id)


@router.post(
    "/sessions/{session_id}/cleaning/preview",
    status_code=202,
    response_model=DataOperationStarted,
)
def preview_cleaning(
    session_id: str,
    body: CleaningPreviewRequest,
    request: Request,
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
) -> DataOperationStarted:
    return _operations(request).start(
        session_id,
        kind="cleaning_preview",
        params=body.model_dump(),
        idempotency_key=idempotency_key,
    )


@router.post(
    "/sessions/{session_id}/cleaning/apply",
    status_code=202,
    response_model=DataOperationStarted,
)
def apply_cleaning(
    session_id: str,
    body: CleaningApplyRequest,
    request: Request,
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    x_eda_session: str | None = Header(None, alias=SESSION_HEADER),
) -> DataOperationStarted:
    # The data operation worker may create a second auto-EDA worker after the
    # cleaned version is written.  Preserve the session's in-memory provider
    # settings through that process tree; they must never be put in job params.
    settings = request.app.state.settings_service.resolve(
        session_id_from_header(x_eda_session)
    )
    return _operations(request).start(
        session_id,
        kind="cleaning_apply",
        params=body.model_dump(),
        idempotency_key=idempotency_key,
        llm_env=settings.env_overlay if body.llm == "env" else None,
    )


@router.get(
    "/jobs/{job_id}/cleaning-preview-result",
    response_model=CleaningPreviewResult,
)
def get_cleaning_preview_result(
    job_id: str,
    request: Request,
) -> CleaningPreviewResult:
    return _operations(request).result(
        job_id,
        expected_kind="cleaning_preview",
        model=CleaningPreviewResult,
    )


@router.get(
    "/jobs/{job_id}/cleaning-apply-result",
    response_model=CleaningApplied,
)
def get_cleaning_apply_result(
    job_id: str,
    request: Request,
) -> CleaningApplied:
    return _operations(request).result(
        job_id,
        expected_kind="cleaning_apply",
        model=CleaningApplied,
    )
