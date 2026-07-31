"""Report endpoints (§7.5): read the run's report, stream an export, and
generate or regenerate it on demand through a `report_generate` job. Sync def:
served from the worker threadpool."""

from __future__ import annotations

import io
from typing import Annotated, Literal

from fastapi import APIRouter, Header, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from eda_platform.api.errors import ApiErrorEnvelope
from eda_platform.api.routers.settings import SESSION_HEADER, session_id_from_header
from eda_platform.application.dto import ReportGenerationStarted, ReportView
from eda_platform.application.services.report_export_service import (
    ReportExportFormat,
    ReportExportService,
)
from eda_platform.application.services.report_generation_service import (
    ReportGenerationService,
)
from eda_platform.application.services.report_service import ReportService

_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    400: {"model": ApiErrorEnvelope},
    404: {"model": ApiErrorEnvelope},
    409: {"model": ApiErrorEnvelope},
    413: {"model": ApiErrorEnvelope},
    422: {"model": ApiErrorEnvelope},
    500: {"model": ApiErrorEnvelope},
}

router = APIRouter(tags=["reports"], responses=_ERROR_RESPONSES)


def _service(request: Request) -> ReportService:
    return request.app.state.report_service


def _export_service(request: Request) -> ReportExportService:
    return request.app.state.report_export_service


@router.get("/sessions/{session_id}/report", response_model=ReportView)
def get_report(session_id: str, request: Request) -> ReportView:
    return _service(request).get_report(session_id)


@router.get(
    "/sessions/{session_id}/report/download",
    responses={**_ERROR_RESPONSES, 503: {"model": ApiErrorEnvelope}},
    response_class=StreamingResponse,
)
def download_report(
    session_id: str,
    request: Request,
    format: Annotated[
        ReportExportFormat, Query(description="html | pdf | md")
    ] = "html",
) -> StreamingResponse:
    """Stream one report export. ``md`` renders the project's decision report."""
    download = _export_service(request).download(session_id, format)
    return StreamingResponse(
        io.BytesIO(download.content),
        media_type=download.media_type,
        headers={
            # export_filename() strips CR/LF and quotes, so this cannot inject.
            "Content-Disposition": f'attachment; filename="{download.filename}"',
            "Content-Length": str(len(download.content)),
        },
    )


class ReportGenerateRequest(BaseModel):
    llm: Literal["env", "offline"] = "env"


@router.post(
    "/sessions/{session_id}/report/generate",
    status_code=201,
    response_model=ReportGenerationStarted,
)
def generate_report(
    session_id: str,
    request: Request,
    body: ReportGenerateRequest | None = None,
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    x_eda_session: str | None = Header(None, alias=SESSION_HEADER),
) -> ReportGenerationStarted:
    """Regenerating replaces the run's report — the reader always serves the
    newest MarkdownReport — so the caller must confirm before invoking this."""
    service: ReportGenerationService = request.app.state.report_generation_service
    effective = request.app.state.settings_service.resolve(session_id_from_header(x_eda_session))
    return service.generate(
        session_id,
        llm=body.llm if body is not None else "env",
        payload_policy=effective.payload_policy,
        llm_env=effective.env_overlay,
        idempotency_key=idempotency_key,
    )
