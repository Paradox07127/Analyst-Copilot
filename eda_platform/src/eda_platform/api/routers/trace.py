"""Read-only trace & cost endpoints (§10.1 Trace) plus the developer inspector.
Sync def: served from the worker threadpool."""

from __future__ import annotations

from fastapi import APIRouter, Header, Query, Request, Response
from fastapi.responses import StreamingResponse

from eda_platform.api.errors import ApiErrorEnvelope, error_response
from eda_platform.application.dto import (
    ClientFailureRecorded,
    ClientFailureRequest,
    LlmDebugRecord,
    Page,
    SessionDebugView,
    SessionMetricsView,
    TraceEventPage,
    WorkspaceUsageView,
)
from eda_platform.application.services.trace_service import (
    DEFAULT_LLM_DEBUG_LIMIT,
    DEFAULT_TRACE_LIMIT,
    DEFAULT_USAGE_WINDOW_DAYS,
    MAX_LLM_DEBUG_LIMIT,
    MAX_TRACE_LIMIT,
    MAX_USAGE_WINDOW_DAYS,
    ClientFailureTooLargeError,
    DebugLogNotFoundError,
    TraceService,
)

_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    400: {"model": ApiErrorEnvelope},
    404: {"model": ApiErrorEnvelope},
    413: {"model": ApiErrorEnvelope},
    422: {"model": ApiErrorEnvelope},
    429: {"model": ApiErrorEnvelope},
    500: {"model": ApiErrorEnvelope},
}

router = APIRouter(tags=["trace"], responses=_ERROR_RESPONSES)
MAX_CLIENT_FAILURE_BODY_BYTES = 512


def _service(request: Request) -> TraceService:
    return request.app.state.trace_service


@router.get("/usage", response_model=WorkspaceUsageView)
def get_workspace_usage(
    request: Request,
    days: int = Query(
        DEFAULT_USAGE_WINDOW_DAYS, ge=1, le=MAX_USAGE_WINDOW_DAYS
    ),
) -> WorkspaceUsageView:
    return _service(request).workspace_usage(days=days)


@router.get("/sessions/{session_id}/metrics", response_model=SessionMetricsView)
def get_metrics(session_id: str, request: Request) -> SessionMetricsView:
    return _service(request).get_metrics(session_id)


@router.get("/sessions/{session_id}/trace", response_model=TraceEventPage)
def list_trace_events(
    session_id: str,
    request: Request,
    limit: int = Query(DEFAULT_TRACE_LIMIT, ge=1, le=MAX_TRACE_LIMIT),
    cursor: str | None = Query(None),
    type: str | None = Query(None, alias="type"),
) -> TraceEventPage:
    return _service(request).list_events(
        session_id, limit=limit, cursor=cursor, event_type=type
    )


@router.post(
    "/sessions/{session_id}/client-failures",
    status_code=201,
    response_model=ClientFailureRecorded,
)
def record_client_failure(
    session_id: str,
    body: ClientFailureRequest,
    request: Request,
    content_length: int | None = Header(None, alias="Content-Length"),
) -> ClientFailureRecorded:
    if content_length is not None and content_length > MAX_CLIENT_FAILURE_BODY_BYTES:
        raise ClientFailureTooLargeError(
            f"Client failure payload exceeds {MAX_CLIENT_FAILURE_BODY_BYTES} bytes."
        )
    return _service(request).record_client_failure(session_id, body)


@router.get("/sessions/{session_id}/debug", response_model=SessionDebugView)
def get_session_debug(
    session_id: str,
    request: Request,
    limit: int = Query(100, ge=1, le=250),
    cursor: str | None = Query(None),
) -> SessionDebugView:
    return _service(request).get_debug(session_id, limit=limit, cursor=cursor)


@router.get("/sessions/{session_id}/debug/log", response_class=StreamingResponse)
def download_debug_log(session_id: str, request: Request) -> Response:
    """Stream the run's debug.jsonl. Handled here rather than by an exception
    handler so the 404 keeps its own code without a global registration."""
    try:
        download = _service(request).open_debug_log(session_id)
    except DebugLogNotFoundError as exc:
        return error_response(404, "debug_log_not_found", str(exc))
    return StreamingResponse(
        download.chunks(),
        media_type=download.media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{download.filename}"',
        },
    )


@router.get("/sessions/{session_id}/debug/llm-calls", response_model=Page[LlmDebugRecord])
def list_llm_debug_calls(
    session_id: str,
    request: Request,
    limit: int = Query(DEFAULT_LLM_DEBUG_LIMIT, ge=1, le=MAX_LLM_DEBUG_LIMIT),
    cursor: str | None = Query(None),
) -> Page[LlmDebugRecord]:
    return _service(request).list_llm_calls(session_id, limit=limit, cursor=cursor)
