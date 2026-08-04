"""E4b exploration control plane and journal-backed SSE projection."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator

from fastapi import APIRouter, Depends, Header, Query, Request, Response
from fastapi.responses import StreamingResponse

from eda_platform.api.errors import ApiErrorEnvelope
from eda_platform.api.routers.settings import SESSION_HEADER, session_id_from_header
from eda_platform.application.services.exploration_service import (
    ExplorationService,
    ExplorationValidationError,
)
from eda_platform.schemas.exploration_api import (
    ExplorationBudgetExtended,
    ExplorationControlRequest,
    ExplorationEventView,
    ExplorationExtendBudgetRequest,
    ExplorationPrepared,
    ExplorationPrepareRequest,
    ExplorationStarted,
    ExplorationStartRequest,
    ExplorationView,
)

POLL_INTERVAL_SECONDS = 0.5
HEARTBEAT_INTERVAL_SECONDS = 15.0

_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    404: {"model": ApiErrorEnvelope},
    409: {"model": ApiErrorEnvelope},
    410: {"model": ApiErrorEnvelope},
    422: {"model": ApiErrorEnvelope},
    503: {"model": ApiErrorEnvelope},
}


def _service(request: Request) -> ExplorationService:
    return request.app.state.exploration_service


def _require_release(request: Request) -> None:
    # Keep the gate on the router as well as the service. A future endpoint
    # cannot accidentally expose exploration state without explicitly passing
    # through the installed production certificate.
    _service(request).require_release_certificate()


router = APIRouter(
    tags=["explorations"],
    responses=_ERROR_RESPONSES,
    dependencies=[Depends(_require_release)],
)


def _effective(request: Request, session: str | None):
    return request.app.state.settings_service.resolve(session_id_from_header(session))


@router.post(
    "/sessions/{session_id}/explorations/prepare",
    response_model=ExplorationPrepared,
)
def prepare_exploration(
    session_id: str,
    body: ExplorationPrepareRequest,
    request: Request,
    x_eda_session: str | None = Header(None, alias=SESSION_HEADER),
) -> ExplorationPrepared:
    effective = _effective(request, x_eda_session)
    return _service(request).prepare(
        session_id,
        mode=body.mode,
        goal=body.goal,
        dataset_ids=body.dataset_ids,
        thinking_level=body.thinking_level,
        provider=effective.llm.provider.value,
    )


@router.post(
    "/sessions/{session_id}/explorations",
    status_code=201,
    response_model=ExplorationStarted,
)
def start_exploration(
    session_id: str,
    body: ExplorationStartRequest,
    request: Request,
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    x_eda_session: str | None = Header(None, alias=SESSION_HEADER),
) -> ExplorationStarted:
    effective = _effective(request, x_eda_session)
    return _service(request).start(
        session_id,
        action_hash=body.action_hash,
        approval_token=body.approval_token,
        provider=effective.llm.provider.value,
        payload_policy=effective.payload_policy,
        llm_env=effective.env_overlay,
        idempotency_key=idempotency_key,
    )


@router.get(
    "/sessions/{session_id}/explorations/{exploration_id}",
    response_model=ExplorationView,
)
def get_exploration(
    session_id: str, exploration_id: str, request: Request
) -> ExplorationView:
    return _service(request).get(session_id, exploration_id)


@router.post(
    "/sessions/{session_id}/explorations/{exploration_id}/pause",
    response_model=ExplorationView,
)
def pause_exploration(
    session_id: str,
    exploration_id: str,
    body: ExplorationControlRequest,
    request: Request,
) -> ExplorationView:
    del body
    return _service(request).pause(session_id, exploration_id)


@router.post(
    "/sessions/{session_id}/explorations/{exploration_id}/resume",
    status_code=201,
    response_model=ExplorationStarted,
)
def resume_exploration(
    session_id: str,
    exploration_id: str,
    body: ExplorationControlRequest,
    request: Request,
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    x_eda_session: str | None = Header(None, alias=SESSION_HEADER),
) -> ExplorationStarted:
    del body
    effective = _effective(request, x_eda_session)
    return _service(request).resume(
        session_id,
        exploration_id,
        provider=effective.llm.provider.value,
        payload_policy=effective.payload_policy,
        llm_env=effective.env_overlay,
        idempotency_key=idempotency_key,
    )


@router.post(
    "/sessions/{session_id}/explorations/{exploration_id}/cancel",
    response_model=ExplorationView,
)
def cancel_exploration(
    session_id: str,
    exploration_id: str,
    body: ExplorationControlRequest,
    request: Request,
) -> ExplorationView:
    del body
    return _service(request).cancel(session_id, exploration_id)


@router.post(
    "/sessions/{session_id}/explorations/{exploration_id}/extend-budget",
    response_model=ExplorationBudgetExtended,
)
def extend_exploration_budget(
    session_id: str,
    exploration_id: str,
    body: ExplorationExtendBudgetRequest,
    request: Request,
    idempotency_key: str = Header(..., alias="Idempotency-Key", min_length=1),
) -> ExplorationBudgetExtended:
    return _service(request).extend_budget(
        session_id,
        exploration_id,
        increase=body.increase,
        reason=body.reason,
        idempotency_key=idempotency_key,
    )


@router.get(
    "/sessions/{session_id}/explorations/{exploration_id}/report",
    response_class=Response,
    responses={200: {"content": {"text/markdown": {}}}},
)
def get_exploration_report(
    session_id: str, exploration_id: str, request: Request
) -> Response:
    """Serve the run's own markdown; the report is not an artifact-store row."""
    markdown = _service(request).read_report(session_id, exploration_id)
    return Response(
        content=markdown,
        media_type="text/markdown; charset=utf-8",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/sessions/{session_id}/explorations/{exploration_id}/events")
def stream_exploration_events(
    session_id: str,
    exploration_id: str,
    request: Request,
    last_event_id_header: str | None = Header(None, alias="Last-Event-ID"),
    last_event_id: str | None = Query(None),
) -> Response:
    service = _service(request)
    service.get(session_id, exploration_id)  # fail before streaming begins
    after_seq = _parse_last_event_id(
        exploration_id, last_event_id_header, last_event_id
    )
    if last_event_id_header is not None or last_event_id is not None:
        page = service.events_after(session_id, exploration_id, after_seq)
        if not page.events and page.exhausted and page.status == "stopped":
            return Response(status_code=204)
    return StreamingResponse(
        _sse_frames(service, session_id, exploration_id, after_seq),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _parse_last_event_id(
    exploration_id: str,
    header_value: str | None,
    query_value: str | None,
) -> int:
    raw = header_value if header_value is not None else query_value
    if raw is None:
        return -1
    prefix, separator, sequence = raw.strip().partition(":")
    if separator != ":" or prefix != exploration_id:
        raise ExplorationValidationError(
            "Last-Event-ID must identify this exploration as '<exploration_id>:<seq>'"
        )
    try:
        value = int(sequence)
    except ValueError as exc:
        raise ExplorationValidationError(
            "Last-Event-ID sequence must be a non-negative integer"
        ) from exc
    if value < 0:
        raise ExplorationValidationError(
            "Last-Event-ID sequence must be a non-negative integer"
        )
    return value


def _sse_frames(
    service: ExplorationService,
    session_id: str,
    exploration_id: str,
    after_seq: int,
) -> Iterator[str]:
    cursor = after_seq
    last_activity = time.monotonic()
    while True:
        page = service.events_after(session_id, exploration_id, cursor)
        cursor = page.cursor
        for event in page.events:
            yield _frame(event)
            last_activity = time.monotonic()
        if not page.exhausted:
            continue
        if page.status == "stopped":
            return
        if time.monotonic() - last_activity >= HEARTBEAT_INTERVAL_SECONDS:
            yield ": heartbeat\n\n"
            last_activity = time.monotonic()
        time.sleep(POLL_INTERVAL_SECONDS)


def _frame(event: ExplorationEventView) -> str:
    data = json.dumps(event.model_dump(mode="json"), ensure_ascii=False)
    return f"id: {event.event_id}\nevent: {event.type}\ndata: {data}\n\n"

