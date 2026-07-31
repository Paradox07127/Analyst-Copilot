"""Chat endpoints (§7.5): transcript paging, accept-then-stream turns, and the
two-step plan approval.

`POST .../chat/messages` answers 202 with a message_id — the turn itself runs on
a worker thread and reports over `GET .../chat/stream`. The SSE generator is a
sync generator for the same reason as jobs.py: Starlette iterates it in the
threadpool, so polling never blocks the event loop.

`GET .../chat/pending-plans` is the read-only recovery half of approval: the
durable pending row retains the same token across reloads and API restarts.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Literal

from fastapi import APIRouter, Header, Query, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from eda_platform.api.errors import ApiErrorEnvelope
from eda_platform.api.routers.settings import SESSION_HEADER, session_id_from_header
from eda_platform.application.dto import (
    ChatMessageAccepted,
    ChatMessagePage,
    ChatPendingPlanList,
    ChatPlanRejected,
    ChatStreamEvent,
)
from eda_platform.application.services.chat_service import ChatService

POLL_INTERVAL_SECONDS = 0.2
HEARTBEAT_INTERVAL_SECONDS = 15.0

_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    400: {"model": ApiErrorEnvelope},
    404: {"model": ApiErrorEnvelope},
    409: {"model": ApiErrorEnvelope},
    410: {"model": ApiErrorEnvelope},
    422: {"model": ApiErrorEnvelope},
    500: {"model": ApiErrorEnvelope},
}

router = APIRouter(tags=["chat"], responses=_ERROR_RESPONSES)


class ChatSendRequest(BaseModel):
    text: str = Field(min_length=1)
    llm: Literal["env", "offline"] = "env"


class ChatPlanDecisionRequest(BaseModel):
    action_hash: str = Field(min_length=1)
    approval_token: str = Field(min_length=1)


def _service(request: Request) -> ChatService:
    return request.app.state.chat_service


@router.get("/sessions/{session_id}/chat/messages", response_model=ChatMessagePage)
def list_chat_messages(
    session_id: str,
    request: Request,
    limit: int = Query(50, ge=1, le=200),
    cursor: str | None = Query(None),
) -> ChatMessagePage:
    return _service(request).list_messages(session_id, limit=limit, cursor=cursor)


@router.post(
    "/sessions/{session_id}/chat/messages", status_code=202, response_model=ChatMessageAccepted
)
def send_chat_message(
    session_id: str,
    body: ChatSendRequest,
    request: Request,
    x_eda_session: str | None = Header(None, alias=SESSION_HEADER),
) -> ChatMessageAccepted:
    effective = request.app.state.settings_service.resolve(
        session_id_from_header(x_eda_session)
    )
    return _service(request).send_message(
        session_id,
        text=body.text,
        llm=body.llm,
        effective_settings=effective,
    )


@router.get("/sessions/{session_id}/chat/pending-plans", response_model=ChatPendingPlanList)
def list_chat_pending_plans(session_id: str, request: Request) -> ChatPendingPlanList:
    """Plans still awaiting approval, using their persisted approval token."""
    return _service(request).list_pending_plans(session_id)


@router.post(
    "/sessions/{session_id}/chat/plans/{plan_id}/approve",
    status_code=202,
    response_model=ChatMessageAccepted,
)
def approve_chat_plan(
    session_id: str, plan_id: str, body: ChatPlanDecisionRequest, request: Request
) -> ChatMessageAccepted:
    return _service(request).approve_plan(
        session_id,
        plan_id,
        action_hash=body.action_hash,
        approval_token=body.approval_token,
    )


@router.post(
    "/sessions/{session_id}/chat/plans/{plan_id}/reject", response_model=ChatPlanRejected
)
def reject_chat_plan(
    session_id: str, plan_id: str, body: ChatPlanDecisionRequest, request: Request
) -> ChatPlanRejected:
    return _service(request).reject_plan(
        session_id,
        plan_id,
        action_hash=body.action_hash,
        approval_token=body.approval_token,
    )


@router.get("/sessions/{session_id}/chat/stream")
def stream_chat_events(
    session_id: str,
    request: Request,
    message_id: str = Query(...),
    last_event_id_header: str | None = Header(None, alias="Last-Event-ID"),
    last_event_id: str | None = Query(None),
) -> Response:
    service = _service(request)
    service.require_session(session_id, message_id)  # 404 before the stream starts
    after_seq = _parse_last_event_id(last_event_id_header, last_event_id)
    # A reconnect on a finished turn with nothing left to replay answers 204:
    # simply closing the stream makes EventSource retry forever.
    if last_event_id_header is not None or last_event_id is not None:
        page = service.events_after(session_id, message_id, after_seq)
        if not page.events and page.done:
            return Response(status_code=204)
    return StreamingResponse(
        _sse_frames(service, session_id, message_id, after_seq),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _parse_last_event_id(header_value: str | None, query_value: str | None) -> int:
    for raw in (header_value, query_value):
        if raw is None:
            continue
        try:
            return max(0, int(raw))
        except ValueError:
            continue
    return 0


def _sse_frames(
    service: ChatService, session_id: str, message_id: str, after_seq: int
) -> Iterator[str]:
    cursor = after_seq
    last_activity = time.monotonic()
    while True:
        page = service.events_after(session_id, message_id, cursor)
        for event in page.events:
            cursor = event.seq
            yield _frame(event)
            last_activity = time.monotonic()
        if page.done and not page.events:
            return
        if not page.events:
            if time.monotonic() - last_activity >= HEARTBEAT_INTERVAL_SECONDS:
                yield ": heartbeat\n\n"
                last_activity = time.monotonic()
            time.sleep(POLL_INTERVAL_SECONDS)


def _frame(event: ChatStreamEvent) -> str:
    data = json.dumps(event.model_dump(mode="json"), ensure_ascii=False)
    return f"id: {event.seq}\nevent: {event.type}\ndata: {data}\n\n"
