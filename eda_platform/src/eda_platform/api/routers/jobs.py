"""Job endpoints (§7.3): create/status/cancel plus the SSE progress stream.

The SSE endpoint is a sync generator on purpose: it polls SQLite, and Starlette
iterates sync generators in its threadpool, so the event loop is never blocked.
`Last-Event-ID` (header or query) replays from the trace_events autoincrement id.
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
from eda_platform.application.dto import JobCreated, JobEvent, JobStatus, PrecleaningOptions
from eda_platform.application.ports import TERMINAL_JOB_STATUSES
from eda_platform.application.services.job_service import (
    TERMINAL_EVENT_TYPES,
    JobService,
)

POLL_INTERVAL_SECONDS = 0.5
HEARTBEAT_INTERVAL_SECONDS = 15.0

_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    400: {"model": ApiErrorEnvelope},
    404: {"model": ApiErrorEnvelope},
    409: {"model": ApiErrorEnvelope},
    422: {"model": ApiErrorEnvelope},
    500: {"model": ApiErrorEnvelope},
}

router = APIRouter(tags=["jobs"], responses=_ERROR_RESPONSES)


class JobCreateRequest(BaseModel):
    kind: Literal["auto_eda"]
    project_id: str = "default"
    """Used only when the run does not exist yet; an existing run wins."""
    datasets: list[str] = Field(min_length=1)
    """Dataset references: an uploaded dataset_id or a workspace-relative CSV path."""
    business_context: str = ""
    ml_target_column: str | None = None
    ml_time_column: str | None = None
    generate_report: bool = True
    llm: Literal["env", "offline"] = "env"
    precleaning: PrecleaningOptions | None = None
    """Optional pre-ingest clean; the uploaded files are never rewritten."""


def _service(request: Request) -> JobService:
    return request.app.state.job_service


@router.post("/sessions/{session_id}/jobs", status_code=201, response_model=JobCreated)
def create_job(
    session_id: str,
    body: JobCreateRequest,
    request: Request,
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    x_eda_session: str | None = Header(None, alias=SESSION_HEADER),
) -> JobCreated:
    # llm="env" means "whatever Settings resolves for this session" (env values
    # until the session overrides them); llm="offline" bypasses providers entirely.
    effective = request.app.state.settings_service.resolve(session_id_from_header(x_eda_session))
    status = _service(request).create_job(
        session_id,
        kind=body.kind,
        project_id=body.project_id,
        datasets=body.datasets,
        business_context=body.business_context,
        ml_target_column=body.ml_target_column,
        ml_time_column=body.ml_time_column,
        generate_report=body.generate_report,
        llm=body.llm,
        payload_policy=effective.payload_policy,
        llm_env=effective.env_overlay,
        precleaning=(
            body.precleaning.model_dump()
            if body.precleaning is not None and body.precleaning.enabled
            else None
        ),
        idempotency_key=idempotency_key,
    )
    return JobCreated(
        job_id=status.job_id,
        session_id=status.session_id,
        status=status.status,
        events_url=status.events_url,
    )


@router.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str, request: Request) -> JobStatus:
    return _service(request).get_job(job_id)


@router.post("/jobs/{job_id}/cancel", response_model=JobStatus)
def cancel_job(job_id: str, request: Request) -> JobStatus:
    return _service(request).cancel_job(job_id)


@router.get("/jobs/{job_id}/events")
def stream_job_events(
    job_id: str,
    request: Request,
    last_event_id_header: str | None = Header(None, alias="Last-Event-ID"),
    last_event_id: str | None = Query(None),
) -> Response:
    service = _service(request)
    service.get_job(job_id)  # 404 before the stream starts, not inside it
    after_id = _parse_last_event_id(last_event_id_header, last_event_id)
    # A reconnect (Last-Event-ID present) on a finished job with nothing left
    # to replay must answer 204: closing the stream alone makes EventSource
    # clients retry forever (review codex-D #5).
    if last_event_id_header is not None or last_event_id is not None:
        page = service.events_after(job_id, after_id)
        if not page.events and page.exhausted and page.status in TERMINAL_JOB_STATUSES:
            return Response(status_code=204)
    return StreamingResponse(
        _sse_frames(service, job_id, after_id),
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


def _sse_frames(service: JobService, job_id: str, after_id: int) -> Iterator[str]:
    cursor = after_id
    last_activity = time.monotonic()
    while True:
        page = service.events_after(job_id, cursor)
        cursor = page.cursor
        for event in page.events:
            yield _frame(event)
            last_activity = time.monotonic()
            if event.type in TERMINAL_EVENT_TYPES and event.name == job_id:
                return
        if not page.exhausted:
            # Full page: more rows are already waiting. Never fall through to
            # the synthetic-terminal branch on a truncated read (review F2).
            continue
        if page.status in TERMINAL_JOB_STATUSES:
            # Terminal row but no terminal trace event past the cursor (e.g. a
            # reconnect after the stream already ended): close with a synthetic
            # terminal frame so EventSource clients stop retrying.
            yield _frame(
                JobEvent(
                    event_id=cursor,
                    job_id=job_id,
                    session_id="",
                    type=f"job.{page.status}",
                    name=job_id,
                    summary={"job_id": job_id, "status": page.status, "synthetic": True},
                )
            )
            return
        if time.monotonic() - last_activity >= HEARTBEAT_INTERVAL_SECONDS:
            yield ": heartbeat\n\n"
            last_activity = time.monotonic()
        time.sleep(POLL_INTERVAL_SECONDS)


def _frame(event: JobEvent) -> str:
    data = json.dumps(event.model_dump(mode="json"), ensure_ascii=False)
    return f"id: {event.event_id}\nevent: {event.type}\ndata: {data}\n\n"
