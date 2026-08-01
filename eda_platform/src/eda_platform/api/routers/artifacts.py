"""Read-only artifact endpoints (§7.4): index-backed summaries, on-demand
detail. Sync def: served from the worker threadpool."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request, Response

from eda_platform.api.errors import ApiErrorEnvelope
from eda_platform.application.dto import AgentHandoffDetail, ArtifactDetail, ArtifactSummary, Page
from eda_platform.application.services.artifact_service import (
    DEFAULT_ARTIFACT_LIMIT,
    MAX_ARTIFACT_LIMIT,
    ArtifactService,
)

_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    400: {"model": ApiErrorEnvelope},
    404: {"model": ApiErrorEnvelope},
    422: {"model": ApiErrorEnvelope},
    500: {"model": ApiErrorEnvelope},
    503: {"model": ApiErrorEnvelope},
}

router = APIRouter(tags=["artifacts"], responses=_ERROR_RESPONSES)


def _service(request: Request) -> ArtifactService:
    return request.app.state.artifact_service


@router.get("/sessions/{session_id}/artifacts", response_model=Page[ArtifactSummary])
def list_artifacts(
    session_id: str,
    request: Request,
    type: str | None = Query(None),
    limit: int = Query(DEFAULT_ARTIFACT_LIMIT, ge=1, le=MAX_ARTIFACT_LIMIT),
    cursor: str | None = Query(None),
) -> Page[ArtifactSummary]:
    return _service(request).list_artifacts(
        session_id, artifact_type=type, limit=limit, cursor=cursor
    )


@router.get(
    "/sessions/{session_id}/agent-handoff",
    response_model=AgentHandoffDetail,
    responses={
        409: {
            "model": ApiErrorEnvelope,
            "headers": {
                "Retry-After": {
                    "description": (
                        "Seconds before retrying while the session is still running; "
                        "absent for terminal failed or cancelled sessions."
                    ),
                    "schema": {"type": "integer", "minimum": 1},
                }
            },
        },
        413: {"model": ApiErrorEnvelope},
        503: {"model": ApiErrorEnvelope},
    },
)
def get_agent_handoff(
    session_id: str, request: Request, response: Response
) -> AgentHandoffDetail:
    # The final handoff keeps a stable per-session artifact id while report
    # regeneration refreshes its content. Prevent intermediaries from treating
    # that URL as an immutable content-addressed object.
    response.headers["Cache-Control"] = "no-store"
    return _service(request).get_agent_handoff(session_id)


@router.get(
    "/sessions/{session_id}/artifacts/{artifact_id}",
    response_model=ArtifactDetail,
    responses={413: {"model": ApiErrorEnvelope}},
)
def get_artifact(session_id: str, artifact_id: str, request: Request) -> ArtifactDetail:
    return _service(request).get_artifact(session_id, artifact_id)
