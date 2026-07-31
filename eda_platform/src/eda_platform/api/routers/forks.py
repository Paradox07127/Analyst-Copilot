"""What-if fork endpoint (§10.3 Compare): re-run a run with exactly one
decision varied. The response carries the `fksess_*` lifecycle run; the forked
analysis's own run id arrives on the job's SSE stream as a `session.forked` event."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Header, Request
from pydantic import BaseModel, Field

from eda_platform.api.errors import ApiErrorEnvelope
from eda_platform.api.routers.settings import SESSION_HEADER, session_id_from_header
from eda_platform.application.dto import SessionForkStarted
from eda_platform.application.services.session_fork_service import SessionForkService

_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    404: {"model": ApiErrorEnvelope},
    409: {"model": ApiErrorEnvelope},
    422: {"model": ApiErrorEnvelope},
    500: {"model": ApiErrorEnvelope},
}

router = APIRouter(tags=["forks"], responses=_ERROR_RESPONSES)


class SessionForkRequest(BaseModel):
    """Exactly one decision varies. `ml_target` re-runs the same inputs with a
    different (or no) baseline target; `dataset` re-runs on the chosen tables."""

    decision: Literal["ml_target", "dataset"]
    ml_target_column: str | None = None
    datasets: list[str] = Field(default_factory=list)
    """Dataset references: an uploaded dataset_id or a workspace-relative CSV path."""
    llm: Literal["env", "offline"] = "env"


@router.post("/sessions/{session_id}/fork", status_code=201, response_model=SessionForkStarted)
def fork_session(
    session_id: str,
    body: SessionForkRequest,
    request: Request,
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    x_eda_session: str | None = Header(None, alias=SESSION_HEADER),
) -> SessionForkStarted:
    service: SessionForkService = request.app.state.session_fork_service
    effective = request.app.state.settings_service.resolve(session_id_from_header(x_eda_session))
    return service.fork(
        session_id,
        decision_kind=body.decision,
        ml_target_column=body.ml_target_column,
        datasets=body.datasets,
        llm=body.llm,
        payload_policy=effective.payload_policy,
        llm_env=effective.env_overlay,
        idempotency_key=idempotency_key,
    )
