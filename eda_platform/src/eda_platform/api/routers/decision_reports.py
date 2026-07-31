"""Decision Story and decision report endpoints (§10.3).

Reading a report is a plain GET: a project with no decision report answers 200
with status="none", not 404. The two write paths mirror the page's two buttons
and both return 201 with a job — neither runs in the request thread.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Header, Request
from pydantic import BaseModel, Field, StringConstraints

from eda_platform.api.errors import ApiErrorEnvelope
from eda_platform.application.dto import (
    DecisionReportGenerationStarted,
    DecisionReportView,
    DecisionStoryDraftStarted,
    DecisionStoryView,
)
from eda_platform.application.services.decision_report_service import DecisionReportService

_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    404: {"model": ApiErrorEnvelope},
    409: {"model": ApiErrorEnvelope},
    422: {"model": ApiErrorEnvelope},
    500: {"model": ApiErrorEnvelope},
    503: {"model": ApiErrorEnvelope},
}

router = APIRouter(tags=["decision-report"], responses=_ERROR_RESPONSES)


class DecisionStoryDraftRequest(BaseModel):
    """The selection is explicit: synthesis never picks findings for the user."""

    finding_artifact_ids: list[Annotated[str, StringConstraints(min_length=1)]] = Field(
        min_length=1, max_length=100
    )
    finding_session_ids: dict[str, str] = Field(default_factory=dict)
    business_context: str = Field(default="", max_length=5000)
    """Unverified user framing; labeled as such and never a source for claims."""


class DecisionReportGenerateRequest(BaseModel):
    brief_artifact_id: str = Field(min_length=1)
    brief_session_id: str | None = Field(default=None, min_length=1)


def _service(request: Request) -> DecisionReportService:
    return request.app.state.decision_report_service


@router.get("/sessions/{session_id}/decision-story", response_model=DecisionStoryView)
def get_decision_story(session_id: str, request: Request) -> DecisionStoryView:
    return _service(request).get_decision_story(session_id)


@router.post(
    "/sessions/{session_id}/decision-story/drafts",
    status_code=201,
    response_model=DecisionStoryDraftStarted,
    description=(
        "Draft a decision story from an explicit selection of report-eligible "
        "validated findings. Queues a background job on a derived sbsess_* "
        "session; the brief artifact lands on the synthesis session the driver "
        "mints, so it is read back through "
        "GET /sessions/{session_id}/decision-story.\n\n"
        "A selection containing an unknown or non-report-eligible finding is "
        "422 before any job is queued. Send Idempotency-Key to make a retry "
        "replay the same job instead of starting a second one."
    ),
)
def create_decision_story_draft(
    session_id: str,
    body: DecisionStoryDraftRequest,
    request: Request,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> DecisionStoryDraftStarted:
    return _service(request).create_draft(
        session_id,
        finding_artifact_ids=body.finding_artifact_ids,
        finding_session_ids=body.finding_session_ids,
        business_context=body.business_context,
        idempotency_key=idempotency_key,
    )


@router.post(
    "/sessions/{session_id}/decision-report/generate",
    status_code=201,
    response_model=DecisionReportGenerationStarted,
    description=(
        "Generate a decision report from one persisted draft. Queues a "
        "background job on a derived drsess_* session; the report artifact "
        "inherits the brief's session and is read back through "
        "GET /sessions/{session_id}/decision-report.\n\n"
        "Generation is deterministic and never spends an LLM."
    ),
)
def generate_decision_report(
    session_id: str,
    body: DecisionReportGenerateRequest,
    request: Request,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> DecisionReportGenerationStarted:
    return _service(request).generate_report(
        session_id,
        brief_artifact_id=body.brief_artifact_id,
        brief_session_id=body.brief_session_id,
        idempotency_key=idempotency_key,
    )


@router.get("/sessions/{session_id}/decision-report", response_model=DecisionReportView)
def get_decision_report(session_id: str, request: Request) -> DecisionReportView:
    return _service(request).get_decision_report(session_id)
