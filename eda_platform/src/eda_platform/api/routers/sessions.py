"""Project/Run endpoints (§7.1). Sync def: the kernel is synchronous, so FastAPI
runs these in its worker threadpool instead of blocking the loop."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request, Response
from pydantic import BaseModel

from eda_platform.api.errors import ApiErrorEnvelope
from eda_platform.application.dto import (
    Page,
    ProjectDeleted,
    ProjectSummary,
    SessionDeleted,
    SessionDetail,
    SessionSummary,
)
from eda_platform.application.services.session_service import (
    DEFAULT_PAGE_LIMIT,
    MAX_PAGE_LIMIT,
    MAX_SEARCH_LENGTH,
    SessionService,
)

# Runtime errors always use the ApiErrorEnvelope shape (api/errors.py); declare
# it so generated clients get typed errors instead of FastAPI's default bodies.
_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    400: {"model": ApiErrorEnvelope},
    404: {"model": ApiErrorEnvelope},
    409: {"model": ApiErrorEnvelope},
    422: {"model": ApiErrorEnvelope},
    500: {"model": ApiErrorEnvelope},
}

router = APIRouter(tags=["sessions"], responses=_ERROR_RESPONSES)


class ProjectCreateRequest(BaseModel):
    project_id: str
    name: str = ""
    """Display name; falls back to project_id when blank."""


class DisplayNameRequest(BaseModel):
    name: str


class ProjectOrderRequest(BaseModel):
    project_ids: list[str]


def _service(request: Request) -> SessionService:
    return request.app.state.session_service


@router.get("/projects", response_model=list[ProjectSummary])
def list_projects(request: Request) -> list[ProjectSummary]:
    return _service(request).list_projects()


@router.post(
    "/projects",
    status_code=201,
    response_model=ProjectSummary,
    responses={200: {"model": ProjectSummary, "description": "Project already existed."}},
)
def create_project(
    body: ProjectCreateRequest, request: Request, response: Response
) -> ProjectSummary:
    summary, created = _service(request).create_project(body.project_id, name=body.name)
    if not created:
        response.status_code = 200
    return summary


@router.put("/projects/order", response_model=list[ProjectSummary])
def reorder_projects(body: ProjectOrderRequest, request: Request) -> list[ProjectSummary]:
    """Persist the order of the user-visible projects in the sidebar."""
    return _service(request).reorder_projects(body.project_ids)


@router.patch("/projects/{project_id}", response_model=ProjectSummary)
def rename_project(
    project_id: str, body: DisplayNameRequest, request: Request
) -> ProjectSummary:
    """Rename the project label without changing its stable id."""
    return _service(request).rename_project(project_id, body.name)


@router.patch("/sessions/{session_id}", response_model=SessionSummary)
def rename_session(
    session_id: str, body: DisplayNameRequest, request: Request
) -> SessionSummary:
    """Rename the session label without changing its stable id."""
    return _service(request).rename_session(session_id, body.name)


# Deliberately not DELETE /projects/{project_id}: HTTP clients collapse ".."
# before the request leaves, so a support-doc delete built from an unvalidated
# id ("/projects/demo/support-docs/..") normalizes onto that URL and the server
# cannot tell it from a deliberate project delete. The trailing segment puts the
# irreversible action somewhere path normalization cannot land. Guarded by
# test_api_support_docs.py::test_delete_with_a_crafted_id_is_404_and_deletes_nothing.
@router.delete("/projects/{project_id}/self", response_model=ProjectDeleted)
def delete_project(project_id: str, request: Request) -> ProjectDeleted:
    """Irreversibly remove a project, including all runs and uploads."""
    return _service(request).delete_project(project_id)


@router.get("/projects/{project_id}/sessions", response_model=Page[SessionSummary])
def list_sessions(
    project_id: str,
    request: Request,
    limit: int = Query(DEFAULT_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT),
    cursor: str | None = Query(None),
    include_derived: bool = Query(
        False,
        description="Include runs derived from another run (question batches, skill replays).",
    ),
    q: str | None = Query(
        None,
        max_length=MAX_SEARCH_LENGTH,
        description="Filter by title, dataset name or run id (contains match).",
    ),
) -> Page[SessionSummary]:
    return _service(request).list_sessions(
        project_id, limit=limit, cursor=cursor, include_derived=include_derived, q=q
    )


@router.get("/sessions/{session_id}", response_model=SessionDetail)
def get_run(session_id: str, request: Request) -> SessionDetail:
    return _service(request).get_session_detail(session_id)


@router.delete(
    "/sessions/{session_id}",
    response_model=SessionDeleted,
    responses={503: {"model": ApiErrorEnvelope}},
)
def delete_session(session_id: str, request: Request) -> SessionDeleted:
    """Irreversible: removes the session directory, its artifacts and its chat
    transcript. Refuses with 409 while a job is still running against it."""
    return _service(request).delete_session(session_id)
