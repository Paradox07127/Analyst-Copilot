"""Support-document endpoints: the optional reference files semantic bootstrap
reads as priors. Multipart is streamed off ``UploadFile.file`` under a cap."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request, Response, UploadFile

from eda_platform.api.errors import ApiErrorEnvelope
from eda_platform.application.dto import SupportDocList, SupportDocView
from eda_platform.application.services.support_doc_service import SupportDocService

_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    404: {"model": ApiErrorEnvelope},
    413: {"model": ApiErrorEnvelope},
    422: {"model": ApiErrorEnvelope},
    500: {"model": ApiErrorEnvelope},
}

router = APIRouter(tags=["support-docs"], responses=_ERROR_RESPONSES)


def _service(request: Request) -> SupportDocService:
    return request.app.state.support_doc_service


@router.get("/projects/{project_id}/support-docs", response_model=SupportDocList)
def list_support_docs(
    project_id: str,
    request: Request,
    limit: int = Query(50, ge=1, le=200),
    cursor: str | None = Query(None),
) -> SupportDocList:
    return _service(request).list_docs(project_id, limit=limit, cursor=cursor)


@router.post(
    "/projects/{project_id}/support-docs",
    response_model=SupportDocView,
    status_code=201,
)
def create_support_doc(
    project_id: str, file: UploadFile, request: Request
) -> SupportDocView:
    return _service(request).create_doc(
        project_id, file.filename or "document.txt", file.file
    )


@router.delete("/projects/{project_id}/support-docs/{doc_id}", status_code=204)
def delete_support_doc(project_id: str, doc_id: str, request: Request) -> Response:
    _service(request).delete_doc(project_id, doc_id)
    return Response(status_code=204)
