"""Upload endpoints (§7.2): multipart streamed to per-upload staging, promoted
atomically into the canonical uploads layout."""

from __future__ import annotations

from fastapi import APIRouter, Request, UploadFile

from eda_platform.api.errors import ApiErrorEnvelope
from eda_platform.application.dto import DatasetHandle, UploadDeleted, UploadStatus
from eda_platform.application.services.upload_service import UploadService

_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    404: {"model": ApiErrorEnvelope},
    413: {"model": ApiErrorEnvelope},
    429: {"model": ApiErrorEnvelope},
    422: {"model": ApiErrorEnvelope},
    500: {"model": ApiErrorEnvelope},
}

router = APIRouter(tags=["uploads"], responses=_ERROR_RESPONSES)


def _service(request: Request) -> UploadService:
    return request.app.state.upload_service


@router.post(
    "/projects/{project_id}/uploads",
    response_model=UploadStatus,
    status_code=201,
)
def create_upload(project_id: str, file: UploadFile, request: Request) -> UploadStatus:
    # UploadFile.file is a SpooledTemporaryFile the service reads in chunks —
    # the request body is never materialised as one bytes object here.
    return _service(request).create_upload(project_id, file.filename or "upload.csv", file.file)


@router.get(
    "/projects/{project_id}/uploads",
    response_model=list[DatasetHandle],
)
def list_uploads(project_id: str, request: Request) -> list[DatasetHandle]:
    """Datasets already in this project, so a new session can reuse one instead
    of the user hunting down the same CSV again."""
    return _service(request).list_uploads(project_id)


@router.delete(
    "/projects/{project_id}/uploads/{dataset_id}",
    response_model=UploadDeleted,
)
def delete_upload(project_id: str, dataset_id: str, request: Request) -> UploadDeleted:
    """Remove an uploaded file that no session is reading from."""
    _service(request).delete_upload(project_id, dataset_id)
    return UploadDeleted(project_id=project_id, dataset_id=dataset_id)
