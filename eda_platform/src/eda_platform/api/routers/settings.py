"""Settings endpoints (§6.0).

Session scope comes from the `X-EDA-Session` header; without it every caller
shares the process-wide "default" session, which is what a single-user loopback
deployment wants. No response here ever carries the API key — `SettingsView`
has no field for it.
"""

from __future__ import annotations

from fastapi import APIRouter, Header, Request, Response

from eda_platform.api.errors import ApiErrorEnvelope
from eda_platform.application.dto import (
    ConnectionTestResult,
    ModelCatalog,
    ProviderInfo,
    SettingsPatch,
    SettingsView,
)
from eda_platform.application.services.settings_service import (
    DEFAULT_SESSION_ID,
    SettingsService,
    SettingsValidationError,
)

SESSION_HEADER = "X-EDA-Session"
MAX_SESSION_ID_LENGTH = 128

_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    422: {"model": ApiErrorEnvelope},
    500: {"model": ApiErrorEnvelope},
}

router = APIRouter(tags=["settings"], responses=_ERROR_RESPONSES)


def _service(request: Request) -> SettingsService:
    return request.app.state.settings_service


def session_id_from_header(raw: str | None) -> str:
    session_id = (raw or "").strip()[:MAX_SESSION_ID_LENGTH]
    return session_id or DEFAULT_SESSION_ID


def version_from_if_match(raw: str | None) -> int | None:
    if raw is None:
        return None
    normalized = raw.strip().strip('"')
    if not normalized.isdigit():
        raise SettingsValidationError('If-Match must be an integer ETag such as "3".')
    return int(normalized)


@router.get("/settings", response_model=SettingsView)
def get_settings(
    request: Request,
    response: Response,
    x_eda_session: str | None = Header(None, alias=SESSION_HEADER),
) -> SettingsView:
    view = _service(request).get_settings(session_id_from_header(x_eda_session))
    response.headers["ETag"] = f'"{view.version}"'
    return view


@router.put("/settings", response_model=SettingsView)
def update_settings(
    body: SettingsPatch,
    request: Request,
    response: Response,
    x_eda_session: str | None = Header(None, alias=SESSION_HEADER),
    if_match: str | None = Header(None, alias="If-Match"),
) -> SettingsView:
    view = _service(request).update_settings(
        body,
        session_id_from_header(x_eda_session),
        expected_version=version_from_if_match(if_match),
    )
    response.headers["ETag"] = f'"{view.version}"'
    return view


@router.delete("/settings", response_model=SettingsView)
def reset_settings(
    request: Request,
    response: Response,
    x_eda_session: str | None = Header(None, alias=SESSION_HEADER),
    if_match: str | None = Header(None, alias="If-Match"),
) -> SettingsView:
    view = _service(request).reset(
        session_id_from_header(x_eda_session),
        expected_version=version_from_if_match(if_match),
    )
    response.headers["ETag"] = f'"{view.version}"'
    return view


@router.post("/settings/test", response_model=ConnectionTestResult)
def test_connection(
    request: Request,
    x_eda_session: str | None = Header(None, alias=SESSION_HEADER),
) -> ConnectionTestResult:
    return _service(request).test_connection(session_id_from_header(x_eda_session))


@router.get("/settings/providers", response_model=list[ProviderInfo])
def list_providers(request: Request) -> list[ProviderInfo]:
    return _service(request).list_providers()


@router.get("/settings/models", response_model=ModelCatalog)
def list_models(
    request: Request,
    x_eda_session: str | None = Header(None, alias=SESSION_HEADER),
) -> ModelCatalog:
    """Models the saved credential can actually see, cached for a few minutes."""
    return _service(request).list_models(session_id_from_header(x_eda_session))


@router.post("/settings/models/refresh", response_model=ModelCatalog)
def refresh_models(
    request: Request,
    x_eda_session: str | None = Header(None, alias=SESSION_HEADER),
) -> ModelCatalog:
    return _service(request).list_models(
        session_id_from_header(x_eda_session), refresh=True
    )
