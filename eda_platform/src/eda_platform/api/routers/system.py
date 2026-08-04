"""System status endpoints. Read-only, never fails the page."""

from __future__ import annotations

from fastapi import APIRouter, Request

from eda_platform.api.errors import ApiErrorEnvelope
from eda_platform.application.dto import (
    HealthStatusView,
    SandboxStatusView,
    SystemCapabilitiesView,
)
from eda_platform.application.services.report_export_service import PDF_INSTALL_HINT
from eda_platform.application.services.sandbox_status_service import SandboxStatusService

_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    500: {"model": ApiErrorEnvelope},
}

router = APIRouter(tags=["system"], responses=_ERROR_RESPONSES)


@router.get("/health", response_model=HealthStatusView)
def get_health() -> HealthStatusView:
    return HealthStatusView()


@router.get("/system/sandbox", response_model=SandboxStatusView)
def get_sandbox_status(request: Request) -> SandboxStatusView:
    service: SandboxStatusService = request.app.state.sandbox_status_service
    return service.get_status()


@router.get("/system/capabilities", response_model=SystemCapabilitiesView)
def get_capabilities() -> SystemCapabilitiesView:
    from eda_platform.application.services.exploration_service import (
        resolve_configured_release_trust,
    )
    from eda_platform.tools.pdf_exporter import is_pdf_available

    available = is_pdf_available()
    exploration_available = resolve_configured_release_trust().certificate is not None
    return SystemCapabilitiesView(
        pdf_export_available=available,
        pdf_export_hint="" if available else PDF_INSTALL_HINT,
        exploration_available=exploration_available,
        exploration_hint=(
            ""
            if exploration_available
            else "Exploration remains unavailable until a verified E4a release is installed."
        ),
    )
