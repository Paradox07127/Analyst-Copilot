"""FastAPI app factory.

Run locally with:
    uv run uvicorn eda_platform.api.main:create_app --factory
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.routing import APIRoute
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from eda_platform.api.errors import ApiErrorEnvelope, register_error_handlers
from eda_platform.api.middleware import (
    BodySizeLimitMiddleware,
    DeploymentSecurityMiddleware,
)
from eda_platform.api.mutation_contract import configure_mutation_contract
from eda_platform.api.routers.analysis import router as analysis_router
from eda_platform.api.routers.artifacts import router as artifacts_router
from eda_platform.api.routers.boards import router as boards_router
from eda_platform.api.routers.chat import router as chat_router
from eda_platform.api.routers.cleaning import router as cleaning_router
from eda_platform.api.routers.compare import router as compare_router
from eda_platform.api.routers.datasets import router as datasets_router
from eda_platform.api.routers.decision_reports import router as decision_reports_router
from eda_platform.api.routers.findings import router as findings_router
from eda_platform.api.routers.forks import router as forks_router
from eda_platform.api.routers.insights import router as insights_router
from eda_platform.api.routers.investigations import router as investigations_router
from eda_platform.api.routers.jobs import router as jobs_router
from eda_platform.api.routers.questions import router as questions_router
from eda_platform.api.routers.relationships import router as relationships_router
from eda_platform.api.routers.reports import router as reports_router
from eda_platform.api.routers.semantic import router as semantic_router
from eda_platform.api.routers.sessions import router as runs_router
from eda_platform.api.routers.settings import router as settings_router
from eda_platform.api.routers.skills import router as skills_router
from eda_platform.api.routers.support_docs import router as support_docs_router
from eda_platform.api.routers.system import router as system_router
from eda_platform.api.routers.trace import router as trace_router
from eda_platform.api.routers.uploads import router as uploads_router
from eda_platform.application.services.analysis_service import AnalysisService
from eda_platform.application.services.approval_service import ApprovalService
from eda_platform.application.services.artifact_service import ArtifactService
from eda_platform.application.services.board_service import BoardService
from eda_platform.application.services.chat_service import ChatService
from eda_platform.application.services.cleaning_service import CleaningService
from eda_platform.application.services.compare_scope_service import CompareScopeService
from eda_platform.application.services.compare_service import CompareService
from eda_platform.application.services.data_operation_service import DataOperationService
from eda_platform.application.services.dataset_service import DatasetService
from eda_platform.application.services.decision_report_service import DecisionReportService
from eda_platform.application.services.finding_service import FindingService
from eda_platform.application.services.insight_service import InsightService
from eda_platform.application.services.investigation_service import InvestigationService
from eda_platform.application.services.job_service import (
    JobService,
    recover_job_lifecycle,
)
from eda_platform.application.services.promotion_service import PromotionService
from eda_platform.application.services.question_service import QuestionService
from eda_platform.application.services.relationship_service import RelationshipService
from eda_platform.application.services.report_export_service import ReportExportService
from eda_platform.application.services.report_generation_service import (
    ReportGenerationService,
)
from eda_platform.application.services.report_service import ReportService
from eda_platform.application.services.sandbox_status_service import SandboxStatusService
from eda_platform.application.services.semantic_service import SemanticService
from eda_platform.application.services.session_fork_service import SessionForkService
from eda_platform.application.services.session_service import SessionService
from eda_platform.application.services.settings_service import SettingsService
from eda_platform.application.services.skill_service import SkillService
from eda_platform.application.services.support_doc_service import SupportDocService
from eda_platform.application.services.trace_service import TraceService
from eda_platform.application.services.upload_service import (
    MAX_UPLOAD_BYTES,
    UploadService,
    sweep_staging,
)
from eda_platform.core.config import (
    DeploymentConfig,
    deployment_config,
    resolve_workspace_path,
)
from eda_platform.core.query import TrustedFileQueryEngine
from eda_platform.core.sandbox_broker import SandboxBroker, sandbox_required_at_startup
from eda_platform.core.session_deletion import (
    SessionDeletionBlockedError,
    SessionDeletionBusyError,
    SessionDeletionCoordinator,
    SessionDeletionRetryableError,
)
from eda_platform.core.store import ArtifactStore
from eda_platform.infrastructure.job_backend import LocalProcessJobBackend

# Multipart framing adds overhead on top of the file payload itself.
_BODY_LIMIT_SLACK = 1 << 20
logger = logging.getLogger(__name__)


def create_app(
    workspace: Path | str | None = None,
    serve_web_dist: Path | None = None,
) -> FastAPI:
    root = resolve_workspace_path(workspace)
    deployment = deployment_config()
    app = FastAPI(title="EDA Agent Platform API", version="0.1.0")
    app.state.workspace = root
    app.state.deployment = deployment
    store = ArtifactStore(root)
    sweep_staging(root)
    store.reconcile_upload_quota()
    # Trusted engine allow-list: projects/* covers per-project uploads (and any
    # cleaned versions written next to them). User SQL never gets this engine.
    engine = TrustedFileQueryEngine([store.root / "projects"])
    dataset_service = DatasetService(store, engine)
    job_backend = LocalProcessJobBackend(root, store)
    job_service = JobService(store, job_backend)
    session_service = SessionService(store)
    app.state.session_service = session_service
    app.state.dataset_service = dataset_service
    app.state.upload_service = UploadService(
        store,
        engine,
        project_byte_quota=deployment.project_byte_quota,
        project_file_quota=deployment.project_file_quota,
        concurrent_upload_quota=deployment.concurrent_upload_quota,
    )
    app.state.job_service = job_service
    app.state.data_operation_service = DataOperationService(store, job_service)
    report_service = ReportService(store)
    app.state.report_service = report_service
    app.state.artifact_service = ArtifactService(store)
    app.state.insight_service = InsightService(store)
    approval_service = ApprovalService(store)
    app.state.cleaning_service = CleaningService(
        store, dataset_service, approval_service, job_service
    )
    question_service = QuestionService(store, approval_service, job_service)
    app.state.question_service = question_service
    app.state.investigation_service = InvestigationService(
        store, approval_service, job_service
    )
    finding_service = FindingService(store)
    app.state.finding_service = finding_service
    semantic_service = SemanticService(store)
    app.state.semantic_service = semantic_service
    app.state.relationship_service = RelationshipService(
        store, dataset_service, approval_service, job_service, semantic_service
    )
    app.state.promotion_service = PromotionService(store, approval_service, semantic_service)
    app.state.report_generation_service = ReportGenerationService(store, job_service)
    app.state.session_fork_service = SessionForkService(store, job_service)
    app.state.compare_service = CompareService(store, session_service)
    app.state.skill_service = SkillService(
        store, dataset_service, approval_service, job_service
    )
    app.state.chat_service = ChatService(store, approval_service)
    app.state.board_service = BoardService(store)
    analysis_service = AnalysisService(store)
    trace_service = TraceService(store)
    app.state.analysis_service = analysis_service
    app.state.trace_service = trace_service
    app.state.compare_scope_service = CompareScopeService(
        store,
        session_service,
        question_service,
        analysis_service,
        finding_service,
        report_service,
        trace_service,
    )
    decision_report_service = DecisionReportService(store, job_service)
    app.state.decision_report_service = decision_report_service
    app.state.report_export_service = ReportExportService(store, decision_report_service)
    app.state.support_doc_service = SupportDocService(store)
    app.state.sandbox_status_service = SandboxStatusService(root)
    if sandbox_required_at_startup():
        app.state.sandbox_backend = SandboxBroker.from_env(
            work_root=root / "_sandbox"
        ).require_safe_backend()
    app.state.settings_service = SettingsService(workspace=root)
    register_error_handlers(app)
    _include_api_router(app, runs_router)
    _include_api_router(app, datasets_router)
    _include_api_router(app, uploads_router)
    _include_api_router(app, jobs_router)
    _include_api_router(app, cleaning_router)
    _include_api_router(app, questions_router)
    _include_api_router(app, reports_router)
    _include_api_router(app, artifacts_router)
    _include_api_router(app, insights_router)
    _include_api_router(app, findings_router)
    _include_api_router(app, semantic_router)
    _include_api_router(app, relationships_router)
    _include_api_router(app, compare_router)
    _include_api_router(app, forks_router)
    _include_api_router(app, skills_router)
    _include_api_router(app, chat_router)
    _include_api_router(app, boards_router)
    _include_api_router(app, analysis_router)
    _include_api_router(app, trace_router)
    _include_api_router(app, decision_reports_router)
    _include_api_router(app, support_docs_router)
    _include_api_router(app, system_router)
    _include_api_router(app, settings_router)
    # Registered last so its paths append to the OpenAPI document instead of
    # reordering every route below them; second path segments are all static,
    # so routing itself does not depend on the order.
    _include_api_router(app, investigations_router)
    _configure_middleware(app, store, deployment)
    # Recover or relaunch durable jobs before run deletion recovery. A delete
    # must never quarantine a run while a recovered worker can still publish.
    recover_job_lifecycle(store, job_backend)
    # Resume every durable delete operation after job ownership is reconciled.
    # Typed non-terminal outcomes deliberately stay durable for a later request
    # or restart; unexpected corruption still fails startup loudly.
    _recover_session_deletions(store)
    if serve_web_dist is not None:
        _mount_web_dist(app, serve_web_dist)
    return app


def _configure_middleware(
    app: FastAPI, store: ArtifactStore, deployment: DeploymentConfig
) -> None:
    """Install inner-to-outer after routes exist so mutation policy is generated."""
    configure_mutation_contract(app, db_path=str(store.db_path))
    # Must sit at the ASGI layer: starlette spools the whole multipart body to
    # temp disk before any handler-level check could run.
    app.add_middleware(
        BodySizeLimitMiddleware,
        max_bytes=MAX_UPLOAD_BYTES + _BODY_LIMIT_SLACK,
    )
    if deployment.remote:
        app.add_middleware(
            DeploymentSecurityMiddleware,
            allowed_origins=deployment.allowed_origins,
            trusted_proxy_ips=deployment.trusted_proxy_ips,
            rate_limit=deployment.upload_rate_limit,
            rate_window_seconds=deployment.upload_rate_window_seconds,
            rate_checker=store.check_upload_rate_limit,
        )
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(deployment.allowed_origins),
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
            allow_headers=[
                "Content-Type",
                "Idempotency-Key",
                "X-EDA-CSRF",
                "X-EDA-Session",
            ],
        )
    # Starlette's last add_middleware call is outermost. Host validation must
    # therefore run before CSRF, rate-limit, body, and mutation replay work.
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=list(deployment.allowed_hosts),
    )


def _recover_session_deletions(store: ArtifactStore) -> None:
    coordinator = SessionDeletionCoordinator(store)
    for op_id in store.list_active_session_deletion_op_ids():
        try:
            coordinator.recover(op_id)
        except SessionDeletionBusyError as exc:
            logger.warning(
                "Run deletion recovery remains busy",
                extra={"op_id": op_id, "session_id": exc.session_id, "job_id": exc.job_id},
            )
        except SessionDeletionRetryableError as exc:
            logger.warning(
                "Run deletion recovery remains retryable",
                extra={"op_id": op_id, "reason": exc.reason},
            )
        except SessionDeletionBlockedError as exc:
            logger.error(
                "Run deletion recovery remains blocked",
                extra={"op_id": op_id, "reason": exc.reason},
            )


def _include_api_router(app: FastAPI, router: APIRouter) -> None:
    """Include one API router after declaring the global limiter on body operations."""
    response = {
        "model": ApiErrorEnvelope,
        "description": "Request body exceeds the global size limit.",
    }
    for route in router.routes:
        if isinstance(route, APIRoute):
            if route.body_field is not None:
                route.responses.setdefault(413, response)
            if (route.methods or set()) & {"POST", "PUT", "PATCH", "DELETE"}:
                route.responses.setdefault(
                    403,
                    {
                        "model": ApiErrorEnvelope,
                        "description": "Remote deployment CSRF policy rejected the request.",
                    },
                )
    app.include_router(router, prefix="/api/v1")


def _mount_web_dist(app: FastAPI, dist: Path) -> None:
    """Serve the built React app (apps/web/dist) with an SPA fallback.

    Registered after every API router, so /api/* keeps first-match priority and
    unknown /api/* paths still 404 instead of returning index.html.
    """
    dist = dist.resolve()
    index_html = dist / "index.html"
    if not index_html.is_file():
        raise FileNotFoundError(f"web dist is missing index.html: {dist}")
    assets_dir = dist / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="web-assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str) -> FileResponse:
        if full_path == "api" or full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not Found")
        candidate = (dist / full_path).resolve()
        if candidate.is_file() and candidate.is_relative_to(dist):
            return FileResponse(candidate)
        # Hashed JavaScript files change on every web build. Keeping the HTML
        # entrypoint fresh prevents a normal reload from reviving an old bundle
        # which references chunks that were removed by the latest build.
        return FileResponse(index_html, headers={"Cache-Control": "no-cache"})
