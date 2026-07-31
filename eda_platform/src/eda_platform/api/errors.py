"""Typed error envelope + handlers. Responses never leak tracebacks."""

from __future__ import annotations

import logging
from collections.abc import Mapping

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from eda_platform.application.job_results import JobResultNotReadyError
from eda_platform.application.services.approval_service import (
    ApprovalConsumedError,
    ApprovalExpiredError,
    ApprovalNotFoundError,
)
from eda_platform.application.services.artifact_service import (
    ArtifactNotFoundError,
    ArtifactTooLargeError,
)
from eda_platform.application.services.board_service import (
    BoardStateError,
    BoardValidationError,
    BoardVersionConflictError,
)
from eda_platform.application.services.chat_service import (
    ChatMessageNotFoundError,
    ChatRunBusyError,
    ChatValidationError,
)
from eda_platform.application.services.cleaning_service import (
    CleaningApplyRefusedError,
    CleaningSourceChangedError,
    CleaningValidationError,
)
from eda_platform.application.services.compare_service import (
    CompareProjectMismatchError,
    CompareSameRunError,
)
from eda_platform.application.services.dataset_service import (
    DatasetNotFoundError,
    DatasetSourceMissingError,
)
from eda_platform.application.services.decision_report_service import (
    DecisionReportCorruptError,
    DecisionReportIdentityInvalidError,
    DecisionReportMissingError,
    DecisionReportTooLargeError,
    DecisionReportUnavailableError,
    DecisionStoryBusyError,
    DecisionStoryDraftNotFoundError,
    DecisionStoryNotDraftableError,
)
from eda_platform.application.services.insight_service import ChartNotFoundError
from eda_platform.application.services.investigation_service import (
    InvestigationNotDecidableError,
    InvestigationNotExecutableError,
    InvestigationNotFoundError,
    InvestigationRunBusyError,
    InvestigationSourceChangedError,
    InvestigationValidationError,
    MacroLoopNotAuthorizedError,
)
from eda_platform.application.services.job_service import (
    JobConflictError,
    JobIdempotencyMismatchError,
    JobNotFoundError,
    JobRunDeletingError,
    JobValidationError,
)
from eda_platform.application.services.promotion_service import (
    PromotionFindingNotFoundError,
    PromotionNotAllowedError,
    PromotionSourceChangedError,
    PromotionValidationError,
)
from eda_platform.application.services.question_service import (
    QuestionNotExecutableError,
    QuestionNotFoundError,
    QuestionRunBusyError,
    QuestionSourceChangedError,
    QuestionValidationError,
    QuestionVersionConflictError,
)
from eda_platform.application.services.relationship_service import (
    RelationshipNotDiscoverableError,
    RelationshipNotFoundError,
    RelationshipNotValidatableError,
    RelationshipRunBusyError,
    RelationshipSourceChangedError,
    RelationshipValidationRequestError,
)
from eda_platform.application.services.report_export_service import (
    ReportExportUnavailableError,
    ReportNotExportableError,
)
from eda_platform.application.services.report_generation_service import (
    ReportNotGeneratableError,
    ReportRunBusyError,
)
from eda_platform.application.services.report_service import ReportTooLargeError
from eda_platform.application.services.session_fork_service import (
    SessionForkBusyError,
    SessionForkNotForkableError,
    SessionForkValidationError,
)
from eda_platform.application.services.session_service import (
    InvalidCursorError,
    ProjectBusyError,
    ProjectConflictError,
    ProjectNotFoundError,
    ProjectValidationError,
    SessionBusyError,
    SessionDeleteBlockedError,
    SessionDeleteRetryableError,
    SessionNotFoundError,
)
from eda_platform.application.services.semantic_service import (
    SemanticJoinNotConfirmableError,
    SemanticJoinNotFoundError,
    SemanticJoinStateError,
    SemanticProposalConflictError,
    SemanticProposalNotFoundError,
    SemanticSeedsInvalidError,
    SemanticSeedsOutOfBandError,
    SemanticStateError,
    SemanticValidationError,
    SemanticVersionConflictError,
)
from eda_platform.application.services.settings_service import (
    SettingsValidationError,
    SettingsVersionConflictError,
)
from eda_platform.application.services.skill_service import (
    SkillBindingInvalidError,
    SkillNotDeletableError,
    SkillNotFoundError,
    SkillNotInLibraryError,
    SkillPlanNotFoundError,
    SkillSqlRejectedError,
    SkillValidationError,
)
from eda_platform.application.services.support_doc_service import (
    SupportDocNotFoundError,
    SupportDocTooLargeError,
    SupportDocValidationError,
)
from eda_platform.application.services.trace_service import (
    ClientFailureRateLimitError,
    ClientFailureTooLargeError,
)
from eda_platform.application.services.upload_service import (
    UploadConcurrentQuotaError,
    UploadFileQuotaError,
    UploadInUseError,
    UploadNotFoundError,
    UploadProjectByteQuotaError,
    UploadTooLargeError,
    UploadValidationError,
)
from eda_platform.core.query import TrustedPathError

logger = logging.getLogger(__name__)


class ApiErrorInfo(BaseModel):
    code: str
    message: str


class ApiErrorEnvelope(BaseModel):
    error: ApiErrorInfo


def error_response(
    status_code: int,
    code: str,
    message: str,
    *,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    envelope = ApiErrorEnvelope(error=ApiErrorInfo(code=code, message=message))
    return JSONResponse(
        status_code=status_code,
        content=envelope.model_dump(),
        headers=headers,
    )


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ProjectNotFoundError)
    def _project_not_found(request: Request, exc: ProjectNotFoundError) -> JSONResponse:
        return error_response(404, "project_not_found", str(exc))

    @app.exception_handler(ProjectValidationError)
    def _project_invalid(request: Request, exc: ProjectValidationError) -> JSONResponse:
        return error_response(422, "project_invalid", str(exc))

    @app.exception_handler(ProjectConflictError)
    def _project_conflict(request: Request, exc: ProjectConflictError) -> JSONResponse:
        return error_response(409, "project_conflict", str(exc))

    @app.exception_handler(ProjectBusyError)
    def _project_busy(request: Request, exc: ProjectBusyError) -> JSONResponse:
        return error_response(409, "project_busy", str(exc))

    @app.exception_handler(SessionNotFoundError)
    def _run_not_found(request: Request, exc: SessionNotFoundError) -> JSONResponse:
        return error_response(404, "session_not_found", str(exc))

    @app.exception_handler(SessionBusyError)
    def _run_busy(request: Request, exc: SessionBusyError) -> JSONResponse:
        return error_response(409, "session_busy", str(exc))

    @app.exception_handler(SessionDeleteRetryableError)
    def _run_delete_retryable(
        request: Request, exc: SessionDeleteRetryableError
    ) -> JSONResponse:
        return error_response(
            503,
            "session_delete_retryable",
            str(exc),
            headers={"Retry-After": "1"},
        )

    @app.exception_handler(SessionDeleteBlockedError)
    def _run_delete_blocked(
        request: Request, exc: SessionDeleteBlockedError
    ) -> JSONResponse:
        return error_response(409, "session_delete_blocked", str(exc))

    @app.exception_handler(SettingsValidationError)
    def _settings_invalid(request: Request, exc: SettingsValidationError) -> JSONResponse:
        return error_response(422, "settings_invalid", str(exc))

    @app.exception_handler(SettingsVersionConflictError)
    def _settings_version_conflict(
        request: Request, exc: SettingsVersionConflictError
    ) -> JSONResponse:
        return error_response(409, "settings_version_conflict", str(exc))

    @app.exception_handler(QuestionVersionConflictError)
    def _question_version_conflict(
        request: Request, exc: QuestionVersionConflictError
    ) -> JSONResponse:
        return error_response(409, "question_version_conflict", str(exc))

    @app.exception_handler(InvalidCursorError)
    def _invalid_cursor(request: Request, exc: InvalidCursorError) -> JSONResponse:
        return error_response(400, "invalid_cursor", str(exc))

    @app.exception_handler(ArtifactNotFoundError)
    def _artifact_not_found(request: Request, exc: ArtifactNotFoundError) -> JSONResponse:
        return error_response(404, "artifact_not_found", str(exc))

    @app.exception_handler(ArtifactTooLargeError)
    def _artifact_too_large(request: Request, exc: ArtifactTooLargeError) -> JSONResponse:
        return error_response(413, "artifact_too_large", str(exc))

    @app.exception_handler(DatasetNotFoundError)
    def _dataset_not_found(request: Request, exc: DatasetNotFoundError) -> JSONResponse:
        return error_response(404, "dataset_not_found", str(exc))

    @app.exception_handler(DatasetSourceMissingError)
    def _dataset_source_missing(request: Request, exc: DatasetSourceMissingError) -> JSONResponse:
        return error_response(404, "dataset_source_missing", str(exc))

    @app.exception_handler(ChartNotFoundError)
    def _chart_not_found(request: Request, exc: ChartNotFoundError) -> JSONResponse:
        return error_response(404, "chart_not_found", str(exc))

    @app.exception_handler(JobNotFoundError)
    def _job_not_found(request: Request, exc: JobNotFoundError) -> JSONResponse:
        return error_response(404, "job_not_found", str(exc))

    @app.exception_handler(JobValidationError)
    def _job_invalid(request: Request, exc: JobValidationError) -> JSONResponse:
        return error_response(422, "job_invalid", str(exc))

    @app.exception_handler(JobIdempotencyMismatchError)
    def _idempotency_key_reused(
        request: Request, exc: JobIdempotencyMismatchError
    ) -> JSONResponse:
        return error_response(422, "idempotency_key_reused", str(exc))

    @app.exception_handler(JobConflictError)
    def _job_conflict(request: Request, exc: JobConflictError) -> JSONResponse:
        return error_response(409, "job_conflict", str(exc))

    @app.exception_handler(JobResultNotReadyError)
    def _job_result_not_ready(
        request: Request, exc: JobResultNotReadyError
    ) -> JSONResponse:
        return error_response(
            409,
            "job_result_not_ready",
            f"Job result is not ready: {exc}",
        )

    @app.exception_handler(JobRunDeletingError)
    def _job_run_deleting(request: Request, exc: JobRunDeletingError) -> JSONResponse:
        return error_response(409, "session_deleting", str(exc))

    @app.exception_handler(UploadNotFoundError)
    def _upload_not_found(request: Request, exc: UploadNotFoundError) -> JSONResponse:
        return error_response(404, "upload_not_found", str(exc))

    @app.exception_handler(UploadInUseError)
    def _upload_in_use(request: Request, exc: UploadInUseError) -> JSONResponse:
        return error_response(409, "upload_in_use", str(exc))

    @app.exception_handler(UploadValidationError)
    def _upload_invalid(request: Request, exc: UploadValidationError) -> JSONResponse:
        return error_response(422, "upload_invalid", str(exc))

    @app.exception_handler(UploadTooLargeError)
    def _upload_too_large(request: Request, exc: UploadTooLargeError) -> JSONResponse:
        return error_response(413, "upload_too_large", str(exc))

    @app.exception_handler(UploadProjectByteQuotaError)
    def _upload_project_bytes(
        request: Request, exc: UploadProjectByteQuotaError
    ) -> JSONResponse:
        return error_response(413, "upload_project_byte_quota", str(exc))

    @app.exception_handler(UploadFileQuotaError)
    def _upload_file_quota(request: Request, exc: UploadFileQuotaError) -> JSONResponse:
        return error_response(429, "upload_file_quota", str(exc))

    @app.exception_handler(UploadConcurrentQuotaError)
    def _upload_concurrent_quota(
        request: Request, exc: UploadConcurrentQuotaError
    ) -> JSONResponse:
        return error_response(
            429,
            "upload_concurrent_quota",
            str(exc),
            headers={"Retry-After": "1"},
        )

    @app.exception_handler(ApprovalNotFoundError)
    def _approval_not_found(request: Request, exc: ApprovalNotFoundError) -> JSONResponse:
        return error_response(404, "approval_not_found", str(exc))

    @app.exception_handler(ApprovalExpiredError)
    def _approval_expired(request: Request, exc: ApprovalExpiredError) -> JSONResponse:
        return error_response(410, "approval_expired", str(exc))

    @app.exception_handler(ApprovalConsumedError)
    def _approval_consumed(request: Request, exc: ApprovalConsumedError) -> JSONResponse:
        return error_response(409, "approval_consumed", str(exc))

    @app.exception_handler(CleaningValidationError)
    def _cleaning_invalid(request: Request, exc: CleaningValidationError) -> JSONResponse:
        return error_response(422, "cleaning_invalid", str(exc))

    @app.exception_handler(CleaningApplyRefusedError)
    def _cleaning_refused(request: Request, exc: CleaningApplyRefusedError) -> JSONResponse:
        return error_response(409, "cleaning_refused", str(exc))

    @app.exception_handler(CleaningSourceChangedError)
    def _cleaning_source_changed(
        request: Request, exc: CleaningSourceChangedError
    ) -> JSONResponse:
        return error_response(409, "cleaning_source_changed", str(exc))

    @app.exception_handler(QuestionNotFoundError)
    def _question_not_found(request: Request, exc: QuestionNotFoundError) -> JSONResponse:
        return error_response(404, "question_not_found", str(exc))

    @app.exception_handler(QuestionNotExecutableError)
    def _question_not_executable(
        request: Request, exc: QuestionNotExecutableError
    ) -> JSONResponse:
        return error_response(409, "question_not_executable", str(exc))

    @app.exception_handler(QuestionSourceChangedError)
    def _question_source_changed(
        request: Request, exc: QuestionSourceChangedError
    ) -> JSONResponse:
        return error_response(409, "question_source_changed", str(exc))

    @app.exception_handler(QuestionRunBusyError)
    def _question_run_busy(request: Request, exc: QuestionRunBusyError) -> JSONResponse:
        return error_response(409, "question_session_busy", str(exc))

    @app.exception_handler(QuestionValidationError)
    def _question_invalid(request: Request, exc: QuestionValidationError) -> JSONResponse:
        return error_response(422, "question_invalid", str(exc))

    @app.exception_handler(InvestigationNotFoundError)
    def _investigation_not_found(
        request: Request, exc: InvestigationNotFoundError
    ) -> JSONResponse:
        return error_response(404, "investigation_not_found", str(exc))

    @app.exception_handler(InvestigationNotDecidableError)
    def _investigation_not_decidable(
        request: Request, exc: InvestigationNotDecidableError
    ) -> JSONResponse:
        return error_response(409, "investigation_not_decidable", str(exc))

    @app.exception_handler(InvestigationNotExecutableError)
    def _investigation_not_executable(
        request: Request, exc: InvestigationNotExecutableError
    ) -> JSONResponse:
        return error_response(409, "investigation_not_executable", str(exc))

    @app.exception_handler(InvestigationSourceChangedError)
    def _investigation_source_changed(
        request: Request, exc: InvestigationSourceChangedError
    ) -> JSONResponse:
        return error_response(409, "investigation_source_changed", str(exc))

    @app.exception_handler(InvestigationRunBusyError)
    def _investigation_run_busy(
        request: Request, exc: InvestigationRunBusyError
    ) -> JSONResponse:
        return error_response(409, "investigation_session_busy", str(exc))

    @app.exception_handler(MacroLoopNotAuthorizedError)
    def _macro_loop_not_authorized(
        request: Request, exc: MacroLoopNotAuthorizedError
    ) -> JSONResponse:
        return error_response(409, "macro_loop_not_authorized", str(exc))

    @app.exception_handler(InvestigationValidationError)
    def _investigation_invalid(
        request: Request, exc: InvestigationValidationError
    ) -> JSONResponse:
        return error_response(422, "investigation_invalid", str(exc))

    @app.exception_handler(PromotionFindingNotFoundError)
    def _promotion_finding_not_found(
        request: Request, exc: PromotionFindingNotFoundError
    ) -> JSONResponse:
        return error_response(404, "finding_not_found", str(exc))

    @app.exception_handler(PromotionNotAllowedError)
    def _promotion_not_allowed(request: Request, exc: PromotionNotAllowedError) -> JSONResponse:
        return error_response(409, "promotion_not_allowed", str(exc))

    @app.exception_handler(PromotionSourceChangedError)
    def _promotion_source_changed(
        request: Request, exc: PromotionSourceChangedError
    ) -> JSONResponse:
        return error_response(409, "promotion_source_changed", str(exc))

    @app.exception_handler(PromotionValidationError)
    def _promotion_invalid(request: Request, exc: PromotionValidationError) -> JSONResponse:
        return error_response(422, "promotion_invalid", str(exc))

    @app.exception_handler(ReportRunBusyError)
    def _report_run_busy(request: Request, exc: ReportRunBusyError) -> JSONResponse:
        return error_response(409, "report_session_busy", str(exc))

    @app.exception_handler(ReportNotGeneratableError)
    def _report_not_generatable(
        request: Request, exc: ReportNotGeneratableError
    ) -> JSONResponse:
        return error_response(409, "report_not_generatable", str(exc))

    @app.exception_handler(ReportTooLargeError)
    def _report_too_large(request: Request, exc: ReportTooLargeError) -> JSONResponse:
        return error_response(413, "report_too_large", str(exc))

    @app.exception_handler(SessionForkBusyError)
    def _run_fork_busy(request: Request, exc: SessionForkBusyError) -> JSONResponse:
        return error_response(409, "session_fork_busy", str(exc))

    @app.exception_handler(SessionForkNotForkableError)
    def _run_fork_not_forkable(request: Request, exc: SessionForkNotForkableError) -> JSONResponse:
        return error_response(409, "session_fork_not_forkable", str(exc))

    @app.exception_handler(SessionForkValidationError)
    def _run_fork_invalid(request: Request, exc: SessionForkValidationError) -> JSONResponse:
        return error_response(422, "session_fork_invalid", str(exc))

    @app.exception_handler(CompareProjectMismatchError)
    def _compare_project_mismatch(
        request: Request, exc: CompareProjectMismatchError
    ) -> JSONResponse:
        return error_response(422, "compare_project_mismatch", str(exc))

    @app.exception_handler(CompareSameRunError)
    def _compare_same_run(request: Request, exc: CompareSameRunError) -> JSONResponse:
        return error_response(422, "compare_same_session", str(exc))

    @app.exception_handler(SkillNotFoundError)
    def _skill_not_found(request: Request, exc: SkillNotFoundError) -> JSONResponse:
        return error_response(404, "skill_not_found", str(exc))

    @app.exception_handler(SkillBindingInvalidError)
    def _skill_binding_invalid(request: Request, exc: SkillBindingInvalidError) -> JSONResponse:
        return error_response(422, "binding_invalid", str(exc))

    @app.exception_handler(SkillSqlRejectedError)
    def _skill_sql_rejected(request: Request, exc: SkillSqlRejectedError) -> JSONResponse:
        return error_response(422, "skill_sql_rejected", str(exc))

    @app.exception_handler(SkillValidationError)
    def _skill_invalid(request: Request, exc: SkillValidationError) -> JSONResponse:
        return error_response(422, "skill_invalid", str(exc))

    @app.exception_handler(SkillNotInLibraryError)
    def _skill_not_in_library(request: Request, exc: SkillNotInLibraryError) -> JSONResponse:
        return error_response(404, "skill_not_found", str(exc))

    @app.exception_handler(SkillNotDeletableError)
    def _skill_not_deletable(request: Request, exc: SkillNotDeletableError) -> JSONResponse:
        return error_response(409, "skill_not_deletable", str(exc))

    @app.exception_handler(SkillPlanNotFoundError)
    def _skill_plan_not_found(request: Request, exc: SkillPlanNotFoundError) -> JSONResponse:
        return error_response(404, "skill_plan_not_found", str(exc))

    @app.exception_handler(RelationshipNotFoundError)
    def _relationship_not_found(request: Request, exc: RelationshipNotFoundError) -> JSONResponse:
        return error_response(404, "relationship_not_found", str(exc))

    @app.exception_handler(RelationshipNotValidatableError)
    def _relationship_not_validatable(
        request: Request, exc: RelationshipNotValidatableError
    ) -> JSONResponse:
        return error_response(409, "relationship_not_validatable", str(exc))

    @app.exception_handler(RelationshipSourceChangedError)
    def _relationship_source_changed(
        request: Request, exc: RelationshipSourceChangedError
    ) -> JSONResponse:
        return error_response(409, "relationship_source_changed", str(exc))

    @app.exception_handler(RelationshipRunBusyError)
    def _relationship_run_busy(request: Request, exc: RelationshipRunBusyError) -> JSONResponse:
        return error_response(409, "relationship_session_busy", str(exc))

    @app.exception_handler(RelationshipValidationRequestError)
    def _relationship_invalid(
        request: Request, exc: RelationshipValidationRequestError
    ) -> JSONResponse:
        return error_response(422, "relationship_invalid", str(exc))

    @app.exception_handler(RelationshipNotDiscoverableError)
    def _relationship_not_discoverable(
        request: Request, exc: RelationshipNotDiscoverableError
    ) -> JSONResponse:
        return error_response(409, "relationship_not_discoverable", str(exc))

    @app.exception_handler(SemanticVersionConflictError)
    def _semantic_version_conflict(
        request: Request, exc: SemanticVersionConflictError
    ) -> JSONResponse:
        return error_response(409, "version_conflict", str(exc))

    @app.exception_handler(SemanticSeedsOutOfBandError)
    def _semantic_out_of_band(
        request: Request, exc: SemanticSeedsOutOfBandError
    ) -> JSONResponse:
        return error_response(409, "version_conflict", str(exc))

    @app.exception_handler(SemanticStateError)
    def _semantic_state_corrupt(request: Request, exc: SemanticStateError) -> JSONResponse:
        logger.error("Corrupt semantic state: %s", exc)
        return error_response(500, "semantic_state_corrupt", str(exc))

    @app.exception_handler(SemanticJoinNotFoundError)
    def _semantic_join_not_found(
        request: Request, exc: SemanticJoinNotFoundError
    ) -> JSONResponse:
        return error_response(404, "join_not_found", str(exc))

    @app.exception_handler(SemanticJoinStateError)
    def _semantic_join_state(request: Request, exc: SemanticJoinStateError) -> JSONResponse:
        return error_response(409, "join_state_invalid", str(exc))

    @app.exception_handler(SemanticJoinNotConfirmableError)
    def _semantic_join_not_confirmable(
        request: Request, exc: SemanticJoinNotConfirmableError
    ) -> JSONResponse:
        return error_response(409, "join_not_confirmable", str(exc))

    @app.exception_handler(SemanticProposalNotFoundError)
    def _semantic_proposal_not_found(
        request: Request, exc: SemanticProposalNotFoundError
    ) -> JSONResponse:
        return error_response(404, "proposal_not_found", str(exc))

    @app.exception_handler(SemanticProposalConflictError)
    def _semantic_proposal_conflict(
        request: Request, exc: SemanticProposalConflictError
    ) -> JSONResponse:
        return error_response(409, "proposal_conflict", str(exc))

    @app.exception_handler(SemanticValidationError)
    def _semantic_invalid(request: Request, exc: SemanticValidationError) -> JSONResponse:
        return error_response(422, "semantic_invalid", str(exc))

    @app.exception_handler(SemanticSeedsInvalidError)
    def _semantic_seeds_invalid(
        request: Request, exc: SemanticSeedsInvalidError
    ) -> JSONResponse:
        return error_response(422, "seeds_invalid", str(exc))

    @app.exception_handler(DecisionStoryBusyError)
    def _decision_story_busy(request: Request, exc: DecisionStoryBusyError) -> JSONResponse:
        return error_response(409, "decision_story_busy", str(exc))

    @app.exception_handler(DecisionStoryNotDraftableError)
    def _decision_story_not_draftable(
        request: Request, exc: DecisionStoryNotDraftableError
    ) -> JSONResponse:
        return error_response(422, "decision_story_not_draftable", str(exc))

    @app.exception_handler(DecisionStoryDraftNotFoundError)
    def _decision_story_draft_not_found(
        request: Request, exc: DecisionStoryDraftNotFoundError
    ) -> JSONResponse:
        return error_response(404, "decision_story_draft_not_found", str(exc))

    @app.exception_handler(DecisionReportMissingError)
    def _decision_report_missing(
        request: Request, exc: DecisionReportMissingError
    ) -> JSONResponse:
        return error_response(404, "decision_report_missing", str(exc))

    @app.exception_handler(DecisionReportCorruptError)
    def _decision_report_corrupt(
        request: Request, exc: DecisionReportCorruptError
    ) -> JSONResponse:
        logger.error("Corrupt stored decision report: %s", exc)
        return error_response(500, "decision_report_corrupt", str(exc))

    @app.exception_handler(DecisionReportIdentityInvalidError)
    def _decision_report_identity_invalid(
        request: Request, exc: DecisionReportIdentityInvalidError
    ) -> JSONResponse:
        logger.error("Invalid stored decision report identity: %s", exc)
        return error_response(500, "decision_report_identity_invalid", str(exc))

    @app.exception_handler(DecisionReportTooLargeError)
    def _decision_report_too_large(
        request: Request, exc: DecisionReportTooLargeError
    ) -> JSONResponse:
        logger.error("Oversized stored decision report: %s", exc)
        return error_response(500, "decision_report_too_large", str(exc))

    @app.exception_handler(DecisionReportUnavailableError)
    def _decision_report_unavailable(
        request: Request, exc: DecisionReportUnavailableError
    ) -> JSONResponse:
        return error_response(
            503,
            "decision_report_unavailable",
            str(exc),
            headers={"Retry-After": "1"},
        )

    @app.exception_handler(ChatValidationError)
    def _chat_invalid(request: Request, exc: ChatValidationError) -> JSONResponse:
        return error_response(422, "chat_invalid", str(exc))

    @app.exception_handler(ChatRunBusyError)
    def _chat_busy(request: Request, exc: ChatRunBusyError) -> JSONResponse:
        return error_response(409, "chat_busy", str(exc))

    @app.exception_handler(ChatMessageNotFoundError)
    def _chat_message_not_found(
        request: Request, exc: ChatMessageNotFoundError
    ) -> JSONResponse:
        return error_response(404, "chat_message_not_found", str(exc))

    @app.exception_handler(BoardValidationError)
    def _board_invalid(request: Request, exc: BoardValidationError) -> JSONResponse:
        return error_response(422, "board_invalid", str(exc))

    @app.exception_handler(BoardVersionConflictError)
    def _board_version_conflict(
        request: Request, exc: BoardVersionConflictError
    ) -> JSONResponse:
        return error_response(409, "version_conflict", str(exc))

    @app.exception_handler(BoardStateError)
    def _board_state_corrupt(request: Request, exc: BoardStateError) -> JSONResponse:
        logger.error("Corrupt board state: %s", exc)
        return error_response(500, "board_state_corrupt", str(exc))

    @app.exception_handler(ReportNotExportableError)
    def _report_not_exportable(
        request: Request, exc: ReportNotExportableError
    ) -> JSONResponse:
        return error_response(404, "report_not_exportable", str(exc))

    @app.exception_handler(ReportExportUnavailableError)
    def _report_export_unavailable(
        request: Request, exc: ReportExportUnavailableError
    ) -> JSONResponse:
        # 503, not 500: the request is valid and the fix is an install step.
        return error_response(503, "report_export_unavailable", str(exc))

    @app.exception_handler(SupportDocNotFoundError)
    def _support_doc_not_found(
        request: Request, exc: SupportDocNotFoundError
    ) -> JSONResponse:
        return error_response(404, "support_doc_not_found", str(exc))

    @app.exception_handler(SupportDocValidationError)
    def _support_doc_invalid(
        request: Request, exc: SupportDocValidationError
    ) -> JSONResponse:
        return error_response(422, "support_doc_invalid", str(exc))

    @app.exception_handler(SupportDocTooLargeError)
    def _support_doc_too_large(
        request: Request, exc: SupportDocTooLargeError
    ) -> JSONResponse:
        return error_response(413, "support_doc_too_large", str(exc))

    @app.exception_handler(TrustedPathError)
    def _path_not_allowed(request: Request, exc: TrustedPathError) -> JSONResponse:
        # Reaching this means a locator bug, never user SQL. The exception text
        # contains the offending server path — log it, never echo it.
        logger.warning("Trusted path rejected: %s", exc)
        return error_response(400, "path_not_allowed", "Requested data is not accessible.")

    @app.exception_handler(ClientFailureTooLargeError)
    def _client_failure_too_large(
        request: Request, exc: ClientFailureTooLargeError
    ) -> JSONResponse:
        return error_response(413, "client_failure_too_large", str(exc))

    @app.exception_handler(ClientFailureRateLimitError)
    def _client_failure_rate_limited(
        request: Request, exc: ClientFailureRateLimitError
    ) -> JSONResponse:
        return error_response(
            429,
            "client_failure_rate_limited",
            str(exc),
            headers={"Retry-After": "60"},
        )

    @app.exception_handler(RequestValidationError)
    def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return error_response(422, "validation_error", "Request validation failed.")

    @app.exception_handler(StarletteHTTPException)
    def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        message = exc.detail if isinstance(exc.detail, str) else "HTTP error."
        return error_response(exc.status_code, "http_error", message)

    @app.exception_handler(Exception)
    def _internal_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled API error", exc_info=exc)
        return error_response(500, "internal_error", "Internal server error.")
