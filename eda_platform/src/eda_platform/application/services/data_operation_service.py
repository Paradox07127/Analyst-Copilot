"""Queue and recover long-running data scan operations."""

from __future__ import annotations

import uuid
from typing import Any, TypeVar

from pydantic import BaseModel

from eda_platform.application.dto import DataOperationStarted, JobCreated
from eda_platform.application.job_results import (
    JobResultNotReadyError,
    read_job_result,
)
from eda_platform.application.services.job_service import (
    JobConflictError,
    JobService,
)
from eda_platform.application.services.session_service import SessionNotFoundError
from eda_platform.core.ids import INTERNAL_SESSION_MARKER
from eda_platform.core.store import ArtifactStore

ResultT = TypeVar("ResultT", bound=BaseModel)


class DataOperationService:
    def __init__(self, store: ArtifactStore, jobs: JobService) -> None:
        self._store = store
        self._jobs = jobs

    def start(
        self,
        source_session_id: str,
        *,
        kind: str,
        params: dict[str, Any],
        idempotency_key: str | None,
        llm_env: dict[str, str] | None = None,
    ) -> DataOperationStarted:
        if INTERNAL_SESSION_MARKER in source_session_id:
            raise SessionNotFoundError(source_session_id)
        row = self._store.get_session_index_row(source_session_id)
        if row is None:
            raise SessionNotFoundError(source_session_id)
        execution_session_id = f"dop_{uuid.uuid4().hex[:20]}"
        job = self._jobs.create_data_operation_job(
            execution_session_id,
            kind=kind,
            project_id=str(row["project_id"]),
            source_session_id=source_session_id,
            params=params,
            idempotency_key=idempotency_key,
            idempotency_content={"source_session_id": source_session_id, **params},
            llm_env=llm_env,
        )
        return DataOperationStarted(
            session_id=source_session_id,
            execution_session_id=job.session_id,
            job=JobCreated(
                job_id=job.job_id,
                session_id=job.session_id,
                status=job.status,
                events_url=job.events_url,
            ),
        )

    def result(
        self,
        job_id: str,
        *,
        expected_kind: str,
        model: type[ResultT],
    ) -> ResultT:
        job = self._jobs.get_job(job_id)
        if job.kind != expected_kind:
            raise JobConflictError(
                job_id,
                f"Job {job_id} is {job.kind}, not {expected_kind}.",
            )
        if job.status != "completed":
            raise JobResultNotReadyError(job_id)
        # The job's own run id is the throwaway dop_ lifecycle run; results are
        # filed under the source run so deleting that run reclaims them.
        row = self._store.get_job(job_id)
        if row is None:
            raise JobResultNotReadyError(job_id)
        return model.model_validate_json(
            read_job_result(
                self._store.root,
                str(row["project_id"]),
                str(row["request_scope"]),
                job_id,
            )
        )
