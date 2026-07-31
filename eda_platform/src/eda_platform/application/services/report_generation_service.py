"""On-demand report generation (§7.5 write side; the read side stays in
`report_service`).

`run_auto_eda` can be started with `generate_report=False` so the user reviews
questions before paying for a report, which leaves runs with no report at all.
This queues the existing `generate_report_on_demand` driver as a
`report_generate` job: it runs on its own derived `rpsess_*` run, while the
report artifacts land on the source run. Regenerating is the same call — the
report reader already prefers the newest MarkdownReport.
"""

from __future__ import annotations

from datetime import UTC, datetime

from eda_platform.application.dto import JobCreated, JobStatus, ReportGenerationStarted
from eda_platform.application.services.job_service import JobConflictError, JobService
from eda_platform.application.services.session_service import SessionNotFoundError
from eda_platform.core.ids import INTERNAL_SESSION_MARKER, stable_hash
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import ArtifactType

REPORT_JOB_KIND = "report_generate"
REPORT_SESSION_PREFIX = "rpsess_"


def generate_report_session_id(source_session_id: str) -> str:
    """Derived run id for one generation; the suffix is deterministic so an
    idempotent retry can prove a job really came from this request."""
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    suffix = stable_hash({"report_source_session_id": source_session_id}, length=6)
    return f"{REPORT_SESSION_PREFIX}{stamp}_{suffix}"


class ReportGenerationError(Exception):
    pass


class ReportRunBusyError(ReportGenerationError):
    def __init__(self, session_id: str, job_id: str) -> None:
        super().__init__(
            f"Session {session_id} has an active job ({job_id}); wait for it to finish "
            "before generating the report."
        )
        self.session_id = session_id
        self.job_id = job_id


class ReportNotGeneratableError(ReportGenerationError):
    def __init__(self, session_id: str, reason: str) -> None:
        super().__init__(f"Session {session_id} cannot generate a report: {reason}")
        self.session_id = session_id


class ReportGenerationService:
    def __init__(self, store: ArtifactStore, jobs: JobService) -> None:
        self._store = store
        self._jobs = jobs

    def generate(
        self,
        session_id: str,
        *,
        llm: str = "env",
        payload_policy: str | None = None,
        llm_env: dict[str, str] | None = None,
        idempotency_key: str | None = None,
    ) -> ReportGenerationStarted:
        project_id = self._project_for_run(session_id)
        execution_session_id = generate_report_session_id(session_id)
        idempotency_content = {
            "source_session_id": session_id,
            "llm": llm,
            "payload_policy": payload_policy,
        }
        # Same ordering as the relationship kinds: an idempotent replay must
        # answer before the busy guard, or a retried request would 409 against
        # the very job it already started.
        if idempotency_key:
            existing = self._store.find_by_idempotency_key(idempotency_key)
            if existing is not None:
                self._check_replay_matches(existing, session_id=session_id)
                self._jobs.assert_idempotent_replay(
                    existing,
                    request_scope=session_id,
                    project_id=project_id,
                    kind=REPORT_JOB_KIND,
                    content=idempotency_content,
                    env=llm_env if llm == "env" else None,
                )
                return self._replayed(session_id, existing)
        self._require_report_lane_free(project_id, session_id)
        self._require_generatable(project_id, session_id)
        regenerated = self._has_report(project_id, session_id)
        job = self._jobs.create_report_generate_job(
            execution_session_id,
            project_id=project_id,
            source_session_id=session_id,
            llm=llm,
            payload_policy=payload_policy,
            llm_env=llm_env,
            idempotency_key=idempotency_key,
            idempotency_content=idempotency_content,
        )
        return ReportGenerationStarted(
            session_id=session_id,
            execution_session_id=job.session_id,
            regenerated=regenerated,
            job=_to_created(job),
        )

    def _require_report_lane_free(self, project_id: str, session_id: str) -> None:
        """One report generation per source run at a time, and never while the
        run itself is still being analysed — two generations would race on the
        same report/report.md and the same MarkdownReport artifact."""
        active = self._store.find_active_job_for_lane(session_id)
        if active is not None:
            raise ReportRunBusyError(session_id, str(active["job_id"]))
        suffix = generate_report_session_id(session_id).rsplit("_", 1)[-1]
        for job in self._store.list_active_jobs():
            if (
                str(job["kind"]) == REPORT_JOB_KIND
                and str(job["project_id"]) == project_id
                and str(job["session_id"]).rsplit("_", 1)[-1] == suffix
            ):
                raise ReportRunBusyError(session_id, str(job["job_id"]))

    def _require_generatable(self, project_id: str, session_id: str) -> None:
        """A report is assembled from the run's evidence artifacts; a run that
        produced none has nothing to report on."""
        if not self._store.list_indexed_artifacts(
            project_id=project_id,
            session_id=session_id,
            artifact_types=(ArtifactType.DATASET_PROFILE,),
        ):
            raise ReportNotGeneratableError(
                session_id, "it has no dataset profiles to build a report from"
            )

    def _has_report(self, project_id: str, session_id: str) -> bool:
        return bool(
            self._store.latest_artifact_index_rows(
                project_id, session_id, ArtifactType.MARKDOWN_REPORT.value
            )
        )

    def _check_replay_matches(self, job_row: dict, *, session_id: str) -> None:
        """A replay is only legitimate for a report_generate job in this run's
        own project, derived from this very run (mirrors relationship discover)."""
        job_id = str(job_row["job_id"])
        run_row = self._store.get_session_index_row(session_id)
        expected_suffix = generate_report_session_id(session_id).rsplit("_", 1)[-1]
        if (
            run_row is None
            or str(job_row["kind"]) != REPORT_JOB_KIND
            or str(job_row["project_id"]) != str(run_row["project_id"])
            or not str(job_row["session_id"]).endswith(f"_{expected_suffix}")
        ):
            raise JobConflictError(
                job_id,
                f"Idempotency key already used by job {job_id} from a "
                "different project, kind or source run.",
            )

    def _replayed(self, session_id: str, job_row: dict) -> ReportGenerationStarted:
        job = self._jobs.get_job(str(job_row["job_id"]))
        return ReportGenerationStarted(
            session_id=session_id,
            execution_session_id=job.session_id,
            regenerated=False,
            job=_to_created(job),
        )

    def _project_for_run(self, session_id: str) -> str:
        if INTERNAL_SESSION_MARKER in session_id:
            raise SessionNotFoundError(session_id)
        row = self._store.get_session_index_row(session_id)
        if row is None:
            raise SessionNotFoundError(session_id)
        return str(row["project_id"])


def _to_created(job: JobStatus) -> JobCreated:
    return JobCreated(
        job_id=job.job_id,
        session_id=job.session_id,
        status=job.status,
        events_url=job.events_url,
    )
