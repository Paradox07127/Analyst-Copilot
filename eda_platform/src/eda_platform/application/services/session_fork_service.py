"""What-if forks (§10.3 Compare): re-run a completed run with exactly one
decision varied.

The existing `drivers.session_fork.fork_session` mints the forked analysis's own run id
(it delegates to `run_auto_eda` with `session_id=None`), so the job cannot run on
it: this queues a `run_fork` job on a lifecycle-only `fksess_*` run, and the
worker reports the forked run id back in a `session.forked` trace event once the
driver has produced it.
"""

from __future__ import annotations

from contextlib import suppress
from datetime import UTC, datetime

from eda_platform.application.dto import JobCreated, JobStatus, SessionForkStarted
from eda_platform.application.services.job_service import JobConflictError, JobService
from eda_platform.application.services.session_service import SessionNotFoundError
from eda_platform.core.ids import INTERNAL_SESSION_MARKER, stable_hash
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import ArtifactType

FORK_JOB_KIND = "session_fork"
FORK_SESSION_PREFIX = "fksess_"
DECISION_KINDS = ("ml_target", "dataset")


def generate_fork_session_id(source_session_id: str) -> str:
    """Lifecycle run id for one fork; the suffix is deterministic so an
    idempotent retry can prove a job really came from this request."""
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    suffix = stable_hash({"fork_source_session_id": source_session_id}, length=6)
    return f"{FORK_SESSION_PREFIX}{stamp}_{suffix}"


class SessionForkError(Exception):
    pass


class SessionForkBusyError(SessionForkError):
    def __init__(self, session_id: str, job_id: str) -> None:
        super().__init__(
            f"Run {session_id} has an active job ({job_id}); wait for it to finish "
            "before forking a variant."
        )
        self.session_id = session_id
        self.job_id = job_id


class SessionForkNotForkableError(SessionForkError):
    def __init__(self, session_id: str, reason: str) -> None:
        super().__init__(f"Run {session_id} cannot be forked: {reason}")
        self.session_id = session_id


class SessionForkValidationError(SessionForkError):
    pass


class SessionForkService:
    def __init__(self, store: ArtifactStore, jobs: JobService) -> None:
        self._store = store
        self._jobs = jobs

    def fork(
        self,
        session_id: str,
        *,
        decision_kind: str,
        ml_target_column: str | None = None,
        datasets: list[str] | None = None,
        llm: str = "env",
        payload_policy: str | None = None,
        llm_env: dict[str, str] | None = None,
        idempotency_key: str | None = None,
    ) -> SessionForkStarted:
        project_id = self._project_for_run(session_id)
        execution_session_id = generate_fork_session_id(session_id)
        chosen = list(datasets or [])
        idempotency_content = {
            "source_session_id": session_id,
            "decision_kind": decision_kind,
            "ml_target_column": ml_target_column or None,
            "datasets": chosen,
            "llm": llm,
            "payload_policy": payload_policy,
        }
        # Same ordering as the other on-demand kinds: an idempotent replay must
        # answer before the busy guard, or a retry would 409 against its own job.
        if idempotency_key:
            existing = self._store.find_by_idempotency_key(idempotency_key)
            if existing is not None:
                self._check_replay_matches(existing, session_id=session_id)
                self._jobs.assert_idempotent_replay(
                    existing,
                    request_scope=session_id,
                    project_id=project_id,
                    kind=FORK_JOB_KIND,
                    content=idempotency_content,
                    env=llm_env if llm == "env" else None,
                )
                return self._replayed(session_id, existing)
        if decision_kind not in DECISION_KINDS:
            raise SessionForkValidationError(
                f"Unsupported fork decision '{decision_kind}'; expected one of "
                + ", ".join(DECISION_KINDS)
            )
        if decision_kind == "dataset" and not chosen:
            raise SessionForkValidationError(
                "A dataset fork needs at least one dataset to re-run on."
            )
        self._require_fork_lane_free(project_id, session_id)
        self._require_forkable(project_id, session_id)
        if decision_kind == "dataset":
            chosen = self._require_source_datasets(project_id, session_id, chosen)
        label = self._dataset_label(project_id, session_id, chosen)
        job = self._jobs.create_run_fork_job(
            execution_session_id,
            project_id=project_id,
            source_session_id=session_id,
            decision_kind=decision_kind,
            ml_target_column=ml_target_column or None,
            dataset_paths=chosen if decision_kind == "dataset" else None,
            dataset_label=label,
            llm=llm,
            payload_policy=payload_policy,
            llm_env=llm_env,
            idempotency_key=idempotency_key,
            idempotency_content=idempotency_content,
        )
        return SessionForkStarted(
            session_id=session_id,
            execution_session_id=job.session_id,
            decision=(
                f"dataset → {label or 'new input'}"
                if decision_kind == "dataset"
                else f"ML target → {ml_target_column or 'none'}"
            ),
            job=_to_created(job),
        )

    def _require_fork_lane_free(self, project_id: str, session_id: str) -> None:
        """One fork per source run at a time, and never while the run itself is
        still being analysed: each fork re-reads the source run's datasets and
        a second one would only duplicate an expensive re-run."""
        active = self._store.find_active_job_for_lane(session_id)
        if active is not None:
            raise SessionForkBusyError(session_id, str(active["job_id"]))
        suffix = generate_fork_session_id(session_id).rsplit("_", 1)[-1]
        for job in self._store.list_active_jobs():
            if (
                str(job["kind"]) == FORK_JOB_KIND
                and str(job["project_id"]) == project_id
                and str(job["session_id"]).rsplit("_", 1)[-1] == suffix
            ):
                raise SessionForkBusyError(session_id, str(job["job_id"]))

    def _require_forkable(self, project_id: str, session_id: str) -> None:
        """`fork_session` re-runs the parent's ingested files, so a run whose
        datasets were never profiled has nothing to fork from."""
        if not self._store.list_indexed_artifacts(
            project_id=project_id,
            session_id=session_id,
            artifact_types=(ArtifactType.DATASET_PROFILE,),
        ):
            raise SessionForkNotForkableError(
                session_id, "its source datasets are not available to re-run"
            )

    def _dataset_label(self, project_id: str, session_id: str, refs: list[str]) -> str:
        """Human-facing name for the chosen input, resolved from the run's own
        dataset profiles so the label never echoes a server path."""
        if not refs:
            return ""
        names: list[str] = []
        for artifact in self._store.list_indexed_artifacts(
            project_id=project_id,
            session_id=session_id,
            artifact_types=(ArtifactType.DATASET_PROFILE,),
        ):
            dataset_id = artifact.payload.get("dataset_id")
            name = artifact.payload.get("name")
            if isinstance(dataset_id, str) and dataset_id in refs and isinstance(name, str):
                names.append(name)
        return ", ".join(dict.fromkeys(names))

    def _require_source_datasets(
        self, project_id: str, session_id: str, refs: list[str]
    ) -> list[str]:
        """Accept only dataset ids actually profiled by the source run."""
        allowed = {
            dataset_id
            for artifact in self._store.list_indexed_artifacts(
                project_id=project_id,
                session_id=session_id,
                artifact_types=(ArtifactType.DATASET_PROFILE,),
            )
            if isinstance((dataset_id := artifact.payload.get("dataset_id")), str)
        }
        unknown = [ref for ref in refs if ref not in allowed]
        if unknown:
            raise SessionForkValidationError(
                "Dataset fork selections must come from the source run: "
                + ", ".join(unknown)
            )
        return list(dict.fromkeys(refs))

    def _check_replay_matches(self, job_row: dict, *, session_id: str) -> None:
        """A replay is only legitimate for a run_fork job in this run's own
        project, derived from this very run (mirrors relationship discover)."""
        job_id = str(job_row["job_id"])
        run_row = self._store.get_session_index_row(session_id)
        expected_suffix = generate_fork_session_id(session_id).rsplit("_", 1)[-1]
        if (
            run_row is None
            or str(job_row["kind"]) != FORK_JOB_KIND
            or str(job_row["project_id"]) != str(run_row["project_id"])
            or not str(job_row["session_id"]).endswith(f"_{expected_suffix}")
        ):
            raise JobConflictError(
                job_id,
                f"Idempotency key already used by job {job_id} from a "
                "different project, kind or source run.",
            )

    def _replayed(self, session_id: str, job_row: dict) -> SessionForkStarted:
        job = self._jobs.get_job(str(job_row["job_id"]))
        return SessionForkStarted(
            session_id=session_id,
            execution_session_id=job.session_id,
            decision=self._replayed_decision(job),
            job=_to_created(job),
        )

    def _replayed_decision(self, job: JobStatus) -> str:
        """The forked run's own manifest title carries the decision summary once
        the worker has written it; before that a neutral label is honest."""
        with suppress(OSError, ValueError):
            manifest = self._store.read_manifest(job.project_id, job.session_id)
            if manifest is not None and manifest.title:
                return manifest.title
        return "fork in progress"

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
