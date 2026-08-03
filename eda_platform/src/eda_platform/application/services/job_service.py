"""Job use cases (§4.3/§7.3): create with idempotency, status, cooperative
cancel, and the SSE read path over the existing trace_events table."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

from eda_platform.application.dto import JobEvent, JobStatus
from eda_platform.application.ports import TERMINAL_JOB_STATUSES, JobBackend, JobCommand
from eda_platform.application.services.session_service import ProjectNotFoundError
from eda_platform.core.ids import validate_session_id
from eda_platform.core.store import ArtifactStore, SessionStorageDeletingError
from eda_platform.infrastructure.job_lifecycle import JobLifecycleRepository
from eda_platform.schemas.resource_metrics import EdaResourcePolicy
from eda_platform.schemas.sessions import TraceEvent

SUPPORTED_JOB_KINDS = frozenset(
    {
        "auto_eda",
        "question_exec",
        "skill_replay",
        "relationship_validate",
        "relationship_discover",
        "report_generate",
        "session_fork",
        "question_draft",
        "investigation_plan",
        "investigation_execute",
        "macro_loop",
        "synthesis_brief_create",
        "decision_report_generate",
        "cleaning_preview",
        "cleaning_apply",
        "dataset_distributions",
        "custom_chart",
        "exploration_run",
    }
)
TERMINAL_EVENT_TYPES = frozenset({"job.completed", "job.failed", "job.cancelled"})
EVENTS_PAGE_LIMIT = 500


class JobServiceError(Exception):
    pass


class JobNotFoundError(JobServiceError):
    def __init__(self, job_id: str) -> None:
        super().__init__(f"Job not found: {job_id}")
        self.job_id = job_id


class JobValidationError(JobServiceError):
    pass


class JobIdempotencyMismatchError(JobServiceError):
    error_code = "idempotency_key_reused"

    def __init__(self, job_id: str) -> None:
        super().__init__(
            "Idempotency key was already used with a different canonical request "
            f"(existing job: {job_id})."
        )
        self.job_id = job_id


class JobConflictError(JobServiceError):
    def __init__(self, job_id: str, message: str) -> None:
        super().__init__(message)
        self.job_id = job_id


class JobRunDeletingError(JobServiceError):
    def __init__(self, session_id: str) -> None:
        super().__init__(f"Session is being deleted: {session_id}")
        self.session_id = session_id


class EventsPage(NamedTuple):
    events: list[JobEvent]
    status: str
    cursor: int
    exhausted: bool
    """False when the underlying page was full — more rows may follow immediately."""


def _resolved_resource_policy(on_exceed: str) -> EdaResourcePolicy:
    env_fields = {
        "EDA_MAX_WORKING_SET_BYTES": "max_working_set_bytes",
        "EDA_MAX_INPUT_BYTES_TOTAL": "max_input_bytes_total",
        "EDA_MAX_SINGLE_INPUT_BYTES": "max_single_input_bytes",
        "EDA_MAX_COLUMNS_PER_DATASET": "max_columns_per_dataset",
        "EDA_MAX_ROWS_PER_DATASET": "max_rows_per_dataset",
        "EDA_PREFLIGHT_SAMPLE_ROWS": "sample_rows",
    }
    values: dict[str, Any] = {"on_exceed": on_exceed}
    for env_name, field_name in env_fields.items():
        raw = os.environ.get(env_name)
        if raw is None or not raw.strip():
            continue
        try:
            values[field_name] = int(raw)
        except ValueError as exc:
            raise JobValidationError(f"{env_name} must be an integer") from exc
    try:
        return EdaResourcePolicy.model_validate(values)
    except ValueError as exc:
        raise JobValidationError(f"Invalid Auto-EDA resource policy: {exc}") from exc


class JobService:
    def __init__(self, store: ArtifactStore, backend: JobBackend) -> None:
        self._store = store
        self._backend = backend
        self._lifecycle = JobLifecycleRepository(store)

    def create_job(
        self,
        session_id: str,
        *,
        kind: str,
        project_id: str = "default",
        datasets: list[str],
        business_context: str = "",
        ml_target_column: str | None = None,
        ml_time_column: str | None = None,
        generate_report: bool = True,
        dataset_workers: int = 1,
        resource_limit_action: str = "limited",
        llm: str = "env",
        payload_policy: str | None = None,
        llm_env: dict[str, str] | None = None,
        precleaning: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
        idempotency_content: dict[str, Any] | None = None,
        idempotency_scope: str | None = None,
    ) -> JobStatus:
        """``payload_policy`` and ``llm_env`` carry the caller's Settings choices
        into the worker; ``llm_env`` holds the API key and therefore goes to the
        child's environment, never into ``params_json``. ``precleaning`` is the
        opt-in pre-ingest clean; omitted or disabled, the worker ingests the
        uploaded files untouched."""
        if kind != "auto_eda":
            raise JobValidationError(f"Unsupported job kind: {kind}")
        if dataset_workers not in {1, 2}:
            raise JobValidationError("dataset_workers must be 1 or 2")
        if resource_limit_action not in {"limited", "reject"}:
            raise JobValidationError("resource_limit_action must be limited or reject")
        resource_policy = _resolved_resource_policy(resource_limit_action)
        return self._create_and_enqueue(
            session_id,
            kind=kind,
            project_id=project_id,
            idempotency_key=idempotency_key,
            env=llm_env if llm == "env" else None,
            idempotency_content=idempotency_content,
            request_scope=idempotency_scope or session_id,
            build_params=lambda resolved_project_id: {
                "dataset_paths": self._resolve_dataset_refs(resolved_project_id, datasets),
                "business_context": business_context,
                "ml_target_column": ml_target_column,
                "ml_time_column": ml_time_column,
                "generate_report": generate_report,
                "dataset_workers": dataset_workers,
                "resource_policy": resource_policy.model_dump(mode="json"),
                "llm": llm,
                "payload_policy": payload_policy,
                "precleaning": precleaning,
            },
        )

    def create_report_generate_job(
        self,
        session_id: str,
        *,
        project_id: str,
        source_session_id: str,
        llm: str = "env",
        payload_policy: str | None = None,
        llm_env: dict[str, str] | None = None,
        idempotency_key: str | None = None,
        idempotency_content: dict[str, Any] | None = None,
    ) -> JobStatus:
        """Queue an on-demand report generation onto a fresh derived run.

        The report artifacts land on ``source_session_id`` (the driver reloads that
        run and writes into it); the derived run only carries the job
        lifecycle, so a failed generation never flips the source run to failed.
        """
        return self._create_and_enqueue(
            session_id,
            kind="report_generate",
            project_id=project_id,
            lane_key=source_session_id,
            idempotency_key=idempotency_key,
            env=llm_env if llm == "env" else None,
            idempotency_content=idempotency_content,
            request_scope=source_session_id,
            build_params=lambda _resolved: {
                "source_session_id": source_session_id,
                "llm": llm,
                "payload_policy": payload_policy,
            },
        )

    def create_data_operation_job(
        self,
        session_id: str,
        *,
        kind: str,
        project_id: str,
        source_session_id: str,
        params: dict[str, Any],
        llm_env: dict[str, str] | None = None,
        idempotency_key: str | None = None,
        idempotency_content: dict[str, Any] | None = None,
    ) -> JobStatus:
        if kind not in {
            "cleaning_preview",
            "cleaning_apply",
            "dataset_distributions",
            "custom_chart",
        }:
            raise JobValidationError(f"Unsupported data operation kind: {kind}")
        return self._create_and_enqueue(
            session_id,
            kind=kind,
            project_id=project_id,
            lane_key=(
                "dop_lane_"
                + hashlib.sha256(f"{source_session_id}:{kind}".encode()).hexdigest()[:32]
            ),
            idempotency_key=idempotency_key,
            env=llm_env,
            idempotency_content=idempotency_content,
            request_scope=source_session_id,
            build_params=lambda _resolved: {
                "source_session_id": source_session_id,
                **params,
            },
        )

    def create_run_fork_job(
        self,
        session_id: str,
        *,
        project_id: str,
        source_session_id: str,
        decision_kind: str,
        ml_target_column: str | None = None,
        dataset_paths: list[str] | None = None,
        dataset_label: str = "",
        llm: str = "env",
        payload_policy: str | None = None,
        llm_env: dict[str, str] | None = None,
        idempotency_key: str | None = None,
        idempotency_content: dict[str, Any] | None = None,
    ) -> JobStatus:
        """Queue a what-if fork of ``source_session_id`` onto a lifecycle run.

        ``session_id`` is the ``fksess_*`` run that carries the job only: `fork_session`
        mints the forked analysis's own run id, which the worker reports back
        in a ``session.forked`` trace event.
        """
        return self._create_and_enqueue(
            session_id,
            kind="session_fork",
            project_id=project_id,
            lane_key=source_session_id,
            idempotency_key=idempotency_key,
            env=llm_env if llm == "env" else None,
            idempotency_content=idempotency_content,
            request_scope=source_session_id,
            build_params=lambda resolved_project_id: {
                "source_session_id": source_session_id,
                "decision_kind": decision_kind,
                "ml_target_column": ml_target_column,
                "dataset_paths": (
                    self._resolve_dataset_refs(resolved_project_id, dataset_paths)
                    if dataset_paths
                    else []
                ),
                "dataset_label": dataset_label,
                "llm": llm,
                "payload_policy": payload_policy,
            },
        )

    def create_question_exec_job(
        self,
        session_id: str,
        *,
        project_id: str,
        source_session_id: str,
        question_id: str,
        candidate_fingerprint: str,
        llm: str = "env",
        idempotency_key: str | None = None,
        idempotency_content: dict[str, Any] | None = None,
    ) -> JobStatus:
        """Queue a single-question execution job onto a fresh derived run.

        ``session_id`` is the pre-generated batch run id; the worker passes it to
        ``run_question_batch`` so artifacts land where the caller expects.
        ``candidate_fingerprint`` travels in the params so the worker can
        recompute it against the source run right before executing."""
        return self._create_and_enqueue(
            session_id,
            kind="question_exec",
            project_id=project_id,
            idempotency_key=idempotency_key,
            idempotency_content=idempotency_content,
            request_scope=source_session_id,
            build_params=lambda _resolved: {
                "source_session_id": source_session_id,
                "question_id": question_id,
                "candidate_fingerprint": candidate_fingerprint,
                "generate_report": False,
                "llm": llm,
            },
        )

    def create_question_draft_job(
        self,
        session_id: str,
        *,
        project_id: str,
        source_session_id: str,
        question: str,
        llm: str = "env",
        payload_policy: str | None = None,
        llm_env: dict[str, str] | None = None,
        idempotency_key: str | None = None,
        idempotency_content: dict[str, Any] | None = None,
    ) -> JobStatus:
        """Queue a free-text question-card draft onto a fresh derived run.

        The drafted candidate is appended to ``source_session_id``'s candidate set
        (the driver owns that write); ``session_id`` is the ``qdsess_*`` run that
        only carries the lifecycle, so a refused draft never fails the source
        run. ``question`` travels in the params because the approval bound that
        exact text — the worker must not re-read a prompt that may have changed.
        """
        return self._create_and_enqueue(
            session_id,
            kind="question_draft",
            project_id=project_id,
            lane_key=source_session_id,
            idempotency_key=idempotency_key,
            env=llm_env if llm == "env" else None,
            idempotency_content=idempotency_content,
            request_scope=source_session_id,
            build_params=lambda _resolved: {
                "source_session_id": source_session_id,
                "question": question,
                "llm": llm,
                "payload_policy": payload_policy,
            },
        )

    def create_investigation_plan_job(
        self,
        session_id: str,
        *,
        project_id: str,
        source_session_id: str,
        question_ids: list[str],
        deep: bool = False,
        idempotency_key: str | None = None,
        idempotency_content: dict[str, Any] | None = None,
    ) -> JobStatus:
        """Queue deterministic plan building onto a fresh derived run.

        The plans land on their own ``investigation_*`` run that the driver
        mints; ``session_id`` is the ``ipsess_*`` lifecycle run, so a failed build
        never flips the source run to failed. Building spends no model budget,
        which is why this kind carries no approval.
        """
        return self._create_and_enqueue(
            session_id,
            kind="investigation_plan",
            project_id=project_id,
            lane_key=source_session_id,
            idempotency_key=idempotency_key,
            idempotency_content=idempotency_content,
            request_scope=source_session_id,
            build_params=lambda _resolved: {
                "source_session_id": source_session_id,
                "question_ids": list(question_ids),
                "deep": deep,
            },
        )

    def create_investigation_execute_job(
        self,
        session_id: str,
        *,
        project_id: str,
        source_session_id: str,
        plan_session_id: str,
        plan_ids: list[str],
        plan_fingerprints: dict[str, str],
        llm: str = "env",
        payload_policy: str | None = None,
        llm_env: dict[str, str] | None = None,
        idempotency_key: str | None = None,
        idempotency_content: dict[str, Any] | None = None,
    ) -> JobStatus:
        """Queue execution of approved plans onto a fresh derived run.

        Findings and records land on ``plan_session_id``; ``session_id`` is the
        ``ixsess_*`` lifecycle run. ``plan_fingerprints`` lets the worker
        recompute what was approved right before executing.
        """
        return self._create_and_enqueue(
            session_id,
            kind="investigation_execute",
            project_id=project_id,
            lane_key=plan_session_id,
            idempotency_key=idempotency_key,
            env=llm_env if llm == "env" else None,
            idempotency_content=idempotency_content,
            request_scope=source_session_id,
            build_params=lambda _resolved: {
                "source_session_id": source_session_id,
                "plan_session_id": plan_session_id,
                "plan_ids": list(plan_ids),
                "plan_fingerprints": dict(plan_fingerprints),
                "llm": llm,
                "payload_policy": payload_policy,
            },
        )

    def create_macro_loop_job(
        self,
        session_id: str,
        *,
        project_id: str,
        source_session_id: str,
        plan_session_id: str,
        depth: int,
        llm: str = "env",
        payload_policy: str | None = None,
        llm_env: dict[str, str] | None = None,
        idempotency_key: str | None = None,
        idempotency_content: dict[str, Any] | None = None,
    ) -> JobStatus:
        """Queue the Ultra macro loop over an executed plan run.

        The ledger and follow-up artifacts land on ``plan_session_id`` and its own
        internal funnel runs; ``session_id`` is the ``mlsess_*`` lifecycle run.
        """
        return self._create_and_enqueue(
            session_id,
            kind="macro_loop",
            project_id=project_id,
            lane_key=plan_session_id,
            idempotency_key=idempotency_key,
            env=llm_env if llm == "env" else None,
            idempotency_content=idempotency_content,
            request_scope=source_session_id,
            build_params=lambda _resolved: {
                "source_session_id": source_session_id,
                "plan_session_id": plan_session_id,
                "depth": depth,
                "llm": llm,
                "payload_policy": payload_policy,
            },
        )

    def create_skill_replay_job(
        self,
        session_id: str,
        *,
        project_id: str,
        source_session_id: str,
        skill_id: str,
        skill: dict[str, Any],
        dataset_ids: list[str],
        idempotency_key: str | None = None,
        idempotency_content: dict[str, Any] | None = None,
    ) -> JobStatus:
        """Queue a deterministic skill replay onto a fresh derived run.

        ``skill`` is the whole approved AnalysisSkill, so the worker replays
        the reviewed content rather than re-reading a library that may have
        changed since the approval."""
        return self._create_and_enqueue(
            session_id,
            kind="skill_replay",
            project_id=project_id,
            idempotency_key=idempotency_key,
            idempotency_content=idempotency_content,
            request_scope=source_session_id,
            build_params=lambda _resolved: {
                "source_session_id": source_session_id,
                "skill_id": skill_id,
                "skill": skill,
                "dataset_ids": list(dataset_ids),
            },
        )

    def create_relationship_validate_job(
        self,
        session_id: str,
        *,
        project_id: str,
        source_session_id: str,
        relationship_id: str,
        pair_label: str,
        candidate_fingerprint: str,
        idempotency_key: str | None = None,
        idempotency_content: dict[str, Any] | None = None,
    ) -> JobStatus:
        """Queue one full relationship validation onto a fresh derived run.

        The validation artifact itself lands on ``source_session_id`` (the driver
        owns that); the derived run only carries the job lifecycle, so a failed
        validation never rewrites the source run's status."""
        return self._create_and_enqueue(
            session_id,
            kind="relationship_validate",
            project_id=project_id,
            lane_key=source_session_id,
            idempotency_key=idempotency_key,
            idempotency_content=idempotency_content,
            request_scope=source_session_id,
            build_params=lambda _resolved: {
                "source_session_id": source_session_id,
                "relationship_id": relationship_id,
                "pair_label": pair_label,
                "candidate_fingerprint": candidate_fingerprint,
            },
        )

    def create_relationship_discover_job(
        self,
        session_id: str,
        *,
        project_id: str,
        source_session_id: str,
        rerun: bool = False,
        idempotency_key: str | None = None,
        idempotency_content: dict[str, Any] | None = None,
    ) -> JobStatus:
        """Queue on-demand relationship discovery onto a fresh derived run.

        The candidate artifacts land on ``source_session_id`` (the driver owns
        that); ``rerun`` tells the worker to rescan a run that already has a
        candidate set instead of returning the existing one unchanged."""
        return self._create_and_enqueue(
            session_id,
            kind="relationship_discover",
            project_id=project_id,
            lane_key=source_session_id,
            idempotency_key=idempotency_key,
            idempotency_content=idempotency_content,
            request_scope=source_session_id,
            build_params=lambda _resolved: {
                "source_session_id": source_session_id,
                "force": rerun,
            },
        )

    def create_synthesis_brief_job(
        self,
        session_id: str,
        *,
        project_id: str,
        source_session_id: str,
        finding_artifact_ids: list[str],
        finding_session_ids: dict[str, str],
        business_context: str = "",
        idempotency_key: str | None = None,
        idempotency_content: dict[str, Any] | None = None,
    ) -> JobStatus:
        """Queue decision-story drafting onto a fresh derived lifecycle run.

        No LLM env: the synthesis orchestrator is deterministic. The brief
        artifact lands on the synthesis run the driver mints, not on
        ``session_id``, which only carries the job lifecycle."""
        return self._create_and_enqueue(
            session_id,
            kind="synthesis_brief_create",
            project_id=project_id,
            lane_key=source_session_id,
            idempotency_key=idempotency_key,
            idempotency_content=idempotency_content,
            request_scope=source_session_id,
            build_params=lambda _resolved: {
                "source_session_id": source_session_id,
                "finding_artifact_ids": list(finding_artifact_ids),
                "finding_session_ids": dict(finding_session_ids),
                "business_context": business_context,
            },
        )

    def create_decision_report_job(
        self,
        session_id: str,
        *,
        project_id: str,
        source_session_id: str,
        brief_artifact_id: str,
        brief_session_id: str,
        idempotency_key: str | None = None,
        idempotency_content: dict[str, Any] | None = None,
    ) -> JobStatus:
        """Queue decision-report generation onto a fresh derived lifecycle run.

        No LLM env on purpose: the driver is called with ``llm=None``, which is
        the product rule for decision reports: generating a
        report never spends tokens. The report artifact inherits the brief's
        run, so it lands on the synthesis run, not on ``session_id``."""
        return self._create_and_enqueue(
            session_id,
            kind="decision_report_generate",
            project_id=project_id,
            lane_key=source_session_id,
            idempotency_key=idempotency_key,
            idempotency_content=idempotency_content,
            request_scope=source_session_id,
            build_params=lambda _resolved: {
                "source_session_id": source_session_id,
                "brief_artifact_id": brief_artifact_id,
                "brief_session_id": brief_session_id,
            },
        )

    def create_exploration_job(
        self,
        session_id: str,
        *,
        project_id: str,
        source_session_id: str,
        exploration_id: str,
        policy: dict[str, Any],
        data_state_witness: str,
        code_fingerprint: str,
        release_certificate_digest: str,
        provider: str,
        payload_policy: str | None,
        llm_env: dict[str, str] | None,
        operation: str,
        idempotency_key: str | None,
        idempotency_content: dict[str, Any],
    ) -> JobStatus:
        """Queue one E4b worker attempt behind the exploration's logical lane.

        The exploration journal remains the state authority. This derived run
        owns only the normal job lifecycle, while every execution-affecting
        identity needed by the worker is frozen in non-secret params.
        """
        if operation not in {"start", "resume"}:
            raise JobValidationError(f"Unsupported exploration operation: {operation}")
        return self._create_and_enqueue(
            session_id,
            kind="exploration_run",
            project_id=project_id,
            lane_key=source_session_id,
            idempotency_key=idempotency_key,
            env=llm_env,
            idempotency_content=idempotency_content,
            request_scope=source_session_id,
            build_params=lambda _resolved: {
                "source_session_id": source_session_id,
                "exploration_id": exploration_id,
                "policy": policy,
                "data_state_witness": data_state_witness,
                "code_fingerprint": code_fingerprint,
                "release_certificate_digest": release_certificate_digest,
                "provider": provider,
                "payload_policy": payload_policy,
                "operation": operation,
            },
        )

    def _create_and_enqueue(
        self,
        session_id: str,
        *,
        kind: str,
        project_id: str,
        lane_key: str | None = None,
        idempotency_key: str | None,
        build_params: Callable[[str], dict[str, Any]],
        request_scope: str,
        env: dict[str, str] | None = None,
        idempotency_content: dict[str, Any] | None = None,
    ) -> JobStatus:
        """Shared create path. Guard order: idempotent replay → project
        resolution → atomic row/lane reservation → queued event → enqueue.

        Both the idempotency key and active logical lane are SQLite unique
        invariants. Integrity races are mapped back to replay or typed 409;
        no process-local check is relied on for correctness.
        """
        try:
            validate_session_id(session_id)
        except ValueError as exc:
            raise JobValidationError(str(exc)) from exc
        run_row = self._store.get_session_index_row(session_id)
        if run_row is not None:
            project_id = str(run_row["project_id"])
        elif not self._store.project_exists(project_id):
            raise ProjectNotFoundError(project_id)
        params = build_params(project_id)
        request_digest = _canonical_request_digest(
            request_scope=request_scope,
            project_id=project_id,
            kind=kind,
            content=params if idempotency_content is None else idempotency_content,
            env=env,
        )
        if idempotency_key:
            existing = self._store.find_by_idempotency_key(idempotency_key)
            if existing is not None:
                return _to_status(
                    _check_idempotent_match(
                        existing,
                        request_scope=request_scope,
                        project_id=project_id,
                        kind=kind,
                        request_digest=request_digest,
                    )
                )
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        params_json = json.dumps(params, ensure_ascii=False)
        try:
            job = self._lifecycle.create_queued_job(
                job_id=job_id,
                session_id=session_id,
                project_id=project_id,
                kind=kind,
                params_json=params_json,
                idempotency_key=idempotency_key,
                lane_key=lane_key or session_id,
                request_digest=request_digest,
                request_scope=request_scope,
            )
        except SessionStorageDeletingError as exc:
            raise JobRunDeletingError(exc.session_id) from None
        except sqlite3.IntegrityError as exc:
            # A concurrent idempotent request wins as a replay.
            existing = (
                self._store.find_by_idempotency_key(idempotency_key) if idempotency_key else None
            )
            if existing is not None:
                return _to_status(
                    _check_idempotent_match(
                        existing,
                        request_scope=request_scope,
                        project_id=project_id,
                        kind=kind,
                        request_digest=request_digest,
                    )
                )
            # Otherwise the DB rejected a second active owner of the lane.
            if "jobs.lane_key" in str(exc):
                active = self._store.find_active_job_for_lane(lane_key or session_id)
                # The winning job can settle between our failed INSERT and
                # this lookup. It still won this reservation attempt, so keep
                # the response a typed 409 instead of leaking SQLite as 500.
                winner = active or self._store.latest_job_for_lane(lane_key or session_id)
            else:
                winner = None
            if winner is not None:
                raise JobConflictError(
                    str(winner["job_id"]),
                    f"Run {lane_key or session_id} already has an active job: {winner['job_id']}",
                ) from None
            raise
        try:
            # The DB row and DB trace are authoritative. Filesystem setup is
            # protected by the same compensation boundary as dispatch, while
            # JSONL materialization remains recoverable and best-effort.
            self._store.start_session(project_id, session_id)
            with suppress(Exception):
                self._lifecycle.materialize_trace(job_id)
            self._backend.enqueue(
                JobCommand(
                    job_id=job_id,
                    session_id=session_id,
                    project_id=project_id,
                    kind=kind,
                    params_json=params_json,
                    env=env,
                )
            )
        except Exception as exc:
            # Never leave a queued row nothing will ever pick up (review F4).
            self._lifecycle.fail_active(
                job_id,
                error_code="enqueue_failed",
                error_message=str(exc)[:500],
                clear_idempotency=bool(idempotency_key),
            )
            with suppress(Exception):
                self._lifecycle.materialize_trace(job_id)
            raise
        return _to_status(job)

    def assert_idempotent_replay(
        self,
        existing: dict,
        *,
        request_scope: str,
        project_id: str,
        kind: str,
        content: dict[str, Any],
        env: dict[str, str] | None = None,
    ) -> None:
        """Apply the same canonical-content guard to producer fast paths.

        Approval-backed producers must inspect a consumed token before replay,
        but they still delegate request binding here so no wrapper invents a
        weaker interpretation of an idempotency key.
        """
        _check_idempotent_match(
            existing,
            request_scope=request_scope,
            project_id=project_id,
            kind=kind,
            request_digest=_canonical_request_digest(
                request_scope=request_scope,
                project_id=project_id,
                kind=kind,
                content=content,
                env=env,
            ),
        )

    def get_job(self, job_id: str) -> JobStatus:
        return _to_status(self._require_job(job_id))

    def cancel_job(self, job_id: str) -> JobStatus:
        job = self._require_job(job_id)
        if job["status"] in TERMINAL_JOB_STATUSES:
            return _to_status(job)
        job = self._lifecycle.request_cancel(job_id)
        with suppress(Exception):
            self._lifecycle.materialize_trace(job_id)
        if str(job["status"]) in TERMINAL_JOB_STATUSES:
            return _to_status(job)
        self._backend.cancel(job_id)
        job = self._require_job(job_id)
        return _to_status(job)

    def events_after(self, job_id: str, after_id: int) -> EventsPage:
        """New trace events with durable ownership by this exact job."""
        job = self._require_job(job_id)
        cursor = max(0, after_id)
        rows = self._store.list_job_trace_rows_after(
            job_id=job_id,
            after_id=cursor,
            limit=EVENTS_PAGE_LIMIT,
        )
        events: list[JobEvent] = []
        for row_id, payload in rows:
            cursor = row_id
            event = _to_event(row_id, payload, job_id=job_id, session_id=str(job["session_id"]))
            events.append(event)
        return EventsPage(
            events=events,
            status=str(job["status"]),
            cursor=cursor,
            exhausted=len(rows) < EVENTS_PAGE_LIMIT,
        )

    def _require_job(self, job_id: str) -> dict:
        job = self._store.get_job(job_id)
        if job is None:
            raise JobNotFoundError(job_id)
        return job

    def _emit_job_event(self, job: dict, event_type: str) -> None:
        self._store.append_trace(
            str(job["project_id"]),
            TraceEvent(
                session_id=str(job["session_id"]),
                event_type=event_type,
                name=str(job["job_id"]),
                job_id=str(job["job_id"]),
                job_generation=int(job.get("launch_attempt") or 0),
                finished_at=datetime.now(UTC),
                summary={"job_id": job["job_id"], "status": job["status"]},
            ),
        )

    def _resolve_dataset_refs(self, project_id: str, refs: list[str]) -> list[str]:
        if not refs:
            raise JobValidationError("At least one dataset reference is required.")
        root = self._store.root.resolve()
        project_root = self._store.project_dir(project_id).resolve()
        resolved: list[str] = []
        for ref in refs:
            path = self._path_for_ref(project_id, ref, root, project_root)
            if path is None:
                raise JobValidationError(f"Unknown dataset reference: {ref}")
            resolved.append(str(path.resolve().relative_to(root)))
        return resolved

    def _path_for_ref(
        self, project_id: str, ref: str, root: Path, project_root: Path
    ) -> Path | None:
        # dataset_id form first: the canonical uploads layout owns these names.
        if "/" not in ref and "\\" not in ref:
            version_dir = self._store.project_dir(project_id) / "uploads" / ref / "v1"
            if not version_dir.is_symlink() and version_dir.is_dir():
                for candidate in sorted(version_dir.glob("*.csv")):
                    if _is_safe_project_csv(candidate, project_root):
                        return candidate
        # Legacy seed/ paths live directly under the workspace. Preserve those,
        # but once a path enters projects/ it must stay in this project.
        if Path(ref).is_absolute():
            return None
        candidate = root / ref
        try:
            resolved = candidate.resolve()
            projects_root = (root / "projects").resolve()
            inside_workspace = resolved != root and resolved.is_relative_to(root)
            crosses_project = resolved.is_relative_to(
                projects_root
            ) and not resolved.is_relative_to(project_root)
        except OSError:
            return None
        if (
            not inside_workspace
            or crosses_project
            or candidate.suffix.lower() != ".csv"
            or not candidate.is_file()
        ):
            return None
        return candidate


def _is_safe_project_csv(candidate: Path, project_root: Path) -> bool:
    try:
        resolved = candidate.resolve()
        return (
            resolved != project_root
            and resolved.is_relative_to(project_root)
            and candidate.suffix.lower() == ".csv"
            and candidate.is_file()
        )
    except OSError:
        return False


def reap_orphan_jobs(store: ArtifactStore) -> int:
    """Fail non-terminal jobs whose worker process is gone (review F4).

    Startup-only sweep: with no enqueue in flight, any active row without a
    live pid can never make progress again.
    """
    lifecycle = JobLifecycleRepository(store)
    active_before = list(store.list_active_jobs())
    count = lifecycle.recover_startup(fail_queued=True)
    for job in active_before:
        with suppress(Exception):
            lifecycle.materialize_trace(str(job["job_id"]))
    return count


def recover_job_lifecycle(store: ArtifactStore, backend: JobBackend) -> int:
    """Recover durable jobs before any run-deletion operation resumes."""
    lifecycle = JobLifecycleRepository(store)
    active_before = list(store.list_active_jobs())
    recovered = lifecycle.recover_startup()
    for job in active_before:
        with suppress(Exception):
            lifecycle.materialize_trace(str(job["job_id"]))
    for job in list(store.list_active_jobs()):
        status = str(job["status"])
        if status == "cancelling":
            resume_cancel = getattr(backend, "resume_cancel", None)
            if callable(resume_cancel):
                resume_cancel(str(job["job_id"]))
            with suppress(Exception):
                lifecycle.materialize_trace(str(job["job_id"]))
            continue
        if status != "queued":
            with suppress(Exception):
                lifecycle.materialize_trace(str(job["job_id"]))
            continue
        params_json = lifecycle.params_json(str(job["job_id"]))
        if params_json is None:
            recovered += int(
                lifecycle.fail_active(
                    str(job["job_id"]),
                    error_code="startup_params_missing",
                    error_message="Durable queued job has no persisted parameters.",
                    clear_idempotency=True,
                )
            )
            continue
        try:
            backend.enqueue(
                JobCommand(
                    job_id=str(job["job_id"]),
                    session_id=str(job["session_id"]),
                    project_id=str(job["project_id"]),
                    kind=str(job["kind"]),
                    params_json=params_json,
                )
            )
            recovered += 1
        except Exception:
            # The backend owns launch-failure terminalization.
            recovered += 1
        with suppress(Exception):
            lifecycle.materialize_trace(str(job["job_id"]))
    return recovered


def _canonical_request_digest(
    *,
    request_scope: str,
    project_id: str,
    kind: str,
    content: dict[str, Any],
    env: dict[str, str] | None,
) -> str:
    """Hash canonical semantic input without persisting raw secret values."""
    secret_digest = None
    if env:
        secret_json = json.dumps(
            env,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        secret_digest = hashlib.sha256(secret_json.encode("utf-8")).hexdigest()
    canonical = json.dumps(
        {
            "version": 2,
            "operation": kind,
            "project_id": project_id,
            "request_scope": request_scope,
            "content": content,
            "secret_env_digest": secret_digest,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _check_idempotent_match(
    existing: dict,
    *,
    request_scope: str,
    project_id: str,
    kind: str,
    request_digest: str,
) -> dict:
    """Same key replays only the exact persisted canonical request."""
    if (
        str(existing["project_id"]) != project_id
        or str(existing["kind"]) != kind
        or not existing.get("request_scope")
        or str(existing["request_scope"]) != request_scope
        or not existing.get("request_digest")
        or str(existing["request_digest"]) != request_digest
    ):
        raise JobIdempotencyMismatchError(str(existing["job_id"]))
    return existing


def events_url_for(job_id: str) -> str:
    return f"/api/v1/jobs/{job_id}/events"


def _to_status(job: dict) -> JobStatus:
    return JobStatus(
        job_id=job["job_id"],
        session_id=job["session_id"],
        project_id=job["project_id"],
        kind=job["kind"],
        status=job["status"],
        cancel_requested=bool(job["cancel_requested"]),
        created_at=_parse_datetime(job.get("created_at")),
        started_at=_parse_datetime(job.get("started_at")),
        finished_at=_parse_datetime(job.get("finished_at")),
        error_code=job.get("error_code"),
        error_message=job.get("error_message"),
        events_url=events_url_for(str(job["job_id"])),
    )


def _to_event(row_id: int, payload: str, *, job_id: str, session_id: str) -> JobEvent:
    event_type = "unknown"
    name = ""
    timestamp = None
    summary: dict[str, Any] = {}
    with suppress(ValueError):
        parsed = TraceEvent.model_validate_json(payload)
        event_type = parsed.event_type
        name = parsed.name
        timestamp = parsed.finished_at or parsed.started_at
        summary = parsed.summary
    return JobEvent(
        event_id=row_id,
        job_id=job_id,
        session_id=session_id,
        type=event_type,
        name=name,
        timestamp=timestamp,
        summary=summary,
    )


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
