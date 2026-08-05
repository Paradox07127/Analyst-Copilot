"""Child-process job entry point.

Runs in its own detached process (``python -m eda_platform.worker.runner``),
so it owns the whole job lifecycle against SQLite: status transitions, run
status, and job.* trace events. It receives only primitive argv and re-reads
the job row for everything else. Tracebacks go to logging, never into the DB.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from time import perf_counter
from typing import Any, cast, get_args

from eda_platform.application.ports import TERMINAL_JOB_STATUSES
from eda_platform.core.budget import SessionBudgetPolicy
from eda_platform.core.cancellation import (
    CancellationError,
    CancellationToken,
    DurableCancellationRecord,
    StorageBackedCancellationToken,
    cancellation_scope,
    current_cancellation_token,
)
from eda_platform.core.config import require_absolute_workspace
from eda_platform.core.env import (
    API_KEY_ENV_VARS,
    load_llm_settings_from_env_file,
    load_report_llm_settings_from_env_file,
)
from eda_platform.core.kernel import SessionCancelled
from eda_platform.core.llm import (
    CancellableLLMClient,
    LLMClient,
    OfflineLLMClient,
    create_llm_client,
)
from eda_platform.core.store import ArtifactStore
from eda_platform.core.trace_correlation import trace_job_scope
from eda_platform.infrastructure.job_lifecycle import (
    Heartbeat,
    JobLifecycleRepository,
    LaunchClaim,
)
from eda_platform.infrastructure.launch_gate import (
    START_ACK_TIMEOUT_SECONDS,
    acknowledge_and_wait,
)
from eda_platform.schemas.sessions import TraceEvent
from eda_platform.tools.evidence import PayloadPolicy

logger = logging.getLogger(__name__)

__all__ = ["TERMINAL_JOB_STATUSES", "emit_job_event", "main", "run_job"]

CancelCheck = Callable[[], bool]


def emit_job_event(
    store: ArtifactStore,
    job: dict,
    event_type: str,
    summary: dict[str, Any] | None = None,
) -> None:
    payload = {"job_id": job["job_id"], "status": job.get("status")}
    if summary:
        payload.update(summary)
    store.append_trace(
        job["project_id"],
        TraceEvent(
            session_id=job["session_id"],
            event_type=event_type,
            name=str(job["job_id"]),
            finished_at=datetime.now(UTC),
            summary=payload,
        ),
    )


def run_job(
    workspace: str,
    job_id: str,
    *,
    launch_token: str,
    launch_attempt: int,
) -> None:
    workspace = str(require_absolute_workspace(workspace))
    store = ArtifactStore(workspace)
    lifecycle = JobLifecycleRepository(store)
    claim = LaunchClaim(job_id=job_id, token=launch_token, attempt=launch_attempt)
    job = store.get_job(job_id)
    if job is None:
        logger.error("Job %s not found in %s", job_id, workspace)
        return
    job = lifecycle.child_start(claim)
    if job is None:
        with suppress(Exception):
            lifecycle.materialize_trace(job_id)
        return
    lifecycle.materialize_trace(job_id)
    params_json = lifecycle.params_json(job_id)
    if params_json is None:
        lifecycle.finish(
            claim,
            "failed",
            error_code="durable_params_missing",
            error_message="Worker claim has no durable params_json.",
        )
        with suppress(Exception):
            lifecycle.materialize_trace(job_id)
        return
    heartbeat = Heartbeat(lifecycle, claim)
    heartbeat.__enter__()
    token = _job_cancellation_token(store, lifecycle, claim)

    def cancel_check() -> bool:
        try:
            token.checkpoint()
        except CancellationError as exc:
            raise SessionCancelled(str(exc)) from exc
        return False

    try:
        with trace_job_scope(claim.job_id, claim.attempt), cancellation_scope(token):
            params = json.loads(params_json)
            cancel_check()
            if job["kind"] == "auto_eda":
                handler = _run_auto_eda_job
            elif job["kind"] == "question_exec":
                handler = _run_question_exec_job
            elif job["kind"] == "skill_replay":
                handler = _run_skill_replay_job
            elif job["kind"] == "relationship_validate":
                handler = _run_relationship_validate_job
            elif job["kind"] == "relationship_discover":
                handler = _run_relationship_discover_job
            elif job["kind"] == "report_generate":
                handler = _run_report_generate_job
            elif job["kind"] == "session_fork":
                handler = _run_run_fork_job
            elif job["kind"] == "question_draft":
                handler = _run_question_draft_job
            elif job["kind"] == "investigation_plan":
                handler = _run_investigation_plan_job
            elif job["kind"] == "investigation_execute":
                handler = _run_investigation_execute_job
            elif job["kind"] == "macro_loop":
                handler = _run_macro_loop_job
            elif job["kind"] == "synthesis_brief_create":
                handler = _run_synthesis_brief_job
            elif job["kind"] == "decision_report_generate":
                handler = _run_decision_report_job
            elif job["kind"] == "cleaning_preview":
                handler = _run_cleaning_preview_job
            elif job["kind"] == "cleaning_apply":
                handler = _run_cleaning_apply_job
            elif job["kind"] == "dataset_distributions":
                handler = _run_dataset_distributions_job
            elif job["kind"] == "custom_chart":
                handler = _run_custom_chart_job
            elif job["kind"] == "exploration_run":
                handler = _run_exploration_job
            else:
                raise ValueError(f"Unsupported job kind: {job['kind']}")
            _invoke_job_handler(
                handler,
                store,
                workspace,
                job,
                params,
                cancel_check=cancel_check,
            )
            cancel_check()
    except (SessionCancelled, CancellationError):
        _finish(store, job, "cancelled", lifecycle=lifecycle, claim=claim)
    except Exception as exc:
        logger.exception("Job %s failed", job_id)
        _finish(
            store,
            job,
            "failed",
            error_code=_durable_error_code(exc),
            error_message=_sanitize_error(str(exc), workspace)[:500],
            lifecycle=lifecycle,
            claim=claim,
        )
    else:
        _finish(store, job, "completed", lifecycle=lifecycle, claim=claim)
    finally:
        heartbeat.__exit__(None, None, None)


def _run_started_at(store: ArtifactStore, project_id: str, session_id: str) -> datetime | None:
    raw = store.earliest_trace_started_at(project_id=project_id, session_id=session_id)
    return None if raw is None else datetime.fromisoformat(raw)


def _durable_error_code(exc: Exception) -> str:
    """Return the stable API code a worker persists across process boundaries."""
    code = getattr(exc, "error_code", None)
    if isinstance(code, str) and code:
        return code
    return type(exc).__name__


def _job_cancellation_token(
    store: ArtifactStore,
    lifecycle: JobLifecycleRepository,
    claim: LaunchClaim,
) -> StorageBackedCancellationToken:
    def reader(job_id: str) -> DurableCancellationRecord | None:
        row = store.get_job(job_id)
        if row is None:
            return None
        return DurableCancellationRecord(
            job_id=str(row["job_id"]),
            generation=int(row["launch_attempt"]),
            owner=str(row["launch_token"] or ""),
            cancel_requested=bool(row["cancel_requested"]),
        )

    return StorageBackedCancellationToken(
        job_id=claim.job_id,
        generation=claim.attempt,
        owner=claim.token,
        reader=reader,
        enter_outer_shield=lambda: lifecycle.enter_critical(claim),
        exit_outer_shield=lambda: lifecycle.exit_critical(claim),
    )


def _invoke_job_handler(
    handler: Callable[..., None],
    store: ArtifactStore,
    workspace: str,
    job: dict,
    params: dict,
    *,
    cancel_check: CancelCheck,
) -> None:
    """Invoke a worker handler through the mandatory cancellation seam."""

    handler(store, workspace, job, params, cancel_check=cancel_check)


def _checkpoint(cancel_check: CancelCheck | None) -> None:
    if cancel_check is not None:
        cancel_check()


def _run_auto_eda_job(
    store: ArtifactStore,
    workspace: str,
    job: dict,
    params: dict,
    *,
    cancel_check: CancelCheck | None = None,
) -> None:
    # Local import: keep the module importable (and the spawn bootstrap cheap)
    # without pulling the full driver stack until a job actually sessions.
    from eda_platform.core.process_metrics import process_peak_rss
    from eda_platform.drivers.auto_eda import run_auto_eda
    from eda_platform.schemas.resource_metrics import EdaResourcePolicy
    from eda_platform.tools.resource_preflight import preflight_csv_resources

    _checkpoint(cancel_check)
    file_paths = [store.root / str(rel) for rel in params.get("dataset_paths", [])]
    # Opt-in pre-clean runs BEFORE ingest: the cleaned copies become the
    # ingested datasets, the originals ride along as raw_file_paths so the
    # lineage keeps both, and the recipes become CleaningRecipe artifacts.
    preclean = _preclean_options(params)
    requested_workers = int(params.get("dataset_workers", 1))
    policy = EdaResourcePolicy.model_validate(params.get("resource_policy") or {})
    baseline_rss = process_peak_rss()
    preprocessing_started = perf_counter()
    resource_preflight = preflight_csv_resources(
        file_paths,
        requested_dataset_workers=requested_workers,
        baseline_peak_rss_bytes=baseline_rss.bytes or 0,
        policy=policy,
        precleaning_enabled=preclean is not None,
        cancel_check=(None if cancel_check is None else lambda: _checkpoint(cancel_check)),
    )
    if resource_preflight.status != "accepted":
        run_auto_eda(
            file_paths,
            workspace=workspace,
            project_id=str(job["project_id"]),
            session_id=str(job["session_id"]),
            business_context=str(params.get("business_context", "")),
            llm=_build_llm(params),
            narrator_llm=_build_report_llm(params),
            payload_policy=_payload_policy(params),
            ml_target_column=params.get("ml_target_column"),
            ml_time_column=params.get("ml_time_column"),
            generate_report=bool(params.get("generate_report", True)),
            dataset_workers=requested_workers,
            resource_policy=policy,
            resource_preflight=resource_preflight,
            preprocessing_duration_seconds=perf_counter() - preprocessing_started,
            baseline_peak_rss=baseline_rss,
            cancel_check=cancel_check,
        )
        return
    ingest_paths = file_paths
    created_paths: list[Path] = []
    recipes = None
    if preclean is not None:
        from eda_platform.tools.precleaning import preclean_csv_files

        _checkpoint(cancel_check)
        batch = preclean_csv_files(file_paths, **preclean)
        _checkpoint(cancel_check)
        ingest_paths = batch.dataset_paths
        created_paths = batch.created_paths
        recipes = batch.recipes
        emit_job_event(
            store,
            job,
            "precleaning_applied",
            {
                "datasets_changed": sum(1 for report in batch.reports if report.changed),
                "datasets_total": len(batch.reports),
                "guards_triggered": sum(1 for report in batch.reports if report.guard_triggered),
            },
        )
    try:
        run_auto_eda(
            ingest_paths,
            workspace=workspace,
            project_id=str(job["project_id"]),
            session_id=str(job["session_id"]),
            business_context=str(params.get("business_context", "")),
            llm=_build_llm(params),
            narrator_llm=_build_report_llm(params),
            payload_policy=_payload_policy(params),
            ml_target_column=params.get("ml_target_column"),
            ml_time_column=params.get("ml_time_column"),
            precleaning=recipes,
            raw_file_paths=file_paths if recipes is not None else None,
            generate_report=bool(params.get("generate_report", True)),
            dataset_workers=requested_workers,
            resource_policy=policy,
            resource_preflight=resource_preflight,
            preprocessing_duration_seconds=perf_counter() - preprocessing_started,
            baseline_peak_rss=baseline_rss,
            cancel_check=cancel_check,
        )
        _checkpoint(cancel_check)
    finally:
        # The cleaned frames are already copied into the project's uploads as a
        # new dataset version; the staging copies next to the source are not.
        for temp_file in created_paths:
            with suppress(OSError):
                temp_file.unlink(missing_ok=True)
            with suppress(OSError):
                temp_file.parent.rmdir()


def _run_exploration_job(
    store: ArtifactStore,
    workspace: str,
    job: dict,
    params: dict,
    *,
    cancel_check: CancelCheck | None = None,
) -> None:
    """Execute the certified E4a root while the journal owns product status."""
    from eda_platform.core.exploration_journal import JsonlExplorationJournal
    from eda_platform.core.exploration_shadow_store import shadow_run_root
    from eda_platform.worker.exploration import run_exploration_worker

    _checkpoint(cancel_check)
    source_session_id = str(params.get("source_session_id", ""))
    exploration_id = str(params.get("exploration_id", ""))
    policy = params.get("policy")
    goal = policy.get("goal") if isinstance(policy, dict) else None
    _start_derived_run(
        store,
        job,
        source_session_id=source_session_id,
        title=str(goal or "Autonomous exploration"),
    )
    run_exploration_worker(
        store,
        workspace,
        job,
        params,
        llm=_build_llm(params),
        cancel_check=cancel_check,
    )
    state = JsonlExplorationJournal(
        shadow_run_root(store.root, exploration_id) / "journal.jsonl"
    ).rebuild()
    if state is None:
        raise RuntimeError("exploration worker returned without a journal state")
    emit_job_event(
        store,
        job,
        "exploration.attempt_finished",
        {
            "exploration_id": exploration_id,
            "exploration_status": state.status,
            "stop_reason": state.stop_reason,
            "journal_seq": state.last_seq,
        },
    )


def _preclean_options(params: dict) -> dict[str, Any] | None:
    """The pre-clean kwargs when the caller asked for one, else None."""
    raw = params.get("precleaning")
    if not isinstance(raw, dict):
        return None
    clean_missing = bool(raw.get("clean_missing_values", False))
    drop_outliers = bool(raw.get("drop_iqr_outliers", False))
    if not clean_missing and not drop_outliers:
        return None
    return {
        "clean_missing_values": clean_missing,
        "missing_threshold_percent": float(raw.get("missing_threshold_percent", 70.0)),
        "min_rows_keep_percent": float(raw.get("min_rows_keep_percent", 50.0)),
        "drop_iqr_outliers": drop_outliers,
    }


# A runaway fuse, not a calibrated tier. The agent loop bounds steps and tool
# calls but nothing stops one question from burning tokens inside them, and the
# job used to run on an empty policy where every ceiling was None. Measured cost
# is ~32k tokens / $0.0066 per question, so these sit far above a healthy run
# and only fire on a loop that has genuinely lost control. Per-tier limits
# belong in settings once real sessions have been measured.
QUESTION_AGENT_BUDGET_FUSE = SessionBudgetPolicy(
    max_total_tokens=500_000,
    max_cost_usd=Decimal("2.00"),
    max_wall_seconds=1_800.0,
)


def _run_question_exec_job(
    store: ArtifactStore,
    workspace: str,
    job: dict,
    params: dict,
    *,
    cancel_check: CancelCheck | None = None,
) -> None:
    # Same local-import rationale as auto_eda. The driver owns the whole batch
    # run lifecycle (start_session, manifest, artifacts, completed status); the job
    # row's session_id IS the batch run id, so SSE follows the same trace stream.
    from eda_platform.drivers.question_exec import run_question_batch

    _checkpoint(cancel_check)
    _verify_question_source(store, job, params)
    _checkpoint(cancel_check)
    run_question_batch(
        project_id=str(job["project_id"]),
        source_session_id=str(params["source_session_id"]),
        question_ids=[str(params["question_id"])],
        workspace=workspace,
        llm=_build_llm(params),
        session_id=str(job["session_id"]),
        generate_report=bool(params.get("generate_report", False)),
        code_backend=(
            None if params.get("llm") == "offline" else _safe_question_code_backend(store, job)
        ),
        budget_policy=QUESTION_AGENT_BUDGET_FUSE,
        cancel_check=cancel_check,
    )
    _checkpoint(cancel_check)


def _safe_question_code_backend(store: ArtifactStore, job: dict) -> Any | None:
    """Resolve Python only through a verified untrusted-code sandbox.

    A missing Docker/trusted executor removes the code tool but does not block
    SQL, artifacts or saved skills. The capability decision is persisted in
    the run trace so an answer never implies that Python was available.
    """
    from eda_platform.core.sandbox import SandboxUnavailableError
    from eda_platform.core.sandbox_broker import SandboxBroker

    try:
        return SandboxBroker.from_env(
            work_root=(
                store.root
                / "_sandbox"
                / "question_agent"
                / str(job["project_id"])
                / str(job["session_id"])
            )
        ).require_safe_backend()
    except (SandboxUnavailableError, OSError, RuntimeError) as exc:
        store.append_trace(
            str(job["project_id"]),
            TraceEvent(
                session_id=str(job["session_id"]),
                event_type="agent_capability_unavailable",
                name="run_open_analysis",
                finished_at=datetime.now(UTC),
                summary={
                    "capability": "sandboxed_python",
                    "reason": str(exc)[:500],
                },
            ),
        )
        return None


def _verify_question_source(store: ArtifactStore, job: dict, params: dict) -> None:
    """Last line of defence: recompute the candidate fingerprint from the
    source run right before executing. The API checked it at execute time, but
    the candidate set can change between enqueue and pickup — fail closed."""
    # Local import mirrors the driver imports: keep spawn bootstrap cheap.
    from eda_platform.application.services.question_service import (
        candidate_fingerprint,
        latest_candidate_set,
    )

    question_id = str(params.get("question_id", ""))
    expected = str(params.get("candidate_fingerprint", ""))
    candidate_set = latest_candidate_set(
        store, str(job["project_id"]), str(params.get("source_session_id", ""))
    )
    candidate = None
    if candidate_set is not None:
        candidate = next(
            (item for item in candidate_set.candidates if item.question_id == question_id),
            None,
        )
    if candidate is None or candidate_fingerprint(candidate) != expected:
        raise ValueError("question source changed since approval")


def _run_skill_replay_job(
    store: ArtifactStore,
    workspace: str,
    job: dict,
    params: dict,
    *,
    cancel_check: CancelCheck | None = None,
) -> None:
    """Replay an approved skill onto a fresh derived run.

    The job's session_id IS the derived run, so a failed replay never rewrites the
    source run's status and SSE follows the derived run's trace stream. No LLM
    is involved: `replay_skill` runs the frozen plan through the read-only gate.
    """
    # Same local-import rationale as the other kinds: keep spawn bootstrap cheap.
    from eda_platform.drivers.analysis_skill import replay_skill
    from eda_platform.schemas.sessions import SessionManifest, clip_run_title
    from eda_platform.schemas.skills import AnalysisSkill

    _checkpoint(cancel_check)
    project_id = str(job["project_id"])
    source_session_id = str(params["source_session_id"])
    derived_session_id = str(job["session_id"])
    skill = AnalysisSkill.model_validate(params["skill"])
    dataset_ids = [str(item) for item in params.get("dataset_ids", [])]
    targets = _replay_targets(project_id, source_session_id, workspace, dataset_ids)

    store.start_session(project_id, derived_session_id)
    # Manifest first: it describes the intent, not the outcome. Written after
    # the replay, a refused or crashed replay left a derived run with no
    # manifest and therefore no source_session_id (review I3).
    store.write_manifest(
        SessionManifest(
            session_id=derived_session_id,
            project_id=project_id,
            input_hashes={dataset.record.name: dataset.record.content_hash for dataset in targets},
            code_version=_source_code_version(store, project_id, source_session_id),
            title=clip_run_title(f"Skill: {skill.name}"),
            source_session_id=source_session_id,
        )
    )
    result = replay_skill(
        skill,
        targets,
        store=store,
        project_id=project_id,
        session_id=derived_session_id,
        cancel_check=cancel_check,
    )
    _checkpoint(cancel_check)
    if result.status != "answer":
        raise ValueError(f"skill replay was refused: {result.message}"[:500])


def _run_relationship_validate_job(
    store: ArtifactStore,
    workspace: str,
    job: dict,
    params: dict,
    *,
    cancel_check: CancelCheck | None = None,
) -> None:
    """Fully validate one relationship candidate of the source run.

    The driver writes its RelationshipValidationSet artifact onto the SOURCE
    run (that is where the graph reads from); this job's own session_id is a
    derived run that only carries the lifecycle, so a failure here never flips
    the source run to failed.
    """
    # Same local-import rationale as the other kinds: keep spawn bootstrap cheap.
    from eda_platform.application.services.relationship_service import (
        candidate_fingerprint,
        latest_candidate_set,
        relationship_id_for,
    )
    from eda_platform.core.session_loader import load_run
    from eda_platform.drivers.auto_eda import validate_relationship_candidate_on_demand
    from eda_platform.schemas.sessions import SessionManifest, clip_run_title

    _checkpoint(cancel_check)
    project_id = str(job["project_id"])
    source_session_id = str(params["source_session_id"])
    derived_session_id = str(job["session_id"])
    relationship_id = str(params["relationship_id"])

    candidate_set = latest_candidate_set(store, project_id, source_session_id)
    candidate = None
    if candidate_set is not None:
        candidate = next(
            (
                item
                for item in candidate_set.candidates
                if relationship_id_for(item.pair.label()) == relationship_id
            ),
            None,
        )
    # Last line of defence: the API checked the fingerprint at execute time, but
    # discovery can rerun between enqueue and pickup — fail closed.
    if candidate is None or candidate_fingerprint(candidate) != str(
        params.get("candidate_fingerprint", "")
    ):
        raise ValueError("relationship source changed since approval")

    loaded = load_run(project_id, source_session_id, workspace=workspace)
    if loaded.result is None:
        raise ValueError("source run could not be reloaded for validation")

    store.start_session(project_id, derived_session_id)
    store.write_manifest(
        SessionManifest(
            session_id=derived_session_id,
            project_id=project_id,
            input_hashes={
                dataset.record.name: dataset.record.content_hash
                for dataset in loaded.result.loaded_datasets
            },
            code_version=_source_code_version(store, project_id, source_session_id),
            title=clip_run_title(f"Validate: {candidate.pair.label()}"),
            source_session_id=source_session_id,
        )
    )
    _checkpoint(cancel_check)
    validation = validate_relationship_candidate_on_demand(
        loaded.result,
        candidate,
        cancel_check=cancel_check,
    )
    _checkpoint(cancel_check)
    emit_job_event(
        store,
        job,
        "relationship.validated",
        {
            "pair": candidate.pair.label(),
            "cardinality": validation.cardinality,
            "verified": validation.verified,
        },
    )


def _run_relationship_discover_job(
    store: ArtifactStore,
    workspace: str,
    job: dict,
    params: dict,
    *,
    cancel_check: CancelCheck | None = None,
) -> None:
    """Discover relationship candidates for the source run on demand.

    Same shape as the validate branch: the driver writes its RELATIONSHIP_*
    artifacts onto the SOURCE run, and this job's own session_id is a derived run
    that only carries the lifecycle.
    """
    # Same local-import rationale as the other kinds: keep spawn bootstrap cheap.
    from dataclasses import replace

    from eda_platform.core.session_loader import load_run
    from eda_platform.drivers.auto_eda import discover_relationships_on_demand
    from eda_platform.schemas.artifacts import ArtifactType
    from eda_platform.schemas.sessions import SessionManifest, clip_run_title

    _checkpoint(cancel_check)
    project_id = str(job["project_id"])
    source_session_id = str(params["source_session_id"])
    derived_session_id = str(job["session_id"])

    loaded = load_run(project_id, source_session_id, workspace=workspace)
    if loaded.result is None:
        raise ValueError("source run could not be reloaded for discovery")
    result = loaded.result
    if params.get("force"):
        # The driver returns the existing candidate set untouched when it finds
        # one, which is what makes a duplicate request cheap; a user-requested
        # rescan has to hide the previous set from it to actually rescan.
        result = replace(
            result,
            artifacts=[
                artifact
                for artifact in result.artifacts
                if artifact.type is not ArtifactType.RELATIONSHIP_CANDIDATE_SET
            ],
        )

    store.start_session(project_id, derived_session_id)
    store.write_manifest(
        SessionManifest(
            session_id=derived_session_id,
            project_id=project_id,
            input_hashes={
                dataset.record.name: dataset.record.content_hash
                for dataset in result.loaded_datasets
            },
            code_version=_source_code_version(store, project_id, source_session_id),
            title=clip_run_title("Discover relationships"),
            source_session_id=source_session_id,
        )
    )
    _checkpoint(cancel_check)
    candidates = discover_relationships_on_demand(result, cancel_check=cancel_check)
    _checkpoint(cancel_check)
    emit_job_event(
        store,
        job,
        "relationship.discovered",
        {
            "candidate_count": len(candidates.candidates),
            "coverage_status": candidates.coverage_status,
            "rescan": bool(params.get("force")),
        },
    )


def _run_report_generate_job(
    store: ArtifactStore,
    workspace: str,
    job: dict,
    params: dict,
    *,
    cancel_check: CancelCheck | None = None,
) -> None:
    """Generate (or regenerate) the source run's final report on demand.

    Same derived-run shape as the relationship kinds: `generate_report_on_demand`
    writes its report artifacts and report/report.md onto the SOURCE run, while
    this job's own session_id only carries the lifecycle.
    """
    # Same local-import rationale as the other kinds: keep spawn bootstrap cheap.
    from eda_platform.core.session_loader import load_run
    from eda_platform.drivers.auto_eda import generate_report_on_demand
    from eda_platform.schemas.sessions import SessionManifest, clip_run_title

    _checkpoint(cancel_check)
    project_id = str(job["project_id"])
    source_session_id = str(params["source_session_id"])
    derived_session_id = str(job["session_id"])

    loaded = load_run(project_id, source_session_id, workspace=workspace)
    if loaded.result is None:
        raise ValueError("source run could not be reloaded for report generation")

    store.start_session(project_id, derived_session_id)
    store.write_manifest(
        SessionManifest(
            session_id=derived_session_id,
            project_id=project_id,
            input_hashes={
                dataset.record.name: dataset.record.content_hash
                for dataset in loaded.result.loaded_datasets
            },
            code_version=_source_code_version(store, project_id, source_session_id),
            title=clip_run_title("Generate report"),
            source_session_id=source_session_id,
        )
    )
    generated = generate_report_on_demand(
        loaded.result,
        llm=_build_llm(params),
        narrator_llm=_build_report_llm(params),
        payload_policy=_payload_policy(params),
        cancel_check=cancel_check,
    )
    _checkpoint(cancel_check)
    emit_job_event(
        store,
        job,
        "report.generated",
        {
            "source_session_id": source_session_id,
            "markdown_chars": len(generated.report_markdown),
        },
    )


def _run_run_fork_job(
    store: ArtifactStore,
    workspace: str,
    job: dict,
    params: dict,
    *,
    cancel_check: CancelCheck | None = None,
) -> None:
    """Re-run the source run with exactly one decision varied.

    `fork_session` mints the forked analysis's own `run_` id, so this job's session_id
    is a lifecycle-only derived run. The forked run's manifest is stamped with
    ``source_session_id`` afterwards — the driver writes it before it knows it is a
    fork — and the new id is reported in a ``session.forked`` event.
    """
    # Same local-import rationale as the other kinds: keep spawn bootstrap cheap.
    from eda_platform.core.session_loader import load_run
    from eda_platform.drivers.session_fork import (
        DatasetDecision,
        ForkDecision,
        MlTargetDecision,
        fork_session,
    )
    from eda_platform.schemas.sessions import SessionManifest, clip_run_title

    _checkpoint(cancel_check)
    project_id = str(job["project_id"])
    source_session_id = str(params["source_session_id"])
    derived_session_id = str(job["session_id"])

    loaded = load_run(project_id, source_session_id, workspace=workspace)
    if loaded.result is None:
        raise ValueError("source run could not be reloaded for forking")

    decision: ForkDecision
    if str(params.get("decision_kind")) == "dataset":
        decision = DatasetDecision(
            file_paths=[store.root / str(rel) for rel in params.get("dataset_paths", [])],
            label=str(params.get("dataset_label", "")),
        )
    else:
        decision = MlTargetDecision(ml_target_column=params.get("ml_target_column"))

    store.start_session(project_id, derived_session_id)
    store.write_manifest(
        SessionManifest(
            session_id=derived_session_id,
            project_id=project_id,
            input_hashes={
                dataset.record.name: dataset.record.content_hash
                for dataset in loaded.result.loaded_datasets
            },
            code_version=_source_code_version(store, project_id, source_session_id),
            title=clip_run_title(f"Fork: {decision.summary()}"),
            source_session_id=source_session_id,
        )
    )
    forked = fork_session(
        loaded.result,
        decision=decision,
        store=store,
        llm=_build_llm(params),
        payload_policy=_payload_policy(params),
        cancel_check=cancel_check,
    )
    _checkpoint(cancel_check)
    forked_manifest = store.read_manifest(project_id, forked.session_id)
    if forked_manifest is None:
        raise ValueError("the forked run wrote no manifest to stamp with its source run")
    store.write_manifest(
        forked_manifest.model_copy(update={"source_session_id": source_session_id})
    )
    emit_job_event(
        store,
        job,
        "session.forked",
        {
            "source_session_id": source_session_id,
            "forked_session_id": forked.session_id,
            "decision": decision.summary(),
        },
    )


def _start_derived_run(
    store: ArtifactStore,
    job: dict,
    *,
    source_session_id: str,
    title: str,
) -> str:
    """Open the job's own derived run with a manifest that points back at its
    source. Written before the work so a crashed job still leaves a run whose
    lineage is readable (same rationale as the skill-replay branch)."""
    from eda_platform.schemas.sessions import SessionManifest, clip_run_title

    project_id = str(job["project_id"])
    derived_session_id = str(job["session_id"])
    store.start_session(project_id, derived_session_id)
    store.write_manifest(
        SessionManifest(
            session_id=derived_session_id,
            project_id=project_id,
            input_hashes={source_session_id: "derived_job_lifecycle"},
            code_version=_source_code_version(store, project_id, source_session_id),
            title=clip_run_title(title),
            source_session_id=source_session_id,
        )
    )
    return derived_session_id


def _data_operation_services(
    store: ArtifactStore,
) -> tuple[Any, Any, Any]:
    """Construct worker-local services; no API process objects cross the boundary."""
    from eda_platform.application.services.approval_service import ApprovalService
    from eda_platform.application.services.cleaning_service import CleaningService
    from eda_platform.application.services.dataset_service import DatasetService
    from eda_platform.application.services.insight_service import InsightService
    from eda_platform.application.services.job_service import JobService
    from eda_platform.core.query import TrustedFileQueryEngine
    from eda_platform.infrastructure.job_backend import LocalProcessJobBackend

    datasets = DatasetService(
        store,
        TrustedFileQueryEngine([store.root / "projects"]),
    )
    jobs = JobService(store, LocalProcessJobBackend(store.root, store))
    cleaning = CleaningService(store, datasets, ApprovalService(store), jobs)
    return datasets, cleaning, InsightService(store)


def _write_data_operation_result(store: ArtifactStore, job: dict, result: Any) -> None:
    from eda_platform.application.job_results import write_job_result

    write_job_result(
        store.root,
        str(job["project_id"]),
        str(job["request_scope"]),
        str(job["job_id"]),
        result.model_dump_json(),
    )


def _run_cleaning_preview_job(
    store: ArtifactStore,
    workspace: str,
    job: dict,
    params: dict,
    *,
    cancel_check: CancelCheck | None = None,
) -> None:
    del workspace
    source_session_id = str(params["source_session_id"])
    _start_derived_run(
        store,
        job,
        source_session_id=source_session_id,
        title="Cleaning preview",
    )
    _, cleaning, _ = _data_operation_services(store)
    result = cleaning.preview(
        source_session_id,
        dataset_id=str(params["dataset_id"]),
        trim_whitespace=bool(params.get("trim_whitespace", True)),
        drop_duplicate_rows=bool(params.get("drop_duplicate_rows", True)),
        drop_missing_rows=bool(params.get("drop_missing_rows", False)),
        drop_outlier_rows=bool(params.get("drop_outlier_rows", False)),
        cancel_check=cancel_check,
    )
    _checkpoint(cancel_check)
    _write_data_operation_result(store, job, result)


def _run_cleaning_apply_job(
    store: ArtifactStore,
    workspace: str,
    job: dict,
    params: dict,
    *,
    cancel_check: CancelCheck | None = None,
) -> None:
    del workspace
    source_session_id = str(params["source_session_id"])
    _start_derived_run(
        store,
        job,
        source_session_id=source_session_id,
        title="Apply cleaning",
    )
    _, cleaning, _ = _data_operation_services(store)
    result = cleaning.apply(
        source_session_id,
        action_hash=str(params["action_hash"]),
        approval_token=str(params["approval_token"]),
        llm=cast(Any, str(params.get("llm", "env"))),
        idempotency_key=f"{job['job_id']}:cleaned-analysis",
        cancel_check=cancel_check,
    )
    _checkpoint(cancel_check)
    _write_data_operation_result(store, job, result)


def _run_dataset_distributions_job(
    store: ArtifactStore,
    workspace: str,
    job: dict,
    params: dict,
    *,
    cancel_check: CancelCheck | None = None,
) -> None:
    del workspace
    source_session_id = str(params["source_session_id"])
    _start_derived_run(
        store,
        job,
        source_session_id=source_session_id,
        title="Dataset distributions",
    )
    datasets, _, _ = _data_operation_services(store)
    result = datasets.get_distributions(
        str(params["dataset_id"]),
        source_session_id,
        cancel_check=cancel_check,
    )
    _checkpoint(cancel_check)
    _write_data_operation_result(store, job, result)


def _run_custom_chart_job(
    store: ArtifactStore,
    workspace: str,
    job: dict,
    params: dict,
    *,
    cancel_check: CancelCheck | None = None,
) -> None:
    del workspace
    from eda_platform.application.dto import CustomChartRequest

    source_session_id = str(params["source_session_id"])
    _start_derived_run(
        store,
        job,
        source_session_id=source_session_id,
        title="Build custom chart",
    )
    datasets, _, insights = _data_operation_services(store)
    result = insights.build_custom_chart(
        source_session_id,
        CustomChartRequest.model_validate(params["chart"]),
        datasets=datasets,
        cancel_check=cancel_check,
    )
    _checkpoint(cancel_check)
    _write_data_operation_result(store, job, result)


def _run_synthesis_brief_job(
    store: ArtifactStore,
    workspace: str,
    job: dict,
    params: dict,
    *,
    cancel_check: CancelCheck | None = None,
) -> None:
    """Draft a decision story from the user's finding selection.

    The orchestrator mints and closes its own `synthesis_*` run (it ends by
    marking that run `ready_for_review`), so this job keeps a separate
    lifecycle run — passing it the job's run would let `_finish` overwrite the
    driver's status.
    """
    # Same local-import rationale as the other kinds: keep spawn bootstrap cheap.
    from eda_platform.drivers.synthesis_orchestrator import create_synthesis_brief

    _checkpoint(cancel_check)
    project_id = str(job["project_id"])
    source_session_id = str(params["source_session_id"])
    _start_derived_run(
        store, job, source_session_id=source_session_id, title="Create decision story draft"
    )
    result = create_synthesis_brief(
        project_id=project_id,
        finding_artifact_ids=list(params["finding_artifact_ids"]),
        finding_session_ids={
            str(artifact_id): str(session_id)
            for artifact_id, session_id in dict(params.get("finding_session_ids", {})).items()
        },
        workspace=workspace,
        business_context=str(params.get("business_context", "")),
        cancel_check=cancel_check,
    )
    _checkpoint(cancel_check)
    emit_job_event(
        store,
        job,
        "synthesis.brief_created",
        {
            "artifact_id": result.artifact.id,
            "brief_session_id": result.session_id,
            "selected_finding_count": len(params["finding_artifact_ids"]),
        },
    )


def _run_decision_report_job(
    store: ArtifactStore,
    workspace: str,
    job: dict,
    params: dict,
    *,
    cancel_check: CancelCheck | None = None,
) -> None:
    """Assemble a decision report from one persisted brief.

    ``llm=None`` keeps generation deterministic and never spends tokens.
    The report artifact
    inherits the brief's run, so it lands on the synthesis run.
    """
    from eda_platform.drivers.decision_report import create_decision_report

    _checkpoint(cancel_check)
    project_id = str(job["project_id"])
    source_session_id = str(params["source_session_id"])
    brief_artifact_id = str(params["brief_artifact_id"])
    raw_brief_session_id = params.get("brief_session_id")
    brief_session_id = str(raw_brief_session_id) if raw_brief_session_id else None
    _start_derived_run(
        store, job, source_session_id=source_session_id, title="Generate decision report"
    )
    artifact_id = create_decision_report(
        store,
        project_id=project_id,
        brief_artifact_id=brief_artifact_id,
        brief_session_id=brief_session_id,
        llm=None,
        cancel_check=cancel_check,
    )
    _checkpoint(cancel_check)
    emit_job_event(
        store,
        job,
        "decision_report.generated",
        {"artifact_id": artifact_id, "brief_artifact_id": brief_artifact_id},
    )


def _run_question_draft_job(
    store: ArtifactStore,
    workspace: str,
    job: dict,
    params: dict,
    *,
    cancel_check: CancelCheck | None = None,
) -> None:
    """Draft one question card from the user's own text and append it to the
    source run's candidate set.

    With a live model the question agent fills the review fields (draft-your-own
    question flow); offline it appends an unscored
    skeleton so the card is still reviewable and editable without a provider.
    """
    # Same local-import rationale as the other kinds: keep spawn bootstrap cheap.
    from eda_platform.agents.question_agent import propose_llm_question_candidates
    from eda_platform.core.llm import is_offline_client
    from eda_platform.drivers.card_edit import append_candidate
    from eda_platform.tools.question_discovery import make_question_id

    _checkpoint(cancel_check)
    project_id = str(job["project_id"])
    source_session_id = str(params["source_session_id"])
    question = str(params["question"]).strip()
    if not question:
        raise ValueError("a drafted question card needs a non-empty question")
    _start_derived_run(store, job, source_session_id=source_session_id, title=f"Draft: {question}")

    artifacts = store.list_artifacts(project_id=project_id, session_id=source_session_id)
    llm = _build_llm(params)
    if is_offline_client(llm):
        candidate = _unscored_user_card(question, artifacts)
    else:
        proposal = propose_llm_question_candidates(
            artifacts,
            llm=llm,
            business_context=(
                f"The user supplied this exact analysis question: {question!r}. "
                "Return exactly one opportunity card for that question and complete "
                "all review fields without changing its intent."
            ),
            max_questions=1,
            payload_policy=_payload_policy(params),
            cancel_check=cancel_check,
        )
        if not proposal.candidates:
            raise ValueError(proposal.error or "the model returned no question card")
        drafted = proposal.candidates[0]
        candidate = drafted.model_copy(
            update={
                "question_en": question,
                "question_id": make_question_id(
                    origin="llm",
                    question_en=question,
                    target_datasets=drafted.target_datasets,
                ),
            }
        )
    append_candidate(
        store, project_id=project_id, session_id=source_session_id, candidate=candidate
    )
    _checkpoint(cancel_check)
    emit_job_event(
        store,
        job,
        "question.drafted",
        {
            "source_session_id": source_session_id,
            "question_id": candidate.question_id,
            "enriched": not is_offline_client(llm),
        },
    )


def _unscored_user_card(question: str, artifacts: list[Any]) -> Any:
    """A user-authored card with no model enrichment.

    Every deterministic score is 0: nothing has scored this question yet, and
    inventing a signal estimate would put an unexamined card above measured
    ones. Feasibility still runs, so the execution gate is real.
    """
    from eda_platform.core.methods import MethodGateContext, evaluate_feasibility
    from eda_platform.schemas.artifacts import ArtifactType, DatasetProfile
    from eda_platform.schemas.questions import QuestionCandidate, QuestionScore
    from eda_platform.tools.question_discovery import make_question_id

    profiles = [
        DatasetProfile.model_validate(artifact.payload)
        for artifact in artifacts
        if artifact.type is ArtifactType.DATASET_PROFILE
    ]
    target_datasets = [profile.name for profile in profiles]
    feasibility = evaluate_feasibility(
        MethodGateContext(
            profiles=profiles,
            target_datasets=target_datasets,
            analysis_mode=None,
            target_column=None,
        )
    )
    return QuestionCandidate(
        question_id=make_question_id(
            origin="llm", question_en=question, target_datasets=target_datasets
        ),
        question_en=question,
        origin="llm",
        target_datasets=target_datasets,
        score=QuestionScore(
            data_availability=0.0,
            statistical_signal=0.0,
            quality_risk=0.0,
            join_risk=0.0,
            deterministic_score=0.0,
        ),
        feasibility=feasibility,
        candidate_methods=([feasibility.method_id] if feasibility.method_id is not None else []),
        priority_rationale=(
            "Drafted from your own question without a model, so it carries no "
            "deterministic score yet. Fill in the review fields before planning it."
        ),
    )


def _run_investigation_plan_job(
    store: ArtifactStore,
    workspace: str,
    job: dict,
    params: dict,
    *,
    cancel_check: CancelCheck | None = None,
) -> None:
    """Build reviewable investigation plans for the selected questions.

    `create_investigation_plans` mints its own ``investigation_*`` plan run and
    leaves it in ``awaiting_approval``; this job's session_id is a separate
    lifecycle run, so finishing here never overwrites that status.
    """
    # Same local-import rationale as the other kinds: keep spawn bootstrap cheap.
    from eda_platform.drivers.investigation_orchestrator import create_investigation_plans
    from eda_platform.schemas.artifacts import ArtifactType

    _checkpoint(cancel_check)
    project_id = str(job["project_id"])
    source_session_id = str(params["source_session_id"])
    question_ids = [str(item) for item in params.get("question_ids", [])]
    if not question_ids:
        raise ValueError("plan building needs at least one question")
    _start_derived_run(
        store,
        job,
        source_session_id=source_session_id,
        title=f"Plan {len(question_ids)} question(s)",
    )
    planned = create_investigation_plans(
        project_id=project_id,
        source_session_id=source_session_id,
        question_ids=question_ids,
        workspace=workspace,
        deep=bool(params.get("deep", False)),
        cancel_check=cancel_check,
    )
    _checkpoint(cancel_check)
    emit_job_event(
        store,
        job,
        "investigation.planned",
        {
            "source_session_id": source_session_id,
            "plan_session_id": planned.session_id,
            "plan_count": sum(
                artifact.type is ArtifactType.INVESTIGATION_PLAN for artifact in planned.artifacts
            ),
        },
    )


def _run_investigation_execute_job(
    store: ArtifactStore,
    workspace: str,
    job: dict,
    params: dict,
    *,
    cancel_check: CancelCheck | None = None,
) -> None:
    """Execute the approved plans of one plan run.

    Findings, records and the run's completed status land on ``plan_session_id``;
    this job's own derived run only carries the lifecycle.
    """
    # Same local-import rationale as the other kinds: keep spawn bootstrap cheap.
    from eda_platform.drivers.investigation_orchestrator import execute_investigation_plans
    from eda_platform.schemas.artifacts import ArtifactType

    _checkpoint(cancel_check)
    project_id = str(job["project_id"])
    source_session_id = str(params["source_session_id"])
    plan_session_id = str(params["plan_session_id"])
    plan_ids = [str(item) for item in params.get("plan_ids", [])]
    _verify_plan_fingerprints(store, project_id, plan_session_id, params)
    _start_derived_run(
        store,
        job,
        source_session_id=source_session_id,
        title=f"Investigate {len(plan_ids)} plan(s)",
    )
    result = execute_investigation_plans(
        project_id=project_id,
        plan_session_id=plan_session_id,
        plan_ids=plan_ids,
        workspace=workspace,
        llm=_build_llm(params),
        cancel_check=cancel_check,
    )
    _checkpoint(cancel_check)
    emit_job_event(
        store,
        job,
        "investigation.executed",
        {
            "plan_session_id": plan_session_id,
            "finding_count": sum(
                artifact.type is ArtifactType.VALIDATED_FINDING for artifact in result.artifacts
            ),
            "skipped": [skip.reason for skip in result.skipped],
        },
    )


def _verify_plan_fingerprints(
    store: ArtifactStore, project_id: str, plan_session_id: str, params: dict
) -> None:
    """Last line of defence: recompute each plan's content fingerprint right
    before executing. The API checked it when the approval was consumed, but a
    rebuilt plan between enqueue and pickup must fail closed."""
    from eda_platform.drivers.investigation_orchestrator import _plan_fingerprint
    from eda_platform.schemas.artifacts import ArtifactType
    from eda_platform.schemas.investigations import InvestigationPlan

    expected = params.get("plan_fingerprints")
    expected = expected if isinstance(expected, dict) else {}
    artifacts = {
        artifact.id: artifact
        for artifact in store.list_artifacts(project_id=project_id, session_id=plan_session_id)
        if artifact.type is ArtifactType.INVESTIGATION_PLAN
    }
    for plan_id in [str(item) for item in params.get("plan_ids", [])]:
        artifact = artifacts.get(plan_id)
        if artifact is None:
            raise ValueError("investigation plan changed since approval")
        plan = InvestigationPlan.model_validate(artifact.payload)
        if _plan_fingerprint(plan) != str(expected.get(plan_id, "")):
            raise ValueError("investigation plan changed since approval")


def _run_macro_loop_job(
    store: ArtifactStore,
    workspace: str,
    job: dict,
    params: dict,
    *,
    cancel_check: CancelCheck | None = None,
) -> None:
    """Run the Ultra macro loop over an executed plan run.

    `run_macro_loop` refuses without a matching pre-authorization artifact on
    the plan run, which the API writes only after the user approves this exact
    depth — so a hand-made job row cannot start follow-up rounds.
    """
    # Same local-import rationale as the other kinds: keep spawn bootstrap cheap.
    from eda_platform.core.budget import SessionBudgetPolicy
    from eda_platform.core.llm_ledger import (
        BUDGET_EVENT_TYPES,
        meter_llm_client,
        restore_run_budget_state,
    )
    from eda_platform.drivers.investigation_orchestrator import run_macro_loop

    _checkpoint(cancel_check)
    project_id = str(job["project_id"])
    source_session_id = str(params["source_session_id"])
    plan_session_id = str(params["plan_session_id"])
    depth = int(params.get("depth", 0))
    _start_derived_run(
        store, job, source_session_id=source_session_id, title=f"Macro loop (depth {depth})"
    )
    budget_policy = SessionBudgetPolicy()
    session_budget = restore_run_budget_state(
        budget_policy,
        store.list_trace_events(
            project_id=project_id,
            session_id=plan_session_id,
            event_types=BUDGET_EVENT_TYPES,
        ),
        run_started_at=_run_started_at(store, project_id, plan_session_id),
    )

    def emit_usage(event: TraceEvent) -> None:
        store.append_trace(project_id, event)

    run_llm = meter_llm_client(
        _build_llm(params),
        session_id=plan_session_id,
        emit=emit_usage,
        budget=session_budget,
        session_dir=store.session_dir(project_id, plan_session_id),
    )
    result = run_macro_loop(
        project_id=project_id,
        plan_session_id=plan_session_id,
        workspace=workspace,
        llm=run_llm,
        depth=depth,
        budget_policy=budget_policy,
        restored_session_budget=session_budget,
        cancel_check=cancel_check,
    )
    _checkpoint(cancel_check)
    if result is None:
        raise ValueError("the macro loop needs analysis depth 2 or higher")
    followup = [row for row in result.ledger.rounds if row.round_id > 0]
    emit_job_event(
        store,
        job,
        "macro_loop.finished",
        {
            "plan_session_id": plan_session_id,
            "exit_reason": result.exit_reason,
            "followup_rounds": len(followup),
            "new_validated_findings": sum(row.new_validated_findings for row in followup),
            "tokens": sum(row.tokens for row in result.ledger.rounds),
            "ledger_artifact_id": result.ledger_artifact_id,
        },
    )


def _replay_targets(
    project_id: str, source_session_id: str, workspace: str, dataset_ids: list[str]
) -> list[Any]:
    """The source run's loaded datasets the approval selected, in that order."""
    from eda_platform.core.session_loader import load_run

    loaded = load_run(project_id, source_session_id, workspace=workspace)
    available = (
        {dataset.record.dataset_id: dataset for dataset in loaded.result.loaded_datasets}
        if loaded.result is not None
        else {}
    )
    targets = [available[dataset_id] for dataset_id in dataset_ids if dataset_id in available]
    if len(targets) != len(dataset_ids):
        raise ValueError("source data for the selected dataset(s) is unavailable")
    return targets


def _source_code_version(store: ArtifactStore, project_id: str, session_id: str) -> str:
    manifest = None
    with suppress(OSError, ValueError):
        manifest = store.read_manifest(project_id, session_id)
    return manifest.code_version if manifest is not None else "unknown"


def _payload_policy(params: dict) -> PayloadPolicy:
    """Caller's Settings choice, falling back to the driver default when the
    job predates the field or sent an unknown value."""
    value = params.get("payload_policy")
    if value in get_args(PayloadPolicy):
        return cast(PayloadPolicy, value)
    return "schema+aggregates"


def _build_llm(
    params: dict,
    *,
    cancellation: CancellationToken | None = None,
) -> LLMClient:
    client: LLMClient
    if params.get("llm") == "offline":
        client = OfflineLLMClient()
    else:
        client = create_llm_client(load_llm_settings_from_env_file())
    return _with_cancellation(client, cancellation)


def _build_report_llm(
    params: dict,
    *,
    cancellation: CancellationToken | None = None,
) -> LLMClient | None:
    """The narrator's client, or None when no report override is configured."""
    if params.get("llm") == "offline":
        return None
    settings = load_report_llm_settings_from_env_file()
    if settings == load_llm_settings_from_env_file():
        return None
    return _with_cancellation(create_llm_client(settings), cancellation)


def _with_cancellation(
    client: LLMClient, cancellation: CancellationToken | None
) -> LLMClient:
    actual_cancellation = cancellation or current_cancellation_token()
    if actual_cancellation is None:
        return client
    return CancellableLLMClient(client, actual_cancellation)


def _sanitize_error(message: str, workspace: str) -> str:
    """Keep server filesystem layout out of API-visible error text (review O2),
    and drop upstream traceback bodies that some LLM endpoints echo back."""
    marker = message.find("Traceback (most recent call last")
    if marker != -1:
        message = message[:marker].rstrip() + " [upstream traceback removed]"
    for prefix in {str(Path(workspace).resolve()), str(workspace)}:
        if prefix and prefix != "/":
            message = message.replace(prefix, "<workspace>")
    for env_name in API_KEY_ENV_VARS:
        secret = os.environ.get(env_name, "")
        if secret:
            message = message.replace(secret, "<redacted>")
    message = re.sub(
        r"(?i)(authorization\s*[:=]\s*)(?:bearer\s+)?[^\s,;]+",
        r"\1<redacted>",
        message,
    )
    message = re.sub(
        r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+",
        "Bearer <redacted>",
        message,
    )
    return message


def _finish(
    store: ArtifactStore,
    job: dict,
    status: str,
    *,
    error_code: str | None = None,
    error_message: str | None = None,
    lifecycle: JobLifecycleRepository | None = None,
    claim: LaunchClaim | None = None,
) -> None:
    if lifecycle is None or claim is None:
        raise RuntimeError("Terminal lifecycle claim is required.")
    lifecycle.finish(
        claim,
        status,
        error_code=error_code,
        error_message=error_message,
    )
    lifecycle.materialize_trace(str(job["job_id"]))


def main(argv: list[str]) -> None:
    """CLI entry: acknowledge, wait for the parent gate, then claim running."""
    if len(argv) != 5:
        raise SystemExit("legacy ungated worker invocation is disabled")
    workspace, job_id, token, attempt, gate_argument = argv
    workspace = str(require_absolute_workspace(workspace, source="worker workspace"))
    if not acknowledge_and_wait(gate_argument, token, START_ACK_TIMEOUT_SECONDS):
        return
    run_job(
        workspace,
        job_id,
        launch_token=token,
        launch_attempt=int(attempt),
    )


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
