from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable

from eda_platform.core.budget import Budget, SessionBudgetPolicy, SessionBudgetState
from eda_platform.core.ids import stable_hash
from eda_platform.core.provenance import code_ref, env_digest
from eda_platform.core.store import ArtifactStore
from eda_platform.core.trace_correlation import current_trace_job
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.sessions import TraceEvent


class StepContractError(RuntimeError):
    """Raised when a workflow step violates its declared artifact contract."""


class SessionCancelled(RuntimeError):
    """Raised at a step boundary after a cooperative cancel request."""


@runtime_checkable
class Step(Protocol):
    name: ClassVar[str]
    requires: ClassVar[tuple[ArtifactType, ...]]
    produces: ClassVar[tuple[ArtifactType, ...]]

    def run(self, ctx: SessionContext) -> list[Artifact]: ...


@dataclass
class SessionContext:
    project_id: str
    session_id: str
    store: ArtifactStore
    max_seconds: float | None = None
    max_tokens: int | None = None
    budget_policy: SessionBudgetPolicy | None = None
    restored_session_budget: SessionBudgetState | None = None
    execution_fingerprint: str = ""
    job_id: str | None = None
    job_generation: int | None = None
    on_trace_event: Callable[[TraceEvent], None] | None = None
    cancel_check: Callable[[], bool] | None = None
    budget: Budget = field(init=False)
    session_budget: SessionBudgetState = field(init=False)

    def __post_init__(self) -> None:
        correlation = current_trace_job()
        if correlation is not None:
            if self.job_id is not None and self.job_id != correlation.job_id:
                raise ValueError("SessionContext job_id conflicts with the active job scope.")
            if (
                self.job_generation is not None
                and self.job_generation != correlation.generation
            ):
                raise ValueError(
                    "SessionContext job_generation conflicts with the active job scope."
                )
            self.job_id = correlation.job_id
            self.job_generation = correlation.generation
        self.budget = Budget(max_seconds=self.max_seconds, max_tokens=self.max_tokens)
        if self.restored_session_budget is not None:
            if (
                self.budget_policy is not None
                and self.restored_session_budget.policy != self.budget_policy
            ):
                raise ValueError("Restored run budget policy does not match budget_policy.")
            self.session_budget = self.restored_session_budget
        else:
            self.session_budget = SessionBudgetState(
                self.budget_policy
                or SessionBudgetPolicy(
                    max_wall_seconds=self.max_seconds,
                    max_total_tokens=self.max_tokens,
                )
            )
        self.store.ensure_project(self.project_id, name=self.project_id)
        self.store.start_session(self.project_id, self.session_id)

    def emit_trace(self, event: TraceEvent) -> None:
        if self.job_id is not None:
            if event.job_id is not None and event.job_id != self.job_id:
                raise ValueError("Trace event job_id conflicts with SessionContext.")
            event = event.model_copy(
                update={
                    "job_id": self.job_id,
                    "job_generation": self.job_generation,
                }
            )
        self.store.append_trace(self.project_id, event)
        if self.on_trace_event is not None:
            self.on_trace_event(event)


@dataclass
class PipelineResult:
    artifacts: list[Artifact]
    skipped_steps: list[str]


def _step_cache_key(step: Step, ctx: SessionContext) -> str:
    """Versioned signature for safe checkpoint reuse."""
    cache_key = getattr(step, "cache_key", None)
    step_specific = ""
    if callable(cache_key):
        step_specific = str(cache_key(ctx))
    try:
        source = inspect.getsource(type(step))
    except (OSError, TypeError):
        source = code_ref(type(step))
    return stable_hash(
        {
            "checkpoint_schema_version": 2,
            "execution_fingerprint": ctx.execution_fingerprint,
            "environment": env_digest(),
            "step": {
                "name": step.name,
                "implementation": stable_hash(source, length=24),
                "requires": [item.value for item in step.requires],
                "produces": [item.value for item in step.produces],
                "artifact_envelope_schema": stable_hash(
                    Artifact.model_json_schema(), length=24
                ),
                "specific": step_specific,
            },
        },
        length=32,
    )


def run_pipeline(
    steps: Sequence[Step],
    ctx: SessionContext,
    *,
    max_workers: int = 1,
) -> PipelineResult:
    """Run workflow steps, optionally parallelizing an explicitly safe batch.

    Worker threads only execute ``step.run``. Artifact writes, checkpoints, and
    trace events are committed by this coordinator in the original step order,
    preserving deterministic lineage and restart behavior.
    """
    if max_workers <= 1 or len(steps) <= 1:
        return _run_pipeline_sequential(steps, ctx)
    unsafe = [step.name for step in steps if not getattr(step, "parallel_safe", False)]
    if unsafe:
        raise ValueError(
            "Parallel pipeline batches require parallel_safe=True on every step; "
            f"unsafe steps: {', '.join(unsafe)}"
        )
    return _run_pipeline_parallel(steps, ctx, max_workers=min(max_workers, 2))


def _run_pipeline_sequential(steps: Sequence[Step], ctx: SessionContext) -> PipelineResult:
    artifacts: list[Artifact] = []
    skipped_steps: list[str] = []

    for index, step in enumerate(steps):
        started_at = datetime.now(UTC)
        try:
            if ctx.cancel_check is not None and ctx.cancel_check():
                raise SessionCancelled(
                    f"Run {ctx.session_id} cancelled before step {step.name!r}."
                )
            ctx.budget.check()
            ctx.session_budget.check_wall_time()
            checkpoint_path = _checkpoint_path(ctx, index, step.name)
            cache_key = _step_cache_key(step, ctx)
            cached = _read_checkpoint(checkpoint_path)
            if cached is not None and cached.cache_key == cache_key:
                cached_artifacts = _load_checkpoint_artifacts(cached, ctx)
                if cached_artifacts is not None:
                    _check_produced_types(step, cached_artifacts, ctx)
                    ctx.emit_trace(
                        TraceEvent(
                            session_id=ctx.session_id,
                            event_type="checkpoint_hit",
                            name=step.name,
                            summary={"artifact_count": len(cached_artifacts)},
                        ),
                    )
                    skipped_steps.append(step.name)
                    artifacts.extend(cached_artifacts)
                    continue
                ctx.emit_trace(
                    TraceEvent(
                        session_id=ctx.session_id,
                        event_type="checkpoint_invalid",
                        name=step.name,
                        summary={"reason": "referenced artifact is missing or unreadable"},
                    ),
                )

            ctx.emit_trace(
                TraceEvent(
                    session_id=ctx.session_id,
                    event_type="step_started",
                    name=step.name,
                    started_at=started_at,
                    summary={"index": index},
                ),
            )
            _check_required_types(step, ctx)
            produced = step.run(ctx)
            _check_produced_types(step, produced, ctx)

            for artifact in produced:
                ctx.store.save_artifact(artifact)
            _write_checkpoint(ctx, checkpoint_path, cache_key, [a.id for a in produced])
            ctx.emit_trace(
                TraceEvent(
                    session_id=ctx.session_id,
                    event_type="step_completed",
                    name=step.name,
                    started_at=started_at,
                    finished_at=datetime.now(UTC),
                    summary={"artifact_count": len(produced)},
                ),
            )
        except SessionCancelled:
            raise
        except Exception as exc:  # noqa: BLE001 - preserve the primary failure
            _record_step_failure(ctx, step, index, started_at, exc)
            raise
        artifacts.extend(produced)

    return PipelineResult(artifacts=artifacts, skipped_steps=skipped_steps)


@dataclass(frozen=True)
class _PendingStep:
    index: int
    step: Step
    started_at: datetime
    checkpoint_path: Path
    cache_key: str


def _run_pipeline_parallel(
    steps: Sequence[Step],
    ctx: SessionContext,
    *,
    max_workers: int,
) -> PipelineResult:
    """Compute independent steps concurrently and serialize every mutation."""
    results: list[list[Artifact] | None] = [None] * len(steps)
    skipped_steps: list[str] = []
    pending: list[_PendingStep] = []

    for index, step in enumerate(steps):
        started_at = datetime.now(UTC)
        try:
            if ctx.cancel_check is not None and ctx.cancel_check():
                raise SessionCancelled(
                    f"Run {ctx.session_id} cancelled before step {step.name!r}."
                )
            ctx.budget.check()
            ctx.session_budget.check_wall_time()
            checkpoint_path = _checkpoint_path(ctx, index, step.name)
            cache_key = _step_cache_key(step, ctx)
            cached = _read_checkpoint(checkpoint_path)
            if cached is not None and cached.cache_key == cache_key:
                cached_artifacts = _load_checkpoint_artifacts(cached, ctx)
                if cached_artifacts is not None:
                    _check_produced_types(step, cached_artifacts, ctx)
                    ctx.emit_trace(
                        TraceEvent(
                            session_id=ctx.session_id,
                            event_type="checkpoint_hit",
                            name=step.name,
                            summary={"artifact_count": len(cached_artifacts)},
                        ),
                    )
                    skipped_steps.append(step.name)
                    results[index] = cached_artifacts
                    continue
                ctx.emit_trace(
                    TraceEvent(
                        session_id=ctx.session_id,
                        event_type="checkpoint_invalid",
                        name=step.name,
                        summary={"reason": "referenced artifact is missing or unreadable"},
                    ),
                )
            ctx.emit_trace(
                TraceEvent(
                    session_id=ctx.session_id,
                    event_type="step_started",
                    name=step.name,
                    started_at=started_at,
                    summary={"index": index, "parallel_workers": max_workers},
                ),
            )
            _check_required_types(step, ctx)
            pending.append(
                _PendingStep(
                    index=index,
                    step=step,
                    started_at=started_at,
                    checkpoint_path=checkpoint_path,
                    cache_key=cache_key,
                )
            )
        except SessionCancelled:
            raise
        except Exception as exc:  # noqa: BLE001 - preserve the primary failure
            _record_step_failure(ctx, step, index, started_at, exc)
            raise

    if not pending:
        return PipelineResult(
            artifacts=[
                artifact
                for step_artifacts in results
                if step_artifacts is not None
                for artifact in step_artifacts
            ],
            skipped_steps=skipped_steps,
        )

    futures: list[tuple[_PendingStep, Future[list[Artifact]]]] = []
    with ThreadPoolExecutor(
        max_workers=min(max_workers, len(pending)),
        thread_name_prefix="eda-step",
    ) as executor:
        futures = [
            (item, executor.submit(item.step.run, ctx))
            for item in pending
        ]
        for position, (item, future) in enumerate(futures):
            try:
                produced = future.result()
                _check_produced_types(item.step, produced, ctx)
                for artifact in produced:
                    ctx.store.save_artifact(artifact)
                _write_checkpoint(
                    ctx,
                    item.checkpoint_path,
                    item.cache_key,
                    [artifact.id for artifact in produced],
                )
                ctx.emit_trace(
                    TraceEvent(
                        session_id=ctx.session_id,
                        event_type="step_completed",
                        name=item.step.name,
                        started_at=item.started_at,
                        finished_at=datetime.now(UTC),
                        summary={
                            "artifact_count": len(produced),
                            "parallel_workers": max_workers,
                        },
                    ),
                )
                results[item.index] = produced
            except SessionCancelled:
                for _, later in futures[position + 1 :]:
                    later.cancel()
                raise
            except Exception as exc:  # noqa: BLE001 - preserve the primary failure
                for _, later in futures[position + 1 :]:
                    later.cancel()
                _record_step_failure(
                    ctx,
                    item.step,
                    item.index,
                    item.started_at,
                    exc,
                )
                raise

    return PipelineResult(
        artifacts=[
            artifact
            for step_artifacts in results
            if step_artifacts is not None
            for artifact in step_artifacts
        ],
        skipped_steps=skipped_steps,
    )


def _record_step_failure(
    ctx: SessionContext,
    step: Step,
    index: int,
    started_at: datetime,
    exc: Exception,
) -> None:
    """Best-effort rich reporting with a DB-only durable fallback."""
    finished_at = datetime.now(UTC)
    event = TraceEvent(
        session_id=ctx.session_id,
        event_type="step_failed",
        name=step.name,
        job_id=ctx.job_id,
        job_generation=ctx.job_generation,
        event_key=(
            "kernel-step-failed:"
            + stable_hash(
                {
                    "project_id": ctx.project_id,
                    "session_id": ctx.session_id,
                    "step": step.name,
                    "index": index,
                    "started_at": started_at.isoformat(),
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                },
                length=32,
            )
        ),
        started_at=started_at,
        finished_at=finished_at,
        summary={
            "index": index,
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        },
    )
    status_reported = False
    trace_reported = False
    try:
        ctx.store.mark_session_status(ctx.project_id, ctx.session_id, "failed")
        status_reported = True
    except Exception:
        pass
    try:
        ctx.emit_trace(event)
        trace_reported = True
    except Exception:
        pass
    if status_reported and trace_reported:
        return
    try:
        ctx.store.persist_step_failure_fallback(ctx.project_id, event)
    except Exception:
        # The primary exception remains authoritative. A completely unavailable
        # SQLite database/filesystem cannot accept a durable fallback.
        pass


def _check_required_types(step: Step, ctx: SessionContext) -> None:
    if not step.requires:
        return
    available_artifacts, _warnings = ctx.store.list_artifacts_safe(
        project_id=ctx.project_id,
        session_id=ctx.session_id,
    )
    available = {artifact.type for artifact in available_artifacts}
    missing = set(step.requires) - available
    if missing:
        _raise_contract_error(
            step,
            ctx,
            summary={"missing_required_types": sorted(item.value for item in missing)},
        )


def _check_produced_types(
    step: Step,
    produced: list[Artifact],
    ctx: SessionContext,
) -> None:
    declared = set(step.produces)
    unexpected = {artifact.type for artifact in produced} - declared if declared else set()
    misbound = [
        artifact.id
        for artifact in produced
        if artifact.project_id != ctx.project_id or artifact.session_id != ctx.session_id
    ]
    if unexpected or misbound:
        _raise_contract_error(
            step,
            ctx,
            summary={
                "unexpected_types": sorted(item.value for item in unexpected),
                "misbound_artifact_ids": misbound,
            },
        )


def _raise_contract_error(step: Step, ctx: SessionContext, *, summary: dict[str, object]) -> None:
    ctx.emit_trace(
        TraceEvent(
            session_id=ctx.session_id,
            event_type="step_contract_violation",
            name=step.name,
            finished_at=datetime.now(UTC),
            summary=summary,
        )
    )
    details = ", ".join(f"{key}={value}" for key, value in summary.items() if value)
    raise StepContractError(f"Step {step.name!r} violated its artifact contract: {details}")


@dataclass(frozen=True)
class _Checkpoint:
    cache_key: str
    artifact_ids: list[str]


def _checkpoint_path(ctx: SessionContext, index: int, step_name: str) -> Path:
    return (
        ctx.store.session_dir(ctx.project_id, ctx.session_id)
        / "checkpoints"
        / f"{index:03d}_{step_name}.txt"
    )


def _read_checkpoint(checkpoint_path: Path) -> _Checkpoint | None:
    if not checkpoint_path.exists():
        return None
    lines = checkpoint_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return _Checkpoint(cache_key="", artifact_ids=[])
    cache_key = lines[0].removeprefix("cache_key:")
    artifact_ids = [line.strip() for line in lines[1:] if line.strip()]
    return _Checkpoint(cache_key=cache_key, artifact_ids=artifact_ids)


def _load_checkpoint_artifacts(
    checkpoint: _Checkpoint,
    ctx: SessionContext,
) -> list[Artifact] | None:
    try:
        artifacts = [
            ctx.store.get_artifact(
                artifact_id,
                project_id=ctx.project_id,
                session_id=ctx.session_id,
            )
            for artifact_id in checkpoint.artifact_ids
        ]
    except (KeyError, OSError, ValueError):
        return None
    if any(
        artifact.project_id != ctx.project_id or artifact.session_id != ctx.session_id
        for artifact in artifacts
    ):
        return None
    return artifacts


def _write_checkpoint(
    ctx: SessionContext,
    checkpoint_path: Path,
    cache_key: str,
    artifact_ids: list[str],
) -> None:
    body = "\n".join([f"cache_key:{cache_key}", *artifact_ids])
    relative = checkpoint_path.relative_to(
        ctx.store.session_dir(ctx.project_id, ctx.session_id)
    )
    ctx.store.write_session_text(
        ctx.project_id,
        ctx.session_id,
        relative.as_posix(),
        body,
    )
