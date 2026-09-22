from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
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
    manage_session_status: bool = True
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
            if self.job_generation is not None and self.job_generation != correlation.generation:
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
        if self.manage_session_status:
            self.store.start_session(self.project_id, self.session_id)
        else:
            existing = self.store.get_session_index_row(self.session_id)
            if existing is None or existing["project_id"] != self.project_id:
                raise KeyError(f"Session not found: {self.session_id}")

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


def artifact_input_fingerprints(
    ctx: SessionContext,
    artifact_ids: Sequence[str],
    *,
    session_ids: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """Bind selected input contents and provenance, ignoring publication time."""
    return [
        {
            "id": artifact_id,
            "session_id": (session_ids or {}).get(artifact_id, ctx.session_id),
            "digest": stable_hash(
                ctx.store.get_artifact(
                    artifact_id,
                    project_id=ctx.project_id,
                    session_id=(session_ids or {}).get(artifact_id, ctx.session_id),
                ).model_dump(mode="json", exclude={"created_at"}),
                length=64,
            ),
        }
        for artifact_id in artifact_ids
    ]


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
            "checkpoint_schema_version": 3,
            "execution_fingerprint": ctx.execution_fingerprint,
            "environment": env_digest(),
            "step": {
                "name": step.name,
                "implementation": stable_hash(source, length=24),
                "requires": [item.value for item in step.requires],
                "produces": [item.value for item in step.produces],
                "artifact_envelope_schema": stable_hash(Artifact.model_json_schema(), length=24),
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
    from eda_platform.core.pipeline_graph import run_pipeline_graph

    if max_workers > 1 and len(steps) > 1:
        unsafe = [step.name for step in steps if not getattr(step, "parallel_safe", False)]
        if unsafe:
            raise ValueError(
                "Parallel pipeline batches require parallel_safe=True on every step; "
                f"unsafe steps: {', '.join(unsafe)}"
            )
    return run_pipeline_graph(tuple(steps), ctx, width=max(1, min(max_workers, 2)))


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
        if ctx.manage_session_status:
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
        ctx.store.persist_step_failure_fallback(
            ctx.project_id, event, update_session_status=ctx.manage_session_status
        )
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
