"""Bounded LangGraph fan-out with replay-safe, ordered artifact commits.

Only the current batch's artifact envelopes enter checkpoints. Dataset frames,
connections and other runtime resources remain in ``PipelineServices``. A completed
compute node can therefore survive a failed sibling or commit without rerunning
the computation. Artifact upserts and keyed completion events make commit replay
idempotent when the process exits before the commit checkpoint is persisted.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Any, TypedDict, cast

from langgraph.cache.sqlite import SqliteCache
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Send

from eda_platform.core.graph_execution import GraphPersistence, graph_execution
from eda_platform.core.ids import stable_hash
from eda_platform.core.kernel import (
    PipelineResult,
    SessionCancelled,
    SessionContext,
    Step,
    _check_produced_types,
    _check_required_types,
    _record_step_failure,
    _step_cache_key,
)
from eda_platform.schemas.artifacts import Artifact
from eda_platform.schemas.sessions import TraceEvent


def _merge(left: dict, right: dict) -> dict:
    result = {**left, **right}
    return {key: value for key, value in result.items() if value is not None}


class ArtifactReference(TypedDict):
    id: str
    digest: str


class PipelineState(TypedDict):
    index: int
    batch: list[int]
    pending: list[int]
    started: dict[int, str]
    outputs: Annotated[dict[int, list[dict[str, Any]] | None], _merge]
    references: dict[int, list[ArtifactReference]]
    skipped: list[str]


class StepWork(TypedDict):
    work_index: int
    started: str


@dataclass
class PipelineServices:
    steps: tuple[Step, ...]
    ctx: SessionContext
    cache: SqliteCache
    keys: tuple[str, ...]
    width: int

    def check_run(self) -> None:
        # Check in every node: resuming a failed compute/commit bypasses prepare.
        if self.ctx.cancel_check is not None and self.ctx.cancel_check():
            raise SessionCancelled(f"Run {self.ctx.session_id} cancelled before next step.")
        self.ctx.budget.check()
        self.ctx.session_budget.check_wall_time()

    def cache_key(self, index: int) -> tuple[tuple[str, ...], str]:
        return (("step", str(index), self.steps[index].name), self.keys[index])

    def load(self, references: list[ArtifactReference]) -> list[Artifact] | None:
        try:
            artifacts: list[Artifact] = []
            for reference in references:
                artifact = self.ctx.store.get_artifact(
                    reference["id"],
                    project_id=self.ctx.project_id,
                    session_id=self.ctx.session_id,
                )
                if (
                    stable_hash(artifact.model_dump(mode="json", exclude={"created_at"}), length=32)
                    != reference["digest"]
                ):
                    return None
                artifacts.append(artifact)
            return artifacts
        except (KeyError, OSError, ValueError, TypeError):
            return None

    def emit(self, index: int, event_type: str, **values: Any) -> None:
        if event_type == "step_completed":
            values["event_key"] = "pipeline-step-completed:" + stable_hash(
                {
                    "project_id": self.ctx.project_id,
                    "session_id": self.ctx.session_id,
                    "index": index,
                    "key": self.keys[index],
                    "started_at": values["started_at"].isoformat(),
                },
                length=32,
            )
        self.ctx.emit_trace(
            TraceEvent(
                session_id=self.ctx.session_id,
                event_type=event_type,
                name=self.steps[index].name,
                **values,
            )
        )

    def failure(self, index: int, started: datetime, exc: Exception) -> None:
        if not isinstance(exc, SessionCancelled):
            _record_step_failure(self.ctx, self.steps[index], index, started, exc)


def _prepare(state: PipelineState, runtime: Runtime[PipelineServices]) -> dict[str, Any]:
    services = runtime.context
    ctx = services.ctx
    batch = list(range(state["index"], min(len(services.steps), state["index"] + services.width)))
    refs, skipped = dict(state["references"]), list(state["skipped"])
    pending, started = [], {}
    for index in batch:
        begin = datetime.now(UTC)
        try:
            services.check_run()
            key = services.cache_key(index)
            cached = services.cache.get([key]).get(key)
            if cached is not None:
                artifacts = services.load(cached)
                if artifacts is not None:
                    _check_produced_types(services.steps[index], artifacts, ctx)
                    refs[index] = cached
                    skipped.append(services.steps[index].name)
                    services.emit(index, "checkpoint_hit", summary={"artifact_count": len(cached)})
                    continue
                services.cache.clear([key[0]])
                services.emit(
                    index,
                    "checkpoint_invalid",
                    summary={
                        "reason": "referenced artifact is missing, unreadable or changed",
                    },
                )
            services.emit(index, "step_started", started_at=begin, summary={"index": index})
            _check_required_types(services.steps[index], ctx)
            pending.append(index)
            started[index] = begin.isoformat()
        except Exception as exc:
            services.failure(index, begin, exc)
            raise
    return {
        "batch": batch,
        "pending": pending,
        "started": started,
        "references": refs,
        "skipped": skipped,
    }


def _dispatch(state: PipelineState) -> str | list[Send]:
    return [
        Send("compute", {"work_index": i, "started": state["started"][i]}) for i in state["pending"]
    ] or "commit"


def _compute(state: StepWork, runtime: Runtime[PipelineServices]) -> dict[str, Any]:
    services, index = runtime.context, state["work_index"]
    try:
        services.check_run()
        _check_required_types(services.steps[index], services.ctx)
        artifacts = services.steps[index].run(services.ctx)
        _check_produced_types(services.steps[index], artifacts, services.ctx)
        return {"outputs": {index: [a.model_dump(mode="json") for a in artifacts]}}
    except Exception as exc:
        services.failure(index, datetime.fromisoformat(state["started"]), exc)
        raise


def _commit(state: PipelineState, runtime: Runtime[PipelineServices]) -> dict[str, Any]:
    services = runtime.context
    refs = dict(state["references"])
    if not state["pending"]:
        # An all-cache-hit batch still has a checkpointed commit node. On resume
        # it must honor cancellation even though it has no artifact writes.
        try:
            services.check_run()
        except Exception as exc:
            services.failure(state["batch"][0], datetime.now(UTC), exc)
            raise
    for index in state["pending"]:
        begin = datetime.fromisoformat(state["started"][index])
        try:
            services.check_run()
            artifacts = [Artifact.model_validate(a) for a in state["outputs"][index] or []]
            _check_produced_types(services.steps[index], artifacts, services.ctx)
            for artifact in artifacts:
                services.ctx.store.save_artifact(artifact)
            references: list[ArtifactReference] = [
                {
                    "id": a.id,
                    "digest": stable_hash(
                        a.model_dump(mode="json", exclude={"created_at"}), length=32
                    ),
                }
                for a in artifacts
            ]
            services.cache.set({services.cache_key(index): (references, None)})
            refs[index] = references
            services.emit(
                index,
                "step_completed",
                started_at=begin,
                finished_at=datetime.now(UTC),
                summary={"artifact_count": len(references)},
            )
        except Exception as exc:
            services.failure(index, begin, exc)
            raise
    return {
        "index": state["batch"][-1] + 1,
        "references": refs,
        "outputs": {i: None for i in state["batch"]},
    }


def build_pipeline_graph() -> StateGraph[PipelineState, PipelineServices]:
    builder = StateGraph(PipelineState, context_schema=PipelineServices)
    builder.add_node("prepare", _prepare)
    builder.add_node("compute", _compute, input_schema=StepWork)
    builder.add_node("commit", _commit)
    builder.add_edge(START, "prepare")
    builder.add_conditional_edges("prepare", _dispatch, ["compute", "commit"])
    builder.add_edge("compute", "commit")

    def after_commit(state: PipelineState, runtime: Runtime[PipelineServices]) -> str:
        return END if state["index"] >= len(runtime.context.steps) else "prepare"

    builder.add_conditional_edges("commit", after_commit, ["prepare", END])
    return builder


def run_pipeline_graph(
    steps: tuple[Step, ...],
    ctx: SessionContext,
    *,
    width: int,
) -> PipelineResult:
    if not steps:
        return PipelineResult([], [])
    keys: list[str] = []
    for index, step in enumerate(steps):
        try:
            keys.append(_step_cache_key(step, ctx))
        except Exception as exc:
            _record_step_failure(ctx, step, index, datetime.now(UTC), exc)
            raise
    directory = ctx.store.session_dir(ctx.project_id, ctx.session_id)
    inputs = {
        "steps": [{"name": s.name, "key": key} for s, key in zip(steps, keys, strict=True)],
        "width": width,
    }
    persistence = GraphPersistence(directory, "pipeline:" + stable_hash(inputs, length=32))
    with graph_execution(
        persistence, definition="pipeline", inputs=inputs, recursion_limit=len(steps) * 3 + 10
    ) as execution:
        cache = SqliteCache(
            path=str(directory / "graphs" / "pipeline-cache.sqlite"),
            serde=JsonPlusSerializer(allowed_msgpack_modules=None),
        )
        try:
            services = PipelineServices(steps, ctx, cache, tuple(keys), width)
            graph = build_pipeline_graph().compile(checkpointer=execution.saver)
            snapshot = graph.get_state(execution.config)
            invalid = bool(snapshot.values) and any(
                services.load(ids) is None for ids in snapshot.values["references"].values()
            )
            if invalid:
                assert execution.saver is not None
                execution.saver.delete_thread(persistence.execution_id)
                snapshot = graph.get_state(execution.config)
            if snapshot.values and not snapshot.next:
                final = snapshot.values
                for index, ids in final["references"].items():
                    begin = datetime.now(UTC)
                    try:
                        services.check_run()
                        services.emit(index, "checkpoint_hit", summary={"artifact_count": len(ids)})
                    except Exception as exc:
                        services.failure(index, begin, exc)
                        raise
                skipped = [s.name for s in steps]
            else:
                initial: PipelineState = {
                    "index": 0,
                    "batch": [],
                    "pending": [],
                    "started": {},
                    "outputs": {},
                    "references": {},
                    "skipped": [],
                }
                final = graph.invoke(
                    None if snapshot.values else initial,
                    execution.config,
                    context=services,
                    durability="sync",
                )
                skipped = final["skipped"]
            artifacts = []
            for index in range(len(steps)):
                begin = datetime.now(UTC)
                try:
                    loaded = services.load(final["references"][index])
                    if loaded is None:
                        raise RuntimeError("Committed pipeline artifacts became unavailable.")
                    _check_produced_types(steps[index], loaded, ctx)
                    artifacts.extend(loaded)
                except Exception as exc:
                    services.failure(index, begin, exc)
                    raise
            return PipelineResult(artifacts, cast(list[str], skipped))
        finally:
            # The pinned official SQLite cache has no public close method.
            cache._conn.close()
