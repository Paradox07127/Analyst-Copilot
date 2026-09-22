"""Execution-local correlation for trace rows emitted by a durable job."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass, replace
from typing import Any

from eda_platform.schemas.sessions import TraceEvent


@dataclass(frozen=True, slots=True)
class TraceJobCorrelation:
    job_id: str
    generation: int


_CURRENT_TRACE_JOB: ContextVar[TraceJobCorrelation | None] = ContextVar(
    "eda_platform_current_trace_job",
    default=None,
)


def current_trace_job() -> TraceJobCorrelation | None:
    return _CURRENT_TRACE_JOB.get()


@contextmanager
def trace_job_scope(job_id: str, generation: int) -> Iterator[TraceJobCorrelation]:
    if not job_id:
        raise ValueError("job_id must be non-empty")
    if generation < 0:
        raise ValueError("generation must be non-negative")
    correlation = TraceJobCorrelation(job_id=job_id, generation=generation)
    reset: Token[TraceJobCorrelation | None] = _CURRENT_TRACE_JOB.set(correlation)
    try:
        yield correlation
    finally:
        _CURRENT_TRACE_JOB.reset(reset)


@dataclass(frozen=True, slots=True)
class TraceExecutionCorrelation:
    session_id: str | None = None
    turn_id: str | None = None
    execution_id: str | None = None
    parent_execution_id: str | None = None
    attempt_id: str | None = None
    graph_node: str | None = None
    checkpoint_ns: str | None = None
    effect_id: str | None = None


_CURRENT_EXECUTION: ContextVar[TraceExecutionCorrelation | None] = ContextVar(
    "eda_platform_execution_trace",
    default=None,
)
_CURRENT_EMITTER: ContextVar[Callable[[TraceEvent], object] | None] = ContextVar(
    "eda_platform_trace_emitter", default=None
)


def current_trace_execution() -> TraceExecutionCorrelation:
    correlation = _CURRENT_EXECUTION.get() or TraceExecutionCorrelation()
    from langgraph.config import get_config

    try:
        metadata = get_config().get("metadata", {})
    except RuntimeError:
        return correlation
    return replace(
        correlation,
        graph_node=metadata.get("langgraph_node") or correlation.graph_node,
        checkpoint_ns=metadata.get("langgraph_checkpoint_ns") or correlation.checkpoint_ns,
    )


@contextmanager
def trace_execution_scope(
    *, emit: Callable[[TraceEvent], object] | None = None, **fields: Any
) -> Iterator[TraceExecutionCorrelation]:
    correlation = replace(current_trace_execution(), **fields)
    token = _CURRENT_EXECUTION.set(correlation)
    emitter_token = _CURRENT_EMITTER.set(emit) if emit is not None else None
    try:
        yield correlation
    finally:
        _CURRENT_EXECUTION.reset(token)
        if emitter_token is not None:
            _CURRENT_EMITTER.reset(emitter_token)


def emit_execution_event(
    event_type: str, name: str, summary: dict[str, Any] | None = None, **fields: Any
) -> None:
    context = asdict(current_trace_execution())
    context.update(fields)
    emitter = _CURRENT_EMITTER.get()
    if emitter is not None and context.get("session_id"):
        emitter(TraceEvent(event_type=event_type, name=name, summary=summary or {}, **context))
