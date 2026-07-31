"""Execution-local correlation for trace rows emitted by a durable job."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass


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
