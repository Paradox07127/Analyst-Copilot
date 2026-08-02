"""Executor-injected identity for one tool invocation.

Only the executor knows the provider call id, run, attempt and call position,
so receipts built under this scope carry an identity a model cannot forge or
replay from tool name + arguments alone.
"""

from __future__ import annotations

import itertools
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from eda_platform.core.ids import stable_hash


@dataclass(frozen=True, slots=True)
class ToolExecutionContext:
    run_id: str
    provider_call_id: str
    logical_step_id: str
    attempt_epoch: int = 0
    sequence_index: int = 0

    def call_identity(self) -> str:
        # provider_call_id is model-emitted text, and both provider adapters fall
        # back to the positional `tool_{i+1}` when the provider omits ids, so it
        # repeats across steps; sequence_index is the executor's own entropy.
        return (
            f"{self.run_id}/{self.provider_call_id}"
            f"#{self.attempt_epoch}.{self.sequence_index}"
        )


_ACTIVE: ContextVar[ToolExecutionContext | None] = ContextVar(
    "tool_execution_context", default=None
)
_LOCAL_SEQUENCE = itertools.count(1)


def current_execution_context() -> ToolExecutionContext | None:
    return _ACTIVE.get()


@contextmanager
def tool_execution_scope(context: ToolExecutionContext) -> Iterator[ToolExecutionContext]:
    token = _ACTIVE.set(context)
    try:
        yield context
    finally:
        _ACTIVE.reset(token)


def make_logical_step_id(run_id: str, provider_call_id: str, sequence_index: int) -> str:
    return "step_" + stable_hash(
        {
            "run_id": run_id,
            "provider_call_id": provider_call_id,
            "sequence_index": sequence_index,
        },
        length=24,
    )


def mint_local_execution_context(run_id: str) -> ToolExecutionContext:
    """Fallback identity for a tool invoked outside any executor scope.

    Unique per invocation (uuid + process-local counter) and never derivable
    from the tool's own name or arguments.
    """
    sequence_index = next(_LOCAL_SEQUENCE)
    provider_call_id = f"local_{uuid.uuid4().hex}"
    return ToolExecutionContext(
        run_id=run_id,
        provider_call_id=provider_call_id,
        logical_step_id=make_logical_step_id(run_id, provider_call_id, sequence_index),
        attempt_epoch=0,
        sequence_index=sequence_index,
    )
