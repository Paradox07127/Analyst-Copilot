"""Local graph progress events without checkpoint payloads or provider prompts."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from threading import Lock
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from eda_platform.core.trace_correlation import current_trace_execution, emit_execution_event


class GraphTraceHandler(BaseCallbackHandler):
    def __init__(self) -> None:
        self._spans: dict[UUID, dict[str, Any]] = {}
        self._lock = Lock()

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: Any,
        *,
        run_id: UUID,
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        metadata = metadata or {}
        node = metadata.get("langgraph_node")
        if not node or name != node:
            return
        fields = asdict(current_trace_execution())
        fields["started_at"] = datetime.now(UTC)
        fields.update(
            graph_node=node,
            checkpoint_ns=metadata.get("langgraph_checkpoint_ns"),
            span_id=str(run_id),
            parent_span_id=str(kwargs["parent_run_id"]) if kwargs.get("parent_run_id") else None,
        )
        with self._lock:
            self._spans[run_id] = fields
        emit_execution_event(
            "graph.node_started", node, {"step": metadata.get("langgraph_step")}, **fields
        )

    def _finish(
        self,
        run_id: UUID,
        status: str,
        error: BaseException | None = None,
        summary: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            fields = self._spans.pop(run_id, None)
        if fields is not None:
            fields["finished_at"] = datetime.now(UTC)
            emit_execution_event(
                f"graph.node_{status}",
                fields["graph_node"],
                {"error_type": type(error).__name__} if error else (summary or {}),
                **fields,
            )

    def on_chain_end(self, outputs: Any, *, run_id: UUID, **kwargs: Any) -> None:
        summary = {}
        if isinstance(outputs, dict):
            for key in ("status", "next_node", "reason", "step", "rewrites", "tool_calls"):
                if key in outputs and isinstance(outputs[key], (str, int, bool, type(None))):
                    summary[key] = outputs[key]
            result = outputs.get("result")
            for key in ("status", "stop_reason"):
                value = getattr(result, key, None)
                if isinstance(value, str):
                    summary[key] = value
        self._finish(run_id, "completed", summary=summary)

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._finish(run_id, "interrupted", error)
