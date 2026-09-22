"""Functional report workflow: bounded task results and journaled provider effects.

Report validation is deterministic and can be replayed. Model calls cannot: each
has a stable logical identity, with its response committed before its task ends.
Only JSON crosses checkpoints; artifact resolvers and clients remain in closures.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.error import HTTPError, URLError

from langgraph.func import entrypoint, task
from pydantic import BaseModel, ValidationError

from eda_platform.core.budget import BudgetExceeded
from eda_platform.core.cancellation import CancellationError
from eda_platform.core.graph_execution import (
    GraphEffectUncertain,
    GraphExecution,
    GraphIdentityError,
    GraphPersistence,
    graph_execution,
)
from eda_platform.core.ids import stable_hash
from eda_platform.core.llm import LLMClient


def model_binding(llm: LLMClient | None) -> dict[str, Any]:
    """Bind behavior-changing settings without persisting credentials."""
    settings = getattr(llm, "settings", None)
    names = (
        "provider", "base_url", "model", "temperature", "max_tokens",
        "structured_output_mode", "timeout_seconds", "usd_per_1k_prompt",
        "usd_per_1k_completion",
    )
    return {
        "client": f"{type(llm).__module__}.{type(llm).__qualname__}",
        "settings": {name: getattr(settings, name, None) for name in names},
        "configuration_digest": (
            stable_hash(settings.model_dump(mode="json", exclude={"api_key"}), length=64)
            if isinstance(settings, BaseModel) else None
        ),
    }


class ReportTaskFailure(Exception):
    """An unexpected client failure is not a model-authored repair request."""


class RecordedReportError(RuntimeError):
    def __init__(self, source_type: str, message: str) -> None:
        self.source_type = source_type
        super().__init__(message)


@dataclass(frozen=True)
class ReportWorkflow:
    execution: GraphExecution

    def step(self, name: str, run: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        # No retries: a provider outcome must be resolved by the effect journal,
        # never by an automatic task retry issuing another billable request.
        @task(name=name)
        def operation() -> dict[str, Any]:
            return run()

        return operation().result()

    def structured(
        self, *, key: str, llm: LLMClient, task_name: str,
        schema: type[BaseModel], payload: dict[str, Any],
    ) -> dict[str, Any]:
        def call() -> dict[str, Any]:
            before = _last_usage(llm)
            result: dict[str, Any] = {
                "started_at": datetime.now(UTC).isoformat(), "operation_id": key,
            }
            try:
                draft = llm.structured(task=task_name, schema=schema, payload=payload)
                result["valid_shape"] = isinstance(draft, schema)
                result["draft"] = draft.model_dump(mode="json") if isinstance(draft, schema) else {}
            except (CancellationError, GraphIdentityError, GraphEffectUncertain):
                raise
            except Exception as exc:
                cause: BaseException | None = exc
                while cause is not None:
                    unknown_transport = (
                        isinstance(cause, (TimeoutError, ConnectionError, URLError))
                        and not isinstance(cause, HTTPError)
                    )
                    unknown_server_error = (
                        isinstance(cause, HTTPError)
                        and cause.code >= 500
                        and cause.code not in {502, 503, 504}
                    )
                    if unknown_transport or unknown_server_error:
                        raise GraphEffectUncertain(
                            "Report provider outcome is unknown; start a new execution."
                        ) from exc
                    cause = cause.__cause__
                result["error"] = {
                    "type": type(exc).__name__, "message": str(exc)[:800],
                    "budget": isinstance(exc, BudgetExceeded),
                    "recoverable": isinstance(exc, (RuntimeError, ValidationError)),
                }
            usage = _last_usage(llm)
            result["usage"] = (
                usage.model_dump(mode="json")
                if usage is not None and ("error" not in result or usage is not before) else None
            )
            result["finished_at"] = datetime.now(UTC).isoformat()
            return result

        return self.execution.model_effect(
            key,
            {"task": task_name, "schema": schema.model_json_schema(),
             "payload": payload, "model": model_binding(llm)},
            call,
        )


def _last_usage(llm: LLMClient) -> Any:
    getter = getattr(llm, "last_usage", None)
    return getter() if callable(getter) else None


def raise_recorded_error(error: dict[str, Any]) -> None:
    if error.get("budget"):
        raise BudgetExceeded(error["message"])
    if error.get("recoverable") is False:
        raise ReportTaskFailure(f"{error['type']}: {error['message']}")
    raise RecordedReportError(error["type"], error["message"])


def run_report_workflow(
    run: Callable[[ReportWorkflow], dict[str, Any]], *,
    persistence: GraphPersistence | None, inputs: dict[str, Any],
    definition: str = "report-functional-v1",
) -> dict[str, Any]:
    # Hash evidence outside the checkpoint: reports may reference large SQL
    # results, but graph state must not retain a duplicate of their contents.
    with graph_execution(
        persistence, definition=definition, inputs={"digest": stable_hash(inputs, length=64)}
    ) as execution:
        workflow = ReportWorkflow(execution)

        @entrypoint(checkpointer=execution.saver)
        def report_workflow(_: dict[str, Any]) -> dict[str, Any]:
            return run(workflow)

        snapshot = report_workflow.get_state(execution.config) if execution.saver else None
        if snapshot is not None and snapshot.created_at is not None and not snapshot.next:
            return snapshot.values
        return report_workflow.invoke(
            None if snapshot is not None and snapshot.created_at is not None else {},
            execution.config, durability="sync" if execution.saver else None,
        )
