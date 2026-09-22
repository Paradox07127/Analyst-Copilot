"""Durable structured-model workflows with one task per request or query.

Planner guard repairs and interpretation validation remain domain policy. Their
individual provider calls and read-only queries are LangGraph tasks, so replay
never hides several paid requests inside one opaque checkpoint boundary.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar
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
from eda_platform.core.llm import (
    LLMClient,
    LLMResultMetadata,
    MalformedProviderResponseError,
)
from eda_platform.schemas.artifacts import Artifact

T = TypeVar("T", bound=BaseModel)


class _BlockedEffect(BaseException):
    """Keep unresolved effects out of domain code's ordinary fallback/retry catches."""

    def __init__(
        self, error: GraphEffectUncertain | GraphIdentityError | CancellationError
    ) -> None:
        self.error = error
        super().__init__(str(error))


def _transport_uncertain(error: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (TimeoutError, ConnectionError, URLError)) and not isinstance(
            current, HTTPError
        ):
            return True
        current = current.__cause__
    return False


@dataclass
class ModelWorkflow:
    execution: GraphExecution
    last_usage_record: dict[str, Any] | None = field(default=None, init=False)

    def step(self, name: str, run: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        @task(name=name)
        def operation() -> dict[str, Any]:
            return run()

        return operation().result()

    def model(
        self, key: str, task_name: str, request: dict[str, Any],
        call: Callable[[], dict[str, Any]],
        usage: Callable[[], dict[str, Any] | None] | None = None,
    ) -> dict[str, Any]:
        self.last_usage_record = None

        def invoke() -> dict[str, Any]:
            try:
                return {"value": call()}
            except (GraphEffectUncertain, GraphIdentityError, CancellationError):
                raise
            except Exception as exc:
                if _transport_uncertain(exc):
                    raise GraphEffectUncertain(
                        "Provider outcome is unknown; start a new execution."
                    ) from exc
                return {"error": {"message": str(exc)[:1000],
                                  "budget": isinstance(exc, BudgetExceeded),
                                  "value": isinstance(exc, ValueError),
                                  "malformed": isinstance(
                                      exc, (MalformedProviderResponseError, ValidationError)
                                  ),
                                  "usage": usage() if usage is not None else None}}

        request_digest = stable_hash(request, length=64)

        @task(name=task_name)
        def request_task() -> dict[str, Any]:
            try:
                return {
                    "request_digest": request_digest,
                    "outcome": self.execution.model_effect(key, request, invoke),
                }
            except (GraphEffectUncertain, GraphIdentityError, CancellationError) as exc:
                raise _BlockedEffect(exc) from exc

        saved = request_task().result()
        # Functional task replay does not call its body, so validate outside
        # the task as well as in the effect ledger's write/adoption boundary.
        if saved["request_digest"] != request_digest:
            raise _BlockedEffect(GraphIdentityError(
                "A checkpointed model task changed its request; start a new execution."
            ))
        result = saved["outcome"]
        error = result.get("error")
        if error is not None:
            self.last_usage_record = error.get("usage")
            if error["budget"]:
                raise BudgetExceeded(error["message"])
            if error.get("malformed"):
                raise MalformedProviderResponseError(error["message"])
            if error["value"]:
                raise ValueError(error["message"])
            raise RuntimeError(error["message"])
        return result["value"]

    def query(self, run: Callable[[], Artifact]) -> Artifact:
        return Artifact.model_validate(
            self.step("question_read_only_sql", lambda: run().model_dump(mode="json"))
        )


@dataclass
class WorkflowModelClient:
    inner: LLMClient
    workflow: ModelWorkflow
    prefix: str = "model"
    request_index: int = field(default=0, init=False)
    _usage: LLMResultMetadata | None = field(default=None, init=False)
    usages: list[LLMResultMetadata] = field(default_factory=list, init=False)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def _call(
        self, task_name: str, request: dict[str, Any], call: Callable[[], Any]
    ) -> Any:
        self.request_index += 1

        usage_getter = getattr(self.inner, "last_usage", None)
        before: LLMResultMetadata | None = None

        def usage_payload(*, successful: bool = False) -> dict[str, Any] | None:
            usage = usage_getter() if callable(usage_getter) else None
            if isinstance(usage, LLMResultMetadata) and (successful or usage is not before):
                return usage.model_dump(mode="json")
            return None

        def invoke() -> dict[str, Any]:
            nonlocal before
            # Provider metadata is thread-local: snapshot it in the actual task
            # thread. Successful clients may legitimately reuse a metadata object;
            # only failed requests require a freshness check against stale usage.
            prior = usage_getter() if callable(usage_getter) else None
            before = prior if isinstance(prior, LLMResultMetadata) else None
            value = call()
            return {"result": value, "usage": usage_payload(successful=True)}

        try:
            saved = self.workflow.model(
                f"{self.prefix}:{self.request_index}", task_name, request, invoke, usage_payload,
            )
        except Exception:
            self.restore_usage(self.workflow.last_usage_record)
            if self._usage is not None:
                self.usages.append(self._usage)
            raise
        self.restore_usage(saved["usage"])
        if self._usage is not None:
            self.usages.append(self._usage)
        return saved["result"]

    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        result = self._call(
            task, {"task": task, "schema": schema.model_json_schema(), "payload": payload},
            lambda: self.inner.structured(task=task, schema=schema, payload=payload).model_dump(
                mode="json"
            ),
        )
        return schema.model_validate(result)

    def text(self, *, task: str, payload: dict) -> str:
        return self._call(
            task, {"task": task, "payload": payload},
            lambda: self.inner.text(task=task, payload=payload),
        )

    def last_usage(self) -> LLMResultMetadata | None:
        return self._usage

    def restore_usage(self, value: dict[str, Any] | LLMResultMetadata | None) -> None:
        self._usage = None if value is None else LLMResultMetadata.model_validate(value)


def run_model_workflow(
    run: Callable[[ModelWorkflow], dict[str, Any]], *,
    persistence: GraphPersistence | None, inputs: dict[str, Any],
    definition: str,
    checkpoint_input: dict[str, Any] | None = None,
) -> dict[str, Any]:
    with graph_execution(
        persistence, definition=definition,
        inputs={"digest": stable_hash(inputs, length=64)},
    ) as execution:
        workflow = ModelWorkflow(execution)

        @entrypoint(checkpointer=execution.saver)
        def model_workflow(_: dict[str, Any]) -> dict[str, Any]:
            return run(workflow)

        snapshot = model_workflow.get_state(execution.config) if execution.saver else None
        if snapshot is not None and snapshot.created_at is not None and not snapshot.next:
            result = snapshot.values
        else:
            try:
                result = model_workflow.invoke(
                    None if snapshot is not None and snapshot.created_at is not None
                    else (checkpoint_input or {}),
                    execution.config, durability="sync" if execution.saver else None,
                )
            except _BlockedEffect as exc:
                raise exc.error from exc
        return result
