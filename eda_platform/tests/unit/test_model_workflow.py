from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from eda_platform.agents.model_workflow import (
    ModelWorkflow,
    WorkflowModelClient,
    run_model_workflow,
)
from eda_platform.core.graph_execution import GraphExecution, GraphPersistence
from eda_platform.core.llm import LLMResultMetadata, LLMUsage


class Answer(BaseModel):
    value: int


class WorkerExit(BaseException):
    pass


class Provider:
    def __init__(self, *, fail: bool) -> None:
        self.fail = fail
        self.calls = 0
        self.usage: LLMResultMetadata | None = None

    def structured[T: BaseModel](self, *, task: str, schema: type[T], payload: dict) -> T:
        self.calls += 1
        self.usage = LLMResultMetadata(
            provider="test", model="test", usage=LLMUsage(total_tokens=17),
        )
        if self.fail:
            raise ValueError("invalid structured response")
        return schema.model_validate({"value": 42})

    def text(self, *, task: str, payload: dict) -> str:
        raise AssertionError("unused")

    def last_usage(self) -> LLMResultMetadata | None:
        return self.usage


@pytest.mark.parametrize("fail", [False, True])
def test_adopts_effect_commit_and_restores_success_or_error_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: bool,
) -> None:
    original = GraphExecution.model_effect

    def crash_after_commit(self: GraphExecution, *args: Any, **kwargs: Any) -> dict[str, Any]:
        original(self, *args, **kwargs)
        raise WorkerExit

    def run(provider: Provider) -> dict[str, Any]:
        def workflow(active: ModelWorkflow) -> dict[str, Any]:
            client = WorkflowModelClient(provider, active)
            try:
                answer = client.structured(task="answer", schema=Answer, payload={})
            except ValueError:
                answer = None
            usage = client.last_usage()
            assert usage is not None
            return {"answer": None if answer is None else answer.value,
                    "tokens": usage.usage.total_tokens}

        return run_model_workflow(
            workflow, persistence=GraphPersistence(tmp_path, "effect-restart"),
            inputs={}, definition="test-model-workflow",
        )

    monkeypatch.setattr(GraphExecution, "model_effect", crash_after_commit)
    first = Provider(fail=fail)
    with pytest.raises(WorkerExit):
        run(first)
    monkeypatch.setattr(GraphExecution, "model_effect", original)
    resumed = Provider(fail=fail)
    assert run(resumed) == {"answer": None if fail else 42, "tokens": 17}
    assert first.calls == 1
    assert resumed.calls == 0


def test_replayed_task_rejects_changed_request_before_any_paid_work(tmp_path: Path) -> None:
    from eda_platform.core.graph_execution import GraphIdentityError

    first = Provider(fail=False)

    def run(provider: Provider, value: int, crash: bool) -> dict[str, Any]:
        def workflow(active: ModelWorkflow) -> dict[str, Any]:
            client = WorkflowModelClient(provider, active)
            answer = client.structured(task="answer", schema=Answer, payload={"value": value})
            if crash:
                raise WorkerExit
            return answer.model_dump(mode="json")

        return run_model_workflow(
            workflow, persistence=GraphPersistence(tmp_path, "task-drift"),
            inputs={}, definition="test-model-workflow",
        )

    with pytest.raises(WorkerExit):
        run(first, 1, True)
    resumed = Provider(fail=False)
    with pytest.raises(GraphIdentityError, match="changed its request"):
        run(resumed, 2, False)
    assert first.calls == 1
    assert resumed.calls == 0


def test_success_accepts_provider_reusing_usage_metadata(tmp_path: Path) -> None:
    class ConstantUsageProvider(Provider):
        reported = LLMResultMetadata(
            provider="test", model="constant", usage=LLMUsage(total_tokens=31),
        )

        def last_usage(self) -> LLMResultMetadata:
            return self.reported

    provider = ConstantUsageProvider(fail=False)

    def workflow(active: ModelWorkflow) -> dict[str, Any]:
        client = WorkflowModelClient(provider, active)
        client.structured(task="answer", schema=Answer, payload={})
        client.structured(task="answer", schema=Answer, payload={})
        return {"tokens": [usage.usage.total_tokens for usage in client.usages]}

    result = run_model_workflow(
        workflow, persistence=GraphPersistence(tmp_path, "reused-usage"),
        inputs={}, definition="test-model-workflow",
    )
    assert result == {"tokens": [31, 31]}


def test_task_failure_without_error_record_preserves_original_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unserializable(*args: Any, **kwargs: Any) -> Any:
        raise TypeError("checkpoint value is not serializable")

    monkeypatch.setattr(GraphExecution, "model_effect", unserializable)

    def workflow(active: ModelWorkflow) -> dict[str, Any]:
        client = WorkflowModelClient(Provider(fail=False), active)
        client.structured(task="answer", schema=Answer, payload={})
        raise AssertionError("unreachable")

    with pytest.raises(TypeError, match="not serializable"):
        run_model_workflow(workflow, persistence=None, inputs={}, definition="test-model-workflow")
