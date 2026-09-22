"""Agent tool effects, artifact hydration and execution identity across worker loss."""
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel

from eda_platform.agents.data_tools import DataToolContext
from eda_platform.agents.runtime import AgentRuntime, AgentTool, AgentToolResult
from eda_platform.core.graph_execution import (
    GraphEffectUncertain,
    GraphExecution,
    GraphIdentityError,
    GraphPersistence,
)
from eda_platform.core.llm import LLMSettings, LLMToolCall, LLMToolResponse, OfflineLLMClient
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType


class WorkerLost(BaseException):
    pass


class NoArguments(BaseModel):
    pass


class Provider(OfflineLLMClient):
    def __init__(self, responses: list[LLMToolResponse]) -> None:
        self.responses = list(responses)
        self.settings = LLMSettings(model="test-model", max_tokens=100)
        self.calls = 0
        self.messages: list[list[dict[str, Any]]] = []

    def tool_call(self, **kwargs: Any) -> LLMToolResponse:
        self.calls += 1
        self.messages.append(kwargs["messages"])
        return self.responses.pop(0)


def setup(tmp_path: Path) -> tuple[ArtifactStore, GraphPersistence, Artifact]:
    store = ArtifactStore(tmp_path)
    store.ensure_project("p", "Project")
    store.start_session("p", "s")
    persistence = GraphPersistence(store.session_dir("p", "s"), "test-agent")
    artifact = Artifact(
        id="sql-result", type=ArtifactType.SQL_RESULT, project_id="p", session_id="s",
        payload={"rows": [{"value": 42}]},
    )
    return store, persistence, artifact


def context(store: ArtifactStore) -> DataToolContext:
    # These tools exercise evidence lookup only; no query engine is needed.
    return DataToolContext(
        datasets=[], catalog=cast(Any, None), project_id="p", session_id="s",
        store=store, payload_policy="schema+aggregates",
    )


@pytest.mark.parametrize("boundary", ["tool_commit", "next_tool"])
def test_tool_commit_survives_checkpoint_loss_and_hydrates_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str,
) -> None:
    store, persistence, artifact = setup(tmp_path)
    produced = consumed = 0
    crashed = False
    original_effect = GraphExecution.effect

    def effect(self: GraphExecution, key: str, request: dict, call: Any, **kwargs: Any) -> dict:
        nonlocal crashed
        result = original_effect(self, key, request, call, **kwargs)
        if boundary == "tool_commit" and key == "tool:1" and not crashed:
            crashed = True
            raise WorkerLost()
        return result

    monkeypatch.setattr(GraphExecution, "effect", effect)

    def run(client: Provider, ctx: DataToolContext):
        def produce(_args: BaseModel) -> AgentToolResult:
            nonlocal produced
            produced += 1
            return AgentToolResult({"artifact_id": artifact.id}, artifacts=[artifact])

        def consume(_args: BaseModel) -> AgentToolResult:
            nonlocal consumed, crashed
            if boundary == "next_tool" and not crashed:
                crashed = True
                raise WorkerLost()
            consumed += 1
            assert ctx.artifact(artifact.id).payload["rows"][0]["value"] == 42
            return AgentToolResult({"value": 42})

        return AgentRuntime(
            llm=client,
            tools=[
                AgentTool("produce", "Produce", NoArguments, produce, retry_safe=True),
                AgentTool("consume", "Consume", NoArguments, consume, retry_safe=True),
            ],
            artifact_store=store, persistence=persistence, restore_artifacts=ctx.restore_artifacts,
            answer_validator=lambda _answer, evidence: (
                any(item.id == artifact.id for item in evidence), "missing evidence",
            ),
        ).run(system_prompt="Analyze", user_message="Describe")

    first = Provider([LLMToolResponse(tool_calls=[
        LLMToolCall(call_id="produce-1", name="produce", arguments={}),
        LLMToolCall(call_id="consume-1", name="consume", arguments={}),
    ])])
    with pytest.raises(WorkerLost):
        run(first, context(store))
    assert produced == 1
    saved_artifact = store.get_artifact(artifact.id, project_id="p", session_id="s")
    assert saved_artifact.payload == artifact.payload
    second = Provider([LLMToolResponse(content="The evidence says 42.")])
    fresh_context = context(store)
    result = run(second, fresh_context)
    assert result.status == "completed"
    assert result.tool_calls == 2
    assert produced == consumed == 1
    assert first.calls == second.calls == 1
    assert [item.id for item in result.artifacts] == [artifact.id]
    assert fresh_context.artifact(artifact.id).payload == artifact.payload
    assert second.messages[0][-2]["tool_call_id"] == "produce-1"
    assert second.messages[0][-1]["tool_call_id"] == "consume-1"
    third = Provider([])
    assert run(third, context(store)) == result
    assert third.calls == 0
    assert produced == consumed == 1


@pytest.mark.parametrize("error_type", [GraphEffectUncertain, GraphIdentityError])
def test_nested_graph_failure_is_terminal_not_a_tool_retry(
    tmp_path: Path, error_type: type[RuntimeError],
) -> None:
    store, persistence, _artifact = setup(tmp_path)
    provider = Provider([LLMToolResponse(tool_calls=[
        LLMToolCall(call_id="nested", name="nested", arguments={}),
    ])])

    def nested(_args: BaseModel) -> AgentToolResult:
        raise error_type("Nested execution cannot safely continue")

    with pytest.raises(error_type):
        AgentRuntime(
            llm=provider, tools=[AgentTool("nested", "Nested", NoArguments, nested)],
            persistence=persistence, artifact_store=store,
        ).run(system_prompt="Analyze", user_message="Describe")
    assert provider.calls == 1


def test_completed_agent_rejects_model_configuration_drift(tmp_path: Path) -> None:
    store, persistence, _artifact = setup(tmp_path)
    provider = Provider([LLMToolResponse(content="Complete")])

    def run(client: Provider):
        return AgentRuntime(
            llm=client, tools=[], persistence=persistence, artifact_store=store,
        ).run(system_prompt="Analyze", user_message="Describe")

    assert run(provider).status == "completed"
    changed = Provider([])
    changed.settings.temperature += 0.1
    with pytest.raises(GraphIdentityError):
        run(changed)
    assert changed.calls == 0


def test_restored_evidence_rejects_overwritten_artifact_payload(tmp_path: Path) -> None:
    store, persistence, artifact = setup(tmp_path)
    provider = Provider([
        LLMToolResponse(tool_calls=[LLMToolCall(call_id="produce", name="produce", arguments={})]),
        LLMToolResponse(content="The evidence says 42."),
    ])

    def run(client: Provider):
        return AgentRuntime(
            llm=client, tools=[AgentTool(
                "produce", "Produce", NoArguments,
                lambda _args: AgentToolResult({"value": 42}, artifacts=[artifact]),
            )], persistence=persistence, artifact_store=store,
        ).run(system_prompt="Analyze", user_message="Describe")

    assert run(provider).status == "completed"
    overwritten = artifact.model_copy(deep=True)
    overwritten.payload["rows"][0]["value"] = 99
    store.save_artifact(overwritten)
    with pytest.raises(GraphIdentityError):
        run(Provider([]))


def test_pending_tool_does_not_gain_retry_permission_on_restart(tmp_path: Path) -> None:
    store, persistence, _artifact = setup(tmp_path)
    client = Provider([LLMToolResponse(tool_calls=[
        LLMToolCall(call_id="effect", name="effect", arguments={}),
    ])])
    calls = 0

    def pending(_args: BaseModel) -> AgentToolResult:
        nonlocal calls
        calls += 1
        raise WorkerLost()

    def run(provider: Provider, *, retry_safe: bool):
        return AgentRuntime(
            llm=provider,
            tools=[AgentTool("effect", "Effect", NoArguments, pending, retry_safe=retry_safe)],
            persistence=persistence, artifact_store=store,
        ).run(system_prompt="Analyze", user_message="Describe")

    with pytest.raises(WorkerLost):
        run(client, retry_safe=False)
    with pytest.raises(GraphIdentityError):
        run(Provider([]), retry_safe=True)
    assert calls == 1


@pytest.mark.parametrize("boundary", ["model", "validator"])
def test_stop_before_final_publication_returns_cancelled(tmp_path: Path, boundary: str) -> None:
    cancelled = False
    events: list[dict[str, Any]] = []

    class StopProvider(Provider):
        def tool_call(self, **kwargs: Any) -> LLMToolResponse:
            nonlocal cancelled
            response = super().tool_call(**kwargs)
            if boundary == "model":
                cancelled = True
            return response

    def validate(answer: str, artifacts: list[Any]) -> tuple[bool, str]:
        nonlocal cancelled
        cancelled = True
        return True, ""

    provider = StopProvider([LLMToolResponse(content="Final answer")])
    result = AgentRuntime(
        llm=provider, tools=[], cancel_check=lambda: cancelled,
        answer_validator=validate if boundary == "validator" else None,
        persistence=GraphPersistence(tmp_path, "cancel-final"),
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        trace=lambda event, name, summary: events.append({"event": event, **summary}),
    ).run(system_prompt="Analyze", user_message="Report")
    assert result.status == "cancelled"
    assert result.answer == ""
    assert provider.calls == 1
    assert any(event["event"] == "agent_cancelled" for event in events)
    assert not any(event["event"] == "agent_completed" for event in events)
