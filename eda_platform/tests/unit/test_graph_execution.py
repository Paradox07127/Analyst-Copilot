from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from eda_platform.agents.runtime import AgentRuntime, AgentTool, AgentToolResult
from eda_platform.core.graph_execution import (
    GraphEffectUncertain,
    GraphIdentityError,
    GraphPersistence,
    graph_execution,
)
from eda_platform.core.llm import LLMToolCall, LLMToolResponse, OfflineLLMClient
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType


class Crash(BaseException):
    pass


class NoArgs(BaseModel):
    pass


class Provider(OfflineLLMClient):
    def __init__(self, responses: list[LLMToolResponse]) -> None:
        self.responses = responses
        self.calls = 0

    def tool_call(self, **kwargs) -> LLMToolResponse:
        self.calls += 1
        return self.responses.pop(0)


def test_agent_reopens_checkpoint_and_resumes_without_repeating_model(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    persistence = GraphPersistence(tmp_path, "project/session/turn")
    first = Provider(
        [
            LLMToolResponse(
                tool_calls=[
                    LLMToolCall(call_id="inspect-1", name="inspect", arguments={}),
                ]
            )
        ]
    )

    def crash(_args):
        raise Crash()

    with pytest.raises(Crash):
        AgentRuntime(
            llm=first,
            tools=[AgentTool("inspect", "Inspect", NoArgs, crash, retry_safe=True)],
            persistence=persistence,
            artifact_store=store,
        ).run(system_prompt="Analyze", user_message="Inspect")
    second = Provider([LLMToolResponse(content="Finished")])
    result = AgentRuntime(
        llm=second,
        tools=[
            AgentTool(
                "inspect",
                "Inspect",
                NoArgs,
                lambda _args: AgentToolResult({"rows": 1}),
                retry_safe=True,
            )
        ],
        persistence=persistence,
        artifact_store=store,
    ).run(system_prompt="Analyze", user_message="Inspect")
    assert result.status == "completed"
    assert result.tool_calls == 1
    assert second.calls == 1
    third = Provider([])
    repeated = AgentRuntime(
        llm=third,
        tools=[AgentTool("inspect", "Inspect", NoArgs, crash, retry_safe=True)],
        persistence=persistence,
        artifact_store=store,
    ).run(system_prompt="Analyze", user_message="Inspect")
    assert repeated == result
    assert third.calls == 0


def test_unknown_remote_outcome_is_not_repeated(tmp_path: Path) -> None:
    persistence = GraphPersistence(tmp_path, "unknown")

    def crash():
        raise Crash()

    with graph_execution(persistence, definition="test", inputs={}) as execution:
        with pytest.raises(Crash):
            execution.model_effect("call", {"prompt": "x"}, crash)
    with graph_execution(persistence, definition="test", inputs={}) as execution:
        with pytest.raises(GraphEffectUncertain):
            execution.model_effect("call", {"prompt": "x"}, lambda: pytest.fail("resent"))


def test_saved_remote_result_survives_missing_graph_checkpoint(tmp_path: Path) -> None:
    persistence = GraphPersistence(tmp_path, "saved")
    with graph_execution(persistence, definition="test", inputs={}) as execution:
        assert execution.model_effect("call", {}, lambda: {"answer": 42}) == {"answer": 42}
    with graph_execution(persistence, definition="test", inputs={}) as execution:
        assert execution.model_effect("call", {}, lambda: pytest.fail("resent")) == {"answer": 42}
        with pytest.raises(GraphIdentityError):
            execution.model_effect("call", {"different": True}, lambda: {})


def test_changed_inputs_cannot_adopt_old_execution(tmp_path: Path) -> None:
    persistence = GraphPersistence(tmp_path, "identity")
    with graph_execution(persistence, definition="test", inputs={"data": "a"}):
        pass
    with (
        pytest.raises(GraphIdentityError),
        graph_execution(
            persistence,
            definition="test",
            inputs={"data": "b"},
        ),
    ):
        pass


def test_policy_epoch_cannot_replay_old_admitted_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import eda_platform.core.graph_execution as runtime

    persistence = GraphPersistence(tmp_path, "report")
    with monkeypatch.context() as previous:
        previous.setattr(runtime, "GRAPH_VERSION", "previous-policy")
        with graph_execution(persistence, definition="report", inputs={}) as execution:
            execution.model_effect("draft", {}, lambda: {"old_narrative": "unverified"})
    with pytest.raises(GraphIdentityError), graph_execution(
        persistence, definition="report", inputs={},
    ):
        pytest.fail("Old policy results must not be adopted or automatically recomputed.")


def test_repeated_content_addressed_tool_output_stays_one_verified_reference(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path)
    store.ensure_project("p", "Project")
    store.start_session("p", "s")
    provider = Provider(
        [
            LLMToolResponse(tool_calls=[LLMToolCall(call_id="one", name="inspect", arguments={})]),
            LLMToolResponse(tool_calls=[LLMToolCall(call_id="two", name="inspect", arguments={})]),
            LLMToolResponse(content="Done"),
        ]
    )

    def inspect(_args):
        return AgentToolResult(
            {"rows": 1},
            [
                Artifact(
                    id="table_same",
                    type=ArtifactType.TABLE,
                    project_id="p",
                    session_id="s",
                    payload={"rows": [{"count": 1}]},
                )
            ],
        )

    runtime = AgentRuntime(
        llm=provider,
        tools=[AgentTool("inspect", "Inspect", NoArgs, inspect, retry_safe=True)],
        persistence=GraphPersistence(tmp_path, "repeated"),
        artifact_store=store,
        restore_artifacts=lambda artifacts: None,
    )
    result = runtime.run(system_prompt="Analyze", user_message="Inspect twice")
    assert result.status == "completed"
    assert len(result.artifacts) == 1
    assert provider.calls == 3
    assert runtime.run(system_prompt="Analyze", user_message="Inspect twice") == result
    assert provider.calls == 3
