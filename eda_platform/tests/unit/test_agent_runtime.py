from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from eda_platform.agents.runtime import AgentRuntime, AgentTool, AgentToolResult
from eda_platform.core.llm import (
    LLMToolCall,
    LLMToolResponse,
    OfflineLLMClient,
    supports_tool_calling,
)


class _NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _ScriptedToolLLM:
    def __init__(self, responses: list[LLMToolResponse]) -> None:
        self.responses = list(responses)
        self.messages: list[list[dict[str, Any]]] = []

    def tool_call(
        self,
        *,
        task: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMToolResponse:
        assert task == "chat_agent_tool_loop"
        assert any(tool["name"] == "inspect" for tool in tools)
        self.messages.append(messages)
        return self.responses.pop(0)


def test_agent_runtime_returns_tool_observation_then_final_answer() -> None:
    events: list[tuple[str, str, dict[str, Any]]] = []
    llm = _ScriptedToolLLM(
        [
            LLMToolResponse(
                tool_calls=[LLMToolCall(call_id="call_1", name="inspect", arguments={})]
            ),
            LLMToolResponse(content="The catalog has one dataset. Sources: artifact_catalog."),
        ]
    )
    runtime = AgentRuntime(
        llm=llm,  # type: ignore[arg-type]
        tools=[
            AgentTool(
                name="inspect",
                description="Inspect the catalog.",
                args_schema=_NoArgs,
                execute=lambda _args: AgentToolResult(
                    content={"dataset_count": 1, "artifact_id": "artifact_catalog"}
                ),
            )
        ],
        trace=lambda event_type, name, summary: events.append((event_type, name, summary)),
    )

    result = runtime.run(system_prompt="Use tools.", user_message="What data is loaded?")

    assert result.status == "completed"
    assert result.tool_calls == 1
    assert "one dataset" in result.answer
    assert [event[0] for event in events] == ["tool_started", "tool_completed", "agent_completed"]
    observation = llm.messages[1][-1]
    assert observation["role"] == "tool"
    assert '"dataset_count": 1' in str(observation["content"])


def test_agent_runtime_replays_provider_reasoning_state() -> None:
    llm = _ScriptedToolLLM(
        [
            LLMToolResponse(
                tool_calls=[LLMToolCall(call_id="call_1", name="inspect", arguments={})],
                provider_state={"reasoning_content": "provider-opaque-state"},
            ),
            LLMToolResponse(content="Done."),
        ]
    )
    runtime = AgentRuntime(
        llm=llm,  # type: ignore[arg-type]
        tools=[
            AgentTool(
                name="inspect",
                description="Inspect.",
                args_schema=_NoArgs,
                execute=lambda _args: AgentToolResult(content={"ok": True}),
            )
        ],
    )

    runtime.run(system_prompt="Use tools.", user_message="Inspect.")

    assistant = llm.messages[1][-2]
    assert assistant["reasoning_content"] == "provider-opaque-state"


def test_agent_runtime_returns_unknown_tool_error_to_model() -> None:
    llm = _ScriptedToolLLM(
        [
            LLMToolResponse(
                tool_calls=[LLMToolCall(call_id="call_bad", name="not_registered", arguments={})]
            ),
            LLMToolResponse(content="That tool is unavailable, so I cannot complete the request."),
        ]
    )
    runtime = AgentRuntime(
        llm=llm,  # type: ignore[arg-type]
        tools=[
            AgentTool(
                name="inspect",
                description="Inspect the catalog.",
                args_schema=_NoArgs,
                execute=lambda _args: AgentToolResult(content={}),
            )
        ],
    )

    result = runtime.run(system_prompt="Use registered tools only.", user_message="Try a tool.")

    assert result.status == "completed"
    assert "unavailable" in result.answer
    observation = str(llm.messages[1][-1]["content"])
    assert "Unknown tool" in observation


def test_capability_check_uses_the_provider_inside_pass_through_wrappers() -> None:
    class _LegacyProvider:
        pass

    class _PassThroughWrapper:
        def __init__(self, inner: object) -> None:
            self.inner = inner

        def tool_call(self, **_kwargs: object) -> None:
            raise AssertionError("A pass-through wrapper cannot add provider capability.")

    assert supports_tool_calling(_PassThroughWrapper(_LegacyProvider())) is False  # type: ignore[arg-type]


def test_offline_client_does_not_advertise_tool_calling() -> None:
    """`OfflineLLMClient` defines `tool_call` only to raise, so a plain
    "is the method there?" check answered True and every agent gate had to
    remember `not is_offline_client(x) and supports_tool_calling(x)`. The
    exclusion belongs in the predicate, not in each caller's memory."""
    offline = OfflineLLMClient()

    assert callable(offline.tool_call)
    assert not supports_tool_calling(offline)
    assert not supports_tool_calling(None)


def test_offline_exclusion_survives_client_decoration() -> None:
    """Clients get wrapped by the spend ledger and the dev-log recorder, so the
    check has to unwrap before deciding."""

    class _Wrapper:
        def __init__(self, inner: Any) -> None:
            self.inner = inner

        def tool_call(self, **kwargs: Any) -> Any:  # pragma: no cover - never run
            return self.inner.tool_call(**kwargs)

    assert not supports_tool_calling(_Wrapper(_Wrapper(OfflineLLMClient())))
