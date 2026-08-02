from __future__ import annotations

from typing import Any, cast

import pytest
from pydantic import BaseModel, ConfigDict

from eda_platform.agents.runtime import AgentRuntime, AgentTool, AgentToolResult
from eda_platform.core.budget import BudgetExceeded
from eda_platform.core.cancellation import (
    CancellationCause,
    CancellationError,
    CancellationSnapshot,
    KillFenceState,
)
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

    assert not supports_tool_calling(
        cast(Any, _Wrapper(_Wrapper(OfflineLLMClient())))
    )


class _CountingToolLLM(_ScriptedToolLLM):
    """Scripted client that also records the tool observations fed back to it."""

    def tool_call(
        self,
        *,
        task: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMToolResponse:
        self.messages.append([dict(message) for message in messages])
        return self.responses.pop(0)


def test_a_budget_error_inside_a_tool_stops_the_loop() -> None:
    """Budget and cancellation are terminal: replaying the call would keep spending."""

    def _overspend(_args: BaseModel) -> AgentToolResult:
        raise BudgetExceeded("token budget exhausted")

    llm = _CountingToolLLM(
        [
            LLMToolResponse(
                tool_calls=[LLMToolCall(call_id="call_1", name="inspect", arguments={})]
            ),
            LLMToolResponse(content="Recovered without the tool."),
        ]
    )
    runtime = AgentRuntime(
        llm=llm,  # type: ignore[arg-type]
        tools=[
            AgentTool(
                name="inspect",
                description="Inspect the catalog.",
                args_schema=_NoArgs,
                execute=_overspend,
            )
        ],
    )

    with pytest.raises(BudgetExceeded):
        runtime.run(system_prompt="Use tools.", user_message="Inspect it.")


def test_a_cancelled_tool_stops_the_loop() -> None:
    def _cancelled(_args: BaseModel) -> AgentToolResult:
        raise CancellationError(
            CancellationSnapshot(
                cause=CancellationCause.CANCEL_REQUESTED,
                reason="the user cancelled the run",
                deadline=None,
                shield_depth=0,
                kill_fence_state=KillFenceState.ELIGIBLE,
            )
        )

    llm = _CountingToolLLM(
        [
            LLMToolResponse(
                tool_calls=[LLMToolCall(call_id="call_1", name="inspect", arguments={})]
            ),
            LLMToolResponse(content="Recovered without the tool."),
        ]
    )
    runtime = AgentRuntime(
        llm=llm,  # type: ignore[arg-type]
        tools=[
            AgentTool(
                name="inspect",
                description="Inspect the catalog.",
                args_schema=_NoArgs,
                execute=_cancelled,
            )
        ],
    )

    with pytest.raises(CancellationError):
        runtime.run(system_prompt="Use tools.", user_message="Inspect it.")


def test_a_rejected_answer_is_returned_to_the_model_once() -> None:
    """The rewrite feedback carries the reason, never the rejected answer itself."""
    llm = _CountingToolLLM(
        [
            LLMToolResponse(content="Revenue was 999."),
            LLMToolResponse(content="Revenue was 42."),
        ]
    )
    runtime = AgentRuntime(
        llm=llm,  # type: ignore[arg-type]
        tools=[
            AgentTool(
                name="inspect",
                description="Inspect the catalog.",
                args_schema=_NoArgs,
                execute=lambda _args: AgentToolResult(content={"ok": True}),
            )
        ],
        answer_validator=lambda answer, _artifacts: (
            ("42" in answer),
            "unsupported_number: 999 not traceable to tool evidence",
        ),
    )

    result = runtime.run(system_prompt="Use tools.", user_message="How much revenue?")

    assert result.status == "completed"
    assert result.answer == "Revenue was 42."
    feedback = llm.messages[1][-1]
    assert feedback["role"] == "user"
    assert "unsupported_number" in str(feedback["content"])
    assert "Revenue was 999." not in str(feedback["content"])


def test_a_twice_rejected_answer_is_never_returned() -> None:
    llm = _CountingToolLLM(
        [
            LLMToolResponse(content="Revenue was 999."),
            LLMToolResponse(content="Revenue was 998."),
        ]
    )
    runtime = AgentRuntime(
        llm=llm,  # type: ignore[arg-type]
        tools=[
            AgentTool(
                name="inspect",
                description="Inspect the catalog.",
                args_schema=_NoArgs,
                execute=lambda _args: AgentToolResult(content={"ok": True}),
            )
        ],
        answer_validator=lambda _answer, _artifacts: (False, "unsupported_number: fabricated"),
    )

    result = runtime.run(system_prompt="Use tools.", user_message="How much revenue?")

    assert result.status == "answer_unverified"
    assert result.answer == ""
    assert result.error is not None
    assert "unsupported_number" in result.error
