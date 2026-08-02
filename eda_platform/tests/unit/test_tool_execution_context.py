"""Executor-injected call identity: receipts must not be able to forge it.

The runtime (executor) is the only party that knows the provider call id, the
run, the attempt and the position of a call inside the run. Tools receive that
identity through an ambient execution scope; a tool invoked outside any
executor gets a locally minted identity that is unique per invocation and
never derivable from the tool name or arguments.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from eda_platform.agents.runtime import AgentRuntime, AgentTool, AgentToolResult
from eda_platform.agents.tool_context import (
    ToolExecutionContext,
    current_execution_context,
    make_logical_step_id,
    mint_local_execution_context,
    tool_execution_scope,
)
from eda_platform.core.llm import LLMToolCall, LLMToolResponse
from eda_platform.schemas.artifacts import Artifact, ArtifactType


class _NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _ScriptedToolLLM:
    def __init__(self, responses: list[LLMToolResponse]) -> None:
        self.responses = list(responses)

    def tool_call(self, **_kwargs: Any) -> LLMToolResponse:
        return self.responses.pop(0)


def test_scope_sets_and_resets_the_active_context() -> None:
    assert current_execution_context() is None
    context = ToolExecutionContext(
        run_id="run_a",
        provider_call_id="call_1",
        logical_step_id="step_x",
        attempt_epoch=0,
        sequence_index=1,
    )
    with tool_execution_scope(context):
        assert current_execution_context() is context
    assert current_execution_context() is None


def test_minted_local_contexts_are_unique_even_for_the_same_run() -> None:
    first = mint_local_execution_context("local:p:s")
    second = mint_local_execution_context("local:p:s")
    assert first.provider_call_id != second.provider_call_id
    assert first.logical_step_id != second.logical_step_id
    assert first.call_identity() != second.call_identity()


def test_call_identity_carries_executor_owned_entropy() -> None:
    """provider_call_id is model/provider text and repeats.

    Anthropic and OpenAI adapters fall back to the positional `tool_{i+1}` when
    the provider omits ids, and a model can emit the same id twice, so the
    position of the call inside the run is the only entropy the executor owns.
    """
    def _context(provider_call_id: str, sequence_index: int) -> ToolExecutionContext:
        return ToolExecutionContext(
            run_id="run_abc",
            provider_call_id=provider_call_id,
            logical_step_id=make_logical_step_id(
                "run_abc", provider_call_id, sequence_index
            ),
            sequence_index=sequence_index,
        )

    first = _context("tool_1", 1)
    fifth = _context("tool_1", 5)
    assert first.call_identity() != fifth.call_identity()
    assert first.call_identity().endswith(".1")
    assert fifth.call_identity().endswith(".5")
    # Replaying the same logical step must reproduce the same identity.
    assert _context("tool_1", 5).call_identity() == fifth.call_identity()


def test_runtime_injects_the_provider_call_identity() -> None:
    seen: list[ToolExecutionContext | None] = []

    def capture(_args: BaseModel) -> AgentToolResult:
        seen.append(current_execution_context())
        return AgentToolResult(content={"ok": True})

    llm = _ScriptedToolLLM(
        [
            LLMToolResponse(
                tool_calls=[
                    LLMToolCall(call_id="prov_call_a", name="capture", arguments={}),
                    LLMToolCall(call_id="prov_call_b", name="capture", arguments={}),
                ]
            ),
            LLMToolResponse(content="Done."),
        ]
    )
    runtime = AgentRuntime(
        llm=llm,  # type: ignore[arg-type]
        tools=[
            AgentTool(
                name="capture",
                description="Capture the execution context.",
                args_schema=_NoArgs,
                execute=capture,
            )
        ],
    )
    runtime.run(system_prompt="Use tools.", user_message="Go.")

    assert len(seen) == 2
    first, second = seen
    assert first is not None and second is not None
    assert first.provider_call_id == "prov_call_a"
    assert second.provider_call_id == "prov_call_b"
    assert first.run_id == second.run_id
    assert first.sequence_index == 1
    assert second.sequence_index == 2
    assert first.logical_step_id != second.logical_step_id
    # The scope must not leak past the invocation.
    assert current_execution_context() is None


def test_two_runs_have_distinct_run_ids() -> None:
    seen: list[ToolExecutionContext | None] = []

    def capture(_args: BaseModel) -> AgentToolResult:
        seen.append(current_execution_context())
        return AgentToolResult(content={"ok": True})

    def _runtime() -> AgentRuntime:
        llm = _ScriptedToolLLM(
            [
                LLMToolResponse(
                    tool_calls=[LLMToolCall(call_id="call_1", name="capture", arguments={})]
                ),
                LLMToolResponse(content="Done."),
            ]
        )
        return AgentRuntime(
            llm=llm,  # type: ignore[arg-type]
            tools=[
                AgentTool(
                    name="capture",
                    description="Capture.",
                    args_schema=_NoArgs,
                    execute=capture,
                )
            ],
        )

    _runtime().run(system_prompt="s", user_message="u")
    _runtime().run(system_prompt="s", user_message="u")
    assert seen[0] is not None and seen[1] is not None
    # Same provider call id in two different runs must still be distinguishable.
    assert seen[0].run_id != seen[1].run_id
    assert seen[0].call_identity() != seen[1].call_identity()


def test_receipt_artifact_is_surfaced_in_trace_and_result() -> None:
    receipt_artifact = Artifact(
        id="receipt_abc",
        type=ArtifactType.EVIDENCE_RECEIPT,
        project_id="p",
        session_id="s",
        payload={"receipt_id": "rcpt_x"},
    )
    primary = Artifact(
        id="table_abc",
        type=ArtifactType.TABLE,
        project_id="p",
        session_id="s",
        payload={},
    )
    events: list[tuple[str, str, dict[str, Any]]] = []
    llm = _ScriptedToolLLM(
        [
            LLMToolResponse(
                tool_calls=[LLMToolCall(call_id="call_1", name="analyze", arguments={})]
            ),
            LLMToolResponse(content="Done."),
        ]
    )
    runtime = AgentRuntime(
        llm=llm,  # type: ignore[arg-type]
        tools=[
            AgentTool(
                name="analyze",
                description="Analyze.",
                args_schema=_NoArgs,
                execute=lambda _args: AgentToolResult(
                    content={"ok": True},
                    artifacts=[primary],
                    receipt_artifact=receipt_artifact,
                ),
            )
        ],
        trace=lambda event_type, name, summary: events.append((event_type, name, summary)),
    )
    result = runtime.run(system_prompt="s", user_message="u")

    completed = next(event for event in events if event[0] == "tool_completed")
    assert completed[2]["receipt_artifact_id"] == "receipt_abc"
    assert "receipt_abc" in completed[2]["artifact_ids"]
    assert "table_abc" in completed[2]["artifact_ids"]
    result_ids = [artifact.id for artifact in result.artifacts]
    assert "receipt_abc" in result_ids and "table_abc" in result_ids
