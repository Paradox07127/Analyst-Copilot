"""A small, provider-neutral runtime for bounded, tool-using agents.

The product already owns important boundaries (workspace scoping, SQL safety,
approval, sandboxing, budget accounting and trace storage).  This module adds
the missing orchestration seam without replacing those boundaries with an
opaque framework.  Providers only choose *which* registered tool to ask for;
the local registry validates and executes every request.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from eda_platform.core.llm import LLMToolCall, ToolCallingLLM

TraceSink = Callable[[str, str, dict[str, Any]], None]
ToolExecutor = Callable[[BaseModel], "AgentToolResult"]


@dataclass(frozen=True, slots=True)
class AgentTool:
    """One local capability exposed to a model as a typed function."""

    name: str
    description: str
    args_schema: type[BaseModel]
    execute: ToolExecutor

    def provider_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.args_schema.model_json_schema(),
        }


@dataclass(slots=True)
class AgentToolResult:
    """A serialisable observation returned to the agent after one tool call."""

    content: dict[str, Any] | str
    artifacts: list[Any] = field(default_factory=list)


@dataclass(slots=True)
class AgentRunResult:
    status: str
    answer: str = ""
    artifacts: list[Any] = field(default_factory=list)
    tool_calls: int = 0
    tool_names: list[str] = field(default_factory=list)
    error: str | None = None


class AgentRuntime:
    """Execute a bounded ReAct-style loop over locally registered tools.

    Bounds are deliberate product policy, not prompt suggestions: a model can
    neither spin indefinitely nor invoke an unregistered function. Every error
    is returned as an observation, letting the model repair arguments once
    without exposing a Python traceback to the chat user.
    """

    def __init__(
        self,
        *,
        llm: ToolCallingLLM,
        tools: list[AgentTool],
        task: str = "chat_agent_tool_loop",
        max_steps: int = 8,
        max_tool_calls: int = 12,
        max_observation_chars: int = 12_000,
        trace: TraceSink | None = None,
    ) -> None:
        if max_steps < 1 or max_tool_calls < 1:
            raise ValueError("Agent runtime limits must be positive.")
        names = [tool.name for tool in tools]
        if len(names) != len(set(names)):
            raise ValueError("Agent tool names must be unique.")
        self._llm = llm
        self._tools = {tool.name: tool for tool in tools}
        self._task = task
        self._max_steps = max_steps
        self._max_tool_calls = max_tool_calls
        self._max_observation_chars = max_observation_chars
        self._trace = trace

    def run(self, *, system_prompt: str, user_message: str) -> AgentRunResult:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        all_artifacts: list[Any] = []
        tool_calls = 0
        tool_names: list[str] = []

        for step in range(1, self._max_steps + 1):
            response = self._llm.tool_call(
                task=self._task,
                messages=messages,
                tools=[tool.provider_schema() for tool in self._tools.values()],
            )
            if not response.tool_calls:
                answer = response.content.strip()
                if answer:
                    self._emit(
                        "agent_completed",
                        "agent_runtime",
                        {"step": step, "tool_calls": tool_calls},
                    )
                    return AgentRunResult(
                        status="completed",
                        answer=answer,
                        artifacts=_unique_artifacts(all_artifacts),
                        tool_calls=tool_calls,
                        tool_names=tool_names,
                    )
                return AgentRunResult(
                    status="failed",
                    artifacts=_unique_artifacts(all_artifacts),
                    tool_calls=tool_calls,
                    tool_names=tool_names,
                    error="The model ended the agent turn without an answer or a tool call.",
                )

            if tool_calls + len(response.tool_calls) > self._max_tool_calls:
                self._emit(
                    "agent_limit_reached",
                    "agent_runtime",
                    {
                        "step": step,
                        "tool_calls": tool_calls,
                        "tool_call_cap": self._max_tool_calls,
                    },
                )
                return AgentRunResult(
                    status="limit_reached",
                    artifacts=_unique_artifacts(all_artifacts),
                    tool_calls=tool_calls,
                    tool_names=tool_names,
                    error=(
                        "The agent reached its tool-call safety limit before producing "
                        "a final answer."
                    ),
                )

            messages.append(
                _assistant_message(
                    response.content,
                    response.tool_calls,
                    response.provider_state,
                )
            )
            for call in response.tool_calls:
                tool_calls += 1
                tool_names.append(call.name)
                observation, artifacts = self._invoke(call, step=step)
                all_artifacts.extend(artifacts)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.call_id,
                        "name": call.name,
                        "content": _observation_text(
                            observation,
                            limit=self._max_observation_chars,
                        ),
                    }
                )

        self._emit(
            "agent_limit_reached",
            "agent_runtime",
            {
                "step": self._max_steps,
                "tool_calls": tool_calls,
                "step_cap": self._max_steps,
            },
        )
        return AgentRunResult(
            status="limit_reached",
            artifacts=_unique_artifacts(all_artifacts),
            tool_calls=tool_calls,
            tool_names=tool_names,
            error=(
                "The agent reached its reasoning-step safety limit before producing "
                "a final answer."
            ),
        )

    def _invoke(
        self,
        call: LLMToolCall,
        *,
        step: int,
    ) -> tuple[dict[str, Any], list[Any]]:
        tool = self._tools.get(call.name)
        self._emit(
            "tool_started",
            call.name,
            {
                "call_id": call.call_id,
                "step": step,
                "arguments": _safe_value(call.arguments),
            },
        )
        if tool is None:
            observation = {
                "ok": False,
                "error": f"Unknown tool '{call.name}'. Choose only a registered tool.",
            }
            self._emit(
                "tool_failed",
                call.name,
                {"call_id": call.call_id, "step": step, "error": observation["error"]},
            )
            return observation, []
        try:
            args = tool.args_schema.model_validate(call.arguments)
        except ValidationError as exc:
            observation = {
                "ok": False,
                "error": "Tool arguments did not match the declared schema.",
                "details": _validation_feedback(exc),
            }
            self._emit(
                "tool_failed",
                tool.name,
                {"call_id": call.call_id, "step": step, "error": observation["error"]},
            )
            return observation, []
        try:
            result = tool.execute(args)
        except Exception as exc:  # tool errors are repairable observations
            observation = {
                "ok": False,
                "error": _safe_error(exc),
            }
            self._emit(
                "tool_failed",
                tool.name,
                {"call_id": call.call_id, "step": step, "error": observation["error"]},
            )
            return observation, []
        content = result.content if isinstance(result.content, dict) else {"result": result.content}
        observation = {"ok": True, **content}
        self._emit(
            "tool_completed",
            tool.name,
            {
                "call_id": call.call_id,
                "step": step,
                "artifact_ids": [getattr(artifact, "id", "") for artifact in result.artifacts],
                "summary": _safe_value(content),
            },
        )
        return observation, result.artifacts

    def _emit(self, event_type: str, name: str, summary: dict[str, Any]) -> None:
        if self._trace is not None:
            self._trace(event_type, name, summary)


def _assistant_message(
    content: str,
    calls: list[LLMToolCall],
    provider_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": call.call_id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call in calls
        ],
    }
    reasoning_content = (provider_state or {}).get("reasoning_content")
    if isinstance(reasoning_content, str):
        message["reasoning_content"] = reasoning_content
    return message


def _observation_text(observation: dict[str, Any], *, limit: int) -> str:
    text = json.dumps(observation, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    return json.dumps(
        {
            "ok": observation.get("ok", False),
            "truncated": True,
            "message": f"Tool observation was clipped to {limit} characters.",
            "preview": text[:limit],
        },
        ensure_ascii=False,
    )


def _validation_feedback(exc: ValidationError) -> list[dict[str, Any]]:
    return [
        {
            "field": ".".join(str(part) for part in error.get("loc", ())),
            "message": str(error.get("msg", "invalid value")),
        }
        for error in exc.errors(include_url=False)[:8]
    ]


def _safe_error(exc: Exception) -> str:
    # Tool guard feedback is intentionally useful to the model; arbitrary
    # transport tracebacks are not. Preserve one concise line only.
    text = " ".join(str(exc).split())
    return text[:800] if text else type(exc).__name__


def _safe_value(value: Any) -> Any:
    if isinstance(value, str):
        return value[:1_000]
    if isinstance(value, list):
        return [_safe_value(item) for item in value[:20]]
    if isinstance(value, dict):
        return {str(key): _safe_value(item) for key, item in list(value.items())[:30]}
    return value


def _unique_artifacts(artifacts: list[Any]) -> list[Any]:
    seen: set[str] = set()
    unique: list[Any] = []
    for artifact in artifacts:
        artifact_id = str(getattr(artifact, "id", ""))
        key = artifact_id or str(id(artifact))
        if key in seen:
            continue
        seen.add(key)
        unique.append(artifact)
    return unique
