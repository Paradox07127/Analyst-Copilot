"""LangGraph tool-agent nodes over the project's typed execution contracts."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypedDict, cast

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from pydantic import BaseModel, ValidationError

from eda_platform.agents.tool_context import (
    ToolExecutionContext,
    make_logical_step_id,
    tool_execution_scope,
)
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
    LLMToolCall,
    LLMToolResponse,
    ToolCallingLLM,
    llm_execution_fingerprint,
)
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact

TraceSink = Callable[[str, str, dict[str, Any]], None]
ToolExecutor = Callable[[BaseModel], "AgentToolResult"]
# (admitted, reason) over the answer and the artifacts the run produced.
AnswerValidator = Callable[[str, list[Any]], tuple[bool, str]]

# Replaying a call that overran the budget or lost a cancellation race spends
# more of the resource that just ran out, so these two never become an
# observation the model is invited to retry.
_TERMINAL_TOOL_ERRORS = (
    BudgetExceeded,
    CancellationError,
    GraphEffectUncertain,
    GraphIdentityError,
)


@dataclass(frozen=True, slots=True)
class AgentTool:
    """One local capability exposed to a model as a typed function."""

    name: str
    description: str
    args_schema: type[BaseModel]
    execute: ToolExecutor
    retry_safe: bool = False

    def provider_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.args_schema.model_json_schema(),
        }


def canonical_tool_arguments(
    args_schema: type[BaseModel], arguments: BaseModel | Mapping[str, Any]
) -> dict[str, Any]:
    """Validate and include schema defaults exactly as the executor does."""
    raw = arguments.model_dump(mode="json") if isinstance(arguments, BaseModel) else dict(arguments)
    return args_schema.model_validate(raw).model_dump(mode="json")


def canonical_tool_arguments_digest(
    args_schema: type[BaseModel], arguments: BaseModel | Mapping[str, Any]
) -> str:
    """Registry/provenance digest over canonical provider arguments."""
    return stable_hash(canonical_tool_arguments(args_schema, arguments), length=32)


def canonical_tool_input_digest(
    args_schema: type[BaseModel], arguments: BaseModel | Mapping[str, Any]
) -> str:
    """Receipt SHA-256 over the same normalized argument document."""
    return canonical_json_sha256(canonical_tool_arguments(args_schema, arguments))


def canonical_json_sha256(value: Any) -> str:
    """Receipt-compatible SHA-256 for a canonical JSON-like value."""
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class AgentToolResult:
    """A serialisable observation returned to the agent after one tool call."""

    content: dict[str, Any] | str
    artifacts: list[Any] = field(default_factory=list)
    # The EvidenceReceipt artifact for this call, so trace/journal/result can
    # reconcile the same call identity without re-scanning the session store.
    receipt_artifact: Any | None = None


@dataclass(slots=True)
class AgentRunResult:
    status: str
    answer: str = ""
    artifacts: list[Any] = field(default_factory=list)
    tool_calls: int = 0
    tool_names: list[str] = field(default_factory=list)
    error: str | None = None


class AgentState(TypedDict):
    messages: list[dict[str, Any]]
    artifacts: list[Any]
    step: int
    tool_calls: int
    tool_names: list[str]
    pending_calls: list[dict[str, Any]]
    rewrites: int
    rejection: str
    answer: str
    status: str
    error: str | None


@dataclass
class AgentServices:
    agent: AgentRuntime
    execution: GraphExecution
    run_id: str


def _cancelled_update(agent: AgentRuntime, state: AgentState) -> dict[str, Any] | None:
    if agent._cancel_check is None or not agent._cancel_check():
        return None
    agent._emit(
        "agent_cancelled", "agent_runtime",
        {"step": state["step"], "tool_calls": state["tool_calls"]},
    )
    return {"status": "cancelled", "answer": "", "error": "The turn was stopped."}


def _model_node(state: AgentState, runtime: Runtime[AgentServices]) -> dict[str, Any]:
    services = runtime.context
    agent = services.agent
    if agent._cancel_check is not None and agent._cancel_check():
        agent._emit(
            "agent_cancelled",
            "agent_runtime",
            {
                "step": state["step"] + 1,
                "tool_calls": state["tool_calls"],
            },
        )
        return {"status": "cancelled", "error": "The turn was stopped before its next step."}
    if state["step"] >= agent._max_steps:
        status = "answer_unverified" if state["rejection"] else "limit_reached"
        agent._emit(
            "agent_limit_reached",
            "agent_runtime",
            {
                "step": state["step"],
                "tool_calls": state["tool_calls"],
                "step_cap": agent._max_steps,
            },
        )
        return {
            "status": status,
            "error": state["rejection"]
            or "The agent reached its reasoning-step safety limit before producing a final answer.",
        }
    step = state["step"] + 1
    request = {
        "task": agent._task,
        "messages": state["messages"],
        "tools": [tool.provider_schema() for tool in agent._tools.values()],
    }
    response = LLMToolResponse.model_validate(
        services.execution.model_effect(
            f"model:{step}",
            request,
            lambda: agent._llm.tool_call(**request).model_dump(mode="json"),
        )
    )
    if cancelled := _cancelled_update(agent, state):
        return {**cancelled, "step": step}
    if not response.tool_calls:
        answer = response.content.strip()
        return {
            "step": step,
            "answer": answer,
            "status": "validate" if answer else "failed",
            "error": None
            if answer
            else "The model ended the agent turn without an answer or a tool call.",
        }
    if state["tool_calls"] + len(response.tool_calls) > agent._max_tool_calls:
        agent._emit(
            "agent_limit_reached",
            "agent_runtime",
            {
                "step": step,
                "tool_calls": state["tool_calls"],
                "tool_call_cap": agent._max_tool_calls,
            },
        )
        return {
            "step": step,
            "status": "limit_reached",
            "error": (
                "The agent reached its tool-call safety limit before producing a final answer."
            ),
        }
    return {
        "step": step,
        "status": "tools",
        "pending_calls": [call.model_dump(mode="json") for call in response.tool_calls],
        "messages": [
            *state["messages"],
            _assistant_message(
                response.content,
                response.tool_calls,
                response.provider_state,
            ),
        ],
    }


def _tool_node(state: AgentState, runtime: Runtime[AgentServices]) -> dict[str, Any]:
    services = runtime.context
    agent = services.agent
    call = LLMToolCall.model_validate(state["pending_calls"][0])
    sequence = state["tool_calls"] + 1
    if agent._cancel_check is not None and agent._cancel_check():
        return {"status": "cancelled", "error": "The turn was stopped before its next tool."}

    def execute() -> dict[str, Any]:
        observation, artifacts = agent._invoke(
            call,
            step=state["step"],
            run_id=services.run_id,
            sequence_index=sequence,
        )
        references = []
        for artifact in artifacts:
            if agent._artifact_store is not None:
                if not isinstance(artifact, Artifact):
                    raise TypeError("Persisted agent outputs must be typed artifacts.")
                agent._artifact_store.save_artifact(artifact)
                ref = {
                    "id": artifact.id,
                    "project_id": artifact.project_id,
                    "session_id": artifact.session_id,
                    "digest": stable_hash(
                        artifact.model_dump(mode="json", exclude={"created_at"}),
                        length=64,
                    ),
                }
            else:
                ref = artifact
            if ref not in references:
                references.append(ref)
        return {
            "artifacts": references,
            "content": _observation_text(observation, limit=agent._max_observation_chars),
        }

    result = services.execution.effect(
        f"tool:{sequence}",
        {"step": state["step"], "call": call.model_dump(mode="json")},
        execute,
        retry_pending=bool(agent._tools.get(call.name) and agent._tools[call.name].retry_safe),
    )
    references = list(state["artifacts"])
    for ref in result["artifacts"]:
        if agent._artifact_store is not None:
            references = [
                old
                for old in references
                if (old["project_id"], old["session_id"], old["id"])
                != (ref["project_id"], ref["session_id"], ref["id"])
            ]
        if ref not in references:
            references.append(ref)
    if agent._restore_artifacts is not None:
        agent._restore_artifacts(agent._load_artifacts(cast(AgentState, {"artifacts": references})))
    pending = state["pending_calls"][1:]
    return {
        "artifacts": references,
        "tool_calls": sequence,
        "tool_names": [*state["tool_names"], call.name],
        "pending_calls": pending,
        "status": "tools" if pending else "model",
        "messages": [
            *state["messages"],
            {
                "role": "tool",
                "tool_call_id": call.call_id,
                "name": call.name,
                "content": result["content"],
            },
        ],
    }


def _validate_node(state: AgentState, runtime: Runtime[AgentServices]) -> dict[str, Any]:
    agent = runtime.context.agent
    if cancelled := _cancelled_update(agent, state):
        return cancelled
    if agent._answer_validator is not None:
        admitted, reason = agent._answer_validator(state["answer"], agent._load_artifacts(state))
        if cancelled := _cancelled_update(agent, state):
            return cancelled
        if not admitted:
            agent._emit(
                "agent_answer_rejected",
                "agent_runtime",
                {
                    "step": state["step"],
                    "rewrite": state["rewrites"],
                    "reason": reason[:800],
                },
            )
            if state["rewrites"] >= agent._max_answer_rewrites:
                return {
                    "status": "answer_unverified",
                    "answer": "",
                    "rejection": reason,
                    "error": (
                        f"The answer could not be verified against the evidence it cites: {reason}"
                    ),
                }
            return {
                "status": "model",
                "answer": "",
                "rejection": reason,
                "rewrites": state["rewrites"] + 1,
                "messages": [
                    *state["messages"],
                    {
                        "role": "user",
                        "content": (
                            f"Validation feedback:\n{reason}\n"
                            "Every figure must come from a tool result in this session. "
                            "Fix the errors and try again."
                        ),
                    },
                ],
            }
    agent._emit(
        "agent_completed",
        "agent_runtime",
        {
            "step": state["step"],
            "tool_calls": state["tool_calls"],
        },
    )
    return {"status": "completed"}


def build_agent_graph() -> StateGraph[AgentState, AgentServices]:
    builder = StateGraph(AgentState, context_schema=AgentServices)
    builder.add_node("model", _model_node)
    builder.add_node("tool", _tool_node)
    builder.add_node("validate", _validate_node)
    builder.add_edge(START, "model")
    routes: dict[Hashable, str] = {
        "model": "model",
        "tools": "tool",
        "validate": "validate",
        "end": END,
    }

    def route(state: AgentState) -> str:
        return state["status"] if state["status"] in routes else "end"

    for name in ("model", "tool", "validate"):
        builder.add_conditional_edges(name, route, routes)
    return builder


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
        answer_validator: AnswerValidator | None = None,
        max_answer_rewrites: int = 1,
        trace: TraceSink | None = None,
        cancel_check: Callable[[], bool] | None = None,
        persistence: GraphPersistence | None = None,
        artifact_store: ArtifactStore | None = None,
        restore_artifacts: Callable[[list[Any]], None] | None = None,
    ) -> None:
        if max_steps < 1 or max_tool_calls < 1:
            raise ValueError("Agent runtime limits must be positive.")
        if max_answer_rewrites < 0:
            raise ValueError("Agent answer rewrites cannot be negative.")
        names = [tool.name for tool in tools]
        if len(names) != len(set(names)):
            raise ValueError("Agent tool names must be unique.")
        self._llm = llm
        self._tools = {tool.name: tool for tool in tools}
        self._task = task
        self._max_steps = max_steps
        self._max_tool_calls = max_tool_calls
        self._max_observation_chars = max_observation_chars
        self._answer_validator = answer_validator
        self._max_answer_rewrites = max_answer_rewrites
        self._trace = trace
        self._cancel_check = cancel_check
        self._persistence = persistence
        self._artifact_store = artifact_store
        self._restore_artifacts = restore_artifacts
        if persistence is not None and artifact_store is None:
            raise ValueError("Durable agents require an artifact store.")

    def run(self, *, system_prompt: str, user_message: str) -> AgentRunResult:
        inputs = {
            "model_fingerprint": llm_execution_fingerprint(self._llm),
            "system_prompt": system_prompt,
            "user_message": user_message,
            "tools": [tool.provider_schema() for tool in self._tools.values()],
            "tool_retry_policy": {tool.name: tool.retry_safe for tool in self._tools.values()},
            "max_steps": self._max_steps,
            "max_tool_calls": self._max_tool_calls,
            "max_answer_rewrites": self._max_answer_rewrites,
            "max_observation_chars": self._max_observation_chars,
        }
        run_id = (
            self._persistence.execution_id if self._persistence else "agentrun_" + uuid.uuid4().hex
        )
        with graph_execution(
            self._persistence,
            definition=self._task,
            inputs=inputs,
            recursion_limit=2 * self._max_steps + self._max_tool_calls + 10,
        ) as execution:
            graph = build_agent_graph().compile(checkpointer=execution.saver)
            initial: AgentState = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "artifacts": [],
                "step": 0,
                "tool_calls": 0,
                "tool_names": [],
                "pending_calls": [],
                "rewrites": 0,
                "rejection": "",
                "answer": "",
                "status": "model",
                "error": None,
            }
            snapshot = graph.get_state(execution.config) if execution.saver else None
            if snapshot is not None and snapshot.values:
                restored = self._load_artifacts(cast(AgentState, snapshot.values))
                if self._restore_artifacts is not None:
                    self._restore_artifacts(restored)
            if snapshot is not None and snapshot.values and not snapshot.next:
                state = snapshot.values
            else:
                state = graph.invoke(
                    None if snapshot is not None and snapshot.values else initial,
                    execution.config,
                    context=AgentServices(self, execution, run_id),
                    durability="sync" if execution.saver else None,
                )
        return AgentRunResult(
            status=state["status"],
            answer=state["answer"],
            artifacts=self._load_artifacts(cast(AgentState, state)),
            tool_calls=state["tool_calls"],
            tool_names=state["tool_names"],
            error=state["error"],
        )

    def _load_artifacts(self, state: AgentState) -> list[Any]:
        if self._artifact_store is None:
            return _unique_artifacts(state["artifacts"])
        artifacts = []
        for ref in state["artifacts"]:
            try:
                artifact = self._artifact_store.get_artifact(
                    ref["id"],
                    project_id=ref["project_id"],
                    session_id=ref["session_id"],
                )
            except (FileNotFoundError, KeyError) as exc:
                raise GraphIdentityError("Saved agent evidence disappeared; start anew.") from exc
            if (
                stable_hash(
                    artifact.model_dump(mode="json", exclude={"created_at"}),
                    length=64,
                )
                != ref["digest"]
            ):
                raise GraphIdentityError("Saved agent evidence changed; start anew.")
            artifacts.append(artifact)
        return artifacts

    def _invoke(
        self,
        call: LLMToolCall,
        *,
        step: int,
        run_id: str,
        sequence_index: int,
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
        execution = ToolExecutionContext(
            run_id=run_id,
            provider_call_id=call.call_id,
            logical_step_id=make_logical_step_id(run_id, call.call_id, sequence_index),
            attempt_epoch=0,
            sequence_index=sequence_index,
        )
        try:
            with tool_execution_scope(execution):
                result = tool.execute(args)
        except _TERMINAL_TOOL_ERRORS:
            self._emit(
                "tool_failed",
                tool.name,
                {"call_id": call.call_id, "step": step, "error": "run_terminated"},
            )
            raise
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
        artifacts = list(result.artifacts)
        if result.receipt_artifact is not None:
            artifacts.append(result.receipt_artifact)
        self._emit(
            "tool_completed",
            tool.name,
            {
                "call_id": call.call_id,
                "step": step,
                "artifact_ids": [getattr(artifact, "id", "") for artifact in artifacts],
                "receipt_artifact_id": getattr(result.receipt_artifact, "id", None),
                "summary": _safe_value(content),
            },
        )
        return observation, artifacts

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
