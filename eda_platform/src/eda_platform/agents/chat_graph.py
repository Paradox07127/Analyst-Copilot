"""Structured-provider chat with durable model tasks and local execution stages."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from langgraph.func import task

from eda_platform.agents.model_workflow import (
    ModelWorkflow,
    WorkflowModelClient,
    run_model_workflow,
)
from eda_platform.agents.tool_context import (
    ToolExecutionContext,
    make_logical_step_id,
    tool_execution_scope,
)
from eda_platform.core.graph_execution import GraphIdentityError, GraphPersistence
from eda_platform.core.ids import stable_hash
from eda_platform.core.llm import LLMClient, StructuredLLM
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact
from eda_platform.schemas.chat import ChatTurnResult


def chat_stage(name: str, run: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    @task(name=name)
    def stage() -> dict[str, Any]:
        return run()

    return stage().result()


def freeze_chat_artifacts(
    workflow: ModelWorkflow,
    artifacts: list[Artifact],
    store: ArtifactStore | None,
) -> list[Artifact]:
    """Keep the original evidence set while a resumed turn has new outputs."""

    if store is None:
        return list(artifacts)

    def references() -> dict[str, Any]:
        return {
            "refs": [
                {
                    "id": artifact.id,
                    "project_id": artifact.project_id,
                    "session_id": artifact.session_id,
                    "digest": stable_hash(
                        artifact.model_dump(mode="json", exclude={"created_at"}), length=64
                    ),
                }
                for artifact in artifacts
            ]
        }

    saved = workflow.step("chat_input_artifacts", references)
    restored = []
    for ref in saved["refs"]:
        try:
            artifact = store.get_artifact(
                ref["id"],
                project_id=ref["project_id"],
                session_id=ref["session_id"],
            )
        except (KeyError, FileNotFoundError) as exc:
            raise GraphIdentityError("Chat source evidence disappeared; start anew.") from exc
        if (
            artifact is None
            or stable_hash(artifact.model_dump(mode="json", exclude={"created_at"}), length=64)
            != ref["digest"]
        ):
            raise GraphIdentityError("Chat source evidence changed; start anew.")
        restored.append(artifact)
    return restored


def execute_chat_stage(
    workflow: ModelWorkflow,
    *,
    name: str,
    execution_id: str,
    request: dict[str, Any],
    run: Callable[[], ChatTurnResult],
) -> ChatTurnResult:
    """Adopt committed local results; code uses its own stable child graph."""

    def invoke() -> dict[str, Any]:
        context = ToolExecutionContext(
            run_id=execution_id,
            provider_call_id=name,
            logical_step_id=make_logical_step_id(execution_id, name, 1),
            sequence_index=1,
        )
        with tool_execution_scope(context):
            return run().model_dump(mode="json")

    result = chat_stage(
        name,
        lambda: workflow.execution.effect(
            f"chat-local:{name}",
            request,
            invoke,
            retry_pending=True,
        ),
    )
    return ChatTurnResult.model_validate(result)


def run_structured_chat_graph(
    run: Callable[[ModelWorkflow, WorkflowModelClient], ChatTurnResult],
    *,
    llm: StructuredLLM,
    store: ArtifactStore | None,
    persistence: GraphPersistence | None,
    inputs: dict[str, Any],
    message: str,
    turn_id: str,
) -> ChatTurnResult:
    saved = run_model_workflow(
        lambda workflow: run(
            workflow,
            WorkflowModelClient(cast(LLMClient, llm), workflow),
        ).model_dump(mode="json"),
        persistence=persistence,
        inputs=inputs,
        definition="chat-structured-functional-v1",
        checkpoint_input={"message": message, "turn_id": turn_id},
    )
    result = ChatTurnResult.model_validate(saved)
    return verify_chat_result(result, store)


def verify_chat_result(result: ChatTurnResult, store: ArtifactStore | None) -> ChatTurnResult:
    if store is not None:
        for artifact in result.artifacts:
            try:
                current = store.get_artifact(
                    artifact.id,
                    project_id=artifact.project_id,
                    session_id=artifact.session_id,
                )
            except (KeyError, FileNotFoundError) as exc:
                raise GraphIdentityError(
                    "Saved chat evidence disappeared; start a new turn."
                ) from exc
            if stable_hash(current.model_dump(mode="json", exclude={"created_at"})) != stable_hash(
                artifact.model_dump(mode="json", exclude={"created_at"})
            ):
                raise GraphIdentityError("Saved chat evidence changed; start a new turn.")
    return result


def run_approved_chat_graph(
    run: Callable[[], ChatTurnResult],
    *,
    store: ArtifactStore | None,
    persistence: GraphPersistence | None,
    inputs: dict[str, Any],
    message: str,
    turn_id: str,
) -> ChatTurnResult:
    saved = run_model_workflow(
        lambda workflow: execute_chat_stage(
            workflow,
            name="approved_execute_plan",
            execution_id="chat-approved:" + turn_id,
            request=inputs,
            run=run,
        ).model_dump(mode="json"),
        persistence=persistence,
        inputs=inputs,
        definition="chat-approved-functional-v1",
        checkpoint_input={"message": message, "turn_id": turn_id},
    )
    return verify_chat_result(ChatTurnResult.model_validate(saved), store)
