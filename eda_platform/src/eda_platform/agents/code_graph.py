"""Bounded code generation, sandbox execution and result validation graph."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypedDict, cast
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from pydantic import TypeAdapter

from eda_platform.agents.code_agent import (
    CodeAgent,
    CodeAgentAttempt,
    CodeAgentResult,
    CodeDraft,
    _cancelled_result,
    _exit_status,
    _feedback,
)
from eda_platform.core.budget import Budget, BudgetExceeded
from eda_platform.core.cancellation import CancellationError, CancellationToken
from eda_platform.core.graph_execution import GraphExecution, GraphIdentityError, graph_execution
from eda_platform.core.ids import hash_file
from eda_platform.core.llm import llm_execution_fingerprint
from eda_platform.core.sandbox import ExecArtifact


class CodeState(TypedDict):
    task: str
    evidence_manifest: dict[str, Any]
    attempts: list[CodeAgentAttempt]
    previous_error: str | None
    draft: CodeDraft | None
    artifact: ExecArtifact | None
    error_category: str | None
    result: CodeAgentResult | None
    output_digests: dict[str, str]
    tokens_used: int


@dataclass
class CodeServices:
    agent: CodeAgent
    execution: GraphExecution
    budget: Budget | None
    cancellation: CancellationToken | None
    budget_execution_id: str


def _check(services: CodeServices) -> None:
    if services.cancellation is not None:
        services.cancellation.checkpoint()
    if services.budget is not None:
        services.budget.check()


def _failure(state: CodeState, exc: Exception) -> CodeAgentResult:
    if isinstance(exc, CancellationError):
        return _cancelled_result(state["attempts"], exc)
    return CodeAgentResult(
        status="failed",
        attempts=state["attempts"],
        final_artifact=state["attempts"][-1].artifact if state["attempts"] else None,
        error=str(exc),
        error_category="budget_exhausted",
    )


def _generate(state: CodeState, runtime: Runtime[CodeServices]) -> dict[str, Any]:
    services = runtime.context
    owner = services.agent
    tokens = state["tokens_used"]
    try:
        _check(services)
        payload: dict[str, Any] = {
            "task": state["task"],
            "evidence_manifest": state["evidence_manifest"],
            "instructions": (
                "Return complete Python code only in the code field. Use only local "
                "mounted data and allowed analytical libraries. Do not access network, "
                "environment variables, subprocesses, or host paths."
            ),
        }
        if state["previous_error"]:
            payload.update(
                previous_error=state["previous_error"],
                repair_instructions=(
                    "Revise the code to address the previous sandbox failure. Keep the "
                    "analysis scoped to the same task and evidence."
                ),
            )

        def request() -> dict[str, Any]:
            draft = owner.llm.structured(
                task="m5_code_agent_generate", schema=CodeDraft, payload=payload
            )
            usage = owner.llm.last_usage()
            return {
                "draft": draft.model_dump(mode="json"),
                "tokens": usage.usage.total_tokens if usage else 0,
            }

        saved = services.execution.model_effect(
            f"draft:{len(state['attempts']) + 1}",
            payload,
            request,
        )
        draft = CodeDraft.model_validate(saved["draft"])
        tokens = state["tokens_used"] + saved["tokens"]
        if services.budget is not None:
            services.budget.account_execution_tokens(services.budget_execution_id, tokens)
        _check(services)
        return {"draft": draft, "tokens_used": tokens}
    except (CancellationError, BudgetExceeded) as exc:
        return {"result": _failure(state, exc), "tokens_used": tokens}


def _sandbox(state: CodeState, runtime: Runtime[CodeServices]) -> dict[str, Any]:
    services = runtime.context
    try:
        _check(services)
    except (CancellationError, BudgetExceeded) as exc:
        return {"result": _failure(state, exc)}
    adapter = TypeAdapter(ExecArtifact)

    def execute() -> dict[str, Any]:
        artifact, category = services.agent._execute_draft(
            cast(CodeDraft, state["draft"]),
            cancellation=services.cancellation,
        )
        paths = [*artifact.output_files]
        if artifact.manifest_path is not None:
            paths.append(artifact.manifest_path)
        return {
            "artifact": adapter.dump_python(artifact, mode="json"),
            "error_category": category,
            "output_digests": {str(path): hash_file(path, length=64) for path in paths},
        }

    saved = services.execution.effect(
        f"sandbox:{len(state['attempts']) + 1}",
        cast(CodeDraft, state["draft"]).model_dump(mode="json"),
        execute,
        retry_pending=True,
    )
    _verify_outputs(saved["output_digests"])
    return {**saved, "artifact": adapter.validate_python(saved["artifact"])}


def _verify_outputs(digests: dict[str, str]) -> None:
    for filename, digest in digests.items():
        path = Path(filename)
        if path.is_symlink() or not path.is_file() or hash_file(path, length=64) != digest:
            raise GraphIdentityError("Saved sandbox outputs changed or disappeared; start anew.")


def _validate(state: CodeState, runtime: Runtime[CodeServices]) -> dict[str, Any]:
    services, owner = runtime.context, runtime.context.agent
    artifact, draft = cast(ExecArtifact, state["artifact"]), cast(CodeDraft, state["draft"])
    attempt = CodeAgentAttempt(
        len(state["attempts"]) + 1, draft.code, artifact, state["previous_error"]
    )
    attempts = [*state["attempts"], attempt]
    try:
        if services.cancellation is not None:
            services.cancellation.checkpoint()
    except CancellationError as exc:
        return {"attempts": attempts, "result": _cancelled_result(attempts, exc)}
    status, stdout_json, contract_error = _exit_status(
        artifact,
        require_stdout_json=owner.require_stdout_json,
    )
    category = "invalid_result_contract" if contract_error else state["error_category"]
    owner._emit(
        {
            "event": "code_agent_attempt",
            "attempt": attempt.attempt,
            "status": status,
            "sandbox_status": artifact.status,
            "duration_seconds": artifact.duration_seconds,
            "error_category": category,
            "error": contract_error or artifact.error or artifact.stderr[:500],
        }
    )
    if status == "succeeded":
        return {
            "attempts": attempts,
            "result": CodeAgentResult(
                status="succeeded",
                attempts=attempts,
                final_artifact=artifact,
                stdout_json=stdout_json,
            ),
        }
    if len(attempts) >= max(1, min(owner.max_repairs + 1, 3)):
        return {
            "attempts": attempts,
            "result": CodeAgentResult(
                status="failed",
                attempts=attempts,
                final_artifact=artifact,
                error=_feedback(artifact),
                error_category="attempt_limit_exceeded",
            ),
        }
    return {"attempts": attempts, "previous_error": contract_error or _feedback(artifact)}


def build_code_graph() -> StateGraph[CodeState, CodeServices]:
    builder = StateGraph(CodeState, context_schema=CodeServices)
    builder.add_node("generate", _generate)
    builder.add_node("sandbox", _sandbox)
    builder.add_node("validate", _validate)
    builder.add_edge(START, "generate")
    builder.add_conditional_edges("generate", lambda s: END if s["result"] else "sandbox")
    builder.add_conditional_edges("sandbox", lambda s: END if s["result"] else "validate")
    builder.add_conditional_edges("validate", lambda s: END if s["result"] else "generate")
    return builder


def _mount_content_hash(source: Path, cancellation: CancellationToken | None) -> str:
    """Bind the full mounted tree, including paths and empty directories."""
    checkpoint = cancellation.checkpoint if cancellation is not None else None
    digest = hashlib.sha256()

    def visit(path: Path) -> None:
        if checkpoint is not None:
            checkpoint()
        if path.is_symlink():
            raise GraphIdentityError("Sandbox inputs must not contain symbolic links.")
        relative = path.relative_to(source).as_posix()
        if path.is_file():
            record = ["file", relative, hash_file(path, length=64, cancel_check=checkpoint)]
        elif path.is_dir():
            record = ["directory", relative]
        else:
            raise GraphIdentityError(
                "Sandbox input disappeared or is not a regular file/directory."
            )
        digest.update(json.dumps(record, ensure_ascii=False).encode("utf-8") + b"\n")
        if path.is_dir():
            for child in sorted(path.iterdir()):
                visit(child)

    visit(source)
    return digest.hexdigest()


def run_code_graph(
    owner: CodeAgent,
    *,
    task: str,
    evidence_manifest: dict[str, Any],
    budget: Budget | None,
    cancellation: CancellationToken | None,
) -> CodeAgentResult:
    budget_execution_id = (
        str(owner.persistence.path.resolve()) if owner.persistence else f"code:{uuid4()}"
    )
    inputs = {
        "task": task,
        "evidence_manifest": evidence_manifest,
        "limits": asdict(owner.limits),
        "max_repairs": owner.max_repairs,
        "require_stdout_json": owner.require_stdout_json,
        "model": llm_execution_fingerprint(owner.llm),
        "backend": asdict(owner.backend.info),
        "mounts": [
            {
                "source": str(m.source),
                "target": m.target,
                "read_only": m.read_only,
                "content_hash": _mount_content_hash(m.source, cancellation),
            }
            for m in owner.mounts
        ],
    }
    with graph_execution(
        owner.persistence,
        definition="code",
        inputs=inputs,
        checkpoint_types=(CodeDraft, CodeAgentAttempt, CodeAgentResult, ExecArtifact),
    ) as execution:
        graph = build_code_graph().compile(checkpointer=execution.saver)
        initial: CodeState = {
            "task": task,
            "evidence_manifest": evidence_manifest,
            "attempts": [],
            "previous_error": None,
            "draft": None,
            "artifact": None,
            "error_category": None,
            "result": None,
            "output_digests": {},
            "tokens_used": 0,
        }
        snapshot = graph.get_state(execution.config) if execution.saver else None
        if snapshot is not None and snapshot.values:
            _verify_outputs(snapshot.values.get("output_digests", {}))
            if budget is not None:
                try:
                    budget.account_execution_tokens(
                        budget_execution_id, snapshot.values["tokens_used"]
                    )
                except BudgetExceeded as exc:
                    return _failure(cast(CodeState, snapshot.values), exc)
        if snapshot is not None and snapshot.values and not snapshot.next:
            result = snapshot.values["result"]
        else:
            final = graph.invoke(
                None if snapshot is not None and snapshot.values else initial,
                execution.config,
                context=CodeServices(owner, execution, budget, cancellation, budget_execution_id),
                durability="sync" if execution.saver else None,
            )
            result = final["result"]
    if not isinstance(result, CodeAgentResult):
        raise RuntimeError("Code graph did not return its execution result.")
    return result
