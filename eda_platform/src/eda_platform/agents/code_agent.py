from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from eda_platform.core.budget import Budget, BudgetExceeded
from eda_platform.core.cancellation import (
    CancellationError,
    CancellationToken,
    current_cancellation_token,
)
from eda_platform.core.llm import LLMClient
from eda_platform.core.permissions import PermissionTier, classify_action
from eda_platform.core.sandbox import (
    ExecArtifact,
    ExecutionBackend,
    SandboxLimits,
    SandboxMount,
)
from eda_platform.core.tool_guard import ToolGuardError, check_non_empty, raise_for_violations


class CodeDraft(BaseModel):
    code: str = Field(description="Complete Python code to execute in the sandbox.")
    notes: str = ""


@dataclass(frozen=True)
class CodeAgentAttempt:
    attempt: int
    code: str
    artifact: ExecArtifact
    previous_error: str | None = None


@dataclass(frozen=True)
class CodeAgentResult:
    status: Literal["succeeded", "failed"]
    attempts: list[CodeAgentAttempt] = field(default_factory=list)
    final_artifact: ExecArtifact | None = None
    stdout_json: dict[str, Any] | None = None
    error: str | None = None
    error_category: str | None = None


class CodeAgent:
    def __init__(
        self,
        *,
        llm: LLMClient,
        backend: ExecutionBackend,
        limits: SandboxLimits | None = None,
        mounts: list[SandboxMount] | None = None,
        max_repairs: int = 2,
        require_stdout_json: bool = False,
        on_event: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self.llm = llm
        self.backend = backend
        self.limits = limits or SandboxLimits()
        self.mounts = mounts or []
        self.max_repairs = max_repairs
        self.require_stdout_json = require_stdout_json
        self.on_event = on_event

    def run(
        self,
        *,
        task: str,
        evidence_manifest: dict[str, Any],
        budget: Budget | None = None,
        cancellation: CancellationToken | None = None,
    ) -> CodeAgentResult:
        cancellation = cancellation or current_cancellation_token()
        attempts: list[CodeAgentAttempt] = []
        previous_error: str | None = None
        max_attempts = max(1, min(self.max_repairs + 1, 3))
        for attempt_number in range(1, max_attempts + 1):
            try:
                if cancellation is not None:
                    cancellation.checkpoint()
                if budget is not None:
                    budget.check()
            except CancellationError as exc:
                return _cancelled_result(attempts, exc)
            except BudgetExceeded as exc:
                return CodeAgentResult(
                    status="failed",
                    attempts=attempts,
                    final_artifact=attempts[-1].artifact if attempts else None,
                    error=str(exc),
                    error_category="budget_exhausted",
                )
            payload: dict[str, Any] = {
                "task": task,
                "evidence_manifest": evidence_manifest,
                "instructions": (
                    "Return complete Python code only in the code field. Use only local "
                    "mounted data and allowed analytical libraries. Do not access network, "
                    "environment variables, subprocesses, or host paths."
                ),
            }
            if previous_error:
                payload["previous_error"] = previous_error
                payload["repair_instructions"] = (
                    "Revise the code to address the previous sandbox failure. Keep the "
                    "analysis scoped to the same task and evidence."
                )

            try:
                draft = self.llm.structured(
                    task="m5_code_agent_generate",
                    schema=CodeDraft,
                    payload=payload,
                )
                if cancellation is not None:
                    cancellation.checkpoint()
                _record_usage(budget, self.llm)
                if budget is not None:
                    budget.check()
            except CancellationError as exc:
                return _cancelled_result(attempts, exc)
            except BudgetExceeded as exc:
                return CodeAgentResult(
                    status="failed",
                    attempts=attempts,
                    final_artifact=attempts[-1].artifact if attempts else None,
                    error=str(exc),
                    error_category="budget_exhausted",
                )
            artifact, error_category = self._execute_draft(
                draft,
                cancellation=cancellation,
            )
            attempt = CodeAgentAttempt(
                attempt=attempt_number,
                code=draft.code,
                artifact=artifact,
                previous_error=previous_error,
            )
            attempts.append(attempt)
            try:
                if cancellation is not None:
                    cancellation.checkpoint()
            except CancellationError as exc:
                return _cancelled_result(attempts, exc)
            status, stdout_json, contract_error = _exit_status(
                artifact,
                require_stdout_json=self.require_stdout_json,
            )
            if contract_error is not None:
                error_category = "invalid_result_contract"
            self._emit(
                {
                    "event": "code_agent_attempt",
                    "attempt": attempt_number,
                    "status": status,
                    "sandbox_status": artifact.status,
                    "duration_seconds": artifact.duration_seconds,
                    "error_category": error_category,
                    "error": contract_error or artifact.error or artifact.stderr[:500],
                }
            )
            if status == "succeeded":
                return CodeAgentResult(
                    status="succeeded",
                    attempts=attempts,
                    final_artifact=artifact,
                    stdout_json=stdout_json,
                )
            previous_error = contract_error or _feedback(artifact)

        return CodeAgentResult(
            status="failed",
            attempts=attempts,
            final_artifact=attempts[-1].artifact if attempts else None,
            error=_feedback(attempts[-1].artifact) if attempts else "No attempts executed.",
            error_category="attempt_limit_exceeded",
        )

    def _emit(self, event: dict[str, object]) -> None:
        if self.on_event is not None:
            self.on_event(event)

    def _execute_draft(
        self,
        draft: CodeDraft,
        *,
        cancellation: CancellationToken | None,
    ) -> tuple[ExecArtifact, str | None]:
        try:
            raise_for_violations(
                "code_agent_draft",
                [
                    check_non_empty(
                        "code",
                        draft.code,
                        fix_hint="Return complete Python code in the `code` field.",
                    )
                ],
            )
        except ToolGuardError as exc:
            return _blocked_artifact(self.backend, exc.to_model_feedback()), "tool_guard"

        permission = classify_action(
            {
                "type": "sandboxed_code",
                "code": draft.code,
                "sandboxed": True,
            }
        )
        if permission.tier is PermissionTier.DENY:
            return _blocked_artifact(self.backend, permission.feedback), "permission_denied"

        if cancellation is None:
            artifact = self.backend.run_python(
                draft.code,
                mounts=self.mounts,
                limits=self.limits,
            )
        else:
            artifact = self.backend.run_python(
                draft.code,
                mounts=self.mounts,
                limits=self.limits,
                cancellation=cancellation,
            )
        return _collect_declared_outputs(artifact), _sandbox_error_category(artifact)


def _feedback(artifact: ExecArtifact) -> str:
    parts = [
        f"status={artifact.status}",
        f"exit_code={artifact.exit_code}",
    ]
    if artifact.error:
        parts.append(f"error={artifact.error}")
    if artifact.stderr:
        parts.append(f"stderr={artifact.stderr[-2000:]}")
    if artifact.stdout:
        parts.append(f"stdout={artifact.stdout[-1000:]}")
    return "\n".join(parts)


def _cancelled_result(
    attempts: list[CodeAgentAttempt],
    exc: CancellationError,
) -> CodeAgentResult:
    return CodeAgentResult(
        status="failed",
        attempts=attempts,
        final_artifact=attempts[-1].artifact if attempts else None,
        error=str(exc),
        error_category="cancelled",
    )


def _record_usage(budget: Budget | None, llm: LLMClient) -> None:
    if budget is None:
        return
    usage = llm.last_usage()
    if usage is None:
        return
    budget.add_tokens(usage.usage.total_tokens)


def _blocked_artifact(backend: ExecutionBackend, error: str) -> ExecArtifact:
    backend_name = str(getattr(backend, "name", type(backend).__name__))
    return ExecArtifact(status="blocked", backend=backend_name, error=error)


def _sandbox_error_category(artifact: ExecArtifact) -> str | None:
    if artifact.status == "succeeded":
        return None
    if artifact.status in {"blocked", "timeout"}:
        return f"sandbox_{artifact.status}"
    return "sandbox_failed"


def _exit_status(
    artifact: ExecArtifact,
    *,
    require_stdout_json: bool,
) -> tuple[Literal["succeeded", "failed"], dict[str, Any] | None, str | None]:
    if artifact.status != "succeeded":
        return "failed", None, None
    if not require_stdout_json:
        return "succeeded", None, None
    stdout_json = _parse_stdout_json(artifact.stdout)
    if stdout_json is None:
        return "failed", None, "Sandbox stdout must end with a JSON object."
    declared = stdout_json.get("result_files", [])
    if not isinstance(declared, list) or not all(isinstance(item, str) for item in declared):
        return "failed", None, "`result_files` must be a list of relative file paths."
    reserved = [item for item in declared if _uses_reserved_output_path(item)]
    if reserved:
        return "failed", None, "Result files cannot use the reserved `inputs/` directory."
    missing = _missing_declared_outputs(artifact, declared)
    if missing:
        return "failed", None, f"Declared result files were not produced: {', '.join(missing)}."
    return "succeeded", stdout_json, None


def _parse_stdout_json(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _missing_declared_outputs(artifact: ExecArtifact, declared: list[str]) -> list[str]:
    if artifact.work_dir is None:
        return list(declared)
    work_dir = artifact.work_dir.resolve()
    missing: list[str] = []
    for item in declared:
        candidate = (work_dir / item).resolve()
        try:
            candidate.relative_to(work_dir)
        except ValueError:
            missing.append(item)
            continue
        if not candidate.is_file():
            missing.append(item)
    return missing


def _uses_reserved_output_path(item: str) -> bool:
    parts = Path(item.replace("\\", "/")).parts
    return bool(parts) and parts[0].casefold() == "inputs"


def _collect_declared_outputs(artifact: ExecArtifact) -> ExecArtifact:
    stdout_json = _parse_stdout_json(artifact.stdout)
    if stdout_json is None or artifact.work_dir is None:
        return artifact
    declared = stdout_json.get("result_files", [])
    if not isinstance(declared, list):
        return artifact
    work_dir = artifact.work_dir.resolve()
    output_files = []
    for item in declared:
        if not isinstance(item, str):
            continue
        if _uses_reserved_output_path(item):
            continue
        candidate = (work_dir / item).resolve()
        try:
            candidate.relative_to(work_dir)
        except ValueError:
            continue
        if candidate.is_file():
            output_files.append(candidate)
    return replace(artifact, output_files=output_files)
