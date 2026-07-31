from __future__ import annotations

import ast
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from eda_platform.core.ids import stable_hash
from eda_platform.core.query import UnsafeQueryError, validate_select_statement
from eda_platform.core.tool_guard import GuardViolation, ToolGuardError
from eda_platform.schemas.plans import AnalysisPlan


class PermissionTier(StrEnum):
    AUTO = "auto"
    CONFIRM = "confirm"
    DENY = "deny"


class PermissionDecision(BaseModel):
    tier: PermissionTier
    action_type: str
    action_hash: str
    description: str
    affects: list[str] = Field(default_factory=list)
    reversible: bool = True
    feedback: str = ""
    approved: bool = False


_NETWORK_IMPORTS = {"ftplib", "http", "requests", "socket", "urllib"}
_BYPASS_IMPORTS = {"ctypes", "importlib", "os", "pathlib", "shutil", "subprocess", "sys"}
_BYPASS_CALLS = {"__import__", "breakpoint", "compile", "eval", "exec", "getattr", "setattr"}
_WRITE_CALLS = {"open"}


def action_hash(action: dict[str, Any]) -> str:
    """Bind an approval to the exact action content the dispatcher will execute."""
    return stable_hash(_canonical_action(action), length=64)


def analysis_plan_action(plan: AnalysisPlan) -> dict[str, Any]:
    """Return the canonical approval payload for an analysis plan."""
    return {"type": "analysis_plan", "plan": plan.model_dump(mode="json")}


def classify_action(action: dict[str, Any]) -> PermissionDecision:
    action_type = _action_type(action)
    digest = action_hash(action)

    if action_type == "duckdb_select":
        sql = str(action.get("sql", ""))
        try:
            validate_select_statement(sql)
        except UnsafeQueryError as exc:
            return _deny(
                action,
                problem=str(exc),
                allowed="a single read-only SELECT/WITH DuckDB query over loaded datasets",
                fix_hint=(
                    "Return a SELECT/WITH query that reads only registered in-memory "
                    "relations. Do not call file readers or mutation statements."
                ),
                got=sql,
            )
        return PermissionDecision(
            tier=PermissionTier.AUTO,
            action_type=action_type,
            action_hash=digest,
            description="Run a read-only DuckDB SELECT over loaded datasets.",
            affects=["SQL result artifact"],
            reversible=True,
        )

    if action_type == "artifact_read":
        artifact_id = str(action.get("artifact_id", "artifact"))
        return PermissionDecision(
            tier=PermissionTier.AUTO,
            action_type=action_type,
            action_hash=digest,
            description=f"Read artifact {artifact_id}.",
            affects=[artifact_id],
            reversible=True,
        )

    if action_type == "sandboxed_code":
        if bool(action.get("bypass_sandbox")) or action.get("sandboxed") is False:
            return _deny(
                action,
                problem="action attempts to bypass the sandbox execution path.",
                allowed="sandboxed computation routed through the configured ExecutionBackend",
                fix_hint="Set `sandboxed` to true and execute only through the sandbox dispatcher.",
                got=action.get("sandboxed"),
            )
        if bool(action.get("network")):
            return _deny(
                action,
                problem="network access is not allowed for CodeAgent execution.",
                allowed="local mounted datasets and writes inside the sandbox output directory",
                fix_hint=(
                    "Remove network access and use only files listed in the evidence manifest."
                ),
                got=action.get("network"),
            )
        code = str(action.get("code", ""))
        violation = _code_permission_violation(code)
        if violation is not None:
            problem, got, fix_hint = violation
            return _deny(
                action,
                problem=problem,
                allowed=(
                    "sandboxed Python that reads mounted data and writes new artifacts "
                    "inside the sandbox output directory"
                ),
                fix_hint=fix_hint,
                got=got,
            )
        return PermissionDecision(
            tier=PermissionTier.AUTO,
            action_type=action_type,
            action_hash=digest,
            description="Run sandboxed Python analysis that produces a new artifact.",
            affects=["new code execution artifact"],
            reversible=True,
        )

    if action_type == "cleaning_apply":
        dataset_id = str(action.get("dataset_id", "dataset"))
        recipe_id = str(action.get("recipe_id", "recipe"))
        transform_ids = [str(item) for item in action.get("transform_ids", [])]
        return PermissionDecision(
            tier=PermissionTier.CONFIRM,
            action_type=action_type,
            action_hash=digest,
            description=(
                f"Apply cleaning recipe {recipe_id} to {dataset_id}; this writes a "
                "new dataset version."
            ),
            affects=[dataset_id, recipe_id, *transform_ids],
            reversible=bool(action.get("reversible", False)),
        )

    if action_type == "artifact_overwrite":
        artifact_id = str(action.get("artifact_id", "artifact"))
        return PermissionDecision(
            tier=PermissionTier.CONFIRM,
            action_type=action_type,
            action_hash=digest,
            description=f"Overwrite artifact {artifact_id}.",
            affects=[artifact_id],
            reversible=bool(action.get("reversible", False)),
        )

    if action_type == "analysis_plan":
        plan = action.get("plan")
        plan_data = plan if isinstance(plan, dict) else {}
        dataset_names = [str(item) for item in plan_data.get("dataset_names", [])]
        return PermissionDecision(
            tier=PermissionTier.CONFIRM,
            action_type=action_type,
            action_hash=digest,
            description=(
                "Run the previewed analysis plan exactly as shown. This produces "
                "a SQL result artifact and does not modify source datasets."
            ),
            affects=[*dataset_names, "SQL result artifact"],
            reversible=True,
        )

    if action_type in {"host_write", "network", "bypass_sandbox"}:
        return _deny(
            action,
            problem=f"{action_type.replace('_', ' ')} is not allowed from chat execution.",
            allowed=(
                "read-only dataset/artifact access, confirmed state-changing actions, "
                "or sandboxed computation"
            ),
            fix_hint="Route the work through an allowed read-only, confirmed, or sandboxed action.",
            got=action_type,
        )

    return _deny(
        action,
        problem=f"unknown action type `{action_type}`.",
        allowed=(
            "duckdb_select, artifact_read, sandboxed_code, cleaning_apply, "
            "artifact_overwrite, analysis_plan"
        ),
        fix_hint="Choose one supported action type and provide the required fields.",
        got=action_type,
    )


def require_permission(
    action: dict[str, Any],
    *,
    approved_hash: str | None = None,
) -> PermissionDecision:
    decision = classify_action(action)
    if decision.tier is PermissionTier.AUTO or decision.tier is PermissionTier.DENY:
        return decision
    if approved_hash is None:
        return _deny_with_hash(
            action,
            problem="action requires confirmation before execution.",
            fix_hint=(
                "Return this pending action to the user and execute only after an "
                "approval carrying the matching action hash."
            ),
            got=None,
        )
    if approved_hash != decision.action_hash:
        return _deny_with_hash(
            action,
            problem="approval hash does not match the action content.",
            fix_hint="Do not execute; request approval again for the exact action content.",
            got=approved_hash,
        )
    return decision.model_copy(update={"approved": True})


def pending_action_payload(decision: PermissionDecision) -> dict[str, Any]:
    return {
        "action_type": decision.action_type,
        "tier": decision.tier.value,
        "action_hash": decision.action_hash,
        "description": decision.description,
        "affects": list(decision.affects),
        "reversible": decision.reversible,
    }


def _action_type(action: dict[str, Any]) -> str:
    return str(action.get("type") or action.get("action_type") or "")


def _canonical_action(action: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in action.items()
        if key not in {"approved", "approved_hash", "approval_hash"}
    }


def _deny(
    action: dict[str, Any],
    *,
    problem: str,
    allowed: str,
    fix_hint: str,
    got: Any,
) -> PermissionDecision:
    error = ToolGuardError(
        "permission_tiering",
        [
            GuardViolation(
                field="action",
                got=got,
                allowed=allowed,
                fix_hint=fix_hint,
                problem=problem,
            )
        ],
    )
    return PermissionDecision(
        tier=PermissionTier.DENY,
        action_type=_action_type(action),
        action_hash=action_hash(action),
        description="Action denied by permission policy.",
        affects=[],
        reversible=True,
        feedback=error.to_model_feedback(),
    )


def _deny_with_hash(
    action: dict[str, Any],
    *,
    problem: str,
    fix_hint: str,
    got: Any,
) -> PermissionDecision:
    return _deny(
        action,
        problem=problem,
        allowed="an approval hash matching the exact canonical action content",
        fix_hint=fix_hint,
        got=got,
    )


def _code_permission_violation(code: str) -> tuple[str, Any, str] | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Syntax failures are normal repair-loop material, not a permission denial.
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                violation = _blocked_import(alias.name)
                if violation is not None:
                    return violation
        elif isinstance(node, ast.ImportFrom):
            violation = _blocked_import(node.module or "")
            if violation is not None:
                return violation
        elif isinstance(node, ast.Call):
            call_name = _call_name(node.func)
            if call_name in _BYPASS_CALLS:
                return (
                    f"bypass primitive `{call_name}` is not allowed in sandboxed code.",
                    call_name,
                    "Remove dynamic execution/reflection primitives from the generated code.",
                )
            if call_name in _WRITE_CALLS:
                return (
                    "direct host file writes outside the sandbox are not allowed.",
                    call_name,
                    (
                        "Write result files with library writers to relative paths inside "
                        "the sandbox output directory, and declare them in stdout JSON."
                    ),
                )
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _host_path_literal(node.value):
                return (
                    "host path literal could write outside the sandbox output directory.",
                    node.value,
                    "Use relative paths under the sandbox working directory only.",
                )
    return None


def _blocked_import(module_name: str) -> tuple[str, Any, str] | None:
    root = module_name.split(".", 1)[0]
    if root in _NETWORK_IMPORTS:
        return (
            f"network import `{root}` is not allowed in sandboxed code.",
            root,
            "Remove network access and use only mounted datasets.",
        )
    if root in _BYPASS_IMPORTS:
        return (
            f"sandbox bypass import `{root}` is not allowed.",
            root,
            "Remove host process, filesystem, or import-system access from the code.",
        )
    return None


def _call_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    return None


def _host_path_literal(value: str) -> bool:
    normalized = value.replace("\\", "/")
    return (
        value.startswith("/")
        or value.startswith("\\")
        or value.startswith("//")
        or value == "~"
        or value.startswith("~/")
        or ("/" in normalized and any(part == ".." for part in normalized.split("/")))
        or _is_windows_absolute_path(value)
    )


def _is_windows_absolute_path(value: str) -> bool:
    return len(value) >= 3 and value[0].isalpha() and value[1] == ":" and value[2] in {"/", "\\"}
