from __future__ import annotations

import ast
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol
from uuid import uuid4

from eda_platform.core.cancellation import (
    CancellationError,
    CancellationToken,
    current_cancellation_token,
)

SandboxStatus = Literal["succeeded", "failed", "timeout", "blocked"]
_PROCESS_POLL_SECONDS = 0.05
_PROCESS_TERMINATE_GRACE_SECONDS = 0.5

_ALLOWED_IMPORT_ROOTS = {
    "collections",
    "datetime",
    "duckdb",
    "functools",
    "itertools",
    "json",
    "math",
    "matplotlib",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "statistics",
    "typing",
}
_BLOCKED_IMPORT_ROOTS = {
    "builtins",
    "ctypes",
    "ftplib",
    "glob",
    "http",
    "importlib",
    "os",
    "pathlib",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "urllib",
}
_BLOCKED_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "getattr",
    "input",
    "open",
    "setattr",
    "vars",
}
_BLOCKED_ATTRS = {
    "__bases__",
    "__class__",
    "__globals__",
    "__mro__",
    "__subclasses__",
    "chmod",
    "chown",
    "exec",
    "fork",
    "kill",
    "popen",
    "remove",
    "rmdir",
    "rmtree",
    "spawn",
    "system",
    "unlink",
}


@dataclass(frozen=True)
class SandboxLimits:
    timeout_seconds: float = 5.0
    max_stdout_bytes: int = 65_536
    max_stderr_bytes: int = 65_536
    max_memory_bytes: int = 1 << 30  # 1 GiB
    max_output_files: int = 32
    max_output_file_bytes: int = 64 << 20
    max_total_output_bytes: int = 128 << 20


@dataclass(frozen=True)
class SandboxMount:
    source: Path
    target: str
    read_only: bool = True


@dataclass(frozen=True)
class SandboxBackendInfo:
    name: str
    safe_for_untrusted_code: bool
    available: bool
    detail: str = ""


@dataclass(frozen=True)
class ExecArtifact:
    status: SandboxStatus
    backend: str
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    error: str | None = None
    timed_out: bool = False
    output_truncated: bool = False
    duration_seconds: float = 0.0
    work_dir: Path | None = None
    output_files: list[Path] = field(default_factory=list)
    output_manifest: list[dict[str, object]] = field(default_factory=list)
    stdout_sha256: str | None = None
    stderr_sha256: str | None = None
    image_digest: str | None = None
    policy_digest: str | None = None
    manifest_path: Path | None = None
    manifest_sha256: str | None = None


class ExecutionBackend(Protocol):
    @property
    def info(self) -> SandboxBackendInfo: ...

    def run_python(
        self,
        code: str,
        *,
        mounts: list[SandboxMount] | None = None,
        limits: SandboxLimits | None = None,
        cancellation: CancellationToken | None = None,
    ) -> ExecArtifact: ...


class PortableSubprocessBackend:
    """Cross-platform *convenience* runner — **NOT a security boundary**."""

    name = "portable_subprocess"

    def __init__(
        self,
        *,
        work_root: Path | str,
        python_executable: str | None = None,
    ) -> None:
        self.work_root = Path(work_root)
        self.python_executable = python_executable or sys.executable

    @property
    def info(self) -> SandboxBackendInfo:
        return SandboxBackendInfo(
            name=self.name,
            safe_for_untrusted_code=False,
            available=True,
            detail="Development/test convenience backend only; not a security boundary.",
        )

    def run_python(
        self,
        code: str,
        *,
        mounts: list[SandboxMount] | None = None,
        limits: SandboxLimits | None = None,
        cancellation: CancellationToken | None = None,
    ) -> ExecArtifact:
        cancellation = cancellation or current_cancellation_token()
        actual_limits = limits or SandboxLimits()
        started = time.monotonic()
        try:
            if cancellation is not None:
                cancellation.checkpoint()
        except CancellationError as exc:
            return _cancelled_artifact(self.name, started, self.work_root, exc)
        blocked = _policy_violation(code)
        exec_dir = self.work_root / f"exec_{uuid4().hex[:12]}"
        exec_dir.mkdir(parents=True, exist_ok=True)
        if blocked is not None:
            return ExecArtifact(
                status="blocked",
                backend=self.name,
                error=blocked,
                duration_seconds=_elapsed(started),
                work_dir=exec_dir,
            )

        try:
            _materialize_mounts(exec_dir, mounts or [])
        except ValueError as exc:
            return ExecArtifact(
                status="blocked",
                backend=self.name,
                error=str(exc),
                duration_seconds=_elapsed(started),
                work_dir=exec_dir,
            )
        script_path = exec_dir / "analysis.py"
        script_path.write_text(code, encoding="utf-8")

        try:
            process = subprocess.Popen(
                [self.python_executable, "-I", str(script_path)],
                cwd=exec_dir,
                env=_sandbox_env(exec_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError as exc:
            return ExecArtifact(
                status="blocked",
                backend=self.name,
                error=f"Failed to launch sandbox process: {exc}",
                duration_seconds=_elapsed(started),
                work_dir=exec_dir,
            )

        deadline = time.monotonic() + actual_limits.timeout_seconds
        while process.poll() is None:
            try:
                if cancellation is not None:
                    cancellation.checkpoint()
            except CancellationError as exc:
                _terminate_process(process)
                stdout, stderr = process.communicate()
                return ExecArtifact(
                    status="blocked",
                    backend=self.name,
                    stdout=stdout[: actual_limits.max_stdout_bytes],
                    stderr=stderr[: actual_limits.max_stderr_bytes],
                    exit_code=process.returncode,
                    error=f"Execution cancelled: {exc}",
                    duration_seconds=_elapsed(started),
                    work_dir=exec_dir,
                )
            if time.monotonic() >= deadline:
                _terminate_process(process)
                stdout, stderr = process.communicate()
                return ExecArtifact(
                    status="timeout",
                    backend=self.name,
                    stdout=stdout[: actual_limits.max_stdout_bytes],
                    stderr=stderr[: actual_limits.max_stderr_bytes],
                    exit_code=process.returncode,
                    timed_out=True,
                    error=f"Execution exceeded {actual_limits.timeout_seconds:.2f}s timeout.",
                    duration_seconds=_elapsed(started),
                    work_dir=exec_dir,
                )
            time.sleep(_PROCESS_POLL_SECONDS)

        stdout, stderr = process.communicate()
        try:
            if cancellation is not None:
                cancellation.checkpoint()
        except CancellationError as exc:
            return ExecArtifact(
                status="blocked",
                backend=self.name,
                stdout=stdout[: actual_limits.max_stdout_bytes],
                stderr=stderr[: actual_limits.max_stderr_bytes],
                exit_code=process.returncode,
                error=f"Execution cancelled: {exc}",
                duration_seconds=_elapsed(started),
                work_dir=exec_dir,
            )

        return _finalize(
            self.name,
            process.returncode,
            stdout,
            stderr,
            actual_limits,
            started,
            exec_dir,
        )


def _terminate_process(
    process: subprocess.Popen[str],
    *,
    grace_seconds: float = _PROCESS_TERMINATE_GRACE_SECONDS,
) -> None:
    """TERM, bounded grace, then KILL; always reap the child."""

    if process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=grace_seconds)
    except OSError:
        return


def _cancelled_artifact(
    backend: str,
    started: float,
    work_dir: Path,
    exc: CancellationError,
) -> ExecArtifact:
    return ExecArtifact(
        status="blocked",
        backend=backend,
        error=f"Execution cancelled: {exc}",
        duration_seconds=_elapsed(started),
        work_dir=work_dir,
    )


class SandboxUnavailableError(RuntimeError):
    """Raised when no configured safe backend can execute untrusted code."""


# Shared static checks are defense in depth. Docker remains the security boundary.


def _policy_violation(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"Invalid Python syntax: {exc.msg}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in _BLOCKED_IMPORT_ROOTS or root not in _ALLOWED_IMPORT_ROOTS:
                    return f"Import is not allowed in sandbox: {root}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root in _BLOCKED_IMPORT_ROOTS or root not in _ALLOWED_IMPORT_ROOTS:
                return f"Import is not allowed in sandbox: {root}"
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in _BLOCKED_CALLS:
                return f"Call is not allowed in sandbox: {name}"
            attr = _call_attr(node.func)
            if attr in _BLOCKED_ATTRS:
                return f"Operation is not allowed in sandbox: {attr}"
        elif isinstance(node, ast.Attribute):
            if node.attr in _BLOCKED_ATTRS:
                return f"Attribute access is not allowed in sandbox: {node.attr}"
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            path_violation = _host_path_literal_violation(node.value)
            if path_violation is not None:
                return path_violation
    return None


def _call_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    return None


def _call_attr(func: ast.expr) -> str | None:
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _host_path_literal_violation(value: str) -> str | None:
    normalized = value.replace("\\", "/")
    if normalized == "/inputs" or normalized.startswith("/inputs/"):
        return None
    if (
        value.startswith("/")
        or value.startswith("\\")
        or value.startswith("\\\\")
        or value.startswith("//")
        or value == "~"
        or value.startswith("~/")
        or value.startswith("~\\")
        or _is_windows_absolute_path(value)
    ):
        return "Host path literal is not allowed in sandbox"
    if "/" in normalized and any(part == ".." for part in normalized.split("/")):
        return "Parent path traversal is not allowed in sandbox"
    return None


def _is_windows_absolute_path(value: str) -> bool:
    return len(value) >= 3 and value[0].isalpha() and value[1] == ":" and value[2] in {"/", "\\"}


def _materialize_mounts(exec_dir: Path, mounts: list[SandboxMount]) -> None:
    exec_root = exec_dir.resolve()
    for mount in mounts:
        target = _resolve_mount_target(exec_root, mount.target)
        source = mount.source
        if not source.exists():
            raise ValueError(f"Sandbox mount source does not exist: {source}")
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def _resolve_mount_target(exec_root: Path, target: str) -> Path:
    raw_target = Path(target)
    if (
        not target.strip()
        or raw_target.is_absolute()
        or target.startswith("\\")
        or target.startswith("//")
        or _is_windows_absolute_path(target)
    ):
        raise ValueError("Sandbox mount target must be a relative path")
    resolved = (exec_root / raw_target).resolve()
    try:
        resolved.relative_to(exec_root)
    except ValueError as exc:
        raise ValueError("Sandbox mount target resolves outside execution directory") from exc
    return resolved


def _sandbox_env(exec_dir: Path) -> dict[str, str]:
    exec_str = str(exec_dir)
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": exec_str,
        "TMPDIR": exec_str,
        "MPLBACKEND": "Agg",
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def _finalize(
    backend_name: str,
    returncode: int,
    stdout: str | None,
    stderr: str | None,
    limits: SandboxLimits,
    started: float,
    exec_dir: Path,
) -> ExecArtifact:
    capped_stdout, stdout_truncated = _cap_text(stdout, limits.max_stdout_bytes)
    capped_stderr, stderr_truncated = _cap_text(stderr, limits.max_stderr_bytes)
    if stdout_truncated or stderr_truncated:
        return ExecArtifact(
            status="blocked",
            backend=backend_name,
            stdout=capped_stdout,
            stderr=capped_stderr,
            exit_code=returncode,
            error="Execution output exceeded configured byte limit.",
            output_truncated=True,
            duration_seconds=_elapsed(started),
            work_dir=exec_dir,
        )
    return ExecArtifact(
        status="succeeded" if returncode == 0 else "failed",
        backend=backend_name,
        stdout=capped_stdout,
        stderr=capped_stderr,
        exit_code=returncode,
        error=None if returncode == 0 else "Python process exited with error.",
        duration_seconds=_elapsed(started),
        work_dir=exec_dir,
    )


def _cap_text(value: str | None, max_bytes: int) -> tuple[str, bool]:
    text = value or ""
    raw = text.encode("utf-8", errors="replace")
    if len(raw) <= max_bytes:
        return text, False
    clipped = raw[:max_bytes].decode("utf-8", errors="ignore")
    return clipped, True


def _elapsed(started: float) -> float:
    return round(time.monotonic() - started, 4)
