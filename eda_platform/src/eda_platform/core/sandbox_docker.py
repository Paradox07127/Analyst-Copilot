from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import IO, cast
from uuid import uuid4

from eda_platform.core.cancellation import (
    CancellationError,
    CancellationToken,
    current_cancellation_token,
)
from eda_platform.core.fs import remove_tree
from eda_platform.core.sandbox import (
    ExecArtifact,
    SandboxBackendInfo,
    SandboxLimits,
    SandboxMount,
    _elapsed,
    _finalize,
    _policy_violation,
    _resolve_mount_target,
)

DEFAULT_DOCKER_IMAGE = "eda-agent-sandbox:py312"
DEFAULT_DOCKER_PIDS_LIMIT = 128
DEFAULT_DOCKER_CPUS = "1"
DEFAULT_DOCKER_TMPFS_SIZE = "64m"
DEFAULT_CONTAINER_USER = "65532:65532"
_DOCKER_CLI_ENV_KEYS = (
    "DOCKER_HOST",
    "DOCKER_CONTEXT",
    "DOCKER_CONFIG",
    "DOCKER_TLS_VERIFY",
    "DOCKER_CERT_PATH",
)
_OUTPUT_CHUNK_BYTES = 8192
_WAIT_POLL_SECONDS = 0.05
_CLEANUP_TIMEOUT_SECONDS = 5.0
_CLEANUP_RETRY_DELAY_SECONDS = 0.05
_CLI_TERMINATE_GRACE_SECONDS = 1.0
_PREFLIGHT_TIMEOUT_SECONDS = 8.0


def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    completed = _run_docker(
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        timeout=2.0,
    )
    return completed is not None and completed.returncode == 0 and bool(completed.stdout.strip())


def docker_security_available() -> tuple[bool, str]:
    completed = _run_docker(
        [
            "docker",
            "info",
            "--format",
            "{{.OSType}}\t{{json .SecurityOptions}}",
        ],
        timeout=2.0,
    )
    if completed is None or completed.returncode != 0:
        return False, "Docker security capabilities could not be inspected."
    try:
        os_type, raw_options = completed.stdout.strip().split("\t", maxsplit=1)
        options = json.loads(raw_options)
    except (ValueError, json.JSONDecodeError):
        return False, "Docker returned an invalid security capability report."
    normalized = {str(option).split(",", maxsplit=1)[0] for option in options}
    if os_type.strip().lower() != "linux":
        return False, "The sandbox requires a Linux-container Docker engine."
    if "name=seccomp" not in normalized:
        return False, "Docker seccomp is unavailable or disabled."
    if "name=cgroupns" not in normalized:
        return False, "Docker private cgroup namespaces are unavailable."
    return True, "Docker seccomp and cgroup namespaces are available."


def docker_image_available(image: str) -> bool:
    return docker_image_digest(image) is not None


def docker_image_digest(image: str) -> str | None:
    if not docker_available():
        return None
    completed = _run_docker(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        timeout=2.0,
    )
    if completed is None or completed.returncode != 0:
        return None
    digest = completed.stdout.strip()
    return digest or None


def _run_docker(
    argv: list[str],
    *,
    timeout: float,
) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            argv,
            capture_output=True,
            text=True,
            env=_docker_cli_env(),
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _docker_cli_env() -> dict[str, str]:
    env = {"PATH": os.environ.get("PATH", os.defpath)}
    for key in _DOCKER_CLI_ENV_KEYS:
        value = os.environ.get(key)
        if value is not None:
            env[key] = value
    return env


class DockerSandboxBackend:
    name = "docker"

    def __init__(
        self,
        *,
        work_root: Path | str,
        image: str = DEFAULT_DOCKER_IMAGE,
        pids_limit: int = DEFAULT_DOCKER_PIDS_LIMIT,
        cpus: str = DEFAULT_DOCKER_CPUS,
        tmpfs_size: str = DEFAULT_DOCKER_TMPFS_SIZE,
    ) -> None:
        self.work_root = Path(work_root)
        self.image = image
        self.pids_limit = pids_limit
        self.cpus = cpus
        self.tmpfs_size = tmpfs_size
        self._verified_info: SandboxBackendInfo | None = None

    @property
    def info(self) -> SandboxBackendInfo:
        if self._verified_info is not None:
            return self._verified_info
        error = self._availability_error()
        return SandboxBackendInfo(
            name=self.name,
            safe_for_untrusted_code=True,
            available=error is None,
            detail=error or "Docker sandbox dependencies are available.",
        )

    def verify_runtime(self) -> SandboxBackendInfo:
        if self._verified_info is not None:
            return self._verified_info
        info = self.info
        if not info.available:
            self._verified_info = info
            return info

        code = (
            "import json, os, socket\n"
            "from pathlib import Path\n"
            "status = {}\n"
            "for line in Path('/proc/self/status').read_text().splitlines():\n"
            "    if ':' in line:\n"
            "        key, value = line.split(':', 1)\n"
            "        if key in {'CapEff', 'NoNewPrivs', 'Seccomp'}:\n"
            "            status[key] = value.strip()\n"
            "root_write = False\n"
            "try:\n"
            "    Path('/sandbox-preflight-write').write_text('blocked')\n"
            "    root_write = True\n"
            "except OSError:\n"
            "    pass\n"
            "network = False\n"
            "try:\n"
            "    socket.create_connection(('1.1.1.1', 53), timeout=0.25)\n"
            "    network = True\n"
            "except OSError:\n"
            "    pass\n"
            "cgroup = {}\n"
            "for name in ('memory.max', 'memory.swap.max', 'pids.max', 'cpu.max'):\n"
            "    try:\n"
            "        cgroup[name] = Path('/sys/fs/cgroup', name).read_text().strip()\n"
            "    except OSError:\n"
            "        cgroup[name] = None\n"
            "print(json.dumps({'uid': os.getuid(), 'status': status, "
            "'root_write': root_write, 'network': network, 'cgroup': cgroup}))\n"
        )
        preflight_limits = SandboxLimits(
            timeout_seconds=_PREFLIGHT_TIMEOUT_SECONDS,
            max_memory_bytes=128 << 20,
            max_output_files=4,
            max_output_file_bytes=1 << 20,
            max_total_output_bytes=2 << 20,
        )
        artifact = self._execute(
            code,
            mounts=[],
            limits=preflight_limits,
            enforce_static_policy=False,
            purpose="preflight",
        )
        try:
            report = json.loads(artifact.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            report = {}
        finally:
            cleanup_root = (
                artifact.manifest_path.parent
                if artifact.manifest_path is not None
                else artifact.work_dir.parent
                if artifact.work_dir is not None
                else None
            )
            if cleanup_root is not None:
                remove_tree(cleanup_root, ignore_errors=True)

        status = report.get("status") if isinstance(report, dict) else None
        valid = (
            artifact.status == "succeeded"
            and isinstance(report, dict)
            and report.get("uid") not in {None, 0}
            and report.get("root_write") is False
            and report.get("network") is False
            and isinstance(status, dict)
            and status.get("CapEff") == "0000000000000000"
            and status.get("NoNewPrivs") == "1"
            and status.get("Seccomp") == "2"
            and _resource_controls_match(
                report,
                memory_bytes=preflight_limits.max_memory_bytes,
                pids_limit=self.pids_limit,
                cpus=self.cpus,
            )
        )
        self._verified_info = SandboxBackendInfo(
            name=self.name,
            safe_for_untrusted_code=True,
            available=valid,
            detail=(
                "Docker sandbox runtime preflight passed."
                if valid
                else "Docker sandbox runtime preflight failed; refusing untrusted code."
            ),
        )
        return self._verified_info

    def run_python(
        self,
        code: str,
        *,
        mounts: list[SandboxMount] | None = None,
        limits: SandboxLimits | None = None,
        cancellation: CancellationToken | None = None,
    ) -> ExecArtifact:
        cancellation = cancellation or current_cancellation_token()
        return self._execute(
            code,
            mounts=mounts or [],
            limits=limits or SandboxLimits(),
            enforce_static_policy=True,
            purpose="agent",
            cancellation=cancellation,
        )

    def _availability_error(self) -> str | None:
        if not docker_available():
            return "Docker unavailable: CLI or daemon is unavailable."
        security_ok, security_detail = docker_security_available()
        if not security_ok:
            return security_detail
        if not docker_image_available(self.image):
            return (
                f"Docker sandbox image {self.image!r} is unavailable. "
                "Build or load the configured image before enabling CodeAgent."
            )
        return None

    def _execute(
        self,
        code: str,
        *,
        mounts: list[SandboxMount],
        limits: SandboxLimits,
        enforce_static_policy: bool,
        purpose: str,
        cancellation: CancellationToken | None = None,
    ) -> ExecArtifact:
        started = time.monotonic()
        exec_dir = self.work_root / f"exec_{uuid4().hex[:12]}"
        output_dir = exec_dir / "outputs"
        output_dir.mkdir(parents=True, exist_ok=False)
        try:
            if cancellation is not None:
                cancellation.checkpoint()
        except CancellationError as exc:
            return ExecArtifact(
                status="blocked",
                backend=self.name,
                error=f"Execution cancelled: {exc}",
                duration_seconds=_elapsed(started),
                work_dir=output_dir,
            )

        availability_error = self._availability_error()
        if availability_error is not None:
            return ExecArtifact(
                status="blocked",
                backend=self.name,
                error=availability_error,
                duration_seconds=_elapsed(started),
                work_dir=output_dir,
            )
        limits_error = _validate_limits(limits, self.pids_limit, self.cpus)
        if limits_error is not None:
            return ExecArtifact(
                status="blocked",
                backend=self.name,
                error=limits_error,
                duration_seconds=_elapsed(started),
                work_dir=output_dir,
            )
        if enforce_static_policy:
            blocked = _policy_violation(code)
            if blocked is not None:
                return ExecArtifact(
                    status="blocked",
                    backend=self.name,
                    error=blocked,
                    duration_seconds=_elapsed(started),
                    work_dir=output_dir,
                )

        try:
            staged_mounts, input_manifest = _stage_mounts(exec_dir, mounts)
        except (OSError, ValueError) as exc:
            return ExecArtifact(
                status="blocked",
                backend=self.name,
                error=f"Sandbox inputs could not be staged safely: {exc}",
                duration_seconds=_elapsed(started),
                work_dir=output_dir,
            )
        resolved_image = docker_image_digest(self.image)
        if resolved_image is None:
            return ExecArtifact(
                status="blocked",
                backend=self.name,
                error="Docker sandbox image digest could not be resolved.",
                duration_seconds=_elapsed(started),
                work_dir=output_dir,
            )

        script_path = exec_dir / "analysis.py"
        container_user = _container_user()
        try:
            script_path.write_text(code, encoding="utf-8")
            script_path.chmod(0o444)
            _prepare_output_dir(output_dir, container_user)
            _prepare_input_mountpoints(output_dir, staged_mounts, container_user)
        except (OSError, ValueError) as exc:
            return ExecArtifact(
                status="blocked",
                backend=self.name,
                error=f"Sandbox workspace could not be prepared safely: {exc}",
                duration_seconds=_elapsed(started),
                work_dir=output_dir,
            )

        container_name = f"eda-agent-{uuid4().hex[:12]}"
        cli_env = _docker_cli_env()
        argv = _docker_argv(
            exec_dir=exec_dir,
            container_name=container_name,
            mounts=staged_mounts,
            limits=limits,
            image=resolved_image,
            pids_limit=self.pids_limit,
            cpus=self.cpus,
            tmpfs_size=self.tmpfs_size,
            container_user=container_user,
        )
        try:
            proc = subprocess.Popen(
                argv,
                cwd=exec_dir,
                env=cli_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            _cleanup_container(container_name, cli_env)
            return ExecArtifact(
                status="blocked",
                backend=self.name,
                error=f"Failed to launch Docker sandbox: {exc}",
                duration_seconds=_elapsed(started),
                work_dir=output_dir,
            )

        stdout = _CappedOutput(limits.max_stdout_bytes)
        stderr = _CappedOutput(limits.max_stderr_bytes)
        stream_limit_exceeded = threading.Event()
        readers = [
            _start_reader(proc.stdout, stdout, stream_limit_exceeded),
            _start_reader(proc.stderr, stderr, stream_limit_exceeded),
        ]
        output_watchdog = _OutputDirectoryWatchdog(output_dir, limits)
        output_watchdog.start()

        deadline = time.monotonic() + limits.timeout_seconds
        returncode: int | None = None
        failure: str | None = None
        timed_out = False
        cancellation_error: CancellationError | None = None
        while returncode is None:
            try:
                if cancellation is not None:
                    cancellation.checkpoint()
            except CancellationError as exc:
                cancellation_error = exc
                break
            if stream_limit_exceeded.is_set():
                failure = "Execution output exceeded configured byte limit."
                break
            if output_watchdog.violation is not None:
                failure = output_watchdog.violation
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            try:
                returncode = proc.wait(timeout=min(_WAIT_POLL_SECONDS, remaining))
            except subprocess.TimeoutExpired:
                continue

        if cancellation_error is not None:
            _cancel_container_run(proc, container_name, cli_env)
        elif failure is not None or timed_out:
            _abort_container_run(proc, container_name, cli_env)
        output_watchdog.stop()
        _join_readers(readers)
        _remove_input_mountpoints(output_dir, staged_mounts)

        if cancellation_error is not None:
            artifact = ExecArtifact(
                status="blocked",
                backend=self.name,
                stdout=stdout.text,
                stderr=stderr.text,
                error=f"Execution cancelled: {cancellation_error}",
                duration_seconds=_elapsed(started),
                work_dir=output_dir,
            )
        elif timed_out:
            artifact = ExecArtifact(
                status="timeout",
                backend=self.name,
                stdout=stdout.text,
                stderr=stderr.text,
                timed_out=True,
                error=f"Execution exceeded {limits.timeout_seconds:.2f}s timeout.",
                duration_seconds=_elapsed(started),
                work_dir=output_dir,
            )
        elif failure is not None or stdout.truncated or stderr.truncated:
            artifact = ExecArtifact(
                status="blocked",
                backend=self.name,
                stdout=stdout.text,
                stderr=stderr.text,
                error=failure or "Execution output exceeded configured byte limit.",
                output_truncated=stdout.truncated or stderr.truncated,
                duration_seconds=_elapsed(started),
                work_dir=output_dir,
            )
        else:
            artifact = _finalize(
                self.name,
                returncode if returncode is not None else -1,
                stdout.text,
                stderr.text,
                limits,
                started,
                output_dir,
            )
            if artifact.exit_code == 137:
                artifact = replace(
                    artifact,
                    error=(
                        "Docker sandbox process was terminated by SIGKILL (exit code 137) "
                        "while cgroup resource limits were active."
                    ),
                )

        output_manifest, output_error = _scan_output_dir(output_dir, limits, include_hashes=True)
        if output_error is not None:
            artifact = replace(artifact, status="blocked", error=output_error)
            output_manifest = []
        return _seal_artifact(
            artifact,
            exec_dir=exec_dir,
            code=code,
            image=self.image,
            image_digest=resolved_image,
            policy_digest=_policy_digest(
                limits=limits,
                pids_limit=self.pids_limit,
                cpus=self.cpus,
                tmpfs_size=self.tmpfs_size,
                container_user=container_user,
            ),
            purpose=purpose,
            inputs=input_manifest,
            outputs=output_manifest,
        )


def _docker_argv(
    *,
    exec_dir: Path,
    container_name: str,
    mounts: list[SandboxMount],
    limits: SandboxLimits,
    image: str = DEFAULT_DOCKER_IMAGE,
    pids_limit: int = DEFAULT_DOCKER_PIDS_LIMIT,
    cpus: str = DEFAULT_DOCKER_CPUS,
    tmpfs_size: str = DEFAULT_DOCKER_TMPFS_SIZE,
    container_user: str | None = None,
) -> list[str]:
    exec_root = exec_dir.resolve()
    script_path = (exec_root / "analysis.py").resolve()
    output_dir = (exec_root / "outputs").resolve()
    bind_mounts = [
        f"type=bind,src={script_path},dst=/sandbox/analysis.py,readonly",
        f"type=bind,src={output_dir},dst=/work",
    ]
    for mount in mounts:
        target = _resolve_mount_target(exec_root, mount.target)
        container_target = _container_mount_target(exec_root, target)
        bind_mounts.append(
            f"type=bind,src={mount.source.resolve()},dst={container_target},readonly"
        )

    argv = [
        "docker",
        "run",
        "--rm",
        "--pull",
        "never",
        "--name",
        container_name,
        "--hostname",
        "eda-sandbox",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges=true",
        "--cgroupns",
        "private",
        "--ipc",
        "none",
        "--pids-limit",
        str(pids_limit),
        "--memory",
        str(limits.max_memory_bytes),
        "--memory-swap",
        str(limits.max_memory_bytes),
        "--cpus",
        str(cpus),
        "--shm-size",
        "64m",
        "--ulimit",
        "core=0:0",
        "--ulimit",
        "nofile=256:256",
        "--tmpfs",
        f"/tmp:rw,noexec,nosuid,nodev,size={tmpfs_size}",
        "--workdir",
        "/work",
        "--log-driver",
        "none",
        "--stop-timeout",
        "1",
        "--user",
        container_user or _container_user(),
    ]
    for bind_mount in bind_mounts:
        argv.extend(["--mount", bind_mount])
    argv.extend([image, "python", "-I", "-B", "-u", "/sandbox/analysis.py"])
    return argv


def _validate_mounts(exec_dir: Path, mounts: list[SandboxMount]) -> list[dict[str, object]]:
    exec_root = exec_dir.resolve()
    seen_targets: set[str] = set()
    manifest: list[dict[str, object]] = []
    for mount in mounts:
        source = mount.source
        if not mount.read_only:
            raise ValueError("Writable external sandbox mounts are forbidden.")
        if not source.exists():
            raise ValueError(f"Sandbox mount source does not exist: {source}")
        if source.is_symlink() or not source.is_file():
            raise ValueError("Sandbox inputs must be non-symlink regular files.")
        target = _resolve_mount_target(exec_root, mount.target)
        relative_target = target.relative_to(exec_root).as_posix()
        if relative_target in seen_targets:
            raise ValueError(f"Duplicate sandbox mount target: {relative_target}")
        seen_targets.add(relative_target)
        manifest.append(
            {
                "target": str(_container_mount_target(exec_root, target)),
                "size": source.stat().st_size,
                "sha256": _hash_file(source),
            }
        )
    return manifest


def _stage_mounts(
    exec_dir: Path,
    mounts: list[SandboxMount],
) -> tuple[list[SandboxMount], list[dict[str, object]]]:
    """Copy approved inputs into a private per-execution staging directory.

    The container never bind-mounts the original host file. Opening with
    O_NOFOLLOW and hashing the staged copy closes symlink and source-mutation
    races between validation and `docker run`.
    """

    _validate_mounts(exec_dir, mounts)
    if not mounts:
        return [], []

    staging_root = exec_dir / "staged-inputs"
    staging_root.mkdir(mode=0o700)
    staged_mounts: list[SandboxMount] = []
    manifest: list[dict[str, object]] = []
    nofollow = getattr(os, "O_NOFOLLOW", 0)

    for index, mount in enumerate(mounts):
        source_fd = os.open(mount.source, os.O_RDONLY | nofollow)
        staged_source = staging_root / f"{index:04d}.input"
        try:
            source_stat = os.fstat(source_fd)
            if not stat.S_ISREG(source_stat.st_mode):
                raise ValueError("Sandbox inputs must be non-symlink regular files.")
            with os.fdopen(source_fd, "rb", closefd=True) as source_stream:
                source_fd = -1
                with staged_source.open("xb") as staged_stream:
                    shutil.copyfileobj(source_stream, staged_stream, length=1 << 20)
            if staged_source.stat().st_size != source_stat.st_size:
                raise ValueError("Sandbox input changed while it was being staged.")
            staged_source.chmod(0o400)
        finally:
            if source_fd >= 0:
                os.close(source_fd)

        target = _resolve_mount_target(exec_dir.resolve(), mount.target)
        staged_mounts.append(
            SandboxMount(
                source=staged_source,
                target=target.relative_to(exec_dir.resolve()).as_posix(),
                read_only=True,
            )
        )
        manifest.append(
            {
                "target": str(_container_mount_target(exec_dir.resolve(), target)),
                "size": staged_source.stat().st_size,
                "sha256": _hash_file(staged_source),
            }
        )

    return staged_mounts, manifest


def _container_mount_target(exec_root: Path, target: Path) -> PurePosixPath:
    relative_target = target.relative_to(exec_root)
    if not relative_target.parts or relative_target.parts[0].casefold() != "inputs":
        raise ValueError("Sandbox mount targets must be under inputs/")
    return PurePosixPath("/work") / PurePosixPath(relative_target.as_posix())


def _container_user() -> str:
    getuid = cast(Callable[[], int] | None, getattr(os, "getuid", None))
    getgid = cast(Callable[[], int] | None, getattr(os, "getgid", None))
    if callable(getuid) and callable(getgid):
        uid = int(getuid())
        gid = int(getgid())
        if uid > 0 and gid >= 0:
            return f"{uid}:{gid}"
    return DEFAULT_CONTAINER_USER


def _prepare_output_dir(output_dir: Path, container_user: str) -> None:
    output_dir.chmod(0o700)
    if container_user == DEFAULT_CONTAINER_USER:
        chown = getattr(os, "chown", None)
        if callable(chown):
            uid, gid = (int(part) for part in DEFAULT_CONTAINER_USER.split(":"))
            try:
                chown(output_dir, uid, gid)
            except PermissionError:
                raise ValueError(
                    "Cannot prepare a non-root sandbox output directory for this host user."
                ) from None


def _prepare_input_mountpoints(
    output_dir: Path,
    mounts: list[SandboxMount],
    container_user: str,
) -> None:
    for mount in mounts:
        target = output_dir / mount.target
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch(exist_ok=False)
        target.chmod(0o400)
    if container_user == DEFAULT_CONTAINER_USER:
        chown = getattr(os, "chown", None)
        if callable(chown):
            uid, gid = (int(part) for part in DEFAULT_CONTAINER_USER.split(":"))
            for path in [output_dir / "inputs", *(output_dir / "inputs").rglob("*")]:
                chown(path, uid, gid)


def _remove_input_mountpoints(output_dir: Path, mounts: list[SandboxMount]) -> None:
    for mount in mounts:
        target = output_dir / mount.target
        try:
            target.unlink()
        except FileNotFoundError:
            pass
    inputs_dir = output_dir / "inputs"
    if inputs_dir.is_dir():
        for directory in sorted(
            (path for path in inputs_dir.rglob("*") if path.is_dir()),
            reverse=True,
        ):
            try:
                directory.rmdir()
            except OSError:
                pass
        try:
            inputs_dir.rmdir()
        except OSError:
            pass


def _validate_limits(limits: SandboxLimits, pids_limit: int, cpus: str) -> str | None:
    numeric_limits = {
        "timeout_seconds": limits.timeout_seconds,
        "max_stdout_bytes": limits.max_stdout_bytes,
        "max_stderr_bytes": limits.max_stderr_bytes,
        "max_memory_bytes": limits.max_memory_bytes,
        "max_output_files": limits.max_output_files,
        "max_output_file_bytes": limits.max_output_file_bytes,
        "max_total_output_bytes": limits.max_total_output_bytes,
        "pids_limit": pids_limit,
    }
    if any(value <= 0 for value in numeric_limits.values()):
        return "Sandbox resource limits must all be positive."
    try:
        if float(cpus) <= 0:
            return "Sandbox CPU limit must be positive."
    except ValueError:
        return "Sandbox CPU limit is invalid."
    return None


def _resource_controls_match(
    report: object,
    *,
    memory_bytes: int,
    pids_limit: int,
    cpus: str,
) -> bool:
    if not isinstance(report, dict):
        return False
    cgroup = report.get("cgroup")
    if not isinstance(cgroup, dict):
        return False
    if cgroup.get("memory.max") != str(memory_bytes):
        return False
    if cgroup.get("memory.swap.max") != "0":
        return False
    if cgroup.get("pids.max") != str(pids_limit):
        return False
    raw_cpu_max = cgroup.get("cpu.max")
    if not isinstance(raw_cpu_max, str):
        return False
    try:
        quota, period = raw_cpu_max.split()
        actual_cpus = float(quota) / float(period)
        expected_cpus = float(cpus)
    except (ValueError, ZeroDivisionError):
        return False
    return abs(actual_cpus - expected_cpus) <= 0.001


class _CappedOutput:
    def __init__(self, max_bytes: int) -> None:
        self._max_bytes = max(0, max_bytes)
        self._data = bytearray()
        self.truncated = False

    def append(self, chunk: bytes) -> bool:
        remaining = self._max_bytes - len(self._data)
        if remaining > 0:
            self._data.extend(chunk[:remaining])
        if len(chunk) > remaining:
            self.truncated = True
        return self.truncated

    @property
    def text(self) -> str:
        return bytes(self._data).decode("utf-8", errors="replace")


class _OutputDirectoryWatchdog(threading.Thread):
    def __init__(self, output_dir: Path, limits: SandboxLimits) -> None:
        super().__init__(daemon=True)
        self._output_dir = output_dir
        self._limits = limits
        self._stop_event = threading.Event()
        self.violation: str | None = None

    def run(self) -> None:
        while not self._stop_event.wait(_WAIT_POLL_SECONDS):
            _, violation = _scan_output_dir(
                self._output_dir,
                self._limits,
                include_hashes=False,
            )
            if violation is not None:
                self.violation = violation
                return

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=1.0)


def _scan_output_dir(
    output_dir: Path,
    limits: SandboxLimits,
    *,
    include_hashes: bool,
) -> tuple[list[dict[str, object]], str | None]:
    manifest: list[dict[str, object]] = []
    total_size = 0
    entry_count = 0
    try:
        entries = sorted(output_dir.rglob("*"))
    except OSError:
        return [], "Sandbox output directory could not be inspected."
    for path in entries:
        entry_count += 1
        if entry_count > limits.max_output_files:
            return [], "Sandbox output entry count exceeded the configured limit."
        try:
            mode = path.lstat().st_mode
        except OSError:
            return [], "Sandbox output changed while it was being inspected."
        if stat.S_ISLNK(mode):
            return [], "Sandbox output cannot contain symbolic links."
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            return [], "Sandbox output can contain only regular files and directories."
        size = path.stat().st_size
        if size > limits.max_output_file_bytes:
            return [], "A sandbox output file exceeded the configured size limit."
        total_size += size
        if total_size > limits.max_total_output_bytes:
            return [], "Sandbox output exceeded the configured total size limit."
        item: dict[str, object] = {
            "path": path.relative_to(output_dir).as_posix(),
            "size": size,
        }
        if include_hashes:
            item["sha256"] = _hash_file(path)
        manifest.append(item)
    return manifest, None


def _start_reader(
    stream: IO[bytes] | None,
    output: _CappedOutput,
    output_exceeded: threading.Event,
) -> threading.Thread | None:
    if stream is None:
        return None
    thread = threading.Thread(
        target=_read_stream,
        args=(stream, output, output_exceeded),
        daemon=True,
    )
    thread.start()
    return thread


def _read_stream(
    stream: IO[bytes],
    output: _CappedOutput,
    output_exceeded: threading.Event,
) -> None:
    try:
        while True:
            chunk = stream.read(_OUTPUT_CHUNK_BYTES)
            if not chunk:
                return
            if output.append(chunk):
                output_exceeded.set()
    except OSError:
        return


def _join_readers(readers: list[threading.Thread | None]) -> None:
    for reader in readers:
        if reader is not None:
            reader.join(timeout=1.0)


def _policy_digest(
    *,
    limits: SandboxLimits,
    pids_limit: int,
    cpus: str,
    tmpfs_size: str,
    container_user: str,
) -> str:
    payload = {
        "version": 2,
        "network": "none",
        "read_only_rootfs": True,
        "cap_drop": ["ALL"],
        "no_new_privileges": True,
        "cgroupns": "private",
        "ipc": "none",
        "pid": "private",
        "user": container_user,
        "pids_limit": pids_limit,
        "cpus": cpus,
        "tmpfs_size": tmpfs_size,
        "limits": {
            "timeout_seconds": limits.timeout_seconds,
            "max_stdout_bytes": limits.max_stdout_bytes,
            "max_stderr_bytes": limits.max_stderr_bytes,
            "max_memory_bytes": limits.max_memory_bytes,
            "max_output_files": limits.max_output_files,
            "max_output_file_bytes": limits.max_output_file_bytes,
            "max_total_output_bytes": limits.max_total_output_bytes,
        },
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _seal_artifact(
    artifact: ExecArtifact,
    *,
    exec_dir: Path,
    code: str,
    image: str,
    image_digest: str | None,
    policy_digest: str,
    purpose: str,
    inputs: list[dict[str, object]],
    outputs: list[dict[str, object]],
) -> ExecArtifact:
    stdout_sha256 = _hash_bytes(artifact.stdout.encode("utf-8", errors="replace"))
    stderr_sha256 = _hash_bytes(artifact.stderr.encode("utf-8", errors="replace"))
    manifest = {
        "version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "purpose": purpose,
        "backend": artifact.backend,
        "status": artifact.status,
        "exit_code": artifact.exit_code,
        "error": artifact.error,
        "timed_out": artifact.timed_out,
        "duration_seconds": artifact.duration_seconds,
        "code_sha256": _hash_bytes(code.encode("utf-8")),
        "image": image,
        "image_digest": image_digest,
        "policy_digest": policy_digest,
        "stdout_sha256": stdout_sha256,
        "stderr_sha256": stderr_sha256,
        "inputs": inputs,
        "outputs": outputs,
    }
    encoded = _canonical_json(manifest)
    manifest_path = exec_dir / "execution-manifest.json"
    temporary_path = exec_dir / ".execution-manifest.tmp"
    temporary_path.write_bytes(encoded)
    os.replace(temporary_path, manifest_path)
    manifest_path.chmod(0o444)
    return replace(
        artifact,
        output_manifest=outputs,
        stdout_sha256=stdout_sha256,
        stderr_sha256=stderr_sha256,
        image_digest=image_digest,
        policy_digest=policy_digest,
        manifest_path=manifest_path,
        manifest_sha256=_hash_bytes(encoded),
    )


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _cleanup_container(container_name: str, cli_env: dict[str, str]) -> None:
    try:
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            env=cli_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=_CLEANUP_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return


def _abort_container_run(
    proc: subprocess.Popen[bytes],
    container_name: str,
    cli_env: dict[str, str],
) -> None:
    _cleanup_container(container_name, cli_env)
    _terminate_docker_cli(proc)
    time.sleep(_CLEANUP_RETRY_DELAY_SECONDS)
    _cleanup_container(container_name, cli_env)


def _cancel_container_run(
    proc: subprocess.Popen[bytes],
    container_name: str,
    cli_env: dict[str, str],
) -> None:
    """Cancellation path: TERM/grace/KILL the CLI, then force-clean container."""

    _terminate_docker_cli(proc)
    _cleanup_container(container_name, cli_env)
    time.sleep(_CLEANUP_RETRY_DELAY_SECONDS)
    _cleanup_container(container_name, cli_env)


def _terminate_docker_cli(proc: subprocess.Popen[bytes]) -> None:
    try:
        proc.terminate()
    except OSError:
        return
    try:
        proc.wait(timeout=_CLI_TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except OSError:
            return
        try:
            proc.wait(timeout=_CLI_TERMINATE_GRACE_SECONDS)
        except (OSError, subprocess.TimeoutExpired):
            return
