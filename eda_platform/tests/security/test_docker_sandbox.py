from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from eda_platform.core.sandbox import SandboxLimits, SandboxMount
from eda_platform.core.sandbox_docker import (
    DEFAULT_DOCKER_IMAGE,
    DockerSandboxBackend,
    _docker_argv,
    _docker_cli_env,
    _resource_controls_match,
    _scan_output_dir,
    _stage_mounts,
    _validate_mounts,
    docker_available,
    docker_image_available,
)


class _BytesStream:
    def __init__(self, data: bytes, *, chunk_size: int | None = None) -> None:
        self._data = data
        self._chunk_size = chunk_size
        self._offset = 0

    def read(self, size: int = -1) -> bytes:
        if self._offset >= len(self._data):
            return b""
        if self._chunk_size is not None:
            size = self._chunk_size if size < 0 else min(size, self._chunk_size)
        if size < 0:
            size = len(self._data) - self._offset
        end = min(len(self._data), self._offset + size)
        chunk = self._data[self._offset : end]
        self._offset = end
        return chunk


class _FakeDockerProcess:
    def __init__(
        self,
        *,
        stdout: bytes = b"",
        stderr: bytes = b"",
        returncode: int = 0,
        timeout: bool = False,
    ) -> None:
        self.stdout = _BytesStream(stdout, chunk_size=4)
        self.stderr = _BytesStream(stderr, chunk_size=4)
        self.returncode = returncode
        self.timeout = timeout
        self.terminated = False
        self.killed = False

    def wait(self, timeout: float | None = None) -> int:
        if self.timeout:
            raise subprocess.TimeoutExpired(["docker", "run"], timeout or 0.0)
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


def _mock_runtime_available(monkeypatch) -> None:
    monkeypatch.setattr("eda_platform.core.sandbox_docker.docker_available", lambda: True)
    monkeypatch.setattr(
        "eda_platform.core.sandbox_docker.docker_security_available",
        lambda: (True, "security available"),
    )
    monkeypatch.setattr(
        "eda_platform.core.sandbox_docker.docker_image_available", lambda image: True
    )
    monkeypatch.setattr(
        "eda_platform.core.sandbox_docker.docker_image_digest",
        lambda image: "sha256:test-image",
    )


def test_docker_backend_reports_unavailable_when_cli_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)

    backend = DockerSandboxBackend(work_root=tmp_path)
    info = backend.info

    assert info.name == "docker"
    assert info.available is False
    assert info.safe_for_untrusted_code is True
    assert "unavailable" in info.detail.lower()


def test_docker_run_python_blocks_when_docker_unavailable(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("eda_platform.core.sandbox_docker.docker_available", lambda: False)

    backend = DockerSandboxBackend(work_root=tmp_path)
    result = backend.run_python("print('should not run')", limits=SandboxLimits(timeout_seconds=1))

    assert result.status == "blocked"
    assert result.backend == "docker"
    assert result.error is not None
    assert "docker unavailable" in result.error.lower()
    assert result.work_dir is not None
    assert result.work_dir.exists()


def test_docker_backend_reports_unavailable_when_image_is_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("eda_platform.core.sandbox_docker.docker_available", lambda: True)
    monkeypatch.setattr(
        "eda_platform.core.sandbox_docker.docker_security_available",
        lambda: (True, "security available"),
    )
    monkeypatch.setattr(
        "eda_platform.core.sandbox_docker.docker_image_available", lambda image: False
    )

    info = DockerSandboxBackend(work_root=tmp_path).info

    assert info.available is False
    assert "image" in info.detail.lower()
    assert DEFAULT_DOCKER_IMAGE in info.detail


def test_docker_run_python_blocks_before_launch_when_image_is_missing(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr("eda_platform.core.sandbox_docker.docker_available", lambda: True)
    monkeypatch.setattr(
        "eda_platform.core.sandbox_docker.docker_security_available",
        lambda: (True, "security available"),
    )
    monkeypatch.setattr(
        "eda_platform.core.sandbox_docker.docker_image_available", lambda image: False
    )

    def fail_if_launched(*args, **kwargs):
        raise AssertionError("Docker process should not launch when the image is unavailable")

    monkeypatch.setattr(subprocess, "Popen", fail_if_launched)
    result = DockerSandboxBackend(work_root=tmp_path).run_python(
        "print('should not run')", limits=SandboxLimits(timeout_seconds=1)
    )

    assert result.status == "blocked"
    assert result.error is not None
    assert "image" in result.error.lower()


@pytest.mark.parametrize(
    ("code", "expected_error"),
    [
        ("import subprocess\nsubprocess.run(['sh', '-lc', 'id'])\n", "subprocess"),
        ("import socket\nsocket.socket().connect(('1.1.1.1', 53))\n", "socket"),
        ("from pathlib import Path\nprint(Path('/etc/passwd').read_text())\n", "path"),
    ],
)
def test_docker_run_python_statically_blocks_escape_code_before_launch(
    monkeypatch, tmp_path, code: str, expected_error: str
) -> None:
    _mock_runtime_available(monkeypatch)
    popen_calls: list[list[str]] = []

    def fake_popen(argv, **kwargs):
        popen_calls.append(argv)
        raise AssertionError("Docker process should not launch for static policy violations")

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    result = DockerSandboxBackend(work_root=tmp_path).run_python(
        code,
        limits=SandboxLimits(timeout_seconds=1),
    )

    assert result.status == "blocked"
    assert result.backend == "docker"
    assert result.error is not None
    assert expected_error in result.error.lower()
    assert popen_calls == []
    assert result.work_dir is not None
    assert result.work_dir.exists()


def test_docker_cli_env_preserves_only_path_and_docker_variables(monkeypatch) -> None:
    monkeypatch.setenv("PATH", "/custom/bin")
    monkeypatch.setenv("DOCKER_HOST", "unix:///tmp/docker.sock")
    monkeypatch.setenv("DOCKER_CONTEXT", "desktop-linux")
    monkeypatch.setenv("DOCKER_CONFIG", "/tmp/docker-config")
    monkeypatch.setenv("DOCKER_TLS_VERIFY", "1")
    monkeypatch.setenv("DOCKER_CERT_PATH", "/tmp/certs")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "should-not-leak")
    monkeypatch.setenv("PYTHONPATH", "should-not-leak")

    env = _docker_cli_env()

    assert env == {
        "PATH": "/custom/bin",
        "DOCKER_HOST": "unix:///tmp/docker.sock",
        "DOCKER_CONTEXT": "desktop-linux",
        "DOCKER_CONFIG": "/tmp/docker-config",
        "DOCKER_TLS_VERIFY": "1",
        "DOCKER_CERT_PATH": "/tmp/certs",
    }


def test_docker_argv_builds_hardened_non_networked_container_command(tmp_path) -> None:
    exec_dir = tmp_path / "exec"
    exec_dir.mkdir()
    (exec_dir / "analysis.py").write_text("print('ok')", encoding="utf-8")
    (exec_dir / "outputs").mkdir()
    source = tmp_path / "source.csv"
    source.write_text("group,value\na,1\n", encoding="utf-8")
    limits = SandboxLimits(timeout_seconds=2, max_memory_bytes=128 << 20)

    argv = _docker_argv(
        exec_dir=exec_dir,
        container_name="eda-agent-test",
        mounts=[SandboxMount(source=source, target="inputs/source.csv", read_only=True)],
        limits=limits,
        image=DEFAULT_DOCKER_IMAGE,
        container_user="1234:1234",
    )

    assert argv[:3] == ["docker", "run", "--rm"]
    assert argv[argv.index("--pull") + 1] == "never"
    assert "--name" in argv
    assert argv[argv.index("--name") + 1] == "eda-agent-test"
    assert "--network" in argv
    assert argv[argv.index("--network") + 1] == "none"
    assert "--read-only" in argv
    assert "--cap-drop" in argv
    assert argv[argv.index("--cap-drop") + 1] == "ALL"
    assert "--security-opt" in argv
    assert argv[argv.index("--security-opt") + 1] == "no-new-privileges=true"
    assert argv[argv.index("--cgroupns") + 1] == "private"
    assert argv[argv.index("--ipc") + 1] == "none"
    assert "--pid" not in argv
    assert "--tmpfs" in argv
    assert argv[argv.index("--tmpfs") + 1].startswith("/tmp:rw,noexec,nosuid,nodev,size=")
    assert "--memory-swap" in argv
    assert argv[argv.index("--memory-swap") + 1] == str(limits.max_memory_bytes)
    assert f"type=bind,src={exec_dir / 'analysis.py'},dst=/sandbox/analysis.py,readonly" in argv
    assert f"type=bind,src={exec_dir / 'outputs'},dst=/work" in argv
    assert f"type=bind,src={source},dst=/work/inputs/source.csv,readonly" in argv
    assert DEFAULT_DOCKER_IMAGE in argv
    assert argv[-6:] == [
        DEFAULT_DOCKER_IMAGE,
        "python",
        "-I",
        "-B",
        "-u",
        "/sandbox/analysis.py",
    ]
    assert argv[argv.index("--user") + 1] == "1234:1234"


def test_runtime_preflight_resource_report_fails_closed_on_any_mismatch() -> None:
    cgroup = {
        "memory.max": str(128 << 20),
        "memory.swap.max": "0",
        "pids.max": "128",
        "cpu.max": "100000 100000",
    }
    report = {"cgroup": cgroup}

    assert _resource_controls_match(
        report,
        memory_bytes=128 << 20,
        pids_limit=128,
        cpus="1",
    )
    for field, bad_value in (
        ("memory.max", "max"),
        ("memory.swap.max", "134217728"),
        ("pids.max", "max"),
        ("cpu.max", "max 100000"),
    ):
        mutated = {"cgroup": {**cgroup, field: bad_value}}
        assert not _resource_controls_match(
            mutated,
            memory_bytes=128 << 20,
            pids_limit=128,
            cpus="1",
        )
    assert not _resource_controls_match(
        {},
        memory_bytes=128 << 20,
        pids_limit=128,
        cpus="1",
    )


@pytest.mark.parametrize(
    "target",
    [
        ".",
        "analysis.py",
        "data/source.csv",
        "work/source.csv",
        "inputs/../analysis.py",
    ],
)
def test_docker_argv_rejects_mount_targets_outside_inputs(
    tmp_path, target: str
) -> None:
    exec_dir = tmp_path / "exec"
    exec_dir.mkdir()
    source = tmp_path / "source.csv"
    source.write_text("group,value\na,1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="under inputs"):
        _docker_argv(
            exec_dir=exec_dir,
            container_name="eda-agent-test",
            mounts=[SandboxMount(source=source, target=target, read_only=True)],
            limits=SandboxLimits(timeout_seconds=2),
            image=DEFAULT_DOCKER_IMAGE,
        )


@pytest.mark.parametrize(
    ("target", "expected_error"),
    [
        ("/host/path.csv", "relative path"),
        ("../outside.csv", "outside execution directory"),
        ("nested/../../outside.csv", "outside execution directory"),
    ],
)
def test_docker_argv_rejects_absolute_and_parent_traversal_mount_targets(
    tmp_path, target: str, expected_error: str
) -> None:
    exec_dir = tmp_path / "exec"
    exec_dir.mkdir()
    source = tmp_path / "source.csv"
    source.write_text("group,value\na,1\n", encoding="utf-8")

    with pytest.raises(ValueError, match=expected_error):
        _docker_argv(
            exec_dir=exec_dir,
            container_name="eda-agent-test",
            mounts=[SandboxMount(source=source, target=target, read_only=True)],
            limits=SandboxLimits(timeout_seconds=2),
            image=DEFAULT_DOCKER_IMAGE,
        )


def test_docker_argv_uses_resolved_mount_target_for_container_destination(tmp_path) -> None:
    exec_dir = tmp_path / "exec"
    exec_dir.mkdir()
    source = tmp_path / "source.csv"
    source.write_text("group,value\na,1\n", encoding="utf-8")

    argv = _docker_argv(
        exec_dir=exec_dir,
        container_name="eda-agent-test",
        mounts=[
            SandboxMount(
                source=source,
                target="inputs/nested/../source.csv",
                read_only=True,
            )
        ],
        limits=SandboxLimits(timeout_seconds=2),
        image=DEFAULT_DOCKER_IMAGE,
    )

    assert f"type=bind,src={source},dst=/work/inputs/source.csv,readonly" in argv
    assert f"type=bind,src={source},dst=/work/inputs/nested/../source.csv,readonly" not in argv


def test_docker_mount_policy_rejects_writable_and_directory_inputs(tmp_path: Path) -> None:
    exec_dir = tmp_path / "exec"
    exec_dir.mkdir()
    source_file = tmp_path / "source.csv"
    source_file.write_text("a\n1\n", encoding="utf-8")
    source_dir = tmp_path / "source-dir"
    source_dir.mkdir()

    with pytest.raises(ValueError, match="Writable external"):
        _validate_mounts(
            exec_dir,
            [SandboxMount(source=source_file, target="inputs/source.csv", read_only=False)],
        )
    with pytest.raises(ValueError, match="regular files"):
        _validate_mounts(
            exec_dir,
            [SandboxMount(source=source_dir, target="inputs/source", read_only=True)],
        )


def test_docker_mount_policy_rejects_symlink_inputs(tmp_path: Path) -> None:
    exec_dir = tmp_path / "exec"
    exec_dir.mkdir()
    source = tmp_path / "source.csv"
    source.write_text("a\n1\n", encoding="utf-8")
    link = tmp_path / "source-link.csv"
    link.symlink_to(source)

    with pytest.raises(ValueError, match="non-symlink regular files"):
        _validate_mounts(
            exec_dir,
            [SandboxMount(source=link, target="inputs/source.csv", read_only=True)],
        )


def test_docker_mounts_only_private_staged_copies(tmp_path: Path) -> None:
    exec_dir = tmp_path / "exec"
    exec_dir.mkdir()
    source = tmp_path / "source.csv"
    source.write_text("a\n1\n", encoding="utf-8")

    staged_mounts, manifest = _stage_mounts(
        exec_dir,
        [SandboxMount(source=source, target="inputs/source.csv", read_only=True)],
    )

    assert len(staged_mounts) == 1
    assert staged_mounts[0].source.parent == exec_dir / "staged-inputs"
    assert staged_mounts[0].source != source
    assert staged_mounts[0].source.read_bytes() == source.read_bytes()
    assert staged_mounts[0].target == "inputs/source.csv"
    assert manifest[0]["target"] == "/work/inputs/source.csv"
    assert "source" not in manifest[0]


def test_output_policy_rejects_symlinks_and_size_excess(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    target = output_dir / "target.txt"
    target.write_text("safe", encoding="utf-8")
    link = output_dir / "link.txt"
    link.symlink_to(target)

    _, symlink_error = _scan_output_dir(
        output_dir,
        SandboxLimits(),
        include_hashes=True,
    )
    assert symlink_error is not None and "symbolic links" in symlink_error

    link.unlink()
    target.write_text("too-large", encoding="utf-8")
    _, size_error = _scan_output_dir(
        output_dir,
        SandboxLimits(max_output_file_bytes=4),
        include_hashes=True,
    )
    assert size_error is not None and "size limit" in size_error


def test_docker_run_python_cleans_container_and_cli_on_timeout(monkeypatch, tmp_path) -> None:
    _mock_runtime_available(monkeypatch)
    fake_proc = _FakeDockerProcess(stdout=b"START\n", timeout=True)
    cleanup_calls: list[tuple[list[str], dict[str, str], bool, bool]] = []
    run_container_names: list[str] = []

    def fake_popen(argv, **kwargs):
        assert "--name" in argv
        run_container_names.append(argv[argv.index("--name") + 1])
        return fake_proc

    def fake_run(argv, **kwargs):
        cleanup_calls.append((argv, kwargs["env"], fake_proc.terminated, fake_proc.killed))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = DockerSandboxBackend(work_root=tmp_path).run_python(
        "print('START')",
        limits=SandboxLimits(timeout_seconds=0.01),
    )

    assert result.status == "timeout"
    assert result.timed_out is True
    assert result.stdout == "START\n"
    assert fake_proc.terminated is True
    assert fake_proc.killed is True
    assert len(cleanup_calls) == 2
    first_cleanup_argv, first_cleanup_env, first_terminated, first_killed = cleanup_calls[0]
    second_cleanup_argv, second_cleanup_env, second_terminated, second_killed = cleanup_calls[1]
    assert first_cleanup_argv == ["docker", "rm", "-f", run_container_names[0]]
    assert second_cleanup_argv == ["docker", "rm", "-f", run_container_names[0]]
    assert run_container_names[0].startswith("eda-agent-")
    assert first_cleanup_env == _docker_cli_env()
    assert second_cleanup_env == _docker_cli_env()
    assert (first_terminated, first_killed) == (False, False)
    assert (second_terminated, second_killed) == (True, True)


def test_docker_run_python_cleans_container_when_output_cap_is_exceeded(
    monkeypatch, tmp_path
) -> None:
    _mock_runtime_available(monkeypatch)
    fake_proc = _FakeDockerProcess(stdout=b"abcdef", stderr=b"error", returncode=0)
    cleanup_calls: list[tuple[list[str], bool]] = []
    run_container_names: list[str] = []

    def fake_popen(argv, **kwargs):
        assert "--name" in argv
        run_container_names.append(argv[argv.index("--name") + 1])
        return fake_proc

    def fake_run(argv, **kwargs):
        cleanup_calls.append((argv, fake_proc.terminated))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = DockerSandboxBackend(work_root=tmp_path).run_python(
        "print('abcdef')",
        limits=SandboxLimits(timeout_seconds=2, max_stdout_bytes=5, max_stderr_bytes=20),
    )

    assert result.status == "blocked"
    assert result.output_truncated is True
    assert result.stdout == "abcde"
    assert result.stderr == "error"
    assert result.error is not None
    assert "output exceeded configured byte limit" in result.error.lower()
    assert fake_proc.terminated is True
    assert len(cleanup_calls) == 2
    assert cleanup_calls[0] == (["docker", "rm", "-f", run_container_names[0]], False)
    assert cleanup_calls[1] == (["docker", "rm", "-f", run_container_names[0]], True)


def test_docker_run_python_cleans_container_when_stderr_cap_is_exceeded(
    monkeypatch, tmp_path
) -> None:
    _mock_runtime_available(monkeypatch)
    fake_proc = _FakeDockerProcess(stdout=b"ok", stderr=b"abcdef", returncode=0)
    cleanup_calls: list[tuple[list[str], bool]] = []
    run_container_names: list[str] = []

    def fake_popen(argv, **kwargs):
        assert "--name" in argv
        run_container_names.append(argv[argv.index("--name") + 1])
        return fake_proc

    def fake_run(argv, **kwargs):
        cleanup_calls.append((argv, fake_proc.terminated))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = DockerSandboxBackend(work_root=tmp_path).run_python(
        "print('stderr cap fixture')\n",
        limits=SandboxLimits(timeout_seconds=2, max_stdout_bytes=20, max_stderr_bytes=5),
    )

    assert result.status == "blocked"
    assert result.output_truncated is True
    assert result.stdout == "ok"
    assert result.stderr == "abcde"
    assert result.error is not None
    assert "output exceeded configured byte limit" in result.error.lower()
    assert fake_proc.terminated is True
    assert len(cleanup_calls) == 2
    assert cleanup_calls[0] == (["docker", "rm", "-f", run_container_names[0]], False)
    assert cleanup_calls[1] == (["docker", "rm", "-f", run_container_names[0]], True)


requires_docker = pytest.mark.skipif(
    not (docker_available() and docker_image_available(DEFAULT_DOCKER_IMAGE)),
    reason="Docker CLI, daemon, or default sandbox image unavailable",
)


def _sandbox_container_names() -> set[str]:
    completed = subprocess.run(
        [
            "docker",
            "ps",
            "--all",
            "--filter",
            "name=eda-agent-",
            "--format",
            "{{.Names}}",
        ],
        capture_output=True,
        text=True,
        env=_docker_cli_env(),
        timeout=5,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return {name for name in completed.stdout.splitlines() if name}


def test_dockerfile_installs_the_repository_locked_runtime() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    dockerfile = (repository_root / "docker" / "eda-agent-sandbox" / "Dockerfile").read_text(
        encoding="utf-8"
    )
    requirements = (
        repository_root / "docker" / "eda-agent-sandbox" / "requirements.lock"
    ).read_text(encoding="utf-8")

    assert "FROM python:3.12.11-slim-bookworm" in dockerfile
    assert "@sha256:" in dockerfile
    assert "COPY requirements.lock" in dockerfile
    assert "-r /tmp/requirements.lock" in dockerfile
    assert "--require-hashes" in dockerfile
    assert "USER sandbox:sandbox" in dockerfile
    assert "matplotlib==3.10.8" in requirements
    assert "pandas==3.0.3" in requirements
    assert "--hash=sha256:" in requirements
    assert "streamlit==" not in requirements
    assert "requests==" not in requirements


@requires_docker
def test_docker_backend_pandas_analysis_with_read_only_input_succeeds(tmp_path: Path) -> None:
    source = tmp_path / "sales.csv"
    source.write_text(
        "region,amount\n"
        "east,10\n"
        "west,7\n"
        "east,5\n",
        encoding="utf-8",
    )
    backend = DockerSandboxBackend(work_root=tmp_path / "work")

    result = backend.run_python(
        "import pandas as pd\n"
        "df = pd.read_csv('inputs/sales.csv')\n"
        "summary = df.groupby('region')['amount'].sum().sort_index()\n"
        "for region, amount in summary.items():\n"
        "    print(f'{region}:{int(amount)}')\n",
        mounts=[SandboxMount(source=source, target="inputs/sales.csv", read_only=True)],
        limits=SandboxLimits(timeout_seconds=30),
    )

    assert result.status == "succeeded", f"stderr={result.stderr!r} error={result.error!r}"
    assert "east:15" in result.stdout
    assert "west:7" in result.stdout
    assert result.image_digest is not None
    assert result.policy_digest is not None
    assert result.stdout_sha256 is not None
    assert result.manifest_path is not None and result.manifest_path.is_file()
    assert result.manifest_sha256 is not None


@requires_docker
def test_docker_backend_minimal_scientific_runtime_is_complete(tmp_path: Path) -> None:
    result = DockerSandboxBackend(work_root=tmp_path / "work").run_python(
        "import duckdb\n"
        "import matplotlib\n"
        "import numpy\n"
        "import pandas\n"
        "import scipy\n"
        "import sklearn\n"
        "print('SCIENTIFIC_STACK_OK')\n",
        limits=SandboxLimits(timeout_seconds=30),
    )

    assert result.status == "succeeded", f"stderr={result.stderr!r} error={result.error!r}"
    assert "SCIENTIFIC_STACK_OK" in result.stdout


@requires_docker
def test_docker_backend_network_attempt_using_socket_fails_or_times_out(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr("eda_platform.core.sandbox_docker._policy_violation", lambda code: None)
    backend = DockerSandboxBackend(work_root=tmp_path)

    result = backend.run_python(
        "import socket\n"
        "print('NETWORK_START')\n"
        "sock = socket.create_connection(('example.com', 80), timeout=1)\n"
        "sock.sendall(b'GET / HTTP/1.0\\r\\n\\r\\n')\n"
        "print('NETWORK_OK')\n",
        limits=SandboxLimits(timeout_seconds=5),
    )

    assert result.status in {"blocked", "failed", "timeout"}
    assert "NETWORK_START" in result.stdout
    assert "NETWORK_OK" not in result.stdout


@requires_docker
def test_docker_backend_read_only_mount_write_does_not_modify_source(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    original = "name,value\nalpha,1\n"
    source.write_text(original, encoding="utf-8")
    backend = DockerSandboxBackend(work_root=tmp_path / "work")

    result = backend.run_python(
        "import pandas as pd\n"
        "print('WRITE_START')\n"
        "df = pd.read_csv('inputs/source.csv')\n"
        "try:\n"
        "    df.assign(value=999).to_csv('inputs/source.csv', index=False)\n"
        "    print('WRITE_OK')\n"
        "except Exception as exc:\n"
        "    print('WRITE_BLOCKED', type(exc).__name__)\n",
        mounts=[SandboxMount(source=source, target="inputs/source.csv", read_only=True)],
        limits=SandboxLimits(timeout_seconds=30),
    )

    assert source.read_text(encoding="utf-8") == original
    assert "WRITE_START" in result.stdout
    assert "WRITE_OK" not in result.stdout
    assert result.status in {"succeeded", "failed"}


@requires_docker
def test_docker_backend_cannot_read_unmounted_host_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = tmp_path / "host-secret.txt"
    secret.write_text("must-not-leak", encoding="utf-8")
    monkeypatch.setattr("eda_platform.core.sandbox_docker._policy_violation", lambda code: None)

    result = DockerSandboxBackend(work_root=tmp_path / "work").run_python(
        "from pathlib import Path\n"
        f"print(Path({str(secret)!r}).read_text(encoding='utf-8'))\n",
        limits=SandboxLimits(timeout_seconds=10),
    )

    assert result.status == "failed"
    assert "must-not-leak" not in result.stdout
    assert secret.read_text(encoding="utf-8") == "must-not-leak"


@requires_docker
def test_later_sandbox_cannot_modify_prior_execution_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("eda_platform.core.sandbox_docker._policy_violation", lambda code: None)
    backend = DockerSandboxBackend(work_root=tmp_path / "work")
    first = backend.run_python(
        "from pathlib import Path\n"
        "Path('result.txt').write_text('sealed', encoding='utf-8')\n",
        limits=SandboxLimits(timeout_seconds=10),
    )
    assert first.status == "succeeded"
    assert first.work_dir is not None
    prior_result = first.work_dir / "result.txt"
    assert prior_result.read_text(encoding="utf-8") == "sealed"

    second = backend.run_python(
        "from pathlib import Path\n"
        f"Path({str(prior_result)!r}).write_text('tampered', encoding='utf-8')\n",
        limits=SandboxLimits(timeout_seconds=10),
    )

    assert second.status == "failed"
    assert prior_result.read_text(encoding="utf-8") == "sealed"


@requires_docker
def test_docker_runtime_preflight_proves_kernel_controls(tmp_path: Path) -> None:
    work_root = tmp_path / "preflight"
    info = DockerSandboxBackend(work_root=work_root).verify_runtime()

    assert info.available is True, info.detail
    assert info.safe_for_untrusted_code is True
    assert "preflight passed" in info.detail.lower()
    assert list(work_root.iterdir()) == []


@requires_docker
def test_docker_runtime_memory_ceiling_kills_bounded_allocation_attack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prove the daemon/cgroup enforces the limit, not merely that argv contains it."""
    monkeypatch.setattr("eda_platform.core.sandbox_docker._policy_violation", lambda code: None)
    containers_before = _sandbox_container_names()

    result = DockerSandboxBackend(work_root=tmp_path / "work").run_python(
        "print('MEMORY_ATTACK_STARTED', flush=True)\n"
        "blocks = []\n"
        "for _ in range(32):\n"
        "    blocks.append(bytearray(8 * 1024 * 1024))\n"
        "print('ALLOCATED', flush=True)\n",
        limits=SandboxLimits(
            timeout_seconds=20,
            max_memory_bytes=64 << 20,
        ),
    )

    assert result.status == "failed", (
        f"status={result.status!r} stdout={result.stdout!r} "
        f"stderr={result.stderr!r} error={result.error!r}"
    )
    assert result.timed_out is False
    assert result.exit_code not in {None, 0}
    assert "MEMORY_ATTACK_STARTED" in result.stdout
    assert "ALLOCATED" not in result.stdout
    assert result.error is not None
    assert result.manifest_path is not None and result.manifest_path.is_file()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["exit_code"] == result.exit_code
    assert manifest["error"] == result.error
    assert _sandbox_container_names() == containers_before


@requires_docker
def test_docker_runtime_pids_ceiling_bounds_child_process_attack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("eda_platform.core.sandbox_docker._policy_violation", lambda code: None)
    configured_limit = 16
    bounded_margin = 8
    containers_before = _sandbox_container_names()

    result = DockerSandboxBackend(
        work_root=tmp_path / "work",
        pids_limit=configured_limit,
    ).run_python(
        "import json, subprocess, sys\n"
        "children = []\n"
        "failure = None\n"
        "try:\n"
        f"    for _ in range({configured_limit + bounded_margin}):\n"
        "        children.append(subprocess.Popen(\n"
        "            [sys.executable, '-c', 'import time; time.sleep(5)'],\n"
        "            stdout=subprocess.DEVNULL,\n"
        "            stderr=subprocess.DEVNULL,\n"
        "        ))\n"
        "except OSError as exc:\n"
        "    failure = type(exc).__name__\n"
        "finally:\n"
        "    print(json.dumps({'spawned': len(children), 'failure': failure}), flush=True)\n"
        "    for child in children:\n"
        "        child.terminate()\n"
        "    for child in children:\n"
        "        child.wait(timeout=2)\n",
        limits=SandboxLimits(timeout_seconds=20),
    )

    assert result.status == "succeeded", (
        f"stdout={result.stdout!r} stderr={result.stderr!r} error={result.error!r}"
    )
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert 0 < report["spawned"] <= configured_limit
    assert report["failure"] == "BlockingIOError"
    assert _sandbox_container_names() == containers_before


@requires_docker
def test_docker_runtime_cpu_quota_bounds_multiworker_busy_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("eda_platform.core.sandbox_docker._policy_violation", lambda code: None)
    configured_cpus = 0.5
    containers_before = _sandbox_container_names()

    result = DockerSandboxBackend(
        work_root=tmp_path / "work",
        cpus=str(configured_cpus),
    ).run_python(
        "import json, multiprocessing, resource, time\n"
        "def burn(stop_at):\n"
        "    value = 0\n"
        "    while time.monotonic() < stop_at:\n"
        "        value = (value + 1) % 1000003\n"
        "if __name__ == '__main__':\n"
        "    started = time.monotonic()\n"
        "    stop_at = started + 2.0\n"
        "    workers = [multiprocessing.Process(target=burn, args=(stop_at,)) for _ in range(4)]\n"
        "    for worker in workers:\n"
        "        worker.start()\n"
        "    for worker in workers:\n"
        "        worker.join()\n"
        "    usage = resource.getrusage(resource.RUSAGE_CHILDREN)\n"
        "    cpu_max = open('/sys/fs/cgroup/cpu.max', encoding='utf-8').read().strip()\n"
        "    print(json.dumps({\n"
        "        'cpu_max': cpu_max,\n"
        "        'child_cpu_seconds': usage.ru_utime + usage.ru_stime,\n"
        "        'wall_seconds': time.monotonic() - started,\n"
        "    }), flush=True)\n",
        limits=SandboxLimits(timeout_seconds=20),
    )

    assert result.status == "succeeded", (
        f"stdout={result.stdout!r} stderr={result.stderr!r} error={result.error!r}"
    )
    report = json.loads(result.stdout.strip().splitlines()[-1])
    quota, period = report["cpu_max"].split()
    assert quota != "max"
    assert float(quota) / float(period) == pytest.approx(configured_cpus, abs=0.05)
    calibrated_ceiling = configured_cpus * report["wall_seconds"] + 0.5
    assert report["child_cpu_seconds"] <= calibrated_ceiling
    assert _sandbox_container_names() == containers_before
