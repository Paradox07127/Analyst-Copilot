"""The worker env allowlist must not strip the sandbox enforcement switches.

2026-08-12 review: `_WORKER_INHERITED_ENV` replaced full os.environ
inheritance, but `EDA_SANDBOX_REQUIRED` / `EDA_SANDBOX_BACKEND` /
`EDA_SANDBOX_DOCKER_IMAGE` are read inside the worker process
(core/sandbox_broker.py), and DOCKER_* CLI plumbing is read by
core/sandbox_docker.py. Stripping them silently downgraded a hard sandbox
requirement back to "auto" — a security-posture regression the operator
could not see.
"""

from __future__ import annotations

import pytest

from eda_platform.infrastructure.job_backend import _worker_environment


def test_sandbox_enforcement_env_reaches_the_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDA_SANDBOX_REQUIRED", "1")
    monkeypatch.setenv("EDA_SANDBOX_BACKEND", "docker")
    monkeypatch.setenv("EDA_SANDBOX_DOCKER_IMAGE", "eda-sandbox:test")
    env = _worker_environment(None)
    assert env["EDA_SANDBOX_REQUIRED"] == "1"
    assert env["EDA_SANDBOX_BACKEND"] == "docker"
    assert env["EDA_SANDBOX_DOCKER_IMAGE"] == "eda-sandbox:test"


def test_docker_cli_plumbing_reaches_the_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCKER_HOST", "unix:///var/run/docker.sock")
    monkeypatch.setenv("DOCKER_CONFIG", "/etc/docker-config")
    env = _worker_environment(None)
    assert env["DOCKER_HOST"] == "unix:///var/run/docker.sock"
    assert env["DOCKER_CONFIG"] == "/etc/docker-config"


def test_debug_capture_opt_in_reaches_the_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    # EDA_LLM_DEBUG_FULL is an explicit operator opt-in read at call time in
    # the worker (core/dev_log.py); stripping it made the opt-in a no-op.
    monkeypatch.setenv("EDA_LLM_DEBUG_FULL", "1")
    env = _worker_environment(None)
    assert env["EDA_LLM_DEBUG_FULL"] == "1"


def test_home_is_inherited(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", "/Users/operator")
    env = _worker_environment(None)
    assert env["HOME"] == "/Users/operator"


def test_credentials_and_desktop_tokens_are_still_stripped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "not-for-workers")
    monkeypatch.setenv("GITHUB_TOKEN", "not-for-workers")
    env = _worker_environment(None)
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert "GITHUB_TOKEN" not in env


def test_unknown_overlay_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported worker environment keys"):
        _worker_environment({"EDA_SANDBOX_REQUIRED": "0"})


def test_llm_overlay_keys_are_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    env = _worker_environment({"EDA_LLM_PROVIDER": "deepseek", "EDA_LLM_API_KEY": "k"})
    assert env["EDA_LLM_PROVIDER"] == "deepseek"
    assert env["EDA_LLM_API_KEY"] == "k"


def test_observability_and_debug_env_reach_the_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex pre-commit review (2026-08-13): worker-side trace persistence
    reads these keys, so stripping them silently disabled span export and
    debug.jsonl for every worker while the API process kept mirroring."""
    keys = (
        "EDA_OBSERVABILITY",
        "PHOENIX_COLLECTOR_ENDPOINT",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "EDA_DEBUG_LOG",
    )
    for key in keys:
        monkeypatch.setenv(key, "1")
    env = _worker_environment(None)
    assert all(key in env for key in keys)


def test_llm_organization_reaches_the_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EDA_LLM_ORGANIZATION", raising=False)
    env = _worker_environment({"EDA_LLM_ORGANIZATION": "org_123"})
    assert env["EDA_LLM_ORGANIZATION"] == "org_123"
