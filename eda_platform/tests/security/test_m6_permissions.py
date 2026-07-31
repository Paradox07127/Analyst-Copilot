from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar, cast

import pytest
from pydantic import BaseModel

from eda_platform.agents.code_agent import CodeAgent
from eda_platform.core import sandbox as sandbox_module
from eda_platform.core import sandbox_broker as sandbox_broker_module
from eda_platform.core.budget import Budget
from eda_platform.core.permissions import (
    PermissionTier,
    action_hash,
    classify_action,
    require_permission,
)
from eda_platform.core.sandbox import (
    ExecArtifact,
    PortableSubprocessBackend,
    SandboxBackendInfo,
    SandboxLimits,
    SandboxMount,
    SandboxUnavailableError,
)
from eda_platform.core.sandbox_broker import SandboxBroker, SandboxSettings
from eda_platform.drivers import chat as chat_driver
from eda_platform.schemas.plans import Intent

T = TypeVar("T", bound=BaseModel)


class _AlwaysBadCodeLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        self.calls.append({"task": task, "payload": payload})
        return schema.model_validate({"code": "raise RuntimeError('still bad')"})

    def text(self, *, task: str, payload: dict) -> str:
        return ""

    def last_usage(self) -> None:
        return None


class _FakeSafeBackend:
    @property
    def info(self) -> SandboxBackendInfo:
        return SandboxBackendInfo(
            name="fake_safe",
            safe_for_untrusted_code=True,
            available=True,
            detail="fake safe backend for tests",
        )

    def run_python(
        self,
        code: str,
        *,
        mounts: list[SandboxMount] | None = None,
        limits: SandboxLimits | None = None,
    ) -> ExecArtifact:
        return ExecArtifact(status="succeeded", backend="fake_safe")


def test_rewritten_select_file_read_is_denied_before_dispatch() -> None:
    decision = require_permission(
        {
            "type": "duckdb_select",
            "sql": "WITH q AS (SELECT * FROM read_csv('/etc/passwd')) SELECT * FROM q",
        }
    )

    assert decision.tier is PermissionTier.DENY
    assert "read-only SELECT" in decision.feedback
    assert "read_csv" in decision.feedback


def test_direct_cleaning_apply_without_approval_is_blocked() -> None:
    decision = require_permission(
        {
            "type": "cleaning_apply",
            "dataset_id": "ds_customers",
            "recipe_id": "recipe_lossy",
            "transform_ids": ["drop_rows_missing_email"],
            "reversible": False,
        }
    )

    assert decision.tier is PermissionTier.DENY
    assert "requires confirmation" in decision.feedback


def test_action_swap_after_confirmation_is_blocked_by_hash() -> None:
    approved = {
        "type": "cleaning_apply",
        "dataset_id": "ds_customers",
        "recipe_id": "recipe_safe",
        "transform_ids": ["trim_whitespace"],
        "reversible": True,
    }
    swapped = {
        "type": "cleaning_apply",
        "dataset_id": "ds_customers",
        "recipe_id": "recipe_drop",
        "transform_ids": ["drop_column_email"],
        "reversible": False,
    }

    decision = require_permission(swapped, approved_hash=action_hash(approved))

    assert decision.tier is PermissionTier.DENY
    assert "approval hash" in decision.feedback.lower()


def test_code_agent_host_write_is_flagged_and_sandbox_blocks_it(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    code = f"open({str(outside)!r}, 'w').write('owned')"

    decision = classify_action({"type": "sandboxed_code", "code": code, "sandboxed": True})
    backend = PortableSubprocessBackend(work_root=tmp_path / "sandbox")
    artifact = backend.run_python(code, limits=SandboxLimits(timeout_seconds=2))

    assert decision.tier is PermissionTier.DENY
    assert "outside the sandbox" in decision.feedback
    assert artifact.status == "blocked"
    assert not outside.exists()


def test_code_agent_loop_terminates_after_three_failed_attempts(tmp_path: Path) -> None:
    llm = _AlwaysBadCodeLLM()
    agent = CodeAgent(
        llm=cast(Any, llm),
        backend=PortableSubprocessBackend(work_root=tmp_path),
        limits=SandboxLimits(timeout_seconds=2),
        max_repairs=99,
    )

    result = agent.run(task="Never succeeds.", evidence_manifest={"datasets": []})

    assert result.status == "failed"
    assert len(result.attempts) == 3
    assert len(llm.calls) == 3


def test_code_agent_budget_exhaustion_returns_failed_result(tmp_path: Path) -> None:
    llm = _AlwaysBadCodeLLM()
    agent = CodeAgent(
        llm=cast(Any, llm),
        backend=PortableSubprocessBackend(work_root=tmp_path),
        limits=SandboxLimits(timeout_seconds=2),
    )

    result = agent.run(
        task="Budget should fail before the model call.",
        evidence_manifest={"datasets": []},
        budget=Budget(max_seconds=0),
    )

    assert result.status == "failed"
    assert result.error_category == "budget_exhausted"
    assert llm.calls == []


def test_portable_backend_declares_it_is_not_safe_for_untrusted_code(tmp_path: Path) -> None:
    backend = PortableSubprocessBackend(work_root=tmp_path)

    info = backend.info

    assert info.name == "portable_subprocess"
    assert info.safe_for_untrusted_code is False
    assert info.available is True
    assert "development/test" in info.detail.lower() or "dev/test" in info.detail.lower()
    assert "not a security boundary" in info.detail.lower()


def test_sandbox_unavailable_error_is_a_runtime_error() -> None:
    assert issubclass(sandbox_module.SandboxUnavailableError, RuntimeError)


class _UnavailableDockerBackend:
    def __init__(self, *, work_root: Path | str, image: str | None = None) -> None:
        self.work_root = Path(work_root)
        self.image = image

    @property
    def info(self) -> SandboxBackendInfo:
        return SandboxBackendInfo(
            name="docker",
            safe_for_untrusted_code=True,
            available=False,
            detail="Docker unavailable for test",
        )


class _AvailableDockerBackend:
    def __init__(self, *, work_root: Path | str, image: str | None = None) -> None:
        self.work_root = Path(work_root)
        self.image = image

    @property
    def info(self) -> SandboxBackendInfo:
        return SandboxBackendInfo(
            name="docker",
            safe_for_untrusted_code=True,
            available=True,
            detail="Docker available for test",
        )


class _BuggyDockerBackend:
    def __init__(self, *, work_root: Path | str, image: str | None = None) -> None:
        raise ValueError("docker backend constructor bug")


def test_broker_env_parser_rejects_unknown_backend(monkeypatch) -> None:
    monkeypatch.setenv("EDA_SANDBOX_BACKEND", " definitely-not-real ")

    with pytest.raises(SandboxUnavailableError, match="EDA_SANDBOX_BACKEND"):
        SandboxBroker.from_env()


@pytest.mark.parametrize("removed_backend", ["seatbelt", "subprocess"])
def test_broker_env_parser_rejects_removed_backends(
    removed_backend: str, monkeypatch
) -> None:
    monkeypatch.setenv("EDA_SANDBOX_BACKEND", removed_backend)
    monkeypatch.setenv("EDA_ALLOW_UNSAFE_SUBPROCESS_SANDBOX", "1")

    with pytest.raises(SandboxUnavailableError, match="auto or docker"):
        SandboxBroker.from_env()


def test_broker_auto_has_no_fallback_when_docker_is_unavailable(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("EDA_ALLOW_UNSAFE_SUBPROCESS_SANDBOX", "1")
    monkeypatch.setattr(
        sandbox_broker_module, "DockerSandboxBackend", _UnavailableDockerBackend
    )
    settings = SandboxSettings(kind="auto", work_root=tmp_path)

    with pytest.raises(SandboxUnavailableError, match="Docker unavailable"):
        SandboxBroker(settings).resolve_backend()


def test_broker_explicit_docker_unavailable_raises(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        sandbox_broker_module, "DockerSandboxBackend", _UnavailableDockerBackend
    )
    settings = SandboxSettings(kind="docker", work_root=tmp_path)

    with pytest.raises(SandboxUnavailableError, match="Docker unavailable"):
        SandboxBroker(settings).resolve_backend()


def test_broker_propagates_settings_docker_image(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sandbox_broker_module, "DockerSandboxBackend", _AvailableDockerBackend)
    settings = SandboxSettings(kind="docker", work_root=tmp_path, docker_image="custom:image")

    backend = SandboxBroker(settings).resolve_backend()

    assert isinstance(backend, _AvailableDockerBackend)
    assert backend.image == "custom:image"


def test_broker_from_env_propagates_docker_image(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sandbox_broker_module, "DockerSandboxBackend", _AvailableDockerBackend)
    monkeypatch.setenv("EDA_SANDBOX_BACKEND", "docker")
    monkeypatch.setenv("EDA_SANDBOX_DOCKER_IMAGE", "env-custom:image")

    backend = SandboxBroker.from_env(work_root=tmp_path).resolve_backend()

    assert isinstance(backend, _AvailableDockerBackend)
    assert backend.image == "env-custom:image"


def test_broker_docker_constructor_bugs_surface(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sandbox_broker_module, "DockerSandboxBackend", _BuggyDockerBackend)
    settings = SandboxSettings(kind="docker", work_root=tmp_path)

    with pytest.raises(ValueError, match="constructor bug"):
        SandboxBroker(settings).resolve_backend()


def test_chat_default_code_backend_requires_broker_safe_backend(monkeypatch) -> None:
    backend = _FakeSafeBackend()
    captured_work_roots: list[Path | None] = []

    class FakeBroker:
        def require_safe_backend(self) -> _FakeSafeBackend:
            return backend

    def fake_from_env(*, work_root: Path | None = None) -> FakeBroker:
        captured_work_roots.append(work_root)
        return FakeBroker()

    monkeypatch.setattr(chat_driver.SandboxBroker, "from_env", staticmethod(fake_from_env))

    resolved = chat_driver._default_code_backend(None, "project_1", "run_1")

    assert resolved is backend
    assert captured_work_roots
    assert captured_work_roots[0] is not None
    assert captured_work_roots[0].name.startswith("eda_code_agent_")


def test_chat_open_analysis_returns_error_when_safe_sandbox_unavailable(
    monkeypatch,
) -> None:
    def fake_from_env(*, work_root: Path | None = None) -> SandboxBroker:
        raise SandboxUnavailableError("No safe sandbox backend is available.")

    def fail_if_code_agent_is_created(*args: object, **kwargs: object) -> None:
        raise AssertionError("CodeAgent should not be constructed without a safe backend")

    monkeypatch.setattr(chat_driver.SandboxBroker, "from_env", staticmethod(fake_from_env))
    monkeypatch.setattr(chat_driver, "CodeAgent", fail_if_code_agent_is_created)

    result = chat_driver._execute_open_analysis(
        "run open analysis",
        intent=Intent(
            kind="open_analysis",
            confidence=1.0,
            raw_message="run open analysis",
        ),
        datasets=[],
        parent_artifacts=[],
        project_id="project_1",
        session_id="run_1",
        llm=cast(Any, object()),
        store=None,
        backend=None,
        limits=None,
        budget=None,
        timeout_seconds=2.0,
    )

    assert result.status == "error"
    assert result.message.startswith("Open-ended Python analysis is disabled")
    assert "no safe sandbox backend is available" in result.message
    assert ".." not in result.message


def test_chat_open_analysis_rejects_explicit_unsafe_backend(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fail_if_code_agent_is_created(*args: object, **kwargs: object) -> None:
        raise AssertionError("CodeAgent should not be constructed with an unsafe backend")

    monkeypatch.setattr(chat_driver, "CodeAgent", fail_if_code_agent_is_created)

    result = chat_driver._execute_open_analysis(
        "run open analysis",
        intent=Intent(
            kind="open_analysis",
            confidence=1.0,
            raw_message="run open analysis",
        ),
        datasets=[],
        parent_artifacts=[],
        project_id="project_1",
        session_id="run_1",
        llm=cast(Any, object()),
        store=None,
        backend=PortableSubprocessBackend(work_root=tmp_path),
        limits=None,
        budget=None,
        timeout_seconds=2.0,
    )

    assert result.status == "error"
    assert "no safe sandbox backend is available" in result.message
    assert "unsafe" in result.message.lower() or "not safe" in result.message.lower()


def test_chat_open_analysis_traces_backend_resolution_failure(monkeypatch) -> None:
    captured: list[dict[str, Any]] = []

    def fake_from_env(*, work_root: Path | None = None) -> SandboxBroker:
        raise SandboxUnavailableError("No safe sandbox backend is available.")

    def capture_trace(
        store: object,
        project_id: str,
        session_id: str,
        **kwargs: Any,
    ) -> None:
        captured.append(
            {
                "store": store,
                "project_id": project_id,
                "session_id": session_id,
                **kwargs,
            }
        )

    monkeypatch.setattr(chat_driver.SandboxBroker, "from_env", staticmethod(fake_from_env))
    monkeypatch.setattr(chat_driver, "_append_trace", capture_trace)

    result = chat_driver._execute_open_analysis(
        "run open analysis",
        intent=Intent(
            kind="open_analysis",
            confidence=1.0,
            raw_message="run open analysis",
        ),
        datasets=[],
        parent_artifacts=[],
        project_id="project_1",
        session_id="run_1",
        llm=cast(Any, object()),
        store=None,
        backend=None,
        limits=None,
        budget=None,
        timeout_seconds=2.0,
    )

    assert result.status == "error"
    assert len(captured) == 1
    trace = captured[0]
    assert trace["event_type"] == "chat_turn_failed"
    assert trace["name"] == "m6_open_analysis_code_agent"
    assert trace["summary"]["stage"] == "backend_resolution"
    assert trace["summary"]["error_type"] == "SandboxUnavailableError"
    assert "No safe sandbox backend" in trace["summary"]["error_message"]
