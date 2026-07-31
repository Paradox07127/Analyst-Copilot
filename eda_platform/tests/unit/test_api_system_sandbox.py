"""Sandbox status endpoint: label mapping, the fail-closed unavailable shape,
and the TTL cache that keeps a poll from re-probing Docker every request."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import eda_platform.api.main as api_main
import eda_platform.application.services.sandbox_status_service as status_module
from eda_platform.api.main import create_app
from eda_platform.application.services.sandbox_status_service import SandboxStatusService
from eda_platform.core.sandbox import SandboxBackendInfo, SandboxUnavailableError
from eda_platform.core.store import ArtifactStore


class _FakeBackend:
    def __init__(self, info: SandboxBackendInfo) -> None:
        self.info = info


class _FakeBroker:
    calls = 0

    def __init__(self, backend: object) -> None:
        self._backend = backend

    def require_safe_backend(self) -> object:
        type(self).calls += 1
        if isinstance(self._backend, Exception):
            raise self._backend
        return self._backend


def _install(monkeypatch: pytest.MonkeyPatch, backend: object) -> None:
    _FakeBroker.calls = 0

    def _from_env(*, work_root: Path | None = None) -> _FakeBroker:
        return _FakeBroker(backend)

    monkeypatch.setattr(
        status_module.SandboxBroker, "from_env", staticmethod(_from_env)
    )


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ArtifactStore(tmp_path).ensure_project("demo", name="Demo")
    return tmp_path


@pytest.fixture
def client(workspace: Path) -> TestClient:
    return TestClient(create_app(workspace))


def test_docker_backend_is_reported_as_usable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(
        monkeypatch,
        _FakeBackend(
            SandboxBackendInfo(
                name="docker",
                safe_for_untrusted_code=True,
                available=True,
                detail="Docker sandbox available",
            )
        ),
    )
    body = client.get("/api/v1/system/sandbox").json()
    assert body["backend"] == "docker"
    assert body["available"] is True
    assert body["safe_for_untrusted_code"] is True
    assert body["open_python_analysis_available"] is True
    assert "available" in body["message"]


def test_no_backend_is_reported_not_raised(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, SandboxUnavailableError("No safe sandbox backend is available."))
    response = client.get("/api/v1/system/sandbox")
    assert response.status_code == 200
    body = response.json()
    assert body["backend"] == "none"
    assert body["open_python_analysis_available"] is False
    assert "No safe sandbox backend" in body["detail"]


def test_unexpected_probe_failure_still_answers(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, OSError("docker socket exploded"))
    body = client.get("/api/v1/system/sandbox").json()
    assert body["backend"] == "none"
    assert body["open_python_analysis_available"] is False
    # A raw OS error text must not reach the client.
    assert "socket exploded" not in body["detail"]


def test_status_is_cached_so_polling_does_not_reprobe(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(
        monkeypatch,
        _FakeBackend(
            SandboxBackendInfo(name="docker", safe_for_untrusted_code=True, available=True)
        ),
    )
    for _ in range(5):
        assert client.get("/api/v1/system/sandbox").status_code == 200
    assert _FakeBroker.calls == 1


def test_status_never_leaks_the_workspace_path(
    client: TestClient, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§14: no response carries a server file path. The probe knows the
    workspace root, so it is the one path that must never surface."""
    _install(
        monkeypatch,
        SandboxUnavailableError(f"Docker sandbox unavailable under {workspace}"),
    )
    body = client.get("/api/v1/system/sandbox").text
    assert str(workspace) not in body
    assert "<path>" in body
    # The actionable part of the message survives the scrub.
    assert "Docker sandbox unavailable" in body


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("requires /var/run/docker.sock", "requires <path>"),
        ("under /tmp/ws/_sandbox now", "under <path> now"),
        ("Install/start Docker", "Install/start Docker"),
        ("Docker sandbox available", "Docker sandbox available"),
    ],
)
def test_scrub_paths(raw: str, expected: str) -> None:
    assert status_module.scrub_paths(raw) == expected


def test_probe_uses_stable_work_root(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The status service supplies one deterministic root instead of mkdtemp."""
    captured: list[Path | None] = []
    backend = _FakeBackend(
        SandboxBackendInfo(name="docker", safe_for_untrusted_code=True, available=True)
    )

    def _from_env(*, work_root: Path | None = None) -> _FakeBroker:
        captured.append(work_root)
        return _FakeBroker(backend)

    monkeypatch.setattr(
        status_module.SandboxBroker, "from_env", staticmethod(_from_env)
    )
    service = SandboxStatusService(workspace)
    service.get_status()
    assert captured == [workspace / "_sandbox"]
    assert not (workspace / "_sandbox").exists()


def test_required_startup_preflight_caches_verified_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _FakeBackend(
        SandboxBackendInfo(name="docker", safe_for_untrusted_code=True, available=True)
    )

    class _StartupBroker:
        def require_safe_backend(self) -> _FakeBackend:
            return backend

    monkeypatch.setenv("EDA_SANDBOX_REQUIRED", "1")
    monkeypatch.setattr(
        api_main.SandboxBroker,
        "from_env",
        staticmethod(lambda *, work_root=None: _StartupBroker()),
    )

    app = create_app(tmp_path)

    assert app.state.sandbox_backend is backend


def test_required_startup_preflight_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _StartupBroker:
        def require_safe_backend(self) -> object:
            raise SandboxUnavailableError("runtime preflight failed")

    monkeypatch.setenv("EDA_SANDBOX_REQUIRED", "true")
    monkeypatch.setattr(
        api_main.SandboxBroker,
        "from_env",
        staticmethod(lambda *, work_root=None: _StartupBroker()),
    )

    with pytest.raises(SandboxUnavailableError, match="preflight failed"):
        create_app(tmp_path)
