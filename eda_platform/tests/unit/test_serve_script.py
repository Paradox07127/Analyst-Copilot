"""scripts/serve.py config resolution: args, dist check, workspace guard."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

_SERVE_PATH = Path(__file__).resolve().parents[3] / "scripts" / "serve.py"


@pytest.fixture(scope="module")
def serve() -> Iterator[ModuleType]:
    spec = importlib.util.spec_from_file_location("serve_script", _SERVE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Must be registered before exec: dataclass field resolution looks the
    # module up in sys.modules.
    sys.modules["serve_script"] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop("serve_script", None)


@pytest.fixture
def dist(tmp_path: Path) -> Path:
    root = tmp_path / "dist"
    root.mkdir()
    (root / "index.html").write_text("<html></html>")
    return root


@pytest.fixture(autouse=True)
def isolated_env(serve: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EDA_WORKSPACE", raising=False)
    # Point the repo anchor at tmp_path so the developer's real .env never leaks in.
    monkeypatch.setattr(serve, "REPO_ROOT", tmp_path)


def test_defaults(serve: ModuleType, dist: Path, tmp_path: Path) -> None:
    config = serve.resolve_config(["--dist", str(dist)])
    assert config.host == "127.0.0.1"
    assert config.port == 8000
    assert config.dist == dist
    assert config.workspace == tmp_path / "eda_platform" / "workspace"


def test_host_port_overrides(serve: ModuleType, dist: Path) -> None:
    config = serve.resolve_config(["--dist", str(dist), "--host", "0.0.0.0", "--port", "8321"])
    assert config.host == "0.0.0.0"
    assert config.port == 8321


def test_missing_dist_mentions_npm_build(serve: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(serve.ServeConfigError, match="npm run build"):
        serve.resolve_config(["--dist", str(tmp_path / "nowhere")])


def test_cli_workspace_absolute_used(
    serve: ModuleType, dist: Path, tmp_path: Path
) -> None:
    workspace = tmp_path / "explicit-ws"
    config = serve.resolve_config(["--dist", str(dist), "--workspace", str(workspace)])
    assert config.workspace == workspace


def test_cli_workspace_relative_rejected(serve: ModuleType, dist: Path) -> None:
    with pytest.raises(serve.ServeConfigError, match="absolute"):
        serve.resolve_config(["--dist", str(dist), "--workspace", "relative/ws"])


def test_env_workspace_absolute_used(
    serve: ModuleType, dist: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("EDA_WORKSPACE", str(tmp_path / "env-ws"))
    config = serve.resolve_config(["--dist", str(dist)])
    assert config.workspace == tmp_path / "env-ws"


def test_env_workspace_relative_rejected(
    serve: ModuleType, dist: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EDA_WORKSPACE", "relative/ws")
    with pytest.raises(serve.ServeConfigError, match="absolute"):
        serve.resolve_config(["--dist", str(dist)])


def test_dotenv_workspace_read_from_repo_root(
    serve: ModuleType,
    dist: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / ".env").write_text(f"EDA_WORKSPACE={tmp_path / 'dotenv-ws'}\n")
    elsewhere = tmp_path / "nested"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    config = serve.resolve_config(["--dist", str(dist)])
    assert config.workspace == tmp_path / "dotenv-ws"


def test_main_returns_1_on_missing_dist(
    serve: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert serve.main(["--dist", str(tmp_path / "nowhere")]) == 1
    assert "npm run build" in capsys.readouterr().out
