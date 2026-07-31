from __future__ import annotations

from pathlib import Path

import pytest

from eda_platform.core import config
from eda_platform.core.config import WorkspaceConfigError, default_workspace
from eda_platform.drivers.auto_eda import run_auto_eda


def test_default_workspace_without_env_is_repo_anchored_and_absolute(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv("EDA_WORKSPACE", raising=False)
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    elsewhere = tmp_path / "nested" / "cwd"
    elsewhere.mkdir(parents=True)
    monkeypatch.chdir(elsewhere)

    result = default_workspace()

    assert result.is_absolute()
    assert result == tmp_path / "eda_platform" / "workspace"


def test_default_workspace_reads_repo_env_file_from_any_cwd(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("EDA_WORKSPACE", raising=False)
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    (tmp_path / ".env").write_text(f"EDA_WORKSPACE={tmp_path / 'from-dotenv'}\n")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    assert default_workspace() == (tmp_path / "from-dotenv").resolve()


def test_default_workspace_env_override_is_absolutized(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("EDA_WORKSPACE", str(tmp_path / "ws"))
    result = default_workspace()
    assert result.is_absolute()
    assert result == (tmp_path / "ws").resolve()


def test_default_workspace_rejects_relative_env_override(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("EDA_WORKSPACE", "relative/workspace")

    with pytest.raises(WorkspaceConfigError, match="absolute"):
        default_workspace()


def test_run_auto_eda_honors_env_workspace(monkeypatch, tmp_path) -> None:
    """Guard the call sites, not just the helper: a run started without an
    explicit workspace must land inside EDA_WORKSPACE (mutation-tested gap)."""
    workspace = tmp_path / "env-ws"
    monkeypatch.setenv("EDA_WORKSPACE", str(workspace))
    monkeypatch.chdir(tmp_path)
    csv = tmp_path / "tiny.csv"
    csv.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    result = run_auto_eda([csv], project_id="wsguard", generate_report=False)

    assert Path(result.workspace).resolve() == workspace.resolve()
    assert (workspace / "projects" / "wsguard" / "sessions").is_dir()
