"""A recovered job must not silently run offline when queued for a live LLM.

Codex pre-commit review (2026-08-13): startup recovery re-enqueues queued jobs
without their per-job env overlay, and the minimal worker environment strips
process-level LLM configuration. `load_llm_settings_from_env_file` then
defaults to offline, so a live job could complete deterministically while
looking successful — the platform's cardinal failure mode.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import eda_platform.core.env as env_module
from eda_platform.core.llm import OfflineLLMClient
from eda_platform.worker import runner


def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(env_module, "DEFAULT_ENV_PATH", tmp_path / "missing.env")
    monkeypatch.delenv("EDA_LLM_PROVIDER", raising=False)


def test_env_job_without_provider_config_fails_instead_of_offline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_env(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match="silently offline"):
        runner._build_llm({"llm": "env"})


def test_explicit_offline_provider_is_still_honored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_env(monkeypatch, tmp_path)
    monkeypatch.setenv("EDA_LLM_PROVIDER", "offline")
    client = runner._build_llm({"llm": "env"})
    assert isinstance(getattr(client, "inner", client), OfflineLLMClient)


def test_offline_param_never_consults_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_env(monkeypatch, tmp_path)
    runner._build_llm({"llm": "offline"})
