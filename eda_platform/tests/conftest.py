"""Test-wide defaults.

Several UI tests build their LLM client the same way the app does — from the
developer's `.env`. On a machine with a real key that turned unit tests into
live, paid, network-bound sessions. Tests declare their own client when they need one, so the
default here is offline.
"""

from __future__ import annotations

import pytest

_LLM_ENV_KEYS = (
    "EDA_LLM_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "DASHSCOPE_API_KEY",
    "QWEN_API_KEY",
    "MOONSHOT_API_KEY",
    "ZHIPU_API_KEY",
    "XAI_API_KEY",
    "MISTRAL_API_KEY",
    "OPENROUTER_API_KEY",
    "TOGETHER_API_KEY",
    "GROQ_API_KEY",
    "FIREWORKS_API_KEY",
    "OLLAMA_API_KEY",
    "LM_STUDIO_API_KEY",
    "OPENAI_COMPATIBLE_API_KEY",
)


@pytest.fixture(autouse=True)
def offline_llm_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Force the offline provider unless a test opts in explicitly."""
    monkeypatch.setenv("EDA_LLM_PROVIDER", "offline")
    for key in _LLM_ENV_KEYS:
        monkeypatch.setenv(key, "")
    # A test that forgets to pass a workspace must never write into the real
    # repository workspace. Point the default at a throwaway dir instead;
    # tests that exercise fallback resolution explicitly delete this override.
    monkeypatch.setenv(
        "EDA_WORKSPACE", str(tmp_path_factory.mktemp("guard-workspace"))
    )
