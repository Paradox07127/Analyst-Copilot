from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from eda_platform.core.llm import LLMProvider, LLMSettings
from eda_platform.core.model_capabilities import selectable_model_ids

DEFAULT_ENV_PATH = Path(__file__).resolve().parents[4] / ".env"

PROVIDER_API_KEY_ENV_VARS: dict[LLMProvider, tuple[str, ...]] = {
    LLMProvider.OPENAI: ("OPENAI_API_KEY",),
    LLMProvider.ANTHROPIC: ("ANTHROPIC_API_KEY",),
    LLMProvider.GEMINI: ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    LLMProvider.AZURE_OPENAI: ("AZURE_OPENAI_API_KEY",),
    LLMProvider.DEEPSEEK: ("DEEPSEEK_API_KEY",),
    LLMProvider.QWEN: ("DASHSCOPE_API_KEY", "QWEN_API_KEY"),
    LLMProvider.MOONSHOT: ("MOONSHOT_API_KEY",),
    LLMProvider.ZHIPU: ("ZHIPU_API_KEY",),
    LLMProvider.XAI: ("XAI_API_KEY",),
    LLMProvider.MISTRAL: ("MISTRAL_API_KEY",),
    LLMProvider.OPENROUTER: ("OPENROUTER_API_KEY",),
    LLMProvider.TOGETHER: ("TOGETHER_API_KEY",),
    LLMProvider.GROQ: ("GROQ_API_KEY",),
    LLMProvider.FIREWORKS: ("FIREWORKS_API_KEY",),
    LLMProvider.OLLAMA: ("OLLAMA_API_KEY",),
    LLMProvider.LM_STUDIO: ("LM_STUDIO_API_KEY",),
    LLMProvider.OPENAI_COMPATIBLE: ("OPENAI_COMPATIBLE_API_KEY",),
}
API_KEY_ENV_VARS = tuple(
    dict.fromkeys(
        (
            "EDA_LLM_API_KEY",
            *(
                name
                for provider_names in PROVIDER_API_KEY_ENV_VARS.values()
                for name in provider_names
            ),
        )
    )
)


def parse_env_file(path: Path | str) -> dict[str, str]:
    env_path = Path(path)
    if not env_path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        key, raw_value = line.split("=", maxsplit=1)
        key = key.strip()
        if not key:
            continue
        values[key] = _clean_env_value(raw_value.strip())
    return values


def load_llm_settings_from_env_file(
    path: Path | str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> LLMSettings:
    merged = parse_env_file(DEFAULT_ENV_PATH if path is None else path)
    merged.update(dict(os.environ if environ is None else environ))
    provider = _provider(merged.get("EDA_LLM_PROVIDER"))
    api_key = _first_non_empty(
        merged.get("EDA_LLM_API_KEY"),
        _provider_api_key(provider, merged),
    )
    return LLMSettings(
        provider=provider,
        api_key=api_key,
        base_url=merged.get("EDA_LLM_BASE_URL", ""),
        model=merged.get("EDA_LLM_MODEL", _default_model(provider)),
        temperature=_float_value(merged.get("EDA_LLM_TEMPERATURE"), 0.2),
        max_tokens=_int_value(merged.get("EDA_LLM_MAX_TOKENS"), 6000),
        timeout_seconds=_float_value(merged.get("EDA_LLM_TIMEOUT_SECONDS"), 180.0),
        organization=merged.get("EDA_LLM_ORGANIZATION", ""),
        structured_output_mode=merged.get("EDA_LLM_STRUCTURED_OUTPUT_MODE", "auto"),
        usd_per_1k_prompt=_float_value(merged.get("EDA_LLM_USD_PER_1K_PROMPT"), 0.0),
        usd_per_1k_completion=_float_value(merged.get("EDA_LLM_USD_PER_1K_COMPLETION"), 0.0),
    )


def load_provider_api_keys_from_env_file(
    path: Path | str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[LLMProvider, str]:
    merged = parse_env_file(DEFAULT_ENV_PATH if path is None else path)
    merged.update(dict(os.environ if environ is None else environ))
    return {
        provider: api_key
        for provider in LLMProvider
        if (api_key := _provider_api_key(provider, merged))
    }


def _clean_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _provider(value: str | None) -> LLMProvider:
    if not value:
        return LLMProvider.OFFLINE
    try:
        return LLMProvider(value)
    except ValueError:
        return LLMProvider.OFFLINE


def _provider_api_key(provider: LLMProvider, values: Mapping[str, str]) -> str:
    return _first_non_empty(
        *(values.get(name) for name in PROVIDER_API_KEY_ENV_VARS.get(provider, ()))
    )


def _default_model(provider: LLMProvider) -> str:
    """First model this provider actually serves.

    This used to return a hardcoded "gpt-4.1-mini" for every provider except
    DeepSeek and offline, so `EDA_LLM_PROVIDER=anthropic` with no model sent an
    OpenAI model id to api.anthropic.com, and the failure only surfaced at the
    first tool call inside a running job.
    """
    models = selectable_model_ids(provider)
    return models[0] if models else ""


def _first_non_empty(*values: str | None) -> str:
    for value in values:
        if value:
            return value
    return ""


def _float_value(value: str | None, default: float) -> float:
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _int_value(value: str | None, default: int) -> int:
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default
