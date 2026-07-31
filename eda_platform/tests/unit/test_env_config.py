from __future__ import annotations

from eda_platform.core import env
from eda_platform.core.env import (
    load_llm_settings_from_env_file,
    load_provider_api_keys_from_env_file,
    parse_env_file,
)
from eda_platform.core.llm import LLMProvider
from eda_platform.core.model_capabilities import (
    selectable_model_ids,
)


def test_default_env_file_is_repo_anchored_not_cwd_relative(
    monkeypatch, tmp_path
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("EDA_LLM_PROVIDER=deepseek\n", encoding="utf-8")
    monkeypatch.setattr(env, "DEFAULT_ENV_PATH", env_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    settings = load_llm_settings_from_env_file(environ={})

    assert settings.provider is LLMProvider.DEEPSEEK


def test_parse_env_file_handles_comments_quotes_and_export_prefix(tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "# local secrets\n"
        "export EDA_LLM_PROVIDER=openai\n"
        'EDA_LLM_MODEL="gpt-4.1-mini"\n'
        "EDA_LLM_API_KEY='sk-local'\n"
        "IGNORED_LINE\n",
        encoding="utf-8",
    )

    values = parse_env_file(env_path)

    assert values["EDA_LLM_PROVIDER"] == "openai"
    assert values["EDA_LLM_MODEL"] == "gpt-4.1-mini"
    assert values["EDA_LLM_API_KEY"] == "sk-local"
    assert "IGNORED_LINE" not in values


def test_load_llm_settings_reads_dotenv_with_provider_specific_key(tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "EDA_LLM_PROVIDER=deepseek\n"
        "EDA_LLM_MODEL=deepseek-v4-flash\n"
        "DEEPSEEK_API_KEY=sk-deepseek\n"
        "EDA_LLM_TEMPERATURE=0.1\n"
        "EDA_LLM_MAX_TOKENS=4096\n"
        "EDA_LLM_TIMEOUT_SECONDS=180\n",
        encoding="utf-8",
    )

    settings = load_llm_settings_from_env_file(env_path, environ={})

    assert settings.provider is LLMProvider.DEEPSEEK
    assert settings.api_key == "sk-deepseek"
    assert settings.model == "deepseek-v4-flash"
    assert settings.temperature == 0.1
    assert settings.max_tokens == 4096
    assert settings.timeout_seconds == 180


def test_load_llm_settings_defaults_deepseek_to_current_v4_flash(tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "EDA_LLM_PROVIDER=deepseek\n"
        "DEEPSEEK_API_KEY=sk-deepseek\n",
        encoding="utf-8",
    )

    settings = load_llm_settings_from_env_file(env_path, environ={})

    assert settings.model == "deepseek-v4-flash"
    assert settings.max_tokens == 6000


def test_load_llm_settings_prefers_process_env_over_dotenv(tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "EDA_LLM_PROVIDER=openai\n"
        "EDA_LLM_API_KEY=sk-file\n"
        "EDA_LLM_MODEL=gpt-file\n",
        encoding="utf-8",
    )

    settings = load_llm_settings_from_env_file(
        env_path,
        environ={
            "EDA_LLM_API_KEY": "sk-process",
            "EDA_LLM_MODEL": "gpt-process",
        },
    )

    assert settings.api_key == "sk-process"
    assert settings.model == "gpt-process"


def test_provider_keys_can_coexist_and_are_loaded_independently(tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "OPENAI_API_KEY=sk-openai\n"
        "DEEPSEEK_API_KEY=sk-deepseek\n"
        "ANTHROPIC_API_KEY=sk-anthropic\n",
        encoding="utf-8",
    )

    keys = load_provider_api_keys_from_env_file(env_path, environ={})

    assert keys == {
        LLMProvider.OPENAI: "sk-openai",
        LLMProvider.ANTHROPIC: "sk-anthropic",
        LLMProvider.DEEPSEEK: "sk-deepseek",
    }


def test_compatible_key_is_not_reused_for_another_provider(tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "EDA_LLM_PROVIDER=anthropic\n"
        "OPENAI_COMPATIBLE_API_KEY=sk-compatible\n",
        encoding="utf-8",
    )

    settings = load_llm_settings_from_env_file(env_path, environ={})

    assert settings.provider is LLMProvider.ANTHROPIC
    assert settings.api_key == ""


def test_env_default_model_stays_inside_each_provider_catalog(tmp_path) -> None:
    """The .env path skips the settings API, so its default has to be one this
    provider can actually run. It used to hand every provider except DeepSeek and
    offline a hardcoded "gpt-4.1-mini", which sent an OpenAI model id to
    api.anthropic.com and only failed at the first tool call inside a job.

    Every provider is covered, not only the ones with a verified catalog: the
    providers without one are exactly the local deployments that regressed."""
    for provider in LLMProvider:
        env_file = tmp_path / f"{provider.value}.env"
        env_file.write_text(f"EDA_LLM_PROVIDER={provider.value}\n", encoding="utf-8")

        settings = load_llm_settings_from_env_file(env_file, environ={})

        assert settings.provider is provider
        known = selectable_model_ids(provider)
        assert not known or settings.model in known, (
            f"{provider.value} defaults to {settings.model!r}, "
            "which is not a model it serves"
        )
