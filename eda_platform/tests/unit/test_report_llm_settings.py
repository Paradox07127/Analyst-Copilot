"""The report may be written by a different model than the workflow uses.

Question discovery runs on a cheap fast model many times per run; the narrative
runs once and is the part a reader actually judges the product by. The override
is narrow on purpose -- an unset variable means "same model as everything else",
never a silent switch.
"""

from __future__ import annotations

from eda_platform.core.env import (
    load_llm_settings_from_env_file,
    load_report_llm_settings_from_env_file,
)
from eda_platform.core.llm import LLMProvider


def _env(**overrides: str) -> dict[str, str]:
    base = {
        "EDA_LLM_PROVIDER": "deepseek",
        "EDA_LLM_MODEL": "deepseek-v4-flash",
        "DEEPSEEK_API_KEY": "dk-workflow",
        "EDA_LLM_TEMPERATURE": "0.2",
    }
    base.update(overrides)
    return base


def test_without_an_override_the_report_uses_the_workflow_model() -> None:
    environ = _env()
    workflow = load_llm_settings_from_env_file(path=None, environ=environ)
    report = load_report_llm_settings_from_env_file(path=None, environ=environ)

    assert report.model == workflow.model == "deepseek-v4-flash"
    assert report.provider == workflow.provider


def test_the_model_can_be_swapped_without_touching_the_provider() -> None:
    environ = _env(EDA_REPORT_LLM_MODEL="deepseek-v4")
    report = load_report_llm_settings_from_env_file(path=None, environ=environ)

    assert report.model == "deepseek-v4"
    assert report.provider is LLMProvider.DEEPSEEK
    # Same provider means the same credential; nothing else shifts.
    assert report.api_key == "dk-workflow"
    assert load_llm_settings_from_env_file(
        path=None, environ=environ
    ).model == "deepseek-v4-flash"


def test_switching_provider_picks_up_that_provider_key() -> None:
    environ = _env(
        EDA_REPORT_LLM_PROVIDER="anthropic",
        EDA_REPORT_LLM_MODEL="claude-opus-5",
        ANTHROPIC_API_KEY="sk-ant-report",
    )
    report = load_report_llm_settings_from_env_file(path=None, environ=environ)

    assert report.provider is LLMProvider.ANTHROPIC
    assert report.model == "claude-opus-5"
    assert report.api_key == "sk-ant-report"


def test_a_provider_switch_without_a_model_uses_that_provider_default() -> None:
    # Carrying deepseek-v4-flash to api.anthropic.com fails at the first call.
    environ = _env(
        EDA_REPORT_LLM_PROVIDER="anthropic", ANTHROPIC_API_KEY="sk-ant-report"
    )
    report = load_report_llm_settings_from_env_file(path=None, environ=environ)

    assert report.provider is LLMProvider.ANTHROPIC
    assert report.model != "deepseek-v4-flash"


def test_the_narrative_may_run_hotter_than_the_workflow() -> None:
    environ = _env(EDA_REPORT_LLM_TEMPERATURE="0.6")
    report = load_report_llm_settings_from_env_file(path=None, environ=environ)

    assert report.temperature == 0.6
    assert load_llm_settings_from_env_file(path=None, environ=environ).temperature == 0.2
