"""Curated model capabilities for the Agent runtime.

This catalog records how a model is known to behave, not who is allowed to run.
It used to be a whitelist, and a whitelist cannot express the case it most
needed to: a self-hosted model whose id is whatever the operator named it.
Twelve of eighteen providers were unreachable as a result, including all three
local ones.

So an entry here means "verified, and here is the dialect that works"; absence
means "unverified, try it and find out" — see ``tool_calling_probe`` for how
that gets decided before a run spends anything.
"""

from __future__ import annotations

from dataclasses import dataclass

from eda_platform.core.provider_registry import LLMProvider, provider_spec

CAPABILITY_CATALOG_VERSION = "agent-models-2026-08-03"


@dataclass(frozen=True)
class AgentModelProfile:
    model_id: str
    tool_calling: bool = True
    parallel_tool_calling: bool = False
    structured_output: str = "json_object"
    temperature_policy: str = "send"
    tool_choice_policy: str = "auto"
    reasoning_state_policy: str = "none"
    # Non-empty: value to send as `reasoning_effort` on tools requests. Empty
    # leaves the provider default alone, which is the only safe answer for
    # reasoning models (sol/terra) and for models that reject the parameter.
    tools_reasoning_effort: str = ""
    docs_url: str = ""
    checked_at: str = "2026-07-30"


def _profiles(
    model_ids: tuple[str, ...],
    *,
    parallel: bool,
    structured_output: str,
    temperature_policy: str = "send",
    tool_choice_policy: str = "auto",
    reasoning_state_policy: str = "none",
    tools_reasoning_effort: str = "",
    docs_url: str,
) -> tuple[AgentModelProfile, ...]:
    return tuple(
        AgentModelProfile(
            model_id=model_id,
            parallel_tool_calling=parallel,
            structured_output=structured_output,
            temperature_policy=temperature_policy,
            tool_choice_policy=tool_choice_policy,
            reasoning_state_policy=reasoning_state_policy,
            tools_reasoning_effort=tools_reasoning_effort,
            docs_url=docs_url,
        )
        for model_id in model_ids
    )


AGENT_MODEL_REGISTRY: dict[LLMProvider, tuple[AgentModelProfile, ...]] = {
    # The whole gpt-5.6 family 400s on function tools at its default reasoning
    # effort on /v1/chat/completions: "Function tools with reasoning_effort are
    # not supported ... set reasoning_effort to 'none'". Observed on luna
    # 2026-08-01 and terra 2026-08-03; sol is pinned by inference from the same
    # family/endpoint, not observation. Lifting it requires /v1/responses.
    LLMProvider.OPENAI: (
        *_profiles(
            ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"),
            parallel=True,
            structured_output="json_schema",
            tools_reasoning_effort="none",
            docs_url="https://developers.openai.com/api/docs/models",
        ),
        *_profiles(
            ("gpt-4.1-mini", "gpt-4.1"),
            parallel=True,
            structured_output="json_schema",
            docs_url="https://developers.openai.com/api/docs/models",
        ),
    ),
    LLMProvider.ANTHROPIC: (
        *_profiles(
            (
                "claude-opus-5",
                "claude-opus-4-8",
                "claude-sonnet-5",
                "claude-fable-5",
            ),
            parallel=True,
            structured_output="provider_tool",
            temperature_policy="omit",
            docs_url=(
                "https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview"
            ),
        ),
        *_profiles(
            ("claude-sonnet-4-6", "claude-haiku-4-5"),
            parallel=True,
            structured_output="provider_tool",
            docs_url=(
                "https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview"
            ),
        ),
    ),
    LLMProvider.DEEPSEEK: _profiles(
        ("deepseek-v4-flash", "deepseek-v4-pro"),
        parallel=True,
        structured_output="json_object",
        tool_choice_policy="omit",
        reasoning_state_policy="reasoning_content",
        docs_url="https://api-docs.deepseek.com/guides/tool_calls",
    ),
    LLMProvider.GROQ: _profiles(
        ("llama-3.3-70b-versatile", "openai/gpt-oss-120b"),
        parallel=True,
        structured_output="json_object",
        docs_url="https://console.groq.com/docs/tool-use/overview",
    ),
    LLMProvider.OPENROUTER: _profiles(
        (
            "anthropic/claude-sonnet-5",
            "openai/gpt-5.6-terra",
            "deepseek/deepseek-v4-pro",
        ),
        parallel=True,
        structured_output="json_schema",
        docs_url="https://openrouter.ai/docs/guides/features/tool-calling",
    ),
}


def agent_model_profiles(provider: LLMProvider) -> tuple[AgentModelProfile, ...]:
    return AGENT_MODEL_REGISTRY.get(provider, ())


def agent_model_ids(provider: LLMProvider) -> tuple[str, ...]:
    """Models this repo has verified for the tool loop. May be empty."""
    if provider is LLMProvider.OFFLINE:
        return ("offline-deterministic",)
    return tuple(
        profile.model_id
        for profile in agent_model_profiles(provider)
        if profile.tool_calling
    )


def selectable_model_ids(provider: LLMProvider) -> tuple[str, ...]:
    """What to offer, and what to default a fresh provider to.

    Verified models first; failing that the registry's own preset list, which
    is why Gemini, Qwen, Ollama and friends have something to start from
    instead of an empty model field. Neither list is a limit — an id typed by
    hand is equally valid, because a self-hosted model can be called anything.
    """
    verified = agent_model_ids(provider)
    if verified:
        return verified
    return tuple(provider_spec(provider).preset_models)


def agent_model_profile(
    provider: LLMProvider,
    model: str,
) -> AgentModelProfile | None:
    normalized = model.strip()
    for profile in agent_model_profiles(provider):
        if normalized == profile.model_id:
            return profile
        prefix = f"{profile.model_id}-"
        if normalized.startswith(prefix):
            suffix = normalized[len(prefix) :]
            if suffix.replace("-", "").isdigit():
                return profile
    return None


def is_verified_agent_model(provider: LLMProvider, model: str) -> bool:
    """Whether this repo has checked the model, not whether it may be used.

    Was ``is_agent_capable_model`` and was wired as a gate in four places. The
    rename is the point: "unverified" is a statement about this catalog, and
    reading it as "incapable" is what locked out every local deployment.
    """
    if provider is LLMProvider.OFFLINE:
        return model.strip() == "offline-deterministic"
    profile = agent_model_profile(provider, model)
    return profile is not None and profile.tool_calling
