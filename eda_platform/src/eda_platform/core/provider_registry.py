"""Single source of truth for LLM provider configuration.

Each provider is one `ProviderSpec`: credentials stay outside this registry,
while public endpoint, request-dialect, model and pricing metadata live here.
Prices are USD per 1M tokens and drift monthly — treat them as overridable
defaults (the UI lets a user override per-token cost), not guarantees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class LLMProvider(StrEnum):
    OFFLINE = "offline"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    AZURE_OPENAI = "azure_openai"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"
    MOONSHOT = "moonshot"
    ZHIPU = "zhipu"
    XAI = "xai"
    MISTRAL = "mistral"
    OPENROUTER = "openrouter"
    TOGETHER = "together"
    GROQ = "groq"
    FIREWORKS = "fireworks"
    OLLAMA = "ollama"
    LM_STUDIO = "lm_studio"
    OPENAI_COMPATIBLE = "openai_compatible"


# Auth styles. "bearer" => Authorization: Bearer <key>; "api_key_header" => Azure's
# `api-key` header; "anthropic" => x-api-key + anthropic-version (native client);
# "none" => local server, key optional (sent as bearer only if provided).
AuthStyle = str
# How to ask the provider what models it serves: "openai" => GET {base}/models,
# "anthropic" => GET {base}/v1/models, "none" => no live list.
ModelDiscovery = str


@dataclass(frozen=True)
class ProviderRequestProfile:
    """Provider-specific wire choices for the shared request adapter."""

    transport: str = "openai_chat_completions"
    output_token_param: str = "max_tokens"
    send_temperature: bool = True
    docs_url: str = ""
    checked_at: str = ""

# Stamped onto every cost derived from the table below, so a stored estimate
# stays interpretable after the table is revised.
PRICING_CATALOG_VERSION = "public-list-prices-2026-07-29"
# The day every number below was read off the provider's own pricing page.
# Surfaced in Settings so a stale table is visible rather than assumed fresh.
PRICING_CHECKED_AT = "2026-07-29"


@dataclass(frozen=True)
class ProviderSpec:
    display_name: str
    default_base_url: str
    auth_style: AuthStyle
    structured_mode: str  # resolved default when settings mode is "auto"
    preset_models: tuple[str, ...] = ()
    # model id -> (input, output) USD per 1M tokens. For DeepSeek the input value
    # is the cache-miss price (cache hits are metered/discounted separately).
    pricing_per_1m: dict[str, tuple[float, float]] = field(default_factory=dict)
    # Only where the provider *publishes* one. An unlisted model falls back to
    # the ordinary input rate rather than a guessed cross-provider discount.
    cache_read_per_1m: dict[str, float] = field(default_factory=dict)
    cache_write_per_1m: dict[str, float] = field(default_factory=dict)
    native: bool = False  # not the OpenAI /chat/completions wire format
    requires_api_key: bool = True
    requires_base_url: bool = False
    model_discovery: ModelDiscovery = "openai"
    pricing_source_url: str = ""
    request: ProviderRequestProfile = field(default_factory=ProviderRequestProfile)


_OPENAI_PRICES: dict[str, tuple[float, float]] = {
    "gpt-5.6-sol": (5.0, 30.0),
    "gpt-5.6-terra": (2.5, 15.0),
    "gpt-5.6-luna": (1.0, 6.0),
    "gpt-5.4-mini": (0.75, 4.5),
    "gpt-5.4-nano": (0.20, 1.25),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4o-mini": (0.15, 0.60),
}
# Verbatim from the "Cached Input" and "Cache Writes" columns, not a multiplier:
# gpt-4.1 reads at 0.25x its input rate while gpt-5.6-sol reads at 0.10x, so one
# ratio cannot cover the family.
_OPENAI_CACHE_READ: dict[str, float] = {
    "gpt-5.6-sol": 0.50,
    "gpt-5.6-terra": 0.25,
    "gpt-5.6-luna": 0.10,
    "gpt-5.4-mini": 0.075,
    "gpt-5.4-nano": 0.02,
    "gpt-4.1-mini": 0.10,
    "gpt-4.1": 0.50,
    "gpt-4o-mini": 0.075,
}
# Only the 5.6 family publishes a cache-write rate.
_OPENAI_CACHE_WRITE: dict[str, float] = {
    "gpt-5.6-sol": 6.25,
    "gpt-5.6-terra": 3.125,
    "gpt-5.6-luna": 1.25,
}

_ANTHROPIC_PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-fable-5": (10.0, 50.0),
}
# Anthropic publishes these as multipliers on the base input rate rather than a
# per-model table: "5-minute cache write 1.25x", "Cache read (hit) 0.1x".
_ANTHROPIC_CACHE_READ = {model: price[0] * 0.10 for model, price in _ANTHROPIC_PRICES.items()}
_ANTHROPIC_CACHE_WRITE = {model: price[0] * 1.25 for model, price in _ANTHROPIC_PRICES.items()}

_GEMINI_PRICES: dict[str, tuple[float, float]] = {
    "gemini-3.5-flash": (1.5, 9.0),
    "gemini-3.1-flash-lite": (0.25, 1.5),
    "gemini-2.5-flash": (0.30, 2.5),
    "gemini-2.5-flash-lite": (0.10, 0.40),
}
# The published "Context caching" column happens to equal 0.1x input for every
# listed model, so the ratio is derived rather than retyped.
_GEMINI_CACHE_READ = {model: price[0] * 0.10 for model, price in _GEMINI_PRICES.items()}

# Input is the cache-MISS price; the hit price is metered separately below.
# deepseek-chat / deepseek-reasoner are gone from the provider's own table.
_DEEPSEEK_PRICES: dict[str, tuple[float, float]] = {
    "deepseek-v4-flash": (0.14, 0.28),
    "deepseek-v4-pro": (0.435, 0.87),
}
_DEEPSEEK_CACHE_READ: dict[str, float] = {
    "deepseek-v4-flash": 0.0028,
    "deepseek-v4-pro": 0.003625,
}

# Below the 200k-prompt tier only. xAI doubles every rate at or above 200k
# input tokens, which this table does not model — a long-context call is
# therefore under-estimated by half. Tracked with the other uncovered
# dimensions in docs/infrastructure/llm-observability-and-billing-audit.
_XAI_PRICES: dict[str, tuple[float, float]] = {
    "grok-4.5": (2.0, 6.0),
    "grok-4.3": (1.25, 2.50),
}
_XAI_CACHE_READ: dict[str, float] = {"grok-4.5": 0.30, "grok-4.3": 0.20}

# Mistral publishes no cache rate, so there is no cache entry here: a guessed
# 0.1x would be a confident wrong number, and _estimate_cost already falls back
# to the ordinary input rate when a model is absent from the cache table.
_MISTRAL_PRICES: dict[str, tuple[float, float]] = {
    "mistral-large-latest": (0.50, 1.50),
    "mistral-medium-latest": (1.50, 7.50),
    "magistral-medium-latest": (2.00, 5.00),
    "ministral-8b-latest": (0.15, 0.15),
}

_GROQ_PRICES: dict[str, tuple[float, float]] = {
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.075, 0.30),
    "llama-3.1-8b-instant": (0.05, 0.08),
}


PROVIDER_REGISTRY: dict[LLMProvider, ProviderSpec] = {
    LLMProvider.OFFLINE: ProviderSpec(
        display_name="Offline (deterministic)",
        default_base_url="",
        auth_style="none",
        structured_mode="json_object",
        preset_models=("offline-deterministic",),
        requires_api_key=False,
        model_discovery="none",
        request=ProviderRequestProfile(transport="offline", send_temperature=False),
    ),
    LLMProvider.OPENAI: ProviderSpec(
        display_name="OpenAI",
        default_base_url="https://api.openai.com/v1",
        auth_style="bearer",
        structured_mode="json_schema",
        preset_models=(
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "gpt-5.6-luna",
            "gpt-5.4-mini",
            "gpt-5.4-nano",
            "gpt-4.1-mini",
            "gpt-4.1",
        ),
        pricing_per_1m=_OPENAI_PRICES,
        cache_read_per_1m=_OPENAI_CACHE_READ,
        cache_write_per_1m=_OPENAI_CACHE_WRITE,
        pricing_source_url="https://developers.openai.com/api/docs/pricing",
        request=ProviderRequestProfile(
            output_token_param="max_completion_tokens",
            docs_url=(
                "https://developers.openai.com/api/reference/resources/chat/"
                "subresources/completions/methods/create"
            ),
            checked_at="2026-07-30",
        ),
    ),
    LLMProvider.ANTHROPIC: ProviderSpec(
        display_name="Anthropic (Claude)",
        default_base_url="https://api.anthropic.com",
        auth_style="anthropic",
        structured_mode="json_schema",
        preset_models=(
            "claude-opus-4-8",
            "claude-sonnet-5",
            "claude-haiku-4-5",
            "claude-fable-5",
            "claude-sonnet-4-6",
        ),
        pricing_per_1m=_ANTHROPIC_PRICES,
        cache_read_per_1m=_ANTHROPIC_CACHE_READ,
        cache_write_per_1m=_ANTHROPIC_CACHE_WRITE,
        native=True,
        model_discovery="anthropic",
        pricing_source_url="https://platform.claude.com/docs/en/about-claude/pricing",
        request=ProviderRequestProfile(
            transport="anthropic_messages",
            output_token_param="max_tokens",
            send_temperature=True,
            docs_url="https://platform.claude.com/docs/en/api/messages",
            checked_at="2026-07-30",
        ),
    ),
    LLMProvider.GEMINI: ProviderSpec(
        display_name="Google Gemini",
        default_base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        auth_style="bearer",
        structured_mode="json_schema",
        preset_models=(
            "gemini-3.5-flash",
            "gemini-3.1-flash-lite",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
        ),
        pricing_per_1m=_GEMINI_PRICES,
        cache_read_per_1m=_GEMINI_CACHE_READ,
        pricing_source_url="https://ai.google.dev/gemini-api/docs/pricing",
    ),
    LLMProvider.AZURE_OPENAI: ProviderSpec(
        display_name="Azure OpenAI",
        default_base_url="",  # https://<resource>.openai.azure.com/openai/v1
        auth_style="api_key_header",
        structured_mode="json_schema",
        requires_base_url=True,
    ),
    LLMProvider.DEEPSEEK: ProviderSpec(
        display_name="DeepSeek",
        default_base_url="https://api.deepseek.com",
        auth_style="bearer",
        structured_mode="json_object",
        preset_models=("deepseek-v4-flash", "deepseek-v4-pro"),
        pricing_per_1m=_DEEPSEEK_PRICES,
        cache_read_per_1m=_DEEPSEEK_CACHE_READ,
        pricing_source_url="https://api-docs.deepseek.com/quick_start/pricing/",
        request=ProviderRequestProfile(
            output_token_param="max_tokens",
            docs_url="https://api-docs.deepseek.com/api/create-chat-completion/",
            checked_at="2026-07-30",
        ),
    ),
    LLMProvider.QWEN: ProviderSpec(
        display_name="Alibaba Qwen (DashScope)",
        default_base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        auth_style="bearer",
        structured_mode="json_object",
        preset_models=("qwen-max", "qwen-plus", "qwen-turbo"),
    ),
    LLMProvider.MOONSHOT: ProviderSpec(
        display_name="Moonshot (Kimi)",
        default_base_url="https://api.moonshot.ai/v1",
        auth_style="bearer",
        structured_mode="json_object",
        preset_models=("kimi-k3", "kimi-k2.7", "kimi-k2.6", "moonshot-v1-128k"),
    ),
    LLMProvider.ZHIPU: ProviderSpec(
        display_name="Zhipu (GLM)",
        default_base_url="https://api.z.ai/api/paas/v4",
        auth_style="bearer",
        structured_mode="json_object",
        preset_models=("glm-4.7", "glm-4.6", "glm-5", "glm-4.5-air"),
    ),
    LLMProvider.XAI: ProviderSpec(
        display_name="xAI (Grok)",
        default_base_url="https://api.x.ai/v1",
        auth_style="bearer",
        structured_mode="json_schema",
        preset_models=("grok-4.5", "grok-4.3"),
        pricing_per_1m=_XAI_PRICES,
        cache_read_per_1m=_XAI_CACHE_READ,
        pricing_source_url="https://docs.x.ai/developers/pricing",
    ),
    LLMProvider.MISTRAL: ProviderSpec(
        display_name="Mistral",
        default_base_url="https://api.mistral.ai/v1",
        auth_style="bearer",
        structured_mode="json_schema",
        preset_models=(
            "mistral-large-latest",
            "mistral-medium-latest",
            "magistral-medium-latest",
            "ministral-8b-latest",
        ),
        pricing_per_1m=_MISTRAL_PRICES,
        pricing_source_url="https://mistral.ai/pricing/api",
    ),
    LLMProvider.OPENROUTER: ProviderSpec(
        display_name="OpenRouter",
        default_base_url="https://openrouter.ai/api/v1",
        auth_style="bearer",
        structured_mode="json_schema",
        preset_models=(
            "anthropic/claude-sonnet-5",
            "openai/gpt-5.6-terra",
            "deepseek/deepseek-v4-pro",
            "x-ai/grok-4.5",
        ),
    ),
    LLMProvider.TOGETHER: ProviderSpec(
        display_name="Together AI",
        default_base_url="https://api.together.xyz/v1",
        auth_style="bearer",
        structured_mode="json_schema",
        preset_models=("meta-llama/Llama-3.3-70B-Instruct-Turbo", "deepseek-ai/DeepSeek-V3"),
    ),
    LLMProvider.GROQ: ProviderSpec(
        display_name="Groq",
        default_base_url="https://api.groq.com/openai/v1",
        auth_style="bearer",
        structured_mode="json_object",  # strict schema only on select models
        preset_models=("llama-3.3-70b-versatile", "openai/gpt-oss-120b"),
        pricing_per_1m=_GROQ_PRICES,
        pricing_source_url="https://console.groq.com/docs/models",
        request=ProviderRequestProfile(
            output_token_param="max_completion_tokens",
            docs_url="https://console.groq.com/docs/api-reference",
            checked_at="2026-07-30",
        ),
    ),
    LLMProvider.FIREWORKS: ProviderSpec(
        display_name="Fireworks AI",
        default_base_url="https://api.fireworks.ai/inference/v1",
        auth_style="bearer",
        structured_mode="json_schema",
        preset_models=("accounts/fireworks/models/deepseek-v3",),
    ),
    LLMProvider.OLLAMA: ProviderSpec(
        display_name="Ollama (local)",
        default_base_url="http://localhost:11434/v1",
        auth_style="none",
        structured_mode="json_object",
        preset_models=("llama3.1:8b", "qwen2.5:7b", "mistral"),
        requires_api_key=False,
    ),
    LLMProvider.LM_STUDIO: ProviderSpec(
        display_name="LM Studio (local)",
        default_base_url="http://localhost:1234/v1",
        auth_style="none",
        structured_mode="json_object",
        requires_api_key=False,
    ),
    LLMProvider.OPENAI_COMPATIBLE: ProviderSpec(
        display_name="OpenAI-compatible (custom / vLLM / llama.cpp)",
        default_base_url="",
        auth_style="none",  # key optional; local servers often need none
        structured_mode="json_object",
        requires_api_key=False,
        requires_base_url=True,
    ),
}


def provider_spec(provider: LLMProvider) -> ProviderSpec:
    return PROVIDER_REGISTRY[provider]


def provider_request_profile(provider: LLMProvider) -> ProviderRequestProfile:
    return provider_spec(provider).request


def default_base_url(provider: LLMProvider) -> str:
    return provider_spec(provider).default_base_url


def default_structured_mode(provider: LLMProvider) -> str:
    return provider_spec(provider).structured_mode


def auth_style(provider: LLMProvider) -> str:
    return provider_spec(provider).auth_style


def requires_api_key(provider: LLMProvider) -> bool:
    return provider_spec(provider).requires_api_key


def requires_base_url(provider: LLMProvider) -> bool:
    return provider_spec(provider).requires_base_url


def pricing_per_1m(provider: LLMProvider, model: str) -> tuple[float, float] | None:
    return provider_spec(provider).pricing_per_1m.get(model)


def cache_read_price_per_1m(provider: LLMProvider, model: str) -> float | None:
    return provider_spec(provider).cache_read_per_1m.get(model)


def cache_write_price_per_1m(provider: LLMProvider, model: str) -> float | None:
    return provider_spec(provider).cache_write_per_1m.get(model)
