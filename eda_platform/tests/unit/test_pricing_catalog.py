"""Prices checked against each provider's own pricing page on 2026-07-29.

These are list prices for a rough estimate, never an invoice. The point of
pinning them is that a silent edit to a number here changes every cost figure
the product shows, and nothing else would catch it.
"""

from __future__ import annotations

import pytest

from eda_platform.core.llm import LLMSettings, LLMUsage, _estimate_cost
from eda_platform.core.provider_registry import (
    LLMProvider,
    cache_read_price_per_1m,
    cache_write_price_per_1m,
    pricing_per_1m,
    provider_spec,
)

# (provider, model, input_per_1m, output_per_1m) — first-party pricing pages.
PUBLISHED = [
    (LLMProvider.OPENAI, "gpt-5.6-sol", 5.0, 30.0),
    (LLMProvider.OPENAI, "gpt-5.6-terra", 2.5, 15.0),
    (LLMProvider.OPENAI, "gpt-5.6-luna", 1.0, 6.0),
    (LLMProvider.OPENAI, "gpt-5.4-mini", 0.75, 4.5),
    (LLMProvider.OPENAI, "gpt-5.4-nano", 0.20, 1.25),
    (LLMProvider.OPENAI, "gpt-4.1", 2.0, 8.0),
    (LLMProvider.OPENAI, "gpt-4.1-mini", 0.40, 1.60),
    (LLMProvider.OPENAI, "gpt-4o-mini", 0.15, 0.60),
    (LLMProvider.ANTHROPIC, "claude-opus-5", 5.0, 25.0),
    (LLMProvider.ANTHROPIC, "claude-opus-4-8", 5.0, 25.0),
    (LLMProvider.ANTHROPIC, "claude-sonnet-4-6", 3.0, 15.0),
    (LLMProvider.ANTHROPIC, "claude-haiku-4-5", 1.0, 5.0),
    (LLMProvider.ANTHROPIC, "claude-fable-5", 10.0, 50.0),
    (LLMProvider.GEMINI, "gemini-3.5-flash", 1.5, 9.0),
    (LLMProvider.GEMINI, "gemini-3.1-flash-lite", 0.25, 1.5),
    (LLMProvider.GEMINI, "gemini-2.5-flash", 0.30, 2.5),
    (LLMProvider.GEMINI, "gemini-2.5-flash-lite", 0.10, 0.40),
    (LLMProvider.DEEPSEEK, "deepseek-v4-flash", 0.14, 0.28),
    (LLMProvider.DEEPSEEK, "deepseek-v4-pro", 0.435, 0.87),
    (LLMProvider.XAI, "grok-4.5", 2.0, 6.0),
    (LLMProvider.XAI, "grok-4.3", 1.25, 2.50),
    (LLMProvider.MISTRAL, "mistral-large-latest", 0.50, 1.50),
    (LLMProvider.MISTRAL, "mistral-medium-latest", 1.50, 7.50),
    (LLMProvider.MISTRAL, "magistral-medium-latest", 2.00, 5.00),
    (LLMProvider.MISTRAL, "ministral-8b-latest", 0.15, 0.15),
    (LLMProvider.GROQ, "llama-3.3-70b-versatile", 0.59, 0.79),
    (LLMProvider.GROQ, "openai/gpt-oss-120b", 0.15, 0.60),
    (LLMProvider.GROQ, "openai/gpt-oss-20b", 0.075, 0.30),
    (LLMProvider.GROQ, "llama-3.1-8b-instant", 0.05, 0.08),
]

# OpenAI publishes a per-model cached-input column; the ratio is not constant
# across the family, so it cannot be derived.
PUBLISHED_CACHE_READ = [
    (LLMProvider.OPENAI, "gpt-5.6-sol", 0.50),
    (LLMProvider.OPENAI, "gpt-5.6-terra", 0.25),
    (LLMProvider.OPENAI, "gpt-4.1", 0.50),  # 0.25x input, not 0.1x
    (LLMProvider.OPENAI, "gpt-4o-mini", 0.075),  # 0.5x input
    (LLMProvider.ANTHROPIC, "claude-opus-5", 0.50),
    (LLMProvider.GEMINI, "gemini-2.5-flash", 0.03),
    (LLMProvider.DEEPSEEK, "deepseek-v4-flash", 0.0028),
    (LLMProvider.DEEPSEEK, "deepseek-v4-pro", 0.003625),
    (LLMProvider.XAI, "grok-4.5", 0.30),
]

PUBLISHED_CACHE_WRITE = [
    (LLMProvider.OPENAI, "gpt-5.6-sol", 6.25),
    (LLMProvider.OPENAI, "gpt-5.6-terra", 3.125),
    (LLMProvider.ANTHROPIC, "claude-opus-5", 6.25),  # 1.25x base input
    (LLMProvider.ANTHROPIC, "claude-fable-5", 12.50),
]


@pytest.mark.parametrize(("provider", "model", "input_price", "output_price"), PUBLISHED)
def test_listed_price_matches_the_provider_page(
    provider: LLMProvider, model: str, input_price: float, output_price: float
) -> None:
    assert pricing_per_1m(provider, model) == (input_price, output_price)


@pytest.mark.parametrize(("provider", "model", "price"), PUBLISHED_CACHE_READ)
def test_cache_read_price_matches_the_provider_page(
    provider: LLMProvider, model: str, price: float
) -> None:
    assert cache_read_price_per_1m(provider, model) == pytest.approx(price)


@pytest.mark.parametrize(("provider", "model", "price"), PUBLISHED_CACHE_WRITE)
def test_cache_write_price_matches_the_provider_page(
    provider: LLMProvider, model: str, price: float
) -> None:
    assert cache_write_price_per_1m(provider, model) == pytest.approx(price)


def test_a_provider_that_publishes_no_cache_rate_carries_none() -> None:
    """Mistral's pricing page has no cache column. A guessed discount would put
    a confidently wrong number on the Trace page."""
    assert cache_read_price_per_1m(LLMProvider.MISTRAL, "mistral-large-latest") is None
    assert provider_spec(LLMProvider.MISTRAL).cache_read_per_1m == {}


def test_every_priced_provider_cites_where_the_numbers_came_from() -> None:
    for provider in LLMProvider:
        spec = provider_spec(provider)
        if spec.pricing_per_1m:
            assert spec.pricing_source_url, f"{provider.value} prices have no source URL"


def test_cache_read_is_billed_at_the_listed_rate_not_the_input_rate() -> None:
    """1M cached tokens on gpt-4.1 cost $0.50, not $2.00."""
    settings = LLMSettings(provider=LLMProvider.OPENAI, model="gpt-4.1", api_key="k")
    usage = LLMUsage(prompt_tokens=1_000_000, cached_tokens=1_000_000, completion_tokens=0)

    cost, basis = _estimate_cost(settings, usage)

    assert basis == "registry_estimate"
    assert cost == pytest.approx(0.50)


def test_anthropic_cache_writes_cost_more_than_plain_input() -> None:
    """A cache write is a 1.25x premium, so it must not be billed as input."""
    settings = LLMSettings(provider=LLMProvider.ANTHROPIC, model="claude-opus-5", api_key="k")
    usage = LLMUsage(prompt_tokens=1_000_000, cache_creation_tokens=1_000_000)

    cost, _ = _estimate_cost(settings, usage)

    assert cost == pytest.approx(6.25)


def test_a_user_override_does_not_borrow_published_cache_rates() -> None:
    """An override is a private contract; applying a public cache discount to it
    would invent a rate the user never gave."""
    settings = LLMSettings(
        provider=LLMProvider.OPENAI,
        model="gpt-4.1",
        api_key="k",
        usd_per_1k_prompt=0.01,
        usd_per_1k_completion=0.03,
    )
    usage = LLMUsage(prompt_tokens=1_000_000, cached_tokens=1_000_000)

    cost, basis = _estimate_cost(settings, usage)

    assert basis == "configured_rates"
    assert cost == pytest.approx(10.0)  # 1M / 1000 * 0.01, the plain input rate
