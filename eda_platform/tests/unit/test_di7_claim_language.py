"""DI7-B item ①: the causal-claim language family has a single source.

These lock the shared ``core.claim_language`` contract that the interpretation,
orchestrator, and decision-report gates now consume.
"""

from __future__ import annotations

from eda_platform.core.claim_language import (
    CAUSAL_PHRASES,
    SAFE_CAUSAL_DISCLAIMERS,
    contains_causal_phrase,
    implies_causation,
)


def test_additional_english_causal_phrase_hits() -> None:
    assert implies_causation("Promotions cause higher revenue because of discounts.")
    assert implies_causation("The promotion causes higher revenue.")
    assert implies_causation("Revenue is driven by the discount.")
    assert implies_causation("The drop leads to churn.")
    assert contains_causal_phrase("This is the effect of the change.")


def test_english_causal_phrase_hits() -> None:
    assert implies_causation("A price reduction caused sales to increase.")
    assert implies_causation("This caused sales to change.")
    assert implies_causation("Lower prices led to higher sales.")


def test_safe_disclaimer_is_exempt() -> None:
    # A deliberate disclaimer must not itself trip the causal gate.
    assert not implies_causation("This is an observed result, not a causal claim.")
    assert not implies_causation("These are non-causal, associative patterns.")
    # The raw scan (no stripping) still flags the bare "causal" substring.
    assert contains_causal_phrase("This is not a causal claim.")


def test_clean_text_is_not_flagged() -> None:
    assert not implies_causation("Revenue rose 12% in the northern region.")
    assert not contains_causal_phrase("Average order value is higher for members.")


def test_disclaimer_strip_preserves_genuine_claim() -> None:
    # Stripping the disclaimer must not mask a genuine causal assertion elsewhere.
    text = "This is not a causal claim, but the outage caused the revenue drop."
    assert implies_causation(text)


def test_family_shape() -> None:
    assert "causes" in CAUSAL_PHRASES
    assert "due to" in CAUSAL_PHRASES
    assert "not a causal" in SAFE_CAUSAL_DISCLAIMERS
