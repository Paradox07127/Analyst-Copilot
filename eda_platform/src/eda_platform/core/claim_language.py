"""Single source of the causal-claim language family."""

from __future__ import annotations

from collections.abc import Iterable

# Causal-phrase family (English + Chinese). A finding or interpretation whose
# text contains any of these phrases implies causation.
CAUSAL_PHRASES: tuple[str, ...] = (
    "causes",
    "caused",
    "causal",
    "drives",
    "driven by",
    "leads to",
    "led to",
    "because of",
    "due to",
    "effect of",
    "results in",
)

# Report prose uses a narrower English-only gate than findings and interpretations.
# Keep these families separate because they guard different trust boundaries.
REPORT_BODY_CAUSAL_TERMS: tuple[str, ...] = (
    "caused",
    "causes",
    "cause ",
    "drives",
    "drove",
    "because of",
    "leads to",
    "led to",
)

# Safe negated forms: a deliberate disclaimer ("not a causal claim") must not
# trip the causal gate, while a genuine causal assertion still must. Ordered
# longest-first so the broadest disclaimer is stripped before its substrings.
SAFE_CAUSAL_DISCLAIMERS: tuple[str, ...] = (
    "not a causal explanation",
    "not a causal",
    "not causal",
    "no causal",
    "non-causal",
)


def contains_causal_phrase(
    text: str,
    *,
    phrases: Iterable[str] = CAUSAL_PHRASES,
) -> bool:
    """Raw phrase scan (no disclaimer stripping) over the given phrase family."""
    lowered = text.lower()
    return any(phrase in lowered for phrase in phrases)


def implies_causation(text: str) -> bool:
    """True when ``text`` asserts causation after removing safe disclaimers."""
    lowered = text.lower()
    for disclaimer in SAFE_CAUSAL_DISCLAIMERS:
        lowered = lowered.replace(disclaimer, " ")
    return any(phrase in lowered for phrase in CAUSAL_PHRASES)
