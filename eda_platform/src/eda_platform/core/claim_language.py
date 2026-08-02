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


# Model/prediction assertions an answer may only make when a ModelCard is in
# evidence. Phrases, not bare words: a substring scan for "model" also fires on
# "the data model", and a gate that noisy gets ignored rather than tightened.
# Deliberately narrow — under-reporting is recoverable, a false rejection
# silently discards a correct answer.
MODEL_ASSERTION_TERMS: tuple[str, ...] = (
    "model produced",
    "model predicts",
    "model predicted",
    "model achieved",
    "model scored",
    "model classified",
    "model was trained",
    "built a model",
    "fitted a model",
    "trained a model",
    "predictive model",
    "classifier",
    "classified as",
    "was predicted",
    "were predicted",
    "prediction accuracy",
    "training accuracy",
)

SAFE_MODEL_DISCLAIMERS: tuple[str, ...] = (
    "no model was trained",
    "no model was built",
    "without building a model",
    "no predictive model",
    "not a predictive model",
)


def contains_causal_phrase(
    text: str,
    *,
    phrases: Iterable[str] = CAUSAL_PHRASES,
) -> bool:
    """Raw phrase scan (no disclaimer stripping) over the given phrase family."""
    lowered = text.lower()
    return any(phrase in lowered for phrase in phrases)


def asserts_model_capability(
    text: str,
    *,
    terms: Iterable[str] = MODEL_ASSERTION_TERMS,
) -> bool:
    """True when ``text`` claims a model exists, after removing safe disclaimers."""
    lowered = text.lower()
    for disclaimer in SAFE_MODEL_DISCLAIMERS:
        lowered = lowered.replace(disclaimer, " ")
    return any(term in lowered for term in terms)


def implies_causation(text: str) -> bool:
    """True when ``text`` asserts causation after removing safe disclaimers."""
    lowered = text.lower()
    for disclaimer in SAFE_CAUSAL_DISCLAIMERS:
        lowered = lowered.replace(disclaimer, " ")
    return any(phrase in lowered for phrase in CAUSAL_PHRASES)
