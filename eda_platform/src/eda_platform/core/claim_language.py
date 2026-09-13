"""Single source of the causal-claim language family."""

from __future__ import annotations

import re
from collections.abc import Iterable

# Causal-phrase family. A finding or interpretation whose text contains any of
# these phrases implies causation. One table, not two: the claim gate used to
# lack the bare infinitive, so "Discounts cause the 42 returns" passed the claim
# gate while the identical sentence was blocked in report prose.
CAUSAL_PHRASES: tuple[str, ...] = (
    "causes",
    "caused",
    "causal",
    "cause ",
    "drives",
    "drove",
    "driven by",
    "leads to",
    "led to",
    "because of",
    "due to",
    "effect of",
    "results in",
    "responsible for",
    "attributable to",
    "a consequence of",
    "triggered",
    # "explains" is causal in prose but is also the standard way to describe R²
    # ("explains 45% of the variance"); it is on the table because the review
    # walked a causal claim through it, and the variance form must be rephrased.
    "explains",
)

# Same family under the name report_validator imports it by.
REPORT_BODY_CAUSAL_TERMS: tuple[str, ...] = CAUSAL_PHRASES

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


# Correlational stand-in for each phrase above. Deleting a causal claim emptied
# two whole report sections in the 2026-08-26 Compare run, and the LLM repair
# round it relied on returned the same sentence three times; downgrading the
# wording keeps the observation and drops only the causal assertion.
CAUSAL_REWRITES: tuple[tuple[str, str], ...] = (
    ("a consequence of", "associated with"),
    ("attributable to", "associated with"),
    ("responsible for", "associated with"),
    ("driven by", "associated with"),
    ("because of", "alongside"),
    ("results in", "coincides with"),
    ("leads to", "is associated with"),
    ("effect of", "association with"),
    ("due to", "alongside"),
    ("led to", "coincided with"),
    ("explains", "is associated with"),
    ("triggered", "coincided with"),
    ("causes", "is associated with"),
    ("caused", "was associated with"),
    ("causal", "correlational"),
    ("drives", "is associated with"),
    ("drove", "was associated with"),
    ("cause ", "coincide with "),
)

# No causal phrase may appear here: the report validator scans claim text
# without stripping disclaimers, so even "not a causal claim" would re-trip it.
ASSOCIATION_QUALIFIER = "This is an observed association; the data does not show what produced it."

_REWRITE_PATTERN = re.compile(
    "|".join(
        re.escape(phrase)
        for phrase, _ in sorted(CAUSAL_REWRITES, key=lambda item: -len(item[0]))
    ),
    re.IGNORECASE,
)
_REWRITE_TABLE = dict(CAUSAL_REWRITES)


def rewrite_causal_language(text: str) -> str | None:
    """Downgrade causal wording to association wording, or None if it cannot be.

    None means "prune this claim": either there was nothing causal to rewrite,
    or a phrase survived the substitution and the claim still asserts a cause.
    Figures are never touched — no replacement carries a digit.
    """

    def _replace(match: re.Match[str]) -> str:
        replacement = _REWRITE_TABLE[match.group(0).lower()]
        if match.group(0)[:1].isupper():
            return replacement[:1].upper() + replacement[1:]
        return replacement

    rewritten, substitutions = _REWRITE_PATTERN.subn(_replace, text)
    if substitutions == 0:
        return None
    rewritten = f"{rewritten.rstrip()} {ASSOCIATION_QUALIFIER}"
    if contains_causal_phrase(rewritten):
        return None
    return rewritten
