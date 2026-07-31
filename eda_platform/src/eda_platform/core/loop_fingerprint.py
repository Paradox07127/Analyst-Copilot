"""Content-based fingerprints for the analysis macro-loop (design doc §8.1).

Finding identity keys on the evidence value set, not wording: rephrasing a
finding over the same evidence keeps the fingerprint; any numeric change breaks
it. Findings without numeric evidence fall back to a digit- and
whitespace-stripped statement hash. Question fingerprints follow the di5
``_probe_fingerprint`` style, applied to question text.
"""

from __future__ import annotations

import hashlib
import re

_FINGERPRINT_HEX_CHARS = 16


def _sha256_16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:_FINGERPRINT_HEX_CHARS]


def finding_fingerprint(
    statements: list[str],
    evidence_values: list[tuple[str, str, float]],
    family_key: str = "",
) -> str:
    """Fingerprint a finding by its evidence set; fall back to normalized text.

    ``family_key`` is the §8.1 question-family key: the same evidence cited by
    different questions no longer collides. Empty keeps the legacy hash.
    """
    prefix = f"{family_key}\x1e" if family_key else ""
    if evidence_values:
        canonical = sorted(
            {
                (artifact_type, locator, repr(round(float(value), 6)))
                for artifact_type, locator, value in evidence_values
            }
        )
        payload = "\n".join("\x1f".join(item) for item in canonical)
        return _sha256_16(prefix + payload)
    merged = " ".join(statements)
    normalized = re.sub(r"[\d\s]+", "", merged).casefold()
    return _sha256_16(prefix + normalized)


def question_fingerprint(question_text: str) -> str:
    """Stable fingerprint of question text: casefold, drop punctuation, collapse spaces."""
    stripped = re.sub(r"[^\w\s]+", " ", question_text.casefold())
    normalized = re.sub(r"\s+", " ", stripped).strip()
    return _sha256_16(normalized)
