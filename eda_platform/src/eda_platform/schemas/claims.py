"""ClaimBundle — typed claims the agent emits instead of prose.

Reports are rendered only from claims that passed the exit gates
(core/claim_gates.py), so the schema is strict by construction: frozen,
extra=forbid, and every evidence reference is a qualified
"<receipt_id>:<fact_id>" pointer that the reachability gate can resolve
against committed receipts without NLP extraction.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# 07-31 plan §5.6 verbatim: the seven claim families.
ClaimType = Literal[
    "observation",
    "comparison",
    "prediction",
    "model",
    "absence",
    "causal",
    "recommendation",
]
ClaimSupportType = Literal["direct", "compression", "inference", "absence"]
EvidenceLane = Literal["exploratory", "confirmatory"]

# Mirrors schemas/receipts._RECEIPT_ID_RE (kept private there on purpose).
_RECEIPT_ID_PATTERN = r"rcpt_[0-9a-f]{24}"
_RECEIPT_ID_RE = re.compile(f"^{_RECEIPT_ID_PATTERN}$")
_QUALIFIED_REF_RE = re.compile(f"^({_RECEIPT_ID_PATTERN}):(.+)$")


def split_evidence_ref(ref: str) -> tuple[str, str]:
    """Split a qualified "<receipt_id>:<fact_id>" reference."""
    match = _QUALIFIED_REF_RE.fullmatch(ref)
    if match is None:
        raise ValueError(
            f"evidence reference {ref!r} must be '<receipt_id>:<fact_id>' "
            "with a rcpt_<24 hex> receipt id."
        )
    return match.group(1), match.group(2)


class ClaimScope(BaseModel):
    """Declared coverage of a claim; the absence gate compares it against the
    resolved scope actually scanned by the cited coverage receipts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_ids: tuple[str, ...] = ()
    columns: tuple[str, ...] = ()
    filters: str | None = None
    time_range: str | None = None


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str = Field(min_length=1)
    claim_type: ClaimType
    claim_text: str = Field(min_length=1)
    support_type: ClaimSupportType
    evidence_fact_ids: tuple[str, ...] = ()
    derivation_ids: tuple[str, ...] = ()
    # Bare receipt ids whose `statistics` field backs this claim.
    statistics_receipt_ids: tuple[str, ...] = ()
    uncertainty: str | None = None
    limitations: tuple[str, ...] = ()
    # Required (non-empty) for recommendation claims — enforced by the gates.
    assumptions: tuple[str, ...] = ()
    scope: ClaimScope | None = None

    @model_validator(mode="after")
    def _well_formed(self) -> Claim:
        for ref in (*self.evidence_fact_ids, *self.derivation_ids):
            split_evidence_ref(ref)
        for receipt_id in self.statistics_receipt_ids:
            if not _RECEIPT_ID_RE.fullmatch(receipt_id):
                raise ValueError(
                    f"statistics receipt id {receipt_id!r} must match rcpt_<24 hex>."
                )
        if not self.evidence_fact_ids and not self.derivation_ids:
            raise ValueError(
                f"claim {self.claim_id!r} cites no evidence facts or derivations."
            )
        if (self.claim_type == "absence") != (self.support_type == "absence"):
            raise ValueError(
                f"claim {self.claim_id!r}: claim_type 'absence' and support_type "
                "'absence' must be used together."
            )
        return self


class ClaimBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_bundle_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    evidence_lane: EvidenceLane
    claims: tuple[Claim, ...] = Field(min_length=1)
    # R4: independent replication is a licence, not a phrasing choice. The
    # gate rejects this flag (and equivalent claim-text phrasing) unless a
    # cited receipt carries replication_kind holdout/external_replication.
    declares_independent_replication: bool = False

    @model_validator(mode="after")
    def _unique_claim_ids(self) -> ClaimBundle:
        claim_ids = [claim.claim_id for claim in self.claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("claim_id values must be unique within a bundle.")
        return self

    def referenced_receipt_ids(self) -> tuple[str, ...]:
        """Every receipt id cited anywhere in the bundle, in first-seen order."""
        seen: dict[str, None] = {}
        for claim in self.claims:
            for ref in (*claim.evidence_fact_ids, *claim.derivation_ids):
                seen.setdefault(split_evidence_ref(ref)[0])
            for receipt_id in claim.statistics_receipt_ids:
                seen.setdefault(receipt_id)
        return tuple(seen)
