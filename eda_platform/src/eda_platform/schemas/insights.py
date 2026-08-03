"""Hypothesis and insight records for the exploration loop (doc §4.5, §5.5).

The split here is deliberate: `TransitionProposal` is what the model is allowed
to say, `InsightRecord` is what the deterministic reducer concludes. Any field
that decides trust lives only on the record.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from eda_platform.schemas.exploration import InsightFamily

InsightStatus = Literal["new", "reinforced", "refuted", "inconclusive"]

# Derived from the evidence on the record, never proposed. `contested` is the
# both-sides case; `unsupported` means no committed evidence at all.
InsightTrustLevel = Literal["supported", "contested", "refuted", "unsupported"]

# Statuses a proposal may ask for. `inconclusive` and `refuted` are reachable
# only as reducer verdicts, so asking for them carries no extra authority.
ProposedStatus = Literal["new", "reinforced", "refuted", "inconclusive"]
ProofComparison = Literal["supports", "contradicts"]


class InsightProof(BaseModel):
    """Fact-level, machine-checkable edge from an insight to one receipt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    receipt_id: str = Field(min_length=1)
    fact_ids: tuple[str, ...] = Field(min_length=1)
    comparison: ProofComparison
    evidence_independence_key: str | None = None


class Hypothesis(BaseModel):
    """One admitted line of enquiry; the unit the scheduler ranks and dedupes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hypothesis_id: str = Field(min_length=1)
    family: InsightFamily
    question: str = Field(min_length=1)
    fingerprint: str = Field(min_length=1)
    created_round: int = Field(ge=0)
    parent_hypothesis_id: str | None = None


class TransitionProposal(BaseModel):
    """What the model may submit. It cites evidence; it does not grade it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hypothesis_id: str = Field(min_length=1)
    insight_id: str = Field(min_length=1)
    family: InsightFamily
    claim_bundle_id: str = Field(min_length=1)
    supporting_receipt_ids: tuple[str, ...] = ()
    contradicting_receipt_ids: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    proposed_status: ProposedStatus

    @model_validator(mode="after")
    def _sides_are_disjoint(self) -> TransitionProposal:
        if len(self.supporting_receipt_ids) != len(set(self.supporting_receipt_ids)):
            raise ValueError("supporting receipt ids must be unique.")
        if len(self.contradicting_receipt_ids) != len(
            set(self.contradicting_receipt_ids)
        ):
            raise ValueError("contradicting receipt ids must be unique.")
        overlap = set(self.supporting_receipt_ids) & set(self.contradicting_receipt_ids)
        if overlap:
            raise ValueError(
                "a receipt cannot both support and contradict the same insight: "
                + ", ".join(sorted(overlap))
            )
        return self


class InsightRecord(BaseModel):
    """The reducer's verdict. Frozen so a later caller cannot edit the outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    insight_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    family: InsightFamily
    status: InsightStatus
    trust_level: InsightTrustLevel
    claim_bundle_id: str = Field(min_length=1)
    supporting_receipt_ids: tuple[str, ...] = ()
    contradicting_receipt_ids: tuple[str, ...] = ()
    proof: tuple[InsightProof, ...] = Field(min_length=1)
    limitations: tuple[str, ...] = ()
    created_round: int = Field(ge=0)
    last_updated_round: int = Field(ge=0)

    @model_validator(mode="after")
    def _rounds_are_ordered(self) -> InsightRecord:
        if self.last_updated_round < self.created_round:
            raise ValueError("last_updated_round cannot precede created_round.")
        if len(self.supporting_receipt_ids) != len(set(self.supporting_receipt_ids)):
            raise ValueError("supporting receipt ids must be unique.")
        if len(self.contradicting_receipt_ids) != len(
            set(self.contradicting_receipt_ids)
        ):
            raise ValueError("contradicting receipt ids must be unique.")
        overlap = set(self.supporting_receipt_ids) & set(self.contradicting_receipt_ids)
        if overlap:
            raise ValueError(
                "a receipt cannot both support and contradict the same insight: "
                + ", ".join(sorted(overlap))
            )
        return self
