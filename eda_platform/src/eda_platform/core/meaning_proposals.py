"""Machine-drafted field-meaning proposals awaiting human review.

Stored beside the join whitelist at <project_dir>/semantic/meaning_proposals.json;
a draft only becomes a FieldMeaning seed when a human accepts it.
"""

from __future__ import annotations

from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from eda_platform.core.semantic import FieldMeaning, SemanticSeeds

ProposalConfidence = Literal["verified", "hypothesis"]
ProposalStatus = Literal["proposed", "accepted", "rejected"]


class MeaningProposal(BaseModel):
    """One drafted column meaning with its review lifecycle state."""

    dataset: str
    column: str
    meaning: str
    unit_guess: str = ""
    # "verified" when the column's role passed a deterministic check.
    confidence: ProposalConfidence = "hypothesis"
    # "document": drafted in a round whose payload carried support-doc text;
    # the review UI badges these cards so provenance is visible at accept time.
    source: Literal["bootstrap", "document"] = "bootstrap"
    status: ProposalStatus = "proposed"
    revision: str = Field(default_factory=lambda: uuid4().hex)


class MeaningProposals(BaseModel):
    version: int = 1
    proposals: list[MeaningProposal] = Field(default_factory=list)

    def find(self, dataset: str, column: str) -> MeaningProposal | None:
        for proposal in self.proposals:
            if proposal.dataset == dataset and proposal.column == column:
                return proposal
        return None


def apply_proposal_to_seeds(
    seeds: SemanticSeeds, proposal: MeaningProposal
) -> None:
    """Write one accepted proposal as a FieldMeaning, replacing any prior entry."""
    unit = proposal.unit_guess.strip() or None
    for index, existing in enumerate(seeds.field_meanings):
        if existing.dataset == proposal.dataset and existing.column == proposal.column:
            seeds.field_meanings[index] = FieldMeaning(
                dataset=proposal.dataset,
                column=proposal.column,
                meaning=proposal.meaning,
                unit=unit,
                aliases=existing.aliases,
            )
            return
    seeds.field_meanings.append(
        FieldMeaning(
            dataset=proposal.dataset,
            column=proposal.column,
            meaning=proposal.meaning,
            unit=unit,
        )
    )
