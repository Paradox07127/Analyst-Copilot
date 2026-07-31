"""Test-only builders for legacy semantic JSON fixtures.

Production code must use ``SemanticSeedsRepository``. These helpers intentionally
live outside the installed package so tests can construct pre-repository fixtures.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

from eda_platform.core.meaning_proposals import (
    MeaningProposal,
    MeaningProposals,
)
from eda_platform.core.semantic import (
    FieldMeaning,
    SemanticSeeds,
    VerifiedRelation,
)
from eda_platform.core.store import ArtifactStore


def load_seeds(project_dir: Path | str) -> SemanticSeeds:
    path = Path(project_dir) / "semantic" / "seeds.json"
    if not path.exists():
        return SemanticSeeds()
    return SemanticSeeds.model_validate_json(path.read_text(encoding="utf-8"))


def save_seeds(project_dir: Path | str, seeds: SemanticSeeds) -> Path:
    path = Path(project_dir) / "semantic" / "seeds.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(seeds.model_dump_json(indent=2), encoding="utf-8")
    os.replace(temporary, path)
    return path


def load_meaning_proposals(project_dir: Path | str) -> MeaningProposals:
    path = Path(project_dir) / "semantic" / "meaning_proposals.json"
    if not path.exists():
        return MeaningProposals()
    return MeaningProposals.model_validate_json(path.read_text(encoding="utf-8"))


def save_meaning_proposals(
    project_dir: Path | str, proposals: MeaningProposals
) -> Path:
    path = Path(project_dir) / "semantic" / "meaning_proposals.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(proposals.model_dump_json(indent=2), encoding="utf-8")
    os.replace(temporary, path)
    return path


def upsert_proposals(
    project_dir: Path | str, drafts: Iterable[MeaningProposal]
) -> MeaningProposals:
    proposals = load_meaning_proposals(project_dir)
    for draft in drafts:
        if not draft.meaning.strip():
            continue
        existing = proposals.find(draft.dataset, draft.column)
        incoming = draft.model_copy(update={"status": "proposed"})
        if existing is None:
            proposals.proposals.append(incoming)
        elif existing.status == "proposed":
            proposals.proposals[proposals.proposals.index(existing)] = incoming
    save_meaning_proposals(project_dir, proposals)
    return proposals


def accept_proposal(
    project_dir: Path | str,
    dataset: str,
    column: str,
    *,
    meaning: str | None = None,
    unit: str | None = None,
) -> MeaningProposals:
    proposals = load_meaning_proposals(project_dir)
    proposal = proposals.find(dataset, column)
    if proposal is None:
        raise ValueError(f"Unknown meaning proposal: {dataset}.{column}")
    proposal.meaning = (proposal.meaning if meaning is None else meaning).strip()
    if not proposal.meaning:
        raise ValueError("A meaning is required to accept a proposal.")
    if unit is not None:
        proposal.unit_guess = unit.strip()
    seeds = load_seeds(project_dir)
    replacement = FieldMeaning(
        dataset=dataset,
        column=column,
        meaning=proposal.meaning,
        unit=proposal.unit_guess.strip() or None,
    )
    for index, field in enumerate(seeds.field_meanings):
        if field.dataset == dataset and field.column == column:
            replacement.aliases = field.aliases
            seeds.field_meanings[index] = replacement
            break
    else:
        seeds.field_meanings.append(replacement)
    proposal.status = "accepted"
    save_seeds(project_dir, seeds)
    save_meaning_proposals(project_dir, proposals)
    return proposals


def reject_proposal(
    project_dir: Path | str, dataset: str, column: str
) -> MeaningProposals:
    proposals = load_meaning_proposals(project_dir)
    proposal = proposals.find(dataset, column)
    if proposal is None:
        raise ValueError(f"Unknown meaning proposal: {dataset}.{column}")
    if proposal.status == "accepted":
        seeds = load_seeds(project_dir)
        seeds.field_meanings = [
            field
            for field in seeds.field_meanings
            if not (
                field.dataset == dataset
                and field.column == column
                and field.meaning == proposal.meaning
            )
        ]
        save_seeds(project_dir, seeds)
    proposal.status = "rejected"
    save_meaning_proposals(project_dir, proposals)
    return proposals


def accept_all_verified(project_dir: Path | str) -> int:
    proposals = load_meaning_proposals(project_dir)
    pending = [
        proposal
        for proposal in proposals.proposals
        if proposal.status == "proposed"
        and proposal.confidence == "verified"
        and proposal.meaning.strip()
    ]
    for proposal in pending:
        accept_proposal(project_dir, proposal.dataset, proposal.column)
    return len(pending)


def add_verified_relation(
    project_dir: Path | str, relation: VerifiedRelation
) -> SemanticSeeds:
    seeds = load_seeds(project_dir)
    for index, existing in enumerate(seeds.verified_relations):
        if existing.left == relation.left and existing.right == relation.right:
            seeds.verified_relations[index] = relation
            break
    else:
        seeds.verified_relations.append(relation)
    save_seeds(project_dir, seeds)
    return seeds


def confirm_promotion(
    store: ArtifactStore,
    project_id: str,
    finding_artifact_id: str,
    *,
    session_id: str | None = None,
) -> Path:
    from eda_platform.core.finding_freshness import assess_finding_freshness
    from eda_platform.drivers.knowledge_promotion import build_promotion_candidate

    if session_id is None:
        row = store.artifact_index_row(
            finding_artifact_id, project_id=project_id
        )
        if row is None:
            raise ValueError("Finding artifact could not be resolved.")
        session_id = str(row["session_id"])
    freshness = assess_finding_freshness(
        store,
        project_id,
        finding_artifact_id,
        finding_session_id=session_id,
    )
    if freshness.status != "fresh":
        raise ValueError(
            f"This finding is {freshness.status} and cannot be promoted yet."
        )
    candidate = build_promotion_candidate(
        store, project_id, finding_artifact_id, session_id=session_id
    )
    seeds = load_seeds(store.project_dir(project_id))
    seeds.verified_answers = [
        answer
        for answer in seeds.verified_answers
        if answer.question != candidate.question
    ]
    seeds.verified_answers.append(candidate)
    return save_seeds(store.project_dir(project_id), seeds)
