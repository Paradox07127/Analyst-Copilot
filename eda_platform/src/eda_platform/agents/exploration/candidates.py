"""Deterministic candidate materialization and coverage accounting (E4a).

The LLM may author :class:`HypothesisProposal`; every identity and coverage
field on :class:`CandidateSeed` is derived here. This keeps wording changes out
of canonical identity while preserving scope changes as distinct hypotheses.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from eda_platform.core.ids import stable_hash
from eda_platform.schemas.exploration import InsightFamily
from eda_platform.schemas.hypotheses import (
    HypothesisPredicate,
    HypothesisProposal,
    HypothesisProposalBatch,
)

FOLLOWUP_ROUND_DECAY = 0.9

HypothesisStatus = Literal[
    "proposed",
    "admitted",
    "running",
    "supported",
    "refuted",
    "inconclusive",
    "rejected_duplicate",
    "rejected_infeasible",
    "rejected_policy",
]
CandidateOrigin = Literal["bootstrap", "agent", "user", "followup", "mandatory"]


class DatasetExplorationProfile(BaseModel):
    """Metadata-only signals used to create the three Eval-0 mandatory probes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str = Field(min_length=1)
    region_dimensions: tuple[str, ...] = ()
    metric_columns: tuple[str, ...] = ()
    missing_value_columns: tuple[str, ...] = ()
    missingness_group_dimensions: tuple[str, ...] = ()
    datetime_columns: tuple[str, ...] = ()
    spike_metric_columns: tuple[str, ...] = ()

    @field_validator(
        "region_dimensions",
        "metric_columns",
        "missing_value_columns",
        "missingness_group_dimensions",
        "datetime_columns",
        "spike_metric_columns",
    )
    @classmethod
    def _canonical_columns(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("profile columns cannot be blank.")
        return tuple(sorted(set(value), key=_normalize))


@dataclass(frozen=True)
class CandidateSeed:
    """Control-plane envelope around one agent-authored proposal."""

    proposal: HypothesisProposal
    hypothesis_id: str
    hypothesis_fingerprint: str
    canonical_group_key: str
    coverage_key: str
    sequence_index: int
    status: HypothesisStatus = "proposed"
    origin: CandidateOrigin = "agent"
    mandatory: bool = False
    priority: float = 0.0


@dataclass(frozen=True)
class CanonicalGroup:
    canonical_group_key: str
    member_fingerprints: tuple[str, ...]
    member_count: int
    kept_count: int


@dataclass(frozen=True)
class CandidateCompression:
    representatives: tuple[CandidateSeed, ...]
    groups: tuple[CanonicalGroup, ...]


@dataclass(frozen=True)
class CoverageRow:
    coverage_key: str
    family: InsightFamily
    dataset_ids: tuple[str, ...]
    columns: tuple[str, ...]
    method_family: str
    probe_kind: str
    mandatory: bool
    explored: bool


@dataclass(frozen=True)
class CoverageMatrix:
    rows: tuple[CoverageRow, ...]


def candidate_seed(
    proposal: HypothesisProposal,
    *,
    sequence_index: int,
    origin: CandidateOrigin = "agent",
    mandatory: bool = False,
) -> CandidateSeed:
    """Attach identities derived only from execution-relevant semantic fields."""
    if sequence_index < 1:
        raise ValueError("sequence_index must be at least 1.")
    signature = _semantic_signature(proposal)
    fingerprint = stable_hash(signature, length=32)
    group_key = "hgrp_" + stable_hash(_group_signature(proposal), length=24)
    coverage_key = "cov_" + stable_hash(signature, length=24)
    return CandidateSeed(
        proposal=proposal,
        hypothesis_id=f"hyp_{fingerprint[:24]}",
        hypothesis_fingerprint=fingerprint,
        canonical_group_key=group_key,
        coverage_key=coverage_key,
        sequence_index=sequence_index,
        origin=origin,
        mandatory=mandatory,
    )


def materialize_proposal_batch(
    batch: HypothesisProposalBatch,
    *,
    first_sequence_index: int,
    origin: CandidateOrigin = "agent",
) -> tuple[CandidateSeed, ...]:
    """Convert one strict structured batch into system-owned candidate envelopes."""
    if first_sequence_index < 1:
        raise ValueError("first_sequence_index must be at least 1.")
    return tuple(
        candidate_seed(
            proposal,
            sequence_index=first_sequence_index + offset,
            origin=origin,
        )
        for offset, proposal in enumerate(batch.proposals)
    )


def followup_candidate_seed(
    proposal: HypothesisProposal,
    *,
    parent: CandidateSeed,
    sequence_index: int,
    rounds_since_parent: int = 1,
) -> CandidateSeed:
    """Convert a follow-up with the existing deterministic 0.9 round decay."""
    if rounds_since_parent < 1:
        raise ValueError("rounds_since_parent must be at least 1.")
    if (
        proposal.parent_hypothesis_id is not None
        and proposal.parent_hypothesis_id != parent.hypothesis_id
    ):
        raise ValueError("follow-up parent_hypothesis_id does not match the parent seed.")
    seeded = candidate_seed(
        proposal,
        sequence_index=sequence_index,
        origin="followup",
    )
    return replace(
        seeded,
        priority=round(parent.priority * FOLLOWUP_ROUND_DECAY**rounds_since_parent, 12),
    )


def compress_canonical_groups(
    candidates: Iterable[CandidateSeed], *, max_per_group: int = 1
) -> CandidateCompression:
    """Compress template spam without depending on proposal/input ordering.

    Mandatory seeds are never discarded. Optional members are retained up to
    ``max_per_group`` after deterministic fingerprint ordering.
    """
    if max_per_group < 1:
        raise ValueError("max_per_group must be at least 1.")
    grouped: dict[str, list[CandidateSeed]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.canonical_group_key].append(candidate)

    representatives: list[CandidateSeed] = []
    records: list[CanonicalGroup] = []
    for group_key in sorted(grouped):
        members = sorted(grouped[group_key], key=lambda item: item.hypothesis_fingerprint)
        mandatory = [item for item in members if item.mandatory]
        optional = sorted(
            (item for item in members if not item.mandatory),
            key=lambda item: (-item.priority, item.hypothesis_fingerprint),
        )
        kept_by_fingerprint = {
            item.hypothesis_fingerprint: item
            for item in (*mandatory, *optional[:max_per_group])
        }
        kept = tuple(kept_by_fingerprint[key] for key in sorted(kept_by_fingerprint))
        representatives.extend(kept)
        records.append(
            CanonicalGroup(
                canonical_group_key=group_key,
                member_fingerprints=tuple(
                    item.hypothesis_fingerprint for item in members
                ),
                member_count=len(members),
                kept_count=len(kept),
            )
        )
    return CandidateCompression(
        representatives=tuple(
            sorted(representatives, key=lambda item: item.hypothesis_fingerprint)
        ),
        groups=tuple(records),
    )


def mandatory_probe_seeds(
    profiles: Sequence[DatasetExplorationProfile], *, first_sequence_index: int = 1
) -> tuple[CandidateSeed, ...]:
    """Generate hard probes for region gaps, missing mechanisms and spike days."""
    if first_sequence_index < 1:
        raise ValueError("first_sequence_index must be at least 1.")
    proposals: list[HypothesisProposal] = []
    for profile in sorted(profiles, key=lambda item: _normalize(item.dataset_id)):
        for region in profile.region_dimensions:
            for metric in profile.metric_columns:
                proposals.append(
                    HypothesisProposal(
                        statement=f"Does {metric} differ materially across {region}?",
                        rationale="Eval-0 requires region-by-metric coverage.",
                        expected_evidence="A group comparison with effect and uncertainty.",
                        falsification_conditions=(
                            f"No material {metric} difference across {region} is observed.",
                        ),
                        family=InsightFamily.DIAGNOSTIC,
                        method_family="compare_groups",
                        dataset_ids=(profile.dataset_id,),
                        columns=(region, metric),
                        probe_kind="region_difference",
                        predicate=HypothesisPredicate(
                            metric=metric,
                            operator="differs",
                            left_operand=region,
                            right_operand="groups",
                        ),
                    )
                )
        for column in profile.missing_value_columns:
            if profile.missingness_group_dimensions:
                for group in profile.missingness_group_dimensions:
                    proposals.append(
                        HypothesisProposal(
                            statement=(
                                f"Is missingness in {column} materially associated "
                                f"with {group} (at least 5 percentage points)?"
                            ),
                            rationale="A missing rate alone does not explain its mechanism.",
                            expected_evidence=(
                                "A missingness diagnostic across an observed group."
                            ),
                            falsification_conditions=(
                                "The observed missing-rate range across "
                                f"{group} is below 5 percentage points.",
                            ),
                            family=InsightFamily.DIAGNOSTIC,
                            method_family="diagnose_missingness",
                            dataset_ids=(profile.dataset_id,),
                            columns=(column, group),
                            probe_kind="missingness_mechanism",
                            predicate=HypothesisPredicate(
                                metric=column,
                                operator="associated_with",
                                right_operand=group,
                                threshold=5.0,
                            ),
                        )
                    )
            else:
                # Fail closed: a rate can motivate future work but is not evidence
                # about a missingness mechanism without an observed grouping scope.
                proposals.append(
                    HypothesisProposal(
                        statement=f"What is the missing rate of {column}?",
                        rationale=(
                            "No grouping dimension is available for a mechanism probe."
                        ),
                        expected_evidence="A direct missing count and rate only.",
                        falsification_conditions=(f"{column} has no missing values.",),
                        family=InsightFamily.DESCRIPTIVE,
                        method_family="diagnose_missingness",
                        dataset_ids=(profile.dataset_id,),
                        columns=(column,),
                        probe_kind="missingness_rate_scan",
                        predicate=HypothesisPredicate(
                            metric=column,
                            operator="greater_than",
                            threshold=0,
                        ),
                    )
                )
        for datetime_column in profile.datetime_columns:
            for metric in profile.spike_metric_columns:
                proposals.append(
                    HypothesisProposal(
                        statement=(
                            f"Does {metric} contain a date-localized spike "
                            f"by {datetime_column}?"
                        ),
                        rationale="Aggregate outliers must be traced to their time scope.",
                        expected_evidence="A dated series with a localized spike assessment.",
                        falsification_conditions=(
                            f"No date-localized {metric} spike is observed.",
                        ),
                        family=InsightFamily.EXPLORATORY,
                        method_family="analyze_time_series",
                        dataset_ids=(profile.dataset_id,),
                        columns=(datetime_column, metric),
                        probe_kind="spike_day",
                        predicate=HypothesisPredicate(
                            metric=metric,
                            operator="has_spike",
                            right_operand=datetime_column,
                            threshold=3.5,
                        ),
                    )
                )

    ordered = sorted(proposals, key=lambda proposal: stable_hash(_semantic_signature(proposal), 32))
    return tuple(
        candidate_seed(
            proposal,
            sequence_index=first_sequence_index + offset,
            origin="mandatory",
            mandatory=True,
        )
        for offset, proposal in enumerate(ordered)
    )


def coverage_matrix(
    targets: Iterable[CandidateSeed], *, executed_coverage_keys: Iterable[str]
) -> CoverageMatrix:
    """Materialize the deterministic coverage target × execution matrix."""
    executed = frozenset(executed_coverage_keys)
    by_key: dict[str, CandidateSeed] = {}
    for target in targets:
        prior = by_key.get(target.coverage_key)
        if prior is None or (target.mandatory and not prior.mandatory):
            by_key[target.coverage_key] = target
    rows = tuple(
        CoverageRow(
            coverage_key=coverage_key,
            family=seed.proposal.family,
            dataset_ids=tuple(sorted(seed.proposal.dataset_ids, key=_normalize)),
            columns=tuple(seed.proposal.columns),
            method_family=seed.proposal.method_family,
            probe_kind=seed.proposal.probe_kind,
            mandatory=seed.mandatory,
            explored=coverage_key in executed,
        )
        for coverage_key, seed in sorted(by_key.items())
    )
    return CoverageMatrix(rows=rows)


def unexplored_coverage(matrix: CoverageMatrix) -> tuple[CoverageRow, ...]:
    """Return ``coverage targets - executed scopes`` without model narration."""
    return tuple(row for row in matrix.rows if not row.explored)


def with_priority(candidate: CandidateSeed, priority: float) -> CandidateSeed:
    """Return a frontier-ready copy without mutating the immutable seed."""
    return replace(candidate, priority=priority)


def _semantic_signature(proposal: HypothesisProposal) -> dict[str, object]:
    return {
        "family": proposal.family.value,
        "method_family": _normalize(proposal.method_family),
        "dataset_ids": sorted(_normalize(item) for item in proposal.dataset_ids),
        "columns": [_normalize(item) for item in proposal.columns],
        "segment": _normalize(proposal.segment or ""),
        "time_scope": _normalize(proposal.time_scope or ""),
        "probe_kind": _normalize(proposal.probe_kind),
        "predicate": {
            "metric": _normalize(proposal.predicate.metric),
            "operator": proposal.predicate.operator,
            "left_operand": _normalize(proposal.predicate.left_operand or ""),
            "right_operand": _normalize(proposal.predicate.right_operand or ""),
            "threshold": proposal.predicate.threshold,
        },
    }


def _group_signature(proposal: HypothesisProposal) -> dict[str, object]:
    """Template identity deliberately omits concrete columns and prose."""
    return {
        "family": proposal.family.value,
        "method_family": _normalize(proposal.method_family),
        "dataset_ids": sorted(_normalize(item) for item in proposal.dataset_ids),
        "column_arity": len(proposal.columns),
        "has_segment": proposal.segment is not None,
        "has_time_scope": proposal.time_scope is not None,
        "probe_kind": _normalize(proposal.probe_kind),
        # Grouping is intentionally a coarse template-spam boundary. Concrete
        # predicate operands and metrics remain in ``_semantic_signature`` so
        # opposite or differently scoped hypotheses still get distinct IDs.
        "predicate_operator": proposal.predicate.operator,
    }


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    return re.sub(r"\s+", " ", normalized)
