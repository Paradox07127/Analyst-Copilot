"""Replayable E4a admission, scoring, quota selection and stop decisions."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from eda_platform.agents.exploration.candidates import CandidateSeed, HypothesisStatus
from eda_platform.schemas.exploration import BranchConstraint, InsightFamily

_HARD_ABANDONMENT_REASONS = frozenset({"refuted", "gate_rejected"})


class CandidateSignals(BaseModel):
    """System-computed inputs; the proposal schema intentionally has no scores."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    business_value: float = Field(default=0.0, ge=0.0, le=1.0)
    information_gain_proxy: float = Field(default=0.0, ge=0.0, le=1.0)
    expected_cost: float = Field(default=0.0, ge=0.0, le=1.0)
    multiplicity_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    query_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{16}$")


class PriorityWeights(BaseModel):
    """Versioned run policy inputs; no purportedly optimal weights are hard-coded."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    business_value: float
    information_gain_proxy: float
    novelty: float
    coverage_gap: float
    feasibility: float
    expected_cost: float
    redundancy: float
    multiplicity_risk: float

    @model_validator(mode="after")
    def _non_negative(self) -> PriorityWeights:
        values = self.model_dump().values()
        if any(not math.isfinite(value) for value in values):
            raise ValueError("priority weights must be finite.")
        if any(value < 0 for value in values):
            raise ValueError("priority weights must be non-negative magnitudes.")
        return self


class SchedulerPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scoring_policy_version: str = Field(min_length=1)
    weights: PriorityWeights
    admission_priority: float
    no_information_priority: float
    max_batch_size: int = Field(ge=1)
    max_multiplicity_risk: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _thresholds_are_finite(self) -> SchedulerPolicy:
        if not math.isfinite(self.admission_priority) or not math.isfinite(
            self.no_information_priority
        ):
            raise ValueError("scheduler priority thresholds must be finite.")
        return self


class PriorityFeatures(BaseModel):
    """The immutable eight-dimensional vector persisted for offline replay."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    business_value: float = Field(ge=0.0, le=1.0)
    information_gain_proxy: float = Field(ge=0.0, le=1.0)
    novelty: float = Field(ge=0.0, le=1.0)
    coverage_gap: float = Field(ge=0.0, le=1.0)
    feasibility: float = Field(ge=0.0, le=1.0)
    expected_cost: float = Field(ge=0.0, le=1.0)
    redundancy: float = Field(ge=0.0, le=1.0)
    multiplicity_risk: float = Field(ge=0.0, le=1.0)


@dataclass(frozen=True)
class AdmissionContext:
    dataset_columns: Mapping[str, frozenset[str]]
    allowed_dataset_ids: frozenset[str]
    supported_method_families: frozenset[str]
    historical_hypothesis_fingerprints: frozenset[str]
    answered_hypothesis_fingerprints: frozenset[str]
    executed_query_fingerprints: frozenset[str]
    remaining_cost: float
    family_quota_remaining: Mapping[InsightFamily, int]
    unexplored_coverage_keys: frozenset[str]
    # E6: "tried + why it failed" facts from abandoned lines. Hard reasons
    # (refuted / gate_rejected) block re-admission; soft ones stay advisory.
    abandoned_constraints: tuple[BranchConstraint, ...] = ()


AdmissionCheckName = Literal[
    "scope_exists",
    "policy_and_capability",
    "novelty",
    "not_already_answered",
    "not_previously_abandoned",
    "falsifiable",
    "within_remaining_budget",
    "coverage_and_quota",
    "multiplicity_acceptable",
]


@dataclass(frozen=True)
class AdmissionCheck:
    name: AdmissionCheckName
    passed: bool
    detail_code: str


@dataclass(frozen=True)
class SchedulingDecision:
    hypothesis_id: str
    hypothesis_fingerprint: str
    family: InsightFamily
    status: HypothesisStatus
    admission_checks: tuple[AdmissionCheck, ...]
    priority_features: PriorityFeatures
    priority: float
    scoring_policy_version: str
    quota_deferred: bool
    chosen: bool


@dataclass(frozen=True)
class SchedulingResult:
    decisions: tuple[SchedulingDecision, ...]
    chosen_hypothesis_ids: tuple[str, ...]


@dataclass(frozen=True)
class NoNewInformationDecision:
    consecutive_no_information_rounds: int
    highest_frontier_priority: float | None
    priority_threshold: float
    required_rounds: int
    should_stop: bool
    reason: str


def canonical_query_fingerprint(query: str) -> str:
    """Canonical query identity from §7.4: trim semicolon/space, lower, sha256:16."""
    normalized = re.sub(r"\s+", " ", query.strip().rstrip(";").strip()).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def family_quotas_for_level(
    thinking_level: Literal["quick", "standard", "deep"],
    coverage_targets: Sequence[InsightFamily],
    *,
    quota_per_family: int = 1,
) -> dict[InsightFamily, int]:
    """Validate the documented quick/full-family policy and materialize quotas."""
    if quota_per_family < 1:
        raise ValueError("quota_per_family must be at least 1.")
    unique = tuple(dict.fromkeys(coverage_targets))
    if len(unique) != len(coverage_targets):
        raise ValueError("coverage targets must be unique.")
    if thinking_level == "quick":
        if len(unique) not in {2, 3}:
            raise ValueError("quick exploration requires exactly 2 or 3 family targets.")
    elif set(unique) != set(InsightFamily):
        raise ValueError("standard and deep exploration require all six family targets.")
    return {
        family: quota_per_family
        for family in sorted(unique, key=lambda item: item.value)
    }


def schedule_candidates(
    candidates: Sequence[CandidateSeed],
    *,
    signals: Mapping[str, CandidateSignals],
    context: AdmissionContext,
    policy: SchedulerPolicy,
) -> SchedulingResult:
    """Run all eight admission checks, score, then reserve family quota slots."""
    if not math.isfinite(context.remaining_cost) or context.remaining_cost < 0:
        raise ValueError("remaining_cost must be finite and non-negative.")
    if any(value < 0 for value in context.family_quota_remaining.values()):
        raise ValueError("family quotas cannot be negative.")

    hard_abandonment: dict[str, str] = {}
    for constraint in context.abandoned_constraints:
        if constraint.reason in _HARD_ABANDONMENT_REASONS:
            current = hard_abandonment.get(constraint.hypothesis_fingerprint)
            code = f"abandoned_{constraint.reason}"
            if current is None or code < current:
                hard_abandonment[constraint.hypothesis_fingerprint] = code

    ordered = tuple(sorted(candidates, key=lambda item: item.hypothesis_fingerprint))
    group_members: dict[str, list[CandidateSeed]] = defaultdict(list)
    for candidate in ordered:
        group_members[candidate.canonical_group_key].append(candidate)
    group_rank = {
        member.hypothesis_id: index
        for members in group_members.values()
        for index, member in enumerate(members)
    }
    exact_counts = Counter(item.hypothesis_fingerprint for item in ordered)
    seen_exact: set[str] = set()
    seen_queries: set[str] = set()
    provisional: list[SchedulingDecision] = []

    for candidate in ordered:
        signal = signals.get(candidate.hypothesis_id)
        if signal is None:
            raise ValueError(
                f"missing system signals for hypothesis {candidate.hypothesis_id!r}."
            )
        scope_exists = _scope_exists(candidate, context)
        policy_capability = _policy_and_capability(candidate, context)
        # Historical dedup exists to stop the model re-proposing its own past
        # hypotheses. A mandatory probe is instead a system-owned coverage
        # obligation replayed until it is explored, so it is exempt from the
        # historical set; within-batch dedup and the query check still apply.
        hypothesis_is_novel = (
            candidate.mandatory
            or candidate.hypothesis_fingerprint
            not in context.historical_hypothesis_fingerprints
        ) and candidate.hypothesis_fingerprint not in seen_exact
        seen_exact.add(candidate.hypothesis_fingerprint)
        query_is_novel = (
            signal.query_fingerprint is None
            or (
                signal.query_fingerprint not in context.executed_query_fingerprints
                and signal.query_fingerprint not in seen_queries
            )
        )
        if signal.query_fingerprint is not None:
            seen_queries.add(signal.query_fingerprint)
        is_novel = hypothesis_is_novel and query_is_novel
        not_answered = (
            candidate.hypothesis_fingerprint
            not in context.answered_hypothesis_fingerprints
        )
        abandonment_code = hard_abandonment.get(candidate.hypothesis_fingerprint)
        not_abandoned = abandonment_code is None
        falsifiable = bool(candidate.proposal.falsification_conditions)
        within_budget = signal.expected_cost <= context.remaining_cost
        quota_remaining = context.family_quota_remaining.get(
            candidate.proposal.family, 0
        )
        coverage_and_quota = (
            candidate.mandatory
            or candidate.coverage_key in context.unexplored_coverage_keys
            or quota_remaining > 0
        )
        multiplicity_ok = signal.multiplicity_risk <= policy.max_multiplicity_risk
        checks = (
            _check("scope_exists", scope_exists, "scope_available", "scope_missing"),
            _check(
                "policy_and_capability",
                policy_capability,
                "policy_allowed",
                "policy_or_tool_denied",
            ),
            _check(
                "novelty",
                is_novel,
                "canonical_novel",
                (
                    "canonical_query_duplicate"
                    if not query_is_novel
                    else "canonical_hypothesis_duplicate"
                ),
            ),
            _check(
                "not_already_answered",
                not_answered,
                "not_answered",
                "stronger_answer_exists",
            ),
            _check(
                "not_previously_abandoned",
                not_abandoned,
                "not_abandoned",
                abandonment_code or "not_abandoned",
            ),
            _check("falsifiable", falsifiable, "falsifiable", "not_falsifiable"),
            _check(
                "within_remaining_budget",
                within_budget,
                "within_budget",
                "estimated_cost_exceeds_remaining",
            ),
            _check(
                "coverage_and_quota",
                coverage_and_quota,
                "coverage_needed",
                "quota_satisfied",
            ),
            _check(
                "multiplicity_acceptable",
                multiplicity_ok,
                "multiplicity_acceptable",
                "multiplicity_deferred",
            ),
        )
        status = _admission_status(
            scope_exists=scope_exists,
            policy_capability=policy_capability,
            is_novel=is_novel,
            not_answered=not_answered,
            not_abandoned=not_abandoned,
            falsifiable=falsifiable,
            within_budget=within_budget,
            multiplicity_ok=multiplicity_ok,
        )
        members = group_members[candidate.canonical_group_key]
        rank = group_rank[candidate.hypothesis_id]
        redundancy = 0.0 if len(members) == 1 else rank / (len(members) - 1)
        if exact_counts[candidate.hypothesis_fingerprint] > 1 and not is_novel:
            redundancy = 1.0
        features = PriorityFeatures(
            business_value=signal.business_value,
            information_gain_proxy=signal.information_gain_proxy,
            novelty=1.0 if is_novel else 0.0,
            coverage_gap=(
                1.0
                if candidate.mandatory
                or candidate.coverage_key in context.unexplored_coverage_keys
                else 0.0
            ),
            feasibility=1.0 if scope_exists and policy_capability and within_budget else 0.0,
            expected_cost=signal.expected_cost,
            redundancy=redundancy,
            multiplicity_risk=signal.multiplicity_risk,
        )
        provisional.append(
            SchedulingDecision(
                hypothesis_id=candidate.hypothesis_id,
                hypothesis_fingerprint=candidate.hypothesis_fingerprint,
                family=candidate.proposal.family,
                status=status,
                admission_checks=checks,
                priority_features=features,
                priority=_priority(features, policy.weights),
                scoring_policy_version=policy.scoring_policy_version,
                quota_deferred=not coverage_and_quota or not multiplicity_ok,
                chosen=False,
            )
        )

    by_id = {candidate.hypothesis_id: candidate for candidate in ordered}
    eligible = [
        decision
        for decision in provisional
        if decision.status == "admitted"
        and (
            decision.priority >= policy.admission_priority
            or by_id[decision.hypothesis_id].mandatory
        )
    ]
    representatives = _canonical_representatives(eligible, by_id=by_id)
    selection_order = _select(
        representatives,
        by_id=by_id,
        quotas=context.family_quota_remaining,
        max_batch_size=policy.max_batch_size,
    )
    chosen = frozenset(selection_order)
    decisions = tuple(
        replace(decision, chosen=decision.hypothesis_id in chosen)
        for decision in provisional
    )
    return SchedulingResult(
        decisions=decisions,
        chosen_hypothesis_ids=tuple(selection_order),
    )


def no_new_information_decision(
    *,
    consecutive_no_information_rounds: int,
    highest_frontier_priority: float | None,
    priority_threshold: float,
    required_rounds: int = 2,
) -> NoNewInformationDecision:
    """Return the exact replay input and verdict for ``no_new_information``."""
    if consecutive_no_information_rounds < 0:
        raise ValueError("consecutive_no_information_rounds cannot be negative.")
    if required_rounds < 1:
        raise ValueError("required_rounds must be at least 1.")
    if not math.isfinite(priority_threshold):
        raise ValueError("priority_threshold must be finite.")
    if highest_frontier_priority is not None and not math.isfinite(
        highest_frontier_priority
    ):
        raise ValueError("highest_frontier_priority must be finite when present.")
    frontier_below = (
        highest_frontier_priority is None
        or highest_frontier_priority < priority_threshold
    )
    should_stop = (
        consecutive_no_information_rounds >= required_rounds and frontier_below
    )
    if should_stop:
        reason = "streak_met_and_frontier_below_threshold"
    elif consecutive_no_information_rounds < required_rounds:
        reason = "streak_not_met"
    else:
        reason = "frontier_at_or_above_threshold"
    return NoNewInformationDecision(
        consecutive_no_information_rounds=consecutive_no_information_rounds,
        highest_frontier_priority=highest_frontier_priority,
        priority_threshold=priority_threshold,
        required_rounds=required_rounds,
        should_stop=should_stop,
        reason=reason,
    )


def _scope_exists(candidate: CandidateSeed, context: AdmissionContext) -> bool:
    for dataset_id in candidate.proposal.dataset_ids:
        available = context.dataset_columns.get(dataset_id)
        if available is None or not set(candidate.proposal.columns).issubset(available):
            return False
    return True


def _policy_and_capability(
    candidate: CandidateSeed, context: AdmissionContext
) -> bool:
    return (
        set(candidate.proposal.dataset_ids).issubset(context.allowed_dataset_ids)
        and candidate.proposal.method_family in context.supported_method_families
    )


def _check(
    name: AdmissionCheckName, passed: bool, passed_code: str, failed_code: str
) -> AdmissionCheck:
    return AdmissionCheck(
        name=name, passed=passed, detail_code=passed_code if passed else failed_code
    )


def _admission_status(
    *,
    scope_exists: bool,
    policy_capability: bool,
    is_novel: bool,
    not_answered: bool,
    not_abandoned: bool,
    falsifiable: bool,
    within_budget: bool,
    multiplicity_ok: bool,
) -> HypothesisStatus:
    if not scope_exists or not within_budget:
        return "rejected_infeasible"
    if (
        not policy_capability
        or not falsifiable
        or not multiplicity_ok
    ):
        return "rejected_policy"
    if not is_novel or not not_answered or not not_abandoned:
        return "rejected_duplicate"
    return "admitted"


def _priority(features: PriorityFeatures, weights: PriorityWeights) -> float:
    value = (
        weights.business_value * features.business_value
        + weights.information_gain_proxy * features.information_gain_proxy
        + weights.novelty * features.novelty
        + weights.coverage_gap * features.coverage_gap
        + weights.feasibility * features.feasibility
        - weights.expected_cost * features.expected_cost
        - weights.redundancy * features.redundancy
        - weights.multiplicity_risk * features.multiplicity_risk
    )
    if not math.isfinite(value):
        raise ValueError("priority calculation must produce a finite value.")
    return round(value, 12)


def _canonical_representatives(
    decisions: Sequence[SchedulingDecision],
    *,
    by_id: Mapping[str, CandidateSeed],
) -> list[SchedulingDecision]:
    """Keep the highest scored eligible candidate per canonical template group."""
    best_by_group: dict[str, SchedulingDecision] = {}
    for decision in decisions:
        group_key = by_id[decision.hypothesis_id].canonical_group_key
        current = best_by_group.get(group_key)
        if current is None or (
            -decision.priority,
            decision.hypothesis_fingerprint,
            decision.hypothesis_id,
        ) < (
            -current.priority,
            current.hypothesis_fingerprint,
            current.hypothesis_id,
        ):
            best_by_group[group_key] = decision
    return list(best_by_group.values())


def _select(
    decisions: Sequence[SchedulingDecision],
    *,
    by_id: Mapping[str, CandidateSeed],
    quotas: Mapping[InsightFamily, int],
    max_batch_size: int,
) -> list[str]:
    ranked = sorted(
        decisions,
        key=lambda item: (-item.priority, item.hypothesis_fingerprint),
    )
    chosen: list[str] = []

    # Mandatory unexplored coverage is the first hard constraint.
    for decision in ranked:
        if by_id[decision.hypothesis_id].mandatory:
            chosen.append(decision.hypothesis_id)
            if len(chosen) == max_batch_size:
                return chosen

    # Reserve remaining slots for family coverage before unconstrained score fill.
    for family in sorted(quotas, key=lambda item: item.value):
        needed = quotas[family]
        already = sum(
            by_id[hypothesis_id].proposal.family == family
            for hypothesis_id in chosen
        )
        for decision in ranked:
            if already >= needed or len(chosen) == max_batch_size:
                break
            if (
                decision.family == family
                and decision.hypothesis_id not in chosen
            ):
                chosen.append(decision.hypothesis_id)
                already += 1

    for decision in ranked:
        if len(chosen) == max_batch_size:
            break
        if decision.hypothesis_id not in chosen:
            chosen.append(decision.hypothesis_id)
    return chosen
