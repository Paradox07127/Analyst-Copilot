"""E4a policy surface: deterministic candidates, coverage, frontier and scheduling."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
from pydantic import ValidationError

from eda_platform.agents.exploration.candidates import (
    DatasetExplorationProfile,
    candidate_seed,
    compress_canonical_groups,
    coverage_matrix,
    followup_candidate_seed,
    mandatory_probe_seeds,
    materialize_proposal_batch,
    unexplored_coverage,
)
from eda_platform.agents.exploration.frontier import Frontier, FrontierTransitionError
from eda_platform.agents.exploration.scheduler import (
    AdmissionContext,
    CandidateSignals,
    PriorityWeights,
    SchedulerPolicy,
    canonical_query_fingerprint,
    family_quotas_for_level,
    no_new_information_decision,
    schedule_candidates,
)
from eda_platform.schemas.exploration import InsightFamily
from eda_platform.schemas.hypotheses import (
    HypothesisPredicate,
    HypothesisProposal,
    HypothesisProposalBatch,
)


def _proposal(
    *,
    statement: str = "Revenue differs by region.",
    family: InsightFamily = InsightFamily.DIAGNOSTIC,
    method_family: str = "compare_groups",
    columns: tuple[str, ...] = ("region", "revenue"),
    probe_kind: str = "region_difference",
) -> HypothesisProposal:
    return HypothesisProposal(
        statement=statement,
        rationale="The dimension is material to the business metric.",
        expected_evidence="A group comparison with effect size and uncertainty.",
        falsification_conditions=("No material group difference is observed.",),
        family=family,
        method_family=method_family,
        dataset_ids=("ds_sales",),
        columns=columns,
        probe_kind=probe_kind,
        predicate=HypothesisPredicate(
            metric=columns[-1] if columns else probe_kind,
            operator="differs",
            left_operand=columns[0] if columns else None,
        ),
    )


def _policy(*, max_batch_size: int = 3) -> SchedulerPolicy:
    return SchedulerPolicy(
        scoring_policy_version="scheduler-test-v1",
        weights=PriorityWeights(
            business_value=2,
            information_gain_proxy=1,
            novelty=1,
            coverage_gap=2,
            feasibility=1,
            expected_cost=1,
            redundancy=1,
            multiplicity_risk=1,
        ),
        admission_priority=0.0,
        no_information_priority=0.25,
        max_batch_size=max_batch_size,
    )


def _context(**updates: object) -> AdmissionContext:
    base = AdmissionContext(
        dataset_columns={
            "ds_sales": frozenset(
                {"region", "revenue", "satisfaction", "order_date", "units"}
            )
        },
        allowed_dataset_ids=frozenset({"ds_sales"}),
        supported_method_families=frozenset(
            {"compare_groups", "diagnose_missingness", "analyze_time_series", "describe"}
        ),
        historical_hypothesis_fingerprints=frozenset(),
        answered_hypothesis_fingerprints=frozenset(),
        executed_query_fingerprints=frozenset(),
        remaining_cost=1.0,
        family_quota_remaining={
            InsightFamily.DIAGNOSTIC: 1,
            InsightFamily.EXPLORATORY: 1,
        },
        unexplored_coverage_keys=frozenset(),
    )
    return replace(base, **updates)


def test_hypothesis_fingerprint_is_semantic_not_wording() -> None:
    a = candidate_seed(_proposal(statement="Revenue differs by region."), sequence_index=1)
    b = candidate_seed(
        _proposal(statement="Regional revenue is not uniform."), sequence_index=2
    )
    changed_scope = candidate_seed(
        _proposal(columns=("region", "units")), sequence_index=3
    )

    assert a.hypothesis_fingerprint == b.hypothesis_fingerprint
    assert a.hypothesis_fingerprint != changed_scope.hypothesis_fingerprint


def test_opposite_typed_predicates_have_distinct_hypothesis_identities() -> None:
    base = _proposal()
    higher = candidate_seed(
        base.model_copy(
            update={
                "predicate": HypothesisPredicate(
                    metric="revenue",
                    operator="greater_than",
                    left_operand="region-east",
                    right_operand="region-west",
                )
            }
        ),
        sequence_index=1,
    )
    lower = candidate_seed(
        base.model_copy(
            update={
                "predicate": HypothesisPredicate(
                    metric="revenue",
                    operator="less_than",
                    left_operand="region-east",
                    right_operand="region-west",
                )
            }
        ),
        sequence_index=2,
    )

    assert higher.hypothesis_id != lower.hypothesis_id
    assert higher.hypothesis_fingerprint != lower.hypothesis_fingerprint


def test_structured_batch_and_followup_conversion_keep_scoring_system_owned() -> None:
    batch = HypothesisProposalBatch(proposals=(_proposal(),))
    seeds = materialize_proposal_batch(
        batch,
        first_sequence_index=4,
        origin="bootstrap",
    )
    followup = followup_candidate_seed(
        _proposal(statement="Recheck regional revenue."),
        parent=replace(seeds[0], priority=0.8),
        sequence_index=5,
        rounds_since_parent=2,
    )

    assert seeds[0].sequence_index == 4
    assert seeds[0].origin == "bootstrap"
    assert followup.origin == "followup"
    assert followup.priority == pytest.approx(0.8 * 0.9**2)
    assert "priority" not in HypothesisProposal.model_fields


def test_query_fingerprint_matches_the_documented_sql_canonicalization() -> None:
    assert canonical_query_fingerprint(" SELECT  *  FROM sales; ") == (
        canonical_query_fingerprint("select * from sales")
    )
    assert canonical_query_fingerprint("select * from returns") != (
        canonical_query_fingerprint("select * from sales")
    )


def test_124_same_template_candidates_are_compressed_to_one_canonical_group() -> None:
    seeds = [
        candidate_seed(
            _proposal(
                statement=f"Describe column {index}.",
                family=InsightFamily.DESCRIPTIVE,
                method_family="describe",
                columns=(f"column_{index}",),
                probe_kind="column_distribution",
            ),
            sequence_index=index + 1,
        )
        for index in range(124)
    ]
    seeds[73] = replace(seeds[73], priority=0.9)
    compressed = compress_canonical_groups(reversed(seeds), max_per_group=1)

    assert len(compressed.representatives) == 1
    assert len(compressed.groups) == 1
    assert compressed.groups[0].member_count == 124
    assert compressed.groups[0].kept_count == 1
    assert compressed.representatives == (seeds[73],)
    # Input ordering is irrelevant; replay selects the same representative.
    replay = compress_canonical_groups(seeds, max_per_group=1)
    assert compressed == replay


def test_124_same_template_candidates_schedule_only_the_highest_scored_one() -> None:
    seeds = tuple(
        candidate_seed(
            _proposal(
                statement=f"Describe column {index}.",
                family=InsightFamily.DESCRIPTIVE,
                method_family="describe",
                columns=(f"column_{index}",),
                probe_kind="column_distribution",
            ),
            sequence_index=index + 1,
        )
        for index in range(124)
    )
    weights = PriorityWeights(
        business_value=1,
        information_gain_proxy=0,
        novelty=0,
        coverage_gap=0,
        feasibility=0,
        expected_cost=0,
        redundancy=0,
        multiplicity_risk=0,
    )
    policy = SchedulerPolicy(
        scoring_policy_version="scheduler-spam-v1",
        weights=weights,
        admission_priority=0,
        no_information_priority=0.25,
        max_batch_size=124,
    )
    result = schedule_candidates(
        tuple(reversed(seeds)),
        signals={
            seed.hypothesis_id: CandidateSignals(business_value=index / 123)
            for index, seed in enumerate(seeds)
        },
        context=_context(
            dataset_columns={
                "ds_sales": frozenset(f"column_{index}" for index in range(124))
            },
            family_quota_remaining={},
        ),
        policy=policy,
    )

    assert len(result.decisions) == 124
    assert result.chosen_hypothesis_ids == (seeds[-1].hypothesis_id,)


def test_mandatory_probes_cover_eval0_failure_shapes_and_unexplored_is_a_difference() -> None:
    profile = DatasetExplorationProfile(
        dataset_id="ds_sales",
        region_dimensions=("region",),
        metric_columns=("revenue",),
        missing_value_columns=("satisfaction",),
        missingness_group_dimensions=("channel",),
        datetime_columns=("order_date",),
        spike_metric_columns=("revenue",),
    )
    seeds = mandatory_probe_seeds((profile,), first_sequence_index=10)
    by_kind = {seed.proposal.probe_kind: seed for seed in seeds}

    assert set(by_kind) == {"region_difference", "missingness_mechanism", "spike_day"}
    assert all(seed.mandatory for seed in seeds)
    assert by_kind["region_difference"].proposal.columns == ("region", "revenue")
    assert by_kind["missingness_mechanism"].proposal.columns == (
        "satisfaction",
        "channel",
    )
    assert by_kind["missingness_mechanism"].proposal.family is InsightFamily.DIAGNOSTIC
    assert by_kind["spike_day"].proposal.columns == ("order_date", "revenue")
    assert by_kind["spike_day"].proposal.family is InsightFamily.EXPLORATORY

    matrix = coverage_matrix(seeds, executed_coverage_keys=())
    assert [row.coverage_key for row in matrix.rows] == sorted(
        row.coverage_key for row in matrix.rows
    )
    assert all(not row.explored for row in matrix.rows)
    executed = (by_kind["region_difference"].coverage_key,)
    replay = coverage_matrix(seeds, executed_coverage_keys=executed)
    remaining = unexplored_coverage(replay)
    assert {row.probe_kind for row in remaining} == {
        "missingness_mechanism",
        "spike_day",
    }


def test_missingness_without_a_group_dimension_never_claims_a_mechanism() -> None:
    profile = DatasetExplorationProfile(
        dataset_id="ds_sales",
        missing_value_columns=("satisfaction",),
    )

    seeds = mandatory_probe_seeds((profile,))

    assert len(seeds) == 1
    assert seeds[0].proposal.probe_kind == "missingness_rate_scan"
    assert seeds[0].proposal.columns == ("satisfaction",)
    assert "mechanism" not in seeds[0].proposal.statement.casefold()


def test_frontier_enforces_state_machine_and_fingerprint_dedup() -> None:
    seed = candidate_seed(_proposal(), sequence_index=1)
    frontier = Frontier()
    frontier.add(seed)
    duplicate = frontier.add(candidate_seed(_proposal(statement="Rephrased."), sequence_index=2))

    assert duplicate.status == "rejected_duplicate"
    frontier.transition(seed.hypothesis_id, "admitted")
    frontier.transition(seed.hypothesis_id, "running")
    frontier.transition(seed.hypothesis_id, "supported")
    with pytest.raises(FrontierTransitionError):
        frontier.transition(seed.hypothesis_id, "running")


def test_scheduler_records_every_feature_check_and_choice_deterministically() -> None:
    seed = candidate_seed(_proposal(), sequence_index=1)
    signals = {
        seed.hypothesis_id: CandidateSignals(
            business_value=0.8,
            information_gain_proxy=0.6,
            expected_cost=0.1,
            multiplicity_risk=0.2,
        )
    }
    context = _context(
        unexplored_coverage_keys=frozenset({seed.coverage_key})
    )

    first = schedule_candidates((seed,), signals=signals, context=context, policy=_policy())
    second = schedule_candidates((seed,), signals=signals, context=context, policy=_policy())

    assert first == second
    assert first.chosen_hypothesis_ids == (seed.hypothesis_id,)
    decision = first.decisions[0]
    assert decision.chosen
    assert decision.scoring_policy_version == "scheduler-test-v1"
    assert tuple(check.name for check in decision.admission_checks) == (
        "scope_exists",
        "policy_and_capability",
        "novelty",
        "not_already_answered",
        "falsifiable",
        "within_remaining_budget",
        "coverage_and_quota",
        "multiplicity_acceptable",
    )
    assert decision.priority_features.coverage_gap == 1.0
    assert decision.priority == pytest.approx(5.9)


def test_scheduler_rejects_canonical_duplicate_before_selection() -> None:
    seed = candidate_seed(_proposal(), sequence_index=1)
    result = schedule_candidates(
        (seed,),
        signals={seed.hypothesis_id: CandidateSignals()},
        context=_context(
            historical_hypothesis_fingerprints=frozenset(
                {seed.hypothesis_fingerprint}
            )
        ),
        policy=_policy(),
    )

    assert result.chosen_hypothesis_ids == ()
    assert result.decisions[0].status == "rejected_duplicate"
    novelty = next(
        check for check in result.decisions[0].admission_checks if check.name == "novelty"
    )
    assert not novelty.passed


def test_multiplicity_failure_cannot_be_selected_but_filled_quota_is_not_a_ban() -> None:
    no_quota = candidate_seed(
        _proposal(
            family=InsightFamily.DESCRIPTIVE,
            method_family="describe",
            columns=("revenue",),
            probe_kind="distribution",
        ),
        sequence_index=1,
    )
    risky = candidate_seed(_proposal(), sequence_index=2)
    result = schedule_candidates(
        (no_quota, risky),
        signals={
            no_quota.hypothesis_id: CandidateSignals(business_value=1.0),
            risky.hypothesis_id: CandidateSignals(
                business_value=1.0, multiplicity_risk=0.8
            ),
        },
        context=_context(
            family_quota_remaining={},
            unexplored_coverage_keys=frozenset(),
        ),
        policy=_policy().model_copy(update={"max_multiplicity_risk": 0.5}),
    )

    assert result.chosen_hypothesis_ids == (no_quota.hypothesis_id,)
    by_id = {decision.hypothesis_id: decision for decision in result.decisions}
    assert by_id[no_quota.hypothesis_id].status == "admitted"
    assert by_id[no_quota.hypothesis_id].quota_deferred
    assert by_id[risky.hypothesis_id].status == "rejected_policy"
    assert by_id[risky.hypothesis_id].quota_deferred


def test_scheduler_rejects_an_already_executed_canonical_query() -> None:
    seed = candidate_seed(_proposal(), sequence_index=1)
    fingerprint = canonical_query_fingerprint("select region, sum(revenue) from sales")
    result = schedule_candidates(
        (seed,),
        signals={
            seed.hypothesis_id: CandidateSignals(query_fingerprint=fingerprint)
        },
        context=_context(executed_query_fingerprints=frozenset({fingerprint})),
        policy=_policy(),
    )

    assert result.decisions[0].status == "rejected_duplicate"
    novelty = next(
        check for check in result.decisions[0].admission_checks if check.name == "novelty"
    )
    assert novelty.detail_code == "canonical_query_duplicate"


def test_family_quota_reserves_slots_before_priority_fill() -> None:
    descriptive = [
        candidate_seed(
            _proposal(
                statement=f"Describe metric {index}",
                family=InsightFamily.DESCRIPTIVE,
                method_family="describe",
                columns=("revenue",),
                probe_kind=f"distribution_{index}",
            ),
            sequence_index=index + 1,
        )
        for index in range(3)
    ]
    diagnostic = candidate_seed(_proposal(), sequence_index=10)
    exploratory = candidate_seed(
        _proposal(
            statement="Why is satisfaction missing?",
            family=InsightFamily.EXPLORATORY,
            method_family="diagnose_missingness",
            columns=("satisfaction",),
            probe_kind="missingness_mechanism",
        ),
        sequence_index=11,
    )
    seeds = (*descriptive, diagnostic, exploratory)
    signals = {
        seed.hypothesis_id: CandidateSignals(business_value=1.0 if seed in descriptive else 0.1)
        for seed in seeds
    }
    result = schedule_candidates(
        seeds,
        signals=signals,
        context=_context(),
        policy=_policy(max_batch_size=3),
    )
    chosen = {
        decision.family
        for decision in result.decisions
        if decision.chosen
    }

    assert chosen == {
        InsightFamily.DESCRIPTIVE,
        InsightFamily.DIAGNOSTIC,
        InsightFamily.EXPLORATORY,
    }


def test_thinking_level_family_quotas_pin_quick_vs_standard_deep() -> None:
    quick = family_quotas_for_level(
        "quick",
        (InsightFamily.DIAGNOSTIC, InsightFamily.EXPLORATORY),
    )
    standard = family_quotas_for_level(
        "standard",
        tuple(InsightFamily),
    )

    assert quick == {
        InsightFamily.DIAGNOSTIC: 1,
        InsightFamily.EXPLORATORY: 1,
    }
    assert set(standard) == set(InsightFamily)
    with pytest.raises(ValueError, match="2 or 3"):
        family_quotas_for_level("quick", (InsightFamily.DIAGNOSTIC,))
    with pytest.raises(ValueError, match="all six"):
        family_quotas_for_level("deep", (InsightFamily.DIAGNOSTIC,))


def test_mandatory_unexplored_probe_is_selected_before_optional_higher_score() -> None:
    mandatory = replace(candidate_seed(_proposal(), sequence_index=1), mandatory=True)
    optional = candidate_seed(
        _proposal(
            family=InsightFamily.DESCRIPTIVE,
            method_family="describe",
            columns=("revenue",),
            probe_kind="distribution",
        ),
        sequence_index=2,
    )
    signals = {
        mandatory.hypothesis_id: CandidateSignals(business_value=0.0),
        optional.hypothesis_id: CandidateSignals(business_value=1.0),
    }
    result = schedule_candidates(
        (optional, mandatory),
        signals=signals,
        context=_context(
            family_quota_remaining={},
            unexplored_coverage_keys=frozenset({mandatory.coverage_key}),
        ),
        policy=_policy(max_batch_size=1),
    )
    assert result.chosen_hypothesis_ids == (mandatory.hypothesis_id,)


@pytest.mark.parametrize(
    ("rounds", "highest", "expected"),
    [(1, 0.1, False), (2, 0.25, False), (2, 0.249, True)],
)
def test_no_new_information_is_a_replayable_explicit_decision(
    rounds: int, highest: float, expected: bool
) -> None:
    decision = no_new_information_decision(
        consecutive_no_information_rounds=rounds,
        highest_frontier_priority=highest,
        priority_threshold=0.25,
        required_rounds=2,
    )

    assert decision.should_stop is expected
    assert decision.consecutive_no_information_rounds == rounds
    assert decision.highest_frontier_priority == highest
    assert decision.priority_threshold == 0.25


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_scheduler_rejects_non_finite_weights_thresholds_and_context(bad: float) -> None:
    weight_fields = _policy().weights.model_dump()
    weight_fields["business_value"] = bad
    with pytest.raises(ValidationError, match="finite"):
        PriorityWeights.model_validate(weight_fields)

    with pytest.raises(ValidationError, match="finite"):
        SchedulerPolicy.model_validate(
            {**_policy().model_dump(), "admission_priority": bad}
        )

    seed = candidate_seed(_proposal(), sequence_index=1)
    with pytest.raises(ValueError, match="remaining_cost must be finite"):
        schedule_candidates(
            (seed,),
            signals={seed.hypothesis_id: CandidateSignals()},
            context=_context(remaining_cost=bad),
            policy=_policy(),
        )

    with pytest.raises(ValueError, match="priority_threshold must be finite"):
        no_new_information_decision(
            consecutive_no_information_rounds=2,
            highest_frontier_priority=0.1,
            priority_threshold=bad,
        )
