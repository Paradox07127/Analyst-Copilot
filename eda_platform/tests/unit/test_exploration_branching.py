"""E6 branch-abandonment constraint derivation: deterministic and issuer-replayable."""

from __future__ import annotations

from typing import Literal

import pytest

from eda_platform.agents.exploration.branching import (
    bundle_hypotheses_from_events,
    derive_branch_constraints,
    receipt_hypothesis_id,
)
from eda_platform.agents.exploration.candidates import CandidateSeed, candidate_seed
from eda_platform.agents.receipts import build_receipt
from eda_platform.core.claim_gates import GateReport, GateVerdict
from eda_platform.schemas.exploration import (
    BranchConstraint,
    GateVerdictEvent,
    InsightFamily,
    ReceiptCommittedEvent,
    RoundSettledEvent,
    RoundStartedEvent,
)
from eda_platform.schemas.hypotheses import HypothesisPredicate, HypothesisProposal
from eda_platform.schemas.receipts import (
    EvidenceReceipt,
    ReceiptExecution,
    ReceiptFact,
    ReceiptMethod,
    ReceiptScope,
    ReceiptStatistics,
)

_WITNESS = "dsw1_branch_test"
_XPL = "xpl_branch"


def _proposal(metric: str) -> HypothesisProposal:
    return HypothesisProposal(
        statement=f"Does {metric} differ by region?",
        rationale="branch test",
        expected_evidence="an observation",
        falsification_conditions=("no difference",),
        family=InsightFamily.DIAGNOSTIC,
        method_family="compare_groups",
        dataset_ids=("ds-1",),
        columns=("region", metric),
        probe_kind="region_difference",
        predicate=HypothesisPredicate(
            metric=metric, operator="differs", left_operand="region"
        ),
    )


def _candidate(metric: str, index: int) -> CandidateSeed:
    return candidate_seed(_proposal(metric), sequence_index=index)


def _receipt(
    candidate: CandidateSeed,
    *,
    round_index: int,
    outcome: Literal["supports", "contradicts"] | None,
    with_statistics: bool = True,
    call: str = "call-1",
) -> EvidenceReceipt:
    statistics = None
    if with_statistics:
        statistics = ReceiptStatistics(
            hypothesis_id=candidate.hypothesis_id,
            hypothesis_outcome=outcome,
            test_name="independent_t_test",
            test_statistic=2.5,
            p_value=0.01,
            adjusted_p_value=0.01,
            effect_size=0.5,
            ci_low=0.1,
            ci_high=0.9,
            sample_size=20,
            sequence_index=1,
        )
    return build_receipt(
        tool_call_id=call,
        tool_name="run_stat_test",
        tool_version="1",
        arguments={},
        raw_output={"value": 1},
        artifact_ids=(),
        result_count=1,
        scope=ReceiptScope(
            dataset_ids=("ds-1",),
            columns=("region", "revenue"),
            scope_resolution="explicit",
        ),
        facts=(
            ReceiptFact(
                fact_id="difference",
                name="difference",
                value=1,
                value_type="number",
                unit="raw",
            ),
        ),
        method=ReceiptMethod(family="compare_groups"),
        statistics=statistics,
        data_state_witness=_WITNESS,
        created_at="2026-08-03T00:00:00Z",
        execution=ReceiptExecution(
            run_id=(
                f"{_XPL}:round:{round_index}:hypothesis:"
                f"{candidate.hypothesis_id}:execute_probes"
            ),
            provider_call_id=call,
            logical_step_id=f"step-{call}",
            attempt_epoch=0,
            sequence_index=1,
        ),
    )


def _rejected_report(bundle_id: str) -> GateReport:
    return GateReport(
        claim_bundle_id=bundle_id,
        claim_bundle_digest="d" * 16,
        run_witness=_WITNESS,
        passed=False,
        verdicts=(
            GateVerdict(gate="structure", passed=True),
            GateVerdict(gate="reachability", passed=False),
        ),
        health_score=0.5,
    )


def test_receipt_hypothesis_id_parses_executor_run_id() -> None:
    candidate = _candidate("revenue", 1)
    receipt = _receipt(candidate, round_index=0, outcome="supports")
    assert receipt_hypothesis_id(receipt) == candidate.hypothesis_id
    unbound = receipt.model_copy(
        update={
            "execution": ReceiptExecution(
                run_id="free-form-run",
                provider_call_id="call-1",
                logical_step_id="step-1",
            )
        }
    )
    with pytest.raises(ValueError, match="hypothesis"):
        receipt_hypothesis_id(unbound)


def test_derive_constraints_covers_refuted_inconclusive_and_gate_rejected() -> None:
    refuted = _candidate("revenue", 1)
    inconclusive = _candidate("cost", 2)
    gated = _candidate("margin", 3)
    candidates = {
        item.hypothesis_id: item for item in (refuted, inconclusive, gated)
    }
    receipts = {
        receipt.receipt_id: receipt
        for receipt in (
            _receipt(refuted, round_index=0, outcome="contradicts", call="call-r"),
            _receipt(
                inconclusive,
                round_index=0,
                outcome=None,
                with_statistics=False,
                call="call-i",
            ),
            _receipt(gated, round_index=0, outcome="supports", call="call-g"),
        )
    }
    constraints = derive_branch_constraints(
        candidates=candidates,
        committed_receipts=receipts,
        gate_reports={"cb_1": _rejected_report("cb_1")},
        bundle_hypotheses={"cb_1": gated.hypothesis_id},
        prior=(),
    )
    by_reason = {item.reason: item for item in constraints}
    assert set(by_reason) == {"refuted", "inconclusive", "gate_rejected"}
    assert by_reason["refuted"].hypothesis_fingerprint == refuted.hypothesis_fingerprint
    assert by_reason["refuted"].coverage_key == refuted.coverage_key
    assert by_reason["gate_rejected"].detail_code == "cb_1:reachability"
    assert constraints == tuple(
        sorted(
            constraints,
            key=lambda item: (item.hypothesis_fingerprint, item.reason, item.detail_code),
        )
    )
    # The recomputation is deterministic and idempotent.
    assert constraints == derive_branch_constraints(
        candidates=candidates,
        committed_receipts=receipts,
        gate_reports={"cb_1": _rejected_report("cb_1")},
        bundle_hypotheses={"cb_1": gated.hypothesis_id},
        prior=(),
    )


def test_derive_constraints_subtracts_prior_and_fails_on_unknown_hypothesis() -> None:
    refuted = _candidate("revenue", 1)
    receipts = {
        receipt.receipt_id: receipt
        for receipt in (_receipt(refuted, round_index=0, outcome="contradicts"),)
    }
    prior = derive_branch_constraints(
        candidates={refuted.hypothesis_id: refuted},
        committed_receipts=receipts,
        gate_reports={},
        bundle_hypotheses={},
        prior=(),
    )
    assert prior and prior[0].reason == "refuted"
    again = derive_branch_constraints(
        candidates={refuted.hypothesis_id: refuted},
        committed_receipts=receipts,
        gate_reports={},
        bundle_hypotheses={},
        prior=prior,
    )
    assert again == ()
    with pytest.raises(ValueError, match="candidate"):
        derive_branch_constraints(
            candidates={},
            committed_receipts=receipts,
            gate_reports={},
            bundle_hypotheses={},
            prior=(),
        )


def test_bundle_hypotheses_follow_deterministic_gate_order() -> None:
    first = _candidate("revenue", 1)
    second = _candidate("cost", 2)
    receipts = {
        receipt.receipt_id: receipt
        for receipt in (
            _receipt(first, round_index=0, outcome="supports", call="call-1"),
            _receipt(second, round_index=0, outcome="contradicts", call="call-2"),
        )
    }
    receipt_ids = list(receipts)
    events = (
        RoundStartedEvent(seq=1, exploration_id=_XPL, round_index=0),
        ReceiptCommittedEvent(
            seq=2,
            exploration_id=_XPL,
            logical_step_id="step-call-1",
            receipt_id=receipt_ids[0],
        ),
        ReceiptCommittedEvent(
            seq=3,
            exploration_id=_XPL,
            logical_step_id="step-call-2",
            receipt_id=receipt_ids[1],
        ),
        GateVerdictEvent(
            seq=4, exploration_id=_XPL, claim_bundle_id="cb_first", verdict="passed"
        ),
        GateVerdictEvent(
            seq=5, exploration_id=_XPL, claim_bundle_id="cb_second", verdict="rejected"
        ),
        RoundSettledEvent(seq=6, exploration_id=_XPL, round_index=0, progress=True),
    )
    mapping = bundle_hypotheses_from_events(events, receipts)
    assert mapping == {
        "cb_first": receipt_hypothesis_id(receipts[receipt_ids[0]]),
        "cb_second": receipt_hypothesis_id(receipts[receipt_ids[1]]),
    }


def test_constraint_reasons_gate_admission() -> None:
    from eda_platform.agents.exploration.scheduler import (
        AdmissionContext,
        CandidateSignals,
        PriorityWeights,
        SchedulerPolicy,
        schedule_candidates,
    )

    candidate = _candidate("revenue", 1)
    soft_candidate = _candidate("cost", 2)

    def context(constraints: tuple[BranchConstraint, ...]) -> AdmissionContext:
        return AdmissionContext(
            dataset_columns={"ds-1": frozenset({"region", "revenue", "cost"})},
            allowed_dataset_ids=frozenset({"ds-1"}),
            supported_method_families=frozenset({"compare_groups"}),
            historical_hypothesis_fingerprints=frozenset(),
            answered_hypothesis_fingerprints=frozenset(),
            executed_query_fingerprints=frozenset(),
            remaining_cost=1.0,
            family_quota_remaining={InsightFamily.DIAGNOSTIC: 2},
            unexplored_coverage_keys=frozenset(
                {candidate.coverage_key, soft_candidate.coverage_key}
            ),
            abandoned_constraints=constraints,
        )

    policy = SchedulerPolicy(
        scoring_policy_version="score-v1",
        weights=PriorityWeights(
            business_value=1.0,
            information_gain_proxy=1.0,
            novelty=1.0,
            coverage_gap=1.0,
            feasibility=1.0,
            expected_cost=1.0,
            redundancy=1.0,
            multiplicity_risk=1.0,
        ),
        admission_priority=0.0,
        no_information_priority=0.0,
        max_batch_size=4,
    )
    signals = {
        candidate.hypothesis_id: CandidateSignals(business_value=1.0),
        soft_candidate.hypothesis_id: CandidateSignals(business_value=1.0),
    }
    hard = BranchConstraint(
        hypothesis_fingerprint=candidate.hypothesis_fingerprint,
        coverage_key=candidate.coverage_key,
        family=InsightFamily.DIAGNOSTIC,
        reason="refuted",
        detail_code="rcpt_1",
    )
    soft = BranchConstraint(
        hypothesis_fingerprint=soft_candidate.hypothesis_fingerprint,
        coverage_key=soft_candidate.coverage_key,
        family=InsightFamily.DIAGNOSTIC,
        reason="inconclusive",
        detail_code="rcpt_2",
    )
    result = schedule_candidates(
        (candidate, soft_candidate),
        signals=signals,
        context=context((hard, soft)),
        policy=policy,
    )
    decisions = {item.hypothesis_id: item for item in result.decisions}
    blocked = decisions[candidate.hypothesis_id]
    assert blocked.status == "rejected_duplicate"
    checks = {check.name: check for check in blocked.admission_checks}
    assert checks["not_previously_abandoned"].passed is False
    assert checks["not_previously_abandoned"].detail_code == "abandoned_refuted"
    allowed = decisions[soft_candidate.hypothesis_id]
    assert allowed.status == "admitted"
    allowed_checks = {check.name: check for check in allowed.admission_checks}
    assert allowed_checks["not_previously_abandoned"].passed is True
