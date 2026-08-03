"""E4a insight reduction is deterministic and fact-level auditable."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from eda_platform.agents.exploration.reducer import reduce_insight
from eda_platform.agents.receipts import build_receipt
from eda_platform.schemas.claims import Claim, ClaimBundle
from eda_platform.schemas.exploration import InsightFamily
from eda_platform.schemas.insights import InsightRecord, TransitionProposal
from eda_platform.schemas.receipts import (
    EvidenceReceipt,
    ReceiptFact,
    ReceiptMethod,
    ReceiptScope,
    ReceiptStatistics,
)

RUN_WITNESS = "dsw1_" + "a" * 32


def _receipt(
    tool_call_id: str,
    *,
    columns: tuple[str, ...] = ("amount",),
    independence_key: str | None = None,
    replication_kind: str | None = None,
    warnings: tuple[str, ...] = (),
    witness: str = RUN_WITNESS,
    filters: str | None = None,
    time_range: str | None = None,
    scope_resolution: str = "explicit",
    statistics: ReceiptStatistics | None = None,
) -> EvidenceReceipt:
    return build_receipt(
        tool_call_id=tool_call_id,
        tool_name="run_sql",
        tool_version="1",
        arguments={"call": tool_call_id},
        raw_output={"rows": []},
        artifact_ids=(),
        result_count=1,
        scope=ReceiptScope(
            dataset_ids=("ds_orders",),
            columns=columns,
            scope_resolution=scope_resolution,  # type: ignore[arg-type]
            filters=filters,
            time_range=time_range,
        ),
        facts=(
            ReceiptFact(
                fact_id="f_n",
                name="row count",
                value=42,
                value_type="count",
                unit="raw",
                support_type="direct",
            ),
        ),
        method=ReceiptMethod(family="sql_aggregation", warnings=warnings),
        statistics=statistics,
        evidence_independence_key=independence_key,
        replication_kind=replication_kind,  # type: ignore[arg-type]
        data_state_witness=witness,
        created_at="2026-08-02T00:00:00Z",
    )


SUPPORT = _receipt("call_support")
SUPPORT_2 = _receipt(
    "call_support_holdout",
    independence_key="holdout-split-2",
    replication_kind="holdout",
)
CONTRA = _receipt("call_contra")
NARROW_CONTRA = _receipt("call_contra_narrow", columns=("other",))
WARNED = _receipt("call_warned", warnings=("assumption violated",))
COMMITTED: dict[str, EvidenceReceipt] = {
    receipt.receipt_id: receipt
    for receipt in (SUPPORT, SUPPORT_2, CONTRA, NARROW_CONTRA, WARNED)
}


def _proposal(**overrides: object) -> TransitionProposal:
    fields: dict[str, object] = {
        "hypothesis_id": "hyp_1",
        "insight_id": "ins_1",
        "family": InsightFamily.DIAGNOSTIC,
        "claim_bundle_id": "clb_1",
        "supporting_receipt_ids": (SUPPORT.receipt_id,),
        "contradicting_receipt_ids": (),
        "limitations": (),
        "proposed_status": "new",
    }
    fields.update(overrides)
    return TransitionProposal.model_validate(fields)


def _bundle(
    proposal: TransitionProposal,
    receipts: dict[str, EvidenceReceipt] = COMMITTED,
    extra_receipt_ids: tuple[str, ...] = (),
) -> ClaimBundle:
    receipt_ids = (
        *proposal.supporting_receipt_ids,
        *proposal.contradicting_receipt_ids,
        *extra_receipt_ids,
    )
    refs = tuple(
        f"{receipt_id}:{receipts[receipt_id].facts[0].fact_id}"
        for receipt_id in receipt_ids
    )
    return ClaimBundle(
        claim_bundle_id=proposal.claim_bundle_id,
        hypothesis_id=proposal.hypothesis_id,
        evidence_lane="exploratory",
        claims=(
            Claim(
                claim_id="claim_1",
                claim_type="comparison",
                claim_text="The evidence has a material direction.",
                support_type="direct",
                evidence_fact_ids=refs,
            ),
        ),
    )


def _reduce(
    proposal: TransitionProposal,
    *,
    prior: InsightRecord | None = None,
    committed: dict[str, EvidenceReceipt] = COMMITTED,
    round_index: int = 0,
    require_typed: bool = False,
    extra_receipt_ids: tuple[str, ...] = (),
    statement: str | None = None,
    rationale: str | None = None,
) -> InsightRecord:
    bundle = _bundle(proposal, committed, extra_receipt_ids)
    return reduce_insight(
        proposal,
        prior=prior,
        committed_receipts=committed,
        admitted_claim_bundles={bundle.claim_bundle_id: bundle},
        expected_witness=RUN_WITNESS,
        round_index=round_index,
        require_typed_hypothesis_outcome=require_typed,
        statement=statement,
        rationale=rationale,
    )


def _typed(
    outcome: str, *, hypothesis_id: str = "hyp_1"
) -> ReceiptStatistics:
    return ReceiptStatistics(
        hypothesis_id=hypothesis_id,
        hypothesis_outcome=outcome,  # type: ignore[arg-type]
        test_name="welch_t",
        test_statistic=2.5,
        p_value=0.01,
        effect_size=0.5,
        ci_low=0.1,
        ci_high=0.9,
        sample_size=40,
    )


def test_a_first_supported_insight_becomes_new_with_fact_level_proof() -> None:
    record = _reduce(_proposal())
    assert record.status == "new"
    assert record.supporting_receipt_ids == (SUPPORT.receipt_id,)
    assert record.proof[0].receipt_id == SUPPORT.receipt_id
    assert record.proof[0].fact_ids == ("f_n",)
    assert record.proof[0].comparison == "supports"
    assert record.statement is None and record.rationale is None


def test_reduce_insight_records_the_proposal_statement_and_rationale() -> None:
    record = _reduce(
        _proposal(),
        statement="Revenue differs by region.",
        rationale="Planted structure.",
    )
    assert record.statement == "Revenue differs by region."
    assert record.rationale == "Planted structure."


def test_a_prior_statement_is_immutable_once_set() -> None:
    first = _reduce(_proposal(), statement="Original claim.", rationale="Original why.")
    second = _reduce(
        _proposal(supporting_receipt_ids=(SUPPORT_2.receipt_id,)),
        prior=first,
        round_index=1,
        statement="Rewritten claim.",
        rationale="Rewritten why.",
    )
    assert second.statement == "Original claim."
    assert second.rationale == "Original why."


def test_an_update_may_backfill_a_missing_prior_statement() -> None:
    first = _reduce(_proposal())
    second = _reduce(
        _proposal(supporting_receipt_ids=(SUPPORT_2.receipt_id,)),
        prior=first,
        round_index=1,
        statement="Backfilled claim.",
        rationale="Backfilled why.",
    )
    assert second.statement == "Backfilled claim."
    assert second.rationale == "Backfilled why."


def test_a_distinct_holdout_receipt_reinforces_an_insight() -> None:
    first = _reduce(_proposal())
    second = _reduce(
        _proposal(supporting_receipt_ids=(SUPPORT_2.receipt_id,)),
        prior=first,
        round_index=1,
    )
    assert second.status == "reinforced"
    assert second.created_round == 0 and second.last_updated_round == 1
    assert len(second.proof) == 2


def test_repeating_the_same_receipt_cannot_fake_reinforcement() -> None:
    first = _reduce(_proposal())
    with pytest.raises(ValueError, match="novel committed evidence"):
        _reduce(_proposal(proposed_status="reinforced"), prior=first, round_index=1)


def test_same_snapshot_support_is_corroboration_not_reinforcement() -> None:
    """R3 (user decision, 2026-08-03): same-snapshot corroboration is not an
    upgrade, but it is not a downgrade either — the prior status survives and
    the caveat is recorded as a limitation instead of a status change."""
    same_snapshot = _receipt(
        "call_same_snapshot",
        independence_key="same-snapshot-query-2",
        replication_kind="same_snapshot_corroboration",
    )
    committed = {**COMMITTED, same_snapshot.receipt_id: same_snapshot}
    first = _reduce(_proposal(), committed=committed)
    second = _reduce(
        _proposal(supporting_receipt_ids=(same_snapshot.receipt_id,)),
        prior=first,
        committed=committed,
        round_index=1,
    )
    assert first.status == "new"
    assert second.status == "new"
    assert second.trust_level == "supported"
    assert any("not an independent replication" in item for item in second.limitations)


def test_same_snapshot_corroboration_keeps_a_reinforced_status() -> None:
    """R3: a prior reinforcement is not walked back by extra same-snapshot work."""
    same_snapshot = _receipt(
        "call_same_snapshot_after_holdout",
        independence_key="same-snapshot-query-3",
        replication_kind="same_snapshot_corroboration",
    )
    committed = {**COMMITTED, same_snapshot.receipt_id: same_snapshot}
    first = _reduce(_proposal(), committed=committed)
    second = _reduce(
        _proposal(supporting_receipt_ids=(SUPPORT_2.receipt_id,)),
        prior=first,
        committed=committed,
        round_index=1,
    )
    assert second.status == "reinforced"
    third = _reduce(
        _proposal(supporting_receipt_ids=(same_snapshot.receipt_id,)),
        prior=second,
        committed=committed,
        round_index=2,
    )
    assert third.status == "reinforced"
    assert third.trust_level == "supported"


def test_an_unadjudicated_bundle_receipt_does_not_join_an_evidence_side() -> None:
    """R1: the bundle keeps every receipt the round produced; only receipts the
    executor typed reach the insight's sides and its proof."""
    typed = _receipt("call_typed_supports", statistics=_typed("supports"))
    unadjudicated = _receipt("call_unadjudicated")
    committed = {
        **COMMITTED,
        typed.receipt_id: typed,
        unadjudicated.receipt_id: unadjudicated,
    }
    record = _reduce(
        _proposal(supporting_receipt_ids=(typed.receipt_id,)),
        committed=committed,
        require_typed=True,
        extra_receipt_ids=(unadjudicated.receipt_id,),
    )
    assert record.status == "new"
    assert record.trust_level == "supported"
    assert record.supporting_receipt_ids == (typed.receipt_id,)
    assert tuple(proof.receipt_id for proof in record.proof) == (typed.receipt_id,)


def test_a_typed_run_rejects_a_contradicting_receipt_on_the_supporting_side() -> None:
    """R2: side placement is not the proposer's to choose."""
    contradicts = _receipt("call_typed_contradicts", statistics=_typed("contradicts"))
    committed = {**COMMITTED, contradicts.receipt_id: contradicts}
    with pytest.raises(ValueError, match="typed outcome"):
        _reduce(
            _proposal(supporting_receipt_ids=(contradicts.receipt_id,)),
            committed=committed,
            require_typed=True,
        )


def test_a_typed_run_rejects_an_unadjudicated_receipt_on_an_evidence_side() -> None:
    with pytest.raises(ValueError, match="typed outcome"):
        _reduce(_proposal(), require_typed=True)


def test_a_typed_run_rejects_a_receipt_adjudicating_another_hypothesis() -> None:
    """R2: the reducer claims to be authoritative, so it checks the binding too."""
    foreign = _receipt(
        "call_typed_foreign",
        statistics=_typed("supports", hypothesis_id="hyp_other"),
    )
    committed = {**COMMITTED, foreign.receipt_id: foreign}
    with pytest.raises(ValueError, match="hypothesis"):
        _reduce(
            _proposal(supporting_receipt_ids=(foreign.receipt_id,)),
            committed=committed,
            require_typed=True,
        )


def test_contradicting_evidence_alone_refutes_despite_agent_proposal() -> None:
    record = _reduce(
        _proposal(
            supporting_receipt_ids=(),
            contradicting_receipt_ids=(CONTRA.receipt_id,),
            proposed_status="reinforced",
        )
    )
    assert record.status == "refuted"
    assert record.proof[0].comparison == "contradicts"


def test_evidence_on_both_sides_is_inconclusive() -> None:
    record = _reduce(
        _proposal(
            supporting_receipt_ids=(SUPPORT.receipt_id,),
            contradicting_receipt_ids=(CONTRA.receipt_id,),
        ),
        round_index=2,
    )
    assert record.status == "inconclusive"
    assert record.trust_level == "contested"


def test_assumption_warning_forces_inconclusive() -> None:
    record = _reduce(_proposal(supporting_receipt_ids=(WARNED.receipt_id,)))
    assert record.status == "inconclusive"
    assert record.trust_level == "unsupported"


def test_no_evidence_or_unpassed_bundle_cannot_create_an_insight() -> None:
    proposal = _proposal(supporting_receipt_ids=())
    with pytest.raises(ValueError, match="has not passed"):
        reduce_insight(
            proposal,
            prior=None,
            committed_receipts=COMMITTED,
            admitted_claim_bundles={},
            expected_witness=RUN_WITNESS,
            round_index=0,
        )


def test_citing_an_uncommitted_receipt_is_rejected() -> None:
    with pytest.raises(ValueError, match="not committed"):
        reduce_insight(
            _proposal(supporting_receipt_ids=("rcpt_" + "f" * 24,)),
            prior=None,
            committed_receipts=COMMITTED,
            admitted_claim_bundles={},
            expected_witness=RUN_WITNESS,
            round_index=0,
        )


def test_a_receipt_failing_its_digest_cannot_support_an_insight() -> None:
    forged = SUPPORT.model_copy(update={"result_count": SUPPORT.result_count + 1})
    with pytest.raises(ValueError, match="digest"):
        _reduce(
            _proposal(),
            committed={**COMMITTED, forged.receipt_id: forged},
        )


def test_claim_bundle_references_must_match_the_transition_assignment() -> None:
    proposal = _proposal()
    mismatched = _bundle(
        proposal.model_copy(
            update={"supporting_receipt_ids": (SUPPORT_2.receipt_id,)}
        )
    )
    with pytest.raises(ValueError, match="must cover"):
        reduce_insight(
            proposal,
            prior=None,
            committed_receipts=COMMITTED,
            admitted_claim_bundles={proposal.claim_bundle_id: mismatched},
            expected_witness=RUN_WITNESS,
            round_index=0,
        )


def test_narrow_contradiction_is_downgraded_to_a_limitation_not_a_refutation() -> None:
    """U7: a narrower-scope contradiction used to raise and kill the whole
    round. It is legitimate exploratory evidence, just not proof against a
    broader prior — so it is dropped from the contradicting side and recorded
    as a limitation instead."""
    first = _reduce(_proposal())
    second = _reduce(
        _proposal(
            supporting_receipt_ids=(),
            contradicting_receipt_ids=(NARROW_CONTRA.receipt_id,),
        ),
        prior=first,
        round_index=1,
    )
    assert second.status == "new"
    assert second.trust_level == "supported"
    assert NARROW_CONTRA.receipt_id not in second.contradicting_receipt_ids
    assert any(
        NARROW_CONTRA.receipt_id in item and "narrower scope" in item
        for item in second.limitations
    )
    assert all(proof.receipt_id != NARROW_CONTRA.receipt_id for proof in second.proof)


def test_filtered_or_time_bounded_contradiction_is_downgraded_not_a_refutation() -> None:
    scoped = _receipt(
        "call_contra_scoped",
        filters="region = 'west'",
        time_range="2026-01-01/2026-01-31",
    )
    committed = {**COMMITTED, scoped.receipt_id: scoped}
    first = _reduce(_proposal(), committed=committed)

    second = _reduce(
        _proposal(
            supporting_receipt_ids=(),
            contradicting_receipt_ids=(scoped.receipt_id,),
        ),
        prior=first,
        committed=committed,
        round_index=1,
    )
    assert second.status == "new"
    assert scoped.receipt_id not in second.contradicting_receipt_ids
    assert any(scoped.receipt_id in item for item in second.limitations)


def test_explicit_scope_downgrade_against_a_whole_dataset_prior_scope() -> None:
    whole = _receipt(
        "call_support_whole",
        columns=(),
        scope_resolution="whole_dataset",
    )
    committed = {**COMMITTED, whole.receipt_id: whole}
    first = _reduce(
        _proposal(supporting_receipt_ids=(whole.receipt_id,)),
        committed=committed,
    )

    second = _reduce(
        _proposal(
            supporting_receipt_ids=(),
            contradicting_receipt_ids=(CONTRA.receipt_id,),
        ),
        prior=first,
        committed=committed,
        round_index=1,
    )
    assert second.status == "new"
    assert CONTRA.receipt_id not in second.contradicting_receipt_ids
    assert any(CONTRA.receipt_id in item for item in second.limitations)


def test_same_scope_contradiction_still_contests_after_the_scope_downgrade_fix() -> None:
    """Control group 1: same-scope contradicting evidence must still land on
    the contradicting side (here making the insight contested, since the
    prior's supporting evidence is sticky) — the fix only changes
    narrower-scope contradictions, not scope-matching ones."""
    first = _reduce(_proposal())
    second = _reduce(
        _proposal(
            supporting_receipt_ids=(),
            contradicting_receipt_ids=(CONTRA.receipt_id,),
        ),
        prior=first,
        round_index=1,
    )
    assert second.status == "inconclusive"
    assert second.trust_level == "contested"
    assert second.contradicting_receipt_ids == (CONTRA.receipt_id,)


def test_narrow_contradiction_on_a_brand_new_insight_is_unaffected() -> None:
    """Control group 2: when there is no prior (first sighting), the scope
    check was already skipped before this fix and must still be skipped —
    narrow-scope contradicting evidence behaves exactly as before."""
    record = _reduce(
        _proposal(
            supporting_receipt_ids=(),
            contradicting_receipt_ids=(NARROW_CONTRA.receipt_id,),
        )
    )
    assert record.status == "refuted"
    assert record.contradicting_receipt_ids == (NARROW_CONTRA.receipt_id,)


def test_receipt_witness_must_match_the_run_witness() -> None:
    drifted = _receipt("call_drifted", witness="dsw1_" + "b" * 32)
    committed = {**COMMITTED, drifted.receipt_id: drifted}
    proposal = _proposal(supporting_receipt_ids=(drifted.receipt_id,))
    bundle = _bundle(proposal, committed)

    with pytest.raises(ValueError, match="data-state witness"):
        reduce_insight(
            proposal,
            prior=None,
            committed_receipts=committed,
            admitted_claim_bundles={bundle.claim_bundle_id: bundle},
            expected_witness=RUN_WITNESS,
            round_index=0,
        )


def test_statistics_only_receipt_cannot_invent_a_fact_level_proof_node() -> None:
    receipt = build_receipt(
        tool_call_id="call_statistics_only",
        tool_name="run_stat_test",
        tool_version="1",
        arguments={"test": "welch_t"},
        raw_output={"p_value": 0.01},
        artifact_ids=(),
        result_count=1,
        scope=ReceiptScope(
            dataset_ids=("ds_orders",),
            columns=("amount",),
            scope_resolution="explicit",
        ),
        facts=(),
        method=ReceiptMethod(family="statistical_test"),
        statistics=ReceiptStatistics(test_name="welch_t", p_value=0.01),
        data_state_witness=RUN_WITNESS,
        created_at="2026-08-02T00:00:00Z",
    )
    proposal = _proposal(
        supporting_receipt_ids=(SUPPORT.receipt_id, receipt.receipt_id)
    )
    bundle = ClaimBundle(
        claim_bundle_id=proposal.claim_bundle_id,
        hypothesis_id=proposal.hypothesis_id,
        evidence_lane="exploratory",
        claims=(
            Claim(
                claim_id="claim_statistics_only",
                claim_type="comparison",
                claim_text="The registered test is material.",
                support_type="direct",
                evidence_fact_ids=(f"{SUPPORT.receipt_id}:f_n",),
                statistics_receipt_ids=(receipt.receipt_id,),
            ),
        ),
    )

    with pytest.raises(ValueError, match="no fact-level proof node"):
        reduce_insight(
            proposal,
            prior=None,
            committed_receipts={**COMMITTED, receipt.receipt_id: receipt},
            admitted_claim_bundles={bundle.claim_bundle_id: bundle},
            expected_witness=RUN_WITNESS,
            round_index=0,
        )


def test_a_refuted_insight_cannot_be_revived_by_support_only() -> None:
    refuted = _reduce(
        _proposal(
            supporting_receipt_ids=(),
            contradicting_receipt_ids=(CONTRA.receipt_id,),
        )
    )
    revived = _reduce(_proposal(), prior=refuted, round_index=1)
    assert revived.status == "inconclusive"
    assert revived.contradicting_receipt_ids == (CONTRA.receipt_id,)


def test_schema_and_reducer_keep_agent_authority_separate() -> None:
    record = _reduce(_proposal(), round_index=3)
    replay = _reduce(_proposal(), round_index=3)
    assert record.model_dump(mode="json") == replay.model_dump(mode="json")
    assert record.trust_level == "supported"
    assert "trust_level" not in TransitionProposal.model_fields
    with pytest.raises(ValidationError):
        record.status = "reinforced"  # type: ignore[misc]


def test_the_same_receipt_cannot_be_on_both_sides() -> None:
    with pytest.raises(ValidationError, match="both"):
        _proposal(
            supporting_receipt_ids=(SUPPORT.receipt_id,),
            contradicting_receipt_ids=(SUPPORT.receipt_id,),
        )


def test_receipt_ids_must_be_unique_on_proposals_and_records() -> None:
    with pytest.raises(ValidationError, match="supporting receipt ids must be unique"):
        _proposal(
            supporting_receipt_ids=(SUPPORT.receipt_id, SUPPORT.receipt_id),
        )

    record = _reduce(_proposal())
    fields = record.model_dump(mode="json")
    fields["supporting_receipt_ids"] = [SUPPORT.receipt_id, SUPPORT.receipt_id]
    with pytest.raises(ValidationError, match="supporting receipt ids must be unique"):
        InsightRecord.model_validate(fields)

    forged = _proposal().model_copy(
        update={"supporting_receipt_ids": (SUPPORT.receipt_id, SUPPORT.receipt_id)}
    )
    bundle = _bundle(_proposal())
    with pytest.raises(ValueError, match="receipt ids must be unique"):
        reduce_insight(
            forged,
            prior=None,
            committed_receipts=COMMITTED,
            admitted_claim_bundles={bundle.claim_bundle_id: bundle},
            expected_witness=RUN_WITNESS,
            round_index=0,
        )
