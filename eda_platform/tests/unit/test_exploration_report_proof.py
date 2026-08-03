"""The report's proof validator against R1 bundles.

Since R1 an admitted claim bundle legitimately carries claims from receipts
that were never adjudicated (a `profile_slice` look-around alongside the
decisive test). Those receipts are not evidence for either side, so they carry
no proof edge — the validator must accept a bundle that is a superset of the
insight's evidence while still proving every adjudicated receipt exactly.
"""

from __future__ import annotations

import pytest

from eda_platform.agents.exploration.workflow import ExplorationWorkflowState
from eda_platform.agents.receipts import build_receipt
from eda_platform.core.exploration_report import _validate_insight_proof
from eda_platform.schemas.claims import Claim, ClaimBundle
from eda_platform.schemas.insights import InsightProof, InsightRecord
from eda_platform.schemas.receipts import (
    EvidenceReceipt,
    ReceiptFact,
    ReceiptMethod,
    ReceiptScope,
    ReceiptStatistics,
)

WITNESS = "dsw1_" + "b" * 60


def _receipt(call: str, fact_id: str, *, adjudicated: bool) -> EvidenceReceipt:
    return build_receipt(
        tool_call_id=call,
        tool_name="run_stat_test" if adjudicated else "profile_slice",
        tool_version="1",
        arguments={},
        raw_output={"value": 1},
        artifact_ids=(),
        result_count=1,
        scope=ReceiptScope(
            dataset_ids=("ds-1",), columns=("region", "revenue"), scope_resolution="explicit"
        ),
        facts=(
            ReceiptFact(
                fact_id=fact_id, name=fact_id, value=1, value_type="number", unit="raw"
            ),
        ),
        method=ReceiptMethod(family="compare_groups" if adjudicated else "slice_profile"),
        statistics=(
            ReceiptStatistics(
                hypothesis_id="hyp_1",
                hypothesis_outcome="supports",
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
            if adjudicated
            else None
        ),
        data_state_witness=WITNESS,
        created_at="2026-08-03T00:00:00Z",
    )


def _claim(receipt: EvidenceReceipt, fact_id: str) -> Claim:
    return Claim(
        claim_id="clm_" + receipt.receipt_id[-12:],
        claim_type="observation",
        claim_text=f"{fact_id}: 1",
        support_type="direct",
        evidence_fact_ids=(f"{receipt.receipt_id}:{fact_id}",),
    )


def _fixture() -> tuple[ExplorationWorkflowState, InsightRecord, tuple[str, ...]]:
    decisive = _receipt("call-decisive", "p_value", adjudicated=True)
    looked_around = _receipt("call-explore", "row_count", adjudicated=False)
    bundle = ClaimBundle(
        claim_bundle_id="cb_mixed",
        hypothesis_id="hyp_1",
        evidence_lane="exploratory",
        claims=(_claim(decisive, "p_value"), _claim(looked_around, "row_count")),
    )
    state = ExplorationWorkflowState(
        committed_receipts={
            decisive.receipt_id: decisive,
            looked_around.receipt_id: looked_around,
        },
        admitted_bundles={bundle.claim_bundle_id: bundle},
    )
    insight = InsightRecord(
        insight_id="ins_1",
        hypothesis_id="hyp_1",
        family="Diagnostic",  # type: ignore[arg-type]
        status="new",
        trust_level="supported",
        claim_bundle_id=bundle.claim_bundle_id,
        supporting_receipt_ids=(decisive.receipt_id,),
        proof=(
            InsightProof(
                receipt_id=decisive.receipt_id,
                comparison="supports",
                fact_ids=("p_value",),
            ),
        ),
        created_round=0,
        last_updated_round=0,
    )
    return state, insight, (decisive.receipt_id, looked_around.receipt_id)


def test_an_unadjudicated_receipt_in_the_bundle_does_not_break_the_proof() -> None:
    state, insight, bundle_ids = _fixture()
    _validate_insight_proof(state, insight, bundle_ids, WITNESS)


def test_a_side_receipt_absent_from_the_bundle_is_still_rejected() -> None:
    state, insight, bundle_ids = _fixture()
    stray = _receipt("call-stray", "stray_fact", adjudicated=True)
    state.committed_receipts[stray.receipt_id] = stray
    forged = insight.model_copy(
        update={"supporting_receipt_ids": (*insight.supporting_receipt_ids, stray.receipt_id)}
    )
    with pytest.raises(ValueError, match="does not match its claim bundle"):
        _validate_insight_proof(state, forged, bundle_ids, WITNESS)


def test_an_adjudicated_receipt_without_its_proof_edge_is_still_rejected() -> None:
    state, insight, bundle_ids = _fixture()
    stripped = insight.model_copy(update={"proof": ()})
    with pytest.raises(ValueError, match="proof"):
        _validate_insight_proof(state, stripped, bundle_ids, WITNESS)
