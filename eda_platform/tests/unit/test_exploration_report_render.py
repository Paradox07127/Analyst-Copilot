"""Human-readable per-insight rendering (statement-led blocks, no claim dump)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eda_platform.agents.exploration.workflow import ExplorationWorkflowState
from eda_platform.agents.receipts import build_receipt
from eda_platform.core.claim_gates import GateReport, claim_bundle_digest
from eda_platform.core.exploration_report import render_exploration_report
from eda_platform.schemas.claims import Claim, ClaimBundle
from eda_platform.schemas.insights import InsightProof, InsightRecord
from eda_platform.schemas.receipts import (
    EvidenceReceipt,
    ReceiptFact,
    ReceiptMethod,
    ReceiptScope,
    ReceiptStatistics,
)

WITNESS = "dsw1_" + "c" * 60

_SEED6_PROJECTION = (
    Path(__file__).resolve().parents[3]
    / "output/e4a/calibration/deepseek-v4-flash/workspace/exploration-eval"
    / "e4a-deep-seed6-x8/projection.json"
)


def _receipt(call: str, *, with_statistics: bool = True) -> EvidenceReceipt:
    return build_receipt(
        tool_call_id=call,
        tool_name="run_stat_test",
        tool_version="1",
        arguments={"call": call},
        raw_output={"value": 1},
        artifact_ids=(),
        result_count=1,
        scope=ReceiptScope(
            dataset_ids=("ds-1",), columns=("region", "revenue"), scope_resolution="explicit"
        ),
        facts=(
            ReceiptFact(
                fact_id="p_value", name="p_value", value=1, value_type="number", unit="raw"
            ),
        ),
        method=ReceiptMethod(family="compare_groups"),
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
            if with_statistics
            else None
        ),
        data_state_witness=WITNESS,
        created_at="2026-08-03T00:00:00Z",
    )


def _insight_fields(receipts: tuple[EvidenceReceipt, ...]) -> dict[str, object]:
    return {
        "insight_id": "ins_1",
        "hypothesis_id": "hyp_1",
        "family": "Diagnostic",
        "status": "new",
        "trust_level": "supported",
        "claim_bundle_id": "cb_1",
        "supporting_receipt_ids": tuple(receipt.receipt_id for receipt in receipts),
        "proof": tuple(
            InsightProof(
                receipt_id=receipt.receipt_id,
                comparison="supports",
                fact_ids=("p_value",),
            )
            for receipt in receipts
        ),
        "created_round": 0,
        "last_updated_round": 0,
    }


def _state(insight: InsightRecord, receipts: tuple[EvidenceReceipt, ...]) -> ExplorationWorkflowState:
    bundle = ClaimBundle(
        claim_bundle_id="cb_1",
        hypothesis_id="hyp_1",
        evidence_lane="exploratory",
        claims=tuple(
            Claim(
                claim_id=f"clm_{index}",
                claim_type="observation",
                claim_text=f"legacy claim dump text {index}: 1",
                support_type="direct",
                evidence_fact_ids=(f"{receipt.receipt_id}:p_value",),
            )
            for index, receipt in enumerate(receipts)
        ),
    )
    report = GateReport(
        claim_bundle_id="cb_1",
        claim_bundle_digest=claim_bundle_digest(bundle),
        run_witness=WITNESS,
        passed=True,
        verdicts=(),
        health_score=1.0,
    )
    return ExplorationWorkflowState(
        committed_receipts={receipt.receipt_id: receipt for receipt in receipts},
        gate_reports={"cb_1": report},
        admitted_bundles={"cb_1": bundle},
        insights={insight.insight_id: insight},
    )


def _render(state: ExplorationWorkflowState) -> str:
    return render_exploration_report(
        state,
        run_metadata={
            "exploration_id": "xpl-render",
            "policy_fingerprint": "policy-v1",
            "witness": WITNESS,
        },
        coverage_targets=(),
        budget_summary={"rounds": 1},
        stop_reason="completed",
    ).markdown


def test_insight_renders_a_statement_led_block_not_a_claim_dump() -> None:
    receipt = _receipt("call-1")
    insight = InsightRecord.model_validate(
        {
            **_insight_fields((receipt,)),
            "statement": "Revenue differs by region.",
            "rationale": "Planted regional structure.",
        }
    )
    markdown = _render(_state(insight, (receipt,)))
    assert "- **Revenue differs by region.** — new/supported" in markdown
    assert (
        f"  - evidence: 1 supporting, 0 contradicting receipt(s); "
        f"{receipt.receipt_id} (supports)" in markdown
    )
    assert "p_value=0.01" in markdown
    assert "effect_size=0.5" in markdown
    assert "sample_size=20" in markdown
    assert "  - why: Planted regional structure." in markdown
    assert "legacy claim dump text" not in markdown
    assert "p_value/" not in markdown  # no fact-id list dump


def test_legacy_record_without_statement_renders_the_fallback() -> None:
    receipt = _receipt("call-1")
    insight = InsightRecord.model_validate(_insight_fields((receipt,)))
    assert insight.statement is None and insight.rationale is None
    markdown = _render(_state(insight, (receipt,)))
    assert "- **(no statement recorded)** — new/supported" in markdown
    assert "why:" not in markdown


def test_key_statistics_come_from_at_most_two_receipts() -> None:
    receipts = tuple(_receipt(f"call-{index}") for index in range(3))
    insight = InsightRecord.model_validate(
        {**_insight_fields(receipts), "statement": "Three receipts, two stat entries."}
    )
    markdown = _render(_state(insight, receipts))
    stats_lines = [line for line in markdown.splitlines() if "key stats:" in line]
    assert len(stats_lines) == 1
    assert stats_lines[0].count("p_value=0.01") == 2


def test_receipt_without_statistics_renders_no_stats_line() -> None:
    receipt = _receipt("call-1", with_statistics=False)
    insight = InsightRecord.model_validate(
        {**_insight_fields((receipt,)), "statement": "No stats recorded."}
    )
    markdown = _render(_state(insight, (receipt,)))
    assert "key stats:" not in markdown


@pytest.mark.skipif(
    not _SEED6_PROJECTION.exists(), reason="seed-6 calibration output not present"
)
def test_seed6_projection_insight_records_still_validate() -> None:
    records = json.loads(_SEED6_PROJECTION.read_text(encoding="utf-8"))["insight_records"]
    assert records
    for raw in records:
        record = InsightRecord.model_validate(raw)
        assert record.statement is None and record.rationale is None
