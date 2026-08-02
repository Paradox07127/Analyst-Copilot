"""Deterministic claim-report renderer (plan R3.4, §4.6, E3-B).

The renderer is the no-LLM fallback: when the SYNTHESIZE polish step is
unavailable, passed ClaimBundles must still ship as a fixed-structure report,
byte-identical across renders, with a final numeric rescan that rejects any
number the renderer itself would introduce.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from eda_platform.agents.receipts import build_receipt
from eda_platform.core import claim_renderer
from eda_platform.core.claim_gates import GateReport, run_claim_gates
from eda_platform.core.claim_renderer import (
    RenderedNumberLeakError,
    assert_rendered_numbers,
    numeric_pool_from_texts,
    render_claim_report,
)
from eda_platform.schemas.claims import Claim, ClaimBundle
from eda_platform.schemas.receipts import (
    EvidenceReceipt,
    ReceiptFact,
    ReceiptMethod,
    ReceiptScope,
    ReceiptStatistics,
)

RUN_WITNESS = "dsw1_" + "a" * 64


def _fact(fact_id: str, value: object, value_type: str = "number") -> ReceiptFact:
    return ReceiptFact(
        fact_id=fact_id,
        name=fact_id,
        value=value,  # type: ignore[arg-type]
        value_type=value_type,  # type: ignore[arg-type]
        unit="raw",
        support_type="direct",
    )


def _receipt(
    *,
    tool_call_id: str,
    facts: tuple[ReceiptFact, ...],
    tool_name: str = "run_sql",
    statistics: ReceiptStatistics | None = None,
    method_family: str = "sql_aggregation",
) -> EvidenceReceipt:
    return build_receipt(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        tool_version="1",
        arguments={"call": tool_call_id},
        raw_output={"rows": []},
        artifact_ids=(),
        result_count=1,
        scope=ReceiptScope(dataset_ids=("ds_orders",), columns=("amount",)),
        facts=facts,
        derivations=(),
        method=ReceiptMethod(family=method_family),
        statistics=statistics,
        data_state_witness=RUN_WITNESS,
        created_at="2026-08-02T00:00:00Z",
    )


R_MAIN = _receipt(
    tool_call_id="call_main",
    facts=(_fact("f_n", 42, "count"), _fact("f_pct", 12.5, "percent")),
)
R_SECOND = _receipt(tool_call_id="call_second", facts=(_fact("f_gaps", 7, "count"),))
R_STAT_BAD = _receipt(
    tool_call_id="call_stat_bad",
    facts=(_fact("f_p2", 0.003, "number"),),
    tool_name="run_stat_test",
    method_family="shapiro_wilk",
    statistics=ReceiptStatistics(test_name="shapiro_wilk", p_value=0.003),
)
COMMITTED: Mapping[str, EvidenceReceipt] = {
    r.receipt_id: r for r in (R_MAIN, R_SECOND, R_STAT_BAD)
}

METADATA = {
    "exploration_id": "exp_demo",
    "policy_fingerprint": "pfp_a1b2c3",
    "witness": RUN_WITNESS,
}


def _claim(**overrides: object) -> Claim:
    fields: dict[str, object] = {
        "claim_id": "c1",
        "claim_type": "observation",
        "claim_text": "There are 42 orders.",
        "support_type": "direct",
        "evidence_fact_ids": (f"{R_MAIN.receipt_id}:f_n",),
    }
    fields.update(overrides)
    return Claim.model_validate(fields)


def _bundle(*claims: Claim, **overrides: object) -> ClaimBundle:
    fields: dict[str, object] = {
        "claim_bundle_id": "clb_explore",
        "hypothesis_id": "hyp_1",
        "evidence_lane": "exploratory",
        "claims": claims or (_claim(),),
    }
    fields.update(overrides)
    return ClaimBundle.model_validate(fields)


def _gated(bundle: ClaimBundle) -> tuple[ClaimBundle, GateReport]:
    report = run_claim_gates(
        bundle, committed_receipts=COMMITTED, run_witness=RUN_WITNESS
    )
    return bundle, report


EXPLORE = _gated(
    _bundle(
        _claim(),
        _claim(
            claim_id="c2",
            claim_text="12.5% of users are affected.",
            evidence_fact_ids=(f"{R_MAIN.receipt_id}:f_pct",),
        ),
    )
)
CONFIRM = _gated(
    _bundle(
        _claim(
            claim_text="A significant difference was observed.",
            evidence_fact_ids=(f"{R_STAT_BAD.receipt_id}:f_p2",),
            statistics_receipt_ids=(R_STAT_BAD.receipt_id,),
        ),
        claim_bundle_id="clb_confirm",
        evidence_lane="confirmatory",
    )
)
REJECTED = _gated(
    _bundle(
        _claim(claim_text="There are 43 orders."),
        claim_bundle_id="clb_zz_rejected",
    )
)
TWO_RECEIPTS = _gated(
    _bundle(
        _claim(
            claim_text="There are 42 orders and 7 gaps.",
            evidence_fact_ids=(
                f"{R_MAIN.receipt_id}:f_n",
                f"{R_SECOND.receipt_id}:f_gaps",
            ),
        ),
        claim_bundle_id="clb_trail",
    )
)

ALL_PAIRS = [EXPLORE, REJECTED, CONFIRM]


def test_fixture_gate_reports_have_the_expected_verdicts() -> None:
    assert EXPLORE[1].passed and CONFIRM[1].passed and TWO_RECEIPTS[1].passed
    assert not REJECTED[1].passed


def test_sections_are_fixed_and_claims_land_in_their_lanes() -> None:
    markdown = render_claim_report(ALL_PAIRS, run_metadata=METADATA).markdown
    positions = [
        markdown.index(section)
        for section in (
            "# Exploration findings",
            "## Confirmed findings",
            "## Exploratory observations",
            "## Statistical caveats",
            "## Evidence trail",
            "## Not rendered",
        )
    ]
    assert positions == sorted(positions)
    assert "- exploration_id: exp_demo" in markdown
    assert "- policy_fingerprint: pfp_a1b2c3" in markdown
    assert "- witness: dsw1_aaaaaaaaaaaa…" in markdown
    confirmed = markdown.index("## Confirmed findings")
    exploratory = markdown.index("## Exploratory observations")
    assert (
        markdown.index("- (clb_confirm/c1) A significant difference was observed.")
        < exploratory
    )
    assert markdown.index("- (clb_explore/c1) There are 42 orders.") > exploratory
    assert "- (clb_explore/c2) 12.5% of users are affected." in markdown
    assert confirmed < exploratory


def test_rejected_bundles_are_counted_but_never_rendered() -> None:
    rendered = render_claim_report(ALL_PAIRS, run_metadata=METADATA)
    assert "43" not in rendered.markdown
    assert "clb_zz_rejected" not in rendered.markdown
    assert "- rendered bundles: 2" in rendered.markdown
    assert "- withheld bundles (rejected or abstained): 1" in rendered.markdown
    assert rendered.rendered_bundle_ids == ("clb_confirm", "clb_explore")
    assert rendered.withheld_bundle_ids == ("clb_zz_rejected",)


def test_statistical_violations_become_fixed_qualifiers() -> None:
    markdown = render_claim_report([CONFIRM], run_metadata=METADATA).markdown
    caveats = markdown[markdown.index("## Statistical caveats") :]
    assert (
        "- clb_confirm/c1: exploratory only: a p-value is reported without "
        "effect size, confidence interval and sample size." in caveats
    )
    assert "sequence index" in caveats
    assert "not confirmatory-ready" in caveats


def test_evidence_trail_lists_sorted_receipt_ids_per_bundle() -> None:
    bundle, _report = TWO_RECEIPTS
    markdown = render_claim_report([TWO_RECEIPTS], run_metadata=METADATA).markdown
    expected = "- clb_trail: " + ", ".join(sorted(bundle.referenced_receipt_ids()))
    assert expected in markdown


def test_evidence_trail_covers_every_rendered_bundle() -> None:
    """One line per rendered bundle: a single-bundle assertion cannot tell a
    truncated trail from a complete one."""
    markdown = render_claim_report(ALL_PAIRS, run_metadata=METADATA).markdown
    for bundle, report in ALL_PAIRS:
        if not report.passed:
            continue
        expected = f"- {bundle.claim_bundle_id}: " + ", ".join(
            sorted(bundle.referenced_receipt_ids())
        )
        assert expected in markdown


def test_two_renders_are_byte_identical() -> None:
    first = render_claim_report(ALL_PAIRS, run_metadata=METADATA)
    second = render_claim_report(ALL_PAIRS, run_metadata=METADATA)
    assert first.markdown.encode() == second.markdown.encode()
    assert first == second


def test_input_order_does_not_change_the_output() -> None:
    forward = render_claim_report(ALL_PAIRS, run_metadata=METADATA)
    reversed_ = render_claim_report(ALL_PAIRS[::-1], run_metadata=METADATA)
    assert forward.markdown == reversed_.markdown


def test_empty_input_renders_placeholders_without_tripping_the_rescan() -> None:
    markdown = render_claim_report([], run_metadata=METADATA).markdown
    assert "(none)" in markdown
    assert "- rendered bundles: 0" in markdown


def test_missing_metadata_keys_render_a_fixed_placeholder() -> None:
    markdown = render_claim_report([EXPLORE], run_metadata={}).markdown
    assert "- exploration_id: (not provided)" in markdown


def test_metadata_numbers_are_whitelisted_by_construction() -> None:
    metadata = {**METADATA, "exploration_id": "exploration 7"}
    markdown = render_claim_report([EXPLORE], run_metadata=metadata).markdown
    assert "- exploration_id: exploration 7" in markdown


def test_a_mismatched_bundle_report_pair_is_refused() -> None:
    bundle, _report = EXPLORE
    _other, wrong_report = CONFIRM
    with pytest.raises(ValueError, match="paired"):
        render_claim_report([(bundle, wrong_report)], run_metadata=METADATA)


# --- §4.6 numeric rescan -------------------------------------------------------


def test_rescan_flags_tokens_outside_the_pool() -> None:
    pool = numeric_pool_from_texts(["There are 42 orders."])
    assert assert_rendered_numbers("There are 42 orders.", pool) == []
    assert assert_rendered_numbers("There are 42 orders and 7 dwarfs.", pool) == ["7"]


def test_rescan_keeps_percent_and_raw_pools_apart() -> None:
    pool = numeric_pool_from_texts(["12.5% of users."])
    assert assert_rendered_numbers("12.5% of users.", pool) == []
    assert assert_rendered_numbers("12.5 users.", pool) == ["12.5"]


def test_rescan_accepts_value_preserving_reformatting() -> None:
    # The E4a polish contract: rewriting "1000" as "1,000" is admissible,
    # adding precision or new numbers is not.
    pool = numeric_pool_from_texts(["1000 total", "42 flagged"])
    assert assert_rendered_numbers("1,000 orders, 42 flagged.", pool) == []
    assert assert_rendered_numbers("1,001 orders.", pool) == ["1001"]


def test_a_mutated_fixed_phrase_trips_the_final_rescan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation probe: inject a pool-foreign number into a renderer literal."""
    monkeypatch.setitem(
        claim_renderer._CAVEAT_PHRASES,
        "p_value_without_effect_ci_n",
        "exploratory only: the 95% confidence interval is missing.",
    )
    with pytest.raises(RenderedNumberLeakError, match="95"):
        render_claim_report([CONFIRM], run_metadata=METADATA)
