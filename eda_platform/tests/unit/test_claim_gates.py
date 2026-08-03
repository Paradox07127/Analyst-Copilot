"""Exit gates over ClaimBundles (plan §6.1-§6.4).

Every rejection path here is the product's grounding guarantee: a claim ships
only when its numbers, entities, coverage and data-state all resolve against
digest-verified receipts. The feedback tests additionally pin the §6.4 rule
that a retry prompt never contains the failed claim text.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from eda_platform.agents.receipts import build_receipt
from eda_platform.core.claim_gates import (
    ABSTAINED_STATUS,
    GATE_ORDER,
    GATE_RETRY_BUDGET,
    GateReport,
    build_gate_feedback,
    retry_decision,
    run_claim_gates,
)
from eda_platform.schemas.claims import Claim, ClaimBundle, ClaimScope
from eda_platform.schemas.receipts import (
    EvidenceReceipt,
    ReceiptDerivation,
    ReceiptFact,
    ReceiptMethod,
    ReceiptScope,
    ReceiptStatistics,
)

RUN_WITNESS = "dsw1_" + "a" * 64
STALE_WITNESS = "dsw1_" + "b" * 64


def _fact(
    fact_id: str,
    value: object,
    value_type: str = "number",
    *,
    unit: str | None = "raw",
    support_type: str = "direct",
) -> ReceiptFact:
    return ReceiptFact(
        fact_id=fact_id,
        name=fact_id,
        value=value,  # type: ignore[arg-type]
        value_type=value_type,  # type: ignore[arg-type]
        unit=unit,
        support_type=support_type,  # type: ignore[arg-type]
    )


def _receipt(
    *,
    tool_call_id: str,
    facts: tuple[ReceiptFact, ...],
    tool_name: str = "run_sql",
    result_count: int = 1,
    derivations: tuple[ReceiptDerivation, ...] = (),
    statistics: ReceiptStatistics | None = None,
    scope: ReceiptScope | None = None,
    witness: str = RUN_WITNESS,
    replication_kind: str | None = None,
    method_family: str = "sql_aggregation",
) -> EvidenceReceipt:
    return build_receipt(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        tool_version="1",
        arguments={"call": tool_call_id},
        raw_output={"rows": []},
        artifact_ids=(),
        result_count=result_count,
        scope=scope or ReceiptScope(dataset_ids=("ds_orders",), columns=("amount",)),
        facts=facts,
        derivations=derivations,
        method=ReceiptMethod(family=method_family),
        statistics=statistics,
        data_state_witness=witness,
        created_at="2026-08-02T00:00:00Z",
        replication_kind=replication_kind,  # type: ignore[arg-type]
    )


R_MAIN = _receipt(
    tool_call_id="call_main",
    facts=(
        _fact("f_n", 42, "count"),
        _fact("f_total", 1000, "count"),
        _fact("f_avg", 3.70423, "number"),
        _fact("f_pct", 12.5, "percent", unit="percent"),
    ),
)
R_DERIV = _receipt(
    tool_call_id="call_deriv",
    facts=(_fact("fa", 42, "count"), _fact("fb", 84, "count")),
    derivations=(
        ReceiptDerivation(derived_fact_id="d_ratio", operator="ratio", input_fact_ids=("fa", "fb")),
        ReceiptDerivation(
            derived_fact_id="d_pct", operator="percentage", input_fact_ids=("fa", "fb")
        ),
    ),
)
R_ZERO_WEIGHTS = _receipt(
    tool_call_id="call_weights",
    facts=(
        _fact("v1", 10, "number"),
        _fact("w1", 0, "number"),
        _fact("v2", 20, "number"),
        _fact("w2", 0, "number"),
    ),
    derivations=(
        ReceiptDerivation(
            derived_fact_id="d_wavg",
            operator="weighted_average",
            input_fact_ids=("v1", "w1", "v2", "w2"),
        ),
    ),
)
R_ABSENCE = _receipt(
    tool_call_id="call_absence",
    facts=(_fact("f_absent", None, "null", unit=None, support_type="absence"),),
    result_count=0,
    scope=ReceiptScope(
        dataset_ids=("ds_orders",), columns=("amount",), scope_resolution="resolved"
    ),
)
R_STAT_OK = _receipt(
    tool_call_id="call_stat_ok",
    facts=(
        _fact("f_p", 0.003, "number"),
        _fact("f_effect", 0.42, "number"),
        _fact("f_n_obs", 200, "count"),
    ),
    tool_name="run_stat_test",
    method_family="independent_t_test",
    statistics=ReceiptStatistics(
        hypothesis_id="fam_orders_amount",
        test_name="independent_t_test",
        p_value=0.003,
        effect_size=0.42,
        ci_low=0.1,
        ci_high=0.7,
        sample_size=200,
        sequence_index=1,
    ),
)
R_STAT_BAD = _receipt(
    tool_call_id="call_stat_bad",
    facts=(_fact("f_p2", 0.003, "number"),),
    tool_name="run_stat_test",
    method_family="shapiro_wilk",
    statistics=ReceiptStatistics(test_name="shapiro_wilk", p_value=0.003),
)
R_MODEL = _receipt(
    tool_call_id="call_model",
    facts=(_fact("f_auc", 0.91, "number"),),
    tool_name="run_baseline_model",
    method_family="ml_baseline",
)
R_HOLDOUT = _receipt(
    tool_call_id="call_holdout",
    facts=(_fact("f_rep", 41, "count"),),
    replication_kind="holdout",
)
R_STALE = _receipt(
    tool_call_id="call_stale",
    facts=(_fact("f_old", 7, "count"),),
    witness=STALE_WITNESS,
)

ALL_RECEIPTS = (
    R_MAIN,
    R_DERIV,
    R_ZERO_WEIGHTS,
    R_ABSENCE,
    R_STAT_OK,
    R_STAT_BAD,
    R_MODEL,
    R_HOLDOUT,
    R_STALE,
)
COMMITTED: Mapping[str, EvidenceReceipt] = {r.receipt_id: r for r in ALL_RECEIPTS}

MISSING_RECEIPT_ID = "rcpt_" + "f" * 24


def _ref(receipt: EvidenceReceipt, fact_id: str) -> str:
    return f"{receipt.receipt_id}:{fact_id}"


def _claim(**overrides: object) -> Claim:
    fields: dict[str, object] = {
        "claim_id": "c1",
        "claim_type": "observation",
        "claim_text": "There are 42 orders.",
        "support_type": "direct",
        "evidence_fact_ids": (_ref(R_MAIN, "f_n"),),
    }
    fields.update(overrides)
    return Claim.model_validate(fields)


def _bundle(*claims: Claim, **overrides: object) -> ClaimBundle:
    fields: dict[str, object] = {
        "claim_bundle_id": "clb_1",
        "hypothesis_id": "hyp_1",
        "evidence_lane": "exploratory",
        "claims": claims or (_claim(),),
    }
    fields.update(overrides)
    return ClaimBundle.model_validate(fields)


def _run(bundle: ClaimBundle | Mapping[str, object], **kwargs: object) -> GateReport:
    return run_claim_gates(
        bundle,
        committed_receipts=kwargs.pop("committed_receipts", COMMITTED),  # type: ignore[arg-type]
        run_witness=kwargs.pop("run_witness", RUN_WITNESS),  # type: ignore[arg-type]
        stat_attempt_counts=kwargs.pop(
            "stat_attempt_counts", {"fam_orders_amount": 1}
        ),  # type: ignore[arg-type]
    )


def _violations(report: GateReport, gate: str) -> list:
    for verdict in report.verdicts:
        if verdict.gate == gate:
            return list(verdict.violations)
    return []


def _codes(report: GateReport, gate: str) -> set[str]:
    return {violation.code for violation in _violations(report, gate)}


# --- structure gate ---------------------------------------------------------


def test_a_fully_grounded_bundle_passes_every_gate() -> None:
    claim = _claim(
        claim_text="There are 42 orders, 12.5% of users are affected.",
        evidence_fact_ids=(_ref(R_MAIN, "f_n"), _ref(R_MAIN, "f_pct")),
    )
    report = _run(_bundle(claim))
    assert report.passed, report.model_dump()
    assert [verdict.gate for verdict in report.verdicts] == list(GATE_ORDER)
    assert all(verdict.passed for verdict in report.verdicts)
    assert report.health_score == 1.0
    assert build_gate_feedback(report) == []


def test_structure_gate_rejects_malformed_payload_and_short_circuits() -> None:
    payload = {
        "claim_bundle_id": "clb_bad",
        "hypothesis_id": "hyp_1",
        "evidence_lane": "exploratory",
        "unknown_field": 1,
        "claims": [
            {
                "claim_id": "c1",
                "claim_type": "observation",
                "claim_text": "MARKER_DO_NOT_ECHO 99 rows",
                "support_type": "direct",
                "evidence_fact_ids": ["not-a-qualified-ref"],
            }
        ],
    }
    report = _run(payload)
    assert not report.passed
    assert len(report.verdicts) == 1
    assert report.verdicts[0].gate == "structure"
    assert not report.verdicts[0].passed
    assert report.verdicts[0].violations
    feedback = build_gate_feedback(report)
    assert feedback and feedback[-1].channel == "user"
    for item in feedback:
        assert "MARKER_DO_NOT_ECHO" not in item.message


# --- reachability gate ------------------------------------------------------


def test_a_fabricated_receipt_reference_is_rejected() -> None:
    claim = _claim(evidence_fact_ids=(f"{MISSING_RECEIPT_ID}:f1",))
    report = _run(_bundle(claim))
    assert not report.passed
    assert "fabricated_receipt" in _codes(report, "reachability")


def test_an_unknown_fact_in_a_real_receipt_is_rejected() -> None:
    claim = _claim(evidence_fact_ids=(_ref(R_MAIN, "f_ghost"),))
    report = _run(_bundle(claim))
    assert not report.passed
    assert "unknown_fact" in _codes(report, "reachability")


def test_an_unknown_derivation_reference_is_rejected() -> None:
    claim = _claim(
        claim_text="Half of orders.",
        evidence_fact_ids=(_ref(R_DERIV, "fa"),),
        derivation_ids=(_ref(R_DERIV, "d_ghost"),),
    )
    report = _run(_bundle(claim))
    assert not report.passed
    assert "unknown_derivation" in _codes(report, "reachability")


def test_a_tampered_receipt_digest_is_rejected() -> None:
    """Reachability loads with load_verified_receipt semantics: digest always
    re-verified, a tampered committed receipt is unusable evidence."""
    tampered = R_MAIN.model_copy(update={"result_count": 5})
    committed = {**COMMITTED, tampered.receipt_id: tampered}
    report = _run(_bundle(), committed_receipts=committed)
    assert not report.passed
    assert "receipt_digest_mismatch" in _codes(report, "reachability")


def test_a_statistics_reference_needs_a_statistics_bearing_receipt() -> None:
    claim = _claim(statistics_receipt_ids=(R_MAIN.receipt_id,))
    report = _run(_bundle(claim))
    assert not report.passed
    assert "missing_statistics" in _codes(report, "reachability")


# --- numeric gate -----------------------------------------------------------


def test_numeric_gate_rejects_untraceable_numbers_and_summarises_the_pool() -> None:
    claim = _claim(claim_text="There are 43 orders.")
    report = _run(_bundle(claim))
    assert not report.passed
    violations = _violations(report, "numeric")
    assert violations and violations[0].code == "numeric_mismatch"
    assert "42" in violations[0].message  # allowed-pool summary
    assert "There are" not in violations[0].message  # never the claim text


def test_the_claim_itself_never_feeds_the_value_pool() -> None:
    """A number repeated inside the bundle must not certify itself."""
    first = _claim(claim_id="c1", claim_text="58 orders were flagged.")
    second = _claim(claim_id="c2", claim_text="58 anomalies were flagged.")
    report = _run(_bundle(first, second))
    assert not report.passed
    assert {violation.claim_id for violation in _violations(report, "numeric")} == {
        "c1",
        "c2",
    }


def test_count_facts_require_exact_matches_and_floats_allow_half_ulp() -> None:
    exact_fail = _claim(
        claim_text="1,001 orders in total.",
        evidence_fact_ids=(_ref(R_MAIN, "f_total"),),
    )
    assert not _run(_bundle(exact_fail)).passed

    exact_pass = _claim(
        claim_text="1,000 orders in total.",
        evidence_fact_ids=(_ref(R_MAIN, "f_total"),),
    )
    rounded_pass = _claim(
        claim_id="c2",
        claim_text="The average is 3.7 items.",
        evidence_fact_ids=(_ref(R_MAIN, "f_avg"),),
    )
    assert _run(_bundle(exact_pass, rounded_pass)).passed


def test_percent_and_raw_pools_do_not_wash_each_other() -> None:
    claim = _claim(
        claim_text="12.5 orders were affected.",  # raw token, only percent evidence
        evidence_fact_ids=(_ref(R_MAIN, "f_pct"),),
    )
    report = _run(_bundle(claim))
    assert not report.passed
    assert "numeric_mismatch" in _codes(report, "numeric")


def test_derivations_are_recomputed_from_input_facts() -> None:
    good = _claim(
        claim_text="50% of orders (a 0.5 ratio).",
        evidence_fact_ids=(_ref(R_DERIV, "fa"),),
        derivation_ids=(_ref(R_DERIV, "d_pct"), _ref(R_DERIV, "d_ratio")),
    )
    assert _run(_bundle(good)).passed

    bad = _claim(
        claim_text="51% of orders.",
        evidence_fact_ids=(_ref(R_DERIV, "fa"),),
        derivation_ids=(_ref(R_DERIV, "d_pct"),),
    )
    report = _run(_bundle(bad))
    assert not report.passed
    assert "numeric_mismatch" in _codes(report, "numeric")


def test_a_derivation_recompute_failure_is_rejected_and_tool_bound() -> None:
    claim = _claim(
        claim_text="The weighted average is 15.",
        evidence_fact_ids=(_ref(R_ZERO_WEIGHTS, "v1"),),
        derivation_ids=(_ref(R_ZERO_WEIGHTS, "d_wavg"),),
    )
    report = _run(_bundle(claim))
    assert not report.passed
    violations = _violations(report, "numeric")
    recompute = [v for v in violations if v.code == "derivation_recompute_failed"]
    assert recompute and recompute[0].tool_call_id == R_ZERO_WEIGHTS.tool_call_id


def test_number_bearing_claims_with_no_numeric_evidence_are_rejected() -> None:
    claim = _claim(
        claim_text="Exactly 7 gaps were found.",
        evidence_fact_ids=(_ref(R_ABSENCE, "f_absent"),),
        claim_type="absence",
        support_type="absence",
        scope=ClaimScope(dataset_ids=("ds_orders",), columns=("amount",)),
    )
    report = _run(_bundle(claim))
    assert not report.passed
    assert "no_evidence_values" in _codes(report, "numeric")


def test_threshold_tokens_verify_against_cited_statistics_only() -> None:
    with_stats = _claim(
        claim_text="The difference is significant at p < 0.05.",
        evidence_fact_ids=(_ref(R_STAT_OK, "f_p"),),
        statistics_receipt_ids=(R_STAT_OK.receipt_id,),
    )
    assert _run(_bundle(with_stats)).passed

    without_stats = _claim(
        claim_text="The difference is significant at p < 0.05.",
        evidence_fact_ids=(_ref(R_MAIN, "f_n"),),
    )
    assert not _run(_bundle(without_stats)).passed


# --- entity / capability gate -----------------------------------------------


def test_model_claims_require_model_capability_evidence() -> None:
    ungrounded = _claim(
        claim_type="model",
        claim_text="A classifier reached 0.91 AUC.",
        evidence_fact_ids=(_ref(R_MAIN, "f_avg"),),
    )
    report = _run(_bundle(ungrounded))
    assert not report.passed
    assert "model_claim_without_model_evidence" in _codes(report, "entity")

    grounded = _claim(
        claim_type="model",
        claim_text="A classifier reached 0.91 AUC.",
        evidence_fact_ids=(_ref(R_MODEL, "f_auc"),),
    )
    assert _run(_bundle(grounded)).passed


def test_model_language_without_model_evidence_is_rejected_even_untyped() -> None:
    claim = _claim(claim_text="The model produced 42 example rows.")
    report = _run(_bundle(claim))
    assert not report.passed
    assert "model_claim_without_model_evidence" in _codes(report, "entity")


def test_model_metrics_must_come_from_the_cited_model_receipt() -> None:
    claim = _claim(
        claim_type="model",
        claim_text="The model produced 42 scored rows.",
        evidence_fact_ids=(_ref(R_MAIN, "f_n"), _ref(R_MODEL, "f_auc")),
    )
    report = _run(_bundle(claim))
    assert not report.passed
    assert "model_metric_not_model_derived" in _codes(report, "entity")


def test_causal_claims_are_rejected_by_default() -> None:
    typed = _claim(claim_type="causal", claim_text="Discounts move volume, 42 rows.")
    report = _run(_bundle(typed))
    assert not report.passed
    assert "causal_claim_rejected" in _codes(report, "entity")

    worded = _claim(claim_text="42 returns happened because of discounts.")
    report = _run(_bundle(worded))
    assert not report.passed
    assert "causal_language" in _codes(report, "entity")

    disclaimed = _claim(
        claim_text="42 returns are associated with discounts; not a causal explanation."
    )
    assert _run(_bundle(disclaimed)).passed


def test_recommendations_must_be_inference_with_assumptions() -> None:
    wrong_support = _claim(
        claim_type="recommendation",
        claim_text="Review the 42 flagged orders.",
        support_type="direct",
    )
    report = _run(_bundle(wrong_support))
    assert not report.passed
    assert "recommendation_not_marked_inference" in _codes(report, "entity")

    no_assumptions = _claim(
        claim_type="recommendation",
        claim_text="Review the 42 flagged orders.",
        support_type="inference",
    )
    report = _run(_bundle(no_assumptions))
    assert not report.passed
    assert "recommendation_without_assumptions" in _codes(report, "entity")

    proper = _claim(
        claim_type="recommendation",
        claim_text="Review the 42 flagged orders.",
        support_type="inference",
        assumptions=("Flagging rules stay unchanged.",),
    )
    assert _run(_bundle(proper)).passed


def test_absence_claims_walk_the_coverage_gate() -> None:
    good = _claim(
        claim_type="absence",
        support_type="absence",
        claim_text="No missing amounts were observed in the checked scope.",
        evidence_fact_ids=(_ref(R_ABSENCE, "f_absent"),),
        scope=ClaimScope(dataset_ids=("ds_orders",), columns=("amount",)),
    )
    assert _run(_bundle(good)).passed

    false_absence = _claim(
        claim_type="absence",
        support_type="absence",
        claim_text="No orders were observed.",
        evidence_fact_ids=(_ref(R_MAIN, "f_n"),),
        scope=ClaimScope(dataset_ids=("ds_orders",), columns=("amount",)),
    )
    report = _run(_bundle(false_absence))
    assert not report.passed
    assert "false_absence" in _codes(report, "entity")

    unscoped = good.model_copy(update={"scope": None})
    report = _run(_bundle(unscoped))
    assert not report.passed
    assert "absence_scope_missing" in _codes(report, "entity")

    over_scoped = good.model_copy(
        update={"scope": ClaimScope(dataset_ids=("ds_orders",), columns=("amount", "tax"))}
    )
    report = _run(_bundle(over_scoped))
    assert not report.passed
    assert "absence_scope_exceeds_coverage" in _codes(report, "entity")


def test_same_snapshot_evidence_cannot_claim_independent_replication() -> None:
    declared = _bundle(declares_independent_replication=True)
    report = _run(declared)
    assert not report.passed
    assert "unlicensed_independence_claim" in _codes(report, "entity")

    worded = _claim(claim_text="The 42-order pattern was independently replicated.")
    report = _run(_bundle(worded))
    assert not report.passed
    assert "unlicensed_independence_claim" in _codes(report, "entity")

    licensed = _bundle(
        _claim(
            claim_text="The pattern held on the 41-row holdout.",
            evidence_fact_ids=(_ref(R_HOLDOUT, "f_rep"),),
        ),
        declares_independent_replication=True,
    )
    assert _run(licensed).passed


def test_holdout_evidence_cannot_license_a_sibling_claim() -> None:
    holdout = _claim(
        claim_id="c_holdout",
        claim_text="The holdout contains 41 rows.",
        evidence_fact_ids=(_ref(R_HOLDOUT, "f_rep"),),
    )
    same_snapshot = _claim(
        claim_id="c_same_snapshot",
        claim_text="The 42-order pattern was independently replicated.",
    )
    report = _run(_bundle(holdout, same_snapshot))
    assert not report.passed
    assert any(
        violation.claim_id == "c_same_snapshot"
        for violation in _violations(report, "entity")
        if violation.code == "unlicensed_independence_claim"
    )


# --- statistical gate ---------------------------------------------------------


def test_statistical_gate_blocks_incomplete_confirmatory_evidence() -> None:
    claim = _claim(
        claim_text="A significant difference was observed.",
        evidence_fact_ids=(_ref(R_STAT_BAD, "f_p2"),),
        statistics_receipt_ids=(R_STAT_BAD.receipt_id,),
    )
    report = _run(_bundle(claim, evidence_lane="confirmatory"))
    statistical = next(v for v in report.verdicts if v.gate == "statistical")
    assert not statistical.passed
    codes = {violation.code for violation in statistical.violations}
    assert "p_value_without_effect_ci_n" in codes
    assert "sequence_index_missing" in codes
    assert "test_not_confirmatory_ready" in codes
    assert not report.passed
    assert report.health_score < 1.0  # registered violations still dent health


def test_confirmatory_claims_need_an_actual_inferential_result() -> None:
    """Citing a statistics receipt is not the same as citing a statistic.

    Every completeness and multiplicity check keys off `p_value`, so a receipt
    that omits it used to skip all of them while still satisfying the
    "confirmatory cites statistics" check.
    """
    hollow = _receipt(
        tool_call_id="call_stat_hollow",
        facts=(_fact("f_hollow", 1, "count"),),
        tool_name="run_stat_test",
        method_family="independent_t_test",
        statistics=ReceiptStatistics(
            hypothesis_id="fam_orders_amount",
            test_name="independent_t_test",
            sequence_index=1,
        ),
    )
    claim = _claim(
        claim_text="Group means differ significantly.",
        evidence_fact_ids=(_ref(hollow, "f_hollow"),),
        statistics_receipt_ids=(hollow.receipt_id,),
    )
    report = _run(
        _bundle(claim, evidence_lane="confirmatory"),
        committed_receipts={**COMMITTED, hollow.receipt_id: hollow},
    )
    assert not report.passed
    assert "confirmatory_without_test_statistic" in _codes(report, "statistical")


def test_confirmatory_claims_without_statistics_are_blocked() -> None:
    report = _run(_bundle(evidence_lane="confirmatory"))
    assert not report.passed
    assert "confirmatory_without_statistics" in _codes(report, "statistical")


def test_a_publishable_confirmatory_claim_registers_cleanly() -> None:
    claim = _claim(
        claim_text="Group means differ (p = 0.003, effect 0.42, n = 200).",
        evidence_fact_ids=(
            _ref(R_STAT_OK, "f_p"),
            _ref(R_STAT_OK, "f_effect"),
            _ref(R_STAT_OK, "f_n_obs"),
        ),
        statistics_receipt_ids=(R_STAT_OK.receipt_id,),
    )
    report = _run(_bundle(claim, evidence_lane="confirmatory"))
    assert report.passed
    assert _violations(report, "statistical") == []


def test_confirmatory_gate_rechecks_the_final_stat_family_size() -> None:
    first = _receipt(
        tool_call_id="call_stat_first",
        facts=(
            _fact("f_p_first", 0.01, "number"),
            _fact("f_effect_first", 0.5, "number"),
            _fact("f_n_first", 100, "count"),
        ),
        tool_name="run_stat_test",
        method_family="independent_t_test",
        statistics=ReceiptStatistics(
            hypothesis_id="fam_repeated",
            test_name="independent_t_test",
            p_value=0.01,
            effect_size=0.5,
            ci_low=0.1,
            ci_high=0.9,
            sample_size=100,
            sequence_index=1,
        ),
    )
    committed = {**COMMITTED, first.receipt_id: first}
    claim = _claim(
        claim_text="The repeated comparison has p = 0.01.",
        evidence_fact_ids=(_ref(first, "f_p_first"),),
        statistics_receipt_ids=(first.receipt_id,),
    )
    report = _run(
        _bundle(claim, evidence_lane="confirmatory"),
        committed_receipts=committed,
        stat_attempt_counts={"fam_repeated": 5},
    )
    assert not report.passed
    assert "stale_multiplicity_adjustment" in _codes(report, "statistical")


def test_statistical_publication_requires_the_registry_family_count() -> None:
    claim = _claim(
        claim_text="The comparison has p = 0.003.",
        evidence_fact_ids=(_ref(R_STAT_OK, "f_p"),),
        statistics_receipt_ids=(R_STAT_OK.receipt_id,),
    )
    report = _run(
        _bundle(claim, evidence_lane="confirmatory"),
        stat_attempt_counts=None,
    )
    assert not report.passed
    assert "stat_family_count_unavailable" in _codes(report, "statistical")


# --- state gate ---------------------------------------------------------------


def test_state_gate_rejects_witness_mismatch() -> None:
    claim = _claim(
        claim_text="7 legacy rows exist.",
        evidence_fact_ids=(_ref(R_STALE, "f_old"),),
    )
    report = _run(_bundle(claim))
    assert not report.passed
    assert "witness_mismatch" in _codes(report, "state")


# --- whole-report behaviour ---------------------------------------------------


def test_all_gates_run_after_structure_for_a_complete_failure_list() -> None:
    claim = _claim(
        claim_type="causal",
        claim_text="43 returns, driven by discounts.",
        evidence_fact_ids=(_ref(R_STALE, "f_old"),),
    )
    report = _run(_bundle(claim))
    assert [verdict.gate for verdict in report.verdicts] == list(GATE_ORDER)
    failed = {verdict.gate for verdict in report.verdicts if not verdict.passed}
    assert {"numeric", "entity", "state"} <= failed


def test_health_score_uses_the_groundeval_multiplier() -> None:
    clean = _claim(claim_id="c_ok")
    dirty = _claim(claim_id="c_bad", claim_text="There are 43 orders.")
    report = _run(_bundle(clean, dirty))
    assert not report.passed
    assert report.health_score == pytest.approx(0.25)  # (1 - 1/2)^2


def test_feedback_is_two_channel_and_never_echoes_claim_text() -> None:
    tool_bound = _claim(
        claim_id="c_tool",
        claim_text="SECRET_MARKER weighted average is 15.",
        evidence_fact_ids=(_ref(R_ZERO_WEIGHTS, "v1"),),
        derivation_ids=(_ref(R_ZERO_WEIGHTS, "d_wavg"),),
    )
    fabricated = _claim(
        claim_id="c_user",
        claim_text="SECRET_MARKER 99 phantom rows.",
        evidence_fact_ids=(f"{MISSING_RECEIPT_ID}:f1",),
    )
    report = _run(_bundle(tool_bound, fabricated))
    assert not report.passed
    feedback = build_gate_feedback(report)
    channels = {item.channel for item in feedback}
    assert channels == {"tool", "user"}
    tool_items = [item for item in feedback if item.channel == "tool"]
    assert tool_items[0].tool_call_id == R_ZERO_WEIGHTS.tool_call_id
    user_item = next(item for item in feedback if item.channel == "user")
    assert user_item.message.startswith("Validation feedback:\n")
    assert user_item.message.rstrip().endswith("Fix the errors and try again.")
    for item in feedback:
        assert "SECRET_MARKER" not in item.message


def test_retry_budget_is_one_and_the_second_failure_abstains() -> None:
    assert GATE_RETRY_BUDGET == 1
    assert retry_decision(1) == "retry"
    assert retry_decision(2) == ABSTAINED_STATUS == "abstained"


# --- R3 review fixes ---------------------------------------------------------

R_ABS_USERS = _receipt(
    tool_call_id="call_absence_users",
    facts=(_fact("f_user", None, "null", unit=None, support_type="absence"),),
    result_count=0,
    scope=ReceiptScope(
        dataset_ids=("ds_users",), columns=("email",), scope_resolution="resolved"
    ),
)
R_ABS_SLICE = _receipt(
    tool_call_id="call_absence_slice",
    facts=(_fact("f_slice", None, "null", unit=None, support_type="absence"),),
    result_count=0,
    scope=ReceiptScope(
        dataset_ids=("ds_orders",),
        columns=("amount",),
        scope_resolution="resolved",
        filters="region = 'ap-south-1'",
        time_range="2026-07-30/2026-08-02",
    ),
)
R3_COMMITTED: Mapping[str, EvidenceReceipt] = {
    **COMMITTED,
    R_ABS_USERS.receipt_id: R_ABS_USERS,
    R_ABS_SLICE.receipt_id: R_ABS_SLICE,
}


def _absence(**overrides: object) -> Claim:
    fields: dict[str, object] = {
        "claim_type": "absence",
        "support_type": "absence",
        "claim_text": "No gaps were observed in the checked scope.",
        "evidence_fact_ids": (_ref(R_ABSENCE, "f_absent"),),
        "scope": ClaimScope(dataset_ids=("ds_orders",), columns=("amount",)),
    }
    fields.update(overrides)
    return _claim(**fields)


def test_absence_coverage_pairs_datasets_with_columns() -> None:
    # ds_orders.amount and ds_users.email were scanned; ds_orders.email was not.
    cross = _absence(
        claim_text="No order is missing a customer email.",
        evidence_fact_ids=(_ref(R_ABSENCE, "f_absent"), _ref(R_ABS_USERS, "f_user")),
        scope=ClaimScope(dataset_ids=("ds_orders",), columns=("email",)),
    )
    report = _run(_bundle(cross), committed_receipts=R3_COMMITTED)
    assert not report.passed
    assert "absence_scope_exceeds_coverage" in _codes(report, "entity")

    # Both scanned pairs together still license the union claim.
    honest = _absence(
        claim_text="No gaps in either checked column.",
        evidence_fact_ids=(_ref(R_ABSENCE, "f_absent"), _ref(R_ABS_USERS, "f_user")),
        scope=ClaimScope(dataset_ids=("ds_orders", "ds_users"), columns=()),
    )
    assert _run(_bundle(honest), committed_receipts=R3_COMMITTED).passed


def test_a_filtered_coverage_receipt_cannot_license_an_unrestricted_absence() -> None:
    unrestricted = _absence(
        claim_text="There are no missing amounts anywhere in ds_orders.",
        evidence_fact_ids=(_ref(R_ABS_SLICE, "f_slice"),),
    )
    report = _run(_bundle(unrestricted), committed_receipts=R3_COMMITTED)
    assert not report.passed
    assert "absence_scope_exceeds_coverage" in _codes(report, "entity")

    contradicting = _absence(
        claim_text="No missing amounts in the EU region last year.",
        evidence_fact_ids=(_ref(R_ABS_SLICE, "f_slice"),),
        scope=ClaimScope(
            dataset_ids=("ds_orders",),
            columns=("amount",),
            filters="region = 'eu-west-1'",
            time_range="2024-01-01/2024-12-31",
        ),
    )
    report = _run(_bundle(contradicting), committed_receipts=R3_COMMITTED)
    assert not report.passed
    assert "absence_scope_exceeds_coverage" in _codes(report, "entity")

    matching = _absence(
        claim_text="No missing amounts in the checked slice.",
        evidence_fact_ids=(_ref(R_ABS_SLICE, "f_slice"),),
        scope=ClaimScope(
            dataset_ids=("ds_orders",),
            columns=("amount",),
            filters="region = 'ap-south-1'",
            time_range="2026-07-30/2026-08-02",
        ),
    )
    assert _run(_bundle(matching), committed_receipts=R3_COMMITTED).passed


def test_an_unfiltered_scan_still_covers_a_narrower_declared_slice() -> None:
    # Absence over the whole table implies absence over any sub-slice.
    narrow = _absence(
        claim_text="No missing amounts in the EU region.",
        scope=ClaimScope(
            dataset_ids=("ds_orders",),
            columns=("amount",),
            filters="region = 'eu-west-1'",
        ),
    )
    assert _run(_bundle(narrow), committed_receipts=R3_COMMITTED).passed


def test_business_thresholds_cannot_launder_through_the_statistics_pool() -> None:
    for text in (
        "Weekly enterprise churn stayed < 0.01 in every cohort.",
        "Total fraud losses this quarter were < 1000000 USD.",
        "Model lift over baseline was > 0.1 on every segment.",
    ):
        claim = _claim(
            claim_text=text,
            evidence_fact_ids=(_ref(R_STAT_OK, "f_p"),),
            statistics_receipt_ids=(R_STAT_OK.receipt_id,),
        )
        report = _run(_bundle(claim, evidence_lane="confirmatory"))
        assert not report.passed, text
        assert "numeric_mismatch" in _codes(report, "numeric"), text


def test_a_threshold_only_verifies_against_the_statistic_it_names() -> None:
    # R_STAT_OK: p_value 0.003, effect_size 0.42. "effect size < 0.01" is false
    # of the effect size and true only of the p-value — the borrowing path.
    borrowed = _claim(
        claim_text="The effect size < 0.01 across segments.",
        evidence_fact_ids=(_ref(R_STAT_OK, "f_p"),),
        statistics_receipt_ids=(R_STAT_OK.receipt_id,),
    )
    report = _run(_bundle(borrowed, evidence_lane="confirmatory"))
    assert not report.passed
    assert "numeric_mismatch" in _codes(report, "numeric")

    truthful = _claim(
        claim_text="The effect size > 0.2 across segments.",
        evidence_fact_ids=(_ref(R_STAT_OK, "f_p"),),
        statistics_receipt_ids=(R_STAT_OK.receipt_id,),
    )
    assert _run(_bundle(truthful, evidence_lane="confirmatory")).passed


def test_p_value_bounds_are_not_false_killed() -> None:
    for text in (
        "The difference is significant at p < 0.05.",
        "The difference is significant (p-value < 0.01).",
    ):
        claim = _claim(
            claim_text=text,
            evidence_fact_ids=(_ref(R_STAT_OK, "f_p"),),
            statistics_receipt_ids=(R_STAT_OK.receipt_id,),
        )
        assert _run(_bundle(claim, evidence_lane="confirmatory")).passed, text


def test_a_forged_bundle_instance_is_revalidated_by_the_structure_gate() -> None:
    honest = _claim(claim_text="Data quality in ds_orders is excellent.")
    ghost = honest.model_copy(update={"evidence_fact_ids": ()})
    forged = ClaimBundle.model_construct(
        claim_bundle_id="clb_ghost",
        hypothesis_id="hyp_1",
        evidence_lane="confirmatory",
        claims=(ghost,),
        declares_independent_replication=False,
    )
    report = _run(forged)
    assert not report.passed
    structure = next(v for v in report.verdicts if v.gate == "structure")
    assert not structure.passed
    assert {violation.code for violation in structure.violations} == {"schema_invalid"}
    assert report.claim_bundle_id == "clb_ghost"


def test_revalidation_keeps_a_well_formed_bundle_instance_working() -> None:
    assert _run(_bundle()).passed
