"""Offline self-checks for the Eval-0 exploration-baseline harness.

Same discipline as the NL2SQL harness tests: every scorer is deterministic and
must prove three properties without an LLM —

1. *Soundness*: perfect agent output scores perfectly.
2. *Discriminative power*: known-bad output (missed insight, claimed absent
   pattern, leaked canary, fabricated receipt ref) must NOT pass.
3. *Fixture integrity*: the planted signals really exist in the vendored CSV,
   the negative patterns really are absent, and the injection payloads really
   are embedded — all recomputed here with pandas, not trusted from JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pandas as pd
import pytest
from pydantic import ValidationError

from .checkers import (
    CHECKER_VERSION,
    SIX_INSIGHT_FAMILIES,
    AbsentPattern,
    InjectionManifest,
    PlantedInsight,
    ReceiptFact,
    ReportedInsight,
    classify_claim_kind,
    load_absent_patterns,
    load_injection_manifest,
    load_planted_ground_truth,
    score_grounding,
    score_injection,
    score_negative,
    score_planted,
)
from .harness import (
    ItemResult,
    RunUsage,
    item_result_path,
    load_item_results,
    load_suite,
    metric_direction,
    summarize,
    write_item_result,
)

BASE_DIR = Path(__file__).parent
FIXTURES = BASE_DIR / "fixtures"
PLANTED_CSV = FIXTURES / "planted" / "planted_retail.csv"
PLANTED_TRUTH = FIXTURES / "planted" / "ground_truth.json"
ABSENT_PATTERNS = FIXTURES / "negative" / "absent_patterns.json"
INJECTION_CSV = FIXTURES / "injection" / "injection_probe.csv"
INJECTION_MANIFEST = FIXTURES / "injection" / "injection_manifest.json"
SUITES = BASE_DIR / "suites"


# ---------------------------------------------------------------- planted


def _truth() -> list[PlantedInsight]:
    return load_planted_ground_truth(PLANTED_TRUTH)


def _perfect_reports() -> list[ReportedInsight]:
    return [
        ReportedInsight(
            insight_id=f"r_{expected.insight_id}",
            kind=expected.kind,
            columns=list(expected.columns),
            direction=expected.direction,
            evidence_refs=["fact_1"],
        )
        for expected in _truth()
    ]


def test_planted_soundness_perfect_reports_score_perfectly() -> None:
    score = score_planted(_truth(), _perfect_reports())
    assert score.precision == 1.0
    assert score.recall == 1.0
    assert score.f1 == 1.0
    assert score.duplicate_rate == 0.0


def test_planted_missed_insight_lowers_recall_not_precision() -> None:
    reports = _perfect_reports()[:-1]
    score = score_planted(_truth(), reports)
    assert score.precision == 1.0
    assert score.recall < 1.0


def test_planted_spurious_report_lowers_precision() -> None:
    reports = [
        *_perfect_reports(),
        ReportedInsight(insight_id="bogus", kind="correlation", columns=["units", "customer_age"]),
    ]
    score = score_planted(_truth(), reports)
    assert score.precision < 1.0
    assert score.recall == 1.0


def test_planted_duplicate_reports_counted_not_credited_twice() -> None:
    reports = [*_perfect_reports(), *_perfect_reports()]
    score = score_planted(_truth(), reports)
    assert score.recall == 1.0
    assert score.duplicate_rate > 0.0


def test_planted_wrong_direction_is_not_a_match() -> None:
    trend = next(e for e in _truth() if e.kind == "trend")
    flipped = "decreasing" if trend.direction == "increasing" else "increasing"
    report = ReportedInsight(insight_id="r", kind="trend", columns=trend.columns, direction=flipped)
    score = score_planted([trend], [report])
    assert score.recall == 0.0


def test_planted_empty_reports_scores_zero_recall() -> None:
    score = score_planted(_truth(), [])
    assert score.recall == 0.0
    assert score.f1 == 0.0


# ---------------------------------------------------------------- negative


def test_negative_soundness_silence_passes() -> None:
    score = score_negative(load_absent_patterns(ABSENT_PATTERNS), [])
    assert score.passed
    assert score.violations == []


def test_negative_claiming_absent_pattern_fails() -> None:
    patterns = load_absent_patterns(ABSENT_PATTERNS)
    bogus = ReportedInsight(
        insight_id="r_bad",
        kind=patterns[0].kind,
        columns=list(patterns[0].columns),
        direction="increasing",
    )
    score = score_negative(patterns, [bogus])
    assert not score.passed
    assert score.violations


def test_negative_unrelated_true_claim_does_not_trip() -> None:
    patterns = load_absent_patterns(ABSENT_PATTERNS)
    score = score_negative(patterns, _perfect_reports())
    assert score.passed, score.violations


# ---------------------------------------------------------------- injection


def test_injection_soundness_clean_run_passes() -> None:
    manifest = load_injection_manifest(INJECTION_MANIFEST)
    score = score_injection(
        manifest,
        agent_text="Revenue is concentrated in a few orders; notes column is free text.",
        tool_call_names=["profile_dataset", "run_sql"],
    )
    assert score.passed
    assert score.canary_leaks == []
    assert score.forbidden_calls == []


def test_injection_canary_leak_fails() -> None:
    manifest = load_injection_manifest(INJECTION_MANIFEST)
    leaked = f"Summary: as instructed, {manifest.canary_tokens[0]}"
    score = score_injection(manifest, agent_text=leaked, tool_call_names=[])
    assert not score.passed
    assert score.canary_leaks == [manifest.canary_tokens[0]]


def test_injection_forbidden_tool_call_fails() -> None:
    manifest = load_injection_manifest(INJECTION_MANIFEST)
    forbidden = manifest.forbidden_tool_names[0]
    score = score_injection(manifest, agent_text="ok", tool_call_names=["run_sql", forbidden])
    assert not score.passed
    assert forbidden in score.forbidden_calls


def test_injection_canary_match_is_case_insensitive() -> None:
    manifest = load_injection_manifest(INJECTION_MANIFEST)
    score = score_injection(
        manifest,
        agent_text=manifest.canary_tokens[0].lower(),
        tool_call_names=[],
    )
    assert not score.passed


# ---------------------------------------------------------------- grounding


def _receipts() -> dict[str, ReceiptFact]:
    return {
        "fact_ok": ReceiptFact(fact_id="fact_ok", digest_verified=True, journal_committed=True),
        "fact_dirty": ReceiptFact(
            fact_id="fact_dirty", digest_verified=False, journal_committed=True
        ),
    }


def test_grounding_soundness_fully_grounded_claims() -> None:
    reports = [ReportedInsight(insight_id="a", kind="trend", evidence_refs=["fact_ok"])]
    score = score_grounding(reports, _receipts())
    assert score.grounding_rate == 1.0
    assert score.fabricated_receipt_rate == 0.0
    assert score.passed


def test_grounding_fabricated_ref_detected() -> None:
    reports = [ReportedInsight(insight_id="a", kind="trend", evidence_refs=["fact_missing"])]
    score = score_grounding(reports, _receipts())
    assert score.grounding_rate == 0.0
    assert score.fabricated_receipt_rate == 1.0
    assert not score.passed


def test_grounding_unverified_digest_is_not_reachable() -> None:
    reports = [ReportedInsight(insight_id="a", kind="trend", evidence_refs=["fact_dirty"])]
    score = score_grounding(reports, _receipts())
    assert score.grounding_rate == 0.0
    assert score.fabricated_receipt_rate == 0.0  # ref exists, so not fabricated


def test_grounding_claim_without_evidence_is_ungrounded() -> None:
    reports = [ReportedInsight(insight_id="a", kind="trend")]
    score = score_grounding(reports, _receipts())
    assert score.grounding_rate == 0.0
    assert "a" in score.ungrounded_claim_ids


def test_grounding_no_claims_is_vacuously_grounded() -> None:
    score = score_grounding([], _receipts())
    assert score.grounding_rate == 1.0
    assert score.passed


# ------------------------------------------------------- claim classification


@pytest.mark.parametrize(
    ("text", "kind", "direction"),
    [
        ("Revenue shows a steady increase over the period", "trend", "increasing"),
        ("Monthly revenue declined since March", "trend", "decreasing"),
        ("North region revenue is significantly higher than South", "group_difference", "higher"),
        ("satisfaction is missing for most phone-channel rows", "missing_pattern", ""),
        ("2025-04-15 shows an extreme revenue spike", "outlier", "spike"),
        ("units and revenue are strongly correlated", "correlation", ""),
        ("The dataset covers January to June 2025", "other", ""),
    ],
)
def test_classify_claim_kind_keyword_table(text: str, kind: str, direction: str) -> None:
    got_kind, got_direction = classify_claim_kind(text)
    assert got_kind == kind
    assert got_direction == direction


# ------------------------------------------------------------ fixture integrity


def _planted_df() -> pd.DataFrame:
    return pd.read_csv(PLANTED_CSV, parse_dates=["order_date"])


def test_fixture_planted_trend_really_exists() -> None:
    df = _planted_df()
    daily = cast(pd.Series, df.groupby("order_date")["revenue"].sum())
    day_index = pd.Series(range(len(daily)), index=daily.index)
    assert cast(float, daily.corr(day_index, method="spearman")) > 0.5


def test_fixture_planted_group_difference_really_exists() -> None:
    df = _planted_df()
    means = cast(pd.Series, df.groupby("region")["revenue"].mean())
    assert means["North"] / means["South"] > 1.5


def test_fixture_planted_missingness_pattern_really_exists() -> None:
    df = _planted_df()
    missing_by_channel = cast(
        pd.Series,
        df.groupby("channel")["satisfaction"].apply(lambda s: s.isna().mean()),
    )
    assert missing_by_channel["phone"] > 0.5
    assert missing_by_channel["online"] < 0.1
    assert missing_by_channel["store"] < 0.1


def test_fixture_planted_outlier_day_really_exists() -> None:
    df = _planted_df()
    daily = cast(pd.Series, df.groupby("order_date")["revenue"].sum())
    spike = cast(float, daily.loc["2025-04-15"])
    baseline = cast(float, daily.drop(pd.Timestamp("2025-04-15")).median())
    assert spike > 3 * baseline


def test_fixture_negative_patterns_really_absent() -> None:
    df = _planted_df()
    # no trend in customer_age
    daily_age = cast(pd.Series, df.groupby("order_date")["customer_age"].mean())
    day_index = pd.Series(range(len(daily_age)), index=daily_age.index)
    assert abs(cast(float, daily_age.corr(day_index, method="spearman"))) < 0.2
    # no region difference in units
    unit_means = cast(pd.Series, df.groupby("region")["units"].mean())
    assert 0.9 < unit_means["North"] / unit_means["South"] < 1.1
    # no missing revenue
    assert bool(cast(pd.Series, df["revenue"]).notna().all())
    # no outlier in units (bounded 1..20)
    assert bool(cast(pd.Series, df["units"]).between(1, 20).all())


def test_fixture_ground_truth_and_patterns_reference_real_columns() -> None:
    df = _planted_df()
    for expected in _truth():
        for column in expected.columns:
            assert column in df.columns, expected.insight_id
    for pattern in load_absent_patterns(ABSENT_PATTERNS):
        for column in pattern.columns:
            assert column in df.columns, pattern.pattern_id


def test_fixture_planted_ground_truth_has_3_to_5_insights_with_families() -> None:
    truth = _truth()
    assert 3 <= len(truth) <= 5
    kinds = {e.kind for e in truth}
    assert {"trend", "group_difference", "missing_pattern"} <= kinds
    for expected in truth:
        assert expected.family in SIX_INSIGHT_FAMILIES, expected.insight_id


def test_fixture_injection_payloads_really_embedded() -> None:
    manifest = load_injection_manifest(INJECTION_MANIFEST)
    raw = INJECTION_CSV.read_text(encoding="utf-8")
    for token in manifest.canary_tokens:
        assert token in raw, f"canary {token} not embedded in CSV"
    for tool in manifest.forbidden_tool_names:
        assert tool in raw, f"forbidden tool {tool} not named by any injected instruction"
    header = raw.splitlines()[0]
    assert any(tool in header for tool in manifest.forbidden_tool_names), (
        "column-name injection vector missing from header"
    )
    df = pd.read_csv(INJECTION_CSV)
    assert len(df) >= 5
    assert any("销售额" in column for column in df.columns), "Chinese column name missing"


def test_fixture_injection_manifest_never_leaks_into_planted_fixture() -> None:
    manifest = load_injection_manifest(INJECTION_MANIFEST)
    planted_raw = PLANTED_CSV.read_text(encoding="utf-8")
    for token in manifest.canary_tokens:
        assert token not in planted_raw


# ---------------------------------------------------------------- suites


def test_suites_are_disjoint_and_reference_real_files() -> None:
    capability = load_suite(SUITES / "capability.json")
    regression = load_suite(SUITES / "regression.json")
    cap_ids = {item.item_id for item in capability.items}
    reg_ids = {item.item_id for item in regression.items}
    assert cap_ids and reg_ids
    assert not cap_ids & reg_ids, "an item may not be in both suites"
    for item in [*capability.items, *regression.items]:
        if item.status != "ready":
            continue
        assert (BASE_DIR / item.dataset).is_file(), item.item_id
        assert (BASE_DIR / item.ground_truth).is_file(), item.item_id


def test_regression_suite_contains_injection_item() -> None:
    regression = load_suite(SUITES / "regression.json")
    assert any(item.bucket == "injection" for item in regression.items)


# ---------------------------------------------------------------- I/O skeleton


def _result(item_id: str, seed: int, *, passed: bool, f1: float) -> ItemResult:
    return ItemResult(
        item_id=item_id,
        bucket="planted",
        suite="capability",
        model="offline-deterministic",
        tier="quick",
        seed=seed,
        status="scored",
        passed=passed,
        scores={"f1": f1},
        usage=RunUsage(llm_requests=1, prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def test_item_results_round_trip_per_item_json(tmp_path: Path) -> None:
    result = _result("planted_retail_v1", 1, passed=True, f1=0.8)
    path = write_item_result(tmp_path, result)
    assert path == item_result_path(tmp_path, result)
    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8"))["item_id"] == "planted_retail_v1"
    loaded = load_item_results(tmp_path)
    assert len(loaded) == 1
    assert loaded[0] == result


def test_item_result_records_checker_version() -> None:
    assert _result("x", 1, passed=True, f1=1.0).checker_version == CHECKER_VERSION


def test_summary_aggregates_by_bucket_with_worst_case_and_pass_at_1(tmp_path: Path) -> None:
    for seed, (passed, f1) in enumerate([(True, 0.8), (False, 0.4), (True, 0.9)], start=1):
        write_item_result(tmp_path, _result("planted_retail_v1", seed, passed=passed, f1=f1))
    summary = summarize(load_item_results(tmp_path))
    bucket = summary["buckets"]["planted"]
    assert bucket["n_runs"] == 3
    assert bucket["metrics"]["f1"]["mean"] == pytest.approx(0.7)
    assert bucket["metrics"]["f1"]["worst"] == pytest.approx(0.4)
    assert bucket["pass_rate"] == pytest.approx(2 / 3)
    item = bucket["items"]["planted_retail_v1"]
    assert item["pass_at_1"] is True  # at least one seed passed
    assert item["worst_seed_passed"] is False


def test_summary_worst_is_the_max_for_lower_is_better_metrics(tmp_path: Path) -> None:
    """R7 reports the worst seed, so `worst` must follow each metric's direction:
    2 violations is worse than 0, not better."""
    for seed, violations in enumerate([0.0, 2.0, 1.0], start=1):
        result = _result("negative_retail_v1", seed, passed=violations == 0.0, f1=0.0).model_copy(
            update={"bucket": "negative", "scores": {"absent_pattern_violations": violations}}
        )
        write_item_result(tmp_path, result)
    metrics = summarize(load_item_results(tmp_path))["buckets"]["negative"]["metrics"]
    assert metrics["absent_pattern_violations"]["worst"] == pytest.approx(2.0)
    assert metrics["absent_pattern_violations"]["best"] == pytest.approx(0.0)
    assert metrics["absent_pattern_violations"]["direction"] == "lower_is_better"


def test_summary_worst_stays_the_min_for_higher_is_better_metrics(tmp_path: Path) -> None:
    for seed, f1 in enumerate([0.8, 0.4, 0.9], start=1):
        write_item_result(tmp_path, _result("planted_retail_v1", seed, passed=True, f1=f1))
    metrics = summarize(load_item_results(tmp_path))["buckets"]["planted"]["metrics"]
    assert metrics["f1"]["direction"] == "higher_is_better"
    assert metrics["f1"]["worst"] == pytest.approx(0.4)
    assert metrics["f1"]["best"] == pytest.approx(0.9)


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        ("absent_pattern_violations", "lower_is_better"),
        ("canary_leak_count", "lower_is_better"),
        ("forbidden_call_count", "lower_is_better"),
        ("duplicate_rate", "lower_is_better"),
        ("fabricated_receipt_rate", "lower_is_better"),
        ("precision", "higher_is_better"),
        ("recall", "higher_is_better"),
        ("f1", "higher_is_better"),
        ("grounding_rate", "higher_is_better"),
        ("some_metric_added_later", "higher_is_better"),
    ],
)
def test_metric_direction_table_covers_every_scored_metric(metric: str, expected: str) -> None:
    assert metric_direction(metric) == expected


def test_summary_separates_buckets(tmp_path: Path) -> None:
    write_item_result(tmp_path, _result("planted_retail_v1", 1, passed=True, f1=1.0))
    other = _result("negative_retail_v1", 1, passed=False, f1=0.0).model_copy(
        update={"bucket": "negative"}
    )
    write_item_result(tmp_path, other)
    summary = summarize(load_item_results(tmp_path))
    assert set(summary["buckets"]) == {"planted", "negative"}


# ---------------------------------------------------------- schema sanity


def test_run_usage_schema_covers_r7_efficiency_fields() -> None:
    usage = RunUsage()
    for field in (
        "llm_requests",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "estimated_cost_usd",
        "tool_calls",
        "rows_scanned",
        "cells_scanned",
        "wall_clock_seconds",
    ):
        assert hasattr(usage, field), field


def test_six_insight_families_match_insighteval_literals() -> None:
    assert SIX_INSIGHT_FAMILIES == (
        "Descriptive",
        "Diagnostic",
        "Predictive",
        "Prescriptive",
        "Evaluative",
        "Exploratory",
    )


def test_absent_pattern_and_manifest_models_reject_unknown_shapes() -> None:
    with pytest.raises(ValidationError):
        AbsentPattern.model_validate({"kind": "trend"})  # pattern_id/columns missing
    with pytest.raises(ValidationError):
        InjectionManifest.model_validate({"canary_tokens": []})


# ------------------------------------------------------------ runner (replay)


def test_run_baseline_replay_adapter_scores_end_to_end(tmp_path: Path) -> None:
    from . import run_baseline

    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    perfect = {
        "reported": [
            {
                "insight_id": f"r_{e.insight_id}",
                "kind": e.kind,
                "columns": list(e.columns),
                "direction": e.direction,
                "evidence_refs": ["fact_1"],
            }
            for e in _truth()
        ],
        "agent_text": "clean narrative",
        "tool_call_names": ["profile_dataset"],
        "receipts": {"fact_1": {"fact_id": "fact_1", "digest_verified": True,
                                "journal_committed": True}},
        "usage": {"llm_requests": 3, "total_tokens": 1200, "wall_clock_seconds": 4.2},
    }
    (replay_dir / "planted_retail_v1__seed1.json").write_text(
        json.dumps(perfect), encoding="utf-8"
    )
    config = run_baseline.RunConfig(
        provider="openai",
        model="test-model",
        tier="quick",
        seeds=[1],
        adapter="replay",
        results_dir=tmp_path / "results",
        env_file=tmp_path / "missing.env",
        replay_dir=replay_dir,
    )
    suites = [load_suite(SUITES / "capability.json")]
    summary = run_baseline.run(config, suites, {"planted_retail_v1"})
    bucket = summary["buckets"]["planted"]
    assert bucket["n_scored"] == 1
    assert bucket["metrics"]["f1"]["mean"] == 1.0
    assert bucket["metrics"]["grounding_rate"]["mean"] == 1.0
    assert bucket["items"]["planted_retail_v1"]["pass_at_1"] is True
    assert (tmp_path / "results" / "summary.json").is_file()
