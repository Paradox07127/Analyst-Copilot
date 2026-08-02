from __future__ import annotations

import re

import pytest

from eda_platform.schemas.anomaly import AnomalyOutlier, AnomalyScreenResult
from eda_platform.schemas.model_card import ModelCard
from eda_platform.schemas.questions import QuestionFinding
from eda_platform.schemas.stats import StatAssumptionCheck, StatTestResult, StatWarning
from eda_platform.tools.method_findings import (
    _stat_deviation,
    anomaly_findings,
    model_findings,
    stat_findings,
)

_NUMBER = re.compile(r"(?<![\w.])-?(?:\d+(?:\.\d+)?|\.\d+)")
_FORBIDDEN_CAUSAL = re.compile(r"\b(?:cause[ds]?|causing|causation|drives?|impacts?)\b", re.I)


def _assert_every_number_resolves(finding: QuestionFinding) -> None:
    evidence_numbers = [
        float(ref.value)
        for ref in finding.evidence
        if ref.value is not None
        and isinstance(ref.value, (int, float, str))
        and _is_number(ref.value)
    ]
    for match in _NUMBER.finditer(finding.text):
        displayed = float(match.group())
        prefix = finding.text[max(0, match.start() - 3) : match.start()]
        if "<" in prefix:
            assert any(0 <= value < displayed for value in evidence_numbers)
        else:
            assert any(value == pytest.approx(displayed) for value in evidence_numbers)
    assert all(ref.artifact_id == "artifact" and ref.locator for ref in finding.evidence)


def _is_number(value: object) -> bool:
    try:
        float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return True


def test_stat_finding_formats_small_p_and_labels_effect_with_warnings() -> None:
    result = StatTestResult(
        dataset_id="orders",
        test_type="one_way_anova",
        group_column="region",
        value_column="amount",
        statistic=8.125,
        p_value=0.0004,
        effect_size=0.08,
        sample_size=120,
        assumptions=[
            StatAssumptionCheck(
                name="variance_homogeneity",
                status="warn",
                statistic=4.2,
                p_value=0.02,
                message="Group variances need review.",
            )
        ],
        warnings=[
            StatWarning(code="low_cells", message="Expected frequency is below 5.")
        ],
    )

    finding = stat_findings(result, "artifact")[0]

    assert "p < 0.001" in finding.text
    assert "medium effect" in finding.text
    assert "Assumption warning: Group variances need review." in finding.text
    assert "Expected frequency is below 5." in finding.text
    assert any(ref.locator == "p_value" and ref.value == 0.0004 for ref in finding.evidence)
    _assert_every_number_resolves(finding)
    assert not _FORBIDDEN_CAUSAL.search(finding.text)


@pytest.mark.parametrize(
    ("test_type", "effect_size", "magnitude"),
    [
        ("independent_t_test", 0.2, "small"),
        ("chi_square_independence", 0.35, "medium"),
        ("kruskal_wallis", 0.2, "large"),
    ],
)
def test_stat_effect_magnitude_uses_test_specific_conventional_cutoffs(
    test_type: str, effect_size: float, magnitude: str
) -> None:
    result = StatTestResult(
        dataset_id="orders",
        test_type=test_type,  # type: ignore[arg-type]
        statistic=2.5,
        p_value=0.04,
        effect_size=effect_size,
        sample_size=40,
    )

    finding = stat_findings(result, "artifact")[0]

    assert f"{magnitude} effect" in finding.text
    _assert_every_number_resolves(finding)
    assert not _FORBIDDEN_CAUSAL.search(finding.text)


def test_fisher_exact_finding_names_the_odds_ratio() -> None:
    result = StatTestResult(
        dataset_id="orders",
        test_type="fisher_exact",
        statistic=0.0,
        p_value=0.01,
        effect_size=3.2,
        sample_size=18,
    )

    finding = stat_findings(result, "artifact")[0]

    assert "odds ratio 3.2" in finding.text
    _assert_every_number_resolves(finding)
    assert not _FORBIDDEN_CAUSAL.search(finding.text)


def _fisher_result(effect_size: float) -> StatTestResult:
    return StatTestResult(
        dataset_id="orders",
        test_type="fisher_exact",
        statistic=0.0,
        p_value=0.01,
        effect_size=effect_size,
        sample_size=18,
    )


def test_fisher_or_near_one_is_not_a_large_effect() -> None:
    finding = stat_findings(_fisher_result(1.05), "artifact")[0]

    assert "large effect" not in finding.text
    assert "small effect" in finding.text
    _assert_every_number_resolves(finding)


def test_fisher_protective_or_bins_like_its_reciprocal() -> None:
    protective = stat_findings(_fisher_result(0.2), "artifact")[0]
    harmful = stat_findings(_fisher_result(5.0), "artifact")[0]

    assert "large effect" in protective.text
    assert "large effect" in harmful.text


def test_fisher_or_two_stays_below_the_medium_cutoff() -> None:
    # Bins are small < 3.0 <= medium < 5.0 <= large, so OR 2.0 lands in small.
    finding = stat_findings(_fisher_result(2.0), "artifact")[0]

    assert "small effect" in finding.text


def test_fisher_or_zero_does_not_crash() -> None:
    finding = stat_findings(_fisher_result(0.0), "artifact")[0]

    assert "effect" in finding.text


def test_fisher_deviation_is_reciprocal_symmetric() -> None:
    protective = _stat_deviation(_fisher_result(0.2))
    harmful = _stat_deviation(_fisher_result(5.0))

    assert protective == pytest.approx(harmful)


def test_fisher_deviation_near_one_sits_near_the_floor() -> None:
    # Symmetric normalization floors fisher deviation at 1/large = 0.2.
    assert _stat_deviation(_fisher_result(1.05)) == pytest.approx(1.05 / 5.0)


def test_fisher_deviation_or_zero_does_not_crash() -> None:
    assert _stat_deviation(_fisher_result(0.0)) == 0.0


def test_model_finding_is_predictive_limited_and_all_metrics_are_evidenced() -> None:
    card = ModelCard(
        dataset_id="orders",
        task_type="regression",
        target_column="amount",
        feature_columns=["quantity"],
        split_strategy="random",
        train_rows=80,
        test_rows=20,
        model_type="baseline",
        metrics={"r2": 0.625, "rmse": 12.5},
    )

    finding = model_findings(card, "artifact")[0]

    assert "R squared 0.625" in finding.text
    assert "baseline estimate within stated performance, not a causal claim" in finding.text
    assert not _FORBIDDEN_CAUSAL.search(finding.text)
    _assert_every_number_resolves(finding)


def test_anomaly_finding_cites_each_number_and_carries_notes() -> None:
    result = AnomalyScreenResult(
        dataset_name="orders",
        column="amount",
        method="robust_zscore",
        threshold=3.5,
        total_rows=100,
        non_null_rows=80,
        outlier_count=20,
        outlier_percent=25.0,
        median=10,
        mad=2,
        q1=8,
        q3=12,
        top_outliers=[AnomalyOutlier(row_index=4, value=100, score=30.35)],
        notes=[
            "The flagged share may indicate distribution shift rather than "
            "individual anomalies."
        ],
    )

    finding = anomaly_findings(result, "artifact")[0]

    assert "distribution shift rather than individual anomalies" in finding.text
    _assert_every_number_resolves(finding)
    assert not _FORBIDDEN_CAUSAL.search(finding.text)
