"""Offline golden correctness checks for the three investigation methods."""

from __future__ import annotations

import numpy as np
import pandas as pd

from eda_platform.tools.anomaly import screen_anomalies
from eda_platform.tools.ml_baseline import run_baseline_model
from eda_platform.tools.stat_tests import run_stat_test


def test_group_comparison_detects_known_difference_and_null() -> None:
    rng = np.random.default_rng(20260717)
    group_a = rng.normal(10.0, 1.0, 60)
    group_b = rng.normal(14.0, 1.0, 60)
    frame = pd.DataFrame(
        {
            "group": ["a"] * 60 + ["b"] * 60,
            "value": np.concatenate([group_a, group_b]),
        }
    )

    result = run_stat_test(
        frame,
        dataset_id="golden_group_difference",
        test_type="independent_t_test",
        group_column="group",
        value_column="value",
    )

    assert result.p_value is not None and result.p_value < 0.001
    assert result.statistic is not None and result.statistic < 0
    # Signed Cohen's d (R4): direction matches the negative t statistic.
    assert result.effect_size is not None and -5.5 <= result.effect_size <= -2.5

    null_frame = pd.DataFrame(
        {
            "group": ["a"] * 60 + ["b"] * 60,
            "value": np.concatenate([group_a, group_a.copy()]),
        }
    )
    null_result = run_stat_test(
        null_frame,
        dataset_id="golden_group_null",
        test_type="independent_t_test",
        group_column="group",
        value_column="value",
    )
    assert null_result.p_value is not None and null_result.p_value >= 0.05


def test_outcome_prediction_separates_target_and_detects_leakage() -> None:
    rng = np.random.default_rng(20260717)
    feature = rng.normal(0.0, 1.0, 500)
    target = (feature > 0.0).astype(int)
    frame = pd.DataFrame(
        {
            "signal": feature,
            "noise": rng.normal(0.0, 1.0, len(feature)),
            "target": target,
        }
    )

    card = run_baseline_model(
        frame,
        dataset_id="golden_separable_target",
        target_column="target",
        random_state=20260717,
    )

    assert card.task_type == "classification"
    assert card.metrics["accuracy"] >= 0.9
    assert any(
        check.code == "target_leakage" and check.action == "passed"
        for check in card.leakage_checks
    )

    leakage_frame = frame.assign(target_copy=target)
    leakage_card = run_baseline_model(
        leakage_frame,
        dataset_id="golden_leaking_target",
        target_column="target",
        random_state=20260717,
    )
    assert "target_copy" in leakage_card.excluded_features
    assert any(
        check.code == "target_leakage"
        and check.column == "target_copy"
        and check.action == "excluded"
        for check in leakage_card.leakage_checks
    )


def test_anomaly_detection_finds_planted_outliers() -> None:
    rng = np.random.default_rng(1)
    values = np.concatenate([rng.normal(0.0, 1.0, 100), [12.0, 15.0, -14.0]])
    frame = pd.DataFrame({"value": values})

    result = screen_anomalies(
        frame,
        dataset_name="golden_anomalies",
        column="value",
    )

    assert {outlier.row_index for outlier in result.top_outliers} == {100, 101, 102}
    assert result.outlier_count >= 3
