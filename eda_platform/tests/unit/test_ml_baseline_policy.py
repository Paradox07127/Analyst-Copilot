"""E1.5 ML hardening: split policy, group/time-aware CV, PR-AUC/calibration,
and aggregated signed permutation importance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eda_platform.core.tool_guard import ToolGuardError
from eda_platform.tools.ml_baseline import (
    _aggregate_importances,
    _PreprocessEncoder,
    _split_indexes,
    run_baseline_model,
)


def _time_frame(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=n, freq="D"),
            "feat": rng.normal(size=n),
            "target": np.arange(n) * 0.7 + rng.normal(0, 0.5, size=n),
        }
    )


def _group_frame(n: int = 90) -> pd.DataFrame:
    rng = np.random.default_rng(4)
    signal = rng.normal(size=n)
    return pd.DataFrame(
        {
            "site": [f"s{i % 6}" for i in range(n)],
            "signal": signal,
            "noise": rng.normal(size=n),
            "label": (signal > 0).astype(int),
        }
    )


# ---------------------------------------------------------------------------
# Split policy
# ---------------------------------------------------------------------------


def test_auto_policy_still_time_orders_temporal_data() -> None:
    card = run_baseline_model(
        _time_frame(), dataset_id="ds", target_column="target", time_column="date"
    )
    assert card.split_strategy == "time_ordered"


def test_random_policy_on_time_data_emits_explicit_warning_field() -> None:
    card = run_baseline_model(
        _time_frame(),
        dataset_id="ds",
        target_column="target",
        time_column="date",
        split_policy="random",
    )
    assert card.split_strategy != "time_ordered"
    warned = [c for c in card.leakage_checks if c.code == "random_split_on_time_data"]
    assert warned and warned[0].action == "warned" and warned[0].severity == "warn"
    assert warned[0].column == "date"


def test_time_policy_without_a_time_column_is_rejected() -> None:
    frame = _group_frame()
    with pytest.raises(ValueError, match="time"):
        run_baseline_model(
            frame, dataset_id="ds", target_column="label", split_policy="time"
        )


def test_group_policy_requires_group_column() -> None:
    with pytest.raises(ToolGuardError, match="group_column"):
        run_baseline_model(
            _group_frame(), dataset_id="ds", target_column="label", split_policy="group"
        )
    with pytest.raises(ToolGuardError):
        run_baseline_model(
            _group_frame(),
            dataset_id="ds",
            target_column="label",
            split_policy="nope",  # type: ignore[arg-type]
        )


def test_group_policy_splits_disjoint_groups_and_excludes_group_column() -> None:
    frame = _group_frame()
    card = run_baseline_model(
        frame,
        dataset_id="ds",
        target_column="label",
        split_policy="group",
        group_column="site",
    )
    assert card.split_strategy == "group"
    assert "site" not in card.feature_columns
    assert any(c.code == "group_split" and c.column == "site" for c in card.leakage_checks)

    groups = frame["site"]
    train_idx, test_idx = _split_indexes(
        frame["label"],
        task_type="classification",
        split_strategy="group",
        random_state=0,
        groups=groups,
    )
    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])
    assert train_groups and test_groups
    assert not train_groups & test_groups


# ---------------------------------------------------------------------------
# Policy-aware cross-validation
# ---------------------------------------------------------------------------


def test_time_policy_cv_reports_mean_and_std() -> None:
    card = run_baseline_model(
        _time_frame(),
        dataset_id="ds",
        target_column="target",
        time_column="date",
        split_policy="time",
        cv_folds=3,
    )
    assert {"cv_r2_mean", "cv_r2_std", "cv_folds"} <= set(card.metrics)
    assert card.metrics["cv_folds"] == 3.0


def test_group_policy_cv_reports_accuracy() -> None:
    card = run_baseline_model(
        _group_frame(),
        dataset_id="ds",
        target_column="label",
        split_policy="group",
        group_column="site",
        cv_folds=3,
    )
    assert {"cv_accuracy_mean", "cv_accuracy_std", "cv_folds"} <= set(card.metrics)
    assert 0.0 <= card.metrics["cv_accuracy_mean"] <= 1.0


# ---------------------------------------------------------------------------
# Classification: PR-AUC + calibration (Brier)
# ---------------------------------------------------------------------------


def test_binary_classification_reports_pr_auc_and_brier() -> None:
    card = run_baseline_model(_group_frame(), dataset_id="ds", target_column="label")
    assert {"pr_auc", "brier"} <= set(card.metrics)
    assert 0.0 <= card.metrics["pr_auc"] <= 1.0
    assert 0.0 <= card.metrics["brier"] <= 1.0


# ---------------------------------------------------------------------------
# Permutation importance: one-hot aggregation, sign, std
# ---------------------------------------------------------------------------


def test_aggregate_importances_sums_one_hot_and_keeps_sign_and_std() -> None:
    importances = np.array(
        [
            [0.20, 0.30],  # color='a'
            [-0.10, -0.20],  # color='b'
            [-0.05, -0.15],  # num
        ]
    )
    rows = _aggregate_importances(
        importances,
        encoded_columns=["color='a'", "color='b'", "num"],
        origin_by_encoded={"color='a'": "color", "color='b'": "color", "num": "num"},
    )
    by_feature = {row["feature"]: row for row in rows}
    assert by_feature["color"]["importance_mean"] == pytest.approx(0.10)
    assert by_feature["color"]["importance_std"] == pytest.approx(0.0)
    # Sign must be preserved, not clipped to zero.
    assert by_feature["num"]["importance_mean"] == pytest.approx(-0.10)
    assert by_feature["num"]["importance_std"] == pytest.approx(0.05)


def test_model_card_importance_uses_original_feature_names() -> None:
    n = 80
    rng = np.random.default_rng(9)
    color = pd.Series([("a", "b", "c")[i % 3] for i in range(n)])
    frame = pd.DataFrame(
        {
            "color": color,
            "noise": rng.normal(size=n),
            "label": (color == "a").astype(int) ^ (rng.random(n) < 0.05).astype(int),
        }
    )
    card = run_baseline_model(frame, dataset_id="ds", target_column="label")
    features = [fi.feature for fi in card.feature_importance]
    assert features, "importance rows must exist"
    assert all("=" not in feature for feature in features)
    assert "color" in features


def test_encoder_exposes_origin_of_every_encoded_column() -> None:
    frame = pd.DataFrame(
        {"color": ["a", "b", None, "a"], "amount": [1.0, 2.0, 3.0, 4.0]}
    )
    encoder = _PreprocessEncoder.fit(frame)
    origin = encoder.origin_by_encoded()
    assert set(origin) == set(encoder.columns)
    assert origin["amount"] == "amount"
    assert all(value == "color" for key, value in origin.items() if key != "amount")
