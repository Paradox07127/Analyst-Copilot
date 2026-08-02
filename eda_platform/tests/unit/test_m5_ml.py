from typing import cast

import numpy as np
import pandas as pd

from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.auto_eda import _select_stat_test, run_auto_eda
from eda_platform.schemas.artifacts import ArtifactType, DatasetProfile
from eda_platform.schemas.model_card import ModelCard
from eda_platform.schemas.stats import StatTestResult
from eda_platform.tools.loader import load_csv
from eda_platform.tools.ml_baseline import (
    _detect_time_column,
    _infer_task_type,
    _is_id_like,
    _looks_like_target_leakage,
    _PreprocessEncoder,
    create_model_card_artifact,
    run_baseline_model,
)
from eda_platform.tools.profiler import profile_dataset


def test_auto_stat_selector_prefers_verified_value_over_sequence(tmp_path) -> None:
    csv_path = tmp_path / "payments.csv"
    rows = ["payment_type,payment_sequential,payment_installments,payment_value"]
    payment_types = ("credit_card", "boleto", "voucher", "debit_card")
    for index in range(80):
        rows.append(
            f"{payment_types[index % 4]},{1 + index % 3},{1 + index % 6},{20 + index * 2.5}"
        )
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    loaded = load_csv(csv_path, dataset_id="ds_payments")
    profile_artifact = profile_dataset(loaded, project_id="project", session_id="run")
    profile = DatasetProfile.model_validate(profile_artifact.payload)

    spec = _select_stat_test(loaded, profile)

    assert spec is not None
    assert spec.test_type == "welch_anova"
    assert spec.group_column == "payment_type"
    assert spec.value_column == "payment_value"


def test_auto_stat_selector_rejects_numeric_postal_code_without_measure(tmp_path) -> None:
    csv_path = tmp_path / "customers.csv"
    rows = ["customer_state,customer_zip_code_prefix"]
    states = ("SP", "RJ", "MG", "BA")
    for index in range(40):
        rows.append(f"{states[index % 4]},{10000 + index}")
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    loaded = load_csv(csv_path, dataset_id="ds_customers")
    profile_artifact = profile_dataset(loaded, project_id="project", session_id="run")
    profile = DatasetProfile.model_validate(profile_artifact.payload)

    assert _select_stat_test(loaded, profile) is None


def test_classification_baseline_excludes_target_leakage_feature() -> None:
    frame = pd.DataFrame(
        {
            "customer_id": [f"C{i:03d}" for i in range(80)],
            "spend": [float(i % 20) for i in range(80)],
            "visits": [i % 7 + 1 for i in range(80)],
            "churned": [1 if i % 5 in {0, 1} else 0 for i in range(80)],
        }
    )
    frame["churned_copy"] = frame["churned"]

    card = run_baseline_model(frame, dataset_id="ds_customers", target_column="churned")
    artifact = create_model_card_artifact(
        card,
        project_id="project_demo",
        session_id="run_demo",
        parents=["prof_123"],
    )
    restored = ModelCard.model_validate(artifact.payload)

    assert card.task_type == "classification"
    assert "churned_copy" not in card.feature_columns
    assert "customer_id" not in card.feature_columns
    assert any(check.code == "target_leakage" for check in card.leakage_checks)
    assert any(check.code == "id_like_feature" for check in card.leakage_checks)
    assert {"accuracy", "f1_weighted"}.issubset(card.metrics)
    assert card.model_type
    assert artifact.type is ArtifactType.MODEL_CARD
    assert restored.target_column == "churned"


def test_time_series_regression_uses_time_ordered_split() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=90, freq="D"),
            "price": [100 + i * 0.5 for i in range(90)],
            "promo": [1 if i % 10 == 0 else 0 for i in range(90)],
            "sales": [200 + i * 1.5 + (10 if i % 10 == 0 else 0) for i in range(90)],
        }
    )

    card = run_baseline_model(
        frame,
        dataset_id="ds_sales",
        target_column="sales",
        time_column="date",
    )

    assert card.task_type == "regression"
    assert card.split_strategy == "time_ordered"
    assert "date" not in card.feature_columns
    assert {"r2", "mae", "rmse"}.issubset(card.metrics)
    assert any(check.code == "time_ordered_split" for check in card.leakage_checks)
    assert card.train_rows > card.test_rows


def test_auto_eda_can_emit_m5_stat_and_model_artifacts(tmp_path) -> None:
    csv_path = tmp_path / "customers.csv"
    pd.DataFrame(
        {
            "segment": ["A"] * 40 + ["B"] * 40,
            "spend": [float(i % 20) for i in range(80)],
            "visits": [i % 6 + 1 for i in range(80)],
            "churned": [1 if i % 5 in {0, 1} else 0 for i in range(80)],
        }
    ).to_csv(csv_path, index=False)

    result = run_auto_eda(
        [csv_path],
        workspace=tmp_path / "workspace",
        project_id="project_demo",
        session_id="run_demo",
        ml_target_column="churned",
    )
    stat_artifacts = [
        artifact for artifact in result.artifacts if artifact.type is ArtifactType.STAT_TEST_RESULT
    ]
    model_artifacts = [
        artifact for artifact in result.artifacts if artifact.type is ArtifactType.MODEL_CARD
    ]

    assert stat_artifacts
    assert model_artifacts
    _stat_p = StatTestResult.model_validate(stat_artifacts[0].payload).p_value
    assert _stat_p is not None and _stat_p >= 0.0
    assert ModelCard.model_validate(model_artifacts[0].payload).target_column == "churned"


def test_auto_eda_skips_ml_baseline_when_target_has_too_few_rows(tmp_path) -> None:
    csv_path = tmp_path / "tiny.csv"
    pd.DataFrame(
        {
            "segment": ["A", "B", "A"],
            "spend": [10.0, 20.0, 30.0],
            "target": [0, 1, 0],
        }
    ).to_csv(csv_path, index=False)

    result = run_auto_eda(
        [csv_path],
        workspace=tmp_path / "workspace",
        project_id="project_demo",
        session_id="run_tiny",
        ml_target_column="target",
    )
    store = ArtifactStore(tmp_path / "workspace")
    events = store.list_trace_events(project_id="project_demo", session_id="run_tiny")

    assert result.report_markdown
    assert not any(artifact.type is ArtifactType.MODEL_CARD for artifact in result.artifacts)
    assert any(
        event.event_type == "ml_baseline_skipped"
        and event.name == "run_baseline_model"
        and "At least 10 labeled rows" in str(event.summary.get("reason", ""))
        for event in events
    )


def test_auto_eda_traces_a_skipped_stat_test(tmp_path) -> None:
    """A ValueError out of run_stat_test used to return [] silently, so an
    auto-selected comparison could vanish with nothing in the trace."""
    csv_path = tmp_path / "customers.csv"
    frame = pd.DataFrame(
        {
            "segment": ["A"] * 40 + ["B"] * 40,
            "spend": [float(i % 20) for i in range(80)],
            "visits": [i % 6 + 1 for i in range(80)],
            "churned": [1 if i % 5 in {0, 1} else 0 for i in range(80)],
        }
    )
    # Group B has no numeric spend at all, so the two-group test loses a group.
    frame.loc[frame["segment"] == "B", "spend"] = None
    frame.to_csv(csv_path, index=False)

    result = run_auto_eda(
        [csv_path],
        workspace=tmp_path / "workspace",
        project_id="project_demo",
        session_id="run_skip",
    )
    store = ArtifactStore(tmp_path / "workspace")
    events = store.list_trace_events(project_id="project_demo", session_id="run_skip")

    assert not any(
        artifact.type is ArtifactType.STAT_TEST_RESULT for artifact in result.artifacts
    )
    assert any(
        event.event_type == "stat_test_skipped"
        and event.name == "run_stat_tests"
        and event.summary.get("group_column") == "segment"
        and "exactly two groups" in str(event.summary.get("reason", ""))
        for event in events
    )


# --------------------------------------------------------------------------------------
# Adversarial regression tests for the M5 part-1 leakage-guard findings (ML-1..ML-7).
# Each test fails on the pre-fix code (commit cb4fb1f) and passes after the fix.
# --------------------------------------------------------------------------------------


def test_ml1_categorical_one_to_one_proxy_is_detected_and_excluded() -> None:
    # ML-1: a categorical feature that maps 1:1 to the class label with DIFFERENT strings
    # coerces to NaN under the numeric-only guard and was silently declared clean.
    n = 80
    target = pd.Series([i % 2 for i in range(n)])
    frame = pd.DataFrame(
        {
            "noise": [float(i % 7) for i in range(n)],
            "diagnosis": target.map({0: "benign", 1: "malignant"}),
            "label": target,
        }
    )

    diagnosis = cast(pd.Series, frame["diagnosis"])
    label = cast(pd.Series, frame["label"])
    assert _looks_like_target_leakage(diagnosis, label, task_type="classification")
    card = run_baseline_model(frame, dataset_id="ds_cat", target_column="label")
    assert "diagnosis" in card.excluded_features
    assert "diagnosis" not in card.feature_columns
    leaks = [c for c in card.leakage_checks if c.code == "target_leakage"]
    assert any(c.column == "diagnosis" and c.action == "excluded" for c in leaks)
    # The ModelCard must NOT certify "passed" for the leaking column.
    assert not any(c.action == "passed" for c in leaks)


def test_ml1_categorical_leakage_for_numeric_target_via_correlation_ratio() -> None:
    # ML-1 (regression arm): a categorical feature that fully determines a numeric target
    # (correlation ratio eta ~ 1) must be flagged even though it is not numeric.
    n = 60
    groups = pd.Series([f"g{i % 6}" for i in range(n)])
    target = groups.map({f"g{k}": float(k * 100) for k in range(6)})
    assert _looks_like_target_leakage(groups, target, task_type="regression")


def test_ml2_datetime_column_detected_regardless_of_name() -> None:
    # ML-2: a real datetime64 column named 'created' (no date/time substring) was skipped
    # before parsing, silently falling back to a random split on temporal data.
    n = 100
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    for name in ["created", "observed_at", "period"]:
        frame = pd.DataFrame({name: dates, "x": range(n)})
        assert _detect_time_column(frame, exclude=set()) == name

    rng = np.random.default_rng(1)
    frame = pd.DataFrame(
        {
            "created": dates,
            "feat": rng.normal(size=n),
            "target": rng.normal(size=n) * 3.0 + 5.0,
        }
    )
    # time_column=None on purpose: detection must fire internally.
    card = run_baseline_model(frame, dataset_id="ds_time", target_column="target")
    assert card.split_strategy == "time_ordered"
    assert "created" not in card.feature_columns
    time_checks = [c for c in card.leakage_checks if c.code == "time_ordered_split"]
    assert time_checks and "enforced" in time_checks[0].message.lower()


def test_ml2_object_dates_parse_but_numeric_epoch_is_not_time() -> None:
    n = 100
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    iso = pd.DataFrame({"stamp": dates.strftime("%Y-%m-%d"), "x": range(n)})
    assert _detect_time_column(iso, exclude=set()) == "stamp"
    # Epoch integers are indistinguishable from an ordinary integer feature -> not a time
    # column (avoids hijacking numeric features), even when named 'timestamp'.
    epoch = pd.DataFrame({"timestamp": (dates.astype("int64") // 10**9), "x": range(n)})
    assert _detect_time_column(epoch, exclude=set()) is None


def test_ml3_preprocessing_is_fit_on_train_only() -> None:
    # ML-3 (A): an early-train NaN must be imputed with the TRAIN-only median, and a test
    # NaN must be imputed with the train median -- never a statistic that saw test rows.
    train = pd.DataFrame({"f": [1.0, 1.0, 1.0, np.nan, 1.0, 1.0, 1.0, 1.0]})
    test = pd.DataFrame({"f": [1000.0, 1000.0, np.nan]})
    encoder = _PreprocessEncoder.fit(train)
    x_train = encoder.transform(train)
    x_test = encoder.transform(test)
    assert x_train["f"].iloc[3] == 1.0  # train NaN -> train median (1.0), not global
    assert x_test["f"].iloc[2] == 1.0  # test NaN -> train median (1.0), not 1000

    # ML-3 (B): a category value present ONLY in test rows must not create a global column.
    train_c = pd.DataFrame({"color": ["red"] * 9})
    test_c = pd.DataFrame({"color": ["blue", "blue"]})
    enc_c = _PreprocessEncoder.fit(train_c)
    assert not any("blue" in col for col in enc_c.columns)
    x_test_c = enc_c.transform(test_c)
    assert list(x_test_c.columns) == list(enc_c.columns)
    assert bool((x_test_c.to_numpy() == 0.0).all())  # unknown category -> all-zero row


def test_ml3_end_to_end_test_only_category_does_not_leak_columns() -> None:
    # End-to-end: the model's feature universe is fixed by train; a test-only category
    # value does not add a column, so train never "sees" a future category.
    n = 60
    rng = np.random.default_rng(3)
    colors = ["red"] * 48 + ["blue"] * 12  # 'blue' concentrated at the tail
    frame = pd.DataFrame(
        {
            "when": pd.date_range("2021-01-01", periods=n, freq="D"),
            "color": colors,
            "value": rng.normal(size=n) + np.arange(n) * 0.1,
        }
    )
    card = run_baseline_model(frame, dataset_id="ds_cat_time", target_column="value")
    encoded_features = [fi.feature for fi in card.feature_importance]
    # With an 80/20 time split, 'blue' lands only in test -> no color=blue feature exists.
    assert not any("blue" in feat for feat in encoded_features)


def test_ml4_classification_numeric_proxy_below_old_threshold_is_flagged() -> None:
    # ML-4: corr ~0.9946 slipped past the old 0.999 classification threshold. The new
    # classification threshold must be no stricter than regression's.
    rng = np.random.default_rng(0)
    target = pd.Series(rng.integers(0, 2, size=200))
    proxy = target + rng.normal(0, 0.05, size=200)
    corr = float(pd.DataFrame({"a": proxy, "b": target}).corr().iloc[0, 1])
    assert 0.99 <= corr < 0.999  # the exact regime that used to escape
    assert _looks_like_target_leakage(proxy, target, task_type="classification")


def test_ml5_wide_range_integer_target_stays_regression() -> None:
    # ML-5: a 1..10 satisfaction score has too many uniques relative to rows to be classes.
    score = pd.Series([(i % 10) + 1 for i in range(120)])
    assert _infer_task_type(score) == "regression"
    # A genuine 2-class label stays classification.
    binary = pd.Series([i % 2 for i in range(120)])
    assert _infer_task_type(binary) == "classification"
    # End-to-end: the score target yields regression metrics, never class metrics.
    rng = np.random.default_rng(5)
    frame = pd.DataFrame({"feat": rng.normal(size=120), "score": score})
    card = run_baseline_model(frame, dataset_id="ds_score", target_column="score")
    assert card.task_type == "regression"
    assert {"r2", "mae", "rmse"}.issubset(card.metrics)
    assert "accuracy" not in card.metrics


def test_ml6_numeric_high_cardinality_id_is_excluded() -> None:
    # ML-6: a numeric surrogate key (unique per row) was never dropped.
    assert _is_id_like(pd.Series(range(1000, 1100)), "account_number")
    assert _is_id_like(pd.Series(range(5000, 5100)), "record_no")
    # A continuous float measurement that is also near-unique must NOT be treated as an ID.
    assert not _is_id_like(pd.Series(np.random.default_rng(0).normal(size=100)), "sensor")

    n = 100
    rng = np.random.default_rng(0)
    target = pd.Series([i % 2 for i in range(n)])
    frame = pd.DataFrame(
        {
            "record_no": range(5000, 5000 + n),
            "real_feat": rng.normal(size=n),
            "label": target,
        }
    )
    card = run_baseline_model(frame, dataset_id="ds_id", target_column="label")
    assert "record_no" in card.excluded_features
    assert "record_no" not in card.feature_columns


# --------------------------------------------------------------------------------------
# A4: the ModelCard must disclose the majority-class baseline so a 0.9996 accuracy on a
# 0.17%-positive dataset cannot masquerade as skill.
# --------------------------------------------------------------------------------------


def test_a4_imbalanced_target_discloses_majority_class_baseline() -> None:
    n = 200
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        {
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
            "label": [1 if i < 4 else 0 for i in range(n)],  # 2% positives
        }
    )
    card = run_baseline_model(frame, dataset_id="ds_imbalanced", target_column="label")
    assert card.baseline_accuracy is not None
    assert card.baseline_accuracy >= 0.9
    assert any("Majority-class baseline accuracy" in item for item in card.limitations)


def test_a4_balanced_informative_target_omits_disclosure() -> None:
    n = 200
    rng = np.random.default_rng(11)
    target = pd.Series([i % 2 for i in range(n)])
    frame = pd.DataFrame(
        {
            "signal": target * 2.0 + rng.normal(0, 0.4, size=n),
            "noise": rng.normal(size=n),
            "label": target,
        }
    )
    card = run_baseline_model(frame, dataset_id="ds_balanced", target_column="label")
    assert card.baseline_accuracy is not None
    assert abs(card.baseline_accuracy - 0.5) < 0.05
    # Clear lift over the baseline AND a balanced target -> no disclosure sentence.
    assert card.metrics["accuracy"] - card.baseline_accuracy >= 0.02
    assert not any("Majority-class baseline" in item for item in card.limitations)


def test_a4_legacy_model_card_payload_without_baseline_field_deserializes() -> None:
    payload = {
        "dataset_id": "ds_legacy",
        "task_type": "classification",
        "target_column": "y",
        "feature_columns": ["x"],
        "split_strategy": "random",
        "train_rows": 8,
        "test_rows": 2,
        "model_type": "RandomForestClassifier",
        "metrics": {"accuracy": 0.9},
    }
    card = ModelCard.model_validate(payload)
    assert card.baseline_accuracy is None


def test_ml7_no_nan_indicator_column_when_no_missing_values() -> None:
    # ML-7: dummy_na=True fabricated a _nan column even with zero missing values.
    clean = pd.DataFrame({"c": list("ababababab")})
    encoder = _PreprocessEncoder.fit(clean)
    assert not any("<NA>" in col or "nan" in col.lower() for col in encoder.columns)
    assert set(encoder.columns) == {"c='a'", "c='b'"}
    # When the train column DOES contain NaN, the indicator is present.
    dirty = pd.DataFrame({"c": ["a", "b", None, "a", "b", "a", "b", "a", "b", "a"]})
    encoder_dirty = _PreprocessEncoder.fit(dirty)
    assert any("<NA>" in col for col in encoder_dirty.columns)
