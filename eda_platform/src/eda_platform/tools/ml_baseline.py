from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_float_dtype,
    is_numeric_dtype,
)

from eda_platform.core.ids import make_artifact_id
from eda_platform.core.provenance import code_ref
from eda_platform.core.tool_guard import (
    GuardViolation,
    check_column_exists,
    check_column_semantic_type,
    check_non_empty,
    check_range,
    raise_for_violations,
)
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.model_card import (
    FeatureImportance,
    LeakageCheck,
    ModelCard,
    SplitStrategy,
    TaskType,
)

# Association thresholds for target-leakage detection. A feature at/above these is
# treated as a near-perfect proxy of the target and excluded from the baseline.
_LEAK_MI_THRESHOLD = 0.9  # adjusted/normalized mutual information (categorical vs class)
_LEAK_PURITY_THRESHOLD = 0.99  # majority-class purity of feature groups vs class target
_LEAK_ETA_THRESHOLD = 0.995  # correlation ratio (categorical feature vs numeric target)
_LEAK_CORR_CLS_THRESHOLD = 0.99  # numeric-proxy Pearson for classification (ML-4)
_LEAK_CORR_REG_THRESHOLD = 0.995  # numeric-proxy Pearson for regression


def run_baseline_model(
    frame: pd.DataFrame,
    *,
    dataset_id: str,
    target_column: str,
    time_column: str | None = None,
    random_state: int = 42,
) -> ModelCard:
    guard_baseline_model_params(
        frame,
        target_column=target_column,
        time_column=time_column,
        random_state=random_state,
    )
    _ensure_sklearn()
    if target_column not in frame.columns:
        raise ValueError(f"Target column not found: {target_column}")

    task_type = _infer_task_type(_series(frame, target_column))
    detected_time_column = time_column or _detect_time_column(frame, exclude={target_column})
    leakage_checks: list[LeakageCheck] = []
    feature_columns, excluded = _select_features(
        frame,
        target_column=target_column,
        task_type=task_type,
        time_column=detected_time_column,
        leakage_checks=leakage_checks,
    )
    if not feature_columns:
        raise ValueError("No usable feature columns remain after leakage checks")

    working = _frame(frame, [*feature_columns, target_column]).copy()
    if detected_time_column is not None and detected_time_column in frame.columns:
        working[detected_time_column] = pd.to_datetime(
            frame[detected_time_column],
            errors="coerce",
            format="mixed",
        )
        working = cast(pd.DataFrame, working.sort_values(by=detected_time_column))
        leakage_checks.append(
            LeakageCheck(
                code="time_ordered_split",
                severity="info",
                column=detected_time_column,
                action="passed",
                message=(
                    "Temporal ordering was enforced: rows are split by time order so the "
                    "model never trains on rows that occur after the test window."
                ),
            )
        )
        split_strategy: SplitStrategy = "time_ordered"
    elif task_type == "classification":
        split_strategy = "random_stratified"
    else:
        split_strategy = "random"

    working = cast(pd.DataFrame, working.dropna(subset=[target_column]))
    raw_features = _frame(working, feature_columns)
    y = _series(working, target_column)
    train_idx, test_idx = _split_indexes(
        y,
        task_type=task_type,
        split_strategy=split_strategy,
        random_state=random_state,
    )
    # ML-3/ML-7: preprocessing is fit on the TRAIN rows only, then applied to test.
    raw_train = cast(pd.DataFrame, raw_features.iloc[train_idx])
    raw_test = cast(pd.DataFrame, raw_features.iloc[test_idx])
    encoder = _PreprocessEncoder.fit(raw_train)
    X_train = encoder.transform(raw_train)
    X_test = encoder.transform(raw_test)
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    if task_type == "classification":
        model, metrics = _fit_classification(
            X_train,
            X_test,
            y_train,
            y_test,
            random_state=random_state,
        )
    else:
        model, metrics = _fit_regression(
            X_train,
            X_test,
            cast(pd.Series, pd.to_numeric(y_train, errors="coerce")),
            cast(pd.Series, pd.to_numeric(y_test, errors="coerce")),
            random_state=random_state,
        )

    limitations = [
        "This is a thin deterministic baseline, not an optimized production model.",
        "Metrics should be interpreted with the recorded leakage checks and split strategy.",
    ]
    baseline_accuracy: float | None = None
    if task_type == "classification":
        baseline_accuracy = _majority_class_baseline(y_train, metrics, limitations)

    return ModelCard(
        dataset_id=dataset_id,
        task_type=task_type,
        target_column=target_column,
        feature_columns=feature_columns,
        excluded_features=excluded,
        split_strategy=split_strategy,
        train_rows=int(len(train_idx)),
        test_rows=int(len(test_idx)),
        model_type=type(model).__name__,
        metrics=metrics,
        baseline_accuracy=baseline_accuracy,
        leakage_checks=leakage_checks,
        feature_importance=_feature_importance(model, list(X_train.columns)),
        limitations=limitations,
    )


def _majority_class_baseline(
    y_train: pd.Series,
    metrics: dict[str, float],
    limitations: list[str],
) -> float:
    """Train-set majority-class accuracy; discloses when accuracy alone is misleading."""
    class_counts = y_train.value_counts()
    total = float(len(y_train))
    baseline = round(float(class_counts.iloc[0]) / total, 6)
    minority_share = float(class_counts.iloc[-1]) / total
    accuracy = metrics.get("accuracy")
    if accuracy is not None and (accuracy - baseline < 0.02 or minority_share < 0.05):
        limitations.append(
            f"Majority-class baseline accuracy is {baseline}; accuracy alone is "
            "not informative for this class balance."
        )
    return baseline


def guard_baseline_model_params(
    frame: pd.DataFrame,
    *,
    target_column: Any,
    time_column: Any = None,
    random_state: Any = 42,
) -> None:
    violations: list[GuardViolation | None] = [
        check_non_empty("target_column", target_column),
        check_range(
            "random_state",
            random_state,
            minimum=0.0,
            maximum=4_294_967_295.0,
            fix_hint=(
                "Set `random_state` to an integer seed from 0 through 4294967295."
            ),
        ),
    ]
    if _is_non_empty_string(target_column):
        violations.append(
            check_column_exists(
                "target_column",
                target_column,
                [str(column) for column in frame.columns],
            )
        )
    if time_column is not None:
        violations.append(check_non_empty("time_column", time_column))
        if _is_non_empty_string(time_column):
            violations.append(
                check_column_semantic_type(
                    "time_column",
                    time_column,
                    frame,
                    allowed_semantic_types=("datetime",),
                    fix_hint=(
                        "Use a datetime column or omit `time_column` so the tool "
                        "can auto-detect one."
                    ),
                )
            )
    raise_for_violations("run_baseline_model", violations)


def create_model_card_artifact(
    card: ModelCard,
    *,
    project_id: str,
    session_id: str,
    parents: list[str] | None = None,
) -> Artifact:
    payload = card.model_dump(mode="json")
    return Artifact(
        id=make_artifact_id("model", payload),
        type=ArtifactType.MODEL_CARD,
        project_id=project_id,
        session_id=session_id,
        parents=parents or [],
        payload=payload,
        code_ref=code_ref(run_baseline_model),
        plain_language=_model_plain_language(card),
    )


def _model_plain_language(card: ModelCard) -> str:
    """One-sentence, deterministic summary of a baseline model card."""
    return (
        f"{card.task_type} baseline ({card.model_type}) predicting {card.target_column} "
        f"from {len(card.feature_columns)} features; "
        f"{card.train_rows} train / {card.test_rows} test rows{_headline_metric(card)}."
    )


def _headline_metric(card: ModelCard) -> str:
    """Pick one representative metric for the plain-language summary, if present."""
    if "accuracy" in card.metrics:
        return f", accuracy={card.metrics['accuracy']:.3f}"
    if "r2" in card.metrics:
        return f", R²={card.metrics['r2']:.3f}"
    return ""


def _infer_task_type(target: pd.Series) -> TaskType:
    # Treat compact integer code sets as classifications; continuous values regress.
    non_null = target.dropna()
    if non_null.empty:
        return "regression"
    if is_bool_dtype(non_null):
        return "classification"
    if not is_numeric_dtype(non_null):
        return "classification"
    numeric = cast(pd.Series, pd.to_numeric(non_null, errors="coerce")).dropna()
    if numeric.empty:
        return "classification"
    if is_float_dtype(non_null) and not bool((numeric == numeric.round()).all()):
        return "regression"
    unique_count = int(numeric.nunique())
    n = int(len(numeric))
    if unique_count < 2:
        return "regression"
    if unique_count > 20 or unique_count > max(2, int(n * 0.05)):
        return "regression"
    spread = float(numeric.max() - numeric.min())
    # Reject sparse code sets, e.g. {0, 500, 999}: range far exceeds cardinality.
    if spread > 3.0 * unique_count:
        return "regression"
    return "classification"


def _detect_time_column(frame: pd.DataFrame, *, exclude: set[str]) -> str | None:
    # Detect time columns by content, using names only to break ties.
    named_candidates: list[str] = []
    parsed_candidates: list[str] = []
    for column in frame.columns:
        column_name = str(column)
        if column_name in exclude:
            continue
        series = _series(frame, column_name)
        if pd.api.types.is_datetime64_any_dtype(series):
            return column_name
        if is_numeric_dtype(series) or is_bool_dtype(series):
            continue
        parsed = pd.to_datetime(series, errors="coerce", format="mixed")
        if float(parsed.notna().mean()) >= 0.8:
            lowered = column_name.lower()
            if "date" in lowered or "time" in lowered:
                named_candidates.append(column_name)
            else:
                parsed_candidates.append(column_name)
    if named_candidates:
        return named_candidates[0]
    if parsed_candidates:
        return parsed_candidates[0]
    return None


def _select_features(
    frame: pd.DataFrame,
    *,
    target_column: str,
    task_type: TaskType,
    time_column: str | None,
    leakage_checks: list[LeakageCheck],
) -> tuple[list[str], list[str]]:
    selected: list[str] = []
    excluded: list[str] = []
    target = _series(frame, target_column)
    for column in map(str, frame.columns):
        if column == target_column:
            continue
        if column == time_column:
            excluded.append(column)
            continue
        feature = _series(frame, column)
        if _is_id_like(feature, column):
            excluded.append(column)
            leakage_checks.append(
                LeakageCheck(
                    code="id_like_feature",
                    severity="info",
                    column=column,
                    action="excluded",
                    message="Identifier-like feature excluded from baseline model.",
                )
            )
            continue
        if _looks_like_target_leakage(feature, target, task_type=task_type):
            excluded.append(column)
            leakage_checks.append(
                LeakageCheck(
                    code="target_leakage",
                    severity="critical",
                    column=column,
                    action="excluded",
                    message=(
                        "Feature is identical to, or near-perfectly associated with, the "
                        "target (numeric correlation, class purity, mutual information, or "
                        "correlation ratio) and was excluded to prevent leakage."
                    ),
                )
            )
            continue
        selected.append(column)
    if not any(check.code == "target_leakage" for check in leakage_checks):
        leakage_checks.append(
            LeakageCheck(
                code="target_leakage",
                severity="info",
                action="passed",
                message="No identical or near-perfect target leakage feature was detected.",
            )
        )
    return selected, excluded


def _is_id_like(series: pd.Series, column: str) -> bool:
    normalized = column.lower().strip()
    if normalized == "id" or normalized.endswith("_id") or normalized.endswith(" id"):
        return True
    non_null = series.dropna()
    # Content-based detection needs enough rows to be confident a near-unique column is
    # an identifier rather than a small sample where every value happens to differ.
    if len(non_null) < 20:
        return False
    unique_percent = float(non_null.nunique()) / max(len(non_null), 1)
    average_length = float(non_null.astype(str).map(len).mean())
    if unique_percent < 0.95 or average_length > 64:
        return False
    # numeric surrogate keys (account_number, record_no) are unique per row and must be
    # dropped too.
    if is_numeric_dtype(non_null):
        numeric = cast(pd.Series, pd.to_numeric(non_null, errors="coerce")).dropna()
        return bool((numeric == numeric.round()).all())
    return True


def _looks_like_target_leakage(
    feature: pd.Series,
    target: pd.Series,
    *,
    task_type: TaskType,
) -> bool:
    comparable = pd.DataFrame({"feature": feature, "target": target}).dropna()
    if comparable.empty:
        return False
    feature_col = _series(comparable, "feature")
    target_col = _series(comparable, "target")
    # Exact 1:1 byte match (same labels).
    if feature_col.astype(str).equals(target_col.astype(str)):
        return True

    # Numeric-vs-numeric proxy (Pearson). ML-4: classification threshold must be no
    # stricter than regression's -- a strong monotone proxy already separates classes.
    feature_numeric = cast(pd.Series, pd.to_numeric(feature_col, errors="coerce"))
    target_numeric = cast(pd.Series, pd.to_numeric(target_col, errors="coerce"))
    valid = pd.DataFrame({"feature": feature_numeric, "target": target_numeric}).dropna()
    if len(valid) >= 3:
        corr = cast(float, _series(valid, "feature").corr(_series(valid, "target")))
        if not math.isnan(corr):
            corr_threshold = (
                _LEAK_CORR_REG_THRESHOLD
                if task_type == "regression"
                else _LEAK_CORR_CLS_THRESHOLD
            )
            if abs(corr) >= corr_threshold:
                return True

    if len(comparable) < 10:
        return False

    if task_type == "classification":
        # categorical (or coerced-categorical) feature that maps ~1:1 to the class label
        # is invisible to the numeric checks above.
        feature_codes = _as_class_codes(feature_col)
        target_codes = _as_class_codes(target_col)
        if feature_codes is None or target_codes is None:
            return False
        purity = _group_purity(feature_codes, target_codes)
        ami = _adjusted_mutual_information(feature_codes, target_codes)
        return purity >= _LEAK_PURITY_THRESHOLD or ami >= _LEAK_MI_THRESHOLD

    # Regression target with a categorical feature: correlation ratio (eta) measures how
    # much of the target variance the feature groups explain.
    if not is_numeric_dtype(feature_col):
        feature_codes = _as_class_codes(feature_col)
        if feature_codes is None:
            return False
        eta = _correlation_ratio(feature_codes, target_numeric)
        return eta >= _LEAK_ETA_THRESHOLD
    return False


def _as_class_codes(series: pd.Series, *, max_groups: int = 200) -> np.ndarray | None:
    # Map arbitrary (categorical or numeric) values to integer group codes.
    codes = pd.factorize(series.astype("object"), sort=False)[0]
    if codes.size == 0:
        return None
    n_groups = int(pd.Series(codes).nunique())
    if n_groups < 2 or n_groups > max_groups or n_groups >= codes.size:
        return None
    return codes


def _group_purity(feature_codes: np.ndarray, target_codes: np.ndarray) -> float:
    # Weighted mean of each feature-group's majority-class fraction. 1.0 means every
    # feature value maps to a single class (a perfect proxy).
    frame = pd.DataFrame({"f": feature_codes, "t": target_codes})
    total = len(frame)
    if total == 0:
        return 0.0
    correct = 0
    for _, group in frame.groupby("f"):
        correct += int(group["t"].value_counts().iloc[0])
    return float(correct) / float(total)


def _adjusted_mutual_information(feature_codes: np.ndarray, target_codes: np.ndarray) -> float:
    from sklearn.metrics import adjusted_mutual_info_score

    return float(adjusted_mutual_info_score(target_codes, feature_codes))


def _correlation_ratio(feature_codes: np.ndarray, target_values: pd.Series) -> float:
    values = cast(pd.Series, pd.to_numeric(target_values, errors="coerce"))
    frame = pd.DataFrame({"f": feature_codes, "y": values.to_numpy()}).dropna()
    y = _series(frame, "y")
    if int(y.nunique()) < 2:
        return 0.0
    grand_mean = float(y.mean())
    ss_total = float(cast(pd.Series, (y - grand_mean) ** 2).sum())
    if ss_total <= 0.0:
        return 0.0
    ss_between = 0.0
    for _, group in frame.groupby("f"):
        group_y = _series(cast(pd.DataFrame, group), "y")
        n_g = int(len(group_y))
        ss_between += n_g * (float(group_y.mean()) - grand_mean) ** 2
    return math.sqrt(max(0.0, min(1.0, ss_between / ss_total)))


class _PreprocessEncoder:
    """Train-only-fit feature encoder (ML-3 / ML-7)."""

    def __init__(
        self,
        *,
        columns: list[str],
        numeric_columns: list[str],
        categorical_columns: list[str],
        categories: dict[str, list[Any]],
        nan_indicator_columns: set[str],
        numeric_medians: dict[str, float],
    ) -> None:
        self.columns = columns
        self._numeric_columns = numeric_columns
        self._categorical_columns = categorical_columns
        self._categories = categories
        self._nan_indicator_columns = nan_indicator_columns
        self._numeric_medians = numeric_medians

    @classmethod
    def fit(cls, features: pd.DataFrame) -> _PreprocessEncoder:
        numeric_columns: list[str] = []
        categorical_columns: list[str] = []
        for column in map(str, features.columns):
            if is_numeric_dtype(_series(features, column)):
                numeric_columns.append(column)
            else:
                categorical_columns.append(column)

        categories: dict[str, list[Any]] = {}
        nan_indicator_columns: set[str] = set()
        output_columns: list[str] = []
        for column in numeric_columns:
            output_columns.append(column)
        for column in categorical_columns:
            series = _series(features, column)
            observed = pd.Index(series.dropna().astype("object").unique()).tolist()
            categories[column] = observed
            for value in observed:
                output_columns.append(f"{column}={value!r}")
            # ML-7: only reserve a missing-indicator column when TRAIN actually has NaNs.
            if bool(series.isna().any()):
                nan_indicator_columns.add(column)
                output_columns.append(f"{column}=<NA>")

        numeric_medians: dict[str, float] = {}
        for column in numeric_columns:
            numeric = cast(pd.Series, pd.to_numeric(_series(features, column), errors="coerce"))
            median = float(numeric.median()) if numeric.notna().any() else 0.0
            numeric_medians[column] = 0.0 if math.isnan(median) else median

        return cls(
            columns=output_columns,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            categories=categories,
            nan_indicator_columns=nan_indicator_columns,
            numeric_medians=numeric_medians,
        )

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        data: dict[str, np.ndarray] = {}
        index = features.index
        for column in self._numeric_columns:
            numeric = cast(pd.Series, pd.to_numeric(_series(features, column), errors="coerce"))
            filled = numeric.fillna(self._numeric_medians[column])
            data[column] = filled.to_numpy(dtype=float)
        for column in self._categorical_columns:
            series = (
                _series(features, column).astype("object")
                if column in features.columns
                else pd.Series([pd.NA] * len(index), index=index, dtype="object")
            )
            for value in self._categories[column]:
                data[f"{column}={value!r}"] = (series == value).to_numpy(dtype=float)
            if column in self._nan_indicator_columns:
                data[f"{column}=<NA>"] = series.isna().to_numpy(dtype=float)
        encoded = pd.DataFrame(data, index=index)
        # Reindex to the train-fit universe: test-only columns are dropped, train-only
        # columns absent from this batch are filled with 0.0.
        return encoded.reindex(columns=self.columns, fill_value=0.0).astype(float)


def _split_indexes(
    y: pd.Series,
    *,
    task_type: TaskType,
    split_strategy: SplitStrategy,
    random_state: int,
) -> tuple[list[int], list[int]]:
    if len(y) < 10:
        raise ValueError("At least 10 labeled rows are required for a baseline model")
    if split_strategy == "time_ordered":
        split_at = max(1, int(len(y) * 0.8))
        split_at = min(split_at, len(y) - 1)
        return list(range(split_at)), list(range(split_at, len(y)))

    from sklearn.model_selection import train_test_split

    indexes = list(range(len(y)))
    stratify = None
    if task_type == "classification":
        value_counts = y.value_counts()
        if not value_counts.empty and int(value_counts.min()) >= 2:
            stratify = y
    train_idx, test_idx = train_test_split(
        indexes,
        test_size=0.2,
        random_state=random_state,
        stratify=stratify,
    )
    return list(train_idx), list(test_idx)


def _fit_classification(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    *,
    random_state: int,
) -> tuple[Any, dict[str, float]]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    model = RandomForestClassifier(n_estimators=80, random_state=random_state)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "f1_weighted": round(float(f1_score(y_test, predictions, average="weighted")), 4),
    }
    if len(model.classes_) == 2 and hasattr(model, "predict_proba"):
        probabilities = cast(Any, model.predict_proba(X_test))[:, 1]
        try:
            metrics["auc"] = round(float(roc_auc_score(y_test, probabilities)), 4)
        except ValueError:
            pass
    return model, metrics


def _fit_regression(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    *,
    random_state: int,
) -> tuple[Any, dict[str, float]]:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    model = RandomForestRegressor(n_estimators=80, random_state=random_state)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = float(mean_squared_error(y_test, predictions))
    return model, {
        "r2": round(float(r2_score(y_test, predictions)), 4),
        "mae": round(float(mean_absolute_error(y_test, predictions)), 4),
        "rmse": round(math.sqrt(mse), 4),
    }


def _feature_importance(
    model: Any,
    feature_names: list[str],
    *,
    limit: int = 10,
) -> list[FeatureImportance]:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []
    rows = [
        FeatureImportance(feature=feature, importance=round(float(importance), 6))
        for feature, importance in zip(feature_names, importances, strict=True)
    ]
    rows.sort(key=lambda row: row.importance, reverse=True)
    return rows[:limit]


def _ensure_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised when dependency missing.
        raise RuntimeError(
            "scikit-learn is required for M5 baseline models. Install project dependencies."
        ) from exc


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    return cast(pd.Series, frame[column])


def _frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return cast(pd.DataFrame, frame.loc[:, columns])
