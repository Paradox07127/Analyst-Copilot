from __future__ import annotations

import re
from collections.abc import Iterable

from eda_platform.core.column_roles import ColumnRoleSet
from eda_platform.schemas.anomaly import AnomalyScreenResult
from eda_platform.schemas.artifacts import EvidenceRef
from eda_platform.schemas.model_card import ModelCard
from eda_platform.schemas.questions import FindingScore, QuestionFinding
from eda_platform.schemas.stats import StatTestResult
from eda_platform.tools.interestingness import (
    ANOMALY_SHARE_SATURATION_PERCENT,
    InterestingnessScore,
    interestingness,
)

# DI8-D insight ranking = impact x significance (QuickInsights-style).
_MODEL_SIGNIFICANCE_METRICS = ("r2", "f1_weighted", "accuracy", "auc")
_DEFAULT_MODEL_SIGNIFICANCE = 0.5


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _impact(
    columns: Iterable[str | None],
    role_set: ColumnRoleSet | None,
) -> float:
    """Business-impact weight over the referenced columns (min-combined)."""
    names = [column for column in columns if column]
    if role_set is None or not names:
        return 1.0
    return min(role_set.impact_weight(column) for column in names)


def _score(
    impact: float,
    significance: float,
    interest: InterestingnessScore | None = None,
) -> FindingScore:
    impact = _clamp01(impact)
    significance = _clamp01(significance)
    final = impact * significance
    interest_value: float | None = None
    if interest is not None:
        interest_value = interest.value
        final *= interest_value
    return FindingScore(
        impact=impact,
        significance=significance,
        interestingness=interest_value,
        final=round(final, 6),
    )


def stat_findings(
    result: StatTestResult,
    artifact_id: str,
    *,
    role_set: ColumnRoleSet | None = None,
    dataset_row_count: int | None = None,
) -> list[QuestionFinding]:
    if result.statistic is None or result.p_value is None:
        return []
    effect_name = _effect_name(result.test_type)
    subject = "association" if result.test_type == "chi_square_independence" else "difference"
    # When a correction was applied, the adjusted p is the decision value; the
    # raw p stays visible so the correction is auditable.
    p_display = (
        f"p {_p_value(result.p_value)}"
        if result.adjusted_p_value is None
        else (
            f"p {_p_value(result.p_value)} "
            f"({result.correction_method}-adjusted p {_p_value(result.adjusted_p_value)})"
        )
    )
    parts = [
        f"The {result.test_type.replace('_', ' ')} shows an observed {subject}: "
        f"statistic {_number(result.statistic)}, {p_display}, "
        f"sample size {result.sample_size}"
    ]
    evidence = [
        _ref(artifact_id, "statistic", result.statistic),
        _ref(artifact_id, "p_value", result.p_value),
        _ref(artifact_id, "sample_size", result.sample_size),
    ]
    if result.adjusted_p_value is not None:
        evidence.append(
            _ref(artifact_id, "adjusted_p_value", result.adjusted_p_value)
        )
    if result.effect_size is not None:
        parts.append(
            f", {effect_name} {_number(result.effect_size)} "
            f"({_effect_magnitude(result.test_type, result.effect_size)} effect)"
        )
        evidence.append(_ref(artifact_id, "effect_size", result.effect_size))
    parts.append(".")

    warning_messages = [
        (f"assumptions[{index}].message", check.message)
        for index, check in enumerate(result.assumptions)
        if check.status == "warn"
    ]
    warning_messages.extend(
        (f"warnings[{index}].message", warning.message)
        for index, warning in enumerate(result.warnings)
    )
    for locator, message in warning_messages:
        parts.append(f" Assumption warning: {message}")
        evidence.extend(_message_number_refs(artifact_id, message, locator))
    interest: InterestingnessScore | None = None
    if dataset_row_count is not None:
        interest = interestingness(
            deviation=_stat_deviation(result),
            rows_involved=result.sample_size,
            dataset_row_count=dataset_row_count,
            degenerate=(
                result.group_column is not None
                and result.group_column == result.value_column
            ),
        )
    score = _score(
        _impact((result.group_column, result.value_column), role_set),
        1.0
        - (
            result.p_value
            if result.adjusted_p_value is None
            else result.adjusted_p_value
        ),
        interest,
    )
    return [QuestionFinding(text="".join(parts), evidence=evidence, score=score)]


def _stat_deviation(result: StatTestResult) -> float | None:
    """Calculate a normalized departure-from-baseline signal."""
    if result.effect_size is not None:
        absolute = abs(result.effect_size)
        if result.test_type == "fisher_exact" and absolute > 0:
            # OR is symmetric on the log scale: OR and 1/OR are equal-strength effects.
            absolute = max(absolute, 1.0 / absolute)
        large = _effect_cutoffs(result.test_type)[1]
        return _clamp01(absolute / large)
    effective_p = (
        result.p_value if result.adjusted_p_value is None else result.adjusted_p_value
    )
    if effective_p is not None:
        return _clamp01(1.0 - effective_p)
    return None


def model_findings(
    card: ModelCard,
    artifact_id: str,
    *,
    role_set: ColumnRoleSet | None = None,
    dataset_row_count: int | None = None,
) -> list[QuestionFinding]:
    metric_items = sorted(card.metrics.items())
    evidence = [
        _ref(artifact_id, f"metrics.{name}", value) for name, value in metric_items
    ]
    if metric_items:
        metrics = ", ".join(
            f"{_metric_label(name)} {_number(value)}" for name, value in metric_items
        )
        text = f"The {card.task_type} baseline estimate reports {metrics}. "
    else:
        text = f"The {card.task_type} baseline produced no reported performance metrics. "
    text += "This is a baseline estimate within stated performance, not a causal claim."
    interest: InterestingnessScore | None = None
    if dataset_row_count is not None:
        interest = interestingness(
            deviation=_model_significance(card),
            rows_involved=card.train_rows + card.test_rows,
            dataset_row_count=dataset_row_count,
            # Identity model: the target leaking into its own features is the
            # degenerate "X predicts X" pattern.
            degenerate=card.target_column in card.feature_columns,
        )
    score = _score(
        _impact((card.target_column, *card.feature_columns), role_set),
        _model_significance(card),
        interest,
    )
    return [QuestionFinding(text=text, evidence=evidence, score=score)]


def _model_significance(card: ModelCard) -> float:
    for name in _MODEL_SIGNIFICANCE_METRICS:
        value = card.metrics.get(name)
        if value is not None:
            return _clamp01(float(value))
    return _DEFAULT_MODEL_SIGNIFICANCE


def anomaly_findings(
    result: AnomalyScreenResult,
    artifact_id: str,
    *,
    role_set: ColumnRoleSet | None = None,
) -> list[QuestionFinding]:
    text = (
        f"Anomaly screening found {result.outlier_count} flagged observations among "
        f"{result.non_null_rows} non-null rows ({_number(result.outlier_percent)} percent), "
        f"using threshold {_number(result.threshold)}."
    )
    evidence = [
        _ref(artifact_id, "outlier_count", result.outlier_count),
        _ref(artifact_id, "non_null_rows", result.non_null_rows),
        _ref(artifact_id, "outlier_percent", result.outlier_percent),
        _ref(artifact_id, "threshold", result.threshold),
    ]
    for index, note in enumerate(result.notes):
        text += f" Note: {note}"
        evidence.extend(
            _message_number_refs(artifact_id, note, f"notes[{index}]")
        )
    # The anomaly result carries its own coverage anchor (``total_rows``), so
    # interestingness inputs are always sufficient for this reducer.
    interest = interestingness(
        deviation=_clamp01(result.outlier_percent / ANOMALY_SHARE_SATURATION_PERCENT),
        rows_involved=result.non_null_rows,
        dataset_row_count=result.total_rows,
        # Degenerate identity pattern: every non-null row flagged means the
        # "anomaly count" is just the row count.
        degenerate=result.non_null_rows > 0
        and result.outlier_count >= result.non_null_rows,
    )
    score = _score(
        _impact((result.column,), role_set),
        result.outlier_percent / 100.0,
        interest,
    )
    return [QuestionFinding(text=text, evidence=evidence, score=score)]


def _ref(artifact_id: str, locator: str, value: str | float | int) -> EvidenceRef:
    return EvidenceRef(kind="stat", artifact_id=artifact_id, locator=locator, value=value)


def _number(value: float) -> str:
    return f"{value:.6g}"


def _p_value(value: float) -> str:
    if value <= 0:
        return "< floating-point resolution"
    if 0 < value < 0.001:
        return "< 0.001"
    return _number(value)


def _effect_name(test_type: str) -> str:
    return {
        "independent_t_test": "Cohen d",
        "paired_t_test": "Cohen d",
        "chi_square_independence": "Cramer V",
        "fisher_exact": "odds ratio",
        "one_way_anova": "eta squared",
        "welch_anova": "descriptive eta squared",
        "mann_whitney_u": "rank-biserial correlation",
        "kruskal_wallis": "epsilon squared",
    }[test_type]


def _effect_cutoffs(test_type: str) -> tuple[float, float]:
    """(medium, large) conventional cutoffs for the test's effect-size scale."""
    if test_type in {"independent_t_test", "paired_t_test"}:
        return 0.5, 0.8
    if test_type in {"one_way_anova", "welch_anova", "kruskal_wallis"}:
        return 0.06, 0.14
    if test_type == "fisher_exact":
        # Odds-ratio cutoffs mapping to Cohen's d 0.5/0.8 (Chen, Cohen & Chen 2010).
        return 3.0, 5.0
    return 0.3, 0.5


def _effect_magnitude(test_type: str, value: float) -> str:
    absolute = abs(value)
    if test_type == "fisher_exact" and absolute > 0:
        # OR is symmetric on the log scale: OR and 1/OR are equal-strength effects.
        absolute = max(absolute, 1.0 / absolute)
    medium, large = _effect_cutoffs(test_type)
    if absolute >= large:
        return "large"
    if absolute >= medium:
        return "medium"
    return "small"


def _metric_label(name: str) -> str:
    labels = {
        "r2": "R squared",
        "f1_weighted": "weighted F score",
    }
    return labels.get(name, name.replace("_", " "))


def _message_number_refs(
    artifact_id: str,
    message: str,
    locator: str,
) -> list[EvidenceRef]:
    return [
        _ref(artifact_id, locator, match.group())
        for match in re.finditer(r"(?<![\w.])-?(?:\d+(?:\.\d+)?|\.\d+)", message)
    ]
