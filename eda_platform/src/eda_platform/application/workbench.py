"""Shared display/projection helpers for API services.

Display/projection helpers shared by application services (and their unit tests).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from eda_platform.schemas.artifacts import Artifact, ArtifactType, DatasetProfile, QualityIssueSet
from eda_platform.schemas.model_card import LeakageCheck, ModelCard

# Below this dataset/test row count, results get a "small sample" caution badge.
SMALL_SAMPLE_THRESHOLD = 30

# Feasibility statuses that block execution: a card with no supported/gated
# method yet, so its checkbox should not be selectable.
_UNSELECTABLE_FEASIBILITY_STATUSES = frozenset({"needs_data", "unsuitable"})

# StatTestResult.test_type -> effect-size metric family for magnitude thresholds.
_EFFECT_SIZE_FAMILY_BY_TEST: dict[str, str] = {
    "independent_t_test": "d",
    "paired_t_test": "d",
    "one_way_anova": "eta",
    "welch_anova": "eta",
    "kruskal_wallis": "eta",
    "chi_square_independence": "r",
    "mann_whitney_u": "r",
}

# (small, medium, large) lower-bound thresholds per effect-size family.
_EFFECT_SIZE_THRESHOLDS: dict[str, tuple[float, float, float]] = {
    "d": (0.2, 0.5, 0.8),
    "eta": (0.01, 0.06, 0.14),
    "r": (0.1, 0.3, 0.5),
}

# ModelCard.task_type -> preferred headline metric names, in priority order.
_HEADLINE_METRIC_PRIORITY: dict[str, tuple[str, ...]] = {
    "classification": ("accuracy", "f1_weighted"),
    "regression": ("r2", "rmse", "mae"),
}


def summarize_session(artifacts: list[Artifact]) -> dict[str, int]:
    summary = {"artifacts": len(artifacts), "datasets": 0, "critical": 0, "warn": 0, "info": 0}
    for artifact in artifacts:
        if artifact.type is ArtifactType.DATASET_PROFILE:
            summary["datasets"] += 1
        if artifact.type is ArtifactType.QUALITY_ISSUE_SET:
            issue_set = QualityIssueSet.model_validate(artifact.payload)
            for issue in issue_set.issues:
                summary[issue.severity] += 1
    return summary


def dataset_display_rows(profile_artifact: Artifact) -> list[dict[str, object]]:
    profile = DatasetProfile.model_validate(profile_artifact.payload)
    columns_detail = getattr(profile, "columns_detail", [])
    if not columns_detail:
        return _legacy_dataset_display_rows(profile)
    return [
        {
            "column": column.name,
            "dtype": column.dtype,
            "semantic_type": column.semantic_type,
            "missing_percent": column.missing_percent,
            "unique_percent": column.unique_percent,
            "sample_values": ", ".join(column.sample_values[:3]),
        }
        for column in columns_detail
    ]


def semantic_type_counts(profile_artifact: Artifact) -> dict[str, int]:
    profile = DatasetProfile.model_validate(profile_artifact.payload)
    counts = getattr(profile, "semantic_type_counts", None)
    if counts:
        return dict(counts)
    columns_detail = getattr(profile, "columns_detail", [])
    if columns_detail:
        derived: dict[str, int] = {}
        for column in columns_detail:
            derived[column.semantic_type] = derived.get(column.semantic_type, 0) + 1
        return derived
    numeric_columns = set(getattr(profile, "numeric_columns", []))
    categorical_columns = set(getattr(profile, "categorical_columns", []))
    unknown_columns = (
        set(getattr(profile, "column_names", [])) - numeric_columns - categorical_columns
    )
    derived = {}
    if numeric_columns:
        derived["numeric"] = len(numeric_columns)
    if categorical_columns:
        derived["categorical"] = len(categorical_columns)
    if unknown_columns:
        derived["unknown"] = len(unknown_columns)
    return derived


def group_quality_issues(artifacts: list[Artifact]) -> dict[str, list[dict[str, str | None]]]:
    """Build display rows for quality issues."""
    grouped: dict[str, list[dict[str, str | None]]] = {"critical": [], "warn": [], "info": []}
    dataset_names = dataset_names_by_id(artifacts)
    for artifact in artifacts:
        if artifact.type is not ArtifactType.QUALITY_ISSUE_SET:
            continue
        issue_set = QualityIssueSet.model_validate(artifact.payload)
        for issue in issue_set.issues:
            grouped[issue.severity].append(
                {
                    "message": issue.message,
                    "column": issue.column,
                    "dataset_name": dataset_names.get(issue_set.dataset_id, issue_set.dataset_id),
                    "code": issue.code,
                    "recommendation": issue.recommendation,
                }
            )
    return grouped


def split_trivial_correlation_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition correlation rows into substantive and trivial groups."""
    substantive: list[dict[str, Any]] = []
    trivial: list[dict[str, Any]] = []
    for row in rows:
        (trivial if bool(row.get("is_trivial_pair")) else substantive).append(dict(row))
    return substantive, trivial


def min_sample_size(rows: Sequence[Mapping[str, Any]]) -> int | None:
    """Return the smallest recorded sample size, or ``None``."""
    sizes = [int(row["sample_size"]) for row in rows if row.get("sample_size") is not None]
    return min(sizes) if sizes else None


def format_p_value(value: Any) -> str:
    """Display a p-value as ``<0.001`` below that threshold, else 3 significant
    digits. Display-only: stored artifacts keep full precision."""
    if value is None:
        return ""
    numeric = float(value)
    if numeric < 0.001:
        return "<0.001"
    return f"{numeric:.3g}"


def run_cost_summary(artifacts: list[Artifact]) -> dict[str, Any] | None:
    """Return whole-run LLM spend.

    SessionMetrics is the run-wide ledger; SessionSummary only covers report generation
    and reading it here understated every run's cost by 40-55%. Old runs without
    SessionMetrics still fall back to it, flagged as partial.
    """
    metrics = _latest_artifact(artifacts, ArtifactType.SESSION_METRICS)
    if metrics is not None:
        payload = metrics.payload
        return {
            "llm_call_count": payload.get("llm_calls"),
            "total_tokens": payload.get("total_tokens"),
            "prompt_tokens": payload.get("prompt_tokens"),
            "cached_tokens": payload.get("cached_tokens"),
            "estimated_cost_usd": payload.get("est_cost_usd"),
            "model": _run_model(artifacts),
            "scope": "session",
        }
    artifact = _latest_artifact(artifacts, ArtifactType.SESSION_SUMMARY)
    if artifact is None:
        return None
    return {**artifact.payload, "scope": "report_only"}


def effect_size_magnitude_badge(
    test_type: str | None, effect_size: float | None
) -> str:
    """Return the conventional magnitude label for an effect size."""
    if effect_size is None:
        return ""
    family = _EFFECT_SIZE_FAMILY_BY_TEST.get(test_type or "", "r")
    small, medium, large = _EFFECT_SIZE_THRESHOLDS[family]
    magnitude = abs(effect_size)
    if magnitude >= large:
        return "large"
    if magnitude >= medium:
        return "medium"
    if magnitude >= small:
        return "small"
    return "negligible"


def leakage_verdict_badge(checks: Sequence[LeakageCheck]) -> str:
    """Return a model card's whole-card leakage verdict label."""
    if not checks:
        return "unchecked"
    if any(check.severity == "critical" and check.action != "excluded" for check in checks):
        return "risk"
    if any(check.action in {"excluded", "warned"} for check in checks):
        return "mitigated"
    return "clean"


def headline_metric(card: ModelCard) -> tuple[str, float] | None:
    """Return the most decision-relevant metric on a model card."""
    for name in _HEADLINE_METRIC_PRIORITY.get(card.task_type, ()):
        if name in card.metrics:
            return name, card.metrics[name]
    return None


def checkbox_disabled_for_feasibility(status: str | None) -> bool:
    """Return whether feasibility blocks candidate selection."""
    return status in _UNSELECTABLE_FEASIBILITY_STATUSES


def dataset_names_by_id(artifacts: list[Artifact]) -> dict[str, str]:
    """Map dataset IDs to names from profile artifacts."""
    names: dict[str, str] = {}
    for artifact in artifacts:
        if artifact.type is not ArtifactType.DATASET_PROFILE:
            continue
        profile = DatasetProfile.model_validate(artifact.payload)
        names[profile.dataset_id] = profile.name
    return names


def _run_model(artifacts: list[Artifact]) -> str:
    summary = _latest_artifact(artifacts, ArtifactType.SESSION_SUMMARY)
    return str(summary.payload.get("model", "")) if summary is not None else ""


def _latest_artifact(artifacts: list[Artifact], artifact_type: ArtifactType) -> Artifact | None:
    for artifact in reversed(artifacts):
        if artifact.type is artifact_type:
            return artifact
    return None


def _legacy_dataset_display_rows(profile: DatasetProfile) -> list[dict[str, object]]:
    numeric_columns = set(getattr(profile, "numeric_columns", []))
    categorical_columns = set(getattr(profile, "categorical_columns", []))
    rows: list[dict[str, object]] = []
    for column in getattr(profile, "column_names", []):
        semantic_type = "unknown"
        if column in numeric_columns:
            semantic_type = "numeric"
        elif column in categorical_columns:
            semantic_type = "categorical"
        rows.append(
            {
                "column": column,
                "dtype": getattr(profile, "dtypes", {}).get(column, ""),
                "semantic_type": semantic_type,
                "missing_percent": getattr(profile, "missing_percent", {}).get(column, 0.0),
                "unique_percent": None,
                "sample_values": "",
            }
        )
    return rows
