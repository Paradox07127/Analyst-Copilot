from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from eda_platform.schemas.artifacts import ColumnProfile, DatasetProfile
from eda_platform.schemas.questions import (
    AnalysisMode,
    OpportunityFeasibility,
)

# Typed prerequisite failures let feasibility distinguish structural gaps from
# missing data or insufficient scale without parsing human-readable messages.
MissingKind = Literal["structural", "data", "scale"]


@dataclass(frozen=True)
class MethodGateContext:
    profiles: list[DatasetProfile]
    target_datasets: list[str]
    analysis_mode: AnalysisMode | None
    target_column: str | None


@dataclass(frozen=True)
class MethodGateResult:
    ok: bool
    reasons: list[str]
    missing: list[str]
    # Parallel to ``missing`` (same length, same order): the typed kind of each
    # missing item. Defaults to empty for ``ok`` results with no missing items.
    missing_kinds: list[MissingKind] = field(default_factory=list)


@dataclass(frozen=True)
class AnalysisMethod:
    method_id: str
    mode: AnalysisMode
    summary: str
    supported: bool
    gate: Callable[[MethodGateContext], MethodGateResult]


def _target_profiles(ctx: MethodGateContext) -> list[DatasetProfile]:
    if not ctx.target_datasets:
        return sorted(ctx.profiles, key=lambda profile: (profile.name, profile.dataset_id))
    targets = set(ctx.target_datasets)
    return sorted(
        (
            profile
            for profile in ctx.profiles
            if profile.name in targets or profile.dataset_id in targets
        ),
        key=lambda profile: (profile.name, profile.dataset_id),
    )


def _columns(profile: DatasetProfile, semantic_types: set[str]) -> list[ColumnProfile]:
    return [column for column in profile.columns_detail if column.semantic_type in semantic_types]


def _descriptive_gate(ctx: MethodGateContext) -> MethodGateResult:
    del ctx
    return MethodGateResult(
        ok=True,
        reasons=["Descriptive SQL is available as a safe baseline."],
        missing=[],
    )


def _domain_metric_pack_gate(ctx: MethodGateContext) -> MethodGateResult:
    """Validate whether registered domain metrics can be computed from the available datasets."""
    if not _target_profiles(ctx):
        return MethodGateResult(
            ok=False,
            reasons=["No target dataset profile is available to resolve metrics against."],
            missing=["target dataset profile"],
            missing_kinds=["data"],
        )
    return MethodGateResult(
        ok=True,
        reasons=[
            "Registered domain metrics are computed with deterministic SQL; "
            "per-metric applicability is decided by the domain-metrics registry."
        ],
        missing=[],
    )


def _group_comparison_gate(ctx: MethodGateContext) -> MethodGateResult:
    profiles = _target_profiles(ctx)
    if not profiles:
        return MethodGateResult(
            ok=False,
            reasons=["No target dataset profile is available for the comparison."],
            missing=["target dataset profile"],
            missing_kinds=["data"],
        )

    has_group = False
    has_numeric = False
    has_paired_columns = False
    for profile in profiles:
        grouping_columns = [
            column
            for column in _columns(profile, {"categorical", "boolean"})
            if 2 <= column.unique_count <= 20
        ]
        numeric_columns = _columns(profile, {"numeric"})
        has_group = has_group or bool(grouping_columns)
        has_numeric = has_numeric or bool(numeric_columns)
        if grouping_columns and numeric_columns:
            has_paired_columns = True
            if any(profile.rows / column.unique_count >= 5 for column in grouping_columns):
                return MethodGateResult(
                    ok=True,
                    reasons=[
                        "A grouping column and numeric measure are available, with at "
                        "least five expected rows per group."
                    ],
                    missing=[],
                )

    missing: list[str] = []
    missing_kinds: list[MissingKind] = []
    # A missing grouping/numeric column (or the pair split across datasets) is a
    # wrong-shape problem → structural.
    if not has_group:
        missing.append("grouping column with 2-20 distinct values")
        missing_kinds.append("structural")
    if not has_numeric:
        missing.append("numeric measure")
        missing_kinds.append("structural")
    if has_group and has_numeric and not has_paired_columns:
        missing.append("grouping column and numeric measure in the same dataset")
        missing_kinds.append("structural")
    if missing:
        return MethodGateResult(
            ok=False,
            reasons=[
                "A group comparison needs a usable grouping column and numeric measure "
                "in one target dataset."
            ],
            missing=missing,
            missing_kinds=missing_kinds,
        )
    # The columns exist but there are too few expected rows per group → scale.
    return MethodGateResult(
        ok=False,
        reasons=["The profile suggests fewer than five expected rows per group."],
        missing=["at least 5 expected rows per group"],
        missing_kinds=["scale"],
    )


def _prediction_gate(ctx: MethodGateContext) -> MethodGateResult:
    target_column = (ctx.target_column or "").strip()
    if not target_column:
        return MethodGateResult(
            ok=False,
            reasons=["Outcome prediction needs a named target column."],
            missing=["target column"],
            missing_kinds=["data"],
        )
    profiles = _target_profiles(ctx)
    matching = [
        (profile, column)
        for profile in profiles
        for column in profile.columns_detail
        if column.name == target_column
    ]
    if not matching:
        return MethodGateResult(
            ok=False,
            reasons=["The proposed target column is not present in a target dataset profile."],
            missing=[f"target column: {target_column}"],
            missing_kinds=["data"],
        )

    for profile, column in matching:
        target_is_usable = column.missing_percent < 99.0
        enough_rows = profile.rows >= 50
        has_classes = column.semantic_type not in {"categorical", "boolean"} or (
            column.unique_count >= 2
        )
        if target_is_usable and enough_rows and has_classes:
            return MethodGateResult(
                ok=True,
                reasons=[
                    "The target and minimum sample size are available; confirm that "
                    "features are known before the outcome to control leakage risk."
                ],
                missing=[],
            )

    missing = []
    missing_kinds: list[MissingKind] = []
    # Target present but all-missing, or with fewer than two classes, is a data
    # prerequisite; too few rows is a scale prerequisite. Both → needs_data.
    if all(column.missing_percent >= 99.0 for _, column in matching):
        missing.append(f"non-missing values for target column: {target_column}")
        missing_kinds.append("data")
    if all(profile.rows < 50 for profile, _ in matching):
        missing.append("at least 50 rows")
        missing_kinds.append("scale")
    if all(
        column.semantic_type in {"categorical", "boolean"} and column.unique_count < 2
        for _, column in matching
    ):
        missing.append("at least two target classes")
        missing_kinds.append("data")
    return MethodGateResult(
        ok=False,
        reasons=["The target profile does not yet support a prediction baseline."],
        missing=missing,
        missing_kinds=missing_kinds,
    )


def _anomaly_gate(ctx: MethodGateContext) -> MethodGateResult:
    profiles = _target_profiles(ctx)
    if not profiles:
        return MethodGateResult(
            ok=False,
            reasons=["No target dataset profile is available for anomaly detection."],
            missing=["target dataset profile"],
            missing_kinds=["data"],
        )
    numeric_profiles = [profile for profile in profiles if _columns(profile, {"numeric"})]
    if not numeric_profiles:
        return MethodGateResult(
            ok=False,
            reasons=["Anomaly detection needs at least one numeric column."],
            missing=["numeric column"],
            missing_kinds=["structural"],
        )
    if not any(profile.rows >= 30 for profile in numeric_profiles):
        return MethodGateResult(
            ok=False,
            reasons=["The numeric data has fewer than 30 rows, so anomalies are unstable."],
            missing=["at least 30 rows"],
            missing_kinds=["scale"],
        )
    return MethodGateResult(
        ok=True,
        reasons=["A numeric column and at least 30 rows support robust anomaly screening."],
        missing=[],
    )


def _forecast_gate(ctx: MethodGateContext) -> MethodGateResult:
    if any(_columns(profile, {"datetime"}) for profile in _target_profiles(ctx)):
        return MethodGateResult(
            ok=True,
            reasons=["forecast method family not implemented yet; start with a descriptive trend"],
            missing=[],
        )
    return MethodGateResult(
        ok=False,
        reasons=["Forecasting needs a time column to order observations."],
        missing=["time column"],
        missing_kinds=["data"],
    )


def _segmentation_gate(ctx: MethodGateContext) -> MethodGateResult:
    if not _target_profiles(ctx):
        return MethodGateResult(
            ok=False,
            reasons=["No target dataset profile is available for segmentation."],
            missing=["target dataset profile"],
            missing_kinds=["data"],
        )
    return MethodGateResult(
        ok=True,
        reasons=[
            "Segmentation method family is not implemented yet; first confirm meaningful "
            "features and segment stability criteria."
        ],
        missing=[],
    )


def _causal_gate(ctx: MethodGateContext) -> MethodGateResult:
    if not _target_profiles(ctx):
        return MethodGateResult(
            ok=False,
            reasons=["No target dataset profile is available to ground an experiment design."],
            missing=["target dataset profile"],
            missing_kinds=["data"],
        )
    return MethodGateResult(
        ok=True,
        reasons=[
            "Causal analysis is not implemented; define a treatment, outcome, and approved "
            "experiment design before making an intervention claim."
        ],
        missing=[],
    )


METHOD_REGISTRY: dict[str, AnalysisMethod] = {
    method.method_id: method
    for method in (
        AnalysisMethod(
            method_id="descriptive_sql",
            mode="descriptive",
            summary="Summarize observed values and trends with deterministic SQL.",
            supported=True,
            gate=_descriptive_gate,
        ),
        AnalysisMethod(
            method_id="group_comparison",
            mode="diagnostic",
            summary="Compare a numeric measure across meaningful groups.",
            supported=True,
            gate=_group_comparison_gate,
        ),
        AnalysisMethod(
            method_id="forecast",
            mode="forecast",
            summary="Estimate a future value from time-ordered history.",
            supported=False,
            gate=_forecast_gate,
        ),
        AnalysisMethod(
            method_id="outcome_prediction",
            mode="prediction",
            summary="Build a baseline that predicts a defined outcome.",
            supported=True,
            gate=_prediction_gate,
        ),
        AnalysisMethod(
            method_id="segmentation",
            mode="segmentation",
            summary="Identify stable groups with meaningfully different behavior.",
            supported=False,
            gate=_segmentation_gate,
        ),
        AnalysisMethod(
            method_id="anomaly_detection",
            mode="anomaly",
            summary="Screen numeric observations for robust statistical outliers.",
            supported=True,
            gate=_anomaly_gate,
        ),
        AnalysisMethod(
            method_id="causal_experiment",
            mode="causal_experiment",
            summary="Design an experiment to test an intervention effect.",
            supported=False,
            gate=_causal_gate,
        ),
        # Keep after descriptive_sql because feasibility selects the first method per mode.
        AnalysisMethod(
            method_id="domain_metric_pack",
            mode="descriptive",
            summary=(
                "Compute registered domain business metrics (GMV, AOV, repeat-"
                "purchase rate, ...) with deterministic SQL."
            ),
            supported=True,
            gate=_domain_metric_pack_gate,
        ),
    )
}


def evaluate_feasibility(ctx: MethodGateContext) -> OpportunityFeasibility:
    requested_mode: AnalysisMode = ctx.analysis_mode or "descriptive"
    methods = [method for method in METHOD_REGISTRY.values() if method.mode == requested_mode]
    if not methods:
        return OpportunityFeasibility(
            status="unsuitable",
            reasons=[f"No method is registered for analysis mode: {requested_mode}."],
            missing=[],
        )

    method = methods[0]
    result = method.gate(ctx)
    if result.ok:
        status = (
            "constrained"
            if not method.supported or method.method_id == "outcome_prediction"
            else "ready"
        )
    elif result.missing_kinds:
        # A wrong-shape (structural) gap means the data cannot support the method
        # as framed → unsuitable; a data/scale gap is a fixable prerequisite →
        # needs_data. Decided from the typed kinds, not missing-item strings.
        status = "unsuitable" if "structural" in result.missing_kinds else "needs_data"
    else:
        status = "unsuitable"
    return OpportunityFeasibility(
        status=status,
        method_id=method.method_id,
        reasons=result.reasons,
        missing=result.missing,
    )
