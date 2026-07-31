from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from eda_platform.core.methods import METHOD_REGISTRY, MethodGateContext, evaluate_feasibility
from eda_platform.schemas.artifacts import DatasetProfile
from eda_platform.schemas.investigations import GateStatus, InvestigationGate
from eda_platform.schemas.questions import (
    FeasibilityStatus,
    OpportunityFeasibility,
    QuestionCandidate,
)


@dataclass(frozen=True)
class MethodSelection:
    method_family: str
    method_recipe: str
    allowed_tools: list[str]
    method_requirements: list[str]
    validation_gates: list[InvestigationGate]
    execution_ready: bool
    feasibility: OpportunityFeasibility


_METHOD_OUTPUTS: dict[str, tuple[str, str, list[str], list[str]]] = {
    "descriptive_sql": (
        "descriptive_analysis",
        "Use approved read-only SQL to summarize observed values and trends.",
        ["read_only_sql"],
        ["a profiled target dataset", "read-only SQL within the approved scope"],
    ),
    "group_comparison": (
        "group_comparison",
        "Use read-only SQL, then run the deterministic statistical test route "
        "to compare eligible groups.",
        ["read_only_sql", "stat_tests"],
        ["a usable grouping column and numeric measure in one dataset"],
    ),
    "forecast": (
        "forecasting",
        "Prepare a reviewed time-series plan before running a forecast.",
        ["time_series_forecast"],
        ["time-ordered observations and a numeric target measure"],
    ),
    "outcome_prediction": (
        "predictive_modeling",
        "Use read-only SQL, then run the deterministic baseline model route "
        "for the defined outcome.",
        ["read_only_sql", "ml_baseline"],
        ["an explicit target column and held-out evaluation"],
    ),
    "segmentation": (
        "segmentation",
        "Prepare a reviewed feature and stability plan before fitting segments.",
        ["segmentation_model"],
        ["meaningful features and segment stability criteria"],
    ),
    "anomaly_detection": (
        "anomaly_detection",
        "Use read-only SQL, then run deterministic anomaly screening on the "
        "approved numeric measure.",
        ["read_only_sql", "anomaly_screen"],
        ["numeric observations and an approved review threshold"],
    ),
    "causal_experiment": (
        "causal_or_experiment_design",
        "Define and review the treatment, outcome, and study design before analysis.",
        ["causal_design"],
        ["an explicit treatment, outcome, and approved study design"],
    ),
}

_TEMPLATE_RECIPES = {
    "trend": "Run the approved deterministic trend SQL.",
    "group_difference": "Run the approved deterministic group comparison SQL.",
    "correlation_probe": "Run the approved deterministic association SQL.",
    "quality_missing": "Run the approved deterministic missing-data SQL.",
}


def select_investigation_method(
    candidate: QuestionCandidate,
    profiles_by_name: Mapping[str, DatasetProfile],
) -> MethodSelection:
    profiles = [
        profiles_by_name[name]
        for name in candidate.target_datasets
        if name in profiles_by_name
    ]
    feasibility = evaluate_feasibility(
        MethodGateContext(
            profiles=profiles,
            target_datasets=candidate.target_datasets,
            analysis_mode=candidate.analysis_mode or "descriptive",
            target_column=_target_column(candidate),
        )
    )
    method_id = feasibility.method_id or "descriptive_sql"
    family, recipe, allowed_tools, requirements = _METHOD_OUTPUTS.get(
        method_id,
        _METHOD_OUTPUTS["descriptive_sql"],
    )
    # Families with a dedicated deterministic executor never fall back to the template SQL
    # wording: the reviewable plan must advertise the route that actually executes.
    executor_backed = method_id in {
        "group_comparison",
        "outcome_prediction",
        "anomaly_detection",
    }
    template_route = (
        candidate.template_id is not None
        and candidate.sql_template is not None
        and not executor_backed
    )
    if template_route:
        recipe = _TEMPLATE_RECIPES.get(
            candidate.template_id or "",
            "Run the approved deterministic read-only SQL.",
        )
        allowed_tools = ["read_only_sql"]
    elif method_id == "descriptive_sql":
        recipe = (
            "Use the guarded planner to create read-only SQL after user approval. "
            "The result may support observed claims only."
        )
        allowed_tools = ["llm_planner", "read_only_sql"]

    method = METHOD_REGISTRY.get(method_id)
    feasible = feasibility.status in {"ready", "constrained"}
    execution_ready = feasible and (template_route or bool(method and method.supported))
    return MethodSelection(
        method_family=family,
        method_recipe=recipe,
        allowed_tools=allowed_tools,
        method_requirements=_unique([*requirements, *feasibility.missing]),
        validation_gates=_validation_gates(feasibility),
        execution_ready=execution_ready,
        feasibility=feasibility,
    )


def _target_column(candidate: QuestionCandidate) -> str | None:
    for requirement in candidate.data_requirements:
        prefix, separator, value = requirement.partition(":")
        if separator and prefix.strip().lower() == "target column" and value.strip():
            return value.strip()
    return None


def _validation_gates(feasibility: OpportunityFeasibility) -> list[InvestigationGate]:
    statuses: dict[FeasibilityStatus, GateStatus] = {
        "ready": "passed",
        "constrained": "warning",
        "needs_data": "failed",
        "unsuitable": "failed",
    }
    status = statuses[feasibility.status]
    reasons = feasibility.reasons or [f"Method feasibility is {feasibility.status}."]
    gates = [
        InvestigationGate(name="feasibility", status=status, reason=reason)
        for reason in reasons
    ]
    gates.extend(
        InvestigationGate(
            name="method",
            status="failed",
            reason=f"Missing method requirement: {missing}.",
        )
        for missing in feasibility.missing
    )
    return gates


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value.strip()))
