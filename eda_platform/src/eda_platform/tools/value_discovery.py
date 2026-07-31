from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Sequence

from eda_platform.core.ids import stable_hash
from eda_platform.core.methods import METHOD_REGISTRY, MethodGateContext, evaluate_feasibility
from eda_platform.core.semantic import SemanticSeeds
from eda_platform.schemas.artifacts import Artifact, ArtifactType, DatasetProfile, QualityIssueSet
from eda_platform.schemas.quality_context import QualityContextSet
from eda_platform.schemas.questions import (
    AnalysisMode,
    OpportunityFeasibility,
    QuestionCandidate,
    ValueCategory,
)
from eda_platform.schemas.value_discovery import (
    DatasetValueProfile,
    KnowledgeSummary,
    ValueMap,
    ValueOpportunity,
)
from eda_platform.tools.question_discovery import default_dataset_display_name

_MONEY_TOKENS = {
    "amount",
    "arr",
    "gmv",
    "income",
    "margin",
    "payment",
    "price",
    "profit",
    "revenue",
    "sales",
    "value",
}
_COST_TOKENS = {"budget", "cost", "expense", "spend"}
_ENTITY_TOKENS = {
    "account",
    "client",
    "company",
    "customer",
    "employee",
    "member",
    "partner",
    "patient",
    "product",
    "store",
    "supplier",
    "user",
    "vendor",
}
_OUTCOME_TOKENS = {
    "cancel",
    "churn",
    "complaint",
    "defect",
    "failure",
    "refund",
    "return",
    "risk",
    "status",
}


def build_value_map(
    artifacts: Sequence[Artifact],
    *,
    business_context: str = "",
    semantic_seeds: SemanticSeeds | None = None,
) -> ValueMap:
    """Build conservative, dataset-grounded possibilities for Role 1."""
    profiles = [
        (artifact, DatasetProfile.model_validate(artifact.payload))
        for artifact in artifacts
        if artifact.type is ArtifactType.DATASET_PROFILE
    ]
    quality_artifacts: dict[str, list[Artifact]] = defaultdict(list)
    quality_counts: dict[str, int] = defaultdict(int)
    quality_context_artifacts: dict[str, list[Artifact]] = defaultdict(list)
    quality_context_counts: dict[str, int] = defaultdict(int)
    for artifact in artifacts:
        if artifact.type is ArtifactType.QUALITY_ISSUE_SET:
            issue_set = QualityIssueSet.model_validate(artifact.payload)
            quality_artifacts[issue_set.dataset_id].append(artifact)
            quality_counts[issue_set.dataset_id] += sum(
                issue.severity in {"critical", "warn"}
                for issue in issue_set.issues
            )
        elif artifact.type is ArtifactType.QUALITY_CONTEXT_SET:
            context_set = QualityContextSet.model_validate(artifact.payload)
            quality_context_artifacts[context_set.dataset_id].append(artifact)
            quality_context_counts[context_set.dataset_id] += len(context_set.contexts)

    dataset_profiles: list[DatasetValueProfile] = []
    opportunities: list[ValueOpportunity] = []
    for artifact, profile in sorted(profiles, key=lambda item: item[1].dataset_id):
        capabilities = _capabilities(profile)
        source_ids = [
            artifact.id,
            *[item.id for item in quality_artifacts[profile.dataset_id]],
            *[item.id for item in quality_context_artifacts[profile.dataset_id]],
        ]
        dataset_profiles.append(
            DatasetValueProfile(
                dataset_name=profile.name,
                dataset_display_name=default_dataset_display_name(profile.name),
                capabilities=capabilities,
                source_artifact_ids=source_ids,
                quality_issue_count=quality_counts[profile.dataset_id],
                quality_context_count=quality_context_counts[profile.dataset_id],
            )
        )
        opportunities.extend(
            _opportunities_for_profile(
                profile,
                capabilities=capabilities,
                quality_issue_count=quality_counts[profile.dataset_id],
                source_artifact_ids=source_ids,
            )
        )

    opportunities.sort(key=lambda item: item.opportunity_id)
    return ValueMap(
        business_context_provided=bool(business_context.strip()),
        datasets=dataset_profiles,
        opportunities=opportunities,
        knowledge=_knowledge_summary(semantic_seeds or SemanticSeeds()),
    )


def enrich_question_candidates(
    candidates: Iterable[QuestionCandidate],
    *,
    value_map: ValueMap,
    quality_context_artifacts: Sequence[Artifact] = (),
) -> list[QuestionCandidate]:
    """Populate the frozen flat Question Card fields from a ValueMap."""
    enriched: list[QuestionCandidate] = []
    for candidate in candidates:
        preferred = _best_opportunity(candidate, value_map)
        quality_context_ids = _quality_context_artifact_ids(
            candidate,
            quality_context_artifacts,
        )
        mode = candidate.analysis_mode or _template_analysis_mode(candidate)
        if mode is None and preferred is not None:
            mode = preferred.analysis_mode
        mode = mode or "descriptive"
        feasibility = candidate.feasibility or _opportunity_feasibility(preferred, mode)
        risks = _unique([*candidate.risks, *(preferred.risks if preferred else [])])
        if candidate.score.join_risk > 0.0:
            risks = _unique(
                [
                    *risks,
                    "Relationship evidence is reference-only until a user confirms a join.",
                ]
            )
        source_ids = candidate.source_artifact_ids or (
            preferred.source_artifact_ids
            if preferred is not None
            else _source_ids(candidate, value_map)
        )
        method_id = feasibility.method_id if feasibility is not None else None
        enriched.append(
            candidate.model_copy(
                update={
                    "business_decision": candidate.business_decision
                    or (preferred.decision_action if preferred else ""),
                    "value_hypothesis": candidate.value_hypothesis
                    or (
                        preferred.value_hypothesis
                        if preferred
                        else "The result may improve a subsequent decision."
                    ),
                    "analysis_mode": mode,
                    "candidate_methods": candidate.candidate_methods
                    or ([method_id] if method_id else []),
                    "feasibility": feasibility,
                    "risks": risks,
                    "proposed_action": candidate.proposed_action
                    or _proposed_action(feasibility, mode),
                    "value_category": candidate.value_category
                    or (preferred.value_category if preferred else "decision_quality"),
                    "data_signal": candidate.data_signal
                    or (preferred.data_signal if preferred else _fallback_signal()),
                    "priority_rationale": candidate.priority_rationale
                    or _priority_rationale(preferred),
                    "source_artifact_ids": _unique([*source_ids, *quality_context_ids]),
                    "quality_context_artifact_ids": _unique(
                        [*candidate.quality_context_artifact_ids, *quality_context_ids]
                    ),
                }
            )
        )
    return enriched


def _capabilities(profile: DatasetProfile) -> list[str]:
    columns = profile.columns_detail
    names = _column_tokens(column.name for column in columns)
    capabilities: list[str] = []
    if any(column.semantic_type == "datetime" for column in columns):
        capabilities.append("time axis")
    if any(column.semantic_type == "numeric" for column in columns):
        capabilities.append("measurable numeric fields")
    if any(
        column.semantic_type in {"categorical", "boolean"} and 2 <= column.unique_count <= 50
        for column in columns
    ):
        capabilities.append("segmentable groups")
    if profile.primary_key_candidates or any(column.semantic_type == "id" for column in columns):
        capabilities.append("entity-level records")
    if names & _MONEY_TOKENS:
        capabilities.append("money-like measure inferred from field names")
    if names & _COST_TOKENS:
        capabilities.append("cost-like measure inferred from field names")
    if names & _ENTITY_TOKENS:
        capabilities.append("named entities inferred from field names")
    if names & _OUTCOME_TOKENS:
        capabilities.append("outcome or status signal inferred from field names")
    return capabilities


def _opportunities_for_profile(
    profile: DatasetProfile,
    *,
    capabilities: list[str],
    quality_issue_count: int,
    source_artifact_ids: list[str],
) -> list[ValueOpportunity]:
    dataset_name = profile.name
    display_name = default_dataset_display_name(dataset_name)
    opportunities: list[ValueOpportunity] = []
    tokens = _column_tokens(column.name for column in profile.columns_detail)
    has_time = "time axis" in capabilities
    has_numeric = "measurable numeric fields" in capabilities
    has_segments = "segmentable groups" in capabilities

    if has_time and has_numeric:
        opportunities.append(
            _opportunity(
                profile=profile,
                category=(
                    "financial_performance" if tokens & _MONEY_TOKENS else "decision_quality"
                ),
                signal=f"{display_name} has a time axis and measurable numeric fields.",
                hypothesis=(
                    "Trend changes may reveal where timing, capacity, or allocation decisions "
                    "deserve attention; this is not a financial impact estimate."
                ),
                action="Review periods with material changes before changing plans or allocations.",
                analysis_mode="forecast",
                source_artifact_ids=source_artifact_ids,
            )
        )
    if has_segments and has_numeric:
        opportunities.append(
            _opportunity(
                profile=profile,
                category=(
                    "customer_or_entity" if tokens & _ENTITY_TOKENS else "decision_quality"
                ),
                signal=f"{display_name} can compare measurable outcomes across groups.",
                hypothesis=(
                    "Differences between groups may help focus the next operational, product, "
                    "or service decision; any cause still requires validation."
                ),
                action="Prioritize the groups that need follow-up or process review.",
                analysis_mode="diagnostic",
                source_artifact_ids=source_artifact_ids,
            )
        )
    if tokens & _COST_TOKENS:
        opportunities.append(
            _opportunity(
                profile=profile,
                category="cost_efficiency",
                signal=f"{display_name} contains a cost-like field inferred from its name.",
                hypothesis=(
                    "Variation in a cost-like measure may identify processes or segments worth "
                    "reviewing for efficiency; the field meaning must be confirmed first."
                ),
                action="Confirm the metric definition, then review high or changing values.",
                analysis_mode="descriptive",
                risks=["Confirm the field meaning before treating it as a cost metric."],
                source_artifact_ids=source_artifact_ids,
            )
        )
    if tokens & _OUTCOME_TOKENS:
        opportunities.append(
            _opportunity(
                profile=profile,
                category="risk_or_service",
                signal=(
                    f"{display_name} contains a status or outcome-like field inferred "
                    "from its name."
                ),
                hypothesis=(
                    "Outcome patterns may point to risk, quality, or service issues that merit "
                    "targeted investigation; the outcome definition must be confirmed first."
                ),
                action="Confirm the outcome definition, then review unfavorable patterns.",
                analysis_mode="segmentation" if has_segments else "descriptive",
                risks=["Confirm the field meaning before using it as an outcome measure."],
                source_artifact_ids=source_artifact_ids,
            )
        )
    if quality_issue_count:
        opportunities.append(
            _opportunity(
                profile=profile,
                category="decision_quality",
                signal=f"{display_name} has {quality_issue_count} recorded data-quality issue(s).",
                hypothesis=(
                    "Resolving material data-quality issues may prevent misleading analysis and "
                    "improve confidence in later decisions."
                ),
                action="Address material quality issues before relying on affected results.",
                analysis_mode="descriptive",
                risks=["Affected fields may bias or limit the analysis."],
                source_artifact_ids=source_artifact_ids,
            )
        )
    return opportunities


def _opportunity(
    *,
    profile: DatasetProfile,
    category: ValueCategory,
    signal: str,
    hypothesis: str,
    action: str,
    analysis_mode: AnalysisMode,
    source_artifact_ids: list[str],
    risks: list[str] | None = None,
) -> ValueOpportunity:
    feasibility = evaluate_feasibility(
        MethodGateContext(
            profiles=[profile],
            target_datasets=[profile.name],
            analysis_mode=analysis_mode,
            target_column=None,
        )
    )
    return ValueOpportunity(
        opportunity_id="value_"
        + stable_hash(
            {
                "dataset": profile.name,
                "category": category,
                "analysis_mode": analysis_mode,
            },
            length=10,
        ),
        value_category=category,
        target_datasets=[profile.name],
        data_signal=signal,
        value_hypothesis=hypothesis,
        decision_action=action,
        analysis_mode=analysis_mode,
        feasibility=feasibility.status,
        feasibility_reasons=feasibility.reasons,
        risks=risks or [],
        source_artifact_ids=source_artifact_ids,
    )


def _knowledge_summary(seeds: SemanticSeeds) -> KnowledgeSummary:
    return KnowledgeSummary(
        field_meanings=[f"{item.dataset}.{item.column}" for item in seeds.field_meanings],
        metric_definitions=[item.name for item in seeds.metric_definitions],
        entity_notes=[item.name for item in seeds.entity_notes],
        verified_relations=[f"{item.left} = {item.right}" for item in seeds.verified_relations],
    )


def _best_opportunity(
    candidate: QuestionCandidate,
    value_map: ValueMap,
) -> ValueOpportunity | None:
    matching = [
        opportunity
        for opportunity in value_map.opportunities
        if set(opportunity.target_datasets) & set(candidate.target_datasets)
    ]
    if candidate.template_id == "quality_missing":
        matching = [
            item for item in matching if item.value_category == "decision_quality"
        ] or matching
    preferred_mode = candidate.analysis_mode or _template_analysis_mode(candidate)
    if preferred_mode is not None:
        matching.sort(
            key=lambda item: (item.analysis_mode != preferred_mode, item.opportunity_id)
        )
    return matching[0] if matching else None


def _template_analysis_mode(candidate: QuestionCandidate) -> AnalysisMode | None:
    modes: dict[str, AnalysisMode] = {
        "trend": "descriptive",
        "group_difference": "diagnostic",
        "correlation_probe": "descriptive",
        "quality_missing": "descriptive",
    }
    return modes.get(candidate.template_id or "")


def _opportunity_feasibility(
    opportunity: ValueOpportunity | None,
    mode: AnalysisMode,
) -> OpportunityFeasibility | None:
    if opportunity is None:
        return None
    method_id = next(
        (method.method_id for method in METHOD_REGISTRY.values() if method.mode == mode),
        None,
    )
    return OpportunityFeasibility(
        status=opportunity.feasibility,
        method_id=method_id,
        reasons=opportunity.feasibility_reasons,
    )


def _quality_context_artifact_ids(
    candidate: QuestionCandidate,
    artifacts: Sequence[Artifact],
) -> list[str]:
    artifact_ids: list[str] = []
    for artifact in artifacts:
        if artifact.type is not ArtifactType.QUALITY_CONTEXT_SET:
            continue
        context_set = QualityContextSet.model_validate(artifact.payload)
        if context_set.dataset_name not in candidate.target_datasets:
            continue
        referenced = set(candidate.referenced_columns.get(context_set.dataset_name, []))
        selected = [
            context
            for context in context_set.contexts
            if not referenced or context.column is None or context.column in referenced
        ]
        if selected:
            artifact_ids.append(artifact.id)
    return _unique(artifact_ids)


def _proposed_action(
    feasibility: OpportunityFeasibility | None,
    mode: AnalysisMode,
) -> str:
    if feasibility is not None and feasibility.status in {"needs_data", "unsuitable"}:
        return "collect_data"
    if mode == "causal_experiment":
        return "design_experiment"
    return "run_analysis"


def _fallback_signal() -> str:
    return "The question is grounded in dataset profiles and EDA evidence."


def _priority_rationale(opportunity: ValueOpportunity | None) -> str:
    if opportunity is not None:
        return (
            "Prioritized because the supporting EDA capability can inform a concrete "
            "next decision, subject to the stated risks."
        )
    return "Prioritized from available data signals; business impact remains a hypothesis."


def _source_ids(candidate: QuestionCandidate, value_map: ValueMap) -> list[str]:
    ids: list[str] = []
    for profile in value_map.datasets:
        if profile.dataset_name in candidate.target_datasets:
            ids.extend(profile.source_artifact_ids)
    return _unique(ids)


def _column_tokens(columns: Iterable[str]) -> set[str]:
    return {
        token
        for column in columns
        for token in re.findall(r"[a-z0-9]+", column.lower())
    }


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value.strip()))
