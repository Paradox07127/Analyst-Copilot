from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

from eda_platform.core.column_roles import ColumnRoleName, ColumnRoleSet
from eda_platform.core.ids import stable_hash
from eda_platform.core.methods import MethodGateContext, evaluate_feasibility
from eda_platform.core.semantic import JoinWhitelist, JoinWhitelistEntry, SemanticSeeds
from eda_platform.schemas.artifacts import (
    AnalysisTable,
    Artifact,
    ArtifactType,
    ColumnProfile,
    DatasetProfile,
    QualityIssue,
    QualityIssueSet,
)
from eda_platform.schemas.questions import (
    QuestionAnswerContract,
    QuestionCandidate,
    QuestionCandidateSet,
    QuestionScore,
)
from eda_platform.schemas.relations import (
    RelationshipCandidate,
    RelationshipCandidateSet,
    RelationshipValidation,
    RelationshipValidationSet,
)
from eda_platform.tools.domain_metrics import (
    applicable_metrics,
    background_section_for,
    metric_definition,
)
from eda_platform.tools.loader import LoadedDataset
from eda_platform.tools.relationship_discovery import _quote_identifier, _relation_name
from eda_platform.tools.sql_names import safe_alias

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "over",
    "the",
    "to",
    "what",
    "which",
    "with",
}
_CORRELATION_THRESHOLD = 0.75
_MISSING_THRESHOLD = 20.0


@dataclass(frozen=True)
class _DatasetContext:
    loaded: LoadedDataset
    profile_artifact: Artifact
    profile: DatasetProfile
    quality_artifacts: list[Artifact]
    quality_issues: list[QualityIssue]
    analysis_tables: list[AnalysisTable]
    # DI8-E: the dataset's semantic role cache (DI8-B), when the caller ran the
    # bootstrap. Roles gate template candidate pools only — never numbers.
    role_set: ColumnRoleSet | None = None

    @property
    def dataset_id(self) -> str:
        return self.profile.dataset_id

    @property
    def dataset_name(self) -> str:
        return self.profile.name

    @property
    def display_name(self) -> str:
        return default_dataset_display_name(self.profile.name)

    @property
    def relation_name(self) -> str:
        return _relation_name(self.profile.dataset_id)


def discover_question_candidates(
    datasets: Sequence[LoadedDataset],
    *,
    profile_artifacts: Sequence[Artifact],
    quality_artifacts: Sequence[Artifact] = (),
    analysis_artifacts: Sequence[Artifact] = (),
    relationship_candidates: RelationshipCandidateSet | None = None,
    relationship_validations: RelationshipValidationSet | None = None,
    llm_candidates: Sequence[QuestionCandidate] = (),
    include_template_candidates: bool = True,
    template_backstop_only: bool = False,
    column_role_sets: Mapping[str, ColumnRoleSet] | None = None,
    join_whitelist: JoinWhitelist | None = None,
    semantic_seeds: SemanticSeeds | None = None,
) -> QuestionCandidateSet:
    """Assemble the run's question candidates."""
    contexts = _contexts(
        datasets,
        profile_artifacts=profile_artifacts,
        quality_artifacts=quality_artifacts,
        analysis_artifacts=analysis_artifacts,
        column_role_sets=column_role_sets,
    )
    relation_map = _relationship_map(relationship_candidates)
    validation_map = _validation_map(relationship_validations)
    profiles = [context.profile for context in contexts]
    llm_list: list[QuestionCandidate] = [
        _ensure_candidate_feasibility(candidate, profiles=profiles) for candidate in llm_candidates
    ]
    trivial_dropped = 0

    template_list: list[QuestionCandidate] = []
    cross_table_list: list[QuestionCandidate] = []
    domain_metric_list: list[QuestionCandidate] = []
    if include_template_candidates or template_backstop_only:
        for context in contexts:
            template_list.extend(
                _trend_questions(
                    context,
                    relation_map=relation_map,
                    validation_map=validation_map,
                )
            )
            group_candidates, group_dropped = _group_difference_questions(
                context,
                relation_map=relation_map,
                validation_map=validation_map,
            )
            template_list.extend(group_candidates)
            trivial_dropped += group_dropped
            correlation_candidates, dropped = _correlation_questions(
                context,
                relation_map=relation_map,
                validation_map=validation_map,
            )
            template_list.extend(correlation_candidates)
            trivial_dropped += dropped
            template_list.extend(
                _quality_questions(
                    context,
                    relation_map=relation_map,
                    validation_map=validation_map,
                )
            )
        cross_table_list = _cross_table_questions(
            contexts,
            join_whitelist=join_whitelist,
            relation_map=relation_map,
            validation_map=validation_map,
        )
        domain_metric_list = _domain_metric_questions(
            contexts,
            column_role_sets=column_role_sets,
            join_whitelist=join_whitelist,
            relation_map=relation_map,
            validation_map=validation_map,
            semantic_seeds=semantic_seeds,
        )

    backstop_used = 0
    backstop_categories: list[str] = []
    if template_backstop_only and llm_list:
        # Only the coverage-backstop template pool is trimmed here.
        template_list, backstop_categories = select_backstop_candidates(llm_list, template_list)
        backstop_used = len(template_list)

    result = rank_and_deduplicate_questions(
        [*llm_list, *template_list, *cross_table_list, *domain_metric_list],
        trivial_dropped=trivial_dropped,
    )
    return result.model_copy(
        update={
            "template_backstop_used": backstop_used,
            "template_backstop_categories": backstop_categories,
        }
    )


# DI8-E coverage backstop checklist.
_REQUIRED_CATEGORIES: tuple[str, ...] = ("trend", "group_difference", "quality")
_TEMPLATE_FAMILY_CATEGORY: dict[str, str] = {
    "trend": "trend",
    "group_difference": "group_difference",
    "quality_missing": "quality",
}
_CATEGORY_TOKENS: dict[str, frozenset[str]] = {
    "trend": frozenset(
        {
            "trend",
            "trending",
            "trends",
            "time",
            "month",
            "monthly",
            "week",
            "weekly",
            "quarter",
            "quarterly",
            "year",
            "yearly",
            "seasonal",
            "seasonality",
            "growth",
            "decline",
            "evolution",
            "change",
            "changing",
        }
    ),
    "group_difference": frozenset(
        {
            "group",
            "groups",
            "segment",
            "segments",
            "difference",
            "differences",
            "differ",
            "compare",
            "comparison",
            "across",
            "versus",
            "vs",
            "highest",
            "lowest",
            "vary",
            "varies",
            "between",
        }
    ),
    "quality": frozenset(
        {
            "missing",
            "incomplete",
            "completeness",
            "null",
            "nulls",
            "quality",
            "gaps",
            "blank",
        }
    ),
}
_BACKSTOP_PER_CATEGORY = 2


def question_coverage(candidate: QuestionCandidate) -> set[str]:
    """Which required categories one (free-form) question covers. Pure."""
    tokens = set(normalized_question_key(candidate.question_en).split())
    covered = {category for category, keywords in _CATEGORY_TOKENS.items() if tokens & keywords}
    if candidate.analysis_mode == "forecast":
        covered.add("trend")
    if candidate.analysis_mode == "segmentation":
        covered.add("group_difference")
    return covered


def select_backstop_candidates(
    llm_candidates: Sequence[QuestionCandidate],
    template_candidates: Sequence[QuestionCandidate],
    *,
    per_category: int = _BACKSTOP_PER_CATEGORY,
) -> tuple[list[QuestionCandidate], list[str]]:
    """Keep only the template questions that fill categories the LLM missed."""
    covered: set[str] = set()
    for candidate in llm_candidates:
        covered |= question_coverage(candidate)
    missing = [category for category in _REQUIRED_CATEGORIES if category not in covered]
    kept: list[QuestionCandidate] = []
    for category in missing:
        family = sorted(
            (
                candidate
                for candidate in template_candidates
                if _TEMPLATE_FAMILY_CATEGORY.get(candidate.template_id or "") == category
            ),
            key=lambda candidate: (
                -candidate.score.deterministic_score,
                candidate.question_id,
            ),
        )
        kept.extend(family[:per_category])
    return kept, missing


def rank_and_deduplicate_questions(
    candidates: Sequence[QuestionCandidate],
    *,
    trivial_dropped: int = 0,
) -> QuestionCandidateSet:
    ranked = sorted(
        candidates,
        key=lambda candidate: (-candidate.score.deterministic_score, candidate.question_id),
    )
    seen: set[str] = set()
    deduped: list[QuestionCandidate] = []
    dropped = 0
    for candidate in ranked:
        candidate = _ensure_answer_contract(candidate)
        key = _candidate_dedup_key(candidate)
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        deduped.append(candidate)
    return QuestionCandidateSet(
        candidates=deduped,
        dedup_dropped=dropped,
        trivial_dropped=trivial_dropped,
    )


def _ensure_answer_contract(candidate: QuestionCandidate) -> QuestionCandidate:
    if candidate.answer_contract is not None:
        return candidate
    if candidate.metric_id:
        return candidate.model_copy(
            update={
                "answer_contract": QuestionAnswerContract(
                    kind="metric",
                    metric_id=candidate.metric_id,
                    abstention_code="metric_contract_failed",
                )
            }
        )
    if "threshold" in normalized_question_key(candidate.question_en).split():
        return candidate.model_copy(
            update={
                "answer_contract": QuestionAnswerContract(
                    kind="threshold",
                    required_column_tokens=["threshold"],
                    abstention_code="answer_schema_mismatch",
                )
            }
        )
    return candidate


def _candidate_dedup_key(candidate: QuestionCandidate) -> str:
    text = candidate.question_en
    for dataset in candidate.target_datasets:
        display_name = candidate.dataset_display_names.get(
            dataset, default_dataset_display_name(dataset)
        )
        text = text.replace(dataset, display_name)
    return normalized_question_key(text)


def normalized_question_key(text: str) -> str:
    tokens = [token for token in re.findall(r"[0-9a-z]+", text.lower()) if token not in _STOPWORDS]
    return " ".join(tokens)


# DI10-W2 auto-execution funnel.
_AUTO_EXEC_MAX_TOTAL = 10
_AUTO_EXEC_TEMPLATE_TOP_N = 2
_AUTO_EXEC_MIN_EXPLORATORY = 1
# Floors guarantee coverage; the fill spends what the floors left over. Without
# it a single-domain-metric dataset executed 2 of 10 allowed slots and left every
# high-scoring LLM business question unrun (2026-07-22 audit). No separate cap:
# every other lane is taken before the fill runs, so none of them can be starved,
# and once the domain-agnostic metrics left the budget a lower ceiling only meant
# idle slots on a dataset with no business metrics at all.
_AUTO_EXEC_MAX_EXPLORATORY = _AUTO_EXEC_MAX_TOTAL
# Business metrics still compete for slots; only the domain-agnostic ones left.
# Uncapped, the metric lane took the World Cup run's top three slots for stadium-
# capacity HHI and a date range while five LLM questions went unrun.
_AUTO_EXEC_MAX_DOMAIN_METRIC = 2


def select_auto_execution_set(
    candidate_set: QuestionCandidateSet,
    *,
    max_total: int = _AUTO_EXEC_MAX_TOTAL,
    template_top_n: int = _AUTO_EXEC_TEMPLATE_TOP_N,
    min_exploratory: int = _AUTO_EXEC_MIN_EXPLORATORY,
    max_exploratory: int = _AUTO_EXEC_MAX_EXPLORATORY,
    max_domain_metric: int = _AUTO_EXEC_MAX_DOMAIN_METRIC,
) -> list[QuestionCandidate]:
    """Pick the questions the run executes automatically (DI10-W2 rules above)."""
    candidates = candidate_set.candidates

    # 1. domain_metric lane: the best few, deterministically ordered. Background
    # metrics leave it: they describe the data, resolve on every dataset, and
    # cost no LLM call, so they run outside the analysis budget rather than
    # winning slots from questions that answer something (2026-08-04 FIFA run).
    resolved_metrics = [
        candidate
        for candidate in candidates
        if candidate.template_id == "domain_metric" and candidate.sql_template is not None
    ]
    background_metrics = [
        candidate for candidate in resolved_metrics if is_background_metric(candidate)
    ]
    domain_metrics = sorted(
        (
            candidate
            for candidate in resolved_metrics
            if not is_background_metric(candidate)
        ),
        key=lambda item: (-item.score.deterministic_score, item.question_id),
    )

    # 2. exploratory floor: LLM free-form questions by composite score.
    exploratory_pool = sorted(
        (
            candidate
            for candidate in candidates
            if candidate.exploratory
            and candidate.origin == "llm"
            and _feasibility_allows_execution(candidate)
        ),
        key=_exploratory_rank_key,
    )

    # 3. other templates: previous eligibility rules, one per template family.
    template_pool = sorted(
        (
            candidate
            for candidate in candidates
            if candidate.template_id != "domain_metric"
            and _template_eligible_for_auto_execution(candidate)
        ),
        key=lambda item: (-item.score.deterministic_score, item.question_id),
    )

    selected: list[QuestionCandidate] = []
    selected_ids: set[str] = set()

    def take(candidate: QuestionCandidate) -> None:
        if candidate.question_id in selected_ids:
            return
        selected.append(candidate.model_copy(update={"status": "auto_selected"}))
        selected_ids.add(candidate.question_id)

    for candidate in background_metrics:
        take(candidate)
    background_ids = {candidate.question_id for candidate in background_metrics}

    def analysis_count() -> int:
        return sum(
            1 for candidate in selected if candidate.question_id not in background_ids
        )

    for candidate in domain_metrics[: min(max_domain_metric, max_total)]:
        take(candidate)
    # The exploratory floor is reserved BEFORE the template top-up so the cap
    # can never squeeze it out.
    for candidate in exploratory_pool[:min_exploratory]:
        if analysis_count() >= max_total:
            break
        take(candidate)
    used_templates: set[str] = set()
    template_taken = 0
    for candidate in template_pool:
        if template_taken >= template_top_n or analysis_count() >= max_total:
            break
        family = candidate.template_id or candidate.origin
        if family in used_templates or candidate.question_id in selected_ids:
            continue
        take(candidate)
        used_templates.add(family)
        template_taken += 1

    # 4. fill: spend what the floors left over on the best remaining questions,
    # so a thin domain-metric catalogue no longer means a two-question run.
    exploratory_ids = {item.question_id for item in exploratory_pool}
    exploratory_taken = sum(1 for candidate in selected if candidate.question_id in exploratory_ids)
    for candidate in exploratory_pool:
        if analysis_count() >= max_total or exploratory_taken >= max_exploratory:
            break
        if candidate.question_id in selected_ids:
            continue
        take(candidate)
        exploratory_taken += 1
    analysis = [c for c in selected if c.question_id not in background_ids][:max_total]
    kept = {c.question_id for c in analysis} | background_ids
    return [c for c in selected if c.question_id in kept]


def is_background_metric(candidate: QuestionCandidate) -> bool:
    return background_section_for(candidate.metric_id) is not None


def auto_execution_composition(
    selected: Sequence[QuestionCandidate],
) -> dict[str, int]:
    """Selection make-up for trace telemetry (n_* keys sum to selected_count)."""
    n_background = sum(1 for candidate in selected if is_background_metric(candidate))
    n_domain_metric = sum(
        1
        for candidate in selected
        if candidate.template_id == "domain_metric" and not is_background_metric(candidate)
    )
    n_exploratory = sum(
        1
        for candidate in selected
        if candidate.exploratory and candidate.template_id != "domain_metric"
    )
    return {
        "selected_count": len(selected),
        "n_background": n_background,
        "n_domain_metric": n_domain_metric,
        "n_exploratory": n_exploratory,
        "n_template": len(selected) - n_background - n_domain_metric - n_exploratory,
    }


def _exploratory_rank_key(candidate: QuestionCandidate) -> tuple[float, str]:
    """Return the composite ranking key for an exploratory question."""
    score = candidate.score
    composite = (
        score.deterministic_score
        + (score.llm_business_relevance or 0.0)
        + (score.llm_actionability or 0.0)
    )
    return (-composite, candidate.question_id)


def _feasibility_allows_execution(candidate: QuestionCandidate) -> bool:
    return candidate.feasibility is None or candidate.feasibility.status not in {
        "needs_data",
        "unsuitable",
    }


def _template_eligible_for_auto_execution(candidate: QuestionCandidate) -> bool:
    """Return whether a template candidate is eligible for automatic execution."""
    if candidate.origin != "template" or candidate.sql_template is None:
        return False
    if not _feasibility_allows_execution(candidate):
        return False
    if candidate.score.join_risk > 0.3 or candidate.score.quality_risk > 0.6:
        return False
    return not candidate.required_relations


def default_dataset_display_name(dataset_name: str) -> str:
    """Turn a file name into a readable fallback when no business label exists."""
    stem = Path(dataset_name).stem
    words = [word for word in re.split(r"[_\-]+", stem) if word]
    return " ".join(word.capitalize() for word in words) or "Dataset"


def make_question_id(
    *,
    origin: str,
    question_en: str,
    target_datasets: Sequence[str],
    required_relations: Sequence[str] = (),
    template_id: str | None = None,
) -> str:
    return "q_" + stable_hash(
        {
            "origin": origin,
            "template_id": template_id,
            "question_en": question_en,
            "target_datasets": list(target_datasets),
            "required_relations": list(required_relations),
        },
        length=10,
    )


def score_question(
    *,
    profile_artifacts: Sequence[Artifact],
    quality_artifacts: Sequence[Artifact] = (),
    target_datasets: Sequence[str],
    referenced_columns: Mapping[str, Sequence[str]] | None = None,
    statistical_signal: float,
    required_relations: Sequence[str] = (),
    relationship_candidates: RelationshipCandidateSet | None = None,
    relationship_validations: RelationshipValidationSet | None = None,
    llm_business_relevance: float | None = None,
    llm_actionability: float | None = None,
    column_role_sets: Mapping[str, ColumnRoleSet] | None = None,
    confirmed_relations: Collection[str] = (),
) -> QuestionScore:
    profiles = [
        DatasetProfile.model_validate(artifact.payload)
        for artifact in profile_artifacts
        if artifact.type is ArtifactType.DATASET_PROFILE
    ]
    issues_by_dataset = _issues_by_dataset(quality_artifacts)
    relation_map = _relationship_map(relationship_candidates)
    validation_map = _validation_map(relationship_validations)
    availability = _data_availability(
        profiles,
        target_datasets=target_datasets,
        referenced_columns=referenced_columns or {},
    )
    quality_risk = _quality_risk(
        issues_by_dataset,
        profiles=profiles,
        target_datasets=target_datasets,
        referenced_columns=referenced_columns or {},
    )
    join_risk = _join_risk(
        required_relations,
        relation_map,
        validation_map,
        confirmed_relations=confirmed_relations,
    )
    # QuickInsights-style impact weighting.
    impact = _impact_weight(column_role_sets, referenced_columns=referenced_columns or {})
    return _score(
        data_availability=availability,
        statistical_signal=statistical_signal * impact,
        quality_risk=quality_risk,
        join_risk=join_risk,
        llm_business_relevance=llm_business_relevance,
        llm_actionability=llm_actionability,
    )


def _impact_weight(
    column_role_sets: Mapping[str, ColumnRoleSet] | None,
    *,
    referenced_columns: Mapping[str, Sequence[str]],
) -> float:
    """Min impact weight over every referenced column (1.0 when unknown)."""
    if not column_role_sets:
        return 1.0
    weights: list[float] = []
    for dataset, columns in referenced_columns.items():
        role_set = column_role_sets.get(dataset)
        if role_set is None:
            continue
        weights.extend(role_set.impact_weight(column) for column in columns)
    return min(weights) if weights else 1.0


def _trend_questions(
    context: _DatasetContext,
    *,
    relation_map: Mapping[str, RelationshipCandidate],
    validation_map: Mapping[str, RelationshipValidation],
) -> list[QuestionCandidate]:
    # DI8-E role gate: verified identifier/sequence columns never become trend metrics —
    # no more "how is order_item_id trending".
    role_excluded = (
        context.role_set.excluded_from_stats() if context.role_set is not None else set()
    )
    datetime_columns = _columns_by_type(context.profile.columns_detail, {"datetime"})
    if context.role_set is not None:
        datetime_columns = [
            column
            for column in datetime_columns
            if (role := context.role_set.role_of(column.name)) is None
            or role.provenance == "unverified"
            or role.role is ColumnRoleName.TIMESTAMP
        ]
    numeric_columns = _eligible_metric_columns(
        context,
        [
            column
            for column in _columns_by_type(context.profile.columns_detail, {"numeric"})
            if column.name not in role_excluded
        ],
    )
    if not datetime_columns or not numeric_columns:
        return []
    candidates: list[QuestionCandidate] = []
    for date_column in datetime_columns[:2]:
        for metric_column in numeric_columns[:2]:
            question = (
                f"How is {metric_column.name} trending over {date_column.name} "
                f"in {context.display_name}?"
            )
            candidates.append(
                _candidate(
                    question_en=question,
                    template_id="trend",
                    target_datasets=[context.dataset_name],
                    referenced_columns={
                        context.dataset_name: [date_column.name, metric_column.name]
                    },
                    sql_template=_trend_sql(context, date_column.name, metric_column.name),
                    statistical_signal=_trend_signal(
                        context.loaded.frame,
                        date_column.name,
                        metric_column.name,
                    ),
                    context_artifacts=[context.profile_artifact],
                    quality_artifacts=context.quality_artifacts,
                    relation_map=relation_map,
                    validation_map=validation_map,
                )
            )
    return candidates


def _group_difference_questions(
    context: _DatasetContext,
    *,
    relation_map: Mapping[str, RelationshipCandidate],
    validation_map: Mapping[str, RelationshipValidation],
) -> tuple[list[QuestionCandidate], int]:
    excluded_columns = _excluded_template_columns(context)
    raw_categories = [
        column
        for column in context.profile.columns_detail
        if column.semantic_type in {"categorical", "boolean"} and 2 <= column.unique_count <= 50
    ]
    categories = [column for column in raw_categories if column.name not in excluded_columns]
    raw_metrics = _columns_by_type(context.profile.columns_detail, {"numeric"})
    metrics = _eligible_metric_columns(
        context,
        [column for column in raw_metrics if column.name not in excluded_columns],
    )
    trivial_dropped = len(raw_categories) - len(categories) + len(raw_metrics) - len(metrics)
    candidates: list[QuestionCandidate] = []
    for category in categories[:2]:
        for metric in metrics[:2]:
            question = (
                f"Which {category.name} groups have the highest average {metric.name} "
                f"in {context.display_name}?"
            )
            candidates.append(
                _candidate(
                    question_en=question,
                    template_id="group_difference",
                    target_datasets=[context.dataset_name],
                    referenced_columns={context.dataset_name: [category.name, metric.name]},
                    sql_template=_group_sql(context, category.name, metric.name),
                    statistical_signal=_group_signal(
                        context.loaded.frame,
                        category.name,
                        metric.name,
                    ),
                    context_artifacts=[context.profile_artifact],
                    quality_artifacts=context.quality_artifacts,
                    relation_map=relation_map,
                    validation_map=validation_map,
                )
            )
    return candidates, trivial_dropped


def _correlation_questions(
    context: _DatasetContext,
    *,
    relation_map: Mapping[str, RelationshipCandidate],
    validation_map: Mapping[str, RelationshipValidation],
) -> tuple[list[QuestionCandidate], int]:
    candidates: list[QuestionCandidate] = []
    trivial_dropped = 0
    excluded_columns = _excluded_template_columns(context)
    for table in context.analysis_tables:
        if table.kind != "correlation":
            continue
        for row in table.rows:
            if bool(row.get("is_trivial_pair")):
                trivial_dropped += 1
                continue
            abs_r = _float(row.get("abs_pearson"))
            if abs_r >= 0.999:
                trivial_dropped += 1
                continue
            if abs_r < _CORRELATION_THRESHOLD:
                continue
            column_a = str(row["column_a"])
            column_b = str(row["column_b"])
            if column_a in excluded_columns or column_b in excluded_columns:
                trivial_dropped += 1
                continue
            question = (
                f"How strongly are {column_a} and {column_b} related in {context.display_name}?"
            )
            candidates.append(
                _candidate(
                    question_en=question,
                    template_id="correlation_probe",
                    target_datasets=[context.dataset_name],
                    referenced_columns={context.dataset_name: [column_a, column_b]},
                    sql_template=_correlation_sql(context, column_a, column_b),
                    statistical_signal=abs_r,
                    context_artifacts=[context.profile_artifact],
                    quality_artifacts=context.quality_artifacts,
                    relation_map=relation_map,
                    validation_map=validation_map,
                )
            )
    return candidates, trivial_dropped


def _quality_questions(
    context: _DatasetContext,
    *,
    relation_map: Mapping[str, RelationshipCandidate],
    validation_map: Mapping[str, RelationshipValidation],
) -> list[QuestionCandidate]:
    candidates: list[QuestionCandidate] = []
    for column in context.profile.columns_detail:
        if column.missing_percent < _MISSING_THRESHOLD:
            continue
        question = f"How much data is missing from {column.name} in {context.display_name}?"
        candidates.append(
            _candidate(
                question_en=question,
                template_id="quality_missing",
                target_datasets=[context.dataset_name],
                referenced_columns={context.dataset_name: [column.name]},
                sql_template=_missing_sql(context, column.name),
                statistical_signal=min(1.0, column.missing_percent / 100.0),
                context_artifacts=[context.profile_artifact],
                quality_artifacts=context.quality_artifacts,
                relation_map=relation_map,
                validation_map=validation_map,
            )
        )
    return candidates


# DI8-C cross-table template family: measures grouped by another table's dimension over a
# **confirmed** whitelist join.
_MAX_CROSS_TABLE_METRICS = 2
_MAX_CROSS_TABLE_DIMENSIONS = 1


def _cross_table_questions(
    contexts: Sequence[_DatasetContext],
    *,
    join_whitelist: JoinWhitelist | None,
    relation_map: Mapping[str, RelationshipCandidate],
    validation_map: Mapping[str, RelationshipValidation],
) -> list[QuestionCandidate]:
    if join_whitelist is None:
        return []
    by_name = {context.dataset_name: context for context in contexts}
    dataset_ids_by_name = {context.dataset_name: context.dataset_id for context in contexts}
    candidates: list[QuestionCandidate] = []
    for entry in join_whitelist.entries_for(set(by_name)):  # DI10-W5 dataset scoping
        if not entry.is_usable(dataset_ids_by_name):
            continue
        if entry.cardinality == "many_to_many":
            continue
        left = by_name.get(entry.left_dataset)
        right = by_name.get(entry.right_dataset)
        if left is None or right is None:
            continue
        if not entry.left_columns or len(entry.left_columns) != len(entry.right_columns):
            continue
        metrics = _cross_table_metrics(left, exclude=set(entry.left_columns))
        dimensions = _cross_table_dimensions(right, exclude=set(entry.right_columns))
        label = entry.label()
        for metric in metrics[:_MAX_CROSS_TABLE_METRICS]:
            for dimension in dimensions[:_MAX_CROSS_TABLE_DIMENSIONS]:
                question = (
                    f"What is the average {metric.name} by {dimension.name} "
                    f"across {left.display_name} and {right.display_name}?"
                )
                candidates.append(
                    _candidate(
                        question_en=question,
                        template_id="cross_table_aggregation",
                        target_datasets=[left.dataset_name, right.dataset_name],
                        referenced_columns={
                            left.dataset_name: [metric.name, *entry.left_columns],
                            right.dataset_name: [dimension.name, *entry.right_columns],
                        },
                        sql_template=_cross_table_sql(
                            left, right, entry, metric.name, dimension.name
                        ),
                        statistical_signal=0.5,
                        context_artifacts=[left.profile_artifact, right.profile_artifact],
                        quality_artifacts=[
                            *left.quality_artifacts,
                            *right.quality_artifacts,
                        ],
                        relation_map=relation_map,
                        validation_map=validation_map,
                        required_relations=[label],
                        confirmed_relations={label},
                    )
                )
    return candidates


def _cross_table_metrics(context: _DatasetContext, *, exclude: set[str]) -> list[ColumnProfile]:
    """Metric candidates on the FK side: verified measures first, else numeric."""
    excluded = _excluded_template_columns(context) | exclude
    numeric = _eligible_metric_columns(
        context,
        [
            column
            for column in _columns_by_type(context.profile.columns_detail, {"numeric"})
            if column.name not in excluded
        ],
    )
    if context.role_set is not None:
        measures = {
            role.column
            for role in context.role_set.roles
            if role.role is ColumnRoleName.MEASURE and role.provenance in ("inferred", "seeded")
        }
        preferred = [column for column in numeric if column.name in measures]
        if preferred:
            return preferred
    return numeric


def _cross_table_dimensions(context: _DatasetContext, *, exclude: set[str]) -> list[ColumnProfile]:
    """Dimension candidates on the PK side: verified dimensions first."""
    excluded = _excluded_template_columns(context) | exclude
    categorical = [
        column
        for column in context.profile.columns_detail
        if column.semantic_type in {"categorical", "boolean"}
        and 2 <= column.unique_count <= 50
        and column.name not in excluded
    ]
    if context.role_set is not None:
        dimensions = {
            role.column
            for role in context.role_set.roles
            if role.role is ColumnRoleName.DIMENSION and role.provenance in ("inferred", "seeded")
        }
        preferred = [column for column in categorical if column.name in dimensions]
        if preferred:
            return preferred
    return categorical


def _cross_table_sql(
    left: _DatasetContext,
    right: _DatasetContext,
    entry: JoinWhitelistEntry,
    metric_column: str,
    dimension_column: str,
) -> str:
    join_condition = " and ".join(
        f"l.{_quote_identifier(left_col)} = r.{_quote_identifier(right_col)}"
        for left_col, right_col in zip(entry.left_columns, entry.right_columns, strict=True)
    )
    metric_expr = _numeric_expr(f"l.{_quote_identifier(metric_column)}")
    dimension_expr = f"r.{_quote_identifier(dimension_column)}"
    alias = safe_alias(metric_column)
    return f"""
select
    cast({dimension_expr} as varchar) as {_quote_identifier(dimension_column)},
    count(*) as row_count,
    avg({metric_expr}) as avg_{alias},
    sum({metric_expr}) as total_{alias}
from {_quote_identifier(left.relation_name)} as l
join {_quote_identifier(right.relation_name)} as r
    on {join_condition}
where {dimension_expr} is not null and {metric_expr} is not null
group by 1
order by avg_{alias} desc, row_count desc, {_quote_identifier(dimension_column)}
limit 50
""".strip()


# H9-C registered domain-metric family (`domain_metric`).
_MAX_DOMAIN_METRIC_QUESTIONS = 6
# Mid-high deterministic baseline: a registered KPI has explicit business value
# even before any statistical signal is measured.
_DOMAIN_METRIC_SIGNAL = 0.65


def _domain_metric_questions(
    contexts: Sequence[_DatasetContext],
    *,
    column_role_sets: Mapping[str, ColumnRoleSet] | None,
    join_whitelist: JoinWhitelist | None,
    relation_map: Mapping[str, RelationshipCandidate],
    validation_map: Mapping[str, RelationshipValidation],
    semantic_seeds: SemanticSeeds | None,
) -> list[QuestionCandidate]:
    if not column_role_sets:
        # Applicability is role-gated by design; without the DI8-B role layer
        # no metric can bind (and legacy callers see no behavior change).
        return []
    resolution = applicable_metrics(
        role_sets=column_role_sets,
        join_whitelist=join_whitelist,
        profiles=[context.profile for context in contexts],
        semantic_seeds=semantic_seeds,
    )
    by_name = {context.dataset_name: context for context in contexts}
    candidates: list[QuestionCandidate] = []
    for metric in resolution.resolved[:_MAX_DOMAIN_METRIC_QUESTIONS]:
        definition = metric_definition(metric.metric_id)
        metric_contexts = [
            by_name[dataset] for dataset in metric.target_datasets if dataset in by_name
        ]
        if len(metric_contexts) != len(metric.target_datasets):
            continue
        candidate = _candidate(
            question_en=metric.question_en,
            template_id="domain_metric",
            target_datasets=metric.target_datasets,
            referenced_columns=metric.referenced_columns,
            sql_template=metric.sql,
            statistical_signal=_DOMAIN_METRIC_SIGNAL,
            context_artifacts=[context.profile_artifact for context in metric_contexts],
            quality_artifacts=[
                artifact for context in metric_contexts for artifact in context.quality_artifacts
            ],
            relation_map=relation_map,
            validation_map=validation_map,
            required_relations=metric.required_relations,
            confirmed_relations=set(metric.required_relations),
        )
        candidates.append(
            candidate.model_copy(
                update={
                    "candidate_methods": ["domain_metric_pack"],
                    "metric_id": metric.metric_id,
                    "answer_contract": QuestionAnswerContract(
                        kind="metric",
                        metric_id=metric.metric_id,
                        expected_units=(definition.units if definition is not None else {}),
                        abstention_code="metric_contract_failed",
                    ),
                    "produced_units": metric.output_units,
                }
            )
        )
    return candidates


def _candidate(
    *,
    question_en: str,
    template_id: str,
    target_datasets: Sequence[str],
    referenced_columns: Mapping[str, Sequence[str]],
    sql_template: str,
    statistical_signal: float,
    context_artifacts: Sequence[Artifact],
    quality_artifacts: Sequence[Artifact],
    relation_map: Mapping[str, RelationshipCandidate],
    validation_map: Mapping[str, RelationshipValidation],
    required_relations: Sequence[str] = (),
    relationship_candidates: RelationshipCandidateSet | None = None,
    relationship_validations: RelationshipValidationSet | None = None,
    confirmed_relations: Collection[str] = (),
) -> QuestionCandidate:
    profile_artifacts = list(context_artifacts)
    score = score_question(
        profile_artifacts=profile_artifacts,
        quality_artifacts=quality_artifacts,
        target_datasets=target_datasets,
        referenced_columns=referenced_columns,
        statistical_signal=statistical_signal,
        required_relations=required_relations,
        relationship_candidates=relationship_candidates
        or RelationshipCandidateSet(candidates=list(relation_map.values())),
        relationship_validations=relationship_validations
        or RelationshipValidationSet(validations=list(validation_map.values())),
        confirmed_relations=confirmed_relations,
    )
    analysis_mode = "diagnostic" if template_id == "group_difference" else "descriptive"
    method_id = "group_comparison" if template_id == "group_difference" else "descriptive_sql"
    profiles = [
        DatasetProfile.model_validate(artifact.payload)
        for artifact in profile_artifacts
        if artifact.type is ArtifactType.DATASET_PROFILE
    ]
    feasibility = evaluate_feasibility(
        MethodGateContext(
            profiles=profiles,
            target_datasets=list(target_datasets),
            analysis_mode=analysis_mode,
            target_column=None,
        )
    )
    return QuestionCandidate(
        question_id=make_question_id(
            origin="template",
            template_id=template_id,
            question_en=question_en,
            target_datasets=target_datasets,
            required_relations=required_relations,
        ),
        question_en=question_en,
        origin="template",
        template_id=template_id,
        target_datasets=list(target_datasets),
        dataset_display_names={
            dataset: default_dataset_display_name(dataset) for dataset in target_datasets
        },
        required_relations=list(required_relations),
        sql_template=sql_template,
        score=score,
        analysis_mode=analysis_mode,
        candidate_methods=[method_id],
        feasibility=feasibility,
        proposed_action=(
            "collect_data" if feasibility.status in {"needs_data", "unsuitable"} else "run_analysis"
        ),
        data_signal=_template_data_signal(template_id, referenced_columns),
        referenced_columns={
            dataset_id: list(columns) for dataset_id, columns in referenced_columns.items()
        },
        source_artifact_ids=[artifact.id for artifact in profile_artifacts],
    )


def _template_data_signal(
    template_id: str,
    referenced_columns: Mapping[str, Sequence[str]],
) -> str:
    columns = [column for columns in referenced_columns.values() for column in columns]
    joined = ", ".join(columns)
    if template_id == "trend":
        return f"Time-ordered movement observed in {joined}." if columns else ""
    if template_id == "group_difference":
        return f"Group-level differences observed across {joined}." if columns else ""
    if template_id == "correlation_probe":
        return f"A numeric association observed between {joined}." if columns else ""
    if template_id == "quality_missing":
        return f"Missing values observed in {joined}." if columns else ""
    if template_id == "domain_metric":
        return (
            f"A registered domain metric computed deterministically from {joined}."
            if columns
            else ""
        )
    return f"An EDA pattern observed in {joined}." if columns else ""


def _ensure_candidate_feasibility(
    candidate: QuestionCandidate,
    *,
    profiles: list[DatasetProfile],
) -> QuestionCandidate:
    if candidate.feasibility is not None:
        return candidate
    feasibility = evaluate_feasibility(
        MethodGateContext(
            profiles=profiles,
            target_datasets=candidate.target_datasets,
            analysis_mode=candidate.analysis_mode,
            target_column=None,
        )
    )
    update: dict[str, Any] = {
        "feasibility": feasibility,
        "proposed_action": (
            "design_experiment"
            if candidate.analysis_mode == "causal_experiment"
            else "collect_data"
            if feasibility.status in {"needs_data", "unsuitable"}
            else "run_analysis"
        ),
    }
    if not candidate.candidate_methods and feasibility.method_id is not None:
        update["candidate_methods"] = [feasibility.method_id]
    return candidate.model_copy(update=update)


def _contexts(
    datasets: Sequence[LoadedDataset],
    *,
    profile_artifacts: Sequence[Artifact],
    quality_artifacts: Sequence[Artifact],
    analysis_artifacts: Sequence[Artifact],
    column_role_sets: Mapping[str, ColumnRoleSet] | None = None,
) -> list[_DatasetContext]:
    loaded_by_id = {dataset.record.dataset_id: dataset for dataset in datasets}
    qualities = _issues_by_dataset(quality_artifacts)
    quality_artifacts_by_dataset: dict[str, list[Artifact]] = {}
    for artifact in quality_artifacts:
        if artifact.type is not ArtifactType.QUALITY_ISSUE_SET:
            continue
        issue_set = QualityIssueSet.model_validate(artifact.payload)
        quality_artifacts_by_dataset.setdefault(issue_set.dataset_id, []).append(artifact)
    analysis_by_dataset: dict[str, list[AnalysisTable]] = {}
    for artifact in analysis_artifacts:
        if artifact.type is not ArtifactType.TABLE:
            continue
        table = AnalysisTable.model_validate(artifact.payload)
        analysis_by_dataset.setdefault(table.dataset_id, []).append(table)

    contexts: list[_DatasetContext] = []
    for artifact in profile_artifacts:
        if artifact.type is not ArtifactType.DATASET_PROFILE:
            continue
        profile = DatasetProfile.model_validate(artifact.payload)
        loaded = loaded_by_id.get(profile.dataset_id)
        if loaded is None:
            continue
        contexts.append(
            _DatasetContext(
                loaded=loaded,
                profile_artifact=artifact,
                profile=profile,
                quality_artifacts=quality_artifacts_by_dataset.get(profile.dataset_id, []),
                quality_issues=qualities.get(profile.dataset_id, []),
                analysis_tables=analysis_by_dataset.get(profile.dataset_id, []),
                role_set=(column_role_sets or {}).get(profile.name),
            )
        )
    contexts.sort(key=lambda context: context.dataset_id)
    return contexts


def _issues_by_dataset(
    quality_artifacts: Sequence[Artifact],
) -> dict[str, list[QualityIssue]]:
    issues: dict[str, list[QualityIssue]] = {}
    for artifact in quality_artifacts:
        if artifact.type is not ArtifactType.QUALITY_ISSUE_SET:
            continue
        issue_set = QualityIssueSet.model_validate(artifact.payload)
        issues.setdefault(issue_set.dataset_id, []).extend(issue_set.issues)
    return issues


def _relationship_map(
    relationship_candidates: RelationshipCandidateSet | None,
) -> dict[str, RelationshipCandidate]:
    if relationship_candidates is None:
        return {}
    return {candidate.pair.label(): candidate for candidate in relationship_candidates.candidates}


def _validation_map(
    relationship_validations: RelationshipValidationSet | None,
) -> dict[str, RelationshipValidation]:
    if relationship_validations is None:
        return {}
    return {
        validation.pair.label(): validation for validation in relationship_validations.validations
    }


def _columns_by_type(
    columns: Sequence[ColumnProfile],
    semantic_types: set[str],
) -> list[ColumnProfile]:
    return [column for column in columns if column.semantic_type in semantic_types]


def _excluded_template_columns(context: _DatasetContext) -> set[str]:
    excluded = {
        issue.column
        for issue in context.quality_issues
        if issue.column is not None and issue.code in {"constant_column", "empty_column"}
    }
    # Exclude verified identifier and sequence columns from statistical templates.
    if context.role_set is not None:
        excluded |= context.role_set.excluded_from_stats()
    for column in context.profile.columns_detail:
        if column.unique_count <= 3:
            excluded.add(column.name)
            continue
        if column.semantic_type != "numeric":
            continue
        coefficient = _coefficient_of_variation(context.loaded.frame, column.name)
        if coefficient is not None and coefficient < 0.01:
            excluded.add(column.name)
    return excluded


def _eligible_metric_columns(
    context: _DatasetContext, columns: Sequence[ColumnProfile]
) -> list[ColumnProfile]:
    """Keep numeric metric slots aligned with verified semantic roles."""
    if context.role_set is None:
        return list(columns)
    eligible: list[ColumnProfile] = []
    for column in columns:
        role = context.role_set.role_of(column.name)
        if (
            role is not None
            and role.provenance != "unverified"
            and role.role is not ColumnRoleName.MEASURE
        ):
            continue
        eligible.append(column)
    return eligible


def _coefficient_of_variation(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    numeric = cast(pd.Series, pd.to_numeric(frame[column], errors="coerce")).dropna()
    if numeric.empty:
        return None
    std = float(cast(float, numeric.std()) or 0.0)
    mean = abs(float(numeric.mean()))
    return std / max(mean, 1.0)


def _trend_sql(context: _DatasetContext, date_column: str, metric_column: str) -> str:
    date_expr = f"try_cast({_quote_identifier(date_column)} as date)"
    metric_expr = _numeric_expr(_quote_identifier(metric_column))
    return f"""
select
    {date_expr} as period,
    count(*) as row_count,
    avg({metric_expr}) as avg_{safe_alias(metric_column)},
    sum({metric_expr}) as total_{safe_alias(metric_column)}
from {_quote_identifier(context.relation_name)}
where {date_expr} is not null and {metric_expr} is not null
group by 1
order by 1
limit 100
""".strip()


def _group_sql(context: _DatasetContext, category_column: str, metric_column: str) -> str:
    category_expr = _quote_identifier(category_column)
    metric_expr = _numeric_expr(_quote_identifier(metric_column))
    return f"""
select
    cast({category_expr} as varchar) as {_quote_identifier(category_column)},
    count(*) as row_count,
    avg({metric_expr}) as avg_{safe_alias(metric_column)},
    sum({metric_expr}) as total_{safe_alias(metric_column)}
from {_quote_identifier(context.relation_name)}
where {category_expr} is not null and {metric_expr} is not null
group by 1
order by avg_{safe_alias(metric_column)} desc, row_count desc, {_quote_identifier(category_column)}
limit 50
""".strip()


def _correlation_sql(context: _DatasetContext, column_a: str, column_b: str) -> str:
    left = _numeric_expr(_quote_identifier(column_a))
    right = _numeric_expr(_quote_identifier(column_b))
    return f"""
select
    corr({left}, {right}) as pearson,
    count(*) as row_count,
    avg({left}) as avg_{safe_alias(column_a)},
    avg({right}) as avg_{safe_alias(column_b)}
from {_quote_identifier(context.relation_name)}
where {left} is not null and {right} is not null
limit 1
""".strip()


def _missing_sql(context: _DatasetContext, column: str) -> str:
    column_expr = _quote_identifier(column)
    return f"""
select
    count(*) as total_rows,
    sum(case when {column_expr} is null then 1 else 0 end) as missing_rows,
    case
        when count(*) = 0 then 0.0
        else 100.0 * sum(case when {column_expr} is null then 1 else 0 end) / count(*)
    end as missing_percent
from {_quote_identifier(context.relation_name)}
limit 1
""".strip()


def _numeric_expr(qualified_identifier: str) -> str:
    """Build a numeric SQL expression for a profiled numeric column."""
    return f"try_cast({qualified_identifier} as double)"


def _trend_signal(frame: pd.DataFrame, date_column: str, metric_column: str) -> float:
    # format="mixed": user date columns have no single known format; this parses
    # per-element (like the default fallback) but without the noisy UserWarning.
    dates = cast(
        pd.Series, pd.to_datetime(frame[date_column], errors="coerce", format="mixed")
    ).dropna()
    metric = cast(pd.Series, pd.to_numeric(frame[metric_column], errors="coerce")).dropna()
    if dates.empty or metric.empty:
        return 0.0
    span_days = max((dates.max() - dates.min()).days, 0)
    span_score = min(1.0, span_days / 30.0)
    variance_score = _variance_signal(metric)
    return _bounded(min(0.9, 0.4 + 0.3 * span_score + 0.3 * variance_score))


def _group_signal(frame: pd.DataFrame, category_column: str, metric_column: str) -> float:
    working = pd.DataFrame(
        {
            "category": cast(pd.Series, frame[category_column]).astype(str),
            "metric": pd.to_numeric(cast(pd.Series, frame[metric_column]), errors="coerce"),
        }
    ).dropna(subset=["metric"])
    if working.empty:
        return 0.0
    grouped = cast(pd.Series, working.groupby("category")["metric"].mean())
    if len(grouped) < 2:
        return 0.0
    spread = float(grouped.max() - grouped.min())
    scale = max(abs(float(grouped.mean())), 1.0)
    return _bounded(min(0.85, spread / scale))


def _variance_signal(series: pd.Series) -> float:
    numeric = cast(pd.Series, pd.to_numeric(series, errors="coerce")).dropna()
    if numeric.empty:
        return 0.0
    mean = abs(float(numeric.mean()))
    std = float(cast(float, numeric.std()) or 0.0)
    if std == 0:
        return 0.0
    return _bounded(min(1.0, std / max(mean, 1.0)))


def _data_availability(
    profiles: Sequence[DatasetProfile],
    *,
    target_datasets: Sequence[str],
    referenced_columns: Mapping[str, Sequence[str]],
) -> float:
    values: list[float] = []
    target_set = set(target_datasets)
    for profile in profiles:
        # referenced_columns is keyed by dataset NAME (DI7 unification); tolerate
        # the legacy dataset_id key so candidates persisted before DI7 still resolve.
        ref_cols = referenced_columns.get(profile.name)
        if ref_cols is None:
            ref_cols = referenced_columns.get(profile.dataset_id)
        if profile.name not in target_set and ref_cols is None:
            continue
        columns = list(ref_cols or ())
        if not columns:
            columns = profile.column_names
        for column in columns:
            missing = float(profile.missing_percent.get(column, 0.0))
            values.append(max(0.0, min(1.0, 1.0 - missing / 100.0)))
    if not values:
        return 1.0
    return round(sum(values) / len(values), 6)


def _quality_risk(
    issues_by_dataset: Mapping[str, Sequence[QualityIssue]],
    *,
    profiles: Sequence[DatasetProfile],
    target_datasets: Sequence[str],
    referenced_columns: Mapping[str, Sequence[str]],
) -> float:
    del target_datasets
    risk = 0.0
    referenced = {key: set(columns) for key, columns in referenced_columns.items()}
    # ``issues_by_dataset`` is keyed by dataset_id; referenced_columns is keyed by dataset
    # NAME (DI7 unification).
    name_by_id = {profile.dataset_id: profile.name for profile in profiles}
    for dataset_id, issues in issues_by_dataset.items():
        columns = referenced.get(name_by_id.get(dataset_id, ""))
        if columns is None:
            columns = referenced.get(dataset_id)
        if columns is None:
            continue
        for issue in issues:
            if issue.column is not None and issue.column not in columns:
                continue
            risk = max(risk, _issue_risk(issue))
    return round(risk, 6)


def _issue_risk(issue: QualityIssue) -> float:
    if issue.severity == "critical":
        return 1.0
    if issue.code == "high_missing":
        return 0.7
    if issue.code in {"outlier_detected", "mixed_type_string", "high_cardinality_category"}:
        return 0.4
    if issue.severity == "warn":
        return 0.5
    if issue.code == "likely_id_column":
        return 0.2
    return 0.1 if issue.severity == "info" else 0.0


def _join_risk(
    required_relations: Sequence[str],
    relation_map: Mapping[str, RelationshipCandidate],
    validation_map: Mapping[str, RelationshipValidation],
    *,
    confirmed_relations: Collection[str] = (),
) -> float:
    if not required_relations:
        return 0.0
    risk = 0.0
    for label in required_relations:
        relationship = relation_map.get(label)
        if label in confirmed_relations:
            # user-confirmed whitelist join — low base risk regardless of this run's
            # candidate map (the whitelist may predate the run).
            risk = max(risk, 0.1)
        elif relationship is None:
            risk = max(risk, 0.8)
            continue
        elif relationship.confidence == "high" and relationship.auto_adopted:
            risk = max(risk, 0.1)
        elif relationship.confidence == "medium":
            risk = max(risk, 0.75)
        else:
            risk = max(risk, 0.9)
        validation = validation_map.get(label)
        if validation is None:
            continue
        if validation.cardinality == "many_to_many" or validation.join_row_multiplier > 1.5:
            risk = max(risk, 0.9)
        elif validation.join_row_multiplier > 1.05:
            risk = max(risk, 0.55)
        if validation.orphan_rate_left > 0.2 or validation.orphan_rate_right > 0.2:
            risk = max(risk, 0.6)
    return round(risk, 6)


def _score(
    *,
    data_availability: float,
    statistical_signal: float,
    quality_risk: float,
    join_risk: float,
    llm_business_relevance: float | None = None,
    llm_actionability: float | None = None,
) -> QuestionScore:
    availability = _bounded(data_availability)
    signal = _bounded(statistical_signal)
    quality = _bounded(quality_risk)
    join = _bounded(join_risk)
    deterministic = _bounded(
        0.35 * availability + 0.45 * signal + 0.10 * (1.0 - quality) + 0.10 * (1.0 - join)
    )
    return QuestionScore(
        data_availability=availability,
        statistical_signal=signal,
        quality_risk=quality,
        join_risk=join,
        deterministic_score=deterministic,
        llm_business_relevance=(
            None if llm_business_relevance is None else _bounded(llm_business_relevance)
        ),
        llm_actionability=None if llm_actionability is None else _bounded(llm_actionability),
    )


def _bounded(value: float) -> float:
    return round(max(0.0, min(1.0, float(value))), 6)


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
