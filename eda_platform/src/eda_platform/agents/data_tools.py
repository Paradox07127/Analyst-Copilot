"""Typed, workspace-scoped tools for autonomous data-analysis agents."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from eda_platform.agents.receipts import build_receipt
from eda_platform.agents.runtime import AgentTool, AgentToolResult
from eda_platform.core.column_roles import ColumnRoleSet
from eda_platform.core.ids import make_artifact_id, stable_hash
from eda_platform.core.permissions import PermissionTier, require_permission
from eda_platform.core.query import DuckDBQueryEngine
from eda_platform.core.skills_store import load_skills
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import (
    Artifact,
    ArtifactType,
    DatasetProfile,
    QualityIssueSet,
)
from eda_platform.schemas.receipts import (
    EvidenceReceipt,
    ReceiptFact,
    ReceiptMethod,
    ReceiptScope,
    ReceiptStatistics,
)
from eda_platform.schemas.relations import (
    RelationshipCandidate,
    RelationshipColumnPair,
    RelationshipSignals,
)
from eda_platform.schemas.stats import StatTestType
from eda_platform.tools.analysis import correlate_column_pairs
from eda_platform.tools.anomaly import create_anomaly_artifact
from eda_platform.tools.anomaly import screen_anomalies as screen_anomaly_column
from eda_platform.tools.cleaning_advice import recommended_cleaning_operations
from eda_platform.tools.domain_metrics import (
    MetricContractResult,
    applicable_metrics,
    validate_metric_result,
)
from eda_platform.tools.evidence import PayloadPolicy
from eda_platform.tools.loader import LoadedDataset
from eda_platform.tools.relationship_discovery import (
    _relation_name as metric_relation_name,
)
from eda_platform.tools.relationship_discovery import (
    discover_relationship_candidates,
    validate_relationships,
)
from eda_platform.tools.slice_profile import compute_slice_profile
from eda_platform.tools.sql_runner import SqlCatalog, rewrite_relation_names, run_sql
from eda_platform.tools.stat_tests import create_stat_test_artifact
from eda_platform.tools.stat_tests import run_stat_test as run_stat_test_frame
from eda_platform.tools.time_series import analyze_series


class _NoArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReadArtifactArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str = Field(min_length=1)


class RunSqlArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sql: str = Field(min_length=1, max_length=20_000)
    purpose: str = Field(min_length=1, max_length=500)


class RunSavedSkillArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill_id: str = Field(min_length=1)
    target_dataset_ids: list[str] = Field(min_length=1, max_length=8)


class OpenAnalysisArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str = Field(min_length=1, max_length=2_000)


class AssessJoinKeysArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    left_dataset_id: str = Field(min_length=1)
    right_dataset_id: str = Field(min_length=1)
    left_columns: list[str] = Field(min_length=1, max_length=4)
    right_columns: list[str] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def _sides_align(self) -> AssessJoinKeysArguments:
        if len(self.left_columns) != len(self.right_columns):
            raise ValueError("left_columns and right_columns must have the same length.")
        return self


class ScreenAnomaliesArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(min_length=1)
    column: str = Field(min_length=1)
    method: Literal["robust_zscore", "iqr"] = "robust_zscore"
    threshold: float | None = Field(default=None, gt=0)


class RunDomainMetricsArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RecommendCleaningArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(min_length=1)


# fisher_exact is result-only (the automatic chi-square fallback), never requested.
AgentStatTestType = Literal[
    "independent_t_test",
    "paired_t_test",
    "chi_square_independence",
    "one_way_anova",
    "welch_anova",
    "mann_whitney_u",
    "kruskal_wallis",
]


class RunStatTestArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(min_length=1)
    test_type: AgentStatTestType
    # Multiple-comparison ledger key; recorded in the receipt for later
    # family-wise auditing, not yet enforced.
    test_family_id: str = Field(min_length=1, max_length=200)
    group_column: str | None = None
    value_column: str | None = None
    category_column: str | None = None
    pair_column: str | None = None
    comparison_count: int = Field(default=1, ge=1, le=1_000)


class CorrelateColumnsArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(min_length=1)
    columns: list[str] | None = Field(default=None, min_length=2, max_length=24)
    correction_method: Literal["holm", "fdr_bh"] = "fdr_bh"


class ProfileSliceArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(min_length=1)
    # WHERE clause body only, never a full statement; the server composes and
    # validates the SELECT around it.
    where_sql: str | None = Field(default=None, min_length=1, max_length=4_000)
    columns: list[str] | None = Field(default=None, min_length=1, max_length=40)


class AnalyzeTimeSeriesArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(min_length=1)
    time_column: str = Field(min_length=1)
    value_column: str = Field(min_length=1)
    freq: str | None = Field(default=None, min_length=1, max_length=16)
    period: int | None = Field(default=None, ge=2, le=1_000)
    agg: Literal["sum", "mean", "count"] = "sum"


OpenAnalysisExecutor = Callable[[OpenAnalysisArguments], AgentToolResult]


@dataclass(slots=True)
class DataToolContext:
    """All values are session-local; tools never accept a host path from a model."""

    datasets: Sequence[LoadedDataset]
    catalog: SqlCatalog
    project_id: str
    session_id: str
    store: ArtifactStore | None
    payload_policy: PayloadPolicy
    artifacts: list[Artifact] = field(default_factory=list)
    open_analysis: OpenAnalysisExecutor | None = None
    _artifacts_by_id: dict[str, Artifact] = field(init=False, repr=False)
    _datasets_by_id: dict[str, LoadedDataset] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._artifacts_by_id = {artifact.id: artifact for artifact in self.artifacts}
        self._datasets_by_id = {dataset.record.dataset_id: dataset for dataset in self.datasets}

    def add_artifact(self, artifact: Artifact, *, persist: bool = True) -> None:
        self._artifacts_by_id[artifact.id] = artifact
        if not any(existing.id == artifact.id for existing in self.artifacts):
            self.artifacts.append(artifact)
        if persist and self.store is not None:
            self.store.save_artifact(artifact)

    def artifact(self, artifact_id: str) -> Artifact:
        artifact = self._artifacts_by_id.get(artifact_id)
        if artifact is None:
            raise ValueError("Artifact is not available in this analysis session.")
        return artifact

    def datasets_for(self, dataset_ids: list[str]) -> list[LoadedDataset]:
        selected: list[LoadedDataset] = []
        seen: set[str] = set()
        for dataset_id in dataset_ids:
            dataset = self._datasets_by_id.get(dataset_id)
            if dataset is None:
                raise ValueError("A selected dataset is not part of this analysis session.")
            if dataset_id in seen:
                raise ValueError("A saved skill target may be selected only once.")
            seen.add(dataset_id)
            selected.append(dataset)
        return selected


def build_data_tools(context: DataToolContext) -> list[AgentTool]:
    """Return the narrow, local-only capability set for one agent task."""
    tools = [
        AgentTool(
            name="inspect_data_catalog",
            description=(
                "Inspect the loaded datasets, their safe relation names, row counts and columns. "
                "Use this before writing SQL or choosing a saved skill."
            ),
            args_schema=_NoArguments,
            execute=lambda _args: _inspect_catalog(context),
        ),
        AgentTool(
            name="list_artifacts",
            description=(
                "List existing evidence artifacts from this session. Use it to find profiles, "
                "quality checks, findings and earlier query results before making a claim."
            ),
            args_schema=_NoArguments,
            execute=lambda _args: _list_artifacts(context),
        ),
        AgentTool(
            name="read_artifact",
            description=(
                "Read one listed artifact by id. The payload is constrained by this session's "
                "data-disclosure policy."
            ),
            args_schema=ReadArtifactArguments,
            execute=lambda args: _read_artifact(context, cast(ReadArtifactArguments, args)),
        ),
        AgentTool(
            name="run_sql",
            description=(
                "Run one read-only DuckDB SELECT/WITH query over the loaded relations. "
                "Never use file readers, mutation statements, network functions or unlisted tables."
            ),
            args_schema=RunSqlArguments,
            execute=lambda args: _run_sql(context, cast(RunSqlArguments, args)),
        ),
        AgentTool(
            name="list_saved_skills",
            description=(
                "List validated, project-local analysis skills that can be replayed against "
                "compatible currently loaded datasets."
            ),
            args_schema=_NoArguments,
            execute=lambda _args: _list_saved_skills(context),
        ),
        AgentTool(
            name="run_saved_skill",
            description=(
                "Replay one listed saved skill on selected loaded dataset ids. This executes "
                "only the skill's frozen read-only plan through the same local safety gates."
            ),
            args_schema=RunSavedSkillArguments,
            execute=lambda args: _run_saved_skill(context, cast(RunSavedSkillArguments, args)),
        ),
        AgentTool(
            name="assess_join_keys",
            description=(
                "Measure whether two loaded datasets can be joined on the given columns: "
                "bidirectional containment, key uniqueness, join row multiplier, orphan "
                "rates and cardinality, with an evidence receipt. Use it before proposing "
                "any join; do not use it for same-table column comparisons."
            ),
            args_schema=AssessJoinKeysArguments,
            execute=lambda args: _assess_join_keys(context, cast(AssessJoinKeysArguments, args)),
        ),
        AgentTool(
            name="screen_anomalies",
            description=(
                "Deterministically screen one numeric column for robust outliers "
                "(robust z-score or IQR) and record an evidence receipt. Use it to "
                "quantify suspected outliers; do not use it on ids, codes or categories."
            ),
            args_schema=ScreenAnomaliesArguments,
            execute=lambda args: _screen_anomalies(context, cast(ScreenAnomaliesArguments, args)),
        ),
        AgentTool(
            name="run_domain_metrics",
            description=(
                "Compute every registered domain metric (GMV, AOV, time coverage, ...) that "
                "deterministically applies to the loaded datasets, validating each result "
                "contract. Use it for standard business metrics after profiling; do not use "
                "it for ad-hoc aggregates - write SQL for those."
            ),
            args_schema=RunDomainMetricsArguments,
            execute=lambda args: _run_domain_metrics(
                context, cast(RunDomainMetricsArguments, args)
            ),
        ),
        AgentTool(
            name="recommend_cleaning",
            description=(
                "Propose cleaning operations derived from this dataset's recorded quality "
                "issues. Proposals only - nothing is ever applied or mutated. Use it when "
                "asked how to clean a dataset; do not use it to execute a cleaning step."
            ),
            args_schema=RecommendCleaningArguments,
            execute=lambda args: _recommend_cleaning(
                context, cast(RecommendCleaningArguments, args)
            ),
        ),
        AgentTool(
            name="run_stat_test",
            description=(
                "Run one guarded statistical test with assumption checks, automatic robust "
                "fallbacks (Welch ANOVA, Fisher exact) and a BCa bootstrap effect-size CI, "
                "recorded in an evidence receipt. Use it for group comparisons and "
                "independence questions; do not use it for correlation screens."
            ),
            args_schema=RunStatTestArguments,
            execute=lambda args: _run_stat_test(context, cast(RunStatTestArguments, args)),
        ),
        AgentTool(
            name="correlate_columns",
            description=(
                "Screen numeric column pairs with Pearson correlation plus "
                "multiplicity-adjusted p-values (holm or fdr_bh across every tested pair). "
                "Use it to find related measures; do not use it to claim causation or to "
                "compare categorical columns."
            ),
            args_schema=CorrelateColumnsArguments,
            execute=lambda args: _correlate_columns(
                context, cast(CorrelateColumnsArguments, args)
            ),
        ),
        AgentTool(
            name="profile_slice",
            description=(
                "Re-profile a WHERE-filtered subgroup of one dataset: per-column "
                "missing/unique/distribution plus numeric summaries, with the "
                "slice's share of the full table. `where_sql` is a bare WHERE "
                "condition (no SELECT, no semicolons). Use it for conditional or "
                "stratified exploration; do not use it to fetch raw rows."
            ),
            args_schema=ProfileSliceArguments,
            execute=lambda args: _profile_slice(context, cast(ProfileSliceArguments, args)),
        ),
        AgentTool(
            name="analyze_time_series",
            description=(
                "Aggregate one time column into a regular series and report trend "
                "direction, seasonal strength, Ljung-Box autocorrelation and a "
                "joint ADF+KPSS stationarity verdict, with gap accounting. Use it "
                "for trend/seasonality questions; do not use it to forecast."
            ),
            args_schema=AnalyzeTimeSeriesArguments,
            execute=lambda args: _analyze_time_series(
                context, cast(AnalyzeTimeSeriesArguments, args)
            ),
        ),
    ]
    open_analysis = context.open_analysis
    if open_analysis is not None:
        tools.append(
            AgentTool(
                name="run_open_analysis",
                description=(
                    "Ask the secured Python analysis tool to perform a custom analysis when "
                    "SQL is insufficient. It receives only mounted local data and runs inside "
                    "the configured sandbox; never request host, network or filesystem access."
                ),
                args_schema=OpenAnalysisArguments,
                execute=lambda args: _run_open_analysis(
                    context,
                    open_analysis,
                    cast(OpenAnalysisArguments, args),
                ),
            )
        )
    return tools


def _run_open_analysis(
    context: DataToolContext,
    execute: OpenAnalysisExecutor,
    args: OpenAnalysisArguments,
) -> AgentToolResult:
    result = execute(args)
    for artifact in result.artifacts:
        # The secured code executor persists its own artifacts. Register them
        # in the live context so a later agent step can inspect and cite them.
        context.add_artifact(artifact, persist=False)
    return result


def _inspect_catalog(context: DataToolContext) -> AgentToolResult:
    rows: list[dict[str, Any]] = []
    for dataset in context.datasets:
        relation = context.catalog.relations[dataset.record.name]
        rows.append(
            {
                "dataset_id": dataset.record.dataset_id,
                "name": dataset.record.name,
                "relation": relation,
                "rows": int(len(dataset.frame)),
                "columns": [str(column) for column in dataset.frame.columns],
            }
        )
    profiles = [
        artifact
        for artifact in context.artifacts
        if artifact.type is ArtifactType.DATASET_PROFILE
    ]
    return AgentToolResult(
        content={
            "datasets": rows,
            "profile_artifact_ids": [artifact.id for artifact in profiles],
        },
        # Catalog claims can now be persisted as evidence even when the answer
        # needs no query beyond the already-produced dataset profiles.
        artifacts=profiles,
    )


def _list_artifacts(context: DataToolContext) -> AgentToolResult:
    return AgentToolResult(
        content={
            "artifacts": [
                {
                    "artifact_id": artifact.id,
                    "type": artifact.type.value,
                    "warnings": artifact.warnings[:5],
                    "evidence_count": len(artifact.evidence),
                }
                for artifact in context.artifacts
            ]
        }
    )


def _read_artifact(context: DataToolContext, args: ReadArtifactArguments) -> AgentToolResult:
    artifact = context.artifact(args.artifact_id)
    payload: dict[str, Any] | None
    if context.payload_policy == "schema_only":
        payload = None
    elif artifact.type is ArtifactType.RAW_DATA_PREVIEW:
        # Raw rows are intentionally never made available just because an agent
        # asked to inspect an artifact. Aggregates/profiles remain available.
        payload = {"notice": "Raw data preview content is withheld by the agent tool policy."}
    else:
        payload = _clip_json(artifact.payload)
    return AgentToolResult(
        content={
            "artifact_id": artifact.id,
            "type": artifact.type.value,
            "payload": payload,
            "warnings": artifact.warnings[:10],
            "evidence": [item.model_dump(mode="json") for item in artifact.evidence[:30]],
        },
        artifacts=[artifact],
    )


def _run_sql(context: DataToolContext, args: RunSqlArguments) -> AgentToolResult:
    decision = require_permission({"type": "duckdb_select", "sql": args.sql})
    if decision.tier is PermissionTier.DENY:
        raise ValueError(decision.feedback)
    artifact = run_sql(
        context.catalog,
        args.sql,
        project_id=context.project_id,
        session_id=context.session_id,
    )
    context.add_artifact(artifact)
    payload = _clip_json(artifact.payload)
    return AgentToolResult(
        content={
            "purpose": args.purpose,
            "artifact_id": artifact.id,
            "result": payload,
        },
        artifacts=[artifact],
    )


def _list_saved_skills(context: DataToolContext) -> AgentToolResult:
    if context.store is None:
        return AgentToolResult(content={"skills": []})
    skills = load_skills(context.store.project_dir(context.project_id))
    return AgentToolResult(
        content={
            "skills": [
                {
                    "skill_id": skill.skill_id,
                    "name": skill.name,
                    "description": skill.description,
                    "question": skill.plan.question,
                    "expected_datasets": skill.expected_datasets,
                    "required_columns": skill.param_columns,
                    # Provenance tiers differ in trust: builtin seeds are
                    # platform-verified, frozen plans carry in-run evidence,
                    # user templates only passed syntax and a trial run.
                    "origin": skill.origin,
                    "when_to_use": skill.when_to_use,
                    "when_not_to_use": skill.when_not_to_use,
                }
                for skill in skills
            ]
        }
    )


def _run_saved_skill(
    context: DataToolContext,
    args: RunSavedSkillArguments,
) -> AgentToolResult:
    if context.store is None:
        raise ValueError("Saved skills are unavailable because this chat has no workspace store.")
    skill = next(
        (
            item
            for item in load_skills(context.store.project_dir(context.project_id))
            if item.skill_id == args.skill_id
        ),
        None,
    )
    if skill is None:
        raise ValueError("Saved skill was not found in this project.")
    # `analysis_skill` re-enters the chat driver for its frozen execution path,
    # so importing it only when the tool is actually invoked keeps the chat
    # driver's module graph acyclic.
    from eda_platform.drivers.analysis_skill import replay_skill

    result = replay_skill(
        skill,
        context.datasets_for(args.target_dataset_ids),
        store=context.store,
        project_id=context.project_id,
        session_id=context.session_id,
    )
    for artifact in result.artifacts:
        # replay_skill persists through the approved-plan execution path.
        context.add_artifact(artifact, persist=False)
    return AgentToolResult(
        content={
            "skill_id": skill.skill_id,
            "skill_name": skill.name,
            "status": result.status,
            "message": result.message,
            "sql": result.sql,
            "artifact_ids": [artifact.id for artifact in result.artifacts],
        },
        artifacts=list(result.artifacts),
    )


_ANALYSIS_TOOL_VERSION = "1"
_MAX_FACT_PROPOSALS = 20
_MAX_FACT_PAIRS = 5
_MAX_SKIP_WARNINGS = 10


def _emit_receipt(
    context: DataToolContext,
    *,
    tool_name: str,
    arguments: dict[str, Any],
    raw_output: Any,
    artifact_ids: tuple[str, ...],
    parent_ids: list[str] | None = None,
    result_count: int,
    scope: ReceiptScope,
    facts: tuple[ReceiptFact, ...],
    method: ReceiptMethod,
    statistics: ReceiptStatistics | None = None,
) -> EvidenceReceipt:
    """Build, register and persist the EvidenceReceipt for one tool call."""
    receipt = build_receipt(
        tool_call_id=f"{tool_name}:{stable_hash(arguments)}",
        tool_name=tool_name,
        tool_version=_ANALYSIS_TOOL_VERSION,
        arguments=arguments,
        raw_output=raw_output,
        artifact_ids=artifact_ids,
        result_count=result_count,
        scope=scope,
        facts=facts,
        method=method,
        statistics=statistics,
        data_state_witness=stable_hash(
            sorted(dataset.record.dataset_id for dataset in context.datasets)
        ),
        created_at=datetime.now(UTC).isoformat(),
    )
    payload = receipt.model_dump(mode="json")
    artifact = Artifact(
        id=make_artifact_id("receipt", payload),
        type=ArtifactType.EVIDENCE_RECEIPT,
        project_id=context.project_id,
        session_id=context.session_id,
        parents=list(artifact_ids) if parent_ids is None else parent_ids,
        payload=payload,
    )
    context.add_artifact(artifact, persist=True)
    return receipt


def _fact(
    fact_id: str,
    value: float | int | str | bool | None,
    value_type: str,
    *,
    unit: str | None = None,
) -> ReceiptFact:
    return ReceiptFact(
        fact_id=fact_id,
        name=fact_id,
        value=value,
        value_type=cast(Any, value_type),
        unit=unit,
    )


def _absence_fact(fact_id: str) -> ReceiptFact:
    return ReceiptFact(
        fact_id=fact_id,
        name=fact_id,
        value=None,
        value_type="null",
        support_type="absence",
    )


def _single_dataset(context: DataToolContext, dataset_id: str) -> LoadedDataset:
    return context.datasets_for([dataset_id])[0]


def _assess_join_keys(context: DataToolContext, args: AssessJoinKeysArguments) -> AgentToolResult:
    left = _single_dataset(context, args.left_dataset_id)
    right = _single_dataset(context, args.right_dataset_id)
    for dataset, columns in ((left, args.left_columns), (right, args.right_columns)):
        for column in columns:
            if column not in dataset.frame.columns:
                raise ValueError(
                    f"Column `{column}` is not in dataset `{dataset.record.name}`."
                )
    engine = DuckDBQueryEngine()
    datasets = [left] if right is left else [left, right]
    candidates = discover_relationship_candidates(datasets, engine)
    match = next(
        (
            candidate
            for candidate in candidates.candidates
            if candidate.pair.left_dataset_id == args.left_dataset_id
            and list(candidate.pair.left_columns) == args.left_columns
            and candidate.pair.right_dataset_id == args.right_dataset_id
            and list(candidate.pair.right_columns) == args.right_columns
        ),
        None,
    )
    method_warnings: list[str] = []
    if match is not None:
        candidate = (
            match
            if match.confidence in {"medium", "high"}
            else match.model_copy(update={"confidence": "medium"})
        )
        signals = match.signals
    else:
        method_warnings.append(
            "Discovery prefiltered this pair as structurally implausible; overlap "
            "signals are unavailable, so only SQL validation facts are reported."
        )
        signals = None
        candidate = RelationshipCandidate(
            pair=RelationshipColumnPair(
                left_dataset_id=args.left_dataset_id,
                left_dataset_name=left.record.name,
                left_columns=args.left_columns,
                right_dataset_id=args.right_dataset_id,
                right_dataset_name=right.record.name,
                right_columns=args.right_columns,
            ),
            signals=RelationshipSignals(
                name_similarity=0.0,
                type_compatible=True,
                overlap_left_in_right=0.0,
                overlap_right_in_left=0.0,
                right_unique_rate=0.0,
                left_null_rate=0.0,
                right_null_rate=0.0,
                format_fingerprint_match=False,
                sampled=False,
            ),
            ensemble_score=0.0,
            confidence="medium",
            auto_adopted=False,
        )
    validation_set = validate_relationships([candidate], engine)
    validation = validation_set.validations[0]
    payload = validation_set.model_dump(mode="json")
    primary = Artifact(
        id=make_artifact_id("relval", payload),
        type=ArtifactType.RELATIONSHIP_VALIDATION_SET,
        project_id=context.project_id,
        session_id=context.session_id,
        payload=payload,
        warnings=list(validation.warnings),
        plain_language=(
            f"Join check {validation.pair.label()}: multiplier "
            f"{validation.join_row_multiplier:.3f}, cardinality {validation.cardinality}."
        ),
    )
    context.add_artifact(primary)

    facts: list[ReceiptFact] = []
    if signals is not None:
        facts.extend(
            [
                _fact("containment_left_in_right", signals.overlap_left_in_right, "number"),
                _fact("containment_right_in_left", signals.overlap_right_in_left, "number"),
                _fact("right_unique_rate", signals.right_unique_rate, "number"),
            ]
        )
    facts.extend(
        [
            _fact("join_row_multiplier", validation.join_row_multiplier, "number"),
            _fact("orphan_rate_left", validation.orphan_rate_left, "number"),
            _fact("orphan_rate_right", validation.orphan_rate_right, "number"),
            _fact("cardinality", validation.cardinality, "string"),
        ]
    )
    receipt = _emit_receipt(
        context,
        tool_name="assess_join_keys",
        arguments=args.model_dump(mode="json"),
        raw_output=payload,
        artifact_ids=(primary.id,),
        result_count=1,
        scope=ReceiptScope(
            dataset_ids=(args.left_dataset_id, args.right_dataset_id),
            columns=tuple([*args.left_columns, *args.right_columns]),
        ),
        facts=tuple(facts),
        method=ReceiptMethod(
            family="join_key_validation",
            parameters={
                "left_dataset_id": args.left_dataset_id,
                "right_dataset_id": args.right_dataset_id,
                "left_columns": ",".join(args.left_columns),
                "right_columns": ",".join(args.right_columns),
                "signals_available": signals is not None,
            },
            warnings=tuple([*method_warnings, *validation.warnings]),
        ),
    )
    return AgentToolResult(
        content={
            "artifact_id": primary.id,
            "receipt_id": receipt.receipt_id,
            "facts": {fact.fact_id: fact.value for fact in facts},
            "warnings": [*method_warnings, *validation.warnings],
        },
        artifacts=[primary],
    )


def _screen_anomalies(context: DataToolContext, args: ScreenAnomaliesArguments) -> AgentToolResult:
    dataset = _single_dataset(context, args.dataset_id)
    result = screen_anomaly_column(
        dataset.frame,
        dataset_name=dataset.record.name,
        column=args.column,
        method=args.method,
        threshold=args.threshold,
    )
    primary = create_anomaly_artifact(
        result,
        project_id=context.project_id,
        session_id=context.session_id,
    )
    context.add_artifact(primary)
    facts = (
        _fact("outlier_count", result.outlier_count, "count"),
        _fact("outlier_percent", round(result.outlier_percent, 6), "percent", unit="percent"),
        _fact("median", result.median, "number"),
        _fact("mad", result.mad, "number"),
        _fact("q1", result.q1, "number"),
        _fact("q3", result.q3, "number"),
    )
    receipt = _emit_receipt(
        context,
        tool_name="screen_anomalies",
        arguments=args.model_dump(mode="json"),
        raw_output=result.model_dump(mode="json"),
        artifact_ids=(primary.id,),
        result_count=result.outlier_count,
        scope=ReceiptScope(dataset_ids=(args.dataset_id,), columns=(args.column,)),
        facts=facts,
        method=ReceiptMethod(
            # result.method reflects the deterministic fallback (e.g. iqr when
            # the MAD collapses), not merely what was requested.
            family=result.method,
            parameters={
                "requested_method": args.method,
                "threshold": result.threshold,
            },
            warnings=tuple(result.notes),
        ),
    )
    return AgentToolResult(
        content={
            "artifact_id": primary.id,
            "receipt_id": receipt.receipt_id,
            "facts": {fact.fact_id: fact.value for fact in facts},
            "method": result.method,
            "notes": result.notes,
        },
        artifacts=[primary],
    )


def _run_domain_metrics(
    context: DataToolContext,
    args: RunDomainMetricsArguments,
) -> AgentToolResult:
    loaded_ids = {dataset.record.dataset_id for dataset in context.datasets}
    profiles: dict[str, DatasetProfile] = {}
    role_sets: dict[str, ColumnRoleSet] = {}
    for artifact in context.artifacts:
        if artifact.type is ArtifactType.DATASET_PROFILE:
            profile = DatasetProfile.model_validate(artifact.payload)
            if profile.dataset_id in loaded_ids:
                profiles[profile.dataset_id] = profile
        elif artifact.type is ArtifactType.COLUMN_ROLE_SET:
            role_set = ColumnRoleSet.model_validate(artifact.payload)
            role_sets[role_set.dataset] = role_set
    if not profiles:
        raise ValueError(
            "run_domain_metrics needs dataset profile artifacts for the loaded "
            "datasets; none are available in this session."
        )
    resolution = applicable_metrics(
        role_sets=role_sets,
        join_whitelist=None,
        profiles=list(profiles.values()),
    )
    skip_warnings = tuple(
        f"{skip.metric_id}: {skip.reason}" for skip in resolution.skipped
    )[:_MAX_SKIP_WARNINGS]
    scope = ReceiptScope(dataset_ids=tuple(sorted(loaded_ids)))
    arguments = args.model_dump(mode="json")

    if not resolution.resolved:
        receipt = _emit_receipt(
            context,
            tool_name="run_domain_metrics",
            arguments=arguments,
            raw_output={"metrics": [], "skipped": skip_warnings},
            artifact_ids=(),
            result_count=0,
            scope=scope,
            facts=(_absence_fact("no_applicable_metrics"),),
            method=ReceiptMethod(
                family="domain_metric_registry",
                parameters={"metric_count": 0},
                warnings=skip_warnings,
            ),
        )
        return AgentToolResult(
            content={
                "receipt_id": receipt.receipt_id,
                "metrics": [],
                "skipped": list(skip_warnings),
            }
        )

    relation_map = {
        metric_relation_name(dataset_id): context.catalog.relations[dataset_id]
        for dataset_id in loaded_ids
        if dataset_id in context.catalog.relations
    }
    facts: list[ReceiptFact] = []
    metric_summaries: list[dict[str, Any]] = []
    sql_artifacts: list[Artifact] = []
    for metric in resolution.resolved:
        sql = rewrite_relation_names(metric.sql, relation_map)
        artifact = run_sql(
            context.catalog,
            sql,
            project_id=context.project_id,
            session_id=context.session_id,
        )
        context.add_artifact(artifact)
        sql_artifacts.append(artifact)
        rows = artifact.payload.get("rows_preview") or []
        row: dict[str, Any] = rows[0] if rows else {}
        if row:
            contract = validate_metric_result(metric.metric_id, row)
        else:
            contract = MetricContractResult(
                valid=False,
                code="empty_metric_result",
                reason="The metric query returned no rows.",
            )
        for field_name in metric.output_units:
            if field_name not in row:
                continue
            value = row[field_name]
            if isinstance(value, bool) or value is None:
                continue
            if isinstance(value, (int, float)):
                facts.append(
                    _fact(
                        f"{metric.metric_id}.{field_name}",
                        value,
                        "number",
                        unit=metric.output_units.get(field_name),
                    )
                )
            else:
                facts.append(_fact(f"{metric.metric_id}.{field_name}", str(value), "string"))
        facts.append(_fact(f"{metric.metric_id}.contract_valid", contract.valid, "bool"))
        metric_summaries.append(
            {
                "metric_id": metric.metric_id,
                "name": metric.name_en,
                "artifact_id": artifact.id,
                "result": {key: str(value) for key, value in row.items()},
                "output_units": metric.output_units,
                "contract_valid": contract.valid,
                "contract_code": contract.code,
                "contract_reason": contract.reason,
            }
        )
    receipt = _emit_receipt(
        context,
        tool_name="run_domain_metrics",
        arguments=arguments,
        raw_output={"metrics": metric_summaries, "skipped": skip_warnings},
        artifact_ids=tuple(artifact.id for artifact in sql_artifacts),
        result_count=len(resolution.resolved),
        scope=scope,
        facts=tuple(facts),
        method=ReceiptMethod(
            family="domain_metric_registry",
            parameters={"metric_count": len(resolution.resolved)},
            warnings=skip_warnings,
        ),
    )
    return AgentToolResult(
        content=cast(
            dict[str, Any],
            _clip_json(
                {
                    "receipt_id": receipt.receipt_id,
                    "metrics": metric_summaries,
                    "skipped": list(skip_warnings),
                }
            ),
        ),
        artifacts=list(sql_artifacts),
    )


def _recommend_cleaning(
    context: DataToolContext,
    args: RecommendCleaningArguments,
) -> AgentToolResult:
    _single_dataset(context, args.dataset_id)
    profile_artifact: Artifact | None = None
    quality_artifact: Artifact | None = None
    for artifact in context.artifacts:
        if (
            artifact.type is ArtifactType.DATASET_PROFILE
            and artifact.payload.get("dataset_id") == args.dataset_id
        ):
            profile_artifact = artifact
        elif (
            artifact.type is ArtifactType.QUALITY_ISSUE_SET
            and artifact.payload.get("dataset_id") == args.dataset_id
        ):
            quality_artifact = artifact
    if profile_artifact is None:
        raise ValueError("recommend_cleaning needs this dataset's profile artifact first.")
    if quality_artifact is None:
        raise ValueError("recommend_cleaning needs this dataset's quality issue artifact first.")
    profile = DatasetProfile.model_validate(profile_artifact.payload)
    issue_set = QualityIssueSet.model_validate(quality_artifact.payload)
    proposals = recommended_cleaning_operations(profile, issue_set)

    facts: list[ReceiptFact] = []
    if proposals:
        facts.append(_fact("proposal_count", len(proposals), "count"))
        for index, proposal in enumerate(proposals[:_MAX_FACT_PROPOSALS]):
            column = proposal["column"]
            facts.extend(
                [
                    _fact(f"op{index}.operation", str(proposal["operation"]), "string"),
                    _fact(
                        f"op{index}.column",
                        str(column) if column is not None else None,
                        "string" if column is not None else "null",
                    ),
                    _fact(f"op{index}.severity", str(proposal["severity"]), "string"),
                    _fact(f"op{index}.lossy", bool(proposal["lossy"]), "bool"),
                ]
            )
    else:
        facts.append(_absence_fact("no_recommended_operations"))
    receipt = _emit_receipt(
        context,
        tool_name="recommend_cleaning",
        arguments=args.model_dump(mode="json"),
        raw_output={"proposals": proposals},
        # Proposals are not artifacts; the receipt's parents point at the
        # evidence the proposals were derived from.
        artifact_ids=(),
        parent_ids=[profile_artifact.id, quality_artifact.id],
        result_count=len(proposals),
        scope=ReceiptScope(dataset_ids=(args.dataset_id,)),
        facts=tuple(facts),
        method=ReceiptMethod(
            family="cleaning_recommendation",
            parameters={"proposal_count": len(proposals)},
            warnings=("Proposals only; no transform was applied.",),
        ),
    )
    return AgentToolResult(
        content=cast(
            dict[str, Any],
            _clip_json(
                {
                    "receipt_id": receipt.receipt_id,
                    "proposals": proposals,
                    "note": "Proposals only; nothing was applied to the data.",
                }
            ),
        )
    )


def _run_stat_test(context: DataToolContext, args: RunStatTestArguments) -> AgentToolResult:
    dataset = _single_dataset(context, args.dataset_id)
    switch_notes: list[str] = []

    def _run(test_type: StatTestType) -> Any:
        return run_stat_test_frame(
            dataset.frame,
            dataset_id=args.dataset_id,
            test_type=test_type,
            group_column=args.group_column,
            value_column=args.value_column,
            category_column=args.category_column,
            pair_column=args.pair_column,
            comparison_count=args.comparison_count,
            effect_ci=True,
        )

    result = _run(args.test_type)
    if args.test_type == "one_way_anova" and any(
        check.name == "variance_homogeneity" and check.status == "warn"
        for check in result.assumptions
    ):
        # Precondition gate: heterogeneous variances invalidate classic ANOVA,
        # so the robust alternative runs instead of a warning being waved through.
        result = _run("welch_anova")
        switch_notes.append(
            "Levene variance-homogeneity check failed for one_way_anova; "
            "automatically switched to Welch ANOVA."
        )
    primary = create_stat_test_artifact(
        result,
        project_id=context.project_id,
        session_id=context.session_id,
    )
    context.add_artifact(primary)
    facts = tuple(
        fact
        for fact in (
            _fact("p_value", result.p_value, "number"),
            _fact("statistic", result.statistic, "number"),
            (
                _fact("effect_size", result.effect_size, "number")
                if result.effect_size is not None
                else None
            ),
            _fact("sample_size", result.sample_size, "count"),
        )
        if fact is not None
    )
    receipt = _emit_receipt(
        context,
        tool_name="run_stat_test",
        arguments=args.model_dump(mode="json"),
        raw_output=result.model_dump(mode="json"),
        artifact_ids=(primary.id,),
        result_count=1,
        scope=ReceiptScope(
            dataset_ids=(args.dataset_id,),
            columns=tuple(
                column
                for column in (
                    args.group_column,
                    args.value_column,
                    args.category_column,
                    args.pair_column,
                )
                if column
            ),
        ),
        facts=facts,
        method=ReceiptMethod(
            # The family names the test that actually ran (fisher_exact /
            # welch_anova after a fallback), never just the requested one.
            family=result.test_type,
            parameters={
                "requested_test_type": args.test_type,
                "test_family_id": args.test_family_id,
                "comparison_count": args.comparison_count,
            },
            assumptions=tuple(
                f"{check.name}={check.status}" for check in result.assumptions
            ),
            warnings=tuple(
                [*switch_notes, *(warning.message for warning in result.warnings)]
            ),
        ),
        statistics=ReceiptStatistics(
            hypothesis_id=args.test_family_id,
            test_name=result.test_type,
            test_statistic=result.statistic,
            p_value=result.p_value,
            adjusted_p_value=result.adjusted_p_value,
            effect_size=result.effect_size,
            ci_low=result.effect_ci_low,
            ci_high=result.effect_ci_high,
            sample_size=result.sample_size,
        ),
    )
    return AgentToolResult(
        content={
            "artifact_id": primary.id,
            "receipt_id": receipt.receipt_id,
            "test_type": result.test_type,
            "requested_test_type": args.test_type,
            "p_value": result.p_value,
            "adjusted_p_value": result.adjusted_p_value,
            "effect_size": result.effect_size,
            "effect_ci": [result.effect_ci_low, result.effect_ci_high],
            "sample_size": result.sample_size,
            "assumptions": [
                f"{check.name}={check.status}" for check in result.assumptions
            ],
            "warnings": [*switch_notes, *(warning.message for warning in result.warnings)],
        },
        artifacts=[primary],
    )


def _correlate_columns(
    context: DataToolContext,
    args: CorrelateColumnsArguments,
) -> AgentToolResult:
    dataset = _single_dataset(context, args.dataset_id)
    table = correlate_column_pairs(
        dataset.frame,
        dataset_id=args.dataset_id,
        dataset_name=dataset.record.name,
        columns=args.columns,
        correction_method=args.correction_method,
    )
    payload = table.model_dump(mode="json")
    primary = Artifact(
        id=make_artifact_id("table", payload),
        type=ArtifactType.TABLE,
        project_id=context.project_id,
        session_id=context.session_id,
        payload=payload,
        plain_language=table.description,
    )
    context.add_artifact(primary)
    rows = table.rows
    pairs_tested = int(rows[0]["pairs_tested"]) if rows else 0
    significant = sum(1 for row in rows if float(row["adjusted_p"]) < 0.05)
    facts: list[ReceiptFact] = [
        _fact("pairs_tested", pairs_tested, "count"),
        _fact("correction_method", args.correction_method, "string"),
        _fact("significant_adjusted_pairs", significant, "count"),
    ]
    for index, row in enumerate(rows[:_MAX_FACT_PAIRS]):
        facts.extend(
            [
                _fact(f"pair{index}.pearson", row["pearson"], "number"),
                _fact(f"pair{index}.adjusted_p", row["adjusted_p"], "number"),
                _fact(
                    f"pair{index}.columns",
                    f"{row['column_a']}~{row['column_b']}",
                    "string",
                ),
            ]
        )
    trivial = sum(1 for row in rows if row.get("is_trivial_pair"))
    receipt = _emit_receipt(
        context,
        tool_name="correlate_columns",
        arguments=args.model_dump(mode="json"),
        raw_output=payload,
        artifact_ids=(primary.id,),
        result_count=pairs_tested,
        scope=ReceiptScope(
            dataset_ids=(args.dataset_id,),
            columns=tuple(args.columns or ()),
        ),
        facts=tuple(facts),
        method=ReceiptMethod(
            family="pearson_correlation_screen",
            parameters={
                "correction_method": args.correction_method,
                "pairs_tested": pairs_tested,
            },
            warnings=(
                (f"{trivial} published pair(s) look trivially coupled (is_trivial_pair).",)
                if trivial
                else ()
            ),
        ),
    )
    return AgentToolResult(
        content=cast(
            dict[str, Any],
            _clip_json(
                {
                    "artifact_id": primary.id,
                    "receipt_id": receipt.receipt_id,
                    "pairs_tested": pairs_tested,
                    "correction_method": args.correction_method,
                    "significant_adjusted_pairs": significant,
                    "top_pairs": rows[:10],
                }
            ),
        ),
        artifacts=[primary],
    )


_MAX_FACT_COLUMNS = 8


def _profile_slice(context: DataToolContext, args: ProfileSliceArguments) -> AgentToolResult:
    dataset = _single_dataset(context, args.dataset_id)
    profile = compute_slice_profile(
        dataset.frame,
        dataset_id=args.dataset_id,
        dataset_name=dataset.record.name,
        where_sql=args.where_sql,
        columns=args.columns,
    )
    scope = ReceiptScope(
        dataset_ids=(args.dataset_id,),
        columns=tuple(args.columns or ()),
        filters=args.where_sql,
    )
    facts: list[ReceiptFact] = [
        _fact("rows_in_slice", profile.rows_in_slice, "count"),
        _fact("rows_total", profile.rows_total, "count"),
        _fact("slice_share_percent", profile.slice_share_percent, "percent", unit="percent"),
    ]
    method_parameters: dict[str, str | int | float | bool | None] = {
        "where_sql": args.where_sql,
        "column_count": None if args.columns is None else len(args.columns),
        "rows_in_slice": profile.rows_in_slice,
    }
    if profile.table is None:
        facts.append(_absence_fact("empty_slice"))
        receipt = _emit_receipt(
            context,
            tool_name="profile_slice",
            arguments=args.model_dump(mode="json"),
            raw_output={"rows_in_slice": 0, "columns": []},
            artifact_ids=(),
            result_count=0,
            scope=scope,
            facts=tuple(facts),
            method=ReceiptMethod(
                family="slice_profile",
                parameters=method_parameters,
                warnings=("The WHERE condition matched no rows.",),
            ),
        )
        return AgentToolResult(
            content={
                "receipt_id": receipt.receipt_id,
                "rows_in_slice": 0,
                "slice_share_percent": profile.slice_share_percent,
                "note": "The WHERE condition matched no rows.",
            }
        )
    payload = profile.table.model_dump(mode="json")
    primary = Artifact(
        id=make_artifact_id("table", payload),
        type=ArtifactType.TABLE,
        project_id=context.project_id,
        session_id=context.session_id,
        payload=payload,
        plain_language=profile.table.description,
    )
    context.add_artifact(primary)
    for row in profile.table.rows[:_MAX_FACT_COLUMNS]:
        column = str(row["column"])
        facts.extend(
            [
                _fact(f"{column}.missing_percent", row["missing_percent"], "percent",
                      unit="percent"),
                _fact(f"{column}.unique_count", row["unique_count"], "count"),
                _fact(f"{column}.distribution_kind", row["distribution_kind"], "string"),
            ]
        )
        if row.get("mean") is not None:
            facts.append(_fact(f"{column}.mean", row["mean"], "number"))
            facts.append(_fact(f"{column}.median", row["median"], "number"))
    receipt = _emit_receipt(
        context,
        tool_name="profile_slice",
        arguments=args.model_dump(mode="json"),
        raw_output=payload,
        artifact_ids=(primary.id,),
        result_count=profile.rows_in_slice,
        scope=scope,
        facts=tuple(facts),
        method=ReceiptMethod(family="slice_profile", parameters=method_parameters),
    )
    return AgentToolResult(
        content=cast(
            dict[str, Any],
            _clip_json(
                {
                    "artifact_id": primary.id,
                    "receipt_id": receipt.receipt_id,
                    "rows_in_slice": profile.rows_in_slice,
                    "rows_total": profile.rows_total,
                    "slice_share_percent": profile.slice_share_percent,
                    "columns": profile.table.rows,
                }
            ),
        ),
        artifacts=[primary],
    )


def _analyze_time_series(
    context: DataToolContext,
    args: AnalyzeTimeSeriesArguments,
) -> AgentToolResult:
    dataset = _single_dataset(context, args.dataset_id)
    result = analyze_series(
        dataset.frame,
        dataset_id=args.dataset_id,
        dataset_name=dataset.record.name,
        time_column=args.time_column,
        value_column=args.value_column,
        freq=args.freq,
        period=args.period,
        agg=args.agg,
    )
    assert result.table is not None  # analyze_series always builds it
    payload = result.table.model_dump(mode="json")
    primary = Artifact(
        id=make_artifact_id("table", payload),
        type=ArtifactType.TABLE,
        project_id=context.project_id,
        session_id=context.session_id,
        payload=payload,
        warnings=list(result.warnings),
        plain_language=result.table.description,
    )
    context.add_artifact(primary)

    def _numeric_fact(fact_id: str, value: float | int | None) -> ReceiptFact:
        if value is None:
            return _fact(fact_id, None, "null")
        return _fact(fact_id, value, "number")

    facts = (
        _fact("n_periods", result.n_periods, "count"),
        _fact("gap_count", result.gap_count, "count"),
        _fact("regular_frequency", result.regular_frequency, "string"),
        _fact("trend_direction", result.trend_direction, "string"),
        _numeric_fact("seasonal_strength", result.seasonal_strength),
        _numeric_fact("ljung_box_p", result.ljung_box_p),
        _numeric_fact("adf_p", result.adf_p),
        _numeric_fact("kpss_p", result.kpss_p),
        _fact("stationarity_verdict", result.stationarity_verdict, "string"),
    )
    receipt = _emit_receipt(
        context,
        tool_name="analyze_time_series",
        arguments=args.model_dump(mode="json"),
        raw_output=payload,
        artifact_ids=(primary.id,),
        result_count=result.n_periods,
        scope=ReceiptScope(
            dataset_ids=(args.dataset_id,),
            columns=(args.time_column, args.value_column),
            time_range=result.time_range,
        ),
        facts=facts,
        method=ReceiptMethod(
            family="time_series_diagnostics",
            parameters={
                "agg": args.agg,
                "freq": result.regular_frequency,
                "period": result.period,
                "decomposition_performed": result.decomposition_performed,
                "log_transformed": result.log_transformed,
                "ljung_box_lag": result.ljung_box_lag,
            },
            warnings=tuple(result.warnings),
        ),
        statistics=ReceiptStatistics(
            test_name="ljung_box",
            p_value=result.ljung_box_p,
            sample_size=result.n_periods,
        ),
    )
    return AgentToolResult(
        content={
            "artifact_id": primary.id,
            "receipt_id": receipt.receipt_id,
            "n_periods": result.n_periods,
            "gap_count": result.gap_count,
            "regular_frequency": result.regular_frequency,
            "trend_direction": result.trend_direction,
            "seasonal_strength": result.seasonal_strength,
            "ljung_box_p": result.ljung_box_p,
            "adf_p": result.adf_p,
            "kpss_p": result.kpss_p,
            "stationarity_verdict": result.stationarity_verdict,
            "warnings": list(result.warnings),
        },
        artifacts=[primary],
    )


def _clip_json(value: Any, *, max_chars: int = 12_000) -> Any:
    """Keep tool observations bounded even when an artifact contains a table."""
    encoded = json.dumps(value, ensure_ascii=False, default=str)
    if len(encoded) <= max_chars:
        return value
    return {
        "truncated": True,
        "preview": encoded[:max_chars],
        "original_chars": len(encoded),
    }
