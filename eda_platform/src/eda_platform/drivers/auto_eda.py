from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from math import ceil
from pathlib import Path
from shutil import copy2
from time import perf_counter
from typing import Any, ClassVar, Literal

import pandas as pd

from eda_platform.agents.question_agent import propose_llm_question_candidates
from eda_platform.agents.reporting import generate_agentic_report
from eda_platform.agents.semantic_bootstrap import bootstrap_semantics
from eda_platform.core.budget import BudgetExceeded, SessionBudgetPolicy
from eda_platform.core.column_roles import (
    ColumnRoleName,
    ColumnRoleSet,
    column_role_set_artifact,
    infer_column_roles,
)
from eda_platform.core.config import resolve_workspace_path
from eda_platform.core.ids import hash_file, make_artifact_id, make_dataset_id, stable_hash
from eda_platform.core.kernel import SessionContext, run_pipeline
from eda_platform.core.llm import (
    LLMClient,
    OfflineLLMClient,
    is_offline_client,
    llm_execution_fingerprint,
    manifest_model_versions,
)
from eda_platform.core.llm_ledger import meter_llm_client, restore_run_budget_state
from eda_platform.core.meaning_proposals import MeaningProposal
from eda_platform.core.methods import MethodGateContext, evaluate_feasibility
from eda_platform.core.process_metrics import PeakRssMeasurement, process_peak_rss
from eda_platform.core.provenance import env_digest
from eda_platform.core.query import DuckDBQueryEngine
from eda_platform.core.semantic import (
    JoinWhitelist,
    join_whitelist_path,
    load_join_whitelist,
    save_join_whitelist,
)
from eda_platform.core.semantic_resources import (
    SemanticSeedsRepository,
    load_semantic_seeds_safe,
)
from eda_platform.core.session_metrics import (
    create_run_metrics_artifact,
    persist_run_metrics,
)
from eda_platform.core.skills_store import catalog_block
from eda_platform.core.store import ArtifactStore
from eda_platform.core.support_docs import extract_support_snippets, load_support_docs
from eda_platform.core.tool_guard import ToolGuardError
from eda_platform.drivers.cancellation import raise_if_cancelled
from eda_platform.drivers.question_exec import execute_question_candidate
from eda_platform.drivers.report_artifacts import build_agentic_report_artifacts
from eda_platform.schemas.artifacts import Artifact, ArtifactType, DatasetProfile
from eda_platform.schemas.cleaning import CleaningRecipe
from eda_platform.schemas.questions import QuestionCandidate, QuestionCandidateSet
from eda_platform.schemas.relations import (
    RelationshipCandidate,
    RelationshipCandidateSet,
    RelationshipValidation,
    RelationshipValidationSet,
)
from eda_platform.schemas.resource_metrics import (
    EdaDataFootprint,
    EdaDatasetEstimate,
    EdaResourcePolicy,
    EdaResourcePreflight,
)
from eda_platform.schemas.sessions import (
    SessionManifest,
    TraceEvent,
    build_run_title,
    clip_run_title,
)
from eda_platform.schemas.stats import StatTestType, StatWarning
from eda_platform.schemas.value_discovery import ValueMap
from eda_platform.tools.agent_handoff import create_agent_handoff_artifact
from eda_platform.tools.analysis import create_analysis_tables
from eda_platform.tools.chart_specs import (
    create_association_chart_specs,
    create_chart_specs,
    create_correlation_chart_specs,
)
from eda_platform.tools.domain_metrics import applicable_metrics
from eda_platform.tools.er_diagram import build_er_diagram
from eda_platform.tools.evidence import PayloadPolicy
from eda_platform.tools.handoff import create_eda_handoff_artifact
from eda_platform.tools.loader import LoadedDataset, load_csv
from eda_platform.tools.ml_baseline import create_model_card_artifact, run_baseline_model
from eda_platform.tools.pii import mask_profile_artifact, pii_labels, tag_pii_columns
from eda_platform.tools.profiler import profile_dataset
from eda_platform.tools.quality import scan_quality
from eda_platform.tools.quality_context import build_quality_context
from eda_platform.tools.question_discovery import (
    auto_execution_composition,
    discover_question_candidates,
    select_auto_execution_set,
)
from eda_platform.tools.relationship_discovery import (
    discover_relationship_candidates,
    eager_validation_candidates,
    propose_join_candidates,
    validate_relationships,
)
from eda_platform.tools.resource_preflight import (
    EdaResourceLimitError,
    preflight_csv_resources,
)
from eda_platform.tools.stat_tests import (
    create_anova_boxplot_artifact,
    create_stat_test_artifact,
    run_stat_test,
)
from eda_platform.tools.value_discovery import build_value_map, enrich_question_candidates

# A resource-limited run publishes preflight/metrics/handoff and nothing else.
# It is terminal but not a completed analysis: marking it "completed" made an
# empty run look like a successful one in the session list and in compare.
LIMITED_SESSION_STATUS = "limited"


@dataclass(frozen=True)
class AutoEDAResult:
    project_id: str
    session_id: str
    business_context: str
    artifacts: list[Artifact]
    report_markdown: str
    workspace: Path
    loaded_datasets: list[LoadedDataset]
    # Non-empty only when this result was rebuilt from disk by ``load_run`` and
    # some source was missing/corrupt; a fresh ``run_auto_eda`` leaves it empty.
    load_warnings: list[str] = field(default_factory=list)


def validate_relationship_candidate_on_demand(
    result: AutoEDAResult,
    candidate: RelationshipCandidate,
    *,
    cancel_check: Callable[[], bool] | None = None,
) -> RelationshipValidation:
    """Fully validate one deferred relationship and persist the merged result."""
    raise_if_cancelled(cancel_check, operation="relationship validation")
    existing = _merged_relationship_validations(result.artifacts)
    label = candidate.pair.label()
    for validation in existing.validations:
        if validation.pair.label() == label and validation.verified:
            return validation

    available_ids = {loaded.record.dataset_id for loaded in result.loaded_datasets}
    required_ids = {
        candidate.pair.left_dataset_id,
        candidate.pair.right_dataset_id,
    }
    missing_ids = sorted(required_ids - available_ids)
    if missing_ids:
        raise ValueError(
            "Full relationship validation requires both source datasets; "
            f"missing dataset id(s): {', '.join(missing_ids)}"
        )

    engine = DuckDBQueryEngine()
    for loaded in result.loaded_datasets:
        raise_if_cancelled(cancel_check, operation="relationship validation")
        engine.register_frame(loaded.record.dataset_id, loaded.frame)
    raise_if_cancelled(cancel_check, operation="relationship validation")
    validated = validate_relationships([candidate], engine)
    raise_if_cancelled(cancel_check, operation="relationship validation")
    if not validated.validations:
        raise ValueError(f"Relationship is not eligible for validation: {label}")
    validation = validated.validations[0]

    merged_by_label = {item.pair.label(): item for item in [*existing.validations, validation]}
    merged = RelationshipValidationSet(
        validations=[merged_by_label[key] for key in sorted(merged_by_label)]
    )
    payload = merged.model_dump(mode="json")
    candidate_parents = [
        artifact.id
        for artifact in result.artifacts
        if artifact.type is ArtifactType.RELATIONSHIP_CANDIDATE_SET
    ]
    artifact = Artifact(
        id=make_artifact_id("relval", payload),
        type=ArtifactType.RELATIONSHIP_VALIDATION_SET,
        project_id=result.project_id,
        session_id=result.session_id,
        parents=candidate_parents,
        payload=payload,
        plain_language="On-demand full validation before relationship confirmation.",
    )
    store = ArtifactStore(result.workspace)
    store.save_artifact(artifact)
    project_dir = store.project_dir(result.project_id)
    whitelist = load_join_whitelist(project_dir)
    whitelist_entry = whitelist.entry(label)
    if whitelist_entry is not None:
        whitelist_entry.left_dataset_id = candidate.pair.left_dataset_id
        whitelist_entry.right_dataset_id = candidate.pair.right_dataset_id
        whitelist_entry.cardinality = validation.cardinality
        whitelist_entry.join_row_multiplier = validation.join_row_multiplier
        whitelist_entry.validation_verified = validation.verified
        save_join_whitelist(project_dir, whitelist)
    store.append_trace(
        result.project_id,
        TraceEvent(
            session_id=result.session_id,
            event_type="relationship_validation_on_demand",
            name="validate_relationship_candidate_on_demand",
            finished_at=datetime.now(UTC),
            summary={
                "pair": label,
                "full_validations_completed": 1,
                "cardinality": validation.cardinality,
                "join_row_multiplier": validation.join_row_multiplier,
                "sampled": validation.sampled,
            },
        ),
    )
    result.artifacts.append(artifact)
    _persist_run_metrics_best_effort(store, result.project_id, result.session_id)
    return validation


def discover_relationships_on_demand(
    result: AutoEDAResult,
    *,
    cancel_check: Callable[[], bool] | None = None,
) -> RelationshipCandidateSet:
    """Materialize deferred relationship artifacts for an existing run."""
    raise_if_cancelled(cancel_check, operation="relationship discovery")
    existing = _relationship_candidates(result.artifacts)
    if existing is not None:
        return existing
    if len(result.loaded_datasets) < 2:
        raise ValueError("Relationship discovery requires at least two datasets.")

    profile_artifacts = [
        artifact for artifact in result.artifacts if artifact.type is ArtifactType.DATASET_PROFILE
    ]
    artifacts, candidates, validations = _build_relationship_artifacts(
        result.loaded_datasets,
        profile_artifact_ids=[artifact.id for artifact in profile_artifacts],
        project_id=result.project_id,
        session_id=result.session_id,
    )
    raise_if_cancelled(cancel_check, operation="relationship discovery")
    store = ArtifactStore(result.workspace)
    for artifact in artifacts:
        raise_if_cancelled(cancel_check, operation="relationship discovery")
        store.save_artifact(artifact)
    result.artifacts.extend(artifacts)
    store.append_trace(
        result.project_id,
        TraceEvent(
            session_id=result.session_id,
            event_type="relationship_discovery_bounded",
            name="discover_relationships_on_demand",
            finished_at=datetime.now(UTC),
            summary={
                **_relationship_trace_summary(candidates, validations),
                "trigger": "relationships_on_demand",
            },
        ),
    )

    role_sets: dict[str, ColumnRoleSet] = {}
    for artifact in result.artifacts:
        if artifact.type is ArtifactType.COLUMN_ROLE_SET:
            role_set = ColumnRoleSet.model_validate(artifact.payload)
            role_sets[role_set.dataset] = role_set
    project_dir = store.project_dir(result.project_id)

    def _record(event: TraceEvent) -> None:
        store.append_trace(result.project_id, event)

    whitelist = _load_join_whitelist_safe(project_dir, session_id=result.session_id, emit=_record)
    proposals = propose_join_candidates(candidates, validations, role_sets=role_sets)
    newly_proposed = whitelist.merge_proposals(proposals)
    if proposals:
        # merge_proposals also refreshes validation dataset identities for an
        # existing label; persist even when the added-label count is zero.
        save_join_whitelist(project_dir, whitelist)
    if newly_proposed:
        store.append_trace(
            result.project_id,
            TraceEvent(
                session_id=result.session_id,
                event_type="join_candidates_proposed",
                name="discover_relationships_on_demand",
                finished_at=datetime.now(UTC),
                summary={
                    "proposed_count": newly_proposed,
                    "labels": [entry.label() for entry in whitelist.entries],
                    "trigger": "relationships_on_demand",
                },
            ),
        )
    _persist_run_metrics_best_effort(store, result.project_id, result.session_id)
    return candidates


@dataclass(frozen=True)
class _StatTestSpec:
    test_type: StatTestType
    group_column: str
    value_column: str


class EmitCleaningRecipeStep:
    """Record an applied pre-clean as a typed ``CleaningRecipe`` artifact."""

    name: ClassVar[str] = "emit_cleaning_recipe"
    requires: ClassVar[tuple[ArtifactType, ...]] = ()
    produces: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.CLEANING_RECIPE,)

    def __init__(self, recipe: CleaningRecipe) -> None:
        self.recipe = recipe

    def run(self, ctx: SessionContext) -> list[Artifact]:
        payload = self.recipe.model_dump(mode="json")
        # A data-changing clean was auto-applied without an interactive HITL
        # gate (spec line 235); flag it on the artifact so it is auditable and a
        # future approval UI can enforce the gate on the recorded recipe.
        warnings = ["auto_applied_without_hitl"] if self.recipe.requires_approval else []
        artifact = Artifact(
            id=make_artifact_id("clean", payload),
            type=ArtifactType.CLEANING_RECIPE,
            project_id=ctx.project_id,
            session_id=ctx.session_id,
            payload=payload,
            warnings=warnings,
        )
        ctx.emit_trace(
            TraceEvent(
                session_id=ctx.session_id,
                event_type="cleaning_applied",
                name=self.name,
                summary={
                    "recipe_id": self.recipe.recipe_id,
                    "dataset_id": self.recipe.dataset_id,
                    "transform_count": len(self.recipe.transforms),
                    "requires_approval": self.recipe.requires_approval,
                },
            )
        )
        return [artifact]


class ProfileDatasetStep:
    name: ClassVar[str] = "profile_dataset"
    parallel_safe: ClassVar[bool] = True
    requires: ClassVar[tuple[ArtifactType, ...]] = ()
    produces: ClassVar[tuple[ArtifactType, ...]] = (
        ArtifactType.DATASET_PROFILE,
        ArtifactType.PII_REPORT,
    )

    def __init__(self, loaded: LoadedDataset, parent_ids: list[str] | None = None) -> None:
        self.loaded = loaded
        self.parent_ids = parent_ids or []

    def run(self, ctx: SessionContext) -> list[Artifact]:
        profile = profile_dataset(
            self.loaded,
            project_id=ctx.project_id,
            session_id=ctx.session_id,
            parents=self.parent_ids,
        )
        pii = tag_pii_columns(
            profile,
            project_id=ctx.project_id,
            session_id=ctx.session_id,
        )
        masked = mask_profile_artifact(profile, pii)
        pii = pii.model_copy(update={"parents": [masked.id]})
        return [masked, pii]


class ScanQualityStep:
    name: ClassVar[str] = "scan_quality"
    parallel_safe: ClassVar[bool] = True
    requires: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.DATASET_PROFILE,)
    produces: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.QUALITY_ISSUE_SET,)

    def __init__(self, profile_artifact_id: str) -> None:
        self.profile_artifact_id = profile_artifact_id

    def run(self, ctx: SessionContext) -> list[Artifact]:
        profile_artifact = ctx.store.get_artifact(
            self.profile_artifact_id,
            project_id=ctx.project_id,
            session_id=ctx.session_id,
        )
        return [
            scan_quality(
                profile_artifact,
                project_id=ctx.project_id,
                session_id=ctx.session_id,
            )
        ]


class BuildQualityContextStep:
    """Turn scanned quality issues into disclosable EDA quality context."""

    name: ClassVar[str] = "build_quality_context"
    parallel_safe: ClassVar[bool] = True
    requires: ClassVar[tuple[ArtifactType, ...]] = (
        ArtifactType.DATASET_PROFILE,
        ArtifactType.QUALITY_ISSUE_SET,
    )
    produces: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.QUALITY_CONTEXT_SET,)

    def __init__(
        self,
        loaded: LoadedDataset,
        profile_artifact_id: str,
        quality_artifact_id: str,
    ) -> None:
        self.loaded = loaded
        self.profile_artifact_id = profile_artifact_id
        self.quality_artifact_id = quality_artifact_id

    def run(self, ctx: SessionContext) -> list[Artifact]:
        profile_artifact = ctx.store.get_artifact(
            self.profile_artifact_id,
            project_id=ctx.project_id,
            session_id=ctx.session_id,
        )
        quality_artifact = ctx.store.get_artifact(
            self.quality_artifact_id,
            project_id=ctx.project_id,
            session_id=ctx.session_id,
        )
        return [
            build_quality_context(
                self.loaded,
                profile_artifact,
                quality_artifact,
                project_id=ctx.project_id,
                session_id=ctx.session_id,
            )
        ]


class BuildValueMapStep:
    """Emit the Role-1 ValueMap of possible, evidence-led value paths."""

    name: ClassVar[str] = "build_value_map"
    requires: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.DATASET_PROFILE,)
    produces: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.VALUE_MAP,)

    def __init__(
        self,
        *,
        source_artifact_ids: list[str],
        business_context: str,
    ) -> None:
        self.source_artifact_ids = source_artifact_ids
        self.business_context = business_context

    def run(self, ctx: SessionContext) -> list[Artifact]:
        source_artifacts = [
            ctx.store.get_artifact(
                artifact_id,
                project_id=ctx.project_id,
                session_id=ctx.session_id,
            )
            for artifact_id in self.source_artifact_ids
        ]
        value_map = build_value_map(
            source_artifacts,
            business_context=self.business_context,
        )
        payload = value_map.model_dump(mode="json")
        return [
            Artifact(
                id=make_artifact_id(
                    "valuemap",
                    {"session_id": ctx.session_id, "value_map": payload},
                ),
                type=ArtifactType.VALUE_MAP,
                project_id=ctx.project_id,
                session_id=ctx.session_id,
                parents=self.source_artifact_ids,
                payload=payload,
                plain_language=(
                    "Possible, evidence-led value paths for Role 1. Interpretations and "
                    "hypotheses only, never asserted business results."
                ),
            )
        ]


class CreateChartSpecsStep:
    name: ClassVar[str] = "create_chart_specs"
    parallel_safe: ClassVar[bool] = True
    requires: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.DATASET_PROFILE,)
    produces: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.CHART_SPEC,)

    def __init__(self, loaded: LoadedDataset, profile_artifact_id: str) -> None:
        self.loaded = loaded
        self.profile_artifact_id = profile_artifact_id

    def run(self, ctx: SessionContext) -> list[Artifact]:
        profile_artifact = ctx.store.get_artifact(
            self.profile_artifact_id,
            project_id=ctx.project_id,
            session_id=ctx.session_id,
        )
        profile = DatasetProfile.model_validate(profile_artifact.payload)
        chart_loaded = _masked_loaded_dataset(self.loaded, profile.pii_columns)
        return create_chart_specs(
            chart_loaded,
            profile_artifact,
            project_id=ctx.project_id,
            session_id=ctx.session_id,
        )


class RecordRawDatasetStep:
    name: ClassVar[str] = "record_raw_dataset"
    requires: ClassVar[tuple[ArtifactType, ...]] = ()
    produces: ClassVar[tuple[ArtifactType, ...]] = (
        ArtifactType.RAW_DATASET_PROFILE,
        ArtifactType.RAW_CHART_SPEC,
        ArtifactType.RAW_DATA_PREVIEW,
        ArtifactType.PII_REPORT,
    )

    def __init__(self, loaded: LoadedDataset) -> None:
        self.loaded = loaded

    def run(self, ctx: SessionContext) -> list[Artifact]:
        profile = profile_dataset(
            self.loaded,
            project_id=ctx.project_id,
            session_id=ctx.session_id,
        )
        pii = tag_pii_columns(
            profile,
            project_id=ctx.project_id,
            session_id=ctx.session_id,
        )
        masked_profile = mask_profile_artifact(profile, pii)
        raw_profile = _retag_raw_artifact(
            masked_profile,
            ArtifactType.RAW_DATASET_PROFILE,
            "raw_prof",
            parents=[],
        )
        labels = pii_labels(pii)
        chart_loaded = _masked_loaded_dataset(self.loaded, labels)
        raw_charts = [
            _retag_raw_artifact(artifact, ArtifactType.RAW_CHART_SPEC, "raw_chart")
            for artifact in create_chart_specs(
                chart_loaded,
                raw_profile,
                project_id=ctx.project_id,
                session_id=ctx.session_id,
            )
        ]
        preview_payload = _raw_preview_payload(chart_loaded)
        raw_preview = Artifact(
            id=make_artifact_id("raw_preview", preview_payload),
            type=ArtifactType.RAW_DATA_PREVIEW,
            project_id=ctx.project_id,
            session_id=ctx.session_id,
            payload=preview_payload,
            warnings=["before_cleaning"],
        )
        pii = pii.model_copy(update={"parents": [raw_profile.id]})
        return [raw_profile, pii, *raw_charts, raw_preview]


class CreateAnalysisTablesStep:
    name: ClassVar[str] = "create_analysis_tables"
    parallel_safe: ClassVar[bool] = True
    requires: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.DATASET_PROFILE,)
    produces: ClassVar[tuple[ArtifactType, ...]] = (
        ArtifactType.TABLE,
        ArtifactType.CHART_SPEC,
    )

    def __init__(self, loaded: LoadedDataset, profile_artifact_id: str) -> None:
        self.loaded = loaded
        self.profile_artifact_id = profile_artifact_id

    def run(self, ctx: SessionContext) -> list[Artifact]:
        profile_artifact = ctx.store.get_artifact(
            self.profile_artifact_id,
            project_id=ctx.project_id,
            session_id=ctx.session_id,
        )
        tables = create_analysis_tables(
            self.loaded,
            profile_artifact,
            project_id=ctx.project_id,
            session_id=ctx.session_id,
        )
        # Tables publish aggregates only; charts inline row values, so scatter
        # generation must run on the PII-masked frame (masked pairs turn
        # non-numeric and simply produce no scatter).
        profile = DatasetProfile.model_validate(profile_artifact.payload)
        chart_loaded = _masked_loaded_dataset(self.loaded, profile.pii_columns)
        charts = [
            chart
            for table in tables
            if table.payload.get("kind") == "correlation"
            for chart in create_correlation_chart_specs(
                chart_loaded,
                table,
                project_id=ctx.project_id,
                session_id=ctx.session_id,
            )
        ]
        charts.extend(
            chart
            for table in tables
            if table.payload.get("kind") == "association"
            for chart in create_association_chart_specs(
                table,
                project_id=ctx.project_id,
                session_id=ctx.session_id,
            )
        )
        return [*tables, *charts]


class SessionStatTestsStep:
    name: ClassVar[str] = "run_stat_tests"
    requires: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.DATASET_PROFILE,)
    produces: ClassVar[tuple[ArtifactType, ...]] = (
        ArtifactType.STAT_TEST_RESULT,
        ArtifactType.CHART_SPEC,
    )

    def __init__(
        self,
        loaded: LoadedDataset,
        profile_artifact_id: str,
        spec: _StatTestSpec,
        *,
        comparison_count: int = 1,
    ) -> None:
        self.loaded = loaded
        self.profile_artifact_id = profile_artifact_id
        self.spec = spec
        self.comparison_count = comparison_count

    def run(self, ctx: SessionContext) -> list[Artifact]:
        profile_artifact = ctx.store.get_artifact(
            self.profile_artifact_id,
            project_id=ctx.project_id,
            session_id=ctx.session_id,
        )
        try:
            result = run_stat_test(
                self.loaded.frame,
                dataset_id=self.loaded.record.dataset_id,
                test_type=self.spec.test_type,
                group_column=self.spec.group_column,
                value_column=self.spec.value_column,
                comparison_count=self.comparison_count,
            )
        except ToolGuardError as exc:
            ctx.emit_trace(
                TraceEvent(
                    session_id=ctx.session_id,
                    event_type="tool_guard_rejected",
                    name=self.name,
                    finished_at=datetime.now(UTC),
                    summary=exc.to_trace_summary(),
                )
            )
            return []
        except ValueError:
            return []
        result.warnings.append(
            StatWarning(
                code="exploratory_auto_selection",
                message=(
                    "This test was auto-selected from the dataset's candidate "
                    "group/measure combinations during exploration; treat it as "
                    "hypothesis-generating, not confirmatory."
                ),
            )
        )
        stat_artifact = create_stat_test_artifact(
            result,
            project_id=ctx.project_id,
            session_id=ctx.session_id,
            parents=[profile_artifact.id],
        )
        boxplot = create_anova_boxplot_artifact(
            self.loaded.frame,
            result,
            project_id=ctx.project_id,
            session_id=ctx.session_id,
            parents=[stat_artifact.id],
        )
        return [stat_artifact, *([boxplot] if boxplot is not None else [])]


class SessionBaselineModelStep:
    name: ClassVar[str] = "run_baseline_model"
    requires: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.DATASET_PROFILE,)
    produces: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.MODEL_CARD,)

    def __init__(
        self,
        loaded: LoadedDataset,
        profile_artifact_id: str,
        *,
        target_column: str,
        time_column: str | None = None,
    ) -> None:
        self.loaded = loaded
        self.profile_artifact_id = profile_artifact_id
        self.target_column = target_column
        self.time_column = time_column

    def run(self, ctx: SessionContext) -> list[Artifact]:
        profile_artifact = ctx.store.get_artifact(
            self.profile_artifact_id,
            project_id=ctx.project_id,
            session_id=ctx.session_id,
        )
        try:
            card = run_baseline_model(
                self.loaded.frame,
                dataset_id=self.loaded.record.dataset_id,
                target_column=self.target_column,
                time_column=self.time_column,
            )
        except ToolGuardError as exc:
            ctx.emit_trace(
                TraceEvent(
                    session_id=ctx.session_id,
                    event_type="tool_guard_rejected",
                    name=self.name,
                    finished_at=datetime.now(UTC),
                    summary=exc.to_trace_summary(),
                )
            )
            return []
        except ValueError as exc:
            ctx.emit_trace(
                TraceEvent(
                    session_id=ctx.session_id,
                    event_type="ml_baseline_skipped",
                    name=self.name,
                    finished_at=datetime.now(UTC),
                    summary={
                        "dataset_id": self.loaded.record.dataset_id,
                        "target_column": self.target_column,
                        "reason": str(exc),
                    },
                )
            )
            return []
        return [
            create_model_card_artifact(
                card,
                project_id=ctx.project_id,
                session_id=ctx.session_id,
                parents=[profile_artifact.id],
            )
        ]


class DiscoverRelationshipsStep:
    name: ClassVar[str] = "discover_relationships"
    requires: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.DATASET_PROFILE,)
    produces: ClassVar[tuple[ArtifactType, ...]] = (
        ArtifactType.RELATIONSHIP_CANDIDATE_SET,
        ArtifactType.RELATIONSHIP_VALIDATION_SET,
        ArtifactType.ER_DIAGRAM,
    )

    def __init__(
        self,
        loaded_datasets: Sequence[LoadedDataset],
        profile_artifact_ids: list[str],
    ) -> None:
        self.loaded_datasets = list(loaded_datasets)
        self.profile_artifact_ids = profile_artifact_ids

    def run(self, ctx: SessionContext) -> list[Artifact]:
        artifacts, candidates, validations = _build_relationship_artifacts(
            self.loaded_datasets,
            profile_artifact_ids=self.profile_artifact_ids,
            project_id=ctx.project_id,
            session_id=ctx.session_id,
        )
        ctx.emit_trace(
            TraceEvent(
                session_id=ctx.session_id,
                event_type="relationship_discovery_bounded",
                name=self.name,
                finished_at=datetime.now(UTC),
                summary=_relationship_trace_summary(candidates, validations),
            )
        )
        return artifacts


def _build_relationship_artifacts(
    loaded_datasets: Sequence[LoadedDataset],
    *,
    profile_artifact_ids: list[str],
    project_id: str,
    session_id: str,
) -> tuple[list[Artifact], RelationshipCandidateSet, RelationshipValidationSet]:
    engine = DuckDBQueryEngine()
    for loaded in loaded_datasets:
        engine.register_frame(loaded.record.dataset_id, loaded.frame)
    candidates = discover_relationship_candidates(loaded_datasets, engine)
    validations = validate_relationships(eager_validation_candidates(candidates), engine)
    diagram = build_er_diagram(candidates, validations)
    candidate_payload = candidates.model_dump(mode="json")
    validation_payload = validations.model_dump(mode="json")
    diagram_payload = diagram.model_dump(mode="json")
    parents = list(profile_artifact_ids)
    return (
        [
            Artifact(
                id=make_artifact_id("relcand", candidate_payload),
                type=ArtifactType.RELATIONSHIP_CANDIDATE_SET,
                project_id=project_id,
                session_id=session_id,
                parents=parents,
                payload=candidate_payload,
            ),
            Artifact(
                id=make_artifact_id("relval", validation_payload),
                type=ArtifactType.RELATIONSHIP_VALIDATION_SET,
                project_id=project_id,
                session_id=session_id,
                parents=parents,
                payload=validation_payload,
            ),
            Artifact(
                id=make_artifact_id("erd", diagram_payload),
                type=ArtifactType.ER_DIAGRAM,
                project_id=project_id,
                session_id=session_id,
                parents=parents,
                payload=diagram_payload,
            ),
        ],
        candidates,
        validations,
    )


def _relationship_trace_summary(
    candidates: RelationshipCandidateSet,
    validations: RelationshipValidationSet,
) -> dict[str, object]:
    return {
        "overlap_pairs_evaluated": candidates.overlap_pairs_evaluated,
        "overlap_pairs_prefiltered": candidates.overlap_pairs_prefiltered,
        "candidate_count": len(candidates.candidates),
        "full_validation_targets": len(eager_validation_candidates(candidates)),
        "full_validations_completed": len(validations.validations),
        "coverage_status": candidates.coverage_status,
        "candidate_payload_bytes": len(candidates.model_dump_json().encode("utf-8")),
    }


# Templates backstop coverage gaps after the primary LLM route.
_LLM_PRIMARY_MAX_QUESTIONS = 12


class DiscoverQuestionsStep:
    name: ClassVar[str] = "discover_questions"
    # Analysis tables are optional: narrow datasets can legitimately produce none.
    requires: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.DATASET_PROFILE,)
    produces: ClassVar[tuple[ArtifactType, ...]] = (
        ArtifactType.QUESTION_CANDIDATE_SET,
        ArtifactType.COLUMN_ROLE_SET,
    )

    def __init__(
        self,
        loaded_datasets: Sequence[LoadedDataset],
        *,
        profile_artifact_ids: list[str],
        quality_artifact_ids: list[str],
        quality_context_artifact_ids: list[str],
        analysis_artifact_ids: list[str],
        relationship_artifact_ids: list[str],
        value_map_artifact_id: str,
        llm: LLMClient,
        business_context: str,
        payload_policy: PayloadPolicy,
    ) -> None:
        self.loaded_datasets = list(loaded_datasets)
        self.profile_artifact_ids = profile_artifact_ids
        self.quality_artifact_ids = quality_artifact_ids
        self.quality_context_artifact_ids = quality_context_artifact_ids
        self.analysis_artifact_ids = analysis_artifact_ids
        self.relationship_artifact_ids = relationship_artifact_ids
        self.value_map_artifact_id = value_map_artifact_id
        self.llm = llm
        self.business_context = business_context
        self.payload_policy: PayloadPolicy = payload_policy

    def run(self, ctx: SessionContext) -> list[Artifact]:
        profile_artifacts = [
            ctx.store.get_artifact(
                artifact_id,
                project_id=ctx.project_id,
                session_id=ctx.session_id,
            )
            for artifact_id in self.profile_artifact_ids
        ]
        quality_artifacts = [
            ctx.store.get_artifact(
                artifact_id,
                project_id=ctx.project_id,
                session_id=ctx.session_id,
            )
            for artifact_id in self.quality_artifact_ids
        ]
        analysis_artifacts = [
            ctx.store.get_artifact(
                artifact_id,
                project_id=ctx.project_id,
                session_id=ctx.session_id,
            )
            for artifact_id in self.analysis_artifact_ids
        ]
        relationship_artifacts = [
            ctx.store.get_artifact(
                artifact_id,
                project_id=ctx.project_id,
                session_id=ctx.session_id,
            )
            for artifact_id in self.relationship_artifact_ids
        ]
        relationship_candidates = _relationship_candidates(relationship_artifacts)
        relationship_validations = _relationship_validations(relationship_artifacts)
        source_artifacts = [
            *profile_artifacts,
            *quality_artifacts,
            *analysis_artifacts,
            *relationship_artifacts,
        ]
        seeds = load_semantic_seeds_safe(ctx.store, ctx.project_id)
        # Optional user reference docs feed bootstrap as priors; none → no-op.
        support_docs = load_support_docs(ctx.store.project_dir(ctx.project_id))

        # Bootstrap semantic roles before question discovery.
        loaded_by_id = {loaded.record.dataset_id: loaded for loaded in self.loaded_datasets}
        role_sets: dict[str, ColumnRoleSet] = {}
        role_artifacts: list[Artifact] = []
        meaning_drafts: list[MeaningProposal] = []
        for profile_artifact in profile_artifacts:
            profile = DatasetProfile.model_validate(profile_artifact.payload)
            loaded = loaded_by_id.get(profile.dataset_id)
            snippets = (
                extract_support_snippets(
                    support_docs, dataset=profile.name, column_names=profile.column_names
                )
                if support_docs
                else None
            )
            bootstrap = bootstrap_semantics(
                profile,
                llm=self.llm,
                frame=loaded.frame if loaded is not None else None,
                seeds=seeds,
                support_doc_snippets=snippets,
            )
            role_sets[profile.name] = bootstrap.role_set
            meaning_drafts.extend(bootstrap.meaning_drafts)
            role_artifacts.append(
                column_role_set_artifact(
                    bootstrap.role_set,
                    project_id=ctx.project_id,
                    session_id=ctx.session_id,
                    parents=[profile_artifact.id],
                )
            )
            ctx.emit_trace(
                TraceEvent(
                    session_id=ctx.session_id,
                    event_type="semantic_bootstrap",
                    name="di8_semantic_bootstrap",
                    finished_at=datetime.now(UTC),
                    summary={
                        "dataset": profile.name,
                        "degraded": bootstrap.degraded,
                        "degraded_reason": bootstrap.degraded_reason,
                        "hypothesis_count": bootstrap.hypothesis_count,
                        "verified_count": bootstrap.verified_count,
                        "unverified_count": bootstrap.unverified_count,
                        "unmapped_labels": bootstrap.unmapped_labels,
                    },
                )
            )
            # Record bootstrap usage separately from its semantic trace event.
            if bootstrap.llm_usage is not None:
                usage = bootstrap.llm_usage
                ctx.emit_trace(
                    TraceEvent(
                        session_id=ctx.session_id,
                        event_type="llm_call",
                        name="semantic_bootstrap",
                        finished_at=datetime.now(UTC),
                        summary={
                            "dataset": profile.name,
                            "provider": usage.provider,
                            "model": usage.model,
                            "prompt_tokens": usage.usage.prompt_tokens,
                            "completion_tokens": usage.usage.completion_tokens,
                            "total_tokens": usage.usage.total_tokens,
                            "cached_tokens": usage.usage.cached_tokens,
                            "estimated_cost_usd": usage.estimated_cost_usd,
                        },
                    )
                )
                ctx.budget.add_tokens(usage.usage.total_tokens)

        # Persist discovered joins as proposals pending confirmation.
        project_dir = ctx.store.project_dir(ctx.project_id)
        # Persist meaning drafts for Knowledge-page review (reviewed entries kept).
        if meaning_drafts:
            SemanticSeedsRepository(ctx.store, ctx.project_id).upsert_proposals(
                meaning_drafts,
                request_key=f"auto-eda-proposals:{ctx.session_id}",
            )
        whitelist = _load_join_whitelist_safe(
            project_dir,
            session_id=ctx.session_id,
            emit=ctx.emit_trace,
        )
        if relationship_candidates is not None:
            proposals = propose_join_candidates(
                relationship_candidates,
                relationship_validations,
                role_sets=role_sets,
            )
            newly_proposed = whitelist.merge_proposals(proposals)
            if proposals:
                save_join_whitelist(project_dir, whitelist)
            if newly_proposed:
                ctx.emit_trace(
                    TraceEvent(
                        session_id=ctx.session_id,
                        event_type="join_candidates_proposed",
                        name="discover_questions",
                        finished_at=datetime.now(UTC),
                        summary={
                            "proposed_count": newly_proposed,
                            "labels": [entry.label() for entry in whitelist.entries],
                        },
                    )
                )
        current_dataset_ids = {
            loaded.record.name: loaded.record.dataset_id for loaded in self.loaded_datasets
        }
        freshness_counts = whitelist.validation_freshness_counts(current_dataset_ids)
        if sum(freshness_counts.values()):
            ctx.emit_trace(
                TraceEvent(
                    session_id=ctx.session_id,
                    event_type="join_authorization_freshness",
                    name="discover_questions",
                    finished_at=datetime.now(UTC),
                    summary=freshness_counts,
                )
            )
        confirmed_joins = whitelist.confirmed_labels(current_dataset_ids)

        llm_result = propose_llm_question_candidates(
            source_artifacts,
            llm=self.llm,
            relationship_candidates=relationship_candidates,
            relationship_validations=relationship_validations,
            business_context=self.business_context,
            max_questions=_LLM_PRIMARY_MAX_QUESTIONS,
            payload_policy=self.payload_policy,
            seeds=seeds,
            skills_catalog=catalog_block(project_dir),
            role_sets=role_sets,
            confirmed_joins=confirmed_joins,
            on_guard_rejected=lambda error: ctx.emit_trace(
                TraceEvent(
                    session_id=ctx.session_id,
                    event_type="tool_guard_rejected",
                    name="m4_question_discovery",
                    finished_at=datetime.now(UTC),
                    summary=error.to_trace_summary(),
                )
            ),
        )
        # Persist route repairs and degradation for run-level metrics.
        route_health = {
            "proposals_dropped": llm_result.dropped_proposals,
            "dataset_names_resolved": llm_result.resolved_dataset_names,
            "list_coercions": llm_result.coerced_list_fields,
            "degraded": llm_result.degraded or llm_result.error is not None,
        }
        if llm_result.error is not None:
            ctx.emit_trace(
                TraceEvent(
                    session_id=ctx.session_id,
                    event_type="question_llm_skipped",
                    name="m4_question_discovery",
                    summary={"error": llm_result.error, **route_health},
                )
            )
        elif llm_result.candidates:
            # Attach usage to the existing event to avoid double counting.
            m4_summary: dict[str, Any] = {
                "candidate_count": len(llm_result.candidates),
                **route_health,
            }
            m4_usage = self.llm.last_usage()
            if m4_usage is not None:
                m4_summary.update(
                    {
                        "provider": m4_usage.provider,
                        "model": m4_usage.model,
                        "prompt_tokens": m4_usage.usage.prompt_tokens,
                        "completion_tokens": m4_usage.usage.completion_tokens,
                        "total_tokens": m4_usage.usage.total_tokens,
                        "cached_tokens": m4_usage.usage.cached_tokens,
                        "estimated_cost_usd": m4_usage.estimated_cost_usd,
                    }
                )
                ctx.budget.add_tokens(m4_usage.usage.total_tokens)
            ctx.emit_trace(
                TraceEvent(
                    session_id=ctx.session_id,
                    event_type="llm_call",
                    name="m4_question_discovery",
                    summary=m4_summary,
                )
            )
        # Use templates only for missing coverage when LLM candidates exist.
        candidates = discover_question_candidates(
            self.loaded_datasets,
            profile_artifacts=profile_artifacts,
            quality_artifacts=quality_artifacts,
            analysis_artifacts=analysis_artifacts,
            relationship_candidates=relationship_candidates,
            relationship_validations=relationship_validations,
            llm_candidates=llm_result.candidates,
            include_template_candidates=True,
            template_backstop_only=bool(llm_result.candidates),
            column_role_sets=role_sets,
            join_whitelist=whitelist,
            semantic_seeds=seeds,
        )
        if candidates.template_backstop_used:
            ctx.emit_trace(
                TraceEvent(
                    session_id=ctx.session_id,
                    event_type="template_backstop",
                    name="discover_questions",
                    finished_at=datetime.now(UTC),
                    summary={
                        "backstop_count": candidates.template_backstop_used,
                        "missing_categories": candidates.template_backstop_categories,
                    },
                )
            )
        profiles = [
            DatasetProfile.model_validate(artifact.payload) for artifact in profile_artifacts
        ]
        # Recompute applicability to trace why registered metrics were skipped.
        metric_resolution = applicable_metrics(
            role_sets=role_sets,
            join_whitelist=whitelist,
            profiles=profiles,
            semantic_seeds=seeds,
        )
        if metric_resolution.skipped:
            ctx.emit_trace(
                TraceEvent(
                    session_id=ctx.session_id,
                    event_type="domain_metrics_skipped",
                    name="discover_questions",
                    finished_at=datetime.now(UTC),
                    summary={
                        "resolved_count": len(metric_resolution.resolved),
                        "skipped_count": len(metric_resolution.skipped),
                        "resolved": [metric.metric_id for metric in metric_resolution.resolved],
                        "skipped": {
                            skip.metric_id: skip.reason for skip in metric_resolution.skipped
                        },
                    },
                )
            )
        quality_context_artifacts = [
            ctx.store.get_artifact(
                artifact_id,
                project_id=ctx.project_id,
                session_id=ctx.session_id,
            )
            for artifact_id in self.quality_context_artifact_ids
        ]
        value_map = ValueMap.model_validate(
            ctx.store.get_artifact(
                self.value_map_artifact_id,
                project_id=ctx.project_id,
                session_id=ctx.session_id,
            ).payload
        )
        enriched = enrich_question_candidates(
            [
                _ensure_question_feasibility(candidate, profiles=profiles)
                for candidate in candidates.candidates
            ],
            value_map=value_map,
            quality_context_artifacts=quality_context_artifacts,
        )
        candidates = candidates.model_copy(
            update={
                "candidates": enriched,
                "value_map_artifact_id": self.value_map_artifact_id,
            }
        )
        payload = candidates.model_dump(mode="json")
        parents = [
            self.value_map_artifact_id,
            *self.relationship_artifact_ids,
            *self.analysis_artifact_ids,
            *self.profile_artifact_ids,
            *self.quality_artifact_ids,
            *self.quality_context_artifact_ids,
            *[artifact.id for artifact in role_artifacts],
        ]
        return [
            *role_artifacts,
            Artifact(
                id=make_artifact_id("qcand", payload),
                type=ArtifactType.QUESTION_CANDIDATE_SET,
                project_id=ctx.project_id,
                session_id=ctx.session_id,
                parents=parents,
                payload=payload,
            ),
        ]


def _load_join_whitelist_safe(
    project_dir: Path,
    *,
    session_id: str = "",
    emit: Callable[[TraceEvent], None] | None = None,
) -> JoinWhitelist:
    """Load the project join whitelist, tolerant of a missing/corrupt file.

    An empty whitelist silently blocks every cross-table metric, so a file that
    exists but cannot be read is recorded rather than swallowed. Callers pass an
    emitter rather than a SessionContext: the on-demand relationship path writes
    traces through the store, not the kernel.
    """
    try:
        return load_join_whitelist(project_dir)
    except (OSError, ValueError) as exc:
        if emit is not None:
            emit(
                TraceEvent(
                    session_id=session_id,
                    event_type="join_whitelist_unreadable",
                    name="join_whitelist",
                    summary={
                        "path": str(join_whitelist_path(project_dir)),
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:300],
                        "effect": "cross-table questions run without confirmed joins",
                    },
                )
            )
        return JoinWhitelist()


def _ensure_question_feasibility(
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
    return candidate.model_copy(
        update={
            "candidate_methods": candidate.candidate_methods
            or ([feasibility.method_id] if feasibility.method_id is not None else []),
            "feasibility": feasibility,
            "proposed_action": (
                "design_experiment"
                if candidate.analysis_mode == "causal_experiment"
                else "collect_data"
                if feasibility.status in {"needs_data", "unsuitable"}
                else "run_analysis"
            ),
        }
    )


class ExecuteTopQuestionsStep:
    name: ClassVar[str] = "execute_top_questions"
    requires: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.QUESTION_CANDIDATE_SET,)
    produces: ClassVar[tuple[ArtifactType, ...]] = (
        ArtifactType.SQL_RESULT,
        ArtifactType.QUESTION_EXECUTION_RESULT,
    )

    def __init__(
        self,
        loaded_datasets: Sequence[LoadedDataset],
        *,
        question_candidate_artifact_id: str,
        relationship_artifact_ids: list[str],
        llm: LLMClient | None = None,
    ) -> None:
        self.loaded_datasets = list(loaded_datasets)
        self.question_candidate_artifact_id = question_candidate_artifact_id
        self.relationship_artifact_ids = relationship_artifact_ids
        self.llm = llm

    def run(self, ctx: SessionContext) -> list[Artifact]:
        question_artifact = ctx.store.get_artifact(
            self.question_candidate_artifact_id,
            project_id=ctx.project_id,
            session_id=ctx.session_id,
        )
        candidate_set = QuestionCandidateSet.model_validate(question_artifact.payload)
        selected = select_auto_execution_set(candidate_set)
        ctx.emit_trace(
            TraceEvent(
                session_id=ctx.session_id,
                event_type="question_auto_execution_selected",
                name="execute_top_questions",
                finished_at=datetime.now(UTC),
                summary={
                    **auto_execution_composition(selected),
                    "question_ids": [candidate.question_id for candidate in selected],
                },
            )
        )
        # Thread confirmed joins through every selected execution.
        exec_whitelist = _load_join_whitelist_safe(
            ctx.store.project_dir(ctx.project_id),
            session_id=ctx.session_id,
            emit=ctx.emit_trace,
        )
        current_dataset_ids = {
            loaded.record.name: loaded.record.dataset_id for loaded in self.loaded_datasets
        }
        confirmed_joins = exec_whitelist.confirmed_labels(current_dataset_ids)
        artifacts: list[Artifact] = []
        parent_ids = [self.question_candidate_artifact_id, *self.relationship_artifact_ids]
        for candidate in selected:
            # Disclose machine-confirmed joins in result risks.
            if candidate.required_relations:
                notes = exec_whitelist.disclosure_notes(candidate.required_relations)
                if notes:
                    candidate = candidate.model_copy(update={"risks": [*candidate.risks, *notes]})
            produced = execute_question_candidate(
                candidate,
                datasets=self.loaded_datasets,
                project_id=ctx.project_id,
                session_id=ctx.session_id,
                parent_ids=parent_ids,
                llm=self.llm,
                confirmed_joins=confirmed_joins,
            )
            artifacts.extend(produced)
            qexec = next(
                (
                    artifact
                    for artifact in produced
                    if artifact.type is ArtifactType.QUESTION_EXECUTION_RESULT
                ),
                None,
            )
            ctx.emit_trace(
                TraceEvent(
                    session_id=ctx.session_id,
                    event_type="question_auto_execution",
                    name="execute_top_questions",
                    summary={
                        "question_id": candidate.question_id,
                        "status": qexec.payload["status"] if qexec is not None else "unknown",
                        "outcome": qexec.payload.get("outcome") if qexec is not None else "unknown",
                        "abstention_code": (
                            qexec.payload.get("abstention_code") if qexec is not None else None
                        ),
                        "interpretation_status": (
                            qexec.payload.get("interpretation_status")
                            if qexec is not None
                            else None
                        ),
                        "template_id": candidate.template_id,
                        "metric_id": candidate.metric_id,
                        "exploratory": candidate.exploratory,
                    },
                )
            )
            if qexec is not None and qexec.payload.get("outcome") == "abstained":
                ctx.emit_trace(
                    TraceEvent(
                        session_id=ctx.session_id,
                        event_type="question_result_contract",
                        name="execute_top_questions",
                        summary={
                            "question_id": candidate.question_id,
                            "metric_id": candidate.metric_id,
                            "verdict": "abstained",
                            "code": qexec.payload.get("abstention_code"),
                        },
                    )
                )
        return artifacts


class ExportAgenticReportStep:
    name: ClassVar[str] = "export_agentic_report"
    requires: ClassVar[tuple[ArtifactType, ...]] = ()
    produces: ClassVar[tuple[ArtifactType, ...]] = (
        ArtifactType.SESSION_SUMMARY,
        ArtifactType.REPORT_BUNDLE,
        ArtifactType.REPORT_AUDIT,
        ArtifactType.MARKDOWN_REPORT,
        ArtifactType.HTML_REPORT,
        ArtifactType.EVIDENCE_INTERLEAVE_TRANSCRIPT,
        ArtifactType.SESSION_METRICS,
        ArtifactType.AGENT_HANDOFF,
    )

    def __init__(
        self,
        artifact_ids: list[str],
        *,
        business_context: str,
        llm: LLMClient,
        payload_policy: PayloadPolicy,
        artifact_session_ids: dict[str, str] | None = None,
    ) -> None:
        self.artifact_ids = artifact_ids
        self.artifact_session_ids = artifact_session_ids or {}
        self.business_context = business_context
        self.llm = llm
        self.payload_policy: PayloadPolicy = payload_policy

    def cache_key(self, ctx: SessionContext) -> str:
        """Bind this report's checkpoint to its inputs and LLM config."""
        return stable_hash(
            {
                "artifact_ids": self.artifact_ids,
                "artifact_session_ids": self.artifact_session_ids,
                "business_context": self.business_context,
                "payload_policy": self.payload_policy,
                "llm": llm_execution_fingerprint(self.llm),
            }
        )

    def run(self, ctx: SessionContext) -> list[Artifact]:
        artifacts = [
            ctx.store.get_artifact(
                artifact_id,
                project_id=ctx.project_id,
                session_id=self.artifact_session_ids.get(artifact_id, ctx.session_id),
            )
            for artifact_id in self.artifact_ids
        ]
        report = generate_agentic_report(
            artifacts,
            project_id=ctx.project_id,
            session_id=ctx.session_id,
            business_context=self.business_context,
            llm=self.llm,
            payload_policy=self.payload_policy,
        )
        for event in report.llm_events:
            call = event.usage
            summary = {
                "attempt": event.attempt,
                "status": event.status,
                "error_type": event.error_type,
                "error": event.error,
            }
            if call is not None:
                summary.update(
                    {
                        "provider": call.provider,
                        "model": call.model,
                        "prompt_tokens": call.usage.prompt_tokens,
                        "completion_tokens": call.usage.completion_tokens,
                        "total_tokens": call.usage.total_tokens,
                        "cached_tokens": call.usage.cached_tokens,
                        "estimated_cost_usd": call.estimated_cost_usd,
                    }
                )
            ctx.emit_trace(
                TraceEvent(
                    session_id=ctx.session_id,
                    event_type="llm_call" if event.status == "success" else "llm_error",
                    name=event.task,
                    started_at=event.started_at,
                    finished_at=event.finished_at,
                    summary=summary,
                ),
            )
            if call is not None:
                ctx.budget.add_tokens(call.usage.total_tokens)
        for event in report.validation_events:
            ctx.emit_trace(
                TraceEvent(
                    session_id=ctx.session_id,
                    event_type="report_validation",
                    name="m2_report_validator",
                    summary={
                        "attempt": event.attempt,
                        "status": event.status,
                        "finding_count": event.finding_count,
                        "critical_count": event.critical_count,
                        "normalized_body_count": event.normalized_body_count,
                        "pruned_claim_count": event.pruned_claim_count,
                        "deterministic_repair_count": event.deterministic_repair_count,
                        "dropped_focus_claim_count": event.dropped_focus_claim_count,
                        "section_coverage": event.section_coverage,
                        "claim_section_coverage": event.claim_section_coverage,
                        "claim_survival_rate": event.claim_survival_rate,
                        "budget_stopped": event.budget_stopped,
                        "selected_attempt": event.selected_attempt,
                        "findings": event.findings,
                        "structured_findings": event.structured_findings,
                    },
                ),
            )
        return build_agentic_report_artifacts(
            report,
            artifacts,
            project_id=ctx.project_id,
            session_id=ctx.session_id,
            payload_policy=self.payload_policy,
        )


def _retag_raw_artifact(
    artifact: Artifact,
    artifact_type: ArtifactType,
    id_prefix: str,
    *,
    parents: list[str] | None = None,
) -> Artifact:
    payload = dict(artifact.payload)
    return Artifact(
        id=make_artifact_id(id_prefix, payload),
        type=artifact_type,
        project_id=artifact.project_id,
        session_id=artifact.session_id,
        parents=list(artifact.parents if parents is None else parents),
        payload=payload,
        warnings=["before_cleaning", *artifact.warnings],
        evidence=list(artifact.evidence),
    )


def _raw_preview_payload(loaded: LoadedDataset, *, limit: int = 20) -> dict[str, Any]:
    frame = loaded.frame
    preview = frame.head(limit)
    records = [
        {str(key): _json_safe_value(value) for key, value in row.items()}
        for row in preview.to_dict("records")
    ]
    return {
        "dataset_id": loaded.record.dataset_id,
        "name": loaded.record.name,
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "column_names": [str(column) for column in frame.columns],
        "rows_preview": records,
        "preview_row_limit": limit,
    }


def _masked_loaded_dataset(
    loaded: LoadedDataset,
    labels: Mapping[str, str],
) -> LoadedDataset:
    if not labels:
        return loaded
    frame = loaded.frame.copy()
    for column, label in labels.items():
        if column in frame.columns:
            frame[column] = frame[column].map(
                lambda value, pii_label=label: value if pd.isna(value) else f"[PII:{pii_label}]"
            )
    return LoadedDataset(record=loaded.record, frame=frame)


def _json_safe_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            return value
    return value


def run_auto_eda(
    file_paths: Sequence[Path | str],
    *,
    workspace: Path | str | None = None,
    project_id: str = "default",
    session_id: str | None = None,
    business_context: str = "",
    llm: LLMClient | None = None,
    payload_policy: PayloadPolicy = "schema+aggregates",
    on_trace_event: Callable[[TraceEvent], None] | None = None,
    ml_target_column: str | None = None,
    ml_time_column: str | None = None,
    precleaning: Sequence[CleaningRecipe | None] | None = None,
    raw_file_paths: Sequence[Path | str] | None = None,
    relationship_discovery: Literal["on_demand", "eager"] = "on_demand",
    generate_report: bool = True,
    dataset_workers: int = 1,
    resource_policy: EdaResourcePolicy | None = None,
    resource_preflight: EdaResourcePreflight | None = None,
    preprocessing_duration_seconds: float = 0.0,
    baseline_peak_rss: PeakRssMeasurement | None = None,
    budget_policy: SessionBudgetPolicy | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> AutoEDAResult:
    if dataset_workers not in {1, 2}:
        raise ValueError("dataset_workers must be 1 or 2")
    driver_started = perf_counter()
    baseline_rss = baseline_peak_rss or process_peak_rss()
    workspace_path = resolve_workspace_path(workspace)
    store = ArtifactStore(workspace_path)
    actual_session_id = session_id or _generate_session_id(file_paths)
    previous_manifest = store.read_manifest(project_id, actual_session_id)
    effective_budget_policy = budget_policy or SessionBudgetPolicy()
    restored_budget = restore_run_budget_state(
        effective_budget_policy,
        store.list_trace_events(project_id=project_id, session_id=actual_session_id),
    )
    ctx = SessionContext(
        project_id=project_id,
        session_id=actual_session_id,
        store=store,
        budget_policy=effective_budget_policy,
        restored_session_budget=restored_budget,
        on_trace_event=on_trace_event,
        cancel_check=cancel_check,
    )
    # Ledger first: every downstream LLM call is metered at this one seam, so a
    # call site that forgets its own llm_call event still shows up in the spend.
    base_llm = llm or OfflineLLMClient()
    llm = meter_llm_client(
        base_llm,
        session_id=actual_session_id,
        emit=ctx.emit_trace,
        budget=ctx.session_budget,
        session_dir=store.session_dir(project_id, actual_session_id),
    )

    run_started_at = datetime.now(UTC)
    preliminary_manifest = SessionManifest(
        session_id=actual_session_id,
        project_id=project_id,
        input_hashes={},
        code_version=_current_code_version(),
        model_versions=manifest_model_versions(llm),
        title=build_run_title([Path(path).name for path in file_paths]) or None,
    )
    store.write_manifest(preliminary_manifest)
    cooperative_cancel = (
        None
        if cancel_check is None
        else lambda: raise_if_cancelled(cancel_check, operation="resource preflight")
    )
    effective_resource_policy = resource_policy or (
        resource_preflight.policy if resource_preflight is not None else EdaResourcePolicy()
    )
    preflight = resource_preflight or preflight_csv_resources(
        file_paths,
        requested_dataset_workers=dataset_workers,
        baseline_peak_rss_bytes=baseline_rss.bytes or 0,
        policy=effective_resource_policy,
        precleaning_enabled=raw_file_paths is not None,
        cancel_check=cooperative_cancel,
    )
    if preflight.requested_dataset_workers != dataset_workers:
        raise ValueError("resource_preflight requested workers do not match dataset_workers")
    preflight_artifact = _resource_preflight_artifact(
        preflight, project_id=project_id, session_id=actual_session_id
    )
    store.save_artifact(preflight_artifact)
    ctx.execution_fingerprint = stable_hash(
        {
            "execution_schema_version": 2,
            "resource_preflight": preflight.model_dump(mode="json"),
            "code_version": preliminary_manifest.code_version,
            "payload_policy": payload_policy,
            "business_context": business_context,
        },
        length=32,
    )
    if preflight.status == "rejected":
        _emit_auto_eda_resource_trace(
            ctx,
            driver_started=driver_started,
            baseline_rss=baseline_rss,
            preprocessing_duration_seconds=preprocessing_duration_seconds,
        )
        store.mark_session_status(project_id, actual_session_id, "failed")
        raise EdaResourceLimitError(preflight)
    if preflight.status == "limited":
        _emit_auto_eda_resource_trace(
            ctx,
            driver_started=driver_started,
            baseline_rss=baseline_rss,
            preprocessing_duration_seconds=preprocessing_duration_seconds,
        )
        final_artifacts = _finalize_agent_artifacts(
            store,
            project_id=project_id,
            session_id=actual_session_id,
            manifest=preliminary_manifest,
            execution_fingerprint=ctx.execution_fingerprint,
            emit_trace=ctx.emit_trace,
        )
        store.mark_session_status(project_id, actual_session_id, LIMITED_SESSION_STATUS)
        return AutoEDAResult(
            project_id=project_id,
            session_id=actual_session_id,
            business_context=business_context,
            artifacts=final_artifacts,
            report_markdown="",
            workspace=workspace_path,
            loaded_datasets=[],
        )

    ingest_started = perf_counter()
    loaded_datasets = [
        _copy_and_load_upload(
            Path(file_path), workspace_path, project_id, cancel_check=cancel_check
        )
        for file_path in file_paths
    ]
    raw_loaded_datasets: list[LoadedDataset] = []
    if raw_file_paths is not None:
        raw_paths = list(raw_file_paths)
        if len(raw_paths) != len(loaded_datasets):
            raise ValueError(
                "raw_file_paths must align with file_paths: "
                f"got {len(raw_paths)} raw path(s) for {len(loaded_datasets)} dataset(s)."
            )
        raw_loaded_datasets = [
            _copy_and_load_upload(
                Path(file_path), workspace_path, project_id, cancel_check=cancel_check
            )
            for file_path in raw_paths
        ]
    analysis_frame_bytes = [
        int(loaded.frame.memory_usage(index=True, deep=True).sum()) for loaded in loaded_datasets
    ]
    raw_frame_bytes = [
        int(loaded.frame.memory_usage(index=True, deep=True).sum())
        for loaded in raw_loaded_datasets
    ]
    preflight = _verify_resource_preflight(
        preflight,
        loaded_datasets,
        raw_loaded_datasets,
        analysis_frame_bytes=analysis_frame_bytes,
        raw_frame_bytes=raw_frame_bytes,
    )
    manifest = SessionManifest(
        session_id=actual_session_id,
        project_id=project_id,
        input_hashes={loaded.record.name: loaded.record.content_hash for loaded in loaded_datasets},
        code_version=_current_code_version(),
        model_versions=manifest_model_versions(llm),
        title=build_run_title([loaded.record.name for loaded in loaded_datasets]) or None,
    )
    if previous_manifest is not None and previous_manifest.input_hashes != manifest.input_hashes:
        store.reset_session_outputs(project_id=project_id, session_id=actual_session_id)
        ctx.session_budget = restore_run_budget_state(effective_budget_policy, [])
        llm = meter_llm_client(
            base_llm,
            session_id=actual_session_id,
            emit=ctx.emit_trace,
            budget=ctx.session_budget,
            session_dir=store.session_dir(project_id, actual_session_id),
        )
    store.save_artifact(
        _resource_preflight_artifact(preflight, project_id=project_id, session_id=actual_session_id)
    )
    store.write_manifest(manifest)
    ctx.emit_trace(
        TraceEvent(
            session_id=actual_session_id,
            event_type="eda_inputs_loaded",
            name="load_inputs",
            started_at=run_started_at,
            finished_at=datetime.now(UTC),
            summary={
                "ingest_duration_seconds": round(perf_counter() - ingest_started, 6),
                "analysis": _resource_footprint(loaded_datasets, analysis_frame_bytes).model_dump(
                    mode="json"
                ),
                "raw_lineage": _resource_footprint(raw_loaded_datasets, raw_frame_bytes).model_dump(
                    mode="json"
                ),
                "unique_file_bytes": _unique_loaded_file_bytes(
                    [*loaded_datasets, *raw_loaded_datasets]
                ),
            },
        )
    )
    if preflight.status != "accepted":
        _emit_auto_eda_resource_trace(
            ctx,
            driver_started=driver_started,
            baseline_rss=baseline_rss,
            preprocessing_duration_seconds=preprocessing_duration_seconds,
        )
        if preflight.status == "rejected":
            store.mark_session_status(project_id, actual_session_id, "failed")
            raise EdaResourceLimitError(preflight)
        final_artifacts = _finalize_agent_artifacts(
            store,
            project_id=project_id,
            session_id=actual_session_id,
            manifest=manifest,
            execution_fingerprint=ctx.execution_fingerprint,
            emit_trace=ctx.emit_trace,
        )
        store.mark_session_status(project_id, actual_session_id, LIMITED_SESSION_STATUS)
        return AutoEDAResult(
            project_id=project_id,
            session_id=actual_session_id,
            business_context=business_context,
            artifacts=final_artifacts,
            report_markdown="",
            workspace=workspace_path,
            loaded_datasets=[],
        )
    effective_dataset_workers = preflight.effective_dataset_workers
    recipes = _align_precleaning(precleaning, len(loaded_datasets))
    ctx.execution_fingerprint = stable_hash(
        {
            "execution_schema_version": 1,
            "inputs": [
                {
                    "name": loaded.record.name,
                    "content_hash": loaded.record.content_hash,
                }
                for loaded in loaded_datasets
            ],
            "raw_inputs": [
                {
                    "name": loaded.record.name,
                    "content_hash": loaded.record.content_hash,
                }
                for loaded in raw_loaded_datasets
            ],
            "code_version": manifest.code_version,
            "llm": llm_execution_fingerprint(llm),
            "payload_policy": payload_policy,
            "business_context": business_context,
            "ml_target_column": ml_target_column,
            "ml_time_column": ml_time_column,
            "precleaning": [
                (
                    None
                    if recipe is None
                    else recipe.model_dump(
                        mode="json",
                        exclude={"recipe_id", "lineage"},
                    )
                )
                for recipe in recipes
            ],
            "relationship_discovery": relationship_discovery,
            "generate_report": generate_report,
            "requested_dataset_workers": dataset_workers,
            "effective_dataset_workers": effective_dataset_workers,
            "resource_policy": preflight.policy.model_dump(mode="json"),
        },
        length=32,
    )

    # Pre-cleaning (opt-in, done before ingest) may have dropped columns/rows.
    # Emit each drop as a CleaningRecipe artifact and make it the lineage parent
    # of the corresponding cleaned dataset's profile, so the evidence chain
    # records what was removed and why.
    raw_artifacts: list[Artifact] = []
    if raw_loaded_datasets:
        raw_result = run_pipeline(
            [RecordRawDatasetStep(loaded) for loaded in raw_loaded_datasets],
            ctx,
        )
        raw_artifacts = raw_result.artifacts
    cleaning_recipe_result = run_pipeline(
        [EmitCleaningRecipeStep(recipe) for recipe in recipes if recipe is not None],
        ctx,
    )
    recipe_artifacts = iter(cleaning_recipe_result.artifacts)
    profile_parent_ids: list[list[str]] = [
        [next(recipe_artifacts).id] if recipe is not None else [] for recipe in recipes
    ]

    profile_result = run_pipeline(
        [
            ProfileDatasetStep(loaded, parent_ids=parent_ids)
            for loaded, parent_ids in zip(loaded_datasets, profile_parent_ids, strict=True)
        ],
        ctx,
        max_workers=effective_dataset_workers,
    )
    profile_artifacts = [
        artifact
        for artifact in profile_result.artifacts
        if artifact.type is ArtifactType.DATASET_PROFILE
    ]
    profile_ids = [artifact.id for artifact in profile_artifacts]

    quality_result = run_pipeline(
        [ScanQualityStep(artifact_id) for artifact_id in profile_ids],
        ctx,
        max_workers=effective_dataset_workers,
    )
    quality_ids = [artifact.id for artifact in quality_result.artifacts]
    quality_context_result = run_pipeline(
        [
            BuildQualityContextStep(loaded, profile_id, quality_id)
            for loaded, profile_id, quality_id in zip(
                loaded_datasets, profile_ids, quality_ids, strict=True
            )
        ],
        ctx,
        max_workers=effective_dataset_workers,
    )
    quality_context_ids = [artifact.id for artifact in quality_context_result.artifacts]
    chart_result = run_pipeline(
        [
            CreateChartSpecsStep(loaded, artifact_id)
            for loaded, artifact_id in zip(loaded_datasets, profile_ids, strict=True)
        ],
        ctx,
        max_workers=effective_dataset_workers,
    )
    analysis_result = run_pipeline(
        [
            CreateAnalysisTablesStep(loaded, artifact_id)
            for loaded, artifact_id in zip(loaded_datasets, profile_ids, strict=True)
        ],
        ctx,
        max_workers=effective_dataset_workers,
    )
    # W-2 revised: a Bonferroni family is a set of tests serving one inferential
    # question. Auto-EDA runs at most one test per dataset, and tests on
    # unrelated datasets are not a family, so no cross-dataset adjustment is
    # applied; every auto result instead carries an explicit
    # exploratory_auto_selection warning.
    profiles = [DatasetProfile.model_validate(artifact.payload) for artifact in profile_artifacts]
    stat_specs = [
        _select_stat_test(loaded, profile)
        for loaded, profile in zip(loaded_datasets, profiles, strict=True)
    ]
    stat_plan = [
        (loaded, artifact_id, spec)
        for loaded, artifact_id, spec in zip(loaded_datasets, profile_ids, stat_specs, strict=True)
        if spec is not None
    ]
    stat_result = run_pipeline(
        [
            SessionStatTestsStep(loaded, artifact_id, spec)
            for loaded, artifact_id, spec in stat_plan
        ],
        ctx,
    )
    model_artifacts: list[Artifact] = []
    if ml_target_column:
        # W-3: only model datasets that actually contain the target column, rather
        # than blindly applying one target name to every table.
        model_result = run_pipeline(
            [
                SessionBaselineModelStep(
                    loaded,
                    artifact_id,
                    target_column=ml_target_column,
                    time_column=ml_time_column,
                )
                for loaded, artifact_id in zip(loaded_datasets, profile_ids, strict=True)
                if ml_target_column in loaded.frame.columns
            ],
            ctx,
        )
        model_artifacts = model_result.artifacts
    relationship_artifacts: list[Artifact] = []
    if len(loaded_datasets) >= 2 and relationship_discovery == "eager":
        relationship_result = run_pipeline(
            [DiscoverRelationshipsStep(loaded_datasets, profile_ids)],
            ctx,
        )
        relationship_artifacts = relationship_result.artifacts
    elif len(loaded_datasets) >= 2:
        ctx.emit_trace(
            TraceEvent(
                session_id=actual_session_id,
                event_type="relationship_discovery_deferred",
                name="discover_relationships",
                finished_at=datetime.now(UTC),
                summary={
                    "dataset_count": len(loaded_datasets),
                    "reason": "default_on_demand_policy",
                    "trigger": "relationships_on_demand",
                },
            )
        )

    core_eda_artifacts = [
        *raw_artifacts,
        *cleaning_recipe_result.artifacts,
        *profile_result.artifacts,
        *quality_result.artifacts,
        *quality_context_result.artifacts,
        *chart_result.artifacts,
        *analysis_result.artifacts,
        *stat_result.artifacts,
        *model_artifacts,
        *relationship_artifacts,
    ]
    handoff_artifact = create_eda_handoff_artifact(
        core_eda_artifacts,
        project_id=project_id,
        session_id=actual_session_id,
        raw_dataset_lineage={
            raw.record.dataset_id: clean.record.dataset_id
            for raw, clean in zip(raw_loaded_datasets, loaded_datasets, strict=False)
        },
    )
    store.save_artifact(handoff_artifact)
    ctx.emit_trace(
        TraceEvent(
            session_id=actual_session_id,
            event_type="eda_handoff_ready",
            name="create_eda_handoff",
            finished_at=datetime.now(UTC),
            summary={
                "dataset_count": len(profile_ids),
                "indexed_artifact_count": len(core_eda_artifacts),
            },
        )
    )

    value_map_source_ids = [
        *profile_ids,
        *quality_ids,
        *quality_context_ids,
        handoff_artifact.id,
    ]
    value_map_result = run_pipeline(
        [
            BuildValueMapStep(
                source_artifact_ids=value_map_source_ids,
                business_context=business_context,
            )
        ],
        ctx,
    )
    value_map_artifact_id = value_map_result.artifacts[0].id
    question_result = run_pipeline(
        [
            DiscoverQuestionsStep(
                loaded_datasets,
                profile_artifact_ids=profile_ids,
                quality_artifact_ids=quality_ids,
                quality_context_artifact_ids=quality_context_ids,
                analysis_artifact_ids=[
                    artifact.id
                    for artifact in [
                        *analysis_result.artifacts,
                        *stat_result.artifacts,
                        *model_artifacts,
                    ]
                    if artifact.type is not ArtifactType.CHART_SPEC
                ],
                relationship_artifact_ids=[artifact.id for artifact in relationship_artifacts],
                value_map_artifact_id=value_map_artifact_id,
                llm=llm,
                business_context=business_context,
                payload_policy=payload_policy,
            )
        ],
        ctx,
    )
    # DiscoverQuestionsStep also emits COLUMN_ROLE_SET artifacts;
    # locate the candidate set by type instead of assuming position 0.
    question_candidate_artifact = next(
        artifact
        for artifact in question_result.artifacts
        if artifact.type is ArtifactType.QUESTION_CANDIDATE_SET
    )
    question_execution_result = run_pipeline(
        [
            ExecuteTopQuestionsStep(
                loaded_datasets,
                question_candidate_artifact_id=question_candidate_artifact.id,
                relationship_artifact_ids=[artifact.id for artifact in relationship_artifacts],
                llm=llm,
            )
        ],
        ctx,
    )

    report_parent_ids = [
        artifact.id
        for artifact in [
            *profile_result.artifacts,
            *quality_result.artifacts,
            *quality_context_result.artifacts,
            *chart_result.artifacts,
            *analysis_result.artifacts,
            *stat_result.artifacts,
            *model_artifacts,
            *relationship_artifacts,
            *value_map_result.artifacts,
            *question_result.artifacts,
            *question_execution_result.artifacts,
            handoff_artifact,
        ]
    ]
    report_result = (
        run_pipeline(
            [
                ExportAgenticReportStep(
                    report_parent_ids,
                    business_context=business_context,
                    llm=llm,
                    payload_policy=payload_policy,
                )
            ],
            ctx,
        )
        if generate_report
        else None
    )
    report_markdown = ""
    if report_result is not None:
        report_artifact = next(
            artifact
            for artifact in report_result.artifacts
            if artifact.type is ArtifactType.MARKDOWN_REPORT
        )
        report_markdown = str(report_artifact.payload["markdown"])
        store.write_session_text(
            project_id,
            actual_session_id,
            "report/report.md",
            report_markdown,
        )
        html_artifact = next(
            (
                artifact
                for artifact in report_result.artifacts
                if artifact.type is ArtifactType.HTML_REPORT
            ),
            None,
        )
        if html_artifact:
            store.write_session_text(
                project_id,
                actual_session_id,
                "report/report.html",
                str(html_artifact.payload["html"]),
            )

    # Cosmetic LLM title upgrade (never a number source): one cheap text call
    # over dataset names + business context + executive-summary claim texts.
    # Any failure keeps the deterministic title already in the manifest.
    llm_title = _llm_session_title(
        ctx,
        llm,
        dataset_names=[loaded.record.name for loaded in loaded_datasets],
        business_context=business_context,
        report_artifacts=report_result.artifacts if report_result is not None else [],
    )
    if llm_title:
        manifest = manifest.model_copy(update={"title": llm_title})
        store.write_manifest(manifest)

    _emit_auto_eda_resource_trace(
        ctx,
        driver_started=driver_started,
        baseline_rss=baseline_rss,
        preprocessing_duration_seconds=preprocessing_duration_seconds,
    )

    # The final AgentHandoff is a publication contract, not observability-only
    # metadata. Persist it (and refresh metrics around it) before the completed
    # status becomes the API publication barrier.
    final_artifacts = _finalize_agent_artifacts(
        store,
        project_id=project_id,
        session_id=actual_session_id,
        manifest=manifest,
        execution_fingerprint=ctx.execution_fingerprint,
        emit_trace=ctx.emit_trace,
    )
    store.mark_session_status(project_id, actual_session_id, "completed")

    return AutoEDAResult(
        project_id=project_id,
        session_id=actual_session_id,
        business_context=business_context,
        artifacts=final_artifacts,
        report_markdown=report_markdown,
        workspace=workspace_path,
        loaded_datasets=loaded_datasets,
    )


def generate_report_on_demand(
    result: AutoEDAResult,
    *,
    llm: LLMClient | None = None,
    payload_policy: PayloadPolicy = "schema+aggregates",
    budget_policy: SessionBudgetPolicy | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> AutoEDAResult:
    """Generate and persist the final report after question review has begun."""
    raise_if_cancelled(cancel_check, operation="report generation")
    store = ArtifactStore(result.workspace)
    effective_budget_policy = budget_policy or SessionBudgetPolicy()
    restored_budget = restore_run_budget_state(
        effective_budget_policy,
        store.list_trace_events(project_id=result.project_id, session_id=result.session_id),
    )
    ctx = SessionContext(
        project_id=result.project_id,
        session_id=result.session_id,
        store=store,
        budget_policy=effective_budget_policy,
        restored_session_budget=restored_budget,
    )
    run_llm = meter_llm_client(
        llm or OfflineLLMClient(),
        session_id=result.session_id,
        emit=ctx.emit_trace,
        budget=ctx.session_budget,
        session_dir=store.session_dir(result.project_id, result.session_id),
    )
    report_types = {
        ArtifactType.SESSION_SUMMARY,
        ArtifactType.REPORT_BUNDLE,
        ArtifactType.REPORT_AUDIT,
        ArtifactType.MARKDOWN_REPORT,
        ArtifactType.HTML_REPORT,
        ArtifactType.EVIDENCE_INTERLEAVE_TRANSCRIPT,
    }
    source_artifacts = [
        artifact for artifact in result.artifacts if artifact.type not in report_types
    ]
    source_ids = {artifact.id for artifact in source_artifacts}
    related: dict[str, Artifact] = {artifact.id: artifact for artifact in source_artifacts}
    project_artifacts: list[Artifact] = []
    for run in store.list_sessions(result.project_id):
        raise_if_cancelled(cancel_check, operation="report generation")
        if run.session_id == result.session_id:
            continue
        try:
            manifest = store.read_manifest(result.project_id, run.session_id)
        except (OSError, ValueError):
            manifest = None
        if manifest is None or manifest.source_session_id != result.session_id:
            continue
        artifacts, _warnings = store.list_artifacts_safe(
            project_id=result.project_id, session_id=run.session_id
        )
        project_artifacts.extend(
            artifact for artifact in artifacts if artifact.type not in report_types
        )
    changed = True
    while changed:
        raise_if_cancelled(cancel_check, operation="report generation")
        changed = False
        related_ids = source_ids | set(related)
        for artifact in project_artifacts:
            if artifact.id in related or not (set(artifact.parents) & related_ids):
                continue
            related[artifact.id] = artifact
            changed = True
    source_artifacts = list(related.values())
    raise_if_cancelled(cancel_check, operation="report generation")
    generated = run_pipeline(
        [
            ExportAgenticReportStep(
                [artifact.id for artifact in source_artifacts],
                business_context=result.business_context,
                llm=run_llm,
                payload_policy=payload_policy,
                artifact_session_ids={
                    artifact.id: artifact.session_id for artifact in source_artifacts
                },
            )
        ],
        ctx,
    ).artifacts
    raise_if_cancelled(cancel_check, operation="report generation")
    markdown_artifact = next(
        artifact for artifact in generated if artifact.type is ArtifactType.MARKDOWN_REPORT
    )
    markdown = str(markdown_artifact.payload["markdown"])
    store.write_session_text(
        result.project_id,
        result.session_id,
        "report/report.md",
        markdown,
    )
    html_artifact = next(
        (artifact for artifact in generated if artifact.type is ArtifactType.HTML_REPORT), None
    )
    if html_artifact is not None:
        store.write_session_text(
            result.project_id,
            result.session_id,
            "report/report.html",
            str(html_artifact.payload["html"]),
        )
    manifest = store.read_manifest(result.project_id, result.session_id)
    if manifest is None:
        raise ValueError("Cannot finalize agent handoff without a session manifest.")
    final_artifacts = _finalize_agent_artifacts(
        store,
        project_id=result.project_id,
        session_id=result.session_id,
        manifest=manifest,
        execution_fingerprint=None,
        referenced_external_artifacts=[
            artifact for artifact in source_artifacts if artifact.session_id != result.session_id
        ],
    )
    return replace(result, artifacts=final_artifacts, report_markdown=markdown)


def _finalize_agent_artifacts(
    store: ArtifactStore,
    *,
    project_id: str,
    session_id: str,
    manifest: SessionManifest,
    execution_fingerprint: str | None,
    emit_trace: Callable[[TraceEvent], None] | None = None,
    referenced_external_artifacts: Sequence[Artifact] = (),
) -> list[Artifact]:
    """Build metrics + handoff from one inventory, then publish each once."""
    persisted = store.list_artifacts(project_id=project_id, session_id=session_id)
    base = [
        artifact
        for artifact in persisted
        if artifact.type not in {ArtifactType.SESSION_METRICS, ArtifactType.AGENT_HANDOFF}
    ]
    effective_fingerprint = execution_fingerprint or _stored_execution_fingerprint(persisted)
    metrics_placeholder = Artifact(
        id=make_artifact_id(
            "session_metrics", {"project_id": project_id, "session_id": session_id}
        ),
        type=ArtifactType.SESSION_METRICS,
        project_id=project_id,
        session_id=session_id,
        payload={},
    )
    handoff_placeholder = Artifact(
        id=make_artifact_id("agent_handoff", {"project_id": project_id, "session_id": session_id}),
        type=ArtifactType.AGENT_HANDOFF,
        project_id=project_id,
        session_id=session_id,
        payload={},
    )
    external = {
        artifact.id: artifact
        for artifact in referenced_external_artifacts
        if artifact.session_id != session_id
        and artifact.type not in {ArtifactType.SESSION_METRICS, ArtifactType.AGENT_HANDOFF}
    }
    generated_at = datetime.now(UTC)
    runtime_env_digest = env_digest()
    handoff = handoff_placeholder
    metrics = metrics_placeholder
    previous_sizes: tuple[int, int, int, int] | None = None
    # Resolve the size/context fixed point in memory. No provisional handoff is
    # externally readable and a metrics failure remains a publication failure.
    for _ in range(4):
        metrics = create_run_metrics_artifact(
            store,
            project_id,
            session_id,
            artifact_snapshot=[*base, metrics_placeholder, handoff],
        )
        metrics.env_digest = runtime_env_digest
        handoff = create_agent_handoff_artifact(
            [*base, metrics],
            external_artifacts=list(external.values()),
            fetch_session_id=session_id,
            project_id=project_id,
            session_id=session_id,
            producer_version=manifest.code_version,
            execution_fingerprint=effective_fingerprint,
            input_hashes=manifest.input_hashes,
            generated_at=generated_at,
        )
        handoff.env_digest = runtime_env_digest
        context = handoff.payload.get("context_policy", {})
        sizes = (
            len(metrics.model_dump_json().encode("utf-8")),
            len(handoff.model_dump_json().encode("utf-8")),
            int(context.get("serialized_bytes", 0)),
            int(context.get("initial_context_bytes", 0)),
        )
        if sizes == previous_sizes:
            break
        previous_sizes = sizes
    store.save_artifact(metrics)
    store.save_artifact(handoff)
    event = TraceEvent(
        session_id=session_id,
        event_type="agent_handoff_ready",
        name="create_agent_handoff",
        finished_at=datetime.now(UTC),
        summary={
            "artifact_id": handoff.id,
            "source_artifact_count": len(base) + 1,
            "referenced_external_artifact_count": len(external),
            "persisted_artifact_count": len(base) + 2,
            "contract_version": "3.0",
        },
    )
    if emit_trace is not None:
        emit_trace(event)
    else:
        store.append_trace(project_id, event)
    published = store.list_artifacts(project_id=project_id, session_id=session_id)
    published_ids = {artifact.id for artifact in published}
    return [
        *(artifact for artifact in published if artifact.type is ArtifactType.CLEANING_RECIPE),
        *(artifact for artifact in published if artifact.type is not ArtifactType.CLEANING_RECIPE),
        *(artifact for aid, artifact in external.items() if aid not in published_ids),
    ]


def _persist_run_metrics_best_effort(
    store: ArtifactStore,
    project_id: str,
    session_id: str,
) -> None:
    """Refresh the coherent metrics + handoff publication after source mutations."""
    legacy_session = False
    try:
        manifest = store.read_manifest(project_id, session_id)
        if manifest is None:
            # A legacy/manual source session has no final handoff manifest.
            # Retain the original best-effort observability refresh.
            legacy_session = True
            persist_run_metrics(store, project_id, session_id)
            return
        _finalize_agent_artifacts(
            store,
            project_id=project_id,
            session_id=session_id,
            manifest=manifest,
            execution_fingerprint=None,
        )
    except Exception as exc:  # noqa: BLE001 - refresh must not hide completed work
        event_type = (
            "session_metrics_error" if legacy_session else "agent_publication_refresh_error"
        )
        event_name = (
            "persist_run_metrics" if legacy_session else "refresh_session_metrics_and_agent_handoff"
        )
        store.append_trace(
            project_id,
            TraceEvent(
                session_id=session_id,
                event_type=event_type,
                name=event_name,
                finished_at=datetime.now(UTC),
                summary={
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                },
            ),
        )


def _stored_execution_fingerprint(artifacts: Sequence[Artifact]) -> str:
    for artifact in reversed(artifacts):
        if artifact.type is not ArtifactType.AGENT_HANDOFF:
            continue
        run = artifact.payload.get("run")
        if isinstance(run, dict) and isinstance(run.get("execution_fingerprint"), str):
            return run["execution_fingerprint"]
    return "unavailable"


def _resource_preflight_artifact(
    decision: EdaResourcePreflight,
    *,
    project_id: str,
    session_id: str,
) -> Artifact:
    return Artifact(
        id=make_artifact_id(
            "resource_preflight", {"project_id": project_id, "session_id": session_id}
        ),
        type=ArtifactType.RESOURCE_PREFLIGHT,
        project_id=project_id,
        session_id=session_id,
        payload=decision.model_dump(mode="json"),
        plain_language=(
            "Bounded resource decision made before expensive full-frame Auto-EDA work."
        ),
    )


def _verify_resource_preflight(
    decision: EdaResourcePreflight,
    analysis: Sequence[LoadedDataset],
    raw_lineage: Sequence[LoadedDataset],
    *,
    analysis_frame_bytes: Sequence[int],
    raw_frame_bytes: Sequence[int],
) -> EdaResourcePreflight:
    policy = decision.policy
    verified_estimates: list[EdaDatasetEstimate] = []
    for index, (loaded, deep_bytes) in enumerate(zip(analysis, analysis_frame_bytes, strict=True)):
        source = decision.datasets[index] if index < len(decision.datasets) else None
        updates = {
            "role": "analysis",
            "name": loaded.record.name,
            "file_bytes": loaded.record.path.stat().st_size,
            "columns": len(loaded.frame.columns),
            "exact_rows": len(loaded.frame),
            "exact_frame_deep_bytes": deep_bytes,
        }
        if source is not None:
            verified_estimates.append(source.model_copy(update=updates))
        else:
            verified_estimates.append(
                EdaDatasetEstimate(
                    **updates,
                    sample_rows=0,
                    sample_frame_deep_bytes=0,
                    sample_serialized_bytes=0,
                    frame_expansion_ratio=0.0,
                    estimated_rows=len(loaded.frame),
                    estimated_frame_deep_bytes=deep_bytes,
                )
            )

    def exact_working_set(workers: int) -> int:
        active_analysis = sum(sorted(analysis_frame_bytes, reverse=True)[:workers])
        active_raw = sum(sorted(raw_frame_bytes, reverse=True)[:workers])
        retained = sum(analysis_frame_bytes) + sum(raw_frame_bytes)
        return ceil(
            decision.baseline_peak_rss_bytes
            + policy.held_frame_multiplier * retained
            + policy.active_frame_multiplier * max(active_analysis, active_raw)
        )

    workers = min(
        decision.effective_dataset_workers,
        decision.requested_dataset_workers,
        policy.max_dataset_workers,
        max(1, len(analysis)),
    )
    worker_reason = decision.worker_adjustment_reason
    verified_working_set = exact_working_set(workers)
    if workers > 1 and verified_working_set > policy.max_working_set_bytes:
        single_worker = exact_working_set(1)
        if single_worker <= policy.max_working_set_bytes:
            workers = 1
            verified_working_set = single_worker
            worker_reason = "memory_budget_worker_downgrade"

    all_loaded = [*analysis, *raw_lineage]
    all_frame_bytes = [*analysis_frame_bytes, *raw_frame_bytes]
    file_sizes = [loaded.record.path.stat().st_size for loaded in all_loaded]
    reasons: list[str] = []
    if any(size > policy.max_single_input_bytes for size in file_sizes):
        reasons.append("single_input_bytes_exceeded")
    if _unique_loaded_file_bytes(all_loaded) > policy.max_input_bytes_total:
        reasons.append("input_bytes_total_exceeded")
    if any(len(loaded.frame.columns) > policy.max_columns_per_dataset for loaded in all_loaded):
        reasons.append("column_count_exceeded")
    if any(len(loaded.frame) > policy.max_rows_per_dataset for loaded in all_loaded):
        reasons.append("row_count_exceeded")
    if verified_working_set > policy.max_working_set_bytes:
        reasons.append("verified_working_set_exceeded")
    if worker_reason is not None:
        reasons.append(worker_reason)
    status = (
        "accepted"
        if not [
            reason
            for reason in reasons
            if reason not in {"memory_budget_worker_downgrade", "dataset_or_policy_worker_cap"}
        ]
        else "limited"
        if policy.on_exceed == "limited"
        else "rejected"
    )
    return decision.model_copy(
        update={
            "status": status,
            "phase": "verified",
            "compute_mode": ("exact_in_memory" if status == "accepted" else "metadata_only"),
            "reason_codes": reasons,
            "effective_dataset_workers": workers,
            "worker_adjustment_reason": worker_reason,
            "input_dataset_count": len(verified_estimates),
            "input_bytes_total": sum(item.file_bytes for item in verified_estimates),
            "input_rows_estimated": sum(item.best_rows for item in verified_estimates),
            "input_columns_total": sum(item.columns for item in verified_estimates),
            "estimated_frame_deep_bytes_total": sum(all_frame_bytes),
            "verified_working_set_bytes": verified_working_set,
            "datasets": verified_estimates,
        }
    )


def _resource_footprint(
    datasets: Sequence[LoadedDataset], frame_bytes: Sequence[int]
) -> EdaDataFootprint:
    return EdaDataFootprint(
        dataset_count=len(datasets),
        file_bytes=sum(loaded.record.path.stat().st_size for loaded in datasets),
        rows=sum(len(loaded.frame) for loaded in datasets),
        columns=sum(len(loaded.frame.columns) for loaded in datasets),
        max_rows=max((len(loaded.frame) for loaded in datasets), default=0),
        max_columns=max((len(loaded.frame.columns) for loaded in datasets), default=0),
        frame_deep_bytes=sum(frame_bytes),
        measurement="exact" if datasets else "unavailable",
    )


def _unique_loaded_file_bytes(datasets: Sequence[LoadedDataset]) -> int:
    paths: dict[Path, int] = {}
    for loaded in datasets:
        path = loaded.record.path.resolve()
        paths.setdefault(path, path.stat().st_size)
    return sum(paths.values())


def _emit_auto_eda_resource_trace(
    ctx: SessionContext,
    *,
    driver_started: float,
    baseline_rss: PeakRssMeasurement,
    preprocessing_duration_seconds: float,
) -> None:
    peak = process_peak_rss()
    ctx.emit_trace(
        TraceEvent(
            session_id=ctx.session_id,
            event_type="eda_resource_usage",
            name="auto_eda",
            finished_at=datetime.now(UTC),
            summary={
                "wall_duration_seconds": round(
                    preprocessing_duration_seconds + perf_counter() - driver_started, 6
                ),
                "preprocessing_duration_seconds": max(
                    0.0, round(preprocessing_duration_seconds, 6)
                ),
                "baseline_peak_rss_bytes": baseline_rss.bytes,
                "peak_rss_bytes": peak.bytes,
                "peak_rss_method": peak.method,
            },
        )
    )


def _align_precleaning(
    precleaning: Sequence[CleaningRecipe | None] | None,
    dataset_count: int,
) -> list[CleaningRecipe | None]:
    """Return one optional recipe per dataset, in ingest order."""
    if precleaning is None:
        return [None] * dataset_count
    recipes = list(precleaning)
    if len(recipes) != dataset_count:
        raise ValueError(
            "precleaning must align with file_paths: "
            f"got {len(recipes)} recipe slot(s) for {dataset_count} dataset(s)."
        )
    return recipes


def _copy_and_load_upload(
    source: Path,
    workspace: Path,
    project_id: str,
    *,
    cancel_check: Callable[[], bool] | None = None,
) -> LoadedDataset:
    # Stream the hash so large uploads do not inflate memory (no bytes->hex->json).
    cooperative_cancel = (
        None
        if cancel_check is None
        else lambda: raise_if_cancelled(cancel_check, operation="input loading")
    )
    content_hash = hash_file(source, cancel_check=cooperative_cancel)
    dataset_id = make_dataset_id(source.name, content_hash)
    upload_dir = workspace / "projects" / project_id / "uploads" / dataset_id / "v1"
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / source.name
    if source.resolve() != destination.resolve():
        copy2(source, destination)
    # The source hash above is also the destination hash: copy2 is byte-for-byte.
    # Pass it through so the loader does not scan the file a second time.
    raise_if_cancelled(cancel_check, operation="input loading")
    loaded = load_csv(destination, dataset_id=dataset_id, content_hash=content_hash)
    raise_if_cancelled(cancel_check, operation="input loading")
    return loaded


_STAT_MEASURE_PRIORITY_TOKENS = frozenset(
    {
        "amount",
        "cost",
        "freight",
        "gmv",
        "price",
        "revenue",
        "sales",
        "spend",
        "value",
    }
)


def _stat_measure_priority(column: str) -> int:
    tokens = {token for token in re.split(r"[^0-9a-z]+", column.strip().lower()) if token}
    return 0 if tokens & _STAT_MEASURE_PRIORITY_TOKENS else 1


def _select_stat_test(loaded: LoadedDataset, profile: DatasetProfile) -> _StatTestSpec | None:
    """Select one conservative, business-relevant automatic group comparison."""
    frame = loaded.frame
    role_set = infer_column_roles(profile, frame=frame)
    # PII columns never enter the test: the boxplot artifact inlines raw
    # measure values and group labels, and a phone number is not a measure.
    pii_columns = set(profile.pii_columns)
    numeric_order = {
        str(column): index
        for index, column in enumerate(frame.columns)
        if pd.api.types.is_numeric_dtype(frame[column])
    }
    numeric_columns = sorted(
        (
            role.column
            for role in role_set.roles
            if role.role is ColumnRoleName.MEASURE
            and role.provenance in ("inferred", "seeded")
            and role.column in numeric_order
            and role.column not in pii_columns
        ),
        key=lambda column: (_stat_measure_priority(column), numeric_order[column]),
    )
    dimension_columns = [
        role.column
        for role in role_set.roles
        if role.role is ColumnRoleName.DIMENSION
        and role.provenance in ("inferred", "seeded")
        and role.column in frame.columns
        and role.column not in pii_columns
    ]
    if not numeric_columns or not dimension_columns:
        return None
    for group_column in dimension_columns:
        counts = frame[group_column].dropna().astype(str).value_counts()
        if len(counts) < 2 or len(counts) > 8 or int(counts.min()) < 5:
            continue
        value_column = numeric_columns[0]
        if len(counts) == 2:
            return _StatTestSpec(
                test_type="independent_t_test",
                group_column=group_column,
                value_column=value_column,
            )
        return _StatTestSpec(
            test_type="welch_anova",
            group_column=group_column,
            value_column=value_column,
        )
    return None


def _sanitize_session_title(raw: str) -> str:
    """Normalize an LLM-proposed title into safe one-line display text."""
    return clip_run_title(" ".join(raw.split()).strip("\"'`").strip())


def _executive_summary_claims(artifacts: Sequence[Artifact], *, limit: int = 3) -> list[str]:
    """Return Executive Summary claims, or an empty list for unexpected input."""
    for artifact in artifacts:
        if artifact.type is not ArtifactType.REPORT_BUNDLE:
            continue
        sections = artifact.payload.get("sections")
        if not isinstance(sections, list):
            return []
        for section in sections:
            if not isinstance(section, dict):
                continue
            if section.get("title") != "Executive Summary":
                continue
            claims = section.get("claims")
            if not isinstance(claims, list):
                return []
            texts = [
                claim["text"]
                for claim in claims
                if isinstance(claim, dict) and isinstance(claim.get("text"), str)
            ]
            return texts[:limit]
    return []


def _llm_session_title(
    ctx: SessionContext,
    llm: LLMClient,
    *,
    dataset_names: Sequence[str],
    business_context: str,
    report_artifacts: Sequence[Artifact],
) -> str | None:
    """Request a run title from the LLM, returning ``None`` on failure."""
    if is_offline_client(llm):
        return None
    started_at = datetime.now(UTC)
    try:
        raw = llm.text(
            task="session_title",
            payload={
                "instruction": (
                    "Write a short English session title for this analysis run: "
                    "3-6 plain words, no quotes, no numbers, title case."
                ),
                "dataset_names": list(dataset_names),
                "business_context": business_context,
                "report_headline_claims": _executive_summary_claims(report_artifacts),
            },
        )
        title = _sanitize_session_title(raw)
        summary: dict[str, Any] = {"title": title}
        usage = llm.last_usage()
        if usage is not None:
            summary.update(
                {
                    "provider": usage.provider,
                    "model": usage.model,
                    "prompt_tokens": usage.usage.prompt_tokens,
                    "completion_tokens": usage.usage.completion_tokens,
                    "total_tokens": usage.usage.total_tokens,
                    "cached_tokens": usage.usage.cached_tokens,
                    "estimated_cost_usd": usage.estimated_cost_usd,
                }
            )
            ctx.budget.add_tokens(usage.usage.total_tokens)
        ctx.emit_trace(
            TraceEvent(
                session_id=ctx.session_id,
                event_type="llm_call",
                name="session_title",
                started_at=started_at,
                finished_at=datetime.now(UTC),
                summary=summary,
            )
        )
        return title or None
    except BudgetExceeded as exc:
        # Cosmetic work is the first budget degradation tier. Preserve the
        # deterministic title and make the budget decision observable.
        with suppress(Exception):
            ctx.emit_trace(
                TraceEvent(
                    session_id=ctx.session_id,
                    event_type="budget_degraded",
                    name="session_title",
                    started_at=started_at,
                    finished_at=datetime.now(UTC),
                    summary={"reason": str(exc)[:500]},
                )
            )
        return None
    except Exception as exc:  # noqa: BLE001 — a cosmetic title must never fail the run
        with suppress(Exception):
            ctx.emit_trace(
                TraceEvent(
                    session_id=ctx.session_id,
                    event_type="llm_error",
                    name="session_title",
                    started_at=started_at,
                    finished_at=datetime.now(UTC),
                    summary={
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:200],
                    },
                )
            )
        return None


def _generate_session_id(
    file_paths: Sequence[Path | str],
    *,
    created_at: datetime | None = None,
) -> str:
    now = created_at or datetime.now(UTC)
    stamp = now.strftime("%Y%m%d_%H%M%S_%f")
    source_hint = stable_hash([str(path) for path in file_paths], length=6)
    return f"sess_{stamp}_{source_hint}"


def _relationship_candidates(
    artifacts: Sequence[Artifact],
) -> RelationshipCandidateSet | None:
    for artifact in artifacts:
        if artifact.type is ArtifactType.RELATIONSHIP_CANDIDATE_SET:
            return RelationshipCandidateSet.model_validate(artifact.payload)
    return None


def _relationship_validations(
    artifacts: Sequence[Artifact],
) -> RelationshipValidationSet | None:
    merged = _merged_relationship_validations(artifacts)
    return merged if merged.validations else None


def _merged_relationship_validations(
    artifacts: Sequence[Artifact],
) -> RelationshipValidationSet:
    """Merge validation artifacts independent of filesystem/list ordering."""
    by_label: dict[str, RelationshipValidation] = {}
    for artifact in artifacts:
        if artifact.type is not ArtifactType.RELATIONSHIP_VALIDATION_SET:
            continue
        validation_set = RelationshipValidationSet.model_validate(artifact.payload)
        for validation in validation_set.validations:
            by_label[validation.pair.label()] = validation
    return RelationshipValidationSet(validations=[by_label[label] for label in sorted(by_label)])


def _current_code_version() -> str:
    """Return the local build marker without spawning a Git subprocess."""
    return "local"
