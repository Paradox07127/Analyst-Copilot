"""Typed, budget-aware final handoff from Auto-EDA to downstream agents."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from eda_platform.schemas.artifacts import ArtifactType


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ReadinessStatus = Literal["ready", "limited", "blocked"]
GateStatus = Literal["pass", "warn", "fail", "not_run"]
CapabilityStatus = Literal[
    "available", "deferred", "not_run", "not_applicable", "failed"
]
PiiLabel = Literal["email", "phone", "name", "id", "unknown"]


class HandoffCapabilities(_StrictModel):
    cleaning: CapabilityStatus
    profiling: CapabilityStatus
    quality: CapabilityStatus
    visualization: CapabilityStatus
    statistics: CapabilityStatus
    modeling: CapabilityStatus
    relationships: CapabilityStatus
    questions: CapabilityStatus
    report: CapabilityStatus
    metrics: CapabilityStatus


class HandoffRun(_StrictModel):
    project_id: str
    session_id: str
    status: Literal["completed", "completed_with_limits"]
    producer_version: str
    execution_fingerprint: str
    input_hashes: dict[str, str] = Field(default_factory=dict)
    pipeline_artifact_count: int = Field(ge=0)
    persisted_source_artifact_count: int = Field(ge=0)
    referenced_external_artifact_count: int = Field(ge=0)
    artifact_count: int = Field(ge=0)
    artifact_counts: dict[str, int] = Field(default_factory=dict)
    source_inventory_count: int = Field(ge=0)
    source_inventory_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    external_inventory_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    lineage_candidate_parent_count: int = Field(ge=0)
    lineage_parent_count: int = Field(ge=0)
    lineage_parents_truncated: bool = False
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @model_validator(mode="after")
    def _validate_inventory_and_lineage(self) -> HandoffRun:
        if self.pipeline_artifact_count > self.persisted_source_artifact_count:
            raise ValueError("pipeline_artifact_count cannot exceed persisted source count")
        if self.source_inventory_count != self.persisted_source_artifact_count:
            raise ValueError("source inventory count must equal persisted source count")
        if self.lineage_parent_count > self.lineage_candidate_parent_count:
            raise ValueError("included lineage parents cannot exceed candidates")
        if self.lineage_parents_truncated != (
            self.lineage_parent_count < self.lineage_candidate_parent_count
        ):
            raise ValueError("lineage truncation flag must match included parent count")
        return self


class HandoffGate(_StrictModel):
    status: GateStatus
    reasons: list[str] = Field(default_factory=list)
    artifact_id: str | None = None


class RelationshipReadiness(_StrictModel):
    status: Literal[
        "not_applicable", "deferred", "materialized", "validated", "rejected"
    ]
    cross_table_claims_allowed: bool
    action: str | None = None
    artifact_ids: list[str] = Field(default_factory=list)


class AgentReadiness(_StrictModel):
    status: ReadinessStatus
    reasons: list[str] = Field(default_factory=list)
    quality_gate: HandoffGate
    report_gate: HandoffGate
    cross_dataset_relationships: RelationshipReadiness


class DatasetQualitySummary(_StrictModel):
    critical: int = Field(ge=0)
    warn: int = Field(ge=0)
    info: int = Field(ge=0)
    material_codes: list[str] = Field(default_factory=list)


class DatasetReadiness(_StrictModel):
    status: ReadinessStatus
    reasons: list[str] = Field(default_factory=list)


class AgentDataset(_StrictModel):
    dataset_id: str
    raw_dataset_id: str | None = None
    name: str
    content_hash: str | None = None
    rows: int = Field(ge=0)
    columns: int = Field(ge=0)
    grain: str | None = None
    semantic_type_counts: dict[str, int] = Field(default_factory=dict)
    primary_key_candidates: list[str] = Field(default_factory=list)
    composite_key_candidates: list[list[str]] = Field(default_factory=list)
    pii_columns: dict[str, PiiLabel] = Field(default_factory=dict)
    pii_column_count: int = Field(ge=0)
    pii_columns_omitted: int = Field(ge=0)
    quality: DatasetQualitySummary
    readiness: DatasetReadiness
    artifact_ids: dict[str, str | list[str]] = Field(default_factory=dict)
    artifact_omitted_counts: dict[str, int] = Field(default_factory=dict)


class ArtifactCatalogEntry(_StrictModel):
    artifact_id: str
    type: ArtifactType
    origin_session_id: str
    stage: Literal[
        "ingest",
        "profile",
        "quality",
        "exploration",
        "statistics",
        "semantic",
        "question_planning",
        "question_execution",
        "reporting",
        "observability",
    ]
    role: Literal[
        "gate",
        "summary",
        "evidence",
        "visual",
        "plan",
        "result",
        "presentation",
        "metric",
    ]
    dataset_id: str | None = None
    title: str | None = None
    priority: Literal["critical", "high", "normal", "on_demand"]
    required: bool
    fetch: str
    content_sha256: str
    content_bytes: int = Field(ge=0)
    estimated_tokens: int = Field(ge=0)
    parent_count: int = Field(ge=0)
    parents: list[str] = Field(default_factory=list)
    evidence_count: int = Field(ge=0)
    warning_count: int = Field(ge=0)
    sensitivity: Literal["public", "internal", "sensitive", "pii_restricted"]


class HandoffQuestionResult(_StrictModel):
    question_id: str
    status: Literal["answered", "abstained", "failed", "awaiting_approval"]
    execution_artifact_id: str
    sql_artifact_id: str | None = None
    chart_artifact_id: str | None = None
    finding_count: int = Field(ge=0)
    limitation_count: int = Field(ge=0)
    exploratory: bool = False


class HandoffReport(_StrictModel):
    status: Literal["ready", "limited", "failed", "not_generated"]
    audit_artifact_id: str | None = None
    bundle_artifact_id: str | None = None
    markdown_artifact_id: str | None = None
    html_artifact_id: str | None = None


class HandoffNextAction(_StrictModel):
    action: str
    priority: Literal["critical", "high", "normal", "low"]
    blocking: bool
    reason: str | None = None
    endpoint: str | None = None


class HandoffContextPolicy(_StrictModel):
    default_artifact_ids: list[str] = Field(default_factory=list)
    on_demand_types: list[ArtifactType] = Field(default_factory=list)
    excluded_by_default_types: list[ArtifactType] = Field(default_factory=list)
    max_initial_bytes: int = Field(gt=0)
    max_initial_estimated_tokens: int = Field(gt=0)
    cataloged_artifact_count: int = Field(ge=0)
    default_artifact_count: int = Field(ge=0)
    omitted_artifact_count: int = Field(ge=0)
    included_question_result_count: int = Field(ge=0)
    omitted_question_result_count: int = Field(ge=0)
    default_artifact_bytes: int = Field(ge=0)
    default_artifact_estimated_tokens: int = Field(ge=0)
    serialized_bytes: int = Field(ge=0)
    estimated_tokens: int = Field(ge=0)
    initial_context_bytes: int = Field(ge=0)
    initial_context_estimated_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_initial_context_budget(self) -> HandoffContextPolicy:
        if self.initial_context_bytes != (
            self.default_artifact_bytes + self.serialized_bytes
        ):
            raise ValueError("initial context bytes must equal defaults plus handoff payload")
        if self.initial_context_estimated_tokens != (
            self.default_artifact_estimated_tokens + self.estimated_tokens
        ):
            raise ValueError("initial context tokens must equal defaults plus handoff payload")
        if self.initial_context_bytes > self.max_initial_bytes:
            raise ValueError("initial context exceeds byte budget")
        if self.initial_context_estimated_tokens > self.max_initial_estimated_tokens:
            raise ValueError("initial context exceeds token budget")
        return self


class AgentHandoffV3(_StrictModel):
    """Final session manifest. Large evidence remains lazy artifact references."""

    contract_version: Literal["3.0"] = "3.0"
    generated_at: datetime
    run: HandoffRun
    readiness: AgentReadiness
    capabilities: HandoffCapabilities
    datasets: list[AgentDataset] = Field(default_factory=list)
    artifact_catalog: list[ArtifactCatalogEntry] = Field(default_factory=list)
    question_results: list[HandoffQuestionResult] = Field(default_factory=list)
    report: HandoffReport
    next_actions: list[HandoffNextAction] = Field(default_factory=list)
    context_policy: HandoffContextPolicy

    @model_validator(mode="after")
    def _validate_references_and_counts(self) -> AgentHandoffV3:
        if self.run.artifact_count != self.run.persisted_source_artifact_count + 1:
            raise ValueError("artifact_count must include exactly one AgentHandoff")
        if sum(self.run.artifact_counts.values()) != self.run.artifact_count:
            raise ValueError("artifact_counts must sum to artifact_count")
        if self.run.artifact_counts.get("AgentHandoff") != 1:
            raise ValueError("artifact_counts must contain exactly one AgentHandoff")
        ids = [entry.artifact_id for entry in self.artifact_catalog]
        if len(ids) != len(set(ids)):
            raise ValueError("artifact_catalog ids must be unique")
        catalog_ids = set(ids)
        if not set(self.context_policy.default_artifact_ids).issubset(catalog_ids):
            raise ValueError("default artifact ids must exist in artifact_catalog")
        if self.context_policy.cataloged_artifact_count != len(self.artifact_catalog):
            raise ValueError("cataloged_artifact_count must match artifact_catalog")
        if self.context_policy.default_artifact_count != len(
            self.context_policy.default_artifact_ids
        ):
            raise ValueError("default_artifact_count must match default_artifact_ids")
        referenced_count = (
            self.run.persisted_source_artifact_count
            + self.run.referenced_external_artifact_count
        )
        if (
            self.context_policy.cataloged_artifact_count
            + self.context_policy.omitted_artifact_count
            != referenced_count
        ):
            raise ValueError("cataloged and omitted artifacts must cover referenced inventory")
        if self.context_policy.included_question_result_count != len(self.question_results):
            raise ValueError("included question result count must match question_results")
        catalog_by_id = {entry.artifact_id: entry for entry in self.artifact_catalog}
        if any(
            catalog_by_id[artifact_id].sensitivity == "pii_restricted"
            for artifact_id in self.context_policy.default_artifact_ids
        ):
            raise ValueError("pii_restricted artifacts cannot enter default context")
        relationship = self.readiness.cross_dataset_relationships
        if relationship.status in {"deferred", "materialized", "rejected"} and (
            relationship.cross_table_claims_allowed or not relationship.action
        ):
            raise ValueError("unvalidated relationships must block claims and name an action")
        if relationship.status == "validated" and not relationship.cross_table_claims_allowed:
            raise ValueError("validated relationships must allow cross-table claims")
        if self.readiness.quality_gate.status == "fail" and self.readiness.status != "blocked":
            raise ValueError("failed quality gate must block readiness")
        if self.readiness.report_gate.status == "fail" and self.readiness.status != "blocked":
            raise ValueError("failed report gate must block readiness")
        if any(dataset.readiness.status == "blocked" for dataset in self.datasets) and (
            self.readiness.status != "blocked"
        ):
            raise ValueError("blocked dataset must block overall readiness")
        expected_run_status = (
            "completed" if self.readiness.status == "ready" else "completed_with_limits"
        )
        if self.run.status != expected_run_status:
            raise ValueError("run status must agree with Agent readiness")
        if self.report.status == "not_generated":
            if any(
                artifact_id is not None
                for artifact_id in (
                    self.report.audit_artifact_id,
                    self.report.bundle_artifact_id,
                    self.report.markdown_artifact_id,
                    self.report.html_artifact_id,
                )
            ):
                raise ValueError("not-generated report cannot reference report artifacts")
            if self.readiness.report_gate.status != "not_run":
                raise ValueError("not-generated report requires a not-run report gate")
        elif self.report.audit_artifact_id is None:
            raise ValueError("generated report must reference its audit artifact")
        return self
