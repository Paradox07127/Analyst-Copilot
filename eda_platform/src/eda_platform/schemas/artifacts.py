from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field, field_validator


class ArtifactType(StrEnum):
    DATASET_PROFILE = "DatasetProfile"
    RAW_DATASET_PROFILE = "RawDatasetProfile"
    QUALITY_ISSUE_SET = "QualityIssueSet"
    QUALITY_CONTEXT_SET = "QualityContextSet"
    CHART_SPEC = "ChartSpec"
    RAW_CHART_SPEC = "RawChartSpec"
    RAW_DATA_PREVIEW = "RawDataPreview"
    MARKDOWN_REPORT = "MarkdownReport"
    REPORT_BUNDLE = "ReportBundle"
    REPORT_AUDIT = "ReportAudit"
    HTML_REPORT = "HtmlReport"
    SQL_RESULT = "SqlResult"
    CODE_EXECUTION_RESULT = "CodeExecutionResult"
    PII_REPORT = "PiiReport"
    CHAT_TURN_PLAN = "ChatTurnPlan"
    TABLE = "Table"
    SESSION_SUMMARY = "SessionSummary"
    RELATIONSHIP_CANDIDATE_SET = "RelationshipCandidateSet"
    RELATIONSHIP_VALIDATION_SET = "RelationshipValidationSet"
    ER_DIAGRAM = "ErDiagram"
    VALUE_MAP = "ValueMap"
    QUESTION_CANDIDATE_SET = "QuestionCandidateSet"
    INVESTIGATION_PLAN = "InvestigationPlan"
    INVESTIGATION_APPROVAL = "InvestigationApproval"
    VALIDATED_FINDING = "ValidatedFinding"
    INVESTIGATION_RECORD = "InvestigationRecord"
    SYNTHESIS_BRIEF = "SynthesisBrief"
    DECISION_REPORT = "DecisionReport"
    DEEP_INVESTIGATION_RESULT = "DeepInvestigationResult"
    DECISION_COVERAGE = "DecisionCoverage"
    QUESTION_EXECUTION_RESULT = "QuestionExecutionResult"
    CLEANING_RECIPE = "CleaningRecipe"
    CLEANING_PREVIEW = "CleaningPreview"
    STAT_TEST_RESULT = "StatTestResult"
    MODEL_CARD = "ModelCard"
    ANOMALY_SCREEN_RESULT = "AnomalyScreenResult"
    SESSION_METRICS = "SessionMetrics"
    RESOURCE_PREFLIGHT = "ResourcePreflight"
    COLUMN_ROLE_SET = "ColumnRoleSet"
    EVIDENCE_INTERLEAVE_TRANSCRIPT = "EvidenceInterleaveTranscript"
    FOLLOW_UP_PROPOSAL_SET = "FollowUpProposalSet"
    LOOP_LEDGER = "LoopLedger"
    EDA_HANDOFF = "EdaHandoff"
    AGENT_HANDOFF = "AgentHandoff"
    EVIDENCE_RECEIPT = "EvidenceReceipt"


class EvidenceRef(BaseModel):
    kind: Literal["sql", "code", "stat", "table", "chart", "profile_field", "artifact"]
    artifact_id: str | None = None
    locator: str
    value: str | float | int | None = None
    unit: Literal["raw", "percent", "currency"] = "raw"
    # Optional exact unit and provenance; unit remains the validator category.
    unit_label: str | None = None
    unit_reference: str | None = None


class Artifact(BaseModel):
    id: str
    type: ArtifactType
    project_id: str
    # Accept artifacts persisted before runs were renamed to sessions.
    session_id: str = Field(validation_alias=AliasChoices("session_id", "run_id"))
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    parents: list[str] = Field(default_factory=list)
    payload: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    # Optional provenance preserves compatibility with existing artifacts.
    env_digest: str | None = None
    code_ref: str | None = None
    plain_language: str | None = None


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    semantic_type: Literal[
        "numeric",
        "categorical",
        "datetime",
        "id",
        "boolean",
        "text",
        "unknown",
    ]
    missing_count: int
    missing_percent: float
    unique_count: int
    unique_percent: float
    sample_values: list[str] = Field(default_factory=list)
    # What the values actually are, independent of dtype: chart form and
    # binning key off this, not off "is the dtype numeric".
    distribution_kind: Literal[
        "empty",
        "constant",
        "binary",
        "discrete",
        "continuous",
    ] = "continuous"
    # Observed levels of a coded column, offered to the user in Knowledge so a
    # value like status="C" gets a confirmed meaning instead of a guess.
    category_levels: list[dict[str, Any]] = Field(default_factory=list)
    parse_success_percent: float | None = None
    parse_failure_count: int = 0
    non_finite_count: int = 0
    whitespace_count: int = 0
    outlier_count: int = 0
    warnings: list[str] = Field(default_factory=list)


class DatasetProfile(BaseModel):
    dataset_id: str
    name: str
    content_hash: str | None = None
    encoding: str | None = None
    delimiter: str | None = None
    rows: int
    columns: int
    column_names: list[str]
    dtypes: dict[str, str]
    missing_values: dict[str, int]
    missing_percent: dict[str, float]
    numeric_columns: list[str]
    categorical_columns: list[str]
    sample_rows: list[dict[str, Any]] = Field(default_factory=list)
    duplicate_rows: int = 0
    exact_duplicate_rows: int = 0
    duplicate_scope_columns: list[str] = Field(default_factory=list)
    columns_detail: list[ColumnProfile] = Field(default_factory=list)
    semantic_type_counts: dict[str, int] = Field(default_factory=dict)
    primary_key_candidates: list[str] = Field(default_factory=list)
    composite_key_candidates: list[list[str]] = Field(default_factory=list)
    # Plain-language answer to "what does one row represent?" — the single most
    # load-bearing sentence in any EDA, and previously never stated.
    grain: str | None = None
    pii_columns: dict[str, Literal["email", "phone", "name", "id", "unknown"]] = Field(
        default_factory=dict
    )


class QualityIssue(BaseModel):
    severity: Literal["critical", "warn", "info"]
    code: str
    column: str | None = None
    message: str
    recommendation: str
    # Structured mirrors of the message figures (analysis-v3 §11.3) so the
    # validator verifies numbers, not prose. None on legacy artifacts.
    metric_value: float | None = None
    metric_unit: Literal["raw", "percent"] = "raw"
    affected_count: int | None = None


class QualityIssueSet(BaseModel):
    dataset_id: str
    issues: list[QualityIssue] = Field(default_factory=list)


class AnalysisTable(BaseModel):
    dataset_id: str
    title: str
    kind: Literal["numeric_summary", "correlation", "association"]
    description: str
    rows: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("rows")
    @classmethod
    def _correlations_are_bounded(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for row in value:
            for key in ("pearson", "abs_pearson"):
                if key not in row or row[key] is None:
                    continue
                try:
                    coefficient = float(row[key])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{key} must be numeric.") from exc
                if not -1.0 <= coefficient <= 1.0:
                    raise ValueError(f"{key} must be in [-1.0, 1.0].")
        return value


class SqlResult(BaseModel):
    sql: str
    columns: list[str]
    dtypes: dict[str, str]
    units: dict[str, str] = Field(
        default_factory=dict,
        description="Case-sensitive output units supplied by the execution plan",
    )
    rows_preview: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int
    truncated: bool = False


class PiiColumn(BaseModel):
    column: str
    label: Literal["email", "phone", "name", "id", "unknown"]
    reason: str


class PiiReport(BaseModel):
    dataset_id: str
    columns: list[PiiColumn] = Field(default_factory=list)
