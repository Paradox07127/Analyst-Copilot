"""Stable DTOs exposed by application services (and thus the HTTP API)."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from eda_platform.schemas.publication import PublicationFreshness, PublicationReadiness


class ProjectSummary(BaseModel):
    project_id: str
    name: str
    session_count: int = 0


class UploadDeleted(BaseModel):
    project_id: str
    dataset_id: str
    deleted: bool = True


class ProjectDeleted(BaseModel):
    project_id: str
    deleted: bool = True


class SessionSummary(BaseModel):
    session_id: str
    project_id: str
    title: str | None = None
    status: str = "unknown"
    created_at: datetime | None = None
    updated_at: datetime | None = None
    dataset_names: list[str] = Field(default_factory=list)
    artifact_count: int = 0
    report_status: str | None = None
    chat_message_count: int = 0


class SessionDetail(SessionSummary):
    """Session metadata only — never artifact payloads, report text, or frames."""

    code_version: str | None = None
    seed: int | None = None
    source_session_id: str | None = None
    artifact_type_counts: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class Page[T](BaseModel):
    items: list[T]
    next_cursor: str | None = None


class DatasetColumn(BaseModel):
    name: str
    dtype: str


class DatasetHandle(BaseModel):
    """Dataset metadata only (§8.2) — never a DataFrame, never an absolute path."""

    # "schema" shadows a (deprecated) BaseModel attribute, hence the alias.
    model_config = ConfigDict(populate_by_name=True)

    dataset_id: str
    project_id: str
    display_name: str
    original_uri: str
    """Source file path relative to the workspace root ('' when missing)."""
    format: str = "csv"
    content_hash: str = ""
    byte_size: int = 0
    row_count: int | None = None
    schema_: list[DatasetColumn] = Field(default_factory=list, alias="schema")
    ingest_status: str = "ready"


class DatasetSchema(BaseModel):
    dataset_id: str
    session_id: str
    columns: list[DatasetColumn]
    source: str
    """"profile" when read from the DatasetProfile payload, "inferred" when
    derived from a header-only DuckDB DESCRIBE of the source file."""


class DatasetPreview(BaseModel):
    dataset_id: str
    session_id: str
    columns: list[str]
    rows: list[list[Any]]
    offset: int
    limit: int
    has_more: bool
    source_format: str = "csv"


class ReportView(BaseModel):
    """Report reader payload. status "none" (empty markdown) means the run has
    no report yet — that is a normal 200, not an error."""

    session_id: str
    status: str
    markdown: str
    generated_at: datetime | None = None


class ArtifactSummary(BaseModel):
    """Index-only listing row — never the payload (§13.2). created_at is
    omitted because the SQLite artifacts index does not store it; including it
    would force a payload read per row."""

    artifact_id: str
    type: str
    created_at: datetime | None = None


class ArtifactDetail(BaseModel):
    artifact_id: str
    type: str
    project_id: str
    session_id: str
    created_at: datetime | None = None
    payload: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)


class JobStatus(BaseModel):
    job_id: str
    session_id: str
    project_id: str
    kind: str
    status: str
    """queued | running | completed | failed | cancelled"""
    cancel_requested: bool = False
    created_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error_code: str | None = None
    error_message: str | None = None
    events_url: str


class JobCreated(BaseModel):
    job_id: str
    session_id: str
    status: str
    events_url: str


class JobEvent(BaseModel):
    """One SSE payload: a trace_events row wrapped with job/run correlation."""

    event_id: int
    job_id: str
    session_id: str
    type: str
    name: str
    timestamp: datetime | None = None
    summary: dict[str, Any] = Field(default_factory=dict)


class CleaningOperation(BaseModel):
    transform_id: str
    type: str
    target_column: str | None = None
    description: str = ""
    lossy: bool = False


class CleaningColumnChange(BaseModel):
    column: str
    before_dtype: str
    after_dtype: str
    changed_rows: int
    before_missing: int
    after_missing: int


class CleaningPreviewView(BaseModel):
    """Row/column diff summary for a recipe — never the cleaned frame itself."""

    dataset_id: str
    recipe_id: str
    source_version: int
    target_version: int
    row_count_before: int
    row_count_after: int
    rows_dropped: int
    rows_edited: int
    cells_changed: int
    column_changes: list[CleaningColumnChange] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CleaningPreviewResult(BaseModel):
    session_id: str
    dataset_id: str
    action_hash: str
    approval_token: str
    expires_at: datetime
    operations: list[CleaningOperation] = Field(default_factory=list)
    preview: CleaningPreviewView


class DataOperationStarted(BaseModel):
    """A data scan queued on a derived lifecycle run."""

    session_id: str
    execution_session_id: str
    job: JobCreated


class CleaningApplied(BaseModel):
    """Apply outcome: the cleaned version plus the fork job analyzing it.

    `target_version` is None on an idempotent replay, where only the job row
    survives to answer from."""

    session_id: str
    new_session_id: str
    dataset_id: str
    target_version: int | None = None
    job: JobCreated


class CleaningLogSummaryRow(BaseModel):
    dataset: str
    recipe_id: str
    rows_before: int | None = None
    rows_after: int | None = None
    rows_removed: int | None = None
    columns_before: int | None = None
    columns_after: int | None = None
    columns_removed: int | None = None
    delete_steps: int = 0
    protection_triggers: int = 0
    requires_approval: bool = False


class CleaningLogOperationRow(BaseModel):
    dataset: str
    operation: str
    column: str = ""
    rows_deleted: int = 0
    columns_deleted: int = 0
    reason: str = ""
    details: str = ""


class CleaningLogGuardrailRow(BaseModel):
    dataset: str
    code: str
    reason: str
    thresholds: str = ""


class CleaningLogSuggestionRow(BaseModel):
    suggestion: str


class CleaningLogView(BaseModel):
    """The four cleaning-transparency tables for one run.

    `recipe_count` is 0 when the run recorded no CleaningRecipe at all, which is
    what separates "nothing was cleaned" from "cleaned, but nothing deleted"."""

    session_id: str
    recipe_count: int = 0
    summary: list[CleaningLogSummaryRow] = Field(default_factory=list)
    deleted_data: list[CleaningLogOperationRow] = Field(default_factory=list)
    protection_triggers: list[CleaningLogGuardrailRow] = Field(default_factory=list)
    suggestions: list[CleaningLogSuggestionRow] = Field(default_factory=list)


class QualityIssueRow(BaseModel):
    severity: str
    """critical | warn | info"""
    dataset_name: str
    dataset_id: str | None = None
    """Nullable for pre-existing clients; the service always sets it."""
    code: str
    column: str | None = None
    message: str
    recommendation: str = ""


class QualityDatasetCard(BaseModel):
    dataset_name: str
    dataset_id: str | None = None
    """Nullable for pre-existing clients; the service always sets it."""
    critical: int = 0
    warn: int = 0
    info: int = 0


class QualityView(BaseModel):
    session_id: str
    critical: int = 0
    warn: int = 0
    info: int = 0
    datasets: list[QualityDatasetCard] = Field(default_factory=list)
    issues: list[QualityIssueRow] = Field(default_factory=list)


class FieldProfileRow(BaseModel):
    column: str
    dtype: str
    semantic_type: str
    missing_percent: float | None = None
    unique_percent: float | None = None
    sample_values: str = ""


class DatasetProfileSummary(BaseModel):
    dataset_id: str
    name: str
    rows: int
    columns: int
    semantic_type_counts: dict[str, int] = Field(default_factory=dict)
    fields: list[FieldProfileRow] = Field(default_factory=list)


class ProfilesView(BaseModel):
    session_id: str
    datasets: list[DatasetProfileSummary] = Field(default_factory=list)


class RawChartView(BaseModel):
    """A before-cleaning chart with its complete vega-lite spec."""

    artifact_id: str
    title: str
    dataset_id: str
    dataset_name: str
    description: str = ""
    plain_language: str | None = None
    spec: dict[str, Any] = Field(default_factory=dict)


class RawDataPreviewView(BaseModel):
    artifact_id: str
    dataset_id: str
    name: str
    rows: int = 0
    columns: int = 0
    column_names: list[str] = Field(default_factory=list)
    rows_preview: list[dict[str, Any]] = Field(default_factory=list)


class CleaningRawView(BaseModel):
    """Before-cleaning snapshot of a run.

    `precleaning_recorded` is False when the run carries no raw profile, chart,
    preview, or cleaning recipe — i.e. pre-cleaning was never enabled. When it is
    True, an empty list means that category alone was not recorded."""

    session_id: str
    precleaning_recorded: bool = False
    profiles: list[DatasetProfileSummary] = Field(default_factory=list)
    charts: list[RawChartView] = Field(default_factory=list)
    previews: list[RawDataPreviewView] = Field(default_factory=list)


class DistributionCategory(BaseModel):
    value: str
    count: int


class ColumnDistribution(BaseModel):
    """One column's mini distribution. Numeric columns fill counts/bin_edges/
    min/max; categorical columns fill top/other_count/unique_count/len_*; an
    all-null column fills neither and reports kind "empty"."""

    name: str
    dtype: str
    kind: str
    """numeric | categorical | empty"""
    missing_percent: float = 0.0
    counts: list[int] | None = None
    bin_edges: list[float] | None = None
    min: float | None = None
    max: float | None = None
    top: list[DistributionCategory] | None = None
    other_count: int | None = None
    unique_count: int | None = None
    len_min: int | None = None
    len_max: int | None = None


class ColumnDistributionsView(BaseModel):
    """Structured per-column distributions; the client draws them.

    `sampled` / `sample_rows` carry the footnote: above `sample_cap` rows the
    stats come from a random sample, not the full table."""

    dataset_id: str
    session_id: str
    row_count: int = 0
    sampled: bool = False
    sample_rows: int = 0
    sample_cap: int = 0
    bins: int = 0
    top_k: int = 0
    columns: list[ColumnDistribution] = Field(default_factory=list)


class ChartSummary(BaseModel):
    """Chart listing row — metadata only, never the vega-lite spec (§13.3)."""

    artifact_id: str
    title: str
    dataset_id: str
    dataset_name: str
    mark: str
    fields: list[str] = Field(default_factory=list)
    description: str = ""


class ChartView(BaseModel):
    artifact_id: str
    session_id: str
    title: str
    dataset_id: str
    dataset_name: str
    description: str = ""
    plain_language: str | None = None
    spec: dict[str, Any] = Field(default_factory=dict)
    """Complete vega-lite spec (ChartSpec.to_vegalite), rendered client-side."""


class UploadStatus(BaseModel):
    upload_id: str
    project_id: str
    status: str
    """"completed" or "failed" (uploads are processed synchronously)."""
    error: str | None = None
    dataset: DatasetHandle | None = None


class QuestionExecutionSummary(BaseModel):
    """Latest execution outcome for one question, across the run and its
    derived question-batch sessions."""

    outcome: str
    """answered | abstained | failed | awaiting_approval"""
    status: str
    """succeeded | failed (legacy driver status)"""
    findings_count: int = 0
    qexec_artifact_id: str
    execution_session_id: str
    abstention_code: str | None = None


class QuestionSummary(BaseModel):
    question_id: str
    question: str
    origin: str
    """template | llm"""
    analysis_mode: str | None = None
    value_category: str | None = None
    feasibility_status: str | None = None
    proposed_action: str | None = None
    priority: float
    """Deterministic score in [0, 1]; display ranking only."""
    exploratory: bool = False
    target_datasets: list[str] = Field(default_factory=list)
    business_decision: str = ""
    executable: bool = True
    """False when deterministic feasibility blocks execution."""
    execution: QuestionExecutionSummary | None = None
    card_version: int = 1
    """Bumped by every card edit; an investigation plan records the version it planned."""
    value_hypothesis: str = ""
    success_criterion: str = ""
    data_signal: str = ""
    priority_rationale: str = ""
    risks: list[str] = Field(default_factory=list)
    data_requirements: list[str] = Field(default_factory=list)


class QuestionsView(BaseModel):
    session_id: str
    questions: list[QuestionSummary] = Field(default_factory=list)


class QuestionExecutionPrepared(BaseModel):
    """Pending approval for executing one question (§6.0 approval contract)."""

    session_id: str
    question_id: str
    action_hash: str
    approval_token: str
    expires_at: datetime
    question: str
    origin: str
    sql_preview: str | None = None
    """Offline deterministic SQL; None when the autonomous agent chooses tools."""
    target_datasets: list[str] = Field(default_factory=list)
    uses_llm: bool = False
    llm_mode: str = "env"
    """The approval-bound LLM mode ("env" | "offline") the execution will use."""


class QuestionExecutionStarted(BaseModel):
    """Execute outcome: the derived batch run plus the job running it."""

    session_id: str
    question_id: str
    execution_session_id: str
    job: JobCreated


class FindingEvidenceRef(BaseModel):
    kind: str
    artifact_id: str | None = None
    locator: str = ""
    session_id: str | None = None
    """Session whose Artifacts page can show this evidence; None → not linkable."""


class FindingStatement(BaseModel):
    text: str
    evidence: list[FindingEvidenceRef] = Field(default_factory=list)


class FindingFreshnessInfo(BaseModel):
    status: str
    """fresh | stale | unverifiable"""
    reasons: list[str] = Field(default_factory=list)


class FindingSummary(BaseModel):
    """One project-level validated finding with provenance and freshness."""

    artifact_id: str
    source_session_id: str
    source_session_navigable: bool = True
    """False for internal runs the UI has no page for; render the id as text."""
    from_current_session: bool = False
    created_at: datetime | None = None
    question: str
    claim_class: str
    evidence_support: str
    analytical_reliability: str
    decision_readiness: str
    report_readiness: str
    report_readiness_reason: str
    statements: list[FindingStatement] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    interpretation: str | None = None
    """Validator-gated interpretation; None unless status is "validated"."""
    value_hypothesis: str | None = None
    method_artifact_id: str | None = None
    freshness: FindingFreshnessInfo


class InvestigationLogEntry(BaseModel):
    artifact_id: str
    source_session_id: str
    from_current_session: bool = False
    question: str
    status: str
    reason_code: str
    reason: str
    next_action: str


class FindingsView(BaseModel):
    """Project-scoped findings library viewed from one run."""

    session_id: str
    project_id: str
    findings: list[FindingSummary] = Field(default_factory=list)
    records: list[InvestigationLogEntry] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class FieldMeaningView(BaseModel):
    dataset: str = Field(max_length=500)
    column: str = Field(max_length=500)
    meaning: str = Field(max_length=500)
    unit: str | None = Field(default=None, max_length=50)
    aliases: list[Annotated[str, StringConstraints(max_length=200)]] = Field(
        default_factory=list, max_length=20
    )


class ColumnRoleRow(BaseModel):
    dataset: str
    column: str
    role: str
    confidence: float


class JoinWhitelistEntryView(BaseModel):
    seeds_version: int = 0
    label: str
    left_dataset: str
    left_columns: list[str]
    right_dataset: str
    right_columns: list[str]
    status: str
    """proposed | confirmed | auto_confirmed"""
    cardinality: str = "unknown"
    validation_verified: bool = False
    freshness: str = "unverifiable"
    """fresh | stale | unverifiable, against the current run's dataset ids."""
    join_row_multiplier: float | None = None
    usage_count: int = 0
    confidence_source: str = ""


class VerifiedRelationView(BaseModel):
    """One user-confirmed join sunk into the semantic seeds. `left`/`right` are
    seed-side keys (`dataset.col[+col...]`), which together identify the row."""

    left: str
    right: str
    cardinality: str
    confirmed_by: str = "user"
    confirmed_at: datetime | None = None
    source_session_id: str | None = None


class VerifiedRelationsUpdated(BaseModel):
    session_id: str
    seeds_version: int
    verified_relations: list[VerifiedRelationView] = Field(default_factory=list)


class MeaningProposalView(BaseModel):
    dataset: str
    column: str
    meaning: str
    unit_guess: str = ""
    confidence: str = "hypothesis"
    source: str = "bootstrap"


class MetricDefinitionView(BaseModel):
    name: str = Field(max_length=200)
    definition: str = Field(max_length=2000)
    formula: str | None = Field(default=None, max_length=1000)
    caveats: str | None = Field(default=None, max_length=2000)


class EntityNoteView(BaseModel):
    name: str = Field(max_length=200)
    note: str = Field(max_length=2000)


class VerifiedAnswerView(BaseModel):
    question: str = Field(max_length=1000)
    answer: str = Field(max_length=5000)
    evidence_note: str | None = Field(default=None, max_length=1000)
    verified_at: datetime | None = None
    """Round-tripped so editing an answer keeps its original verification date;
    omitted on a new answer, which the server then stamps."""


class SemanticView(BaseModel):
    """The project semantic layer viewed from one run."""

    session_id: str
    project_id: str
    seeds_version: int = 0
    field_meanings: list[FieldMeaningView] = Field(default_factory=list)
    metric_definitions: list[MetricDefinitionView] = Field(default_factory=list)
    entity_notes: list[EntityNoteView] = Field(default_factory=list)
    verified_answers: list[VerifiedAnswerView] = Field(default_factory=list)
    verified_relations: list[VerifiedRelationView] = Field(default_factory=list)
    column_roles: list[ColumnRoleRow] = Field(default_factory=list)
    join_whitelist: list[JoinWhitelistEntryView] = Field(default_factory=list)
    proposals: list[MeaningProposalView] = Field(default_factory=list)
    next_cursor: str | None = None


class SemanticSeedsUpdated(BaseModel):
    session_id: str
    version: int
    field_meanings: list[FieldMeaningView] = Field(default_factory=list)
    metric_definitions: list[MetricDefinitionView] = Field(default_factory=list)
    entity_notes: list[EntityNoteView] = Field(default_factory=list)
    verified_answers: list[VerifiedAnswerView] = Field(default_factory=list)


class ProposalReviewed(BaseModel):
    session_id: str
    dataset: str
    column: str
    status: str
    """accepted | rejected"""
    seeds_version: int


class CompareSessionSide(BaseModel):
    """Index-level identity of one side of a comparison."""

    session_id: str
    project_id: str
    title: str | None = None
    status: str = "unknown"
    created_at: datetime | None = None
    dataset_names: list[str] = Field(default_factory=list)
    artifact_count: int = 0
    report_status: str | None = None


class CompareValue[T](BaseModel):
    """A comparison value whose absence has an explicit, non-numeric meaning."""

    state: Literal["value", "missing", "unavailable", "not_applicable"]
    value: T | None = None
    reason: str | None = None


class CompareRuntimeView(BaseModel):
    input_hashes: CompareValue[dict[str, str]]
    code_version: CompareValue[str]
    seed: CompareValue[int]
    model_versions: CompareValue[dict[str, str]]
    source_session_id: CompareValue[str]
    prompt_template_version: CompareValue[str]


class ComparabilityView(BaseModel):
    verdict: Literal[
        "controlled", "partially_controlled", "not_directly_comparable", "unknown"
    ]
    left: CompareRuntimeView
    right: CompareRuntimeView
    changed_dimensions: list[str] = Field(default_factory=list)
    unknown_dimensions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CompareLineageView(BaseModel):
    relation: Literal[
        "siblings", "direct_parent", "ancestor_descendant", "unrelated", "unknown"
    ]
    common_ancestor_session_id: str | None = None
    left_path: list[str] = Field(default_factory=list)
    right_path: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CompareMetricRow(BaseModel):
    key: str
    label: str
    left: CompareValue[float]
    right: CompareValue[float]
    delta: float | None = None
    """right - left; None unless both sides contain different numeric values."""
    optimization_direction: Literal["maximize", "minimize", "none"] = "none"
    verdict: Literal["improved", "regressed", "unchanged", "tradeoff", "unknown"] = "unknown"
    higher_is_better: bool | None = None
    """Compatibility field; new clients use optimization_direction + verdict."""


class CompareTextRow(BaseModel):
    key: str
    label: str
    left: CompareValue[str]
    right: CompareValue[str]
    changed: bool | None = None


class CompareArtifactDelta(BaseModel):
    type: str
    left: CompareValue[int]
    right: CompareValue[int]
    delta: int | None = None


class CompareDatasetDiff(BaseModel):
    left: CompareValue[list[str]]
    right: CompareValue[list[str]]
    shared: list[str] = Field(default_factory=list)
    only_left: list[str] = Field(default_factory=list)
    only_right: list[str] = Field(default_factory=list)


class CompareView(BaseModel):
    """Two runs of one project side by side (§10.3 Compare)."""

    project_id: str
    left: CompareSessionSide
    right: CompareSessionSide
    comparability: ComparabilityView
    lineage: CompareLineageView
    metrics: list[CompareMetricRow] = Field(default_factory=list)
    text_rows: list[CompareTextRow] = Field(default_factory=list)
    artifact_deltas: list[CompareArtifactDelta] = Field(default_factory=list)
    datasets: CompareDatasetDiff


class SkillPlanCandidate(BaseModel):
    """A validated plan artifact of this run that can be frozen into a skill."""

    artifact_id: str
    question: str
    sql: str
    method: str = ""
    dataset_names: list[str] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    created_at: datetime | None = None


class SkillParamSpec(BaseModel):
    """One placeholder a seed template requires before it can be replayed."""

    name: str
    role: str
    """measure | dimension | timestamp | identifier | any"""
    description: str = ""


class SkillSummary(BaseModel):
    skill_id: str
    source: str
    """library (saved in this project) | seed (builtin template)"""
    name: str
    description: str = ""
    question: str
    sql: str
    method: str = ""
    param_columns: list[str] = Field(default_factory=list)
    expected_datasets: list[str] = Field(default_factory=list)
    params: list[SkillParamSpec] = Field(default_factory=list)
    """Always empty for library skills; seeds need every one bound to a column."""
    source_session_id: str | None = None
    created_at: datetime | None = None


class SkillTargetColumn(BaseModel):
    """A column of a replay target, and whether a seed may bind to it."""

    name: str
    bindable: bool = True
    """False when the name is not a plain SQL identifier: bound values are
    interpolated into SQL, so the guard would refuse it at prepare time."""


class SkillTargetDataset(BaseModel):
    """A replay target in the current run, with the columns bindings may use."""

    dataset_id: str
    name: str
    relation: str
    """SQL relation name when this dataset is the sole replay target."""
    columns: list[SkillTargetColumn] = Field(default_factory=list)


class SkillsView(BaseModel):
    session_id: str
    project_id: str
    skills: list[SkillSummary] = Field(default_factory=list)
    datasets: list[SkillTargetDataset] = Field(default_factory=list)
    savable_plans: list[SkillPlanCandidate] = Field(default_factory=list)
    """Plan artifacts of this run a POST /skills can freeze into a new skill."""
    next_cursor: str | None = None


class SkillReplayPrepared(BaseModel):
    """Pending approval for replaying one skill (§6.0 approval contract)."""

    session_id: str
    skill_id: str
    action_hash: str
    approval_token: str
    expires_at: datetime
    name: str
    question: str
    sql_preview: str
    dataset_ids: list[str] = Field(default_factory=list)
    dataset_names: list[str] = Field(default_factory=list)
    bindings: dict[str, str] = Field(default_factory=dict)
    uses_llm: bool = False
    """Always False: replay is deterministic SQL through the read-only gate."""


class SkillReplayStarted(BaseModel):
    """Replay outcome: the derived run holding the result plus its job."""

    session_id: str
    skill_id: str
    execution_session_id: str
    job: JobCreated


class ChatMessageView(BaseModel):
    """One persisted transcript line. `seq` is the 0-based JSONL line index and
    doubles as the reverse-pagination cursor."""

    seq: int
    role: str
    content: str
    status: str = "answer"
    sql: str | None = None
    artifact_refs: list[str] = Field(default_factory=list)
    created_at: datetime | None = None


class ChatMessagePage(BaseModel):
    session_id: str
    messages: list[ChatMessageView] = Field(default_factory=list)
    next_cursor: str | None = None
    """Cursor for the next OLDER page; None once the transcript start is reached."""
    total: int = 0


class ChatMessageAccepted(BaseModel):
    """202 body: the turn runs in the background, progress arrives over SSE."""

    session_id: str
    message_id: str
    stream_url: str


class ChatStreamEvent(BaseModel):
    seq: int
    session_id: str
    message_id: str
    type: str
    """turn.started | progress | tool.call | plan.pending | message.completed | turn.failed"""
    data: dict[str, Any] = Field(default_factory=dict)


class ChatPlanRejected(BaseModel):
    session_id: str
    plan_id: str
    status: str = "rejected"


class ChatPendingPlan(BaseModel):
    """A plan still awaiting approval, with a freshly re-issued token.

    Same fields as the `plan.pending` stream frame, so a client that lost its
    stream (reload, API restart, session eviction) can rebuild the approval
    card without replaying the turn.
    """

    plan_id: str
    action_hash: str
    approval_token: str
    expires_at: datetime
    question: str = ""
    method: str = ""
    sql: str = ""
    dataset_names: list[str] = Field(default_factory=list)
    estimated_scan: str = "unknown"


class ChatPendingPlanList(BaseModel):
    session_id: str
    plans: list[ChatPendingPlan] = Field(default_factory=list)


class BoardCard(BaseModel):
    id: str = Field(max_length=64)
    title: str = Field(max_length=300)
    ref_type: str = Field(default="none", max_length=32)
    """none | finding | question | artifact"""
    ref_id: str = Field(default="", max_length=200)
    note: str = Field(default="", max_length=2000)


class BoardColumn(BaseModel):
    id: str = Field(max_length=64)
    title: str = Field(max_length=120)
    card_ids: list[str] = Field(default_factory=list, max_length=500)


class BoardView(BaseModel):
    project_id: str
    board_id: str
    version: int = 0
    """0 means the board has never been written; PUT with expected_version=0 creates it."""
    columns: list[BoardColumn] = Field(default_factory=list)
    cards: list[BoardCard] = Field(default_factory=list)


class AnalysisTableView(BaseModel):
    """One AnalysisTable artifact reshaped for display — rows verbatim."""

    artifact_id: str
    dataset_id: str
    dataset_name: str
    title: str
    kind: str
    """numeric_summary | correlation"""
    description: str
    question: str
    """Source question resolved through artifact lineage, or the baseline disclosure."""
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    trivial_rows: list[dict[str, Any]] = Field(default_factory=list)
    """Correlation pairs flagged is_trivial_pair; empty for other kinds."""
    min_sample_size: int | None = None
    small_sample: bool = False


class StatTestRow(BaseModel):
    """One StatTestResult; p_value/statistic stay at stored precision."""

    artifact_id: str
    dataset_id: str
    dataset_name: str
    test_type: str
    group_column: str | None = None
    value_column: str | None = None
    statistic: float | None = None
    p_value: float | None = None
    p_value_display: str = ""
    effect_size: float | None = None
    effect_size_magnitude: str | None = None
    degrees_of_freedom: int | None = None
    sample_size: int = 0
    significant: bool | None = None
    """p_value < 0.05; None when the test reported no p-value."""
    conclusion: str = ""
    groups: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    small_sample: bool = False


class ModelCardFeature(BaseModel):
    feature: str
    importance: float


class ModelCardLeakageCheck(BaseModel):
    code: str
    severity: str
    column: str | None = None
    action: str
    message: str


class ModelCardView(BaseModel):
    artifact_id: str
    dataset_id: str
    dataset_name: str
    task_type: str
    target_column: str
    model_type: str
    split_strategy: str
    train_rows: int = 0
    test_rows: int = 0
    feature_columns: list[str] = Field(default_factory=list)
    excluded_features: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    headline_metric: str | None = None
    headline_metric_value: float | None = None
    baseline_accuracy: float | None = None
    leakage_verdict: str = "unknown"
    leakage_checks: list[ModelCardLeakageCheck] = Field(default_factory=list)
    feature_importance: list[ModelCardFeature] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class AnalysisView(BaseModel):
    """Deep analysis read model (§10.2 P1) — shaped from persisted artifacts only."""

    session_id: str
    tables: list[AnalysisTableView] = Field(default_factory=list)
    stat_tests: list[StatTestRow] = Field(default_factory=list)
    model_cards: list[ModelCardView] = Field(default_factory=list)


class SessionStepMetricRow(BaseModel):
    step_name: str
    llm_calls: int = 0
    tool_calls: int = 0
    tokens: int = 0
    duration_seconds: float = 0.0


class UsageDay(BaseModel):
    """One UTC calendar day of the activity window."""

    date: str
    sessions: int = 0


class UsageRecentSession(BaseModel):
    """A session on the home page's cross-project recency list."""

    session_id: str
    project_id: str
    title: str | None = None
    status: str = "unknown"
    created_at: datetime | None = None
    updated_at: datetime | None = None


class WorkspaceUsageView(BaseModel):
    """Period-scoped cross-run rollup for the workspace home.

    Every other trace figure is per-run, so a dashboard had to fan out one
    request per session to say anything about the workspace. Cost is summed
    only over sessions that persisted a metrics artifact; `unpriced_sessions`
    reports the rest rather than letting them vanish into the total.
    ``project_count``, ``data_bytes``, and ``recent`` remain workspace-wide;
    session-derived figures are scoped by last activity to ``window_days``.
    """

    schema_version: int = 1
    generated_at: datetime
    window_days: int
    project_count: int = 0
    session_count: int = 0
    status_counts: dict[str, int] = Field(default_factory=dict)
    daily: list[UsageDay] = Field(default_factory=list)
    llm_calls: int = 0
    total_tokens: int = 0
    est_cost_usd: float = 0.0
    priced_sessions: int = 0
    unpriced_sessions: int = 0
    artifact_count: int = 0
    dataset_count: int = 0
    profiled_rows: int = 0
    data_bytes: int = 0
    # Non-zero only if a project held more runs than one sweep reads. The
    # figures above are then partial, and say so rather than looking complete.
    truncated_sessions: int = 0
    # Served here rather than fetched per project by the client: the rollup
    # already walks every run, so the home page needs one request, not one per
    # project.
    recent: list[UsageRecentSession] = Field(default_factory=list)


class SessionMetricsView(BaseModel):
    """Trace & cost rollup. `source` is "artifact" when the run persisted a
    SessionMetrics artifact, "aggregated" when it was recomputed from trace events."""

    schema_version: int = 5
    session_id: str
    source: str
    llm_calls: int = 0
    tool_calls: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cache_creation_tokens: int = 0
    reasoning_tokens: int = 0
    cache_hit_rate: float = 0.0
    est_cost_usd: float | None = None
    costed_calls: int = 0
    uncosted_calls: int = 0
    usage_known_calls: int = 0
    usage_unknown_calls: int = 0
    cost_estimate_status: Literal[
        "complete_estimate", "partial_estimate", "unavailable", "not_applicable"
    ] = "not_applicable"
    llm_calls_by_model: dict[str, int] = Field(default_factory=dict)
    llm_calls_by_kind: dict[str, int] = Field(default_factory=dict)
    llm_calls_by_task: dict[str, int] = Field(default_factory=dict)
    llm_calls_by_status: dict[str, int] = Field(default_factory=dict)
    budget_est_cost_usd: float | None = None
    budget_reserved_calls: int = 0
    budget_settled_calls: int = 0
    budget_rejected_calls: int = 0
    budget_uncertain_calls: int = 0
    budget_total_tokens: int = 0
    budget_reconciliation: Literal[
        "verified", "unverifiable", "not_applicable"
    ] = "not_applicable"
    duration_seconds: float = 0.0
    event_count: int = 0
    trace_status: Literal["verified", "unverifiable"] = "verified"
    failures_count: int = 0
    findings_count: int = 0
    report_gate_verdict: str | None = None
    publication_readiness: PublicationReadiness = "draft"
    publication_freshness: PublicationFreshness = "not_applicable"
    question_llm_skipped: bool = False
    question_proposals_dropped: int = 0
    question_dataset_names_resolved: int = 0
    question_list_coercions: int = 0
    degraded: bool = False
    semantic_bootstrap_degraded: bool = False
    column_roles_unverified: int = 0
    template_backstop_used: int = 0
    join_candidates_proposed: int = 0
    join_authorizations_fresh: int = 0
    join_authorizations_stale: int = 0
    join_authorizations_unverifiable: int = 0
    relationship_overlap_pairs_evaluated: int = 0
    relationship_overlap_pairs_prefiltered: int = 0
    relationship_full_validations: int = 0
    relationship_coverage_limited: bool = False
    relationship_candidate_payload_bytes: int = 0
    relationship_discovery_deferred: bool = False
    semantic_degraded_claims: int = 0
    time_boundary_truncations: int = 0
    numeric_unverified_claims: int = 0
    quantitative_coverage_gaps: int = 0
    evidence_interleave_granted: int = 0
    evidence_interleave_rejected: int = 0
    findings_dedup_clusters: int = 0
    findings_dedup_merged: int = 0
    domain_metric_questions: int = 0
    domain_metrics_skipped: int = 0
    macro_loop_rounds: int = 0
    macro_loop_new_findings: int = 0
    macro_loop_discard_rounds: int = 0
    question_answered: int = 0
    question_abstained: int = 0
    question_failed: int = 0
    question_awaiting_approval: int = 0
    result_contract_failures: dict[str, int] = Field(default_factory=dict)
    interpretation_validated: int = 0
    interpretation_fallbacks: int = 0
    report_eligible_findings: int = 0
    question_answer_rate: float = 0.0
    question_abstention_rate: float = 0.0
    tokens_per_answered_question: float = 0.0
    seconds_per_answered_question: float = 0.0
    report_token_share: float = 0.0
    report_duration_share: float = 0.0
    coverage_limited: bool = False
    publication_blocked: bool = False
    steps: list[SessionStepMetricRow] = Field(default_factory=list)
    artifact_counts: dict[str, int] = Field(default_factory=dict)
    generated_at: datetime | None = None


class TraceEventRow(BaseModel):
    event_id: int
    event_type: str
    name: str
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_seconds: float | None = None
    summary: dict[str, Any] = Field(default_factory=dict)


class TraceEventPage(BaseModel):
    """Trace page plus the run's event-type histogram, so the client filter does
    not need a second request."""

    session_id: str
    items: list[TraceEventRow] = Field(default_factory=list)
    next_cursor: str | None = None
    event_types: dict[str, int] = Field(default_factory=dict)
    total: int = 0


class ClientFailureRequest(BaseModel):
    """Privacy-minimal handled UI failure. Free text and request data are
    deliberately absent; extra keys are rejected instead of silently stored."""

    model_config = ConfigDict(extra="forbid")

    error_code: Literal[
        "access_forbidden",
        "client_error",
        "conflict",
        "http_error",
        "network_error",
        "not_found",
        "rate_limited",
        "server_error",
        "validation_error",
    ]
    operation: Literal["mutation", "render", "retry"]
    dedupe_key: str = Field(
        min_length=36,
        max_length=36,
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    )


class ClientFailureRecorded(BaseModel):
    event_type: Literal["failure_recorded"] = "failure_recorded"
    recorded: bool


class SessionDebugSummary(BaseModel):
    events: int = 0
    artifacts: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    errors: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class ReportQualitySummary(BaseModel):
    section_coverage: float = 0.0
    claim_section_coverage: float = 0.0
    claim_survival_rate: float = 0.0
    deterministic_repair_count: int = 0
    prompt_tokens_by_attempt: str = ""


class DebugTimelineRow(BaseModel):
    event_type: str
    name: str
    started_at: str
    duration_ms: int | None = None
    summary: str = ""


class DebugLlmCallRow(BaseModel):
    """One billed call. "schema" is aliased because the bare name shadows a
    BaseModel attribute."""

    model_config = ConfigDict(populate_by_name=True)

    task: str = ""
    provider: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float | None = None
    schema_: str = Field("", alias="schema")
    duration_ms: int | None = None
    status: str = ""
    attempt: str = ""
    error_type: str = ""
    error: str = ""


class DebugToolCallRow(BaseModel):
    event_type: str
    tool: str
    duration_ms: int | None = None
    row_count: int | None = None
    truncated: bool | None = None
    artifact_id: str = ""
    summary: str = ""


class DebugErrorRow(BaseModel):
    event_type: str
    name: str
    error_type: str = ""
    error: str = ""


class DebugArtifactRow(BaseModel):
    artifact_id: str
    type: str
    parents: int = 0
    warnings: int = 0


class SessionDebugView(BaseModel):
    """Developer-inspector rollup for one session (Trace / debug panel)."""

    session_id: str
    code_version: str | None = None
    summary: SessionDebugSummary
    report_quality: ReportQualitySummary
    timeline: list[DebugTimelineRow] = Field(default_factory=list)
    llm_calls: list[DebugLlmCallRow] = Field(default_factory=list)
    tool_calls: list[DebugToolCallRow] = Field(default_factory=list)
    errors: list[DebugErrorRow] = Field(default_factory=list)
    artifacts: list[DebugArtifactRow] = Field(default_factory=list)
    next_cursor: str | None = None


class LlmDebugRecord(BaseModel):
    """One captured LLM call from llm_debug.jsonl. Previews are truncated at
    capture time and redacted again on the way out."""

    index: int
    ts: str = ""
    kind: str = ""
    # What actually went over the wire; differs from kind when a provider has no
    # native structured mode and the schema rides a forced tool call.
    transport_kind: str = ""
    task: str = ""
    provider: str = ""
    model: str = ""
    endpoint_host: str = ""
    status: str = ""
    finish_reason: str = ""
    duration_s: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    cache_creation_tokens: int | None = None
    reasoning_tokens: int | None = None
    estimated_cost_usd: float | None = None
    cost_basis: str = ""
    pricing_version: str = ""
    usage_reported: bool = True
    request_id: str = ""
    response_id: str = ""
    request_bytes: int | None = None
    response_bytes: int | None = None
    payload_preview: str = ""
    response_preview: str = ""


class SCQAView(BaseModel):
    situation: str
    complication: str
    question: str
    answer: str


class DecisionReportSectionView(BaseModel):
    title: str
    body: str
    finding_artifact_ids: list[str] = Field(default_factory=list)


class MetaInsightView(BaseModel):
    commonality_statements: list[str] = Field(default_factory=list)
    exception_statements: list[str] = Field(default_factory=list)


class CandidateDecisionView(BaseModel):
    """A source finding's proposed action. Hypothesis context supplied by the
    model/template — never validated evidence, and labeled as such in the UI."""

    finding_artifact_id: str
    question: str
    decision_action: str
    decision_readiness: str
    analytical_reliability: str
    report_readiness: str


class DecisionEvidenceRefView(BaseModel):
    artifact_id: str
    kind: str
    locator: str = ""
    session_id: str | None = None
    """The run whose Artifacts page can show it; None when it is not navigable."""


class DecisionReportFreshnessView(BaseModel):
    status: str
    reasons: list[str] = Field(default_factory=list)


class DecisionStoryFindingView(BaseModel):
    """One report-eligible finding offered for curation, including freshness
    (clients may hide or surface stale items)."""

    artifact_id: str
    source_session_id: str
    question: str
    analytical_reliability: str
    report_readiness: str
    freshness: str
    """fresh | stale | unverifiable"""


class DecisionStoryBeatView(BaseModel):
    title: str
    body: str


class DecisionStoryDraftView(BaseModel):
    """One persisted synthesis brief. `session_id` is the driver's own synthesis
    run, never the job's lifecycle run."""

    artifact_id: str
    session_id: str
    created_at: datetime
    brief_id: str
    headline: str
    decision_context: str
    storyline: list[DecisionStoryBeatView] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    investigation_gaps: list[str] = Field(default_factory=list)
    business_context: str = ""
    """Unverified user framing; never a source for claims or numbers."""
    report_eligible: bool = False
    report_readiness: str = "not_eligible"
    selected_finding_artifact_ids: list[str] = Field(default_factory=list)


class DecisionStoryView(BaseModel):
    """Curation surface: what may be selected, and what has been drafted."""

    session_id: str
    project_id: str
    eligible_findings: list[DecisionStoryFindingView] = Field(default_factory=list)
    drafts: list[DecisionStoryDraftView] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class DecisionStoryDraftStarted(BaseModel):
    """Create-draft outcome: the derived lifecycle run plus the job running it."""

    session_id: str
    execution_session_id: str
    job: JobCreated


class DecisionReportGenerationStarted(BaseModel):
    session_id: str
    brief_artifact_id: str
    execution_session_id: str
    job: JobCreated


class DecisionReportView(BaseModel):
    """Role-3 decision report for the Report page header. status "none" means the
    project has no decision report yet — a normal 200, not an error."""

    session_id: str
    status: str
    """none | available"""
    artifact_id: str | None = None
    report_session_id: str | None = None
    report_id: str | None = None
    brief_id: str | None = None
    title: str | None = None
    generated_at: datetime | None = None
    scqa: SCQAView | None = None
    sections: list[DecisionReportSectionView] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    investigation_gaps: list[str] = Field(default_factory=list)
    meta_insight: MetaInsightView | None = None
    candidate_decisions: list[CandidateDecisionView] = Field(default_factory=list)
    evidence_refs: list[DecisionEvidenceRefView] = Field(default_factory=list)
    source_finding_artifact_ids: list[str] = Field(default_factory=list)
    granted_evidence_artifact_ids: list[str] = Field(default_factory=list)
    report_readiness: str | None = None
    narrative_status: str | None = None
    narrative_fallback_reason: str = ""
    freshness: DecisionReportFreshnessView | None = None
    publication_status: str | None = None
    """"published" whenever a decision report exists (§publication_state)."""
    gate_verdict: str | None = None
    """Worst report-gate verdict recorded for the viewed run, if audited."""
    confidence_label: str | None = None
    """Weakest analytical_reliability across the source findings (high|medium|low)."""
    export_available: bool = False
    """Export stays disabled unless freshness is "fresh" (mirrors the legacy UI)."""


class RelationshipNode(BaseModel):
    """One dataset of the run, as a node of the relationship graph."""

    dataset_id: str
    name: str
    row_count: int | None = None
    column_count: int = 0
    source_available: bool = True
    """False when the CSV behind the profile is gone: validation cannot run."""


class RelationshipEdge(BaseModel):
    """One directed join hypothesis with everything the Inspector shows.

    `state` is the render tier: candidate (scored only) → validated (DuckDB
    verified) → confirmed (a user or auto-confirmation promoted the join).
    """

    relationship_id: str
    label: str
    state: str
    """candidate | validated | confirmed"""
    left_dataset_id: str
    left_dataset: str
    left_columns: list[str] = Field(default_factory=list)
    right_dataset_id: str
    right_dataset: str
    right_columns: list[str] = Field(default_factory=list)
    confidence: str = "low"
    ensemble_score: float = 0.0
    name_similarity: float = 0.0
    overlap_left_in_right: float = 0.0
    overlap_right_in_left: float = 0.0
    right_unique_rate: float = 0.0
    signals_sampled: bool = False
    verified: bool = False
    cardinality: str | None = None
    join_row_multiplier: float | None = None
    orphan_rate_left: float | None = None
    orphan_rate_right: float | None = None
    verification_sql: str = ""
    validation_sampled: bool = False
    warnings: list[str] = Field(default_factory=list)
    join_status: str | None = None
    """proposed | auto_confirmed | confirmed; None when no whitelist entry exists."""
    freshness: str | None = None
    """fresh | stale | unverifiable, against this run's dataset identities."""
    can_validate: bool = False
    can_confirm: bool = False
    can_revoke: bool = False
    candidate_artifact_id: str | None = None
    validation_artifact_id: str | None = None


class RelationshipGraphView(BaseModel):
    """Nodes + edges for one run, shaped from existing artifacts only."""

    session_id: str
    project_id: str
    seeds_version: int = 0
    discovered: bool = False
    """False when the run never produced a RelationshipCandidateSet."""
    nodes: list[RelationshipNode] = Field(default_factory=list)
    edges: list[RelationshipEdge] = Field(default_factory=list)
    coverage_status: str = "complete"
    coverage_reason: str = ""
    overlap_pairs_evaluated: int = 0
    overlap_pairs_prefiltered: int = 0
    truncated_pairs: int = 0


class RelationshipValidationPrepared(BaseModel):
    """Pending approval for one full relationship validation (§6.0 contract)."""

    session_id: str
    relationship_id: str
    action_hash: str
    approval_token: str
    expires_at: datetime
    label: str
    left_dataset: str
    right_dataset: str
    confidence: str
    uses_llm: bool = False
    """Always False: validation is DuckDB joins over the source tables."""


class RelationshipValidationStarted(BaseModel):
    """The derived run carrying the validation job, plus the job itself."""

    session_id: str
    relationship_id: str
    execution_session_id: str
    job: JobCreated


class SessionDeleted(BaseModel):
    session_id: str
    project_id: str
    deleted: bool
    """False when the run row existed but nothing was left on disk to remove."""


class ProviderInfo(BaseModel):
    """One LLM provider's static capabilities, straight from provider_registry."""

    provider: str
    display_name: str
    requires_api_key: bool
    requires_base_url: bool
    default_base_url: str = ""
    preset_models: list[str] = Field(default_factory=list)
    structured_mode: str = "auto"
    native: bool = False
    # How a request actually reaches this provider. Anthropic has no
    # response_format, so a schema rides a forced tool call — worth stating,
    # because it is why its structured calls are metered as provider_tool.
    text_transport: str = "messages"
    structured_transport: str = "response_format"
    # Data/SQL/Python tools are chosen and executed by this host, never handed
    # to the model as provider-native tools.
    tool_execution: str = "host_orchestrated"
    pricing_catalog_version: str = ""
    pricing_source_url: str = ""
    pricing_checked_at: str = ""
    capability_catalog_version: str = ""
    agent_model_count: int = 0


class ModelCatalogItem(BaseModel):
    """One model from the live provider catalog or the local fallback."""

    id: str
    owned_by: str = ""
    created: int | None = None
    input_usd_per_1m: float | None = None
    output_usd_per_1m: float | None = None
    cache_read_usd_per_1m: float | None = None
    cache_write_usd_per_1m: float | None = None
    pricing_source: str = ""
    capabilities: list[str] = Field(default_factory=list)
    parallel_tool_calling: bool = False
    structured_output: str = ""
    temperature_policy: str = ""
    verified: bool = False
    """This repo has checked the model against its tool loop. False means
    untested, not unusable — the run probes it before spending anything."""


class ModelCatalog(BaseModel):
    """Models visible to the saved credential, with freshness stated rather
    than implied. `source` is "live" when the provider answered and "snapshot"
    when this fell back to the built-in preset list."""

    provider: str
    models: list[ModelCatalogItem] = Field(default_factory=list)
    source: str
    fetched_at: datetime
    endpoint: str = ""
    warning: str = ""
    truncated: bool = False
    pricing_catalog_version: str = ""
    pricing_notice: str = "List prices for a rough estimate, not an invoice."


class AboutInfo(BaseModel):
    app_version: str
    workspace_is_default: bool
    workspace_label: str
    """Relativized on purpose (§14): the API never returns server file paths."""


class SettingsView(BaseModel):
    """Effective settings for the calling session. Carries no API key — only
    whether one is set and its last 4 characters."""

    version: int = 0
    provider: str
    model: str
    base_url: str = ""
    resolved_base_url: str = ""
    temperature: float = 0.2
    max_tokens: int = 6000
    timeout_seconds: float = 180.0
    structured_output_mode: str = "auto"
    payload_policy: str = "schema+aggregates"
    usd_per_1k_prompt: float = 0.0
    usd_per_1k_completion: float = 0.0
    """0 means "use the built-in model pricing defaults" (core.llm)."""
    analysis_depth: int = 0
    """Thinking level: 0 Standard, 1 Deep (deep investigation), 2+ Ultra (macro loop)."""
    dev_mode: bool = False
    api_key_set: bool = False
    api_key_last4: str = ""
    is_ready_for_live_calls: bool = False
    status_state: str = "offline"
    """ready | offline | incomplete"""
    status_message: str = ""
    missing_fields: list[str] = Field(default_factory=list)
    model_verified: bool = True
    """False when the chosen model is outside the verified catalog. It still
    runs; the tool-calling probe decides before the run spends anything."""
    warnings: list[str] = Field(default_factory=list)
    """Advisory, unlike missing_fields — these never block a save."""
    source: str = "env"
    """env when untouched this session, session once overridden."""
    about: AboutInfo


class SettingsPatch(BaseModel):
    """Partial update; omitted fields keep their current value.

    ``api_key`` is write-only and never echoed back — repr is suppressed so it
    cannot reach a log line through an exception or debug dump.
    """

    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    timeout_seconds: float | None = None
    structured_output_mode: str | None = None
    payload_policy: str | None = None
    usd_per_1k_prompt: float | None = None
    usd_per_1k_completion: float | None = None
    analysis_depth: int | None = None
    """0-3. Raising it to 1 enables deep investigation, 2+ additionally
    authorizes the macro loop's automatic follow-up rounds."""
    dev_mode: bool | None = None
    api_key: str | None = Field(default=None, repr=False)
    clear_api_key: bool | None = None
    """Explicit erase; a blank ``api_key`` means "keep what is stored"."""


class ConnectionTestResult(BaseModel):
    ok: bool
    provider: str
    model: str
    elapsed_ms: int = 0
    message: str = ""
    error_code: str = ""
    # The probe is a real paid call that belongs to no session, so its cost is
    # reported here instead of silently missing from every total.
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    estimated_cost_usd: float | None = None
    cost_basis: str = ""
    request_id: str = ""
    usage_reported: bool | None = None


class RelationshipDiscoveryStarted(BaseModel):
    """The derived run carrying an on-demand relationship discovery job.

    Discovery reads every source CSV, so it never runs on page load; the
    candidate artifacts it produces land on ``session_id`` (the source run), while
    ``execution_session_id`` is the ``rdsess_*`` run that only carries the lifecycle.
    """

    session_id: str
    execution_session_id: str
    job: JobCreated


class SupportDocView(BaseModel):
    """One user-supplied reference document under ``<project>/semantic/docs``."""

    doc_id: str
    """Derived from the stored name; the API never rebuilds a path from it."""
    name: str
    byte_size: int
    modified_at: datetime | None = None


class SupportDocList(BaseModel):
    project_id: str
    docs: list[SupportDocView] = Field(default_factory=list)
    next_cursor: str | None = None


class HealthStatusView(BaseModel):
    status: Literal["ok"] = "ok"


class SystemCapabilitiesView(BaseModel):
    """Optional-dependency features, so the UI can disable what this host
    cannot do instead of offering a button that 503s."""

    pdf_export_available: bool
    pdf_export_hint: str = ""
    """Install instructions; empty when PDF export already works."""


class SandboxStatusView(BaseModel):
    """Which execution backend untrusted analysis code would get right now."""

    backend: str
    """docker | none"""
    available: bool
    safe_for_untrusted_code: bool
    open_python_analysis_available: bool
    """available AND safe — the only combination CodeAgent may run under."""
    detail: str = ""
    message: str = ""


class PrecleaningOptions(BaseModel):
    """Opt-in pre-clean applied before ingest. Originals are never rewritten:
    the cleaned frame becomes a new dataset version and every drop is recorded
    as a CleaningRecipe artifact."""

    clean_missing_values: bool = False
    missing_threshold_percent: float = Field(default=70.0, ge=0.0, le=100.0)
    min_rows_keep_percent: float = Field(default=50.0, ge=0.0, le=100.0)
    drop_iqr_outliers: bool = False

    @property
    def enabled(self) -> bool:
        return self.clean_missing_values or self.drop_iqr_outliers


class KnowledgePromotionPrepared(BaseModel):
    """Preview of the VerifiedAnswer a promotion would write into the project's
    semantic seeds, plus the one-time approval that authorises the write."""

    session_id: str
    finding_id: str
    action_hash: str
    approval_token: str
    expires_at: datetime
    question: str
    answer: str
    evidence_note: str
    replaces_existing: bool = False
    """True when a verified answer for the same question is already stored."""


class KnowledgePromoted(BaseModel):
    session_id: str
    finding_id: str
    question: str
    answer: str
    verified_answer_count: int


class ReportGenerationStarted(BaseModel):
    """The derived run carrying an on-demand report generation job.

    The report artifacts land on ``session_id`` (the source run); ``execution_session_id``
    is the ``rpsess_*`` run that only carries the job lifecycle.
    """

    session_id: str
    execution_session_id: str
    regenerated: bool
    """True when the run already had a report that this job will replace."""
    job: JobCreated


class SessionForkStarted(BaseModel):
    """The derived run carrying a what-if fork job.

    ``execution_session_id`` is the ``fksess_*`` lifecycle run; the forked analysis
    mints its own ordinary run id, which the job reports in its ``session.forked``
    trace event once the fork starts producing it.
    """

    session_id: str
    execution_session_id: str
    decision: str
    """Human-readable summary of the single varied decision."""
    job: JobCreated


class QuestionDraftPrepared(BaseModel):
    """Pending approval for drafting one question card from free text.

    Drafting calls the model, so it goes through the same two-step approval as
    execution; the approval binds the exact prompt, not a request id.
    """

    session_id: str
    action_hash: str
    approval_token: str
    expires_at: datetime
    question: str
    llm_mode: str = "env"


class QuestionDraftStarted(BaseModel):
    """The derived ``qdsess_*`` run carrying a card-drafting job.

    The new candidate is appended to the SOURCE run's QuestionCandidateSet; the
    derived run only carries the job lifecycle.
    """

    session_id: str
    execution_session_id: str
    question: str
    job: JobCreated


class InvestigationGateView(BaseModel):
    name: str
    status: str
    """passed | warning | failed"""
    reason: str = ""


class InvestigationPlanView(BaseModel):
    """One investigation plan plus its decision and terminal outcome."""

    plan_id: str
    """Artifact id of the InvestigationPlan; approval targets this artifact."""
    plan_session_id: str
    investigation_id: str
    question_id: str
    question: str
    method_family: str = ""
    method_recipe: str = ""
    card_version: int = 1
    status: str = "pending"
    """pending | approved | rejected | executed"""
    plan_status: str = "planned"
    """The plan artifact's own status (planned | needs_data | ...)."""
    execution_ready: bool = False
    allowed_tools: list[str] = Field(default_factory=list)
    target_datasets: list[str] = Field(default_factory=list)
    method_requirements: list[str] = Field(default_factory=list)
    validation_gates: list[InvestigationGateView] = Field(default_factory=list)
    candidate_fingerprint: str = ""
    deep_investigation: bool = False
    """True when the plan carries the deep-probe marker tool."""
    decision_reason: str = ""
    outcome_status: str | None = None
    """InvestigationRecord.status once execution produced one."""
    outcome_reason: str = ""
    finding_texts: list[str] = Field(default_factory=list)
    report_readiness: str | None = None
    can_approve: bool = False
    can_reject: bool = False
    can_execute: bool = False


class MacroLoopRoundView(BaseModel):
    round_id: int
    new_validated_findings: int = 0
    redundant_findings: int = 0
    discarded_findings: int = 0
    executed_questions: int = 0
    tokens: int = 0
    exit_reason: str = "continue"
    disposition: str = "keep"


class MacroLoopView(BaseModel):
    """One persisted macro-loop ledger for a plan run."""

    plan_session_id: str
    depth: int = 0
    rounds: list[MacroLoopRoundView] = Field(default_factory=list)
    admitted_finding_count: int = 0
    total_tokens: int = 0
    exit_reason: str = ""


class InvestigationsView(BaseModel):
    """Every plan derived from this run, newest plan run first."""

    session_id: str
    project_id: str
    analysis_depth: int = 0
    deep_investigation_enabled: bool = False
    macro_loop_authorized: bool = False
    """True when the session's thinking level authorizes the macro loop."""
    plans: list[InvestigationPlanView] = Field(default_factory=list)
    macro_loops: list[MacroLoopView] = Field(default_factory=list)


class InvestigationPlanBuildStarted(BaseModel):
    """The derived ``ipsess_*`` run carrying a plan-building job.

    Plan building is deterministic and spends no model budget, so it needs no
    approval; the plans it writes land on their own ``investigation_*`` run,
    which the job reports in an ``investigation.planned`` trace event.
    """

    session_id: str
    execution_session_id: str
    question_ids: list[str] = Field(default_factory=list)
    deep: bool = False
    job: JobCreated


class InvestigationDecisionPrepared(BaseModel):
    """Pending approval for approving or rejecting one plan.

    The approval binds the plan's content fingerprint: a plan rebuilt after
    this call no longer matches and the decision fails closed.
    """

    session_id: str
    plan_id: str
    plan_session_id: str
    decision: str
    """approved | rejected"""
    reason: str = ""
    action_hash: str
    approval_token: str
    expires_at: datetime
    plan: InvestigationPlanView


class InvestigationDecisionRecorded(BaseModel):
    session_id: str
    plan_id: str
    decision: str
    approval_artifact_id: str
    plan: InvestigationPlanView


class InvestigationExecutionPrepared(BaseModel):
    """Pending approval for executing a set of approved plans.

    Execution runs the approved methods and interprets findings with the model,
    so the approval binds the exact plan set and the LLM mode.
    """

    session_id: str
    plan_session_id: str
    plan_ids: list[str] = Field(default_factory=list)
    action_hash: str
    approval_token: str
    expires_at: datetime
    llm_mode: str = "env"
    plans: list[InvestigationPlanView] = Field(default_factory=list)


class InvestigationExecutionStarted(BaseModel):
    """The derived ``ixsess_*`` run carrying a plan-execution job.

    Findings and records land on the plan run; the derived run carries only the
    lifecycle, so a failed execution never flips the source run to failed.
    """

    session_id: str
    plan_session_id: str
    execution_session_id: str
    plan_ids: list[str] = Field(default_factory=list)
    job: JobCreated


class MacroLoopPrepared(BaseModel):
    """Pending approval for the Ultra macro loop over one executed plan run.

    Approving this authorizes the loop to generate and execute follow-up
    questions on its own for up to ``rounds_cap`` rounds, spending model budget
    without a further prompt.
    """

    session_id: str
    plan_session_id: str
    action_hash: str
    approval_token: str
    expires_at: datetime
    depth: int
    rounds_cap: int
    questions_per_round: int
    llm_mode: str = "env"


class MacroLoopStarted(BaseModel):
    """The derived ``mlsess_*`` run carrying a macro-loop job."""

    session_id: str
    plan_session_id: str
    execution_session_id: str
    depth: int
    rounds_cap: int
    job: JobCreated


class CustomChartRequest(BaseModel):
    """One ad-hoc chart over a run's dataset. Column names are validated against
    the dataset before they reach the spec; `y_column` null means row count."""

    dataset_id: str = Field(min_length=1)
    chart_type: Literal["bar", "line", "point", "area", "histogram"]
    x_column: str = Field(min_length=1)
    y_column: str | None = None
    color_column: str | None = None
    aggregate: Literal["none", "count", "mean", "sum", "median"] | None = None
    """None picks the default: sum for a numeric Y, count otherwise."""
    drop_missing: bool = True
    drop_outliers: bool = False


class CustomChartView(BaseModel):
    """A ready-to-render vega-lite spec with its rows inlined at
    `spec.data.values`. `row_count` 0 is the empty state, not an error."""

    session_id: str
    dataset_id: str
    chart_type: str
    aggregate: str
    row_count: int
    source_row_count: int
    """Rows after the cleaning switches, before the inline cap."""
    truncated: bool
    row_limit: int
    spec: dict[str, Any]


class DecisionCoverageView(BaseModel):
    """Deterministic coverage across every run of the viewed run's project,
    computed on read. `top_cards_total` 0 means nothing to assess yet."""

    session_id: str
    project_id: str
    top_cards_total: int
    top_cards_terminal: int
    uninvestigated_high_value: list[str] = Field(default_factory=list)
    findings_not_eligible: int
    validated_findings: int
    coverage_ready: bool
    gaps: list[str] = Field(default_factory=list)
