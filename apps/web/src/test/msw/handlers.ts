/* Default MSW handlers: a small consistent workspace (project p1 → run r1 →
 * dataset "sample") so shell/routing tests render real-shaped data. Tests
 * override individual endpoints with server.use(...). */

import { http, HttpResponse } from "msw";
import type {
  AnalysisView,
  ArtifactDetail,
  ArtifactPage,
  ArtifactSummary,
  BoardView,
  DecisionReportView,
  SessionMetricsView,
  TraceEventPage,
  TraceEventRow,
  ChatMessageAccepted,
  ChatMessagePage,
  CompareView,
  CompareScopeName,
  CompareScopeView,
  FindingsView,
  SemanticView,
  RelationshipGraphView,
  RelationshipValidationPrepared,
  RelationshipValidationStarted,
  SkillReplayPrepared,
  SkillReplayStarted,
  SkillsView,
  ChartPage,
  ChartSummary,
  ChartView,
  CleaningApplied,
  CleaningPreviewResult,
  DatasetHandle,
  DatasetPreview,
  DatasetSchema,
  JobCreated,
  JobStatus,
  ProfilesView,
  ProjectSummary,
  QualityView,
  InvestigationDecisionPrepared,
  InvestigationPlanView,
  InvestigationsView,
  QuestionExecutionPrepared,
  QuestionExecutionStarted,
  QuestionsView,
  SessionDetail,
  SessionPage,
  SessionDeleted,
  SettingsView,
  SettingsPatch,
  ProviderInfo,
  ConnectionTestResult,
  UploadStatus,
  SandboxStatusView,
  SupportDocList,
  SupportDocView,
  SystemCapabilitiesView,
  CleaningLogView,
  CleaningRawView,
  ColumnDistributionsView,
  CustomChartRequest,
  CustomChartView,
  DataOperationStarted,
  DecisionCoverageView,
  DecisionStoryView,
  VerifiedRelationsUpdated,
  LlmDebugPage,
  SessionDebugView,
} from "../../api/client";

export const SAMPLE_TOTAL_ROWS = 250;
const dataOperationResults = new Map<string, unknown>();
const dataOperationFailures = new Map<
  string,
  { code: string; message: string }
>();

/* Module-level like the other fixture state, so a queued result or failure must
 * be cleared between tests or it satisfies an unrelated job poll. */
export function resetDataOperations(): void {
  dataOperationResults.clear();
  dataOperationFailures.clear();
}

function dataOperationStarted(
  sessionId: string,
  jobId: string,
): DataOperationStarted {
  return {
    session_id: sessionId,
    execution_session_id: `dop_${jobId}`,
    job: {
      job_id: jobId,
      session_id: `dop_${jobId}`,
      status: "queued",
      events_url: `/api/v1/jobs/${jobId}/events`,
    },
  };
}

export function queueDataOperation(
  sessionId: string,
  jobId: string,
  result: unknown,
) {
  dataOperationResults.set(jobId, result);
  return HttpResponse.json(dataOperationStarted(sessionId, jobId), { status: 202 });
}

export function queueFailedDataOperation(
  sessionId: string,
  jobId: string,
  code: string,
  message: string,
) {
  dataOperationFailures.set(jobId, { code, message });
  return HttpResponse.json(dataOperationStarted(sessionId, jobId), { status: 202 });
}

const sampleColumns = [
  { name: "id", dtype: "int64" },
  { name: "name", dtype: "string" },
  { name: "value", dtype: "float64" },
];

export function sampleRows(offset: number, limit: number): unknown[][] {
  const end = Math.min(offset + limit, SAMPLE_TOTAL_ROWS);
  return Array.from({ length: Math.max(end - offset, 0) }, (_, i) => {
    const n = offset + i;
    return [n, `row-${n}`, n * 1.5];
  });
}

export function uploadedDataset(name: string): DatasetHandle {
  return {
    dataset_id: `ds_${name.replace(/\.csv$/i, "")}`,
    project_id: "p1",
    display_name: name,
    original_uri: `upload://${name}`,
    format: "csv",
    content_hash: "cafebabe",
    byte_size: 2048,
    row_count: 42,
    schema: sampleColumns,
    ingest_status: "ready",
  };
}

export function jobStatus(
  jobId: string,
  patch: Partial<JobStatus> = {},
): JobStatus {
  return {
    job_id: jobId,
    session_id: "r_new",
    project_id: "p1",
    kind: "auto_eda",
    status: "running",
    cancel_requested: false,
    created_at: "2026-07-25T10:00:00Z",
    started_at: "2026-07-25T10:00:01Z",
    finished_at: null,
    error_code: null,
    error_message: null,
    events_url: `/api/v1/jobs/${jobId}/events`,
    ...patch,
  };
}

export const SAMPLE_REPORT_MARKDOWN = [
  "# Demo report",
  "",
  "Revenue grew steadily across the quarter.",
  "",
  "| Segment | Revenue |",
  "|---------|---------|",
  "| North | 1200 |",
  "| South | 800 |",
  "",
  "```sql",
  "select segment, sum(revenue) from orders group by 1",
  "```",
].join("\n");

export const SAMPLE_ARTIFACTS: ArtifactSummary[] = [
  { artifact_id: "chart_1", type: "ChartSpec", created_at: null },
  { artifact_id: "chart_2", type: "ChartSpec", created_at: null },
  { artifact_id: "prof_1", type: "DatasetProfile", created_at: null },
  { artifact_id: "quality_1", type: "QualityIssueSet", created_at: null },
  { artifact_id: "chart_3", type: "ChartSpec", created_at: null },
];

export const SAMPLE_QUALITY: QualityView = {
  session_id: "r1",
  critical: 1,
  warn: 1,
  info: 1,
  datasets: [
    { dataset_name: "sample.csv", dataset_id: "sample", critical: 1, warn: 1, info: 0 },
    { dataset_name: "other.csv", dataset_id: "other", critical: 0, warn: 0, info: 1 },
  ],
  issues: [
    {
      severity: "critical",
      dataset_name: "sample.csv",
      dataset_id: "sample",
      code: "empty_column",
      column: "value",
      message: "Column value is empty.",
      recommendation: "Drop the empty column.",
    },
    {
      severity: "warn",
      dataset_name: "sample.csv",
      dataset_id: "sample",
      code: "high_missing",
      column: "name",
      message: "Column name has 40% missing values.",
      recommendation: "Review missingness.",
    },
    {
      severity: "info",
      dataset_name: "other.csv",
      dataset_id: "other",
      code: "likely_id",
      column: "id",
      message: "Column id looks like an identifier.",
      recommendation: "Exclude from aggregation.",
    },
  ],
};

export const SAMPLE_PROFILES: ProfilesView = {
  session_id: "r1",
  datasets: [
    {
      dataset_id: "sample",
      name: "sample.csv",
      rows: 250,
      columns: 3,
      semantic_type_counts: { id: 1, categorical: 1, numeric: 1 },
      fields: [
        {
          column: "id",
          dtype: "int64",
          semantic_type: "id",
          missing_percent: 0,
          unique_percent: 100,
          sample_values: "1, 2, 3",
        },
        {
          column: "name",
          dtype: "string",
          semantic_type: "categorical",
          missing_percent: 40,
          unique_percent: 12,
          sample_values: "row-1, row-2",
        },
        {
          column: "value",
          dtype: "float64",
          semantic_type: "numeric",
          missing_percent: 100,
          unique_percent: null,
          sample_values: "",
        },
      ],
    },
  ],
};

export const SAMPLE_CHART_SUMMARIES: ChartSummary[] = [
  {
    artifact_id: "chart_1",
    title: "Value by name",
    dataset_id: "sample",
    dataset_name: "sample.csv",
    mark: "bar",
    fields: ["name", "value"],
    description: "Demo bar chart.",
  },
  {
    artifact_id: "chart_2",
    title: "Value over id",
    dataset_id: "sample",
    dataset_name: "sample.csv",
    mark: "line",
    fields: ["id", "value"],
    description: "Demo line chart.",
  },
];

export function chartView(chartId: string): ChartView {
  const summary = SAMPLE_CHART_SUMMARIES.find(
    (item) => item.artifact_id === chartId,
  );
  return {
    artifact_id: chartId,
    session_id: "r1",
    title: summary?.title ?? chartId,
    dataset_id: "sample",
    dataset_name: "sample.csv",
    description: summary?.description ?? "",
    plain_language: `Plain reading of ${chartId}.`,
    spec: {
      $schema: "https://vega.github.io/schema/vega-lite/v6.json",
      title: summary?.title ?? chartId,
      mark: summary?.mark ?? "bar",
      encoding: {
        x: { field: summary?.fields?.[0] ?? "name", type: "nominal" },
        y: { field: summary?.fields?.[1] ?? "value", type: "quantitative" },
      },
      data: { values: [{ name: "a", value: 1 }] },
    },
  };
}

export const SAMPLE_ACTION_HASH = "a".repeat(64);
export const SAMPLE_APPROVAL_TOKEN = "c".repeat(32);

export function cleaningPreviewResult(
  sessionId: string,
  datasetId: string,
): CleaningPreviewResult {
  return {
    session_id: sessionId,
    dataset_id: datasetId,
    action_hash: SAMPLE_ACTION_HASH,
    approval_token: SAMPLE_APPROVAL_TOKEN,
    expires_at: "2026-07-25T12:00:00Z",
    operations: [
      {
        transform_id: "trim_0",
        type: "trim_whitespace",
        target_column: "name",
        description: "Trim leading/trailing whitespace in name.",
        lossy: false,
      },
      {
        transform_id: "dedupe",
        type: "drop_duplicate_rows",
        target_column: null,
        description: "Remove exact duplicate rows.",
        lossy: false,
      },
    ],
    preview: {
      dataset_id: datasetId,
      recipe_id: `api_${datasetId}`,
      source_version: 1,
      target_version: 2,
      row_count_before: 250,
      row_count_after: 245,
      rows_dropped: 5,
      rows_edited: 12,
      cells_changed: 12,
      column_changes: [
        {
          column: "name",
          before_dtype: "str",
          after_dtype: "str",
          changed_rows: 12,
          before_missing: 0,
          after_missing: 0,
        },
      ],
      warnings: [],
    },
  };
}

export function cleaningApplied(sessionId: string): CleaningApplied {
  return {
    session_id: sessionId,
    new_session_id: "run_cleaned_1",
    dataset_id: "sample",
    target_version: 2,
    job: {
      job_id: "job_clean_1",
      session_id: "run_cleaned_1",
      status: "queued",
      events_url: "/api/v1/jobs/job_clean_1/events",
    },
  };
}

export function questionsView(sessionId: string): QuestionsView {
  return {
    session_id: sessionId,
    questions: [
      {
        question_id: "q_trend",
        question: "How is value trending over time?",
        origin: "template",
        analysis_mode: "descriptive",
        value_category: "financial_performance",
        feasibility_status: "ready",
        proposed_action: "run_analysis",
        priority: 0.82,
        exploratory: false,
        target_datasets: ["sample.csv"],
        business_decision: "Plan next quarter's inventory",
        executable: true,
        card_version: 1,
        value_hypothesis: "",
        success_criterion: "",
        data_signal: "",
        priority_rationale: "",
        risks: [],
        data_requirements: [],
        execution: {
          outcome: "answered",
          status: "succeeded",
          findings_count: 2,
          qexec_artifact_id: "qexec_1",
          execution_session_id: "qsess_1",
          abstention_code: null,
        },
      },
      {
        question_id: "q_segment",
        question: "Can rows be segmented by value patterns?",
        origin: "llm",
        analysis_mode: "segmentation",
        value_category: "customer_or_entity",
        feasibility_status: "constrained",
        proposed_action: "run_analysis",
        priority: 0.55,
        exploratory: true,
        target_datasets: ["sample.csv"],
        business_decision: "",
        executable: true,
        card_version: 1,
        value_hypothesis: "",
        success_criterion: "",
        data_signal: "",
        priority_rationale: "",
        risks: [],
        data_requirements: [],
        execution: null,
      },
      {
        question_id: "q_blocked",
        question: "Can churn be predicted from labels we lack?",
        origin: "llm",
        analysis_mode: "prediction",
        value_category: null,
        feasibility_status: "needs_data",
        proposed_action: "collect_data",
        priority: 0.3,
        exploratory: true,
        target_datasets: ["sample.csv"],
        business_decision: "",
        executable: false,
        card_version: 1,
        value_hypothesis: "",
        success_criterion: "",
        data_signal: "",
        priority_rationale: "",
        risks: [],
        data_requirements: [],
        execution: null,
      },
    ],
  };
}

export function questionPrepared(
  sessionId: string,
  questionId: string,
  llmMode: string = "env",
): QuestionExecutionPrepared {
  return {
    session_id: sessionId,
    question_id: questionId,
    action_hash: SAMPLE_ACTION_HASH,
    approval_token: SAMPLE_APPROVAL_TOKEN,
    expires_at: "2026-07-25T12:00:00Z",
    question: "How is value trending over time?",
    origin: "template",
    sql_preview:
      llmMode === "offline"
        ? "select month, sum(value) from sample group by 1"
        : undefined,
    target_datasets: ["sample.csv"],
    uses_llm: llmMode !== "offline",
    llm_mode: llmMode,
  };
}

export function questionStarted(
  sessionId: string,
  questionId: string,
): QuestionExecutionStarted {
  return {
    session_id: sessionId,
    question_id: questionId,
    execution_session_id: "qsess_new_1",
    job: {
      job_id: "job_q_1",
      session_id: "qsess_new_1",
      status: "queued",
      events_url: "/api/v1/jobs/job_q_1/events",
    },
  };
}

export function findingsView(sessionId: string): FindingsView {
  return {
    session_id: sessionId,
    project_id: "p1",
    findings: [
      {
        artifact_id: "finding_1",
        source_session_id: "r_lib",
        source_session_navigable: true,
        from_current_session: false,
        created_at: "2026-07-20T10:00:00Z",
        question: "What was average order value?",
        claim_class: "observed",
        evidence_support: "high",
        analytical_reliability: "high",
        decision_readiness: "medium",
        report_readiness: "eligible_with_limitations",
        report_readiness_reason: "The descriptive claim is supported.",
        statements: [
          {
            text: "Average order value was $42.",
            evidence: [
              {
                kind: "table",
                artifact_id: "sqlres_1",
                locator: "rows[0]",
                session_id: "r_lib",
              },
            ],
          },
        ],
        limitations: ["Refunds were not identified."],
        interpretation: null,
        value_hypothesis: "Could increase profit.",
        method_artifact_id: null,
        freshness: {
          status: "fresh",
          reasons: ["The source datasets still match the saved finding."],
        },
      },
      {
        artifact_id: "finding_2",
        source_session_id: sessionId,
        source_session_navigable: true,
        from_current_session: true,
        created_at: "2026-07-21T10:00:00Z",
        question: "Do regions differ in revenue?",
        claim_class: "observed",
        evidence_support: "medium",
        analytical_reliability: "medium",
        decision_readiness: "low",
        report_readiness: "eligible",
        report_readiness_reason: "Supported.",
        statements: [{ text: "North leads revenue.", evidence: [] }],
        limitations: [],
        interpretation: null,
        value_hypothesis: null,
        method_artifact_id: null,
        freshness: {
          status: "stale",
          reasons: ["Dataset 'orders' changed since the finding was saved."],
        },
      },
    ],
    records: [
      {
        artifact_id: "record_open",
        source_session_id: "r_lib",
        from_current_session: false,
        question: "Is churn predictable?",
        status: "inconclusive",
        reason_code: "insufficient_signal",
        reason: "Not enough labelled data.",
        next_action: "Collect churn labels.",
      },
    ],
    warnings: [],
  };
}

export function semanticView(sessionId: string): SemanticView {
  return {
    session_id: sessionId,
    project_id: "p1",
    seeds_version: 3,
    field_meanings: [
      {
        dataset: "sample.csv",
        column: "value",
        meaning: "Gross order value before refunds.",
        unit: "USD",
        aliases: ["gmv"],
      },
    ],
    metric_definitions: [
      {
        name: "Active user",
        definition: "A user with at least one session in 28 days.",
        formula: "count(distinct user_id)",
        caveats: null,
      },
    ],
    entity_notes: [
      { name: "customer", note: "One row per billing account, not per person." },
    ],
    verified_answers: [
      {
        question: "What was Q3 revenue?",
        answer: "$4.2M, up 12% QoQ.",
        evidence_note: null,
        verified_at: "2026-01-02T03:04:05Z",
      },
    ],
    verified_relations: SAMPLE_VERIFIED_RELATIONS,
    column_roles: [
      { dataset: "sample.csv", column: "value", role: "measure", confidence: 0.9 },
    ],
    join_whitelist: [
      {
        seeds_version: 3,
        label: "sample.csv.id -> other.csv.sample_id",
        left_dataset: "sample.csv",
        left_columns: ["id"],
        right_dataset: "other.csv",
        right_columns: ["sample_id"],
        status: "proposed",
        cardinality: "one_to_many",
        validation_verified: true,
        freshness: "fresh",
        join_row_multiplier: 1.2,
        usage_count: 0,
        confidence_source: "relationship_discovery: confidence=high",
      },
      {
        seeds_version: 3,
        label: "sample.csv.ref -> other.csv.sample_id",
        left_dataset: "sample.csv",
        left_columns: ["ref"],
        right_dataset: "other.csv",
        right_columns: ["sample_id"],
        status: "auto_confirmed",
        cardinality: "one_to_one",
        validation_verified: true,
        freshness: "fresh",
        join_row_multiplier: 1.0,
        usage_count: 2,
        confidence_source: "relationship_discovery: confidence=high",
      },
    ],
    proposals: [
      {
        dataset: "sample.csv",
        column: "name",
        meaning: "Display name of the row.",
        unit_guess: "",
        confidence: "verified",
        source: "bootstrap",
      },
    ],
  };
}

export function compareView(left: string, right: string): CompareView {
  const value = <T,>(item: T) => ({ state: "value" as const, value: item });
  return {
    project_id: "p1",
    left: {
      session_id: left,
      project_id: "p1",
      title: "Baseline run",
      status: "complete",
      created_at: "2026-07-20T10:00:00Z",
      dataset_names: ["sample.csv"],
      artifact_count: 12,
      report_status: "final",
    },
    right: {
      session_id: right,
      project_id: "p1",
      title: "Cleaned run",
      status: "complete",
      created_at: "2026-07-22T10:00:00Z",
      dataset_names: ["sample.csv", "extra.csv"],
      artifact_count: 15,
      report_status: "draft",
    },
    metrics: [
      {
        key: "rows",
        label: "Total rows",
        left: value(250),
        right: value(245),
        delta: -5,
        optimization_direction: "none",
        verdict: "unknown",
        higher_is_better: null,
      },
      {
        key: "critical",
        label: "Critical issues",
        left: value(3),
        right: value(1),
        delta: -2,
        optimization_direction: "minimize",
        verdict: "improved",
        higher_is_better: false,
      },
      {
        key: "charts",
        label: "Charts",
        left: value(2),
        right: value(2),
        delta: null,
        optimization_direction: "none",
        verdict: "unchanged",
        higher_is_better: null,
      },
    ],
    text_rows: [
      {
        key: "report_status",
        label: "Report status",
        left: value("final"),
        right: value("draft"),
        changed: true,
      },
      {
        key: "ml_target",
        label: "ML target",
        left: {
          state: "not_applicable",
          reason: "This session produced no ML model.",
        },
        right: value("value"),
        changed: null,
      },
    ],
    artifact_deltas: [
      { type: "ChartSpec", left: value(2), right: value(2), delta: null },
      { type: "ModelCard", left: value(0), right: value(1), delta: 1 },
    ],
    datasets: {
      left: value(["sample.csv"]),
      right: value(["sample.csv", "extra.csv"]),
      shared: ["sample.csv"],
      only_left: [],
      only_right: ["extra.csv"],
    },
    comparability: {
      verdict: "partially_controlled",
      left: {
        input_hashes: value({ sample: "hash-a" }),
        code_version: value("v1"),
        seed: value(42),
        model_versions: value({ llm: "model-a" }),
        source_session_id: {
          state: "not_applicable",
          reason: "Root session",
        },
        prompt_template_version: {
          state: "missing",
          reason: "Not persisted",
        },
      },
      right: {
        input_hashes: value({ sample: "hash-a" }),
        code_version: value("v1"),
        seed: value(42),
        model_versions: value({ llm: "model-b" }),
        source_session_id: value(left),
        prompt_template_version: {
          state: "missing",
          reason: "Not persisted",
        },
      },
      changed_dimensions: ["model_versions"],
      unknown_dimensions: ["prompt_template_version"],
      warnings: [],
    },
    lineage: {
      relation: "direct_parent",
      common_ancestor_session_id: left,
      left_path: [left],
      right_path: [right, left],
      warnings: [],
    },
  };
}

export function compareScopeView(
  scope: CompareScopeName,
  left: string,
  right: string,
  differencesOnly = false,
): CompareScopeView {
  const side = (
    sessionId: string,
    title: string,
    value: string,
  ) => ({
    record_id: `${sessionId}:${scope}:shared`,
    title,
    kind: `${scope} record`,
    status: "ready",
    summary: `${scope} summary for ${title}`,
    source_session_id: sessionId,
    artifact_id: `${scope}_${sessionId}`,
    tags: [scope],
    evidence_ids: [],
    fields: [
      { key: "value", label: "Value", value, value_kind: "text" as const },
    ],
  });
  const changed = {
    match_key: `${scope}:changed`,
    matcher_version: "fixture-v1",
    reason: "same deterministic scope identity",
    confidence: "high" as const,
    match_status: "strong" as const,
    change: "changed" as const,
    left: side(left, `Baseline ${scope}`, "before"),
    right: side(right, `Variant ${scope}`, "after"),
    changed_fields: ["value"],
    warnings: [],
  };
  const same = {
    match_key: `${scope}:same`,
    matcher_version: "fixture-v1",
    reason: "same deterministic scope identity",
    confidence: "high" as const,
    match_status: "strong" as const,
    change: "same" as const,
    left: {
      ...side(left, `Shared ${scope}`, "same"),
      record_id: `${left}:${scope}:same`,
    },
    right: {
      ...side(right, `Shared ${scope}`, "same"),
      record_id: `${right}:${scope}:same`,
    },
    changed_fields: [],
    warnings: [],
  };
  return {
    project_id: "p1",
    scope,
    left: compareView(left, right).left,
    right: compareView(left, right).right,
    left_state: { state: "value", reason: null },
    right_state: { state: "value", reason: null },
    counts: { added: 0, removed: 0, changed: 1, same: 1, unavailable: 0 },
    items: differencesOnly ? [changed] : [changed, same],
    next_cursor: null,
    warnings: [],
  };
}

export function skillsView(sessionId: string): SkillsView {
  return {
    session_id: sessionId,
    project_id: "p1",
    skills: [
      {
        skill_id: "skill_saved_1",
        source: "library",
        name: "Revenue by name",
        description: "Saved from a validated chat plan.",
        question: "How does value vary by name?",
        sql: "SELECT name, SUM(value) AS total FROM sample GROUP BY 1",
        method: "aggregation",
        param_columns: ["name", "value"],
        expected_datasets: ["sample"],
        params: [],
        source_session_id: "r_lib",
        created_at: "2026-07-21T10:00:00Z",
      },
      {
        skill_id: "group_value_comparison",
        source: "seed",
        name: "Group totals and averages",
        description: "Totals per segment.",
        question: "How does {value_col} vary across {group_col} segments?",
        sql: "SELECT {group_col}, SUM({value_col}) FROM {dataset} GROUP BY 1",
        method: "aggregation",
        param_columns: [],
        expected_datasets: [],
        params: [
          {
            name: "group_col",
            role: "dimension",
            description: "Categorical column to group by.",
          },
          {
            name: "value_col",
            role: "measure",
            description: "Numeric column to aggregate.",
          },
        ],
        source_session_id: null,
        created_at: null,
      },
    ],
    datasets: [
      {
        dataset_id: "sample",
        name: "sample.csv",
        relation: "sample",
        columns: [
          { name: "id", bindable: true },
          { name: "name", bindable: true },
          { name: "value", bindable: true },
          { name: "unit price", bindable: false },
        ],
      },
    ],
    savable_plans: [
      {
        artifact_id: "plan_abc123",
        question: "How does value vary by name?",
        sql: "SELECT name, SUM(value) AS total FROM sample GROUP BY 1",
        method: "aggregation",
        dataset_names: ["sample"],
        columns: ["name", "value"],
        created_at: "2026-07-24T09:00:00Z",
      },
    ],
  };
}

export function relationshipGraph(sessionId: string): RelationshipGraphView {
  return {
    session_id: sessionId,
    project_id: "p1",
    seeds_version: 0,
    discovered: true,
    nodes: [
      {
        dataset_id: "sample",
        name: "sample.csv",
        row_count: 120,
        column_count: 3,
        source_available: true,
      },
      {
        dataset_id: "lookup",
        name: "lookup.csv",
        row_count: 4,
        column_count: 2,
        source_available: true,
      },
    ],
    edges: [
      {
        relationship_id: "rel_candidate",
        label: "sample.csv.code -> lookup.csv.code",
        state: "candidate",
        left_dataset_id: "sample",
        left_dataset: "sample.csv",
        left_columns: ["code"],
        right_dataset_id: "lookup",
        right_dataset: "lookup.csv",
        right_columns: ["code"],
        confidence: "high",
        ensemble_score: 0.95,
        name_similarity: 1,
        overlap_left_in_right: 1,
        overlap_right_in_left: 1,
        right_unique_rate: 1,
        signals_sampled: false,
        verified: false,
        cardinality: null,
        join_row_multiplier: null,
        orphan_rate_left: null,
        orphan_rate_right: null,
        verification_sql: "",
        validation_sampled: false,
        warnings: [],
        join_status: "proposed",
        freshness: "unverifiable",
        can_validate: true,
        can_confirm: false,
        can_revoke: false,
        candidate_artifact_id: "relcand_1",
        validation_artifact_id: null,
      },
      {
        relationship_id: "rel_validated",
        label: "sample.csv.lookup_id -> lookup.csv.lookup_id",
        state: "validated",
        left_dataset_id: "sample",
        left_dataset: "sample.csv",
        left_columns: ["lookup_id"],
        right_dataset_id: "lookup",
        right_dataset: "lookup.csv",
        right_columns: ["lookup_id"],
        confidence: "high",
        ensemble_score: 1,
        name_similarity: 1,
        overlap_left_in_right: 1,
        overlap_right_in_left: 1,
        right_unique_rate: 1,
        signals_sampled: false,
        verified: true,
        cardinality: "many_to_one",
        join_row_multiplier: 1,
        orphan_rate_left: 0,
        orphan_rate_right: 0,
        verification_sql: "select 1",
        validation_sampled: false,
        warnings: [],
        join_status: "proposed",
        freshness: "fresh",
        can_validate: false,
        can_confirm: true,
        can_revoke: false,
        candidate_artifact_id: "relcand_1",
        validation_artifact_id: "relval_1",
      },
    ],
    coverage_status: "complete",
    coverage_reason: "",
    overlap_pairs_evaluated: 2,
    overlap_pairs_prefiltered: 0,
    truncated_pairs: 0,
  };
}

export function relationshipPrepared(
  sessionId: string,
  relationshipId: string,
): RelationshipValidationPrepared {
  return {
    session_id: sessionId,
    relationship_id: relationshipId,
    action_hash: SAMPLE_ACTION_HASH,
    approval_token: SAMPLE_APPROVAL_TOKEN,
    expires_at: "2026-07-25T12:00:00Z",
    label: "sample.csv.code -> lookup.csv.code",
    left_dataset: "sample.csv",
    right_dataset: "lookup.csv",
    confidence: "high",
    uses_llm: false,
  };
}

export function relationshipStarted(
  sessionId: string,
  relationshipId: string,
): RelationshipValidationStarted {
  return {
    session_id: sessionId,
    relationship_id: relationshipId,
    execution_session_id: "rvsess_1",
    job: {
      job_id: "job_rel_1",
      session_id: "rvsess_1",
      status: "queued",
      events_url: "/api/v1/jobs/job_rel_1/events",
    },
  };
}

export function skillPrepared(
  sessionId: string,
  skillId: string,
): SkillReplayPrepared {
  return {
    session_id: sessionId,
    skill_id: skillId,
    action_hash: SAMPLE_ACTION_HASH,
    approval_token: SAMPLE_APPROVAL_TOKEN,
    expires_at: "2026-07-25T12:00:00Z",
    name: "Group totals and averages",
    question: "How does value vary across name segments?",
    sql_preview: "SELECT name, SUM(value) FROM sample GROUP BY 1",
    dataset_ids: ["sample"],
    dataset_names: ["sample.csv"],
    bindings: { group_col: "name", value_col: "value" },
    uses_llm: false,
  };
}

export function skillStarted(
  sessionId: string,
  skillId: string,
): SkillReplayStarted {
  return {
    session_id: sessionId,
    skill_id: skillId,
    execution_session_id: "ssess_new_1",
    job: {
      job_id: "job_s_1",
      session_id: "ssess_new_1",
      status: "queued",
      events_url: "/api/v1/jobs/job_s_1/events",
    },
  };
}

export function chatAccepted(
  sessionId: string,
  messageId: string = "msg_1",
): ChatMessageAccepted {
  return {
    session_id: sessionId,
    message_id: messageId,
    stream_url: `/api/v1/sessions/${sessionId}/chat/stream?message_id=${messageId}`,
  };
}

export function emptyBoard(projectId: string, boardId: string): BoardView {
  return {
    project_id: projectId,
    board_id: boardId,
    version: 0,
    columns: [],
    cards: [],
  };
}

export function analysisView(sessionId: string): AnalysisView {
  return {
    session_id: sessionId,
    tables: [
      {
        artifact_id: "table_summary",
        dataset_id: "sample",
        dataset_name: "sample.csv",
        title: "Numeric summary",
        kind: "numeric_summary",
        description: "Per-column descriptive statistics.",
        question: "Baseline EDA (not tied to a selected question card)",
        columns: ["column", "mean", "sample_size"],
        rows: [{ column: "value", mean: 12.5, sample_size: 250 }],
        trivial_rows: [],
        min_sample_size: 250,
        small_sample: false,
      },
      {
        artifact_id: "table_corr",
        dataset_id: "sample",
        dataset_name: "sample.csv",
        title: "Correlations",
        kind: "correlation",
        description: "Pairwise Pearson correlations.",
        question: "Which drivers move together?",
        columns: ["left", "right", "pearson"],
        rows: [{ left: "value", right: "total", pearson: 0.62 }],
        trivial_rows: [{ left: "value", right: "value_cents", pearson: 1 }],
        min_sample_size: 250,
        small_sample: false,
      },
    ],
    stat_tests: [
      {
        artifact_id: "stat_1",
        dataset_id: "sample",
        dataset_name: "sample.csv",
        test_type: "independent_t_test",
        group_column: "segment",
        value_column: "value",
        statistic: 5.82,
        p_value: 1.3e-8,
        p_value_display: "<0.001",
        effect_size: 0.44,
        effect_size_magnitude: "small",
        degrees_of_freedom: null,
        sample_size: 1470,
        significant: true,
        conclusion: "Significant at alpha=0.05 (small effect)",
        groups: { a: 1233, b: 237 },
        warnings: [],
        small_sample: false,
      },
    ],
    model_cards: [
      {
        artifact_id: "card_1",
        dataset_id: "sample",
        dataset_name: "sample.csv",
        task_type: "classification",
        target_column: "churn",
        model_type: "logistic_regression",
        split_strategy: "random_stratified",
        train_rows: 800,
        test_rows: 200,
        feature_columns: ["value", "tenure"],
        excluded_features: ["id"],
        metrics: { accuracy: 0.83 },
        headline_metric: "accuracy",
        headline_metric_value: 0.83,
        baseline_accuracy: 0.7,
        leakage_verdict: "mitigated",
        leakage_checks: [
          {
            code: "target_leak",
            severity: "critical",
            column: "id",
            action: "excluded",
            message: "Identifier dropped.",
          },
        ],
        feature_importance: [{ feature: "tenure", importance: 0.6 }],
        limitations: ["Single split; no cross-validation."],
      },
    ],
  };
}

export function runMetricsView(sessionId: string): SessionMetricsView {
  return {
    schema_version: 5,
    session_id: sessionId,
    source: "artifact",
    llm_calls: 22,
    tool_calls: 4,
    total_tokens: 68868,
    prompt_tokens: 33625,
    completion_tokens: 35243,
    cached_tokens: 18816,
    cache_creation_tokens: 0,
    reasoning_tokens: 0,
    cache_hit_rate: 0.5595,
    est_cost_usd: 0.012205,
    costed_calls: 22,
    uncosted_calls: 0,
    usage_known_calls: 22,
    usage_unknown_calls: 0,
    cost_estimate_status: "complete_estimate",
    llm_calls_by_model: { "offline-stub": 22 },
    llm_calls_by_kind: { structured: 18, text: 4 },
    llm_calls_by_task: { draft_report: 22 },
    llm_calls_by_status: { success: 22 },
    budget_est_cost_usd: 0.012205,
    budget_reserved_calls: 0,
    budget_settled_calls: 0,
    budget_rejected_calls: 0,
    budget_uncertain_calls: 0,
    budget_total_tokens: 0,
    budget_reconciliation: "verified",
    duration_seconds: 215.97,
    event_count: 194,
    trace_status: "verified",
    failures_count: 0,
    findings_count: 3,
    report_gate_verdict: "pass",
    publication_readiness: "analysis_available",
    publication_freshness: "not_applicable",
    question_llm_skipped: false,
    question_proposals_dropped: 0,
    question_dataset_names_resolved: 0,
    question_list_coercions: 0,
    degraded: false,
    semantic_bootstrap_degraded: false,
    column_roles_unverified: 0,
    template_backstop_used: 0,
    join_candidates_proposed: 0,
    join_authorizations_fresh: 0,
    join_authorizations_stale: 0,
    join_authorizations_unverifiable: 0,
    relationship_overlap_pairs_evaluated: 0,
    relationship_overlap_pairs_prefiltered: 0,
    relationship_full_validations: 0,
    relationship_coverage_limited: false,
    relationship_candidate_payload_bytes: 0,
    relationship_discovery_deferred: false,
    semantic_degraded_claims: 0,
    time_boundary_truncations: 0,
    numeric_unverified_claims: 0,
    quantitative_coverage_gaps: 0,
    evidence_interleave_granted: 0,
    evidence_interleave_rejected: 0,
    findings_dedup_clusters: 0,
    findings_dedup_merged: 0,
    domain_metric_questions: 0,
    domain_metrics_skipped: 0,
    macro_loop_rounds: 0,
    macro_loop_new_findings: 0,
    macro_loop_discard_rounds: 0,
    question_answered: 0,
    question_abstained: 0,
    question_failed: 0,
    question_awaiting_approval: 0,
    result_contract_failures: {},
    interpretation_validated: 0,
    interpretation_fallbacks: 0,
    report_eligible_findings: 0,
    question_answer_rate: 0,
    question_abstention_rate: 0,
    tokens_per_answered_question: 0,
    seconds_per_answered_question: 0,
    report_token_share: 0,
    report_duration_share: 0,
    coverage_limited: false,
    publication_blocked: false,
    steps: [
      {
        step_name: "profile_dataset",
        llm_calls: 0,
        tool_calls: 0,
        tokens: 0,
        duration_seconds: 20.4,
      },
      {
        step_name: "export_agentic_report",
        llm_calls: 8,
        tool_calls: 0,
        tokens: 42000,
        duration_seconds: 120.5,
      },
    ],
    artifact_counts: { ChartSpec: 12 },
    generated_at: "2026-07-25T03:41:00Z",
  };
}

/* Echoes the caller's picks the way the server does, so tests can assert the
 * builder sent what the controls showed. */
export function customChartView(
  sessionId: string,
  body: CustomChartRequest,
): CustomChartView {
  const aggregate = body.aggregate ?? (body.y_column ? "sum" : "count");
  const yEncoding = body.y_column
    ? { field: body.y_column, type: "quantitative", aggregate }
    : { aggregate: "count", type: "quantitative" };
  return {
    session_id: sessionId,
    dataset_id: body.dataset_id,
    chart_type: body.chart_type,
    aggregate,
    row_count: 3,
    source_row_count: 3,
    truncated: false,
    series_truncated: false,
    row_limit: 5000,
    spec: {
      $schema: "https://vega.github.io/schema/vega-lite/v6.json",
      mark: body.chart_type === "histogram" ? "bar" : body.chart_type,
      encoding: {
        x: { field: body.x_column, type: "nominal" },
        y: yEncoding,
        ...(body.color_column
          ? { color: { field: body.color_column, type: "nominal" } }
          : {}),
      },
      data: {
        values: [
          { [body.x_column]: "a", value: 1 },
          { [body.x_column]: "b", value: 2 },
          { [body.x_column]: "c", value: 3 },
        ],
      },
    },
  };
}

const INITIAL_VERIFIED_RELATIONS = [
  {
    left: "orders.csv.customer_id",
    right: "customers.csv.customer_id",
    cardinality: "many_to_one",
    confirmed_by: "user",
    confirmed_at: "2026-07-25T09:00:00Z",
    source_session_id: "r1",
  },
  {
    left: "orders.csv.product_id",
    right: "products.csv.product_id",
    cardinality: "many_to_one",
    confirmed_by: "user",
    confirmed_at: "2026-07-25T09:05:00Z",
    source_session_id: "r1",
  },
];

/* Mutable so a delete is visible to the refetch it triggers, the way the server
 * behaves; the afterEach hook restores it via resetVerifiedRelations(). */
export let SAMPLE_VERIFIED_RELATIONS = INITIAL_VERIFIED_RELATIONS.map((r) => ({
  ...r,
}));

/* freshness is an annotation, not a filter: stale findings still come back and
 * the page decides whether to show them. */
export function decisionStoryView(sessionId: string): DecisionStoryView {
  return {
    session_id: sessionId,
    project_id: "p1",
    eligible_findings: [
      {
        artifact_id: "vf_1",
        source_session_id: sessionId,
        question: "Which region drives the revenue drop?",
        analytical_reliability: "high",
        report_readiness: "eligible",
        freshness: "fresh",
      },
      {
        artifact_id: "vf_2",
        source_session_id: sessionId,
        question: "Does the refund spike track a single channel?",
        analytical_reliability: "medium",
        report_readiness: "eligible_with_limitations",
        freshness: "stale",
      },
    ],
    drafts: [
      {
        artifact_id: "brief_1",
        session_id: "synthesis_20260725_000000_aaaa",
        created_at: "2026-07-25T09:30:00Z",
        brief_id: "brief_1",
        headline: "Refunds, not demand, drove the Q3 dip.",
        decision_context: "Q3 review.",
        storyline: [
          { title: "What happened", body: "Revenue fell 8% QoQ." },
          {
            title: "Why",
            body: "Refunds explain 6 of those 8 points.",
          },
        ],
        limitations: ["Channel attribution is incomplete before May."],
        investigation_gaps: ["No terminal outcome for the top-ranked question."],
        business_context: "Q3 review.",
        report_eligible: true,
        report_readiness: "eligible",
        selected_finding_artifact_ids: ["vf_1", "vf_2"],
      },
    ],
    warnings: [],
  };
}

export function decisionCoverageView(sessionId: string): DecisionCoverageView {
  return {
    session_id: sessionId,
    project_id: "p1",
    top_cards_total: 8,
    top_cards_terminal: 2,
    uninvestigated_high_value: [
      "Which region drives the revenue drop?",
      "Does the refund spike track a single channel?",
    ],
    findings_not_eligible: 1,
    validated_findings: 3,
    coverage_ready: false,
    gaps: ["No terminal outcome for the top-ranked question."],
  };
}

export function cleaningLogView(sessionId: string): CleaningLogView {
  return {
    session_id: sessionId,
    recipe_count: 1,
    summary: [
      {
        dataset: "sample",
        recipe_id: "recipe_1",
        rows_before: 250,
        rows_after: 244,
        rows_removed: 6,
        columns_before: 4,
        columns_after: 3,
        columns_removed: 1,
        delete_steps: 2,
        protection_triggers: 1,
        requires_approval: true,
      },
    ],
    deleted_data: [
      {
        dataset: "sample",
        operation: "drop_duplicate_rows",
        column: "",
        rows_deleted: 6,
        columns_deleted: 0,
        reason: "exact duplicates",
        details: "Removed 6 exact duplicate rows.",
      },
      {
        dataset: "sample",
        operation: "drop_column",
        column: "notes",
        rows_deleted: 0,
        columns_deleted: 1,
        reason: "all values missing",
        details: "Dropped notes: every value was null.",
      },
    ],
    protection_triggers: [
      {
        dataset: "sample",
        code: "row_loss_ratio",
        reason: "Would delete more than 20% of rows.",
        thresholds: "max_row_loss=0.2",
      },
    ],
    suggestions: [{ suggestion: "Consider normalising the region column." }],
  };
}

export function cleaningRawView(sessionId: string): CleaningRawView {
  return {
    session_id: sessionId,
    precleaning_recorded: true,
    profiles: SAMPLE_PROFILES.datasets,
    charts: [
      {
        artifact_id: "art_raw_chart_1",
        title: "Raw value distribution",
        dataset_id: "sample",
        dataset_name: "sample",
        description: "Before cleaning.",
        plain_language: "Values were skewed before cleaning.",
        spec: {
          mark: "bar",
          encoding: {
            x: { field: "value", type: "quantitative", bin: true },
            y: { aggregate: "count", type: "quantitative" },
          },
          data: { values: [{ value: 1 }, { value: 2 }, { value: 2 }] },
        },
      },
    ],
    previews: [
      {
        artifact_id: "art_raw_preview_1",
        dataset_id: "sample",
        name: "sample",
        rows: 250,
        columns: 4,
        column_names: ["id", "name", "value", "notes"],
        rows_preview: [
          { id: 1, name: "row-1", value: 1.5, notes: null },
          { id: 2, name: "row-2", value: 2.5, notes: null },
        ],
      },
    ],
  };
}

export const SAMPLE_DEBUG_LOG = [
  '{"event":"step_started","name":"profile_dataset"}',
  '{"event":"step_completed","name":"profile_dataset"}',
].join("\n");

export function runDebugView(sessionId: string): SessionDebugView {
  return {
    session_id: sessionId,
    code_version: "abc1234",
    summary: {
      events: 194,
      artifacts: 21,
      llm_calls: 22,
      tool_calls: 4,
      errors: 1,
      total_tokens: 68868,
      estimated_cost_usd: 0.012205,
    },
    report_quality: {
      section_coverage: 0.9,
      claim_section_coverage: 0.85,
      claim_survival_rate: 0.75,
      deterministic_repair_count: 2,
      prompt_tokens_by_attempt: "1:12000, 2:9000",
    },
    timeline: [
      {
        event_type: "step_completed",
        name: "profile_dataset",
        started_at: "2026-07-25T03:40:00Z",
        duration_ms: 20400,
        summary: "profiled 1 dataset",
      },
    ],
    llm_calls: [
      {
        task: "draft_report",
        provider: "offline",
        model: "offline-stub",
        prompt_tokens: 12000,
        completion_tokens: 900,
        total_tokens: 12900,
        estimated_cost_usd: 0.0041,
        schema: "ReportDraft",
        duration_ms: 4200,
        status: "success",
        attempt: "1",
        error_type: "",
        error: "",
      },
    ],
    tool_calls: [
      {
        event_type: "tool_completed",
        tool: "sql_runner",
        duration_ms: 120,
        row_count: 244,
        truncated: false,
        artifact_id: "art_1",
        summary: "select ... from sample",
      },
    ],
    errors: [
      {
        event_type: "step_failed",
        name: "validate_report",
        error_type: "ValidationError",
        error: "claim 3 lost its evidence reference",
      },
    ],
    artifacts: [
      { artifact_id: "art_1", type: "DatasetProfile", parents: 0, warnings: 0 },
    ],
  };
}

export function llmDebugPage(cursor: string | null): LlmDebugPage {
  const start = cursor ? Number(cursor) : 0;
  const items = [0, 1].map((offset) => ({
    index: start + offset + 1,
    ts: "2026-07-25T03:40:00Z",
    kind: "structured",
    transport_kind: "structured",
    task: "draft_report",
    provider: "offline",
    model: "offline-stub",
    endpoint_host: "localhost",
    status: "success",
    finish_reason: "stop",
    duration_s: 4.2,
    prompt_tokens: 12000,
    completion_tokens: 900,
    cached_tokens: 0,
    cache_creation_tokens: 0,
    reasoning_tokens: 0,
    estimated_cost_usd: 0.0041,
    cost_basis: "registry_estimate",
    pricing_version: "public-list-prices-2026-07-28",
    usage_reported: true,
    request_id: "req_stub",
    response_id: "resp_stub",
    request_bytes: 4096,
    response_bytes: 2048,
    payload_preview: `{"system": "You are ...", "call": ${start + offset + 1}}`,
    response_preview: '{"sections": [...]}',
  }));
  return { items, next_cursor: start === 0 ? "2" : null };
}

export function columnDistributionsView(
  sessionId: string,
  datasetId: string,
): ColumnDistributionsView {
  return {
    dataset_id: datasetId,
    session_id: sessionId,
    row_count: SAMPLE_TOTAL_ROWS,
    sampled: false,
    sample_rows: SAMPLE_TOTAL_ROWS,
    sample_cap: 10000,
    bins: 10,
    top_k: 5,
    columns: [
      {
        name: "value",
        dtype: "float64",
        kind: "numeric",
        missing_percent: 0,
        counts: [4, 9, 20, 40, 60, 50, 30, 20, 12, 5],
        bin_edges: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        min: 0.5,
        max: 9.5,
        top: null,
        other_count: null,
        unique_count: null,
        len_min: null,
        len_max: null,
      },
      {
        name: "name",
        dtype: "object",
        kind: "categorical",
        missing_percent: 1.2,
        counts: null,
        bin_edges: null,
        min: null,
        max: null,
        top: [
          { value: "row-1", count: 30 },
          { value: "row-2", count: 25 },
        ],
        other_count: 195,
        unique_count: 120,
        len_min: 5,
        len_max: 7,
      },
    ],
  };
}

const TRACE_EVENT_TYPES = { step_started: 2, step_completed: 2, llm_usage: 1 };

export function traceEvents(sessionId: string, type?: string): TraceEventPage {
  const all: TraceEventRow[] = [
    {
      event_id: 1,
      event_type: "step_started",
      name: "profile_dataset",
      started_at: "2026-07-25T03:37:24Z",
      finished_at: null,
      duration_seconds: null,
      summary: { index: 0 },
    },
    {
      event_id: 2,
      event_type: "llm_usage",
      name: "profile",
      started_at: "2026-07-25T03:37:25Z",
      finished_at: null,
      duration_seconds: null,
      summary: { total_tokens: 1000, estimated_cost_usd: 0.002 },
    },
    {
      event_id: 3,
      event_type: "step_completed",
      name: "profile_dataset",
      started_at: "2026-07-25T03:37:24Z",
      finished_at: "2026-07-25T03:37:44Z",
      duration_seconds: 20.4,
      summary: {},
    },
  ];
  const items = type ? all.filter((event) => event.event_type === type) : all;
  return {
    session_id: sessionId,
    items,
    next_cursor: null,
    event_types: TRACE_EVENT_TYPES,
    total: items.length,
  };
}

export function decisionReportView(sessionId: string): DecisionReportView {
  return {
    session_id: sessionId,
    status: "available",
    artifact_id: "dreport_1",
    report_session_id: "synthesis_abc",
    report_id: "dreport_1",
    brief_id: "brief_1",
    title: "Channel mix decision story",
    generated_at: "2026-07-25T04:00:00Z",
    scqa: {
      situation: "Order volume grew across all channels.",
      complication: "Average order value fell in two channels.",
      question: "Where should channel spend move next?",
      answer: "Rebalance toward the channel holding value.",
    },
    sections: [
      {
        title: "How do order values vary by channel?",
        body: "The observed average order value is 125.5.",
        finding_artifact_ids: ["vf_1"],
      },
    ],
    limitations: ["Channel labels require review."],
    investigation_gaps: ["Validate return patterns in later periods."],
    meta_insight: null,
    candidate_decisions: [
      {
        finding_artifact_id: "vf_1",
        question: "How do order values vary by channel?",
        decision_action: "Rebalance channel spend once labels are reviewed.",
        decision_readiness: "medium",
        analytical_reliability: "high",
        report_readiness: "eligible_with_limitations",
      },
    ],
    evidence_refs: [
      {
        artifact_id: "table_orders",
        kind: "table",
        locator: "rows[0].average_order_value",
        session_id: "r1",
      },
    ],
    source_finding_artifact_ids: ["vf_1"],
    granted_evidence_artifact_ids: [],
    report_readiness: "eligible_with_limitations",
    narrative_status: "deterministic",
    narrative_fallback_reason: "",
    freshness: { status: "stale", reasons: ["vf_1: source dataset changed."] },
    publication_status: "published",
    gate_verdict: "degraded",
    confidence_label: "high",
    export_available: false,
  };
}

export const PROVIDERS: ProviderInfo[] = [
  {
    provider: "offline",
    display_name: "Offline (deterministic)",
    requires_api_key: false,
    requires_base_url: false,
    default_base_url: "",
    preset_models: ["offline-deterministic"],
    structured_mode: "json_object",
    native: false,
    text_transport: "messages",
    structured_transport: "response_format:json_object",
    tool_execution: "host_orchestrated",
    pricing_catalog_version: "",
    pricing_source_url: "",
    pricing_checked_at: "",
    capability_catalog_version: "agent-models-2026-07-30",
    agent_model_count: 1,
  },
  {
    provider: "deepseek",
    display_name: "DeepSeek",
    requires_api_key: true,
    requires_base_url: false,
    default_base_url: "https://api.deepseek.com",
    preset_models: ["deepseek-v4-flash", "deepseek-v4-pro"],
    structured_mode: "json_object",
    native: false,
    text_transport: "messages",
    structured_transport: "response_format:json_object",
    tool_execution: "host_orchestrated",
    pricing_catalog_version: "public-list-prices-2026-07-29",
    pricing_source_url: "https://api-docs.deepseek.com/quick_start/pricing/",
    pricing_checked_at: "2026-07-29",
    capability_catalog_version: "agent-models-2026-07-30",
    agent_model_count: 2,
  },
];

/* List prices for the models the fixtures use, so a catalog row and the
 * provider table cannot drift apart inside the tests. */
const PRICED_MODELS: Record<string, Record<string, unknown>> = {
  "deepseek-v4-flash": {
    id: "deepseek-v4-flash",
    owned_by: "deepseek",
    created: null,
    input_usd_per_1m: 0.14,
    output_usd_per_1m: 0.28,
    cache_read_usd_per_1m: 0.0028,
    cache_write_usd_per_1m: null,
    pricing_source: "public_list_price_snapshot",
    capabilities: ["tool_calling"],
    parallel_tool_calling: true,
    structured_output: "json_object",
    temperature_policy: "send",
    verified: true,
  },
  "deepseek-v4-pro": {
    id: "deepseek-v4-pro",
    owned_by: "deepseek",
    created: null,
    input_usd_per_1m: 0.435,
    output_usd_per_1m: 0.87,
    cache_read_usd_per_1m: 0.003625,
    cache_write_usd_per_1m: null,
    pricing_source: "public_list_price_snapshot",
    capabilities: ["tool_calling"],
    parallel_tool_calling: true,
    structured_output: "json_object",
    temperature_policy: "send",
    verified: true,
  },
};

export function defaultSettings(): SettingsView {
  return {
    version: 0,
    provider: "offline",
    model: "offline-deterministic",
    base_url: "",
    resolved_base_url: "",
    temperature: 0.2,
    max_tokens: 6000,
    timeout_seconds: 180,
    structured_output_mode: "auto",
    payload_policy: "schema+aggregates",
    usd_per_1k_prompt: 0,
    usd_per_1k_completion: 0,
    analysis_depth: 0,
    dev_mode: false,
    api_key_set: false,
    api_key_last4: "",
    is_ready_for_live_calls: false,
    status_state: "offline",
    status_message: "Offline mode: deterministic fallback is used.",
    missing_fields: [],
    model_verified: true,
    warnings: [],
    source: "env",
    about: {
      app_version: "0.2.0",
      workspace_is_default: true,
      workspace_label: "default workspace",
    },
  };
}

const supportDocs: SupportDocView[] = [];

export function resetSupportDocs(): void {
  supportDocs.length = 0;
}

export function resetVerifiedRelations(): void {
  SAMPLE_VERIFIED_RELATIONS = INITIAL_VERIFIED_RELATIONS.map((r) => ({ ...r }));
}

/* Module-level so a PUT is visible to the next GET, the way the server behaves.
 * `resetSettingsState()` runs from the global test setup. */
let settingsState: SettingsView = defaultSettings();

export function resetSettingsState(): void {
  settingsState = defaultSettings();
}

/* Investigation governance fixtures: one plan in each lifecycle state so the
 * panel's pending / approved / outcome sections all render from defaults. */
export const SAMPLE_PLAN_ID = "invplan_pending";
export const SAMPLE_PLAN_RUN_ID = "investigation_20260725_000000_000000_abcd1234";

function planView(
  overrides: Partial<InvestigationPlanView> & { plan_id: string },
): InvestigationPlanView {
  return {
    plan_session_id: SAMPLE_PLAN_RUN_ID,
    investigation_id: `inv_${overrides.plan_id}`,
    question_id: "q_trend",
    question: "How is value trending over time?",
    method_family: "descriptive_analysis",
    method_recipe: "read-only aggregate over the approved scope",
    card_version: 1,
    status: "pending",
    plan_status: "planned",
    execution_ready: true,
    allowed_tools: ["sql_select"],
    target_datasets: ["sample.csv"],
    method_requirements: [],
    validation_gates: [
      { name: "scope", status: "passed", reason: "Limited to the named datasets." },
    ],
    candidate_fingerprint: "f".repeat(64),
    deep_investigation: false,
    decision_reason: "",
    outcome_status: null,
    outcome_reason: "",
    finding_texts: [],
    report_readiness: null,
    can_approve: false,
    can_reject: false,
    can_execute: false,
    ...overrides,
  };
}

export function investigationsView(
  sessionId: string,
  overrides: Partial<InvestigationsView> = {},
): InvestigationsView {
  return {
    session_id: sessionId,
    project_id: "p1",
    analysis_depth: 0,
    deep_investigation_enabled: false,
    macro_loop_authorized: false,
    plans: [
      planView({ plan_id: SAMPLE_PLAN_ID, can_approve: true, can_reject: true }),
      planView({
        plan_id: "invplan_approved",
        question: "What is total value by segment?",
        status: "approved",
        decision_reason: "Reviewed by the analyst.",
        can_execute: true,
      }),
      planView({
        plan_id: "invplan_executed",
        question: "How large is the seasonal swing?",
        status: "executed",
        outcome_status: "validated",
        outcome_reason: "Gates passed.",
        finding_texts: ["Q4 value is 18% above the yearly mean."],
        report_readiness: "eligible_with_limitations",
      }),
    ],
    macro_loops: [],
    ...overrides,
  };
}

export function investigationDecisionPrepared(
  sessionId: string,
  planId: string,
  decision: "approved" | "rejected",
  reason = "",
): InvestigationDecisionPrepared {
  return {
    session_id: sessionId,
    plan_id: planId,
    plan_session_id: SAMPLE_PLAN_RUN_ID,
    decision,
    reason,
    action_hash: SAMPLE_ACTION_HASH,
    approval_token: SAMPLE_APPROVAL_TOKEN,
    expires_at: "2026-07-25T12:00:00Z",
    plan: planView({ plan_id: planId, can_approve: true, can_reject: true }),
  };
}

export const defaultHandlers = [
  http.get("/api/v1/projects", () =>
    HttpResponse.json([
      { project_id: "p1", name: "Project p1", session_count: 1 },
    ] satisfies ProjectSummary[]),
  ),

  http.get("/api/v1/usage", ({ request }) => {
    const days = Number(new URL(request.url).searchParams.get("days") ?? 180);
    const today = new Date();
    return HttpResponse.json({
      schema_version: 1,
      generated_at: today.toISOString(),
      window_days: days,
      project_count: 1,
      session_count: 2,
      status_counts: { completed: 2 },
      daily: Array.from({ length: days }, (_, index) => {
        const day = new Date(today);
        day.setUTCDate(day.getUTCDate() - (days - 1 - index));
        return {
          date: day.toISOString().slice(0, 10),
          sessions: index === days - 1 ? 2 : 0,
        };
      }),
      llm_calls: 4,
      total_tokens: 12345,
      est_cost_usd: 0.0125,
      priced_sessions: 1,
      unpriced_sessions: 1,
      artifact_count: 3,
      dataset_count: 2,
      profiled_rows: 12500,
      data_bytes: 2048,
      recent: [
        {
          session_id: "r1",
          project_id: "p1",
          title: "Demo run",
          status: "complete",
          created_at: "2026-07-20T10:00:00Z",
          updated_at: "2026-07-21T10:00:00Z",
        },
      ],
    });
  }),

  http.post("/api/v1/projects", async ({ request }) => {
    const body = (await request.json()) as { project_id: string; name?: string };
    return HttpResponse.json(
      {
        project_id: body.project_id,
        name: body.name || body.project_id,
        session_count: 0,
      } satisfies ProjectSummary,
      { status: 201 },
    );
  }),

  http.get("/api/v1/projects/:projectId/sessions", ({ params, request }) => {
    const projectId = String(params["projectId"]);
    /* The bucket holding sessions with no project is empty by default, so the
     * rail's standalone group hides itself; tests that want one override this. */
    if (projectId === "unfiled-sessions") {
      return HttpResponse.json({ items: [], next_cursor: null });
    }
    const q = new URL(request.url).searchParams.get("q")?.toLowerCase() ?? "";
    const items = [
      {
        session_id: "r1",
        project_id: projectId,
        title: "Demo run",
        status: "complete",
        created_at: "2026-07-20T10:00:00Z",
        updated_at: "2026-07-21T10:00:00Z",
        dataset_names: ["sample"],
        artifact_count: 3,
        report_status: "final",
        chat_message_count: 0,
      },
      {
        session_id: "r2",
        project_id: projectId,
        title: "Churn deep dive",
        status: "complete",
        created_at: "2026-07-19T10:00:00Z",
        updated_at: "2026-07-19T10:00:00Z",
        dataset_names: ["churn"],
        artifact_count: 1,
        report_status: null,
        chat_message_count: 0,
      },
    ];
    const matches = (run: (typeof items)[number]) =>
      !q ||
      run.session_id.toLowerCase().includes(q) ||
      (run.title ?? "").toLowerCase().includes(q) ||
      run.dataset_names.some((name) => name.toLowerCase().includes(q));
    return HttpResponse.json({
      items: items.filter(matches),
      next_cursor: null,
    } satisfies SessionPage);
  }),

  http.delete("/api/v1/sessions/:sessionId", ({ params }) =>
    HttpResponse.json({
      session_id: String(params["sessionId"]),
      project_id: "p1",
      deleted: true,
    } satisfies SessionDeleted),
  ),

  http.get("/api/v1/sessions/:sessionId", ({ params }) =>
    HttpResponse.json({
      session_id: String(params["sessionId"]),
      project_id: "p1",
      title: "Demo run",
      status: "complete",
      created_at: "2026-07-20T10:00:00Z",
      updated_at: "2026-07-21T10:00:00Z",
      dataset_names: ["sample"],
      artifact_count: 3,
      report_status: "final",
      chat_message_count: 0,
      code_version: "abc123",
      seed: 42,
      source_session_id: null,
      artifact_type_counts: { chart: 2, table: 1 },
      warnings: ["one warning"],
    } satisfies SessionDetail),
  ),

  http.get("/api/v1/sessions/:sessionId/datasets", () =>
    HttpResponse.json([
      {
        dataset_id: "sample",
        project_id: "p1",
        display_name: "sample.csv",
        original_uri: "upload://sample.csv",
        format: "csv",
        content_hash: "deadbeef",
        byte_size: 1024,
        row_count: SAMPLE_TOTAL_ROWS,
        schema: sampleColumns,
        ingest_status: "ready",
      },
    ] satisfies DatasetHandle[]),
  ),

  http.get("/api/v1/sessions/:sessionId/datasets/:datasetId/schema", ({ params }) =>
    HttpResponse.json({
      dataset_id: String(params["datasetId"]),
      session_id: String(params["sessionId"]),
      columns: sampleColumns,
      source: "duckdb",
    } satisfies DatasetSchema),
  ),

  http.get(
    "/api/v1/sessions/:sessionId/datasets/:datasetId/preview",
    ({ params, request }) => {
      const url = new URL(request.url);
      const offset = Number(url.searchParams.get("offset") ?? "0");
      const limit = Number(url.searchParams.get("limit") ?? "100");
      return HttpResponse.json({
        dataset_id: String(params["datasetId"]),
        session_id: String(params["sessionId"]),
        columns: sampleColumns.map((col) => col.name),
        rows: sampleRows(offset, limit),
        offset,
        limit,
        has_more: offset + limit < SAMPLE_TOTAL_ROWS,
        source_format: "csv",
      } satisfies DatasetPreview);
    },
  ),

  http.get("/api/v1/sessions/:sessionId/report", ({ params }) =>
    HttpResponse.json({
      session_id: String(params["sessionId"]),
      status: "validated",
      markdown: SAMPLE_REPORT_MARKDOWN,
      generated_at: "2026-07-22T12:00:00Z",
    }),
  ),

  /* Cursor is the numeric index of the next item, mirroring the API's opaque
   * rowid cursor closely enough for paging tests. */
  http.get("/api/v1/sessions/:sessionId/artifacts", ({ request }) => {
    const url = new URL(request.url);
    const type = url.searchParams.get("type");
    const limit = Number(url.searchParams.get("limit") ?? "50");
    const start = Number(url.searchParams.get("cursor") ?? "0");
    const filtered = type
      ? SAMPLE_ARTIFACTS.filter((item) => item.type === type)
      : SAMPLE_ARTIFACTS;
    const items = filtered.slice(start, start + limit);
    const nextStart = start + limit;
    return HttpResponse.json({
      items,
      next_cursor: nextStart < filtered.length ? String(nextStart) : null,
    } satisfies ArtifactPage);
  }),

  http.get("/api/v1/sessions/:sessionId/artifacts/:artifactId", ({ params }) => {
    const artifactId = String(params["artifactId"]);
    const summary = SAMPLE_ARTIFACTS.find(
      (item) => item.artifact_id === artifactId,
    );
    if (!summary) {
      return HttpResponse.json(
        {
          error: {
            code: "artifact_not_found",
            message: `Artifact not found: ${artifactId}`,
          },
        },
        { status: 404 },
      );
    }
    return HttpResponse.json({
      artifact_id: artifactId,
      type: summary.type,
      project_id: "p1",
      session_id: "r1",
      created_at: "2026-07-22T12:00:00Z",
      payload: { title: `Payload of ${artifactId}` },
      warnings: [],
    } satisfies ArtifactDetail);
  }),

  http.get("/api/v1/sessions/:sessionId/quality", ({ params }) =>
    HttpResponse.json({
      ...SAMPLE_QUALITY,
      session_id: String(params["sessionId"]),
    } satisfies QualityView),
  ),

  http.get("/api/v1/sessions/:sessionId/profiles", ({ params }) =>
    HttpResponse.json({
      ...SAMPLE_PROFILES,
      session_id: String(params["sessionId"]),
    } satisfies ProfilesView),
  ),

  http.get("/api/v1/sessions/:sessionId/charts", () =>
    HttpResponse.json({
      items: SAMPLE_CHART_SUMMARIES,
      next_cursor: null,
    } satisfies ChartPage),
  ),

  http.get("/api/v1/sessions/:sessionId/charts/:chartId", ({ params }) =>
    HttpResponse.json(chartView(String(params["chartId"]))),
  ),

  /* Empty by default so the Data step shows only the upload path; tests that
   * exercise reuse override this. */
  http.get("/api/v1/projects/:projectId/uploads", () => HttpResponse.json([])),

  http.post("/api/v1/projects/:projectId/uploads", async ({ request, params }) => {
    const form = await request.formData();
    const file = form.get("file");
    const name = file instanceof File ? file.name : "upload.csv";
    return HttpResponse.json(
      {
        upload_id: "up_1",
        project_id: String(params["projectId"]),
        status: "completed",
        error: null,
        dataset: uploadedDataset(name),
      } satisfies UploadStatus,
      { status: 201 },
    );
  }),

  http.post("/api/v1/sessions/:sessionId/jobs", ({ params }) =>
    HttpResponse.json(
      {
        job_id: "job_1",
        session_id: String(params["sessionId"]),
        status: "queued",
        events_url: "/api/v1/jobs/job_1/events",
      } satisfies JobCreated,
      { status: 201 },
    ),
  ),

  http.get("/api/v1/jobs/:jobId", ({ params }) => {
    const jobId = String(params["jobId"]);
    const failure = dataOperationFailures.get(jobId);
    return HttpResponse.json(
      jobStatus(
        jobId,
        failure
          ? {
              status: "failed",
              finished_at: "2026-07-25T10:00:02Z",
              error_code: failure.code,
              error_message: failure.message,
            }
          : dataOperationResults.has(jobId)
            ? {
                status: "completed",
                finished_at: "2026-07-25T10:00:02Z",
              }
            : {},
      ),
    );
  }),

  http.post("/api/v1/jobs/:jobId/cancel", ({ params }) =>
    HttpResponse.json(
      jobStatus(String(params["jobId"]), { cancel_requested: true }),
    ),
  ),

  http.post(
    "/api/v1/sessions/:sessionId/cleaning/preview",
    async ({ params, request }) => {
      const body = (await request.json()) as { dataset_id: string };
      const jobId = "job_cleaning_preview";
      dataOperationResults.set(
        jobId,
        cleaningPreviewResult(String(params["sessionId"]), body.dataset_id),
      );
      return HttpResponse.json(
        dataOperationStarted(String(params["sessionId"]), jobId),
        { status: 202 },
      );
    },
  ),

  http.post("/api/v1/sessions/:sessionId/cleaning/apply", ({ params }) => {
    const jobId = "job_cleaning_apply";
    dataOperationResults.set(jobId, cleaningApplied(String(params["sessionId"])));
    return HttpResponse.json(
      dataOperationStarted(String(params["sessionId"]), jobId),
      { status: 202 },
    );
  }),

  http.get("/api/v1/jobs/:jobId/cleaning-preview-result", ({ params }) =>
    HttpResponse.json(
      dataOperationResults.get(String(params["jobId"])) as Record<string, unknown>,
    ),
  ),

  http.get("/api/v1/jobs/:jobId/cleaning-apply-result", ({ params }) =>
    HttpResponse.json(
      dataOperationResults.get(String(params["jobId"])) as Record<string, unknown>,
    ),
  ),

  http.get("/api/v1/sessions/:sessionId/questions", ({ params }) =>
    HttpResponse.json(questionsView(String(params["sessionId"]))),
  ),

  http.post(
    "/api/v1/sessions/:sessionId/questions/:questionId/prepare",
    async ({ params, request }) => {
      /* Echo the requested llm mode back, like the real server does. */
      const body = (await request.json().catch(() => null)) as {
        llm?: string;
      } | null;
      return HttpResponse.json(
        questionPrepared(
          String(params["sessionId"]),
          String(params["questionId"]),
          body?.llm ?? "env",
        ),
      );
    },
  ),

  http.post(
    "/api/v1/sessions/:sessionId/questions/:questionId/execute",
    ({ params }) =>
      HttpResponse.json(
        questionStarted(
          String(params["sessionId"]),
          String(params["questionId"]),
        ),
        { status: 201 },
      ),
  ),

  http.get("/api/v1/sessions/:sessionId/findings", ({ params }) =>
    HttpResponse.json(findingsView(String(params["sessionId"]))),
  ),

  http.get("/api/v1/sessions/:sessionId/semantic", ({ params }) =>
    HttpResponse.json(semanticView(String(params["sessionId"]))),
  ),

  http.get("/api/v1/compare", ({ request }) => {
    const url = new URL(request.url);
    return HttpResponse.json(
      compareView(
        url.searchParams.get("left") ?? "r1",
        url.searchParams.get("right") ?? "r2",
      ),
    );
  }),

  http.get("/api/v1/compare/:scope", ({ params, request }) => {
    const url = new URL(request.url);
    return HttpResponse.json(
      compareScopeView(
        String(params["scope"]) as CompareScopeName,
        url.searchParams.get("left") ?? "r1",
        url.searchParams.get("right") ?? "r2",
        url.searchParams.get("filter") === "differences",
      ),
    );
  }),

  http.get("/api/v1/sessions/:sessionId/skills", ({ params }) =>
    HttpResponse.json(skillsView(String(params["sessionId"]))),
  ),

  http.post("/api/v1/sessions/:sessionId/skills/:seedId/import", async ({ request }) => {
    const body = (await request.json()) as {
      bindings: Record<string, string>;
      name?: string;
    };
    return HttpResponse.json(
      {
        skill_id: "skill_imported_seed",
        source: "library",
        name: body.name || "Group totals and averages",
        description: "From seed 'group_value_comparison' on sample.",
        question: "How does value vary across name segments?",
        sql: "SELECT name, SUM(value) FROM sample GROUP BY 1",
        method: "aggregation",
        param_columns: Object.values(body.bindings),
        expected_datasets: ["sample"],
        params: [],
        source_session_id: null,
        created_at: "2026-07-25T09:00:00Z",
      },
      { status: 201 },
    );
  }),

  http.post("/api/v1/sessions/:sessionId/skills/:skillId/prepare", ({ params }) =>
    HttpResponse.json(
      skillPrepared(String(params["sessionId"]), String(params["skillId"])),
    ),
  ),

  http.post("/api/v1/sessions/:sessionId/skills/:skillId/execute", ({ params }) =>
    HttpResponse.json(
      skillStarted(String(params["sessionId"]), String(params["skillId"])),
      { status: 201 },
    ),
  ),

  http.post("/api/v1/sessions/:sessionId/skills", async ({ request }) => {
    const body = (await request.json()) as {
      source_artifact_id: string;
      name: string;
      description: string;
    };
    return HttpResponse.json(
      {
        skill_id: "skill_saved_2",
        source: "library",
        name: body.name,
        description: body.description,
        question: "How does value vary by name?",
        sql: "SELECT name, SUM(value) AS total FROM sample GROUP BY 1",
        method: "aggregation",
        param_columns: ["name", "value"],
        expected_datasets: ["sample"],
        params: [],
        source_session_id: "r1",
        created_at: "2026-07-25T09:00:00Z",
      },
      { status: 201 },
    );
  }),

  http.delete(
    "/api/v1/projects/:projectId/skills/:skillId",
    () => new HttpResponse(null, { status: 204 }),
  ),

  http.get("/api/v1/sessions/:sessionId/relationships", ({ params }) =>
    HttpResponse.json(relationshipGraph(String(params["sessionId"]))),
  ),

  http.post(
    "/api/v1/sessions/:sessionId/relationships/:relationshipId/prepare-validate",
    ({ params }) =>
      HttpResponse.json(
        relationshipPrepared(
          String(params["sessionId"]),
          String(params["relationshipId"]),
        ),
      ),
  ),

  http.post(
    "/api/v1/sessions/:sessionId/relationships/:relationshipId/validate",
    ({ params }) =>
      HttpResponse.json(
        relationshipStarted(
          String(params["sessionId"]),
          String(params["relationshipId"]),
        ),
        { status: 201 },
      ),
  ),

  http.post(
    "/api/v1/sessions/:sessionId/relationships/:relationshipId/confirm",
    ({ params }) => {
      const graph = relationshipGraph(String(params["sessionId"]));
      const edge = (graph.edges ?? []).find(
        (item) => item.relationship_id === String(params["relationshipId"]),
      );
      return HttpResponse.json({
        ...edge,
        state: "confirmed",
        join_status: "confirmed",
        can_confirm: false,
      });
    },
  ),

  http.post(
    "/api/v1/sessions/:sessionId/relationships/:relationshipId/revoke",
    ({ params }) => {
      const graph = relationshipGraph(String(params["sessionId"]));
      const edge = (graph.edges ?? []).find(
        (item) => item.relationship_id === String(params["relationshipId"]),
      );
      return HttpResponse.json({ ...edge, join_status: "proposed" });
    },
  ),

  http.put("/api/v1/sessions/:sessionId/semantic/seeds", async ({ params, request }) => {
    const body = (await request.json()) as {
      expected_version: number;
      field_meanings: SemanticView["field_meanings"];
      metric_definitions?: SemanticView["metric_definitions"];
      entity_notes?: SemanticView["entity_notes"];
      verified_answers?: SemanticView["verified_answers"];
    };
    const stored = semanticView(String(params["sessionId"]));
    return HttpResponse.json({
      session_id: String(params["sessionId"]),
      version: body.expected_version + 1,
      field_meanings: body.field_meanings,
      /* Mirrors the server: a class the PUT omits comes back unchanged. */
      metric_definitions: body.metric_definitions ?? stored.metric_definitions,
      entity_notes: body.entity_notes ?? stored.entity_notes,
      verified_answers: body.verified_answers ?? stored.verified_answers,
    });
  }),

  http.post(
    "/api/v1/sessions/:sessionId/semantic/joins/confirm",
    async ({ params, request }) => {
      const body = (await request.json()) as { label: string };
      const entry = semanticView(String(params["sessionId"])).join_whitelist?.find(
        (item) => item.label === body.label,
      );
      return HttpResponse.json({ ...entry, status: "confirmed" });
    },
  ),

  http.post(
    "/api/v1/sessions/:sessionId/semantic/joins/revoke",
    async ({ params, request }) => {
      const body = (await request.json()) as { label: string };
      const entry = semanticView(String(params["sessionId"])).join_whitelist?.find(
        (item) => item.label === body.label,
      );
      return HttpResponse.json({ ...entry, status: "proposed" });
    },
  ),

  http.post(
    "/api/v1/sessions/:sessionId/semantic/proposals/accept",
    async ({ params, request }) => {
      const body = (await request.json()) as { dataset: string; column: string };
      return HttpResponse.json({
        session_id: String(params["sessionId"]),
        dataset: body.dataset,
        column: body.column,
        status: "accepted",
        seeds_version: 4,
      });
    },
  ),

  http.post(
    "/api/v1/sessions/:sessionId/semantic/proposals/reject",
    async ({ params, request }) => {
      const body = (await request.json()) as { dataset: string; column: string };
      return HttpResponse.json({
        session_id: String(params["sessionId"]),
        dataset: body.dataset,
        column: body.column,
        status: "rejected",
        seeds_version: 3,
      });
    },
  ),

  http.get("/api/v1/sessions/:sessionId/chat/messages", ({ params }) =>
    HttpResponse.json({
      session_id: String(params["sessionId"]),
      messages: [],
      next_cursor: null,
      total: 0,
    } satisfies ChatMessagePage),
  ),

  http.post("/api/v1/sessions/:sessionId/chat/messages", ({ params }) =>
    HttpResponse.json(chatAccepted(String(params["sessionId"])), { status: 202 }),
  ),

  http.get("/api/v1/sessions/:sessionId/chat/pending-plans", ({ params }) =>
    HttpResponse.json({ session_id: String(params["sessionId"]), plans: [] }),
  ),

  http.post("/api/v1/sessions/:sessionId/chat/plans/:planId/approve", ({ params }) =>
    HttpResponse.json(chatAccepted(String(params["sessionId"]), "msg_approved"), {
      status: 202,
    }),
  ),

  http.post("/api/v1/sessions/:sessionId/chat/plans/:planId/reject", ({ params }) =>
    HttpResponse.json({
      session_id: String(params["sessionId"]),
      plan_id: String(params["planId"]),
      status: "rejected",
    }),
  ),

  http.get("/api/v1/projects/:projectId/boards/:boardId", ({ params }) =>
    HttpResponse.json(
      emptyBoard(String(params["projectId"]), String(params["boardId"])),
    ),
  ),

  http.put(
    "/api/v1/projects/:projectId/boards/:boardId",
    async ({ params, request }) => {
      const body = (await request.json()) as {
        expected_version: number;
        columns: BoardView["columns"];
        cards: BoardView["cards"];
      };
      return HttpResponse.json({
        project_id: String(params["projectId"]),
        board_id: String(params["boardId"]),
        version: body.expected_version + 1,
        columns: body.columns,
        cards: body.cards,
      } satisfies BoardView);
    },
  ),

  http.get("/api/v1/sessions/:sessionId/analysis", ({ params }) =>
    HttpResponse.json(analysisView(String(params["sessionId"]))),
  ),

  http.get("/api/v1/sessions/:sessionId/metrics", ({ params }) =>
    HttpResponse.json(runMetricsView(String(params["sessionId"]))),
  ),

  http.get("/api/v1/sessions/:sessionId/trace", ({ params, request }) => {
    const url = new URL(request.url);
    return HttpResponse.json(
      traceEvents(String(params["sessionId"]), url.searchParams.get("type") ?? undefined),
    );
  }),

  http.post("/api/v1/sessions/:sessionId/client-failures", () =>
    HttpResponse.json(
      { event_type: "failure_recorded", recorded: true },
      { status: 201 },
    ),
  ),

  http.get("/api/v1/sessions/:sessionId/decision-report", ({ params }) =>
    HttpResponse.json({
      session_id: String(params["sessionId"]),
      status: "none",
      sections: [],
    } satisfies Partial<DecisionReportView>),
  ),

  http.post("/api/v1/sessions/:sessionId/charts/custom", async ({ params, request }) => {
    const body = (await request.json()) as CustomChartRequest;
    const jobId = "job_custom_chart";
    dataOperationResults.set(
      jobId,
      customChartView(String(params["sessionId"]), body),
    );
    return HttpResponse.json(
      dataOperationStarted(String(params["sessionId"]), jobId),
      { status: 202 },
    );
  }),

  http.get("/api/v1/jobs/:jobId/custom-chart-result", ({ params }) =>
    HttpResponse.json(
      dataOperationResults.get(String(params["jobId"])) as Record<string, unknown>,
    ),
  ),

  http.get("/api/v1/sessions/:sessionId/decision-coverage", ({ params }) =>
    HttpResponse.json(decisionCoverageView(String(params["sessionId"]))),
  ),

  http.get("/api/v1/sessions/:sessionId/decision-story", ({ params }) =>
    HttpResponse.json(decisionStoryView(String(params["sessionId"]))),
  ),

  http.post("/api/v1/sessions/:sessionId/decision-story/drafts", ({ params }) =>
    HttpResponse.json(
      {
        session_id: String(params["sessionId"]),
        execution_session_id: "sbsess_20260725_000000_abc123",
        job: jobStatus("job_story_draft", {
          status: "queued",
          kind: "synthesis_brief_create",
        }),
      },
      { status: 201 },
    ),
  ),

  http.post("/api/v1/sessions/:sessionId/decision-report/generate", ({ params }) =>
    HttpResponse.json(
      {
        session_id: String(params["sessionId"]),
        brief_artifact_id: "brief_1",
        execution_session_id: "drsess_20260725_000000_def456",
        job: jobStatus("job_report_gen", {
          status: "queued",
          kind: "decision_report_generate",
        }),
      },
      { status: 201 },
    ),
  ),

  http.post(
    "/api/v1/sessions/:sessionId/semantic/verified-relations/delete",
    async ({ params, request }) => {
      const body = (await request.json()) as { left: string; right: string };
      SAMPLE_VERIFIED_RELATIONS = SAMPLE_VERIFIED_RELATIONS.filter(
        (relation) =>
          !(relation.left === body.left && relation.right === body.right),
      );
      return HttpResponse.json({
        session_id: String(params["sessionId"]),
        seeds_version: 3,
        verified_relations: SAMPLE_VERIFIED_RELATIONS,
      } satisfies VerifiedRelationsUpdated);
    },
  ),

  http.get("/api/v1/sessions/:sessionId/cleaning/log", ({ params }) =>
    HttpResponse.json(cleaningLogView(String(params["sessionId"]))),
  ),

  http.get("/api/v1/sessions/:sessionId/cleaning/raw", ({ params }) =>
    HttpResponse.json(cleaningRawView(String(params["sessionId"]))),
  ),

  http.get("/api/v1/sessions/:sessionId/debug", ({ params }) =>
    HttpResponse.json(runDebugView(String(params["sessionId"]))),
  ),

  http.get("/api/v1/sessions/:sessionId/debug/log", () =>
    HttpResponse.text(SAMPLE_DEBUG_LOG, {
      headers: {
        "Content-Type": "application/x-ndjson",
        "Content-Disposition": 'attachment; filename="r1-debug.jsonl"',
      },
    }),
  ),

  http.get("/api/v1/sessions/:sessionId/debug/llm-calls", ({ request }) => {
    const url = new URL(request.url);
    return HttpResponse.json(llmDebugPage(url.searchParams.get("cursor")));
  }),

  http.post(
    "/api/v1/sessions/:sessionId/datasets/:datasetId/distributions",
    ({ params }) => {
      const jobId = `job_distributions_${String(params["datasetId"])}`;
      dataOperationResults.set(
        jobId,
        columnDistributionsView(
          String(params["sessionId"]),
          String(params["datasetId"]),
        ),
      );
      return HttpResponse.json(
        dataOperationStarted(String(params["sessionId"]), jobId),
        { status: 202 },
      );
    },
  ),

  http.get("/api/v1/jobs/:jobId/dataset-distributions-result", ({ params }) =>
    HttpResponse.json(
      dataOperationResults.get(String(params["jobId"])) as Record<string, unknown>,
    ),
  ),

  http.get("/api/v1/settings", () => HttpResponse.json(settingsState)),

  /* Mirrors the server contract: the key goes up and only ever comes back as
     api_key_set + api_key_last4. */
  http.put("/api/v1/settings", async ({ request }) => {
    const patch = (await request.json()) as SettingsPatch;
    const key = patch.api_key ?? "";
    const provider = patch.provider ?? settingsState.provider;
    const spec = PROVIDERS.find((item) => item.provider === provider);
    /* Same re-seed rule as the server: a provider switch without an explicit
       model adopts that provider's first preset and its default base URL. */
    const switched = provider !== settingsState.provider;
    settingsState = {
      ...settingsState,
      provider,
      model:
        patch.model ??
        (switched ? (spec?.preset_models?.[0] ?? "") : settingsState.model),
      base_url: patch.base_url ?? (switched ? "" : settingsState.base_url),
      resolved_base_url: patch.base_url || (spec?.default_base_url ?? ""),
      ...(patch.payload_policy ? { payload_policy: patch.payload_policy } : {}),
      ...(patch.temperature != null ? { temperature: patch.temperature } : {}),
      ...(patch.usd_per_1k_prompt != null
        ? { usd_per_1k_prompt: patch.usd_per_1k_prompt }
        : {}),
      ...(patch.usd_per_1k_completion != null
        ? { usd_per_1k_completion: patch.usd_per_1k_completion }
        : {}),
      ...(patch.dev_mode != null ? { dev_mode: patch.dev_mode } : {}),
      ...(key ? { api_key_set: true, api_key_last4: key.slice(-4) } : {}),
      ...(patch.clear_api_key ? { api_key_set: false, api_key_last4: "" } : {}),
      source: "session",
    };
    return HttpResponse.json(settingsState);
  }),

  http.delete("/api/v1/settings", () => {
    settingsState = defaultSettings();
    return HttpResponse.json(settingsState);
  }),

  http.post("/api/v1/settings/test", () =>
    HttpResponse.json({
      ok: true,
      provider: settingsState.provider,
      model: settingsState.model,
      elapsed_ms: 42,
      message: "Provider responded.",
      error_code: "",
      prompt_tokens: 12,
      completion_tokens: 4,
      estimated_cost_usd: 0.0000029,
      cost_basis: "registry_estimate",
      request_id: "req_probe",
      usage_reported: true,
    } satisfies ConnectionTestResult),
  ),

  http.get("/api/v1/settings/models", () => {
    const spec = PROVIDERS.find((item) => item.provider === settingsState.provider);
    return HttpResponse.json({
      provider: settingsState.provider,
      /* Snapshot fallback: the catalog is the provider's preset list, so these
       * mirror PROVIDERS above rather than being an independent fixture. */
      models: (spec?.preset_models ?? []).map((id) => PRICED_MODELS[id] ?? { id }),
      source: "snapshot",
      fetched_at: "2026-07-29T00:00:00Z",
      endpoint: "",
      warning: "Save an API key before refreshing models.",
      truncated: false,
      pricing_catalog_version: "public-list-prices-2026-07-29",
      pricing_notice: "List prices for a rough estimate, not an invoice.",
    });
  }),

  http.get("/api/v1/settings/providers", () =>
    HttpResponse.json(PROVIDERS satisfies ProviderInfo[]),
  ),

  http.get("/api/v1/system/sandbox", () =>
    HttpResponse.json({
      backend: "docker",
      available: true,
      safe_for_untrusted_code: true,
      open_python_analysis_available: true,
      detail: "Docker sandbox available",
      message: "docker sandbox active; open-ended Python analysis is available.",
    } satisfies SandboxStatusView),
  ),

  http.get("/api/v1/system/capabilities", () =>
    HttpResponse.json({
      pdf_export_available: true,
      pdf_export_hint: "",
      exploration_available: false,
      exploration_hint: "Exploration release is not installed.",
    } satisfies SystemCapabilitiesView),
  ),

  /* In-memory support-doc store, so upload → list → delete behaves like the
     server across a single test. `resetSupportDocs()` runs from global setup. */
  http.get("/api/v1/projects/:projectId/support-docs", ({ params }) =>
    HttpResponse.json({
      project_id: String(params["projectId"]),
      docs: [...supportDocs],
    } satisfies SupportDocList),
  ),

  http.post("/api/v1/projects/:projectId/support-docs", async ({ request }) => {
    const form = await request.formData();
    const file = form.get("file");
    const name = file instanceof File ? file.name : "document.txt";
    const basename = name.replace(/\\/g, "/").split("/").pop() || "document.txt";
    const doc: SupportDocView = {
      doc_id: `doc_${supportDocs.length + 1}`,
      name: basename,
      byte_size: file instanceof File ? file.size : 0,
      modified_at: "2026-07-25T12:00:00Z",
    };
    supportDocs.push(doc);
    return HttpResponse.json(doc, { status: 201 });
  }),

  http.delete("/api/v1/projects/:projectId/support-docs/:docId", ({ params }) => {
    const docId = String(params["docId"]);
    const index = supportDocs.findIndex((doc) => doc.doc_id === docId);
    if (index < 0) {
      return HttpResponse.json(
        { error: { code: "support_doc_not_found", message: "not found" } },
        { status: 404 },
      );
    }
    supportDocs.splice(index, 1);
    return new HttpResponse(null, { status: 204 });
  }),

  http.get("/api/v1/sessions/:sessionId/investigations", ({ params }) =>
    HttpResponse.json(investigationsView(String(params["sessionId"]))),
  ),

  http.post("/api/v1/sessions/:sessionId/investigations/plan", ({ params }) =>
    HttpResponse.json(
      {
        session_id: String(params["sessionId"]),
        execution_session_id: "ipsess_1",
        question_ids: ["q_trend"],
        deep: false,
        job: {
          job_id: "job_plan_1",
          session_id: "ipsess_1",
          status: "queued",
          events_url: "/api/v1/jobs/job_plan_1/events",
        },
      },
      { status: 201 },
    ),
  ),

  http.post(
    "/api/v1/sessions/:sessionId/investigations/:planId/prepare-decision",
    async ({ params, request }) => {
      const body = (await request.json().catch(() => null)) as {
        decision?: "approved" | "rejected";
        reason?: string;
      } | null;
      return HttpResponse.json(
        investigationDecisionPrepared(
          String(params["sessionId"]),
          String(params["planId"]),
          body?.decision ?? "approved",
          body?.reason ?? "",
        ),
      );
    },
  ),

  http.post("/api/v1/sessions/:sessionId/investigations/:planId/approve", ({ params }) =>
    HttpResponse.json({
      session_id: String(params["sessionId"]),
      plan_id: String(params["planId"]),
      decision: "approved",
      approval_artifact_id: "invappr_1",
      plan: planView({ plan_id: String(params["planId"]), status: "approved" }),
    }),
  ),

  http.post("/api/v1/sessions/:sessionId/investigations/:planId/reject", ({ params }) =>
    HttpResponse.json({
      session_id: String(params["sessionId"]),
      plan_id: String(params["planId"]),
      decision: "rejected",
      approval_artifact_id: "invappr_2",
      plan: planView({ plan_id: String(params["planId"]), status: "rejected" }),
    }),
  ),

  http.post(
    "/api/v1/sessions/:sessionId/investigations/prepare-execute",
    async ({ params, request }) => {
      const body = (await request.json().catch(() => null)) as {
        plan_ids?: string[];
        llm?: string;
      } | null;
      return HttpResponse.json({
        session_id: String(params["sessionId"]),
        plan_session_id: SAMPLE_PLAN_RUN_ID,
        plan_ids: body?.plan_ids ?? [],
        action_hash: SAMPLE_ACTION_HASH,
        approval_token: SAMPLE_APPROVAL_TOKEN,
        expires_at: "2026-07-25T12:00:00Z",
        llm_mode: body?.llm ?? "env",
        plans: (body?.plan_ids ?? []).map((planId) =>
          planView({ plan_id: planId, status: "approved", can_execute: true }),
        ),
      });
    },
  ),

  http.post("/api/v1/sessions/:sessionId/investigations/execute", ({ params }) =>
    HttpResponse.json(
      {
        session_id: String(params["sessionId"]),
        plan_session_id: SAMPLE_PLAN_RUN_ID,
        execution_session_id: "ixsess_1",
        plan_ids: ["invplan_approved"],
        job: {
          job_id: "job_exec_1",
          session_id: "ixsess_1",
          status: "queued",
          events_url: "/api/v1/jobs/job_exec_1/events",
        },
      },
      { status: 201 },
    ),
  ),

  http.post(
    "/api/v1/sessions/:sessionId/investigations/prepare-macro-loop",
    async ({ params, request }) => {
      const body = (await request.json().catch(() => null)) as {
        plan_session_id?: string;
        llm?: string;
      } | null;
      return HttpResponse.json({
        session_id: String(params["sessionId"]),
        plan_session_id: body?.plan_session_id ?? SAMPLE_PLAN_RUN_ID,
        action_hash: SAMPLE_ACTION_HASH,
        approval_token: SAMPLE_APPROVAL_TOKEN,
        expires_at: "2026-07-25T12:00:00Z",
        depth: 2,
        rounds_cap: 1,
        questions_per_round: 4,
        llm_mode: body?.llm ?? "env",
      });
    },
  ),

  http.post("/api/v1/sessions/:sessionId/investigations/macro-loop", ({ params }) =>
    HttpResponse.json(
      {
        session_id: String(params["sessionId"]),
        plan_session_id: SAMPLE_PLAN_RUN_ID,
        execution_session_id: "mlsess_1",
        depth: 2,
        rounds_cap: 1,
        job: {
          job_id: "job_loop_1",
          session_id: "mlsess_1",
          status: "queued",
          events_url: "/api/v1/jobs/job_loop_1/events",
        },
      },
      { status: 201 },
    ),
  ),

  http.patch(
    "/api/v1/sessions/:sessionId/questions/:questionId",
    async ({ params, request }) => {
      const body = (await request.json().catch(() => null)) as Record<
        string,
        unknown
      > | null;
      const base = (questionsView(String(params["sessionId"])).questions ?? [])[0]!;
      return HttpResponse.json({
        ...base,
        question_id: String(params["questionId"]),
        card_version: base.card_version + 1,
        ...(body ?? {}),
        question:
          typeof body?.["question_en"] === "string"
            ? (body["question_en"] as string)
            : base.question,
      });
    },
  ),

  http.post(
    "/api/v1/sessions/:sessionId/questions/prepare-draft",
    async ({ params, request }) => {
      const body = (await request.json().catch(() => null)) as {
        question?: string;
        llm?: string;
      } | null;
      return HttpResponse.json({
        session_id: String(params["sessionId"]),
        action_hash: SAMPLE_ACTION_HASH,
        approval_token: SAMPLE_APPROVAL_TOKEN,
        expires_at: "2026-07-25T12:00:00Z",
        question: body?.question ?? "",
        llm_mode: body?.llm ?? "env",
      });
    },
  ),

  http.post("/api/v1/sessions/:sessionId/questions", ({ params }) =>
    HttpResponse.json(
      {
        session_id: String(params["sessionId"]),
        execution_session_id: "qdsess_1",
        question: "Which region has the highest average order amount?",
        job: {
          job_id: "job_draft_1",
          session_id: "qdsess_1",
          status: "queued",
          events_url: "/api/v1/jobs/job_draft_1/events",
        },
      },
      { status: 201 },
    ),
  ),

  http.get("/api/v1/sessions/:sessionId/report/download", ({ request }) => {
    const format = new URL(request.url).searchParams.get("format") ?? "html";
    const bodies: Record<string, [string, string]> = {
      html: ["<!doctype html><html><body>report</body></html>", "text/html"],
      pdf: ["%PDF-1.7 fake", "application/pdf"],
      md: ["# Decision report\n", "text/markdown"],
    };
    const [body, type] = bodies[format] ?? bodies["html"]!;
    return new HttpResponse(body, {
      headers: {
        "Content-Type": type,
        "Content-Disposition": `attachment; filename="report.${format}"`,
      },
    });
  }),
];
