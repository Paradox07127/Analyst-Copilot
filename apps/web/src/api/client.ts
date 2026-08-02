/* Thin typed fetch wrapper over the FastAPI backend (§6.2 of the migration
 * plan). Response/DTO types come from the OpenAPI-generated schema; error
 * bodies follow the ApiErrorEnvelope contract. */

import type { components, operations } from "./generated/schema";
import { SESSION_HEADER, sessionId } from "./session";

type Schemas = components["schemas"];
type JsonRequestBody<Operation extends keyof operations> =
  operations[Operation] extends {
    requestBody?: { content: { "application/json": infer Body } };
  }
    ? Body
    : never;

export type ProjectSummary = Schemas["ProjectSummary"];
export type ProjectCreateRequest = Schemas["ProjectCreateRequest"];
export type ProjectDeleted = Schemas["ProjectDeleted"];
export type UploadDeleted = Schemas["UploadDeleted"];
export type SessionSummary = Schemas["SessionSummary"];
export type SessionDetail = Schemas["SessionDetail"];
export type SessionPage = Schemas["Page_SessionSummary_"];
export type DatasetHandle = Schemas["DatasetHandle"];
export type DatasetColumn = Schemas["DatasetColumn"];
export type DatasetSchema = Schemas["DatasetSchema"];
export type DatasetPreview = Schemas["DatasetPreview"];
export type ApiErrorEnvelope = Schemas["ApiErrorEnvelope"];
export type ReportView = Schemas["ReportView"];
export type ArtifactSummary = Schemas["ArtifactSummary"];
export type ArtifactDetail = Schemas["ArtifactDetail"];
export type AgentHandoffDetail = Schemas["AgentHandoffDetail"];
export type ArtifactPage = Schemas["Page_ArtifactSummary_"];
export type UploadStatus = Schemas["UploadStatus"];
export type JobCreateRequest = Schemas["JobCreateRequest"];
export type JobCreated = Schemas["JobCreated"];
export type JobStatus = Schemas["JobStatus"];
export type CleaningPreviewRequest = Schemas["CleaningPreviewRequest"];
export type CleaningApplyRequest = Schemas["CleaningApplyRequest"];
export type CleaningPreviewResult = Schemas["CleaningPreviewResult"];
export type CleaningOperation = Schemas["CleaningOperation"];
export type CleaningApplied = Schemas["CleaningApplied"];
export type DataOperationStarted = Schemas["DataOperationStarted"];
export type QualityView = Schemas["QualityView"];
export type QualityIssueRow = Schemas["QualityIssueRow"];
export type QualityDatasetCard = Schemas["QualityDatasetCard"];
export type ProfilesView = Schemas["ProfilesView"];
export type DatasetProfileSummary = Schemas["DatasetProfileSummary"];
export type FieldProfileRow = Schemas["FieldProfileRow"];
export type ChartSummary = Schemas["ChartSummary"];
export type ChartPage = Schemas["Page_ChartSummary_"];
export type ChartView = Schemas["ChartView"];
export type QuestionsView = Schemas["QuestionsView"];
export type QuestionSummary = Schemas["QuestionSummary"];
export type QuestionExecutionSummary = Schemas["QuestionExecutionSummary"];
export type QuestionExecutionPrepared = Schemas["QuestionExecutionPrepared"];
export type QuestionExecutionStarted = Schemas["QuestionExecutionStarted"];
export type FindingsView = Schemas["FindingsView"];
export type FindingSummary = Schemas["FindingSummary"];
export type InvestigationLogEntry = Schemas["InvestigationLogEntry"];
export type SemanticView = Schemas["SemanticView"];
export type FieldMeaningView = Schemas["FieldMeaningView"];
export type MetricDefinitionView = Schemas["MetricDefinitionView"];
export type EntityNoteView = Schemas["EntityNoteView"];
export type VerifiedAnswerView = Schemas["VerifiedAnswerView"];
export type JoinWhitelistEntryView = Schemas["JoinWhitelistEntryView"];
export type MeaningProposalView = Schemas["MeaningProposalView"];
export type SemanticSeedsUpdated = Schemas["SemanticSeedsUpdated"];
export type ProposalReviewed = Schemas["ProposalReviewed"];
export type CompareView = Schemas["CompareView"];
export type CompareSessionSide = Schemas["CompareSessionSide"];
export type CompareMetricRow = Schemas["CompareMetricRow"];
export type CompareTextRow = Schemas["CompareTextRow"];
export type CompareArtifactDelta = Schemas["CompareArtifactDelta"];
export type CompareScopeName =
  | "questions"
  | "analysis"
  | "findings"
  | "report"
  | "artifacts"
  | "execution";
export type CompareScopeView = Schemas["CompareScopeView"];
export type CompareScopeItem = Schemas["CompareScopeItem"];
export type CompareScopeRecord = Schemas["CompareScopeRecord"];
export type CompareScopeCounts = Schemas["CompareScopeCounts"];
export type CompareNumberValue = Schemas["CompareValue_float_"];
export type CompareIntegerValue = Schemas["CompareValue_int_"];
export type CompareStringValue = Schemas["CompareValue_str_"];
export type CompareStringListValue = Schemas["CompareValue_list_str__"];
export type SkillsView = Schemas["SkillsView"];
export type SkillSummary = Schemas["SkillSummary"];
export type SkillParamSpec = Schemas["SkillParamSpec"];
export type SkillTargetDataset = Schemas["SkillTargetDataset"];
export type SkillReplayPrepared = Schemas["SkillReplayPrepared"];
export type SkillReplayStarted = Schemas["SkillReplayStarted"];
export type ChatMessageView = Schemas["ChatMessageView"];
export type ChatMessagePage = Schemas["ChatMessagePage"];
export type ChatMessageAccepted = Schemas["ChatMessageAccepted"];
export type ChatPlanRejected = Schemas["ChatPlanRejected"];
export type ChatPendingPlan = Schemas["ChatPendingPlan"];
export type ChatPendingPlanList = Schemas["ChatPendingPlanList"];
export type BoardView = Schemas["BoardView"];
export type BoardColumn = Schemas["BoardColumn"];
export type BoardCard = Schemas["BoardCard"];
export type AnalysisView = Schemas["AnalysisView"];
export type AnalysisTableView = Schemas["AnalysisTableView"];
export type StatTestRow = Schemas["StatTestRow"];
export type ModelCardView = Schemas["ModelCardView"];
export type SessionMetricsView = Schemas["SessionMetricsView"];
export type WorkspaceUsageView = Schemas["WorkspaceUsageView"];
export type UsageDay = Schemas["UsageDay"];
export type UsageRecentSession = Schemas["UsageRecentSession"];
export type SessionStepMetricRow = Schemas["SessionStepMetricRow"];
export type TraceEventPage = Schemas["TraceEventPage"];
export type TraceEventRow = Schemas["TraceEventRow"];
export type ClientFailureRequest = Schemas["ClientFailureRequest"];
export type ClientFailureRecorded = Schemas["ClientFailureRecorded"];
export type DecisionReportView = Schemas["DecisionReportView"];
export type DecisionReportSectionView = Schemas["DecisionReportSectionView"];
export type CandidateDecisionView = Schemas["CandidateDecisionView"];
export type DecisionEvidenceRefView = Schemas["DecisionEvidenceRefView"];
export type SkillPlanCandidate = Schemas["SkillPlanCandidate"];
export type RelationshipGraphView = Schemas["RelationshipGraphView"];
export type RelationshipNode = Schemas["RelationshipNode"];
export type RelationshipEdge = Schemas["RelationshipEdge"];
export type RelationshipValidationPrepared =
  Schemas["RelationshipValidationPrepared"];
export type RelationshipValidationStarted =
  Schemas["RelationshipValidationStarted"];
export type RelationshipDiscoveryStarted =
  Schemas["RelationshipDiscoveryStarted"];
export type SessionDeleted = Schemas["SessionDeleted"];
export type SettingsView = Schemas["SettingsView"];
export type SettingsPatch = Schemas["SettingsPatch"];
export type ProviderInfo = Schemas["ProviderInfo"];
export type ModelCatalog = Schemas["ModelCatalog"];
export type ConnectionTestResult = Schemas["ConnectionTestResult"];
export type AboutInfo = Schemas["AboutInfo"];
export type PrecleaningOptions = Schemas["PrecleaningOptions"];
export type KnowledgePromotionPrepared = Schemas["KnowledgePromotionPrepared"];
export type KnowledgePromoted = Schemas["KnowledgePromoted"];
export type ReportGenerationStarted = Schemas["ReportGenerationStarted"];
export type SessionForkStarted = Schemas["SessionForkStarted"];
export type SessionForkRequest = Schemas["SessionForkRequest"];
export type SupportDocView = Schemas["SupportDocView"];
export type SupportDocList = Schemas["SupportDocList"];
export type SandboxStatusView = Schemas["SandboxStatusView"];
export type SystemCapabilitiesView = Schemas["SystemCapabilitiesView"];
export type QuestionDraftPrepared = Schemas["QuestionDraftPrepared"];
export type QuestionDraftStarted = Schemas["QuestionDraftStarted"];
export type InvestigationsView = Schemas["InvestigationsView"];
export type InvestigationPlanView = Schemas["InvestigationPlanView"];
export type InvestigationGateView = Schemas["InvestigationGateView"];
export type CleaningLogView = Schemas["CleaningLogView"];
export type CleaningLogSummaryRow = Schemas["CleaningLogSummaryRow"];
export type CleaningLogOperationRow = Schemas["CleaningLogOperationRow"];
export type CleaningLogGuardrailRow = Schemas["CleaningLogGuardrailRow"];
export type CleaningLogSuggestionRow = Schemas["CleaningLogSuggestionRow"];
export type CleaningRawView = Schemas["CleaningRawView"];
export type RawChartView = Schemas["RawChartView"];
export type RawDataPreviewView = Schemas["RawDataPreviewView"];
export type SessionDebugView = Schemas["SessionDebugView"];
export type SessionDebugSummary = Schemas["SessionDebugSummary"];
export type ReportQualitySummary = Schemas["ReportQualitySummary"];
export type DebugTimelineRow = Schemas["DebugTimelineRow"];
export type DebugLlmCallRow = Schemas["DebugLlmCallRow"];
export type DebugToolCallRow = Schemas["DebugToolCallRow"];
export type DebugErrorRow = Schemas["DebugErrorRow"];
export type DebugArtifactRow = Schemas["DebugArtifactRow"];
export type LlmDebugRecord = Schemas["LlmDebugRecord"];
export type LlmDebugPage = Schemas["Page_LlmDebugRecord_"];
export type ColumnDistributionsView = Schemas["ColumnDistributionsView"];
export type ColumnDistribution = Schemas["ColumnDistribution"];
export type CustomChartRequest = Schemas["CustomChartRequest"];
export type CustomChartView = Schemas["CustomChartView"];
export type DecisionCoverageView = Schemas["DecisionCoverageView"];
export type VerifiedRelationView = Schemas["VerifiedRelationView"];
export type VerifiedRelationDeleteRequest =
  Schemas["VerifiedRelationDeleteRequest"];
export type VerifiedRelationsUpdated = Schemas["VerifiedRelationsUpdated"];
export type DecisionStoryView = Schemas["DecisionStoryView"];
export type DecisionStoryFindingView = Schemas["DecisionStoryFindingView"];
export type DecisionStoryDraftRequest = Schemas["DecisionStoryDraftRequest"];
export type DecisionStoryDraftStarted = Schemas["DecisionStoryDraftStarted"];
export type DecisionReportGenerateRequest =
  Schemas["DecisionReportGenerateRequest"];
export type DecisionReportGenerationStarted =
  Schemas["DecisionReportGenerationStarted"];
export type InvestigationPlanBuildStarted = Schemas["InvestigationPlanBuildStarted"];
export type InvestigationDecisionPrepared = Schemas["InvestigationDecisionPrepared"];
export type InvestigationDecisionRecorded = Schemas["InvestigationDecisionRecorded"];
export type InvestigationExecutionPrepared = Schemas["InvestigationExecutionPrepared"];
export type InvestigationExecutionStarted = Schemas["InvestigationExecutionStarted"];
export type MacroLoopPrepared = Schemas["MacroLoopPrepared"];
export type MacroLoopStarted = Schemas["MacroLoopStarted"];
export type MacroLoopView = Schemas["MacroLoopView"];
export type QuestionPrepareRequest = JsonRequestBody<
  "prepare_question_execution_api_v1_sessions__session_id__questions__question_id__prepare_post"
>;
export type ReportGenerateRequest = JsonRequestBody<
  "generate_report_api_v1_sessions__session_id__report_generate_post"
>;
export type SemanticSeedsUpdateRequest = Schemas["SemanticSeedsUpdateRequest"];
export type JoinReviewRequest = Schemas["JoinReviewRequest"];
export type ProposalAcceptRequest = Schemas["ProposalAcceptRequest"];
export type ProposalRejectRequest = Schemas["ProposalRejectRequest"];
export type SeedImportRequest = Schemas["SeedImportRequest"];
export type SkillReplayPrepareRequest = Schemas["SkillReplayPrepareRequest"];
export type SkillReplayExecuteRequest = Schemas["SkillReplayExecuteRequest"];
export type QuestionExecuteRequest = Schemas["QuestionExecuteRequest"];
export type ChatSendRequest = Schemas["ChatSendRequest"];
export type ChatPlanDecisionRequest = Schemas["ChatPlanDecisionRequest"];
export type BoardUpdateRequest = Schemas["BoardUpdateRequest"];
export type SkillSaveRequest = Schemas["SkillSaveRequest"];
export type RelationshipValidateRequest = Schemas["RelationshipValidateRequest"];
export type FindingPromoteRequest = Schemas["FindingPromoteRequest"];
export type QuestionCardEdit = Schemas["QuestionCardEditRequest"];
export type QuestionDraftPrepareRequest = Schemas["QuestionDraftPrepareRequest"];
export type QuestionDraftRequest = Schemas["QuestionDraftRequest"];
export type InvestigationPlanRequest = Schemas["InvestigationPlanRequest"];
export type InvestigationDecisionPrepareRequest =
  Schemas["InvestigationDecisionPrepareRequest"];
export type InvestigationDecisionRequest = Schemas["InvestigationDecisionRequest"];
export type InvestigationExecutePrepareRequest =
  Schemas["InvestigationExecutePrepareRequest"];
export type InvestigationExecuteRequest = Schemas["InvestigationExecuteRequest"];
export type MacroLoopPrepareRequest = Schemas["MacroLoopPrepareRequest"];
export type MacroLoopRequest = Schemas["MacroLoopRequest"];

export type ReportExportFormat = "html" | "pdf" | "md";

const BASE = "/api/v1";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

function parseErrorEnvelope(body: unknown): { code: string; message: string } | null {
  if (typeof body !== "object" || body === null) return null;
  const error = (body as Partial<ApiErrorEnvelope>).error;
  if (
    typeof error === "object" &&
    error !== null &&
    typeof error.code === "string" &&
    typeof error.message === "string"
  ) {
    return { code: error.code, message: error.message };
  }
  return null;
}

export async function apiFetch<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const headers = new Headers(init.headers);
  /* Correlates the request with its server-side Settings session (§6.0). */
  if (!headers.has(SESSION_HEADER)) headers.set(SESSION_HEADER, sessionId());
  /* Remote mode requires an explicit non-simple-request signal on every
   * mutation. Local mode ignores it, keeping the desktop contract unchanged. */
  if (
    ["POST", "PUT", "PATCH", "DELETE"].includes(
      (init.method ?? "GET").toUpperCase(),
    ) &&
    !headers.has("X-EDA-CSRF")
  ) {
    headers.set("X-EDA-CSRF", "1");
  }
  /* FormData must keep its browser-generated multipart boundary header. */
  if (
    init.body !== undefined &&
    !(init.body instanceof FormData) &&
    !headers.has("Content-Type")
  ) {
    headers.set("Content-Type", "application/json");
  }

  /* Absolute URL: Node's fetch (used by vitest/jsdom) rejects relative ones. */
  const url = new URL(`${BASE}${path}`, window.location.origin);
  const response = await fetch(url, { ...init, headers });

  if (!response.ok) {
    let body: unknown = null;
    try {
      body = await response.json();
    } catch {
      /* non-JSON error body */
    }
    const parsed = parseErrorEnvelope(body);
    throw new ApiError(
      response.status,
      parsed?.code ?? "http_error",
      parsed?.message ?? `API request failed with status ${response.status}`,
    );
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

/* Binary sibling of apiFetch: report exports come back as a file stream, not
 * JSON, so the body is read as a Blob and the server-chosen name is taken from
 * Content-Disposition. Error bodies still follow the JSON envelope. */
export async function apiDownload(
  path: string,
  fallbackFilename: string,
  init: RequestInit = {},
): Promise<{ blob: Blob; filename: string }> {
  const headers = new Headers(init.headers);
  if (!headers.has(SESSION_HEADER)) headers.set(SESSION_HEADER, sessionId());
  const url = new URL(`${BASE}${path}`, window.location.origin);
  const response = await fetch(url, { ...init, headers });

  if (!response.ok) {
    let body: unknown = null;
    try {
      body = await response.json();
    } catch {
      /* non-JSON error body */
    }
    const parsed = parseErrorEnvelope(body);
    throw new ApiError(
      response.status,
      parsed?.code ?? "http_error",
      parsed?.message ?? `Download failed with status ${response.status}`,
    );
  }

  return {
    blob: await response.blob(),
    filename:
      filenameFromContentDisposition(response.headers.get("Content-Disposition")) ??
      fallbackFilename,
  };
}

export function filenameFromContentDisposition(
  header: string | null,
): string | null {
  if (!header) return null;
  const match = /filename="?([^";]+)"?/i.exec(header);
  const raw = match?.[1]?.trim();
  if (!raw) return null;
  /* The server already sanitizes this, but the value decides a local file
   * name — keep it a basename here too rather than trusting the header. */
  const basename = raw.replace(/\\/g, "/").split("/").pop();
  return basename || null;
}

/* Hands the blob to the browser's own download flow. Kept out of components so
 * they never touch object URLs (which leak until revoked). */
export function saveBlob(blob: Blob, filename: string): void {
  const objectUrl = URL.createObjectURL(blob);
  try {
    const anchor = document.createElement("a");
    anchor.href = objectUrl;
    anchor.download = filename;
    anchor.rel = "noopener";
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

const enc = encodeURIComponent;

export const api = {
  listProjects: (signal?: AbortSignal) =>
    apiFetch<ProjectSummary[]>("/projects", { signal }),

  /* 201 when created, 200 when the id already existed — both return the
     project, so the caller does not branch on status. */
  createProject: (body: ProjectCreateRequest, idempotencyKey: string) =>
    apiFetch<ProjectSummary>("/projects", {
      method: "POST",
      body: JSON.stringify(body),
      headers: { "Idempotency-Key": idempotencyKey },
    }),

  reorderProjects: (projectIds: string[]) =>
    apiFetch<ProjectSummary[]>("/projects/order", {
      method: "PUT",
      body: JSON.stringify({ project_ids: projectIds }),
    }),

  renameProject: (projectId: string, name: string) =>
    apiFetch<ProjectSummary>(`/projects/${enc(projectId)}`, {
      method: "PATCH",
      body: JSON.stringify({ name }),
    }),

  renameSession: (sessionId: string, name: string) =>
    apiFetch<SessionSummary>(`/sessions/${enc(sessionId)}`, {
      method: "PATCH",
      body: JSON.stringify({ name }),
    }),

  deleteProject: (projectId: string) =>
    apiFetch<ProjectDeleted>(`/projects/${enc(projectId)}/self`, { method: "DELETE" }),

  listUploads: (projectId: string, signal?: AbortSignal) =>
    apiFetch<DatasetHandle[]>(`/projects/${enc(projectId)}/uploads`, { signal }),

  deleteUpload: (projectId: string, datasetId: string) =>
    apiFetch<UploadDeleted>(
      `/projects/${enc(projectId)}/uploads/${enc(datasetId)}`,
      { method: "DELETE" },
    ),

  listSessions: (
    projectId: string,
    opts: { limit?: number; cursor?: string; q?: string } = {},
    signal?: AbortSignal,
  ) => {
    const params = new URLSearchParams();
    if (opts.limit !== undefined) params.set("limit", String(opts.limit));
    if (opts.cursor) params.set("cursor", opts.cursor);
    if (opts.q) params.set("q", opts.q);
    const qs = params.size > 0 ? `?${params.toString()}` : "";
    return apiFetch<SessionPage>(`/projects/${enc(projectId)}/sessions${qs}`, {
      signal,
    });
  },

  getSession: (sessionId: string, signal?: AbortSignal) =>
    apiFetch<SessionDetail>(`/sessions/${enc(sessionId)}`, { signal }),

  /* Irreversible — the caller must confirm before invoking this. */
  deleteSession: (sessionId: string) =>
    apiFetch<SessionDeleted>(`/sessions/${enc(sessionId)}`, { method: "DELETE" }),

  listDatasets: (sessionId: string, signal?: AbortSignal) =>
    apiFetch<DatasetHandle[]>(`/sessions/${enc(sessionId)}/datasets`, { signal }),

  getDatasetSchema: (sessionId: string, datasetId: string, signal?: AbortSignal) =>
    apiFetch<DatasetSchema>(
      `/sessions/${enc(sessionId)}/datasets/${enc(datasetId)}/schema`,
      { signal },
    ),

  getDatasetPreview: (
    sessionId: string,
    datasetId: string,
    opts: { limit?: number; offset?: number } = {},
    signal?: AbortSignal,
  ) => {
    const params = new URLSearchParams();
    if (opts.limit !== undefined) params.set("limit", String(opts.limit));
    if (opts.offset !== undefined) params.set("offset", String(opts.offset));
    const qs = params.size > 0 ? `?${params.toString()}` : "";
    return apiFetch<DatasetPreview>(
      `/sessions/${enc(sessionId)}/datasets/${enc(datasetId)}/preview${qs}`,
      { signal },
    );
  },

  getReport: (sessionId: string, signal?: AbortSignal) =>
    apiFetch<ReportView>(`/sessions/${enc(sessionId)}/report`, { signal }),

  /* "md" renders the project's decision report, matching the three downloads
     the Report page download formats. */
  downloadReport: (sessionId: string, format: ReportExportFormat) =>
    apiDownload(
      `/sessions/${enc(sessionId)}/report/download?format=${enc(format)}`,
      `${sessionId}.${format}`,
    ),

  listArtifacts: (
    sessionId: string,
    opts: { type?: string; limit?: number; cursor?: string } = {},
    signal?: AbortSignal,
  ) => {
    const params = new URLSearchParams();
    if (opts.type) params.set("type", opts.type);
    if (opts.limit !== undefined) params.set("limit", String(opts.limit));
    if (opts.cursor) params.set("cursor", opts.cursor);
    const qs = params.size > 0 ? `?${params.toString()}` : "";
    return apiFetch<ArtifactPage>(`/sessions/${enc(sessionId)}/artifacts${qs}`, {
      signal,
    });
  },

  getArtifact: (sessionId: string, artifactId: string, signal?: AbortSignal) =>
    apiFetch<ArtifactDetail>(
      `/sessions/${enc(sessionId)}/artifacts/${enc(artifactId)}`,
      { signal },
    ),

  getQuality: (sessionId: string, signal?: AbortSignal) =>
    apiFetch<QualityView>(`/sessions/${enc(sessionId)}/quality`, { signal }),

  getProfiles: (sessionId: string, signal?: AbortSignal) =>
    apiFetch<ProfilesView>(`/sessions/${enc(sessionId)}/profiles`, { signal }),

  listCharts: (
    sessionId: string,
    opts: { limit?: number; cursor?: string } = {},
    signal?: AbortSignal,
  ) => {
    const params = new URLSearchParams();
    if (opts.limit !== undefined) params.set("limit", String(opts.limit));
    if (opts.cursor) params.set("cursor", opts.cursor);
    const qs = params.size > 0 ? `?${params.toString()}` : "";
    return apiFetch<ChartPage>(`/sessions/${enc(sessionId)}/charts${qs}`, { signal });
  },

  getChart: (sessionId: string, chartId: string, signal?: AbortSignal) =>
    apiFetch<ChartView>(`/sessions/${enc(sessionId)}/charts/${enc(chartId)}`, {
      signal,
    }),

  createUpload: (projectId: string, file: File, signal?: AbortSignal) => {
    const form = new FormData();
    form.append("file", file);
    return apiFetch<UploadStatus>(`/projects/${enc(projectId)}/uploads`, {
      method: "POST",
      body: form,
      signal,
    });
  },

  createJob: (sessionId: string, body: JobCreateRequest, idempotencyKey: string) =>
    apiFetch<JobCreated>(`/sessions/${enc(sessionId)}/jobs`, {
      method: "POST",
      body: JSON.stringify(body),
      headers: { "Idempotency-Key": idempotencyKey },
    }),

  getJob: (jobId: string, signal?: AbortSignal) =>
    apiFetch<JobStatus>(`/jobs/${enc(jobId)}`, { signal }),

  cancelJob: (jobId: string) =>
    apiFetch<JobStatus>(`/jobs/${enc(jobId)}/cancel`, { method: "POST" }),

  previewCleaning: (
    sessionId: string,
    body: CleaningPreviewRequest,
    idempotencyKey: string,
  ) =>
    apiFetch<DataOperationStarted>(`/sessions/${enc(sessionId)}/cleaning/preview`, {
      method: "POST",
      body: JSON.stringify(body),
      headers: { "Idempotency-Key": idempotencyKey },
    }),

  applyCleaning: (
    sessionId: string,
    body: CleaningApplyRequest,
    idempotencyKey: string,
  ) =>
    apiFetch<DataOperationStarted>(`/sessions/${enc(sessionId)}/cleaning/apply`, {
      method: "POST",
      body: JSON.stringify(body),
      headers: { "Idempotency-Key": idempotencyKey },
    }),

  getCleaningPreviewResult: (jobId: string, signal?: AbortSignal) =>
    apiFetch<CleaningPreviewResult>(
      `/jobs/${enc(jobId)}/cleaning-preview-result`,
      { signal },
    ),

  getCleaningApplyResult: (jobId: string, signal?: AbortSignal) =>
    apiFetch<CleaningApplied>(`/jobs/${enc(jobId)}/cleaning-apply-result`, {
      signal,
    }),

  listQuestions: (sessionId: string, signal?: AbortSignal) =>
    apiFetch<QuestionsView>(`/sessions/${enc(sessionId)}/questions`, { signal }),

  prepareQuestion: (
    sessionId: string,
    questionId: string,
    body?: QuestionPrepareRequest,
  ) =>
    apiFetch<QuestionExecutionPrepared>(
      `/sessions/${enc(sessionId)}/questions/${enc(questionId)}/prepare`,
      { method: "POST", body: JSON.stringify(body) },
    ),

  listFindings: (sessionId: string, signal?: AbortSignal) =>
    apiFetch<FindingsView>(`/sessions/${enc(sessionId)}/findings`, { signal }),

  getSemantic: (
    sessionId: string,
    opts: { limit?: number; cursor?: string } = {},
    signal?: AbortSignal,
  ) => {
    const params = new URLSearchParams();
    if (opts.limit !== undefined) params.set("limit", String(opts.limit));
    if (opts.cursor) params.set("cursor", opts.cursor);
    const qs = params.size > 0 ? `?${params.toString()}` : "";
    return apiFetch<SemanticView>(`/sessions/${enc(sessionId)}/semantic${qs}`, { signal });
  },

  /* One seeds.json, one version: a PUT may carry several classes at once.
   * field_meanings is required; the other three are left untouched when
   * omitted, so an editor for one class cannot clear the others. */
  updateSemanticSeeds: (
    sessionId: string,
    body: SemanticSeedsUpdateRequest,
  ) =>
    apiFetch<SemanticSeedsUpdated>(`/sessions/${enc(sessionId)}/semantic/seeds`, {
      method: "PUT",
      body: JSON.stringify(body),
    }),

  confirmSemanticJoin: (
    sessionId: string,
    label: string,
    expectedVersion: number,
  ) => {
    const body: JoinReviewRequest = {
      label,
      expected_version: expectedVersion,
    };
    return apiFetch<JoinWhitelistEntryView>(
      `/sessions/${enc(sessionId)}/semantic/joins/confirm`,
      { method: "POST", body: JSON.stringify(body) },
    );
  },

  revokeSemanticJoin: (
    sessionId: string,
    label: string,
    expectedVersion: number,
  ) => {
    const body: JoinReviewRequest = {
      label,
      expected_version: expectedVersion,
    };
    return apiFetch<JoinWhitelistEntryView>(
      `/sessions/${enc(sessionId)}/semantic/joins/revoke`,
      { method: "POST", body: JSON.stringify(body) },
    );
  },

  acceptMeaningProposal: (
    sessionId: string,
    body: ProposalAcceptRequest,
  ) =>
    apiFetch<ProposalReviewed>(`/sessions/${enc(sessionId)}/semantic/proposals/accept`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  rejectMeaningProposal: (sessionId: string, body: ProposalRejectRequest) =>
    apiFetch<ProposalReviewed>(`/sessions/${enc(sessionId)}/semantic/proposals/reject`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  compareSessions: (left: string, right: string, signal?: AbortSignal) =>
    apiFetch<CompareView>(
      `/compare?left=${enc(left)}&right=${enc(right)}`,
      { signal },
    ),

  compareScope: (
    scope: CompareScopeName,
    left: string,
    right: string,
    opts: {
      filter?: "all" | "differences";
      limit?: number;
      cursor?: string;
    } = {},
    signal?: AbortSignal,
  ) => {
    const params = new URLSearchParams({ left, right });
    if (opts.filter) params.set("filter", opts.filter);
    if (opts.limit !== undefined) params.set("limit", String(opts.limit));
    if (opts.cursor) params.set("cursor", opts.cursor);
    return apiFetch<CompareScopeView>(
      `/compare/${enc(scope)}?${params.toString()}`,
      { signal },
    );
  },

  listSkills: (
    sessionId: string,
    opts: { limit?: number; cursor?: string } = {},
    signal?: AbortSignal,
  ) => {
    const params = new URLSearchParams();
    if (opts.limit !== undefined) params.set("limit", String(opts.limit));
    if (opts.cursor) params.set("cursor", opts.cursor);
    const qs = params.size > 0 ? `?${params.toString()}` : "";
    return apiFetch<SkillsView>(`/sessions/${enc(sessionId)}/skills${qs}`, { signal });
  },

  /* Binds a builtin seed to this session's data and saves it into the project
   * library; idempotent on the (seed, relation, bindings) triple. */
  importSeedSkill: (
    sessionId: string,
    seedId: string,
    body: SeedImportRequest,
    idempotencyKey: string,
  ) =>
    apiFetch<SkillSummary>(`/sessions/${enc(sessionId)}/skills/${enc(seedId)}/import`, {
      method: "POST",
      body: JSON.stringify(body),
      headers: { "Idempotency-Key": idempotencyKey },
    }),

  prepareSkillReplay: (
    sessionId: string,
    skillId: string,
    body: SkillReplayPrepareRequest,
  ) =>
    apiFetch<SkillReplayPrepared>(
      `/sessions/${enc(sessionId)}/skills/${enc(skillId)}/prepare`,
      { method: "POST", body: JSON.stringify(body) },
    ),

  /* No skill content: execute replays exactly what the approval froze. */
  executeSkillReplay: (
    sessionId: string,
    skillId: string,
    body: SkillReplayExecuteRequest,
    idempotencyKey: string,
  ) =>
    apiFetch<SkillReplayStarted>(
      `/sessions/${enc(sessionId)}/skills/${enc(skillId)}/execute`,
      {
        method: "POST",
        body: JSON.stringify(body),
        headers: { "Idempotency-Key": idempotencyKey },
      },
    ),

  /* No llm field: execution runs the mode bound into the approval at prepare. */
  executeQuestion: (
    sessionId: string,
    questionId: string,
    body: QuestionExecuteRequest,
    idempotencyKey: string,
  ) =>
    apiFetch<QuestionExecutionStarted>(
      `/sessions/${enc(sessionId)}/questions/${enc(questionId)}/execute`,
      {
        method: "POST",
        body: JSON.stringify(body),
        headers: { "Idempotency-Key": idempotencyKey },
      },
    ),

  listChatMessages: (
    sessionId: string,
    opts: { limit?: number; cursor?: string } = {},
    signal?: AbortSignal,
  ) => {
    const params = new URLSearchParams();
    if (opts.limit !== undefined) params.set("limit", String(opts.limit));
    if (opts.cursor) params.set("cursor", opts.cursor);
    const qs = params.size > 0 ? `?${params.toString()}` : "";
    return apiFetch<ChatMessagePage>(`/sessions/${enc(sessionId)}/chat/messages${qs}`, {
      signal,
    });
  },

  /* 202: the turn runs in the background; follow stream_url for progress. */
  sendChatMessage: (sessionId: string, body: ChatSendRequest) =>
    apiFetch<ChatMessageAccepted>(`/sessions/${enc(sessionId)}/chat/messages`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  /* Recovery path: the approval token only ever reaches the client over SSE,
     so a reload or a restarted API asks for the outstanding plans again. */
  listChatPendingPlans: (sessionId: string, signal?: AbortSignal) =>
    apiFetch<ChatPendingPlanList>(`/sessions/${enc(sessionId)}/chat/pending-plans`, {
      signal,
    }),

  approveChatPlan: (
    sessionId: string,
    planId: string,
    body: ChatPlanDecisionRequest,
  ) =>
    apiFetch<ChatMessageAccepted>(
      `/sessions/${enc(sessionId)}/chat/plans/${enc(planId)}/approve`,
      { method: "POST", body: JSON.stringify(body) },
    ),

  rejectChatPlan: (
    sessionId: string,
    planId: string,
    body: ChatPlanDecisionRequest,
  ) =>
    apiFetch<ChatPlanRejected>(
      `/sessions/${enc(sessionId)}/chat/plans/${enc(planId)}/reject`,
      { method: "POST", body: JSON.stringify(body) },
    ),

  getBoard: (projectId: string, boardId: string, signal?: AbortSignal) =>
    apiFetch<BoardView>(
      `/projects/${enc(projectId)}/boards/${enc(boardId)}`,
      { signal },
    ),

  putBoard: (
    projectId: string,
    boardId: string,
    body: BoardUpdateRequest,
    idempotencyKey: string,
  ) =>
    apiFetch<BoardView>(`/projects/${enc(projectId)}/boards/${enc(boardId)}`, {
      method: "PUT",
      body: JSON.stringify(body),
      headers: { "Idempotency-Key": idempotencyKey },
    }),

  getAnalysis: (sessionId: string, signal?: AbortSignal) =>
    apiFetch<AnalysisView>(`/sessions/${enc(sessionId)}/analysis`, { signal }),

  getSessionMetrics: (sessionId: string, signal?: AbortSignal) =>
    apiFetch<SessionMetricsView>(`/sessions/${enc(sessionId)}/metrics`, { signal }),

  getWorkspaceUsage: (days: number, signal?: AbortSignal) =>
    apiFetch<WorkspaceUsageView>(`/usage?days=${days}`, { signal }),

  listTraceEvents: (
    sessionId: string,
    opts: { limit?: number; cursor?: string; type?: string } = {},
    signal?: AbortSignal,
  ) => {
    const params = new URLSearchParams();
    if (opts.limit !== undefined) params.set("limit", String(opts.limit));
    if (opts.cursor) params.set("cursor", opts.cursor);
    if (opts.type) params.set("type", opts.type);
    const qs = params.size > 0 ? `?${params.toString()}` : "";
    return apiFetch<TraceEventPage>(`/sessions/${enc(sessionId)}/trace${qs}`, { signal });
  },

  recordClientFailure: (sessionId: string, body: ClientFailureRequest) =>
    apiFetch<ClientFailureRecorded>(`/sessions/${enc(sessionId)}/client-failures`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  getDecisionReport: (sessionId: string, signal?: AbortSignal) =>
    apiFetch<DecisionReportView>(`/sessions/${enc(sessionId)}/decision-report`, { signal }),

  getCleaningLog: (sessionId: string, signal?: AbortSignal) =>
    apiFetch<CleaningLogView>(`/sessions/${enc(sessionId)}/cleaning/log`, { signal }),

  getCleaningRaw: (sessionId: string, signal?: AbortSignal) =>
    apiFetch<CleaningRawView>(`/sessions/${enc(sessionId)}/cleaning/raw`, { signal }),

  getSessionDebug: (
    sessionId: string,
    opts: { limit?: number; cursor?: string } = {},
    signal?: AbortSignal,
  ) => {
    const params = new URLSearchParams();
    if (opts.limit !== undefined) params.set("limit", String(opts.limit));
    if (opts.cursor) params.set("cursor", opts.cursor);
    const qs = params.size > 0 ? `?${params.toString()}` : "";
    return apiFetch<SessionDebugView>(`/sessions/${enc(sessionId)}/debug${qs}`, { signal });
  },

  downloadDebugLog: (sessionId: string) =>
    apiDownload(`/sessions/${enc(sessionId)}/debug/log`, `${sessionId}-debug.jsonl`),

  listLlmDebugCalls: (
    sessionId: string,
    opts: { limit?: number; cursor?: string } = {},
    signal?: AbortSignal,
  ) => {
    const params = new URLSearchParams();
    if (opts.limit !== undefined) params.set("limit", String(opts.limit));
    if (opts.cursor) params.set("cursor", opts.cursor);
    const qs = params.size > 0 ? `?${params.toString()}` : "";
    return apiFetch<LlmDebugPage>(`/sessions/${enc(sessionId)}/debug/llm-calls${qs}`, {
      signal,
    });
  },

  deleteVerifiedRelation: (
    sessionId: string,
    body: VerifiedRelationDeleteRequest,
  ) =>
    apiFetch<VerifiedRelationsUpdated>(
      `/sessions/${enc(sessionId)}/semantic/verified-relations/delete`,
      { method: "POST", body: JSON.stringify(body) },
    ),

  getDecisionStory: (sessionId: string, signal?: AbortSignal) =>
    apiFetch<DecisionStoryView>(`/sessions/${enc(sessionId)}/decision-story`, {
      signal,
    }),

  createDecisionStoryDraft: (sessionId: string, body: DecisionStoryDraftRequest) =>
    apiFetch<DecisionStoryDraftStarted>(
      `/sessions/${enc(sessionId)}/decision-story/drafts`,
      { method: "POST", body: JSON.stringify(body) },
    ),

  generateDecisionReport: (
    sessionId: string,
    body: DecisionReportGenerateRequest,
  ) =>
    apiFetch<DecisionReportGenerationStarted>(
      `/sessions/${enc(sessionId)}/decision-report/generate`,
      { method: "POST", body: JSON.stringify(body) },
    ),

  buildCustomChart: (
    sessionId: string,
    body: CustomChartRequest,
    idempotencyKey: string,
  ) =>
    apiFetch<DataOperationStarted>(`/sessions/${enc(sessionId)}/charts/custom`, {
      method: "POST",
      body: JSON.stringify(body),
      headers: { "Idempotency-Key": idempotencyKey },
    }),

  getCustomChartResult: (jobId: string, signal?: AbortSignal) =>
    apiFetch<CustomChartView>(`/jobs/${enc(jobId)}/custom-chart-result`, {
      signal,
    }),

  getDecisionCoverage: (sessionId: string, signal?: AbortSignal) =>
    apiFetch<DecisionCoverageView>(`/sessions/${enc(sessionId)}/decision-coverage`, {
      signal,
    }),

  startDatasetDistributions: (
    sessionId: string,
    datasetId: string,
    idempotencyKey: string,
  ) =>
    apiFetch<DataOperationStarted>(
      `/sessions/${enc(sessionId)}/datasets/${enc(datasetId)}/distributions`,
      {
        method: "POST",
        headers: { "Idempotency-Key": idempotencyKey },
      },
    ),

  getDatasetDistributionsResult: (jobId: string, signal?: AbortSignal) =>
    apiFetch<ColumnDistributionsView>(
      `/jobs/${enc(jobId)}/dataset-distributions-result`,
      { signal },
    ),

  saveSkill: (
    sessionId: string,
    body: SkillSaveRequest,
    idempotencyKey: string,
  ) =>
    apiFetch<SkillSummary>(`/sessions/${enc(sessionId)}/skills`, {
      method: "POST",
      body: JSON.stringify(body),
      headers: { "Idempotency-Key": idempotencyKey },
    }),

  /* 204: the library shrinks; seed templates are not deletable (409). */
  deleteSkill: (projectId: string, skillId: string, idempotencyKey: string) =>
    apiFetch<void>(`/projects/${enc(projectId)}/skills/${enc(skillId)}`, {
      method: "DELETE",
      headers: { "Idempotency-Key": idempotencyKey },
    }),

  getRelationships: (sessionId: string, signal?: AbortSignal) =>
    apiFetch<RelationshipGraphView>(`/sessions/${enc(sessionId)}/relationships`, {
      signal,
    }),

  /* Discovery reads every source CSV, so this only queues a job. */
  discoverRelationships: (sessionId: string, idempotencyKey: string) =>
    apiFetch<RelationshipDiscoveryStarted>(
      `/sessions/${enc(sessionId)}/relationships/discover`,
      {
        method: "POST",
        headers: { "Idempotency-Key": idempotencyKey },
      },
    ),

  prepareRelationshipValidation: (sessionId: string, relationshipId: string) =>
    apiFetch<RelationshipValidationPrepared>(
      `/sessions/${enc(sessionId)}/relationships/${enc(relationshipId)}/prepare-validate`,
      { method: "POST" },
    ),

  /* No candidate content: validation runs exactly what the approval froze. */
  validateRelationship: (
    sessionId: string,
    relationshipId: string,
    body: RelationshipValidateRequest,
    idempotencyKey: string,
  ) =>
    apiFetch<RelationshipValidationStarted>(
      `/sessions/${enc(sessionId)}/relationships/${enc(relationshipId)}/validate`,
      {
        method: "POST",
        body: JSON.stringify(body),
        headers: { "Idempotency-Key": idempotencyKey },
      },
    ),

  confirmRelationship: (
    sessionId: string,
    relationshipId: string,
    expectedVersion: number,
  ) =>
    apiFetch<RelationshipEdge>(
      `/sessions/${enc(sessionId)}/relationships/${enc(relationshipId)}/confirm`,
      {
        method: "POST",
        body: JSON.stringify({ expected_version: expectedVersion }),
      },
    ),

  revokeRelationship: (
    sessionId: string,
    relationshipId: string,
    expectedVersion: number,
  ) =>
    apiFetch<RelationshipEdge>(
      `/sessions/${enc(sessionId)}/relationships/${enc(relationshipId)}/revoke`,
      {
        method: "POST",
        body: JSON.stringify({ expected_version: expectedVersion }),
      },
    ),

  getSettings: (signal?: AbortSignal) =>
    apiFetch<SettingsView>("/settings", { signal }),

  /* The API key is write-only: it goes up in this patch and never comes back
     down — SettingsView reports only api_key_set + api_key_last4. */
  updateSettings: (body: SettingsPatch, expectedVersion?: number) =>
    apiFetch<SettingsView>("/settings", {
      method: "PUT",
      body: JSON.stringify(body),
      headers:
        expectedVersion === undefined
          ? undefined
          : { "If-Match": `"${expectedVersion}"` },
    }),

  resetSettings: (expectedVersion?: number) =>
    apiFetch<SettingsView>("/settings", {
      method: "DELETE",
      headers:
        expectedVersion === undefined
          ? undefined
          : { "If-Match": `"${expectedVersion}"` },
    }),

  testConnection: () =>
    apiFetch<ConnectionTestResult>("/settings/test", { method: "POST" }),

  listModels: (signal?: AbortSignal) =>
    apiFetch<ModelCatalog>("/settings/models", { signal }),

  refreshModels: () =>
    apiFetch<ModelCatalog>("/settings/models/refresh", { method: "POST" }),

  listProviders: (signal?: AbortSignal) =>
    apiFetch<ProviderInfo[]>("/settings/providers", { signal }),

  listSupportDocs: (
    projectId: string,
    opts: { limit?: number; cursor?: string } = {},
    signal?: AbortSignal,
  ) => {
    const params = new URLSearchParams();
    if (opts.limit !== undefined) params.set("limit", String(opts.limit));
    if (opts.cursor) params.set("cursor", opts.cursor);
    const qs = params.size > 0 ? `?${params.toString()}` : "";
    return apiFetch<SupportDocList>(
      `/projects/${enc(projectId)}/support-docs${qs}`,
      { signal },
    );
  },

  createSupportDoc: (
    projectId: string,
    file: File,
    idempotencyKey: string,
    signal?: AbortSignal,
  ) => {
    const form = new FormData();
    form.append("file", file);
    return apiFetch<SupportDocView>(`/projects/${enc(projectId)}/support-docs`, {
      method: "POST",
      body: form,
      signal,
      headers: { "Idempotency-Key": idempotencyKey },
    });
  },

  /* 204: the document is removed from semantic/docs and stops seeding future
     runs. Already-finished runs keep whatever they read. */
  deleteSupportDoc: (
    projectId: string,
    docId: string,
    idempotencyKey: string,
  ) =>
    apiFetch<void>(
      `/projects/${enc(projectId)}/support-docs/${enc(docId)}`,
      {
        method: "DELETE",
        headers: { "Idempotency-Key": idempotencyKey },
      },
    ),

  getSandboxStatus: (signal?: AbortSignal) =>
    apiFetch<SandboxStatusView>("/system/sandbox", { signal }),

  getCapabilities: (signal?: AbortSignal) =>
    apiFetch<SystemCapabilitiesView>("/system/capabilities", { signal }),

  prepareFindingPromotion: (sessionId: string, findingId: string) =>
    apiFetch<KnowledgePromotionPrepared>(
      `/sessions/${enc(sessionId)}/findings/${enc(findingId)}/prepare-promote`,
      { method: "POST" },
    ),

  /* No knowledge text: promote writes exactly what the approval froze. */
  promoteFinding: (
    sessionId: string,
    findingId: string,
    body: FindingPromoteRequest,
  ) =>
    apiFetch<KnowledgePromoted>(
      `/sessions/${enc(sessionId)}/findings/${enc(findingId)}/promote`,
      { method: "POST", body: JSON.stringify(body) },
    ),

  /* Regenerating replaces the session's report, so the caller must confirm first. */
  generateReport: (
    sessionId: string,
    body: ReportGenerateRequest | undefined,
    idempotencyKey: string,
  ) =>
    apiFetch<ReportGenerationStarted>(`/sessions/${enc(sessionId)}/report/generate`, {
      method: "POST",
      body: JSON.stringify(body),
      headers: { "Idempotency-Key": idempotencyKey },
    }),

  /* The forked run mints its own id; it arrives on the job's SSE stream as a
     `session.forked` event, not in this response. */
  forkSession: (sessionId: string, body: SessionForkRequest, idempotencyKey: string) =>
    apiFetch<SessionForkStarted>(`/sessions/${enc(sessionId)}/fork`, {
      method: "POST",
      body: JSON.stringify(body),
      headers: { "Idempotency-Key": idempotencyKey },
    }),

  /* Card editing is deterministic and inline: the server bumps card_version
     and re-runs feasibility, then returns the card at its new version. */
  editQuestionCard: (sessionId: string, questionId: string, body: QuestionCardEdit) =>
    apiFetch<QuestionSummary>(
      `/sessions/${enc(sessionId)}/questions/${enc(questionId)}`,
      { method: "PATCH", body: JSON.stringify(body) },
    ),

  prepareQuestionDraft: (
    sessionId: string,
    body: QuestionDraftPrepareRequest,
  ) =>
    apiFetch<QuestionDraftPrepared>(`/sessions/${enc(sessionId)}/questions/prepare-draft`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  /* No question text: drafting runs exactly what the approval froze. */
  draftQuestionCard: (
    sessionId: string,
    body: QuestionDraftRequest,
    idempotencyKey: string,
  ) =>
    apiFetch<QuestionDraftStarted>(`/sessions/${enc(sessionId)}/questions`, {
      method: "POST",
      body: JSON.stringify(body),
      headers: { "Idempotency-Key": idempotencyKey },
    }),
};
