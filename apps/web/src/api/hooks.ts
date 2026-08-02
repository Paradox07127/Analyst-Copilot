/* TanStack Query hooks over the typed API client. Query keys are centralized
 * here so invalidation stays consistent across features. */

import {
  useInfiniteQuery,
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import {
  ApiError,
  api,
  saveBlob,
  type ColumnDistributionsView,
  type ProjectSummary,
  type ReportExportFormat,
  type SettingsPatch,
  type SettingsView,
  type SessionForkRequest,
  type CustomChartRequest,
  type DecisionStoryDraftRequest,
  type DecisionReportGenerateRequest,
  type VerifiedRelationDeleteRequest,
} from "./client";
import {
  dataOperationStorageKey,
  operationActivity,
  runDataOperation,
} from "./data-operations";
import { useJobActivity } from "../app/job-activity";

export const SESSIONS_PAGE_SIZE = 30;
export const PREVIEW_PAGE_SIZE = 100;
export const ARTIFACTS_PAGE_SIZE = 50;
export const RESOURCE_PAGE_SIZE = 50;

export const queryKeys = {
  projects: ["projects"] as const,
  /* Rooted so a delete can invalidate every window at once without knowing
   * which one the dashboard happens to be showing. */
  workspaceUsageRoot: ["workspace-usage"] as const,
  workspaceUsage: (days: number) => ["workspace-usage", days] as const,
  /* A search term extends the key rather than replacing it, so invalidating
   * ["sessions", projectId] still covers every filtered list of that project. */
  sessions: (projectId: string, q = "") =>
    (q ? ["sessions", projectId, { q }] : ["sessions", projectId]) as readonly unknown[],
  session: (sessionId: string) => ["session", sessionId] as const,
  datasets: (sessionId: string) => ["datasets", sessionId] as const,
  datasetSchema: (sessionId: string, datasetId: string) =>
    ["dataset-schema", sessionId, datasetId] as const,
  datasetPreview: (
    sessionId: string,
    datasetId: string,
    offset: number,
    limit: number,
  ) => ["dataset-preview", sessionId, datasetId, { offset, limit }] as const,
  report: (sessionId: string) => ["report", sessionId] as const,
  artifactsRoot: (sessionId: string) => ["artifacts", sessionId] as const,
  artifacts: (sessionId: string, type: string | undefined) =>
    ["artifacts", sessionId, { type: type ?? null }] as const,
  artifact: (sessionId: string, artifactId: string) =>
    ["artifact", sessionId, artifactId] as const,
  quality: (sessionId: string) => ["quality", sessionId] as const,
  edaHandoff: (sessionId: string) => ["eda-handoff", sessionId] as const,
  agentHandoff: (sessionId: string) => ["agent-handoff", sessionId] as const,
  profiles: (sessionId: string) => ["profiles", sessionId] as const,
  charts: (sessionId: string) => ["charts", sessionId] as const,
  chart: (sessionId: string, chartId: string) => ["chart", sessionId, chartId] as const,
  questions: (sessionId: string) => ["questions", sessionId] as const,
  investigations: (sessionId: string) => ["investigations", sessionId] as const,
  findings: (sessionId: string) => ["findings", sessionId] as const,
  semantic: (sessionId: string) => ["semantic", sessionId] as const,
  compare: (left: string, right: string) => ["compare", left, right] as const,
  compareScope: (
    scope: string,
    left: string,
    right: string,
    filter: "all" | "differences",
  ) => ["compare", "scope", scope, left, right, { filter }] as const,
  skills: (sessionId: string) => ["skills", sessionId] as const,
  relationships: (sessionId: string) => ["relationships", sessionId] as const,
  job: (jobId: string) => ["job", jobId] as const,
  chatMessages: (sessionId: string) => ["chat-messages", sessionId] as const,
  chatPendingPlans: (sessionId: string) => ["chat-pending-plans", sessionId] as const,
  board: (projectId: string, boardId: string) =>
    ["board", projectId, boardId] as const,
  analysis: (sessionId: string) => ["analysis", sessionId] as const,
  sessionMetrics: (sessionId: string) => ["session-metrics", sessionId] as const,
  trace: (sessionId: string, type: string | undefined) =>
    ["trace", sessionId, { type: type ?? null }] as const,
  traceRoot: (sessionId: string) => ["trace", sessionId] as const,
  decisionReport: (sessionId: string) => ["decision-report", sessionId] as const,
  cleaningLog: (sessionId: string) => ["cleaning-log", sessionId] as const,
  cleaningRaw: (sessionId: string) => ["cleaning-raw", sessionId] as const,
  sessionDebug: (sessionId: string) => ["session-debug", sessionId] as const,
  llmDebugCalls: (sessionId: string) => ["llm-debug-calls", sessionId] as const,
  datasetDistributions: (sessionId: string, datasetId: string) =>
    ["dataset-distributions", sessionId, datasetId] as const,
  decisionCoverage: (sessionId: string) => ["decision-coverage", sessionId] as const,
  decisionStory: (sessionId: string) => ["decision-story", sessionId] as const,
  settings: ["settings"] as const,
  providers: ["settings", "providers"] as const,
  supportDocs: (projectId: string) => ["support-docs", projectId] as const,
  projectUploads: (projectId: string) => ["project-uploads", projectId] as const,
  sandboxStatus: ["system", "sandbox"] as const,
  capabilities: ["system", "capabilities"] as const,
};

export const CHAT_PAGE_SIZE = 50;

export const CHARTS_PAGE_SIZE = 50;

export function useProjects() {
  return useQuery({
    queryKey: queryKeys.projects,
    queryFn: ({ signal }) => api.listProjects(signal),
  });
}

/* The one cross-run aggregate; everything else in the trace surface is scoped
 * to a single run. */
export function useWorkspaceUsage(days: number) {
  return useQuery({
    queryKey: queryKeys.workspaceUsage(days),
    queryFn: ({ signal }) => api.getWorkspaceUsage(days, signal),
  });
}

/* Project-scoped, not session-scoped: uploads live under the project, so a new
 * session can start from a table an earlier one already loaded. */
export function useProjectUploads(projectId: string, enabled = true) {
  return useQuery({
    queryKey: queryKeys.projectUploads(projectId),
    queryFn: ({ signal }) => api.listUploads(projectId, signal),
    enabled: enabled && Boolean(projectId),
  });
}

export function useDeleteUpload(projectId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (datasetId: string) => api.deleteUpload(projectId, datasetId),
    /* Uploads are project-scoped storage, so a removal changes the project's
     * quota headroom that the list is read against. */
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.projects });
      queryClient.invalidateQueries({
        queryKey: queryKeys.projectUploads(projectId),
      });
    },
  });
}

export function useDeleteProject() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (projectId: string) => api.deleteProject(projectId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.projects });
      /* The dashboard totals count this project's sessions; leaving them cached
       * shows a workspace that still contains what was just deleted. */
      queryClient.invalidateQueries({ queryKey: queryKeys.workspaceUsageRoot });
    },
  });
}

export function useRenameProject() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ projectId, name }: { projectId: string; name: string }) =>
      api.renameProject(projectId, name),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: queryKeys.projects }),
  });
}

export function useReorderProjects() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (projectIds: string[]) => api.reorderProjects(projectIds),
    onMutate: async (projectIds) => {
      await queryClient.cancelQueries({ queryKey: queryKeys.projects });
      const previous = queryClient.getQueryData<ProjectSummary[]>(queryKeys.projects);
      if (previous) {
        const byId = new Map(previous.map((project) => [project.project_id, project]));
        queryClient.setQueryData(
          queryKeys.projects,
          projectIds.flatMap((projectId) => {
            const project = byId.get(projectId);
            return project ? [project] : [];
          }),
        );
      }
      return { previous };
    },
    onError: (_error, _projectIds, context) => {
      if (context?.previous) {
        queryClient.setQueryData(queryKeys.projects, context.previous);
      }
    },
    onSettled: () => queryClient.invalidateQueries({ queryKey: queryKeys.projects }),
  });
}

export function useRenameSession(projectId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ sessionId, name }: { sessionId: string; name: string }) =>
      api.renameSession(sessionId, name),
    onSuccess: (session) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.sessions(projectId) });
      queryClient.invalidateQueries({ queryKey: queryKeys.session(session.session_id) });
    },
  });
}

/* enabled: the rail mounts one of these per project, so a workspace with a
 * dozen projects fired a dozen list requests on every navigation — including
 * for groups the user has collapsed and cannot see. */
export function useSessions(projectId: string, q = "", enabled = true) {
  return useInfiniteQuery({
    queryKey: queryKeys.sessions(projectId, q),
    queryFn: ({ pageParam, signal }) =>
      api.listSessions(
        projectId,
        { limit: SESSIONS_PAGE_SIZE, cursor: pageParam, q: q || undefined },
        signal,
      ),
    initialPageParam: undefined as string | undefined,
    getNextPageParam: (lastPage) => lastPage.next_cursor ?? undefined,
    enabled,
  });
}

/* enabled: the app shell renders on routes with no run, and an unguarded call
 * there sends GET /sessions/ on every navigation. */
export function useSessionDetail(sessionId: string) {
  return useQuery({
    queryKey: queryKeys.session(sessionId),
    queryFn: ({ signal }) => api.getSession(sessionId, signal),
    enabled: Boolean(sessionId),
  });
}

export function useDatasets(sessionId: string) {
  return useQuery({
    queryKey: queryKeys.datasets(sessionId),
    queryFn: ({ signal }) => api.listDatasets(sessionId, signal),
  });
}

export function useDatasetSchema(sessionId: string, datasetId: string) {
  return useQuery({
    queryKey: queryKeys.datasetSchema(sessionId, datasetId),
    queryFn: ({ signal }) => api.getDatasetSchema(sessionId, datasetId, signal),
  });
}

export function useReport(sessionId: string) {
  return useQuery({
    queryKey: queryKeys.report(sessionId),
    queryFn: ({ signal }) => api.getReport(sessionId, signal),
  });
}

export function useArtifacts(sessionId: string, type?: string) {
  return useInfiniteQuery({
    queryKey: queryKeys.artifacts(sessionId, type),
    queryFn: ({ pageParam, signal }) =>
      api.listArtifacts(
        sessionId,
        { type, limit: ARTIFACTS_PAGE_SIZE, cursor: pageParam },
        signal,
      ),
    initialPageParam: undefined as string | undefined,
    getNextPageParam: (lastPage) => lastPage.next_cursor ?? undefined,
  });
}

/* enabled gates the fetch to expanded rows: details are on-demand (§13.2). */
export function useArtifact(sessionId: string, artifactId: string, enabled: boolean) {
  return useQuery({
    queryKey: queryKeys.artifact(sessionId, artifactId),
    queryFn: ({ signal }) => api.getArtifact(sessionId, artifactId, signal),
    enabled,
  });
}

export function useQuality(sessionId: string) {
  return useQuery({
    queryKey: queryKeys.quality(sessionId),
    queryFn: ({ signal }) => api.getQuality(sessionId, signal),
  });
}

/* The EdaHandoff artifact is the pipeline's own downstream-readiness gate;
 * the Data Map surfaces it per table. Sessions from before the handoff
 * existed simply resolve to null. */
export function useEdaHandoff(sessionId: string) {
  return useQuery({
    queryKey: queryKeys.edaHandoff(sessionId),
    queryFn: async ({ signal }) => {
      const page = await api.listArtifacts(
        sessionId,
        { type: "EdaHandoff", limit: 1 },
        signal,
      );
      const first = page.items[0];
      if (!first) return null;
      return api.getArtifact(sessionId, first.artifact_id, signal);
    },
  });
}

/* Final, versioned contract for downstream analysis agents. Unlike the
 * early EdaHandoff gate, this exists only after the completed publish barrier. */
export function useAgentHandoff(sessionId: string, enabled = true) {
  return useQuery({
    queryKey: queryKeys.agentHandoff(sessionId),
    queryFn: ({ signal }) => api.getAgentHandoff(sessionId, signal),
    enabled: enabled && Boolean(sessionId),
  });
}

export function useProfiles(sessionId: string) {
  return useQuery({
    queryKey: queryKeys.profiles(sessionId),
    queryFn: ({ signal }) => api.getProfiles(sessionId, signal),
  });
}

export function useCharts(sessionId: string) {
  return useInfiniteQuery({
    queryKey: queryKeys.charts(sessionId),
    queryFn: ({ pageParam, signal }) =>
      api.listCharts(sessionId, { limit: CHARTS_PAGE_SIZE, cursor: pageParam }, signal),
    initialPageParam: undefined as string | undefined,
    getNextPageParam: (lastPage) => lastPage.next_cursor ?? undefined,
  });
}

/* enabled gates the spec fetch to in-viewport or zoomed cards (§13.3). */
export function useChart(sessionId: string, chartId: string, enabled: boolean) {
  return useQuery({
    queryKey: queryKeys.chart(sessionId, chartId),
    queryFn: ({ signal }) => api.getChart(sessionId, chartId, signal),
    enabled,
  });
}

export function useQuestions(sessionId: string, enabled = true) {
  return useQuery({
    queryKey: queryKeys.questions(sessionId),
    queryFn: ({ signal }) => api.listQuestions(sessionId, signal),
    enabled: enabled && Boolean(sessionId),
  });
}

export function useInvestigations(sessionId: string) {
  return useQuery({
    queryKey: queryKeys.investigations(sessionId),
    queryFn: ({ signal }) => api.getInvestigations(sessionId, signal),
  });
}

export function useFindings(sessionId: string) {
  return useQuery({
    queryKey: queryKeys.findings(sessionId),
    queryFn: ({ signal }) => api.listFindings(sessionId, signal),
  });
}

export function useSemantic(sessionId: string) {
  return useInfiniteQuery({
    queryKey: queryKeys.semantic(sessionId),
    queryFn: ({ pageParam, signal }) =>
      api.getSemantic(
        sessionId,
        { limit: RESOURCE_PAGE_SIZE, cursor: pageParam },
        signal,
      ),
    initialPageParam: undefined as string | undefined,
    getNextPageParam: (lastPage) => lastPage.next_cursor ?? undefined,
  });
}

/* enabled gates the fetch until the user has picked a run to compare against. */
export function useCompare(left: string, right: string) {
  return useQuery({
    queryKey: queryKeys.compare(left, right),
    queryFn: ({ signal }) => api.compareSessions(left, right, signal),
    enabled: Boolean(left) && Boolean(right),
  });
}

export function useCompareScope(
  scope: import("./client").CompareScopeName,
  left: string,
  right: string,
  filter: "all" | "differences",
) {
  return useInfiniteQuery({
    queryKey: queryKeys.compareScope(scope, left, right, filter),
    queryFn: ({ pageParam, signal }) =>
      api.compareScope(
        scope,
        left,
        right,
        {
          filter,
          limit: RESOURCE_PAGE_SIZE,
          cursor: pageParam,
        },
        signal,
      ),
    initialPageParam: undefined as string | undefined,
    getNextPageParam: (lastPage) => lastPage.next_cursor ?? undefined,
    enabled: Boolean(scope && left && right),
  });
}

export function useSkills(sessionId: string) {
  return useInfiniteQuery({
    queryKey: queryKeys.skills(sessionId),
    queryFn: ({ pageParam, signal }) =>
      api.listSkills(
        sessionId,
        { limit: RESOURCE_PAGE_SIZE, cursor: pageParam },
        signal,
      ),
    initialPageParam: undefined as string | undefined,
    getNextPageParam: (lastPage) => lastPage.next_cursor ?? undefined,
  });
}

export function useRelationships(sessionId: string) {
  return useQuery({
    queryKey: queryKeys.relationships(sessionId),
    queryFn: ({ signal }) => api.getRelationships(sessionId, signal),
  });
}

/* Kind never changes for a job, so one fetch per job id is enough. */
export function useJob(jobId: string) {
  return useQuery({
    queryKey: queryKeys.job(jobId),
    queryFn: ({ signal }) => api.getJob(jobId, signal),
    enabled: Boolean(jobId),
    staleTime: Infinity,
  });
}

/* Transcript pages walk backwards from the newest message, so getNextPageParam
 * returns the cursor for the next OLDER page and pages render in reverse. */
export function useChatMessages(sessionId: string) {
  return useInfiniteQuery({
    queryKey: queryKeys.chatMessages(sessionId),
    queryFn: ({ pageParam, signal }) =>
      api.listChatMessages(
        sessionId,
        { limit: CHAT_PAGE_SIZE, cursor: pageParam },
        signal,
      ),
    initialPageParam: undefined as string | undefined,
    getNextPageParam: (lastPage) => lastPage.next_cursor ?? undefined,
  });
}

/* Only asked for when the transcript ends on an unapproved plan and no stream
 * is live. The GET is read-only and returns the durable token unchanged, so
 * normal query caching/refetch cannot invalidate an already-rendered card. */
export function useChatPendingPlans(sessionId: string, enabled: boolean) {
  return useQuery({
    queryKey: queryKeys.chatPendingPlans(sessionId),
    queryFn: ({ signal }) => api.listChatPendingPlans(sessionId, signal),
    enabled: Boolean(sessionId) && enabled,
    staleTime: 30_000,
  });
}

export function useBoard(projectId: string, boardId: string) {
  return useQuery({
    queryKey: queryKeys.board(projectId, boardId),
    queryFn: ({ signal }) => api.getBoard(projectId, boardId, signal),
    enabled: Boolean(projectId) && Boolean(boardId),
  });
}

export function useDatasetPreview(
  sessionId: string,
  datasetId: string,
  offset: number,
  limit: number = PREVIEW_PAGE_SIZE,
) {
  return useQuery({
    queryKey: queryKeys.datasetPreview(sessionId, datasetId, offset, limit),
    queryFn: ({ signal }) =>
      api.getDatasetPreview(sessionId, datasetId, { limit, offset }, signal),
    placeholderData: (previous) => previous,
  });
}

export const TRACE_PAGE_SIZE = 100;

export function useAnalysis(sessionId: string) {
  return useQuery({
    queryKey: queryKeys.analysis(sessionId),
    queryFn: ({ signal }) => api.getAnalysis(sessionId, signal),
  });
}

export function useSessionMetrics(sessionId: string) {
  return useQuery({
    queryKey: queryKeys.sessionMetrics(sessionId),
    queryFn: ({ signal }) => api.getSessionMetrics(sessionId, signal),
    enabled: Boolean(sessionId),
  });
}

/* Type filter is part of the key: a cursor is only valid for the filter it was
 * minted under, so switching filters must start a new page chain. */
export function useTraceEvents(sessionId: string, type: string | undefined) {
  return useInfiniteQuery({
    queryKey: queryKeys.trace(sessionId, type),
    queryFn: ({ pageParam, signal }) =>
      api.listTraceEvents(
        sessionId,
        { limit: TRACE_PAGE_SIZE, cursor: pageParam, type },
        signal,
      ),
    initialPageParam: undefined as string | undefined,
    getNextPageParam: (lastPage) => lastPage.next_cursor ?? undefined,
  });
}

export function useDecisionReport(sessionId: string) {
  return useQuery({
    queryKey: queryKeys.decisionReport(sessionId),
    queryFn: ({ signal }) => api.getDecisionReport(sessionId, signal),
  });
}

export function useCleaningLog(sessionId: string) {
  return useQuery({
    queryKey: queryKeys.cleaningLog(sessionId),
    queryFn: ({ signal }) => api.getCleaningLog(sessionId, signal),
  });
}

export function useCleaningRaw(sessionId: string) {
  return useQuery({
    queryKey: queryKeys.cleaningRaw(sessionId),
    queryFn: ({ signal }) => api.getCleaningRaw(sessionId, signal),
  });
}

export function useSessionDebug(sessionId: string) {
  return useInfiniteQuery({
    queryKey: queryKeys.sessionDebug(sessionId),
    queryFn: ({ pageParam, signal }) =>
      api.getSessionDebug(
        sessionId,
        { limit: 100, cursor: pageParam },
        signal,
      ),
    initialPageParam: undefined as string | undefined,
    getNextPageParam: (lastPage) => lastPage.next_cursor ?? undefined,
  });
}

export const LLM_DEBUG_PAGE_SIZE = 25;

export function useLlmDebugCalls(sessionId: string, enabled = true) {
  return useInfiniteQuery({
    queryKey: queryKeys.llmDebugCalls(sessionId),
    queryFn: ({ pageParam, signal }) =>
      api.listLlmDebugCalls(
        sessionId,
        { limit: LLM_DEBUG_PAGE_SIZE, cursor: pageParam },
        signal,
      ),
    initialPageParam: undefined as string | undefined,
    getNextPageParam: (lastPage) => lastPage.next_cursor ?? undefined,
    enabled,
  });
}

const DISTRIBUTIONS_KIND = "dataset-distributions";

/* One idempotency key per (run, dataset) rather than per visit: the dataset is
 * immutable inside its run, so the server replays the finished scan instead of
 * re-reading the whole table every time the memory cache is gone. A key that
 * ended up on a failed or cancelled job would otherwise pin the dataset to it
 * forever, so that one case retries under a throwaway key — but only once the
 * job actually existed, since a rejected start (a busy lane, also a 409) would
 * just lose the same race again.
 *
 * No activity tracking: this is a page-level read, not a run the user started.
 */
export async function fetchDatasetDistributions(
  sessionId: string,
  datasetId: string,
  signal?: AbortSignal,
): Promise<ColumnDistributionsView> {
  const storageKey = dataOperationStorageKey(
    DISTRIBUTIONS_KIND,
    sessionId,
    datasetId,
  );
  let queued = false;
  const scan = (idempotencyKey: string) =>
    runDataOperation(
      storageKey,
      () => api.startDatasetDistributions(sessionId, datasetId, idempotencyKey),
      (jobId, resultSignal) =>
        api.getDatasetDistributionsResult(jobId, resultSignal),
      () => {
        queued = true;
      },
      signal,
    );
  try {
    return await scan(`${DISTRIBUTIONS_KIND}:${sessionId}:${datasetId}`);
  } catch (error) {
    if (
      !queued ||
      signal?.aborted ||
      !(error instanceof ApiError) ||
      error.status !== 409
    ) {
      throw error;
    }
    return await scan(crypto.randomUUID());
  }
}

export function useDatasetDistributions(
  sessionId: string,
  datasetId: string,
  enabled = true,
) {
  return useQuery({
    queryKey: queryKeys.datasetDistributions(sessionId, datasetId),
    queryFn: ({ signal }) =>
      fetchDatasetDistributions(sessionId, datasetId, signal),
    enabled: enabled && Boolean(datasetId),
    staleTime: Infinity,
    /* The stable-key retry above is the only retry worth having here; a second
     * one on top of it would queue a third scan of the same table. */
    retry: false,
  });
}

export function useDecisionStory(sessionId: string) {
  return useQuery({
    queryKey: queryKeys.decisionStory(sessionId),
    queryFn: ({ signal }) => api.getDecisionStory(sessionId, signal),
  });
}

/* Both writes settle asynchronously through a job, so the caller drives the
 * refresh once the job reports done rather than on mutation success. */
export function useCreateDecisionStoryDraft(sessionId: string) {
  return useMutation({
    mutationFn: (body: DecisionStoryDraftRequest) =>
      api.createDecisionStoryDraft(sessionId, body),
  });
}

export function useGenerateDecisionReport(sessionId: string) {
  return useMutation({
    mutationFn: (body: DecisionReportGenerateRequest) =>
      api.generateDecisionReport(sessionId, body),
  });
}

export function useDeleteVerifiedRelation(sessionId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (body: VerifiedRelationDeleteRequest) =>
      api.deleteVerifiedRelation(sessionId, body),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: queryKeys.semantic(sessionId) }),
  });
}

export function useDecisionCoverage(sessionId: string) {
  return useQuery({
    queryKey: queryKeys.decisionCoverage(sessionId),
    queryFn: ({ signal }) => api.getDecisionCoverage(sessionId, signal),
  });
}

/* A chart is built from the user's current picks, not cached per run: the same
 * run yields a different spec every time the controls change. */
export function useBuildCustomChart(sessionId: string, projectId = "") {
  const { startTracking } = useJobActivity();
  return useMutation({
    mutationFn: (body: CustomChartRequest) =>
      runDataOperation(
        dataOperationStorageKey(
          "custom-chart",
          sessionId,
          encodeURIComponent(JSON.stringify(body)),
        ),
        () => api.buildCustomChart(sessionId, body, crypto.randomUUID()),
        (jobId, signal) => api.getCustomChartResult(jobId, signal),
        (started) => startTracking(operationActivity(started, projectId)),
      ),
  });
}

export function useDownloadDebugLog(sessionId: string) {
  return useMutation({
    mutationFn: async () => {
      const { blob, filename } = await api.downloadDebugLog(sessionId);
      saveBlob(blob, filename);
    },
  });
}

export function useSettings() {
  return useQuery({
    queryKey: queryKeys.settings,
    queryFn: ({ signal }) => api.getSettings(signal),
  });
}

/* Static per build — the registry never changes at runtime. */
export function useProviders() {
  return useQuery({
    queryKey: queryKeys.providers,
    queryFn: ({ signal }) => api.listProviders(signal),
    staleTime: Infinity,
  });
}

/* Keyed on the saved connection, not the draft: discovery always uses what the
 * server has stored, so an unsaved key edit must not look like a new list. */
export function useModels(provider: string, version: number | undefined) {
  return useQuery({
    queryKey: ["settings-models", provider, version] as const,
    queryFn: ({ signal }) => api.listModels(signal),
  });
}

export function useRefreshModels(provider: string, version: number | undefined) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => api.refreshModels(),
    onSuccess: (catalog) =>
      queryClient.setQueryData(["settings-models", provider, version], catalog),
  });
}

export function useUpdateSettings() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (patch: SettingsPatch) =>
      api.updateSettings(
        patch,
        queryClient.getQueryData<SettingsView>(queryKeys.settings)?.version,
      ),
    onSuccess: (view: SettingsView) =>
      queryClient.setQueryData(queryKeys.settings, view),
  });
}

export function useResetSettings() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () =>
      api.resetSettings(
        queryClient.getQueryData<SettingsView>(queryKeys.settings)?.version,
      ),
    onSuccess: (view: SettingsView) =>
      queryClient.setQueryData(queryKeys.settings, view),
  });
}

export function useTestConnection() {
  return useMutation({ mutationFn: () => api.testConnection() });
}

/* Downloads are one-shot side effects, not cached state: each click refetches
 * so the file always matches the session's current report. */
export function useDownloadReport(sessionId: string) {
  return useMutation({
    mutationFn: async (format: ReportExportFormat) => {
      const { blob, filename } = await api.downloadReport(sessionId, format);
      saveBlob(blob, filename);
      return format;
    },
  });
}

export function useSandboxStatus() {
  return useQuery({
    queryKey: queryKeys.sandboxStatus,
    queryFn: ({ signal }) => api.getSandboxStatus(signal),
    /* The server caches its probe for 30s; matching that keeps navigation
     * between pages from re-asking. */
    staleTime: 30_000,
  });
}

/* Optional-dependency probe: fixed per server process, so one fetch is enough. */
export function useCapabilities() {
  return useQuery({
    queryKey: queryKeys.capabilities,
    queryFn: ({ signal }) => api.getCapabilities(signal),
    staleTime: Infinity,
  });
}

export function useSupportDocs(projectId: string) {
  return useInfiniteQuery({
    queryKey: queryKeys.supportDocs(projectId),
    queryFn: ({ pageParam, signal }) =>
      api.listSupportDocs(
        projectId,
        { limit: RESOURCE_PAGE_SIZE, cursor: pageParam },
        signal,
      ),
    initialPageParam: undefined as string | undefined,
    getNextPageParam: (lastPage) => lastPage.next_cursor ?? undefined,
    enabled: Boolean(projectId),
  });
}

export function useUploadSupportDoc(projectId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (file: File) =>
      api.createSupportDoc(projectId, file, crypto.randomUUID()),
    onSuccess: () =>
      queryClient.invalidateQueries({
        queryKey: queryKeys.supportDocs(projectId),
      }),
  });
}

export function useDeleteSupportDoc(projectId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (docId: string) =>
      api.deleteSupportDoc(projectId, docId, crypto.randomUUID()),
    onSuccess: () =>
      queryClient.invalidateQueries({
        queryKey: queryKeys.supportDocs(projectId),
      }),
  });
}

/* Two-step like question execution: prepare returns the exact knowledge text
 * plus a one-time approval, promote consumes it. Promotion rewrites the
 * project's semantic seeds, so the Knowledge view is invalidated too. */
export function usePrepareFindingPromotion(sessionId: string) {
  return useMutation({
    mutationFn: (findingId: string) =>
      api.prepareFindingPromotion(sessionId, findingId),
  });
}

export function usePromoteFinding(sessionId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (vars: {
      findingId: string;
      action_hash: string;
      approval_token: string;
    }) =>
      api.promoteFinding(sessionId, vars.findingId, {
        action_hash: vars.action_hash,
        approval_token: vars.approval_token,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.semantic(sessionId) });
      queryClient.invalidateQueries({ queryKey: queryKeys.findings(sessionId) });
    },
  });
}

/* Report generation and forking both return a job; the caller hands it to the
 * activity drawer, which invalidates the session's queries once it settles. */
export function useGenerateReport(sessionId: string) {
  return useMutation({
    mutationFn: (vars: { llm: "env" | "offline"; idempotencyKey: string }) =>
      api.generateReport(sessionId, { llm: vars.llm }, vars.idempotencyKey),
  });
}

export function useForkSession(sessionId: string) {
  return useMutation({
    mutationFn: (vars: { body: SessionForkRequest; idempotencyKey: string }) =>
      api.forkSession(sessionId, vars.body, vars.idempotencyKey),
  });
}

/* Destructive: the caller owns the confirmation step. Invalidates the project's
 * session lists (every search variant) and the project index, whose session_count moved. */
export function useDeleteSession(projectId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (sessionId: string) => api.deleteSession(sessionId),
    onSuccess: (_result, sessionId) => {
      queryClient.removeQueries({ queryKey: queryKeys.session(sessionId) });
      queryClient.invalidateQueries({ queryKey: ["sessions", projectId] });
      queryClient.invalidateQueries({ queryKey: queryKeys.projects });
      queryClient.invalidateQueries({ queryKey: queryKeys.workspaceUsageRoot });
    },
  });
}
