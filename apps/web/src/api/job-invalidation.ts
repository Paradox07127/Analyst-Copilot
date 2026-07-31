import type { QueryClient, QueryKey } from "@tanstack/react-query";
import { queryKeys } from "./hooks";

export const JOB_KINDS = [
  "auto_eda",
  "question_exec",
  "skill_replay",
  "relationship_validate",
  "relationship_discover",
  "report_generate",
  "session_fork",
  "question_draft",
  "investigation_plan",
  "investigation_execute",
  "macro_loop",
  "synthesis_brief_create",
  "decision_report_generate",
  "cleaning_preview",
  "cleaning_apply",
  "dataset_distributions",
  "custom_chart",
] as const;

export type JobKind = (typeof JOB_KINDS)[number];

export interface JobInvalidationContext {
  projectId: string;
  sessionId: string;
  sourceSessionId: string;
  /** Session that owns result artifacts when it differs from both source and
   * lifecycle runs (for example an investigation plan run). */
  resultSessionId?: string;
}

type KeyFactory = (context: JobInvalidationContext) => readonly QueryKey[];

const baseKeys: KeyFactory = ({ projectId, sessionId }) => [
  queryKeys.session(sessionId),
  queryKeys.sessions(projectId),
  queryKeys.projects,
  queryKeys.workspaceUsageRoot,
  queryKeys.sessionMetrics(sessionId),
  queryKeys.traceRoot(sessionId),
];

const JOB_RESULT_KEYS: Record<JobKind, KeyFactory> = {
  auto_eda: ({ sessionId }) => [
    queryKeys.datasets(sessionId),
    queryKeys.quality(sessionId),
    queryKeys.profiles(sessionId),
    queryKeys.charts(sessionId),
    queryKeys.report(sessionId),
    queryKeys.artifactsRoot(sessionId),
    queryKeys.questions(sessionId),
    queryKeys.findings(sessionId),
    queryKeys.semantic(sessionId),
    queryKeys.analysis(sessionId),
    queryKeys.skills(sessionId),
    queryKeys.cleaningLog(sessionId),
    queryKeys.cleaningRaw(sessionId),
    queryKeys.sessionDebug(sessionId),
    queryKeys.decisionCoverage(sessionId),
  ],
  question_exec: ({ sessionId, sourceSessionId }) => [
    queryKeys.questions(sourceSessionId),
    queryKeys.artifactsRoot(sessionId),
    queryKeys.findings(sessionId),
  ],
  skill_replay: ({ sessionId, sourceSessionId }) => [
    queryKeys.skills(sourceSessionId),
    queryKeys.artifactsRoot(sessionId),
    queryKeys.findings(sessionId),
  ],
  relationship_validate: ({ sourceSessionId }) => [
    queryKeys.relationships(sourceSessionId),
    queryKeys.artifactsRoot(sourceSessionId),
  ],
  relationship_discover: ({ sourceSessionId }) => [
    queryKeys.relationships(sourceSessionId),
    queryKeys.artifactsRoot(sourceSessionId),
  ],
  report_generate: ({ sourceSessionId }) => [
    queryKeys.report(sourceSessionId),
    queryKeys.artifactsRoot(sourceSessionId),
    queryKeys.sessionMetrics(sourceSessionId),
  ],
  /* The new session id arrives in the terminal event, so the project session list is
   * the only result cache addressable from the tracked lifecycle job. */
  session_fork: () => [],
  question_draft: ({ sourceSessionId }) => [
    queryKeys.questions(sourceSessionId),
    queryKeys.artifactsRoot(sourceSessionId),
  ],
  investigation_plan: ({ sourceSessionId }) => [
    queryKeys.investigations(sourceSessionId),
  ],
  investigation_execute: ({ sourceSessionId, resultSessionId }) => [
    queryKeys.investigations(sourceSessionId),
    queryKeys.artifactsRoot(resultSessionId ?? sourceSessionId),
    queryKeys.findings(resultSessionId ?? sourceSessionId),
  ],
  macro_loop: ({ sourceSessionId, resultSessionId }) => [
    queryKeys.investigations(sourceSessionId),
    queryKeys.artifactsRoot(resultSessionId ?? sourceSessionId),
    queryKeys.findings(resultSessionId ?? sourceSessionId),
  ],
  synthesis_brief_create: ({ sourceSessionId }) => [
    queryKeys.decisionStory(sourceSessionId),
  ],
  decision_report_generate: ({ sourceSessionId, resultSessionId }) => [
    queryKeys.decisionReport(sourceSessionId),
    ...(resultSessionId ? [queryKeys.artifactsRoot(resultSessionId)] : []),
  ],
  cleaning_preview: () => [],
  cleaning_apply: ({ sourceSessionId }) => [
    queryKeys.cleaningLog(sourceSessionId),
    queryKeys.cleaningRaw(sourceSessionId),
  ],
  dataset_distributions: () => [],
  custom_chart: () => [],
};

export function jobInvalidationKeys(
  kind: JobKind,
  context: JobInvalidationContext,
): readonly QueryKey[] {
  const unique = new Map<string, QueryKey>();
  for (const key of [...baseKeys(context), ...JOB_RESULT_KEYS[kind](context)]) {
    unique.set(JSON.stringify(key), key);
  }
  return [...unique.values()];
}

export async function invalidateJobResultQueries(
  queryClient: QueryClient,
  kind: JobKind,
  context: JobInvalidationContext,
): Promise<void> {
  await Promise.all(
    jobInvalidationKeys(kind, context).map((queryKey) =>
      queryClient.invalidateQueries({ queryKey }),
    ),
  );
}

export function isJobKind(value: string | undefined): value is JobKind {
  return JOB_KINDS.some((kind) => kind === value);
}
