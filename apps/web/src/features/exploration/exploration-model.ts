export type ExplorationTier = "quick" | "standard" | "deep";
export type ExplorationStatus =
  | "running"
  | "pause_requested"
  | "paused"
  | "stopped";
export type ExplorationStopReason =
  | "completed"
  | "budget_exhausted"
  | "cancelled"
  | "failed"
  | "state_witness_changed"
  | "no_new_information";
export type EvidenceLane = "exploratory" | "confirmatory";
export type InsightStatus = "new" | "reinforced" | "refuted" | "inconclusive";
export type InsightTrustLevel =
  | "supported"
  | "contested"
  | "refuted"
  | "unsupported";
export type ProofComparison = "supports" | "contradicts";

export interface ExplorationProofView {
  receiptId: string;
  factIds: readonly string[];
  comparison: ProofComparison;
}

export interface ExplorationInsightView {
  insightId: string;
  hypothesisId: string;
  statement: string;
  family: string;
  status: InsightStatus;
  trustLevel: InsightTrustLevel;
  evidenceLane: EvidenceLane;
  proof: readonly ExplorationProofView[];
  limitations: readonly string[];
}

export interface ExplorationHypothesisView {
  hypothesisId: string;
  statement: string;
  whySelected: string;
  status: string;
}

export interface ExplorationEvidenceView {
  receiptId: string;
  summary: string;
  factIds: readonly string[];
}

export interface ExplorationBudgetCaps {
  modelRequests: number;
  successfulToolCalls: number;
  rounds: number;
  costUsd: number;
}

export interface ExplorationBudgetUsage {
  modelRequests: number;
  successfulToolCalls: number;
  rounds: number;
  costUsd: number;
}

export interface ExplorationBudgetAmendmentView {
  amendmentId: string;
  reason: string;
  increase: Partial<ExplorationBudgetCaps>;
}

export interface ExplorationBudgetView {
  base: ExplorationBudgetCaps;
  amendments: readonly ExplorationBudgetAmendmentView[];
  used: ExplorationBudgetUsage;
}

export interface ExplorationRunView {
  explorationId: string;
  goal: string;
  tier: ExplorationTier;
  status: ExplorationStatus;
  stopReason: ExplorationStopReason | null;
  currentHypothesis: ExplorationHypothesisView | null;
  currentEvidence: readonly ExplorationEvidenceView[];
  insights: readonly ExplorationInsightView[];
  limitations: readonly string[];
  coverageTargets: readonly string[];
  coverageCompleted: readonly string[];
  coverageProjectionConsistent?: boolean;
  budget: ExplorationBudgetView;
  report?: { available: boolean; artifactRef: string | null };
}

const BUDGET_DIMENSIONS = [
  "modelRequests",
  "successfulToolCalls",
  "rounds",
  "costUsd",
] as const satisfies readonly (keyof ExplorationBudgetCaps)[];

function compareText(left: string, right: string): number {
  if (left < right) return -1;
  if (left > right) return 1;
  return 0;
}

function uniqueSorted(values: readonly string[]): string[] {
  return [...new Set(values)].sort(compareText);
}

/** The report's not-explored section is data, never generated prose. */
export function coverageGaps(
  targets: readonly string[],
  completed: readonly string[],
): string[] {
  const completedSet = new Set(completed);
  return uniqueSorted(targets).filter((target) => !completedSet.has(target));
}

export function amendedBudgetIncrease(
  budget: ExplorationBudgetView,
): ExplorationBudgetCaps {
  const total: ExplorationBudgetCaps = {
    modelRequests: 0,
    successfulToolCalls: 0,
    rounds: 0,
    costUsd: 0,
  };
  for (const amendment of budget.amendments) {
    for (const dimension of BUDGET_DIMENSIONS) {
      total[dimension] += amendment.increase[dimension] ?? 0;
    }
  }
  return total;
}

export function effectiveBudgetCaps(
  budget: ExplorationBudgetView,
): ExplorationBudgetCaps {
  const increase = amendedBudgetIncrease(budget);
  return {
    modelRequests: budget.base.modelRequests + increase.modelRequests,
    successfulToolCalls:
      budget.base.successfulToolCalls + increase.successfulToolCalls,
    rounds: budget.base.rounds + increase.rounds,
    costUsd: budget.base.costUsd + increase.costUsd,
  };
}

export function remainingBudget(
  budget: ExplorationBudgetView,
): ExplorationBudgetCaps {
  const effective = effectiveBudgetCaps(budget);
  return {
    modelRequests: Math.max(0, effective.modelRequests - budget.used.modelRequests),
    successfulToolCalls: Math.max(
      0,
      effective.successfulToolCalls - budget.used.successfulToolCalls,
    ),
    rounds: Math.max(0, effective.rounds - budget.used.rounds),
    costUsd: Math.max(0, effective.costUsd - budget.used.costUsd),
  };
}

export const EXPLORATION_REPORT_SECTION_ORDER = [
  "supported_insights",
  "refuted_hypotheses",
  "inconclusive_questions",
  "limitations",
  "coverage_gaps",
  "cost_and_stop",
] as const;

export type ExplorationReportSectionId =
  (typeof EXPLORATION_REPORT_SECTION_ORDER)[number];

export const EXPLORATION_REPORT_SECTION_TITLES: Readonly<
  Record<ExplorationReportSectionId, string>
> = {
  supported_insights: "Supported insights",
  refuted_hypotheses: "Refuted hypotheses",
  inconclusive_questions: "Inconclusive questions",
  limitations: "Data and method limitations",
  coverage_gaps: "Coverage gaps / not explored",
  cost_and_stop: "Cost and structured stop reason",
};

export interface ExplorationReportGroups {
  supportedInsights: readonly ExplorationInsightView[];
  refutedHypotheses: readonly ExplorationInsightView[];
  inconclusiveQuestions: readonly ExplorationInsightView[];
  limitations: readonly string[];
  coverageGaps: readonly string[];
}

/** Deterministically group terminal records into the fixed six-section report. */
export function buildExplorationReportGroups(
  run: ExplorationRunView,
): ExplorationReportGroups {
  const insights = [...run.insights].sort((left, right) =>
    compareText(left.insightId, right.insightId),
  );
  return {
    supportedInsights: insights.filter(
      (insight) => insight.status === "new" || insight.status === "reinforced",
    ),
    refutedHypotheses: insights.filter((insight) => insight.status === "refuted"),
    inconclusiveQuestions: insights.filter(
      (insight) => insight.status === "inconclusive",
    ),
    limitations: uniqueSorted([
      ...run.limitations,
      ...insights.flatMap((insight) => insight.limitations),
    ]),
    coverageGaps: coverageGaps(run.coverageTargets, run.coverageCompleted),
  };
}

export function stopReasonLabel(reason: ExplorationStopReason | null): string {
  if (reason === null) return "Run has not stopped";
  const labels: Record<ExplorationStopReason, string> = {
    completed: "Completed",
    budget_exhausted: "Budget exhausted",
    cancelled: "Cancelled",
    failed: "Failed",
    state_witness_changed: "Data version changed",
    no_new_information: "No new information",
  };
  return labels[reason];
}

/** How much attention a terminal stop deserves.
 *
 * `problem` means the run did not finish its own way and the reader has
 * something to act on; `incomplete` means the results are valid but partial.
 * Rendering every stop identically hid a provider outage behind the same grey
 * line as a converged run (seed 8, 2026-08-03). */
export type StopSeverity = "problem" | "incomplete" | "healthy";

export function stopSeverity(reason: ExplorationStopReason | null): StopSeverity {
  if (reason === null) return "healthy";
  const severities: Record<ExplorationStopReason, StopSeverity> = {
    completed: "healthy",
    no_new_information: "healthy",
    // The operator asked for it; nothing to act on.
    cancelled: "healthy",
    budget_exhausted: "incomplete",
    failed: "problem",
    state_witness_changed: "problem",
  };
  return severities[reason];
}

/** Plain-language consequence, not a restatement of the code. */
export function stopReasonGuidance(
  reason: ExplorationStopReason | null,
): string | null {
  if (reason === null) return null;
  const guidance: Record<ExplorationStopReason, string | null> = {
    completed: null,
    no_new_information: null,
    cancelled: "You stopped this run. Its committed evidence is kept.",
    budget_exhausted:
      "A budget cap was reached, so the exploration is incomplete: what it did " +
      "prove still holds, but unexplored questions remain.",
    failed:
      "The run hit a fault it could not continue past. Evidence already " +
      "committed is kept and verifiable; re-running resumes from the journal " +
      "rather than starting over.",
    state_witness_changed:
      "The underlying data changed while the run was in flight, so these " +
      "results describe the earlier snapshot. Re-run against the current data " +
      "before acting on them.",
  };
  return guidance[reason];
}
