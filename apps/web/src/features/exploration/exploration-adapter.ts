import type { ExplorationViewDto } from "../../api/client";
import {
  coverageGaps,
  type ExplorationBudgetAmendmentView,
  type ExplorationRunView,
} from "./exploration-model";

function decimal(value: string | number | null | undefined): number {
  if (value === null || value === undefined) return 0;
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : 0;
}

/** Map the API's snake_case journal projection into the presentation model.
 * No UI-only fallback is used for analytical content. */
export function explorationRunFromDto(dto: ExplorationViewDto): ExplorationRunView {
  const amendments: ExplorationBudgetAmendmentView[] = dto.budget.amendments.map(
    (amendment) => ({
      amendmentId: amendment.amendment_id,
      reason: amendment.reason,
      increase: {
        modelRequests: amendment.increase.max_requests ?? 0,
        successfulToolCalls:
          amendment.increase.max_successful_tool_calls ?? 0,
        rounds: amendment.increase.max_rounds ?? 0,
        costUsd: decimal(amendment.increase.max_cost_usd),
      },
    }),
  );
  const computedUnexplored = coverageGaps(
    dto.coverage_targets,
    dto.coverage_completed,
  );
  /* coverage_unexplored is consumed as a consistency check; the report itself
   * deliberately derives set difference from targets/completed. */
  const coverageProjectionConsistent = !(
    dto.coverage_unexplored.length !== computedUnexplored.length ||
    dto.coverage_unexplored.some((item) => !computedUnexplored.includes(item))
  );

  return {
    explorationId: dto.exploration_id,
    goal: dto.goal,
    tier: dto.thinking_level,
    status: dto.status,
    stopReason: dto.stop_reason ?? null,
    currentHypothesis: dto.current_hypothesis
      ? {
          hypothesisId: dto.current_hypothesis.hypothesis_id,
          statement: dto.current_hypothesis.statement,
          whySelected: dto.current_hypothesis.why_selected,
          status: dto.current_hypothesis.status,
        }
      : null,
    currentEvidence: dto.current_evidence.map((evidence) => ({
      receiptId: evidence.receipt_id,
      toolName: evidence.tool_name,
      summary: evidence.summary,
      factIds: evidence.fact_ids,
      facts: (evidence.facts ?? []).map((fact) => ({
        factId: fact.fact_id,
        name: fact.name,
        value: fact.value ?? null,
        unit: fact.unit ?? null,
      })),
      statistics: evidence.statistics
        ? {
            testName: evidence.statistics.test_name,
            outcome: evidence.statistics.outcome ?? null,
            testStatistic: evidence.statistics.test_statistic ?? null,
            pValue: evidence.statistics.p_value ?? null,
            adjustedPValue: evidence.statistics.adjusted_p_value ?? null,
            effectSize: evidence.statistics.effect_size ?? null,
            ciLow: evidence.statistics.ci_low ?? null,
            ciHigh: evidence.statistics.ci_high ?? null,
            sampleSize: evidence.statistics.sample_size ?? null,
          }
        : null,
    })),
    insights: dto.insights.map((insight) => ({
      insightId: insight.insight_id,
      hypothesisId: insight.hypothesis_id,
      statement: insight.statement,
      family: insight.family,
      status: insight.status,
      trustLevel: insight.trust_level,
      evidenceLane: insight.evidence_lane,
      proof: insight.proof.map((proof) => ({
        receiptId: proof.receipt_id,
        factIds: proof.fact_ids,
        comparison: proof.comparison,
      })),
      limitations: insight.limitations,
    })),
    limitations: dto.limitations,
    coverageTargets: dto.coverage_targets,
    coverageCompleted: dto.coverage_completed,
    coverageProjectionConsistent,
    budget: {
      base: {
        modelRequests:
          dto.budget.base.llm.max_requests ?? dto.budget.max_llm_requests ?? 0,
        successfulToolCalls: dto.budget.base.max_successful_tool_calls,
        rounds: dto.budget.base.max_rounds,
        costUsd: decimal(
          dto.budget.base.llm.max_cost_usd ?? dto.budget.max_cost_usd,
        ),
      },
      amendments,
      used: {
        modelRequests: dto.budget.llm_requests_used,
        successfulToolCalls: dto.budget.successful_tool_calls_used,
        rounds: dto.budget.rounds_used,
        costUsd: decimal(dto.budget.cost_usd),
      },
    },
    report: {
      available: dto.report.available,
      artifactRef: dto.report.artifact_ref ?? null,
    },
  };
}
