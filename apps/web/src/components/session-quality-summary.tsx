import type { SessionMetricsView } from "../api/client";
import { Disclosure } from "./ui";

interface QualitySignal {
  label: string;
  value: string;
}

function nonZeroNumber(
  signals: QualitySignal[],
  label: string,
  value: number | null | undefined,
  format: (value: number) => string = String,
) {
  if (typeof value === "number" && value !== 0) {
    signals.push({ label, value: format(value) });
  }
}

function activeFlag(
  signals: QualitySignal[],
  label: string,
  value: boolean | null | undefined,
) {
  if (value) signals.push({ label, value: "yes" });
}

export function qualitySignals(metrics: SessionMetricsView): QualitySignal[] {
  const signals: QualitySignal[] = [];
  nonZeroNumber(signals, "Failures recorded", metrics.failures_count);
  activeFlag(signals, "Session degraded", metrics.degraded);
  activeFlag(signals, "Question LLM skipped", metrics.question_llm_skipped);
  nonZeroNumber(
    signals,
    "Question proposals dropped",
    metrics.question_proposals_dropped,
  );
  nonZeroNumber(
    signals,
    "Question dataset names resolved",
    metrics.question_dataset_names_resolved,
  );
  nonZeroNumber(
    signals,
    "Question list coercions",
    metrics.question_list_coercions,
  );
  activeFlag(
    signals,
    "Semantic bootstrap degraded",
    metrics.semantic_bootstrap_degraded,
  );
  nonZeroNumber(
    signals,
    "Column roles unverified",
    metrics.column_roles_unverified,
  );
  nonZeroNumber(
    signals,
    "Template backstop used",
    metrics.template_backstop_used,
  );
  nonZeroNumber(
    signals,
    "Join authorizations stale",
    metrics.join_authorizations_stale,
  );
  nonZeroNumber(
    signals,
    "Join authorizations unverifiable",
    metrics.join_authorizations_unverifiable,
  );
  activeFlag(
    signals,
    "Relationship coverage limited",
    metrics.relationship_coverage_limited,
  );
  activeFlag(
    signals,
    "Relationship discovery deferred",
    metrics.relationship_discovery_deferred,
  );
  nonZeroNumber(
    signals,
    "Semantic claims degraded",
    metrics.semantic_degraded_claims,
  );
  nonZeroNumber(
    signals,
    "Time boundaries truncated",
    metrics.time_boundary_truncations,
  );
  nonZeroNumber(
    signals,
    "Numeric figures unverified",
    metrics.numeric_unverified_claims,
  );
  nonZeroNumber(
    signals,
    "Quantitative coverage gaps",
    metrics.quantitative_coverage_gaps,
  );
  nonZeroNumber(
    signals,
    "Evidence requests rejected",
    metrics.evidence_interleave_rejected,
  );
  nonZeroNumber(
    signals,
    "Evidence requests granted",
    metrics.evidence_interleave_granted,
  );
  nonZeroNumber(
    signals,
    "Domain metric questions",
    metrics.domain_metric_questions,
  );
  nonZeroNumber(
    signals,
    "Domain metrics skipped",
    metrics.domain_metrics_skipped,
  );
  nonZeroNumber(
    signals,
    "Macro-loop rounds discarded",
    metrics.macro_loop_discard_rounds,
  );
  nonZeroNumber(signals, "Questions answered", metrics.question_answered);
  nonZeroNumber(signals, "Questions abstained", metrics.question_abstained);
  nonZeroNumber(signals, "Questions failed", metrics.question_failed);
  nonZeroNumber(
    signals,
    "Questions awaiting approval",
    metrics.question_awaiting_approval,
  );
  const contractFailures = Object.values(
    metrics.result_contract_failures ?? {},
  ).reduce((sum, value) => sum + value, 0);
  nonZeroNumber(signals, "Result contract failures", contractFailures);
  nonZeroNumber(
    signals,
    "Interpretations validated",
    metrics.interpretation_validated,
  );
  nonZeroNumber(
    signals,
    "Interpretation fallbacks",
    metrics.interpretation_fallbacks,
  );
  nonZeroNumber(
    signals,
    "Report-eligible findings",
    metrics.report_eligible_findings,
  );
  nonZeroNumber(
    signals,
    "Question answer rate",
    metrics.question_answer_rate,
    (value) => `${(value * 100).toFixed(1)}%`,
  );
  nonZeroNumber(
    signals,
    "Question abstention rate",
    metrics.question_abstention_rate,
    (value) => `${(value * 100).toFixed(1)}%`,
  );
  nonZeroNumber(
    signals,
    "Budget calls rejected",
    metrics.budget_rejected_calls,
  );
  nonZeroNumber(
    signals,
    "Budget calls uncertain",
    metrics.budget_uncertain_calls,
  );
  activeFlag(signals, "Coverage limited", metrics.coverage_limited);
  activeFlag(signals, "Publication blocked", metrics.publication_blocked);
  if (metrics.trace_status && metrics.trace_status !== "verified") {
    signals.push({ label: "Trace status", value: metrics.trace_status });
  }
  if (
    metrics.budget_reconciliation &&
    !["verified", "not_applicable"].includes(metrics.budget_reconciliation)
  ) {
    signals.push({
      label: "Budget reconciliation",
      value: metrics.budget_reconciliation,
    });
  }
  if (
    metrics.report_gate_verdict &&
    metrics.report_gate_verdict !== "pass"
  ) {
    signals.push({
      label: "Report gate",
      value: metrics.report_gate_verdict,
    });
  }
  if (
    metrics.publication_freshness &&
    !["fresh", "not_applicable", "unknown"].includes(
      metrics.publication_freshness,
    )
  ) {
    signals.push({
      label: "Publication freshness",
      value: metrics.publication_freshness,
    });
  }
  return signals;
}

export function SessionQualitySummary({
  metrics,
  compact = false,
}: {
  metrics: SessionMetricsView;
  compact?: boolean;
}) {
  const signals = qualitySignals(metrics);
  if (signals.length === 0) return null;

  return (
    <section
      aria-label="Session quality signals"
      className={
        compact
          ? "rounded-base border border-status-warn/30 px-2"
          : "rounded-base border border-status-warn/30 bg-status-warn/5 px-3 py-1"
      }
    >
      <Disclosure summary={`${signals.length} non-zero quality signals`}>
        <dl className="grid gap-x-4 gap-y-1 pb-2 text-xs sm:grid-cols-2">
          {signals.map((signal) => (
            <div
              key={signal.label}
              className="flex items-baseline justify-between gap-3"
            >
              <dt className="text-status-neutral">{signal.label}</dt>
              <dd className="font-mono font-medium">{signal.value}</dd>
            </div>
          ))}
        </dl>
        {Object.keys(metrics.result_contract_failures ?? {}).length > 0 && (
          <pre className="mb-2 overflow-x-auto rounded-base bg-code-bg p-2 text-xs">
            {JSON.stringify(metrics.result_contract_failures, null, 2)}
          </pre>
        )}
      </Disclosure>
    </section>
  );
}
