import { Badge, Card, Disclosure, MetricStrip, MetricTile, SectionHeader, type Tone } from "../../components/ui";
import {
  EXPLORATION_REPORT_SECTION_ORDER,
  EXPLORATION_REPORT_SECTION_TITLES,
  amendedBudgetIncrease,
  buildExplorationReportGroups,
  effectiveBudgetCaps,
  remainingBudget,
  stopReasonGuidance,
  stopReasonLabel,
  stopSeverity,
  type EvidenceLane,
  type ExplorationInsightView,
  type ExplorationReportSectionId,
  type ExplorationRunView,
  type ExplorationStatus,
  type ExplorationStopReason,
} from "./exploration-model";

const STATUS_PRESENTATION: Record<
  ExplorationStatus,
  { label: string; tone: Tone; detail: string }
> = {
  running: {
    label: "Running",
    tone: "info",
    detail: "The read-only exploration is actively evaluating hypotheses.",
  },
  pause_requested: {
    label: "Pause requested",
    tone: "warn",
    detail: "The current operation is draining before the run becomes resumable.",
  },
  paused: {
    label: "Paused · resumable",
    tone: "warn",
    detail: "The journal remains resumable. Paused is not a terminal stop.",
  },
  stopped: {
    label: "Stopped · terminal",
    tone: "neutral",
    detail: "The run is terminal and cannot be resumed.",
  },
};

const LANE_PRESENTATION: Record<
  EvidenceLane,
  { label: string; tone: Tone; meaning: string }
> = {
  exploratory: {
    label: "Exploratory",
    tone: "warn",
    meaning: "Hypothesis-generating evidence that needs independent follow-up.",
  },
  confirmatory: {
    label: "Confirmatory evidence",
    tone: "info",
    meaning:
      "Evidence from a designated confirmation lane. Read its proof and limitations; the label is not a claim of certainty.",
  },
};

function formatUsd(value: number): string {
  return `$${value.toFixed(2)}`;
}

export function EvidenceLaneBadge({
  lane,
  label,
}: {
  lane: EvidenceLane;
  label?: string;
}) {
  const presentation = LANE_PRESENTATION[lane];
  return (
    <Badge tone={presentation.tone} variant="outline" title={presentation.meaning}>
      {label ?? presentation.label}
    </Badge>
  );
}

/** A stop the run did not choose gets an alert, not a grey line. */
function StopNotice({ reason }: { reason: ExplorationStopReason | null }) {
  const severity = stopSeverity(reason);
  const guidance = stopReasonGuidance(reason);
  const label = stopReasonLabel(reason);
  const heading = (
    <p className="text-sm">
      Stop reason: <strong>{label}</strong>
      {reason && <code className="ml-2">{reason}</code>}
    </p>
  );
  if (severity === "healthy") return heading;
  const isProblem = severity === "problem";
  return (
    <div
      role={isProblem ? "alert" : "status"}
      aria-label={`Stop reason: ${label}`}
      className={`flex flex-col gap-1 rounded-base border p-3 ${
        isProblem ? "border-status-critical/40" : "border-status-warning/40"
      }`}
    >
      <p
        className={`text-sm font-medium ${
          isProblem ? "text-status-critical" : "text-status-warning"
        }`}
      >
        {label}
        {reason && <code className="ml-2 font-normal">{reason}</code>}
      </p>
      {guidance && <p className="text-sm text-status-neutral">{guidance}</p>}
    </div>
  );
}

function RunStatus({ status }: { status: ExplorationStatus }) {
  const presentation = STATUS_PRESENTATION[status];
  return (
    <div className="flex flex-col items-start gap-1">
      <Badge tone={presentation.tone} caps>
        {presentation.label}
      </Badge>
      <span className="text-xs text-status-neutral">{presentation.detail}</span>
    </div>
  );
}

function CostChip({ run }: { run: ExplorationRunView }) {
  const effective = effectiveBudgetCaps(run.budget);
  const exhausted = run.budget.used.costUsd >= effective.costUsd;
  return (
    <Badge
      tone={exhausted ? "critical" : "brand"}
      title="Observed cost against the effective hard cap; this is not a billing forecast."
    >
      {`Cost ${formatUsd(run.budget.used.costUsd)} / ${formatUsd(effective.costUsd)} cap`}
    </Badge>
  );
}

function BudgetSummary({ run }: { run: ExplorationRunView }) {
  const increase = amendedBudgetIncrease(run.budget);
  const remaining = remainingBudget(run.budget);
  const hint = (base: number, amended: number, used: number) =>
    `base ${base} + ${amended} amended · ${used} used`;
  return (
    <section aria-label="Exploration budget" className="flex flex-col gap-2">
      <SectionHeader
        level={3}
        title="Remaining budget"
        description="Base caps and approved amendments stay visible separately."
      />
      <MetricStrip>
        <MetricTile
          label="Model requests"
          value={`${remaining.modelRequests} left`}
          hint={hint(
            run.budget.base.modelRequests,
            increase.modelRequests,
            run.budget.used.modelRequests,
          )}
        />
        <MetricTile
          label="Tool calls"
          value={`${remaining.successfulToolCalls} left`}
          hint={hint(
            run.budget.base.successfulToolCalls,
            increase.successfulToolCalls,
            run.budget.used.successfulToolCalls,
          )}
        />
        <MetricTile
          label="Rounds"
          value={`${remaining.rounds} left`}
          hint={hint(run.budget.base.rounds, increase.rounds, run.budget.used.rounds)}
        />
        <MetricTile
          label="Cost"
          value={`${formatUsd(remaining.costUsd)} left`}
          hint={`base ${formatUsd(run.budget.base.costUsd)} + ${formatUsd(increase.costUsd)} amended · ${formatUsd(run.budget.used.costUsd)} used`}
        />
      </MetricStrip>
      {run.budget.amendments.length > 0 && (
        <ul aria-label="Budget amendments" className="flex flex-col gap-1 text-xs text-status-neutral">
          {run.budget.amendments.map((amendment) => (
            <li key={amendment.amendmentId}>
              <code>{amendment.amendmentId}</code> · {amendment.reason}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function ProofTrail({ insights }: { insights: readonly ExplorationInsightView[] }) {
  const withProof = insights.filter((insight) => insight.proof.length > 0);
  return (
    <section aria-label="Proof trail" className="flex flex-col gap-2">
      <SectionHeader
        level={3}
        title="Proof trail"
        description="Machine-checkable receipt → fact edges for terminal insight records."
      />
      {withProof.length === 0 ? (
        <p className="text-sm text-status-neutral">No terminal proof records yet.</p>
      ) : (
        <div className="flex flex-col gap-2">
          {withProof.map((insight) => (
            <Card key={insight.insightId} tone="quiet" className="px-3 py-2">
              <Disclosure
                summary={insight.statement}
                meta={`${insight.proof.length} receipt edge${insight.proof.length === 1 ? "" : "s"}`}
                defaultOpen
              >
                <ol
                  aria-label={`${insight.statement} proof edges`}
                  className="flex flex-col gap-2 text-xs"
                >
                  {insight.proof.map((proof) => (
                    <li
                      key={`${proof.receiptId}:${proof.comparison}:${proof.factIds.join(",")}`}
                      className="grid gap-1 sm:grid-cols-[auto_1fr]"
                    >
                      <Badge
                        tone={proof.comparison === "supports" ? "ok" : "critical"}
                        variant="outline"
                      >
                        {proof.comparison}
                      </Badge>
                      <span>
                        <code>{proof.receiptId}</code>
                        {" → "}
                        {proof.factIds.map((factId) => (
                          <code key={factId} className="mr-1">
                            {factId}
                          </code>
                        ))}
                      </span>
                    </li>
                  ))}
                </ol>
              </Disclosure>
            </Card>
          ))}
        </div>
      )}
    </section>
  );
}

export function ExplorationRunPanel({ run }: { run: ExplorationRunView }) {
  const current = run.currentHypothesis;
  return (
    <Card
      as="section"
      aria-label="Exploration run"
      className="flex flex-col gap-5 px-4 py-4"
    >
      <SectionHeader
        title={run.goal}
        description={`${run.tier} exploration · ${run.explorationId}`}
        actions={<CostChip run={run} />}
      />
      <RunStatus status={run.status} />
      {run.status === "stopped" && <StopNotice reason={run.stopReason} />}

      <div className="grid gap-4 lg:grid-cols-2">
        <section aria-label="Current hypothesis" className="flex flex-col gap-2">
          <SectionHeader level={3} title="Current hypothesis" />
          {current ? (
            <Card tone="quiet" className="flex flex-col gap-2 px-3 py-2">
              <div className="flex flex-wrap items-center gap-2">
                <Badge tone="brand" variant="outline">
                  {current.status}
                </Badge>
                <code className="text-xs text-status-neutral">{current.hypothesisId}</code>
              </div>
              <p className="text-sm font-medium">{current.statement}</p>
              <p className="text-xs text-status-neutral">
                Why this hypothesis: {current.whySelected}
              </p>
            </Card>
          ) : (
            <p className="text-sm text-status-neutral">No hypothesis is active.</p>
          )}
        </section>

        <section aria-label="Current evidence" className="flex flex-col gap-2">
          <SectionHeader level={3} title="Current evidence" />
          {run.currentEvidence.length === 0 ? (
            <p className="text-sm text-status-neutral">No evidence committed yet.</p>
          ) : (
            <ul className="flex flex-col gap-2">
              {run.currentEvidence.map((evidence) => (
                <Card as="li" key={evidence.receiptId} tone="quiet" className="px-3 py-2">
                  <code className="text-xs">{evidence.receiptId}</code>
                  <p className="text-sm">{evidence.summary}</p>
                  <p className="text-xs text-status-neutral">
                    Facts: {evidence.factIds.join(", ")}
                  </p>
                </Card>
              ))}
            </ul>
          )}
        </section>
      </div>

      <BudgetSummary run={run} />
      <ProofTrail insights={run.insights} />
    </Card>
  );
}

function InsightList({
  insights,
  empty,
}: {
  insights: readonly ExplorationInsightView[];
  empty: string;
}) {
  if (insights.length === 0) {
    return <p className="text-sm text-status-neutral">{empty}</p>;
  }
  return (
    <ul className="flex flex-col gap-2">
      {insights.map((insight) => (
        <Card as="li" key={insight.insightId} tone="quiet" className="px-3 py-2">
          <div className="flex flex-wrap items-center gap-2">
            <EvidenceLaneBadge lane={insight.evidenceLane} />
            <Badge tone={insight.status === "refuted" ? "critical" : "neutral"}>
              {insight.status}
            </Badge>
            <span className="text-xs text-status-neutral">{insight.family}</span>
          </div>
          <p className="mt-1 text-sm">{insight.statement}</p>
        </Card>
      ))}
    </ul>
  );
}

function ReportSectionContent({
  id,
  run,
}: {
  id: ExplorationReportSectionId;
  run: ExplorationRunView;
}) {
  const groups = buildExplorationReportGroups(run);
  if (id === "supported_insights") {
    return (
      <InsightList
        insights={groups.supportedInsights}
        empty="No supported terminal insights."
      />
    );
  }
  if (id === "refuted_hypotheses") {
    return (
      <InsightList
        insights={groups.refutedHypotheses}
        empty="No hypotheses were refuted."
      />
    );
  }
  if (id === "inconclusive_questions") {
    return (
      <InsightList
        insights={groups.inconclusiveQuestions}
        empty="No inconclusive questions were recorded."
      />
    );
  }
  if (id === "limitations") {
    return groups.limitations.length === 0 ? (
      <p className="text-sm text-status-neutral">No limitations were recorded.</p>
    ) : (
      <ul className="list-disc pl-5 text-sm">
        {groups.limitations.map((limitation) => (
          <li key={limitation}>{limitation}</li>
        ))}
      </ul>
    );
  }
  if (id === "coverage_gaps") {
    return (
      <div className="flex flex-col gap-2">
        {run.coverageProjectionConsistent === false && (
          <p role="alert" className="text-sm text-status-warn">
            The server coverage summary was inconsistent; this list is derived from declared targets minus completed targets.
          </p>
        )}
        {groups.coverageGaps.length === 0 ? (
          <p className="text-sm text-status-neutral">All declared coverage targets were explored.</p>
        ) : (
          <ul aria-label="Not explored" className="list-disc pl-5 text-sm">
            {groups.coverageGaps.map((gap) => (
              <li key={gap}>
                <code>{gap}</code>
              </li>
            ))}
          </ul>
        )}
      </div>
    );
  }
  const effective = effectiveBudgetCaps(run.budget);
  return (
    <div className="flex flex-col gap-2 text-sm">
      <CostChip run={run} />
      <p>
        Structured stop reason: <strong>{stopReasonLabel(run.stopReason)}</strong>
        {run.stopReason && <code className="ml-2">{run.stopReason}</code>}
      </p>
      <p className="text-status-neutral">
        Effective hard cap {formatUsd(effective.costUsd)}; observed cost {formatUsd(run.budget.used.costUsd)}.
      </p>
    </div>
  );
}

export function ExplorationReport({ run }: { run: ExplorationRunView }) {
  return (
    <article aria-label="Exploration report" className="flex flex-col gap-4">
      <SectionHeader
        level={2}
        title="Exploration report"
        description="Exploratory leads are hypothesis-generating. Confirmatory evidence comes from a designated evidence lane and still carries limitations; neither label is a claim of certainty."
      />
      {EXPLORATION_REPORT_SECTION_ORDER.map((id, index) => (
        <Card
          as="section"
          key={id}
          data-section-id={id}
          className="flex flex-col gap-3 px-4 py-3"
        >
          <SectionHeader
            level={3}
            title={`${index + 1}. ${EXPLORATION_REPORT_SECTION_TITLES[id]}`}
          />
          <ReportSectionContent id={id} run={run} />
        </Card>
      ))}
    </article>
  );
}
