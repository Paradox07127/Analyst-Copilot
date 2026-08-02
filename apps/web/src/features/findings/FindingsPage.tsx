/* Findings slice (§7.5 slice H): the project-level findings library viewed
 * from this session. Findings are cross-session project state — cards carry a source
 * run link, evidence links into that run's Artifacts page, and a freshness
 * badge from the server-side assessor. A fresh finding can be promoted into
 * the project's verified answers through the same prepare → approve two-step
 * question execution uses, since it writes the semantic layer.
 *
 * Everything on this page is project-scoped while the URL is run-scoped, so
 * the scope is stated in the open rather than left for the reader to infer
 * from a source-run link. */

import { useState } from "react";
import { Link, useParams, useSearchParams } from "react-router";
import type {
  DecisionCoverageView,
  FindingSummary,
  InvestigationLogEntry,
  KnowledgePromotionPrepared,
} from "../../api/client";
import {
  useDecisionCoverage,
  useFindings,
  usePrepareFindingPromotion,
  usePromoteFinding,
} from "../../api/hooks";
import { artifactPath, sessionSectionPath } from "../../app/paths";
import {
  EmptyState,
  ErrorState,
  LoadingSkeleton,
} from "../../components/async-states";
import {
  Badge,
  Card,
  Disclosure,
  Dot,
  Hint,
  MetricStrip,
  MetricTile,
  SectionHeader,
  formatCompact,
  formatPercent,
  type Tone,
} from "../../components/ui";

const FRESHNESS_TONE: Record<string, Tone> = {
  fresh: "ok",
  stale: "critical",
  unverifiable: "warn",
};

const READINESS_TONE: Record<string, Tone> = {
  eligible: "ok",
  eligible_with_limitations: "warn",
  not_eligible: "neutral",
};

const RELIABILITY_TONE: Record<string, Tone> = {
  high: "ok",
  medium: "warn",
  low: "critical",
};

const OUTCOME_TONE: Record<string, Tone> = {
  inconclusive: "warn",
  needs_data: "info",
};

function PromoteControl({
  sessionId,
  finding,
}: {
  sessionId: string;
  finding: FindingSummary;
}) {
  const prepare = usePrepareFindingPromotion(sessionId);
  const promote = usePromoteFinding(sessionId);
  const [prepared, setPrepared] = useState<KnowledgePromotionPrepared | null>(
    null,
  );
  const [done, setDone] = useState<string | null>(null);
  const fresh = finding.freshness.status === "fresh";

  if (done) {
    return (
      <p role="status" className="text-xs text-status-ok">
        {done}
      </p>
    );
  }

  if (prepared) {
    return (
      <Card tone="warn" className="flex flex-col gap-2 p-3 text-xs">
        <p className="font-medium text-status-warn">
          {prepared.replaces_existing
            ? "This replaces the verified answer already stored for the same question."
            : "This will be stored as a verified answer for the whole project."}
        </p>
        <dl className="flex flex-col gap-1">
          <dt className="text-status-neutral">Question</dt>
          <dd>{prepared.question}</dd>
          <dt className="text-status-neutral">Answer</dt>
          <dd>{prepared.answer}</dd>
          <dt className="text-status-neutral">Evidence</dt>
          <dd className="font-mono">{prepared.evidence_note}</dd>
        </dl>
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            disabled={promote.isPending}
            onClick={() =>
              promote.mutate(
                {
                  findingId: finding.artifact_id,
                  action_hash: prepared.action_hash,
                  approval_token: prepared.approval_token,
                },
                {
                  onSuccess: (result) => {
                    setPrepared(null);
                    setDone(
                      `Promoted. The project now has ${result.verified_answer_count} verified answer(s).`,
                    );
                  },
                },
              )
            }
            className="rounded-base bg-primary px-3 py-1 font-medium text-bg hover:opacity-90 disabled:opacity-50"
          >
            {promote.isPending ? "Promoting…" : "Confirm promotion"}
          </button>
          <button
            type="button"
            onClick={() => setPrepared(null)}
            className="rounded-base border border-border px-3 py-1 hover:bg-code-bg"
          >
            Cancel
          </button>
        </div>
        {promote.isError && (
          <p role="alert" className="text-status-critical">
            {promote.error instanceof Error
              ? promote.error.message
              : "Promotion failed."}
          </p>
        )}
      </Card>
    );
  }

  return (
    <div className="flex flex-col gap-1">
      <button
        type="button"
        disabled={!fresh || prepare.isPending}
        title={
          fresh
            ? undefined
            : "Only a fresh finding can become verified knowledge."
        }
        onClick={() =>
          prepare.mutate(finding.artifact_id, { onSuccess: setPrepared })
        }
        className="self-start rounded-base border border-border px-2 py-1 text-xs hover:bg-code-bg disabled:opacity-50"
      >
        {prepare.isPending ? "Preparing…" : "Promote to verified answer"}
      </button>
      {prepare.isError && (
        <p role="alert" className="text-xs text-status-critical">
          {prepare.error instanceof Error
            ? prepare.error.message
            : "Could not prepare the promotion."}
        </p>
      )}
    </div>
  );
}

function FindingCard({
  projectId,
  sessionId,
  finding,
}: {
  projectId: string;
  sessionId: string;
  finding: FindingSummary;
}) {
  /* Dedupe evidence by artifact id; keep the run the server says can show it
   * (null session_id → the artifact is unreachable or lives in an internal run). */
  const evidenceRuns = new Map<string, string | null>();
  for (const evidence of (finding.statements ?? []).flatMap(
    (statement) => statement.evidence ?? [],
  )) {
    if (evidence.artifact_id && !evidenceRuns.has(evidence.artifact_id)) {
      evidenceRuns.set(evidence.artifact_id, evidence.session_id ?? null);
    }
  }
  const sourceSessionNavigable = finding.source_session_navigable !== false;
  return (
    <li>
      <Card className="flex flex-col gap-2 p-4">
        <h3 className="text-sm font-semibold">{finding.question}</h3>
        {/* Outcome first, provenance second: the report/freshness pair is what
         * decides whether this finding can be used at all. */}
        <div className="flex flex-wrap items-center gap-1.5">
          <Badge tone={READINESS_TONE[finding.report_readiness]}>
            {`report ${finding.report_readiness.replaceAll("_", " ")}`}
          </Badge>
          <Badge tone={FRESHNESS_TONE[finding.freshness.status]}>
            {`freshness ${finding.freshness.status}`}
          </Badge>
          {finding.from_current_session ? (
            <Badge tone="brand">this session</Badge>
          ) : (
            <Badge variant="outline">{`run ${finding.source_session_id}`}</Badge>
          )}
        </div>
        <ul className="flex list-disc flex-col gap-1 border-l-2 border-primary/30 py-1 pl-6 text-sm">
          {(finding.statements ?? []).map((statement, index) => (
            <li key={index}>{statement.text}</li>
          ))}
        </ul>
        {finding.interpretation && (
          <p className="text-xs text-status-neutral">
            Interpretation (validated): {finding.interpretation}
          </p>
        )}
        {finding.value_hypothesis && (
          <p className="text-xs text-status-neutral">
            Hypothesis (unvalidated): {finding.value_hypothesis}
          </p>
        )}
        {(finding.limitations ?? []).length > 0 && (
          <p className="text-xs text-status-neutral">
            Limitations: {(finding.limitations ?? []).join(" | ")}
          </p>
        )}
        {/* The readiness badge alone cannot say whether the blocker is the data
         * or the method, so always print the reason. */}
        <p className="text-xs text-status-neutral">
          Report status reason: {finding.report_readiness_reason}
        </p>
        {finding.freshness.status !== "fresh" &&
          (finding.freshness.reasons ?? []).length > 0 && (
            <ul className="flex flex-col gap-0.5 text-xs text-status-warn">
              {(finding.freshness.reasons ?? []).map((reason, index) => (
                <li key={index}>{reason}</li>
              ))}
            </ul>
          )}
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
          <Badge tone="neutral">{finding.claim_class}</Badge>
          <Badge tone={RELIABILITY_TONE[finding.evidence_support]}>
            {`evidence ${finding.evidence_support}`}
          </Badge>
          <Badge tone={RELIABILITY_TONE[finding.analytical_reliability]}>
            {`reliability ${finding.analytical_reliability}`}
          </Badge>
          <Badge tone={RELIABILITY_TONE[finding.decision_readiness]}>
            {`decision ${finding.decision_readiness}`}
          </Badge>
        </div>
        <div className="flex flex-wrap items-center gap-3 text-xs">
          {sourceSessionNavigable ? (
            <Link
              to={sessionSectionPath(projectId, finding.source_session_id, "artifacts")}
              className="text-primary underline-offset-2 hover:underline"
            >
              {`Source session ${finding.source_session_id}`}
            </Link>
          ) : (
            <span className="text-status-neutral">
              {`Source session ${finding.source_session_id}`}
            </span>
          )}
          {Array.from(evidenceRuns, ([artifactId, evidenceRunId]) =>
            evidenceRunId ? (
              <Link
                key={artifactId}
                to={artifactPath(projectId, evidenceRunId, artifactId)}
                className="font-mono text-primary underline-offset-2 hover:underline"
              >
                {`evidence ${artifactId}`}
              </Link>
            ) : (
              <span key={artifactId} className="font-mono text-status-neutral">
                {`evidence ${artifactId}`}
              </span>
            ),
          )}
          {finding.method_artifact_id && (
            sourceSessionNavigable ? (
              <Link
                to={artifactPath(
                  projectId,
                  finding.source_session_id,
                  finding.method_artifact_id,
                )}
                className="font-mono text-primary underline-offset-2 hover:underline"
              >
                {`method ${finding.method_artifact_id}`}
              </Link>
            ) : (
              <span className="font-mono text-status-neutral">
                {`method ${finding.method_artifact_id}`}
              </span>
            )
          )}
        </div>
        <PromoteControl sessionId={sessionId} finding={finding} />
      </Card>
    </li>
  );
}

/* Count investigation log rows by outcome status. */
function outcomeCounts(records: InvestigationLogEntry[]): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const record of records) {
    counts[record.status] = (counts[record.status] ?? 0) + 1;
  }
  return counts;
}

/** Empty states teach the interface: five zeroes at the top of the page say
 *  nothing about how a finding is produced or what to do next. */
function FindingsPrimer({
  projectId,
  sessionId,
  records,
}: {
  projectId: string;
  sessionId: string;
  records: InvestigationLogEntry[];
}) {
  return (
    <Card tone="quiet" className="flex flex-col gap-2 p-4">
      <p className="text-base font-semibold">No validated findings yet</p>
      <p className="text-sm text-status-neutral">
        Validated findings come from investigations, not from Auto EDA
        alone — Auto EDA profiles and screens the data but does not validate a
        finding. Run an investigation on a question to get one here.
      </p>
      {records.length > 0 && (
        <p className="text-sm text-status-neutral">
          {records.length} investigation(s) in this project already reached an
          outcome without producing one. The investigation log below says why
          and what each needs next.
        </p>
      )}
      <Disclosure summary="What counts as a validated finding?">
        <p className="text-sm text-status-neutral">
          A statement an investigation checked against the data and stored with
          the artifacts that produced it: the query or table it came from, the
          limitations found while checking, and a freshness check that turns
          stale when the source data changes. Anything not checked that way
          stays a question.
        </p>
      </Disclosure>
      <Link
        to={sessionSectionPath(projectId, sessionId, "questions")}
        className="self-start text-sm text-primary underline-offset-2 hover:underline"
      >
        Go to Questions to start an investigation
      </Link>
    </Card>
  );
}

/* One banded strip carrying its own scope: the counts are the project's,
 * not this session's. */
function FindingsMetrics({
  findings,
  records,
}: {
  findings: FindingSummary[];
  records: InvestigationLogEntry[];
}) {
  const counts = outcomeCounts(records);
  const fromThisRun = findings.filter((item) => item.from_current_session).length;
  const eligible = findings.filter(
    (item) => item.report_readiness === "eligible",
  ).length;
  const limited = findings.filter(
    (item) => item.report_readiness === "eligible_with_limitations",
  ).length;

  return (
    <MetricStrip>
      <MetricTile
        label="Validated findings"
        value={formatCompact(findings.length)}
        tone="ok"
        emphasis={findings.length > 0}
        hint={
          findings.length === 0
            ? "None in this project yet."
            : `${fromThisRun} from this session · ${findings.length - fromThisRun} from other sessions`
        }
      />
      <MetricTile
        label="Direct report"
        value={formatCompact(eligible)}
        hint="Usable in a report as they stand."
      />
      <MetricTile
        label="With limitations"
        value={formatCompact(limited)}
        hint="Usable once the caveat travels with the claim."
      />
      <MetricTile
        label="Inconclusive"
        value={formatCompact(counts["inconclusive"] ?? 0)}
        hint="Investigated, no answer the data supports."
      />
      <MetricTile
        label="Needs data"
        value={formatCompact(counts["needs_data"] ?? 0)}
        hint="Blocked on a column or table that is missing."
      />
    </MetricStrip>
  );
}

function DecisionCoverageBody({
  projectId,
  sessionId,
  coverage,
}: {
  projectId: string;
  sessionId: string;
  coverage: DecisionCoverageView;
}) {
  if (coverage.top_cards_total === 0) {
    return (
      <p className="text-xs text-status-neutral">
        Decision coverage: no candidate questions found yet.
      </p>
    );
  }
  const resolved =
    coverage.top_cards_total > 0
      ? formatPercent(coverage.top_cards_terminal / coverage.top_cards_total, 0)
      : "—";
  return (
    <div className="flex flex-col gap-3">
      <div className="flex">
        <Badge tone={coverage.coverage_ready ? "ok" : "warn"}>
          {coverage.coverage_ready
            ? "Report-ready coverage"
            : "Coverage gaps remain"}
        </Badge>
      </div>
      <MetricStrip>
        <MetricTile
          label="Top questions resolved"
          value={`${coverage.top_cards_terminal}/${coverage.top_cards_total}`}
          hint={`${resolved} of the highest-ranked questions reached an outcome.`}
        />
        <MetricTile
          label="Coverage: validated findings"
          value={formatCompact(coverage.validated_findings)}
        />
        <MetricTile
          label="Coverage: findings not eligible"
          value={formatCompact(coverage.findings_not_eligible)}
        />
      </MetricStrip>
      <div className="flex flex-col gap-1">
        <p className="flex items-center gap-1.5 text-xs font-semibold">
          Uninvestigated high-value questions
          <Hint label="Where these come from">
            Ranked across every run in this project, including runs over other
            datasets. A question listed here may not be answerable from the data
            loaded into the session you have open.
          </Hint>
        </p>
        {(coverage.uninvestigated_high_value ?? []).length === 0 ? (
          <EmptyState title="No uninvestigated high-value questions — every top card has reached an outcome." />
        ) : (
          <>
            <ul className="flex list-disc flex-col gap-1 pl-5 text-sm">
              {(coverage.uninvestigated_high_value ?? []).map((question) => (
                <li key={question}>{question}</li>
              ))}
            </ul>
            <Link
              to={sessionSectionPath(projectId, sessionId, "questions")}
              className="self-start text-xs text-primary underline-offset-2 hover:underline"
            >
              See the questions raised on this session's data
            </Link>
          </>
        )}
      </div>
      <div className="flex flex-col gap-1">
        <p className="text-xs font-semibold">Gaps</p>
        {(coverage.gaps ?? []).length === 0 ? (
          <EmptyState title="No coverage gaps." />
        ) : (
          <ul className="flex list-disc flex-col gap-1 pl-5 text-sm">
            {(coverage.gaps ?? []).map((gap) => (
              <li key={gap}>{gap}</li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

function DecisionCoverageSection({
  projectId,
  sessionId,
}: {
  projectId: string;
  sessionId: string;
}) {
  const coverage = useDecisionCoverage(sessionId);
  return (
    <section className="flex flex-col gap-2">
      <SectionHeader
        title="Decision coverage"
        description="Measured across every session in this project, not just the one you have open — the questions below can come from a different session's datasets."
      />
      {coverage.isPending && (
        <LoadingSkeleton lines={3} label="Loading decision coverage" />
      )}
      {coverage.isError && (
        <ErrorState error={coverage.error} onRetry={() => coverage.refetch()} />
      )}
      {coverage.data && (
        <DecisionCoverageBody
          projectId={projectId}
          sessionId={sessionId}
          coverage={coverage.data}
        />
      )}
    </section>
  );
}

function InvestigationLog({
  projectId,
  sessionId,
  records,
  selected,
  onSelectedChange,
}: {
  projectId: string;
  sessionId: string;
  records: InvestigationLogEntry[];
  selected: Set<string>;
  onSelectedChange: (selected: Set<string>) => void;
}) {
  const statuses = [...new Set(records.map((record) => record.status))].sort();
  if (records.length === 0) {
    /* Dropping the whole section would make "nothing recorded" and "failed to
     * load" look identical, so say which one it is. */
    return (
      <section className="flex flex-col gap-2">
        <SectionHeader
          title="Investigation log"
          description="Investigations that ended without a validated finding, and what each one needs next."
        />
        <EmptyState
          title="No investigation outcomes have been recorded yet."
          description="An investigation lands here when it finishes without an answer the data supports — start one from Questions to see how it reports back."
        />
        <Link
          to={sessionSectionPath(projectId, sessionId, "questions")}
          className="self-start text-xs text-primary underline-offset-2 hover:underline"
        >
          Go to Questions
        </Link>
      </section>
    );
  }

  const counts = outcomeCounts(records);
  const visible = records.filter((record) => selected.has(record.status));
  const toggle = (status: string) => {
    const next = new Set(selected);
    if (next.has(status)) next.delete(status);
    else next.add(status);
    onSelectedChange(next);
  };

  return (
    <section className="flex flex-col gap-2">
      <SectionHeader
        title="Investigation log"
        description="Investigations that ended without a validated finding, and what each one needs next."
      />
      <fieldset className="flex flex-col gap-1">
        <legend className="text-xs text-status-neutral">Outcome status</legend>
        <div className="flex flex-wrap gap-2">
          {statuses.map((status) => (
            <label
              key={status}
              className="flex items-center gap-1.5 rounded-base border border-border px-2 py-1 text-xs"
            >
              <input
                type="checkbox"
                checked={selected.has(status)}
                onChange={() => toggle(status)}
              />
              {`${status} (${counts[status]})`}
            </label>
          ))}
        </div>
      </fieldset>
      {visible.length === 0 ? (
        <EmptyState
          title="No investigation outcomes match the selected statuses"
          description="Select at least one outcome status to see log entries."
        />
      ) : (
        <ul className="flex flex-col gap-2">
          {visible.map((record) => (
            <li key={record.artifact_id}>
              <Card tone="quiet" className="flex flex-col gap-1 p-3 text-sm">
                <div className="flex flex-wrap items-center gap-2">
                  <Dot tone={OUTCOME_TONE[record.status] ?? "neutral"} />
                  <Badge tone={OUTCOME_TONE[record.status]}>
                    {record.status}
                  </Badge>
                  <span className="text-xs text-status-neutral">
                    {record.reason_code}
                  </span>
                  <span className="ml-auto text-xs text-status-neutral">
                    {record.from_current_session
                      ? "this session"
                      : record.source_session_id}
                  </span>
                </div>
                <p className="font-medium">{record.question}</p>
                <p className="text-xs text-status-neutral">{record.reason}</p>
                <p className="text-xs text-status-neutral">
                  Next: {record.next_action}
                </p>
              </Card>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

const RELIABILITY_OPTIONS = ["All", "high", "medium", "low"];

function FindingsList({
  projectId,
  sessionId,
  findings,
  reliability,
  onReliabilityChange,
}: {
  projectId: string;
  sessionId: string;
  findings: FindingSummary[];
  reliability: string;
  onReliabilityChange: (reliability: string) => void;
}) {
  if (findings.length === 0) return null;

  const visible = findings.filter(
    (finding) =>
      reliability === "All" || finding.analytical_reliability === reliability,
  );

  return (
    <section className="flex flex-col gap-3" aria-labelledby="finding-library-title">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div className="flex flex-col gap-1">
          <h2 id="finding-library-title" className="text-base font-semibold">
            Validated result library
          </h2>
          <p role="status" className="text-xs text-status-neutral">
            Showing {visible.length} of {findings.length} project finding(s)
          </p>
        </div>
        <label
          htmlFor="findings-reliability-filter"
          className="flex w-fit flex-col gap-1 text-xs text-status-neutral"
        >
          Analytical reliability
          <select
            id="findings-reliability-filter"
            value={reliability}
            onChange={(event) => onReliabilityChange(event.target.value)}
            className="rounded-base border border-border bg-bg px-2 py-1 text-sm text-text"
          >
            {RELIABILITY_OPTIONS.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </label>
      </div>
      {visible.length === 0 ? (
        <EmptyState
          title="No findings at this reliability level"
          description="Choose a different analytical reliability to see more findings."
        />
      ) : (
        <ul className="flex flex-col gap-3">
          {visible.map((finding) => (
            <FindingCard
              key={finding.artifact_id}
              projectId={projectId}
              sessionId={sessionId}
              finding={finding}
            />
          ))}
        </ul>
      )}
    </section>
  );
}

export function Component() {
  const { projectId = "", sessionId = "" } = useParams();
  const [searchParams, setSearchParams] = useSearchParams();
  const findings = useFindings(sessionId);

  if (findings.isPending) {
    return <LoadingSkeleton lines={4} label="Loading findings" />;
  }
  if (findings.isError) {
    return (
      <div className="p-6">
        <ErrorState error={findings.error} onRetry={() => findings.refetch()} />
      </div>
    );
  }
  const { findings: items, records, warnings } = findings.data;
  const findingItems = items ?? [];
  const recordItems = records ?? [];
  const recordStatuses = [
    ...new Set(recordItems.map((record) => record.status)),
  ].sort();
  const requestedReliability = searchParams.get("reliability") ?? "All";
  const reliability = RELIABILITY_OPTIONS.includes(requestedReliability)
    ? requestedReliability
    : "All";
  const requestedOutcomes = searchParams.get("outcome");
  const parsedOutcomes =
    requestedOutcomes?.split(",").filter((status) =>
      recordStatuses.includes(status),
    ) ?? [];
  const selectedOutcomes =
    requestedOutcomes === "none"
      ? new Set<string>()
      : requestedOutcomes === null || parsedOutcomes.length === 0
        ? new Set(recordStatuses)
        : new Set(parsedOutcomes);

  const updateParam = (key: "reliability" | "outcome", value?: string) =>
    setSearchParams(
      (current) => {
        const next = new URLSearchParams(current);
        if (value) next.set(key, value);
        else next.delete(key);
        return next;
      },
      { replace: true },
    );

  const updateOutcomes = (next: Set<string>) => {
    if (
      next.size === recordStatuses.length &&
      recordStatuses.every((status) => next.has(status))
    ) {
      updateParam("outcome");
      return;
    }
    updateParam(
      "outcome",
      next.size === 0 ? "none" : [...next].sort().join(","),
    );
  };

  return (
    <div className="mx-auto flex w-[95%] max-w-data flex-col gap-5 p-6">
      <header className="flex flex-col gap-1">
        <SectionHeader
          level={1}
          title="Findings"
          description="Validated conclusions and investigation outcomes collected across this project."
          actions={<Badge tone="brand">Project scope</Badge>}
        />
      </header>
      <Card tone="quiet" className="flex flex-wrap items-start gap-x-3 gap-y-1 p-3">
        <Badge variant="outline">Current session {sessionId}</Badge>
        <p className="min-w-64 flex-1 text-sm text-status-neutral">
          Results may come from other sessions in this project. Each card names
          its source session and opens evidence in that session, while Questions
          links keep the session you currently have open.
        </p>
      </Card>
      {(warnings ?? []).length > 0 && (
        <Card tone="warn" className="p-3">
          <div role="alert" className="text-xs text-status-warn">
            {(warnings ?? []).map((warning, index) => (
              <p key={index}>{warning}</p>
            ))}
          </div>
        </Card>
      )}
      {findingItems.length === 0 && (
        <FindingsPrimer
          projectId={projectId}
          sessionId={sessionId}
          records={recordItems}
        />
      )}
      <FindingsMetrics findings={findingItems} records={recordItems} />
      {/* Keyed by sessionId: filter state (reliability, outcome status) is local
       * to each of these components and must reset when the user switches
       * run. Distinct key prefixes keep the three siblings' keys unique. */}
      <FindingsList
        key={`findings-${sessionId}`}
        projectId={projectId}
        sessionId={sessionId}
        findings={findingItems}
        reliability={reliability}
        onReliabilityChange={(next) =>
          updateParam("reliability", next === "All" ? undefined : next)
        }
      />
      <DecisionCoverageSection
        key={`coverage-${sessionId}`}
        projectId={projectId}
        sessionId={sessionId}
      />
      <InvestigationLog
        key={`log-${sessionId}`}
        projectId={projectId}
        sessionId={sessionId}
        records={recordItems}
        selected={selectedOutcomes}
        onSelectedChange={updateOutcomes}
      />
    </div>
  );
}
