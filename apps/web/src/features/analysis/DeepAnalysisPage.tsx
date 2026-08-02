/* Deep analysis slice (§10.2 P1): the deterministic analysis the session already
 * produced — analysis tables, statistical tests and ML model cards. Read-only;
 * every number comes from a persisted artifact. */

import { useState } from "react";
import { Link, useParams } from "react-router";
import type {
  AnalysisTableView,
  ModelCardView,
  StatTestRow,
} from "../../api/client";
import { useAnalysis } from "../../api/hooks";
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
  Marquee,
  MetricStrip,
  MetricTile,
  SectionHeader,
  type Tone,
} from "../../components/ui";

const MAGNITUDE_TONE: Record<string, Tone> = {
  negligible: "neutral",
  small: "ok",
  medium: "warn",
  large: "brand",
};

const LEAKAGE_TONE: Record<string, Tone> = {
  unchecked: "neutral",
  clean: "ok",
  mitigated: "warn",
  risk: "critical",
};

const CHECK_TONE: Record<string, Tone> = {
  critical: "critical",
  warn: "warn",
  warning: "warn",
  info: "info",
};

function Count({ value, noun }: { value: number; noun: string }) {
  return (
    <span className="tabular text-sm text-status-neutral">
      {value} {noun}
      {value === 1 ? "" : "s"}
    </span>
  );
}

function SmallSampleNote({ sample }: { sample: number | null | undefined }) {
  if (sample === null || sample === undefined) return null;
  return (
    <p className="text-xs text-status-warn">
      Small sample (n={sample}) — interpret these results with care.
    </p>
  );
}

function cellText(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function RowTable({
  label,
  columns,
  rows,
  markTrivialFrom,
}: {
  label: string;
  columns: string[];
  rows: Record<string, unknown>[];
  /* Index at which the appended trivial pairs start, so they stay visually
   * separable once they are mixed into the same table. */
  markTrivialFrom?: number;
}) {
  return (
    <div className="overflow-x-auto rounded-base border border-border">
      <table className="tabular w-full text-sm">
        <caption className="sr-only">{label}</caption>
        <thead className="bg-table-header-bg text-left">
          <tr>
            {columns.map((column) => (
              <th
                key={column}
                scope="col"
                className="px-3 py-2 font-medium whitespace-nowrap"
              >
                {column}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => {
            const trivial =
              markTrivialFrom !== undefined && index >= markTrivialFrom;
            return (
              <tr
                key={index}
                className={`border-t border-table-border align-top ${
                  trivial ? "text-status-neutral" : ""
                }`}
              >
                {columns.map((column, cell) => (
                  <td key={column} className="px-3 py-2 whitespace-nowrap">
                    {cellText(row[column])}
                    {trivial && cell === 0 && (
                      <span className="ml-2 text-xs">(trivial)</span>
                    )}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/* One disclosure level per table, not two: the trivial pairs used to sit in a
 * <details> nested inside this one, and folding them into the same table as
 * marked rows keeps the comparison in one place (NN/g: summary → detail). */
function EvidenceLink({
  projectId,
  sessionId,
  artifactId,
  label = "Open evidence",
}: {
  projectId: string;
  sessionId: string;
  artifactId: string;
  label?: string;
}) {
  return (
    <Link
      to={artifactPath(projectId, sessionId, artifactId)}
      className="font-mono text-xs text-primary underline-offset-2 hover:underline"
    >
      {label}
    </Link>
  );
}

function AnalysisTableCard({
  projectId,
  sessionId,
  table,
}: {
  projectId: string;
  sessionId: string;
  table: AnalysisTableView;
}) {
  const [showTrivial, setShowTrivial] = useState(false);
  const columns = table.columns ?? [];
  const rows = (table.rows ?? []) as Record<string, unknown>[];
  const trivial = (table.trivial_rows ?? []) as Record<string, unknown>[];
  const shown = showTrivial ? [...rows, ...trivial] : rows;

  return (
    <Card className="px-4 py-3">
      <Disclosure
        summary={table.title}
        meta={`${table.dataset_name} · ${rows.length + trivial.length} row(s)`}
      >
        <div className="flex flex-col gap-3">
          <p className="text-sm text-status-neutral">{table.description}</p>
          <p className="text-xs text-status-neutral">
            Question: {table.question}
          </p>
          <EvidenceLink
            projectId={projectId}
            sessionId={sessionId}
            artifactId={table.artifact_id}
          />
          {table.small_sample && (
            <SmallSampleNote sample={table.min_sample_size} />
          )}
          {shown.length > 0 ? (
            <RowTable
              label={table.title}
              columns={columns}
              rows={shown}
              markTrivialFrom={showTrivial ? rows.length : undefined}
            />
          ) : (
            <p className="text-sm text-status-neutral">
              {trivial.length > 0
                ? `All ${trivial.length} correlation pair(s) here are trivial or degenerate — no substantive relationships to show.`
                : "This table has no rows."}
            </p>
          )}
          {trivial.length > 0 && (
            <div className="flex flex-col gap-1">
              <label className="flex w-fit items-center gap-2 text-xs">
                <input
                  type="checkbox"
                  checked={showTrivial}
                  onChange={(event) => setShowTrivial(event.target.checked)}
                />
                {`Show ${trivial.length} trivial/degenerate pair(s)`}
              </label>
              <p className="text-xs text-status-neutral">
                Trivial pairs are perfect or near-perfect correlations that add
                no new signal (complementary or rescaled columns).
              </p>
            </div>
          )}
        </div>
      </Disclosure>
    </Card>
  );
}

function StatTestTable({
  projectId,
  sessionId,
  tests,
}: {
  projectId: string;
  sessionId: string;
  tests: StatTestRow[];
}) {
  return (
    <div className="overflow-x-auto rounded-base border border-border">
      <table className="tabular w-full text-sm">
        <caption className="sr-only">
          Statistical tests with their p-values, effect sizes and verdicts
        </caption>
        <thead className="bg-table-header-bg text-left">
          <tr>
            <th scope="col" className="px-3 py-2 font-medium">
              Test
            </th>
            <th scope="col" className="px-3 py-2 font-medium">
              Dataset
            </th>
            <th scope="col" className="px-3 py-2 font-medium">
              Group / Value
            </th>
            <th scope="col" className="px-3 py-2 text-right font-medium">
              Statistic
            </th>
            <th scope="col" className="px-3 py-2 text-right font-medium">
              p-value
            </th>
            <th scope="col" className="px-3 py-2 font-medium">
              Effect size
            </th>
            <th scope="col" className="px-3 py-2 text-right font-medium">
              n
            </th>
            <th scope="col" className="px-3 py-2 font-medium">
              Conclusion
            </th>
          </tr>
        </thead>
        <tbody>
          {tests.map((test) => (
            <tr
              key={test.artifact_id}
              className="border-t border-table-border align-top"
            >
              <th
                scope="row"
                className="px-3 py-2 text-left font-mono text-xs font-normal whitespace-nowrap"
              >
                <EvidenceLink
                  projectId={projectId}
                  sessionId={sessionId}
                  artifactId={test.artifact_id}
                  label={test.test_type}
                />
              </th>
              <td className="px-3 py-2 whitespace-nowrap">{test.dataset_name}</td>
              <td className="px-3 py-2 whitespace-nowrap">
                {[test.group_column, test.value_column].filter(Boolean).join(" / ")}
              </td>
              <td className="px-3 py-2 text-right">{cellText(test.statistic)}</td>
              <td className="px-3 py-2 text-right whitespace-nowrap">
                {test.p_value_display || "—"}
              </td>
              <td className="px-3 py-2 whitespace-nowrap">
                {cellText(test.effect_size)}{" "}
                {test.effect_size_magnitude && (
                  <Badge tone={MAGNITUDE_TONE[test.effect_size_magnitude] ?? "neutral"}>
                    {test.effect_size_magnitude}
                  </Badge>
                )}
              </td>
              <td className="px-3 py-2 text-right whitespace-nowrap">
                {test.sample_size}
                {test.small_sample && (
                  <span className="ml-1 text-status-warn" title="Small sample">
                    <span aria-hidden>⚠</span>
                    <span className="sr-only">small sample</span>
                  </span>
                )}
              </td>
              <td className="px-3 py-2">
                <span
                  className={
                    test.significant === true
                      ? "font-medium text-status-ok"
                      : "text-status-neutral"
                  }
                >
                  {test.conclusion}
                </span>
                {(test.warnings ?? []).length > 0 && (
                  <p className="text-xs text-status-warn">
                    {(test.warnings ?? []).join(", ")}
                  </p>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Ranked bar plus the printed number: the bar carries the ordering, the
 *  number carries the value, because importances routinely span three orders
 *  of magnitude and a proportional bar hides everything but the top feature. */
function FeatureImportance({
  items,
}: {
  items: { feature: string; importance: number }[];
}) {
  const peak = Math.max(...items.map((item) => item.importance), 0);
  return (
    <div className="flex flex-col gap-1">
      <h4 className="text-xs font-medium text-status-neutral">
        Feature importance
      </h4>
      {items.map((item) => (
        <div key={item.feature} className="flex items-center gap-2 text-xs">
          <Marquee className="w-40 font-mono" title={item.feature}>
            {item.feature}
          </Marquee>
          <span
            aria-hidden
            className="inline-block h-1.5 w-24 shrink-0 overflow-hidden rounded-sm bg-track"
          >
            <span
              className="block h-full min-w-px rounded-sm bg-primary/60"
              style={{
                width: `${peak > 0 ? (item.importance / peak) * 100 : 0}%`,
              }}
            />
          </span>
          <span className="tabular text-status-neutral">{item.importance}</span>
        </div>
      ))}
    </div>
  );
}

function ModelCard({
  projectId,
  sessionId,
  card,
}: {
  projectId: string;
  sessionId: string;
  card: ModelCardView;
}) {
  const importance = card.feature_importance ?? [];
  const leakageTone = LEAKAGE_TONE[card.leakage_verdict] ?? "neutral";
  return (
    <Card as="li" className="flex flex-col gap-3 p-4">
      <div className="flex flex-wrap items-center gap-2">
        <h3 className="text-sm font-semibold">
          {card.target_column} · {card.task_type}
        </h3>
        <Badge>{card.model_type}</Badge>
        <Badge>{card.split_strategy}</Badge>
        <Badge tone={leakageTone} caps>
          <Dot tone={leakageTone} />
          {`leakage ${card.leakage_verdict}`}
        </Badge>
      </div>
      <p className="text-xs text-status-neutral">
        {card.dataset_name} · train {card.train_rows} / test {card.test_rows} rows ·{" "}
        {(card.feature_columns ?? []).length} feature(s)
      </p>
      <EvidenceLink
        projectId={projectId}
        sessionId={sessionId}
        artifactId={card.artifact_id}
      />

      <MetricStrip>
        {Object.entries(card.metrics ?? {}).map(([name, value]) => (
          <MetricTile
            key={name}
            label={name}
            value={value}
            tone="brand"
            emphasis={name === card.headline_metric}
            hint={name === card.headline_metric ? "headline metric" : undefined}
          />
        ))}
        {card.baseline_accuracy !== null && card.baseline_accuracy !== undefined && (
          <MetricTile
            label="baseline"
            value={card.baseline_accuracy}
            hint="what always guessing the majority class would score"
          />
        )}
      </MetricStrip>

      {importance.length > 0 && <FeatureImportance items={importance} />}
      {(card.excluded_features ?? []).length > 0 && (
        <p className="text-xs text-status-neutral">
          Excluded features: {(card.excluded_features ?? []).join(", ")}
        </p>
      )}
      {(card.leakage_checks ?? []).length > 0 && (
        <ul className="flex flex-col gap-1 text-xs">
          {(card.leakage_checks ?? []).map((check, index) => (
            <li key={index} className="flex flex-wrap items-baseline gap-x-2">
              <Badge tone={CHECK_TONE[check.severity] ?? "neutral"}>
                {check.severity}
              </Badge>
              <span>{check.message}</span>
              <span className="font-mono text-status-neutral">
                {check.code} · {check.action}
              </span>
            </li>
          ))}
        </ul>
      )}
      {(card.limitations ?? []).length > 0 && (
        <p className="text-xs text-status-warn">
          Limitations: {(card.limitations ?? []).join(" | ")}
        </p>
      )}
    </Card>
  );
}

export function Component() {
  const { projectId = "", sessionId = "" } = useParams();
  const analysis = useAnalysis(sessionId);

  if (analysis.isPending) {
    return <LoadingSkeleton lines={5} label="Loading deep analysis" />;
  }
  if (analysis.isError) {
    return (
      <div className="p-6">
        <ErrorState error={analysis.error} onRetry={() => analysis.refetch()} />
      </div>
    );
  }
  const tables = analysis.data.tables ?? [];
  const statTests = analysis.data.stat_tests ?? [];
  const modelCards = analysis.data.model_cards ?? [];
  const empty =
    tables.length === 0 && statTests.length === 0 && modelCards.length === 0;
  const significant = statTests.filter((test) => test.significant === true).length;
  const smallestStatSample = statTests
    .filter((test) => test.small_sample)
    .reduce<number | null>(
      (smallest, test) =>
        smallest === null ? test.sample_size : Math.min(smallest, test.sample_size),
      null,
    );

  return (
    <div className="mx-auto flex w-[95%] max-w-data flex-col gap-6 p-6">
      <SectionHeader
        level={1}
        title="Deep analysis"
        description="Review deterministic tables, statistical tests, and baseline models already produced by this session."
        actions={
          <Link
            to={sessionSectionPath(projectId, sessionId, "questions")}
            className="rounded-base bg-primary px-3 py-1.5 text-sm font-medium text-bg hover:opacity-90"
          >
            Run a question
          </Link>
        }
      />

      <Card tone="quiet" className="flex flex-wrap items-start gap-x-4 gap-y-2 p-3">
        <Badge tone="neutral">Review only</Badge>
        <p className="min-w-64 flex-1 text-sm text-status-neutral">
          This page reads persisted output and never starts work on its own.
          Approve and run a question to create more analysis. Baseline EDA is
          produced before question review and is labelled accordingly.
        </p>
      </Card>

      {empty ? (
        <div className="flex flex-col gap-3">
          <EmptyState
            title="No deterministic analysis artifacts"
            description="This session produced no analysis tables, statistical tests, or model cards."
          />
          <Link
            to={sessionSectionPath(projectId, sessionId, "questions")}
            className="self-start text-sm text-primary underline-offset-2 hover:underline"
          >
            Go to Questions to start an investigation
          </Link>
        </div>
      ) : (
        <>
          <section className="flex flex-col gap-2">
            <SectionHeader
              level={2}
              title="Analysis tables"
              description="Deterministic tables, each one collapsed to its title until you open it."
              actions={<Count value={tables.length} noun="table" />}
            />
            {tables.length === 0 ? (
              <p className="text-sm text-status-neutral">
                No analysis tables in this session.
              </p>
            ) : (
              tables.map((table) => (
                <AnalysisTableCard
                  key={table.artifact_id}
                  projectId={projectId}
                  sessionId={sessionId}
                  table={table}
                />
              ))
            )}
          </section>

          <section className="flex flex-col gap-2">
            <SectionHeader
              level={2}
              title="Statistical tests"
              description={
                statTests.length === 0
                  ? undefined
                  : `${significant} of ${statTests.length} reached significance at the stated alpha.`
              }
              actions={<Count value={statTests.length} noun="test" />}
            />
            {statTests.length === 0 ? (
              <p className="text-sm text-status-neutral">
                No statistical tests in this session.
              </p>
            ) : (
              <>
                <SmallSampleNote sample={smallestStatSample} />
                <StatTestTable
                  projectId={projectId}
                  sessionId={sessionId}
                  tests={statTests}
                />
              </>
            )}
          </section>

          <section className="flex flex-col gap-2">
            <SectionHeader
              level={2}
              title="ML baseline model cards"
              description="A baseline says whether the target is predictable at all — it is not a tuned model."
              actions={<Count value={modelCards.length} noun="card" />}
            />
            {modelCards.length === 0 ? (
              <p className="text-sm text-status-neutral">
                No model cards in this session.
              </p>
            ) : (
              <ul className="flex flex-col gap-3">
                {modelCards.map((card) => (
                  <ModelCard
                    key={card.artifact_id}
                    projectId={projectId}
                    sessionId={sessionId}
                    card={card}
                  />
                ))}
              </ul>
            )}
          </section>
        </>
      )}
    </div>
  );
}
