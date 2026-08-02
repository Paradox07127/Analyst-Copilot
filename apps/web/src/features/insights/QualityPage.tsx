import { useMemo } from "react";
import { Link, useParams } from "react-router";
import type { QualityIssueRow, QualityView } from "../../api/client";
import { useQuality } from "../../api/hooks";
import {
  EmptyState,
  ErrorState,
  LoadingSkeleton,
} from "../../components/async-states";
import {
  Badge,
  Card,
  Dot,
  Marquee,
  MetricStrip,
  MetricTile,
  SectionHeader,
  buttonClass,
  type Tone,
} from "../../components/ui";
import { sessionSectionPath, tablePath } from "../../app/paths";
import {
  parseCsvParam,
  serializeCsvParam,
  useRouteSearchParam,
} from "../../app/route-state";

const SEVERITIES = ["critical", "warn", "info"] as const;
type Severity = (typeof SEVERITIES)[number];

const SEVERITY_LABEL: Record<Severity, string> = {
  critical: "Critical",
  warn: "Warning",
  info: "Info",
};

const SEVERITY_TONE: Record<Severity, Tone> = {
  critical: "critical",
  warn: "warn",
  info: "ok",
};

/* Static class strings so Tailwind sees every severity variant. */
const SEVERITY_BADGE: Record<Severity, string> = {
  critical: "bg-status-critical/15 text-status-critical",
  warn: "bg-status-warn/15 text-status-warn",
  info: "bg-status-ok/15 text-status-ok",
};

const SEVERITY_MEANING: Record<Severity, string> = {
  critical: "Review before relying on the affected data.",
  warn: "Check whether this changes the analysis you plan to run.",
  info: "Context recorded by the profiler.",
};

function severityOf(issue: QualityIssueRow): Severity {
  return (SEVERITIES as readonly string[]).includes(issue.severity)
    ? (issue.severity as Severity)
    : "info";
}

/* Ranked, not gridded: nine equal cards said nothing about which table to open
 * first. Clicking one scopes the issue list below to that dataset. */
function AffectedDatasets({
  quality,
  selectedDataset,
  onSelect,
}: {
  quality: QualityView;
  selectedDataset: string;
  onSelect: (datasetId: string) => void;
}) {
  const cards = quality.datasets ?? [];
  if (cards.length === 0) return null;
  const ranked = [...cards].sort(
    (a, b) =>
      (b.critical ?? 0) - (a.critical ?? 0) ||
      (b.warn ?? 0) - (a.warn ?? 0) ||
      (b.info ?? 0) - (a.info ?? 0) ||
      a.dataset_name.localeCompare(b.dataset_name),
  );

  return (
    <section aria-labelledby="dataset-scope-heading" className="flex flex-col gap-2">
      <SectionHeader
        level={2}
        title={<span id="dataset-scope-heading">Dataset scope</span>}
        description="Ranked by recorded severity. Select a dataset to narrow the issue queue."
      />
      <Card tone="quiet" className="flex flex-col gap-1 p-2">
        {ranked.map((card) => {
          const id = card.dataset_id ?? card.dataset_name;
          const active = selectedDataset === id;
          const counts = SEVERITIES.map(
            (severity) => [severity, card[severity] ?? 0] as const,
          );
          const issueCount = counts.reduce((sum, [, count]) => sum + count, 0);
          return (
            <button
              key={id}
              type="button"
              aria-pressed={active}
              onClick={() => onSelect(active ? "" : id)}
              className={`flex min-w-0 flex-wrap items-center gap-x-3 gap-y-1 rounded-base border-l-2 px-2 py-1.5 text-left hover:bg-bg ${
                active ? "border-primary bg-bg" : "border-transparent"
              }`}
            >
              <Marquee className="min-w-0 flex-1 text-sm font-medium">
                {card.dataset_name}
              </Marquee>
              {issueCount === 0 ? (
                <span className="text-xs text-status-neutral">
                  No flags recorded
                </span>
              ) : (
                <>
                  <span className="tabular text-xs text-status-neutral">
                    {issueCount} {issueCount === 1 ? "flag" : "flags"}
                  </span>
                  {counts.map(([severity, count]) =>
                    count > 0 ? (
                      <span
                        key={severity}
                        className={`tabular rounded-base px-1.5 py-0.5 text-xs font-medium ${SEVERITY_BADGE[severity]}`}
                      >
                        {SEVERITY_LABEL[severity]}: {count}
                      </span>
                    ) : null,
                  )}
                </>
              )}
            </button>
          );
        })}
      </Card>
    </section>
  );
}

function IssueRow({
  issue,
  projectId,
  sessionId,
}: {
  issue: QualityIssueRow;
  projectId: string;
  sessionId: string;
}) {
  const datasetId = issue.dataset_id ?? "";
  return (
    <li className="flex min-w-0 flex-col gap-2 border-t border-hairline px-3 py-2.5 first:border-t-0">
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        <Badge tone="neutral">
          <span className="font-mono">{issue.code}</span>
        </Badge>
        <Marquee className="text-sm font-medium">
          {issue.dataset_name}
        </Marquee>
        {issue.column && (
          <span className="font-mono text-xs text-status-neutral">
            {issue.column}
          </span>
        )}
        {datasetId && (
          <Link
            to={tablePath(projectId, sessionId, datasetId)}
            className={`${buttonClass({ variant: "ghost", size: "sm" })} ml-auto`}
          >
            Inspect rows
          </Link>
        )}
      </div>
      <p className="break-words text-sm">{issue.message}</p>
      {issue.recommendation && (
        <div className="flex min-w-0 gap-2 rounded-base bg-surface px-2 py-1.5 text-xs">
          <span className="shrink-0 font-medium text-status-neutral">
            Profiler recommendation
          </span>
          <span className="min-w-0 break-words">{issue.recommendation}</span>
        </div>
      )}
    </li>
  );
}

function TriageSummary({
  issues,
  projectId,
  sessionId,
}: {
  issues: QualityIssueRow[];
  projectId: string;
  sessionId: string;
}) {
  const critical = issues.filter(
    (issue) => severityOf(issue) === "critical",
  );
  const warnings = issues.filter((issue) => severityOf(issue) === "warn");
  const priority = critical.length > 0 ? critical : warnings;
  const first = priority[0];
  const datasetCount = new Set(
    issues.map((issue) => issue.dataset_id ?? issue.dataset_name),
  ).size;
  const fieldCount = new Set(
    issues
      .filter((issue) => issue.column)
      .map(
        (issue) =>
          `${issue.dataset_id ?? issue.dataset_name}:${issue.column ?? ""}`,
      ),
  ).size;
  const title =
    critical.length > 0
      ? `Review ${critical.length} critical ${
          critical.length === 1 ? "flag" : "flags"
        } first`
      : warnings.length > 0
        ? `Review ${warnings.length} ${
            warnings.length === 1 ? "warning" : "warnings"
          }`
        : "Review the recorded context";

  return (
    <Card
      tone={critical.length > 0 ? "critical" : warnings.length > 0 ? "warn" : "quiet"}
      className="grid min-w-0 gap-3 p-4 lg:grid-cols-[minmax(0,1fr)_auto] lg:items-center"
    >
      <div className="min-w-0">
        <p className="mb-1 text-xs font-medium tracking-wide text-status-neutral uppercase">
          Next review
        </p>
        <h2 className="text-base font-semibold">{title}</h2>
        <p className="mt-1 max-w-content text-sm text-status-neutral">
          The profiler recorded {issues.length}{" "}
          {issues.length === 1 ? "flag" : "flags"} across {datasetCount}{" "}
          {datasetCount === 1 ? "dataset" : "datasets"}
          {fieldCount > 0
            ? ` and ${fieldCount} named ${fieldCount === 1 ? "field" : "fields"}`
            : ""}
          . Treat each flag as something to verify, not proof that the data is
          unusable.
        </p>
      </div>
      {/* Stacked, not side-by-side: two wide buttons in one row squeeze the
        * headline into a sliver at laptop widths. */}
      <div className="flex flex-col items-start gap-2 lg:items-end">
        {first?.dataset_id && (
          <Link
            to={tablePath(projectId, sessionId, first.dataset_id)}
            className={buttonClass({ variant: "primary" })}
          >
            Inspect highest-priority table
          </Link>
        )}
        {(critical.length > 0 || warnings.length > 0) && (
          <Link
            to={sessionSectionPath(projectId, sessionId, "cleaning")}
            className={buttonClass({ variant: "secondary" })}
          >
            Review cleaning options
          </Link>
        )}
      </div>
    </Card>
  );
}

function SeverityGroup({
  severity,
  issues,
  projectId,
  sessionId,
}: {
  severity: Severity;
  issues: QualityIssueRow[];
  projectId: string;
  sessionId: string;
}) {
  if (issues.length === 0) return null;
  return (
    <section className="flex flex-col gap-2">
      <header className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
        <Dot tone={SEVERITY_TONE[severity]} />
        <h3 className="text-sm font-semibold">{SEVERITY_LABEL[severity]}</h3>
        <span className="tabular text-sm text-status-neutral">
          {issues.length}
        </span>
        <span className="text-xs text-status-neutral">
          {SEVERITY_MEANING[severity]}
        </span>
      </header>
      <Card
        tone={severity === "critical" ? "critical" : severity === "warn" ? "warn" : "default"}
      >
        <ul className="flex flex-col">
          {issues.map((issue, index) => (
            <IssueRow
              key={`${issue.code}:${issue.dataset_name}:${issue.column ?? ""}:${index}`}
              issue={issue}
              projectId={projectId}
              sessionId={sessionId}
            />
          ))}
        </ul>
      </Card>
    </section>
  );
}

export function Component() {
  const { projectId = "", sessionId = "" } = useParams();
  const quality = useQuality(sessionId);
  const [dataset, setDataset] = useRouteSearchParam("dataset");
  const [severityParam, setSeverity] = useRouteSearchParam("severity");
  const severity = SEVERITIES.includes(severityParam as Severity)
    ? severityParam
    : "";
  /* Code multiselect: options
   * deduped from the loaded issues, default = every code selected. `null`
   * stands for that default so no effect is needed to seed it once data
   * loads; an explicit empty Set means the same thing as `null` (unchecking
   * every code falls back to "no filter"
   * `not selected_codes or row.code in selected_codes`). */
  const [codesParam, setCodesParam] = useRouteSearchParam("codes");
  const selectedCodes = codesParam
    ? new Set(parseCsvParam(codesParam))
    : null;

  /* Filter compares dataset ids, not display names: same-named datasets keep
   * distinct entries. The name fallback covers rows from older servers. */
  const datasetOptions = useMemo(
    () =>
      (quality.data?.datasets ?? []).map((card) => ({
        id: card.dataset_id ?? card.dataset_name,
        label: card.dataset_name,
      })),
    [quality.data],
  );

  const codeOptions = useMemo(() => {
    const counts = new Map<string, number>();
    for (const issue of quality.data?.issues ?? []) {
      counts.set(issue.code, (counts.get(issue.code) ?? 0) + 1);
    }
    return Array.from(counts, ([code, count]) => ({ code, count })).sort(
      (a, b) => a.code.localeCompare(b.code),
    );
  }, [quality.data]);

  const toggleCode = (code: string) => {
    const base =
      selectedCodes ?? new Set(codeOptions.map((option) => option.code));
    const next = new Set(base);
    if (next.has(code)) next.delete(code);
    else next.add(code);
    const allCodes = codeOptions.map((option) => option.code);
    setCodesParam(
      next.size === 0 || allCodes.every((option) => next.has(option))
        ? ""
        : serializeCsvParam(next),
    );
  };

  const allIssues = quality.data?.issues ?? [];
  const visible = allIssues.filter(
    (issue) =>
      (!dataset || (issue.dataset_id ?? issue.dataset_name) === dataset) &&
      (!severity || severityOf(issue) === severity) &&
      (!selectedCodes || selectedCodes.size === 0 || selectedCodes.has(issue.code)),
  );
  const filtered = visible.length !== allIssues.length;
  const affectedDatasets = new Set(
    allIssues.map((issue) => issue.dataset_id ?? issue.dataset_name),
  ).size;
  const affectedFields = new Set(
    allIssues
      .filter((issue) => issue.column)
      .map(
        (issue) =>
          `${issue.dataset_id ?? issue.dataset_name}:${issue.column ?? ""}`,
      ),
  ).size;
  const summaryTotal =
    (quality.data?.critical ?? 0) +
    (quality.data?.warn ?? 0) +
    (quality.data?.info ?? 0);

  return (
    <div className="mx-auto flex w-[95%] max-w-data min-w-0 flex-col gap-4 p-6">
      <SectionHeader
        level={1}
        title="Quality"
        description="Triage profiler flags by severity and scope, then inspect the evidence before deciding whether to clean."
      />

      {quality.isPending && <LoadingSkeleton lines={4} label="Loading quality" />}
      {quality.isError && (
        <ErrorState error={quality.error} onRetry={() => quality.refetch()} />
      )}
      {quality.data && (
        <>
          {allIssues.length > 0 && (
            <TriageSummary
              issues={allIssues}
              projectId={projectId}
              sessionId={sessionId}
            />
          )}

          <MetricStrip>
            <MetricTile
              label="Critical"
              value={quality.data.critical ?? 0}
              tone="critical"
              emphasis={(quality.data.critical ?? 0) > 0}
              hint="Highest-priority profiler flags"
            />
            <MetricTile
              label="Warnings"
              value={quality.data.warn ?? 0}
              tone="warn"
              emphasis={(quality.data.warn ?? 0) > 0}
              hint="Review for analysis impact"
            />
            <MetricTile
              label="Info"
              value={quality.data.info ?? 0}
              tone="ok"
              emphasis={(quality.data.info ?? 0) > 0}
              hint="Profiler context"
            />
            <MetricTile
              label="Affected datasets"
              value={affectedDatasets}
              hint="Datasets named in issue details"
            />
            <MetricTile
              label="Affected fields"
              value={affectedFields}
              hint="Named fields in issue details"
            />
          </MetricStrip>

          <AffectedDatasets
            quality={quality.data}
            selectedDataset={dataset}
            onSelect={setDataset}
          />

          {allIssues.length === 0 && summaryTotal > 0 ? (
            <EmptyState
              title="Issue details were not returned"
              description={`The quality summary reports ${summaryTotal} ${
                summaryTotal === 1 ? "flag" : "flags"
              }, but this response contains no issue-level evidence. Retry before treating the session as clear.`}
            />
          ) : allIssues.length === 0 ? (
            <EmptyState
              title="No quality flags recorded"
              description="The profiler did not record critical, warning, or informational flags for this session."
            />
          ) : (
            <>
              <Card tone="quiet" className="flex flex-col gap-2 px-3 py-2">
                <SectionHeader
                  level={2}
                  title="Issue queue"
                  description="Filters are saved in the URL so this review can be shared or resumed."
                />
                <div className="flex flex-wrap items-center gap-3">
                  <label className="flex min-w-0 flex-wrap items-center gap-2 text-sm">
                    <span className="text-status-neutral">Dataset</span>
                    <select
                      value={dataset}
                      onChange={(event) => setDataset(event.target.value)}
                      className="max-w-full min-w-0 rounded-base border border-border bg-bg px-2 py-1 text-sm"
                    >
                      <option value="">All datasets</option>
                      {datasetOptions.map((option) => (
                        <option key={option.id} value={option.id}>
                          {option.label}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="flex min-w-0 flex-wrap items-center gap-2 text-sm">
                    <span className="text-status-neutral">Severity</span>
                    <select
                      value={severity}
                      onChange={(event) => setSeverity(event.target.value)}
                      className="rounded-base border border-border bg-bg px-2 py-1 text-sm"
                    >
                      <option value="">All severities</option>
                      {SEVERITIES.map((option) => (
                        <option key={option} value={option}>
                          {SEVERITY_LABEL[option]}
                        </option>
                      ))}
                    </select>
                  </label>
                  <span className="tabular text-xs text-status-neutral sm:ml-auto">
                    {visible.length} of {allIssues.length} issues
                  </span>
                </div>
                <fieldset className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm">
                  <legend className="text-status-neutral">Code type</legend>
                  {codeOptions.map((option) => (
                    <label
                      key={option.code}
                      className="flex items-center gap-1 rounded-sm border border-border px-1.5 py-0.5"
                    >
                      <input
                        type="checkbox"
                        checked={selectedCodes?.has(option.code) ?? true}
                        onChange={() => toggleCode(option.code)}
                      />
                      <span className="font-mono text-xs">
                        {option.code} ({option.count})
                      </span>
                    </label>
                  ))}
                </fieldset>
              </Card>

              {visible.length === 0 ? (
                <EmptyState
                  title="No issues match the selected filters"
                  description={
                    filtered
                      ? "Clear a filter above to widen the list."
                      : undefined
                  }
                />
              ) : (
                <div className="flex flex-col gap-4">
                  {SEVERITIES.map((level) => (
                    <SeverityGroup
                      key={level}
                      severity={level}
                      issues={visible.filter(
                        (issue) => severityOf(issue) === level,
                      )}
                      projectId={projectId}
                      sessionId={sessionId}
                    />
                  ))}
                </div>
              )}
            </>
          )}
        </>
      )}
    </div>
  );
}
