import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router";
import type { QualityIssueRow } from "../../api/client";
import { useQuality } from "../../api/hooks";
import {
  EmptyState,
  ErrorState,
  LoadingSkeleton,
} from "../../components/async-states";
import {
  DataWorkspacePage,
  DatasetScopeBar,
  SegmentedControl,
} from "../../components/data-workspace";
import {
  Badge,
  Dot,
  Marquee,
  MetricStrip,
  MetricTile,
  buttonClass,
  type Tone,
} from "../../components/ui";
import { tablePath } from "../../app/paths";
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

function SeverityGroup({
  severity,
  issues,
  projectId,
  sessionId,
  defaultOpen,
}: {
  severity: Severity;
  issues: QualityIssueRow[];
  projectId: string;
  sessionId: string;
  defaultOpen: boolean;
}) {
  if (issues.length === 0) return null;
  return (
    <details
      open={defaultOpen}
      className="group overflow-hidden rounded-base border border-border bg-bg"
    >
      <summary className="flex cursor-pointer list-none flex-wrap items-center gap-x-2 gap-y-0.5 px-3 py-2.5 hover:bg-surface">
        <span className="text-status-neutral transition-transform group-open:rotate-90" aria-hidden>›</span>
        <Dot tone={SEVERITY_TONE[severity]} />
        <h3 className="text-sm font-semibold">{SEVERITY_LABEL[severity]}</h3>
        <Badge tone={SEVERITY_TONE[severity]}>{issues.length}</Badge>
        <span className="min-w-0 flex-1 text-xs text-status-neutral">
          {SEVERITY_MEANING[severity]}
        </span>
      </summary>
      <div className={`border-t ${severity === "critical" ? "border-status-critical/30" : severity === "warn" ? "border-status-warn/30" : "border-hairline"}`}>
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
      </div>
    </details>
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
  /* Missing param means all codes. `none` represents a genuine empty
   * selection; without the sentinel, clearing the last checkbox silently
   * turned every code back on. */
  const [codesParam, setCodesParam] = useRouteSearchParam("codes");
  const [issueTypesOpen, setIssueTypesOpen] = useState(false);
  const issueTypesRef = useRef<HTMLDetailsElement>(null);
  const selectedCodes =
    codesParam === "none"
      ? new Set<string>()
      : codesParam
        ? new Set(parseCsvParam(codesParam))
        : null;

  /* Filter compares dataset ids, not display names: same-named datasets keep
   * distinct entries. The name fallback covers rows from older servers. */
  const datasetOptions = useMemo(
    () => {
      const options = new Map<string, string>();
      for (const card of quality.data?.datasets ?? []) {
        options.set(card.dataset_id ?? card.dataset_name, card.dataset_name);
      }
      for (const issue of quality.data?.issues ?? []) {
        options.set(issue.dataset_id ?? issue.dataset_name, issue.dataset_name);
      }
      return Array.from(options, ([value, label]) => ({ value, label }));
    },
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

  useEffect(() => {
    if (!issueTypesOpen) return;

    const closeIfOutside = (event: PointerEvent) => {
      if (!issueTypesRef.current?.contains(event.target as Node)) {
        setIssueTypesOpen(false);
      }
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setIssueTypesOpen(false);
    };

    document.addEventListener("pointerdown", closeIfOutside);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("pointerdown", closeIfOutside);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [issueTypesOpen]);

  const toggleCode = (code: string) => {
    const base =
      selectedCodes ?? new Set(codeOptions.map((option) => option.code));
    const next = new Set(base);
    if (next.has(code)) next.delete(code);
    else next.add(code);
    const allCodes = codeOptions.map((option) => option.code);
    setCodesParam(
      next.size === 0
        ? "none"
        : allCodes.every((option) => next.has(option))
          ? ""
          : serializeCsvParam(next),
    );
  };

  const allIssues = quality.data?.issues ?? [];
  const visible = allIssues.filter(
    (issue) =>
      (!dataset || (issue.dataset_id ?? issue.dataset_name) === dataset) &&
      (!severity || severityOf(issue) === severity) &&
      (!selectedCodes || selectedCodes.has(issue.code)),
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
    <DataWorkspacePage
      title="Quality"
      description="Triage profiler flags by severity and scope, then inspect the evidence before deciding whether to clean."
    >

      {quality.isPending && <LoadingSkeleton lines={4} label="Loading quality" />}
      {quality.isError && (
        <ErrorState error={quality.error} onRetry={() => quality.refetch()} />
      )}
      {quality.data && (
        <>
          <DatasetScopeBar
            value={dataset}
            onChange={setDataset}
            options={datasetOptions}
            allLabel="All datasets"
          />
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
              tone="info"
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
              <section aria-labelledby="issue-queue-heading" className="flex min-w-0 flex-col gap-3">
                <div className="flex min-w-0 flex-wrap items-end gap-x-4 gap-y-3 border-b border-hairline pb-3">
                  <div className="flex flex-col gap-1">
                    <span className="text-sm text-status-neutral">Severity</span>
                    <SegmentedControl
                      label="Filter by severity"
                      value={severity}
                      onChange={setSeverity}
                      options={[
                        { value: "", label: "All" },
                        ...SEVERITIES.map((option) => ({
                          value: option,
                          label: SEVERITY_LABEL[option],
                        })),
                      ]}
                    />
                  </div>
                  <details
                    ref={issueTypesRef}
                    open={issueTypesOpen}
                    onToggle={(event) => setIssueTypesOpen(event.currentTarget.open)}
                    className="relative"
                  >
                    <summary className="cursor-pointer list-none rounded-base border border-border px-3 py-2 text-sm font-medium hover:bg-surface">Issue types <span className="tabular text-status-neutral">· {selectedCodes?.size ?? codeOptions.length}</span></summary>
                    <fieldset className="absolute right-0 z-30 mt-1 flex max-h-72 w-72 flex-col gap-1 overflow-y-auto rounded-base border border-border bg-surface p-2 shadow-overlay text-sm">
                      <legend className="sr-only">Code type</legend>
                      <div className="mb-1 flex items-center justify-between border-b border-hairline pb-1">
                        <span className="text-xs text-status-neutral">Issue types</span>
                        <span className="flex items-center gap-2">
                          <button type="button" onClick={() => setCodesParam("")} className="text-xs font-medium text-primary hover:underline">Select all</button>
                          <button type="button" onClick={() => setCodesParam("none")} className="text-xs font-medium text-primary hover:underline">Clear</button>
                        </span>
                      </div>
                      {codeOptions.map((option) => (
                        <label key={option.code} className="flex items-center gap-2 rounded-sm px-1 py-1 hover:bg-surface">
                          <input type="checkbox" aria-label={`${option.code} (${option.count})`} checked={selectedCodes?.has(option.code) ?? true} onChange={() => toggleCode(option.code)} />
                          <span className="min-w-0 flex-1 truncate font-mono text-xs">{option.code}</span>
                          <span className="tabular text-xs text-status-neutral">{option.count}</span>
                        </label>
                      ))}
                    </fieldset>
                  </details>
                </div>
                <div>
                  <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
                    <h2 id="issue-queue-heading" className="text-base font-semibold">Issue queue</h2>
                    <span className="tabular text-sm text-status-neutral">{visible.length} of {allIssues.length} flags</span>
                  </div>
                  <p className="text-sm text-status-neutral">Open the highest severity first; lower-priority groups stay collapsed until needed.</p>
                </div>
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
                <div className="flex flex-col gap-2">
                  {SEVERITIES.map((level) => (
                    <SeverityGroup
                      key={level}
                      severity={level}
                      issues={visible.filter(
                        (issue) => severityOf(issue) === level,
                      )}
                      projectId={projectId}
                      sessionId={sessionId}
                      defaultOpen={
                        level === "critical" ||
                        (level === "warn" && !visible.some((issue) => severityOf(issue) === "critical"))
                      }
                    />
                  ))}
                </div>
                )}
              </section>
          )}
        </>
      )}
    </DataWorkspacePage>
  );
}
