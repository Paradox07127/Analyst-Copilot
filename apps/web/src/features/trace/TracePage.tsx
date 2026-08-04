/* Trace & cost slice (§10.1 Trace): what the session spent and what it did. Cost
 * cards and stage bars come from the SessionMetrics rollup; the event feed is a
 * cursor-paginated read over the trace_events table. */

import { useMemo } from "react";
import { Link, useParams } from "react-router";
import {
  ApiError,
  type LlmDebugRecord,
  type ReportQualitySummary,
  type SessionDebugSummary,
  type SessionMetricsView,
  type TraceEventRow,
} from "../../api/client";
import {
  useDownloadDebugLog,
  useLlmDebugCalls,
  useSessionDebug,
  useSessionMetrics,
  useSettings,
  useTraceEvents,
} from "../../api/hooks";
import { ErrorState, LoadingSkeleton } from "../../components/async-states";
import { SessionQualitySummary } from "../../components/session-quality-summary";
import {
  Hint,
  Marquee,
  MetricStrip,
  MetricTile,
  SectionHeader,
} from "../../components/ui";
import { JOB_PHASES } from "../../api/job-events";
import {
  parseCsvParam,
  serializeCsvParam,
  useRouteSearchParam,
} from "../../app/route-state";

function formatCost(value: number | null | undefined): string {
  if (value === null || value === undefined) return "n/a";
  return `$${value.toFixed(4)}`;
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${Math.round(seconds - minutes * 60)}s`;
}

function CostCards({ metrics }: { metrics: SessionMetricsView }) {
  const cacheHit = `${((metrics.cache_hit_rate ?? 0) * 100).toFixed(1)}%`;
  const calls = metrics.llm_calls ?? 0;
  const priced = calls > 0 ? `${metrics.costed_calls ?? 0}/${calls} calls priced` : "no model calls";
  const reported =
    calls > 0
      ? `${metrics.usage_known_calls ?? 0}/${calls} calls reported usage`
      : "no model calls";
  return (
    <MetricStrip>
      <MetricTile
        label="Estimated cost"
        value={formatCost(metrics.est_cost_usd)}
        hint={`${metrics.cost_estimate_status ?? "not_applicable"} · ${priced}`}
        title="Derived from provider-reported tokens and configured or built-in rates. This is an estimate, not an invoice: contract discounts, service tiers, tools and storage are not included."
      />
      <MetricTile
        label="LLM calls"
        value={String(calls)}
        hint={`${metrics.tool_calls ?? 0} tool call(s)`}
      />
      <MetricTile
        label="Total tokens"
        value={(metrics.total_tokens ?? 0).toLocaleString()}
        hint={`${(metrics.prompt_tokens ?? 0).toLocaleString()} in · ${(metrics.completion_tokens ?? 0).toLocaleString()} out`}
        title={reported}
      />
      <MetricTile
        label="Prompt cache hit"
        value={cacheHit}
        hint={`${(metrics.cached_tokens ?? 0).toLocaleString()} read · ${(metrics.cache_creation_tokens ?? 0).toLocaleString()} written`}
      />
      <MetricTile
        label="Reasoning tokens"
        value={(metrics.reasoning_tokens ?? 0).toLocaleString()}
        hint="already inside output tokens"
      />
      <MetricTile
        label="Duration"
        value={formatDuration(metrics.duration_seconds ?? 0)}
        hint={`${metrics.event_count ?? 0} trace event(s)`}
      />
      <MetricTile
        label="Findings recorded"
        value={String(metrics.findings_count ?? 0)}
        hint="not the Findings page count"
        title="Counts validated findings plus findings listed inside each question's execution result. The Findings page only counts validated findings, so the two totals can differ."
      />
    </MetricStrip>
  );
}

/* The task cut is keyed by the internal task id the agent tags each call with
 * (`m2_report_claim_plan`, `di8_semantic_bootstrap`). The prefixes are
 * milestone codes with no meaning outside this repository, so the panel that
 * answers "what did the run spend its calls on" was unreadable. Unknown ids
 * fall through to the raw name — a new task shows up honestly rather than
 * silently mislabelled. */
const TASK_LABELS: Record<string, string> = {
  di4_l1_interpretation: "Interpreting results",
  di8_semantic_bootstrap: "Learning column meanings",
  m2_report_claim_plan: "Planning report claims",
  m3_build_plan: "Planning the analysis",
  m4_question_discovery: "Proposing questions",
  session_title: "Naming the session",
};

function taskLabel(name: string): string {
  return TASK_LABELS[name] ?? name;
}

/* Four cuts of the same call set. Which model actually served a request and
 * which transport carried it are the two that most often explain a cost
 * surprise, so they lead. */
function LlmBreakdowns({ metrics }: { metrics: SessionMetricsView }) {
  const groups = [
    ["Model", metrics.llm_calls_by_model, false],
    ["Transport", metrics.llm_calls_by_kind, false],
    ["Status", metrics.llm_calls_by_status, false],
    ["What the calls were for", metrics.llm_calls_by_task, true],
  ] as const;
  if (groups.every(([, values]) => Object.keys(values ?? {}).length === 0)) {
    return null;
  }
  return (
    <section className="flex flex-col gap-2">
      <SectionHeader title="LLM request breakdown" level={3} />
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        {groups.map(([label, values, humanize]) => (
          <div key={label} className="rounded-base border border-border p-3">
            <h4 className="mb-2 text-xs font-medium text-status-neutral">{label}</h4>
            <dl className="flex flex-col gap-1 text-sm">
              {Object.entries(values ?? {})
                .sort((a, b) => b[1] - a[1])
                .map(([name, count]) => (
                  <div key={name} className="flex justify-between gap-3">
                    <dt
                      className={`min-w-0 text-xs ${humanize ? "" : "font-mono"}`}
                    >
                      <Marquee title={name}>
                        {humanize ? taskLabel(name) : name}
                      </Marquee>
                    </dt>
                    <dd className="tabular">{count}</dd>
                  </div>
                ))}
            </dl>
          </div>
        ))}
      </div>
    </section>
  );
}

/* Raw trace step name -> the phase label the activity drawer shows, so the
 * same run reads the same way live and afterwards. */
const PHASE_OF_STEP = new Map<string, string>(
  JOB_PHASES.flatMap((phase) =>
    phase.steps.map((step) => [step, phase.label] as const),
  ),
);

/* Below this share of the run a bar is drawn at a fixed minimum and labelled
 * "<0.1%": measured steps span 0.016s to 137s (~8000x), so a purely
 * proportional bar renders most of the pipeline as an invisible hairline and
 * makes 20.1s indistinguishable from 0.2s. The share is printed either way, so
 * the number stays exact even where the bar cannot be. */
const NEGLIGIBLE_SHARE = 0.001;

function StageBars({ metrics }: { metrics: SessionMetricsView }) {
  const steps = [...(metrics.steps ?? [])].sort(
    (a, b) => b.duration_seconds - a.duration_seconds,
  );
  if (steps.length === 0) return null;
  const total = steps.reduce((sum, step) => sum + step.duration_seconds, 0);

  return (
    <section className="flex flex-col gap-2">
      <SectionHeader
        title="Stage duration"
        level={3}
        actions={
          <Hint label="Stage duration">
            Wall-clock time per pipeline step, longest first, with its share of
            the summed step time. Bars are proportional; steps under 0.1% are
            drawn at a fixed minimum so they stay visible.
          </Hint>
        }
      />
      <ul className="flex flex-col gap-1">
        {steps.map((step) => {
          const share = total > 0 ? step.duration_seconds / total : 0;
          const negligible = share < NEGLIGIBLE_SHARE;
          const phase = PHASE_OF_STEP.get(step.step_name);
          return (
            <li
              key={step.step_name}
              className="grid min-w-0 grid-cols-[minmax(0,1fr)_auto] items-center gap-x-3 gap-y-1 text-xs sm:grid-cols-[minmax(10rem,16rem)_minmax(8rem,1fr)_auto]"
            >
              <span className="flex min-w-0 flex-col">
                <Marquee>{phase ?? step.step_name}</Marquee>
                <Marquee className="font-mono text-[11px] text-status-neutral" title={step.step_name}>
                  {step.step_name}
                </Marquee>
              </span>
              <span className="col-span-2 h-2.5 min-w-0 rounded-base bg-track sm:col-span-1">
                <span
                  className={`block h-full rounded-base ${
                    negligible ? "bg-primary/35" : "bg-primary/70"
                  }`}
                  style={{
                    width: negligible ? "3px" : `${(share * 100).toFixed(2)}%`,
                  }}
                />
              </span>
              <span className="tabular col-start-2 row-start-1 text-right whitespace-nowrap text-status-neutral sm:col-start-3">
                {formatDuration(step.duration_seconds)}
                {" · "}
                {negligible ? "<0.1%" : `${(share * 100).toFixed(1)}%`}
                {step.tokens > 0
                  ? ` · ${step.tokens.toLocaleString()} tok`
                  : ""}
              </span>
            </li>
          );
        })}
      </ul>
    </section>
  );
}

function EventRow({ event }: { event: TraceEventRow }) {
  const summary = event.summary ?? {};
  const hasSummary = Object.keys(summary).length > 0;
  return (
    <tr className="border-t border-table-border align-top">
      <td className="px-3 py-2 font-mono text-xs whitespace-nowrap">
        {event.event_id}
      </td>
      <td className="px-3 py-2 font-mono text-xs whitespace-nowrap">
        {event.event_type}
      </td>
      <td className="px-3 py-2 whitespace-nowrap">{event.name}</td>
      <td className="px-3 py-2 text-xs whitespace-nowrap text-status-neutral">
        {event.started_at ? new Date(event.started_at).toLocaleTimeString() : ""}
      </td>
      <td className="px-3 py-2 text-xs whitespace-nowrap">
        {event.duration_seconds === null || event.duration_seconds === undefined
          ? ""
          : `${event.duration_seconds.toFixed(3)}s`}
      </td>
      <td className="px-3 py-2">
        {hasSummary ? (
          <details>
            <summary className="cursor-pointer text-xs text-status-neutral">
              {Object.keys(summary).length} field(s)
            </summary>
            <pre className="mt-1 max-w-xl overflow-x-auto rounded-base bg-code-bg p-2 text-xs">
              {JSON.stringify(summary, null, 2)}
            </pre>
          </details>
        ) : (
          <span className="text-xs text-status-neutral">—</span>
        )}
      </td>
    </tr>
  );
}

function debugCellText(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "boolean") return value ? "true" : "false";
  return String(value);
}

function DebugTable({ rows }: { rows: Record<string, unknown>[] }) {
  const columns = Object.keys(rows[0] ?? {});
  return (
    <div className="overflow-x-auto rounded-base border border-border">
      <table className="w-full text-sm">
        <thead className="bg-table-header-bg text-left">
          <tr>
            {columns.map((column) => (
              <th key={column} className="px-3 py-2 font-medium whitespace-nowrap">
                {column}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index} className="border-t border-table-border align-top">
              {columns.map((column) => (
                <td key={column} className="px-3 py-2 whitespace-nowrap">
                  {debugCellText(row[column])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DebugExpander({
  title,
  rows,
  open,
  onOpenChange,
}: {
  title: string;
  rows: Record<string, unknown>[];
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  return (
    <details
      className="rounded-base border border-border"
      open={open}
      onToggle={(event) => onOpenChange(event.currentTarget.open)}
    >
      <summary className="cursor-pointer px-3 py-2 text-sm font-medium">
        {title} ({rows.length})
      </summary>
      <div className="border-t border-border p-3">
        {rows.length > 0 ? (
          <DebugTable rows={rows} />
        ) : (
          <p className="text-sm text-status-neutral">No rows.</p>
        )}
      </div>
    </details>
  );
}

function DebugSummaryTiles({ summary }: { summary: SessionDebugSummary }) {
  return (
    <MetricStrip>
      <MetricTile label="Events" value={String(summary.events)} />
      <MetricTile label="Artifacts" value={String(summary.artifacts)} />
      <MetricTile label="LLM calls" value={String(summary.llm_calls)} />
      <MetricTile label="Tool calls" value={String(summary.tool_calls)} />
      <MetricTile label="Errors" value={String(summary.errors)} />
      <MetricTile label="Total tokens" value={summary.total_tokens.toLocaleString()} />
      <MetricTile label="Estimated cost" value={formatCost(summary.estimated_cost_usd)} />
    </MetricStrip>
  );
}

function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function ReportQualityTiles({ quality }: { quality: ReportQualitySummary }) {
  return (
    <MetricStrip>
      <MetricTile label="Rendered sections" value={formatPercent(quality.section_coverage)} />
      <MetricTile
        label="Claim sections"
        value={formatPercent(quality.claim_section_coverage)}
      />
      <MetricTile label="Claim survival" value={formatPercent(quality.claim_survival_rate)} />
      <MetricTile label="Auto repairs" value={String(quality.deterministic_repair_count)} />
      <MetricTile
        label="Prompt tokens"
        value={quality.prompt_tokens_by_attempt || "n/a"}
      />
    </MetricStrip>
  );
}

/* Header mirrors trace_ui._render_llm_debug_details: status only shows when the
 * call did not succeed, so a clean list stays scannable. */
function llmCallHeader(record: LlmDebugRecord): string {
  const tokens = `tok=${record.prompt_tokens ?? "?"}→${record.completion_tokens ?? "?"}`;
  /* A priced $0 call is not the same as an unpriced one, so test for null
   * rather than falsiness. */
  const cost =
    record.estimated_cost_usd !== null && record.estimated_cost_usd !== undefined
      ? ` · est. $${record.estimated_cost_usd}`
      : "";
  const status = record.status === "success" ? "" : ` · ${record.status || "?"}`;
  return `${record.index}. ${record.task || "?"} · ${tokens}${cost} · ${record.duration_s ?? "?"}s${status}`;
}

function LlmDebugCallItem({ record }: { record: LlmDebugRecord }) {
  return (
    <details className="rounded-base border border-border">
      <summary className="cursor-pointer px-3 py-2 text-sm">
        {llmCallHeader(record)}
      </summary>
      <div className="flex flex-col gap-2 border-t border-border p-3">
        <p className="text-xs text-status-neutral">
          {record.ts} · {record.transport_kind || record.kind || "unknown transport"} ·{" "}
          {record.provider || "unknown provider"} ·{" "}
          {record.endpoint_host || "endpoint unavailable"} · {record.model}
        </p>
        <p className="text-xs text-status-neutral">
          {record.request_id || record.response_id || "no request id"} ·{" "}
          {record.finish_reason || "no finish reason"} ·{" "}
          {record.cost_basis || "cost basis unavailable"}
          {record.pricing_version ? ` (${record.pricing_version})` : ""} ·{" "}
          {record.request_bytes ?? "?"} B in / {record.response_bytes ?? "?"} B out
        </p>
        <div>
          <p className="text-xs font-medium">Payload</p>
          <pre className="max-h-64 overflow-auto whitespace-pre rounded-base bg-code-bg p-2 font-mono text-xs">
            {record.payload_preview}
          </pre>
        </div>
        <div>
          <p className="text-xs font-medium">Response</p>
          <pre className="max-h-64 overflow-auto whitespace-pre rounded-base bg-code-bg p-2 font-mono text-xs">
            {record.response_preview || "(none)"}
          </pre>
        </div>
      </div>
    </details>
  );
}

function LlmDebugForensics({ sessionId }: { sessionId: string }) {
  const calls = useLlmDebugCalls(sessionId);
  const rows = useMemo(
    () => (calls.data?.pages ?? []).flatMap((page) => page.items ?? []),
    [calls.data],
  );
  return (
    <div className="flex flex-col gap-2">
      {calls.isPending && (
        <LoadingSkeleton lines={3} label="Loading LLM call details" />
      )}
      {calls.isError && (
        <ErrorState error={calls.error} onRetry={() => calls.refetch()} />
      )}
      {calls.data && rows.length === 0 && (
        <p className="text-sm text-status-neutral">
          No captured LLM payloads for this session.
        </p>
      )}
      {rows.map((record) => (
        <LlmDebugCallItem key={record.index} record={record} />
      ))}
      {calls.hasNextPage && (
        <button
          type="button"
          onClick={() => calls.fetchNextPage()}
          disabled={calls.isFetchingNextPage}
          className="self-start rounded-base border border-border px-2 py-1 text-sm hover:bg-surface disabled:opacity-60"
        >
          {calls.isFetchingNextPage ? "Loading…" : "Load more"}
        </button>
      )}
    </div>
  );
}

function DebugLogDownload({ sessionId }: { sessionId: string }) {
  const download = useDownloadDebugLog(sessionId);
  const notFound =
    download.isError &&
    download.error instanceof ApiError &&
    download.error.code === "debug_log_not_found";
  return (
    <div className="flex flex-col gap-2">
      <button
        type="button"
        onClick={() => download.mutate()}
        disabled={download.isPending}
        className="self-start rounded-base border border-border px-2 py-1 text-sm hover:bg-surface disabled:opacity-60"
      >
        {download.isPending ? "Downloading…" : "Download debug log (JSONL)"}
      </button>
      {notFound && (
        <p className="text-sm text-status-neutral">This session has no debug log.</p>
      )}
      {download.isError && !notFound && <ErrorState error={download.error} />}
    </div>
  );
}

function DeveloperInspectorBody({ sessionId }: { sessionId: string }) {
  const debug = useSessionDebug(sessionId);
  const [expandedParam, setExpandedParam] = useRouteSearchParam(
    "debug",
    "errors,timeline",
  );
  const expanded = new Set(
    expandedParam === "none" ? [] : parseCsvParam(expandedParam),
  );
  const setExpanded = (title: string, open: boolean) => {
    const key = title.toLowerCase().replaceAll(" ", "-");
    const next = new Set(expanded);
    if (open) next.add(key);
    else next.delete(key);
    setExpandedParam(next.size === 0 ? "none" : serializeCsvParam(next));
  };
  const expander = (title: string) => ({
    open: expanded.has(title.toLowerCase().replaceAll(" ", "-")),
    onOpenChange: (open: boolean) => setExpanded(title, open),
  });
  const pages = debug.data?.pages ?? [];
  const first = pages[0];
  type DebugRowKey =
    | "timeline"
    | "llm_calls"
    | "tool_calls"
    | "errors"
    | "artifacts";
  const rows = (key: DebugRowKey): Record<string, unknown>[] =>
    pages.flatMap(
      (page) => (page[key] ?? []) as Record<string, unknown>[],
    );
  return (
    <div className="flex flex-col gap-4">
      {debug.isPending && (
        <LoadingSkeleton lines={4} label="Loading developer inspector" />
      )}
      {debug.isError && (
        <ErrorState error={debug.error} onRetry={() => debug.refetch()} />
      )}
      {first && (
        <>
          <DebugSummaryTiles summary={first.summary} />
          <ReportQualityTiles quality={first.report_quality} />
          <div className="flex flex-col gap-2">
            <DebugExpander
              title="Timeline"
              rows={rows("timeline")}
              {...expander("Timeline")}
            />
            <DebugExpander
              title="LLM calls"
              rows={rows("llm_calls")}
              {...expander("LLM calls")}
            />
            <DebugExpander
              title="Tool calls"
              rows={rows("tool_calls")}
              {...expander("Tool calls")}
            />
            <DebugExpander
              title="Errors"
              rows={rows("errors")}
              {...expander("Errors")}
            />
            <DebugExpander
              title="Artifacts"
              rows={rows("artifacts")}
              {...expander("Artifacts")}
            />
          </div>
          {debug.hasNextPage && (
            <button
              type="button"
              onClick={() => void debug.fetchNextPage()}
              disabled={debug.isFetchingNextPage}
              className="self-start rounded-base border border-border px-2 py-1 text-sm hover:bg-surface disabled:opacity-60"
            >
              {debug.isFetchingNextPage ? "Loading…" : "Load more debug rows"}
            </button>
          )}
        </>
      )}
      <div className="flex flex-col gap-3 border-t border-border pt-4">
        <h3 className="text-sm font-semibold">LLM call details</h3>
        <LlmDebugForensics sessionId={sessionId} />
      </div>
      <div className="border-t border-border pt-4">
        <DebugLogDownload sessionId={sessionId} />
      </div>
    </div>
  );
}

/* Gated by the shared developer-mode settings field. */
function DeveloperInspector({ sessionId }: { sessionId: string }) {
  const settings = useSettings();
  const [view, setView] = useRouteSearchParam("view");
  const open = view === "developer";
  return (
    <section className="flex flex-col gap-3 border-t border-border pt-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex min-w-0 flex-col gap-1">
          <h2 className="text-sm font-semibold">Developer inspector</h2>
          <p className="max-w-content text-xs text-status-neutral">
            Raw timelines, payload previews, request metadata, and debug
            downloads for diagnosing the workbench itself.
          </p>
        </div>
        {settings.data?.dev_mode && (
          <button
            type="button"
            aria-expanded={open}
            onClick={() => setView(open ? "" : "developer")}
            className="shrink-0 rounded-base border border-border px-3 py-1.5 text-sm hover:bg-surface"
          >
            {open ? "Close developer inspector" : "Open developer inspector"}
          </button>
        )}
      </div>
      {settings.isPending && (
        <LoadingSkeleton lines={2} label="Loading developer inspector settings" />
      )}
      {settings.isError && (
        <ErrorState error={settings.error} onRetry={() => settings.refetch()} />
      )}
      {settings.data && !settings.data.dev_mode && (
        <div className="flex flex-wrap items-center gap-2 text-sm text-status-neutral">
          <span>
            Developer inspector is off. Enable it in Settings to inspect raw
            calls and payloads.
          </span>
          <Link
            to="/settings?section=about"
            className="text-primary underline-offset-2 hover:underline"
          >
            Open Settings
          </Link>
        </div>
      )}
      {settings.data?.dev_mode && open && (
        <DeveloperInspectorBody sessionId={sessionId} />
      )}
    </section>
  );
}

export function Component() {
  const { sessionId = "" } = useParams();
  const [type, setType] = useRouteSearchParam("type");
  const metrics = useSessionMetrics(sessionId);
  const events = useTraceEvents(sessionId, type || undefined);

  const rows = useMemo(
    () => (events.data?.pages ?? []).flatMap((page) => page.items ?? []),
    [events.data],
  );
  /* The histogram describes the whole run, so it stays stable while filtered. */
  const eventTypes = Object.entries(events.data?.pages[0]?.event_types ?? {});
  const total = events.data?.pages[0]?.total ?? 0;

  return (
    <div className="mx-auto flex w-[95%] max-w-data min-w-0 flex-col gap-6 p-6">
      <header className="flex flex-wrap items-center gap-3">
        <h1 className="text-xl font-semibold">Trace &amp; cost</h1>
        {metrics.data && (
          <span className="text-xs text-status-neutral">
            {metrics.data.source === "artifact"
              ? "from the session's SessionMetrics artifact"
              : "aggregated from trace events"}
            {metrics.data.trace_status !== "verified" &&
              ` · trace ${metrics.data.trace_status}`}
          </span>
        )}
      </header>

      {metrics.isPending && <LoadingSkeleton lines={3} label="Loading session metrics" />}
      {metrics.isError && (
        <ErrorState error={metrics.error} onRetry={() => metrics.refetch()} />
      )}
      {metrics.data && (
        <>
          <CostCards metrics={metrics.data} />
          <LlmBreakdowns metrics={metrics.data} />
          <SessionQualitySummary metrics={metrics.data} />
          <StageBars metrics={metrics.data} />
        </>
      )}

      <section className="flex flex-col gap-2">
        <div className="flex flex-wrap items-center gap-3">
          <h2 className="text-sm font-semibold">Events</h2>
          <label className="flex items-center gap-2 text-sm">
            <span className="text-status-neutral">Type</span>
            <select
              value={type}
              onChange={(event) => setType(event.target.value)}
              className="rounded-base border border-border bg-bg px-2 py-1 text-sm"
            >
              <option value="">All types</option>
              {eventTypes.map(([name, count]) => (
                <option key={name} value={name}>
                  {name} ({count})
                </option>
              ))}
            </select>
          </label>
          <span className="text-xs text-status-neutral">
            {rows.length} of {total} shown
          </span>
        </div>

        {events.isPending && (
          <LoadingSkeleton lines={4} label="Loading trace events" />
        )}
        {events.isError && (
          <ErrorState error={events.error} onRetry={() => events.refetch()} />
        )}
        {events.data && rows.length === 0 && (
          <p className="text-sm text-status-neutral">
            No trace events recorded for this filter.
          </p>
        )}
        {rows.length > 0 && (
          <div className="overflow-x-auto rounded-base border border-border">
            <table className="w-full text-sm">
              <thead className="bg-table-header-bg text-left">
                <tr>
                  <th className="px-3 py-2 font-medium">#</th>
                  <th className="px-3 py-2 font-medium">Type</th>
                  <th className="px-3 py-2 font-medium">Name</th>
                  <th className="px-3 py-2 font-medium">Started</th>
                  <th className="px-3 py-2 font-medium">Duration</th>
                  <th className="px-3 py-2 font-medium">Summary</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((event) => (
                  <EventRow key={event.event_id} event={event} />
                ))}
              </tbody>
            </table>
          </div>
        )}
        {events.hasNextPage && (
          <button
            type="button"
            onClick={() => events.fetchNextPage()}
            disabled={events.isFetchingNextPage}
            className="self-start rounded-base border border-border px-2 py-1 text-sm hover:bg-surface"
          >
            {events.isFetchingNextPage ? "Loading…" : "Load more events"}
          </button>
        )}
      </section>

      <DeveloperInspector sessionId={sessionId} />
    </div>
  );
}
