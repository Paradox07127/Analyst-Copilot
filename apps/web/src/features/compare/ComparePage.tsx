/* Compare slice (§10.3): the current run on the left, any other run of the
 * same project on the right. The right-hand run lives in `?right=`, so a
 * comparison is a shareable deep link. Deltas are coloured only where the
 * server declared a direction — more charts is neither good nor bad. */

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type FocusEvent,
  type PointerEvent,
} from "react";
import { Link, useParams, useSearchParams } from "react-router";
import {
  Panel,
  PanelGroup,
  PanelResizeHandle,
} from "react-resizable-panels";
import type {
  CompareArtifactDelta,
  CompareIntegerValue,
  CompareMetricRow,
  CompareNumberValue,
  CompareSessionSide,
  CompareStringListValue,
  CompareStringValue,
  CompareTextRow,
  SessionForkStarted,
  SessionSummary,
} from "../../api/client";
import {
  useCompare,
  useDatasets,
  useForkSession,
  useSessionDetail,
  useSessions,
} from "../../api/hooks";
import { useJobEvents } from "../../api/job-events";
import { useJobActivity } from "../../app/job-activity";
import { sessionSectionPath } from "../../app/paths";
import {
  useWorkspaceFocus,
  type WorkspacePane,
} from "../../app/workspace-focus";
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
  SectionHeader,
  type Tone,
} from "../../components/ui";
import {
  clearBaseline,
  readBaseline,
  writeBaseline,
} from "./baseline-storage";
import {
  COMPARE_SCOPES,
  SPLIT_SECTIONS,
  readCompareRouteState,
  swapCompareRouteState,
  writeCompareRouteState,
  type CompareRouteState,
  type SplitSection,
} from "./compare-route-state";

function formatNumber(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  if (Number.isInteger(value)) return value.toLocaleString();
  return value.toLocaleString(undefined, { maximumSignificantDigits: 4 });
}

function formatDelta(value: number): string {
  const shown = formatNumber(Math.abs(value));
  return value > 0 ? `+${shown}` : `−${shown}`;
}

type DisplayValue =
  | CompareNumberValue
  | CompareIntegerValue
  | CompareStringValue
  | CompareStringListValue;

const VALUE_STATE_LABEL: Record<DisplayValue["state"], string> = {
  value: "Value",
  missing: "Missing",
  unavailable: "Unavailable",
  not_applicable: "Not applicable",
};

function compareValueText(value: DisplayValue): string {
  if (value.state !== "value") return VALUE_STATE_LABEL[value.state];
  if (typeof value.value === "number") return formatNumber(value.value);
  if (Array.isArray(value.value)) return value.value.join(", ") || "—";
  return value.value || "—";
}

function CompareValueCell({
  value,
  align = "left",
}: {
  value: DisplayValue;
  align?: "left" | "right";
}) {
  if (value.state === "value") {
    return <span className="tabular">{compareValueText(value)}</span>;
  }
  return (
    <span
      title={value.reason ?? VALUE_STATE_LABEL[value.state]}
      className={`inline-flex flex-col text-xs text-status-neutral ${
        align === "right" ? "items-end" : "items-start"
      }`}
    >
      <span>{VALUE_STATE_LABEL[value.state]}</span>
      {value.reason && <span className="sr-only">: {value.reason}</span>}
    </span>
  );
}

/* Every table in this page has a left column and a right column, and until now
 * they were headed "Left" and "Right" — positions, not runs. Both sides carry a
 * human title, so the tables are headed with it and fall back to the id. */
function runLabel(side: CompareSessionSide): string {
  return side.title?.trim() || side.session_id;
}

const STATUS_TONE: Record<string, Tone> = {
  complete: "ok",
  completed: "ok",
  failed: "critical",
  error: "critical",
  running: "info",
  queued: "neutral",
};

function SessionHeader({ side, label }: { side: CompareSessionSide; label: string }) {
  const title = runLabel(side);
  const tone = STATUS_TONE[side.status] ?? "neutral";
  return (
    <div className="flex min-w-0 flex-col gap-0.5">
      <span className="text-[10px] font-semibold tracking-wide text-status-neutral uppercase">
        {label}
      </span>
      <Marquee className="text-sm font-medium" title={title}>
        {title}
      </Marquee>
      <Marquee className="font-mono text-xs text-status-neutral" title={side.session_id}>
        {side.session_id}
      </Marquee>
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 pt-1">
        <Badge tone={tone}>
          <Dot tone={tone} />
          {side.status}
        </Badge>
        {side.created_at && (
          <span className="tabular text-xs text-status-neutral">
            started {new Date(side.created_at).toLocaleDateString()}
          </span>
        )}
      </div>
    </div>
  );
}

/** Colour alone cannot carry the verdict, so a declared direction also emits
 *  the word a screen reader needs. */
function DeltaCell({ row }: { row: CompareMetricRow }) {
  if (row.delta === null || row.delta === undefined) {
    return (
      <span className="text-status-neutral">
        —
        {row.verdict === "unknown" && (
          <span className="sr-only"> unavailable</span>
        )}
      </span>
    );
  }
  const directional = row.verdict === "improved" || row.verdict === "regressed";
  const tone = !directional
    ? "text-status-neutral"
    : row.verdict === "improved"
      ? "text-status-ok"
      : "text-status-critical";
  return (
    <>
      <span className={`tabular ${tone}`}>{formatDelta(row.delta)}</span>
      {directional && (
        <span className="sr-only">
          {row.verdict === "improved" ? " improvement" : " regression"}
        </span>
      )}
    </>
  );
}

function SideHeading({ label }: { label: string }) {
  return (
    <Marquee className="ml-auto block max-w-48" title={label}>
      {label}
    </Marquee>
  );
}

function MetricTable({
  rows,
  leftLabel,
  rightLabel,
}: {
  rows: CompareMetricRow[];
  leftLabel: string;
  rightLabel: string;
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[36rem] border-collapse text-sm">
        <caption className="sr-only">Headline metrics side by side</caption>
        <thead>
          <tr className="border-b border-border text-left text-xs text-status-neutral">
            <th scope="col" className="py-1.5 pr-3 font-medium">
              Metric
            </th>
            <th scope="col" className="py-1.5 pr-3 text-right font-medium">
              <SideHeading label={leftLabel} />
            </th>
            <th scope="col" className="py-1.5 pr-3 text-right font-medium">
              <SideHeading label={rightLabel} />
            </th>
            <th scope="col" className="py-1.5 text-right font-medium">
              Change
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.key} className="border-b border-hairline">
              <th scope="row" className="py-1.5 pr-3 text-left font-normal">
                {row.label}
              </th>
              <td className="tabular py-1.5 pr-3 text-right">
                <CompareValueCell value={row.left} align="right" />
              </td>
              <td className="tabular py-1.5 pr-3 text-right">
                <CompareValueCell value={row.right} align="right" />
              </td>
              <td className="py-1.5 text-right">
                <DeltaCell row={row} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function TextTable({
  rows,
  leftLabel,
  rightLabel,
}: {
  rows: CompareTextRow[];
  leftLabel: string;
  rightLabel: string;
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[36rem] border-collapse text-sm">
        <caption className="sr-only">Session properties side by side</caption>
        <thead>
          <tr className="border-b border-border text-left text-xs text-status-neutral">
            <th scope="col" className="py-1.5 pr-3 font-medium">
              Property
            </th>
            <th scope="col" className="py-1.5 pr-3 font-medium">
              <Marquee className="block max-w-48" title={leftLabel}>
                {leftLabel}
              </Marquee>
            </th>
            <th scope="col" className="py-1.5 pr-3 font-medium">
              <Marquee className="block max-w-48" title={rightLabel}>
                {rightLabel}
              </Marquee>
            </th>
            <th scope="col" className="py-1.5 text-right font-medium">
              Change
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.key} className="border-b border-hairline">
              <th scope="row" className="py-1.5 pr-3 text-left font-normal">
                {row.label}
              </th>
              <td className="py-1.5 pr-3">
                <CompareValueCell value={row.left} />
              </td>
              <td className="py-1.5 pr-3">
                <CompareValueCell value={row.right} />
              </td>
              <td className="py-1.5 text-right">
                {row.changed === true ? (
                  <Badge tone="info">changed</Badge>
                ) : row.changed === false ? (
                  <span className="text-xs text-status-neutral">same</span>
                ) : (
                  <span className="text-xs text-status-neutral">unavailable</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ArtifactDeltas({
  rows,
  leftLabel,
  rightLabel,
}: {
  rows: CompareArtifactDelta[];
  leftLabel: string;
  rightLabel: string;
}) {
  const changed = rows.filter(
    (row) =>
      row.left.state !== "value" ||
      row.right.state !== "value" ||
      (row.delta !== null && row.delta !== undefined),
  );
  if (changed.length === 0) {
    return (
      <p className="text-sm text-status-neutral">
        Both sessions produced the same artifact mix.
      </p>
    );
  }
  return (
    <div className="flex flex-col gap-2">
      <p className="text-xs text-status-neutral">
        Counts read {leftLabel} → {rightLabel}. Unchanged types are omitted.
      </p>
      <ul className="flex flex-wrap gap-2">
        {changed.map((row) => (
          <li
            key={row.type}
            className="rounded-base border border-border px-2 py-1 text-xs"
          >
            <span className="font-medium">{row.type}</span>{" "}
            <span
              className="tabular text-status-neutral"
              title={[row.left.reason, row.right.reason].filter(Boolean).join(" · ")}
            >
              {compareValueText(row.left)} → {compareValueText(row.right)}
            </span>{" "}
            {/* Neutral on purpose: the server declares no direction for artifact
             * counts, and more (or fewer) artifacts is neither good nor bad. */}
            <span className="tabular text-status-neutral">
              {row.delta === null || row.delta === undefined
                ? "—"
                : formatDelta(row.delta)}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function DatasetList({ title, names }: { title: string; names: string[] }) {
  return (
    <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
      <span className="text-xs font-medium text-status-neutral">{title}</span>
      {names.length === 0 ? (
        <span className="text-sm text-status-neutral">—</span>
      ) : (
        <span className="text-sm">{names.join(", ")}</span>
      )}
    </div>
  );
}

function SessionPicker({
  label,
  runs,
  value,
  unlistedLabel,
  onChange,
}: {
  label: string;
  runs: SessionSummary[];
  value: string;
  unlistedLabel: string | null;
  onChange: (sessionId: string) => void;
}) {
  /* A ?right= from a deep link — or a run we just forked — can name a run this
   * list does not carry. Without an option for it the select renders blank,
   * which reads as "nothing selected"; the comparison below resolves its title
   * once it loads, so the option can read like every other one. */
  const unlisted = value && !runs.some((run) => run.session_id === value);
  const unlistedText = [unlistedLabel, value, "not in this list"]
    .filter(Boolean)
    .join(" · ");
  return (
    <label className="flex items-center gap-2 text-sm">
      {label}
      <select
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="rounded-base border border-border bg-surface px-2 py-1 text-sm"
      >
        <option value="">Select a session…</option>
        {unlisted && <option value={value}>{unlistedText}</option>}
        {runs.map((run) => (
          <option key={run.session_id} value={run.session_id}>
            {run.title ? `${run.title} · ${run.session_id}` : run.session_id}
          </option>
        ))}
      </select>
    </label>
  );
}

function Comparison({ compare }: { compare: ReturnType<typeof useCompare> }) {
  if (compare.isPending) {
    return <LoadingSkeleton lines={5} label="Loading comparison" />;
  }
  if (compare.isError) {
    return (
      <ErrorState error={compare.error} onRetry={() => compare.refetch()} />
    );
  }
  const data = compare.data;
  const leftLabel = runLabel(data.left);
  const rightLabel = runLabel(data.right);

  return (
    <div className="flex flex-col gap-5">
      <Card
        as="section"
        tone="quiet"
        aria-label="Compared sessions"
      className="grid grid-cols-1 gap-4 p-3 sm:grid-cols-2"
    >
        <SessionHeader side={data.left} label="Baseline" />
        <SessionHeader side={data.right} label="Variant" />
      </Card>

      <Card as="section" tone="quiet" className="flex flex-col gap-3 p-3">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs font-semibold uppercase text-status-neutral">
            Comparability
          </span>
          <Badge
            tone={
              data.comparability.verdict === "controlled"
                ? "ok"
                : data.comparability.verdict === "not_directly_comparable"
                  ? "warn"
                  : "neutral"
            }
          >
            {data.comparability.verdict.replaceAll("_", " ")}
          </Badge>
          {(data.comparability.changed_dimensions ?? []).length > 0 && (
            <span className="text-xs text-status-neutral">
              Changed: {(data.comparability.changed_dimensions ?? []).join(", ")}
            </span>
          )}
          {(data.comparability.unknown_dimensions ?? []).length > 0 && (
            <span className="text-xs text-status-neutral">
              Unknown: {(data.comparability.unknown_dimensions ?? []).join(", ")}
            </span>
          )}
        </div>
        <Disclosure
          summary="Session lineage"
          meta={data.lineage.relation.replaceAll("_", " ")}
        >
          <div className="grid gap-2 text-xs sm:grid-cols-2">
            <p>
              <span className="font-medium">Baseline path:</span>{" "}
              {(data.lineage.left_path ?? []).join(" ← ") || "Unknown"}
            </p>
            <p>
              <span className="font-medium">Variant path:</span>{" "}
              {(data.lineage.right_path ?? []).join(" ← ") || "Unknown"}
            </p>
          </div>
          {data.lineage.common_ancestor_session_id && (
            <p className="mt-2 text-xs text-status-neutral">
              Common ancestor:{" "}
              <span className="font-mono">
                {data.lineage.common_ancestor_session_id}
              </span>
            </p>
          )}
          {(data.lineage.warnings ?? []).length > 0 && (
            <ul className="mt-2 list-inside list-disc text-xs text-status-warn">
              {(data.lineage.warnings ?? []).map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          )}
        </Disclosure>
      </Card>

      <section className="flex flex-col gap-2">
        <SectionHeader
          level={2}
          title="Headline metrics"
          description="Coloured only where the server declared which direction is better."
        />
        <MetricTable
          rows={data.metrics ?? []}
          leftLabel={leftLabel}
          rightLabel={rightLabel}
        />
      </section>

      <section className="flex flex-col gap-2">
        <SectionHeader level={2} title="Session properties" />
        <TextTable
          rows={data.text_rows ?? []}
          leftLabel={leftLabel}
          rightLabel={rightLabel}
        />
      </section>

      <section className="flex flex-col gap-2">
        <SectionHeader level={2} title="Artifact differences" />
        <ArtifactDeltas
          rows={data.artifact_deltas ?? []}
          leftLabel={leftLabel}
          rightLabel={rightLabel}
        />
      </section>

      <section className="flex flex-col gap-2">
        <SectionHeader level={2} title="Datasets" />
        {data.datasets.left.state === "value" &&
        data.datasets.right.state === "value" ? (
          <Card tone="quiet" className="flex flex-col gap-1.5 px-3 py-2">
            <DatasetList title="In both sessions" names={data.datasets.shared ?? []} />
            <DatasetList
              title={`Only in ${leftLabel}`}
              names={data.datasets.only_left ?? []}
            />
            <DatasetList
              title={`Only in ${rightLabel}`}
              names={data.datasets.only_right ?? []}
            />
          </Card>
        ) : (
          <Card tone="quiet" className="grid gap-2 px-3 py-2 sm:grid-cols-2">
            <div>
              <p className="text-xs font-medium">{leftLabel}</p>
              <CompareValueCell value={data.datasets.left} />
            </div>
            <div>
              <p className="text-xs font-medium">{rightLabel}</p>
              <CompareValueCell value={data.datasets.right} />
            </div>
          </Card>
        )}
      </section>
    </div>
  );
}

/* Variant creation re-runs this session with exactly one decision varied. The forked session
 * mints its own id inside the driver, so it only becomes known when the job
 * emits `session.forked` — this panel follows the job's own stream to catch it and
 * drops the new run straight into the right-hand side of the comparison. */
function ForkPanel({
  projectId,
  sessionId,
  onForked,
}: {
  projectId: string;
  sessionId: string;
  onForked: (forkedRunId: string) => void;
}) {
  const fork = useForkSession(sessionId);
  const datasets = useDatasets(sessionId);
  const { startTracking } = useJobActivity();
  const [decision, setDecision] = useState<"ml_target" | "dataset">("ml_target");
  const [mlTarget, setMlTarget] = useState("");
  const [selectedDatasets, setSelectedDatasets] = useState<string[]>([]);
  const [started, setStarted] = useState<SessionForkStarted | null>(null);
  const keyRef = useRef<string | null>(null);

  const events = useJobEvents(
    started ? started.job.job_id : null,
    started ? started.job.events_url : null,
  );

  /* The forked id arrives once, in the summary of a `session.forked` frame. */
  const forkedRunId = events.events.reduce<string | null>((found, event) => {
    if (event.type !== "session.forked") return found;
    const id = event.summary.forked_session_id;
    return typeof id === "string" && id ? id : found;
  }, null);

  useEffect(() => {
    if (forkedRunId) onForked(forkedRunId);
  }, [forkedRunId, onForked]);

  const handles = datasets.data ?? [];
  const columnNames = Array.from(
    new Set(handles.flatMap((handle) => (handle.schema ?? []).map((c) => c.name))),
  );

  const start = () => {
    keyRef.current ??= crypto.randomUUID();
    fork.mutate(
      {
        body: {
          decision,
          ml_target_column: decision === "ml_target" ? mlTarget || null : null,
          datasets: decision === "dataset" ? selectedDatasets : [],
          llm: "env",
        },
        idempotencyKey: keyRef.current,
      },
      {
        onSuccess: (result) => {
          keyRef.current = null;
          setStarted(result);
          startTracking({
            jobId: result.job.job_id,
            sessionId: result.job.session_id,
            sourceSessionId: sessionId,
            projectId,
            eventsUrl: result.job.events_url,
          });
        },
        onError: () => {
          keyRef.current = null;
        },
      },
    );
  };

  const ready =
    decision === "ml_target" ? true : selectedDatasets.length > 0;

  return (
    <Card className="px-3 py-2">
      <Disclosure
        summary="Create a variant"
        meta="re-run with one decision changed"
      >
        <p className="text-xs text-status-neutral">
          Re-runs the whole analysis with exactly one decision changed and
          everything else held fixed. The new run appears on the right-hand side
          as soon as it exists; this session is left untouched.
        </p>

        <div className="mt-3 flex flex-col gap-3">
          <label className="flex items-center gap-2 text-sm">
            What to vary
            <select
              value={decision}
              onChange={(event) =>
                setDecision(event.target.value as "ml_target" | "dataset")
              }
              className="rounded-base border border-border bg-surface px-2 py-1 text-sm"
            >
              <option value="ml_target">Prediction target column</option>
              <option value="dataset">Input tables</option>
            </select>
          </label>

          {decision === "ml_target" ? (
            <label className="flex items-center gap-2 text-sm">
              Target column
              <select
                value={mlTarget}
                onChange={(event) => setMlTarget(event.target.value)}
                className="rounded-base border border-border bg-surface px-2 py-1 text-sm"
              >
                <option value="">No prediction baseline</option>
                {columnNames.map((column) => (
                  <option key={column} value={column}>
                    {column}
                  </option>
                ))}
              </select>
            </label>
          ) : (
            <fieldset className="flex flex-col gap-1 text-sm">
              <legend className="text-xs text-status-neutral">
                Tables to re-run on
              </legend>
              {handles.map((handle) => (
                <label key={handle.dataset_id} className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={selectedDatasets.includes(handle.dataset_id)}
                    onChange={(event) =>
                      setSelectedDatasets((current) =>
                        event.target.checked
                          ? [...current, handle.dataset_id]
                          : current.filter((id) => id !== handle.dataset_id),
                      )
                    }
                  />
                  {handle.display_name}
                </label>
              ))}
              {handles.length === 0 && (
                <span className="text-xs text-status-neutral">
                  This session has no listed tables to fork onto.
                </span>
              )}
            </fieldset>
          )}

          <div className="flex flex-wrap items-center gap-3">
            <button
              type="button"
              onClick={start}
              disabled={!ready || fork.isPending || Boolean(started)}
              className="self-start rounded-base bg-primary px-3 py-1.5 text-sm font-medium text-bg hover:opacity-90 disabled:opacity-50"
            >
              {fork.isPending ? "Starting…" : "Run variant"}
            </button>
            {started && (
              <span className="text-xs text-status-neutral">
                {forkedRunId
                  ? `Variant run ${forkedRunId} is ready to compare.`
                  : `${started.decision} · running, follow progress in the activity drawer.`}
              </span>
            )}
          </div>

          {fork.isError && (
            <p role="alert" className="text-xs text-status-critical">
              {fork.error instanceof Error
                ? fork.error.message
                : "Could not start the fork."}
            </p>
          )}
        </div>
      </Disclosure>
    </Card>
  );
}

const SECTION_LABELS: Record<SplitSection, string> = {
  overview: "Overview",
  questions: "Questions",
  "deep-analysis": "Deep analysis",
  findings: "Findings",
  report: "Report",
  artifacts: "Artifacts",
  trace: "Trace & cost",
  chat: "Chat",
};

function SplitPane({
  pane,
  projectId,
  sessionId,
  section,
  active,
  onFocus,
  onSectionChange,
  paneRef,
}: {
  pane: WorkspacePane;
  projectId: string;
  sessionId: string;
  section: SplitSection;
  active: boolean;
  onFocus: () => void;
  onSectionChange: (section: SplitSection) => void;
  paneRef: React.RefObject<HTMLElement | null>;
}) {
  const run = useSessionDetail(sessionId);
  const focus = (
    event: PointerEvent<HTMLElement> | FocusEvent<HTMLElement>,
  ) => {
    if (event.currentTarget.contains(event.target as Node)) onFocus();
  };

  return (
    <section
      ref={paneRef}
      tabIndex={-1}
      aria-label={`${pane === "left" ? "Left" : "Right"} session pane`}
      onPointerDown={focus}
      onFocusCapture={focus}
      className={`flex h-full min-h-0 min-w-0 flex-col border-2 ${
        active ? "border-primary" : "border-transparent"
      }`}
    >
      <header className="flex flex-wrap items-center gap-2 border-b border-border bg-surface px-3 py-2">
        <span
          className={`rounded-base border px-1.5 py-0.5 text-[10px] font-semibold uppercase ${
            active
              ? "border-primary text-primary"
              : "border-border text-status-neutral"
          }`}
        >
          {active ? "Active" : pane}
        </span>
        <label className="ml-auto flex items-center gap-1.5 text-xs text-status-neutral">
          Section
          <select
            aria-label={`${pane === "left" ? "Left" : "Right"} section`}
            value={section}
            onChange={(event) =>
              onSectionChange(event.target.value as SplitSection)
            }
            className="rounded-base border border-border bg-bg px-2 py-1 text-sm text-text"
          >
            {SPLIT_SECTIONS.map((value) => (
              <option key={value} value={value}>
                {SECTION_LABELS[value]}
              </option>
            ))}
          </select>
        </label>
      </header>

      <div className="min-h-0 flex-1 overflow-auto p-4">
        {!sessionId ? (
          <EmptyState
            title="Choose a session"
            description={`Select the ${pane} session above to use this pane.`}
          />
        ) : run.isPending ? (
          <LoadingSkeleton
            lines={4}
            label={`Loading ${pane} session workspace`}
          />
        ) : run.isError ? (
          <ErrorState error={run.error} onRetry={() => run.refetch()} />
        ) : (
          <div className="flex flex-col gap-4">
            <div>
              <p className="text-xs font-semibold tracking-wide text-status-neutral uppercase">
                {SECTION_LABELS[section]}
              </p>
              <h2 className="mt-1 text-lg font-semibold">
                {run.data.title?.trim() || run.data.session_id}
              </h2>
              <p className="mt-1 font-mono text-xs text-status-neutral">
                {run.data.session_id}
              </p>
            </div>
            <Card tone="quiet" className="grid grid-cols-2 gap-3 p-3 text-sm">
              <div>
                <p className="text-xs text-status-neutral">Status</p>
                <p className="font-medium">{run.data.status}</p>
              </div>
              <div>
                <p className="text-xs text-status-neutral">Latest report</p>
                <p className="font-medium">
                  {run.data.report_status ??
                    (run.data.status === "failed" ? "Unavailable" : "Not generated")}
                </p>
              </div>
              <div>
                <p className="text-xs text-status-neutral">Artifacts</p>
                <p className="font-medium">
                  {run.data.artifact_count ?? "Unavailable"}
                </p>
              </div>
              <div>
                <p className="text-xs text-status-neutral">Datasets</p>
                <p className="font-medium">
                  {(run.data.dataset_names ?? []).length}
                </p>
              </div>
            </Card>
            <p className="text-sm text-status-neutral">
              This pane keeps its own section and scroll state. Open the full
              session page for editing and detailed controls.
            </p>
            <Link
              to={sessionSectionPath(
                projectId,
                sessionId,
                section === "overview" ? "data-map" : section,
              )}
              className="self-start rounded-base border border-border px-3 py-1.5 text-sm font-medium hover:border-primary hover:text-primary"
            >
              Open full {SECTION_LABELS[section]}
            </Link>
          </div>
        )}
      </div>
    </section>
  );
}

function SplitWorkspace({
  state,
  projectId,
  activePane,
  onFocusPane,
  onSectionChange,
}: {
  state: CompareRouteState;
  projectId: string;
  activePane: WorkspacePane;
  onFocusPane: (pane: WorkspacePane) => void;
  onSectionChange: (pane: WorkspacePane, section: SplitSection) => void;
}) {
  const leftRef = useRef<HTMLElement>(null);
  const rightRef = useRef<HTMLElement>(null);
  const [narrow, setNarrow] = useState(
    () => window.matchMedia("(max-width: 767px)").matches,
  );

  useEffect(() => {
    const query = window.matchMedia("(max-width: 767px)");
    const update = () => setNarrow(query.matches);
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, []);

  const focusPane = (pane: WorkspacePane) => {
    onFocusPane(pane);
    (pane === "left" ? leftRef.current : rightRef.current)?.focus();
  };

  const pane = (side: WorkspacePane) => (
    <SplitPane
      pane={side}
      projectId={projectId}
      sessionId={side === "left" ? state.left : state.right}
      section={side === "left" ? state.leftSection : state.rightSection}
      active={activePane === side}
      onFocus={() => onFocusPane(side)}
      onSectionChange={(section) => onSectionChange(side, section)}
      paneRef={side === "left" ? leftRef : rightRef}
    />
  );

  return (
    <div className="flex min-h-[36rem] flex-col gap-2">
      <div className="flex justify-end gap-2">
        <button
          type="button"
          onClick={() => focusPane("left")}
          className="rounded-base border border-border px-2 py-1 text-xs hover:border-primary"
        >
          Focus left pane
        </button>
        <button
          type="button"
          onClick={() => focusPane("right")}
          className="rounded-base border border-border px-2 py-1 text-xs hover:border-primary"
        >
          Focus right pane
        </button>
      </div>

      {!narrow ? (
      <div className="min-h-0 flex-1">
        <PanelGroup direction="horizontal" className="min-h-0">
          <Panel defaultSize={50} minSize={25}>
            {pane("left")}
          </Panel>
          <PanelResizeHandle
            aria-label="Resize split panes"
            className="w-1 bg-border transition-colors hover:bg-primary"
          />
          <Panel defaultSize={50} minSize={25}>
            {pane("right")}
          </Panel>
        </PanelGroup>
      </div>
      ) : (
      <div className="min-h-0 flex-1">
        {pane(activePane)}
      </div>
      )}
    </div>
  );
}

export function Component() {
  const { projectId = "" } = useParams();
  const [searchParams, setSearchParams] = useSearchParams();
  const state = useMemo(
    () => readCompareRouteState(searchParams),
    [searchParams],
  );
  const runs = useSessions(projectId);
  const compare = useCompare(
    state.mode === "compare" ? state.left : "",
    state.mode === "compare" ? state.right : "",
  );
  const workspace = useWorkspaceFocus();
  const [pinnedBaseline, setPinnedBaseline] = useState(() =>
    readBaseline(projectId),
  );
  const [baselineNotice, setBaselineNotice] = useState("");

  const updateState = useCallback(
    (
      update:
        | Partial<CompareRouteState>
        | ((current: CompareRouteState) => CompareRouteState),
      replace = false,
    ) => {
      setSearchParams(
        (currentParams) => {
          const current = readCompareRouteState(currentParams);
          const next =
            typeof update === "function"
              ? update(current)
              : { ...current, ...update };
          return writeCompareRouteState(next, currentParams);
        },
        { replace },
      );
    },
    [setSearchParams],
  );

  const allRuns = useMemo(
    () => (runs.data?.pages ?? []).flatMap((page) => page.items),
    [runs.data],
  );

  useEffect(() => {
    setPinnedBaseline(readBaseline(projectId));
    setBaselineNotice("");
  }, [projectId]);

  useEffect(() => {
    if (state.left || !runs.isSuccess || allRuns.length === 0) return;
    const pinnedVisible = allRuns.some(
      (run) => run.session_id === pinnedBaseline,
    );
    if (pinnedBaseline && (pinnedVisible || runs.hasNextPage)) {
      updateState(
        pinnedBaseline === state.right
          ? { left: pinnedBaseline, right: "" }
          : { left: pinnedBaseline },
        true,
      );
      return;
    }
    if (pinnedBaseline) {
      clearBaseline(projectId);
      setPinnedBaseline("");
      setBaselineNotice(
        "The pinned baseline is no longer available in this project.",
      );
    }
    /* Backfill has to honour the same "two different sessions" rule the manual
     * selectors enforce (decision 0.6). Without this, clearing the baseline on
     * a pair whose variant is the newest run refills left with that same run
     * and the request comes back as compare_same_session. */
    const fallback = allRuns.find((run) => run.session_id !== state.right);
    updateState({ left: fallback?.session_id ?? "" }, true);
  }, [
    allRuns,
    pinnedBaseline,
    projectId,
    runs.hasNextPage,
    runs.isSuccess,
    state.left,
    state.right,
    updateState,
  ]);

  const changeSection = useCallback(
    (pane: WorkspacePane, section: string) => {
      if (!SPLIT_SECTIONS.includes(section as SplitSection)) return;
      updateState(
        pane === "left"
          ? { leftSection: section as SplitSection }
          : { rightSection: section as SplitSection },
      );
    },
    [updateState],
  );

  const leftContext = useMemo(
    () => ({
      projectId,
      sessionId: state.left,
      section: state.mode === "split" ? state.leftSection : state.scope,
      onSectionChange: (section: string) => changeSection("left", section),
    }),
    [
      changeSection,
      projectId,
      state.left,
      state.leftSection,
      state.mode,
      state.scope,
    ],
  );
  const rightContext = useMemo(
    () => ({
      projectId,
      sessionId: state.right,
      section: state.mode === "split" ? state.rightSection : state.scope,
      onSectionChange: (section: string) => changeSection("right", section),
    }),
    [
      changeSection,
      projectId,
      state.mode,
      state.right,
      state.rightSection,
      state.scope,
    ],
  );

  useEffect(() => {
    workspace.configure(state.mode, leftContext, rightContext);
  }, [leftContext, rightContext, state.mode, workspace.configure]);
  useEffect(() => () => workspace.reset(), [workspace.reset]);

  const selectLeft = useCallback(
    (left: string) => {
      updateState((current) => ({
        ...current,
        left,
        right: left && left === current.right ? "" : current.right,
      }));
    },
    [updateState],
  );
  const selectRight = useCallback(
    (right: string) => {
      updateState((current) => ({
        ...current,
        right,
        left: right && right === current.left ? "" : current.left,
      }));
    },
    [updateState],
  );

  const togglePin = () => {
    if (pinnedBaseline === state.left) {
      clearBaseline(projectId);
      setPinnedBaseline("");
      return;
    }
    writeBaseline(projectId, state.left);
    setPinnedBaseline(state.left);
  };

  const swap = () => {
    workspace.focusPane(workspace.activePane === "left" ? "right" : "left");
    updateState(swapCompareRouteState);
  };

  const leftCandidates = allRuns.filter(
    (run) => run.session_id !== state.right,
  );
  const rightCandidates = allRuns.filter(
    (run) => run.session_id !== state.left,
  );

  return (
    <div className="mx-auto flex w-[90%] max-w-data flex-col gap-5 p-4 sm:p-6">
      <SectionHeader
        level={1}
        title="Compare"
        description="Compare generated results or place two sessions side by side. Create variant keeps the original session untouched."
      />

      {baselineNotice && (
        <p role="status" className="text-xs text-status-warn">
          {baselineNotice}
        </p>
      )}

      {runs.isError ? (
        <ErrorState error={runs.error} onRetry={() => runs.refetch()} />
      ) : (
        <Card tone="quiet" className="flex flex-col gap-3 p-3">
          <div className="grid gap-3 lg:grid-cols-2">
            <SessionPicker
              label="Baseline"
              runs={leftCandidates}
              value={state.left}
              unlistedLabel={
                compare.data?.left.session_id === state.left
                  ? (compare.data.left.title?.trim() ?? null)
                  : null
              }
              onChange={selectLeft}
            />
            <SessionPicker
              label="Compare against"
              runs={rightCandidates}
              value={state.right}
              unlistedLabel={
                compare.data?.right.session_id === state.right
                  ? (compare.data.right.title?.trim() ?? null)
                  : null
              }
              onChange={selectRight}
            />
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <div
              role="group"
              aria-label="Workspace mode"
              className="inline-flex rounded-base border border-border p-0.5"
            >
              {(["compare", "split"] as const).map((mode) => (
                <button
                  key={mode}
                  type="button"
                  aria-pressed={state.mode === mode}
                  onClick={() => updateState({ mode })}
                  className={`rounded-sm px-2.5 py-1 text-sm capitalize ${
                    state.mode === mode
                      ? "bg-primary text-bg"
                      : "text-status-neutral hover:text-text"
                  }`}
                >
                  {mode}
                </button>
              ))}
            </div>
            <button
              type="button"
              onClick={togglePin}
              disabled={!state.left}
              aria-pressed={pinnedBaseline === state.left}
              className="rounded-base border border-border px-2.5 py-1 text-sm hover:border-primary disabled:opacity-50"
            >
              {pinnedBaseline === state.left
                ? "Unpin baseline"
                : "Pin as baseline"}
            </button>
            <button
              type="button"
              onClick={swap}
              disabled={!state.left || !state.right}
              className="rounded-base border border-border px-2.5 py-1 text-sm hover:border-primary disabled:opacity-50"
            >
              Swap sides
            </button>
          </div>
        </Card>
      )}

      {state.mode === "compare" ? (
        <>
          <nav
            aria-label="Compare scopes"
            className="flex min-w-0 flex-wrap gap-1 border-b border-border pb-2"
          >
            {COMPARE_SCOPES.map((scope) => (
              <button
                key={scope}
                type="button"
                aria-current={state.scope === scope ? "page" : undefined}
                onClick={() => updateState({ scope })}
                className={`rounded-base px-2.5 py-1 text-sm capitalize ${
                  state.scope === scope
                    ? "bg-surface font-medium text-primary"
                    : "text-status-neutral hover:text-text"
                }`}
              >
                {scope}
              </button>
            ))}
            <label className="ml-auto flex items-center gap-2 text-xs text-status-neutral">
              <input
                type="checkbox"
                checked={state.filter === "differences"}
                onChange={(event) =>
                  updateState({
                    filter: event.target.checked ? "differences" : "all",
                  })
                }
              />
              Only differences
            </label>
          </nav>

          {state.left && (
            <ForkPanel
              projectId={projectId}
              sessionId={state.left}
              onForked={selectRight}
            />
          )}

          {!state.right ? (
            <EmptyState
              title="Pick a session to compare against"
              description={
                rightCandidates.length === 0
                  ? "This project has no other session yet. Start a second analysis to compare them."
                  : "Choose another session of this project; the complete workspace state stays in the URL."
              }
            />
          ) : state.scope === "overview" ? (
            <Comparison compare={compare} />
          ) : (
            <Card tone="quiet" className="p-4">
              <h2 className="text-sm font-semibold capitalize">{state.scope}</h2>
              <p className="mt-1 text-sm text-status-neutral">
                The {state.scope} comparison workspace is ready for its typed
                scope endpoint. Overview remains available while that data loads
                independently.
              </p>
            </Card>
          )}
        </>
      ) : (
        <SplitWorkspace
          state={state}
          projectId={projectId}
          activePane={workspace.activePane}
          onFocusPane={workspace.focusPane}
          onSectionChange={(pane, section) => changeSection(pane, section)}
        />
      )}
    </div>
  );
}
