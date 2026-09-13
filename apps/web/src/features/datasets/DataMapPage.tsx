import { useEffect } from "react";
import { Link, useParams } from "react-router";
import type {
  DatasetHandle,
  QualityDatasetCard,
} from "../../api/client";
import {
  useDatasets,
  useEdaHandoff,
  usePrimaryRunFailure,
  useQuality,
  useSessionDetail,
  type PrimaryRunFailure,
} from "../../api/hooks";
import { useJobActivity } from "../../app/job-activity";
import { RunAgainButton } from "../../components/run-again";
import {
  EmptyState,
  ErrorState,
  LoadingSkeleton,
  PartialState,
} from "../../components/async-states";
import { qualityCodeLabel } from "../../api/quality-codes";
import { ResourceLimitNotice } from "../../components/resource-limit-notice";
import { DataWorkspacePage } from "../../components/data-workspace";
import {
  Badge,
  Card,
  Dot,
  Marquee,
  MetricStrip,
  MetricTile,
  SectionHeader,
  formatCompact,
} from "../../components/ui";
import { sessionSectionPath, tablePath } from "../../app/paths";
import {
  KindCompositionBar,
  countColumnKinds,
  kindCompositionText,
} from "./mini-charts";
import "./data-map.css";

/* "limited" belongs here: a run the resource gate stopped is over, and leaving
 * it out kept this page polling for tables that were never going to arrive. */
const TERMINAL_SESSION_STATUSES = new Set([
  "complete",
  "completed",
  "failed",
  "cancelled",
  "limited",
]);

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(0)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
}

/* Per-table gate published by the pipeline's EdaHandoff artifact: whether
 * downstream analysis should treat the table as ready, and which material
 * quality conditions apply if not. */
type DatasetGate = {
  analysisReady: boolean;
  materialCodes: string[];
  piiColumns: string[];
};

function handoffGates(payload: unknown): Map<string, DatasetGate> {
  const gates = new Map<string, DatasetGate>();
  if (typeof payload !== "object" || payload === null) return gates;
  const datasets = (payload as { datasets?: unknown }).datasets;
  if (!Array.isArray(datasets)) return gates;
  for (const entry of datasets) {
    if (typeof entry !== "object" || entry === null) continue;
    const record = entry as Record<string, unknown>;
    if (typeof record["dataset_id"] !== "string") continue;
    const quality =
      typeof record["quality"] === "object" && record["quality"] !== null
        ? (record["quality"] as Record<string, unknown>)
        : {};
    const codes = Array.isArray(quality["material_codes"])
      ? quality["material_codes"].filter((code): code is string => typeof code === "string")
      : [];
    const pii =
      typeof record["pii_columns"] === "object" && record["pii_columns"] !== null
        ? Object.keys(record["pii_columns"] as Record<string, unknown>)
        : [];
    gates.set(record["dataset_id"], {
      analysisReady: record["analysis_ready"] === true,
      materialCodes: codes,
      piiColumns: pii,
    });
  }
  return gates;
}

function ReadinessBadge({ gate }: { gate: DatasetGate }) {
  if (gate.analysisReady) {
    return (
      <Badge
        tone="ok"
        title="The pipeline marked this table ready for downstream analysis."
      >
        analysis-ready
      </Badge>
    );
  }
  const conditions =
    gate.materialCodes.length > 0
      ? `Open conditions: ${gate.materialCodes.map(qualityCodeLabel).join(", ")}`
      : "Critical quality flags are open.";
  return (
    <Badge
      tone="critical"
      title={`Findings on this table carry quality conditions until these are reviewed. ${conditions}`}
    >
      <Dot tone="critical" />
      limited for analysis
    </Badge>
  );
}

function isEditedDataset(dataset: DatasetHandle): boolean {
  /* Cleaning produces a derived file in this project-scoped location. Keep
   * the check path-based instead of guessing from a display name, which users
   * may edit freely. */
  const source = dataset.original_uri.replaceAll("\\", "/").replace(/^\.?\//, "");
  return source.startsWith(`projects/${dataset.project_id}/cleaned/`);
}

function SessionSummary({
  datasets,
  inputCount,
}: {
  datasets: DatasetHandle[] | undefined;
  inputCount: number;
}) {
  const rowsKnown =
    datasets != null &&
    datasets.length > 0 &&
    datasets.every((dataset) => dataset.row_count != null);
  const totalRows = rowsKnown
    ? datasets.reduce((sum, dataset) => sum + (dataset.row_count ?? 0), 0)
    : undefined;
  const totalColumns = datasets?.reduce(
    (sum, dataset) => sum + (dataset.schema?.length ?? 0),
    0,
  );
  const totalBytes = datasets?.reduce(
    (sum, dataset) => sum + (dataset.byte_size ?? 0),
    0,
  );
  const cleanedTables = datasets?.filter(isEditedDataset).length;
  const datasetCount = Math.max(
    datasets?.length ?? 0,
    inputCount,
  );
  return (
    <MetricStrip>
      <MetricTile label="Tables" value={datasetCount} />
      <MetricTile
        label="Rows"
        value={totalRows == null ? "—" : formatCompact(totalRows)}
        hint={
          totalRows == null
            ? "Available after every table is profiled"
            : undefined
        }
      />
      <MetricTile label="Columns" value={totalColumns ?? "—"} />
      <MetricTile
        label="Cleaned tables"
        value={cleanedTables ?? "—"}
        title="Tables derived by a cleaning operation"
      />
      <MetricTile
        label="Stored size"
        value={totalBytes == null || totalBytes === 0 ? "—" : formatBytes(totalBytes)}
      />
    </MetricStrip>
  );
}

/* The failed counterpart of the empty state: sending the user off to upload
 * data would hide that the analysis stopped, and why. */
function FailedRunState({
  failure,
  cancelled,
  projectId,
  sessionId,
}: {
  /** Null when the job history no longer names the failed run. */
  failure: PrimaryRunFailure | null;
  cancelled: boolean;
  projectId: string;
  sessionId: string;
}) {
  return (
    <Card role="alert" className="flex flex-col gap-2 border-status-critical/40 p-4">
      <p className="text-sm font-semibold text-status-critical">
        {(failure?.cancelled ?? cancelled)
          ? "The analysis was stopped before data was ready"
          : "The analysis failed before data was ready"}
      </p>
      {failure?.reason && (
        <p className="text-sm text-status-neutral">{failure.reason}</p>
      )}
      <div className="flex flex-wrap items-center gap-3">
        {failure && (
          <RunAgainButton
            jobId={failure.job.job_id}
            sessionId={sessionId}
            projectId={projectId}
            sourceSessionId={failure.job.source_session_id}
          />
        )}
        <Link
          to={sessionSectionPath(projectId, sessionId, "trace")}
          className="text-sm font-medium text-primary hover:underline"
        >
          See what happened on the Trace page
        </Link>
      </div>
    </Card>
  );
}

function RunInputs({ names }: { names: string[] }) {
  return (
    <Card className="flex flex-col gap-2 border-primary/30 bg-primary/5 p-4">
      <div className="flex flex-wrap items-center gap-2">
        <Badge tone="info">
          <Dot tone="info" />
          Preparing data
        </Badge>
        <span className="text-sm text-status-neutral">
          The table map fills in as profiling completes.
        </span>
      </div>
      {names.length > 0 && (
        <p className="text-sm">
          <span className="font-medium">Selected input{names.length === 1 ? "" : "s"}:</span>{" "}
          {names.join(", ")}
        </p>
      )}
    </Card>
  );
}

function DatasetHealth({
  dataset,
  issues,
  qualityState,
}: {
  dataset: DatasetHandle;
  issues: QualityDatasetCard | undefined;
  qualityState: "loading" | "pending" | "available" | "unavailable";
}) {
  if (dataset.ingest_status !== "ready") {
    return (
      <Badge tone="warn">
        <Dot tone="warn" />
        {dataset.ingest_status}
      </Badge>
    );
  }
  if (qualityState === "loading" || qualityState === "pending") {
    return <Badge tone="neutral">quality pending</Badge>;
  }
  if (qualityState === "unavailable") {
    return (
      <Badge tone="warn">
        <Dot tone="warn" />
        quality unavailable
      </Badge>
    );
  }
  const critical = issues?.critical ?? 0;
  const warn = issues?.warn ?? 0;
  const info = issues?.info ?? 0;
  if (critical > 0) {
    return <Badge tone="critical"><Dot tone="critical" />{critical} critical</Badge>;
  }
  if (warn > 0) {
    return <Badge tone="warn"><Dot tone="warn" />{warn} warning{warn > 1 ? "s" : ""}</Badge>;
  }
  if (info > 0) return <Badge tone="info">{info} note{info > 1 ? "s" : ""}</Badge>;
  return <Badge tone="ok"><Dot tone="ok" />no issues</Badge>;
}

function DatasetOverviewRow({
  dataset,
  issues,
  qualityState,
  gate,
  projectId,
  sessionId,
}: {
  dataset: DatasetHandle;
  issues: QualityDatasetCard | undefined;
  qualityState: "loading" | "pending" | "available" | "unavailable";
  gate: DatasetGate | undefined;
  projectId: string;
  sessionId: string;
}) {
  const columns = dataset.schema ?? [];
  const kinds = countColumnKinds(columns);
  const composition =
    kinds.length > 0 ? kindCompositionText(kinds) : "schema unavailable";
  const rows = dataset.row_count;
  const edited = isEditedDataset(dataset);
  return (
    <article className="dataset-inventory-row grid min-w-0 gap-3 border-t border-hairline px-3 py-3 first:border-t-0">
      <div className="min-w-0">
        <h2 className="min-w-0 text-sm font-semibold">
          <Marquee title={dataset.display_name}>{dataset.display_name}</Marquee>
        </h2>
        <div className="mt-1 flex flex-wrap items-center gap-1.5 text-xs text-status-neutral">
          <span>{dataset.format.toUpperCase()}</span>
          {edited && <Badge tone="brand">edited</Badge>}
          {gate && gate.piiColumns.length > 0 && (
            <Badge
              tone="warn"
              title={`Values in these columns are masked in shared artifacts: ${gate.piiColumns.join(", ")}`}
            >
              {gate.piiColumns.length} PII column{gate.piiColumns.length === 1 ? "" : "s"}
            </Badge>
          )}
        </div>
      </div>
      <div className="flex flex-col gap-0.5 text-sm">
        <span className="tabular font-medium">
          <span>{rows == null ? "Rows unknown" : `${formatCompact(rows)} rows`}</span>
          <span aria-hidden> · </span>
          <span>{columns.length} columns</span>
        </span>
        <span className="text-xs text-status-neutral">
          {formatBytes(dataset.byte_size ?? 0)}
        </span>
      </div>
      <div className="flex min-w-0 flex-col gap-1.5">
        <KindCompositionBar counts={kinds} />
        <p className="truncate text-xs text-status-neutral" title={composition}>
          {composition}
        </p>
      </div>
      <div className="flex flex-wrap items-center gap-1.5">
        <DatasetHealth dataset={dataset} issues={issues} qualityState={qualityState} />
        {gate && <ReadinessBadge gate={gate} />}
      </div>
      <div className="dataset-inventory-actions flex flex-wrap items-center gap-2">
        <Link
          to={tablePath(projectId, sessionId, dataset.dataset_id)}
          aria-label={`Preview ${dataset.display_name}`}
          className="inline-flex items-center rounded-base border border-border bg-bg px-2.5 py-1.5 text-xs font-medium hover:border-primary hover:text-primary"
        >
          Preview
        </Link>
        <Link
          to={`${sessionSectionPath(projectId, sessionId, "quality")}?dataset=${encodeURIComponent(dataset.dataset_id)}`}
          className="inline-flex items-center rounded-base border border-border bg-bg px-2.5 py-1.5 text-xs font-medium hover:border-primary hover:text-primary"
        >
          Review quality
        </Link>
      </div>
    </article>
  );
}

function DatasetOverview({
  datasets,
  issuesByDataset,
  qualityState,
  gates,
  projectId,
  sessionId,
}: {
  datasets: DatasetHandle[];
  issuesByDataset: Map<string, QualityDatasetCard>;
  qualityState: "loading" | "pending" | "available" | "unavailable";
  gates: Map<string, DatasetGate>;
  projectId: string;
  sessionId: string;
}) {
  return (
    <section aria-labelledby="table-inventory-heading" className="flex min-w-0 flex-col gap-2">
      <div className="flex min-w-0">
        <SectionHeader
          level={2}
          title={
            <span id="table-inventory-heading" className="flex flex-wrap items-baseline gap-x-2">
              <span>Table inventory</span>
              <span className="tabular text-sm font-normal text-status-neutral">
                {datasets.length} {datasets.length === 1 ? "table" : "tables"}
              </span>
            </span>
          }
        />
      </div>
      <Card className="dataset-inventory min-w-0 overflow-hidden">
        <div className="dataset-inventory-header gap-3 border-b border-table-border bg-table-header-bg px-3 py-2 text-xs font-medium text-status-neutral">
          <span>Table</span>
          <span>Shape</span>
          <span>Field mix</span>
          <span>Health & readiness</span>
          <span className="text-right">Open</span>
        </div>
        {datasets.map((dataset) => (
          <DatasetOverviewRow
            key={dataset.dataset_id}
            dataset={dataset}
            issues={issuesByDataset.get(dataset.dataset_id) ?? issuesByDataset.get(dataset.display_name)}
            qualityState={qualityState}
            gate={gates.get(dataset.dataset_id)}
            projectId={projectId}
            sessionId={sessionId}
          />
        ))}
      </Card>
    </section>
  );
}

export function Component() {
  const { projectId = "", sessionId = "" } = useParams();
  const { activeJob } = useJobActivity();
  const trackedJob = activeJob?.sourceSessionId === sessionId ? activeJob : null;
  const run = useSessionDetail(sessionId);
  const sessionStillBuilding =
    !run.data || !TERMINAL_SESSION_STATUSES.has(run.data.status);
  const sessionLive = Boolean(trackedJob) && sessionStillBuilding;
  const datasets = useDatasets(sessionId);
  const quality = useQuality(sessionId);
  const handoff = useEdaHandoff(sessionId);
  const stopped =
    run.data?.status === "failed" || run.data?.status === "cancelled";
  const failure = usePrimaryRunFailure(sessionId, stopped);
  const resourceLimit = run.data?.resource_limit ?? null;

  useEffect(() => {
    if (!sessionLive && !run.data?.status) return;
    if (!sessionStillBuilding) return;
    const timer = window.setInterval(() => {
      void run.refetch();
      void datasets.refetch();
      void quality.refetch();
    }, 1500);
    return () => window.clearInterval(timer);
  }, [
    sessionLive,
    sessionStillBuilding,
    run.data?.status,
    run.refetch,
    datasets.refetch,
    quality.refetch,
  ]);

  const issuesByDataset = new Map<string, QualityDatasetCard>();
  for (const card of quality.data?.datasets ?? []) issuesByDataset.set(card.dataset_id ?? card.dataset_name, card);
  const gates = handoffGates(handoff.data?.payload);
  const inputNames = trackedJob?.inputDatasets?.map((dataset) => dataset.displayName) ?? [];
  const qualityState = quality.isError
    ? "unavailable"
    : sessionStillBuilding
      ? "pending"
      : quality.isSuccess
        ? "available"
        : "loading";

  return (
    <DataWorkspacePage
      title="Data Map"
      description="The tables available in this analysis, their shape and current health."
    >
      {run.isPending && <LoadingSkeleton lines={2} label="Loading session" />}
      {run.isError && <ErrorState error={run.error} onRetry={() => run.refetch()} />}
      {run.data && (
        <SessionSummary
          datasets={datasets.data}
          inputCount={Math.max(
            inputNames.length,
            (run.data.dataset_names ?? []).length,
          )}
        />
      )}
      {sessionLive && <RunInputs names={inputNames} />}
      {resourceLimit && <ResourceLimitNotice limit={resourceLimit} />}
      {run.data && (run.data.warnings ?? []).length > 0 && (
        <Card className="flex flex-col gap-1 border-status-warn/40 p-4">
          <span className="font-semibold">Session warnings</span>
          {(run.data.warnings ?? []).map((warning) => <p key={warning} className="text-sm text-status-neutral">{warning}</p>)}
        </Card>
      )}
      {datasets.isPending && <LoadingSkeleton lines={4} label="Loading datasets" />}
      {datasets.isError && <ErrorState error={datasets.error} onRetry={() => datasets.refetch()} />}
      {quality.isError && <PartialState error={quality.error} onRetry={() => quality.refetch()} />}
      {datasets.data && (datasets.data.length === 0 ? (
        sessionStillBuilding ? null : stopped ? (
          <FailedRunState
            failure={failure}
            cancelled={run.data?.status === "cancelled"}
            projectId={projectId}
            sessionId={sessionId}
          />
        ) : resourceLimit ? null : (
          /* The stop is already explained above; "upload data to see datasets"
           * would send the user to fix a problem they do not have. */
          <EmptyState title="No datasets in this session" description="Upload data and start a session to see its datasets here." />
        )
      ) : (
        <DatasetOverview
          datasets={datasets.data}
          issuesByDataset={issuesByDataset}
          qualityState={qualityState}
          gates={gates}
          projectId={projectId}
          sessionId={sessionId}
        />
      ))}
    </DataWorkspacePage>
  );
}
