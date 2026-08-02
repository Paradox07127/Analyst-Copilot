import { Link, useLocation, useParams } from "react-router";
import {
  useDeleteSupportDoc,
  useDeleteUpload,
  useEdaHandoff,
  useProjectUploads,
  useSessionDetail,
  useSupportDocs,
} from "../../api/hooks";
import { sessionSectionPath } from "../paths";
import { isUnfiled } from "../unfiled";
import { Marquee } from "../../components/ui";
import { useWorkspaceFocus } from "../workspace-focus";

const SECTION_LABELS: Record<string, string> = {
  "data-map": "Data map",
  table: "Table preview",
  quality: "Quality",
  profiles: "Profiles",
  charts: "Charts",
  cleaning: "Cleanup",
  relationships: "Relationships",
  semantic: "Knowledge",
  questions: "Questions",
  "deep-analysis": "Deep analysis",
  findings: "Findings",
  compare: "Compare",
  chat: "Chat",
  skills: "Skills",
  report: "Report",
  board: "Board",
  trace: "Trace & cost",
  artifacts: "Artifacts",
};

/* The pages that read the data itself. On these the question in the user's
 * head is "can I trust this table, and what is a row?", so the inspector
 * answers that instead of repeating session counters. */
const DATA_SECTIONS = new Set([
  "data-map",
  "table",
  "quality",
  "profiles",
  "charts",
  "cleaning",
  "relationships",
  "semantic",
]);

type HandoffDataset = {
  name: string;
  grain: string | null;
  analysisReady: boolean;
  materialCodes: string[];
  piiCount: number;
};

function readHandoffDatasets(payload: unknown): HandoffDataset[] {
  if (typeof payload !== "object" || payload === null) return [];
  const datasets = (payload as { datasets?: unknown }).datasets;
  if (!Array.isArray(datasets)) return [];
  const parsed: HandoffDataset[] = [];
  for (const entry of datasets) {
    if (typeof entry !== "object" || entry === null) continue;
    const record = entry as Record<string, unknown>;
    const quality =
      typeof record["quality"] === "object" && record["quality"] !== null
        ? (record["quality"] as Record<string, unknown>)
        : {};
    parsed.push({
      name: typeof record["name"] === "string" ? record["name"] : "table",
      grain: typeof record["grain"] === "string" ? record["grain"] : null,
      analysisReady: record["analysis_ready"] === true,
      materialCodes: Array.isArray(quality["material_codes"])
        ? quality["material_codes"].filter(
            (code): code is string => typeof code === "string",
          )
        : [],
      piiCount:
        typeof record["pii_columns"] === "object" && record["pii_columns"] !== null
          ? Object.keys(record["pii_columns"] as Record<string, unknown>).length
          : 0,
    });
  }
  return parsed;
}

/** What one row is, and whether the table may be relied on — the two things
 *  every other number on these pages is conditional on. */
function DataReadiness({ sessionId }: { sessionId: string }) {
  const handoff = useEdaHandoff(sessionId);
  const datasets = readHandoffDatasets(handoff.data?.payload);
  if (datasets.length === 0) return null;
  return (
    <section className="flex flex-col gap-2">
      <p className="text-[10px] font-medium uppercase text-status-neutral">
        Grain &amp; readiness
      </p>
      {datasets.map((dataset) => (
        <div
          key={dataset.name}
          className="flex flex-col gap-1 rounded-base border border-border bg-bg p-2"
        >
          <div className="flex items-baseline justify-between gap-2">
            <Marquee title={dataset.name}>
              <span className="text-xs font-medium">{dataset.name}</span>
            </Marquee>
            <span
              className={`shrink-0 text-[10px] font-medium uppercase ${
                dataset.analysisReady ? "text-status-ok" : "text-status-critical"
              }`}
            >
              {dataset.analysisReady ? "ready" : "limited"}
            </span>
          </div>
          <p className="text-xs text-status-neutral">
            {dataset.grain ?? "Grain not established."}
          </p>
          {dataset.materialCodes.length > 0 && (
            <p className="font-mono text-[10px] text-status-neutral">
              {dataset.materialCodes.slice(0, 4).join(" · ")}
            </p>
          )}
          {dataset.piiCount > 0 && (
            <p className="text-[10px] text-status-warn">
              {dataset.piiCount} PII column
              {dataset.piiCount > 1 ? "s" : ""} masked in shared artifacts
            </p>
          )}
        </div>
      ))}
    </section>
  );
}

function Metric({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-base border border-border bg-bg p-2">
      <dt className="text-[10px] font-medium uppercase text-status-neutral">
        {label}
      </dt>
      <dd className="mt-0.5 text-sm font-medium">
        {typeof value === "number"
          ? `${value.toLocaleString()} ${label.toLowerCase()}`
          : `${label}: ${value}`}
      </dd>
    </div>
  );
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value < 10 ? value.toFixed(1) : Math.round(value)} ${units[unit]}`;
}

/* Files belong to the project, not to the session that happened to upload
 * them, and nothing else in the app shows what a project is holding. This is
 * the management surface: what is stored, how big it is, and how to get rid of
 * it — the same delete the Launchpad offers, reachable without starting a run. */
function ProjectData({ projectId }: { projectId: string }) {
  const uploads = useProjectUploads(projectId);
  const remove = useDeleteUpload(projectId);
  const files = uploads.data ?? [];
  const totalBytes = files.reduce((sum, file) => sum + file.byte_size, 0);

  return (
    <section className="flex flex-col gap-1.5">
      <p className="flex items-baseline gap-2 text-[10px] font-medium uppercase text-status-neutral">
        Project data
        {files.length > 0 && (
          <span className="tabular ml-auto normal-case">
            {files.length} file{files.length === 1 ? "" : "s"} · {formatBytes(totalBytes)}
          </span>
        )}
      </p>
      {uploads.isPending && (
        <p role="status" className="text-xs text-status-neutral">
          Loading project data…
        </p>
      )}
      {uploads.isError && (
        <p className="text-xs text-status-warn">Could not list project data.</p>
      )}
      {uploads.data && files.length === 0 && (
        <p className="text-xs text-status-neutral">
          No data uploaded to this project yet.
        </p>
      )}
      <ul className="flex flex-col gap-0.5">
        {files.map((file) => (
          <li
            key={file.dataset_id}
            className="group/file flex items-center gap-2 rounded-base px-1.5 py-1 hover:bg-bg"
          >
            <Marquee className="min-w-0 flex-1 font-mono text-xs" title={file.display_name}>
              {file.display_name}
            </Marquee>
            <span className="tabular shrink-0 text-[11px] text-status-neutral">
              {formatBytes(file.byte_size)}
            </span>
            <button
              type="button"
              onClick={() => remove.mutate(file.dataset_id)}
              disabled={remove.isPending}
              aria-label={`Delete ${file.display_name}`}
              className="shrink-0 rounded-sm px-1 text-[11px] text-status-neutral opacity-0 transition-opacity hover:text-status-critical focus-visible:opacity-100 group-hover/file:opacity-100 disabled:opacity-40"
            >
              Delete
            </button>
          </li>
        ))}
      </ul>
      {remove.isError && (
        <p role="alert" className="text-xs text-status-critical">
          {remove.error instanceof Error
            ? remove.error.message
            : "Could not delete that file."}
        </p>
      )}
    </section>
  );
}

function ProjectSupportDocuments({ projectId }: { projectId: string }) {
  const docs = useSupportDocs(projectId);
  const remove = useDeleteSupportDoc(projectId);
  const files = docs.data?.pages.flatMap((page) => page.docs ?? []) ?? [];

  return (
    <section className="flex flex-col gap-1.5">
      <p className="flex items-baseline gap-2 text-[10px] font-medium uppercase text-status-neutral">
        Support documents
        {files.length > 0 && (
          <span className="tabular ml-auto normal-case">
            {files.length} document{files.length === 1 ? "" : "s"}
          </span>
        )}
      </p>
      {docs.isPending && (
        <p role="status" className="text-xs text-status-neutral">
          Loading support documents…
        </p>
      )}
      {docs.isError && (
        <p className="text-xs text-status-warn">
          Could not list support documents.
        </p>
      )}
      {docs.data && files.length === 0 && (
        <p className="text-xs text-status-neutral">No support documents yet.</p>
      )}
      <ul className="flex max-h-48 flex-col gap-0.5 overflow-y-auto">
        {files.map((file) => (
          <li
            key={file.doc_id}
            className="group/doc flex items-center gap-2 rounded-base px-1.5 py-1 hover:bg-bg"
          >
            <Marquee className="min-w-0 flex-1 font-mono text-xs" title={file.name}>
              {file.name}
            </Marquee>
            <span className="tabular shrink-0 text-[11px] text-status-neutral">
              {formatBytes(file.byte_size)}
            </span>
            <button
              type="button"
              onClick={() => remove.mutate(file.doc_id)}
              disabled={remove.isPending}
              aria-label={`Delete support document ${file.name}`}
              className="shrink-0 rounded-sm px-1 text-[11px] text-status-neutral opacity-0 transition-opacity hover:text-status-critical focus-visible:opacity-100 group-hover/doc:opacity-100 disabled:opacity-40"
            >
              Delete
            </button>
          </li>
        ))}
      </ul>
      {docs.hasNextPage && (
        <button
          type="button"
          onClick={() => void docs.fetchNextPage()}
          disabled={docs.isFetchingNextPage}
          className="self-start text-xs text-link hover:underline disabled:opacity-50"
        >
          {docs.isFetchingNextPage ? "Loading…" : "Load more"}
        </button>
      )}
      {remove.isError && (
        <p role="alert" className="text-xs text-status-critical">
          {remove.error instanceof Error
            ? remove.error.message
            : "Could not delete that document."}
        </p>
      )}
    </section>
  );
}

export function Inspector() {
  const route = useParams();
  const { pathname } = useLocation();
  const workspace = useWorkspaceFocus();
  const focused = workspace.mode === "single" ? null : workspace.activeContext;
  const projectId = focused?.projectId ?? route.projectId ?? "";
  const sessionId = focused?.sessionId ?? route.sessionId ?? "";
  const run = useSessionDetail(sessionId);
  // "table/:datasetId" ends in an id, not a section name; fall back to the
  // preceding segment so the page keeps its label and its data context.
  const segments = pathname.split("/").filter(Boolean);
  const lastSegment = segments.at(-1) ?? "";
  const pathSection =
    lastSegment in SECTION_LABELS ? lastSegment : (segments.at(-2) ?? lastSegment);
  const section = focused?.section ?? pathSection;
  const sectionLabel =
    SECTION_LABELS[section] ?? (section && sessionId ? section : "Home");

  return (
    <aside
      aria-label="Context Inspector"
      className="flex h-full flex-col overflow-auto border-l border-border bg-surface"
    >
      <h2 className="px-4 pt-4 pb-2 text-xs font-semibold tracking-wide text-status-neutral uppercase">
        Context Inspector
      </h2>
      <div className="flex flex-col gap-4 px-4 pb-4 text-sm">
        <section className="flex flex-col gap-1">
          <p className="text-[10px] font-medium uppercase text-status-neutral">
            Current page
          </p>
          <p className="font-medium">{sectionLabel}</p>
        </section>
        {sessionId && DATA_SECTIONS.has(section) && (
          <DataReadiness sessionId={sessionId} />
        )}
        {projectId && !isUnfiled(projectId) && (
          <>
            <ProjectData projectId={projectId} />
            <ProjectSupportDocuments projectId={projectId} />
          </>
        )}
        {!sessionId && !projectId && (
          <p className="text-status-neutral">
            Open a session to inspect its evidence, activity and publication state.
          </p>
        )}
        {run.isPending && sessionId && (
          <p role="status" className="text-status-neutral">
            Loading session context…
          </p>
        )}
        {run.data && (
          <>
            <section className="flex flex-col gap-1">
              <p className="font-medium">{run.data.title ?? "Untitled session"}</p>
              <p className="text-xs text-status-neutral">
                {(run.data.dataset_names ?? []).join(", ") || "No datasets"} ·{" "}
                {run.data.status}
              </p>
              {run.data.source_session_id && (
                <p className="text-xs text-status-neutral">
                  Derived from{" "}
                  <Link
                    className="font-mono text-primary hover:underline"
                    to={sessionSectionPath(
                      projectId,
                      run.data.source_session_id,
                      "artifacts",
                    )}
                  >
                    {run.data.source_session_id}
                  </Link>
                </p>
              )}
            </section>
            <dl className="grid grid-cols-2 gap-2">
              <Metric label="Artifacts" value={run.data.artifact_count} />
              <Metric label="Messages" value={run.data.chat_message_count} />
              <Metric
                label="Report"
                value={run.data.report_status ?? "none"}
              />
              <Metric
                label="Warnings"
                value={(run.data.warnings ?? []).length}
              />
            </dl>
            {(run.data.warnings ?? []).length > 0 && (
              <section>
                <p className="mb-1 text-[10px] font-medium uppercase text-status-neutral">
                  Session warnings
                </p>
                <ul className="list-inside list-disc text-xs text-status-warn">
                  {(run.data.warnings ?? []).map((warning) => (
                    <li key={warning}>{warning}</li>
                  ))}
                </ul>
              </section>
            )}
            <nav aria-label="Inspector shortcuts" className="grid gap-1">
              {[
                ["Open artifacts", "artifacts"],
                ["Open trace & cost", "trace"],
                ["Open report", "report"],
                ["Open chat", "chat"],
              ].map(([label, target]) =>
                workspace.mode === "split" && focused?.onSectionChange ? (
                  <button
                    key={target}
                    type="button"
                    onClick={() => focused.onSectionChange?.(target!)}
                    className="rounded-base border border-border bg-bg px-2 py-1.5 text-left text-xs hover:border-primary hover:text-primary"
                  >
                    {label}
                  </button>
                ) : (
                  <Link
                    key={target}
                    to={sessionSectionPath(projectId, sessionId, target!)}
                    className="rounded-base border border-border bg-bg px-2 py-1.5 text-xs hover:border-primary hover:text-primary"
                  >
                    {label}
                  </Link>
                ),
              )}
            </nav>
          </>
        )}
      </div>
    </aside>
  );
}
