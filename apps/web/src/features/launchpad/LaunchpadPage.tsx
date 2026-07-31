/* New session. The file list is the page.
 *
 * Data is the only mandatory input — `blockedReason` cannot clear without a
 * selected dataset — while the context sentence is optional. The composer this
 * replaces inverted that: six rows of canvas for the optional prompt, one eighth
 * of a toolbar for the mandatory files, and project files hidden inside a
 * <select> labelled "Add project file".
 *
 * One screen, no numbered steps. The frequent path is a returning user reusing
 * files already in the project, and a staged wizard turns that one click into
 * three. Uploads and project files are siblings in a single list so that
 * dropping a CSV and reusing a stored one are the same gesture.
 *
 * `NewSessionPanel` is the whole surface and takes no page chrome, so Home and
 * the per-project + button mount it directly. */

import {
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
} from "react";
import { Link, useNavigate, useParams } from "react-router";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  api,
  ApiError,
  type DatasetHandle,
  type JobCreated,
  type PrecleaningOptions,
} from "../../api/client";
import {
  queryKeys,
  useProjectUploads,
  useProjects,
  useSettings,
  useSupportDocs,
  useUploadSupportDoc,
} from "../../api/hooks";
import { useJobActivity } from "../../app/job-activity";
import { sessionSectionPath } from "../../app/paths";
import { UNFILED_PROJECT_ID } from "../../app/unfiled";
import { useOpenSettingsDialog } from "../../app/settings-dialog";
import {
  Badge,
  Button,
  Card,
  Dot,
  Hint,
  IconButton,
  Marquee,
  SectionHeader,
} from "../../components/ui";
import { liveState } from "../settings/live-status";

interface UploadEntry {
  id: number;
  fileName: string;
  status: "staged" | "uploading" | "completed" | "failed";
  selected: boolean;
  file?: File;
  dataset?: DatasetHandle;
  error?: string;
}

function uploadErrorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.code === "upload_file_quota")
      return `Project file quota reached. ${error.message}`;
    if (error.code === "upload_project_byte_quota")
      return `Project storage quota reached. ${error.message}`;
    if (error.code === "upload_concurrent_quota")
      return `Too many uploads are already running. ${error.message}`;
    if (error.code === "upload_rate_limited")
      return `Uploads are being submitted too quickly. ${error.message}`;
  }
  return error instanceof Error ? error.message : String(error);
}

/* Same defaults as the pre-clean controls the desktop app shipped, so a run
 * started here behaves identically to one started there. */
const PRECLEAN_DEFAULTS: PrecleaningOptions = {
  clean_missing_values: false,
  missing_threshold_percent: 70,
  min_rows_keep_percent: 50,
  drop_iqr_outliers: false,
};

function precleanEnabled(options: PrecleaningOptions): boolean {
  return Boolean(options.clean_missing_values || options.drop_iqr_outliers);
}

function newRunId(): string {
  return `sess_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

const MAX_PROJECT_ID_LENGTH = 64;

function deriveProjectId(name: string): string {
  return name
    .replace(/[^A-Za-z0-9 _.-]/g, "")
    .replace(/\s+/g, " ")
    .replace(/^[^A-Za-z0-9]+/, "")
    .trim()
    .slice(0, MAX_PROJECT_ID_LENGTH);
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

function GearGlyph() {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 24 24"
      width="16"
      height="16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.75"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.7 1.7 0 0 0 .34 1.88l.06.06-2.12 2.12-.06-.06a1.7 1.7 0 0 0-1.88-.34 1.7 1.7 0 0 0-1.03 1.56V20.3h-3v-.08A1.7 1.7 0 0 0 10.68 18.66a1.7 1.7 0 0 0-1.88.34l-.06.06-2.12-2.12.06-.06A1.7 1.7 0 0 0 7.02 15a1.7 1.7 0 0 0-1.56-1.03H5.4v-3h.06A1.7 1.7 0 0 0 7.02 9.94a1.7 1.7 0 0 0-.34-1.88l-.06-.06 2.12-2.12.06.06a1.7 1.7 0 0 0 1.88.34 1.7 1.7 0 0 0 1.03-1.56V4.7h3v.08a1.7 1.7 0 0 0 1.03 1.56 1.7 1.7 0 0 0 1.88-.34l.06-.06L19.8 8l-.06.06a1.7 1.7 0 0 0-.34 1.88 1.7 1.7 0 0 0 1.56 1.03h.06v3h-.06A1.7 1.7 0 0 0 19.4 15Z" />
    </svg>
  );
}

/* ------------------------------------------------------------- the file list */

/** One row of the data list, whether it arrived by drop or was already stored.
 *  `toggle` is null for anything that cannot be analysed, which is what keeps a
 *  failed upload and an unreadable stored file visible but unselectable. */
interface FileRowModel {
  key: string;
  name: string;
  state: "ready" | "staged" | "uploading" | "failed" | "unreadable";
  selected: boolean;
  rowCount: number | null;
  columnCount: number | null;
  byteSize: number | null;
  detail: string | null;
  inProject: boolean;
  toggle: ((next: boolean) => void) | null;
  remove: (() => void) | null;
}

const ROW_TONE = {
  ready: "neutral",
  staged: "neutral",
  uploading: "warn",
  failed: "critical",
  unreadable: "warn",
} as const;

function FileRow({ row }: { row: FileRowModel }) {
  /* Figures rather than a name chip: catching "wrong CSV" here costs nothing,
   * and catching it after the run costs a whole run. */
  const stats = [
    row.rowCount !== null ? `${row.rowCount.toLocaleString()} rows` : null,
    row.columnCount !== null ? `${row.columnCount} cols` : null,
    row.byteSize !== null ? formatBytes(row.byteSize) : null,
  ]
    .filter(Boolean)
    .join(" · ");

  const checkboxId = `launchpad-${row.key}`;

  return (
    <li className="animate-enter">
      <div
        className={`flex min-w-0 items-center gap-2.5 px-3 py-2 ${
          row.toggle ? "hover:bg-surface" : ""
        }`}
      >
        <input
          id={checkboxId}
          type="checkbox"
          checked={row.selected}
          disabled={row.toggle === null}
          onChange={(event) => row.toggle?.(event.target.checked)}
          aria-label={`${row.selected ? "Exclude" : "Include"} ${row.name}`}
          className="shrink-0 disabled:opacity-40"
        />
        <label
          htmlFor={checkboxId}
          className={`flex min-w-0 flex-1 items-center gap-2.5 ${
            row.toggle ? "cursor-pointer" : "cursor-default"
          }`}
        >
          <Marquee className="min-w-0 flex-1 font-mono text-xs" title={row.name}>
            {row.name}
          </Marquee>
          {stats && (
            <span className="tabular hidden shrink-0 text-xs text-status-neutral sm:inline">
              {stats}
            </span>
          )}
          {row.detail && (
            <span
              className={`flex min-w-0 max-w-[55%] items-center gap-1.5 text-xs ${
                row.state === "failed"
                  ? "text-status-critical"
                  : row.state === "ready"
                    ? "text-status-neutral"
                    : "text-status-warn"
              }`}
            >
              <Dot
                tone={ROW_TONE[row.state]}
                motion={row.state === "uploading" ? "working" : undefined}
              />
              <Marquee title={row.detail}>
                {row.detail}
              </Marquee>
            </span>
          )}
          {row.inProject && (
            <Badge tone="neutral" title="Already stored in this project">
              in project
            </Badge>
          )}
        </label>
        {row.remove && (
          <button
            type="button"
            onClick={row.remove}
            aria-label={`Remove ${row.name} from this session`}
            title="Remove from this session"
            className="flex size-6 shrink-0 items-center justify-center rounded-base text-status-neutral hover:bg-bg hover:text-text"
          >
            ×
          </button>
        )}
      </div>
    </li>
  );
}

/* -------------------------------------------------------------- pre-cleaning */

/* Only the thresholds and the outlier switch: turning cleaning on at all is the
 * toolbar's job, so repeating that checkbox here would give one setting two
 * controls that can disagree. */
function PrecleaningFields({
  value,
  onChange,
}: {
  value: PrecleaningOptions;
  onChange: (next: PrecleaningOptions) => void;
}) {
  return (
    <div className="flex flex-wrap items-end gap-x-6 gap-y-3 border-t border-hairline px-3 py-2.5">
      <p className="w-full text-xs text-status-neutral">
        Creates cleaned copies for this run. Your uploads are never changed.
      </p>
      <label className="flex flex-col gap-1 text-xs text-status-neutral">
        {`Missing threshold: ${value.missing_threshold_percent ?? 70}%`}
        <input
          type="range"
          min={0}
          max={100}
          step={5}
          value={value.missing_threshold_percent ?? 70}
          onChange={(event) =>
            onChange({
              ...value,
              missing_threshold_percent: Number(event.target.value),
            })
          }
          className="w-40"
        />
      </label>
      <label className="flex flex-col gap-1 text-xs text-status-neutral">
        {`Minimum rows to keep: ${value.min_rows_keep_percent ?? 50}%`}
        <input
          type="range"
          min={0}
          max={100}
          step={5}
          value={value.min_rows_keep_percent ?? 50}
          onChange={(event) =>
            onChange({
              ...value,
              min_rows_keep_percent: Number(event.target.value),
            })
          }
          className="w-40"
        />
      </label>
      <label className="flex items-center gap-2 text-xs">
        <input
          type="checkbox"
          checked={Boolean(value.drop_iqr_outliers)}
          onChange={(event) =>
            onChange({ ...value, drop_iqr_outliers: event.target.checked })
          }
        />
        <span className="flex items-center gap-1.5">
          Drop IQR outlier rows
          <Hint label="Outlier cleaning">
            Skipped when it would take the table below the minimum-rows guard.
          </Hint>
        </span>
      </label>
    </div>
  );
}

/* ---------------------------------------------------------- support documents */

/* Project-level and persistent across sessions, so this owns its own server
 * state rather than joining the per-launch upload list. Rendered as one toolbar
 * line: it is optional context, not a step. */
function SupportDocsBar({
  projectId,
  disabled = false,
}: {
  projectId: string;
  disabled?: boolean;
}) {
  const docs = useSupportDocs(projectId);
  const upload = useUploadSupportDoc(projectId);
  const inputRef = useRef<HTMLInputElement>(null);
  const dragDepth = useRef(0);
  const [dragging, setDragging] = useState(false);

  const uploadFiles = async (files: File[]) => {
    for (const file of files) {
      try {
        await upload.mutateAsync(file);
      } catch {
        /* The error line below renders it; keep going with the rest. */
      }
    }
  };

  const items = docs.data?.pages.flatMap((page) => page.docs ?? []) ?? [];

  return (
    <span
      role="region"
      aria-label="Support documents"
      onDragEnter={(event: DragEvent<HTMLSpanElement>) => {
        if (disabled || event.dataTransfer.files.length === 0) return;
        event.preventDefault();
        dragDepth.current += 1;
        setDragging(true);
      }}
      onDragOver={(event: DragEvent<HTMLSpanElement>) => {
        if (disabled || event.dataTransfer.files.length === 0) return;
        event.preventDefault();
      }}
      onDragLeave={() => {
        dragDepth.current = Math.max(0, dragDepth.current - 1);
        if (dragDepth.current === 0) setDragging(false);
      }}
      onDrop={(event: DragEvent<HTMLSpanElement>) => {
        if (disabled) return;
        event.preventDefault();
        dragDepth.current = 0;
        setDragging(false);
        void uploadFiles(Array.from(event.dataTransfer.files));
      }}
      className={`flex flex-wrap items-center gap-1.5 rounded-base px-1 text-xs text-status-neutral transition-colors ${
        dragging ? "bg-primary/5" : ""
      }`}
    >
      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        disabled={disabled || upload.isPending}
        className="rounded-base px-1.5 py-1 underline-offset-2 hover:text-text hover:underline disabled:opacity-50"
      >
        {upload.isPending
          ? "Uploading…"
          : items.length === 0
            ? "Add support docs"
            : `${items.length} support doc${items.length === 1 ? "" : "s"}`}
      </button>
      <Hint label="Support documents">
        MD, TXT, CSV or text-based PDF, up to 10 MB per file. The agent may read
        them to guess what a column means. Scanned PDFs are not supported.
        Nothing in them can confirm a join or enter a report figure — only the
        data itself does that.
      </Hint>
      <input
        ref={inputRef}
        id="launchpad-support-docs"
        aria-label="Support documents (optional)"
        type="file"
        accept=".md,.txt,.csv,.pdf"
        multiple
        onChange={(event) => {
          const files = Array.from(event.target.files ?? []);
          event.target.value = "";
          void uploadFiles(files);
        }}
        disabled={disabled}
        className="sr-only"
      />
      {upload.isError && (
        <span role="alert" className="text-status-critical">
          {upload.error instanceof Error
            ? upload.error.message
            : "Upload failed."}
        </span>
      )}
    </span>
  );
}

/* ------------------------------------------------------------------- panel */

export function NewSessionPanel({
  projectId: fixedProjectId,
  layout = "inline",
}: {
  /** Set by the per-project route: the destination is decided, so the picker
   *  collapses to a label instead of offering to move the session. */
  projectId?: string;
  /** The route keeps launch confirmation visible beside the composer. Home
   * embeds the same panel inline inside its quick-start card. */
  layout?: "inline" | "route";
}) {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { startTracking } = useJobActivity();
  const openSettings = useOpenSettingsDialog();
  const settings = useSettings();
  const projects = useProjects();

  const [uploads, setUploads] = useState<UploadEntry[]>([]);
  const [projectChoice, setProjectChoice] = useState<
    "unfiled" | "existing" | "new"
  >(fixedProjectId ? "existing" : "unfiled");
  const [selectedProjectId, setSelectedProjectId] = useState(
    fixedProjectId ?? "",
  );
  const [newProjectName, setNewProjectName] = useState("");
  const [businessContext, setBusinessContext] = useState("");
  const [preclean, setPreclean] = useState<PrecleaningOptions>(PRECLEAN_DEFAULTS);
  const [reusedIds, setReusedIds] = useState<string[]>([]);
  const [dragging, setDragging] = useState(false);
  const [fileFeedback, setFileFeedback] = useState<string | null>(null);
  const [conflictRunId, setConflictRunId] = useState<string | null>(null);
  const nextId = useRef(0);
  const dragDepth = useRef(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (fixedProjectId) {
      setProjectChoice("existing");
      setSelectedProjectId(fixedProjectId);
    }
  }, [fixedProjectId]);

  /* One idempotency key + generated run id per launch attempt: retries replay
   * the same job instead of creating a second session. */
  const launchRef = useRef<{
    sessionId: string;
    key: string;
    input: string;
    projectKey: string;
    projectCreated: boolean;
  } | null>(null);

  const newProjectId = deriveProjectId(newProjectName);
  const projectId =
    projectChoice === "unfiled"
      ? UNFILED_PROJECT_ID
      : projectChoice === "existing"
        ? selectedProjectId
        : newProjectId;
  const isUnfiled = projectChoice === "unfiled";
  const unfiledProject = useMutation({
    mutationFn: () =>
      api.createProject(
        { project_id: UNFILED_PROJECT_ID, name: "Unfiled sessions" },
        "unfiled-sessions-v1",
      ),
  });

  useEffect(() => {
    if (
      isUnfiled &&
      !unfiledProject.isSuccess &&
      !unfiledProject.isPending &&
      !unfiledProject.isError
    ) {
      unfiledProject.mutate();
    }
    // The private bucket is idempotently provisioned once per mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isUnfiled]);

  const workspacePending = isUnfiled
    ? !unfiledProject.isSuccess
    : projectId === "";
  const projectReady = !workspacePending && projectId !== null;
  const activeProjectId = projectId ?? UNFILED_PROJECT_ID;
  /* Files already uploaded belong to the chosen project, so moving the session
   * elsewhere afterwards would orphan them. */
  const projectLocked = uploads.length > 0;
  const availableProjects = (projects.data ?? []).filter(
    (project) => project.project_id !== UNFILED_PROJECT_ID,
  );
  const fixedProject = availableProjects.find(
    (project) => project.project_id === fixedProjectId,
  );

  const updateEntry = (id: number, patch: Partial<UploadEntry>) => {
    setUploads((entries) =>
      entries.map((entry) => (entry.id === id ? { ...entry, ...patch } : entry)),
    );
  };

  const ingest = async (files: File[]) => {
    if (!projectReady) return;
    setFileFeedback(null);
    if (projectChoice === "new") {
      /* The project does not exist yet, so these are held locally and uploaded
       * by the launch itself once it has been created. */
      setUploads((entries) => [
        ...entries,
        ...files.map((file) => ({
          id: nextId.current++,
          fileName: file.name,
          file,
          status: "staged" as const,
          selected: true,
        })),
      ]);
      return;
    }
    for (const file of files) {
      const id = nextId.current++;
      setUploads((entries) => [
        ...entries,
        { id, fileName: file.name, status: "uploading", selected: true },
      ]);
      try {
        const result = await api.createUpload(activeProjectId, file);
        if (result.status === "completed" && result.dataset) {
          updateEntry(id, { status: "completed", dataset: result.dataset });
        } else {
          updateEntry(id, {
            status: "failed",
            error: result.error ?? `upload ${result.status}`,
          });
        }
      } catch (error) {
        updateEntry(id, { status: "failed", error: uploadErrorMessage(error) });
      }
    }
  };

  const onPickFiles = async (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files ?? []);
    event.target.value = "";
    setFileFeedback(null);
    await ingest(files);
  };

  const freshDatasets = uploads.filter(
    (entry) => entry.status === "completed" && entry.dataset,
  );
  const uploading = uploads.some((entry) => entry.status === "uploading");

  /* Files belong to the project, not to whoever uploaded them first, so a new
   * session starts from what is already there instead of asking for the same
   * CSV again. Only fetched once a project is settled — the standalone bucket is
   * created on demand. */
  const projectUploads = useProjectUploads(
    activeProjectId,
    projectReady && projectChoice !== "new",
  );
  const freshIds = new Set(
    freshDatasets.map((entry) => entry.dataset!.dataset_id),
  );
  /* Anything uploaded in this visit already has a row below; listing it twice
   * would let the same table be selected twice. */
  const existing = (projectUploads.data ?? []).filter(
    (handle) => !freshIds.has(handle.dataset_id),
  );

  const setDatasetSelected = (datasetId: string, selected: boolean) => {
    setUploads((entries) =>
      entries.map((entry) =>
        entry.dataset?.dataset_id === datasetId ? { ...entry, selected } : entry,
      ),
    );
    setReusedIds((current) =>
      selected
        ? current.includes(datasetId)
          ? current
          : [...current, datasetId]
        : current.filter((id) => id !== datasetId),
    );
  };

  const reused = existing.filter(
    (handle) =>
      handle.ingest_status === "ready" &&
      Boolean(handle.original_uri) &&
      reusedIds.includes(handle.dataset_id),
  );
  const selectedDatasets: DatasetHandle[] = [
    ...reused,
    ...freshDatasets.filter((entry) => entry.selected).map((entry) => entry.dataset!),
  ];
  const stagedEntries = uploads.filter(
    (entry) => entry.status === "staged" && entry.selected && entry.file,
  );
  const selectedFileCount = selectedDatasets.length + stagedEntries.length;

  /* Uploads first, then what was already stored: this visit's work stays where
   * the eye landed after the drop. */
  const rows: FileRowModel[] = [
    ...uploads.map((entry): FileRowModel => {
      const handle = entry.dataset;
      const selectable = entry.status === "completed" || entry.status === "staged";
      return {
        key: `upload-${entry.id}`,
        name: entry.fileName,
        state:
          entry.status === "completed"
            ? "ready"
            : entry.status === "staged"
              ? "staged"
              : entry.status,
        selected: selectable && entry.selected,
        rowCount: handle?.row_count ?? null,
        columnCount: handle?.schema?.length ?? null,
        byteSize: handle?.byte_size ?? entry.file?.size ?? null,
        detail:
          entry.status === "uploading"
            ? "Uploading…"
            : entry.status === "failed"
              ? `Upload failed${entry.error ? `: ${entry.error}` : ""}`
              : entry.status === "completed" && handle
                ? `Ready · ${handle.dataset_id}`
                : null,
        inProject: false,
        remove: () => {
          setUploads((entries) =>
            entries.filter((item) => item.id !== entry.id),
          );
          if (handle) {
            queryClient.invalidateQueries({
              queryKey: queryKeys.projectUploads(activeProjectId),
            });
          }
        },
        toggle: selectable
          ? (next) =>
              handle
                ? setDatasetSelected(handle.dataset_id, next)
                : updateEntry(entry.id, { selected: next })
          : null,
      };
    }),
    ...existing.map((handle): FileRowModel => {
      const readable =
        handle.ingest_status === "ready" && Boolean(handle.original_uri);
      return {
        key: `project-${handle.dataset_id}`,
        name: handle.display_name,
        state: readable ? "ready" : "unreadable",
        selected: reusedIds.includes(handle.dataset_id),
        rowCount: handle.row_count ?? null,
        columnCount: handle.schema?.length ?? null,
        byteSize: handle.byte_size ?? null,
        detail: readable ? null : "Stored file is missing or still ingesting",
        inProject: true,
        remove: null,
        toggle: readable
          ? (next) => setDatasetSelected(handle.dataset_id, next)
          : null,
      };
    }),
  ];

  const launch = useMutation({
    mutationFn: async () => {
      const input = projectChoice === "new" ? newProjectId : activeProjectId;
      if (!launchRef.current || launchRef.current.input !== input) {
        launchRef.current = {
          sessionId: newRunId(),
          key: crypto.randomUUID(),
          input,
          projectKey: crypto.randomUUID(),
          projectCreated: false,
        };
      }
      const attempt = launchRef.current;
      const { sessionId, key } = attempt;
      let datasets = selectedDatasets;
      if (projectChoice === "new") {
        if (!attempt.projectCreated) {
          await api.createProject(
            {
              project_id: activeProjectId,
              name: newProjectName.trim() || activeProjectId,
            },
            attempt.projectKey,
          );
          attempt.projectCreated = true;
          queryClient.invalidateQueries({ queryKey: queryKeys.projects });
        }
        const uploaded: DatasetHandle[] = [];
        for (const entry of stagedEntries) {
          const result = await api.createUpload(activeProjectId, entry.file!);
          if (result.status !== "completed" || !result.dataset) {
            throw new Error(
              result.error ?? `Could not upload ${entry.fileName}.`,
            );
          }
          uploaded.push(result.dataset);
          updateEntry(entry.id, {
            status: "completed",
            dataset: result.dataset,
          });
        }
        datasets = [...datasets, ...uploaded];
      }
      const created = await api.createJob(
        sessionId,
        {
          kind: "auto_eda",
          project_id: activeProjectId,
          datasets: datasets.map((handle) => handle.dataset_id),
          business_context: businessContext,
          generate_report: true,
          llm: "env",
          precleaning: precleanEnabled(preclean) ? preclean : null,
        },
        key,
      );
      return { created, datasets };
    },
    onSuccess: ({
      created,
      datasets,
    }: {
      created: JobCreated;
      datasets: DatasetHandle[];
    }) => {
      launchRef.current = null;
      startTracking({
        jobId: created.job_id,
        sessionId: created.session_id,
        sourceSessionId: created.session_id,
        projectId: activeProjectId,
        eventsUrl: created.events_url,
        inputDatasets: datasets.map((dataset) => ({
          datasetId: dataset.dataset_id,
          displayName: dataset.display_name,
          byteSize: dataset.byte_size,
          rowCount: dataset.row_count,
          format: dataset.format,
        })),
      });
      navigate(
        sessionSectionPath(activeProjectId, created.session_id, "data-map"),
      );
    },
    onError: (error: unknown) => {
      /* A 409 means the key/run already has a job — a retry with the same key
       * cannot help, so the next click starts a fresh attempt. */
      if (error instanceof ApiError && error.status === 409) {
        setConflictRunId(launchRef.current?.sessionId ?? null);
        launchRef.current = null;
      }
    },
  });

  const conflict =
    launch.error instanceof ApiError && launch.error.code === "job_conflict"
      ? launch.error
      : null;
  const llmConnection = settings.data ? liveState(settings.data) : null;

  /* One reason at a time, in the order the user can act on them. The composer
   * this replaces disabled the button on six conditions and named none of them,
   * so a dead primary button was the only feedback. */
  const blockedReason: string | null = !settings.data
    ? settings.isError
      ? "Could not load LLM settings."
      : "Checking the LLM connection…"
    : llmConnection === "incomplete"
      ? "Add an API key in Settings to start."
      : unfiledProject.isError && isUnfiled
        ? "Could not prepare private storage."
        : !projectReady
          ? projectChoice === "new"
            ? "Name the new project to start."
            : "Choose a project to start."
          : uploading
            ? "Waiting for the upload to finish…"
            : selectedFileCount === 0
              ? "Select at least one file to start."
              : null;
  const blocked = blockedReason !== null || launch.isPending;

  const dropHandlers = {
    onDragEnter: (event: DragEvent) => {
      if (!projectReady) return;
      event.preventDefault();
      dragDepth.current += 1;
      setDragging(true);
    },
    onDragOver: (event: DragEvent) => {
      if (!projectReady) return;
      event.preventDefault();
      setDragging(true);
    },
    onDragLeave: () => {
      dragDepth.current = Math.max(0, dragDepth.current - 1);
      if (dragDepth.current === 0) setDragging(false);
    },
    onDrop: (event: DragEvent) => {
      event.preventDefault();
      dragDepth.current = 0;
      setDragging(false);
      if (!projectReady) return;
      const dropped = Array.from(event.dataTransfer.files);
      const csvFiles = dropped.filter((file) => /\.csv$/i.test(file.name));
      if (csvFiles.length > 0) void ingest(csvFiles);
      if (csvFiles.length !== dropped.length) {
        setFileFeedback(
          csvFiles.length === 0
            ? "Add CSV files only."
            : "Some files were ignored. This analysis accepts CSV files only.",
        );
      }
    },
  };

  const setDestination = (value: string) => {
    setReusedIds([]);
    setConflictRunId(null);
    launch.reset();
    launchRef.current = null;
    if (value === "unfiled" || value === "new") {
      setProjectChoice(value);
      return;
    }
    setProjectChoice("existing");
    setSelectedProjectId(value);
  };

  const modelLabel =
    llmConnection === "offline"
      ? "Offline"
      : settings.data?.model || "LLM";
  const supportDocsDisabled = !projectReady || projectChoice === "new";
  const runSummary = (
    <dl className="grid grid-cols-2 gap-x-3 gap-y-2 text-xs">
      <div className="flex flex-col gap-0.5">
        <dt className="text-status-neutral">Data</dt>
        <dd className="tabular font-medium">
          {selectedFileCount} file{selectedFileCount === 1 ? "" : "s"}
        </dd>
      </div>
      <div className="flex flex-col gap-0.5">
        <dt className="text-status-neutral">Output</dt>
        <dd className="font-medium">Report on</dd>
      </div>
      <div className="col-span-2 flex min-w-0 items-center gap-1.5">
        <Dot tone={llmConnection === "ready" ? "ok" : "neutral"} />
        <dt className="text-status-neutral">Model</dt>
        <dd className="min-w-0 font-medium">
          <Marquee title={modelLabel}>{modelLabel}</Marquee>
        </dd>
        <IconButton label="Configure LLM in Settings" onClick={openSettings}>
          <GearGlyph />
        </IconButton>
      </div>
    </dl>
  );
  const runOptions = (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
      <label className="flex items-center gap-1.5 text-xs">
        <input
          type="checkbox"
          checked={precleanEnabled(preclean)}
          onChange={(event) =>
            setPreclean(
              event.target.checked
                ? { ...preclean, clean_missing_values: true }
                : {
                    ...preclean,
                    clean_missing_values: false,
                    drop_iqr_outliers: false,
                  },
            )
          }
        />
        Clean data first
      </label>
      <SupportDocsBar
        projectId={supportDocsDisabled ? "" : activeProjectId}
        disabled={supportDocsDisabled}
      />
      {projectChoice === "new" && (
        <span className="text-xs text-status-neutral">
          Support docs can be added after the project is created.
        </span>
      )}
    </div>
  );
  const startButton = (
    <Button
      variant="primary"
      onClick={() => launch.mutate()}
      disabled={blocked}
    >
      {launch.isPending ? "Starting…" : "Run analysis"}
    </Button>
  );

  return (
    <div
      className={
        layout === "route"
          ? "grid min-w-0 grid-cols-1 items-start gap-5 @2xl/launchpad:grid-cols-[minmax(0,1fr)_19rem]"
          : "flex min-w-0 flex-col gap-3"
      }
    >
      <div className="flex min-w-0 flex-col gap-4">
        <section
          aria-labelledby="launchpad-project-title"
          className="flex min-w-0 flex-col gap-2"
        >
          <div className="flex flex-wrap items-center gap-2">
            <h2 id="launchpad-project-title" className="text-sm font-semibold">
              Project
            </h2>
            <Hint label="Projects">
              Projects keep related sessions, uploads and support documents
              together. "No project" keeps this session in a private bucket.
            </Hint>
            {fixedProjectId ? (
              <span className="text-sm font-medium">
                {fixedProject?.name ?? fixedProjectId}
              </span>
            ) : (
              <>
                <select
                  aria-label="Project"
                  value={
                    projectChoice === "existing"
                      ? selectedProjectId
                      : projectChoice
                  }
                  disabled={projectLocked}
                  onChange={(event) => setDestination(event.target.value)}
                  className="min-w-48 rounded-base border border-border bg-bg px-2 py-1.5 text-sm disabled:opacity-50"
                >
                  <option value="unfiled">No project</option>
                  {availableProjects.map((project) => (
                    <option key={project.project_id} value={project.project_id}>
                      {project.name}
                    </option>
                  ))}
                  <option value="new">+ New project</option>
                </select>
                {projectChoice === "new" && (
                  <input
                    aria-label="Project name"
                    value={newProjectName}
                    maxLength={200}
                    autoFocus
                    disabled={projectLocked}
                    onChange={(event) => setNewProjectName(event.target.value)}
                    placeholder="Project name"
                    className="min-w-40 flex-1 rounded-base border border-border bg-bg px-2 py-1.5 text-sm disabled:opacity-50"
                  />
                )}
              </>
            )}
            {projectLocked && !fixedProjectId && (
              <span className="text-xs text-status-neutral">
                Locked after adding data.
              </span>
            )}
          </div>
          {projects.isError && !fixedProjectId && (
            <p role="alert" className="text-xs text-status-warn">
              Existing projects could not be loaded. You can still start without
              a project.
            </p>
          )}
          {unfiledProject.isError && isUnfiled && (
            <div
              role="alert"
              className="flex flex-wrap items-center gap-2 text-xs text-status-critical"
            >
              <span>Private storage could not be prepared.</span>
              <Button
                size="sm"
                variant="secondary"
                onClick={() => unfiledProject.mutate()}
                disabled={unfiledProject.isPending}
              >
                Retry
              </Button>
            </div>
          )}
        </section>

        {/* Uploads and reusable project files share one list: the required
         * input stays primary, regardless of where it came from. */}
        <section
          aria-labelledby="launchpad-data-title"
          {...dropHandlers}
          className={`flex min-w-0 flex-col overflow-hidden rounded-base border transition-colors ${
            dragging
              ? "border-primary bg-primary/5 ring-1 ring-primary/30"
              : "border-border bg-bg"
          }`}
        >
          <div className="flex items-center justify-between gap-3 border-b border-hairline px-3 py-2.5">
            <h2 id="launchpad-data-title" className="text-sm font-semibold">
              Data
            </h2>
            <Badge tone="brand">Required</Badge>
          </div>
          {rows.length > 0 && (
            <ul
              aria-label="Data files in this session"
              className="flex max-h-72 flex-col overflow-y-auto"
            >
              {rows.map((row) => (
                <FileRow key={row.key} row={row} />
              ))}
            </ul>
          )}
          {projectUploads.isPending && projectReady && (
            <p
              role="status"
              className="border-b border-hairline px-3 py-2 text-xs text-status-neutral"
            >
              Loading project files…
            </p>
          )}
          <div
            className={`flex flex-wrap items-center gap-x-3 gap-y-2 px-3 ${
              rows.length > 0 || projectUploads.isPending
                ? "border-t border-hairline py-2.5"
                : "py-8"
            }`}
          >
            <Button
              variant={rows.length > 0 ? "secondary" : "primary"}
              onClick={() => fileInputRef.current?.click()}
              disabled={!projectReady}
            >
              Choose CSV files
            </Button>
            <span className="text-sm text-status-neutral">
              {rows.length > 0
                ? "or drop more here"
                : "or drop CSVs here. One table per file; joins are detected automatically."}
            </span>
            <input
              ref={fileInputRef}
              aria-label="Data files (.csv)"
              type="file"
              accept=".csv"
              multiple
              onChange={onPickFiles}
              disabled={!projectReady}
              className="sr-only"
            />
          </div>
          {fileFeedback && (
            <p
              role="alert"
              className="border-t border-hairline px-3 py-2 text-xs text-status-warn"
            >
              {fileFeedback}
            </p>
          )}
          {projectUploads.isError && projectReady && (
            <p
              role="alert"
              className="border-t border-hairline px-3 py-2 text-xs text-status-warn"
            >
              Project files could not be listed. You can still upload a CSV.
            </p>
          )}
        </section>

        <label className="flex flex-col gap-1.5">
          <span className="text-sm font-semibold">
            Context{" "}
            <span className="font-normal text-status-neutral">(optional)</span>
          </span>
          <textarea
            aria-label="Business context"
            value={businessContext}
            onChange={(event) => setBusinessContext(event.target.value)}
            rows={3}
            placeholder="What is this data about, and what decision should the analysis support?"
            className="w-full resize-y rounded-base border border-border bg-bg p-2.5 text-sm outline-none placeholder:text-status-neutral"
          />
        </label>

        {layout === "inline" && (
          <div className="flex flex-col overflow-hidden rounded-base border border-border bg-surface">
            <div className="flex flex-wrap items-center gap-x-3 gap-y-2 px-3 py-2.5">
              {runOptions}
              <span className="ml-auto flex flex-wrap items-center gap-3">
                <span aria-live="polite" className="text-xs text-status-neutral">
                  {blockedReason ??
                    `${selectedFileCount} file${
                      selectedFileCount === 1 ? "" : "s"
                    } · report on · ${modelLabel}`}
                </span>
                {startButton}
              </span>
            </div>
            {precleanEnabled(preclean) && (
              <PrecleaningFields value={preclean} onChange={setPreclean} />
            )}
          </div>
        )}
      </div>

      {layout === "route" && (
        <Card
          as="section"
          tone="quiet"
          aria-labelledby="launchpad-run-title"
          className="flex min-w-0 flex-col gap-4 p-4 @2xl/launchpad:sticky @2xl/launchpad:top-4"
        >
          <SectionHeader
            level={2}
            title={<span id="launchpad-run-title">Run analysis</span>}
            description="Profile the data, check quality and generate a report."
          />
          {runSummary}
          <div className="border-t border-border pt-3">{runOptions}</div>
          {precleanEnabled(preclean) && (
            <PrecleaningFields value={preclean} onChange={setPreclean} />
          )}
          <div className="flex flex-wrap items-center justify-between gap-3 border-t border-border pt-3">
            <span
              aria-live="polite"
              className="min-w-0 flex-1 text-xs text-status-neutral"
            >
              {blockedReason ?? "Ready to start."}
            </span>
            {startButton}
          </div>
        </Card>
      )}

      <div
        className={
          layout === "route"
            ? "flex flex-col gap-3 @2xl/launchpad:col-span-2"
            : ""
        }
      >
        {conflict ? (
          <div
            role="alert"
            className="flex flex-wrap items-center gap-2 rounded-base border border-status-warn/45 bg-status-warn/5 p-3 text-sm"
          >
            <span>This session already has an active job.</span>
            <span className="text-status-neutral">{conflict.message}</span>
            {conflictRunId && (
              <Link
                to={sessionSectionPath(
                  activeProjectId,
                  conflictRunId,
                  "data-map",
                )}
                className="text-primary underline-offset-2 hover:underline"
              >
                Open session
              </Link>
            )}
          </div>
        ) : (
          launch.isError && (
            <p role="alert" className="text-sm text-status-critical">
              {launch.error instanceof Error
                ? launch.error.message
                : "Failed to start the session."}
            </p>
          )
        )}
      </div>
    </div>
  );
}

/* --------------------------------------------------------------- route shell */

/* The route gets a quiet sticky confirmation panel; Home embeds the same
 * composer inline, where a second column would compete with the dashboard. */
export function Component() {
  const { projectId } = useParams();
  return (
    <div className="mx-auto grid w-[90%] max-w-data grid-cols-1 items-start gap-5 p-6 lg:grid-cols-[minmax(0,1fr)_19rem]">
      <header className="flex flex-col gap-1 lg:col-span-2">
        <h1 className="text-xl font-semibold">New session</h1>
        <p className="text-sm text-status-neutral">
          Choose data, add any useful context, then run the analysis.
        </p>
      </header>
      <div className="@container/launchpad lg:col-span-2">
        <NewSessionPanel projectId={projectId} layout="route" />
      </div>
    </div>
  );
}
