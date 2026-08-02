import { useEffect, useId, useRef, useState, type DragEvent, type ReactNode } from "react";
import { useDraggable } from "@dnd-kit/core";
import { NavLink, useNavigate, useParams } from "react-router";
import type { ProjectSummary, SessionSummary } from "../../api/client";
import {
  useDeleteSession,
  useProjects,
  useRenameProject,
  useRenameSession,
  useReorderProjects,
  useSessions,
} from "../../api/hooks";
import {
  ErrorState,
  LoadingSkeleton,
} from "../../components/async-states";
import { Button, Chevron, IconButton, Marquee } from "../../components/ui";
import { newProjectSessionPath, newSessionPath, sessionBasePath } from "../paths";
import { getCollapsedProjects, setCollapsedProjects } from "./rail-prefs";
import {
  UNFILED_PROJECT_ID,
  UNFILED_RAIL_HEADING,
  UNFILED_ROW_LABEL,
} from "../unfiled";
import { useDialogFocus } from "../../components/use-dialog-focus";
import { DeleteProjectDialog } from "../../features/projects/DeleteProjectDialog";
import { sessionDragData, sessionDragId } from "../session-drag";
import { useWorkspaceFocus } from "../workspace-focus";

const SEARCH_DEBOUNCE_MS = 250;

/* One row shape for everything in the rail — the global entries, the project
 * headers and the session rows all share it, so the list reads as one column
 * instead of three stacked widgets. Hover uses --color-rail-hover rather than
 * bg-surface, which is only 2/255 from the rail in light and 4/255 in dark;
 * see the token's own note for the measurements. */
const ROW_BASE =
  "flex w-full min-w-0 items-center gap-2 truncate rounded-base px-2 py-1.5 text-left text-sm transition-colors duration-150 ease-out-quart";

const PROJECT_ROW_BASE =
  "flex w-full min-w-0 items-center gap-2 truncate rounded-base p-[2px] text-left text-sm transition-colors duration-150 ease-out-quart";

const rowClass = (active: boolean) =>
  `${ROW_BASE} ${
    active
      ? "bg-primary/10 font-medium text-primary"
      : "hover:bg-rail-hover"
  }`;

const navLinkClass = ({ isActive }: { isActive: boolean }) => rowClass(isActive);

function statusDotClass(status: string): string {
  const s = status.toLowerCase();
  if (["complete", "completed", "succeeded", "success", "ready"].includes(s))
    return "bg-status-ok";
  if (["running", "in_progress", "active", "pending"].includes(s))
    return "bg-status-warn";
  if (["failed", "error", "cancelled"].includes(s)) return "bg-status-critical";
  return "bg-status-neutral";
}

function isRunning(status: string): boolean {
  return ["running", "in_progress", "active"].includes(status.toLowerCase());
}

function formatTime(iso: string | null | undefined): string | null {
  if (!iso) return null;
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return null;
  return date.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}

/* Buckets are UTC calendar days: SessionSummary timestamps are always RFC 3339 UTC
 * (run_service._parse_datetime declares naive legacy values UTC), so comparing
 * in the browser's local day would move a run across a boundary.
 *
 * Keyed on updated_at, matching the order the API returns
 * (store.query_run_index_rows orders by updated_at desc). Bucketing by
 * created_at while the list is ordered by updated_at made a re-run land at the
 * top of the list under an "Earlier" caption, and let a caption repeat further
 * down the same list. */
function timeGroup(timestamp: string | null | undefined): string {
  if (!timestamp) return "Earlier";
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return "Earlier";
  const utcDay = (d: Date) => Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate());
  const days = Math.round((utcDay(new Date()) - utcDay(date)) / 86_400_000);
  if (days <= 0) return "Today";
  if (days === 1) return "Yesterday";
  if (days < 7) return "This week";
  return "Earlier";
}

function DeleteRunDialog({
  run,
  onClose,
}: {
  run: SessionSummary;
  onClose: () => void;
}) {
  const navigate = useNavigate();
  const { sessionId: activeRunId } = useParams();
  const remove = useDeleteSession(run.project_id);
  const { dialogRef, onKeyDown } = useDialogFocus(onClose);

  const confirm = () => {
    remove.mutate(run.session_id, {
      onSuccess: () => {
        onClose();
        /* Leaving a deleted run open would render 404s in every panel. */
        if (activeRunId === run.session_id) navigate("/projects");
      },
    });
  };

  return (
    <div
      ref={dialogRef}
      role="dialog"
      aria-modal="true"
      aria-label={`Delete session ${run.title ?? run.session_id}`}
      onKeyDown={onKeyDown}
      className="animate-fade fixed inset-0 z-50 flex items-center justify-center bg-scrim p-4"
    >
      <div className="animate-enter flex max-w-md flex-col gap-3 rounded-base border border-border bg-bg p-4">
        <h2 className="text-base font-semibold">Delete this session?</h2>
        <p className="text-sm">
          <span className="font-medium">{run.title ?? run.session_id}</span> and all
          of its artifacts, charts, report and chat transcript are removed from
          disk. This cannot be undone.
        </p>
        {remove.isError && <ErrorState error={remove.error} />}
        <div className="flex justify-end gap-2">
          <Button onClick={onClose}>Cancel</Button>
          <Button variant="danger" onClick={confirm} disabled={remove.isPending}>
            {remove.isPending ? "Deleting…" : "Delete session"}
          </Button>
        </div>
      </div>
    </div>
  );
}

function SessionItem({ run }: { run: SessionSummary }) {
  /* The date shown is the one the row is bucketed and sorted by, so the caption
   * above a row can never contradict the date printed inside it. */
  const time = formatTime(run.updated_at ?? run.created_at);
  const [confirming, setConfirming] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const label = run.title ?? run.session_id;
  const dragPayload = {
    projectId: run.project_id,
    sessionId: run.session_id,
    label,
  };
  const { attributes, isDragging, listeners, setNodeRef } = useDraggable({
    id: sessionDragId(dragPayload),
    data: sessionDragData(dragPayload),
    attributes: {
      role: "link",
      roleDescription: "draggable session",
    },
  });
  const workspace = useWorkspaceFocus();
  const focused =
    workspace.mode === "split" &&
    workspace.activeContext?.sessionId === run.session_id;

  const meta = [time, (run.dataset_names ?? []).join(", ")]
    .filter(Boolean)
    .join(" · ");

  return (
    <div className="group/run relative">
      <NavLink
        ref={setNodeRef}
        to={sessionBasePath(run.project_id, run.session_id)}
        draggable={false}
        {...attributes}
        {...listeners}
        aria-label={label}
        title="Open session, or drag it to the left or right workspace pane"
        className={({ isActive }) =>
          `${rowClass(isActive || focused)} select-none flex-col items-start gap-[1.5px] py-[2px] pr-2 ${isDragging ? "cursor-grabbing opacity-45" : "cursor-default"}`
        }
      >
        <span className="flex w-full min-w-0 items-center gap-1.5">
          <span
            aria-hidden
            className={`h-1.5 w-1.5 shrink-0 rounded-full ${statusDotClass(run.status)}`}
          />
          <Marquee className="flex-1">{label}</Marquee>
          {isRunning(run.status) && (
            <span className="ml-auto shrink-0 rounded-sm bg-status-warn/15 px-1 text-[10px] font-medium text-status-warn uppercase">
              Running
            </span>
          )}
        </span>
        {meta && (
          <Marquee className="w-full pl-3 text-xs font-normal text-status-neutral">
            {meta}
          </Marquee>
        )}
      </NavLink>
      {/* One masked cluster rather than two free-floating buttons. The row had
        * `pr-7`, which reserves 28px — enough for the ✕ but not for Rename, so a
        * long title ran underneath it. Opaque background over the row's own
        * hover fill hides the name behind the controls instead of reserving
        * dead space on every row. Same pattern as the project header below. */}
      <span className="absolute top-1.5 right-1 flex items-center gap-0.5 rounded-sm bg-rail-hover pl-1 opacity-0 transition-opacity focus-within:opacity-100 group-hover/run:opacity-100">
        <button
          type="button"
          aria-label={`Rename session ${label}`}
          onClick={() => setRenaming(true)}
          className="rounded-sm px-1 text-xs text-status-neutral hover:bg-primary/10 hover:text-primary"
        >
          Rename
        </button>
        <button
          type="button"
          aria-label={`Delete session ${label}`}
          onClick={() => setConfirming(true)}
          className="inline-flex size-5 items-center justify-center rounded-sm text-xs text-status-neutral hover:bg-status-critical/10 hover:text-status-critical"
        >
          ✕
        </button>
      </span>
      {confirming && (
        <DeleteRunDialog run={run} onClose={() => setConfirming(false)} />
      )}
      {renaming && <RenameSessionDialog run={run} onClose={() => setRenaming(false)} />}
    </div>
  );
}

function RenameSessionDialog({ run, onClose }: { run: SessionSummary; onClose: () => void }) {
  const [name, setName] = useState(run.title ?? run.session_id);
  const rename = useRenameSession(run.project_id);
  const { dialogRef, onKeyDown } = useDialogFocus(onClose);

  return (
    <div className="animate-fade fixed inset-0 z-50 grid place-items-center bg-scrim p-4" role="presentation">
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={`rename-session-${run.session_id}`}
        className="animate-enter w-full max-w-sm rounded-base border border-border bg-bg p-4 shadow-overlay"
        onKeyDown={onKeyDown}
      >
      <form
        onSubmit={(event) => {
          event.preventDefault();
          if (!name.trim()) return;
          rename.mutate(
            { sessionId: run.session_id, name },
            { onSuccess: onClose },
          );
        }}
      >
        <h2 id={`rename-session-${run.session_id}`} className="text-base font-semibold">Rename session</h2>
        <label className="mt-3 flex flex-col gap-1 text-sm font-medium">
          Session name
          <input
            autoFocus
            value={name}
            onChange={(event) => setName(event.target.value)}
            className="rounded-base border border-border bg-bg px-2 py-1.5 text-sm font-normal"
          />
        </label>
        {rename.isError && <p role="alert" className="mt-2 text-xs text-status-critical">Could not rename session.</p>}
        <div className="mt-4 flex justify-end gap-2">
          <Button type="button" onClick={onClose}>Cancel</Button>
          <Button type="submit" disabled={rename.isPending || !name.trim()}>
            {rename.isPending ? "Renaming…" : "Save name"}
          </Button>
        </div>
      </form>
      </div>
    </div>
  );
}

const RAIL_ICON_BUTTON =
  "inline-flex size-6 shrink-0 items-center justify-center rounded-sm text-status-neutral transition-colors duration-150 ease-out-quart hover:bg-primary/12 hover:text-primary";

/* A popover anchored to the button, not a block spliced into the list: the
 * inline version pushed every session below it down by its own height, so
 * opening a menu re-laid-out the thing you were aiming at. */
function ProjectMenu({
  project,
  open,
  onOpenChange,
  onDelete,
}: {
  project: ProjectSummary;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onDelete: () => void;
}) {
  const wrapRef = useRef<HTMLSpanElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const menuId = useId();
  const [name, setName] = useState(project.name);
  const rename = useRenameProject();

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (event: KeyboardEvent) => {
      /* Focus goes back to the trigger, not to the document: the menu item is
       * unmounted on close, so a keyboard user who escapes out of it would
       * otherwise land at the top of the page. */
      if (event.key === "Escape") {
        onOpenChange(false);
        triggerRef.current?.focus();
      }
    };
    const onPointerDown = (event: PointerEvent) => {
      if (!wrapRef.current?.contains(event.target as Node)) onOpenChange(false);
    };
    document.addEventListener("keydown", onKeyDown);
    document.addEventListener("pointerdown", onPointerDown);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.removeEventListener("pointerdown", onPointerDown);
    };
  }, [open, onOpenChange]);

  return (
    <span ref={wrapRef} className="relative inline-flex">
      <button
        ref={triggerRef}
        type="button"
        aria-label={`Manage ${project.name}`}
        aria-expanded={open}
        aria-controls={open ? menuId : undefined}
        onClick={() => onOpenChange(!open)}
        className={RAIL_ICON_BUTTON}
      >
        <MoreGlyph />
      </button>
      {open && (
        /* A plain popover, not role="menu": that role promises arrow-key
         * navigation and focus-on-open to a screen reader, and this is one
         * button. Declaring a contract we do not keep is worse than none. */
        <div
          id={menuId}
          aria-label={`Manage ${project.name}`}
          className="absolute top-full right-0 z-50 mt-1 flex min-w-36 flex-col rounded-base border border-border bg-bg p-1 shadow-overlay"
        >
          <form
            className="flex min-w-48 flex-col gap-1 p-1"
            onSubmit={(event) => {
              event.preventDefault();
              const nextName = name.trim();
              if (!nextName || nextName === project.name) {
                onOpenChange(false);
                return;
              }
              rename.mutate(
                { projectId: project.project_id, name: nextName },
                { onSuccess: () => onOpenChange(false) },
              );
            }}
          >
            <label className="text-xs text-status-neutral" htmlFor={`project-name-${menuId}`}>
              Project name
            </label>
            <input
              id={`project-name-${menuId}`}
              value={name}
              onChange={(event) => setName(event.target.value)}
              disabled={rename.isPending}
              className="w-full rounded-sm border border-border bg-bg px-2 py-1 text-xs"
            />
            <button
              type="submit"
              disabled={rename.isPending || name.trim() === "" || name.trim() === project.name}
              className="rounded-sm px-2 py-1 text-left text-xs text-primary hover:bg-primary/10 disabled:opacity-50"
            >
              {rename.isPending ? "Renaming…" : "Rename project"}
            </button>
            {rename.isError && (
              <p role="alert" className="px-1 text-xs text-status-critical">
                Could not rename project.
              </p>
            )}
          </form>
          <button
            type="button"
            onClick={() => {
              onDelete();
              onOpenChange(false);
            }}
            className="rounded-sm px-2 py-1.5 text-left text-xs whitespace-nowrap text-status-critical hover:bg-status-critical/10"
          >
            Delete project
          </button>
        </div>
      )}
    </span>
  );
}

/* Mirrors render_sidebar_sessions: a project header caption shows only when
 * several projects share the sidebar, and _time_group captions (dedup'd
 * against the previous session, like the source's `last_group`) show only
 * when it doesn't — never nest both. Unlike a hard-capped sidebar
 * (cap at 12 and point elsewhere for more),
 * this keeps the app's existing per-project pagination intact. */
function ProjectRunGroup({
  project,
  collapsed,
  onToggle,
  active,
  dragging,
  onDragStart,
  onDragOver,
  onDrop,
  onDragEnd,
}: {
  project: ProjectSummary;
  collapsed: boolean;
  onToggle: () => void;
  active: boolean;
  dragging: boolean;
  onDragStart: (event: DragEvent<HTMLDivElement>) => void;
  onDragOver: (event: DragEvent<HTMLElement>) => void;
  onDrop: (event: DragEvent<HTMLElement>) => void;
  onDragEnd: () => void;
}) {
  const runs = useSessions(project.project_id, "", !collapsed);
  const [menuOpen, setMenuOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);

  return (
    <section
      className="group/project flex flex-col"
      onDragOver={onDragOver}
      onDrop={onDrop}
    >
      {/* The whole header is the disclosure control — a 14px chevron was the
       * only hit target before, beside a name that looked clickable and was
       * not. The two actions overlay the row's right edge on hover rather than
       * reserving padding for it: at the rail's minimum width the 56px held
       * back for buttons nobody can see left the project name ~30px to live
       * in, measured before AppShell raised the rail's default size. */}
      <div
        draggable
        onDragStart={onDragStart}
        onDragEnd={onDragEnd}
        title="Drag to reorder projects"
        className={`group/header relative flex cursor-grab items-center rounded-base hover:bg-rail-hover active:cursor-grabbing ${
          dragging ? "opacity-45" : ""
        }`}
      >
        <button
          type="button"
          onClick={onToggle}
          aria-label={project.name}
          aria-expanded={!collapsed}
          className={`${PROJECT_ROW_BASE} ${active ? "font-semibold text-primary" : "font-medium"}`}
        >
          <Chevron open={!collapsed} />
          <span aria-hidden className="shrink-0 text-status-neutral/60">
            <DragGlyph />
          </span>
          <Marquee className="min-w-0 flex-1">{project.name}</Marquee>
        </button>
        {/* Manage before New: the destructive-capable menu is the rarer of the
         * two, and putting it outermost is what makes people mis-click it. */}
        <span className="absolute right-1 flex items-center gap-0.5 rounded-sm bg-rail-hover pl-1 opacity-0 transition-opacity focus-within:opacity-100 group-hover/header:opacity-100">
          <ProjectMenu
            project={project}
            open={menuOpen}
            onOpenChange={setMenuOpen}
            onDelete={() => setDeleting(true)}
          />
          <NavLink
            to={newProjectSessionPath(project.project_id)}
            className={RAIL_ICON_BUTTON}
            aria-label={`New session in ${project.name}`}
            title="New session"
          >
            <PlusGlyph />
          </NavLink>
        </span>
      </div>
      {!collapsed && <SessionGroupBody runs={runs} />}
      {deleting && <DeleteProjectDialog project={project} onClose={() => setDeleting(false)} />}
    </section>
  );
}

/* Shared by the project groups and the standalone group below them: same rows,
 * same time buckets, same pagination — only the header above differs. */
function SessionGroupBody({
  runs,
  nested = true,
}: {
  runs: ReturnType<typeof useSessions>;
  /* False for the Recent group: it is not inside a project, so it drops both
   * the guide rule and the day headers. "RECENT" over "TODAY" at the same
   * caption weight reads as a rendering mistake, and Recent already says the
   * list is newest-first. */
  nested?: boolean;
}) {
  let lastGroup: string | null = null;

  return (
    /* Indented behind a guide rule: without it a session row and a project
     * row sit at the same offset and the list reads as one flat sequence,
     * not as sessions belonging to a project. The sessions under Recent are
     * exactly that flat sequence — they belong to nothing — so they opt out. */
    <div
      className={
        nested
          ? "ml-4 flex flex-col border-l border-hairline pl-1"
          : "flex flex-col"
      }
    >
      {runs.isPending && <LoadingSkeleton lines={2} label="Loading sessions" />}
      {runs.isError && (
        <ErrorState error={runs.error} onRetry={() => runs.refetch()} />
      )}
      {runs.data &&
        (runs.data.pages[0]?.items.length === 0 ? (
          <p className="px-2 pb-1 text-xs text-status-neutral">
            No sessions yet.
          </p>
        ) : (
          <>
            {runs.data.pages.flatMap((page) =>
              page.items.map((run) => {
                /* Time buckets only for a single project,
                 * but that is an artifact of its 12-session cap: after
                 * capping, one project usually remains. Paginating instead
                 * of capping means both groupings can coexist. */
                let header: string | null = null;
                if (nested) {
                  const group = timeGroup(run.updated_at ?? run.created_at);
                  if (group !== lastGroup) {
                    header = group;
                    lastGroup = group;
                  }
                }
                return (
                  <div key={run.session_id} className="flex flex-col">
                    {header && (
                      <p className="px-2 pt-1.5 pb-0.5 text-[11px] font-medium tracking-wide text-status-neutral/70 uppercase">
                        {header}
                      </p>
                    )}
                    <SessionItem run={run} />
                  </div>
                );
              }),
            )}
            {/* Not <Button variant="ghost">: its hover is bg-surface,
              * which is 2/255 from the rail and so reads as no hover at
              * all here. The rail owns its own hover token. */}
            {runs.hasNextPage && (
              <button
                type="button"
                onClick={() => runs.fetchNextPage()}
                disabled={runs.isFetchingNextPage}
                className="ml-2 w-fit rounded-base px-2 py-1 text-xs font-medium text-status-neutral transition-colors duration-150 ease-out-quart hover:bg-rail-hover hover:text-text disabled:pointer-events-none disabled:opacity-50"
              >
                {runs.isFetchingNextPage ? "Loading…" : "Load more"}
              </button>
            )}
          </>
        ))}
    </div>
  );
}

/* Sessions that belong to no project, below every project group. They carry no
 * project menu: there is no folder here to rename or delete, and the global
 * "New session" at the top of the rail is already the way to add one. */
function StandaloneSessionGroup({
  collapsed,
  onToggle,
}: {
  collapsed: boolean;
  onToggle: () => void;
}) {
  /* Fetched even while collapsed: the group hides itself when empty, and it
   * cannot know it is empty without asking. */
  const runs = useSessions(UNFILED_PROJECT_ID);
  const count = runs.data?.pages.reduce((n, page) => n + page.items.length, 0) ?? 0;
  /* Nothing at all until there is something to show — rendering the header
   * while the request is in flight flashed an empty "Recent" heading into
   * every rail on every load. The error case still renders so its retry stays
   * reachable. */
  if (count === 0 && !runs.isError) return null;

  return (
    <section className="flex flex-col">
      {/* Caption weight, no hover fill, no row chrome: a project row and this
        * one must not read as the same kind of thing. Recent is a heading over
        * sessions that belong to nothing — there is no folder here to open,
        * rename or delete. */}
      <button
        type="button"
        onClick={onToggle}
        aria-label={UNFILED_RAIL_HEADING}
        aria-expanded={!collapsed}
        className="flex items-center gap-1 px-2 pt-3 pb-1 text-left text-[11px] font-medium tracking-wide text-status-neutral/70 uppercase hover:text-text"
      >
        <Chevron open={!collapsed} />
        <Marquee className="min-w-0 flex-1">{UNFILED_RAIL_HEADING}</Marquee>
      </button>
      {!collapsed && <SessionGroupBody runs={runs} nested={false} />}
    </section>
  );
}

function RailGlyph({ children }: { children: ReactNode }) {
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
      className="shrink-0"
    >
      {children}
    </svg>
  );
}

function SearchGlyph() {
  return (
    <RailGlyph>
      <circle cx="10.5" cy="10.5" r="5.5" />
      <path d="m15 15 4 4" />
    </RailGlyph>
  );
}

function PlusGlyph() {
  return (
    <RailGlyph>
      <path d="M12 5v14M5 12h14" />
    </RailGlyph>
  );
}

function HomeGlyph() {
  return (
    <RailGlyph>
      <path d="M3 7.5A1.5 1.5 0 0 1 4.5 6h4L10 8h9.5A1.5 1.5 0 0 1 21 9.5v8A1.5 1.5 0 0 1 19.5 19h-15A1.5 1.5 0 0 1 3 17.5z" />
    </RailGlyph>
  );
}

function MoreGlyph() {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24" width="16" height="16" className="shrink-0">
      <circle cx="5.5" cy="12" r="1.6" fill="currentColor" />
      <circle cx="12" cy="12" r="1.6" fill="currentColor" />
      <circle cx="18.5" cy="12" r="1.6" fill="currentColor" />
    </svg>
  );
}

function DragGlyph() {
  return (
    <svg aria-hidden="true" viewBox="0 0 12 16" width="8" height="12" fill="currentColor">
      <circle cx="3" cy="2.5" r="1" />
      <circle cx="9" cy="2.5" r="1" />
      <circle cx="3" cy="8" r="1" />
      <circle cx="9" cy="8" r="1" />
      <circle cx="3" cy="13.5" r="1" />
      <circle cx="9" cy="13.5" r="1" />
    </svg>
  );
}

/* Keyed by bucket id rather than a ProjectSummary: the sessions that belong to
 * no project are searchable too, and they have no summary row to pass. */
function SessionSearchResults({
  projectId,
  label,
  query,
  onChoose,
}: {
  projectId: string;
  label: string;
  query: string;
  onChoose: () => void;
}) {
  const runs = useSessions(projectId, query);
  const items = runs.data?.pages.flatMap((page) => page.items) ?? [];
  if (runs.isPending) return null;
  return items.map((run) => (
    <SessionSearchResult
      key={`${projectId}-${run.session_id}`}
      run={run}
      projectLabel={label}
      onChoose={onChoose}
    />
  ));
}

function SessionSearchResult({
  run,
  projectLabel,
  onChoose,
}: {
  run: SessionSummary;
  projectLabel: string;
  onChoose: () => void;
}) {
  const label = run.title ?? run.session_id;
  const payload = {
    projectId: run.project_id,
    sessionId: run.session_id,
    label,
  };
  const { attributes, isDragging, listeners, setNodeRef } = useDraggable({
    id: sessionDragId(payload, "search"),
    data: sessionDragData(payload),
    attributes: {
      role: "link",
      roleDescription: "draggable session",
    },
  });

  return (
    <NavLink
      ref={setNodeRef}
      to={sessionBasePath(run.project_id, run.session_id)}
      draggable={false}
      {...attributes}
      {...listeners}
      onClick={onChoose}
      className={`flex select-none flex-col gap-0.5 rounded-base px-3 py-2 transition-colors duration-150 ease-out-quart hover:bg-surface ${isDragging ? "cursor-grabbing opacity-45" : "cursor-default"}`}
    >
      <Marquee className="text-sm font-medium">{label}</Marquee>
      <Marquee className="text-xs text-status-neutral">
        {projectLabel} · {run.session_id}
        {run.dataset_names?.length ? ` · ${run.dataset_names.join(", ")}` : ""}
      </Marquee>
    </NavLink>
  );
}

function SessionSearchDialog({ onClose }: { onClose: () => void }) {
  const projects = useProjects();
  const [query, setQuery] = useState("");
  const debounced = useDebounced(query.trim(), SEARCH_DEBOUNCE_MS);
  const { dialogRef, onKeyDown } = useDialogFocus(onClose);

  return (
    <div ref={dialogRef} role="dialog" aria-modal="true" aria-label="Session search" onKeyDown={onKeyDown} className="animate-fade fixed inset-0 z-50 flex items-start justify-center bg-scrim p-4 pt-[10vh]">
      <div className="animate-enter flex max-h-[75vh] w-full max-w-xl flex-col overflow-hidden rounded-xl border border-border bg-bg shadow-overlay">
        <div className="flex items-center gap-2.5 border-b border-border px-4 py-3">
          <span className="text-status-neutral">
            <SearchGlyph />
          </span>
          <input
            autoFocus
            type="search"
            aria-label="Search sessions"
            placeholder="Search title, session ID, or dataset name"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            className="min-w-0 flex-1 bg-transparent text-sm outline-none"
          />
          <IconButton label="Close search" onClick={onClose}>
            ✕
          </IconButton>
        </div>
        <p className="border-b border-border px-4 py-2 text-xs text-status-neutral">
          Matches session titles, IDs and uploaded dataset names across every
          project, and the sessions that belong to none.
        </p>
        <div className="min-h-0 overflow-y-auto p-2">
          {!debounced ? (
            <p className="px-2 py-1.5 text-sm text-status-neutral">
              Type to search all sessions.
            </p>
          ) : projects.isPending ? (
            <LoadingSkeleton lines={3} label="Loading projects" />
          ) : (
            <div className="flex flex-col">
              {projects.data?.map((project) => (
                <SessionSearchResults
                  key={project.project_id}
                  projectId={project.project_id}
                  label={project.name}
                  query={debounced}
                  onChoose={onClose}
                />
              ))}
              {/* "across all projects" has to include the sessions that are in
                * none of them, or the promise above this list is false. */}
              <SessionSearchResults
                projectId={UNFILED_PROJECT_ID}
                label={UNFILED_ROW_LABEL}
                query={debounced}
                onChoose={onClose}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function SessionList() {
  const projects = useProjects();
  const reorder = useReorderProjects();
  const [collapsed, setCollapsed] = useState(getCollapsedProjects);
  const [draggingProjectId, setDraggingProjectId] = useState<string | null>(null);
  /* The list is re-rendered in the order the drop would produce, so the ghost
   * IS the preview. Highlighting a target row instead left the user guessing
   * whether the dragged project would land above or below it. */
  const [previewIds, setPreviewIds] = useState<string[] | null>(null);
  const { projectId } = useParams();
  const workspace = useWorkspaceFocus();
  const activeProjectId = workspace.activeContext?.projectId ?? projectId;

  const toggleProject = (projectId: string) => {
    const next = new Set(collapsed);
    if (!next.delete(projectId)) next.add(projectId);
    setCollapsed(next);
    setCollapsedProjects(next);
  };

  const serverIds = (projects.data ?? []).map((project) => project.project_id);
  const orderedProjects = previewIds
    ? previewIds.flatMap((id) => {
        const project = projects.data?.find((item) => item.project_id === id);
        return project ? [project] : [];
      })
    : (projects.data ?? []);

  const previewMove = (targetProjectId: string) => {
    if (!draggingProjectId || draggingProjectId === targetProjectId) return;
    const current = previewIds ?? serverIds;
    const from = current.indexOf(draggingProjectId);
    const to = current.indexOf(targetProjectId);
    if (from < 0 || to < 0 || from === to) return;
    const next = [...current];
    next.splice(from, 1);
    next.splice(to, 0, draggingProjectId);
    setPreviewIds(next);
  };

  const endDrag = () => {
    setDraggingProjectId(null);
    setPreviewIds(null);
  };

  const commitOrder = () => {
    const next = previewIds;
    endDrag();
    if (!next || next.every((id, index) => id === serverIds[index])) return;
    reorder.mutate(next);
  };

  if (projects.isPending) {
    return <LoadingSkeleton lines={4} label="Loading projects" />;
  }
  if (projects.isError) {
    return (
      <div className="px-2">
        <ErrorState error={projects.error} onRetry={() => projects.refetch()} />
      </div>
    );
  }
  if (projects.data.length === 0) {
    /* Not an empty state: a workspace can hold standalone sessions and no
     * project at all, and the old copy told those users they had nothing. */
    return (
      <div className="flex flex-col gap-1 px-2">
        <StandaloneSessionGroup
          collapsed={collapsed.has(UNFILED_PROJECT_ID)}
          onToggle={() => toggleProject(UNFILED_PROJECT_ID)}
        />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-1 px-2">
      {reorder.isError && (
        <p role="alert" className="px-1 text-xs text-status-critical">
          Could not save project order. Please try again.
        </p>
      )}
      {orderedProjects.map((project) => (
        <ProjectRunGroup
          key={project.project_id}
          project={project}
          collapsed={collapsed.has(project.project_id)}
          onToggle={() => toggleProject(project.project_id)}
          active={project.project_id === activeProjectId}
          dragging={draggingProjectId === project.project_id}
          onDragStart={(event) => {
            if (event.dataTransfer) {
              event.dataTransfer.effectAllowed = "move";
              event.dataTransfer.setData("text/plain", project.project_id);
            }
            setDraggingProjectId(project.project_id);
          }}
          onDragOver={(event) => {
            if (!draggingProjectId) return;
            // Without preventDefault the browser refuses the drop outright,
            // which is why releasing outside the header strip used to do
            // nothing at all.
            event.preventDefault();
            if (event.dataTransfer) event.dataTransfer.dropEffect = "move";
            previewMove(project.project_id);
          }}
          onDrop={(event) => {
            event.preventDefault();
            commitOrder();
          }}
          onDragEnd={endDrag}
        />
      ))}
      <StandaloneSessionGroup
        collapsed={collapsed.has(UNFILED_PROJECT_ID)}
        onToggle={() => toggleProject(UNFILED_PROJECT_ID)}
      />
    </div>
  );
}

/* Debounced so typing does not fire one request per keystroke; clearing the box
 * resets to the unfiltered list on the next tick. */
function useDebounced(value: string, delayMs: number): string {
  const [debounced, setDebounced] = useState(value);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => setDebounced(value), delayMs);
    return () => {
      if (timer.current) clearTimeout(timer.current);
    };
  }, [value, delayMs]);

  return debounced;
}

/* No footer: connection state and the settings entry moved to the top bar, so
 * there is exactly one of each in the app rather than one per surface. */
export function SessionRail() {
  const [searchOpen, setSearchOpen] = useState(false);

  return (
    <aside
      aria-label="Sessions"
      className="flex h-full flex-col bg-rail-bg"
    >
      {/* Start, find, browse — the three things you can do that are not "open
       * this session", in one block above the list they act on. Search was an
       * unlabelled icon alone on its own row and read as an orphan control. */}
      <nav
        className="flex shrink-0 flex-col gap-0.5 px-2 pt-3 pb-2"
      >
        {/* Search rides the New session row rather than owning one: it is one
         * glyph, and a full row for it cost as much height as the primary
         * action beside it. */}
        <div className="flex items-center gap-1">
          <NavLink
            aria-label="New session"
            to={newSessionPath()}
            className={({ isActive }) =>
              `${rowClass(isActive)} font-medium ${isActive ? "" : "text-primary"}`
            }
          >
            <PlusGlyph />
            New session
          </NavLink>
          <button
            type="button"
            onClick={() => setSearchOpen(true)}
            aria-label="Search sessions"
            title="Search sessions"
            className={RAIL_ICON_BUTTON}
          >
            <SearchGlyph />
          </button>
        </div>
        <NavLink to="/projects" end className={navLinkClass}>
          <HomeGlyph />
          Home
        </NavLink>
      </nav>
      <div className="min-h-0 flex-1 overflow-y-auto pb-2">
        <SessionList />
      </div>
      {searchOpen && <SessionSearchDialog onClose={() => setSearchOpen(false)} />}
    </aside>
  );
}

/* Collapsed rail: a hairline, not a 48px icon stub. The stub carried a second
 * expand control and a second settings gear next to the top bar's own; the top
 * bar owns both, so the stub only duplicated them. Connection state does go
 * away with the rail — the gear is the way back to it. */
export function CollapsedSessionRail() {
  return (
    <aside
      aria-label="Sessions (collapsed)"
      className="h-full w-1 shrink-0 border-r border-rail-border bg-rail-bg"
    />
  );
}
