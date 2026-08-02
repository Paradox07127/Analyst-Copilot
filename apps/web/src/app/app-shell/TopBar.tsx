import {
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
  type RefObject,
} from "react";
import { createPortal } from "react-dom";
import { Link, useLocation, useParams } from "react-router";
import { useSessionDetail } from "../../api/hooks";
import {
  Badge,
  Button,
  IconButton,
  Marquee,
} from "../../components/ui";
import {
  JOB_KIND_ACTIVITY,
  phaseProgress,
  type JobPhase,
  type PhaseState,
} from "../../api/job-events";
import {
  useJobActivity,
  type ActiveJob,
  type JobActivitySnapshot,
} from "../job-activity";
import { LlmStatusDot, useLlmStatus } from "./LlmStatus";
import { NavIcon } from "./nav-icons";
import {
  getEffectiveTheme,
  hasStoredTheme,
  setTheme,
  type Theme,
} from "../theme";
import { useWorkspaceFocus } from "../workspace-focus";

const HEADER_SEGMENT: Record<PhaseState, string> = {
  done: "bg-status-ok",
  active: "animate-breathe bg-status-warn",
  failed: "bg-status-critical",
  pending: "bg-track",
  skipped: "bg-track",
};

function matchesSession(job: ActiveJob, sessionId: string): boolean {
  return (
    job.sourceSessionId === sessionId ||
    job.sessionId === sessionId ||
    job.resultSessionId === sessionId
  );
}

function jobKindLabel(kind: string | undefined): string {
  if (kind === "auto_eda") return "EDA";
  if (kind === "report_generate") return "Report";
  if (kind === "question_exec") return "Question";
  if (kind === "cleaning_apply" || kind === "cleaning_preview") return "Cleanup";
  return kind ? "Task" : "Job";
}

function isDegraded(snapshot: JobActivitySnapshot | undefined): boolean {
  return Boolean(snapshot?.state.events.some(
    (event) =>
      event.type === "budget_degraded" || event.summary["degraded"] === true,
  ));
}

function jobPhaseLabel(
  phase: JobPhase,
  kind: string | undefined,
  currentPhase: ReturnType<typeof phaseProgress>[number] | undefined,
  degraded: boolean,
): string {
  const name = jobKindLabel(kind);
  if (degraded) return `${name} · degraded`;
  if (phase === "completed") return `${name} · complete`;
  if (phase === "failed" || phase === "cancelled") return `${name} · stopped`;
  if (phase === "queued" || phase === "connecting") return `${name} · queued`;
  return `${name} · ${currentPhase?.label ?? "working"}`;
}

function jobStatusLabel(phase: JobPhase, degraded: boolean): string {
  if (degraded) return "Degraded";
  if (phase === "completed") return "Completed";
  if (phase === "failed") return "Failed";
  if (phase === "cancelled") return "Stopped";
  if (phase === "queued" || phase === "connecting") return "Queued";
  if (phase === "disconnected") return "Connection lost";
  return "Running";
}

function readableEvent(type: string | undefined): string {
  if (!type) return "Waiting for job events";
  return type.replace(/[._]/g, " ");
}

function detailFromSummary(snapshot: JobActivitySnapshot | undefined): string | null {
  const event = [...(snapshot?.state.events ?? [])]
    .reverse()
    .find((candidate) => candidate.type === "budget_degraded") ??
    snapshot?.state.events.at(-1);
  if (!event) return null;
  for (const key of ["message", "reason", "detail", "error"]) {
    const value = event.summary[key];
    if (typeof value === "string" && value.trim()) return value;
  }
  return null;
}

/* Header content sits before the session navigation in DOM order. An absolute
 * child can therefore still be painted under later app content in some nested
 * layout contexts, even with a high local z-index. This portal has a single,
 * documented layer in the viewport instead:
 *
 *   base content / session navigation (0) < top bar (20) < Activity (40)
 *   < header tooltip (45) < modal dialogs (50).
 *
 * It is intentionally pointer-events-none: a hover card explains the status;
 * it must not steal focus or block the navigation below it. */
function HeaderJobTooltip({
  anchorRef,
  id,
  children,
}: {
  anchorRef: RefObject<HTMLDivElement | null>;
  id: string;
  children: ReactNode;
}) {
  const [style, setStyle] = useState<CSSProperties | null>(null);

  useLayoutEffect(() => {
    const updatePosition = () => {
      const rect = anchorRef.current?.getBoundingClientRect();
      if (!rect) return;
      const width = 320;
      const gap = 8;
      const margin = 12;
      setStyle({
        top: rect.bottom + gap,
        left: Math.max(
          margin,
          Math.min(rect.left, window.innerWidth - width - margin),
        ),
      });
    };
    updatePosition();
    window.addEventListener("resize", updatePosition);
    /* The app has independent scroll containers, so capture catches a scroll
     * from any one of them rather than only document scrolling. */
    window.addEventListener("scroll", updatePosition, true);
    return () => {
      window.removeEventListener("resize", updatePosition);
      window.removeEventListener("scroll", updatePosition, true);
    };
  }, [anchorRef]);

  if (!style) return null;
  return createPortal(
    <span
      id={id}
      role="tooltip"
      style={style}
      className="pointer-events-none animate-enter fixed z-[45] flex w-80 flex-col gap-2 rounded-base border border-border bg-bg p-3 text-left text-xs leading-relaxed text-text shadow-overlay"
    >
      {children}
    </span>,
    document.body,
  );
}

function TopBarJobProgress({
  job,
  snapshot,
}: {
  job: ActiveJob;
  snapshot: JobActivitySnapshot | undefined;
}) {
  const [detailsOpen, setDetailsOpen] = useState(false);
  const tooltipId = useId();
  const jobRef = useRef<HTMLDivElement>(null);
  const phase = snapshot?.state.phase ?? "connecting";
  const kind = snapshot?.kind;
  const phases = kind === "auto_eda" && snapshot ? phaseProgress(snapshot.state) : [];
  const current = phases.find((phase) => phase.state === "active");
  const mostRecentPhase = current ?? [...phases].reverse().find(
    (candidate) => candidate.state === "done" || candidate.state === "failed",
  );
  const degraded = isDegraded(snapshot);

  const states: PhaseState[] =
    kind === "auto_eda"
      ? phases.map((phase) => phase.state)
      : phase === "completed" || phase === "failed" || phase === "cancelled"
        ? Array.from({ length: 3 }, () =>
            phase === "completed" ? "done" : "failed",
          )
      : phase === "running"
        ? ["done", "active", "pending"]
        : ["active", "pending", "pending"];
  const label = jobPhaseLabel(phase, kind, current, degraded);
  const status = jobStatusLabel(phase, degraded);
  const lastEvent = snapshot?.state.events.at(-1);
  const detail =
    detailFromSummary(snapshot) ??
    (degraded ? "This job continued with a reduced capability or budget." : undefined) ??
    mostRecentPhase?.activity ??
    (kind ? JOB_KIND_ACTIVITY[kind] : undefined) ??
    "Waiting for this job to begin.";
  const dot = degraded
    ? "bg-status-warn"
    : phase === "completed"
      ? "bg-status-ok"
      : phase === "failed" || phase === "cancelled"
        ? "bg-status-critical"
        : phase === "running"
          ? "animate-breathe bg-status-warn"
          : "bg-status-neutral";

  return (
    <div
      ref={jobRef}
      className="flex w-52 min-w-0 cursor-help flex-col gap-1 rounded-base px-2 py-1 transition-colors hover:bg-surface focus:bg-surface xl:w-60"
      tabIndex={0}
      aria-label={`${label}. ${status}. Focus for job details.`}
      aria-describedby={detailsOpen ? tooltipId : undefined}
      onMouseEnter={() => setDetailsOpen(true)}
      onMouseLeave={() => setDetailsOpen(false)}
      onFocus={() => setDetailsOpen(true)}
      onBlur={() => setDetailsOpen(false)}
    >
      <span className="flex min-w-0 items-center gap-1.5">
        <span aria-hidden className={`size-2 shrink-0 rounded-full ${dot}`} />
        <Marquee className="text-xs leading-tight font-medium text-text xl:text-sm">
          {label}
        </Marquee>
      </span>
      <span className="flex items-center gap-1" aria-hidden="true">
        {states.map((state, index) => (
          <span
            key={index}
            className={`h-1 flex-1 rounded-full transition-colors duration-slow ease-out-quart ${HEADER_SEGMENT[state]}`}
          />
        ))}
      </span>
      {detailsOpen && <HeaderJobTooltip anchorRef={jobRef} id={tooltipId}>
        <span className="flex items-center justify-between gap-3">
          <span className="text-sm font-semibold">{jobKindLabel(kind)} job</span>
          <span className={`rounded-full px-2 py-0.5 text-[11px] font-medium ${
            degraded
              ? "bg-status-warn/15 text-status-warn"
              : phase === "completed"
                ? "bg-status-ok/15 text-status-ok"
                : phase === "failed" || phase === "cancelled"
                  ? "bg-status-critical/15 text-status-critical"
                  : "bg-surface text-status-neutral"
          }`}>
            {status}
          </span>
        </span>
        <span className="grid grid-cols-[76px_minmax(0,1fr)] gap-x-2 gap-y-1">
          <span className="text-status-neutral">Step</span>
          <span>{mostRecentPhase?.label ?? "Not started"}</span>
          <span className="text-status-neutral">Activity</span>
          <span>{detail}</span>
          <span className="text-status-neutral">Last event</span>
          <span className="capitalize">{readableEvent(lastEvent?.type)}</span>
          <span className="text-status-neutral">Job ID</span>
          <span className="truncate font-mono" title={job.jobId}>{job.jobId}</span>
        </span>
        {current?.items && (
          <span className="text-status-neutral">
            {current.items.total
              ? `${current.items.current} of ${current.items.total} items`
              : `${current.items.current} item${current.items.current === 1 ? "" : "s"} processed`}
          </span>
        )}
      </HeaderJobTooltip>}
    </div>
  );
}

/* The focused session can run several small jobs. Keep their terminal bars in
 * place instead of collapsing them after success, so the header is a compact
 * record of completed, queued, and degraded work. */
function TopBarProgress({
  jobs,
  snapshots,
  maxVisible = 2,
}: {
  jobs: ActiveJob[];
  snapshots: ReadonlyMap<string, JobActivitySnapshot>;
  maxVisible?: number;
}) {
  if (jobs.length === 0) return null;
  const visible = jobs.slice(0, maxVisible);
  return (
    <div aria-label="Current session job progress" className="flex items-center gap-2">
      {visible.map((job) => (
        <TopBarJobProgress
          key={job.jobId}
          job={job}
          snapshot={snapshots.get(job.jobId)}
        />
      ))}
      {jobs.length > visible.length && (
        <span className="text-xs font-medium text-status-neutral">+{jobs.length - visible.length}</span>
      )}
    </div>
  );
}

/* The header remains useful on a laptop: one widened card at ordinary desktop
 * widths, two only once there is enough room for both without overlapping the
 * route context or the controls on the right. */
function useHeaderJobLimit(): number {
  const media = "(min-width: 1536px)";
  const [limit, setLimit] = useState(() =>
    typeof window !== "undefined" && window.matchMedia?.(media).matches ? 2 : 1,
  );

  useEffect(() => {
    const query = window.matchMedia?.(media);
    if (!query) return;
    const sync = () => setLimit(query.matches ? 2 : 1);
    sync();
    query.addEventListener("change", sync);
    return () => query.removeEventListener("change", sync);
  }, []);

  return limit;
}

interface TopBarProps {
  inspectorCollapsed: boolean;
  onToggleInspector: () => void;
  onOpenSessions: () => void;
  onToggleSessions: () => void;
  sessionsCollapsed: boolean;
  onOpenSettings: () => void;
  showInspector: boolean;
}

export function TopBar({
  inspectorCollapsed,
  onToggleInspector,
  onOpenSessions,
  onToggleSessions,
  sessionsCollapsed,
  onOpenSettings,
  showInspector,
}: TopBarProps) {
  const route = useParams();
  const { pathname } = useLocation();
  const workspace = useWorkspaceFocus();
  const splitContext =
    workspace.mode === "split" ? workspace.activeContext : null;
  const projectId = splitContext?.projectId ?? route.projectId;
  const sessionId = splitContext?.sessionId ?? route.sessionId;
  const atHome = pathname === "/projects";
  const llm = useLlmStatus();
  const { trackedJobs, jobSnapshots } = useJobActivity();
  const [theme, setThemeState] = useState<Theme>(() => getEffectiveTheme());
  const runDetail = useSessionDetail(sessionId ?? "");

  /* A manual choice is persistent. Otherwise, update the default as local
   * time crosses the daytime/night-time boundary without a page reload. */
  useEffect(() => {
    const syncTimeTheme = () => {
      if (hasStoredTheme()) return;
      const next = getEffectiveTheme();
      document.documentElement.dataset["theme"] = next;
      setThemeState(next);
    };
    syncTimeTheme();
    const timer = window.setInterval(syncTimeTheme, 60_000);
    return () => window.clearInterval(timer);
  }, []);

  const toggleTheme = () => {
    const next: Theme = theme === "dark" ? "light" : "dark";
    setTheme(next);
    setThemeState(next);
  };

  /* The run id was the headline here, which meant the bar read
   * "run_1785039866191_k1det2" while the run's actual name sat unused in the
   * same payload. Title leads; the id stays visible but subordinate, because
   * it is what you paste into a bug report. */
  const runTitle = runDetail.data?.title;
  const headerJobLimit = useHeaderJobLimit();
  const focusedJobs = sessionId
    ? trackedJobs.filter((job) => matchesSession(job, sessionId))
    : [];

  return (
    /* Named because a page's own <header> also maps to the banner role, and
     * an unnamed query then matches two landmarks. */
    <header
      aria-label="Workbench"
      className="relative z-20 flex min-h-12 shrink-0 items-center gap-2 border-b border-border bg-surface px-3 py-2 sm:h-12 sm:gap-3 sm:px-4 sm:py-0"
    >
      {/* Wrapper, not `className="hidden md:…"` on the control: IconButton's
        * base sets inline-flex and Tailwind resolves that conflict by CSS
        * order, so the hidden never applied and the button stayed clickable on
        * mobile — where it wrote a rail-collapsed preference for a rail that
        * is not even rendered. */}
      <span className="hidden md:inline-flex">
        <IconButton
          label={sessionsCollapsed ? "Expand sessions" : "Collapse sessions"}
          onClick={onToggleSessions}
        >
          <RailGlyph collapsed={sessionsCollapsed} />
        </IconButton>
      </span>
      <nav
        aria-label="Context"
        className="hidden min-w-0 items-baseline gap-1.5 text-sm md:flex"
      >
        {atHome ? (
          <span className="font-medium">Home</span>
        ) : (
          <>
            <Link
              to="/projects"
              className="shrink-0 text-status-neutral hover:text-primary hover:underline"
            >
              {projectId ?? "—"}
            </Link>
            <span aria-hidden className="shrink-0 text-status-neutral/60">
              /
            </span>
            {sessionId ? (
              <>
                <Marquee className="font-medium" title={runTitle ?? sessionId}>
                  {runTitle ?? "Untitled session"}
                </Marquee>
                <span
                  className="shrink-0 font-mono text-xs text-status-neutral"
                  title="Session id"
                >
                  {sessionId}
                </span>
              </>
            ) : (
              <span className="text-status-neutral">no session open</span>
            )}
            {workspace.mode === "split" && (
              <Badge tone="neutral" variant="outline">
                Split · {workspace.activePane}
              </Badge>
            )}
          </>
        )}
      </nav>
      <div className="pointer-events-none absolute left-1/2 hidden -translate-x-1/2 lg:flex">
        {/* One generous status card fits without colliding with the context and
         * controls on ordinary laptops. Ultra-wide screens expose a second
         * card; the +N marker keeps every remaining job accounted for. */}
        <span className="pointer-events-auto">
          <TopBarProgress
            jobs={focusedJobs}
            snapshots={jobSnapshots}
            maxVisible={headerJobLimit}
          />
        </span>
      </div>
      <div className="ml-auto flex shrink-0 items-center gap-1 text-sm sm:gap-2">
        <Button size="sm" onClick={onOpenSessions} className="md:hidden">
          Sessions
        </Button>
        {/* The rail footer used to own this, which meant collapsing the rail
         * hid the one signal that says whether a session will call a paid model. */}
        {llm && (
          /* The dot never hides — it is the only signal for "will this session
           * call a paid model", and it costs 8px. Only its label drops. */
          <span className="flex items-center gap-1.5 text-xs text-status-neutral">
            <LlmStatusDot status={llm} />
            <Marquee className="hidden max-w-40 font-mono lg:inline">
              {llm.label}
            </Marquee>
          </span>
        )}
        <IconButton
          label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
          onClick={toggleTheme}
        >
          <ThemeGlyph theme={theme} />
        </IconButton>
        {showInspector && (
          <span className="hidden md:inline-flex">
            <IconButton
              label="Inspector"
              onClick={onToggleInspector}
              aria-pressed={!inspectorCollapsed}
            >
              <InspectorGlyph />
            </IconButton>
          </span>
        )}
        <IconButton label="Settings" onClick={onOpenSettings}>
          <NavIcon name="settings" />
        </IconButton>
      </div>
    </header>
  );
}

function InspectorGlyph() {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3.5" y="4" width="17" height="16" rx="2" />
      <path d="M15 4v16" />
    </svg>
  );
}

function RailGlyph({ collapsed }: { collapsed: boolean }) {
  return <svg aria-hidden="true" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round"><rect x="3.5" y="4" width="17" height="16" rx="2" /><path d="M9 4v16" />{collapsed ? <path d="m13 9 3 3-3 3" /> : <path d="m16 9-3 3 3 3" />}</svg>;
}

function ThemeGlyph({ theme }: { theme: Theme }) {
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
      {theme === "dark" ? (
        <path d="M20 14.5A8.5 8.5 0 1 1 9.5 4a7 7 0 0 0 10.5 10.5z" />
      ) : (
        <>
          <circle cx="12" cy="12" r="4" />
          <path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M19.1 4.9l-1.4 1.4M6.3 17.7l-1.4 1.4" />
        </>
      )}
    </svg>
  );
}
