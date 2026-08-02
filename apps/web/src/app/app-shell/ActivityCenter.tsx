import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
  type RefObject,
  type ReactNode,
} from "react";
import { createPortal } from "react-dom";
import { Link } from "react-router";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "../../api/client";
import { useJob, useSessionMetrics } from "../../api/hooks";
import {
  invalidateJobResultQueries,
  isJobKind,
} from "../../api/job-invalidation";
import {
  phaseProgress,
  TERMINAL_PHASES,
  useJobEvents,
  type JobEvent,
  type JobEventsState,
  type JobPhase,
} from "../../api/job-events";
import { SessionQualitySummary } from "../../components/session-quality-summary";
import {
  useJobActivity,
  type ActiveJob,
  type JobActivitySnapshot,
} from "../job-activity";
import { sessionSectionPath } from "../paths";
import { PipelineProgress } from "./PipelineProgress";
import { Marquee } from "../../components/ui";

const PHASE_LABELS: Record<JobPhase, string> = {
  connecting: "Connecting…",
  queued: "Queued",
  running: "Running",
  completed: "Completed",
  failed: "Failed",
  cancelled: "Cancelled",
  disconnected: "Stream lost",
};

/* Terminal and asking nothing of the user — the only runs safe to fold away. */
const QUIET_PHASES: ReadonlySet<JobPhase> = new Set(["completed", "cancelled"]);

const CANCELLABLE_KINDS = new Set([
  "auto_eda",
  "cleaning_preview",
  "cleaning_apply",
  "dataset_distributions",
  "custom_chart",
]);

const LAUNCHER_POSITION_KEY = "eda.layout.activity-position";
const LAUNCHER_SIZE = 48;
const VIEWPORT_GAP = 16;

const ACTIVITY_SECTIONS = [
  { id: "runs", label: "Runs" },
  { id: "activity", label: "Activity" },
  { id: "events", label: "Event log" },
] as const;

type PanelSection = (typeof ACTIVITY_SECTIONS)[number]["id"];

function phaseTone(phase: JobPhase): string {
  if (phase === "completed") return "text-status-ok";
  if (phase === "failed" || phase === "disconnected")
    return "text-status-critical";
  if (phase === "cancelled") return "text-status-neutral";
  return "text-status-warn";
}

function phaseDot(phase: JobPhase): string {
  if (phase === "completed") return "bg-status-ok";
  if (phase === "failed" || phase === "disconnected")
    return "bg-status-critical";
  if (phase === "cancelled") return "bg-status-neutral";
  return "bg-status-warn";
}

function launcherProgress(
  job: JobEventsState,
  kind: string | undefined,
): number | null {
  if (job.phase === "completed") return 100;
  if (kind !== "auto_eda") return null;

  const phases = phaseProgress(job);
  const units = phases.reduce((total, phase) => {
    if (phase.state === "done") return total + 1;
    if (
      phase.state === "active" &&
      phase.items?.total &&
      phase.items.total > 0
    ) {
      return total + Math.min(1, phase.items.current / phase.items.total);
    }
    return total;
  }, 0);
  return Math.round((units / phases.length) * 100);
}

function formatEventTime(event: JobEvent): string {
  if (!event.timestamp) return "";
  const date = new Date(event.timestamp);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleTimeString(undefined, { hour12: false });
}

function ActivityGlyph({ active }: { active: boolean }) {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 24 24"
      width="22"
      height="22"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="12" cy="12" r="8.5" />
      <path d="M7.5 13h2l1.4-4 2.2 7 1.4-4H17" />
      {active && <path d="M12 3.5a8.5 8.5 0 0 1 8.5 8.5" />}
    </svg>
  );
}

function CompletionGlyph() {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 24 24"
      width="24"
      height="24"
      fill="none"
      stroke="currentColor"
      strokeWidth="3"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="m5 12.5 4.2 4.2L19 7.5" />
    </svg>
  );
}

function CloseGlyph() {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 24 24"
      width="18"
      height="18"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
    >
      <path d="M6 6l12 12M18 6 6 18" />
    </svg>
  );
}

function EventLog({ events }: { events: JobEvent[] }) {
  const scrollRef = useRef<HTMLOListElement>(null);
  const [following, setFollowing] = useState(true);
  /* Rows already present when the panel opened must not animate: on a finished
   * run that is 200 of them highlighting at once, which says "everything just
   * happened". Only arrivals after mount get the tint. */
  const backlog = useRef(events.length);

  useEffect(() => {
    const element = scrollRef.current;
    if (element && following) element.scrollTop = element.scrollHeight;
  }, [events.length, following]);

  if (events.length === 0) {
    return (
      <p className="px-3 py-4 font-mono text-xs text-status-neutral">
        Waiting for events…
      </p>
    );
  }

  const jumpToLatest = () => {
    const element = scrollRef.current;
    if (element) element.scrollTop = element.scrollHeight;
    setFollowing(true);
  };

  return (
    <div className="relative flex min-h-0 flex-1 flex-col">
      <ol
        ref={scrollRef}
        aria-label="Job event log"
        role="log"
        onScroll={(event) => {
          const element = event.currentTarget;
          const distance =
            element.scrollHeight - element.scrollTop - element.clientHeight;
          setFollowing(distance < 24);
        }}
        className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-3 py-2 text-xs"
      >
        {events.map((event, index) => (
          <li
            key={event.event_id}
            className={`grid grid-cols-[4.5rem_auto_minmax(0,1fr)] items-start gap-x-2 gap-y-1 border-b border-border/60 py-2 last:border-b-0 ${
              index >= backlog.current ? "animate-arrive" : ""
            }`}
          >
            <span className="pt-0.5 font-mono text-[11px] tabular-nums text-status-neutral">
              {formatEventTime(event)}
            </span>
            <span className="rounded-sm bg-code-bg px-1.5 py-0.5 font-mono text-[10px] font-medium text-text">
              {event.type}
            </span>
            <Marquee className="min-w-0 py-0.5 font-mono leading-5 text-status-neutral">
              {event.name}
            </Marquee>
            {Object.keys(event.summary).length > 0 && (
              <details className="col-span-2 col-start-2 min-w-0">
                <summary className="cursor-pointer text-[11px] text-status-neutral hover:text-text">
                  Event details
                </summary>
                <pre className="mt-1 max-h-32 overflow-auto rounded-base border border-border bg-code-bg p-2 font-mono text-[11px] leading-4 text-status-neutral">
                  {JSON.stringify(event.summary, null, 2)}
                </pre>
              </details>
            )}
          </li>
        ))}
      </ol>
      {!following && (
        <button
          type="button"
          onClick={jumpToLatest}
          className="absolute right-3 bottom-3 rounded-full border border-border bg-bg px-3 py-1 text-xs shadow-panel hover:bg-surface"
        >
          Jump to latest ↓
        </button>
      )}
    </div>
  );
}

function CancelButton({ jobId, disabled }: { jobId: string; disabled: boolean }) {
  const [confirming, setConfirming] = useState(false);
  const cancel = useMutation({
    mutationFn: () => api.cancelJob(jobId),
    onSettled: () => setConfirming(false),
  });

  if (disabled) return null;
  if (!confirming) {
    return (
      <button
        type="button"
        onClick={() => setConfirming(true)}
        className="rounded-base border border-status-critical/50 px-2 py-1 text-xs text-status-critical hover:bg-status-critical/10"
      >
        Cancel job
      </button>
    );
  }

  return (
    <span className="flex flex-wrap items-center gap-1.5">
      <span className="text-xs text-status-critical">Cancel this job?</span>
      <button
        type="button"
        onClick={() => cancel.mutate()}
        disabled={cancel.isPending}
        className="rounded-base border border-status-critical/50 px-2 py-1 text-xs font-medium text-status-critical hover:bg-status-critical/10 disabled:opacity-60"
      >
        {cancel.isPending ? "Cancelling…" : "Confirm cancel"}
      </button>
      <button
        type="button"
        onClick={() => setConfirming(false)}
        className="rounded-base border border-border px-2 py-1 text-xs hover:bg-code-bg"
      >
        Keep running
      </button>
    </span>
  );
}

interface ActivityPanelFrameProps {
  children: ReactNode;
  eventCount: number;
  launcherVisible: boolean;
  onClose: () => void;
  onToggleLauncher: () => void;
  onKeyDown: (event: ReactKeyboardEvent<HTMLElement>) => void;
  status: ReactNode;
  trackingLabel?: string;
  geometry: OverlayGeometry;
  section: PanelSection;
  onSectionChange: (section: PanelSection) => void;
}

function ActivityPanelFrame({
  children,
  eventCount,
  launcherVisible,
  onClose,
  onToggleLauncher,
  onKeyDown,
  status,
  trackingLabel,
  geometry,
  section,
  onSectionChange,
}: ActivityPanelFrameProps) {
  return (
    <section
      id="activity-center-panel"
      role="dialog"
      aria-label="Activity"
      aria-modal="false"
      onKeyDown={onKeyDown}
      style={geometry.panelStyle}
      className="animate-enter pointer-events-auto absolute flex h-[min(30rem,calc(100dvh-1.5rem))] max-h-[min(35rem,calc(100dvh-1.5rem))] max-w-[calc(100vw-1.5rem)] resize flex-col overflow-hidden rounded-xl border border-border bg-bg shadow-overlay"
    >
      <header className="flex shrink-0 items-start gap-3 border-b border-border bg-surface px-4 py-3">
        <span className="mt-0.5 text-primary">
          <ActivityGlyph active={eventCount > 0} />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <h2 id="activity-center-title" className="text-sm font-semibold">
              Agent activity
            </h2>
            {eventCount > 0 && (
              <span className="rounded-full bg-code-bg px-2 py-0.5 font-mono text-[11px] text-status-neutral">
                {eventCount} events
              </span>
            )}
          </div>
          <div className="mt-0.5 text-xs">{status}</div>
          {trackingLabel && (
            <p className="mt-0.5 min-w-0 font-mono text-[11px] text-status-neutral">
              <Marquee title={trackingLabel}>{trackingLabel}</Marquee>
            </p>
          )}
        </div>
        <button
          type="button"
          aria-label="Close activity"
          onClick={onClose}
          className="rounded-base p-1 text-status-neutral hover:bg-code-bg hover:text-text focus-visible:outline-2 focus-visible:outline-primary"
        >
          <CloseGlyph />
        </button>
      </header>
      <div
        role="tablist"
        aria-label="Activity center sections"
        className="flex shrink-0 gap-1 border-b border-border bg-surface px-3"
      >
        {ACTIVITY_SECTIONS.map((item, index) => (
          <button
            key={item.id}
            id={`activity-center-${item.id}-tab`}
            type="button"
            role="tab"
            aria-selected={section === item.id}
            aria-controls={`activity-center-${item.id}-panel`}
            tabIndex={section === item.id ? 0 : -1}
            onClick={() => onSectionChange(item.id)}
            onKeyDown={(event) => {
              if (event.key !== "ArrowLeft" && event.key !== "ArrowRight")
                return;
              event.preventDefault();
              const offset = event.key === "ArrowRight" ? 1 : -1;
              const next =
                ACTIVITY_SECTIONS[
                  (index + offset + ACTIVITY_SECTIONS.length) %
                    ACTIVITY_SECTIONS.length
                ]!;
              onSectionChange(next.id);
              requestAnimationFrame(() =>
                document
                  .getElementById(`activity-center-${next.id}-tab`)
                  ?.focus(),
              );
            }}
            className={`border-b-2 px-3 py-2 text-xs font-medium transition-colors ${
              section === item.id
                ? "border-primary text-text"
                : "border-transparent text-status-neutral hover:text-text"
            }`}
          >
            {item.label}
            {item.id === "events" && eventCount > 0 && (
              <span className="ml-1.5 rounded-full bg-code-bg px-1.5 py-0.5 font-mono text-[10px]">
                {eventCount}
              </span>
            )}
          </button>
        ))}
      </div>
      <div
        id={`activity-center-${section}-panel`}
        role="tabpanel"
        aria-labelledby={`activity-center-${section}-tab`}
        className="flex min-h-0 flex-1 flex-col"
      >
        {children}
      </div>
      <footer className="flex shrink-0 items-center justify-between border-t border-border bg-surface px-3 py-2">
        <span className="text-[11px] text-status-neutral">
          Ctrl/⌘ + Shift + L
        </span>
        <button
          type="button"
          onClick={onToggleLauncher}
          className="rounded-base px-2 py-1 text-xs text-status-neutral hover:bg-code-bg hover:text-text"
        >
          {launcherVisible ? "Hide floating button" : "Show floating button"}
        </button>
      </footer>
    </section>
  );
}

function Launcher({
  buttonRef,
  runCount,
  geometry,
  onClick,
  open,
  phase,
  progress,
}: {
  buttonRef?: RefObject<HTMLButtonElement | null>;
  /* Runs still needing attention, not events seen. A finished analysis streams
   * hundreds of events; counting those pinned "99+" to the launcher for the
   * rest of the session. */
  runCount: number;
  geometry: OverlayGeometry;
  onClick: () => void;
  open: boolean;
  phase?: JobPhase;
  progress?: number | null;
}) {
  const active = phase !== undefined && !TERMINAL_PHASES.has(phase);
  const radius = 20;
  const circumference = 2 * Math.PI * radius;
  const progressLabel =
    phase === "completed"
      ? "✓"
      : phase === "failed" || phase === "disconnected"
        ? "!"
        : phase === "cancelled"
          ? "–"
          : progress !== null && progress !== undefined
            ? `${progress}%`
            : null;

  return (
    <button
      ref={buttonRef}
      type="button"
      aria-label={
        open ? "Close activity from floating button" : "Open activity"
      }
      aria-expanded={open}
      aria-controls="activity-center-panel"
      onClick={onClick}
      onPointerDown={geometry.dragHandlers.onPointerDown}
      onPointerMove={geometry.dragHandlers.onPointerMove}
      onPointerUp={geometry.dragHandlers.onPointerUp}
      onPointerCancel={geometry.dragHandlers.onPointerCancel}
      title="Drag to move · click to open Activity"
      style={{
        left: geometry.launcherPosition.x,
        top: geometry.launcherPosition.y,
        borderRadius: "9999px",
        touchAction: "none",
      }}
      className={`pointer-events-auto absolute flex h-12 w-12 cursor-grab items-center justify-center rounded-full border bg-bg shadow-overlay transition-[transform,box-shadow] duration-200 hover:scale-105 active:cursor-grabbing focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary ${phase ? phaseTone(phase) : "border-primary/40 text-primary"}`}
    >
      {active && (
        <svg
          aria-hidden="true"
          data-testid="activity-working-orbit"
          viewBox="0 0 56 56"
          className="pointer-events-none absolute -inset-1 size-14 animate-orbit"
        >
          <circle
            cx="28"
            cy="28"
            r="25"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeDasharray="18 139"
          />
        </svg>
      )}
      <svg
        aria-hidden="true"
        viewBox="0 0 48 48"
        className="absolute inset-0 h-full w-full -rotate-90"
      >
        <circle
          cx="24"
          cy="24"
          r={radius}
          fill="none"
          stroke="currentColor"
          strokeOpacity="0.16"
          strokeWidth="3"
        />
        {progress !== null && progress !== undefined && (
          <circle
            cx="24"
            cy="24"
            r={radius}
            fill="none"
            stroke="currentColor"
            strokeWidth="3"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={circumference * (1 - progress / 100)}
            /* No breathe here: a ring that is visibly advancing already says
             * "alive", and throbbing at the same time fights the sweep. */
            className="motion-progress transition-[stroke-dashoffset] duration-slow ease-out-quart"
          />
        )}
      </svg>
      {phase === "completed" ? (
        <span className="relative flex items-center justify-center">
          <CompletionGlyph />
        </span>
      ) : progressLabel ? (
        <span className="relative font-mono text-[11px] font-semibold tabular-nums">
          {progressLabel}
          {progress !== null && progress !== undefined && (
            <span className="sr-only"> complete</span>
          )}
        </span>
      ) : (
        <span className={active ? "relative animate-breathe" : "relative"}>
          <ActivityGlyph active={active} />
        </span>
      )}
      {runCount > 0 && (
        <span
          aria-hidden="true"
          data-testid="activity-run-count"
          className={`absolute -top-1 -right-1 flex min-h-5 min-w-5 items-center justify-center rounded-full border-2 border-bg px-1 text-[10px] font-semibold text-white ${
            phase ? phaseDot(phase) : "bg-status-neutral"
          }`}
        >
          {runCount > 99 ? "99+" : runCount}
        </span>
      )}
    </button>
  );
}

interface LauncherPosition {
  x: number;
  y: number;
}

interface OverlayGeometry {
  launcherPosition: LauncherPosition;
  panelStyle: CSSProperties;
  dragHandlers: {
    onPointerDown: (event: ReactPointerEvent<HTMLButtonElement>) => void;
    onPointerMove: (event: ReactPointerEvent<HTMLButtonElement>) => void;
    onPointerUp: (event: ReactPointerEvent<HTMLButtonElement>) => void;
    onPointerCancel: (event: ReactPointerEvent<HTMLButtonElement>) => void;
  };
  consumeDraggedClick: () => boolean;
}

function clampLauncherPosition(
  position: LauncherPosition,
): LauncherPosition {
  return {
    x: Math.min(
      Math.max(VIEWPORT_GAP, position.x),
      Math.max(VIEWPORT_GAP, window.innerWidth - LAUNCHER_SIZE - VIEWPORT_GAP),
    ),
    y: Math.min(
      Math.max(VIEWPORT_GAP, position.y),
      Math.max(VIEWPORT_GAP, window.innerHeight - LAUNCHER_SIZE - VIEWPORT_GAP),
    ),
  };
}

function readLauncherPosition(): LauncherPosition {
  const fallback = {
    x: window.innerWidth - LAUNCHER_SIZE - VIEWPORT_GAP,
    y: window.innerHeight - LAUNCHER_SIZE - VIEWPORT_GAP,
  };
  try {
    const stored = JSON.parse(
      window.localStorage.getItem(LAUNCHER_POSITION_KEY) ?? "null",
    ) as Partial<LauncherPosition> | null;
    if (
      stored &&
      typeof stored.x === "number" &&
      typeof stored.y === "number"
    ) {
      return clampLauncherPosition({ x: stored.x, y: stored.y });
    }
  } catch {
    // A malformed preference should only reset the launcher to its default.
  }
  return clampLauncherPosition(fallback);
}

function useOverlayGeometry(): OverlayGeometry {
  const [launcherPosition, setLauncherPosition] = useState(readLauncherPosition);
  const positionRef = useRef(launcherPosition);
  positionRef.current = launcherPosition;
  const drag = useRef<{
    pointerId: number;
    startX: number;
    startY: number;
    origin: LauncherPosition;
    moved: boolean;
  } | null>(null);
  const draggedClick = useRef(false);
  const [, setViewportVersion] = useState(0);

  useEffect(() => {
    const update = () => {
      setLauncherPosition((current) => {
        const next = clampLauncherPosition(current);
        positionRef.current = next;
        return next;
      });
      setViewportVersion((current) => current + 1);
    };
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, []);

  const finishDrag = (event: ReactPointerEvent<HTMLButtonElement>) => {
    const current = drag.current;
    if (!current || current.pointerId !== event.pointerId) return;
    if (current.moved) {
      draggedClick.current = true;
      window.localStorage.setItem(
        LAUNCHER_POSITION_KEY,
        JSON.stringify(positionRef.current),
      );
    }
    drag.current = null;
    if (event.currentTarget.hasPointerCapture?.(event.pointerId)) {
      event.currentTarget.releasePointerCapture?.(event.pointerId);
    }
  };

  const dragHandlers = {
    onPointerDown: (event: ReactPointerEvent<HTMLButtonElement>) => {
      if (event.button !== 0) return;
      drag.current = {
        pointerId: event.pointerId,
        startX: event.clientX,
        startY: event.clientY,
        origin: launcherPosition,
        moved: false,
      };
      event.currentTarget.setPointerCapture?.(event.pointerId);
    },
    onPointerMove: (event: ReactPointerEvent<HTMLButtonElement>) => {
      const current = drag.current;
      if (!current || current.pointerId !== event.pointerId) return;
      const dx = event.clientX - current.startX;
      const dy = event.clientY - current.startY;
      if (!current.moved && Math.hypot(dx, dy) < 4) return;
      current.moved = true;
      const next = clampLauncherPosition({
        x: current.origin.x + dx,
        y: current.origin.y + dy,
      });
      positionRef.current = next;
      setLauncherPosition(next);
    },
    onPointerUp: finishDrag,
    onPointerCancel: finishDrag,
  };

  const launcherOnRight =
    launcherPosition.x + LAUNCHER_SIZE / 2 >= window.innerWidth / 2;
  const launcherOnBottom =
    launcherPosition.y + LAUNCHER_SIZE / 2 >= window.innerHeight / 2;
  const horizontalInset = launcherOnRight
    ? Math.max(
        12,
        window.innerWidth - launcherPosition.x - LAUNCHER_SIZE,
      )
    : Math.max(12, launcherPosition.x);
  const panelWidth = Math.max(
    240,
    Math.min(416, window.innerWidth - horizontalInset - 12),
  );
  const panelStyle: CSSProperties = {
    width: panelWidth,
    maxHeight: Math.max(
      180,
      launcherOnBottom
        ? launcherPosition.y - 24
        : window.innerHeight -
            launcherPosition.y -
            LAUNCHER_SIZE -
            24,
    ),
    ...(launcherOnRight
      ? { right: horizontalInset }
      : { left: horizontalInset }),
    ...(launcherOnBottom
      ? { bottom: window.innerHeight - launcherPosition.y + 12 }
      : { top: launcherPosition.y + LAUNCHER_SIZE + 12 }),
  };

  return {
    launcherPosition,
    panelStyle,
    dragHandlers,
    consumeDraggedClick: () => {
      if (!draggedClick.current) return false;
      draggedClick.current = false;
      return true;
    },
  };
}

function ActivityPortal({
  children,
  onDismiss,
}: {
  children: ReactNode;
  onDismiss?: () => void;
}) {
  return createPortal(
    /* The layer itself stays click-through so the page underneath keeps
     * working; this one span of it does not, purely to catch a click aimed
     * past the panel. Without it the only way out is the panel's own X, while
     * the panel covers page controls. */
    <div className="pointer-events-none fixed inset-0 z-40">
      {onDismiss && (
        <div
          aria-hidden
          onPointerDown={onDismiss}
          className="pointer-events-auto absolute inset-0"
        />
      )}
      {children}
    </div>,
    document.body,
  );
}

function useActivityKeyboardShortcut() {
  const { panelOpen, setPanelOpen } = useJobActivity();

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (
        (event.ctrlKey || event.metaKey) &&
        event.shiftKey &&
        event.key.toLowerCase() === "l"
      ) {
        event.preventDefault();
        setPanelOpen(!panelOpen);
        return;
      }
      /* Measured at 1440x900 the open panel sits over four page elements,
       * two of them buttons — so it has to be dismissible the way every other
       * overlay in the app is, not only by its own close control. */
      if (event.key === "Escape" && panelOpen) setPanelOpen(false);
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [panelOpen, setPanelOpen]);
}

function activityKindLabel(kind: string | undefined): string {
  if (!kind) return "Background task";
  return kind
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function JobObserver({
  trackedJob,
  onSnapshot,
}: {
  trackedJob: ActiveJob;
  onSnapshot: (jobId: string, snapshot: JobActivitySnapshot) => void;
}) {
  const state = useJobEvents(trackedJob.jobId, trackedJob.eventsUrl);
  const kind = useJob(trackedJob.jobId).data?.kind;
  const queryClient = useQueryClient();
  const { claimSettlement } = useJobActivity();

  useEffect(() => {
    onSnapshot(trackedJob.jobId, { state, kind });
  }, [kind, onSnapshot, state, trackedJob.jobId]);

  useEffect(() => {
    if (
      !TERMINAL_PHASES.has(state.phase) ||
      !isJobKind(kind) ||
      !claimSettlement(trackedJob.jobId)
    )
      return;
    void invalidateJobResultQueries(queryClient, kind, trackedJob);
  }, [
    state.phase,
    kind,
    queryClient,
    trackedJob,
    claimSettlement,
  ]);

  return null;
}

function RunRow({
  trackedJob,
  snapshot,
  selected,
  onSelect,
  onDismiss,
}: {
  trackedJob: ActiveJob;
  snapshot: JobActivitySnapshot | undefined;
  selected: boolean;
  onSelect: (jobId: string) => void;
  onDismiss: (jobId: string) => void;
}) {
  const phase = snapshot?.state.phase ?? "connecting";
  const terminal = TERMINAL_PHASES.has(phase);
  const resultSessionId = trackedJob.resultSessionId ?? trackedJob.sessionId;
  return (
    <li
      className={`rounded-lg border p-3 transition-colors ${
        selected
          ? "border-primary/50 bg-primary/5"
          : "border-border bg-surface hover:border-primary/30"
      }`}
    >
      <button
        type="button"
        aria-label={`View run ${trackedJob.jobId}`}
        onClick={() => onSelect(trackedJob.jobId)}
        className="flex w-full items-start gap-3 text-left focus-visible:outline-2 focus-visible:outline-primary"
      >
        <span
          aria-hidden
          className={`mt-1.5 h-2 w-2 shrink-0 rounded-full ${phaseDot(phase)}`}
        />
        <span className="min-w-0 flex-1">
          <span className="flex items-center justify-between gap-2">
            <Marquee className="text-sm font-medium">
              {activityKindLabel(snapshot?.kind)}
            </Marquee>
            <span className={`shrink-0 text-xs font-medium ${phaseTone(phase)}`}>
              {PHASE_LABELS[phase]}
            </span>
          </span>
          <Marquee className="mt-1 block font-mono text-[11px] text-status-neutral">
            {trackedJob.jobId} · {resultSessionId}
          </Marquee>
          {snapshot && snapshot.state.events.length > 0 && (
            <span className="mt-1 block text-[11px] text-status-neutral">
              {snapshot.state.events.length} events received
            </span>
          )}
        </span>
      </button>
      <div className="mt-2 flex items-center gap-2 border-t border-border/70 pt-2">
        {trackedJob.projectId && (
          <Link
            to={sessionSectionPath(
              trackedJob.projectId,
              resultSessionId,
              "data-map",
            )}
            className="text-xs font-medium text-primary hover:underline"
          >
            Open result
          </Link>
        )}
        {terminal && (
          <button
            type="button"
            onClick={() => onDismiss(trackedJob.jobId)}
            className="ml-auto text-xs text-status-neutral hover:text-text"
          >
            Dismiss
          </button>
        )}
      </div>
    </li>
  );
}

/* Runs that finished cleanly drop into a collapsed history rather than staying
 * in the live list; their result links are still worth reaching, so nothing is
 * auto-deleted. `failed` and `disconnected` are terminal-ish but stay up top —
 * they are exactly what the user needs to see, and folding them away would
 * contradict the launcher badge that counts them. */
function RunsInbox({
  jobs,
  snapshots,
  selectedJobId,
  onSelect,
  onDismiss,
}: {
  jobs: ActiveJob[];
  snapshots: ReadonlyMap<string, JobActivitySnapshot>;
  selectedJobId: string | null;
  onSelect: (jobId: string) => void;
  onDismiss: (jobId: string) => void;
}) {
  const [historyOpen, setHistoryOpen] = useState(false);

  if (jobs.length === 0) {
    return (
      <div className="flex min-h-56 items-center justify-center px-6 py-8 text-center">
        <div className="max-w-xs">
          <p className="text-sm font-medium">No background runs</p>
          <p className="mt-1 text-xs leading-5 text-status-neutral">
            Long-running analyses, cleaning jobs, relationship checks and
            reports will stay visible here while you continue working.
          </p>
        </div>
      </div>
    );
  }

  const settled = (trackedJob: ActiveJob) =>
    QUIET_PHASES.has(snapshots.get(trackedJob.jobId)?.state.phase ?? "connecting");
  const live = jobs.filter((trackedJob) => !settled(trackedJob));
  const finished = jobs.filter(settled);
  const row = (trackedJob: ActiveJob) => (
    <RunRow
      key={trackedJob.jobId}
      trackedJob={trackedJob}
      snapshot={snapshots.get(trackedJob.jobId)}
      selected={trackedJob.jobId === selectedJobId}
      onSelect={onSelect}
      onDismiss={onDismiss}
    />
  );

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-auto p-3">
      {live.length > 0 ? (
        <ol aria-label="Tracked runs" className="flex flex-col gap-2">
          {live.map(row)}
        </ol>
      ) : (
        <p className="px-1 py-2 text-xs text-status-neutral">
          Nothing running right now.
        </p>
      )}
      {finished.length > 0 && (
        <>
          <button
            type="button"
            aria-expanded={historyOpen}
            onClick={() => setHistoryOpen(!historyOpen)}
            className="self-start rounded-base px-1 py-1 text-xs font-medium text-status-neutral hover:text-text focus-visible:outline-2 focus-visible:outline-primary"
          >
            {historyOpen ? "Hide" : "Show"} {finished.length} finished run
            {finished.length > 1 ? "s" : ""}
          </button>
          {historyOpen && (
            <ol aria-label="Finished runs" className="flex flex-col gap-2">
              {finished.map(row)}
            </ol>
          )}
        </>
      )}
    </div>
  );
}

function AgentActivityCenter() {
  const {
    activeJob,
    dismissJob,
    launcherVisible,
    panelOpen,
    selectedJobId,
    jobSnapshots: snapshots,
    selectJob,
    setJobSnapshot,
    setLauncherVisible,
    setPanelOpen,
    trackedJobs,
  } = useJobActivity();
  const metrics = useSessionMetrics(activeJob?.sessionId ?? "");
  const launcherRef = useRef<HTMLButtonElement>(null);
  const geometry = useOverlayGeometry();
  const [section, setSection] = useState<PanelSection>(
    trackedJobs.length === 1 ? "activity" : "runs",
  );
  const previousJobCount = useRef(trackedJobs.length);
  useActivityKeyboardShortcut();

  const onSnapshot = useCallback(
    (jobId: string, snapshot: JobActivitySnapshot) => {
      setJobSnapshot(jobId, snapshot);
    },
    [setJobSnapshot],
  );

  useEffect(() => {
    if (trackedJobs.length === 0) {
      setSection("runs");
    } else if (previousJobCount.current === 0 && trackedJobs.length === 1) {
      setSection("activity");
    } else if (previousJobCount.current <= 1 && trackedJobs.length > 1) {
      setSection("runs");
    }
    previousJobCount.current = trackedJobs.length;
  }, [trackedJobs.length]);

  const selectedSnapshot = activeJob
    ? snapshots.get(activeJob.jobId)
    : undefined;
  const selectedState = selectedSnapshot?.state;
  const selectedKind = selectedSnapshot?.kind;
  const selectedPhase = selectedState?.phase;
  const selectedTerminal =
    selectedPhase !== undefined && TERMINAL_PHASES.has(selectedPhase);
  const totalEvents = [...snapshots.values()].reduce(
    (total, snapshot) => total + snapshot.state.events.length,
    0,
  );
  const activeCount = trackedJobs.filter((trackedJob) => {
    const phase = snapshots.get(trackedJob.jobId)?.state.phase ?? "connecting";
    return !TERMINAL_PHASES.has(phase) && phase !== "disconnected";
  }).length;
  const attentionCount = trackedJobs.filter((trackedJob) => {
    const phase = snapshots.get(trackedJob.jobId)?.state.phase;
    return phase === "failed" || phase === "disconnected";
  }).length;

  const close = () => {
    setPanelOpen(false);
    requestAnimationFrame(() => launcherRef.current?.focus());
  };
  const toggleFloatingButton = () => {
    setLauncherVisible(!launcherVisible);
  };

  const sectionContent: Record<PanelSection, ReactNode> = {
    runs: (
      <RunsInbox
        jobs={trackedJobs}
        snapshots={snapshots}
        selectedJobId={selectedJobId}
        onSelect={(jobId) => {
          selectJob(jobId);
          setSection("activity");
        }}
        onDismiss={dismissJob}
      />
    ),
    activity: (
      activeJob && selectedState ? (
        <div className="flex shrink-0 flex-col gap-2 px-4 py-3">
          {trackedJobs.length > 1 && (
            <button
              type="button"
              onClick={() => setSection("runs")}
              className="mb-1 flex items-center gap-1 self-start text-xs font-medium text-primary hover:underline"
            >
              ← All runs
            </button>
          )}
          <PipelineProgress job={selectedState} kind={selectedKind} />
          {metrics.data && (
            <SessionQualitySummary metrics={metrics.data} compact />
          )}
          <div className="flex flex-wrap items-center gap-2">
            {selectedKind !== undefined &&
              CANCELLABLE_KINDS.has(selectedKind) && (
                <CancelButton
                  jobId={activeJob.jobId}
                  disabled={selectedTerminal}
                />
              )}
            {selectedKind !== undefined &&
              !CANCELLABLE_KINDS.has(selectedKind) &&
              !selectedTerminal && (
                <span className="text-xs text-status-neutral">
                  Cannot be cancelled mid-run.
                </span>
              )}
            {selectedTerminal && (
              <button
                type="button"
                onClick={() => dismissJob(activeJob.jobId)}
                className="rounded-base border border-border px-2 py-1 text-xs hover:bg-code-bg"
              >
                Dismiss
              </button>
            )}
          </div>
        </div>
      ) : (
        <div className="flex min-h-52 items-center justify-center px-6 py-8 text-center">
          <p className="max-w-xs text-sm text-status-neutral">
            {activeJob
              ? "Connecting to this run’s activity stream…"
              : "Select a run to inspect its live progress and quality signals."}
          </p>
        </div>
      )
    ),
    events: (
      <div className="flex min-h-64 flex-1 flex-col bg-code-bg/40">
        <EventLog events={selectedState?.events ?? []} />
      </div>
    ),
  };

  return (
    <ActivityPortal onDismiss={panelOpen ? () => setPanelOpen(false) : undefined}>
      {trackedJobs.map((trackedJob) => (
        <JobObserver
          key={trackedJob.jobId}
          trackedJob={trackedJob}
          onSnapshot={onSnapshot}
        />
      ))}
      {!panelOpen && activeJob && (
        <>
          <p className="sr-only">
            Tracking {activeJob.jobId} · session {activeJob.sessionId}
          </p>
          <span className="sr-only">{activeJob.jobId}</span>
        </>
      )}
      {panelOpen && (
        <ActivityPanelFrame
          eventCount={totalEvents}
          geometry={geometry}
          launcherVisible={launcherVisible}
          section={section}
          onSectionChange={setSection}
          onClose={close}
          onToggleLauncher={toggleFloatingButton}
          onKeyDown={(event) => {
            if (event.key === "Escape") close();
          }}
          status={
            trackedJobs.length === 0 ? (
              <span className="text-status-neutral">No background runs</span>
            ) : trackedJobs.length === 1 && selectedPhase ? (
              <span
                className={`flex items-center gap-1.5 font-semibold ${phaseTone(selectedPhase)}`}
              >
                <span
                  aria-hidden
                  className={`h-1.5 w-1.5 rounded-full ${phaseDot(selectedPhase)}`}
                />
                {PHASE_LABELS[selectedPhase]}
                {selectedState?.cancelRequested && !selectedTerminal
                  ? " · cancel requested"
                  : ""}
              </span>
            ) : (
              <span className="flex flex-wrap items-center gap-x-2 text-status-neutral">
                <span>{activeCount} active</span>
                {attentionCount > 0 && (
                  <span className="font-semibold text-status-critical">
                    {attentionCount} need attention
                  </span>
                )}
                <span>{trackedJobs.length} total</span>
              </span>
            )
          }
          trackingLabel={
            activeJob
              ? `Selected ${activeJob.jobId} · session ${activeJob.sessionId}`
              : undefined
          }
        >
          {sectionContent[section]}
        </ActivityPanelFrame>
      )}
      {launcherVisible && (
        <Launcher
          buttonRef={launcherRef}
          runCount={activeCount + attentionCount}
          geometry={geometry}
          open={panelOpen}
          phase={
            attentionCount > 0
              ? "failed"
              : activeCount > 0
                ? "running"
                : selectedPhase
          }
          progress={
            selectedState
              ? launcherProgress(selectedState, selectedKind)
              : null
          }
          onClick={() => {
            if (!geometry.consumeDraggedClick()) setPanelOpen(!panelOpen);
          }}
        />
      )}
    </ActivityPortal>
  );
}

export function ActivityCenter() {
  return <AgentActivityCenter />;
}
