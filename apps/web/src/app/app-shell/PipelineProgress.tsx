/* Live run progress.
 *
 * Shape follows the Dagster run-page split (segmented progress above, filterable
 * event log below) rather than a proportional waterfall: measured stage times on
 * a real run span 0.02s to 137s, so length-encoded bars would render four of the
 * six phases as invisible slivers. Segments are therefore equal width and the
 * duration is printed as text. */

import { useEffect, useState } from "react";
import {
  JOB_KIND_ACTIVITY,
  phaseProgress,
  TERMINAL_PHASES,
  type JobEventsState,
  type PhaseProgress,
  type PhaseState,
} from "../../api/job-events";
import { Marquee, formatDuration } from "../../components/ui";

const SEGMENT: Record<PhaseState, string> = {
  done: "bg-status-ok",
  active: "bg-status-warn",
  failed: "bg-status-critical",
  pending: "bg-track",
  skipped: "bg-track",
};

const LABEL: Record<PhaseState, string> = {
  done: "text-status-neutral",
  active: "font-medium text-text",
  failed: "font-medium text-status-critical",
  pending: "text-status-neutral/70",
  /* Dimmed, not struck through: a run that stopped before the report is a
   * normal outcome, and six struck-out labels read as six failures. */
  skipped: "text-status-neutral/45",
};

/** Ticks once a second while a start time is set, so the active phase shows a
 *  moving number instead of a spinner that says nothing about progress. */
function useElapsed(since: number | undefined): number | null {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    if (since === undefined) return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [since]);

  if (since === undefined) return null;
  return Math.max(0, (now - since) / 1000);
}

function PhaseSegment({ phase }: { phase: PhaseProgress }) {
  /* Fraction of the running step's items that are done. Only the active
   * segment fills; a finished one is solid and a pending one is bare track. */
  const fraction =
    phase.state === "active" && phase.items && phase.items.total
      ? Math.min(1, phase.items.current / phase.items.total)
      : null;

  return (
    <li
      aria-label={`${phase.label}: ${phase.state}${
        phase.items
          ? `, item ${phase.items.current}${phase.items.total ? ` of ${phase.items.total}` : ""}`
          : ""
      }`}
      className="flex min-w-0 flex-1 flex-col gap-1"
    >
      {fraction === null ? (
        <span
          aria-hidden
          className={`h-1 rounded-full transition-colors duration-slow ease-out-quart ${
            SEGMENT[phase.state]
          } ${phase.state === "active" ? "animate-breathe" : ""}`}
        />
      ) : (
        /* scaleX, not width: width is a layout property and this updates on
         * every item. transform stays on the compositor. */
        <span aria-hidden className="h-1 overflow-hidden rounded-full bg-track">
          <span
            className="motion-progress block h-full origin-left rounded-full bg-status-warn transition-transform duration-slow ease-out-quart"
            style={{ transform: `scaleX(${fraction})` }}
          />
        </span>
      )}
      {/* Six labels across a drawer that shrinks with the inspector: the title
       * keeps the full name reachable when the segment clips it. */}
      <Marquee
        title={`${phase.label} — ${phase.activity}`}
        className={`text-xs ${LABEL[phase.state]}`}
      >
        {phase.label}
      </Marquee>
    </li>
  );
}

/* Everything that is not auto_eda: one honest sentence beats a phase strip
 * whose phases this job will never enter. */
function SingleStepProgress({
  job,
  kind,
}: {
  job: JobEventsState;
  kind: string | undefined;
}) {
  const running = !TERMINAL_PHASES.has(job.phase);
  const activity =
    (kind && JOB_KIND_ACTIVITY[kind]) ?? "Working on this session";

  /* No per-investigation breakdown here on purpose. investigation_started /
   * _completed are written to `plan_session_id`
   * (drivers/investigation_orchestrator.py), while the SSE stream is filtered
   * to the job's own derived run — whose docstring says it "only carries the
   * lifecycle" (worker/runner.py _run_investigation_execute_job). Those frames
   * therefore never reach this client, so rendering them would be dead code. */
  return (
    <div className="flex flex-col gap-1.5">
      <p role="status" className="flex items-center gap-2 text-xs">
        <span
          aria-hidden
          className={`h-1 w-16 shrink-0 rounded-full ${
            job.phase === "failed"
              ? "bg-status-critical"
              : running
                ? "animate-breathe bg-status-warn"
                : "bg-status-ok"
          }`}
        />
        <span className={running ? "text-text" : "text-status-neutral"}>
          {activity}
          {running ? "…" : ""}
        </span>
      </p>
    </div>
  );
}

/* Undefined kind means the job-status fetch has not landed yet, and this now
 * actually waits rather than guessing: it used to fall through to the strip,
 * so for one request round-trip every question execution flashed six greyed-out
 * EDA phases — the exact wrong-information problem this component exists to
 * fix. The branch lives here so neither child sits behind a conditional return
 * in a component that also calls hooks. */
export function PipelineProgress({
  job,
  kind,
}: {
  job: JobEventsState;
  kind: string | undefined;
}) {
  if (kind === undefined) {
    return (
      <p role="status" className="flex items-center gap-2 text-xs">
        <span
          aria-hidden
          /* Sweep, not breathe: the job kind has not landed yet, so this bar is
           * an empty placeholder rather than a process reporting itself. */
          className="skeleton h-1 w-16 shrink-0 rounded-full"
        />
        <span className="text-status-neutral">Starting…</span>
      </p>
    );
  }
  return kind === "auto_eda" ? (
    <PhaseStrip job={job} />
  ) : (
    <SingleStepProgress job={job} kind={kind} />
  );
}

function PhaseStrip({ job }: { job: JobEventsState }) {
  const phases = phaseProgress(job);
  const active = phases.find((phase) => phase.state === "active");
  const failed = phases.find((phase) => phase.state === "failed");
  const current = failed ?? active;
  const elapsed = useElapsed(
    current?.currentStep
      ? job.stepStartedAt.get(current.currentStep)
      : undefined,
  );

  return (
    <div className="flex flex-col gap-2">
      <ol
        aria-label="Pipeline phases"
        className="flex items-start gap-1.5"
      >
        {phases.map((phase) => (
          <PhaseSegment key={phase.key} phase={phase} />
        ))}
      </ol>
      {current && (
        /* role="status" belongs on this one line and nowhere else: the strip
         * and the raw log would both flood a screen reader. */
        <p role="status" className="flex items-baseline gap-2 text-xs">
          <span className="text-text">{current.activity}</span>
          {current.items && (
            <span className="tabular shrink-0 font-medium text-status-warn">
              {current.items.total
                ? `${current.items.current} of ${current.items.total}`
                : `item ${current.items.current}`}
            </span>
          )}
          {elapsed !== null && (
            <span className="tabular shrink-0 text-status-neutral">
              {formatDuration(elapsed)}
            </span>
          )}
          {current.currentStep && (
            <span
              className="ml-auto shrink-0 font-mono text-status-neutral/70"
              title="Trace step name — search for this on the Trace page"
            >
              {current.currentStep}
            </span>
          )}
        </p>
      )}
    </div>
  );
}
