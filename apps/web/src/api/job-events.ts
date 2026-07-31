/* SSE stream for job progress (§7.3). Wraps the browser-native EventSource so
 * reconnects replay from Last-Event-ID automatically; frames carry `id:` and a
 * named `event:` type, so we register a listener per known type. */

import { useEffect, useReducer } from "react";

export interface JobEvent {
  event_id: number;
  job_id: string;
  session_id: string;
  type: string;
  name: string;
  timestamp?: string | null;
  summary: Record<string, unknown>;
}

/* Pipeline phases. `steps` are the kernel's step_started/step_completed trace
 * names, in the order drivers/auto_eda.py declares them.
 *
 * This used to be a flat six-entry list that drifted from the driver stages:
 * which had drifted from the driver: it named run_stat_tests (conditional — it
 * does not run without test targets) and omitted discover_questions and
 * export_agentic_report, the two longest steps there are. With no entry to
 * match, the stepper fell through to "first stage not yet done" and displayed
 * Stats as the running step for both. Measured on the Olist run that was 270s
 * of a 426s run naming the wrong step, so grouping is by phase now and phase
 * state is driven only by step names the backend actually emitted.
 *
 * Several steps are conditional (cleaning only with precleaning on, baseline
 * only with an ml target, relationships only for multi-table runs); a phase is
 * never blocked on a step that never arrives because state is derived from the
 * furthest step seen, not from every step completing. */
export const JOB_PHASES = [
  {
    key: "read",
    label: "Reading data",
    /* Present tense, and about the data rather than the code path — this line
     * is what the user reads while waiting. */
    activity: "Loading the tables and recording their raw state",
    steps: ["emit_cleaning_recipe", "profile_dataset", "record_raw_dataset"],
  },
  {
    key: "quality",
    label: "Checking quality",
    activity: "Scanning for missing values, duplicates and type problems",
    steps: ["scan_quality", "build_quality_context"],
  },
  {
    key: "describe",
    label: "Profiling & stats",
    activity: "Building column profiles, charts, summary tables and tests",
    steps: [
      "build_value_map",
      "create_chart_specs",
      "create_analysis_tables",
      "run_stat_tests",
      "run_baseline_model",
      "discover_relationships",
    ],
  },
  {
    key: "questions",
    label: "Finding questions",
    activity: "Proposing questions this data can actually answer",
    steps: ["discover_questions"],
  },
  {
    key: "answers",
    label: "Answering questions",
    activity: "Running the top questions and collecting findings",
    steps: ["execute_top_questions"],
  },
  {
    key: "report",
    label: "Writing report",
    activity: "Drafting the report and validating every claim against evidence",
    steps: ["export_agentic_report"],
  },
] as const;

export type JobPhaseKey = (typeof JOB_PHASES)[number]["key"];

/* Only auto_eda walks the phases above; the other twelve kinds worker/runner.py
 * dispatches are single-purpose jobs. They get a sentence instead of a phase
 * strip, because rendering six greyed-out EDA phases for a question execution
 * says nothing true about it. Keys match runner.py's dispatch. */
export const JOB_KIND_ACTIVITY: Record<string, string> = {
  auto_eda: "Running the full analysis pipeline",
  question_exec: "Answering one approved question",
  question_draft: "Drafting a question card from your text",
  investigation_plan: "Building investigation plans for the selected questions",
  investigation_execute: "Working through an approved investigation plan",
  macro_loop: "Running the investigation loop until its gates are met",
  relationship_validate: "Validating a candidate join against the data",
  relationship_discover: "Searching for candidate joins between tables",
  report_generate: "Regenerating the report and revalidating its claims",
  decision_report_generate: "Assembling the decision report",
  synthesis_brief_create: "Summarising findings into a synthesis brief",
  skill_replay: "Replaying a saved skill",
  session_fork: "Forking this session",
  cleaning_preview: "Scanning the dataset for a cleaning preview",
  cleaning_apply: "Applying the approved cleaning recipe",
  dataset_distributions: "Scanning the dataset for column distributions",
  custom_chart: "Scanning the dataset to build the custom chart",
};

/* step name -> index into JOB_PHASES. */
const PHASE_OF_STEP = new Map<string, number>(
  JOB_PHASES.flatMap((phase, index) =>
    phase.steps.map((step) => [step, index] as const),
  ),
);

const STAGE_KEYS = new Set<string>(PHASE_OF_STEP.keys());

/* EventSource only dispatches named events it has listeners for; this is the
 * full set of trace event_type values the backend emits today (grep of
 * eda_platform `event_type="..."` plus worker/runner job.* frames). */
export const SSE_EVENT_TYPES = [
  "job.queued",
  "job.started",
  "job.cancel_requested",
  "job.completed",
  "job.failed",
  "job.cancelled",
  "step_started",
  "step_completed",
  "step_failed",
  "step_contract_violation",
  "agent_intent",
  "agent_plan",
  "budget_degraded",
  "chat_turn_failed",
  "checkpoint_hit",
  "checkpoint_invalid",
  "cleaning_applied",
  "code_agent_attempt",
  "domain_metrics_skipped",
  "investigation_completed",
  "investigation_plans_created",
  "investigation_started",
  "join_authorization_freshness",
  "join_candidates_proposed",
  "join_whitelist_unreadable",
  "llm_call",
  "llm_error",
  "loop_started",
  "ml_baseline_skipped",
  "precleaning_applied",
  "report.generated",
  "session.forked",
  "question_auto_execution",
  "question_auto_execution_selected",
  "question_execution_completed",
  "question_llm_skipped",
  "question_result_contract",
  "relationship_discovery_bounded",
  "relationship_discovery_deferred",
  "relationship_validation_on_demand",
  "report_validation",
  "session_metrics_error",
  "run_metrics_failed",
  "semantic_bootstrap",
  "synthesis_brief_created",
  "template_backstop",
  "tool_completed",
  "tool_guard_rejected",
  "validator_result",
  "unknown",
] as const;

export type JobPhase =
  | "connecting"
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "cancelled"
  | "disconnected";

export const TERMINAL_PHASES: ReadonlySet<JobPhase> = new Set([
  "completed",
  "failed",
  "cancelled",
]);

export interface JobEventsState {
  events: JobEvent[];
  phase: JobPhase;
  stagesDone: ReadonlySet<string>;
  activeStage: string | null;
  failedStage: string | null;
  cancelRequested: boolean;
  /* step name -> epoch ms of its step_started, for the elapsed readout. */
  stepStartedAt: ReadonlyMap<string, number>;
  /* step name -> how far through its items it is. Steps that run once per
   * dataset re-emit step_started with `summary.index`, so profile_dataset
   * fires nine times on a nine-table run; without this the strip showed one
   * undifferentiated "Reading data" for all nine. */
  stepItems: ReadonlyMap<string, ItemProgress>;
}

export interface ItemProgress {
  /** 1-based position of the item currently being worked. */
  current: number;
  /** Null until the run reveals a total; grows monotonically, never shrinks. */
  total: number | null;
}

const MAX_LOG_EVENTS = 200;

const initialState: JobEventsState = {
  events: [],
  phase: "connecting",
  stagesDone: new Set(),
  activeStage: null,
  failedStage: null,
  cancelRequested: false,
  stepStartedAt: new Map(),
  stepItems: new Map(),
};

export type PhaseState = "done" | "active" | "failed" | "pending" | "skipped";

export interface PhaseProgress {
  key: JobPhaseKey;
  label: string;
  activity: string;
  state: PhaseState;
  /* The step name the backend actually reported, when one is running. Shown so
   * a stalled phase can still be matched against the trace. */
  currentStep: string | null;
  /* Per-item position inside the running step, when it works item by item. */
  items: ItemProgress | null;
}

/* Phase state comes from the furthest phase the run has reached, never from
 * "have all this phase's steps completed" — most steps are conditional, so
 * waiting for all of them would strand a phase as pending forever. */
export function phaseProgress(job: JobEventsState): PhaseProgress[] {
  const seen = [...job.stagesDone, job.activeStage, job.failedStage]
    .filter((step): step is string => step !== null)
    .map((step) => PHASE_OF_STEP.get(step))
    .filter((index): index is number => index !== undefined);

  const furthest = seen.length > 0 ? Math.max(...seen) : -1;
  const failedPhase =
    job.failedStage !== null
      ? (PHASE_OF_STEP.get(job.failedStage) ?? -1)
      : -1;
  const activePhase =
    job.activeStage !== null ? (PHASE_OF_STEP.get(job.activeStage) ?? -1) : -1;

  const finished = TERMINAL_PHASES.has(job.phase);
  /* A killed worker emits step_started with no matching completed/failed, so
   * activeStage survives into a terminal job. Left as "active" the strip kept
   * an amber segment and an elapsed counter ticking upward forever (measured
   * at 32h on a reaped job) while the header already said Failed. On a
   * finished run the step that was in flight is the one that stopped. */
  const stalledPhase = finished && failedPhase === -1 ? activePhase : -1;

  return JOB_PHASES.map((phase, index) => {
    let state: PhaseState;
    if (index === stalledPhase) state = job.phase === "completed" ? "done" : "failed";
    else if (index === failedPhase) state = "failed";
    else if (!finished && index === activePhase) state = "active";
    else if (index <= furthest) state = "done";
    /* A run that stopped never "will" reach the rest, so they are skipped
     * rather than pending — pending on a finished run reads as a hang. */
    else state = finished ? "skipped" : "pending";

    return {
      key: phase.key,
      label: phase.label,
      activity: phase.activity,
      state,
      currentStep: state === "active" ? job.activeStage : null,
      items:
        state === "active" && job.activeStage !== null
          ? (job.stepItems.get(job.activeStage) ?? null)
          : null,
    };
  });
}

type Action =
  | { kind: "reset" }
  | { kind: "event"; event: JobEvent; jobId: string }
  | { kind: "disconnected" };

function readCount(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) && value >= 0
    ? value
    : null;
}

/* `index` is 0-based and arrives once per item. The kernel emits ONLY the index
 * (core/kernel.py step_started: `summary={"index": index}`), so a total is
 * normally unknown and must stay null — deriving it from the highest index seen
 * made every update render "N of N", i.e. permanently 100% complete. A total is
 * used only when a step actually declares one, and then it never shrinks. */
function advanceItems(
  items: ReadonlyMap<string, ItemProgress>,
  event: JobEvent,
): ReadonlyMap<string, ItemProgress> {
  const index = readCount(event.summary["index"]);
  if (index === null) return items;
  const declared =
    readCount(event.summary["dataset_count"]) ??
    readCount(event.summary["total"]);
  const previous = items.get(event.name);
  const current = index + 1;
  const known = Math.max(declared ?? 0, previous?.total ?? 0);
  return new Map(items).set(event.name, {
    current,
    total: known > 0 ? Math.max(known, current) : null,
  });
}

function reduce(state: JobEventsState, action: Action): JobEventsState {
  if (action.kind === "reset") return initialState;
  if (action.kind === "disconnected") {
    return TERMINAL_PHASES.has(state.phase)
      ? state
      : { ...state, phase: "disconnected" };
  }

  const { event, jobId } = action;
  const next: JobEventsState = {
    ...state,
    events: [...state.events, event].slice(-MAX_LOG_EVENTS),
  };

  switch (event.type) {
    case "job.queued":
      if (event.name === jobId) next.phase = "queued";
      break;
    case "job.started":
      if (event.name === jobId) next.phase = "running";
      break;
    case "job.cancel_requested":
      if (event.name === jobId) next.cancelRequested = true;
      break;
    case "job.completed":
    case "job.failed":
    case "job.cancelled":
      if (event.name === jobId) {
        next.phase = event.type.slice("job.".length) as JobPhase;
      }
      break;
    case "step_started":
      if (STAGE_KEYS.has(event.name)) {
        next.activeStage = event.name;
        const startedAt = event.timestamp
          ? Date.parse(event.timestamp)
          : Number.NaN;
        if (!Number.isNaN(startedAt)) {
          next.stepStartedAt = new Map(next.stepStartedAt).set(
            event.name,
            startedAt,
          );
        }
        next.stepItems = advanceItems(next.stepItems, event);
        if (next.phase === "queued" || next.phase === "connecting") {
          next.phase = "running";
        }
      }
      break;
    case "step_completed":
      if (STAGE_KEYS.has(event.name)) {
        const done = new Set(next.stagesDone);
        done.add(event.name);
        next.stagesDone = done;
        if (next.activeStage === event.name) next.activeStage = null;
      }
      break;
    case "step_failed":
      if (STAGE_KEYS.has(event.name)) next.failedStage = event.name;
      break;
  }
  return next;
}

function parseEvent(raw: string): JobEvent | null {
  try {
    const data = JSON.parse(raw) as Partial<JobEvent>;
    if (typeof data !== "object" || data === null) return null;
    return {
      event_id: typeof data.event_id === "number" ? data.event_id : 0,
      job_id: String(data.job_id ?? ""),
      session_id: String(data.session_id ?? ""),
      type: String(data.type ?? "unknown"),
      name: String(data.name ?? ""),
      timestamp: data.timestamp ?? null,
      summary:
        typeof data.summary === "object" && data.summary !== null
          ? (data.summary as Record<string, unknown>)
          : {},
    };
  } catch {
    return null;
  }
}

/* Streams a job's events into derived progress state. Pass null to idle.
 * The terminal frame ends the server stream; we close the source so the
 * browser does not auto-reconnect after a finished job. */
export function useJobEvents(
  jobId: string | null,
  eventsUrl: string | null,
): JobEventsState {
  const [state, dispatch] = useReducer(reduce, initialState);

  useEffect(() => {
    dispatch({ kind: "reset" });
    if (!jobId || !eventsUrl) return;

    const source = new EventSource(eventsUrl);
    let terminal = false;

    const onEvent = (message: MessageEvent) => {
      const event = parseEvent(String(message.data));
      if (!event) return;
      dispatch({ kind: "event", event, jobId });
      if (
        event.name === jobId &&
        (event.type === "job.completed" ||
          event.type === "job.failed" ||
          event.type === "job.cancelled")
      ) {
        terminal = true;
        source.close();
      }
    };

    for (const type of SSE_EVENT_TYPES) {
      source.addEventListener(type, onEvent);
    }
    source.onerror = () => {
      /* CLOSED means the browser gave up (e.g. 404) — no auto-retry coming. */
      if (!terminal && source.readyState === EventSource.CLOSED) {
        dispatch({ kind: "disconnected" });
      }
    };

    return () => source.close();
  }, [jobId, eventsUrl]);

  return state;
}
