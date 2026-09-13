/* App-level job tracking: every launch surface can add a run, and Activity
 * follows the bounded inbox from any route (§5.1: navigation must not kill
 * background work). Persisted to localStorage so a refresh re-attaches to all
 * streams instead of silently replacing the previous run. */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { api, type SessionJobSummary } from "../api/client";
import type { JobEventsState } from "../api/job-events";

export interface ActiveJob {
  jobId: string;
  /** Derived run that owns only the asynchronous job lifecycle. */
  sessionId: string;
  /** Run from which the operation was requested. */
  sourceSessionId: string;
  /** Run that owns result artifacts, when neither source nor lifecycle does. */
  resultSessionId?: string;
  projectId: string;
  eventsUrl: string;
  /** Inputs selected when the job was launched. They keep the destination
   * surface honest while profiles are still being written by the worker. */
  inputDatasets?: TrackedDataset[];
}

export interface TrackedDataset {
  datasetId: string;
  displayName: string;
  byteSize?: number | null;
  rowCount?: number | null;
  format?: string | null;
}

export interface JobActivitySnapshot {
  state: JobEventsState;
  kind: string | undefined;
}

interface JobActivityValue {
  trackedJobs: ActiveJob[];
  jobSnapshots: ReadonlyMap<string, JobActivitySnapshot>;
  activeJob: ActiveJob | null;
  selectedJobId: string | null;
  panelOpen: boolean;
  launcherVisible: boolean;
  startTracking: (job: ActiveJob) => void;
  /** Merge server-known runs without stealing the selection (recovery path). */
  adoptJobs: (jobs: ActiveJob[]) => void;
  selectJob: (jobId: string) => void;
  dismissJob: (jobId: string) => void;
  /** Transitional alias for callers that act on the selected run. */
  clearActiveJob: () => void;
  setPanelOpen: (open: boolean) => void;
  setLauncherVisible: (visible: boolean) => void;
  setJobSnapshot: (jobId: string, snapshot: JobActivitySnapshot) => void;
  /** True the first time a job settles; false on SSE replays after route
   * changes, so toast + invalidation fire once per job per app session. */
  claimSettlement: (jobId: string) => boolean;
}

const JOB_KEY = "eda.activity.job";
const JOBS_KEY = "eda.activity.jobs";
const SELECTED_JOB_KEY = "eda.activity.selected-job";
const MAX_TRACKED_JOBS = 12;
const PANEL_KEY = "eda.layout.activity-open";
const LEGACY_DRAWER_KEY = "eda.layout.drawer-open";
const LAUNCHER_KEY = "eda.layout.activity-launcher";

function readStoredBoolean(key: string): boolean | null {
  const value = window.localStorage.getItem(key);
  if (value === null) return null;
  return value === "true";
}

function parseStoredJob(value: unknown): ActiveJob | null {
  if (typeof value !== "object" || value === null) return null;
  const parsed = value as Partial<ActiveJob>;
  if (
    typeof parsed.jobId !== "string" ||
    typeof parsed.sessionId !== "string" ||
    typeof parsed.projectId !== "string" ||
    typeof parsed.eventsUrl !== "string"
  ) {
    return null;
  }
  return {
    jobId: parsed.jobId,
    sessionId: parsed.sessionId,
    sourceSessionId:
      typeof parsed.sourceSessionId === "string"
        ? parsed.sourceSessionId
        : parsed.sessionId,
    resultSessionId:
      typeof parsed.resultSessionId === "string"
        ? parsed.resultSessionId
        : undefined,
    projectId: parsed.projectId,
    eventsUrl: parsed.eventsUrl,
    inputDatasets: Array.isArray(parsed.inputDatasets)
      ? parsed.inputDatasets.filter(
          (dataset): dataset is TrackedDataset =>
            typeof dataset === "object" &&
            dataset !== null &&
            typeof (dataset as Partial<TrackedDataset>).datasetId === "string" &&
            typeof (dataset as Partial<TrackedDataset>).displayName === "string",
        )
      : undefined,
  };
}

function readStoredJobs(): ActiveJob[] {
  try {
    const raw = window.localStorage.getItem(JOBS_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as unknown;
      if (Array.isArray(parsed)) {
        return parsed
          .map(parseStoredJob)
          .filter((job): job is ActiveJob => job !== null)
          .slice(0, MAX_TRACKED_JOBS);
      }
    }
    const legacy = window.localStorage.getItem(JOB_KEY);
    if (!legacy) return [];
    const job = parseStoredJob(JSON.parse(legacy));
    return job ? [job] : [];
  } catch {
    /* corrupted entry: ignore */
    return [];
  }
}

function persistJobs(jobs: ActiveJob[], selectedJobId: string | null) {
  try {
    window.localStorage.setItem(JOBS_KEY, JSON.stringify(jobs));
    if (selectedJobId) {
      window.localStorage.setItem(SELECTED_JOB_KEY, selectedJobId);
      const selected = jobs.find((job) => job.jobId === selectedJobId);
      if (selected) {
        /* Keep one release of compatibility for older clients and tests that
         * still restore the original single-job key. */
        window.localStorage.setItem(JOB_KEY, JSON.stringify(selected));
      }
    } else {
      window.localStorage.removeItem(SELECTED_JOB_KEY);
      window.localStorage.removeItem(JOB_KEY);
    }
  } catch {
    // Recovery storage is best effort; tracking in this tab must still work.
  }
}

const JobActivityContext = createContext<JobActivityValue | null>(null);

export function JobActivityProvider({ children }: { children: ReactNode }) {
  const [trackedJobs, setTrackedJobs] = useState<ActiveJob[]>(readStoredJobs);
  const [jobSnapshots, setJobSnapshots] = useState<
    ReadonlyMap<string, JobActivitySnapshot>
  >(new Map());
  const [selectedJobId, setSelectedJobId] = useState<string | null>(() => {
    const stored = window.localStorage.getItem(SELECTED_JOB_KEY);
    return stored && trackedJobs.some((job) => job.jobId === stored)
      ? stored
      : (trackedJobs[0]?.jobId ?? null);
  });
  const activeJob =
    trackedJobs.find((job) => job.jobId === selectedJobId) ??
    trackedJobs[0] ??
    null;

  /* Opening a legacy workspace is itself enough to complete the migration;
   * the user should not have to launch another job before multi-run recovery
   * becomes durable. */
  useEffect(() => {
    persistJobs(trackedJobs, activeJob?.jobId ?? null);
  }, [activeJob?.jobId, trackedJobs]);
  const [panelOpen, setPanelOpenState] = useState(
    () =>
      readStoredBoolean(PANEL_KEY) ??
      readStoredBoolean(LEGACY_DRAWER_KEY) ??
      false,
  );
  const [launcherVisible, setLauncherVisibleState] = useState(
    () => readStoredBoolean(LAUNCHER_KEY) ?? true,
  );

  const setPanelOpen = useCallback((open: boolean) => {
    setPanelOpenState(open);
    window.localStorage.setItem(PANEL_KEY, String(open));
  }, []);

  const setLauncherVisible = useCallback((visible: boolean) => {
    setLauncherVisibleState(visible);
    window.localStorage.setItem(LAUNCHER_KEY, String(visible));
  }, []);

  const startTracking = useCallback(
    (job: ActiveJob) => {
      setTrackedJobs((current) => {
        const next = [
          job,
          ...current.filter((tracked) => tracked.jobId !== job.jobId),
        ].slice(0, MAX_TRACKED_JOBS);
        persistJobs(next, job.jobId);
        return next;
      });
      setSelectedJobId(job.jobId);
    },
    [],
  );

  const adoptJobs = useCallback((jobs: ActiveJob[]) => {
    setTrackedJobs((current) => {
      const known = new Set(current.map((job) => job.jobId));
      const added = jobs.filter((job) => !known.has(job.jobId));
      if (added.length === 0) return current;
      /* Recovered runs go after what this tab already tracks: local entries
       * are what the user just launched, and adoption must not evict them. */
      return [...current, ...added].slice(0, MAX_TRACKED_JOBS);
    });
  }, []);

  const selectJob = useCallback((jobId: string) => {
    setTrackedJobs((current) => {
      if (!current.some((job) => job.jobId === jobId)) return current;
      setSelectedJobId(jobId);
      persistJobs(current, jobId);
      return current;
    });
  }, []);

  const dismissJob = useCallback(
    (jobId: string) => {
      setJobSnapshots((current) => {
        if (!current.has(jobId)) return current;
        const next = new Map(current);
        next.delete(jobId);
        return next;
      });
      setTrackedJobs((current) => {
        const next = current.filter((job) => job.jobId !== jobId);
        const nextSelected =
          selectedJobId === jobId
            ? (next[0]?.jobId ?? null)
            : selectedJobId;
        setSelectedJobId(nextSelected);
        persistJobs(next, nextSelected);
        return next;
      });
    },
    [selectedJobId],
  );

  const clearActiveJob = useCallback(() => {
    if (activeJob) dismissJob(activeJob.jobId);
  }, [activeJob, dismissJob]);

  const setJobSnapshot = useCallback(
    (jobId: string, snapshot: JobActivitySnapshot) => {
      setJobSnapshots((current) => {
        if (current.get(jobId) === snapshot) return current;
        const next = new Map(current);
        next.set(jobId, snapshot);
        return next;
      });
    },
    [],
  );

  const settledJobs = useRef(new Set<string>());
  const claimSettlement = useCallback((jobId: string) => {
    if (settledJobs.current.has(jobId)) return false;
    settledJobs.current.add(jobId);
    return true;
  }, []);

  const value = useMemo(
    () => ({
      trackedJobs,
      jobSnapshots,
      activeJob,
      selectedJobId,
      panelOpen,
      launcherVisible,
      startTracking,
      adoptJobs,
      selectJob,
      dismissJob,
      clearActiveJob,
      setPanelOpen,
      setLauncherVisible,
      setJobSnapshot,
      claimSettlement,
    }),
    [
      trackedJobs,
      jobSnapshots,
      activeJob,
      selectedJobId,
      panelOpen,
      launcherVisible,
      startTracking,
      adoptJobs,
      selectJob,
      dismissJob,
      clearActiveJob,
      setPanelOpen,
      setLauncherVisible,
      setJobSnapshot,
      claimSettlement,
    ],
  );

  return (
    <JobActivityContext.Provider value={value}>
      {children}
    </JobActivityContext.Provider>
  );
}

export function useJobActivity(): JobActivityValue {
  const value = useContext(JobActivityContext);
  if (!value) {
    throw new Error("useJobActivity must be used inside JobActivityProvider");
  }
  return value;
}

const SETTLED_JOB_STATUSES = new Set(["completed", "failed", "cancelled"]);

function recoveredJob(job: SessionJobSummary): ActiveJob {
  return {
    jobId: job.job_id,
    sessionId: job.session_id,
    sourceSessionId: job.source_session_id ?? job.session_id,
    resultSessionId: job.source_session_id ?? undefined,
    projectId: job.project_id,
    eventsUrl: job.events_url,
  };
}

/* Server-side recovery: localStorage only knows about jobs this browser
 * launched, so after a cleared cache or a device switch a run still executing
 * would be invisible — and uncancellable. Entering a session asks the backend
 * for that session's live jobs and adopts the ones not already tracked. */
export function useServerJobRecovery(sessionId: string | undefined) {
  const { adoptJobs } = useJobActivity();

  useEffect(() => {
    if (!sessionId) return;
    const controller = new AbortController();
    api
      .listSessionJobs(sessionId, controller.signal)
      .then((listing) => {
        const live = (listing.jobs ?? []).filter(
          (job) => !SETTLED_JOB_STATUSES.has(job.status),
        );
        if (live.length > 0) adoptJobs(live.map(recoveredJob));
      })
      .catch(() => {
        /* Recovery is best-effort; local tracking keeps working without it. */
      });
    return () => controller.abort();
  }, [sessionId, adoptJobs]);
}
