import {
  ApiError,
  api,
  type DataOperationStarted,
  type JobStatus,
} from "./client";
import type { ActiveJob } from "../app/job-activity";

const POLL_MS = 400;

function isDataOperationStarted(value: unknown): value is DataOperationStarted {
  if (typeof value !== "object" || value === null) return false;
  const candidate = value as Partial<DataOperationStarted>;
  return (
    typeof candidate.session_id === "string" &&
    candidate.session_id.length > 0 &&
    typeof candidate.execution_session_id === "string" &&
    candidate.execution_session_id.length > 0 &&
    typeof candidate.job?.job_id === "string" &&
    candidate.job.job_id.length > 0 &&
    candidate.job.session_id === candidate.execution_session_id &&
    typeof candidate.job.events_url === "string" &&
    candidate.job.events_url.startsWith("/api/v1/jobs/")
  );
}

function storedOperation(key: string): DataOperationStarted | null {
  try {
    const raw = window.localStorage.getItem(key);
    if (!raw) return null;
    const value: unknown = JSON.parse(raw);
    if (isDataOperationStarted(value)) return value;
  } catch {
    // Corrupt recovery state is equivalent to no pending operation.
  }
  try {
    window.localStorage.removeItem(key);
  } catch {
    // Recovery storage is optional.
  }
  return null;
}

function pause(signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(signal.reason);
      return;
    }
    const timer = window.setTimeout(resolve, POLL_MS);
    signal?.addEventListener(
      "abort",
      () => {
        window.clearTimeout(timer);
        reject(signal.reason);
      },
      { once: true },
    );
  });
}

async function waitForTerminal(
  jobId: string,
  signal?: AbortSignal,
): Promise<JobStatus> {
  for (;;) {
    const job = await api.getJob(jobId, signal);
    if (job.status === "completed") return job;
    if (job.status === "failed" || job.status === "cancelled") {
      throw new ApiError(
        409,
        job.error_code || `job_${job.status}`,
        job.error_message || `The data operation ${job.status}.`,
      );
    }
    await pause(signal);
  }
}

export function operationActivity(
  started: DataOperationStarted,
  projectId: string,
): ActiveJob {
  return {
    jobId: started.job.job_id,
    sessionId: started.execution_session_id,
    sourceSessionId: started.session_id,
    projectId,
    eventsUrl: started.job.events_url,
  };
}

/** Persist before waiting so a route reload can resume the same durable job.
 * The activity drawer owns the SSE connection; polling here only gates the
 * typed result fetch and survives an SSE reconnect without losing the result. */
export async function runDataOperation<Result>(
  storageKey: string,
  start: () => Promise<DataOperationStarted>,
  getResult: (jobId: string, signal?: AbortSignal) => Promise<Result>,
  onStarted: (started: DataOperationStarted) => void,
  signal?: AbortSignal,
): Promise<Result> {
  let started = storedOperation(storageKey);
  if (!started) {
    const response: unknown = await start();
    if (!isDataOperationStarted(response)) {
      throw new ApiError(
        502,
        "data_operation_contract_invalid",
        "The server returned an invalid data-operation job contract.",
      );
    }
    started = response;
    try {
      window.localStorage.setItem(storageKey, JSON.stringify(started));
    } catch {
      // Storage can be unavailable; the durable server job remains reachable
      // in this session through the activity drawer.
    }
  }
  try {
    // Inside the try: a throwing callback must still reach the cleanup path,
    // or the job it just queued runs with nothing watching it.
    onStarted(started);
    await waitForTerminal(started.job.job_id, signal);
    const result = await getResult(started.job.job_id, signal);
    try {
      window.localStorage.removeItem(storageKey);
    } catch {
      // Storage is an optional recovery aid, not part of job correctness.
    }
    return result;
  } catch (error) {
    if (!signal?.aborted) {
      const status = await api.getJob(started.job.job_id).catch(() => null);
      if (status?.status === "failed" || status?.status === "cancelled") {
        try {
          window.localStorage.removeItem(storageKey);
        } catch {
          // See the successful cleanup path above.
        }
      }
    }
    throw error;
  }
}

export function dataOperationStorageKey(
  kind: string,
  sessionId: string,
  discriminator = "",
): string {
  return `eda.data-operation.${kind}.${sessionId}.${discriminator}`;
}
