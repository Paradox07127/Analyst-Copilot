/* "Run again" for a failed or cancelled main analysis: the server re-queues
 * the job with its persisted parameters, and the new run lands in Activity. */

import { useRef } from "react";
import { useRetryJob } from "../api/hooks";
import { useJobActivity } from "../app/job-activity";
import { Button } from "./ui";

export function RunAgainButton({
  jobId,
  sessionId,
  projectId,
  sourceSessionId,
}: {
  jobId: string;
  sessionId: string;
  projectId: string;
  sourceSessionId?: string | null;
}) {
  const retry = useRetryJob(sessionId);
  const { startTracking, setPanelOpen } = useJobActivity();
  /* One key per attempt: a retry after a network failure replays the same
   * queued job instead of enqueuing a second run. */
  const keyRef = useRef<string | null>(null);

  const start = () => {
    keyRef.current ??= crypto.randomUUID();
    retry.mutate(
      { jobId, idempotencyKey: keyRef.current },
      {
        onSuccess: (created) => {
          keyRef.current = null;
          startTracking({
            jobId: created.job_id,
            sessionId: created.session_id,
            sourceSessionId: sourceSessionId ?? created.session_id,
            projectId,
            eventsUrl: created.events_url,
          });
          setPanelOpen(true);
        },
        onError: () => {
          keyRef.current = null;
        },
      },
    );
  };

  return (
    <span className="flex flex-col items-start gap-1">
      <Button size="sm" onClick={start} disabled={retry.isPending}>
        {retry.isPending ? "Starting…" : "Run again"}
      </Button>
      {retry.isError && (
        <span role="alert" className="text-xs text-status-critical">
          {retry.error instanceof Error
            ? retry.error.message
            : "The analysis could not be restarted."}
        </span>
      )}
    </span>
  );
}
