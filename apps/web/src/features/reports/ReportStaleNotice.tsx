/* A question executed after the report was written leaves the report silently
 * out of date — the exec job lands on a derived run and never regenerates the
 * report. This banner compares the newest completed question_exec job against
 * the report's own generated_at and offers the existing regenerate path. */

import { useRef, useState } from "react";
import { Link } from "react-router";
import type { ReportView, SessionJobList } from "../../api/client";
import { useGenerateReport, useReport, useSessionJobs } from "../../api/hooks";
import { useJobActivity } from "../../app/job-activity";
import { sessionSectionPath } from "../../app/paths";
import { Button, Card } from "../../components/ui";

export function isReportStale(
  report: ReportView | undefined,
  jobs: SessionJobList | undefined,
): boolean {
  if (!report || report.status === "none" || !report.generated_at) return false;
  const generated = Date.parse(report.generated_at);
  if (Number.isNaN(generated)) return false;
  return (jobs?.jobs ?? []).some(
    (job) =>
      job.kind === "question_exec" &&
      job.status === "completed" &&
      typeof job.finished_at === "string" &&
      Date.parse(job.finished_at) > generated,
  );
}

export function ReportStaleNotice({
  projectId,
  sessionId,
  showReportLink = false,
}: {
  projectId: string;
  sessionId: string;
  /** Set on pages other than the report itself. */
  showReportLink?: boolean;
}) {
  const report = useReport(sessionId);
  const jobs = useSessionJobs(sessionId);
  const generate = useGenerateReport(sessionId);
  const { startTracking } = useJobActivity();
  const [started, setStarted] = useState(false);
  /* One key per attempt, same as the page's own regenerate control: a retry
   * after a network failure replays the job instead of queuing a second one. */
  const keyRef = useRef<string | null>(null);

  if (!isReportStale(report.data, jobs.data)) return null;

  const start = () => {
    keyRef.current ??= crypto.randomUUID();
    generate.mutate(
      { llm: "env", idempotencyKey: keyRef.current },
      {
        onSuccess: (startedJob) => {
          keyRef.current = null;
          setStarted(true);
          startTracking({
            jobId: startedJob.job.job_id,
            sessionId: startedJob.job.session_id,
            sourceSessionId: sessionId,
            projectId,
            eventsUrl: startedJob.job.events_url,
          });
        },
        onError: () => {
          keyRef.current = null;
        },
      },
    );
  };

  return (
    <Card
      role="status"
      tone="warn"
      aria-label="Report out of date"
      className="flex flex-wrap items-center gap-x-3 gap-y-2 p-3"
    >
      <p className="min-w-52 flex-1 text-sm text-status-warn">
        {started
          ? "Regenerating the report — it will refresh when the run finishes."
          : "The report does not include your latest question results."}
      </p>
      {!started && (
        <Button size="sm" onClick={start} disabled={generate.isPending}>
          {generate.isPending ? "Starting…" : "Regenerate report"}
        </Button>
      )}
      {showReportLink && (
        <Link
          to={sessionSectionPath(projectId, sessionId, "report")}
          className="text-sm font-medium text-primary hover:underline"
        >
          Open the report
        </Link>
      )}
      {generate.isError && (
        <p role="alert" className="w-full text-xs text-status-critical">
          {generate.error instanceof Error
            ? generate.error.message
            : "Could not start report generation."}
        </p>
      )}
    </Card>
  );
}
