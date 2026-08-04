import { useMemo, useRef, useState } from "react";
import { useParams } from "react-router";
import {
  ApiError,
  type DecisionReportView,
  type ReportExportFormat,
} from "../../api/client";
import {
  useCapabilities,
  useDecisionReport,
  useDownloadReport,
  useGenerateReport,
  useReport,
} from "../../api/hooks";
import { useJobActivity } from "../../app/job-activity";
import {
  EmptyState,
  ErrorState,
  LoadingSkeleton,
} from "../../components/async-states";
import {
  Badge,
  Button,
  Card,
  Hint,
  SectionHeader,
  type Tone,
} from "../../components/ui";
import { DecisionReportCard } from "./DecisionReportCard";
import { DecisionStoryPanel } from "./DecisionStoryPanel";
import { EvidenceInspector } from "./EvidenceInspector";
import { ReportBody } from "./ReportBody";
import { ReportContents } from "./ReportContents";
import {
  citationsFor,
  readReportOutline,
  type ReportHeading,
} from "./report-outline";
import "./report-markdown.css";

const OK_STATUSES = ["validated", "final", "generated"];

const GATE_TONE: Record<string, Tone> = {
  pass: "ok",
  degraded: "warn",
  rejected: "critical",
};

function StatusBadge({ status }: { status: string }) {
  const ok = OK_STATUSES.includes(status.toLowerCase());
  return (
    <Badge tone={ok ? "ok" : "neutral"} variant="outline" caps>
      {status}
    </Badge>
  );
}

/* The exporter prints the gate verdict as prose in the report body; it is the
 * run's trust signal, so it is hoisted here and explained on demand. */
function GateSignal({ verdict }: { verdict: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <Badge tone={GATE_TONE[verdict] ?? "neutral"} caps>
        {`Gate ${verdict}`}
      </Badge>
      <Hint label="Evidence gate">
        The check the report runs against its own claims after writing them.{" "}
        <strong>pass</strong> — at least 60% of claims reached the strongest
        evidence tier. <strong>degraded</strong> — the report was published, but
        fewer than that did; most claims are indicative or exploratory.{" "}
        <strong>rejected</strong> — a validator raised a critical finding.
        Degraded is a legitimate outcome, not an error: read the Claim Ledger
        before quoting a figure.
      </Hint>
    </span>
  );
}

function formatGeneratedAt(iso: string | null | undefined): string | null {
  if (!iso) return null;
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return null;
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function DecisionReportReadError({
  error,
  onRetry,
}: {
  error: unknown;
  onRetry: () => void;
}) {
  const code = error instanceof ApiError ? error.code : "";
  const message =
    code === "decision_report_missing"
      ? "A stored decision report exists, but its file is missing."
      : code === "decision_report_unavailable"
        ? "The stored decision report is temporarily unavailable."
        : code.startsWith("decision_report_")
          ? "The stored decision report is unreadable or invalid."
          : "The stored decision report could not be loaded.";
  return (
    <Card
      role="alert"
      aria-label="Stored decision report unavailable"
      className="flex flex-col gap-2 border-status-critical/40"
    >
      <h2 className="text-sm font-semibold text-status-critical">
        Stored decision report unavailable
      </h2>
      <p className="text-sm text-status-neutral">{message}</p>
      <button
        type="button"
        onClick={onRetry}
        className="self-start rounded-base border border-border px-3 py-1.5 text-sm hover:bg-surface"
      >
        Retry stored report
      </button>
    </Card>
  );
}

/* The page has two different documents. Keep their formats in one compact
 * export group, but label the document each format belongs to so "Markdown"
 * cannot be mistaken for another rendering of the technical report. */
function DownloadButtons({
  sessionId,
  hasReport,
  decisionReport,
  decisionReportPending,
}: {
  sessionId: string;
  hasReport: boolean;
  decisionReport: DecisionReportView | undefined;
  decisionReportPending: boolean;
}) {
  const download = useDownloadReport(sessionId);
  const capabilities = useCapabilities();

  const pdfChecking = capabilities.isPending;
  const pdfAvailable =
    capabilities.isSuccess && capabilities.data.pdf_export_available;
  /* Freshness gates the button, not the endpoint — the same place the old
   * Decision Story panel put it. */
  const decisionExportable =
    decisionReport?.status === "available" && decisionReport.export_available;
  const noReportReason = "Generate a report before exporting it.";
  const decisionReason =
    decisionReportPending
      ? "Checking the decision report…"
      : decisionReport?.status === "available"
      ? "The decision report is out of date with its source findings."
      : "No decision story has been published for this project yet.";

  const options: {
    format: ReportExportFormat;
    label: string;
    disabled: boolean;
    reason?: string;
  }[] = [
    {
      format: "html",
      label: "HTML",
      disabled: !hasReport,
      reason: hasReport ? undefined : noReportReason,
    },
    {
      format: "pdf",
      label: "PDF",
      disabled: !hasReport || pdfChecking || !pdfAvailable,
      reason: !hasReport
        ? noReportReason
        : pdfChecking
          ? "Checking PDF export support…"
        : pdfAvailable
          ? undefined
          : (capabilities.data?.pdf_export_hint ??
            "This host has no PDF renderer installed."),
    },
    {
      format: "md",
      label: "Decision report (MD)",
      disabled: !decisionExportable,
      reason: decisionExportable ? undefined : decisionReason,
    },
  ];

  const blockedReasons = [
    ...new Set(
      options
        .filter((option) => option.disabled && option.reason)
        .map((option) => option.reason as string),
    ),
  ];

  const button = (option: (typeof options)[number]) => (
    <Button
      key={option.format}
      size="sm"
      className="min-w-20 flex-1 sm:flex-none"
      disabled={option.disabled || download.isPending}
      title={option.reason}
      onClick={() => download.mutate(option.format)}
    >
      {download.isPending && download.variables === option.format
        ? "Preparing…"
        : `Download ${option.label}`}
    </Button>
  );

  return (
    <div className="flex w-full flex-col items-stretch gap-1 sm:w-auto sm:items-end">
      {/* flex-wrap + intrinsic-width sections: compressing the sections let
        * the PDF button overprint the "Decision story" label at laptop
        * widths; wrapping keeps each export group intact instead. */}
      <Card
        tone="quiet"
        role="group"
        aria-label="Report exports"
        className="flex w-full flex-col gap-2 px-2.5 py-2 sm:w-auto sm:flex-row sm:flex-wrap sm:items-center"
      >
        <div className="flex items-center gap-2">
          <span className="w-24 shrink-0 text-xs text-status-neutral sm:w-auto">
            Technical
          </span>
          <div className="flex gap-2">
            {options.slice(0, 2).map(button)}
          </div>
        </div>
        <span aria-hidden className="hidden h-5 w-px shrink-0 bg-hairline sm:block" />
        <div className="flex items-center gap-2">
          <span className="w-24 shrink-0 text-xs text-status-neutral sm:w-auto">
            Decision story
          </span>
          <div className="flex">
            {options.slice(2).map(button)}
          </div>
        </div>
      </Card>
      {blockedReasons.map((reason) => (
        <p
          key={reason}
          className="max-w-sm text-left text-xs text-status-neutral sm:text-right"
        >
          {reason}
        </p>
      ))}
      {download.isError && (
        <p role="alert" className="text-xs text-status-critical">
          {download.error instanceof Error
            ? download.error.message
            : "Download failed."}
        </p>
      )}
    </div>
  );
}

/* The chart inventory, the claim ledger and the per-figure verification trace
 * are what make a claim checkable, and together they are about half the
 * report's words — a reader looking for the findings had to scroll past ~116
 * chart entries to reach them. Open on demand, never removed. */
function EvidenceRecord({
  markdown,
  headings,
  inspectableIds,
  selectedId,
  onInspect,
}: {
  markdown: string;
  headings: ReportHeading[];
  inspectableIds: Set<string>;
  selectedId: string | null;
  onInspect: (artifactId: string) => void;
}) {
  /* Every level, not just h2: the exporter writes Claim Ledger and its
   * neighbours as h3, and they are the reason to open this. */
  const names = headings.map((heading) =>
    heading.text.replace(/^Appendix:\s*/, ""),
  );
  return (
    <details className="group min-w-0 rounded-base border border-border bg-surface">
      <summary className="flex cursor-pointer list-none flex-wrap items-baseline gap-x-2 gap-y-1 rounded-base px-4 py-3 hover:bg-bg">
        <span className="text-sm font-semibold">Evidence record</span>
        <span className="text-xs text-status-neutral">
          How every figure above was checked
        </span>
        <span className="ml-auto text-xs text-primary group-open:hidden">
          Show
        </span>
        <span className="ml-auto hidden text-xs text-primary group-open:inline">
          Hide
        </span>
        {names.length > 0 && (
          <span className="w-full text-xs text-status-neutral">
            {names.join(" · ")}
          </span>
        )}
      </summary>
      <div className="min-w-0 border-t border-hairline p-4 sm:p-6">
        <ReportBody
          markdown={markdown}
          inspectableIds={inspectableIds}
          selectedId={selectedId}
          onInspect={onInspect}
        />
      </div>
    </details>
  );
}

function GenerateReportControl({
  projectId,
  sessionId,
  hasReport,
}: {
  projectId: string;
  sessionId: string;
  hasReport: boolean;
}) {
  const generate = useGenerateReport(sessionId);
  const { startTracking } = useJobActivity();
  const [confirming, setConfirming] = useState(false);
  /* One key per attempt: a retry after a network failure replays the same job
   * instead of queuing a second generation. */
  const keyRef = useRef<string | null>(null);

  const start = () => {
    keyRef.current ??= crypto.randomUUID();
    generate.mutate(
      { llm: "env", idempotencyKey: keyRef.current },
      {
        onSuccess: (started) => {
          keyRef.current = null;
          setConfirming(false);
          startTracking({
            jobId: started.job.job_id,
            sessionId: started.job.session_id,
            sourceSessionId: sessionId,
            projectId,
            eventsUrl: started.job.events_url,
          });
        },
        onError: () => {
          /* A conflict cannot be helped by replaying the same key. */
          keyRef.current = null;
        },
      },
    );
  };

  if (hasReport && confirming) {
    return (
      <Card
        tone="warn"
        className="flex flex-wrap items-center gap-2 p-2 sm:w-auto"
      >
        <span className="min-w-52 flex-1 text-xs text-status-warn">
          Regenerating replaces this report — the run keeps only the newest one.
        </span>
        <Button
          variant="danger"
          size="sm"
          onClick={start}
          disabled={generate.isPending}
        >
          {generate.isPending ? "Starting…" : "Replace report"}
        </Button>
        <Button
          size="sm"
          onClick={() => setConfirming(false)}
        >
          Keep current
        </Button>
      </Card>
    );
  }

  return (
    <div className="flex w-full flex-col items-start gap-1 sm:w-auto sm:items-end">
      <Button
        variant={hasReport ? "ghost" : "primary"}
        onClick={() => (hasReport ? setConfirming(true) : start())}
        disabled={generate.isPending}
      >
        {generate.isPending
          ? "Starting…"
          : hasReport
            ? "Regenerate report"
            : "Generate report"}
      </Button>
      {generate.isError && (
        <p role="alert" className="text-xs text-status-critical">
          {generate.error instanceof Error
            ? generate.error.message
            : "Could not start report generation."}
        </p>
      )}
    </div>
  );
}

export function Component() {
  const { projectId = "", sessionId = "" } = useParams();
  const report = useReport(sessionId);
  /* Decision report is project-level and optional: it renders above the report
   * body when one exists, and is silent otherwise (including on error). */
  const decisionReport = useDecisionReport(sessionId);
  const generatedAt = formatGeneratedAt(report.data?.generated_at);
  const hasReport = Boolean(report.data && report.data.status !== "none");
  const [inspecting, setInspecting] = useState<{
    artifactId: string;
    sessionId: string;
  } | null>(null);

  const outline = useMemo(
    () => (report.data ? readReportOutline(report.data.markdown) : null),
    [report.data],
  );

  return (
    <div className="mx-auto flex w-[95%] max-w-data flex-col gap-5 p-4 sm:p-6">
      <header className="flex flex-col gap-4 border-b border-hairline pb-5 lg:flex-row lg:items-start">
        <div className="flex min-w-0 flex-col gap-1">
          <div className="flex flex-wrap items-center gap-2">
            <h1 className="text-xl font-semibold">Report</h1>
            {report.data && <StatusBadge status={report.data.status} />}
            {outline?.gate && <GateSignal verdict={outline.gate} />}
          </div>
          <p className="max-w-2xl text-sm text-status-neutral">
            Read the decision-ready story first, then verify its claims in the
            current run&apos;s technical report and evidence.
          </p>
          {generatedAt && (
            <span className="text-xs text-status-neutral">
              Generated {generatedAt}
            </span>
          )}
        </div>
        <div className="flex w-full flex-col items-stretch gap-2 lg:ml-auto lg:w-auto lg:items-end">
          <DownloadButtons
            sessionId={sessionId}
            hasReport={hasReport}
            decisionReport={decisionReport.data}
            decisionReportPending={decisionReport.isPending}
          />
          {report.data && (
            <GenerateReportControl
              projectId={projectId}
              sessionId={sessionId}
              hasReport={hasReport}
            />
          )}
        </div>
      </header>

      {/* The inspector is a sibling of everything readable, so evidence cited
          by the decision story and by the report body opens in one place. */}
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start">
        <div className="flex min-w-0 flex-1 flex-col gap-8">
          {decisionReport.isError && (
            <DecisionReportReadError
              error={decisionReport.error}
              onRetry={() => decisionReport.refetch()}
            />
          )}

          {decisionReport.data?.status === "available" && (
            <section
              aria-labelledby="decision-report-heading"
              className="flex min-w-0 flex-col gap-3"
            >
              <SectionHeader
                title={
                  <span id="decision-report-heading">Decision report</span>
                }
                description="The concise, shareable narrative synthesized from approved findings."
              />
              <DecisionReportCard
                report={decisionReport.data}
                selectedEvidenceId={inspecting?.artifactId ?? null}
                selectedEvidenceSessionId={inspecting?.sessionId ?? null}
                onInspect={(artifactId, sourceSessionId) =>
                  setInspecting({
                    artifactId,
                    sessionId: sourceSessionId ?? sessionId,
                  })
                }
              />
            </section>
          )}

          <section
            aria-labelledby="technical-report-heading"
            className="flex min-w-0 flex-col gap-3"
          >
            <SectionHeader
              title={<span id="technical-report-heading">Technical report</span>}
              description="The current analysis run’s complete evidence-backed narrative. Evidence links open without losing your reading position."
            />
            {report.isPending && (
              <LoadingSkeleton lines={6} label="Loading report" />
            )}
            {report.isError && (
              <ErrorState error={report.error} onRetry={() => report.refetch()} />
            )}
            {report.data &&
              (report.data.status === "none" ? (
                <EmptyState
                  title="No technical report yet"
                  description="Generate one to re-read this run’s evidence and write an evidence-checked report."
                />
              ) : (
                outline && (
                  <>
                    <ReportContents
                      outline={outline}
                      hasEvidence={outline.inspectableIds.size > 0}
                    />
                    <Card className="min-w-0 p-4 sm:p-6">
                      <ReportBody
                        markdown={outline.narrative}
                        inspectableIds={outline.inspectableIds}
                        selectedId={
                          inspecting?.sessionId === sessionId
                            ? inspecting.artifactId
                            : null
                        }
                        onInspect={(artifactId) =>
                          setInspecting({ artifactId, sessionId })
                        }
                      />
                    </Card>
                    {outline.reference && (
                      <EvidenceRecord
                        markdown={outline.reference}
                        headings={outline.referenceHeadings}
                        inspectableIds={outline.inspectableIds}
                        selectedId={
                          inspecting?.sessionId === sessionId
                            ? inspecting.artifactId
                            : null
                        }
                        onInspect={(artifactId) =>
                          setInspecting({ artifactId, sessionId })
                        }
                      />
                    )}
                  </>
                )
              ))}
          </section>

          <DecisionStoryPanel projectId={projectId} sessionId={sessionId} />
        </div>
        {inspecting && (
          <EvidenceInspector
            projectId={projectId}
            sessionId={inspecting.sessionId}
            artifactId={inspecting.artifactId}
            citations={
              outline && inspecting.sessionId === sessionId
                ? citationsFor(outline.ledger, inspecting.artifactId)
                : []
            }
            onClose={() => setInspecting(null)}
          />
        )}
      </div>
    </div>
  );
}
