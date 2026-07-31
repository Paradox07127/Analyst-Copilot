/* Role-3 decision report, shown above the technical report (§10.3, the block
 * rendered at the top of the Report page). Renders nothing
 * when the project has no decision report — no empty shell. */

import type { DecisionReportView } from "../../api/client";

const FRESHNESS_STYLE: Record<string, string> = {
  fresh: "bg-status-ok/15 text-status-ok",
  stale: "bg-status-critical/15 text-status-critical",
  unverifiable: "bg-status-warn/15 text-status-warn",
};

const READINESS_STYLE: Record<string, string> = {
  eligible: "bg-status-ok/15 text-status-ok",
  eligible_with_limitations: "bg-status-warn/15 text-status-warn",
  not_eligible: "bg-code-bg text-status-neutral",
};

const GATE_STYLE: Record<string, string> = {
  pass: "bg-status-ok/15 text-status-ok",
  degraded: "bg-status-warn/15 text-status-warn",
  rejected: "bg-status-critical/15 text-status-critical",
};

const CONFIDENCE_STYLE: Record<string, string> = {
  high: "bg-status-ok/15 text-status-ok",
  medium: "bg-status-warn/15 text-status-warn",
  low: "bg-status-critical/15 text-status-critical",
};

function Badge({ tone, children }: { tone?: string; children: string }) {
  return (
    <span
      className={`rounded-base px-1.5 py-0.5 text-[10px] font-medium uppercase ${
        tone ?? "bg-code-bg text-status-neutral"
      }`}
    >
      {children}
    </span>
  );
}

function Block({ label, body }: { label: string; body: string }) {
  return (
    <div className="flex min-w-0 flex-col gap-1">
      <h3 className="text-xs font-semibold text-status-neutral uppercase">
        {label}
      </h3>
      <p className="text-sm leading-relaxed">{body}</p>
    </div>
  );
}

export function DecisionReportCard({
  report,
  selectedEvidenceId,
  selectedEvidenceSessionId,
  onInspect,
}: {
  report: DecisionReportView;
  selectedEvidenceId: string | null;
  selectedEvidenceSessionId: string | null;
  onInspect: (artifactId: string, sessionId: string | null) => void;
}) {
  if (report.status !== "available") return null;
  const freshness = report.freshness?.status ?? "unverifiable";
  const reasons = report.freshness?.reasons ?? [];
  const candidates = report.candidate_decisions ?? [];
  const evidence = report.evidence_refs ?? [];

  return (
    <article className="flex min-w-0 flex-col gap-5 rounded-base border border-border bg-bg p-4 shadow-card sm:p-5">
      <div className="flex flex-col gap-2">
        <h2 className="text-lg font-semibold leading-snug">
          {report.title ?? "Decision story"}
        </h2>
        <div className="flex flex-wrap items-center gap-2">
          {report.report_readiness && (
            <Badge tone={READINESS_STYLE[report.report_readiness]}>
              {`report ${report.report_readiness.replaceAll("_", " ")}`}
            </Badge>
          )}
          <Badge tone={FRESHNESS_STYLE[freshness]}>{`freshness ${freshness}`}</Badge>
          {report.gate_verdict && (
            <Badge tone={GATE_STYLE[report.gate_verdict]}>
              {`gate ${report.gate_verdict}`}
            </Badge>
          )}
          {report.confidence_label && (
            <Badge tone={CONFIDENCE_STYLE[report.confidence_label]}>
              {`confidence ${report.confidence_label}`}
            </Badge>
          )}
          {report.narrative_status && <Badge>{report.narrative_status}</Badge>}
        </div>
        {(report.publication_status || report.report_session_id) && (
          <p className="text-xs text-status-neutral">
            {report.publication_status && (
              <>
                Publication: <span>{report.publication_status}</span>
              </>
            )}
            {report.publication_status && report.report_session_id && " · "}
            {report.report_session_id &&
              `Synthesis run ${report.report_session_id}`}
          </p>
        )}
        {freshness !== "fresh" && (
          <div
            role="alert"
            className="rounded-base border border-status-warn/50 p-3 text-xs text-status-warn"
          >
            <p>
              {freshness === "stale"
                ? "This report is published history but is stale for current reuse. Re-run the affected investigation before exporting it."
                : "This report is published history, but current freshness cannot be verified. Export remains disabled."}
            </p>
            {reasons[0] && <p className="mt-1">{reasons[0]}</p>}
          </div>
        )}
      </div>

      {report.scqa && (
        <div className="flex flex-col gap-4">
          <div className="rounded-base border border-primary/30 bg-primary/5 p-4">
            <Block label="Answer" body={report.scqa.answer} />
          </div>
          <div className="grid gap-4 sm:grid-cols-3">
            <Block label="Situation" body={report.scqa.situation} />
            <Block label="Complication" body={report.scqa.complication} />
            <Block label="Question" body={report.scqa.question} />
          </div>
        </div>
      )}

      {(report.sections ?? []).length > 0 && (
        <div className="grid gap-4 border-t border-hairline pt-4 md:grid-cols-2">
          {(report.sections ?? []).map((section, index) => (
            <div key={index} className="flex min-w-0 flex-col gap-1">
              <h3 className="text-sm font-semibold">{section.title}</h3>
              <p className="text-sm leading-relaxed">{section.body}</p>
            </div>
          ))}
        </div>
      )}

      {candidates.length > 0 && (
        <section className="flex flex-col gap-2 border-t border-hairline pt-4">
          <h3 className="text-sm font-semibold">Candidate decisions</h3>
          <p className="text-xs text-status-neutral">
            Proposed actions carried by the source findings. Hypothesis context —
            not validated evidence, and never a source for the numbers above.
          </p>
          <ul className="flex flex-col gap-2">
            {candidates.map((candidate) => (
              <li
                key={candidate.finding_artifact_id}
                className="flex flex-col gap-2 rounded-base border border-border p-3"
              >
                <p className="text-sm font-medium leading-relaxed">
                  {candidate.decision_action}
                </p>
                <div className="flex flex-wrap items-center gap-2">
                  <Badge tone={CONFIDENCE_STYLE[candidate.analytical_reliability]}>
                    {`reliability ${candidate.analytical_reliability}`}
                  </Badge>
                  <Badge>{`decision ${candidate.decision_readiness}`}</Badge>
                </div>
                <p className="text-xs text-status-neutral">
                  Source question: {candidate.question}
                </p>
              </li>
            ))}
          </ul>
        </section>
      )}

      {((report.limitations ?? []).length > 0 ||
        (report.investigation_gaps ?? []).length > 0) && (
        <section className="grid gap-4 rounded-base bg-surface p-3 sm:grid-cols-2">
          {(report.limitations ?? []).length > 0 && (
            <div className="flex flex-col gap-1">
              <h3 className="text-xs font-semibold">Limitations</h3>
              <ul className="list-disc space-y-1 pl-4 text-xs text-status-neutral">
                {(report.limitations ?? []).map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </div>
          )}
          {(report.investigation_gaps ?? []).length > 0 && (
            <div className="flex flex-col gap-1">
              <h3 className="text-xs font-semibold">Open investigations</h3>
              <ul className="list-disc space-y-1 pl-4 text-xs text-status-neutral">
                {(report.investigation_gaps ?? []).map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </div>
          )}
        </section>
      )}

      {/* Opens beside the story rather than navigating to Artifacts. Keep the
          source run attached: decision stories can cite project findings from
          a different analysis run. */}
      {evidence.length > 0 && (
        <div className="flex flex-wrap items-center gap-2 border-t border-hairline pt-4 text-xs">
          <span className="font-medium text-status-neutral">
            Supporting evidence
          </span>
          {evidence.map((ref) => (
            <button
              key={`${ref.session_id ?? "current"}:${ref.artifact_id}`}
              type="button"
              onClick={() => onInspect(ref.artifact_id, ref.session_id ?? null)}
              aria-pressed={
                ref.artifact_id === selectedEvidenceId &&
                (ref.session_id ?? report.session_id) ===
                  selectedEvidenceSessionId
              }
              title="Open this artifact beside the story"
              className={`rounded-sm border px-1 py-0.5 font-mono text-xs hover:border-primary ${
                ref.artifact_id === selectedEvidenceId &&
                (ref.session_id ?? report.session_id) ===
                  selectedEvidenceSessionId
                  ? "border-primary bg-primary/10 text-primary"
                  : "border-border text-primary"
              }`}
            >
              {ref.artifact_id}
            </button>
          ))}
        </div>
      )}

      <p className="text-xs text-status-neutral">
        {`Source findings: ${(report.source_finding_artifact_ids ?? []).length} · export ${
          report.export_available ? "available" : "disabled"
        }`}
      </p>
    </article>
  );
}
