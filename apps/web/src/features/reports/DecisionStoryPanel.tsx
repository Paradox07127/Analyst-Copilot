/* Role-3 curation surface (§10.3): pick
 * report-eligible findings into a decision story draft, then generate a
 * decision report from one. Both writes are async jobs — completion is
 * caught locally via SSE so the story/report queries refresh once the job
 * actually lands, not on mutation success. */

import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  ApiError,
  type DecisionReportGenerationStarted,
  type DecisionStoryDraftStarted,
  type DecisionStoryFindingView,
  type DecisionStoryView,
} from "../../api/client";
import {
  queryKeys,
  useCreateDecisionStoryDraft,
  useDecisionStory,
  useGenerateDecisionReport,
} from "../../api/hooks";
import { TERMINAL_PHASES, useJobEvents } from "../../api/job-events";
import { useJobActivity } from "../../app/job-activity";
import { ErrorState, LoadingSkeleton } from "../../components/async-states";
import { Button, Card, SectionHeader } from "../../components/ui";

/* Not exported by client.ts (only the parent view is) — derived here rather
 * than editing api/client.ts, which is out of scope for this slice. */
type DecisionStoryDraftView = NonNullable<DecisionStoryView["drafts"]>[number];

const READINESS_STYLE: Record<string, string> = {
  eligible: "bg-status-ok/15 text-status-ok",
  eligible_with_limitations: "bg-status-warn/15 text-status-warn",
  not_eligible: "bg-code-bg text-status-neutral",
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

/* Mirrors the codes decision_report_service.py raises before queueing a job
 * (busy lane, un-draftable selection, unknown draft) — never show the raw
 * code to the user. */
function decisionStoryGuidance(error: unknown): string | null {
  if (!(error instanceof ApiError)) return null;
  if (error.code === "decision_story_busy")
    return "A decision story job is already running for this session. Wait for it to finish, then try again.";
  if (error.code === "decision_story_not_draftable")
    return "Some selected findings can no longer become a decision story. Refresh and re-select.";
  if (error.code === "decision_story_draft_not_found")
    return "That draft could not be found — it may have been removed. Refresh the page.";
  return error.message;
}

function findingLabel(item: DecisionStoryFindingView): string {
  const label = `${item.question} | reliability: ${item.analytical_reliability} | report: ${item.report_readiness}`;
  return item.freshness === "stale"
    ? `${label} (stale — re-run before reporting)`
    : label;
}

function FindingOption({
  item,
  checked,
  onChange,
}: {
  item: DecisionStoryFindingView;
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <label
      className={`flex cursor-pointer items-start gap-3 rounded-base border p-3 ${
        checked
          ? "border-primary bg-primary/5"
          : "border-border hover:border-primary/50"
      }`}
    >
      <input
        type="checkbox"
        className="mt-1 shrink-0"
        checked={checked}
        aria-label={findingLabel(item)}
        onChange={(event) => onChange(event.target.checked)}
      />
      <span className="flex min-w-0 flex-1 flex-col gap-1.5">
        <span className="text-sm font-medium leading-snug">{item.question}</span>
        <span className="flex flex-wrap gap-1.5">
          <Badge>{`reliability ${item.analytical_reliability}`}</Badge>
          <Badge tone={READINESS_STYLE[item.report_readiness]}>
            {`report ${item.report_readiness.replaceAll("_", " ")}`}
          </Badge>
          {item.freshness === "stale" && (
            <Badge tone="bg-status-warn/15 text-status-warn">stale</Badge>
          )}
        </span>
      </span>
    </label>
  );
}

function CurationPanel({
  projectId,
  sessionId,
  story,
}: {
  projectId: string;
  sessionId: string;
  story: DecisionStoryView;
}) {
  const queryClient = useQueryClient();
  const { startTracking } = useJobActivity();
  const createDraft = useCreateDecisionStoryDraft(sessionId);
  const [showStale, setShowStale] = useState(false);
  const [selected, setSelected] = useState<Map<string, string>>(new Map());
  const [businessContext, setBusinessContext] = useState("");
  const [started, setStarted] = useState<DecisionStoryDraftStarted | null>(
    null,
  );

  const events = useJobEvents(
    started ? started.job.job_id : null,
    started ? started.job.events_url : null,
  );
  const settledJob = useRef<string | null>(null);

  useEffect(() => {
    if (!started || !TERMINAL_PHASES.has(events.phase)) return;
    if (settledJob.current === started.job.job_id) return;
    settledJob.current = started.job.job_id;
    if (events.phase === "completed") {
      void queryClient.invalidateQueries({
        queryKey: queryKeys.decisionStory(sessionId),
      });
      setSelected(new Map());
    }
  }, [started, events.phase, queryClient, sessionId]);

  const eligible = story.eligible_findings ?? [];
  const visible = eligible.filter(
    (item) => showStale || item.freshness !== "stale",
  );

  const toggleSelected = (
    artifactId: string,
    sourceSessionId: string,
    checked: boolean,
  ) => {
    setSelected((prev) => {
      const next = new Map(prev);
      if (checked) next.set(artifactId, sourceSessionId);
      else if (next.get(artifactId) === sourceSessionId) next.delete(artifactId);
      return next;
    });
  };

  const submit = () => {
    createDraft.mutate(
      {
        finding_artifact_ids: Array.from(selected.keys()),
        finding_session_ids: Object.fromEntries(selected),
        business_context: businessContext,
      },
      {
        onSuccess: (result) => {
          setStarted(result);
          startTracking({
            jobId: result.job.job_id,
            sessionId: result.execution_session_id,
            sourceSessionId: sessionId,
            projectId,
            eventsUrl: result.job.events_url,
          });
        },
      },
    );
  };

  const guidance = decisionStoryGuidance(createDraft.error);

  return (
    <Card className="flex min-w-0 flex-col gap-4 p-4">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold">Choose approved findings</h3>
          <p className="text-xs text-status-neutral">
            {`${selected.size} selected · ${visible.length} available`}
          </p>
        </div>
        <label className="flex items-center gap-2 text-xs text-status-neutral">
          <input
            type="checkbox"
            checked={showStale}
            onChange={(event) => setShowStale(event.target.checked)}
          />
          Show stale findings
        </label>
      </div>
      <ul className="grid gap-2 lg:grid-cols-2">
        {visible.map((item) => (
          <li key={`${item.artifact_id}:${item.source_session_id}`}>
            <FindingOption
              item={item}
              checked={
                selected.get(item.artifact_id) === item.source_session_id
              }
              onChange={(checked) =>
                toggleSelected(item.artifact_id, item.source_session_id, checked)
              }
            />
          </li>
        ))}
      </ul>
      {visible.length === 0 && (
        <p className="rounded-base bg-surface p-3 text-xs text-status-neutral">
          Only stale findings are available. Show them to review why they need
          another run before reporting.
        </p>
      )}
      <label className="flex flex-col gap-1 text-sm">
        <span className="font-medium">Business context (unverified)</span>
        <span className="text-xs text-status-neutral">
          Supplied by you, not derived from validated evidence. Never a source
          for claims or numbers in the story.
        </span>
        <textarea
          value={businessContext}
          onChange={(event) => setBusinessContext(event.target.value)}
          rows={2}
          className="min-h-20 resize-y rounded-base border border-border bg-bg p-2 text-sm"
        />
      </label>
      <Button
        variant="primary"
        onClick={submit}
        disabled={selected.size === 0 || createDraft.isPending}
      >
        {createDraft.isPending ? "Creating…" : "Create decision story draft"}
      </Button>
      {guidance && (
        <p role="alert" className="text-xs text-status-warn">
          {guidance}
        </p>
      )}
    </Card>
  );
}

function DraftCard({
  projectId,
  sessionId,
  draft,
}: {
  projectId: string;
  sessionId: string;
  draft: DecisionStoryDraftView;
}) {
  const queryClient = useQueryClient();
  const { startTracking } = useJobActivity();
  const generate = useGenerateDecisionReport(sessionId);
  const [started, setStarted] =
    useState<DecisionReportGenerationStarted | null>(null);

  const events = useJobEvents(
    started ? started.job.job_id : null,
    started ? started.job.events_url : null,
  );
  const settledJob = useRef<string | null>(null);

  useEffect(() => {
    if (!started || !TERMINAL_PHASES.has(events.phase)) return;
    if (settledJob.current === started.job.job_id) return;
    settledJob.current = started.job.job_id;
    if (events.phase === "completed") {
      void queryClient.invalidateQueries({
        queryKey: queryKeys.decisionReport(sessionId),
      });
    }
  }, [started, events.phase, queryClient, sessionId]);

  const submit = () => {
    generate.mutate(
      {
        brief_artifact_id: draft.artifact_id,
        brief_session_id: draft.session_id,
      },
      {
        onSuccess: (result) => {
          setStarted(result);
          startTracking({
            jobId: result.job.job_id,
            sessionId: result.execution_session_id,
            sourceSessionId: sessionId,
            resultSessionId: draft.session_id,
            projectId,
            eventsUrl: result.job.events_url,
          });
        },
      },
    );
  };

  const guidance = decisionStoryGuidance(generate.error);
  const storyline = draft.storyline ?? [];
  const limitations = draft.limitations ?? [];
  const gaps = draft.investigation_gaps ?? [];

  return (
    <Card as="li" className="flex min-w-0 flex-col gap-3 p-4">
      <div className="flex flex-wrap items-center gap-2">
        <h4 className="text-base font-semibold leading-snug">
          {draft.headline}
        </h4>
        <Badge tone={READINESS_STYLE[draft.report_readiness]}>
          {`report ${draft.report_readiness.replaceAll("_", " ")}`}
        </Badge>
      </div>
      {storyline.map((beat, index) => (
        <div key={index} className="flex flex-col gap-0.5">
          <p className="text-sm font-medium">{beat.title}</p>
          <p className="text-sm">{beat.body}</p>
        </div>
      ))}
      {limitations.length > 0 && (
        <p className="text-xs text-status-neutral">
          Limitations: {limitations.join(" | ")}
        </p>
      )}
      {gaps.length > 0 && (
        <p className="text-xs text-status-neutral">
          Open investigations: {gaps.join(" | ")}
        </p>
      )}
      {draft.business_context.trim() && (
        <div className="rounded-base border border-border p-3">
          <p className="text-xs font-semibold">
            User-provided context (unverified)
          </p>
          <p className="text-xs text-status-neutral">
            Supplied by the user, not derived from validated evidence. Never a
            source for claims or numbers in the story above.
          </p>
          <p className="mt-1 text-sm">{draft.business_context}</p>
        </div>
      )}
      <p className="text-xs text-status-neutral">
        {`Synthesis run ${draft.session_id} · ${
          (draft.selected_finding_artifact_ids ?? []).length
        } selected finding(s) · report: ${draft.report_readiness}`}
      </p>
      <Button
        onClick={submit}
        disabled={generate.isPending}
      >
        {generate.isPending ? "Starting…" : "Generate decision report"}
      </Button>
      {guidance && (
        <p role="alert" className="text-xs text-status-warn">
          {guidance}
        </p>
      )}
    </Card>
  );
}

export function DecisionStoryPanel({
  projectId,
  sessionId,
}: {
  projectId: string;
  sessionId: string;
}) {
  const story = useDecisionStory(sessionId);

  if (story.isPending)
    return <LoadingSkeleton lines={4} label="Loading decision story" />;
  if (story.isError)
    return <ErrorState error={story.error} onRetry={() => story.refetch()} />;
  if (!story.data) return null;

  const eligible = story.data.eligible_findings ?? [];
  const drafts = story.data.drafts ?? [];
  const warnings = story.data.warnings ?? [];

  /* Nothing to curate and nothing drafted: a muted line, not an empty panel
   * (Decision Story selection rules). */
  if (eligible.length === 0 && drafts.length === 0) {
    return (
      <section
        aria-labelledby="decision-story-heading"
        className="flex flex-col gap-2 border-t border-hairline pt-6"
      >
        <SectionHeader
          title={<span id="decision-story-heading">Decision Story</span>}
          description="Optional authoring workspace for turning approved findings into the concise decision report above."
        />
        <p className="text-xs text-status-neutral">
          No story is drafted yet. Approve findings on the Questions page to
          build one.
        </p>
      </section>
    );
  }

  return (
    <section
      aria-labelledby="decision-story-heading"
      className="flex min-w-0 flex-col gap-4 border-t border-hairline pt-6"
    >
      <SectionHeader
        title={<span id="decision-story-heading">Decision Story</span>}
        description="Optional authoring workspace. Select approved findings, review the draft, then publish the concise decision report above."
      />
      {eligible.length > 0 ? (
        <CurationPanel projectId={projectId} sessionId={sessionId} story={story.data} />
      ) : (
        <p className="text-xs text-status-neutral">
          No report-eligible validated findings are available in this
          project.
        </p>
      )}
      {warnings.map((warning) => (
        <p key={warning} role="alert" className="text-xs text-status-warn">
          {warning}
        </p>
      ))}
      {drafts.length > 0 && (
        <div className="flex flex-col gap-2">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <h3 className="text-sm font-semibold">Drafts</h3>
            <span className="text-xs text-status-neutral">
              {`${drafts.length} saved`}
            </span>
          </div>
          <ul className="grid gap-3 xl:grid-cols-2">
            {drafts.map((draft) => (
              <DraftCard
                key={draft.artifact_id}
                projectId={projectId}
                sessionId={sessionId}
                draft={draft}
              />
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
