import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router";
import { useQueryClient } from "@tanstack/react-query";
import type {
  ExplorationBudgetIncrease,
  ExplorationEventDto,
  ExplorationPreparedDto,
  ExplorationTierDto,
} from "../../api/client";
import { useExplorationEvents } from "../../api/exploration-events";
import {
  queryKeys,
  useCancelExploration,
  useDatasets,
  useExploration,
  useExplorationReport,
  useExtendExplorationBudget,
  usePauseExploration,
  usePrepareExploration,
  useResumeExploration,
  useStartExploration,
} from "../../api/hooks";
import { explorationRunPath } from "../../app/paths";
import {
  ErrorState,
  LoadingSkeleton,
  formatUnknownError,
} from "../../components/async-states";
import { Badge, Card, MetricStrip, MetricTile, SectionHeader } from "../../components/ui";
import { explorationRunFromDto } from "./exploration-adapter";
import {
  readExplorationGoal,
  writeLastExplorationId,
} from "./exploration-goal-storage";
import { ExplorationReport, ExplorationRunPanel } from "./ExplorationView";

function usd(value: string | number | null): string {
  const parsed = Number(value ?? 0);
  return `$${(Number.isFinite(parsed) ? parsed : 0).toFixed(2)}`;
}

function HardCaps({ prepared }: { prepared: ExplorationPreparedDto }) {
  const budget = prepared.policy.budget;
  return (
    <section aria-label="Exploration hard caps" className="flex flex-col gap-2">
      <SectionHeader
        level={3}
        title={`${prepared.policy.thinking_level} hard caps`}
        description="These are server-sealed ceilings, not targets or estimates."
      />
      <MetricStrip>
        <MetricTile label="Model requests" value={budget.llm.max_requests ?? "—"} />
        <MetricTile label="Tool calls" value={budget.max_successful_tool_calls} />
        <MetricTile label="Rounds" value={budget.max_rounds} />
        <MetricTile
          label="Cost range"
          value={`${usd(prepared.cost_range.minimum_usd)}–${usd(prepared.cost_range.maximum_usd)}`}
          hint="policy hard cap · not exact"
        />
      </MetricStrip>
    </section>
  );
}

function ExplorationLaunch({ projectId, sessionId }: { projectId: string; sessionId: string }) {
  const navigate = useNavigate();
  const datasets = useDatasets(sessionId);
  const prepare = usePrepareExploration(sessionId);
  const start = useStartExploration(sessionId);
  /* A goal typed on the new-session screen decides the mode too: arriving with
   * "Explore freely" pre-selected would silently discard what the user asked
   * for. */
  const carriedGoal = useRef(readExplorationGoal(sessionId)).current;
  const [mode, setMode] = useState<"open" | "goal_directed">(
    carriedGoal ? "goal_directed" : "open",
  );
  const [goal, setGoal] = useState(carriedGoal);
  const [tier, setTier] = useState<ExplorationTierDto>("standard");
  const [selected, setSelected] = useState<string[]>([]);
  const initializedScope = useRef(false);
  const startKey = useRef<string | null>(null);

  useEffect(() => {
    if (!initializedScope.current && datasets.data?.length) {
      initializedScope.current = true;
      setSelected(datasets.data.map((dataset) => dataset.dataset_id));
    }
  }, [datasets.data]);

  const resetApproval = () => {
    prepare.reset();
    start.reset();
    startKey.current = null;
  };
  const submitPrepare = () => {
    prepare.mutate(
      {
        mode,
        goal: mode === "goal_directed" ? goal.trim() : null,
        dataset_ids: selected,
        thinking_level: tier,
      },
      { onSuccess: () => { startKey.current = crypto.randomUUID(); } },
    );
  };
  const authorize = () => {
    const prepared = prepare.data;
    if (!prepared) return;
    startKey.current ??= crypto.randomUUID();
    start.mutate(
      {
        action_hash: prepared.action_hash,
        approval_token: prepared.approval_token,
        idempotencyKey: startKey.current,
      },
      {
        onSuccess: (result) => {
          const { exploration_id: explorationId } = result.exploration;
          writeLastExplorationId(sessionId, explorationId);
          navigate(explorationRunPath(projectId, sessionId, explorationId));
        },
      },
    );
  };

  if (datasets.isLoading) return <LoadingSkeleton label="Loading exploration datasets" />;
  if (datasets.error) return <ErrorState error={datasets.error} onRetry={() => void datasets.refetch()} />;

  return (
    <main className="mx-auto flex w-full max-w-5xl flex-col gap-5 p-4 sm:p-6">
      <SectionHeader
        title="Read-only exploration"
        description="Explore freely or investigate a goal without modifying source datasets."
      />
      <Card as="section" className="flex flex-col gap-4 p-4">
        <fieldset className="flex flex-col gap-2">
          <legend className="text-sm font-medium">Mode</legend>
          <div className="flex gap-2">
            {(["open", "goal_directed"] as const).map((value) => (
              <button
                key={value}
                type="button"
                aria-pressed={mode === value}
                onClick={() => { setMode(value); resetApproval(); }}
                className={`rounded-base border px-3 py-2 text-sm ${mode === value ? "border-primary bg-primary/10 text-primary" : "border-border"}`}
              >
                {value === "open" ? "Explore freely" : "Investigate a goal"}
              </button>
            ))}
          </div>
        </fieldset>
        {mode === "goal_directed" && (
          <label className="flex flex-col gap-1 text-sm font-medium">
            Goal
            <textarea
              value={goal}
              onChange={(event) => { setGoal(event.target.value); resetApproval(); }}
              rows={3}
              maxLength={4000}
              className="rounded-base border border-border bg-bg px-3 py-2 font-normal"
              placeholder="What should the exploration investigate?"
            />
          </label>
        )}
        <fieldset className="flex flex-col gap-2">
          <legend className="text-sm font-medium">Thinking level</legend>
          <div className="flex flex-wrap gap-2">
            {(["quick", "standard", "deep"] as const).map((value) => (
              <button
                key={value}
                type="button"
                aria-pressed={tier === value}
                onClick={() => { setTier(value); resetApproval(); }}
                className={`rounded-base border px-3 py-2 text-sm capitalize ${tier === value ? "border-primary bg-primary/10 text-primary" : "border-border"}`}
              >
                {value}
              </button>
            ))}
          </div>
        </fieldset>
        <fieldset className="flex flex-col gap-2">
          <legend className="text-sm font-medium">Datasets in scope</legend>
          {(datasets.data ?? []).map((dataset) => (
            <label key={dataset.dataset_id} className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={selected.includes(dataset.dataset_id)}
                onChange={(event) => {
                  setSelected((current) => event.target.checked
                    ? [...current, dataset.dataset_id]
                    : current.filter((id) => id !== dataset.dataset_id));
                  resetApproval();
                }}
              />
              {dataset.display_name}
            </label>
          ))}
        </fieldset>
        {!prepare.data && (
          <button
            type="button"
            onClick={submitPrepare}
            disabled={prepare.isPending || selected.length === 0 || (mode === "goal_directed" && !goal.trim())}
            className="self-start rounded-base bg-primary px-4 py-2 text-sm font-medium text-bg disabled:opacity-50"
          >
            {prepare.isPending ? "Preparing…" : "Review authorization"}
          </button>
        )}
        {prepare.error && <p role="alert" className="text-sm text-status-critical">{formatUnknownError(prepare.error)}</p>}
      </Card>

      {prepare.data && (
        <Card as="section" aria-label="One-time read-only authorization" className="flex flex-col gap-4 border-primary/30 p-4">
          <SectionHeader
            title="One-time read-only authorization"
            description="Confirm once to consume this short-lived approval. A retry reuses the same idempotency key."
            actions={<Badge tone="ok">Read-only</Badge>}
          />
          <p className="text-sm text-status-neutral">
            The sealed capability digest allows analytical reads only. Source datasets and the source session are not modified.
          </p>
          <HardCaps prepared={prepare.data} />
          <div className="text-xs text-status-neutral">
            <p>Policy <code>{prepare.data.policy.policy_fingerprint}</code></p>
            <p>Data witness <code>{prepare.data.data_state_witness}</code></p>
            <p>Approval expires {new Date(prepare.data.expires_at).toLocaleString()}</p>
          </div>
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={authorize}
              disabled={start.isPending}
              className="rounded-base bg-primary px-4 py-2 text-sm font-medium text-bg disabled:opacity-50"
            >
              {start.isPending ? "Starting…" : "Authorize and start"}
            </button>
            <button type="button" onClick={resetApproval} className="rounded-base border border-border px-4 py-2 text-sm">
              Change setup
            </button>
          </div>
          {start.error && <p role="alert" className="text-sm text-status-critical">{formatUnknownError(start.error)}</p>}
        </Card>
      )}
    </main>
  );
}

function BudgetExtension({ sessionId, explorationId }: { sessionId: string; explorationId: string }) {
  const extend = useExtendExplorationBudget(sessionId, explorationId);
  const [requests, setRequests] = useState(0);
  const [tools, setTools] = useState(0);
  const [rounds, setRounds] = useState(0);
  const [cost, setCost] = useState(0);
  const [reason, setReason] = useState("");
  const idempotencyKey = useRef<string | null>(null);
  const edit = <T,>(setter: (value: T) => void, value: T) => {
    idempotencyKey.current = null;
    setter(value);
  };
  const increase: ExplorationBudgetIncrease = {
    max_requests: requests,
    max_successful_tool_calls: tools,
    max_rounds: rounds,
    max_cost_usd: cost,
  };
  const valid = requests + tools + rounds + cost > 0 && Boolean(reason.trim());
  return (
    <details className="rounded-base border border-border p-3">
      <summary className="cursor-pointer text-sm font-medium">Extend hard caps</summary>
      <form
        className="mt-3 grid gap-3 sm:grid-cols-2"
        onSubmit={(event) => {
          event.preventDefault();
          if (!valid) return;
          idempotencyKey.current ??= crypto.randomUUID();
          extend.mutate(
            { increase, reason: reason.trim(), idempotencyKey: idempotencyKey.current },
            { onSuccess: () => { idempotencyKey.current = null; } },
          );
        }}
      >
        {[
          ["Model requests", requests, setRequests],
          ["Tool calls", tools, setTools],
          ["Rounds", rounds, setRounds],
          ["Cost USD", cost, setCost],
        ].map(([label, value, setter]) => (
          <label key={String(label)} className="flex flex-col gap-1 text-xs">
            {String(label)} increase
            <input
              type="number"
              min="0"
              step={label === "Cost USD" ? "0.01" : "1"}
              value={Number(value)}
              onChange={(event) => edit(
                setter as (value: number) => void,
                Number(event.target.value),
              )}
              className="rounded-base border border-border bg-bg px-2 py-1.5 text-sm"
            />
          </label>
        ))}
        <label className="flex flex-col gap-1 text-xs sm:col-span-2">
          Reason
          <input value={reason} onChange={(event) => edit(setReason, event.target.value)} maxLength={2000} className="rounded-base border border-border bg-bg px-2 py-1.5 text-sm" />
        </label>
        <button type="submit" disabled={!valid || extend.isPending} className="self-start rounded-base border border-primary px-3 py-1.5 text-sm text-primary disabled:opacity-50">
          {extend.isPending ? "Extending…" : "Approve additive increase"}
        </button>
        {extend.error && <p role="alert" className="text-sm text-status-critical">{formatUnknownError(extend.error)}</p>}
      </form>
    </details>
  );
}

/** The report is a file in the run directory, so it is fetched and shown here
 *  rather than linked as an artifact that never existed. */
function FinalReport({ sessionId, explorationId }: { sessionId: string; explorationId: string }) {
  const report = useExplorationReport(sessionId, explorationId);
  return (
    <Card as="section" aria-label="Final exploration report" className="flex flex-col gap-2 p-4">
      <SectionHeader
        level={3}
        title="Deterministic final report"
        description="Rendered by the run itself from its own journal, not re-summarized."
      />
      {report.isLoading && <LoadingSkeleton label="Loading the final report" lines={4} />}
      {report.error && (
        <ErrorState error={report.error} onRetry={() => void report.refetch()} />
      )}
      {report.data !== undefined && (
        <pre className="max-h-[32rem] overflow-auto rounded-base bg-code-bg p-3 font-mono text-xs whitespace-pre-wrap text-code-text">
          {report.data}
        </pre>
      )}
    </Card>
  );
}

function ExplorationRun({ sessionId, explorationId }: { sessionId: string; explorationId: string }) {
  const queryClient = useQueryClient();
  const exploration = useExploration(sessionId, explorationId);
  const pause = usePauseExploration(sessionId, explorationId);
  const resume = useResumeExploration(sessionId, explorationId);
  const cancel = useCancelExploration(sessionId, explorationId);
  const [lastEvent, setLastEvent] = useState<ExplorationEventDto | null>(null);
  const [confirmCancel, setConfirmCancel] = useState(false);
  const resumeKey = useRef<string | null>(null);
  const onEvent = useCallback((event: ExplorationEventDto) => {
    setLastEvent(event);
    void queryClient.invalidateQueries({ queryKey: queryKeys.exploration(sessionId, explorationId) });
  }, [explorationId, queryClient, sessionId]);
  const stream = useExplorationEvents({
    explorationId,
    eventsUrl: exploration.data?.events_url ?? "",
    initialLastSeq: exploration.data?.last_seq ?? -1,
    enabled: Boolean(exploration.data && exploration.data.status !== "stopped"),
    onEvent,
  });
  const run = useMemo(
    () => exploration.data ? explorationRunFromDto(exploration.data) : null,
    [exploration.data],
  );

  if (exploration.isLoading || !run) return exploration.error
    ? <ErrorState error={exploration.error} onRetry={() => void exploration.refetch()} />
    : <LoadingSkeleton label="Loading exploration" lines={6} />;

  const controlError = pause.error ?? resume.error ?? cancel.error;
  return (
    <main className="mx-auto flex w-full max-w-6xl flex-col gap-5 p-4 sm:p-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <SectionHeader title="Exploration" description="Live journal-backed status and evidence projection." />
        <div className="flex flex-wrap items-center gap-2">
          <Badge tone={stream.phase === "disconnected" ? "warn" : "neutral"}>SSE {stream.phase}</Badge>
          {lastEvent && <code className="text-xs text-status-neutral">{lastEvent.type} · {lastEvent.event_id}</code>}
        </div>
      </div>
      <div className="flex flex-wrap gap-2">
        {run.status === "running" && <button type="button" onClick={() => pause.mutate()} disabled={pause.isPending} className="rounded-base border border-border px-3 py-1.5 text-sm">Pause</button>}
        {run.status === "pause_requested" && <button type="button" disabled className="rounded-base border border-border px-3 py-1.5 text-sm opacity-50">Pause requested…</button>}
        {run.status === "paused" && <button type="button" onClick={() => { resumeKey.current ??= crypto.randomUUID(); resume.mutate(resumeKey.current, { onSuccess: () => { resumeKey.current = null; } }); }} disabled={resume.isPending} className="rounded-base bg-primary px-3 py-1.5 text-sm font-medium text-bg">Resume</button>}
        {run.status !== "stopped" && !confirmCancel && <button type="button" onClick={() => setConfirmCancel(true)} className="rounded-base border border-status-critical/50 px-3 py-1.5 text-sm text-status-critical">Cancel…</button>}
        {run.status !== "stopped" && confirmCancel && (
          <>
            <button type="button" onClick={() => cancel.mutate()} disabled={cancel.isPending} className="rounded-base bg-status-critical px-3 py-1.5 text-sm font-medium text-white">Confirm terminal cancel</button>
            <button type="button" onClick={() => setConfirmCancel(false)} className="rounded-base border border-border px-3 py-1.5 text-sm">Keep running</button>
          </>
        )}
      </div>
      {controlError && <p role="alert" className="text-sm text-status-critical">{formatUnknownError(controlError)}</p>}
      <ExplorationRunPanel run={run} />
      {run.status !== "stopped" && <BudgetExtension sessionId={sessionId} explorationId={explorationId} />}
      {run.report?.available && (
        <FinalReport sessionId={sessionId} explorationId={explorationId} />
      )}
      <ExplorationReport run={run} />
    </main>
  );
}

export function Component() {
  const { projectId = "", sessionId = "", explorationId } = useParams();
  return explorationId
    ? <ExplorationRun sessionId={sessionId} explorationId={explorationId} />
    : <ExplorationLaunch projectId={projectId} sessionId={sessionId} />;
}
