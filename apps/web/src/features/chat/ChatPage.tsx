/* Chat slice (§10.3): the persisted transcript (newest page first, older pages
 * on demand), an input that accepts a turn (202) and follows its SSE stream,
 * live tool/SQL cards from the driver's own trace, and the plan-approval card —
 * a plan that needs approval never runs until Approve consumes its token. */

import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { useParams } from "react-router";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  api,
  ApiError,
  type ChatMessageAccepted,
  type ChatMessageView,
  type ChatPendingPlan,
  type ArtifactDetail,
} from "../../api/client";
import {
  useChatStream,
  type PendingPlan,
  type ToolCall,
} from "../../api/chat-events";
import {
  queryKeys,
  useChatMessages,
  useChatPendingPlans,
  useArtifact,
  useSandboxStatus,
} from "../../api/hooks";
import { ErrorState, LoadingSkeleton } from "../../components/async-states";
import { useDialogFocus } from "../../components/use-dialog-focus";
import { SimpleTable } from "../cleaning/SimpleTable";
import { VegaChart } from "../insights/VegaChart";

const STAGE_LABEL: Record<string, string> = {
  loading_datasets: "Loading this session's data…",
  planning: "Planning the analysis…",
  agent: "Agent is investigating with tools…",
  executing: "Running the approved plan…",
  starting: "Starting…",
};

const TRACE_LABEL: Record<string, string> = {
  agent_intent: "Routed intent",
  agent_plan: "Built plan",
  agent_completed: "Agent completed",
  agent_limit_reached: "Agent safety limit",
  tool_started: "Starting tool",
  tool_completed: "Ran tool",
  tool_failed: "Tool failed",
  validator_result: "Validated result",
  tool_guard_rejected: "Tool guard rejected",
  permission_denied: "Permission denied",
  code_agent_attempt: "Sandbox attempt",
  chat_turn_failed: "Turn failed",
  llm_call: "Model call",
};

const FOLLOW_LATEST_THRESHOLD_PX = 72;

function isNearLatest(element: HTMLDivElement): boolean {
  return (
    element.scrollHeight - element.scrollTop - element.clientHeight <=
    FOLLOW_LATEST_THRESHOLD_PX
  );
}

function recoveredPlan(plan: ChatPendingPlan): PendingPlan {
  return {
    planId: plan.plan_id,
    actionHash: plan.action_hash,
    approvalToken: plan.approval_token,
    question: plan.question ?? "",
    method: plan.method ?? "",
    sql: plan.sql ?? "",
    datasetNames: plan.dataset_names ?? [],
    estimatedScan: plan.estimated_scan ?? "unknown",
  };
}

function summaryText(value: unknown): string | null {
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  return null;
}

function ToolCallCard({ call }: { call: ToolCall }) {
  const sql = summaryText(call.summary["sql"]);
  const rows = summaryText(call.summary["row_count"]);
  const status = summaryText(call.summary["status"]);
  return (
    <li className="rounded-base border border-border bg-surface/60 p-2 text-xs">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-medium">
          {TRACE_LABEL[call.traceType] ?? call.traceType}
        </span>
        <span className="font-mono text-status-neutral">{call.name}</span>
        {status && <span className="text-status-neutral">status: {status}</span>}
        {rows !== null && (
          <span className="text-status-neutral">{rows} rows</span>
        )}
      </div>
      {sql && (
        <pre className="mt-1 overflow-x-auto rounded-base bg-code-bg p-2 font-mono">
          {sql}
        </pre>
      )}
    </li>
  );
}

function PlanApprovalCard({
  plan,
  sessionId,
  pending,
  onApprove,
  onReject,
  error,
}: {
  plan: PendingPlan;
  sessionId: string;
  pending: boolean;
  onApprove: () => void;
  onReject: () => void;
  error: unknown;
}) {
  void sessionId;
  const { dialogRef, onKeyDown } = useDialogFocus(onReject);
  return (
    <div
      ref={dialogRef}
      role="alertdialog"
      aria-label="Approve analysis plan"
      onKeyDown={onKeyDown}
      className="flex flex-col gap-2 rounded-base border border-status-warn/50 p-3 text-sm"
    >
      <p className="font-medium text-status-warn">
        This plan needs your approval before it runs.
      </p>
      <p>{plan.question}</p>
      <p className="text-xs text-status-neutral">
        Method: {plan.method || "n/a"} · Datasets:{" "}
        {plan.datasetNames.join(", ") || "n/a"} · Estimated scan:{" "}
        {plan.estimatedScan}
      </p>
      <pre className="overflow-x-auto rounded-base bg-code-bg p-2 font-mono text-xs">
        {plan.sql}
      </pre>
      {error instanceof Error && (
        <p role="alert" className="text-xs text-status-critical">
          {error.message}
        </p>
      )}
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={onApprove}
          disabled={pending}
          className="rounded-base bg-primary px-3 py-1.5 text-sm font-medium text-bg hover:opacity-90 disabled:opacity-50"
        >
          {pending ? "Working…" : "Approve & run"}
        </button>
        <button
          type="button"
          onClick={onReject}
          disabled={pending}
          className="rounded-base border border-border px-3 py-1.5 text-sm hover:bg-surface"
        >
          Reject
        </button>
      </div>
    </div>
  );
}

interface SqlResultPayload {
  sql?: string;
  columns: string[];
  rows_preview: Record<string, unknown>[];
  row_count: number;
  truncated?: boolean;
}

function sqlResult(payload: Record<string, unknown>): SqlResultPayload | null {
  const columns = payload["columns"];
  const rows = payload["rows_preview"];
  if (
    !Array.isArray(columns) ||
    !columns.every((value): value is string => typeof value === "string") ||
    !Array.isArray(rows) ||
    !rows.every(
      (value): value is Record<string, unknown> =>
        typeof value === "object" && value !== null && !Array.isArray(value),
    )
  ) {
    return null;
  }
  return {
    sql: typeof payload["sql"] === "string" ? payload["sql"] : undefined,
    columns,
    rows_preview: rows,
    row_count:
      typeof payload["row_count"] === "number"
        ? payload["row_count"]
        : rows.length,
    truncated: payload["truncated"] === true,
  };
}

function inferredChart(
  result: SqlResultPayload,
  title: string,
): Record<string, unknown> | null {
  if (
    result.columns.length !== 2 ||
    result.rows_preview.length < 2 ||
    result.rows_preview.length > 30
  ) {
    return null;
  }
  const [x, y] = result.columns;
  const values = result.rows_preview.map((row) => row[y!]);
  if (
    values.some(
      (value) =>
        typeof value !== "number" || typeof value === "boolean",
    )
  ) {
    return null;
  }
  const dates = result.rows_preview.every((row) => {
    const value = row[x!];
    return (
      typeof value === "string" &&
      /^\d{4}([-\/]\d{2})([-\/]\d{2})?(?:[T ]\d{2}:\d{2}:\d{2})?$/.test(
        value.trim(),
      )
    );
  });
  return {
    title,
    mark: dates ? "line" : "bar",
    data: { values: result.rows_preview },
    encoding: {
      x: {
        field: x,
        type: dates ? "temporal" : "nominal",
        ...(dates ? {} : { sort: "-y" }),
      },
      y: { field: y, type: "quantitative" },
    },
  };
}

function ArtifactResult({
  sessionId,
  artifactId,
  answer,
}: {
  sessionId: string;
  artifactId: string;
  answer: string;
}) {
  const detail = useArtifact(sessionId, artifactId, true);
  if (detail.isPending) {
    return <LoadingSkeleton lines={2} label={`Loading result ${artifactId}`} />;
  }
  if (detail.isError) {
    return (
      <p role="alert" className="text-xs text-status-warn">
        Result {artifactId} could not be loaded.
      </p>
    );
  }
  const artifact = detail.data as ArtifactDetail | undefined;
  if (!artifact) return null;
  const result = artifact.type === "SqlResult" ? sqlResult(artifact.payload) : null;
  const artifactSpec =
    artifact.type === "ChartSpec" || artifact.type === "RawChartSpec"
      ? artifact.payload
      : null;
  const quickSpec = result ? inferredChart(result, answer) : null;

  return (
    <section
      aria-label={`Result ${artifactId}`}
      className="flex min-w-0 flex-col gap-2 rounded-base border border-border bg-surface/40 p-2"
    >
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-status-neutral">
        <span>{artifact.type}</span>
        <span className="font-mono">{artifactId}</span>
      </div>
      {result && result.rows_preview.length > 0 && (
        <>
          <SimpleTable
            ariaLabel={`SQL result ${artifactId}`}
            columns={result.columns.map((column) => ({
              key: column,
              label: column,
            }))}
            rows={result.rows_preview}
          />
          <p className="text-xs text-status-neutral">
            {result.row_count.toLocaleString()} rows
            {result.truncated ? " · preview truncated" : ""}
          </p>
        </>
      )}
      {artifactSpec && (
        <VegaChart spec={artifactSpec} label={`${answer} chart`} />
      )}
      {quickSpec && (
        <div className="flex min-w-0 flex-col gap-1">
          <p className="text-xs font-medium">Quick visualization</p>
          <p className="text-xs text-status-neutral">
            Automatically inferred from the two-column preview.
          </p>
          <VegaChart spec={quickSpec} label={`${answer} quick visualization`} />
        </div>
      )}
      {(artifact.warnings ?? []).length > 0 && (
        <ul className="list-inside list-disc text-xs text-status-warn">
          {(artifact.warnings ?? []).map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      )}
    </section>
  );
}

function MessageRow({
  message,
  sessionId,
  validation,
}: {
  message: ChatMessageView;
  sessionId: string;
  validation?: { status: string; findings: string[] } | null;
}) {
  const isUser = message.role === "user";
  const tone =
    message.status === "error"
      ? "border-status-critical/40"
      : message.status === "refused" || message.status === "awaiting_approval"
        ? "border-status-warn/40"
        : "border-border";
  return (
    <li
      className={`flex flex-col gap-1 rounded-base border ${tone} p-3 text-sm ${
        isUser ? "ml-10 bg-surface" : ""
      }`}
    >
      <span className="text-[10px] font-medium uppercase text-status-neutral">
        {message.role}
      </span>
      {/* The page shell is now the same width as every other page, so the
        * measure has to live here instead. Report prose is already held at
        * 72ch by report-markdown.css; this is the only other running text. */}
      <p className="max-w-content whitespace-pre-wrap">{message.content}</p>
      {message.sql && (
        <pre className="overflow-x-auto rounded-base bg-code-bg p-2 font-mono text-xs">
          {message.sql}
        </pre>
      )}
      {(message.artifact_refs ?? []).length > 0 && (
        <div className="flex min-w-0 flex-col gap-2">
          {(message.artifact_refs ?? []).map((artifactId) => (
            <ArtifactResult
              key={artifactId}
              sessionId={sessionId}
              artifactId={artifactId}
              answer={message.content}
            />
          ))}
        </div>
      )}
      {validation && (
        <div
          className={`rounded-base border p-2 text-xs ${
            validation.status === "pass"
              ? "border-status-ok/40 text-status-ok"
              : "border-status-warn/50 text-status-warn"
          }`}
        >
          <p className="font-medium">Validation: {validation.status}</p>
          {validation.findings.length > 0 && (
            <ul className="mt-1 list-inside list-disc">
              {validation.findings.map((finding) => (
                <li key={finding}>{finding}</li>
              ))}
            </ul>
          )}
        </div>
      )}
    </li>
  );
}

export function Component() {
  const { sessionId = "" } = useParams();
  const queryClient = useQueryClient();
  const transcript = useChatMessages(sessionId);
  const [turn, setTurn] = useState<ChatMessageAccepted | null>(null);
  const [draft, setDraft] = useState("");
  const [llmMode, setLlmMode] = useState<"env" | "offline">("env");
  const sandbox = useSandboxStatus();
  const stream = useChatStream(turn?.message_id ?? null, turn?.stream_url ?? null);
  const transcriptRef = useRef<HTMLDivElement>(null);
  const followLatestRef = useRef(true);
  const olderPagePositionRef = useRef<{
    scrollHeight: number;
    scrollTop: number;
  } | null>(null);
  const [showJumpToLatest, setShowJumpToLatest] = useState(false);

  const refreshTranscript = useCallback(() => {
    void queryClient.invalidateQueries({
      queryKey: queryKeys.chatMessages(sessionId),
    });
  }, [queryClient, sessionId]);

  /* The transcript is the durable record; refresh it once the turn settles. */
  useEffect(() => {
    if (stream.phase === "completed" || stream.phase === "failed") {
      refreshTranscript();
    }
  }, [stream.phase, refreshTranscript]);

  const scrollToLatest = useCallback((behavior: ScrollBehavior = "smooth") => {
    const viewport = transcriptRef.current;
    if (!viewport) return;
    followLatestRef.current = true;
    setShowJumpToLatest(false);
    if (typeof viewport.scrollTo === "function") {
      viewport.scrollTo({ top: viewport.scrollHeight, behavior });
    } else {
      /* jsdom and older embedded webviews do not implement scrollTo. */
      viewport.scrollTop = viewport.scrollHeight;
    }
  }, []);

  const handleTranscriptScroll = useCallback(() => {
    const viewport = transcriptRef.current;
    if (!viewport) return;
    const nearLatest = isNearLatest(viewport);
    followLatestRef.current = nearLatest;
    setShowJumpToLatest(!nearLatest);
  }, []);

  const send = useMutation({
    mutationFn: (text: string) =>
      api.sendChatMessage(sessionId, { text, llm: llmMode }),
    onSuccess: (accepted) => {
      /* Sending is an explicit return to the live edge of the conversation. */
      followLatestRef.current = true;
      setShowJumpToLatest(false);
      setTurn(accepted);
      setDraft("");
      refreshTranscript();
    },
  });

  const decide = useMutation({
    mutationFn: ({ plan, approve }: { plan: PendingPlan; approve: boolean }) => {
      const body = {
        action_hash: plan.actionHash,
        approval_token: plan.approvalToken,
      };
      return approve
        ? api.approveChatPlan(sessionId, plan.planId, body)
        : api
            .rejectChatPlan(sessionId, plan.planId, body)
            .then(() => null as ChatMessageAccepted | null);
    },
    onSuccess: (accepted) => {
      setTurn(accepted ?? null);
      refreshTranscript();
      /* The token this card carried is spent either way; a cached copy must
       * not come back if another plan later needs approving. */
      void queryClient.invalidateQueries({
        queryKey: queryKeys.chatPendingPlans(sessionId),
      });
    },
  });

  /* Pages walk backwards from the newest, so render them in reverse order. */
  const messages = [...(transcript.data?.pages ?? [])]
    .reverse()
    .flatMap((page) => page.messages ?? []);
  const completedAlreadyPersisted =
    stream.completed !== null &&
    messages.some(
      (message) =>
        message.role === "assistant" &&
        message.content === stream.completed?.content &&
        message.sql === stream.completed?.sql,
    );

  /* A plan the stream left unapproved (reload, API restart, evicted session)
   * is only reachable through the recovery endpoint — the approval token was
   * never persisted. */
  const stranded =
    turn === null && messages.at(-1)?.status === "awaiting_approval";
  const pendingPlans = useChatPendingPlans(sessionId, stranded);
  const approvalPlan =
    stream.phase === "awaiting_approval" && stream.pendingPlan
      ? stream.pendingPlan
      : stranded && pendingPlans.data?.plans?.[0]
        ? recoveredPlan(pendingPlans.data.plans[0])
        : null;

  useLayoutEffect(() => {
    const viewport = transcriptRef.current;
    if (!viewport) return;

    const olderPosition = olderPagePositionRef.current;
    if (olderPosition) {
      /* Older pages are prepended. Keep the same message under the user's
       * eyes instead of jumping by the height of the inserted content. */
      viewport.scrollTop =
        viewport.scrollHeight -
        olderPosition.scrollHeight +
        olderPosition.scrollTop;
      olderPagePositionRef.current = null;
      return;
    }

    if (followLatestRef.current) {
      scrollToLatest("auto");
    }
  }, [
    transcript.data,
    stream.phase,
    stream.stage,
    stream.toolCalls.length,
    stream.completed?.content,
    approvalPlan?.planId,
    scrollToLatest,
  ]);

  if (transcript.isPending) {
    return <LoadingSkeleton lines={4} label="Loading chat" />;
  }
  if (transcript.isError) {
    return (
      <div className="p-6">
        <ErrorState
          error={transcript.error}
          onRetry={() => transcript.refetch()}
        />
      </div>
    );
  }

  const busy = stream.phase === "connecting" || stream.phase === "running";

  return (
    <div className="mx-auto flex h-full min-h-0 w-[95%] max-w-data flex-col gap-3 overflow-hidden p-3 sm:p-6">
      <header className="flex shrink-0 flex-col gap-1">
        <h1 className="text-xl font-semibold">Chat</h1>
        <p className="text-sm text-status-neutral">
          Ask about this session. The agent can inspect evidence, run bounded
          read-only queries, and reuse saved skills; state-changing work still
          requires approval.
        </p>
      </header>

      <div className="relative min-h-0 flex-1">
        <div
          ref={transcriptRef}
          role="region"
          aria-label="Conversation"
          onScroll={handleTranscriptScroll}
          className="h-full min-h-0 overflow-y-auto overscroll-contain pr-1"
        >
          <div className="flex min-h-full flex-col gap-3 pb-2">
            {transcript.hasNextPage && (
              <button
                type="button"
                onClick={() => {
                  const viewport = transcriptRef.current;
                  if (viewport) {
                    olderPagePositionRef.current = {
                      scrollHeight: viewport.scrollHeight,
                      scrollTop: viewport.scrollTop,
                    };
                  }
                  followLatestRef.current = false;
                  setShowJumpToLatest(true);
                  void transcript.fetchNextPage();
                }}
                disabled={transcript.isFetchingNextPage}
                className="self-center rounded-base border border-border px-2 py-1 text-xs hover:bg-surface disabled:opacity-50"
              >
                {transcript.isFetchingNextPage
                  ? "Loading…"
                  : "Load older messages"}
              </button>
            )}

            {messages.length === 0 && (
              <p className="my-auto text-center text-sm text-status-neutral">
                No messages yet. Ask a question below.
              </p>
            )}
            <ul className="flex flex-col gap-3">
              {messages.map((message) => (
                <MessageRow
                  key={message.seq}
                  message={message}
                  sessionId={sessionId}
                />
              ))}
              {stream.completed && !completedAlreadyPersisted && (
                <MessageRow
                  sessionId={sessionId}
                  validation={stream.completed.validation}
                  message={{
                    seq: -1,
                    role: "assistant",
                    content: stream.completed.content,
                    status: stream.completed.status,
                    sql: stream.completed.sql,
                    artifact_refs: stream.completed.artifactRefs,
                    created_at: null,
                  }}
                />
              )}
            </ul>

            {stream.toolCalls.length > 0 && (
              <section className="flex flex-col gap-1">
                <h2 className="text-xs font-medium uppercase text-status-neutral">
                  Tool activity
                </h2>
                <ul className="flex flex-col gap-1">
                  {stream.toolCalls.map((call) => (
                    <ToolCallCard key={call.seq} call={call} />
                  ))}
                </ul>
              </section>
            )}

            {busy && (
              <p role="status" className="text-sm text-status-neutral">
                {STAGE_LABEL[stream.stage ?? ""] ?? "Working…"}
              </p>
            )}

            {stream.phase === "disconnected" && (
              <p role="alert" className="text-sm text-status-warn">
                Lost the connection to this turn. Reload the page to see the
                recorded result.
              </p>
            )}

            {stream.phase === "failed" && (
              <p role="alert" className="text-sm text-status-critical">
                {stream.error}
              </p>
            )}

            {approvalPlan && (
              <PlanApprovalCard
                plan={approvalPlan}
                sessionId={sessionId}
                pending={decide.isPending}
                error={decide.error}
                onApprove={() =>
                  decide.mutate({ plan: approvalPlan, approve: true })
                }
                onReject={() =>
                  decide.mutate({ plan: approvalPlan, approve: false })
                }
              />
            )}
          </div>
        </div>

        {showJumpToLatest && (
          <button
            type="button"
            onClick={() => scrollToLatest()}
            className="absolute right-3 bottom-3 rounded-base border border-border bg-bg px-3 py-1.5 text-xs font-medium shadow-sm hover:bg-surface"
          >
            Jump to latest
          </button>
        )}
      </div>

      <form
        aria-label="Ask about this session"
        className="flex shrink-0 flex-col gap-2 border-t border-border bg-bg pt-3"
        onSubmit={(event) => {
          event.preventDefault();
          const text = draft.trim();
          if (text && !busy) send.mutate(text);
        }}
      >
        <label className="flex flex-col gap-1 text-sm">
          <span className="sr-only">Message</span>
          <textarea
            aria-label="Message"
            rows={3}
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            placeholder="Ask a data question, e.g. 'which column has the most missing values?'"
            className="rounded-base border border-border bg-surface px-3 py-2 text-sm"
          />
        </label>
        <div className="flex items-center gap-2">
          <button
            type="submit"
            disabled={busy || send.isPending || draft.trim().length === 0}
            className="rounded-base bg-primary px-3 py-1.5 text-sm font-medium text-bg hover:opacity-90 disabled:opacity-50"
          >
            {busy ? "Streaming…" : "Send"}
          </button>
          <label className="flex items-center gap-1 text-xs text-status-neutral">
            LLM mode
            <select
              value={llmMode}
              onChange={(event) =>
                setLlmMode(event.target.value as "env" | "offline")
              }
              className="rounded-base border border-border bg-surface px-1.5 py-1 text-xs"
            >
              <option value="env">env (live model)</option>
              <option value="offline">offline</option>
            </select>
          </label>
        </div>
        {/* Open-ended Python analysis needs a safe sandbox backend; without one
         * the agent refuses those requests rather than running them unprotected.
         * SQL and deterministic answers are unaffected. */}
        {sandbox.data && !sandbox.data.open_python_analysis_available && (
          <p className="text-xs text-status-warn">
            {sandbox.data.message ||
              "Open-ended Python analysis is unavailable: no safe sandbox backend. SQL questions still work."}
          </p>
        )}
        {send.isError && (
          <p role="alert" className="text-sm text-status-critical">
            {send.error instanceof ApiError && send.error.code === "chat_busy"
              ? "A turn is already running for this session. Wait for it to finish."
              : send.error instanceof Error
                ? send.error.message
                : "Failed to send the message."}
          </p>
        )}
      </form>
    </div>
  );
}
