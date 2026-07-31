/* SSE stream for one chat turn (§7.5). Same shape as job-events: EventSource
 * with `id:` frames so a reconnect replays from Last-Event-ID, and one listener
 * per named event type.
 *
 * The kernel's chat driver is synchronous and has no token callback, so frames
 * are turn-level, not token-level: lifecycle, the driver's own trace events
 * (intent/plan/SQL/validation), and one final message. */

import { useEffect, useReducer } from "react";

export interface ChatStreamEvent {
  seq: number;
  session_id: string;
  message_id: string;
  type: string;
  data: Record<string, unknown>;
}

export const CHAT_EVENT_TYPES = [
  "turn.started",
  "progress",
  "tool.call",
  "plan.pending",
  "message.completed",
  "turn.failed",
] as const;

const TERMINAL_TYPES = new Set<string>([
  "plan.pending",
  "message.completed",
  "turn.failed",
]);

export interface ToolCall {
  seq: number;
  traceType: string;
  name: string;
  summary: Record<string, unknown>;
}

export interface PendingPlan {
  planId: string;
  actionHash: string;
  approvalToken: string;
  question: string;
  method: string;
  sql: string;
  datasetNames: string[];
  estimatedScan: string;
}

export interface CompletedMessage {
  content: string;
  status: string;
  sql: string | null;
  artifactRefs: string[];
  validation: {
    status: string;
    findings: string[];
  } | null;
}

export type ChatPhase =
  | "idle"
  | "connecting"
  | "running"
  | "completed"
  | "awaiting_approval"
  | "failed"
  | "disconnected";

export interface ChatTurnState {
  phase: ChatPhase;
  stage: string | null;
  toolCalls: ToolCall[];
  completed: CompletedMessage | null;
  pendingPlan: PendingPlan | null;
  error: string | null;
}

export const initialChatTurnState: ChatTurnState = {
  phase: "idle",
  stage: null,
  toolCalls: [],
  completed: null,
  pendingPlan: null,
  error: null,
};

const MAX_TOOL_CALLS = 100;

type Action =
  | { kind: "reset" }
  | { kind: "connecting" }
  | { kind: "event"; event: ChatStreamEvent }
  | { kind: "disconnected" };

function str(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

function strList(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((v): v is string => typeof v === "string") : [];
}

function validation(
  value: unknown,
): CompletedMessage["validation"] {
  if (typeof value !== "object" || value === null) return null;
  const record = value as Record<string, unknown>;
  return {
    status: str(record["status"], "unknown"),
    findings: strList(record["findings"]),
  };
}

function reduce(state: ChatTurnState, action: Action): ChatTurnState {
  if (action.kind === "reset") return initialChatTurnState;
  if (action.kind === "connecting") {
    return { ...initialChatTurnState, phase: "connecting" };
  }
  if (action.kind === "disconnected") {
    return state.phase === "completed" ||
      state.phase === "failed" ||
      state.phase === "awaiting_approval"
      ? state
      : { ...state, phase: "disconnected" };
  }

  const { data } = action.event;
  switch (action.event.type) {
    case "turn.started":
      return { ...state, phase: "running", stage: str(data["stage"], "starting") };
    case "progress":
      return { ...state, phase: "running", stage: str(data["stage"], null as never) || null };
    case "tool.call":
      return {
        ...state,
        phase: "running",
        toolCalls: [
          ...state.toolCalls,
          {
            seq: action.event.seq,
            traceType: str(data["trace_type"], "unknown"),
            name: str(data["name"]),
            summary:
              typeof data["summary"] === "object" && data["summary"] !== null
                ? (data["summary"] as Record<string, unknown>)
                : {},
          },
        ].slice(-MAX_TOOL_CALLS),
      };
    case "plan.pending":
      return {
        ...state,
        phase: "awaiting_approval",
        stage: null,
        pendingPlan: {
          planId: str(data["plan_id"]),
          actionHash: str(data["action_hash"]),
          approvalToken: str(data["approval_token"]),
          question: str(data["question"]),
          method: str(data["method"]),
          sql: str(data["sql"]),
          datasetNames: strList(data["dataset_names"]),
          estimatedScan: str(data["estimated_scan"], "unknown"),
        },
      };
    case "message.completed":
      return {
        ...state,
        phase: "completed",
        stage: null,
        completed: {
          content: str(data["content"]),
          status: str(data["status"], "answer"),
          sql: typeof data["sql"] === "string" ? data["sql"] : null,
          artifactRefs: strList(data["artifact_refs"]),
          validation: validation(data["validation"]),
        },
      };
    case "turn.failed":
      return {
        ...state,
        phase: "failed",
        stage: null,
        error: str(data["message"], "The chat turn failed."),
      };
    default:
      return state;
  }
}

function parseEvent(raw: string): ChatStreamEvent | null {
  try {
    const data = JSON.parse(raw) as Partial<ChatStreamEvent>;
    if (typeof data !== "object" || data === null) return null;
    return {
      seq: typeof data.seq === "number" ? data.seq : 0,
      session_id: String(data.session_id ?? ""),
      message_id: String(data.message_id ?? ""),
      type: String(data.type ?? "unknown"),
      data:
        typeof data.data === "object" && data.data !== null
          ? (data.data as Record<string, unknown>)
          : {},
    };
  } catch {
    return null;
  }
}

/* Streams one turn. Pass null to idle. The terminal frame ends the server
 * stream, so the source is closed to stop the browser auto-reconnecting. */
export function useChatStream(
  messageId: string | null,
  streamUrl: string | null,
): ChatTurnState {
  const [state, dispatch] = useReducer(reduce, initialChatTurnState);

  useEffect(() => {
    if (!messageId || !streamUrl) {
      dispatch({ kind: "reset" });
      return;
    }
    dispatch({ kind: "connecting" });

    const source = new EventSource(streamUrl);
    let terminal = false;

    const onEvent = (message: MessageEvent) => {
      const event = parseEvent(String(message.data));
      if (!event) return;
      dispatch({ kind: "event", event });
      if (TERMINAL_TYPES.has(event.type)) {
        terminal = true;
        source.close();
      }
    };

    for (const type of CHAT_EVENT_TYPES) {
      source.addEventListener(type, onEvent);
    }
    source.onerror = () => {
      if (!terminal && source.readyState === EventSource.CLOSED) {
        dispatch({ kind: "disconnected" });
      }
    };

    return () => source.close();
  }, [messageId, streamUrl]);

  return state;
}
