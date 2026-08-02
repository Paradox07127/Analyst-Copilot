/* Sidebar footer status: a state dot
 * followed by the selected model. The dot uses is_ready_for_live_calls rather
 * than that footer's is_live_provider, so a provider picked without its API key
 * reads as "not ready" instead of green — the footer's looser predicate hides
 * the one failure the dot exists to warn about. */

import { useSettings } from "../../api/hooks";

type LlmState = "ready" | "incomplete" | "offline";

interface LlmStatus {
  state: LlmState;
  label: string;
  title: string;
}

const STATE_LABEL: Record<LlmState, string> = {
  ready: "LLM ready",
  incomplete: "LLM not ready",
  offline: "LLM offline",
};

const STATE_DOT: Record<LlmState, string> = {
  ready: "bg-status-ok",
  incomplete: "bg-status-warn",
  offline: "bg-status-neutral",
};

export function useLlmStatus(): LlmStatus | null {
  const settings = useSettings();
  const data = settings.data;
  if (!data) return null;
  const state: LlmState = data.is_ready_for_live_calls
    ? "ready"
    : data.status_state === "offline"
      ? "offline"
      : "incomplete";
  const missing = data.missing_fields ?? [];
  return {
    state,
    label: data.model || "No model selected",
    /* missing_fields is the only place that names what to go fix. */
    title:
      state === "incomplete" && missing.length > 0
        ? `${data.status_message} Missing: ${missing.join(", ")}.`
        : data.status_message,
  };
}

export function LlmStatusDot({ status }: { status: LlmStatus }) {
  return (
    <span
      role="img"
      aria-label={STATE_LABEL[status.state]}
      title={status.title}
      className={`h-2 w-2 shrink-0 rounded-full ${STATE_DOT[status.state]}`}
    />
  );
}
