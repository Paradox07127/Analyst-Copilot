/* Journal-backed exploration SSE. The server accepts Last-Event-ID both as an
 * SSE header and a query parameter; EventSource cannot set headers, so every
 * explicit connection includes the cursor in `last_event_id`. Native retries
 * then continue to use the frame ids as usual. */

import { useEffect, useRef, useState } from "react";
import type { ExplorationEventDto } from "./exploration-types";

export const EXPLORATION_EVENT_TYPES = [
  "exploration_started",
  "attempt_started",
  "round_started",
  "llm_call_started",
  "llm_call_completed",
  "llm_call_rejected",
  "llm_call_uncertain",
  "tool_call_started",
  "receipt_prepared",
  "receipt_committed",
  "tool_call_failed",
  "gate_verdict",
  "reduction_committed",
  "round_settled",
  "pause_requested",
  "paused",
  "resumed",
  "budget_amended",
  "exploration_stopped",
] as const;

export type ExplorationStreamPhase =
  | "idle"
  | "connecting"
  | "live"
  | "disconnected"
  | "stopped";

export function explorationEventsUrl(
  eventsUrl: string,
  lastEventId: string | null,
): string {
  if (!lastEventId) return eventsUrl;
  const url = new URL(eventsUrl, window.location.origin);
  url.searchParams.set("last_event_id", lastEventId);
  return eventsUrl.startsWith("http")
    ? url.toString()
    : `${url.pathname}${url.search}`;
}

function parseEvent(raw: string, explorationId: string): ExplorationEventDto | null {
  try {
    const data = JSON.parse(raw) as Partial<ExplorationEventDto>;
    if (
      data.exploration_id !== explorationId ||
      typeof data.seq !== "number" ||
      data.event_id !== `${explorationId}:${data.seq}` ||
      typeof data.type !== "string"
    ) {
      return null;
    }
    return {
      event_id: data.event_id,
      exploration_id: data.exploration_id,
      seq: data.seq,
      type: data.type,
      occurred_at:
        typeof data.occurred_at === "string" ? data.occurred_at : "",
      data:
        typeof data.data === "object" && data.data !== null
          ? (data.data as Record<string, unknown>)
          : {},
    };
  } catch {
    return null;
  }
}

export function useExplorationEvents({
  explorationId,
  eventsUrl,
  initialLastSeq,
  enabled,
  onEvent,
}: {
  explorationId: string;
  eventsUrl: string;
  initialLastSeq: number;
  enabled: boolean;
  onEvent: (event: ExplorationEventDto) => void;
}): { phase: ExplorationStreamPhase; lastEventId: string | null } {
  const initialId =
    initialLastSeq >= 0 ? `${explorationId}:${initialLastSeq}` : null;
  const cursorRef = useRef<string | null>(initialId);
  const onEventRef = useRef(onEvent);
  const [phase, setPhase] = useState<ExplorationStreamPhase>(
    enabled ? "connecting" : "idle",
  );
  const [lastEventId, setLastEventId] = useState<string | null>(initialId);

  onEventRef.current = onEvent;

  useEffect(() => {
    cursorRef.current = initialLastSeq >= 0
      ? `${explorationId}:${initialLastSeq}`
      : null;
    setLastEventId(cursorRef.current);
  }, [explorationId, initialLastSeq]);

  useEffect(() => {
    if (!enabled || !explorationId || !eventsUrl) {
      setPhase("idle");
      return;
    }
    setPhase("connecting");
    const source = new EventSource(
      explorationEventsUrl(eventsUrl, cursorRef.current),
    );
    let terminal = false;

    source.onopen = () => setPhase("live");
    const receive = (message: MessageEvent) => {
      const event = parseEvent(String(message.data), explorationId);
      if (!event) return;
      cursorRef.current = event.event_id;
      setLastEventId(event.event_id);
      onEventRef.current(event);
      if (event.type === "exploration_stopped") {
        terminal = true;
        setPhase("stopped");
        source.close();
      } else {
        setPhase("live");
      }
    };
    for (const type of EXPLORATION_EVENT_TYPES) {
      source.addEventListener(type, receive);
    }
    source.onerror = () => {
      if (!terminal && source.readyState === EventSource.CLOSED) {
        setPhase("disconnected");
      }
    };
    return () => source.close();
  }, [enabled, eventsUrl, explorationId]);

  return { phase, lastEventId };
}
