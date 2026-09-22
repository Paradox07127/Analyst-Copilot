import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { useExplorationEvents } from "../api/exploration-events";
import { FakeEventSource } from "./fake-event-source";

function frame(seq: number, type = "receipt_committed") {
  return { event_id: `x1:${seq}`, exploration_id: "x1", seq, type,
    occurred_at: "2026-09-22T00:00:00Z", data: {} };
}

describe("exploration event cursor", () => {
  it("ignores reconnect duplicates and keeps a newer cursor than an older GET snapshot", () => {
    const onEvent = vi.fn();
    const { result, rerender } = renderHook(({ seq, enabled }) => useExplorationEvents({
      explorationId: "x1", eventsUrl: "/events", initialLastSeq: seq, enabled, onEvent,
    }), { initialProps: { seq: 3, enabled: true } });
    const source = FakeEventSource.latest();
    act(() => source.emit("receipt_committed", frame(5)));
    rerender({ seq: 4, enabled: true });
    act(() => {
      source.emit("receipt_committed", frame(5));
      source.emit("receipt_committed", frame(4));
    });
    expect(onEvent).toHaveBeenCalledTimes(1);
    expect(result.current.lastEventId).toBe("x1:5");
    rerender({ seq: 4, enabled: false });
    rerender({ seq: 4, enabled: true });
    expect(FakeEventSource.latest().url).toBe("/events?last_event_id=x1%3A5");
    act(() => FakeEventSource.latest().emit("branch_abandoned", frame(6, "branch_abandoned")));
    expect(onEvent).toHaveBeenCalledTimes(2);
  });
});
