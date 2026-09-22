import { act, renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { useChatStream } from "../api/chat-events";
import { FakeEventSource } from "./fake-event-source";

function event(seq: number, type: string, data: Record<string, unknown> = {}) {
  return { seq, session_id: "r1", message_id: "m1", type, data };
}

describe("durable chat stream", () => {
  it.each([
    ["answer", "completed"], ["error", "failed"],
    ["cancelled", "cancelled"], ["refused", "refused"],
  ])("keeps message status %s distinct from successful completion", (status, phase) => {
    const { result } = renderHook(() => useChatStream("m1", "/stream?message_id=m1"));
    const source = FakeEventSource.latest();
    act(() => source.emit("message.completed", event(1, "message.completed", {
      content: "Recorded outcome", status,
    })));
    expect(result.current.phase).toBe(phase);
    expect(result.current.completed?.status).toBe(status);
    expect(source.readyState).toBe(FakeEventSource.CLOSED);
    act(() => source.failFatally());
    expect(result.current.phase).toBe(phase);
  });

  it("deduplicates replayed frames and rejects stale, invalid, and other-turn frames", () => {
    const { result } = renderHook(() => useChatStream("m1", "/stream?message_id=m1"));
    const source = FakeEventSource.latest();
    act(() => {
      source.emit("turn.started", event(1, "turn.started"));
      source.emit("tool.call", event(2, "tool.call", { name: "inspect" }));
      // A native EventSource reconnect replays on the same connection object.
      source.onerror?.(new Event("error"));
      source.emit("tool.call", event(2, "tool.call", { name: "inspect" }));
      source.emit("progress", event(1, "progress", { stage: "stale" }));
      source.emit("message.completed", { ...event(5, "message.completed"), message_id: "other" });
      source.emit("message.completed", event(-1, "message.completed"));
      source.emit("message.completed", { ...event(9, "message.completed"), seq: 1.5 });
    });
    expect(result.current.toolCalls).toHaveLength(1);
    expect(result.current.phase).toBe("running");
    expect(result.current.stage).not.toBe("stale");
    expect(source.readyState).not.toBe(FakeEventSource.CLOSED);
    act(() => source.emit("message.completed", event(3, "message.completed", {
      content: "Done", status: "answer",
    })));
    act(() => source.emit("progress", event(4, "progress", { stage: "late" })));
    expect(result.current.phase).toBe("completed");
  });

  it("rebuilds from durable history on explicit recovery of the same turn", () => {
    const { result, rerender } = renderHook(
      ({ attempt }) => useChatStream("m1", "/stream?message_id=m1", attempt),
      { initialProps: { attempt: 0 } },
    );
    const old = FakeEventSource.latest();
    act(() => old.emit("tool.call", event(2, "tool.call", { name: "inspect" })));
    act(() => old.failFatally());
    rerender({ attempt: 1 });
    const resumed = FakeEventSource.latest();
    expect(resumed).not.toBe(old);
    act(() => {
      old.emit("tool.call", event(99, "tool.call", { name: "late old connection" }));
      resumed.emit("tool.call", event(2, "tool.call", { name: "inspect" }));
      resumed.emit("tool.call", event(2, "tool.call", { name: "inspect" }));
    });
    expect(result.current.toolCalls).toHaveLength(1);
  });
});
