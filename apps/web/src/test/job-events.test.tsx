import { act, renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { explorationJobOutcome, projectExplorationJobState, useJobEvents } from "../api/job-events";
import { FakeEventSource } from "./fake-event-source";

function frame(id: number, type: string, summary: Record<string, unknown>) {
  return { event_id: id, job_id: "j1", session_id: "derived", type, name: "j1", summary };
}

describe("exploration job outcomes", () => {
  it.each([
    ["paused", null, "paused"],
    ["stopped", "failed", "failed"],
    ["stopped", "state_witness_changed", "failed"],
    ["stopped", "cancelled", "cancelled"],
    ["stopped", "budget_exhausted", "limited"],
    ["stopped", "no_new_information", "limited"],
    ["stopped", "completed", "completed"],
  ])("preserves %s/%s when the worker completes", (status, reason, phase) => {
    const { result } = renderHook(() => useJobEvents("j1", "/events"));
    const source = FakeEventSource.latest();
    act(() => {
      source.emit("job.started", frame(1, "job.started", {}));
      source.emit("exploration.attempt_finished", frame(2, "exploration.attempt_finished", {
        exploration_id: "x1", exploration_status: status, stop_reason: reason, journal_seq: 20,
      }));
    });
    expect(result.current.phase).toBe("running");
    act(() => source.emit("job.completed", frame(3, "job.completed", {})));
    expect(result.current.phase).toBe(phase);
    expect(explorationJobOutcome(result.current)).not.toBeNull();
    expect(source.readyState).toBe(FakeEventSource.CLOSED);
  });

  it("does not infer successful exploration from a completed job snapshot alone", () => {
    const { result } = renderHook(() => useJobEvents("j1", "/events"));
    expect(projectExplorationJobState(result.current, "exploration_run", "completed").phase)
      .toBe("outcome_unknown");
    expect(projectExplorationJobState(result.current, "auto_eda", "completed"))
      .toBe(result.current);
    act(() => FakeEventSource.latest().emit("exploration.attempt_finished", frame(4, "exploration.attempt_finished", {
      exploration_status: "paused", stop_reason: null,
    })));
    expect(projectExplorationJobState(result.current, "exploration_run", "completed").phase)
      .toBe("paused");
  });
});

it.each([
  ["paused", null, "paused"],
  ["stopped", "failed", "failed"],
  ["stopped", "budget_exhausted", "limited"],
])("restores persisted domain outcome %s/%s without synthesizing trace events", (status, reason, phase) => {
  const { result } = renderHook(() => useJobEvents("j1", "/events"));
  const restored = projectExplorationJobState(result.current, "exploration_run", "completed", {
    status: status!, stop_reason: reason, exploration_id: "x1",
  });
  expect(restored.phase).toBe(phase);
  expect(restored.events).toHaveLength(0);
  expect(explorationJobOutcome(restored)?.phase).toBe(phase);
});
