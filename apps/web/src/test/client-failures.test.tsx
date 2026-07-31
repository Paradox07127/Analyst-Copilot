import { useEffect } from "react";
import { useMutation } from "@tanstack/react-query";
import { render, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router";
import { afterEach, describe, expect, it, vi } from "vitest";
import { createAppQueryClient, AppProviders } from "../app/providers";
import {
  reportHandledClientFailure,
  resetClientFailureReporterForTests,
} from "../api/client-failures";
import { api, ApiError } from "../api/client";
import { ErrorState } from "../components/async-states";

afterEach(() => {
  vi.restoreAllMocks();
  resetClientFailureReporterForTests();
  window.history.replaceState({}, "", "/");
});

function recorded() {
  return Promise.resolve({ event_type: "failure_recorded" as const, recorded: true });
}

describe("handled client failure reporting", () => {
  it("ErrorState sends only allowlisted metadata and deduplicates the error", async () => {
    const send = vi.spyOn(api, "recordClientFailure").mockImplementation(recorded);
    const error = new ApiError(
      503,
      "provider_secret_failure",
      "customer@example.com token=never-store-this",
    );

    const view = render(
      <MemoryRouter initialEntries={["/projects/demo/sessions/run_safe/report"]}>
        <Routes>
          <Route
            path="/projects/:projectId/sessions/:sessionId/report"
            element={<ErrorState error={error} />}
          />
        </Routes>
      </MemoryRouter>,
    );
    view.rerender(
      <MemoryRouter initialEntries={["/projects/demo/sessions/run_safe/report"]}>
        <Routes>
          <Route
            path="/projects/:projectId/sessions/:sessionId/report"
            element={<ErrorState error={error} />}
          />
        </Routes>
      </MemoryRouter>,
    );

    await waitFor(() => expect(send).toHaveBeenCalledTimes(1));
    const [sessionId, payload] = send.mock.calls[0]!;
    expect(sessionId).toBe("run_safe");
    expect(payload).toEqual({
      error_code: "server_error",
      operation: "render",
      dedupe_key: expect.stringMatching(/^[0-9a-f-]{36}$/),
    });
    expect(JSON.stringify(payload)).not.toContain("customer@example.com");
    expect(JSON.stringify(payload)).not.toContain("never-store-this");
    expect(JSON.stringify(payload)).not.toContain("provider_secret_failure");
  });

  it("the production mutation cache reports a handled mutation rejection", async () => {
    window.history.replaceState(
      {},
      "",
      "/projects/demo/sessions/run_mutation/cleaning",
    );
    const send = vi.spyOn(api, "recordClientFailure").mockImplementation(recorded);
    const error = new ApiError(409, "approval_consumed", "private detail");

    function FailedMutation() {
      const mutation = useMutation({
        mutationFn: async () => {
          throw error;
        },
      });
      useEffect(() => mutation.mutate(), []); // eslint-disable-line react-hooks/exhaustive-deps
      return null;
    }

    render(
      <AppProviders client={createAppQueryClient()}>
        <FailedMutation />
      </AppProviders>,
    );

    await waitFor(() => expect(send).toHaveBeenCalledTimes(1));
    expect(send).toHaveBeenCalledWith(
      "run_mutation",
      expect.objectContaining({
        error_code: "conflict",
        operation: "mutation",
      }),
    );
  });

  it("does nothing outside a validated run route", () => {
    const send = vi.spyOn(api, "recordClientFailure").mockImplementation(recorded);
    window.history.replaceState({}, "", "/settings");
    reportHandledClientFailure(new Error("private"), "render");
    expect(send).not.toHaveBeenCalled();
  });

  it("never lets a synchronous telemetry failure replace the product error", () => {
    window.history.replaceState({}, "", "/projects/demo/sessions/run_safe/report");
    vi.spyOn(api, "recordClientFailure").mockImplementation(() => {
      throw new Error("telemetry unavailable");
    });

    expect(() =>
      reportHandledClientFailure(new Error("visible product error"), "render"),
    ).not.toThrow();
  });
});
