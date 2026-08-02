/* The four capabilities that previously existed only in the desktop app:
 * pre-cleaning on the Launchpad, knowledge promotion on Findings, on-demand
 * report generation, and what-if forks on Compare. */

import { describe, expect, it } from "vitest";
import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import type { SessionPage } from "../api/client";
import { server } from "./msw/server";
import { FakeEventSource } from "./fake-event-source";
import { renderAppAt } from "./render";

const csvFile = () =>
  new File(["id,name\n1,a\n"], "orders.csv", { type: "text/csv" });

function jobCreated(jobId: string, sessionId: string) {
  return {
    job_id: jobId,
    session_id: sessionId,
    status: "queued",
    events_url: `/api/v1/jobs/${jobId}/events`,
  };
}

describe("Launchpad pre-cleaning", () => {
  it("is off by default and sends no precleaning block", async () => {
    let body: Record<string, unknown> | null = null;
    server.use(
      http.post("/api/v1/sessions/:sessionId/jobs", async ({ request, params }) => {
        body = (await request.json()) as Record<string, unknown>;
        return HttpResponse.json(
          jobCreated("job_1", String(params["sessionId"])),
          { status: 201 },
        );
      }),
    );

    const user = userEvent.setup();
    renderAppAt("/projects/p1/new-session");
    await screen.findByRole("heading", { name: "New session" });
    await user.upload(screen.getByLabelText("Data files (.csv)"), csvFile());
    await screen.findByRole("checkbox", { name: "Exclude orders.csv" });
    await user.click(screen.getByRole("button", { name: "Run analysis" }));

    await screen.findByRole("heading", { name: "Data Map" });
    expect(body).toMatchObject({ precleaning: null });
  });

  it("sends the configured thresholds once a clean is switched on", async () => {
    let body: Record<string, unknown> | null = null;
    server.use(
      http.post("/api/v1/sessions/:sessionId/jobs", async ({ request, params }) => {
        body = (await request.json()) as Record<string, unknown>;
        return HttpResponse.json(
          jobCreated("job_1", String(params["sessionId"])),
          { status: 201 },
        );
      }),
    );

    const user = userEvent.setup();
    renderAppAt("/projects/p1/new-session");
    await screen.findByRole("heading", { name: "New session" });
    await user.upload(screen.getByLabelText("Data files (.csv)"), csvFile());
    await screen.findByRole("checkbox", { name: "Exclude orders.csv" });

    /* Cleaning is optional tuning and the compact checkbox is its only entry
     * point. Thresholds appear only after it is enabled. */
    await user.click(
      screen.getByRole("checkbox", { name: "Clean data first" }),
    );

    /* The promise that the originals survive is the whole reason this control
     * is safe to expose, so it is asserted, not assumed — and it must be
     * readable without opening anything, which is why this does not click. */
    expect(
      screen.getByText(/Your uploads are never changed/),
    ).toBeInTheDocument();

    await user.click(
      screen.getByRole("checkbox", {
        name: /Drop IQR outlier rows/,
      }),
    );
    await user.click(screen.getByRole("button", { name: "Run analysis" }));

    await screen.findByRole("heading", { name: "Data Map" });
    expect(body).toMatchObject({
      precleaning: {
        clean_missing_values: true,
        missing_threshold_percent: 70,
        min_rows_keep_percent: 50,
        drop_iqr_outliers: true,
      },
    });
  });
});

describe("Findings knowledge promotion", () => {
  it("previews the exact knowledge text, then promotes it in a second step", async () => {
    let promoteBody: Record<string, unknown> | null = null;
    server.use(
      http.post(
        "/api/v1/sessions/:sessionId/findings/:findingId/prepare-promote",
        ({ params }) =>
          HttpResponse.json({
            session_id: String(params["sessionId"]),
            finding_id: String(params["findingId"]),
            action_hash: "a".repeat(64),
            approval_token: "c".repeat(32),
            expires_at: "2099-01-01T00:00:00Z",
            question: "What was average order value?",
            answer: "Average order value was $42.",
            evidence_note: "Source artifacts: sqlres_1.",
            replaces_existing: false,
          }),
      ),
      http.post(
        "/api/v1/sessions/:sessionId/findings/:findingId/promote",
        async ({ request, params }) => {
          promoteBody = (await request.json()) as Record<string, unknown>;
          return HttpResponse.json(
            {
              session_id: String(params["sessionId"]),
              finding_id: String(params["findingId"]),
              question: "What was average order value?",
              answer: "Average order value was $42.",
              verified_answer_count: 3,
            },
            { status: 201 },
          );
        },
      ),
    );

    const user = userEvent.setup();
    renderAppAt("/projects/p1/sessions/r1/findings");
    await screen.findByRole("heading", { name: "Findings" });

    const cards = screen.getAllByRole("listitem");
    const freshCard = cards.find((card) =>
      within(card).queryByText("What was average order value?"),
    )!;
    const staleCard = cards.find((card) =>
      within(card).queryByText("Do regions differ in revenue?"),
    )!;

    /* Only a fresh finding may become knowledge; the stale one is refused in
     * the UI as well as on the server. */
    expect(
      within(staleCard).getByRole("button", {
        name: "Promote to verified answer",
      }),
    ).toBeDisabled();

    await user.click(
      within(freshCard).getByRole("button", {
        name: "Promote to verified answer",
      }),
    );

    /* Step one shows what would be written, before anything is written. */
    expect(
      await within(freshCard).findByText(
        "This will be stored as a verified answer for the whole project.",
      ),
    ).toBeInTheDocument();
    expect(
      within(freshCard).getByText("Source artifacts: sqlres_1."),
    ).toBeInTheDocument();
    expect(promoteBody).toBeNull();

    await user.click(
      within(freshCard).getByRole("button", { name: "Confirm promotion" }),
    );

    expect(
      await within(freshCard).findByText(
        "Promoted. The project now has 3 verified answer(s).",
      ),
    ).toBeInTheDocument();
    expect(promoteBody).toEqual({
      action_hash: "a".repeat(64),
      approval_token: "c".repeat(32),
    });
  });

  it("surfaces a server refusal instead of claiming success", async () => {
    server.use(
      http.post("/api/v1/sessions/:sessionId/findings/:findingId/prepare-promote", () =>
        HttpResponse.json(
          {
            error: {
              code: "promotion_not_allowed",
              message: "Finding finding_1 cannot be promoted: it is stale.",
            },
          },
          { status: 409 },
        ),
      ),
    );

    const user = userEvent.setup();
    renderAppAt("/projects/p1/sessions/r1/findings");
    await screen.findByRole("heading", { name: "Findings" });
    const freshCard = screen
      .getAllByRole("listitem")
      .find((card) =>
        within(card).queryByText("What was average order value?"),
      )!;

    await user.click(
      within(freshCard).getByRole("button", {
        name: "Promote to verified answer",
      }),
    );

    expect(
      await within(freshCard).findByText(
        "Finding finding_1 cannot be promoted: it is stale.",
      ),
    ).toBeInTheDocument();
  });
});

describe("Report generation on demand", () => {
  function noReport() {
    return http.get("/api/v1/sessions/:sessionId/report", ({ params }) =>
      HttpResponse.json({
        session_id: String(params["sessionId"]),
        status: "none",
        markdown: "",
        generated_at: null,
      }),
    );
  }

  it("generates a report for a run that has none and tracks the job", async () => {
    let key: string | null = null;
    server.use(
      noReport(),
      http.post("/api/v1/sessions/:sessionId/report/generate", ({ request, params }) => {
        key = request.headers.get("Idempotency-Key");
        return HttpResponse.json(
          {
            session_id: String(params["sessionId"]),
            execution_session_id: "rpsess_1",
            regenerated: false,
            job: jobCreated("job_report", "rpsess_1"),
          },
          { status: 201 },
        );
      }),
    );

    const user = userEvent.setup();
    renderAppAt("/projects/p1/sessions/r1/report");
    await user.click(
      await screen.findByRole("button", { name: "Generate report" }),
    );

    expect(
      await screen.findByText(/Tracking job_report · session rpsess_1/),
    ).toBeInTheDocument();
    expect(key).toMatch(/^[0-9a-f-]{36}$/);
  });

  it("asks for confirmation before replacing an existing report", async () => {
    let generateCalls = 0;
    server.use(
      http.post("/api/v1/sessions/:sessionId/report/generate", ({ params }) => {
        generateCalls += 1;
        return HttpResponse.json(
          {
            session_id: String(params["sessionId"]),
            execution_session_id: "rpsess_2",
            regenerated: true,
            job: jobCreated("job_regen", "rpsess_2"),
          },
          { status: 201 },
        );
      }),
    );

    const user = userEvent.setup();
    renderAppAt("/projects/p1/sessions/r1/report");
    await user.click(
      await screen.findByRole("button", { name: "Regenerate report" }),
    );
    expect(generateCalls).toBe(0);
    expect(
      screen.getByText(/Regenerating replaces this report/),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Keep current" }));
    expect(generateCalls).toBe(0);

    await user.click(screen.getByRole("button", { name: "Regenerate report" }));
    await user.click(screen.getByRole("button", { name: "Replace report" }));
    expect(
      await screen.findByText(/Tracking job_regen · session rpsess_2/),
    ).toBeInTheDocument();
    expect(generateCalls).toBe(1);
  });
});

describe("Compare what-if fork", () => {
  function twoRuns() {
    return http.get("/api/v1/projects/:projectId/sessions", ({ params }) =>
      HttpResponse.json({
        items: [
          {
            session_id: "r1",
            project_id: String(params["projectId"]),
            title: "Baseline run",
            status: "complete",
            created_at: "2026-07-20T10:00:00Z",
            updated_at: "2026-07-20T10:00:00Z",
            dataset_names: ["sample"],
            artifact_count: 12,
            report_status: "final",
            chat_message_count: 0,
          },
        ],
        next_cursor: null,
      } satisfies SessionPage),
    );
  }

  it("forks on an ML target and pulls the new run into the right-hand side", async () => {
    let forkBody: Record<string, unknown> | null = null;
    server.use(
      twoRuns(),
      http.post("/api/v1/sessions/:sessionId/fork", async ({ request, params }) => {
        forkBody = (await request.json()) as Record<string, unknown>;
        return HttpResponse.json(
          {
            session_id: String(params["sessionId"]),
            execution_session_id: "fksess_1",
            decision: "ML target → value",
            job: jobCreated("job_fork", "fksess_1"),
          },
          { status: 201 },
        );
      }),
    );

    const user = userEvent.setup();
    renderAppAt("/projects/p1/sessions/r1/compare");
    await screen.findByRole("heading", { name: "Compare" });

    await user.click(screen.getByText("Create a variant"));
    await user.selectOptions(screen.getByLabelText("Target column"), "value");
    await user.click(screen.getByRole("button", { name: "Run variant" }));

    expect(forkBody).toEqual({
      decision: "ml_target",
      ml_target_column: "value",
      datasets: [],
      llm: "env",
    });
    expect(
      await screen.findByText(/running, follow progress/),
    ).toBeInTheDocument();

    /* The forked run mints its own id inside the driver; the only place it
     * reaches the client is the job's `session.forked` frame. */
    const stream = FakeEventSource.instances.find((source) =>
      source.url.includes("job_fork"),
    )!;
    stream.emit("session.forked", {
      event_id: 7,
      job_id: "job_fork",
      session_id: "fksess_1",
      type: "session.forked",
      name: "job_fork",
      summary: { forked_session_id: "run_forked_9", decision: "ML target → value" },
    });

    expect(
      await screen.findByText(/Variant run run_forked_9 is ready to compare/),
    ).toBeInTheDocument();
    /* waitFor, not a bare assertion: the ?right= navigation lands one router
     * tick after the event that produced the id. */
    await waitFor(() =>
      expect(screen.getByLabelText(/Compare against/)).toHaveValue(
        "run_forked_9",
      ),
    );
  });

  it("refuses a dataset fork until at least one table is picked", async () => {
    server.use(twoRuns());
    const user = userEvent.setup();
    renderAppAt("/projects/p1/sessions/r1/compare");
    await screen.findByRole("heading", { name: "Compare" });

    await user.click(screen.getByText("Create a variant"));
    await user.selectOptions(screen.getByLabelText("What to vary"), "dataset");
    expect(screen.getByRole("button", { name: "Run variant" })).toBeDisabled();

    await user.click(await screen.findByRole("checkbox", { name: "sample.csv" }));
    expect(screen.getByRole("button", { name: "Run variant" })).toBeEnabled();
  });
});
