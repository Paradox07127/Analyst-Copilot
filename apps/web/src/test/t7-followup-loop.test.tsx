/* T7 follow-up loop: contextual "Ask about this" entries into chat, and the
 * report-staleness banner after a question execution completes. */

import { describe, expect, it } from "vitest";
import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { server } from "./msw/server";
import { renderAppAt, renderAppWithRouterAt } from "./render";
import { jobStatus } from "./msw/handlers";

/* The default report fixture carries generated_at 2026-07-22T12:00:00Z. */
function questionExecJobs(finishedAt: string) {
  return http.get("/api/v1/sessions/:sessionId/jobs", ({ params }) =>
    HttpResponse.json({
      session_id: String(params["sessionId"]),
      jobs: [
        {
          ...jobStatus("job_qexec_1", {
            session_id: "qsess_r1_1",
            kind: "question_exec",
            status: "completed",
            finished_at: finishedAt,
          }),
          source_session_id: String(params["sessionId"]),
        },
      ],
    }),
  );
}

describe("report staleness after a question execution", () => {
  it("offers one-click regeneration when a question finished after the report", async () => {
    let generateCalls = 0;
    server.use(
      questionExecJobs("2026-07-23T09:00:00Z"),
      http.post("/api/v1/sessions/:sessionId/report/generate", ({ params }) => {
        generateCalls += 1;
        return HttpResponse.json(
          {
            session_id: String(params["sessionId"]),
            execution_session_id: "rpsess_t7",
            regenerated: true,
            job: {
              job_id: "job_regen_t7",
              session_id: "rpsess_t7",
              status: "queued",
              events_url: "/api/v1/jobs/job_regen_t7/events",
            },
          },
          { status: 201 },
        );
      }),
    );

    const user = userEvent.setup();
    renderAppAt("/projects/p1/sessions/r1/report");

    const banner = await screen.findByRole("status", {
      name: "Report out of date",
    });
    expect(
      within(banner).getByText(
        "The report does not include your latest question results.",
      ),
    ).toBeInTheDocument();

    /* One click, no confirm step: the banner already says what changed. */
    await user.click(
      within(banner).getByRole("button", { name: "Regenerate report" }),
    );
    expect(generateCalls).toBe(1);
    expect(
      await within(banner).findByText(/Regenerating the report/),
    ).toBeInTheDocument();
  });

  it("stays silent when the report is newer than every question execution", async () => {
    server.use(questionExecJobs("2026-07-21T09:00:00Z"));

    renderAppAt("/projects/p1/sessions/r1/report");
    await screen.findByRole("heading", { name: "Demo report" });

    expect(
      screen.queryByRole("status", { name: "Report out of date" }),
    ).not.toBeInTheDocument();
  });

  it("surfaces the same notice on the Questions page with a report link", async () => {
    server.use(questionExecJobs("2026-07-23T09:00:00Z"));

    renderAppAt("/projects/p1/sessions/r1/questions");

    const banner = await screen.findByRole("status", {
      name: "Report out of date",
    });
    expect(
      within(banner).getByRole("link", { name: "Open the report" }),
    ).toHaveAttribute("href", "/projects/p1/sessions/r1/report");
    expect(
      within(banner).getByRole("button", { name: "Regenerate report" }),
    ).toBeInTheDocument();
  });
});

describe("Ask about this entries into chat", () => {
  it("prefills chat from a finding card without sending", async () => {
    const user = userEvent.setup();
    const { router } = renderAppWithRouterAt("/projects/p1/sessions/r1/findings");

    /* Wait for the page (and its sibling queries) to settle before clicking:
     * a link found mid-load can be replaced by a re-render before the click. */
    await screen.findByRole("heading", { name: "Decision coverage" });
    const card = (await screen.findAllByRole("listitem")).find((item) =>
      within(item).queryByText("What was average order value?"),
    )!;
    await user.click(within(card).getByRole("link", { name: "Ask about this" }));

    await waitFor(() =>
      expect(router.state.location.pathname).toBe(
        "/projects/p1/sessions/r1/chat",
      ),
    );
    const input = await screen.findByLabelText("Message");
    expect((input as HTMLTextAreaElement).value).toBe(
      'About finding finding_1 ("What was average order value?") — ' +
        '"Average order value was $42.": why does the data support this, ' +
        "and what should I check before relying on it?",
    );
    expect((input as HTMLTextAreaElement).value).toContain(
      "why does the data support this",
    );
    /* Prefilled, not sent: the transcript stays empty until the user sends. */
    expect(screen.queryByText(/^user$/)).not.toBeInTheDocument();
  });

  it("prefills chat from a report section heading", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/report", ({ params }) =>
        HttpResponse.json({
          session_id: String(params["sessionId"]),
          status: "validated",
          markdown: "# Demo report\n\n## Key Findings\n\nRevenue grew.",
          generated_at: "2026-07-22T12:00:00Z",
        }),
      ),
    );

    const user = userEvent.setup();
    const { router } = renderAppWithRouterAt("/projects/p1/sessions/r1/report");

    await user.click(
      await screen.findByRole("button", { name: "Ask about Key Findings" }),
    );

    expect(router.state.location.pathname).toBe(
      "/projects/p1/sessions/r1/chat",
    );
    const input = await screen.findByLabelText("Message");
    expect((input as HTMLTextAreaElement).value).toContain(
      'About the "Key Findings" section',
    );
  });
});
