import { describe, expect, it } from "vitest";
import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import type { DecisionStoryDraftRequest } from "../api/client";
import { decisionReportView } from "./msw/handlers";
import { server } from "./msw/server";
import { renderAppAt } from "./render";

const PATH = "/projects/p1/sessions/r1/report";

function serveDecisionReport() {
  server.use(
    http.get("/api/v1/sessions/:sessionId/decision-report", ({ params }) =>
      HttpResponse.json(decisionReportView(String(params["sessionId"]))),
    ),
  );
}

describe("Decision report on the Report page", () => {
  it("renders SCQA, candidate decisions, evidence, and status badges above the report", async () => {
    serveDecisionReport();
    renderAppAt(PATH);

    expect(
      await screen.findByRole("heading", { name: "Channel mix decision story" }),
    ).toBeInTheDocument();
    expect(screen.getByText("published")).toBeInTheDocument();
    expect(screen.getByText("freshness stale")).toBeInTheDocument();
    expect(screen.getByText("gate degraded")).toBeInTheDocument();
    expect(screen.getByText("confidence high")).toBeInTheDocument();
    expect(screen.getByText("report eligible with limitations")).toBeInTheDocument();

    for (const label of ["Situation", "Complication", "Question", "Answer"]) {
      expect(screen.getByText(label)).toBeInTheDocument();
    }
    expect(
      screen.getByText("Rebalance toward the channel holding value."),
    ).toBeInTheDocument();

    expect(screen.getByText("Candidate decisions")).toBeInTheDocument();
    expect(
      screen.getByText("Rebalance channel spend once labels are reviewed."),
    ).toBeInTheDocument();
    /* Candidate actions are unverified hypothesis context; the page says so. */
    expect(screen.getByText(/Hypothesis context/)).toBeInTheDocument();

    /* Evidence is a control that opens the artifact beside the story, not a
     * link that takes the reader off the page. */
    expect(
      screen.getByRole("button", { name: "table_orders" }),
    ).toHaveAttribute("aria-pressed", "false");

    expect(
      screen.getByText(/Source findings: 1 · export disabled/),
    ).toBeInTheDocument();

    /* The technical report still renders underneath. */
    expect(
      await screen.findByRole("heading", { name: "Demo report" }),
    ).toBeInTheDocument();

    const decision = screen.getByRole("region", { name: "Decision report" });
    const technical = screen.getByRole("region", { name: "Technical report" });
    expect(
      decision.compareDocumentPosition(technical) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  it("opens a cited artifact beside the story instead of navigating away", async () => {
    serveDecisionReport();
    server.use(
      http.get("/api/v1/sessions/:sessionId/artifacts/:artifactId", ({ params }) =>
        HttpResponse.json({
          artifact_id: String(params["artifactId"]),
          type: "Table",
          project_id: "p1",
          session_id: "r1",
          created_at: "2026-07-22T12:00:00Z",
          payload: { title: "Orders by channel", rows: [1, 2, 3] },
          warnings: ["Sampled to 10k rows."],
        }),
      ),
    );

    const user = userEvent.setup();
    renderAppAt(PATH);
    await user.click(
      await screen.findByRole("button", { name: "table_orders" }),
    );

    const panel = await screen.findByRole("complementary", {
      name: "Evidence inspector",
    });
    expect(within(panel).getByText("Table")).toBeInTheDocument();
    /* Payload keys are readable without opening the JSON. */
    expect(within(panel).getByText("Orders by channel")).toBeInTheDocument();
    expect(within(panel).getByText("3 items")).toBeInTheDocument();
    /* Warnings ride along with evidence and are shown, not swallowed. */
    expect(within(panel).getByText("Sampled to 10k rows.")).toBeInTheDocument();
    expect(
      within(panel).getByRole("link", { name: "Open in Artifacts" }),
    ).toHaveAttribute(
      "href",
      "/projects/p1/sessions/r1/artifacts?artifact=table_orders",
    );
    /* The reader never left the report. */
    expect(
      screen.getByRole("heading", { name: "Demo report" }),
    ).toBeInTheDocument();

    await user.click(within(panel).getByRole("button", { name: "Close" }));
    expect(
      screen.queryByRole("complementary", { name: "Evidence inspector" }),
    ).not.toBeInTheDocument();
  });

  it("warns and disables export when the report is not fresh", async () => {
    serveDecisionReport();
    renderAppAt(PATH);
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/stale for current reuse/);
    expect(alert).toHaveTextContent("vf_1: source dataset changed.");
  });

  it("renders nothing when the project has no decision report", async () => {
    renderAppAt(PATH);
    expect(
      await screen.findByRole("heading", { name: "Demo report" }),
    ).toBeInTheDocument();
    expect(screen.queryByText("Candidate decisions")).not.toBeInTheDocument();
    expect(screen.queryByText("Situation")).not.toBeInTheDocument();
    expect(screen.queryByText("published")).not.toBeInTheDocument();
  });

  it.each([
    ["decision_report_missing", 404, "file is missing"],
    ["decision_report_corrupt", 500, "unreadable or invalid"],
    ["decision_report_identity_invalid", 500, "unreadable or invalid"],
    ["decision_report_too_large", 500, "unreadable or invalid"],
    ["decision_report_unavailable", 503, "temporarily unavailable"],
  ])("shows a stored-report error for %s", async (code, status, message) => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/decision-report", () =>
        HttpResponse.json(
          { error: { code, message: "Safe server message." } },
          { status },
        ),
      ),
    );
    renderAppAt(PATH);
    expect(
      await screen.findByRole("heading", { name: "Demo report" }),
    ).toBeInTheDocument();
    const alert = await screen.findByRole("alert", {
      name: "Stored decision report unavailable",
    });
    expect(alert).toHaveTextContent(message);
    expect(screen.queryByText("Candidate decisions")).not.toBeInTheDocument();
    expect(screen.queryByText("No decision report yet")).not.toBeInTheDocument();
  });
});

describe("Decision Story curation on the Report page", () => {
  it("hides stale findings by default", async () => {
    renderAppAt(PATH);
    await screen.findByRole("heading", { name: "Decision Story" });
    expect(
      screen.getByRole("checkbox", {
        name: /Which region drives the revenue drop\?/,
      }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("checkbox", {
        name: /Does the refund spike track a single channel\?/,
      }),
    ).not.toBeInTheDocument();
  });

  it("shows stale findings once the toggle is switched on", async () => {
    const user = userEvent.setup();
    renderAppAt(PATH);
    await screen.findByRole("heading", { name: "Decision Story" });
    await user.click(screen.getByLabelText("Show stale findings"));
    expect(
      screen.getByRole("checkbox", {
        name: /Does the refund spike track a single channel\?.*stale/,
      }),
    ).toBeInTheDocument();
  });

  it("sends exactly the selected finding ids when creating a draft", async () => {
    let capturedBody: DecisionStoryDraftRequest | null = null;
    server.use(
      http.post(
        "/api/v1/sessions/:sessionId/decision-story/drafts",
        async ({ request, params }) => {
          capturedBody = (await request.json()) as DecisionStoryDraftRequest;
          return HttpResponse.json(
            {
              session_id: String(params["sessionId"]),
              execution_session_id: "sbsess_test",
              job: {
                job_id: "job_story_draft",
                session_id: "sbsess_test",
                project_id: "p1",
                kind: "synthesis_brief_create",
                status: "queued",
                cancel_requested: false,
                created_at: "2026-07-25T10:00:00Z",
                started_at: null,
                finished_at: null,
                error_code: null,
                error_message: null,
                events_url: "/api/v1/jobs/job_story_draft/events",
              },
            },
            { status: 201 },
          );
        },
      ),
    );

    const user = userEvent.setup();
    renderAppAt(PATH);
    await screen.findByRole("heading", { name: "Decision Story" });
    /* vf_2 is stale and hidden by default — show it so both can be selected. */
    await user.click(screen.getByLabelText("Show stale findings"));
    await user.click(
      screen.getByRole("checkbox", {
        name: /Which region drives the revenue drop\?/,
      }),
    );
    await user.click(
      screen.getByRole("checkbox", {
        name: /Does the refund spike track a single channel\?/,
      }),
    );
    await user.click(
      screen.getByRole("button", { name: "Create decision story draft" }),
    );

    expect(capturedBody).not.toBeNull();
    expect(capturedBody!.finding_artifact_ids).toEqual(["vf_1", "vf_2"]);
    expect(capturedBody!.finding_session_ids).toEqual({
      vf_1: "r1",
      vf_2: "r1",
    });
  });

  it("renders a draft's storyline as title/body beats", async () => {
    renderAppAt(PATH);
    expect(
      await screen.findByRole("heading", {
        name: "Refunds, not demand, drove the Q3 dip.",
      }),
    ).toBeInTheDocument();
    expect(screen.getByText("What happened")).toBeInTheDocument();
    expect(screen.getByText("Revenue fell 8% QoQ.")).toBeInTheDocument();
    expect(screen.getByText("Why")).toBeInTheDocument();
    expect(
      screen.getByText("Refunds explain 6 of those 8 points."),
    ).toBeInTheDocument();
  });

  it("shows plain-language guidance, not the raw code, on a 409 busy conflict", async () => {
    server.use(
      http.post("/api/v1/sessions/:sessionId/decision-story/drafts", () =>
        HttpResponse.json(
          {
            error: {
              code: "decision_story_busy",
              message: "Session r1 has an active job (job_1); wait for it to finish.",
            },
          },
          { status: 409 },
        ),
      ),
    );

    const user = userEvent.setup();
    renderAppAt(PATH);
    await screen.findByRole("heading", { name: "Decision Story" });
    await user.click(
      screen.getByRole("checkbox", {
        name: /Which region drives the revenue drop\?/,
      }),
    );
    await user.click(
      screen.getByRole("button", { name: "Create decision story draft" }),
    );

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(
      "A decision story job is already running for this session. Wait for it to finish, then try again.",
    );
    expect(alert).not.toHaveTextContent("decision_story_busy");
  });

  it("shows plain-language guidance, not the raw code, on a 422 not-draftable selection", async () => {
    server.use(
      http.post("/api/v1/sessions/:sessionId/decision-story/drafts", () =>
        HttpResponse.json(
          {
            error: {
              code: "decision_story_not_draftable",
              message:
                "Only report-eligible validated findings can enter a decision story: vf_1",
            },
          },
          { status: 422 },
        ),
      ),
    );

    const user = userEvent.setup();
    renderAppAt(PATH);
    await screen.findByRole("heading", { name: "Decision Story" });
    await user.click(
      screen.getByRole("checkbox", {
        name: /Which region drives the revenue drop\?/,
      }),
    );
    await user.click(
      screen.getByRole("button", { name: "Create decision story draft" }),
    );

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(
      "Some selected findings can no longer become a decision story. Refresh and re-select.",
    );
    expect(alert).not.toHaveTextContent("decision_story_not_draftable");
  });
});
