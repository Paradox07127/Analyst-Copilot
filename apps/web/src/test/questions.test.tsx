import { describe, expect, it } from "vitest";
import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { server } from "./msw/server";
import { renderAppAt } from "./render";
import { questionsView } from "./msw/handlers";

const PAGE_PATH = "/projects/p1/sessions/r1/questions";

describe("Questions page", () => {
  it("offers only the direct single-question workflow", async () => {
    renderAppAt(`${PAGE_PATH}?selected=q_segment,q_trend`);

    await screen.findByRole("heading", { name: "Questions" });
    expect(
      screen.getByText(
        "Review a suggested question or ask your own, then approve what runs.",
      ),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("checkbox", {
        name: /Select for an investigation plan/,
      }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("heading", { name: "Plan an investigation" }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("heading", { name: "Investigation plans" }),
    ).not.toBeInTheDocument();
  });

  it("renders candidate cards with badges, execution status, and findings link", async () => {
    renderAppAt(PAGE_PATH);

    await screen.findByRole("heading", { name: "Questions" });
    expect(
      screen.getByText("How is value trending over time?"),
    ).toBeInTheDocument();

    const cards = screen.getAllByRole("listitem");
    const trendCard = cards.find((card) =>
      within(card).queryByText("How is value trending over time?"),
    )!;
    expect(within(trendCard).getByText("template")).toBeInTheDocument();
    expect(within(trendCard).getByText("descriptive")).toBeInTheDocument();
    expect(within(trendCard).getByText("High priority")).toBeInTheDocument();
    expect(within(trendCard).getByText("answered")).toBeInTheDocument();
    const findingsLink = within(trendCard).getByRole("link", {
      name: "2 findings",
    });
    expect(findingsLink).toHaveAttribute(
      "href",
      "/projects/p1/sessions/qsess_1/artifacts",
    );

    /* Feasibility-blocked cards expose no approve control. */
    const blockedCard = cards.find((card) =>
      within(card).queryByText("Can churn be predicted from labels we lack?"),
    )!;
    expect(
      within(blockedCard).queryByRole("button", { name: "Approve & run" }),
    ).not.toBeInTheDocument();
    expect(
      within(blockedCard).getByText(/Not executable: feasibility is needs_data/),
    ).toBeInTheDocument();
  });

  it("shows the review step before the user commits to running a question", async () => {
    const user = userEvent.setup();
    renderAppAt(PAGE_PATH);

    await screen.findByRole("heading", { name: "Questions" });
    const trendCard = screen
      .getAllByRole("listitem")
      .find((card) =>
        within(card).queryByText("How is value trending over time?"),
      )!;

    /* Before any click: the whole sequence is on screen, so "Approve & run"
     * cannot read as the point of no return. */
    const chain = within(trendCard).getByRole("list", {
      name: "Run this question",
    });
    expect(within(chain).getAllByRole("listitem")).toHaveLength(3);
    expect(within(chain).getByText("Approve")).toHaveAttribute(
      "aria-current",
      "step",
    );
    expect(within(chain).getByText("Review what will run")).toBeInTheDocument();
    expect(within(chain).getByText("Execute")).toBeInTheDocument();

    await user.click(
      within(trendCard).getByRole("button", { name: "Approve & run" }),
    );
    const dialog = await screen.findByRole("alertdialog", {
      name: "Confirm question execution",
    });
    expect(within(dialog).getByText("Review what will run")).toHaveAttribute(
      "aria-current",
      "step",
    );
    expect(within(dialog).getByText("Nothing has run yet.")).toBeInTheDocument();
  });

  it("weights the outcome above the question's genre", async () => {
    renderAppAt(PAGE_PATH);

    await screen.findByRole("heading", { name: "Questions" });
    const trendCard = screen
      .getAllByRole("listitem")
      .find((card) =>
        within(card).queryByText("How is value trending over time?"),
      )!;

    /* State shouts; genre does not. Five equal small-caps chips was the bug. */
    expect(within(trendCard).getByText("answered").className).toContain(
      "uppercase",
    );
    for (const genre of ["template", "descriptive", "financial performance"]) {
      expect(within(trendCard).getByText(genre).className).not.toContain(
        "uppercase",
      );
    }
  });

  it("approves, confirms the prepared content, executes, and tracks the job", async () => {
    let idempotencyKey: string | null = null;
    let executeBody: Record<string, unknown> | null = null;
    server.use(
      http.post(
        "/api/v1/sessions/:sessionId/questions/:questionId/execute",
        async ({ request, params }) => {
          idempotencyKey = request.headers.get("Idempotency-Key");
          executeBody = (await request.json()) as Record<string, unknown>;
          return HttpResponse.json(
            {
              session_id: String(params["sessionId"]),
              question_id: String(params["questionId"]),
              execution_session_id: "qsess_new_1",
              job: {
                job_id: "job_q_1",
                session_id: "qsess_new_1",
                status: "queued",
                events_url: "/api/v1/jobs/job_q_1/events",
              },
            },
            { status: 201 },
          );
        },
      ),
    );

    const user = userEvent.setup();
    renderAppAt(PAGE_PATH);

    await screen.findByRole("heading", { name: "Questions" });
    const approveButton = screen.getAllByRole("button", {
      name: "Approve & run",
    })[0]!;
    await user.click(approveButton);

    /* The confirm card shows the approved scope and autonomous capabilities. */
    const dialog = await screen.findByRole("alertdialog", {
      name: "Confirm question execution",
    });
    expect(
      within(dialog).getByText(/The agent chooses the analysis at execution time/),
    ).toBeInTheDocument();
    expect(within(dialog).getByText(/replay a compatible Skill/)).toBeInTheDocument();
    expect(within(dialog).getByText(/sample\.csv/)).toBeInTheDocument();
    /* env mode shows the cost warning even for template questions. */
    expect(within(dialog).getByText(/LLM mode:/)).toBeInTheDocument();
    expect(within(dialog).getByText("env")).toBeInTheDocument();
    expect(
      within(dialog).getByText(
        "Executes with the live model and may incur cost.",
      ),
    ).toBeInTheDocument();

    await user.click(
      within(dialog).getByRole("button", { name: "Confirm & execute" }),
    );

    expect(
      await screen.findByText(/Tracking job_q_1 · session qsess_new_1/),
    ).toBeInTheDocument();
    expect(idempotencyKey).toMatch(/^[0-9a-f-]{36}$/);
    expect(executeBody).toMatchObject({
      action_hash: "a".repeat(64),
      approval_token: "c".repeat(32),
    });
    /* The client must not try to pick the llm at execute time. */
    expect(executeBody).not.toHaveProperty("llm");
  });

  it("binds the selected offline mode at prepare and hides the cost hint", async () => {
    let prepareBody: Record<string, unknown> | null = null;
    server.use(
      http.post(
        "/api/v1/sessions/:sessionId/questions/:questionId/prepare",
        async ({ request, params }) => {
          prepareBody = (await request.json()) as Record<string, unknown>;
          return HttpResponse.json({
            session_id: String(params["sessionId"]),
            question_id: String(params["questionId"]),
            action_hash: "a".repeat(64),
            approval_token: "c".repeat(32),
            expires_at: "2026-07-25T12:00:00Z",
            question: "How is value trending over time?",
            origin: "template",
            sql_preview: "select month, sum(value) from sample group by 1",
            target_datasets: ["sample.csv"],
            uses_llm: false,
            llm_mode: "offline",
          });
        },
      ),
    );

    const user = userEvent.setup();
    renderAppAt(PAGE_PATH);

    await screen.findByRole("heading", { name: "Questions" });
    await user.selectOptions(
      screen.getAllByRole("combobox", { name: "LLM mode" })[0]!,
      "offline",
    );
    await user.click(
      screen.getAllByRole("button", { name: "Approve & run" })[0]!,
    );

    const dialog = await screen.findByRole("alertdialog", {
      name: "Confirm question execution",
    });
    expect(prepareBody).toMatchObject({ llm: "offline" });
    expect(within(dialog).getByText("offline")).toBeInTheDocument();
    expect(
      within(dialog).queryByText(
        "Executes with the live model and may incur cost.",
      ),
    ).not.toBeInTheDocument();
  });

  it("guides recovery when the retry key hits a conflicting job", async () => {
    server.use(
      http.post("/api/v1/sessions/:sessionId/questions/:questionId/execute", () =>
        HttpResponse.json(
          {
            error: {
              code: "job_conflict",
              message: "Idempotency key already used.",
            },
          },
          { status: 409 },
        ),
      ),
    );

    const user = userEvent.setup();
    renderAppAt(PAGE_PATH);

    await screen.findByRole("heading", { name: "Questions" });
    await user.click(
      screen.getAllByRole("button", { name: "Approve & run" })[0]!,
    );
    await user.click(
      await screen.findByRole("button", { name: "Confirm & execute" }),
    );

    expect(
      await screen.findByText(
        "This request conflicts with an earlier execution.",
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Approve again" }),
    ).toBeInTheDocument();
  });

  it("explains an already-used approval on 409 approval_consumed", async () => {
    server.use(
      http.post("/api/v1/sessions/:sessionId/questions/:questionId/execute", () =>
        HttpResponse.json(
          {
            error: {
              code: "approval_consumed",
              message: "Approval was already used.",
            },
          },
          { status: 409 },
        ),
      ),
    );

    const user = userEvent.setup();
    renderAppAt(PAGE_PATH);

    await screen.findByRole("heading", { name: "Questions" });
    const approveButton = screen.getAllByRole("button", {
      name: "Approve & run",
    })[0]!;
    await user.click(approveButton);
    await user.click(
      await screen.findByRole("button", { name: "Confirm & execute" }),
    );

    expect(
      await screen.findByText("This approval was already used."),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Approve again" }),
    ).toBeInTheDocument();
  });

  it("keeps free-text question drafting available when there are no suggestions", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/questions", ({ params }) =>
        HttpResponse.json({
          ...questionsView(String(params["sessionId"])),
          questions: [],
        }),
      ),
    );
    const user = userEvent.setup();
    renderAppAt(PAGE_PATH);

    expect(
      await screen.findByText("No suggested questions yet"),
    ).toBeInTheDocument();
    expect(
      screen.getByText("You can still draft and run your own question above."),
    ).toBeInTheDocument();

    const input = screen.getByRole("textbox", { name: "Your question" });
    await user.type(input, "Which region has the highest return rate?");
    await user.click(
      screen.getByRole("button", { name: "Draft question card" }),
    );

    const dialog = await screen.findByRole("alertdialog", {
      name: "Confirm question drafting",
    });
    expect(
      within(dialog).getByText("Which region has the highest return rate?"),
    ).toBeInTheDocument();
  });
});
