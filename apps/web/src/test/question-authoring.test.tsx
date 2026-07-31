import { describe, expect, it } from "vitest";
import { fireEvent, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { server } from "./msw/server";
import { renderAppAt } from "./render";

const PAGE_PATH = "/projects/p1/sessions/r1/questions";

async function openPage() {
  const user = userEvent.setup();
  renderAppAt(PAGE_PATH);
  await screen.findByRole("heading", { name: "Questions" });
  return user;
}

function questionCard(question: string) {
  const list = screen.getByRole("list", { name: "Question candidates" });
  return within(list)
    .getAllByRole("listitem")
    .find((card) => within(card).queryByText(question))!;
}

describe("Question card editing and drafting", () => {
  it("saves an edited card without changing execution-defining fields", async () => {
    let patchBody: Record<string, unknown> | null = null;
    server.use(
      http.patch(
        "/api/v1/sessions/:sessionId/questions/:questionId",
        async ({ request, params }) => {
          patchBody = (await request.json()) as Record<string, unknown>;
          return HttpResponse.json({
            question_id: String(params["questionId"]),
            question: "How is value trending over time?",
            origin: "template",
            priority: 0.82,
            executable: true,
            card_version: 2,
            business_decision: "Plan next quarter's inventory",
            value_hypothesis: "",
            success_criterion: "",
            data_signal: "",
            priority_rationale: "",
            risks: ["Seasonality is not controlled for"],
            data_requirements: [],
            target_datasets: ["sample.csv"],
            exploratory: false,
            execution: null,
          });
        },
      ),
    );

    const user = await openPage();
    const card = questionCard("How is value trending over time?");
    await user.click(within(card).getByRole("button", { name: "Edit card" }));
    const form = await screen.findByRole("form", { name: "Edit question card" });
    fireEvent.change(within(form).getByLabelText("Risks (one per line)"), {
      target: { value: "Seasonality is not controlled for" },
    });
    await user.click(within(form).getByRole("button", { name: "Save card" }));

    expect(patchBody).toMatchObject({
      risks: ["Seasonality is not controlled for"],
    });
    expect(Object.keys(patchBody ?? {})).not.toContain("sql_template");
    expect(Object.keys(patchBody ?? {})).not.toContain("target_datasets");
  });

  it("drafts one card from free text through prepare then confirm", async () => {
    let draftBody: Record<string, unknown> | null = null;
    server.use(
      http.post(
        "/api/v1/sessions/:sessionId/questions",
        async ({ request, params }) => {
          draftBody = (await request.json()) as Record<string, unknown>;
          return HttpResponse.json(
            {
              session_id: String(params["sessionId"]),
              execution_session_id: "qdsess_1",
              question: "Which region returns the most?",
              job: {
                job_id: "job_draft_1",
                session_id: "qdsess_1",
                status: "queued",
                events_url: "/api/v1/jobs/job_draft_1/events",
              },
            },
            { status: 201 },
          );
        },
      ),
    );

    const user = await openPage();
    fireEvent.change(screen.getByLabelText("Your question"), {
      target: { value: "Which region returns the most?" },
    });
    await user.click(
      screen.getByRole("button", { name: "Draft question card" }),
    );

    const dialog = await screen.findByRole("alertdialog", {
      name: "Confirm question drafting",
    });
    expect(
      within(dialog).getByText("Which region returns the most?"),
    ).toBeInTheDocument();
    await user.click(
      within(dialog).getByRole("button", { name: "Confirm & draft" }),
    );

    expect(await screen.findByText("job_draft_1")).toBeInTheDocument();
    expect(Object.keys(draftBody ?? {}).sort()).toEqual([
      "action_hash",
      "approval_token",
    ]);
  });
});
