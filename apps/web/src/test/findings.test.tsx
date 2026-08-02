import { describe, expect, it } from "vitest";
import { fireEvent, screen, within } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { server } from "./msw/server";
import { renderAppAt, renderAppWithRouterAt } from "./render";
import { decisionCoverageView, findingsView } from "./msw/handlers";

const PAGE_PATH = "/projects/p1/sessions/r1/findings";

describe("Findings page", () => {
  it("renders finding cards with badges, source-run and evidence links", async () => {
    renderAppAt(PAGE_PATH);

    await screen.findByRole("heading", { name: "Findings" });
    expect(screen.getByText("Project scope")).toBeInTheDocument();
    expect(screen.getByText("Current session r1")).toBeInTheDocument();
    const cards = screen.getAllByRole("listitem");
    const libCard = cards.find((card) =>
      within(card).queryByText("What was average order value?"),
    )!;
    expect(within(libCard).getByText("observed")).toBeInTheDocument();
    expect(within(libCard).getByText("reliability high")).toBeInTheDocument();
    expect(
      within(libCard).getByText("report eligible with limitations"),
    ).toBeInTheDocument();
    expect(within(libCard).getByText("freshness fresh")).toBeInTheDocument();
    expect(
      within(libCard).getByText("Average order value was $42."),
    ).toBeInTheDocument();
    /* Cross-run provenance: the card links to its source run's artifacts. */
    expect(
      within(libCard).getByRole("link", { name: "Source session r_lib" }),
    ).toHaveAttribute("href", "/projects/p1/sessions/r_lib/artifacts");
    /* Evidence deep-links to the artifact itself, not just to the page. */
    expect(
      within(libCard).getByRole("link", { name: "evidence sqlres_1" }),
    ).toHaveAttribute(
      "href",
      "/projects/p1/sessions/r_lib/artifacts?artifact=sqlres_1",
    );

    /* Stale current-run finding surfaces its freshness reasons. */
    const currentCard = cards.find((card) =>
      within(card).queryByText("Do regions differ in revenue?"),
    )!;
    expect(within(currentCard).getByText("this session")).toBeInTheDocument();
    expect(within(currentCard).getByText("freshness stale")).toBeInTheDocument();
    expect(
      within(currentCard).getByText(
        "Dataset 'orders' changed since the finding was saved.",
      ),
    ).toBeInTheDocument();
  });

  it("renders the investigation log", async () => {
    renderAppAt(PAGE_PATH);

    await screen.findByRole("heading", { name: "Investigation log" });
    expect(screen.getByText("Is churn predictable?")).toBeInTheDocument();
    expect(screen.getByText("inconclusive")).toBeInTheDocument();
    expect(screen.getByText("Next: Collect churn labels.")).toBeInTheDocument();
  });

  it("renders internal-run findings without dead links", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/findings", ({ params }) => {
        const view = findingsView(String(params["sessionId"]));
        const base = view.findings![0]!;
        const finding = {
          ...base,
          source_session_id: "run_probe__internal_x",
          source_session_navigable: false,
          statements: [
            {
              ...base.statements![0]!,
              evidence: [
                { ...base.statements![0]!.evidence![0]!, session_id: null },
              ],
            },
          ],
        };
        return HttpResponse.json({ ...view, findings: [finding], records: [] });
      }),
    );
    renderAppAt(PAGE_PATH);

    await screen.findByRole("heading", { name: "Findings" });
    /* Source session and evidence render as plain text — there is no page for an
     * internal run, so a link would 404. */
    expect(
      screen.getByText("Source session run_probe__internal_x"),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("link", { name: "Source session run_probe__internal_x" }),
    ).not.toBeInTheDocument();
    expect(screen.getByText("evidence sqlres_1")).toBeInTheDocument();
    expect(
      screen.queryByRole("link", { name: "evidence sqlres_1" }),
    ).not.toBeInTheDocument();
  });

  it("explains the report status instead of leaving a bare badge", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/findings", ({ params }) => {
        const view = findingsView(String(params["sessionId"]));
        return HttpResponse.json({
          ...view,
          findings: [
            {
              ...view.findings![0]!,
              report_readiness: "not_eligible",
              report_readiness_reason:
                "sample size below the pre-registered minimum",
            },
          ],
          records: [],
        });
      }),
    );
    renderAppAt(PAGE_PATH);

    await screen.findByRole("heading", { name: "Findings" });
    expect(
      screen.getByText(
        "Report status reason: sample size below the pre-registered minimum",
      ),
    ).toBeInTheDocument();
  });

  it("renders the evidence, decision-readiness and hypothesis fields", async () => {
    renderAppAt(PAGE_PATH);

    await screen.findByRole("heading", { name: "Findings" });
    const cards = screen.getAllByRole("listitem");
    const libCard = cards.find((card) =>
      within(card).queryByText("What was average order value?"),
    )!;
    expect(within(libCard).getByText("evidence high")).toBeInTheDocument();
    expect(within(libCard).getByText("decision medium")).toBeInTheDocument();
    expect(
      within(libCard).getByText("Hypothesis (unvalidated): Could increase profit."),
    ).toBeInTheDocument();

    /* value_hypothesis is optional — a finding without one shows no caption. */
    const currentCard = cards.find((card) =>
      within(card).queryByText("Do regions differ in revenue?"),
    )!;
    expect(
      within(currentCard).queryByText(/Hypothesis \(unvalidated\)/),
    ).not.toBeInTheDocument();
  });

  it("links a finding's method artifact to the source session", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/findings", ({ params }) => {
        const view = findingsView(String(params["sessionId"]));
        return HttpResponse.json({
          ...view,
          findings: [
            {
              ...view.findings![0]!,
              method_artifact_id: "method_1",
            },
          ],
          records: [],
        });
      }),
    );
    renderAppAt(PAGE_PATH);

    expect(
      await screen.findByRole("link", { name: "method method_1" }),
    ).toHaveAttribute(
      "href",
      "/projects/p1/sessions/r_lib/artifacts?artifact=method_1",
    );
  });

  it("shows the empty state when the library has no findings", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/findings", ({ params }) =>
        HttpResponse.json({
          ...findingsView(String(params["sessionId"])),
          findings: [],
          records: [],
        }),
      ),
    );
    renderAppAt(PAGE_PATH);

    await screen.findByRole("heading", { name: "Findings" });
    expect(
      await screen.findByText("No validated findings yet"),
    ).toBeInTheDocument();
    /* Explains why the page is empty (validated findings come from
     * investigations, not Auto EDA alone) and gives a next-step entry point. */
    expect(
      screen.getByText(/Validated findings come from completed investigation workflows/),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("link", {
        name: "Review suggested questions",
      }),
    ).toHaveAttribute("href", "/projects/p1/sessions/r1/questions");
  });
});

describe("Findings reliability filter", () => {
  it("defaults to All and narrows the list by analytical_reliability", async () => {
    renderAppAt(PAGE_PATH);
    await screen.findByText("What was average order value?");
    expect(screen.getByText("Do regions differ in revenue?")).toBeInTheDocument();

    const select = screen.getByLabelText("Analytical reliability");
    expect(select).toHaveValue("All");

    fireEvent.change(select, { target: { value: "high" } });
    /* finding_1 is high, finding_2 is medium — only the high one remains. */
    expect(
      screen.getByText("What was average order value?"),
    ).toBeInTheDocument();
    expect(
      screen.queryByText("Do regions differ in revenue?"),
    ).not.toBeInTheDocument();
  });

  it("shows a distinct empty state when no finding matches the reliability", async () => {
    renderAppAt(PAGE_PATH);
    await screen.findByText("What was average order value?");

    fireEvent.change(screen.getByLabelText("Analytical reliability"), {
      target: { value: "low" },
    });
    expect(
      await screen.findByText("No findings at this reliability level"),
    ).toBeInTheDocument();
  });

  it("restores reliability from the URL and removes the default from it", async () => {
    const { router } = renderAppWithRouterAt(
      `${PAGE_PATH}?reliability=high`,
    );
    await screen.findByText("What was average order value?");

    expect(screen.getByLabelText("Analytical reliability")).toHaveValue("high");
    expect(
      screen.queryByText("Do regions differ in revenue?"),
    ).not.toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Analytical reliability"), {
      target: { value: "All" },
    });
    expect(router.state.location.search).toBe("");
  });
});

describe("Findings outcome status filter", () => {
  function recordsFixture() {
    return [
      {
        artifact_id: "record_a",
        source_session_id: "r_lib",
        from_current_session: false,
        question: "Is churn predictable?",
        status: "inconclusive",
        reason_code: "insufficient_signal",
        reason: "Not enough labelled data.",
        next_action: "Collect churn labels.",
      },
      {
        artifact_id: "record_b",
        source_session_id: "r_lib",
        from_current_session: false,
        question: "Does region drive margin?",
        status: "inconclusive",
        reason_code: "insufficient_signal",
        reason: "Sample too small.",
        next_action: "Collect more regions.",
      },
      {
        artifact_id: "record_c",
        source_session_id: "r1",
        from_current_session: true,
        question: "Can price elasticity be estimated?",
        status: "needs_data",
        reason_code: "missing_column",
        reason: "No historical price column.",
        next_action: "Instrument pricing changes.",
      },
    ];
  }

  it("shows each outcome option with its count, mirroring _outcome_counts", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/findings", ({ params }) =>
        HttpResponse.json({
          ...findingsView(String(params["sessionId"])),
          records: recordsFixture(),
        }),
      ),
    );
    renderAppAt(PAGE_PATH);

    await screen.findByRole("heading", { name: "Investigation log" });
    expect(
      screen.getByRole("checkbox", { name: "inconclusive (2)" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("checkbox", { name: "needs_data (1)" }),
    ).toBeInTheDocument();
  });

  it("defaults to every status selected, then narrows on uncheck", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/findings", ({ params }) =>
        HttpResponse.json({
          ...findingsView(String(params["sessionId"])),
          records: recordsFixture(),
        }),
      ),
    );
    renderAppAt(PAGE_PATH);

    await screen.findByText("Is churn predictable?");
    expect(screen.getByText("Does region drive margin?")).toBeInTheDocument();
    expect(
      screen.getByText("Can price elasticity be estimated?"),
    ).toBeInTheDocument();

    const inconclusive = screen.getByRole("checkbox", {
      name: "inconclusive (2)",
    });
    expect(inconclusive).toBeChecked();
    fireEvent.click(inconclusive);

    expect(screen.queryByText("Is churn predictable?")).not.toBeInTheDocument();
    expect(
      screen.queryByText("Does region drive margin?"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByText("Can price elasticity be estimated?"),
    ).toBeInTheDocument();
  });

  it("restores the selected outcome statuses from the URL", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/findings", ({ params }) =>
        HttpResponse.json({
          ...findingsView(String(params["sessionId"])),
          records: recordsFixture(),
        }),
      ),
    );
    renderAppAt(`${PAGE_PATH}?outcome=needs_data`);

    await screen.findByText("Can price elasticity be estimated?");
    expect(
      screen.queryByText("Is churn predictable?"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByRole("checkbox", { name: "inconclusive (2)" }),
    ).not.toBeChecked();
    expect(
      screen.getByRole("checkbox", { name: "needs_data (1)" }),
    ).toBeChecked();
  });

  it("keeps the log section visible when nothing has been recorded", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/findings", ({ params }) =>
        HttpResponse.json({
          ...findingsView(String(params["sessionId"])),
          records: [],
        }),
      ),
    );
    renderAppAt(PAGE_PATH);

    /* "No data" and "failed to load" must not look the same
     * (the Findings API keeps the section and states it). */
    expect(
      await screen.findByRole("heading", { name: "Investigation log" }),
    ).toBeInTheDocument();
    expect(
      screen.getByText("No investigation outcomes have been recorded yet."),
    ).toBeInTheDocument();
  });

  it("shows a distinct empty state when every status is unchecked", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/findings", ({ params }) =>
        HttpResponse.json({
          ...findingsView(String(params["sessionId"])),
          records: recordsFixture(),
        }),
      ),
    );
    renderAppAt(PAGE_PATH);

    await screen.findByText("Is churn predictable?");
    fireEvent.click(screen.getByRole("checkbox", { name: "inconclusive (2)" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "needs_data (1)" }));

    expect(
      await screen.findByText(
        "No investigation outcomes match the selected statuses",
      ),
    ).toBeInTheDocument();
  });
});

describe("Findings top metrics", () => {
  it("renders the 5 headline metrics from findings and investigation records", async () => {
    renderAppAt(PAGE_PATH);
    await screen.findByRole("heading", { name: "Findings" });

    /* findingsView() fixture: finding_1 is eligible_with_limitations, finding_2
     * is eligible; 1 investigation record with status inconclusive. */
    const metric = (label: string) =>
      screen.getByText(label).nextElementSibling as HTMLElement;
    expect(metric("Validated findings").textContent).toBe("2");
    expect(metric("Direct report").textContent).toBe("1");
    expect(metric("With limitations").textContent).toBe("1");
    expect(metric("Inconclusive").textContent).toBe("1");
    expect(metric("Needs data").textContent).toBe("0");
  });
});

describe("Decision coverage", () => {
  it("renders the terminal/total ratio and the coverage-gaps badge", async () => {
    renderAppAt(PAGE_PATH);

    await screen.findByRole("heading", { name: "Decision coverage" });
    /* decisionCoverageView() fixture: top_cards_terminal=2, top_cards_total=8, coverage_ready=false.
     * findByText (not getByText) waits out the section's own async query. */
    expect(await screen.findByText("2/8")).toBeInTheDocument();
    expect(screen.getByText("Coverage gaps remain")).toBeInTheDocument();
    expect(
      screen.queryByText("Report-ready coverage"),
    ).not.toBeInTheDocument();
  });

  it("shows the report-ready badge instead when coverage_ready is true", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/decision-coverage", ({ params }) =>
        HttpResponse.json({
          ...decisionCoverageView(String(params["sessionId"])),
          coverage_ready: true,
        }),
      ),
    );
    renderAppAt(PAGE_PATH);

    await screen.findByRole("heading", { name: "Decision coverage" });
    expect(
      await screen.findByText("Report-ready coverage"),
    ).toBeInTheDocument();
    expect(
      screen.queryByText("Coverage gaps remain"),
    ).not.toBeInTheDocument();
  });

  it("lists each uninvestigated high-value question", async () => {
    renderAppAt(PAGE_PATH);

    await screen.findByRole("heading", { name: "Decision coverage" });
    expect(
      await screen.findByText("Which region drives the revenue drop?"),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Does the refund spike track a single channel?"),
    ).toBeInTheDocument();
  });

  it("shows an empty state instead of a blank list when there are no uninvestigated questions", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/decision-coverage", ({ params }) =>
        HttpResponse.json({
          ...decisionCoverageView(String(params["sessionId"])),
          uninvestigated_high_value: [],
        }),
      ),
    );
    renderAppAt(PAGE_PATH);

    await screen.findByRole("heading", { name: "Decision coverage" });
    expect(
      await screen.findByText(
        "No uninvestigated high-value questions — every top card has reached an outcome.",
      ),
    ).toBeInTheDocument();
  });

  it("lists each coverage gap", async () => {
    renderAppAt(PAGE_PATH);

    await screen.findByRole("heading", { name: "Decision coverage" });
    expect(
      await screen.findByText("No terminal outcome for the top-ranked question."),
    ).toBeInTheDocument();
  });

  it("shows an empty state instead of a blank list when there are no gaps", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/decision-coverage", ({ params }) =>
        HttpResponse.json({
          ...decisionCoverageView(String(params["sessionId"])),
          gaps: [],
        }),
      ),
    );
    renderAppAt(PAGE_PATH);

    await screen.findByRole("heading", { name: "Decision coverage" });
    expect(await screen.findByText("No coverage gaps.")).toBeInTheDocument();
  });

  it("shows a plain caption instead of a badge when there are no candidate questions", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/decision-coverage", ({ params }) =>
        HttpResponse.json({
          ...decisionCoverageView(String(params["sessionId"])),
          top_cards_total: 0,
          top_cards_terminal: 0,
          uninvestigated_high_value: [],
          gaps: [],
        }),
      ),
    );
    renderAppAt(PAGE_PATH);

    await screen.findByRole("heading", { name: "Decision coverage" });
    expect(
      await screen.findByText(
        "Decision coverage: no candidate questions found yet.",
      ),
    ).toBeInTheDocument();
    expect(
      screen.queryByText("Coverage gaps remain"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByText("Report-ready coverage"),
    ).not.toBeInTheDocument();
  });
});
