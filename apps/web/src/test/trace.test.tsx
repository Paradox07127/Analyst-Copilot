import { describe, expect, it } from "vitest";
import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { defaultSettings, llmDebugPage, runDebugView } from "./msw/handlers";
import { server } from "./msw/server";
import { renderAppAt } from "./render";
import { objectUrls } from "./setup";

const PATH = "/projects/p1/sessions/r1/trace";

function enableDevMode() {
  server.use(
    http.get("/api/v1/settings", () =>
      HttpResponse.json({ ...defaultSettings(), dev_mode: true }),
    ),
  );
}

async function findInspector() {
  const heading = await screen.findByRole("heading", {
    name: "Developer inspector",
  });
  const inspector = within(heading.closest("section") as HTMLElement);
  await waitFor(() => {
    expect(
      inspector.queryByRole("button", {
        name: "Open developer inspector",
      }) ??
        inspector.queryByRole("button", {
          name: "Close developer inspector",
        }) ??
        inspector.queryByRole("link", {
          name: "Open Settings",
        }),
    ).not.toBeNull();
  });
  const open = inspector.queryByRole("button", {
    name: "Open developer inspector",
  });
  if (open) {
    fireEvent.click(open);
    await inspector.findByRole("button", {
      name: "Close developer inspector",
    });
  }
  return inspector;
}

describe("Trace & cost page", () => {
  it("restores the event filter from a shared URL", async () => {
    renderAppAt(`${PATH}?type=step_completed`);

    expect(await screen.findByText("1 of 1 shown")).toBeInTheDocument();
    expect(screen.getByLabelText("Type")).toHaveValue("step_completed");
    expect(screen.queryByText("llm_usage")).not.toBeInTheDocument();
  });

  it("renders cost cards, stage bars, and the event feed", async () => {
    renderAppAt(PATH);

    // Scope to main: the TopBar shows the same run cost, so an unscoped query
    // would match twice. Await main itself — the route is lazy.
    const main = await screen.findByRole("main");
    expect(await within(main).findByText("$0.0122")).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "Trace & cost" }),
    ).toBeInTheDocument();
    /* Scoped: the request breakdown lists the same call count per model, so an
     * unscoped "22" now matches several nodes. */
    const callsKpi = screen.getByText("LLM calls").closest("div") as HTMLElement;
    expect(within(callsKpi).getByText("22")).toBeInTheDocument();
    expect(screen.getByText("68,868")).toBeInTheDocument();
    expect(screen.getByText("56.0%")).toBeInTheDocument();
    expect(screen.getByText("18,816 read · 0 written")).toBeInTheDocument();
    expect(screen.getByText("3m 36s")).toBeInTheDocument();
    expect(screen.getByText("194 trace event(s)")).toBeInTheDocument();
    expect(
      screen.getByText("from the session's SessionMetrics artifact"),
    ).toBeInTheDocument();

    /* "Findings recorded" disambiguates from the Findings page's validated-only
     * count (run_metrics.py:483 also tallies findings embedded in question
     * execution results), via a distinct label and a hover explanation. */
    const findingsKpi = screen
      .getByText("Findings recorded")
      .closest("div") as HTMLElement;
    expect(within(findingsKpi).getByText("3")).toBeInTheDocument();
    expect(within(findingsKpi).getByText("not the Findings page count")).toBeInTheDocument();
    expect(findingsKpi.getAttribute("title")).toContain("Findings page");

    /* A cost total that covers only some calls must say so on the card, not
     * only in the payload. */
    const costKpi = screen.getByText("Estimated cost").closest("div") as HTMLElement;
    expect(
      within(costKpi).getByText("complete_estimate · 22/22 calls priced"),
    ).toBeInTheDocument();

    const breakdown = screen
      .getByText("LLM request breakdown")
      .closest("section") as HTMLElement;
    expect(within(breakdown).getByText("offline-stub")).toBeInTheDocument();
    expect(within(breakdown).getByText("structured")).toBeInTheDocument();

    /* Stage bars are ordered longest-first. */
    const stages = screen.getByText("Stage duration").closest("section");
    const stageNames = within(stages as HTMLElement)
      .getAllByText(/profile_dataset|export_agentic_report/)
      .map((node) => node.textContent);
    expect(stageNames).toEqual(["export_agentic_report", "profile_dataset"]);

    expect(screen.getByText("llm_usage")).toBeInTheDocument();
    expect(screen.getByText("3 of 3 shown")).toBeInTheDocument();
  });

  it("renders a handled React failure as failure_recorded", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/trace", ({ params }) =>
        HttpResponse.json({
          session_id: String(params["sessionId"]),
          items: [
            {
              event_id: 195,
              event_type: "failure_recorded",
              name: "mutation",
              summary: {
                source: "react",
                error_code: "conflict",
                operation: "mutation",
              },
            },
          ],
          event_types: { failure_recorded: 1 },
          total: 1,
        }),
      ),
    );
    renderAppAt(PATH);

    expect(await screen.findByText("failure_recorded")).toBeInTheDocument();
    expect(screen.getByText("mutation")).toBeInTheDocument();
    expect(screen.getByText("3 field(s)")).toBeInTheDocument();
  });

  it("filters events by type and requests that type from the server", async () => {
    const requested: (string | null)[] = [];
    server.events.on("request:start", ({ request }) => {
      const url = new URL(request.url);
      if (url.pathname.endsWith("/trace")) {
        requested.push(url.searchParams.get("type"));
      }
    });
    const user = userEvent.setup();
    renderAppAt(PATH);
    await screen.findByText("llm_usage");

    await user.selectOptions(screen.getByLabelText("Type"), "step_completed");
    expect(await screen.findByText("1 of 1 shown")).toBeInTheDocument();
    expect(screen.queryByText("llm_usage")).not.toBeInTheDocument();
    expect(requested).toContain("step_completed");
    /* The histogram still lists every type so the filter can be cleared. */
    expect(
      within(screen.getByLabelText<HTMLSelectElement>("Type")).getByText(
        "llm_usage (1)",
      ),
    ).toBeInTheDocument();
  });

  it("expands an event summary as JSON", async () => {
    const user = userEvent.setup();
    renderAppAt(PATH);
    const disclosure = await screen.findByText("2 field(s)");

    const details = disclosure.closest("details") as HTMLElement;
    expect(details).not.toHaveAttribute("open");
    await user.click(disclosure);
    expect(details).toHaveAttribute("open");
    expect(within(details).getByText(/"total_tokens": 1000/)).toBeInTheDocument();
  });

  it("keeps the cost cards when the event feed fails", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/trace", () =>
        HttpResponse.json(
          { error: { code: "invalid_cursor", message: "Invalid cursor." } },
          { status: 400 },
        ),
      ),
    );
    renderAppAt(PATH);
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Request failed (invalid_cursor)",
    );
    expect(
      within(screen.getByRole("main")).getByText("$0.0122"),
    ).toBeInTheDocument();
  });

  it("labels a recomputed rollup as aggregated", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/metrics", ({ params }) =>
        HttpResponse.json({
          session_id: String(params["sessionId"]),
          source: "aggregated",
          llm_calls: 0,
          total_tokens: 0,
          est_cost_usd: null,
          duration_seconds: 0,
          event_count: 0,
          trace_status: "unverifiable",
          steps: [],
        }),
      ),
    );
    renderAppAt(PATH);
    expect(
      await screen.findByText(/aggregated from trace events/),
    ).toBeInTheDocument();
    expect(screen.getByText("n/a")).toBeInTheDocument();
  });

  it("summarizes only non-zero quality metrics with a drilldown", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/metrics", ({ params }) =>
        HttpResponse.json({
          session_id: String(params["sessionId"]),
          source: "artifact",
          llm_calls: 0,
          total_tokens: 0,
          duration_seconds: 0,
          event_count: 0,
          question_proposals_dropped: 2,
          question_abstained: 1,
          semantic_degraded_claims: 3,
          numeric_unverified_claims: 0,
          evidence_interleave_granted: 2,
          domain_metric_questions: 4,
          result_contract_failures: { invalid_payload: 2 },
          publication_blocked: true,
          steps: [],
        }),
      ),
    );
    const user = userEvent.setup();
    renderAppAt(PATH);

    const summary = await screen.findByText("7 non-zero quality signals");
    expect(screen.queryByText("Numeric figures unverified")).not.toBeInTheDocument();

    await user.click(summary);
    expect(screen.getByText("Question proposals dropped")).toBeInTheDocument();
    expect(screen.getByText("Semantic claims degraded")).toBeInTheDocument();
    expect(screen.getByText("Evidence requests granted")).toBeInTheDocument();
    expect(screen.getByText("Domain metric questions")).toBeInTheDocument();
    expect(screen.getByText("Result contract failures")).toBeInTheDocument();
    expect(screen.getByText("Publication blocked")).toBeInTheDocument();
  });
});

describe("Trace page — developer inspector gate", () => {
  it("stays off and points to Settings when dev mode is disabled", async () => {
    renderAppAt(PATH);
    const dev = await findInspector();
    expect(
      await dev.findByText(/Developer inspector is off/),
    ).toBeInTheDocument();
    expect(dev.getByRole("link", { name: "Open Settings" })).toHaveAttribute(
      "href",
      "/settings?section=about",
    );
    expect(dev.queryByText(/Timeline \(/)).not.toBeInTheDocument();
  });

  it("restores the developer inspector only when explicitly requested", async () => {
    enableDevMode();
    renderAppAt(`${PATH}?view=developer`);
    const dev = await findInspector();

    expect(
      dev.getByRole("button", { name: "Close developer inspector" }),
    ).toHaveAttribute("aria-expanded", "true");
    expect(await dev.findByText("Timeline (1)")).toBeInTheDocument();
  });
});

describe("Trace page — debug inspector tables", () => {
  it("restores the expanded debug panes from the URL", async () => {
    enableDevMode();
    renderAppAt(`${PATH}?debug=artifacts,llm-calls`);
    const dev = await findInspector();

    expect(
      (await dev.findByText("Artifacts (1)")).closest("details"),
    ).toHaveAttribute("open");
    expect(dev.getByText("LLM calls (1)").closest("details")).toHaveAttribute(
      "open",
    );
    expect(dev.getByText("Timeline (1)").closest("details")).not.toHaveAttribute(
      "open",
    );
    expect(dev.getByText("Errors (1)").closest("details")).not.toHaveAttribute(
      "open",
    );
  });

  it("persists an explicitly collapsed debug inspector instead of restoring defaults", async () => {
    enableDevMode();
    renderAppAt(`${PATH}?debug=none`);
    const dev = await findInspector();

    for (const title of [
      "Timeline (1)",
      "LLM calls (1)",
      "Tool calls (1)",
      "Errors (1)",
      "Artifacts (1)",
    ]) {
      expect((await dev.findByText(title)).closest("details")).not.toHaveAttribute(
        "open",
      );
    }
  });

  it("expands only Timeline and Errors by default", async () => {
    enableDevMode();
    renderAppAt(PATH);
    const dev = await findInspector();

    const timeline = await dev.findByText("Timeline (1)");
    expect(timeline.closest("details")).toHaveAttribute("open");
    const errors = dev.getByText("Errors (1)");
    expect(errors.closest("details")).toHaveAttribute("open");

    const llmCalls = dev.getByText("LLM calls (1)");
    expect(llmCalls.closest("details")).not.toHaveAttribute("open");
    const toolCalls = dev.getByText("Tool calls (1)");
    expect(toolCalls.closest("details")).not.toHaveAttribute("open");
    const artifacts = dev.getByText("Artifacts (1)");
    expect(artifacts.closest("details")).not.toHaveAttribute("open");
  });

  it("shows the run summary and report-quality tiles from useSessionDebug", async () => {
    enableDevMode();
    renderAppAt(PATH);
    const dev = await findInspector();

    await dev.findByText("Timeline (1)");
    expect(dev.getByText("68,868")).toBeInTheDocument(); // summary.total_tokens
    expect(dev.getByText("$0.0122")).toBeInTheDocument(); // summary.estimated_cost_usd
    expect(dev.getByText("90%")).toBeInTheDocument(); // report_quality.section_coverage
    expect(dev.getByText("1:12000, 2:9000")).toBeInTheDocument(); // prompt_tokens_by_attempt
  });

  it("renders 'No rows.' for an empty table", async () => {
    enableDevMode();
    server.use(
      http.get("/api/v1/sessions/:sessionId/debug", ({ params }) =>
        HttpResponse.json({
          ...runDebugView(String(params["sessionId"])),
          errors: [],
        }),
      ),
    );
    renderAppAt(PATH);
    const dev = await findInspector();

    const errors = await dev.findByText("Errors (0)");
    const details = errors.closest("details") as HTMLElement;
    expect(within(details).getByText("No rows.")).toBeInTheDocument();
  });
});

describe("Trace page — LLM payload forensics", () => {
  it("expands a call to show its payload and response", async () => {
    enableDevMode();
    const user = userEvent.setup();
    renderAppAt(PATH);
    const dev = await findInspector();

    const item = await dev.findByText(/1\. draft_report · tok=12000→900/);
    await user.click(item);
    const details = item.closest("details") as HTMLElement;
    expect(
      within(details).getByText('{"system": "You are ...", "call": 1}'),
    ).toBeInTheDocument();
    expect(within(details).getByText('{"sections": [...]}')).toBeInTheDocument();
  });

  it("shows '(none)' when response_preview is empty", async () => {
    enableDevMode();
    server.use(
      http.get("/api/v1/sessions/:sessionId/debug/llm-calls", ({ request }) => {
        const cursor = new URL(request.url).searchParams.get("cursor");
        const page = llmDebugPage(cursor);
        return HttpResponse.json({
          ...page,
          items: page.items.map((item) => ({ ...item, response_preview: "" })),
        });
      }),
    );
    const user = userEvent.setup();
    renderAppAt(PATH);
    const dev = await findInspector();

    const item = await dev.findByText(/1\. draft_report · tok=12000→900/);
    await user.click(item);
    const details = item.closest("details") as HTMLElement;
    expect(within(details).getByText("(none)")).toBeInTheDocument();
  });

  /* trace_ui._render_llm_debug_details appends the status only when the call
   * failed, so a clean run's list stays scannable. */
  it("shows the status only for calls that did not succeed", async () => {
    enableDevMode();
    server.use(
      http.get("/api/v1/sessions/:sessionId/debug/llm-calls", ({ request }) => {
        const cursor = new URL(request.url).searchParams.get("cursor");
        const page = llmDebugPage(cursor);
        const items = page.items.map((item, index) =>
          index === 0 ? { ...item, status: "schema_error" } : item,
        );
        return HttpResponse.json({ ...page, items });
      }),
    );
    renderAppAt(PATH);
    const dev = await findInspector();

    expect(await dev.findByText(/1\..*· schema_error$/)).toBeInTheDocument();
    expect(dev.queryByText(/2\..*· success$/)).not.toBeInTheDocument();
    expect(await dev.findByText(/2\. draft_report · tok=12000→900/)).toBeInTheDocument();
  });

  it("loads the next page of LLM calls on demand", async () => {
    enableDevMode();
    const user = userEvent.setup();
    renderAppAt(PATH);
    const dev = await findInspector();

    await dev.findByText(/1\. draft_report · tok=12000→900/);
    expect(dev.queryByText(/3\. draft_report · tok=12000→900/)).not.toBeInTheDocument();

    await user.click(dev.getByRole("button", { name: "Load more" }));
    expect(await dev.findByText(/3\. draft_report · tok=12000→900/)).toBeInTheDocument();
    expect(
      dev.queryByRole("button", { name: "Load more" }),
    ).not.toBeInTheDocument();
  });
});

describe("Trace page — debug log download", () => {
  it("downloads the debug.jsonl file", async () => {
    enableDevMode();
    const user = userEvent.setup();
    renderAppAt(PATH);
    const dev = await findInspector();

    await user.click(
      await dev.findByRole("button", { name: "Download debug log (JSONL)" }),
    );
    expect(objectUrls.created).toHaveLength(1);
  });

  it("shows a friendly message instead of throwing when there is no debug log", async () => {
    enableDevMode();
    server.use(
      http.get("/api/v1/sessions/:sessionId/debug/log", () =>
        HttpResponse.json(
          {
            error: {
              code: "debug_log_not_found",
              message: "No debug log recorded for this session.",
            },
          },
          { status: 404 },
        ),
      ),
    );
    const user = userEvent.setup();
    renderAppAt(PATH);
    const dev = await findInspector();

    await user.click(
      await dev.findByRole("button", { name: "Download debug log (JSONL)" }),
    );
    expect(
      await dev.findByText("This session has no debug log."),
    ).toBeInTheDocument();
    expect(dev.queryByText(/Request failed/)).not.toBeInTheDocument();
    expect(objectUrls.created).toHaveLength(0);
  });
});
