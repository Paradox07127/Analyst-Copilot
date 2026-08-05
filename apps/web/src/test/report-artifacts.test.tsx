import { describe, expect, it, vi } from "vitest";
import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { server } from "./msw/server";
import { SAMPLE_ARTIFACTS } from "./msw/handlers";
import { renderAppAt, renderAppWithRouterAt } from "./render";

describe("Report page with real API", () => {
  it("renders markdown with tables and code plus the status badge", async () => {
    renderAppAt("/projects/p1/sessions/r1/report");

    expect(
      await screen.findByRole("heading", { name: "Demo report" }),
    ).toBeInTheDocument();
    expect(screen.getByText("validated")).toBeInTheDocument();
    /* GFM table rendered as a real table, not literal pipes. */
    expect(screen.getByRole("cell", { name: "North" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Segment" })).toBeInTheDocument();
    expect(
      screen.getByText(/select segment, sum\(revenue\)/),
    ).toBeInTheDocument();
    expect(screen.getByText(/Generated/)).toBeInTheDocument();
  });

  it("links the Quality page a grouped limitation defers to", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/report", ({ params }) =>
        HttpResponse.json({
          session_id: String(params["sessionId"]),
          status: "validated",
          markdown:
            "## Limitations and Risks\n\n- An identifier repeats in 4 places "
            + "across 4 datasets: matches.csv (full list on the Quality page).",
          generated_at: "2026-07-22T12:00:00Z",
        }),
      ),
    );

    renderAppAt("/projects/p1/sessions/r1/report");

    const link = await screen.findByRole("link", { name: "the Quality page" });
    expect(link).toHaveAttribute("href", "/projects/p1/sessions/r1/quality");
  });

  it("renders markdown images as links, never as <img> beacons", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/report", ({ params }) =>
        HttpResponse.json({
          session_id: String(params["sessionId"]),
          status: "validated",
          markdown: "# With image\n\n![p](https://example.com/x.png)",
          generated_at: "2026-07-22T12:00:00Z",
        }),
      ),
    );

    renderAppAt("/projects/p1/sessions/r1/report");

    expect(
      await screen.findByRole("heading", { name: "With image" }),
    ).toBeInTheDocument();
    expect(document.querySelector("img")).toBeNull();
    const link = screen.getByRole("link", { name: "p" });
    expect(link).toHaveAttribute("href", "https://example.com/x.png");
  });

  it("shows an empty state when the run has no report", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/report", ({ params }) =>
        HttpResponse.json({
          session_id: String(params["sessionId"]),
          status: "none",
          markdown: "",
          generated_at: null,
        }),
      ),
    );

    renderAppAt("/projects/p1/sessions/r1/report");

    expect(
      await screen.findByText("No technical report yet"),
    ).toBeInTheDocument();
    expect(screen.getByText(/Generate one to re-read this run’s evidence/)).toBeInTheDocument();
    expect(screen.getByText("none")).toBeInTheDocument();
    /* The empty state is actionable now: generation is a job, not a dead end. */
    expect(
      screen.getByRole("button", { name: "Generate report" }),
    ).toBeEnabled();
  });

  it("shows a typed error with retry when the report fails", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/report", () =>
        HttpResponse.json(
          { error: { code: "session_not_found", message: "Session r1 is missing" } },
          { status: 404 },
        ),
      ),
    );

    renderAppAt("/projects/p1/sessions/r1/report");
    expect(
      await screen.findByText("Request failed (session_not_found)"),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });
});

/* Shaped like exporter.py's own output: hoisted status line, a Data Map line
 * that opens with a backticked artifact id, and a Claim Ledger table. */
const EVIDENCE_REPORT = [
  "# EDA Agent Report",
  "",
  "Status: validated · 4 of 12 claims measured over a whole table, the rest from analysis queries",
  "",
  "## Executive Summary",
  "",
  "- [Indicative] Refunds explain 6 of the 8 points of the Q3 drop.",
  "- [Unverified figures] Cover held at 12.4973 days.",
  "",
  "## Data Map",
  "",
  "- `prof_1` `orders`: 1000 rows, 12 columns, 0 duplicate rows.",
  "",
  "### Claim Ledger",
  "",
  "| Section | Claim | Evidence | Coverage |",
  "|---|---|---|---|",
  "| Executive Summary | claim_refunds | quality_1, chart_1 | ok |",
  "| Executive Summary | claim_cover | prof_1 | gap |",
  "",
].join("\n");

function serveEvidenceReport(overrides: { markdown?: string } = {}) {
  server.use(
    http.get("/api/v1/sessions/:sessionId/report", ({ params }) =>
      HttpResponse.json({
        session_id: String(params["sessionId"]),
        status: "validated",
        markdown: overrides.markdown ?? EVIDENCE_REPORT,
        generated_at: "2026-07-22T12:00:00Z",
      }),
    ),
  );
}

describe("Report reading chrome", () => {
  it("hoists the evidence mix out of the body and explains it", async () => {
    /* Reworded 2026-08-05: the exporter no longer prints a blanket
     * `Gate: degraded`, which every real run wore and which read as a
     * malfunction. It states how much of the report was measured over a whole
     * table instead, and this chrome has to keep hoisting that line. */
    const user = userEvent.setup();
    serveEvidenceReport();
    renderAppAt("/projects/p1/sessions/r1/report");

    expect(
      await screen.findByRole("heading", { name: "EDA Agent Report" }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/4 of 12 claims measured over a whole table/),
    ).toBeInTheDocument();
    /* The prose line is not left behind to say it a second time. */
    expect(screen.queryByText(/Status: validated/)).not.toBeInTheDocument();
    expect(screen.getByText("validated")).toBeInTheDocument();

    await user.click(
      screen.getByRole("button", { name: "What is Evidence mix?" }),
    );
    expect(screen.getByRole("note")).toHaveTextContent(
      /queries the agent planned for this run/,
    );
  });

  it("still explains a stored report's old gate verdict", async () => {
    /* A report's markdown is frozen in its artifact, so every run written
     * before 2026-08-05 keeps `Gate: degraded` forever. The badge must go on
     * meaning what it meant then, not the rejection wording that replaced it. */
    const user = userEvent.setup();
    serveEvidenceReport({
      markdown: EVIDENCE_REPORT.replace(
        /^Status:.*$/m,
        "Status: validated · Gate: degraded",
      ),
    });
    renderAppAt("/projects/p1/sessions/r1/report");

    expect(await screen.findByText("Gate degraded")).toBeInTheDocument();
    await user.click(
      screen.getByRole("button", { name: "What is Evidence gate?" }),
    );
    expect(screen.getByRole("note")).toHaveTextContent(
      /legitimate outcome, not an error/,
    );
  });

  it("gives the report a contents strip with its length", async () => {
    serveEvidenceReport();
    renderAppAt("/projects/p1/sessions/r1/report");

    const nav = await screen.findByRole("navigation", {
      name: "Report sections",
    });
    for (const name of ["Executive Summary", "Data Map", "Claim Ledger"]) {
      expect(within(nav).getByRole("button", { name })).toBeInTheDocument();
    }
    expect(screen.getByText(/2 sections · ≈\d+ words/)).toBeInTheDocument();
    /* The jump targets are still real headings. */
    expect(
      screen.getByRole("heading", { name: "Executive Summary" }),
    ).toBeInTheDocument();
  });

  it("renders the strength qualifier as a label, not a bracketed debug tag", async () => {
    serveEvidenceReport();
    renderAppAt("/projects/p1/sessions/r1/report");

    expect(await screen.findByText("Indicative")).toBeInTheDocument();
    expect(screen.getByText("Unverified figures")).toBeInTheDocument();
    expect(screen.queryByText(/\[Indicative\]/)).not.toBeInTheDocument();
    expect(
      screen.getByText(/Refunds explain 6 of the 8 points/),
    ).toBeInTheDocument();
  });

  it("leaves the generated figures exactly as the exporter wrote them", async () => {
    serveEvidenceReport();
    renderAppAt("/projects/p1/sessions/r1/report");
    /* Reformatting the body would be rewriting the author's numbers. */
    expect(await screen.findByText(/12\.4973 days/)).toBeInTheDocument();
  });

  it("opens a cited artifact beside the report and names the claims citing it", async () => {
    const user = userEvent.setup();
    serveEvidenceReport();
    renderAppAt("/projects/p1/sessions/r1/report");

    await user.click(await screen.findByRole("button", { name: "quality_1" }));

    const panel = await screen.findByRole("complementary", {
      name: "Evidence inspector",
    });
    expect(within(panel).getByText("QualityIssueSet")).toBeInTheDocument();
    expect(within(panel).getByText("Cited by 1 claim")).toBeInTheDocument();
    expect(within(panel).getByText(/claim_refunds/)).toBeInTheDocument();
    expect(
      await within(panel).findByText("Payload of quality_1"),
    ).toBeInTheDocument();
    /* The report is still on screen: the journey never left the page. */
    expect(
      screen.getByRole("heading", { name: "Executive Summary" }),
    ).toBeInTheDocument();
  });

  it("opens decision-story evidence from its source session", async () => {
    const user = userEvent.setup();
    let requestedSession = "";
    server.use(
      http.get("/api/v1/sessions/:sessionId/decision-report", ({ params }) =>
        HttpResponse.json({
          session_id: String(params["sessionId"]),
          status: "available",
          title: "Cross-run decision",
          sections: [],
          limitations: [],
          investigation_gaps: [],
          candidate_decisions: [],
          evidence_refs: [
            {
              artifact_id: "sql_cross_run",
              kind: "table",
              locator: "rows[0]",
              session_id: "r_source",
            },
          ],
          source_finding_artifact_ids: [],
          granted_evidence_artifact_ids: [],
          freshness: { status: "fresh", reasons: [] },
          export_available: true,
        }),
      ),
      http.get(
        "/api/v1/sessions/:sessionId/artifacts/:artifactId",
        ({ params }) => {
          requestedSession = String(params["sessionId"]);
          return HttpResponse.json({
            artifact_id: String(params["artifactId"]),
            type: "SqlResult",
            project_id: "p1",
            session_id: requestedSession,
            created_at: "2026-07-22T12:00:00Z",
            payload: { title: "Cross-run payload" },
            warnings: [],
          });
        },
      ),
    );
    renderAppAt("/projects/p1/sessions/r1/report");

    await user.click(
      await screen.findByRole("button", { name: "sql_cross_run" }),
    );

    const panel = await screen.findByRole("complementary", {
      name: "Evidence inspector",
    });
    expect(requestedSession).toBe("r_source");
    expect(await within(panel).findByText("Cross-run payload")).toBeInTheDocument();
    expect(within(panel).getByRole("link", { name: "Open in Artifacts" })).toHaveAttribute(
      "href",
      "/projects/p1/sessions/r_source/artifacts?artifact=sql_cross_run",
    );
  });

  it("makes an id inspectable wherever the report printed it", async () => {
    serveEvidenceReport();
    renderAppAt("/projects/p1/sessions/r1/report");

    /* Once in the Data Map line, once in the Claim Ledger evidence cell. */
    await screen.findByRole("heading", { name: "Data Map" });
    expect(screen.getAllByRole("button", { name: "prof_1" })).toHaveLength(2);
    /* Claim ids look like artifact ids but are not evidence — not buttons. */
    expect(
      screen.queryByRole("button", { name: "claim_refunds" }),
    ).not.toBeInTheDocument();
  });
});

describe("Artifacts page with real API", () => {
  it("lists artifacts and filters by type server-side", async () => {
    const user = userEvent.setup();
    renderAppAt("/projects/p1/sessions/r1/artifacts");

    expect(await screen.findByText("chart_1")).toBeInTheDocument();
    expect(screen.getByText("quality_1")).toBeInTheDocument();

    await user.selectOptions(
      screen.getByRole("combobox"),
      "ChartSpec",
    );

    expect(await screen.findByText("chart_3")).toBeInTheDocument();
    expect(screen.queryByText("quality_1")).not.toBeInTheDocument();
    expect(screen.queryByText("prof_1")).not.toBeInTheDocument();
    /* Dropdown still offers every type seen in the unfiltered list. */
    expect(
      screen.getByRole("option", { name: "Quality issues" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "Chart" })).toBeInTheDocument();
  });

  it("restores type and loaded-id filters from a shared URL", async () => {
    const user = userEvent.setup();
    const { router } = renderAppWithRouterAt(
      "/projects/p1/sessions/r1/artifacts?type=ChartSpec&q=chart",
    );

    expect(await screen.findByText("chart_3")).toBeInTheDocument();
    expect(screen.getByRole("combobox")).toHaveValue("ChartSpec");
    expect(screen.getByLabelText("Find id")).toHaveValue("chart");
    expect(screen.queryByText("quality_1")).not.toBeInTheDocument();
    expect(
      screen.getByText(/ID search checks only the records loaded below/),
    ).toBeInTheDocument();

    await user.clear(screen.getByLabelText("Find id"));
    expect(router.state.location.search).toBe("?type=ChartSpec");
    await user.selectOptions(screen.getByRole("combobox"), "");
    expect(router.state.location.search).toBe("");
  });

  it("pages with the cursor and appends items", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/artifacts", ({ request }) => {
        const start = Number(
          new URL(request.url).searchParams.get("cursor") ?? "0",
        );
        const items = SAMPLE_ARTIFACTS.slice(start, start + 2);
        return HttpResponse.json({
          items,
          next_cursor: start + 2 < SAMPLE_ARTIFACTS.length ? String(start + 2) : null,
        });
      }),
    );

    const user = userEvent.setup();
    renderAppAt("/projects/p1/sessions/r1/artifacts");

    expect(await screen.findByText("chart_1")).toBeInTheDocument();
    expect(screen.queryByText("prof_1")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Load more" }));
    expect(await screen.findByText("prof_1")).toBeInTheDocument();
    expect(screen.getByText("chart_1")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Load more" }));
    expect(await screen.findByText("chart_3")).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Load more" }),
    ).not.toBeInTheDocument();
  });

  it("surfaces the caveats recorded on an artifact", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/artifacts/:artifactId", ({ params }) =>
        HttpResponse.json({
          artifact_id: String(params["artifactId"]),
          type: "DatasetProfile",
          project_id: "p1",
          session_id: "r1",
          created_at: "2026-07-22T12:00:00Z",
          payload: { title: "Payload of chart_1" },
          warnings: ["Profile sampled to 10k rows."],
        }),
      ),
    );

    const user = userEvent.setup();
    renderAppAt("/projects/p1/sessions/r1/artifacts");
    await user.click(await screen.findByRole("button", { name: /chart_1/ }));

    expect(await screen.findByText("1 warning")).toBeInTheDocument();
    expect(
      screen.getByText("Profile sampled to 10k rows."),
    ).toBeInTheDocument();
  });

  it("finds a cited id among the loaded rows", async () => {
    const user = userEvent.setup();
    renderAppAt("/projects/p1/sessions/r1/artifacts");
    await screen.findByText("chart_1");

    /* clear() focuses; a bare click does not move focus under jsdom. */
    await user.clear(screen.getByLabelText("Find id"));
    await user.keyboard("quality");
    expect(screen.getByText("quality_1")).toBeInTheDocument();
    expect(screen.queryByText("chart_1")).not.toBeInTheDocument();
    expect(screen.getByText("1 of 5 loaded")).toBeInTheDocument();

    await user.keyboard("-nope");
    expect(screen.getByText("No id matches")).toBeInTheDocument();
  });

  it("expands an artifact to fetch and show its payload", async () => {
    const user = userEvent.setup();
    renderAppAt("/projects/p1/sessions/r1/artifacts");

    await user.click(await screen.findByRole("button", { name: /chart_1/ }));
    expect(
      await screen.findByText(/"Payload of chart_1"/),
    ).toBeInTheDocument();

    /* Clicking again collapses the payload. */
    await user.click(screen.getByRole("button", { name: /chart_1/ }));
    expect(screen.queryByText(/"Payload of chart_1"/)).not.toBeInTheDocument();
  });
});

describe("Artifacts deep link (?artifact=)", () => {
  it("opens and reveals the linked row when it is on the page", async () => {
    const scrollIntoView = vi.fn();
    /* jsdom has no scrollIntoView; the page calls it optionally. */
    Object.defineProperty(Element.prototype, "scrollIntoView", {
      value: scrollIntoView,
      configurable: true,
      writable: true,
    });

    renderAppAt("/projects/p1/sessions/r1/artifacts?artifact=quality_1");

    /* Expanded without a click, and its payload fetched. */
    expect(
      await screen.findByText(/"Payload of quality_1"/),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /quality_1/ }),
    ).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByRole("button", { name: /chart_1/ })).toHaveAttribute(
      "aria-expanded",
      "false",
    );
    expect(scrollIntoView).toHaveBeenCalled();
  });

  it("fetches by id when the linked artifact is not among the loaded rows", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/artifacts/:artifactId", ({ params }) =>
        HttpResponse.json({
          artifact_id: String(params["artifactId"]),
          type: "RelationshipValidationSet",
          project_id: "p1",
          session_id: "r1",
          created_at: "2026-07-22T12:00:00Z",
          payload: { title: "Payload of relval_9" },
          warnings: [],
        }),
      ),
    );

    renderAppAt("/projects/p1/sessions/r1/artifacts?artifact=relval_9");

    const panel = await screen.findByRole("region", {
      name: "Linked artifact",
    });
    expect(
      await within(panel).findByText(/"Payload of relval_9"/),
    ).toBeInTheDocument();
    expect(
      within(panel).getByText(/Not on the loaded page of this session/),
    ).toBeInTheDocument();
  });

  it("says so when the linked artifact belongs to another session", async () => {
    const user = userEvent.setup();
    server.use(
      http.get("/api/v1/sessions/:sessionId/artifacts/:artifactId", ({ params }) =>
        HttpResponse.json({
          artifact_id: String(params["artifactId"]),
          type: "SqlResult",
          project_id: "p1",
          session_id: "qsess_other",
          created_at: "2026-07-22T12:00:00Z",
          payload: { title: "elsewhere" },
          warnings: [],
        }),
      ),
    );

    renderAppAt("/projects/p1/sessions/r1/artifacts?artifact=sqlres_1");

    const panel = await screen.findByRole("region", {
      name: "Linked artifact",
    });
    expect(
      await within(panel).findByText(/belongs to session qsess_other/),
    ).toBeInTheDocument();

    await user.click(within(panel).getByRole("button", { name: "Dismiss" }));
    expect(
      screen.queryByRole("region", { name: "Linked artifact" }),
    ).not.toBeInTheDocument();
  });

  it("surfaces a broken evidence link instead of failing silently", async () => {
    renderAppAt("/projects/p1/sessions/r1/artifacts?artifact=missing_9");

    const panel = await screen.findByRole("region", {
      name: "Linked artifact",
    });
    expect(
      await within(panel).findByText("Request failed (artifact_not_found)"),
    ).toBeInTheDocument();
  });
});
