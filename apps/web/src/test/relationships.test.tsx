import { describe, expect, it } from "vitest";
import { act, fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient } from "@tanstack/react-query";
import { createMemoryRouter, RouterProvider } from "react-router";
import { http, HttpResponse } from "msw";
import { server } from "./msw/server";
import { renderAppAt } from "./render";
import { AppProviders } from "../app/providers";
import { routes } from "../app/router";
import { relationshipGraph } from "./msw/handlers";

const PAGE_PATH = "/projects/p1/sessions/r1/relationships";

/* renderAppAt hides its router, and switching run is exactly what the filter
 * reset has to survive — so this variant hands the router back. */
function renderAppWithRouter(path: string) {
  const router = createMemoryRouter(routes, { initialEntries: [path] });
  render(
    <AppProviders
      client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}
    >
      <RouterProvider router={router} />
    </AppProviders>,
  );
  return router;
}
/* A row is no longer a dotted path string: it renders table + column on each
 * side, and carries the flat form as its accessible name. */
const CANDIDATE_NAME =
  /^sample\.csv\.code → lookup\.csv\.code, high confidence, candidate$/;
const VALIDATED_NAME =
  /^sample\.csv\.lookup_id → lookup\.csv\.lookup_id, high confidence, validated$/;

/* Distinct column names per row: the list renders table + column, so the rows
 * have to differ there rather than in the old dotted label. */
function scoredEdges() {
  const base = relationshipGraph("r1").edges![0]!;
  return [
    {
      ...base,
      relationship_id: "rel_high",
      left_columns: ["high_left"],
      right_columns: ["high_right"],
      confidence: "high",
      ensemble_score: 0.9,
    },
    {
      ...base,
      relationship_id: "rel_medium",
      left_columns: ["medium_left"],
      right_columns: ["medium_right"],
      confidence: "medium",
      ensemble_score: 0.5,
    },
    {
      ...base,
      relationship_id: "rel_low",
      left_columns: ["low_left"],
      right_columns: ["low_right"],
      confidence: "low",
      ensemble_score: 0.2,
    },
  ];
}

async function waitForRelationshipWorkspace() {
  await screen.findByText("Column candidates");
}

async function openInspector(name: RegExp) {
  const user = userEvent.setup();
  renderAppAt(PAGE_PATH);
  await waitForRelationshipWorkspace();
  await user.click(screen.getByRole("button", { name }));
  return user;
}

describe("Relationships page", () => {
  it("keeps pair, candidate, and confirmed-join units explicit", async () => {
    renderAppAt(PAGE_PATH);
    await waitForRelationshipWorkspace();

    expect(screen.getByText("Datasets").parentElement).toHaveTextContent("2");
    expect(screen.getByText("Dataset pairs").parentElement).toHaveTextContent("1");
    expect(screen.getByText("Column candidates").parentElement).toHaveTextContent("2");
    expect(screen.getByText("Confirmed joins").parentElement).toHaveTextContent("0");
    expect(screen.getByText("Discovery").parentElement).toHaveTextContent(
      "Complete",
    );
  });

  it("defaults large table sets to the matrix and keeps view state in the URL", async () => {
    const base = relationshipGraph("r1");
    const seed = base.nodes![0]!;
    server.use(
      http.get("/api/v1/sessions/:sessionId/relationships", () =>
        HttpResponse.json({
          ...base,
          nodes: [
            ...base.nodes!,
            ...Array.from({ length: 7 }, (_, index) => ({
              ...seed,
              dataset_id: `extra_${index}`,
              name: `extra_${index}.csv`,
            })),
          ],
        }),
      ),
    );
    const user = userEvent.setup();
    const router = renderAppWithRouter(PAGE_PATH);
    await waitForRelationshipWorkspace();

    expect(
      screen.getByRole("table", { name: "Table relationship matrix" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Matrix" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );

    await user.click(screen.getByRole("button", { name: "Overview" }));
    expect(
      document.querySelectorAll(".relationships-flow .react-flow__node"),
    ).toHaveLength(9);
    expect(new URLSearchParams(router.state.location.search).get("view")).toBe(
      "overview",
    );

    await user.click(screen.getByRole("button", { name: "Neighborhood" }));
    expect(document.querySelector(".relationships-flow")).not.toBeNull();
    expect(new URLSearchParams(router.state.location.search).get("view")).toBe(
      "neighborhood",
    );
  });

  it("shows pair-level explanation before column-level validation", async () => {
    const user = userEvent.setup();
    const router = renderAppWithRouter(
      `${PAGE_PATH}?pair=${encodeURIComponent("sample→lookup")}`,
    );
    await waitForRelationshipWorkspace();

    expect(screen.getByText("Many → one lookup")).toBeInTheDocument();
    expect(screen.getByText("Why this pair surfaced")).toBeInTheDocument();
    expect(screen.getByText("Target key uniqueness")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Candidates · 2" }));
    expect(screen.getByRole("button", { name: CANDIDATE_NAME })).toBeInTheDocument();
    expect(new URLSearchParams(router.state.location.search).get("detail")).toBe(
      "candidates",
    );
  });

  it("restores score, confidence, pair, and edge context from the URL", async () => {
    const user = userEvent.setup();
    renderAppAt(
      `${PAGE_PATH}?score=0.9&confidence=high&pair=${encodeURIComponent("sample→lookup")}&edge=rel_candidate`,
    );
    await waitForRelationshipWorkspace();

    expect(
      screen.getByRole("list", { name: "Relationship review progress" }),
    ).toBeInTheDocument();
    expect(screen.getByText("relcand_1")).toBeInTheDocument();
    await user.click(
      screen.getByRole("button", { name: "Back to candidates" }),
    );
    expect(screen.getByLabelText(/Minimum score/)).toHaveValue("0.9");
    expect(screen.getByLabelText("high")).toBeChecked();
    expect(screen.getByLabelText("medium")).not.toBeChecked();
    expect(screen.getByRole("button", { name: "Show all pairs" })).toBeInTheDocument();
  });

  it("updates pair and edge route context atomically", async () => {
    const user = userEvent.setup();
    const router = renderAppWithRouter(PAGE_PATH);
    await waitForRelationshipWorkspace();

    await user.click(screen.getByRole("button", { name: CANDIDATE_NAME }));
    let params = new URLSearchParams(router.state.location.search);
    expect(params.get("pair")).toBe("sample→lookup");
    expect(params.get("edge")).toBe("rel_candidate");

    await user.click(screen.getByRole("button", { name: "Show all pairs" }));
    params = new URLSearchParams(router.state.location.search);
    expect(params.has("pair")).toBe(false);
    expect(params.has("edge")).toBe(false);
    expect(
      screen.queryByText("relcand_1"),
    ).not.toBeInTheDocument();
  });

  it("lists edges with their three states and the legend", async () => {
    renderAppAt(PAGE_PATH);
    await waitForRelationshipWorkspace();

    const inspector = screen.getByRole("complementary", {
      name: "Relationship inspector",
    });
    const candidateRow = within(inspector).getByRole("button", {
      name: CANDIDATE_NAME,
    });
    expect(
      within(inspector).getByRole("button", { name: VALIDATED_NAME }),
    ).toBeInTheDocument();
    expect(within(inspector).getByText("candidate")).toBeInTheDocument();
    expect(within(inspector).getByText("validated")).toBeInTheDocument();

    /* Table and column on each side, not the dotted path string. */
    expect(within(candidateRow).getAllByText("code")).toHaveLength(2);
    expect(within(candidateRow).getByText("sample.csv")).toBeInTheDocument();
    expect(within(candidateRow).getByText("lookup.csv")).toBeInTheDocument();
    expect(
      within(inspector).queryByText("sample.csv.code -> lookup.csv.code"),
    ).not.toBeInTheDocument();

    /* Candidates are grouped under the table pair they join. */
    expect(
      within(inspector).getByRole("button", {
        name: /sample\.csv ↔ lookup\.csv/,
      }),
    ).toBeInTheDocument();

    /* Legend explains what the edge styles on the canvas mean. */
    expect(screen.getByText("scored only — dashed")).toBeInTheDocument();
    expect(screen.getByText("DuckDB verified")).toBeInTheDocument();
  });

  it("supports keyboard move cancel, commit, undo, and edge selection on the canvas", async () => {
    const user = userEvent.setup();
    renderAppAt(PAGE_PATH);
    await waitForRelationshipWorkspace();

    /* jsdom does not lay out handles, so React Flow cannot materialise its SVG
     * edges. Add the same focusable edge target React Flow renders in-browser
     * and exercise the delegated canvas keyboard handler. */
    const flow = document.querySelector<HTMLElement>(".relationships-flow");
    expect(flow).not.toBeNull();
    const edge = document.createElement("div");
    edge.className = "react-flow__edge";
    edge.dataset.id = "sample→lookup";
    edge.tabIndex = 0;
    flow!.append(edge);
    edge.focus();
    await user.keyboard("{Enter}");
    const showAll = await screen.findByRole("button", { name: "Show all pairs" });
    await user.click(showAll);

    const node = document.querySelector<HTMLElement>(
      ".relationships-flow .react-flow__node",
    );
    expect(node).not.toBeNull();
    node!.focus();
    expect(node).toHaveFocus();

    await user.keyboard("{Enter}{ArrowRight}{Escape}");
    expect(screen.getByText("Dataset move cancelled.")).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Undo graph move" }),
    ).not.toBeInTheDocument();

    const restoredNode = document.querySelector<HTMLElement>(
      ".relationships-flow .react-flow__node",
    );
    expect(restoredNode).not.toBeNull();
    const restoredNodeId = restoredNode!.dataset.id;
    restoredNode!.focus();
    await user.keyboard("{Enter}");
    expect(
      screen.getByText(
        "Dataset grabbed. Use arrow keys to move it, Enter to confirm, Escape to cancel.",
      ),
    ).toBeInTheDocument();
    const grabbedNode = document.querySelector<HTMLElement>(
      `.relationships-flow .react-flow__node[data-id="${restoredNodeId}"]`,
    );
    expect(grabbedNode).not.toBeNull();
    grabbedNode!.focus();
    await user.keyboard("{ArrowDown}");
    expect(
      screen.getByText(
        "Dataset moved down. Press Enter to confirm or Escape to cancel.",
      ),
    ).toBeInTheDocument();
    const movedNode = document.querySelector<HTMLElement>(
      `.relationships-flow .react-flow__node[data-id="${restoredNodeId}"]`,
    );
    expect(movedNode).not.toBeNull();
    movedNode!.focus();
    await user.keyboard("{Enter}");
    const undo = await screen.findByRole("button", { name: "Undo graph move" });
    expect(
      screen.getByText("Dataset position saved. Undo is available."),
    ).toBeInTheDocument();
    await user.click(undo);
    expect(screen.getByText("Last dataset move undone.")).toBeInTheDocument();
  });

  it("shows the evidence of a validated edge in the Inspector", async () => {
    await openInspector(VALIDATED_NAME);

    expect(await screen.findByText("Cardinality")).toBeInTheDocument();
    expect(screen.getByText("many_to_one")).toBeInTheDocument();
    expect(screen.getByText("Orphan rate left")).toBeInTheDocument();
    expect(screen.getAllByText("0.0%").length).toBeGreaterThan(0);
    /* Evidence must be traceable back to the artifact that produced it. */
    expect(screen.getByRole("link", { name: "relval_1" })).toHaveAttribute(
      "href",
      "/projects/p1/sessions/r1/artifacts?artifact=relval_1",
    );
    expect(screen.getByText("Verification SQL")).toBeInTheDocument();
  });

  it("shows only the next eligible review action", async () => {
    await openInspector(CANDIDATE_NAME);

    expect(
      await screen.findByRole("button", {
        name: "Validate against full tables",
      }),
    ).toBeEnabled();
    expect(
      screen.queryByRole("button", { name: "Use as project join" }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Remove from project joins" }),
    ).not.toBeInTheDocument();
  });

  it("prepares, confirms and starts a validation job", async () => {
    let idempotencyKey: string | null = null;
    let validateBody: Record<string, unknown> | null = null;
    server.use(
      http.post(
        "/api/v1/sessions/:sessionId/relationships/:relationshipId/validate",
        async ({ request }) => {
          idempotencyKey = request.headers.get("Idempotency-Key");
          validateBody = (await request.json()) as Record<string, unknown>;
          return HttpResponse.json(
            {
              session_id: "r1",
              relationship_id: "rel_candidate",
              execution_session_id: "rvsess_new_1",
              job: {
                job_id: "job_rel_9",
                session_id: "rvsess_new_1",
                status: "queued",
                events_url: "/api/v1/jobs/job_rel_9/events",
              },
            },
            { status: 201 },
          );
        },
      ),
    );

    const user = await openInspector(CANDIDATE_NAME);
    await user.click(
      await screen.findByRole("button", {
        name: "Validate against full tables",
      }),
    );

    const dialog = await screen.findByRole("alertdialog", {
      name: "Confirm relationship validation",
    });
    expect(
      within(dialog).getByText(/Both source tables are read in full/),
    ).toBeInTheDocument();

    await user.click(
      within(dialog).getByRole("button", { name: "Confirm & validate" }),
    );

    expect(
      await screen.findByText(/Tracking job_rel_9 · session rvsess_new_1/),
    ).toBeInTheDocument();
    expect(idempotencyKey).toMatch(/^[0-9a-f-]{36}$/);
    expect(validateBody).toMatchObject({
      action_hash: "a".repeat(64),
      approval_token: "c".repeat(32),
    });
  });

  it("confirms a validated edge and reflects the new join state", async () => {
    const user = await openInspector(VALIDATED_NAME);
    const confirm = await screen.findByRole("button", {
      name: "Use as project join",
    });
    expect(confirm).toBeEnabled();

    server.use(
      http.get("/api/v1/sessions/:sessionId/relationships", ({ params }) => {
        const graph = relationshipGraph(String(params["sessionId"]));
        return HttpResponse.json({
          ...graph,
          edges: (graph.edges ?? []).map((edge) =>
            edge.relationship_id === "rel_validated"
              ? {
                  ...edge,
                  state: "confirmed",
                  join_status: "confirmed",
                  can_confirm: false,
                }
              : edge,
          ),
        });
      }),
    );
    await user.click(confirm);

    expect(await screen.findByText("join: confirmed")).toBeInTheDocument();
  });

  it("explains a refused confirmation instead of failing silently", async () => {
    server.use(
      http.post(
        "/api/v1/sessions/:sessionId/relationships/:relationshipId/confirm",
        () =>
          HttpResponse.json(
            {
              error: {
                code: "join_not_confirmable",
                message: "Join is many-to-many and cannot be confirmed.",
              },
            },
            { status: 409 },
          ),
      ),
    );

    const user = await openInspector(VALIDATED_NAME);
    await user.click(
      await screen.findByRole("button", { name: "Use as project join" }),
    );

    expect(
      await screen.findByText("Request failed (join_not_confirmable)"),
    ).toBeInTheDocument();
  });

  it("filters the candidate list by minimum score and confidence", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/relationships", ({ params }) =>
        HttpResponse.json({
          ...relationshipGraph(String(params["sessionId"])),
          edges: scoredEdges(),
        }),
      ),
    );
    const user = userEvent.setup();
    renderAppAt(PAGE_PATH);
    await waitForRelationshipWorkspace();
    const inspector = screen.getByRole("complementary", {
      name: "Relationship inspector",
    });

    // Default filters: high + medium confidence, min score 0 — low is hidden.
    expect(within(inspector).getByText("high_left")).toBeInTheDocument();
    expect(within(inspector).getByText("medium_left")).toBeInTheDocument();
    expect(within(inspector).queryByText("low_left")).not.toBeInTheDocument();
    expect(
      within(inspector).getByText("Showing 2 of 3 candidates."),
    ).toBeInTheDocument();
    /* The excluded candidates are accounted for, not merely subtracted. */
    expect(
      within(inspector).getByText(
        "Hidden: 0 below score 0.00 · 1 outside the confidence filter.",
      ),
    ).toBeInTheDocument();

    await user.click(within(inspector).getByRole("checkbox", { name: "low" }));
    expect(within(inspector).getByText("low_left")).toBeInTheDocument();
    expect(
      within(inspector).getByText("Showing 3 of 3 candidates."),
    ).toBeInTheDocument();

    fireEvent.change(within(inspector).getByLabelText(/Minimum score/), {
      target: { value: "0.6" },
    });
    expect(within(inspector).queryByText("medium_left")).not.toBeInTheDocument();
    expect(within(inspector).getByText("high_left")).toBeInTheDocument();
    expect(
      within(inspector).getByText(
        "Hidden: 2 below score 0.60 · 0 outside the confidence filter.",
      ),
    ).toBeInTheDocument();

    fireEvent.change(within(inspector).getByLabelText(/Minimum score/), {
      target: { value: "0.95" },
    });
    expect(
      within(inspector).getByText("No candidates match the current filters."),
    ).toBeInTheDocument();
  });

  it("filters candidates by validation status", async () => {
    const user = userEvent.setup();
    const router = renderAppWithRouter(PAGE_PATH);
    await waitForRelationshipWorkspace();
    const inspector = screen.getByRole("complementary", {
      name: "Relationship inspector",
    });

    await user.click(within(inspector).getByLabelText("Candidate"));
    expect(
      within(inspector).queryByRole("button", { name: CANDIDATE_NAME }),
    ).not.toBeInTheDocument();
    expect(
      within(inspector).getByRole("button", { name: VALIDATED_NAME }),
    ).toBeInTheDocument();
    expect(new URLSearchParams(router.state.location.search).get("state")).toBe(
      "confirmed,validated",
    );
  });

  it("resets the candidate filters when the user switches run", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/relationships", ({ params }) =>
        HttpResponse.json({
          ...relationshipGraph(String(params["sessionId"])),
          edges: scoredEdges(),
        }),
      ),
    );
    const user = userEvent.setup();
    const router = renderAppWithRouter(PAGE_PATH);
    await waitForRelationshipWorkspace();

    // Narrow run r1 down to nothing: high min score plus medium unchecked.
    await user.click(screen.getByRole("checkbox", { name: "medium" }));
    fireEvent.change(screen.getByLabelText(/Minimum score/), {
      target: { value: "0.95" },
    });
    expect(
      screen.getByText("No candidates match the current filters."),
    ).toBeInTheDocument();

    await act(async () => {
      await router.navigate("/projects/p1/sessions/r2/relationships");
    });
    await screen.findByText("high_left");

    /* Run r2 was never filtered by the user: it must show the defaults, not
     * r1's leftovers (filter widgets are keyed by session_id). */
    expect(screen.getByLabelText(/Minimum score/)).toHaveValue("0");
    expect(screen.getByRole("checkbox", { name: "medium" })).toBeChecked();
    expect(screen.getByText("medium_left")).toBeInTheDocument();
    expect(screen.getByText("Showing 2 of 3 candidates.")).toBeInTheDocument();
  });

  it("reports the candidates the output-size cap dropped", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/relationships", ({ params }) =>
        HttpResponse.json({
          ...relationshipGraph(String(params["sessionId"])),
          truncated_pairs: 140,
        }),
      ),
    );
    renderAppAt(PAGE_PATH);
    await waitForRelationshipWorkspace();

    /* The prose line is now a figure in the search-coverage strip. */
    expect(screen.getByText("Omitted by size cap").nextElementSibling)
      .toHaveTextContent("140");
    expect(
      screen.getByText(
        "Scored, then dropped from the displayed artifact by the output-size cap.",
      ),
    ).toBeInTheDocument();
  });

  it("stays quiet about truncation when nothing was dropped", async () => {
    renderAppAt(PAGE_PATH);
    await waitForRelationshipWorkspace();

    expect(screen.queryByText(/output-size cap/)).not.toBeInTheDocument();
  });

  it("warns when discovery found no high-confidence relationship", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/relationships", ({ params }) => {
        const graph = relationshipGraph(String(params["sessionId"]));
        const base = graph.edges![0]!;
        return HttpResponse.json({
          ...graph,
          edges: [
            { ...base, relationship_id: "rel_medium", label: "medium pair", confidence: "medium", ensemble_score: 0.6 },
            { ...base, relationship_id: "rel_low", label: "low pair", confidence: "low", ensemble_score: 0.2 },
          ],
        });
      }),
    );
    renderAppAt(PAGE_PATH);
    await waitForRelationshipWorkspace();

    expect(
      screen.getByText(
        "No high-confidence relationship was found. Medium-confidence candidates are hypotheses only: validate them against the full tables before using a join.",
      ),
    ).toBeInTheDocument();
  });

  it("does not warn when a high-confidence relationship exists", async () => {
    renderAppAt(PAGE_PATH);
    await waitForRelationshipWorkspace();

    expect(
      screen.queryByText(/No high-confidence relationship was found/),
    ).not.toBeInTheDocument();
  });

  it("says so when discovery never ran for this session", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/relationships", ({ params }) =>
        HttpResponse.json({
          ...relationshipGraph(String(params["sessionId"])),
          discovered: false,
          edges: [],
        }),
      ),
    );
    renderAppAt(PAGE_PATH);
    expect(
      await screen.findByText("Relationship discovery has not run"),
    ).toBeInTheDocument();
  });
});

describe("Relationship discovery", () => {
  const discoveryStarted = {
    session_id: "r1",
    execution_session_id: "rdsess_new_1",
    job: {
      job_id: "job_disc_1",
      session_id: "rdsess_new_1",
      status: "queued",
      events_url: "/api/v1/jobs/job_disc_1/events",
    },
  };

  function undiscoveredGraph(overrides: Record<string, unknown> = {}) {
    server.use(
      http.get("/api/v1/sessions/:sessionId/relationships", ({ params }) =>
        HttpResponse.json({
          ...relationshipGraph(String(params["sessionId"])),
          discovered: false,
          edges: [],
          ...overrides,
        }),
      ),
    );
  }

  it("offers discovery from the empty state and tracks the job", async () => {
    let idempotencyKey: string | null = null;
    undiscoveredGraph();
    server.use(
      http.post("/api/v1/sessions/:sessionId/relationships/discover", ({ request }) => {
        idempotencyKey = request.headers.get("Idempotency-Key");
        return HttpResponse.json(discoveryStarted, { status: 201 });
      }),
    );

    const user = userEvent.setup();
    renderAppAt(PAGE_PATH);
    await screen.findByText("Relationship discovery has not run");

    await user.click(
      screen.getByRole("button", { name: "Discover relationships" }),
    );

    expect(
      await screen.findByText(/Tracking job_disc_1 · session rdsess_new_1/),
    ).toBeInTheDocument();
    expect(idempotencyKey).toMatch(/^[0-9a-f-]{36}$/);
  });

  it("disables discovery when fewer than two source tables are readable", async () => {
    undiscoveredGraph({
      nodes: relationshipGraph("r1").nodes?.map((node, index) =>
        index === 0 ? node : { ...node, source_available: false },
      ),
    });

    renderAppAt(PAGE_PATH);
    expect(
      await screen.findByRole("button", { name: "Discover relationships" }),
    ).toBeDisabled();
  });

  it("explains a refused discovery instead of failing silently", async () => {
    undiscoveredGraph();
    server.use(
      http.post("/api/v1/sessions/:sessionId/relationships/discover", () =>
        HttpResponse.json(
          {
            error: {
              code: "relationship_session_busy",
              message: "Session r1 has an active job.",
            },
          },
          { status: 409 },
        ),
      ),
    );

    const user = userEvent.setup();
    renderAppAt(PAGE_PATH);
    await user.click(
      await screen.findByRole("button", { name: "Discover relationships" }),
    );

    expect(
      await screen.findByText("Request failed (relationship_session_busy)"),
    ).toBeInTheDocument();
  });

  it("offers a re-run once the run already has candidates", async () => {
    const user = userEvent.setup();
    renderAppAt(PAGE_PATH);
    await user.click(
      await screen.findByRole("button", { name: /Discovery controls/ }),
    );
    expect(
      await screen.findByRole("button", { name: "Re-run discovery" }),
    ).toBeEnabled();
    expect(
      screen.getByRole("link", {
        name: "Review the project join policy in Knowledge.",
      }),
    ).toHaveAttribute(
      "href",
      "/projects/p1/sessions/r1/semantic?view=joins",
    );
    expect(
      screen.queryByRole("button", { name: "Discover relationships" }),
    ).not.toBeInTheDocument();
  });
});
