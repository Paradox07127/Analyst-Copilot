import { describe, expect, it } from "vitest";
import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { server } from "./msw/server";
import { SAMPLE_VERIFIED_RELATIONS, semanticView } from "./msw/handlers";
import { renderAppAt } from "./render";

const PAGE_PATH = "/projects/p1/sessions/r1/semantic";

function renderKnowledgeView(view: "definitions" | "joins") {
  renderAppAt(`${PAGE_PATH}?view=${view}`);
}

describe("Knowledge page", () => {
  it("renders field meanings, join whitelist, proposals, and column roles", async () => {
    const user = userEvent.setup();
    renderAppAt(PAGE_PATH);

    await screen.findByRole("heading", { name: "Knowledge" });
    expect(
      screen.getByText("Gross order value before refunds."),
    ).toBeInTheDocument();
    expect(screen.getByText(/Version 3/)).toBeInTheDocument();

    /* Pending proposal card with accept/dismiss. */
    expect(screen.getByText("Display name of the row.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Accept" })).toBeInTheDocument();

    /* Column roles from this run. */
    expect(screen.getByText("· measure")).toBeInTheDocument();

    /* Join policy is a separate project-wide task, restored from URL state. */
    await user.click(screen.getByRole("button", { name: "Join policy" }));
    expect(
      screen.getByText("sample.csv.id -> other.csv.sample_id"),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Confirm" })).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Revoke auto-confirmation" }),
    ).toBeInTheDocument();
  });

  /* The semantic layer is project-scoped, so /semantic returns proposals for
   * every table the project ever loaded. On the real Olist run that was 452
   * proposals of which only 52 touched a table this run had, and the other 400
   * (a Stack Overflow survey, a World Cup set, an LLM-usage set) rendered
   * inline with nothing marking them as foreign. */
  it("separates proposals for tables this session did not load", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/semantic", ({ params }) => {
        const view = semanticView(String(params["sessionId"]));
        return HttpResponse.json({
          ...view,
          proposals: [
            ...(view.proposals ?? []),
            {
              dataset: "unrelated_survey.csv",
              column: "respondent_id",
              meaning: "Identifier from a different upload.",
              unit_guess: "",
              confidence: "hypothesis",
              source: "bootstrap",
            },
          ],
        });
      }),
    );

    renderAppAt(PAGE_PATH);
    await screen.findByRole("heading", { name: "Knowledge" });

    /* This run's table leads, ungrouped behind any toggle. */
    expect(screen.getByText("Display name of the row.")).toBeInTheDocument();

    /* The foreign one is counted and put behind its own disclosure. */
    const foreign = screen.getByRole("button", {
      name: /From other tables in this project \(1\)/,
    });
    expect(foreign).toHaveAttribute("aria-expanded", "false");

    await userEvent.setup().click(foreign);
    expect(
      await screen.findByText("Identifier from a different upload."),
    ).toBeInTheDocument();
  });

  it("saves an inline field-meaning edit with the loaded expected_version", async () => {
    let putBody: Record<string, unknown> | null = null;
    server.use(
      http.put(
        "/api/v1/sessions/:sessionId/semantic/seeds",
        async ({ params, request }) => {
          putBody = (await request.json()) as Record<string, unknown>;
          return HttpResponse.json({
            session_id: String(params["sessionId"]),
            version: 4,
            field_meanings: putBody["field_meanings"],
          });
        },
      ),
    );

    const user = userEvent.setup();
    renderAppAt(PAGE_PATH);

    await screen.findByRole("heading", { name: "Knowledge" });
    await user.click(screen.getByRole("button", { name: "Edit" }));
    const meaningInput = screen.getByRole("textbox", {
      name: "Meaning of sample.csv.value",
    });
    await user.clear(meaningInput);
    await user.type(meaningInput, "Net order value.");
    await user.click(screen.getByRole("button", { name: "Save" }));

    await screen.findByRole("button", { name: "Edit" });
    expect(putBody).toMatchObject({
      expected_version: 3,
      field_meanings: [
        expect.objectContaining({
          dataset: "sample.csv",
          column: "value",
          meaning: "Net order value.",
          unit: "USD",
        }),
      ],
    });
  });

  it("shows the reload prompt on a version conflict", async () => {
    server.use(
      http.put("/api/v1/sessions/:sessionId/semantic/seeds", () =>
        HttpResponse.json(
          {
            error: {
              code: "version_conflict",
              message:
                "Semantic seeds changed since they were loaded: expected " +
                "version 3, current version is 5. Reload and retry.",
            },
          },
          { status: 409 },
        ),
      ),
    );

    const user = userEvent.setup();
    renderAppAt(PAGE_PATH);

    await screen.findByRole("heading", { name: "Knowledge" });
    await user.click(screen.getByRole("button", { name: "Edit" }));
    await user.click(screen.getByRole("button", { name: "Save" }));

    const alert = await screen.findByRole("alert");
    expect(
      within(alert).getByText(
        "The semantic layer changed while you were editing.",
      ),
    ).toBeInTheDocument();
    /* Reload clears the conflict and leaves edit mode. */
    await user.click(within(alert).getByRole("button", { name: "Reload" }));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(
      await screen.findByRole("button", { name: "Edit" }),
    ).toBeInTheDocument();
  });

  it("locks proposal review while a field-meaning edit is in progress", async () => {
    const user = userEvent.setup();
    renderAppAt(PAGE_PATH);

    await screen.findByRole("heading", { name: "Knowledge" });
    const accept = screen.getByRole("button", { name: "Accept" });
    const dismiss = screen.getByRole("button", { name: "Dismiss" });
    expect(accept).toBeEnabled();

    await user.click(screen.getByRole("button", { name: "Edit" }));
    expect(accept).toBeDisabled();
    expect(dismiss).toBeDisabled();
    expect(
      screen.getByText(/Save or cancel the field-meaning edit/),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(accept).toBeEnabled();
    expect(dismiss).toBeEnabled();
  });

  it("accepts a meaning proposal and refreshes the view", async () => {
    let acceptBody: Record<string, unknown> | null = null;
    server.use(
      http.post(
        "/api/v1/sessions/:sessionId/semantic/proposals/accept",
        async ({ params, request }) => {
          acceptBody = (await request.json()) as Record<string, unknown>;
          return HttpResponse.json({
            session_id: String(params["sessionId"]),
            dataset: "sample.csv",
            column: "name",
            status: "accepted",
            seeds_version: 4,
          });
        },
      ),
    );

    const user = userEvent.setup();
    renderAppAt(PAGE_PATH);

    await screen.findByRole("heading", { name: "Knowledge" });
    await user.click(screen.getByRole("button", { name: "Accept" }));

    await screen.findByRole("heading", { name: "Knowledge" });
    expect(acceptBody).toMatchObject({
      dataset: "sample.csv",
      column: "name",
    });
    /* Plain Accept takes the machine draft verbatim: no override travels. */
    expect(acceptBody).not.toHaveProperty("meaning");
    expect(acceptBody).not.toHaveProperty("unit");
  });

  it("accepts a proposal with an edited meaning and unit", async () => {
    let acceptBody: Record<string, unknown> | null = null;
    server.use(
      http.post(
        "/api/v1/sessions/:sessionId/semantic/proposals/accept",
        async ({ params, request }) => {
          acceptBody = (await request.json()) as Record<string, unknown>;
          return HttpResponse.json({
            session_id: String(params["sessionId"]),
            dataset: "sample.csv",
            column: "name",
            status: "accepted",
            seeds_version: 4,
          });
        },
      ),
    );

    const user = userEvent.setup();
    renderAppAt(PAGE_PATH);

    await screen.findByRole("heading", { name: "Knowledge" });
    await user.click(screen.getByRole("button", { name: "Edit & accept" }));

    /* The editor opens on the machine guess, not on an empty field. */
    const meaning = screen.getByRole("textbox", {
      name: "Proposed meaning of sample.csv.name",
    });
    expect(meaning).toHaveValue("Display name of the row.");
    /* fireEvent, not user.type: react-resizable-panels' document-level
     * pointerdown handler preventDefaults clicks at jsdom's all-zero rects,
     * so user-event's click never moves focus between the two inputs. */
    fireEvent.change(meaning, { target: { value: "Customer display name." } });
    fireEvent.change(
      screen.getByRole("textbox", { name: "Proposed unit of sample.csv.name" }),
      { target: { value: "text" } },
    );
    await user.click(screen.getByRole("button", { name: "Accept edited" }));

    await screen.findByRole("heading", { name: "Knowledge" });
    expect(acceptBody).toMatchObject({
      dataset: "sample.csv",
      column: "name",
      meaning: "Customer display name.",
      unit: "text",
    });
  });

  it("cancels an edit without accepting anything", async () => {
    let accepted = 0;
    server.use(
      http.post("/api/v1/sessions/:sessionId/semantic/proposals/accept", () => {
        accepted += 1;
        return HttpResponse.json({
          session_id: "r1",
          dataset: "sample.csv",
          column: "name",
          status: "accepted",
          seeds_version: 4,
        });
      }),
    );

    const user = userEvent.setup();
    renderAppAt(PAGE_PATH);

    await screen.findByRole("heading", { name: "Knowledge" });
    await user.click(screen.getByRole("button", { name: "Edit & accept" }));
    await user.click(screen.getByRole("button", { name: "Cancel" }));

    expect(accepted).toBe(0);
    expect(screen.getByText("Display name of the row.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Accept" })).toBeInTheDocument();
  });
});

describe("Knowledge page seed classes", () => {
  /* metric definitions / entity notes / verified answers used to be editable
   * via API; they share seeds.json and its version with the field
   * meanings, so every save goes through the one seeds PUT. */

  async function seedsPut(): Promise<Record<string, unknown>> {
    return await new Promise((resolve) => {
      server.use(
        http.put(
          "/api/v1/sessions/:sessionId/semantic/seeds",
          async ({ params, request }) => {
            const body = (await request.json()) as Record<string, unknown>;
            resolve(body);
            return HttpResponse.json({
              session_id: String(params["sessionId"]),
              version: 4,
              field_meanings: body["field_meanings"],
              metric_definitions: body["metric_definitions"] ?? [],
              entity_notes: body["entity_notes"] ?? [],
              verified_answers: body["verified_answers"] ?? [],
            });
          },
        ),
      );
    });
  }

  it("renders the three hand-edited seed classes", async () => {
    renderKnowledgeView("definitions");

    await screen.findByRole("heading", { name: "Knowledge" });
    expect(screen.getByText("Active user")).toBeInTheDocument();
    expect(
      screen.getByText("A user with at least one session in 28 days."),
    ).toBeInTheDocument();
    expect(
      screen.getByText("One row per billing account, not per person."),
    ).toBeInTheDocument();
    expect(screen.getByText("$4.2M, up 12% QoQ.")).toBeInTheDocument();
    expect(screen.getByText("verified 2026-01-02")).toBeInTheDocument();
  });

  it("adds a metric definition, carrying the loaded field meanings along", async () => {
    const user = userEvent.setup();
    const put = seedsPut();
    renderKnowledgeView("definitions");

    await screen.findByRole("heading", { name: "Knowledge" });
    fireEvent.change(screen.getByRole("textbox", { name: "New metric name" }), {
      target: { value: "Churn" },
    });
    fireEvent.change(
      screen.getByRole("textbox", { name: "New metric definition" }),
      { target: { value: "No session in 90 days." } },
    );
    await user.click(screen.getByRole("button", { name: "Add metric" }));

    const body = await put;
    expect(body["expected_version"]).toBe(3);
    expect(body["metric_definitions"]).toEqual([
      {
        name: "Active user",
        definition: "A user with at least one session in 28 days.",
        formula: "count(distinct user_id)",
        caveats: null,
      },
      { name: "Churn", definition: "No session in 90 days.", formula: null, caveats: null },
    ]);
    /* field_meanings is required by the contract, so it always travels. */
    expect(body["field_meanings"]).toHaveLength(1);
    /* The classes this save did not edit are omitted, so the server leaves
     * them alone instead of clearing them. */
    expect(body).not.toHaveProperty("entity_notes");
    expect(body).not.toHaveProperty("verified_answers");
  });

  it("saves an inline entity-note edit", async () => {
    const user = userEvent.setup();
    renderKnowledgeView("definitions");

    await screen.findByRole("heading", { name: "Knowledge" });
    await user.click(
      screen.getByRole("button", { name: "Edit entity note customer" }),
    );
    const put = seedsPut();
    fireEvent.change(screen.getByRole("textbox", { name: "Note of customer" }), {
      target: { value: "One row per person after the 2026 migration." },
    });
    await user.click(screen.getByRole("button", { name: "Save entity note" }));

    const body = await put;
    expect(body["entity_notes"]).toEqual([
      { name: "customer", note: "One row per person after the 2026 migration." },
    ]);
    expect(body).not.toHaveProperty("metric_definitions");
  });

  it("keeps the original verification date when an answer is edited", async () => {
    const user = userEvent.setup();
    renderKnowledgeView("definitions");

    await screen.findByRole("heading", { name: "Knowledge" });
    await user.click(
      screen.getByRole("button", {
        name: "Edit verified answer What was Q3 revenue?",
      }),
    );
    const put = seedsPut();
    fireEvent.change(
      screen.getByRole("textbox", { name: "Answer of What was Q3 revenue?" }),
      { target: { value: "$4.4M, restated." } },
    );
    await user.click(
      screen.getByRole("button", { name: "Save verified answer" }),
    );

    const body = await put;
    expect(body["verified_answers"]).toEqual([
      {
        question: "What was Q3 revenue?",
        answer: "$4.4M, restated.",
        evidence_note: null,
        verified_at: "2026-01-02T03:04:05Z",
      },
    ]);
  });

  it("requires a second click before a delete leaves the page", async () => {
    const bodies: Record<string, unknown>[] = [];
    server.use(
      http.put("/api/v1/sessions/:sessionId/semantic/seeds", async ({ request }) => {
        const body = (await request.json()) as Record<string, unknown>;
        bodies.push(body);
        return HttpResponse.json({
          session_id: "r1",
          version: 4,
          field_meanings: body["field_meanings"],
          metric_definitions: [],
          entity_notes: [],
          verified_answers: [],
        });
      }),
    );

    const user = userEvent.setup();
    renderKnowledgeView("definitions");

    await screen.findByRole("heading", { name: "Knowledge" });
    await user.click(
      screen.getByRole("button", { name: "Delete metric Active user" }),
    );
    /* The first click only arms the confirmation. */
    expect(bodies).toHaveLength(0);
    expect(
      screen.getByText("Delete \u201cActive user\u201d?"),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Keep metric" }));
    expect(bodies).toHaveLength(0);

    await user.click(
      screen.getByRole("button", { name: "Delete metric Active user" }),
    );
    await user.click(
      screen.getByRole("button", { name: "Confirm deleting metric" }),
    );
    await waitFor(() => expect(bodies).toHaveLength(1));
    expect(bodies[0]?.["metric_definitions"]).toEqual([]);
  });

  it("shows the reload prompt when a seed-class save hits a version conflict", async () => {
    server.use(
      http.put("/api/v1/sessions/:sessionId/semantic/seeds", () =>
        HttpResponse.json(
          {
            error: {
              code: "version_conflict",
              message: "Semantic seeds changed since they were loaded.",
            },
          },
          { status: 409 },
        ),
      ),
    );

    const user = userEvent.setup();
    renderKnowledgeView("definitions");

    await screen.findByRole("heading", { name: "Knowledge" });
    await user.click(
      screen.getByRole("button", { name: "Delete entity note customer" }),
    );
    await user.click(
      screen.getByRole("button", { name: "Confirm deleting entity note" }),
    );

    const alert = await screen.findByRole("alert");
    expect(
      within(alert).getByText(
        "The semantic layer changed while you were editing.",
      ),
    ).toBeInTheDocument();
  });
});

describe("Knowledge page verified relations", () => {
  const FIRST_LABEL =
    "orders.csv.customer_id → customers.csv.customer_id";
  const SECOND_LABEL = "orders.csv.product_id → products.csv.product_id";

  it("renders both verified relations with their fields", async () => {
    renderKnowledgeView("joins");

    await screen.findByRole("heading", { name: "Knowledge" });
    const first = screen.getByText(FIRST_LABEL).closest("li");
    const second = screen.getByText(SECOND_LABEL).closest("li");
    expect(first).not.toBeNull();
    expect(second).not.toBeNull();

    expect(
      within(first as HTMLElement).getByText("many to one"),
    ).toBeInTheDocument();
    expect(first?.textContent).toContain("confirmed by user");
    expect(first?.textContent).toContain("2026-07-25");
    expect(first?.textContent).toContain("from session r1");

    expect(
      within(second as HTMLElement).getByText("many to one"),
    ).toBeInTheDocument();
    expect(second?.textContent).toContain("confirmed by user");
    expect(second?.textContent).toContain("2026-07-25");
    expect(second?.textContent).toContain("from session r1");
  });

  it("deletes the relation that was clicked, not the other one", async () => {
    let deleteBody: Record<string, unknown> | null = null;
    server.use(
      http.post(
        "/api/v1/sessions/:sessionId/semantic/verified-relations/delete",
        async ({ request }) => {
          deleteBody = (await request.json()) as Record<string, unknown>;
          return HttpResponse.json({
            session_id: "r1",
            seeds_version: 4,
            verified_relations: [SAMPLE_VERIFIED_RELATIONS[0]],
          });
        },
      ),
    );

    const user = userEvent.setup();
    renderKnowledgeView("joins");

    await screen.findByRole("heading", { name: "Knowledge" });
    /* Click delete on the SECOND relation — the request body must carry the
     * second relation's keys, not the first's. */
    await user.click(
      screen.getByRole("button", { name: `Delete relation ${SECOND_LABEL}` }),
    );
    await user.click(
      screen.getByRole("button", { name: "Confirm deleting relation" }),
    );

    await waitFor(() => expect(deleteBody).not.toBeNull());
    expect(deleteBody).toEqual({
      left: "orders.csv.product_id",
      right: "products.csv.product_id",
      expected_version: 3,
    });
  });

  it("requires a second click before a relation delete leaves the page", async () => {
    let deletes = 0;
    server.use(
      http.post(
        "/api/v1/sessions/:sessionId/semantic/verified-relations/delete",
        () => {
          deletes += 1;
          return HttpResponse.json({
            session_id: "r1",
            seeds_version: 4,
            verified_relations: [],
          });
        },
      ),
    );

    const user = userEvent.setup();
    renderKnowledgeView("joins");

    await screen.findByRole("heading", { name: "Knowledge" });
    await user.click(
      screen.getByRole("button", { name: `Delete relation ${FIRST_LABEL}` }),
    );
    expect(deletes).toBe(0);
    expect(screen.getByText(`Delete “${FIRST_LABEL}”?`)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Keep relation" }));
    expect(deletes).toBe(0);
    expect(
      screen.getByRole("button", { name: `Delete relation ${FIRST_LABEL}` }),
    ).toBeInTheDocument();
  });

  /* No handler override: the default delete handler is stateful, so this also
   * covers that the refetch sees the deletion. */
  it("removes the relation from the list once the delete succeeds", async () => {
    const user = userEvent.setup();
    renderKnowledgeView("joins");

    await screen.findByRole("heading", { name: "Knowledge" });
    expect(screen.getByText(FIRST_LABEL)).toBeInTheDocument();

    await user.click(
      screen.getByRole("button", { name: `Delete relation ${FIRST_LABEL}` }),
    );
    await user.click(
      screen.getByRole("button", { name: "Confirm deleting relation" }),
    );

    await waitFor(() =>
      expect(screen.queryByText(FIRST_LABEL)).not.toBeInTheDocument(),
    );
    expect(screen.getByText(SECOND_LABEL)).toBeInTheDocument();
  });

  it("shows the empty-state hint when there are no verified relations", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/semantic", ({ params }) =>
        HttpResponse.json({
          ...semanticView(String(params["sessionId"])),
          verified_relations: [],
        }),
      ),
    );

    renderKnowledgeView("joins");

    await screen.findByRole("heading", { name: "Knowledge" });
    expect(screen.getByText("No verified relations yet")).toBeInTheDocument();
    expect(
      screen.getByText(
        "Confirming a join on the Relationships page sinks it here " +
          "automatically. Remove one below if it turns out to be wrong.",
      ),
    ).toBeInTheDocument();
  });
});
