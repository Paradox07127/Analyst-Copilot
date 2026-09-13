import { describe, expect, it } from "vitest";
import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { server } from "./msw/server";
import { renderAppAt, renderAppWithRouterAt } from "./render";
import { skillPrepared, skillsView } from "./msw/handlers";

const PAGE_PATH = "/projects/p1/sessions/r1/skills";

function cardFor(name: string): HTMLElement {
  return screen
    .getAllByRole("listitem")
    .find((card) => within(card).queryByText(name))!;
}

describe("Skills page", () => {
  it("lists library skills and seed templates with their parameter signature", async () => {
    renderAppAt(PAGE_PATH);

    await screen.findByRole("heading", { name: "Skills" });

    const saved = cardFor("Revenue by name");
    /* Badge copy, not the storage word: `library` / `seed` said nothing about
     * what the reader may do with the row. */
    expect(within(saved).getByText("Saved skill")).toBeInTheDocument();
    expect(within(saved).getByText("Reads: name, value")).toBeInTheDocument();
    /* A saved skill takes a target dataset and nothing else. */
    expect(
      within(saved).queryByRole("combobox", { name: /group_col/ }),
    ).not.toBeInTheDocument();
    expect(within(saved).getByRole("button", { name: "Replay" })).toBeEnabled();

    const seed = cardFor("Group totals and averages");
    expect(within(seed).getByText("Built-in template")).toBeInTheDocument();
    expect(
      within(seed).getByRole("combobox", { name: /group_col/ }),
    ).toBeInTheDocument();
    /* Unbound placeholders block the replay until they are bound. */
    expect(within(seed).getByRole("button", { name: "Replay" })).toBeDisabled();
    expect(
      within(seed).getByText(/Bind \{group_col\}, \{value_col\} to a column/),
    ).toBeInTheDocument();
  });

  /* One flat list made a project-owned skill and a shipped template look like
   * the same kind of row, though only one of them can be deleted. */
  it("separates skills saved here from templates that ship with the app", async () => {
    renderAppAt(PAGE_PATH);
    await screen.findByRole("heading", { name: "Skills" });

    const savedSection = screen
      .getByRole("heading", { name: /^Saved in this project/ })
      .closest("section")!;
    expect(within(savedSection).getByText("Revenue by name")).toBeInTheDocument();
    expect(
      within(savedSection).queryByText("Group totals and averages"),
    ).not.toBeInTheDocument();

    const seedSection = screen
      .getByRole("heading", { name: /^Built-in templates/ })
      .closest("section")!;
    expect(
      within(seedSection).getByText("Group totals and averages"),
    ).toBeInTheDocument();
  });

  it("says which run's data a replay binds to", async () => {
    renderAppAt(PAGE_PATH);
    await screen.findByRole("heading", { name: "Skills" });
    expect(
      screen.getByText("A replay binds to this session's data: sample.csv."),
    ).toBeInTheDocument();
  });

  it("restores a shareable skill search and clears only that filter", async () => {
    const user = userEvent.setup();
    const { router } = renderAppWithRouterAt(`${PAGE_PATH}?q=averages`);

    expect(
      await screen.findByRole("heading", { name: /^Built-in templates/ }),
    ).toBeInTheDocument();
    expect(screen.getByLabelText("Find a skill")).toHaveValue("averages");
    expect(screen.getByText("Group totals and averages")).toBeInTheDocument();
    expect(screen.queryByText("Revenue by name")).not.toBeInTheDocument();
    expect(screen.getByText("1 of 2")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Clear" }));
    expect(router.state.location.search).toBe("");
    expect(await screen.findByText("Revenue by name")).toBeInTheDocument();
  });

  it("keeps the SQL one click away instead of expanded on every card", async () => {
    const user = userEvent.setup();
    renderAppAt(PAGE_PATH);
    await screen.findByRole("heading", { name: "Skills" });

    const saved = cardFor("Revenue by name");
    const toggle = within(saved).getByRole("button", { name: /^SQL/ });
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    await user.click(toggle);
    expect(toggle).toHaveAttribute("aria-expanded", "true");
  });

  it("offers only columns a binding may actually use", async () => {
    /* A bound column is interpolated into the SQL, so a header with a space
     * would be refused at prepare time — the form must say so up front. */
    renderAppAt(PAGE_PATH);

    await screen.findByRole("heading", { name: "Skills" });
    const seed = cardFor("Group totals and averages");
    const picker = within(seed).getByRole("combobox", { name: /group_col/ });

    expect(within(picker).getByRole("option", { name: "value" })).toBeEnabled();
    const unbindable = within(picker).getByRole("option", {
      name: /unit price — not a plain identifier/,
    });
    expect(unbindable).toBeDisabled();
    expect(
      within(seed).getByText(/unit price cannot be bound/),
    ).toBeInTheDocument();
  });

  it("binds a seed, prepares, confirms the SQL, executes, and tracks the job", async () => {
    let prepareBody: Record<string, unknown> | null = null;
    let idempotencyKey: string | null = null;
    let executeBody: Record<string, unknown> | null = null;
    server.use(
      http.post(
        "/api/v1/sessions/:sessionId/skills/:skillId/prepare",
        async ({ request, params }) => {
          prepareBody = (await request.json()) as Record<string, unknown>;
          return HttpResponse.json({
            session_id: String(params["sessionId"]),
            skill_id: String(params["skillId"]),
            action_hash: "a".repeat(64),
            approval_token: "c".repeat(32),
            expires_at: "2026-07-25T12:00:00Z",
            name: "Group totals and averages",
            question: "How does value vary across name segments?",
            sql_preview: "SELECT name, SUM(value) FROM sample GROUP BY 1",
            dataset_ids: ["sample"],
            dataset_names: ["sample.csv"],
            bindings: { group_col: "name", value_col: "value" },
            uses_llm: false,
          });
        },
      ),
      http.post(
        "/api/v1/sessions/:sessionId/skills/:skillId/execute",
        async ({ request, params }) => {
          idempotencyKey = request.headers.get("Idempotency-Key");
          executeBody = (await request.json()) as Record<string, unknown>;
          return HttpResponse.json(
            {
              session_id: String(params["sessionId"]),
              skill_id: String(params["skillId"]),
              execution_session_id: "ssess_new_1",
              job: {
                job_id: "job_s_1",
                session_id: "ssess_new_1",
                status: "queued",
                events_url: "/api/v1/jobs/job_s_1/events",
              },
            },
            { status: 201 },
          );
        },
      ),
    );

    const user = userEvent.setup();
    renderAppAt(PAGE_PATH);

    await screen.findByRole("heading", { name: "Skills" });
    const seed = cardFor("Group totals and averages");
    await user.selectOptions(
      within(seed).getByRole("combobox", { name: /group_col/ }),
      "name",
    );
    await user.selectOptions(
      within(seed).getByRole("combobox", { name: /value_col/ }),
      "value",
    );
    await user.click(within(seed).getByRole("button", { name: "Replay" }));

    let dialog = await screen.findByRole("alertdialog", {
      name: "Confirm skill replay",
    });
    expect(
      within(dialog).getByText(/SELECT name, SUM\(value\) FROM sample/),
    ).toBeInTheDocument();
    expect(within(dialog).getByText(/Datasets: sample\.csv/)).toBeInTheDocument();
    /* Replay is deterministic SQL: the card must say so, not hide it. */
    expect(within(dialog).getByText("none")).toBeInTheDocument();
    expect(prepareBody).toMatchObject({
      dataset_ids: ["sample"],
      bindings: { group_col: "name", value_col: "value" },
    });

    const initialConfirm = within(dialog).getByRole("button", {
      name: "Confirm & replay",
    });
    await waitFor(() => expect(initialConfirm).toHaveFocus());
    await user.keyboard("{Escape}");
    expect(
      screen.queryByRole("alertdialog", { name: "Confirm skill replay" }),
    ).not.toBeInTheDocument();
    await waitFor(() =>
      expect(
        within(seed).getByRole("button", { name: "Replay" }),
      ).toHaveFocus(),
    );

    await user.click(within(seed).getByRole("button", { name: "Replay" }));
    dialog = await screen.findByRole("alertdialog", {
      name: "Confirm skill replay",
    });
    await user.click(
      within(dialog).getByRole("button", { name: "Confirm & replay" }),
    );

    expect(
      await screen.findByText(/Tracking job_s_1 · session ssess_new_1/),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: "Open replay results" }),
    ).toHaveAttribute(
      "href",
      "/projects/p1/sessions/ssess_new_1/artifacts",
    );
    expect(
      screen.getByRole("link", { name: "Source session artifacts" }),
    ).toHaveAttribute("href", "/projects/p1/sessions/r1/artifacts");
    expect(idempotencyKey).toMatch(/^[0-9a-f-]{36}$/);
    expect(executeBody).toMatchObject({
      action_hash: "a".repeat(64),
      approval_token: "c".repeat(32),
    });
  });

  it("replays a two-table skill against two selected datasets", async () => {
    /* A saved skill that referenced two relations: the server takes either 1
     * target (whole analysis) or exactly 2, mapped in order. */
    const twoTableView = (sessionId: string) => {
      const view = skillsView(sessionId);
      return {
        ...view,
        skills: [
          {
            ...view.skills![0]!,
            name: "Orders joined to customers",
            expected_datasets: ["orders", "customers"],
          },
        ],
        datasets: [
          { ...view.datasets![0]!, dataset_id: "orders", name: "orders.csv" },
          {
            ...view.datasets![0]!,
            dataset_id: "customers",
            name: "customers.csv",
            relation: "customers",
          },
        ],
      };
    };
    let prepareBody: Record<string, unknown> | null = null;
    server.use(
      http.get("/api/v1/sessions/:sessionId/skills", ({ params }) =>
        HttpResponse.json(twoTableView(String(params["sessionId"]))),
      ),
      http.post(
        "/api/v1/sessions/:sessionId/skills/:skillId/prepare",
        async ({ request, params }) => {
          prepareBody = (await request.json()) as Record<string, unknown>;
          return HttpResponse.json({
            ...skillPrepared(String(params["sessionId"]), String(params["skillId"])),
            dataset_ids: ["orders", "customers"],
            dataset_names: ["orders.csv", "customers.csv"],
          });
        },
      ),
    );

    const user = userEvent.setup();
    renderAppAt(PAGE_PATH);

    await screen.findByRole("heading", { name: "Skills" });
    const card = cardFor("Orders joined to customers");
    expect(within(card).getByText(/1 selected/)).toBeInTheDocument();
    expect(
      within(card).getByText(
        /Select 1 dataset to run the whole analysis on it, or 2 to map onto the 2 tables/,
      ),
    ).toBeInTheDocument();

    await user.click(within(card).getByRole("checkbox", { name: "customers.csv" }));
    expect(within(card).getByText(/2 selected/)).toBeInTheDocument();
    await user.click(within(card).getByRole("button", { name: "Replay" }));

    const dialog = await screen.findByRole("alertdialog", {
      name: "Confirm skill replay",
    });
    expect(
      within(dialog).getByText(/Datasets: orders\.csv, customers\.csv/),
    ).toBeInTheDocument();
    expect(prepareBody).toMatchObject({
      dataset_ids: ["orders", "customers"],
    });
  });

  it("blocks a target count the server would refuse", async () => {
    const twoDatasetSeedView = (sessionId: string) => {
      const view = skillsView(sessionId);
      return {
        ...view,
        datasets: [
          view.datasets![0]!,
          { ...view.datasets![0]!, dataset_id: "other", name: "other.csv" },
        ],
      };
    };
    server.use(
      http.get("/api/v1/sessions/:sessionId/skills", ({ params }) =>
        HttpResponse.json(twoDatasetSeedView(String(params["sessionId"]))),
      ),
    );

    const user = userEvent.setup();
    renderAppAt(PAGE_PATH);

    await screen.findByRole("heading", { name: "Skills" });

    /* A seed template instantiates on exactly one dataset. */
    const seed = cardFor("Group totals and averages");
    expect(
      within(seed).getByText("A seed template is instantiated on exactly 1 dataset."),
    ).toBeInTheDocument();
    await user.selectOptions(
      within(seed).getByRole("combobox", { name: /group_col/ }),
      "name",
    );
    await user.selectOptions(
      within(seed).getByRole("combobox", { name: /value_col/ }),
      "value",
    );
    expect(within(seed).getByRole("button", { name: "Replay" })).toBeEnabled();
    await user.click(within(seed).getByRole("checkbox", { name: "other.csv" }));
    expect(within(seed).getByRole("button", { name: "Replay" })).toBeDisabled();

    /* A one-table saved skill likewise only takes one target. */
    const saved = cardFor("Revenue by name");
    await user.click(within(saved).getByRole("checkbox", { name: "other.csv" }));
    expect(within(saved).getByRole("button", { name: "Replay" })).toBeDisabled();
    await user.click(within(saved).getByRole("checkbox", { name: "sample.csv" }));
    expect(within(saved).getByRole("button", { name: "Replay" })).toBeEnabled();
  });

  it("explains an already-used approval on 409 approval_consumed", async () => {
    server.use(
      http.post("/api/v1/sessions/:sessionId/skills/:skillId/execute", () =>
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

    await screen.findByRole("heading", { name: "Skills" });
    const saved = cardFor("Revenue by name");
    await user.click(within(saved).getByRole("button", { name: "Replay" }));
    await user.click(
      await screen.findByRole("button", { name: "Confirm & replay" }),
    );

    expect(
      await screen.findByText("This approval was already used."),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Prepare again" }),
    ).toBeInTheDocument();
  });

  it("surfaces an invalid binding as a retryable error", async () => {
    server.use(
      http.post("/api/v1/sessions/:sessionId/skills/:skillId/prepare", () =>
        HttpResponse.json(
          {
            error: {
              code: "binding_invalid",
              message: "Column(s) nope do not exist in the selected dataset(s).",
            },
          },
          { status: 422 },
        ),
      ),
    );

    const user = userEvent.setup();
    renderAppAt(PAGE_PATH);

    await screen.findByRole("heading", { name: "Skills" });
    const saved = cardFor("Revenue by name");
    await user.click(within(saved).getByRole("button", { name: "Replay" }));

    expect(
      await screen.findByText("binding_invalid"),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Column(s) nope do not exist in the selected dataset(s)."),
    ).toBeInTheDocument();
  });

  it("saves a validated plan of this session as a named skill", async () => {
    let saveBody: Record<string, unknown> | null = null;
    server.use(
      http.post("/api/v1/sessions/:sessionId/skills", async ({ request }) => {
        saveBody = (await request.json()) as Record<string, unknown>;
        return HttpResponse.json({ skill_id: "skill_new" }, { status: 201 });
      }),
    );

    const user = userEvent.setup();
    renderAppAt(PAGE_PATH);

    await screen.findByRole("heading", { name: "Skills" });
    /* fireEvent, not user.type: react-resizable-panels' document-level
     * pointerdown handler preventDefaults clicks at jsdom's all-zero rects,
     * so user-event's click never focuses the input. */
    fireEvent.change(screen.getByRole("textbox", { name: /Skill name/ }), {
      target: { value: "Revenue by region" },
    });
    await user.click(screen.getByRole("button", { name: "Save as skill" }));

    await waitFor(() =>
      expect(saveBody).toMatchObject({
        source_artifact_id: "plan_abc123",
        name: "Revenue by region",
      }),
    );
    /* A successful save clears the form so the next plan starts fresh. */
    expect(screen.getByRole("textbox", { name: /Skill name/ })).toHaveValue("");
  });

  it("requires a name before a plan can be saved", async () => {
    renderAppAt(PAGE_PATH);
    await screen.findByRole("heading", { name: "Skills" });
    expect(
      screen.getByRole("button", { name: "Save as skill" }),
    ).toBeDisabled();
  });

  it("deletes a library skill behind a confirmation, and never a seed", async () => {
    let deletedPath: string | null = null;
    server.use(
      http.delete("/api/v1/projects/:projectId/skills/:skillId", ({ request }) => {
        deletedPath = new URL(request.url).pathname;
        return new HttpResponse(null, { status: 204 });
      }),
    );

    const user = userEvent.setup();
    renderAppAt(PAGE_PATH);

    await screen.findByRole("heading", { name: "Skills" });
    const seed = cardFor("Group totals and averages");
    expect(
      within(seed).queryByRole("button", { name: "Delete" }),
    ).not.toBeInTheDocument();

    const saved = cardFor("Revenue by name");
    await user.click(within(saved).getByRole("button", { name: "Delete" }));
    expect(
      within(saved).getByText(/Delete “Revenue by name”\?/),
    ).toBeInTheDocument();
    await user.click(within(saved).getAllByRole("button", { name: "Delete" })[0]!);

    await waitFor(() =>
      expect(deletedPath).toBe("/api/v1/projects/p1/skills/skill_saved_1"),
    );
  });

  it("surfaces a refused seed deletion instead of failing silently", async () => {
    server.use(
      http.delete("/api/v1/projects/:projectId/skills/:skillId", () =>
        HttpResponse.json(
          {
            error: {
              code: "skill_not_deletable",
              message: "Builtin seed templates cannot be deleted.",
            },
          },
          { status: 409 },
        ),
      ),
    );

    const user = userEvent.setup();
    renderAppAt(PAGE_PATH);

    await screen.findByRole("heading", { name: "Skills" });
    const saved = cardFor("Revenue by name");
    await user.click(within(saved).getByRole("button", { name: "Delete" }));
    await user.click(within(saved).getAllByRole("button", { name: "Delete" })[0]!);

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Builtin seed templates cannot be deleted.",
    );
  });

  it("shows the empty state when neither library nor seeds are available", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/skills", ({ params }) =>
        HttpResponse.json({
          ...skillsView(String(params["sessionId"])),
          skills: [],
        }),
      ),
    );
    renderAppAt(PAGE_PATH);
    expect(
      await screen.findByText("No skills available"),
    ).toBeInTheDocument();
  });
});

describe("Skills page seed import", () => {
  it("imports a bound seed into the project library", async () => {
    let importBody: Record<string, unknown> | null = null;
    let importedPath: string | null = null;
    server.use(
      http.post(
        "/api/v1/sessions/:sessionId/skills/:seedId/import",
        async ({ request, params }) => {
          importBody = (await request.json()) as Record<string, unknown>;
          importedPath = String(params["seedId"]);
          return HttpResponse.json(
            {
              skill_id: "skill_imported_seed",
              source: "library",
              name: "Group totals and averages",
              description: "From seed 'group_value_comparison' on sample.",
              question: "How does value vary across name segments?",
              sql: "SELECT name, SUM(value) FROM sample GROUP BY 1",
              method: "aggregation",
              param_columns: ["name", "value"],
              expected_datasets: ["sample"],
              params: [],
              source_session_id: null,
              created_at: "2026-07-25T09:00:00Z",
            },
            { status: 201 },
          );
        },
      ),
    );

    const user = userEvent.setup();
    renderAppAt(PAGE_PATH);

    await screen.findByRole("heading", { name: "Skills" });
    const seed = cardFor("Group totals and averages");
    /* Import is gated on the same bindings a replay needs. */
    expect(
      within(seed).getByRole("button", { name: "Import as skill" }),
    ).toBeDisabled();

    await user.selectOptions(
      within(seed).getByRole("combobox", { name: /group_col/ }),
      "name",
    );
    await user.selectOptions(
      within(seed).getByRole("combobox", { name: /value_col/ }),
      "value",
    );
    await user.click(
      within(seed).getByRole("button", { name: "Import as skill" }),
    );

    expect(
      await screen.findByText(
        /Imported .Group totals and averages. into the skill library/,
      ),
    ).toBeInTheDocument();
    expect(importedPath).toBe("group_value_comparison");
    expect(importBody).toMatchObject({
      dataset_ids: ["sample"],
      bindings: { group_col: "name", value_col: "value" },
    });
  });

  it("never offers import on a saved library skill", async () => {
    renderAppAt(PAGE_PATH);

    await screen.findByRole("heading", { name: "Skills" });
    const saved = cardFor("Revenue by name");
    expect(
      within(saved).queryByRole("button", { name: "Import as skill" }),
    ).not.toBeInTheDocument();
  });

  it("surfaces a refused import instead of failing silently", async () => {
    server.use(
      http.post("/api/v1/sessions/:sessionId/skills/:seedId/import", () =>
        HttpResponse.json(
          {
            error: {
              code: "binding_invalid",
              message: "Column(s) nope do not exist in the selected dataset(s).",
            },
          },
          { status: 422 },
        ),
      ),
    );

    const user = userEvent.setup();
    renderAppAt(PAGE_PATH);

    await screen.findByRole("heading", { name: "Skills" });
    const seed = cardFor("Group totals and averages");
    await user.selectOptions(
      within(seed).getByRole("combobox", { name: /group_col/ }),
      "name",
    );
    await user.selectOptions(
      within(seed).getByRole("combobox", { name: /value_col/ }),
      "value",
    );
    await user.click(
      within(seed).getByRole("button", { name: "Import as skill" }),
    );

    const alert = await within(seed).findByRole("alert");
    expect(
      within(alert).getByText("binding_invalid"),
    ).toBeInTheDocument();
  });
});

describe("Skills page user templates", () => {
  it("creates a template from the New template form", async () => {
    let createBody: Record<string, unknown> | null = null;
    server.use(
      http.post(
        "/api/v1/projects/:projectId/skill-templates",
        async ({ request }) => {
          createBody = (await request.json()) as Record<string, unknown>;
          return HttpResponse.json(
            {
              template_id: "user_tmpl_new",
              source: "user",
              name: "Top values",
              question: "Which {group_col} leads?",
              sql: "SELECT {group_col} FROM {dataset} LIMIT 5",
              method: "ranking",
              rationale: "A sorted head answers leader questions.",
              params: [{ name: "group_col", role: "dimension", description: "" }],
              when_to_use: "",
              when_not_to_use: "",
            },
            { status: 201 },
          );
        },
      ),
    );

    const user = userEvent.setup();
    renderAppAt(PAGE_PATH);
    await screen.findByRole("heading", { name: "Skills" });

    expect(
      screen.getByRole("button", { name: "Save template" }),
    ).toBeDisabled();

    fireEvent.change(screen.getByRole("textbox", { name: /Template name/ }), {
      target: { value: "Top values" },
    });
    fireEvent.change(
      screen.getByRole("textbox", { name: /Question it answers/ }),
      { target: { value: "Which {group_col} leads?" } },
    );
    fireEvent.change(screen.getByRole("textbox", { name: /^SQL/ }), {
      target: { value: "SELECT {group_col} FROM {dataset} LIMIT 5" },
    });
    fireEvent.change(screen.getByRole("textbox", { name: "Method" }), {
      target: { value: "ranking" },
    });
    fireEvent.change(screen.getByRole("textbox", { name: /Why this method/ }), {
      target: { value: "A sorted head answers leader questions." },
    });
    fireEvent.change(
      screen.getByRole("textbox", { name: "Placeholder 1 name" }),
      { target: { value: "group_col" } },
    );
    await user.selectOptions(
      screen.getByRole("combobox", { name: "Placeholder 1 role" }),
      "dimension",
    );
    await user.click(screen.getByRole("button", { name: "Save template" }));

    expect(
      await screen.findByText(/Saved template .Top values./),
    ).toBeInTheDocument();
    expect(createBody).toMatchObject({
      name: "Top values",
      question: "Which {group_col} leads?",
      sql: "SELECT {group_col} FROM {dataset} LIMIT 5",
      method: "ranking",
      rationale: "A sorted head answers leader questions.",
      params: [{ name: "group_col", role: "dimension" }],
    });
    /* A successful save clears the form for the next template. */
    expect(
      screen.getByRole("textbox", { name: /Template name/ }),
    ).toHaveValue("");
  });

  it("lists only user templates here and deletes one behind a confirmation", async () => {
    let deletedPath: string | null = null;
    server.use(
      http.delete(
        "/api/v1/projects/:projectId/skill-templates/:templateId",
        ({ request }) => {
          deletedPath = new URL(request.url).pathname;
          return new HttpResponse(null, { status: 204 });
        },
      ),
    );

    const user = userEvent.setup();
    renderAppAt(PAGE_PATH);
    await screen.findByRole("heading", { name: "Skills" });
    await screen.findByText("Share per category");

    const card = cardFor("Share per category");
    expect(within(card).getByText("Your template")).toBeInTheDocument();
    /* The builtin template renders once, as a seed card — not again here. */
    expect(
      screen.getAllByRole("listitem").filter((item) =>
        within(item).queryByText("Group totals and averages"),
      ),
    ).toHaveLength(1);

    await user.click(within(card).getByRole("button", { name: "Delete" }));
    expect(
      within(card).getByText(/Delete .Share per category.\?/),
    ).toBeInTheDocument();
    await user.click(
      within(card).getAllByRole("button", { name: "Delete" })[0]!,
    );

    await waitFor(() =>
      expect(deletedPath).toBe(
        "/api/v1/projects/p1/skill-templates/user_tmpl_1",
      ),
    );
  });

  it("imports a bound template and shows the trial-run preview", async () => {
    let importBody: Record<string, unknown> | null = null;
    let importedTemplate: string | null = null;
    server.use(
      http.post(
        "/api/v1/sessions/:sessionId/skill-templates/:templateId/import",
        async ({ request, params }) => {
          importBody = (await request.json()) as Record<string, unknown>;
          importedTemplate = String(params["templateId"]);
          return HttpResponse.json(
            {
              skill: {
                skill_id: "skill_imported_template",
                source: "library",
                name: "Share per category",
                description: "From template 'user_tmpl_1' on sample.",
                question: "What share of value does each name hold?",
                sql: "SELECT name FROM sample",
                method: "windowed share",
                param_columns: ["name", "value"],
                expected_datasets: ["sample"],
                params: [],
                source_session_id: null,
                created_at: "2026-08-25T09:00:00Z",
              },
              row_count: 2,
              columns: ["name", "share"],
              rows_preview: [
                { name: "alpha", share: 0.75 },
                { name: "beta", share: 0.25 },
              ],
              truncated: false,
            },
            { status: 201 },
          );
        },
      ),
    );

    const user = userEvent.setup();
    renderAppAt(PAGE_PATH);
    await screen.findByRole("heading", { name: "Skills" });
    await screen.findByText("Share per category");

    const card = cardFor("Share per category");
    const importButton = within(card).getByRole("button", {
      name: /Import & trial-run/,
    });
    /* Unbound placeholders gate the import, like a replay. */
    expect(importButton).toBeDisabled();

    await user.selectOptions(
      within(card).getByRole("combobox", { name: /group_col/ }),
      "name",
    );
    await user.selectOptions(
      within(card).getByRole("combobox", { name: /value_col/ }),
      "value",
    );
    await user.click(importButton);

    expect(
      await within(card).findByText(
        /Imported .Share per category. into the skill library. Trial run returned 2 row/,
      ),
    ).toBeInTheDocument();
    expect(within(card).getByText("alpha")).toBeInTheDocument();
    expect(within(card).getByText("0.75")).toBeInTheDocument();
    expect(importedTemplate).toBe("user_tmpl_1");
    expect(importBody).toMatchObject({
      dataset_ids: ["sample"],
      bindings: { group_col: "name", value_col: "value" },
    });
  });
});
