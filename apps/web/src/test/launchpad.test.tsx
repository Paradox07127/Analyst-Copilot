import { describe, expect, it } from "vitest";
import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { server } from "./msw/server";
import { defaultSettings } from "./msw/handlers";
import { FakeEventSource } from "./fake-event-source";
import { renderAppAt } from "./render";

const csvFile = () =>
  new File(["id,name\n1,a\n"], "orders.csv", { type: "text/csv" });

const existing = {
  dataset_id: "ds_prior",
  project_id: "p1",
  display_name: "orders_2025.csv",
  original_uri: "projects/p1/uploads/ds_prior/v1/orders_2025.csv",
  format: "csv",
  content_hash: "abc",
  byte_size: 2048,
  row_count: 42,
  schema: [{ name: "order_id", dtype: "VARCHAR" }],
  ingest_status: "ready",
};

describe("Launchpad", () => {
  it("keeps required data primary and launch confirmation visible", async () => {
    renderAppAt("/projects/p1/new-session");

    await screen.findByRole("heading", { name: "New session" });
    expect(
      screen.queryByRole("list", { name: "New session setup" }),
    ).not.toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Project" })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Data" })).toBeInTheDocument();
    expect(
      screen.getByRole("region", { name: "Run analysis" }),
    ).toBeInTheDocument();
    expect(screen.getByLabelText("Business context")).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /^Prediction baseline/ }),
    ).not.toBeInTheDocument();
  });

  it("uses the whole data card as the drop target and explains rejected files", async () => {
    renderAppAt("/projects/p1/new-session");
    const dataCard = await screen.findByRole("region", { name: "Data" });

    fireEvent.dragEnter(dataCard);
    expect(dataCard).toHaveClass("border-primary", "bg-primary/5");

    fireEvent.dragLeave(dataCard);
    expect(dataCard).not.toHaveClass("border-primary", "bg-primary/5");

    fireEvent.drop(dataCard, {
      dataTransfer: {
        files: [new File(["notes"], "notes.txt", { type: "text/plain" })],
      },
    });
    expect(await screen.findByText("Choose CSV files only.")).toBeInTheDocument();
  });

  it("keeps the run panel focused on the current LLM connection", async () => {
    renderAppAt("/projects/p1/new-session");

    await screen.findByRole("heading", { name: "New session" });
    const runPanel = screen.getByRole("region", { name: "Run analysis" });
    expect(await within(runPanel).findByText("Offline")).toBeInTheDocument();
    expect(
      within(runPanel).getByRole("button", {
        name: "Configure LLM in Settings",
      }),
    ).toBeInTheDocument();
    expect(screen.queryByText("For comparison")).not.toBeInTheDocument();
    expect(
      screen.queryByRole("list", { name: "Phases this session will execute" }),
    ).toBeNull();
  });

  it("reports a connected live model without surfacing pricing", async () => {
    server.use(
      http.get("/api/v1/settings", () =>
        HttpResponse.json({
          ...defaultSettings(),
          provider: "deepseek",
          model: "deepseek-chat",
          is_ready_for_live_calls: true,
          status_state: "ready",
          missing_fields: [],
        }),
      ),
    );
    renderAppAt("/projects/p1/new-session");
    const runPanel = await screen.findByRole("region", {
      name: "Run analysis",
    });

    expect(await within(runPanel).findByText("deepseek-chat")).toBeInTheDocument();
    expect(screen.queryByText("Calls a paid model")).not.toBeInTheDocument();
  });

  it("opens Settings in a dialog without leaving the page", async () => {
    const user = userEvent.setup();
    renderAppAt("/projects/p1/new-session");
    await screen.findByRole("heading", { name: "New session" });

    await user.click(
      screen.getByRole("button", { name: "Configure LLM in Settings" }),
    );
    expect(
      await screen.findByRole("dialog", { name: "Settings" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "New session" }),
    ).toBeInTheDocument();
  });

  it("lets a new session stay standalone, use a project, or stage a new one", async () => {
    const user = userEvent.setup();
    renderAppAt("/new-session");

    await screen.findByRole("heading", { name: "New session" });
    const project = await screen.findByRole("combobox", { name: "Project" });
    expect(project).toHaveValue("unfiled");

    // The select renders before the project list resolves; selecting an option
    // that has not arrived yet fails intermittently under a loaded suite.
    await within(project).findByRole("option", { name: "Project p1" });
    await user.selectOptions(project, "p1");
    expect(screen.getByLabelText("Data files (.csv)")).toBeEnabled();

    await user.selectOptions(project, "new");
    const name = screen.getByLabelText("Project name");
    await user.type(name, "Revenue review");
    expect(screen.queryByRole("button", { name: "Create project" })).toBeNull();
    expect(screen.getByRole("button", { name: "Add support docs" })).toBeDisabled();

    await user.upload(screen.getByLabelText("Data files (.csv)"), csvFile());
    expect(name).toBeDisabled();
    expect(screen.getByText("Locked after adding data.")).toBeInTheDocument();
  });

  it("creates a staged project only when the analysis starts", async () => {
    let projectBody: Record<string, unknown> | null = null;
    let jobBody: Record<string, unknown> | null = null;
    server.use(
      http.post("/api/v1/projects", async ({ request }) => {
        const body = (await request.json()) as Record<string, unknown>;
        projectBody = body;
        return HttpResponse.json(
          {
            project_id: String(body["project_id"]),
            name: String(body["name"]),
            session_count: 0,
          },
          { status: 201 },
        );
      }),
      http.post("/api/v1/sessions/:sessionId/jobs", async ({ request, params }) => {
        jobBody = (await request.json()) as Record<string, unknown>;
        return HttpResponse.json(
          {
            job_id: "job_new_project",
            session_id: String(params["sessionId"]),
            status: "queued",
            events_url: "/api/v1/jobs/job_new_project/events",
          },
          { status: 201 },
        );
      }),
    );
    const user = userEvent.setup();
    renderAppAt("/new-session");

    const project = await screen.findByRole("combobox", { name: "Project" });
    await user.selectOptions(project, "new");
    await user.type(screen.getByLabelText("Project name"), "Revenue review");
    await user.upload(screen.getByLabelText("Data files (.csv)"), csvFile());
    expect(projectBody).toBeNull();

    await user.click(screen.getByRole("button", { name: "Run analysis" }));
    await screen.findByRole("heading", { name: "Data Map" });
    expect(projectBody).toEqual({
      project_id: "Revenue review",
      name: "Revenue review",
    });
    expect(jobBody).toMatchObject({
      project_id: "Revenue review",
      datasets: ["ds_orders"],
    });
  });

  it("uses the server-provisioned private bucket without creating it", async () => {
    let createAttempts = 0;
    server.use(
      http.post("/api/v1/projects", () => {
        createAttempts += 1;
        return HttpResponse.json({ error: { code: "unexpected" } }, { status: 500 });
      }),
    );
    renderAppAt("/new-session");

    expect(await screen.findByLabelText("Data files (.csv)")).toBeEnabled();
    expect(createAttempts).toBe(0);
  });

  it("refreshes project data after a successful upload", async () => {
    let listAttempts = 0;
    server.use(
      http.get("/api/v1/projects/p1/uploads", () => {
        listAttempts += 1;
        return HttpResponse.json(
          listAttempts === 1
            ? []
            : [{ ...existing, dataset_id: "ds_orders", display_name: "orders.csv" }],
        );
      }),
    );
    const user = userEvent.setup();
    renderAppAt("/projects/p1/new-session");

    await user.upload(await screen.findByLabelText("Data files (.csv)"), csvFile());

    const inspector = screen.getByRole("complementary", {
      name: "Context Inspector",
    });
    expect(await within(inspector).findByText("orders.csv")).toBeInTheDocument();
    expect(listAttempts).toBeGreaterThanOrEqual(2);
  });

  it("uploads a CSV, starts an idempotent job and hands off to activity", async () => {
    let idempotencyKey: string | null = null;
    let jobBody: Record<string, unknown> | null = null;
    server.use(
      http.post("/api/v1/sessions/:sessionId/jobs", async ({ request, params }) => {
        idempotencyKey = request.headers.get("Idempotency-Key");
        jobBody = (await request.json()) as Record<string, unknown>;
        return HttpResponse.json(
          {
            job_id: "job_1",
            session_id: String(params["sessionId"]),
            status: "queued",
            events_url: "/api/v1/jobs/job_1/events",
          },
          { status: 201 },
        );
      }),
    );

    const user = userEvent.setup();
    renderAppAt("/projects/p1/new-session");

    await screen.findByRole("heading", { name: "New session" });
    const runButton = screen.getByRole("button", { name: "Run analysis" });
    expect(runButton).toBeDisabled();

    await user.upload(screen.getByLabelText("Data files (.csv)"), csvFile());
    expect(await screen.findByText("Saved to project")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Business context"), {
      target: { value: "E-commerce orders" },
    });
    expect(runButton).toBeEnabled();
    await user.click(runButton);

    expect(
      await screen.findByRole("heading", { name: "Data Map" }),
    ).toBeInTheDocument();
    expect(screen.getByText(/Tracking job_1 · session sess_/)).toBeInTheDocument();
    expect(idempotencyKey).toMatch(/^[0-9a-f-]{36}$/);
    expect(jobBody).toMatchObject({
      kind: "auto_eda",
      project_id: "p1",
      datasets: ["ds_orders"],
      business_context: "E-commerce orders",
      llm: "env",
    });
    expect(FakeEventSource.latest().url).toBe("/api/v1/jobs/job_1/events");
  });

  it("deselects a saved fresh upload without deleting project data", async () => {
    let deletes = 0;
    server.use(
      http.delete("/api/v1/projects/:projectId/uploads/:datasetId", () => {
        deletes += 1;
        return HttpResponse.json({ dataset_id: "ds_orders" });
      }),
    );
    const user = userEvent.setup();
    renderAppAt("/projects/p1/new-session");

    await user.upload(
      await screen.findByLabelText("Data files (.csv)"),
      csvFile(),
    );
    const uploaded = await screen.findByRole("checkbox", {
      name: "Exclude orders.csv",
    });
    await user.click(uploaded);

    expect(deletes).toBe(0);
    expect(uploaded).toHaveAccessibleName("Include orders.csv");
    expect(uploaded).not.toBeChecked();
    expect(screen.getByRole("button", { name: "Run analysis" })).toBeDisabled();
  });

  it("replays the same run id and key when an unchanged retry succeeds", async () => {
    const attempts: { sessionId: string; key: string | null }[] = [];
    server.use(
      http.post("/api/v1/sessions/:sessionId/jobs", ({ request, params }) => {
        const sessionId = String(params["sessionId"]);
        attempts.push({
          sessionId,
          key: request.headers.get("Idempotency-Key"),
        });
        if (attempts.length === 1) {
          return HttpResponse.json(
            { error: { code: "internal", message: "Boom." } },
            { status: 500 },
          );
        }
        return HttpResponse.json(
          {
            job_id: "job_1",
            session_id: sessionId,
            status: "queued",
            events_url: "/api/v1/jobs/job_1/events",
          },
          { status: 201 },
        );
      }),
    );

    const user = userEvent.setup();
    renderAppAt("/projects/p1/new-session");
    await user.upload(
      await screen.findByLabelText("Data files (.csv)"),
      csvFile(),
    );
    await screen.findByText("Saved to project");
    await user.click(screen.getByRole("button", { name: "Run analysis" }));
    await screen.findByText("Boom.");
    await user.click(screen.getByRole("button", { name: "Run analysis" }));

    await screen.findByRole("heading", { name: "Data Map" });
    expect(attempts[0]!.sessionId).toBe(attempts[1]!.sessionId);
    expect(attempts[0]!.key).toBe(attempts[1]!.key);
  });

  it("rotates the run id and key after launch inputs change", async () => {
    const attempts: { sessionId: string; key: string | null }[] = [];
    server.use(
      http.post("/api/v1/sessions/:sessionId/jobs", ({ request, params }) => {
        const sessionId = String(params["sessionId"]);
        attempts.push({
          sessionId,
          key: request.headers.get("Idempotency-Key"),
        });
        if (attempts.length === 1) {
          return HttpResponse.json(
            { error: { code: "internal", message: "Boom." } },
            { status: 500 },
          );
        }
        return HttpResponse.json(
          {
            job_id: "job_changed",
            session_id: sessionId,
            status: "queued",
            events_url: "/api/v1/jobs/job_changed/events",
          },
          { status: 201 },
        );
      }),
    );
    const user = userEvent.setup();
    renderAppAt("/projects/p1/new-session");
    await user.upload(await screen.findByLabelText("Data files (.csv)"), csvFile());
    await screen.findByText("Saved to project");

    await user.click(screen.getByRole("button", { name: "Run analysis" }));
    await screen.findByText("Boom.");
    fireEvent.change(screen.getByLabelText("Business context"), {
      target: { value: "Use the revised scope" },
    });
    await user.click(screen.getByRole("button", { name: "Run analysis" }));
    await screen.findByRole("heading", { name: "Data Map" });

    expect(attempts).toHaveLength(2);
    expect(attempts[1]!.sessionId).not.toBe(attempts[0]!.sessionId);
    expect(attempts[1]!.key).not.toBe(attempts[0]!.key);
  });

  it("shows the conflict branch with a jump link", async () => {
    server.use(
      http.post("/api/v1/sessions/:sessionId/jobs", () =>
        HttpResponse.json(
          {
            error: {
              code: "job_conflict",
              message: "Run already has an active job job_9.",
            },
          },
          { status: 409 },
        ),
      ),
    );

    const user = userEvent.setup();
    renderAppAt("/projects/p1/new-session");
    await user.upload(
      await screen.findByLabelText("Data files (.csv)"),
      csvFile(),
    );
    await screen.findByText("Saved to project");
    await user.click(screen.getByRole("button", { name: "Run analysis" }));

    expect(
      await screen.findByText("This session already has an active job."),
    ).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Open session" })).toHaveAttribute(
      "href",
      expect.stringMatching(
        /^\/projects\/p1\/sessions\/sess_.*\/data-map$/,
      ),
    );
  });

  it.each([
    [
      "upload_project_byte_quota",
      "Project upload storage quota is 1024 bytes.",
      /Project storage quota reached/,
      413,
    ],
    [
      "upload_rate_limited",
      "Upload rate limit exceeded. Retry after 30 seconds.",
      /Uploads are being submitted too quickly/,
      429,
    ],
  ])("explains the %s upload rejection", async (code, message, expected, status) => {
    server.use(
      http.post("/api/v1/projects/:projectId/uploads", () =>
        HttpResponse.json(
          { error: { code, message } },
          { status },
        ),
      ),
    );
    const user = userEvent.setup();
    renderAppAt("/projects/p1/new-session");

    await user.upload(
      await screen.findByLabelText("Data files (.csv)"),
      csvFile(),
    );
    expect(await screen.findByText(expected)).toBeInTheDocument();
  });
});

describe("Reusing project data", () => {
  it("select all and clear selection include fresh uploads", async () => {
    const user = userEvent.setup();
    const customerFile = new File(["id,name\n1,b\n"], "customers.csv", {
      type: "text/csv",
    });
    renderAppAt("/projects/p1/new-session");

    await user.upload(await screen.findByLabelText("Data files (.csv)"), [
      csvFile(),
      customerFile,
    ]);
    await waitFor(() =>
      expect(screen.getAllByText("Saved to project")).toHaveLength(2),
    );
    await user.click(screen.getByRole("button", { name: "Clear selection" }));
    expect(screen.getByRole("checkbox", { name: "Include orders.csv" })).not.toBeChecked();
    expect(screen.getByRole("checkbox", { name: "Include customers.csv" })).not.toBeChecked();

    await user.click(screen.getByRole("button", { name: "Select all" }));
    expect(screen.getByRole("checkbox", { name: "Exclude orders.csv" })).toBeChecked();
    expect(screen.getByRole("checkbox", { name: "Exclude customers.csv" })).toBeChecked();
  });

  it("launches from a table an earlier session uploaded", async () => {
    let jobBody: Record<string, unknown> | null = null;
    server.use(
      http.get("/api/v1/projects/:projectId/uploads", () =>
        HttpResponse.json([existing]),
      ),
      http.post("/api/v1/sessions/:sessionId/jobs", async ({ request, params }) => {
        jobBody = (await request.json()) as Record<string, unknown>;
        return HttpResponse.json(
          {
            job_id: "job_1",
            session_id: String(params["sessionId"]),
            status: "queued",
            events_url: "/api/v1/jobs/job_1/events",
          },
          { status: 201 },
        );
      }),
    );

    const user = userEvent.setup();
    renderAppAt("/projects/p1/new-session");
    const file = await screen.findByRole("checkbox", {
      name: "Include orders_2025.csv",
    });
    expect(screen.getByRole("button", { name: "Run analysis" })).toBeDisabled();

    await user.click(file);
    expect(file).toHaveAccessibleName("Exclude orders_2025.csv");
    await user.click(screen.getByRole("button", { name: "Run analysis" }));

    await waitFor(() => expect(jobBody).not.toBeNull());
    expect(jobBody).toMatchObject({ datasets: ["ds_prior"] });
  });

  it("shows an unreadable stored file without making it selectable", async () => {
    server.use(
      http.get("/api/v1/projects/:projectId/uploads", () =>
        HttpResponse.json([
          {
            ...existing,
            dataset_id: "ds_unreadable",
            display_name: "broken.csv",
            schema: [],
            ingest_status: "unreadable",
          },
        ]),
      ),
    );

    renderAppAt("/projects/p1/new-session");
    const file = await screen.findByRole("checkbox", {
      name: "Include broken.csv",
    });

    expect(file).toBeDisabled();
    expect(
      screen.getByText("Stored file is missing or still ingesting"),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run analysis" })).toBeDisabled();
  });

  it("keeps a long project file list bounded and summarizes selection", async () => {
    const files = [
      "orders.csv",
      "customers.csv",
      "products.csv",
      "regions.csv",
    ].map((display_name, index) => ({
      ...existing,
      dataset_id: `ds_${index}`,
      display_name,
    }));
    server.use(
      http.get("/api/v1/projects/:projectId/uploads", () =>
        HttpResponse.json(files),
      ),
    );
    const user = userEvent.setup();
    renderAppAt("/projects/p1/new-session");

    const list = await screen.findByRole("list", {
      name: "Data files in this session",
    });
    expect(list).toHaveClass(
      "max-h-[min(13rem,28dvh)]",
      "overflow-y-auto",
    );
    for (const file of files) {
      await user.click(
        screen.getByRole("checkbox", {
          name: `Include ${file.display_name}`,
        }),
      );
    }

    const runPanel = screen.getByRole("region", { name: "Run analysis" });
    expect(within(runPanel).getByText("4 files")).toBeInTheDocument();
  });

  it("selects or clears all reusable project files in one action", async () => {
    const files = ["orders.csv", "customers.csv", "products.csv"].map(
      (display_name, index) => ({
        ...existing,
        dataset_id: `ds_${index}`,
        display_name,
      }),
    );
    server.use(
      http.get("/api/v1/projects/:projectId/uploads", () =>
        HttpResponse.json(files),
      ),
    );
    const user = userEvent.setup();
    renderAppAt("/projects/p1/new-session");

    await user.click(await screen.findByRole("button", { name: "Select all" }));
    for (const file of files) {
      expect(screen.getByRole("checkbox", { name: `Exclude ${file.display_name}` })).toBeChecked();
    }
    expect(screen.getByRole("button", { name: "Run analysis" })).toBeEnabled();

    await user.click(screen.getByRole("button", { name: "Clear selection" }));
    for (const file of files) {
      expect(screen.getByRole("checkbox", { name: `Include ${file.display_name}` })).not.toBeChecked();
    }
  });

  it("deselects a project file without deleting it", async () => {
    let deletes = 0;
    server.use(
      http.get("/api/v1/projects/:projectId/uploads", () =>
        HttpResponse.json([existing]),
      ),
      http.delete("/api/v1/projects/:projectId/uploads/:datasetId", () => {
        deletes += 1;
        return HttpResponse.json({ dataset_id: existing.dataset_id });
      }),
    );
    const user = userEvent.setup();
    renderAppAt("/projects/p1/new-session");

    const file = await screen.findByRole("checkbox", {
      name: "Include orders_2025.csv",
    });
    await user.click(file);
    await user.click(file);

    expect(deletes).toBe(0);
    expect(file).not.toBeChecked();
    expect(screen.getByRole("button", { name: "Run analysis" })).toBeDisabled();
  });

  it("clears reused selection when the destination changes", async () => {
    server.use(
      http.get("/api/v1/projects", () =>
        HttpResponse.json([
          { project_id: "p1", name: "Project p1", session_count: 1 },
          { project_id: "p2", name: "Project p2", session_count: 0 },
        ]),
      ),
      http.get("/api/v1/projects/:projectId/uploads", ({ params }) =>
        HttpResponse.json([
          {
            ...existing,
            project_id: String(params["projectId"]),
            display_name:
              params["projectId"] === "p1" ? "orders.csv" : "customers.csv",
          },
        ]),
      ),
    );
    const user = userEvent.setup();
    renderAppAt("/new-session");
    const project = await screen.findByRole("combobox", { name: "Project" });
    await screen.findByRole("option", { name: "Project p1" });

    await user.selectOptions(project, "p1");
    await user.click(
      await screen.findByRole("checkbox", { name: "Include orders.csv" }),
    );
    expect(screen.getByRole("button", { name: "Run analysis" })).toBeEnabled();

    await user.selectOptions(project, "p2");
    const second = await screen.findByRole("checkbox", {
      name: "Include customers.csv",
    });
    expect(second).not.toBeChecked();
    expect(screen.getByRole("button", { name: "Run analysis" })).toBeDisabled();
  });
});
