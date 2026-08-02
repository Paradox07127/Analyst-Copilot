import { describe, expect, it } from "vitest";
import { fireEvent, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { server } from "./msw/server";
import { renderAppAt } from "./render";

/* fireEvent, not user.type: react-resizable-panels' document-level pointerdown
 * handler preventDefaults clicks at jsdom's all-zero rects, so user-event never
 * focuses the input (same workaround as chat.test). */
async function fillNewProject(name: string, projectId?: string) {
  /* The form is disclosed on demand now; the fixture workspace has projects,
   * so the list renders and the form starts closed. */
  fireEvent.click(await screen.findByRole("button", { name: "New project" }));
  fireEvent.change(await screen.findByLabelText("Name"), {
    target: { value: name },
  });
  if (projectId !== undefined) {
    fireEvent.change(screen.getByLabelText("Project id"), {
      target: { value: projectId },
    });
  }
}

describe("Project list with real API", () => {
  it("keeps overview, recent work, and projects in one aligned top row", async () => {
    renderAppAt("/projects");

    const overview = await screen.findByRole("heading", {
      level: 1,
      name: "Overview",
    });
    expect(
      overview.closest('[data-home-layout="workspace"]'),
    ).toHaveClass("lg:grid-cols-5");
    expect(
      overview.closest('[data-home-column="overview"]'),
    ).toHaveClass("lg:col-span-2");
    expect(
      overview
        .closest('[data-home-column="overview"]')
        ?.querySelector("[data-dashboard-metrics]"),
    ).toHaveClass("md:grid-cols-4");
    expect(
      overview.closest('[data-home-column="overview"]'),
    ).not.toContainElement(
      screen.getByRole("heading", { name: "Start an analysis" }),
    );

    const recent = screen.getByRole("complementary", {
      name: "Recent projects and sessions",
    });
    expect(recent).toHaveClass("w-full", "lg:col-span-3", "lg:grid-cols-2");
    expect(recent).not.toContainElement(
      screen.getByRole("heading", { name: "Start an analysis" }),
    );
    expect(overview.closest('[data-home-layout="workspace"]')).toContainElement(
      recent,
    );
    expect(overview.closest('[data-home-layout="workspace"]')?.parentElement).toContainElement(
      screen.getByRole("heading", { name: "Start an analysis" }),
    );
    expect(recent).toContainElement(
      screen.getByRole("heading", { name: "Recent work" }),
    );
    expect(recent).toContainElement(
      screen.getByRole("heading", { name: "Projects" }),
    );
    const activity = screen.getByRole("img", {
      name: /Active sessions per day over the last 180 days/,
    });
    expect(activity).toHaveClass("grid", "w-full", "auto-cols-fr");
    expect(activity.querySelectorAll("[title]")).not.toHaveLength(0);
    expect(
      Array.from(activity.querySelectorAll("[title]")).every((node) =>
        node.closest('[aria-hidden="true"]'),
      ),
    ).toBe(true);
    expect(screen.queryByText(/last 180 days/)).not.toBeInTheDocument();
    expect(screen.queryByText(/no recorded cost/)).not.toBeInTheDocument();
  });

  it("renders project cards with session counts, latest session link and New session entry", async () => {
    renderAppAt("/projects");

    // level 2 = the card heading. The rail's project header is a button,
    // not a heading, so it cannot collide with this query.
    expect(
      await screen.findByRole("heading", { level: 2, name: "Project p1" }),
    ).toBeInTheDocument();
    expect(screen.getByText("1 session")).toBeInTheDocument();

    const latest = await screen.findByRole("link", { name: "Latest: Demo run" });
    expect(latest).toHaveAttribute("href", "/projects/p1/sessions/r1");

    expect(screen.getAllByRole("link", { name: "New session in Project p1" }).at(-1)).toHaveAttribute(
      "href",
      "/projects/p1/new-session",
    );
    // The 3a-era hardcoded demo link is gone.
    expect(screen.queryByText("Open demo run")).not.toBeInTheDocument();
  });

  it("defaults workspace activity to 180 days and keeps 30d/7d switching", async () => {
    const requestedDays: string[] = [];
    server.use(
      http.get("/api/v1/usage", ({ request }) => {
        const days = new URL(request.url).searchParams.get("days") ?? "";
        requestedDays.push(days);
        return HttpResponse.json({
          schema_version: 1,
          generated_at: "2026-07-29T12:00:00Z",
          window_days: Number(days),
          project_count: 1,
          session_count: 1,
          status_counts: { complete: 1 },
          daily: Array.from({ length: Number(days) }, (_, index) => ({
            date: `2026-07-${String(index + 1).padStart(2, "0")}`,
            sessions: 0,
          })),
          llm_calls: 1,
          total_tokens: 10,
          est_cost_usd: 0.001,
          priced_sessions: 1,
          unpriced_sessions: 0,
          artifact_count: 1,
          dataset_count: 1,
          profiled_rows: 250,
          data_bytes: 2048,
          truncated_sessions: 0,
          recent: [],
        });
      }),
    );
    const user = userEvent.setup();
    renderAppAt("/projects");

    expect(
      await screen.findByRole("button", { name: "180d", pressed: true }),
    ).toBeInTheDocument();
    expect(requestedDays).toContain("180");

    await user.click(screen.getByRole("button", { name: "30d" }));
    expect(
      await screen.findByRole("button", { name: "30d", pressed: true }),
    ).toBeInTheDocument();
    expect(requestedDays).toContain("30");
    expect(
      screen.getByRole("img", {
        name: /Active sessions per day over the last 180 days/,
      }),
    ).toHaveClass("w-full", "auto-cols-fr");
    expect(
      screen.queryByRole("img", {
        name: /Active sessions per day over the last 30 days/,
      }),
    ).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "7d" }));
    expect(
      await screen.findByRole("button", { name: "7d", pressed: true }),
    ).toBeInTheDocument();
    expect(requestedDays).toContain("7");
    expect(
      screen.getByRole("img", {
        name: /Active sessions per day over the last 180 days/,
      }),
    ).toHaveClass("w-full", "auto-cols-fr");
  });

  it("keeps standalone history visible when there are no projects", async () => {
    server.use(
      http.get("/api/v1/projects", () => HttpResponse.json([])),
      http.get("/api/v1/usage", () =>
        HttpResponse.json({
          schema_version: 1,
          generated_at: "2026-07-29T12:00:00Z",
          window_days: 180,
          project_count: 0,
          session_count: 1,
          status_counts: { complete: 1 },
          daily: [{ date: "2026-07-29", sessions: 1 }],
          llm_calls: 1,
          total_tokens: 10,
          est_cost_usd: 0.001,
          priced_sessions: 1,
          unpriced_sessions: 0,
          artifact_count: 1,
          dataset_count: 1,
          profiled_rows: 250,
          data_bytes: 2048,
          truncated_sessions: 0,
          recent: [
            {
              session_id: "solo",
              project_id: "unfiled-sessions",
              title: "Standalone analysis",
              status: "complete",
              created_at: "2026-07-28T12:00:00Z",
              updated_at: "2026-07-29T12:00:00Z",
            },
          ],
        }),
      ),
    );
    renderAppAt("/projects");

    expect(
      await screen.findByRole("link", { name: /Standalone analysis/ }),
    ).toHaveAttribute(
      "href",
      "/projects/unfiled-sessions/sessions/solo",
    );
    expect(screen.getByText("No projects yet")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Overview" })).toBeInTheDocument();
    expect(screen.queryByRole("textbox", { name: "Name" })).toBeNull();
  });
});

describe("New project form", () => {
  it("derives an id from the name, creates the project and lands on its new-run page", async () => {
    let posted: Record<string, unknown> | null = null;
    server.use(
      http.post("/api/v1/projects", async ({ request }) => {
        posted = (await request.json()) as Record<string, unknown>;
        return HttpResponse.json(
          { project_id: "Brazilian E-Commerce", name: "Brazilian E-Commerce", session_count: 0 },
          { status: 201 },
        );
      }),
    );
    const user = userEvent.setup();
    renderAppAt("/projects");

    await fillNewProject("Brazilian E-Commerce!");
    // The id is derived from the name with server-rejected characters dropped.
    expect(screen.getByLabelText("Project id")).toHaveValue(
      "Brazilian E-Commerce",
    );

    await user.click(screen.getByRole("button", { name: "Create project" }));

    expect(await screen.findByRole("heading", { name: "New session" })).toBeInTheDocument();
    expect(posted).toEqual({
      project_id: "Brazilian E-Commerce",
      name: "Brazilian E-Commerce!",
    });
  });

  it("keeps a hand-edited id instead of re-deriving it from the name", async () => {
    renderAppAt("/projects");

    await fillNewProject("Sales", "sales_2026");
    fireEvent.change(screen.getByLabelText("Name"), {
      target: { value: "Sales Report" },
    });

    expect(screen.getByLabelText("Project id")).toHaveValue("sales_2026");
  });

  it("surfaces a 409 case conflict without navigating away", async () => {
    server.use(
      http.post("/api/v1/projects", () =>
        HttpResponse.json(
          {
            error: {
              code: "project_conflict",
              message:
                "Project id 'Demo' only differs in case from the existing project 'demo'.",
            },
          },
          { status: 409 },
        ),
      ),
    );
    const user = userEvent.setup();
    renderAppAt("/projects");

    await fillNewProject("Demo");
    await user.click(screen.getByRole("button", { name: "Create project" }));

    const alert = (await screen.findByText(
      "That id clashes with an existing project.",
    )).closest('[role="alert"]') as HTMLElement;
    expect(alert).toHaveTextContent("That id clashes with an existing project.");
    expect(alert).toHaveTextContent("only differs in case");
    // Still on the project list, not the new-run page.
    expect(screen.getByRole("heading", { level: 1, name: "Overview" })).toBeInTheDocument();
  });

  it("surfaces a 422 invalid id", async () => {
    server.use(
      http.post("/api/v1/projects", () =>
        HttpResponse.json(
          {
            error: {
              code: "project_invalid",
              message: "project_id must be a single path segment.",
            },
          },
          { status: 422 },
        ),
      ),
    );
    const user = userEvent.setup();
    renderAppAt("/projects");

    await fillNewProject("ok");
    await user.click(screen.getByRole("button", { name: "Create project" }));

    const alert = (await screen.findByText(
      "Could not create the project.",
    )).closest('[role="alert"]') as HTMLElement;
    expect(alert).toHaveTextContent("Could not create the project.");
    expect(alert).toHaveTextContent("single path segment");
  });
});
