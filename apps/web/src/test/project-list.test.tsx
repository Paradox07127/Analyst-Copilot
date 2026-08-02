import { describe, expect, it } from "vitest";
import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { server } from "./msw/server";
import { renderAppAt } from "./render";

describe("Project list with real API", () => {
  it("keeps overview, recent work, and projects in one aligned top row", async () => {
    renderAppAt("/projects");

    const overview = await screen.findByRole("heading", {
      level: 1,
      name: "Overview",
    });
    expect(
      overview.closest('[data-home-layout="workspace"]'),
    ).toHaveClass("lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]");
    expect(
      overview.closest('[data-home-column="overview"]'),
    ).toHaveClass("flex", "min-w-0");
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
    expect(recent).toHaveClass("min-w-0");
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
    expect(activity).toHaveClass("grid", "grid-flow-col", "grid-rows-7", "auto-cols-[var(--activity-cell-size)]", "gap-1");
    expect(activity).toHaveAttribute("data-activity-window", "180");
    expect(activity).toHaveAttribute("data-activity-cell-size", "12");
    expect(activity.querySelector("[title]")).toHaveClass("size-[var(--activity-cell-size)]");
    expect(activity.querySelectorAll("[title]")).not.toHaveLength(0);
    expect(
      Array.from(activity.querySelectorAll("[title]")).every(
        (node) => node.tagName === "SPAN",
      ),
    ).toBe(true);
    expect(screen.queryByText(/last 180 days/)).not.toBeInTheDocument();
    expect(screen.queryByText(/no recorded cost/)).not.toBeInTheDocument();
  });

  it("renders compact project destinations without repeating recent sessions", async () => {
    renderAppAt("/projects");

    expect((await screen.findAllByText("Project p1")).length).toBeGreaterThan(0);
    expect(screen.getByText("1 session")).toBeInTheDocument();

    const latest = await screen.findByRole("link", { name: /Demo run/ });
    expect(latest).toHaveAttribute("href", "/projects/p1/sessions/r1");

    expect(screen.getAllByRole("link", { name: "New session in Project p1" }).at(-1)).toHaveAttribute(
      "href",
      "/projects/p1/new-session",
    );
    expect(screen.queryByText("Latest: Demo run")).not.toBeInTheDocument();
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
        name: /Active sessions per day over the last 30 days/,
      }),
    ).toHaveClass("grid-flow-col", "grid-rows-7", "auto-cols-[var(--activity-cell-size)]", "gap-1");
    expect(
      screen.getByRole("img", {
        name: /Active sessions per day over the last 30 days/,
      }),
    ).toHaveAttribute("data-activity-cell-size", "20");
    expect(
      screen.queryByRole("img", {
        name: /Active sessions per day over the last 180 days/,
      }),
    ).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "7d" }));
    expect(
      await screen.findByRole("button", { name: "7d", pressed: true }),
    ).toBeInTheDocument();
    expect(requestedDays).toContain("7");
    expect(
      screen.getByRole("img", {
        name: /Active sessions per day over the last 7 days/,
      }),
    ).toHaveClass("grid-cols-[repeat(7,var(--activity-cell-size))]", "gap-1");
    expect(
      screen.getByRole("img", {
        name: /Active sessions per day over the last 7 days/,
      }),
    ).toHaveAttribute("data-activity-cell-size", "20");
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
    expect(screen.getByText(/No projects yet/)).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Overview" })).toBeInTheDocument();
    expect(screen.queryByRole("textbox", { name: "Name" })).toBeNull();
  });

  it("shows only the six most recently updated sessions", async () => {
    server.use(
      http.get("/api/v1/usage", ({ request }) => {
        const days = Number(new URL(request.url).searchParams.get("days") ?? 180);
        return HttpResponse.json({
          schema_version: 1,
          generated_at: "2026-08-02T12:00:00Z",
          window_days: days,
          project_count: 1,
          session_count: 7,
          status_counts: { complete: 7 },
          daily: Array.from({ length: days }, (_, index) => ({
            date: `2026-01-${String(index + 1).padStart(2, "0")}`,
            sessions: 0,
          })),
          llm_calls: 0,
          total_tokens: 0,
          est_cost_usd: 0,
          priced_sessions: 0,
          unpriced_sessions: 7,
          artifact_count: 0,
          dataset_count: 7,
          profiled_rows: 0,
          data_bytes: 0,
          truncated_sessions: 0,
          recent: Array.from({ length: 7 }, (_, index) => ({
            session_id: `recent-${index + 1}`,
            project_id: "p1",
            title: `Recent session ${index + 1}`,
            status: "complete",
            created_at: `2026-08-0${Math.max(1, 7 - index)}T12:00:00Z`,
            updated_at: `2026-08-0${Math.max(1, 7 - index)}T12:00:00Z`,
          })),
        });
      }),
    );
    renderAppAt("/projects");

    const recentHeading = await screen.findByRole("heading", {
      name: "Recent work",
    });
    const recent = recentHeading.closest("section") as HTMLElement;
    expect(within(recent).getAllByRole("link")).toHaveLength(6);
    expect(within(recent).queryByText("Recent session 7")).not.toBeInTheDocument();
  });
});
