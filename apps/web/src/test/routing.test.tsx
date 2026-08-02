import { describe, expect, it } from "vitest";
import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { isDynamicImportFailure } from "../app/router";
import { renderAppAt, renderAppWithRouterAt } from "./render";

const RUN_DEEP_LINKS = [
  ["data-map", "Data map", "Data Map"],
  ["table/sample", "Table preview", "Table Preview"],
  ["quality", "Quality", "Quality"],
  ["profiles", "Profiles & charts", "Profiles & charts"],
  ["charts", "Profiles & charts", "Profiles & charts"],
  ["relationships", "Relationships", "Relationships"],
  ["questions", "Questions", "Questions"],
  ["findings", "Findings", "Findings"],
  ["semantic", "Knowledge", "Knowledge"],
  ["cleaning", "Cleanup", "Cleanup"],
  ["deep-analysis", "Deep analysis", "Deep analysis"],
  ["trace", "Trace & cost", "Trace & cost"],
  ["report", "Report", "Report"],
  ["artifacts", "Artifacts", "Artifacts"],
  ["skills", "Skills", "Skills"],
  ["chat", "Chat", "Chat"],
  ["board", "Board", "Investigation board"],
] as const;

describe("Routing", () => {
  it("recognizes stale lazy-route chunks so the shell can recover once", () => {
    expect(
      isDynamicImportFailure(
        new TypeError("Failed to fetch dynamically imported module: /assets/TablePreviewPage-old.js"),
      ),
    ).toBe(true);
    expect(isDynamicImportFailure(new Error("Request failed (not_found)"))).toBe(false);
  });

  it.each([
    ["/settings", "Settings", "—", "no session open"],
    ["/projects/p1/new-session", "New session", "p1", "no session open"],
    ["/new-session", "New session", "—", "no session open"],
  ])(
    "restores the shell context on a hard refresh of %s",
    async (path, heading, project, run) => {
      renderAppAt(path);
      expect(
        await screen.findByRole("heading", { name: heading, level: 1 }),
      ).toBeInTheDocument();
      const context = await screen.findByRole("navigation", {
        name: "Context",
      });
      expect(within(context).getByText(project)).toBeInTheDocument();
      expect(within(context).getByText(run)).toBeInTheDocument();
    },
  );

  it("uses a clean Home context without an empty session placeholder", async () => {
    renderAppAt("/projects");
    expect(
      await screen.findByRole("heading", { name: "Overview", level: 1 }),
    ).toBeInTheDocument();
    const context = await screen.findByRole("navigation", {
      name: "Context",
    });
    expect(within(context).getByText("Home")).toBeInTheDocument();
    expect(within(context).queryByText("no session open")).toBeNull();
    expect(screen.queryByRole("button", { name: "Inspector" })).toBeNull();
    expect(
      screen.queryByRole("complementary", { name: "Context Inspector" }),
    ).toBeNull();
    expect(
      screen.queryByRole("dialog", { name: "Activity" }),
    ).toBeNull();
  });

  it.each(RUN_DEEP_LINKS)(
    "restores run shell navigation and URL context on hard refresh of %s",
    async (section, currentLabel, heading) => {
      renderAppAt(`/projects/p1/sessions/r1/${section}`);

      expect(
        await screen.findByRole("heading", { name: heading, level: 1 }),
      ).toBeInTheDocument();
      const context = await screen.findByRole("navigation", {
        name: "Context",
      });
      expect(within(context).getByText("p1")).toBeInTheDocument();
      expect(within(context).getByText("Demo run")).toBeInTheDocument();
      expect(within(context).getByText("r1")).toBeInTheDocument();

      const runNavigation = await screen.findByRole("navigation", {
        name: "Session sections",
      });
      expect(
        await within(runNavigation).findByText(currentLabel),
      ).toHaveTextContent(currentLabel);
      expect(
        within(runNavigation).getByText(currentLabel).closest(
          '[aria-current="page"]',
        ),
      ).not.toBeNull();
    },
  );

  it("keeps /settings reachable directly and from the top bar", async () => {
    renderAppAt("/settings");
    expect(
      await screen.findByRole("heading", { name: "Settings", level: 1 }),
    ).toBeInTheDocument();
    const rail = screen.getByRole("complementary", { name: "Sessions" });
    expect(within(rail).queryByRole("link", { name: "Settings" })).toBeNull();
    expect(within(rail).queryByRole("button", { name: "Settings" })).toBeNull();
    /* Exactly one settings entry in the app, and it is the one that survives
     * collapsing the rail. */
    expect(screen.getAllByRole("button", { name: "Settings" })).toHaveLength(1);
  });

  it("redirects / to the project list", async () => {
    renderAppAt("/");
    expect(
      await screen.findByRole("heading", { name: "Overview" }),
    ).toBeInTheDocument();
  });

  it("redirects a bare run URL to data-map", async () => {
    renderAppAt("/projects/p1/sessions/r1");
    expect(
      await screen.findByRole("heading", { name: "Data Map" }),
    ).toBeInTheDocument();
  });

  it("survives the bare-run redirect for ids needing URL encoding", async () => {
    renderAppAt(`/projects/${encodeURIComponent("sales#2026")}/sessions/r1`);
    expect(
      await screen.findByRole("heading", { name: "Data Map" }),
    ).toBeInTheDocument();
    expect(screen.getByText("sales#2026")).toBeInTheDocument();
  });

  it("upgrades a legacy session split URL to the shell-level workspace", async () => {
    const { router } = renderAppWithRouterAt(
      "/projects/p1/sessions/r1/compare?right=r2&mode=split&leftSection=questions&rightSection=report",
    );

    await waitFor(() =>
      expect(router.state.location.pathname).toBe("/split"),
    );
    expect(
      await screen.findByRole("region", { name: "Left workspace pane" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("region", { name: "Right workspace pane" }),
    ).toBeInTheDocument();
    const params = new URLSearchParams(router.state.location.search);
    expect(params.get("left")).toBe("/projects/p1/sessions/r1/questions");
    expect(params.get("right")).toBe("/projects/p1/sessions/r2/report");
    expect(params.get("active")).toBe("left");
  });

  it("navigates between run sections via the grouped top navigation", async () => {
    const user = userEvent.setup();
    renderAppAt("/projects/p1/sessions/r1/data-map");
    await screen.findByRole("heading", { name: "Data Map" });
    const understand = () =>
      screen.getByRole("button", { name: "Understand the data" });

    await user.click(understand());
    await user.click(screen.getByRole("link", { name: "Relationships" }));
    expect(
      await screen.findByRole("heading", { name: "Relationships" }),
    ).toBeInTheDocument();

    // The Table preview link appears once the run's datasets load (MSW).
    await user.click(understand());
    await user.click(await screen.findByRole("link", { name: "Table preview" }));
    expect(
      await screen.findByRole("heading", { name: "Table Preview" }),
    ).toBeInTheDocument();
    expect(screen.getByLabelText("Table")).toHaveValue("sample");
  });

  it("keeps Table preview active for a non-default dataset route", async () => {
    renderAppAt("/projects/p1/sessions/r1/table/another-dataset");
    expect(
      await screen.findByRole("heading", { name: "Table Preview", level: 1 }),
    ).toBeInTheDocument();

    const navigation = await screen.findByRole("navigation", {
      name: "Session sections",
    });
    expect(
      within(navigation).getByRole("link", { name: "Table preview" }),
    ).toHaveAttribute("aria-current", "page");
  });
});
