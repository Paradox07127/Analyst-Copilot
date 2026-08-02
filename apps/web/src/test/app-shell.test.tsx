import { describe, expect, it } from "vitest";
import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { server } from "./msw/server";
import { renderAppAt } from "./render";

/* Session nav section titles and page order, including React-only pages
 * (Board; Trace & cost + Artifacts historically served as one
 * "Trace & evidence" page) appended to the group they belong to. */
const GROUPS: Record<string, string[]> = {
  "Understand the data": [
    "Data map",
    "Table preview",
    "Quality",
    "Profiles & charts",
    "Cleaning info and raw data",
    "Relationships",
    "Knowledge",
  ],
  "Investigate with the agent": [
    "Questions",
    "Deep analysis",
    "Findings",
    "Compare",
    "Chat",
    "Skills",
    "Report",
    "Board",
  ],
  "Trust & trace": ["Trace & cost", "Artifacts"],
};

async function chooseStage(
  user: ReturnType<typeof userEvent.setup>,
  title: string,
) {
  const navigation = screen.getByRole("navigation", {
    name: "Session sections",
  });
  await user.click(within(navigation).getByRole("button", { expanded: false }));
  await user.click(
    within(navigation).getByRole("button", { name: `Show ${title} pages` }),
  );
}

describe("App Shell", () => {
  it("answers grain and readiness in the inspector on data pages", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/artifacts", ({ request }) => {
        const url = new URL(request.url);
        if (url.searchParams.get("type") !== "EdaHandoff") {
          return HttpResponse.json({ items: [], next_cursor: null });
        }
        return HttpResponse.json({
          items: [{ artifact_id: "h1", type: "EdaHandoff", created_at: null }],
          next_cursor: null,
        });
      }),
      http.get("/api/v1/sessions/:sessionId/artifacts/h1", () =>
        HttpResponse.json({
          artifact_id: "h1",
          type: "EdaHandoff",
          project_id: "p1",
          session_id: "r1",
          created_at: "2026-07-31T12:00:00Z",
          payload: {
            datasets: [
              {
                name: "visits.csv",
                grain: "One row per unique (customer_id + visit_date) combination.",
                analysis_ready: false,
                quality: { material_codes: ["id_not_unique"] },
                pii_columns: { email: "email" },
              },
            ],
          },
          warnings: [],
        }),
      ),
    );

    renderAppAt("/projects/p1/sessions/r1/data-map");
    const inspector = await screen.findByRole("complementary", {
      name: "Context Inspector",
    });

    expect(
      await within(inspector).findByText(
        "One row per unique (customer_id + visit_date) combination.",
      ),
    ).toBeInTheDocument();
    expect(within(inspector).getByText("limited")).toBeInTheDocument();
    expect(within(inspector).getByText("id_not_unique")).toBeInTheDocument();
    expect(
      within(inspector).getByText("1 PII column masked in shared artifacts"),
    ).toBeInTheDocument();
  });

  it("renders the workbench with a floating Activity launcher", async () => {
    renderAppAt("/projects/p1/sessions/r1/data-map");

    expect(
      await screen.findByRole("heading", { name: "Data Map" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("banner")).toBeInTheDocument();
    expect(
      screen.getByRole("complementary", { name: "Sessions" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("complementary", { name: "Sessions" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "Context Inspector" }),
    ).toBeInTheDocument();
    const inspector = screen.getByRole("complementary", {
      name: "Context Inspector",
    });
    expect(await within(inspector).findByText("Demo run")).toBeInTheDocument();
    expect(within(inspector).getByText("Data map")).toBeInTheDocument();
    expect(within(inspector).getByText("sample · complete")).toBeInTheDocument();
    expect(within(inspector).getByText("3 artifacts")).toBeInTheDocument();
    expect(
      within(inspector).getByRole("link", { name: "Open trace & cost" }),
    ).toHaveAttribute("href", "/projects/p1/sessions/r1/trace");
    expect(
      screen.getByRole("button", { name: "Open activity" }),
    ).toHaveAttribute("aria-expanded", "false");
    expect(screen.getByText("p1")).toBeInTheDocument();
    expect(screen.getByText("r1")).toBeInTheDocument();
  });

  it("keeps page navigation out of the session rail", async () => {
    renderAppAt("/projects/p1/sessions/r1/data-map");
    await screen.findByRole("heading", { name: "Data Map" });
    const rail = screen.getByRole("complementary", { name: "Sessions" });

    for (const label of ["Quality", "Data map", "Findings", "Report"]) {
      expect(
        within(rail).queryByRole("link", { name: label }),
      ).not.toBeInTheDocument();
    }
    /* The rail keeps its own global entries and the session list. */
    expect(
      within(rail).getByRole("link", { name: "Home" }),
    ).toBeInTheDocument();
    expect(await within(rail).findByText("Demo run")).toBeInTheDocument();
  });

  it("toggles the Activity panel and persists the choice", async () => {
    const user = userEvent.setup();
    renderAppAt("/projects/p1/sessions/r1/data-map");
    await screen.findByRole("heading", { name: "Data Map" });

    await user.click(screen.getByRole("button", { name: "Open activity" }));
    expect(
      screen.getByRole("dialog", { name: "Activity" }),
    ).toBeInTheDocument();
    expect(window.localStorage.getItem("eda.layout.activity-open")).toBe("true");
  });

  it("drags the launcher within the viewport and remembers its position", async () => {
    const pointerDescriptor = Object.getOwnPropertyDescriptor(
      window,
      "PointerEvent",
    );
    Object.defineProperty(window, "PointerEvent", {
      configurable: true,
      value: MouseEvent,
    });
    try {
      renderAppAt("/projects/p1/sessions/r1/data-map");
      await screen.findByRole("heading", { name: "Data Map" });
      const launcher = screen.getByRole("button", { name: "Open activity" });

      fireEvent.pointerDown(launcher, {
        button: 0,
        clientX: 980,
        clientY: 720,
      });
      fireEvent.pointerMove(launcher, {
        clientX: 620,
        clientY: 420,
      });
      fireEvent.pointerUp(launcher, {
        clientX: 620,
        clientY: 420,
      });
      fireEvent.click(launcher);

      expect(screen.queryByRole("dialog", { name: "Activity" })).toBeNull();
      expect(
        JSON.parse(
          window.localStorage.getItem("eda.layout.activity-position") ?? "{}",
        ),
      ).toMatchObject({ x: expect.any(Number), y: expect.any(Number) });
    } finally {
      if (pointerDescriptor) {
        Object.defineProperty(window, "PointerEvent", pointerDescriptor);
      } else {
        Reflect.deleteProperty(window, "PointerEvent");
      }
    }
  });

  it("can hide and restore the floating launcher from the panel", async () => {
    const user = userEvent.setup();
    renderAppAt("/projects/p1/sessions/r1/data-map");
    await screen.findByRole("heading", { name: "Data Map" });

    await user.click(screen.getByRole("button", { name: "Open activity" }));
    await user.click(
      within(screen.getByRole("dialog", { name: "Activity" })).getByRole(
        "button",
        { name: "Hide floating button" },
      ),
    );

    expect(
      screen.queryByRole("button", { name: "Open activity" }),
    ).not.toBeInTheDocument();
    expect(window.localStorage.getItem("eda.layout.activity-launcher")).toBe(
      "false",
    );
    const reopened = screen.getByRole("dialog", { name: "Activity" });
    await user.click(
      within(reopened).getByRole("button", { name: "Show floating button" }),
    );
    expect(
      screen.getByRole("button", {
        name: "Close activity from floating button",
      }),
    ).toBeInTheDocument();
    expect(window.localStorage.getItem("eda.layout.activity-launcher")).toBe(
      "true",
    );
  });

  it("does not duplicate the Activity launcher in the top bar", async () => {
    renderAppAt("/projects/p1/sessions/r1/data-map");
    await screen.findByRole("heading", { name: "Data Map" });

    expect(
      within(screen.getByRole("banner", { name: "Workbench" })).queryByRole(
        "button",
        { name: "Activity" },
      ),
    ).toBeNull();
    expect(
      screen.getByRole("button", { name: "Open activity" }),
    ).toBeInTheDocument();
  });

  it("uses a single-column workbench at a 390px viewport", async () => {
    const user = userEvent.setup();
    const original = window.matchMedia;
    window.matchMedia = (query: string) =>
      ({
        matches: query === "(max-width: 767px)",
        media: query,
        onchange: null,
        addEventListener: () => {},
        removeEventListener: () => {},
        addListener: () => {},
        removeListener: () => {},
        dispatchEvent: () => false,
      }) as MediaQueryList;
    try {
      renderAppAt("/projects/p1/sessions/r1/data-map");
      expect(
        await screen.findByRole("heading", { name: "Data Map" }),
      ).toBeInTheDocument();
      expect(
        screen.queryByRole("complementary", { name: "Sessions" }),
      ).not.toBeInTheDocument();
      expect(
        screen.queryByRole("heading", { name: "Context Inspector" }),
      ).not.toBeInTheDocument();
      /* The compact session nav stays one row high; its page links scroll
       * inside their own strip rather than creating a second page-wide wrap. */
      expect(
        within(
          screen.getByRole("navigation", { name: "Session sections" }),
        ).getByRole("list", { name: "Understand the data" }),
      ).toHaveClass("overflow-x-auto");
      expect(
        screen.getByRole("navigation", { name: "Session sections" }),
      ).not.toHaveClass("overflow-x-auto");

      const trigger = screen.getByRole("button", { name: "Sessions" });
      await user.click(trigger);
      let dialog = screen.getByRole("dialog", { name: "Mobile sessions" });
      expect(
        within(dialog).getByRole("link", { name: "Home" }),
      ).toHaveAttribute("href", "/projects");
      expect(await within(dialog).findByText("Demo run")).toBeInTheDocument();
      await waitFor(() =>
        expect(within(dialog).getByRole("link", { name: "New session" })).toHaveFocus(),
      );

      await user.keyboard("{Escape}");
      await waitFor(() =>
        expect(
          screen.queryByRole("dialog", { name: "Mobile sessions" }),
        ).not.toBeInTheDocument(),
      );
      await waitFor(() => expect(trigger).toHaveFocus());

      await user.click(trigger);
      dialog = screen.getByRole("dialog", { name: "Mobile sessions" });
      await user.click(
        within(dialog).getByRole("link", { name: "Home" }),
      );
      expect(
        await screen.findByRole("heading", { name: "Overview" }),
      ).toBeInTheDocument();
      expect(
        screen.queryByRole("dialog", { name: "Mobile sessions" }),
      ).not.toBeInTheDocument();
    } finally {
      window.matchMedia = original;
    }
  });
});

describe("Run navigation groups", () => {
  it("puts every session page under its section, in order", async () => {
    const user = userEvent.setup();
    renderAppAt("/projects/p1/sessions/r1/data-map");
    await screen.findByRole("heading", { name: "Data Map" });

    for (const [title, pages] of Object.entries(GROUPS)) {
      await chooseStage(user, title);
      const list = await screen.findByRole("list", { name: title });
      /* Table preview only appears once the run's datasets resolve. */
      await waitFor(() =>
        expect(
          within(list).getAllByRole("link").map((link) => link.textContent),
        ).toEqual(pages),
      );
    }
  });

  it("gives every page an icon", async () => {
    const user = userEvent.setup();
    renderAppAt("/projects/p1/sessions/r1/data-map");
    await screen.findByRole("heading", { name: "Data Map" });

    for (const title of Object.keys(GROUPS)) {
      await chooseStage(user, title);
      const list = await screen.findByRole("list", { name: title });
      for (const link of within(list).getAllByRole("link")) {
        expect(
          link.querySelector("svg"),
          `${link.textContent} has no icon`,
        ).not.toBeNull();
      }
    }
  });

  it("navigates through a stage and moves `current` with the route", async () => {
    const user = userEvent.setup();
    renderAppAt("/projects/p1/sessions/r1/data-map");
    await screen.findByRole("heading", { name: "Data Map" });

    const understand = screen.getByRole("button", {
      name: "Understand the data",
    });
    expect(understand).toHaveAttribute("aria-current", "true");

    await chooseStage(user, "Trust & trace");
    await user.click(await screen.findByRole("link", { name: "Trace & cost" }));

    expect(
      await screen.findByRole("heading", { name: "Trace & cost" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Trust & trace" }),
    ).toHaveAttribute("aria-current", "true");
  });

  it("expands stages on demand and does not navigate on pick", async () => {
    const user = userEvent.setup();
    renderAppAt("/projects/p1/sessions/r1/data-map");
    await screen.findByRole("heading", { name: "Data Map" });

    expect(
      screen.getByRole("list", { name: "Understand the data" }),
    ).toBeInTheDocument();

    const trigger = screen.getByRole("button", { name: "Understand the data" });
    expect(trigger).toHaveAttribute("aria-expanded", "false");
    await user.click(trigger);
    expect(
      screen.getByRole("button", { name: "Show Investigate with the agent pages" }),
    ).toBeInTheDocument();
    await user.click(
      screen.getByRole("button", { name: "Show Investigate with the agent pages" }),
    );
    expect(
      await screen.findByRole("list", { name: "Investigate with the agent" }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("list", { name: "Understand the data" }),
    ).not.toBeInTheDocument();
    /* Picking a stage is a look-ahead; the open page is unchanged and the
     * picker folds away immediately. */
    expect(screen.getByRole("heading", { name: "Data Map" })).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Show Understand the data pages" }),
    ).not.toBeInTheDocument();
  });

  it("hides session navigation without an open run", async () => {
    renderAppAt("/projects/p1/new-session");
    await screen.findByRole("heading", { name: "New session" });
    expect(screen.queryByRole("link", { name: "Launchpad" })).not.toBeInTheDocument();
    for (const title of Object.keys(GROUPS)) {
      expect(
        screen.queryByRole("button", { name: title }),
      ).not.toBeInTheDocument();
    }
    /* The dead links this replaces: no run page may be reachable from here. */
    for (const label of ["Data map", "Quality", "Trace & cost", "Findings"]) {
      expect(screen.queryByRole("link", { name: label })).not.toBeInTheDocument();
    }
  });
});
