import { describe, expect, it } from "vitest";
import {
  cleanup,
  fireEvent,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { defaultSettings } from "./msw/handlers";
import { server } from "./msw/server";
import { renderAppAt } from "./render";

/* Every wildcard override below also answers for the bucket holding sessions
 * with no project. Left unguarded, the rail's standalone group renders a copy
 * of whatever the test staged for its projects. */
const NO_STANDALONE = http.get(
  "/api/v1/projects/unfiled-sessions/sessions",
  () => HttpResponse.json({ items: [], next_cursor: null }),
);

async function railRuns(): Promise<string[]> {
  const rail = await screen.findByRole("complementary", { name: "Sessions" });
  return within(rail)
    .queryAllByRole("link")
    .map((link) => link.textContent ?? "")
    .filter((text) => text.includes("Demo run") || text.includes("Churn"));
}

describe("Session rail search", () => {
  it("lists every run before a search term is typed", async () => {
    renderAppAt("/projects");
    /* A single project shows no project heading (sessions_ui.py's
     * multi_project check) — wait on a run itself instead. */
    await screen.findByText("Demo run");
    await waitFor(async () => expect(await railRuns()).toHaveLength(2));
  });

  it("filters sessions server-side and passes q to the API", async () => {
    const queries: (string | null)[] = [];
    server.use(
      NO_STANDALONE,
      http.get("/api/v1/projects/:projectId/sessions", ({ request }) => {
        const q = new URL(request.url).searchParams.get("q");
        queries.push(q);
        const items =
          q && !"churn deep dive".includes(q.toLowerCase())
            ? []
            : [
                {
                  session_id: "r2",
                  project_id: "p1",
                  title: "Churn deep dive",
                  status: "complete",
                  created_at: "2026-07-19T10:00:00Z",
                  updated_at: "2026-07-19T10:00:00Z",
                  dataset_names: ["churn"],
                  artifact_count: 1,
                  report_status: null,
                  chat_message_count: 0,
                },
              ];
        return HttpResponse.json({ items, next_cursor: null });
      }),
    );
    renderAppAt("/projects");
    fireEvent.click(await screen.findByRole("button", { name: "Search sessions" }));
    const search = await screen.findByRole("searchbox", { name: "Search sessions" });
    fireEvent.change(search, { target: { value: "churn" } });

    await waitFor(() => expect(queries).toContain("churn"));
    expect(await screen.findAllByText("Churn deep dive")).not.toHaveLength(0);
  });

  it("debounces so a burst of keystrokes issues one filtered request", async () => {
    const queries: (string | null)[] = [];
    server.use(
      NO_STANDALONE,
      http.get("/api/v1/projects/:projectId/sessions", ({ request }) => {
        queries.push(new URL(request.url).searchParams.get("q"));
        return HttpResponse.json({ items: [], next_cursor: null });
      }),
    );
    renderAppAt("/projects");
    fireEvent.click(await screen.findByRole("button", { name: "Search sessions" }));
    const search = await screen.findByRole("searchbox", { name: "Search sessions" });
    for (const value of ["c", "ch", "chu", "chur", "churn"]) {
      fireEvent.change(search, { target: { value } });
    }
    await waitFor(() => expect(queries).toContain("churn"));
    // Only the unfiltered first load and the settled term reach the server.
    expect(queries.filter(Boolean)).toEqual(["churn"]);
  });

  it("clearing the box restores the unfiltered list", async () => {
    renderAppAt("/projects");
    fireEvent.click(await screen.findByRole("button", { name: "Search sessions" }));
    const search = await screen.findByRole("searchbox", { name: "Search sessions" });
    fireEvent.change(search, { target: { value: "churn" } });
    expect(await screen.findAllByText("Churn deep dive")).not.toHaveLength(0);

    fireEvent.change(search, { target: { value: "" } });
    expect(search).toHaveValue("");
    expect(screen.getByText("Type to search all sessions.")).toBeInTheDocument();
  });

  it("shows a distinct empty state when nothing matches", async () => {
    renderAppAt("/projects");
    fireEvent.click(await screen.findByRole("button", { name: "Search sessions" }));
    fireEvent.change(await screen.findByRole("searchbox", { name: "Search sessions" }), {
      target: { value: "zzzz" },
    });
    expect(await screen.findByText(/Matches session titles/)).toBeInTheDocument();
  });
});

describe("Session rail run deletion", () => {
  it("asks for confirmation and spells out what is lost", async () => {
    renderAppAt("/projects");
    fireEvent.click(await screen.findByLabelText("Delete session Demo run"));

    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByText(/cannot be undone/i)).toBeInTheDocument();
    expect(within(dialog).getByText(/artifacts/i)).toBeInTheDocument();
  });

  it("cancelling deletes nothing", async () => {
    let deletes = 0;
    server.use(
      http.delete("/api/v1/sessions/:sessionId", () => {
        deletes += 1;
        return HttpResponse.json({
          session_id: "r1",
          project_id: "p1",
          deleted: true,
        });
      }),
    );
    renderAppAt("/projects");
    fireEvent.click(await screen.findByLabelText("Delete session Demo run"));
    fireEvent.click(
      within(await screen.findByRole("dialog")).getByRole("button", {
        name: "Cancel",
      }),
    );
    await waitFor(() =>
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument(),
    );
    expect(deletes).toBe(0);
  });

  it("confirming calls DELETE and refetches the run list", async () => {
    let deleted: string | null = null;
    let listCalls = 0;
    server.use(
      http.delete("/api/v1/sessions/:sessionId", ({ params }) => {
        deleted = String(params["sessionId"]);
        return HttpResponse.json({
          session_id: deleted,
          project_id: "p1",
          deleted: true,
        });
      }),
      NO_STANDALONE,
      http.get("/api/v1/projects/:projectId/sessions", () => {
        listCalls += 1;
        const items = deleted
          ? []
          : [
              {
                session_id: "r1",
                project_id: "p1",
                title: "Demo run",
                status: "complete",
                created_at: "2026-07-20T10:00:00Z",
                updated_at: "2026-07-21T10:00:00Z",
                dataset_names: ["sample"],
                artifact_count: 3,
                report_status: "final",
                chat_message_count: 0,
              },
            ];
        return HttpResponse.json({ items, next_cursor: null });
      }),
    );
    renderAppAt("/projects");
    fireEvent.click(await screen.findByLabelText("Delete session Demo run"));
    const before = listCalls;
    fireEvent.click(
      within(await screen.findByRole("dialog")).getByRole("button", {
        name: "Delete session",
      }),
    );

    await waitFor(() => expect(deleted).toBe("r1"));
    await waitFor(() => expect(listCalls).toBeGreaterThan(before));
    await waitFor(() =>
      expect(screen.queryByLabelText("Delete session Demo run")).not.toBeInTheDocument(),
    );
  });

  it("surfaces a 409 from an active job instead of pretending it worked", async () => {
    server.use(
      http.delete("/api/v1/sessions/:sessionId", () =>
        HttpResponse.json(
          {
            error: {
              code: "session_busy",
              message: "Session r1 has an active job (job_1); cancel it first.",
            },
          },
          { status: 409 },
        ),
      ),
    );
    renderAppAt("/projects");
    fireEvent.click(await screen.findByLabelText("Delete session Demo run"));
    fireEvent.click(
      within(await screen.findByRole("dialog")).getByRole("button", {
        name: "Delete session",
      }),
    );
    expect(await screen.findByText(/active job/)).toBeInTheDocument();
    // The dialog stays open so the user can cancel the job and retry.
    expect(screen.getByRole("dialog")).toBeInTheDocument();
  });

  it("navigates away when the open run is the one deleted", async () => {
    renderAppAt("/projects/p1/sessions/r1/data-map");
    const rail = await screen.findByRole("complementary", { name: "Sessions" });
    fireEvent.click(await within(rail).findByLabelText("Delete session Demo run"));
    fireEvent.click(
      within(await screen.findByRole("dialog")).getByRole("button", {
        name: "Delete session",
      }),
    );
    expect(
      await screen.findByRole("heading", { name: "Overview" }),
    ).toBeInTheDocument();
  });
});

describe("Session rail navigation", () => {
  /* Settings and connection state live in the top bar and nowhere else: a
   * second copy in the rail meant collapsing the rail hid one of them. */
  it("keeps settings out of the rail entirely", async () => {
    renderAppAt("/projects/p1/new-session");
    const rail = await screen.findByRole("complementary", { name: "Sessions" });
    expect(within(rail).queryByRole("button", { name: "Settings" })).toBeNull();
    expect(within(rail).queryByRole("link", { name: "Settings" })).toBeNull();
    expect(
      within(
        await screen.findByRole("banner", { name: "Workbench" }),
      ).getByRole("button", { name: "Settings" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "New session" })).toBeInTheDocument();
    expect(
      within(rail).getByRole("link", { name: "Home" }),
    ).toHaveAttribute("href", "/projects");
  });
});

describe("Top bar LLM status", () => {
  it("shows provider · model with an offline indicator", async () => {
    renderAppAt("/projects");
    const bar = await screen.findByRole("banner");
    expect(
      await within(bar).findByText("offline · offline-deterministic"),
    ).toBeInTheDocument();
    expect(within(bar).getByLabelText("LLM offline")).toBeInTheDocument();
    expect(within(bar).queryByLabelText("LLM ready")).not.toBeInTheDocument();
    expect(
      within(
        screen.getByRole("complementary", { name: "Sessions" }),
      ).queryByLabelText("LLM offline"),
    ).toBeNull();
  });

  it("shows a different indicator once a live provider is configured", async () => {
    server.use(
      http.get("/api/v1/settings", () =>
        HttpResponse.json({
          ...defaultSettings(),
          provider: "deepseek",
          model: "deepseek-v4-flash",
          status_state: "ready",
          is_ready_for_live_calls: true,
          api_key_set: true,
        }),
      ),
    );
    renderAppAt("/projects");
    const bar = await screen.findByRole("banner");
    expect(
      await within(bar).findByText("deepseek · deepseek-v4-flash"),
    ).toBeInTheDocument();
    expect(within(bar).getByLabelText("LLM ready")).toBeInTheDocument();
    expect(within(bar).queryByLabelText("LLM offline")).not.toBeInTheDocument();
  });

  /* The whole point of the strict predicate: a provider picked without its key
   * must not read as ready, and the dot must name what is missing. */
  it("warns when a provider is selected but not usable yet", async () => {
    server.use(
      http.get("/api/v1/settings", () =>
        HttpResponse.json({
          ...defaultSettings(),
          provider: "deepseek",
          model: "deepseek-v4-flash",
          status_state: "incomplete",
          is_ready_for_live_calls: false,
          api_key_set: false,
          status_message: "Live calls are disabled.",
          missing_fields: ["api_key"],
        }),
      ),
    );
    renderAppAt("/projects");
    const bar = await screen.findByRole("banner");
    const dot = await within(bar).findByLabelText("LLM not ready");
    expect(within(bar).queryByLabelText("LLM ready")).not.toBeInTheDocument();
    expect(within(bar).queryByLabelText("LLM offline")).not.toBeInTheDocument();
    expect(dot.getAttribute("title")).toContain("api_key");
  });

  it("keeps connection state and settings repair side by side in the top bar", async () => {
    renderAppAt("/projects");
    const bar = await screen.findByRole("banner");
    expect(await within(bar).findByLabelText("LLM offline")).toBeInTheDocument();
    expect(within(bar).getByRole("button", { name: "Settings" })).toBeInTheDocument();
  });
});

describe("Session rail collapse", () => {
  it("collapses to a strip, persists the choice and restores it on remount", async () => {
    renderAppAt("/projects");
    await screen.findByRole("complementary", { name: "Sessions" });

    fireEvent.click(screen.getByRole("button", { name: "Collapse sessions" }));
    await waitFor(() =>
      expect(
        screen.queryByRole("complementary", { name: "Sessions" }),
      ).not.toBeInTheDocument(),
    );
    expect(window.localStorage.getItem("eda.layout.rail-collapsed")).toBe("true");

    cleanup();
    renderAppAt("/projects");
    expect(
      await screen.findByRole("button", { name: "Expand sessions" }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("complementary", { name: "Sessions" }),
    ).not.toBeInTheDocument();
  });

  it("restores the project rail from the top-bar control", async () => {
    window.localStorage.setItem("eda.layout.rail-collapsed", "true");
    renderAppAt("/projects");
    await screen.findByRole("complementary", { name: "Sessions (collapsed)" });
    fireEvent.click(screen.getByRole("button", { name: "Expand sessions" }));
    expect(
      await screen.findByRole("complementary", { name: "Sessions" }),
    ).toBeInTheDocument();
    expect(window.localStorage.getItem("eda.layout.rail-collapsed")).toBe("false");
  });
});

describe("Session rail project groups", () => {
  it("keeps a compact, even session row inside a project", async () => {
    renderAppAt("/projects");
    const rail = await screen.findByRole("complementary", { name: "Sessions" });
    const row = (await within(rail).findByText("Demo run")).closest("a")!;

    expect(row).toHaveClass("gap-[1.5px]", "py-[2px]");
    expect(within(rail).getByText("Earlier")).toHaveClass("pt-1.5", "pb-0.5");
    expect(rail.querySelector(".overflow-y-auto > div")).toHaveClass("gap-1");
  });

  it("hides a project's sessions when collapsed and brings them back", async () => {
    renderAppAt("/projects");
    const rail = await screen.findByRole("complementary", { name: "Sessions" });
    await within(rail).findByText("Demo run");

    const toggle = within(rail).getByRole("button", { name: "Project p1" });
    expect(toggle).toHaveAttribute("aria-expanded", "true");

    fireEvent.click(toggle);
    await waitFor(() =>
      expect(within(rail).queryByText("Demo run")).not.toBeInTheDocument(),
    );
    expect(
      within(rail).getByRole("button", { name: "Project p1" }),
    ).toHaveAttribute("aria-expanded", "false");

    fireEvent.click(within(rail).getByRole("button", { name: "Project p1" }));
    expect(await within(rail).findByText("Demo run")).toBeInTheDocument();
  });

  it("remembers each project group separately", async () => {
    server.use(
      http.get("/api/v1/projects", () =>
        HttpResponse.json([
          { project_id: "p1", name: "Alpha", session_count: 1 },
          { project_id: "p2", name: "Beta", session_count: 1 },
        ]),
      ),
      NO_STANDALONE,
      http.get("/api/v1/projects/:projectId/sessions", ({ params }) =>
        HttpResponse.json({
          items: [
            {
              session_id: `${params["projectId"]}_run`,
              project_id: String(params["projectId"]),
              title: `${params["projectId"]} run`,
              status: "complete",
              created_at: new Date().toISOString(),
              updated_at: null,
              dataset_names: [],
              artifact_count: 0,
              report_status: null,
              chat_message_count: 0,
            },
          ],
          next_cursor: null,
        }),
      ),
    );

    renderAppAt("/projects");
    const rail = await screen.findByRole("complementary", { name: "Sessions" });
    await within(rail).findByText("p1 run");
    fireEvent.click(within(rail).getByRole("button", { name: "Alpha" }));
    await waitFor(() =>
      expect(within(rail).queryByText("p1 run")).not.toBeInTheDocument(),
    );
    /* Only the collapsed one hides — the sibling group is untouched. */
    expect(within(rail).getByText("p2 run")).toBeInTheDocument();

    cleanup();
    renderAppAt("/projects");
    const restored = await screen.findByRole("complementary", {
      name: "Sessions",
    });
    expect(await within(restored).findByText("p2 run")).toBeInTheDocument();
    expect(within(restored).queryByText("p1 run")).not.toBeInTheDocument();
  });
});

/* Two projects, Alpha expanded with one run, and a reorder endpoint that
 * records what it was sent. Shared by every drag test below. */
function useReorderableProjects() {
  const savedOrders: string[][] = [];
  let currentOrder = ["p1", "p2"];
  const summary = (project_id: string) => ({
    project_id,
    name: project_id === "p1" ? "Alpha" : "Beta",
    session_count: project_id === "p1" ? 1 : 0,
  });
  server.use(
    NO_STANDALONE,
    http.get("/api/v1/projects", () =>
      HttpResponse.json(currentOrder.map(summary)),
    ),
    http.put("/api/v1/projects/order", async ({ request }) => {
      const body = (await request.json()) as { project_ids: string[] };
      savedOrders.push(body.project_ids);
      currentOrder = body.project_ids;
      return HttpResponse.json(body.project_ids.map(summary));
    }),
    http.get("/api/v1/projects/:projectId/sessions", ({ params }) =>
      HttpResponse.json({
        items:
          params.projectId === "p1"
            ? [
                {
                  session_id: "alpha-run",
                  project_id: "p1",
                  title: "Alpha run",
                  status: "complete",
                  created_at: "2026-07-29T10:00:00Z",
                  updated_at: "2026-07-29T10:00:00Z",
                  dataset_names: [],
                  artifact_count: 0,
                  report_status: null,
                  chat_message_count: 0,
                },
              ]
            : [],
        next_cursor: null,
      }),
    ),
  );
  return { savedOrders };
}

describe("Session rail time grouping", () => {
  it("buckets sessions into Today/Yesterday/This week/Earlier exactly like _time_group", async () => {
    /* _time_group compares UTC calendar dates (created_at is always UTC), so
     * the offsets below are anchored to today's UTC midnight, not local time. */
    const utcMidnight = Date.UTC(
      new Date().getUTCFullYear(),
      new Date().getUTCMonth(),
      new Date().getUTCDate(),
    );
    const atDaysAgo = (days: number) =>
      new Date(utcMidnight - days * 86_400_000 + 12 * 3_600_000).toISOString();
    const run = (id: string, title: string, createdAt: string | null) => ({
      session_id: id,
      project_id: "p1",
      title,
      status: "complete",
      created_at: createdAt,
      updated_at: createdAt,
      dataset_names: [],
      artifact_count: 0,
      report_status: null,
      chat_message_count: 0,
    });

    server.use(
      NO_STANDALONE,
      http.get("/api/v1/projects/:projectId/sessions", () =>
        HttpResponse.json({
          items: [
            run("today_run", "Today run", atDaysAgo(0)),
            run("yesterday_run", "Yesterday run", atDaysAgo(1)),
            run("week_run", "This week run", atDaysAgo(6)),
            run("earlier_run", "Earlier run", atDaysAgo(10)),
            run("no_date_run", "No date run", null),
          ],
          next_cursor: null,
        }),
      ),
    );

    renderAppAt("/projects");
    const rail = await screen.findByRole("complementary", { name: "Sessions" });
    await within(rail).findByText("Today run");
    const text = rail.textContent ?? "";
    const at = (needle: string) => {
      const index = text.indexOf(needle);
      expect(index, `expected to find "${needle}" in the rail`).toBeGreaterThan(-1);
      return index;
    };

    /* Each caption must precede its own bucket's run(s) and precede the next
     * caption, so this also proves no session drifts into the wrong group. */
    expect(at("Today")).toBeLessThan(at("Today run"));
    expect(at("Today run")).toBeLessThan(at("Yesterday"));
    expect(at("Yesterday")).toBeLessThan(at("Yesterday run"));
    expect(at("Yesterday run")).toBeLessThan(at("This week"));
    expect(at("This week")).toBeLessThan(at("This week run"));
    expect(at("This week run")).toBeLessThan(at("Earlier"));
    expect(at("Earlier")).toBeLessThan(at("Earlier run"));
    expect(at("Earlier run")).toBeLessThan(at("No date run"));

    /* A null created_at buckets to "Earlier" too, and — like the
     * source — only the first session of a contiguous bucket gets a caption. */
    expect(
      within(rail).getAllByText(/^(Today|Yesterday|This week|Earlier)$/),
    ).toHaveLength(4);
  });

  /* The invariant is unchanged — the visible date must be the field the row is
   * bucketed and ordered by — but that field is now updated_at, matching
   * store.query_run_index_rows. Grouping by created_at while the API orders by
   * updated_at put a re-run at the top of the list under an "Earlier" caption. */
  it("dates each run by updated_at, the field it is grouped by", async () => {
    const created = "2026-07-10T12:00:00Z";
    const updated = new Date().toISOString();
    server.use(
      NO_STANDALONE,
      http.get("/api/v1/projects/:projectId/sessions", () =>
        HttpResponse.json({
          items: [
            {
              session_id: "old_run",
              project_id: "p1",
              title: "Old run",
              status: "complete",
              created_at: created,
              updated_at: updated,
              dataset_names: [],
              artifact_count: 0,
              report_status: null,
              chat_message_count: 0,
            },
          ],
          next_cursor: null,
        }),
      ),
    );

    renderAppAt("/projects");
    const rail = await screen.findByRole("complementary", { name: "Sessions" });
    const row = (await within(rail).findByText("Old run")).closest("a")!;
    const label = (iso: string) =>
      new Date(iso).toLocaleDateString(undefined, {
        month: "short",
        day: "numeric",
      });

    /* updated_at is today, so the row buckets under Today and must print that
     * same date — showing created_at here would read as July 10 under a Today
     * caption, which is the contradiction this test exists to prevent. */
    expect(within(rail).getByText("Today")).toBeInTheDocument();
    expect(row.textContent).toContain(label(updated));
    expect(row.textContent).not.toContain(label(created));
  });

  it("keeps empty projects visible so a new session can start there", async () => {
    server.use(
      http.get("/api/v1/projects", () =>
        HttpResponse.json([
          { project_id: "p1", name: "Demo", session_count: 1 },
          { project_id: "p2", name: "Scratch", session_count: 0 },
        ]),
      ),
      NO_STANDALONE,
      http.get("/api/v1/projects/:projectId/sessions", ({ params }) => {
        const items =
          String(params["projectId"]) === "p1"
            ? [
                {
                  session_id: "r1",
                  project_id: "p1",
                  title: "Demo run",
                  status: "complete",
                  created_at: new Date().toISOString(),
                  updated_at: null,
                  dataset_names: [],
                  artifact_count: 0,
                  report_status: null,
                  chat_message_count: 0,
                },
              ]
            : [];
        return HttpResponse.json({ items, next_cursor: null });
      }),
    );

    renderAppAt("/projects");
    const rail = await screen.findByRole("complementary", { name: "Sessions" });
    await within(rail).findByText("Demo run");

    /* Project cards are always visible: an empty project still needs its
     * management link and the + entry for starting its first session. */
    expect(within(rail).getByText("Today")).toBeInTheDocument();
    expect(within(rail).getByRole("button", { name: "Demo" })).toBeInTheDocument();
    expect(within(rail).getByRole("button", { name: "Scratch" })).toBeInTheDocument();
    expect(within(rail).getByText("No sessions yet.")).toBeInTheDocument();
  });

  it("uses nested project groups in the API's creation order", async () => {
    const makeRuns = (projectId: string, prefix: string) =>
      [0, 1].map((i) => ({
        session_id: `${prefix}_${i}`,
        project_id: projectId,
        title: `${prefix} run ${i}`,
        status: "complete",
        created_at: new Date(Date.now() - i * 3_600_000).toISOString(),
        updated_at: null,
        dataset_names: [],
        artifact_count: 0,
        report_status: null,
        chat_message_count: 0,
      }));

    server.use(
      http.get("/api/v1/projects", () =>
        HttpResponse.json([
          { project_id: "p1", name: "Alpha", session_count: 2 },
          { project_id: "p2", name: "Beta", session_count: 2 },
        ]),
      ),
      NO_STANDALONE,
      http.get("/api/v1/projects/:projectId/sessions", ({ params }) => {
        const projectId = String(params["projectId"]);
        const items =
          projectId === "p1"
            ? makeRuns("p1", "alpha")
            : makeRuns("p2", "beta");
        return HttpResponse.json({ items, next_cursor: null });
      }),
    );

    /* Opening Beta must not reorder projects: the backend's creation order is
     * stable, so Alpha stays first. */
    renderAppAt("/projects/p2/new-session");
    const rail = await screen.findByRole("complementary", { name: "Sessions" });
    await within(rail).findByText("beta run 0");
    await within(rail).findByText("alpha run 0");
    const alphaHeading = within(rail).getByRole("button", { name: "Alpha" });
    const betaHeading = within(rail).getByRole("button", { name: "Beta" });
    expect(
      alphaHeading.compareDocumentPosition(betaHeading) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();

    /* Deliberate deviation: sessions_ui.py suppresses time buckets once several
     * projects show, but only because its 12-session cap usually leaves one
     * project. Paginating instead of capping lets both groupings coexist, so
     * each project keeps its own time buckets. */
    expect(
      within(rail).getAllByText(/^(Today|Yesterday|This week|Earlier)$/).length,
    ).toBeGreaterThanOrEqual(2);
    expect(within(rail).getAllByText(/^beta run \d$/)).toHaveLength(2);
    expect(within(rail).getAllByText(/^alpha run \d$/)).toHaveLength(2);
  });

  it("reorders projects by dragging their headers and saves the complete order", async () => {
    const { savedOrders } = useReorderableProjects();

    renderAppAt("/projects");
    const rail = await screen.findByRole("complementary", { name: "Sessions" });
    const alpha = await within(rail).findByRole("button", { name: "Alpha" });
    const beta = within(rail).getByRole("button", { name: "Beta" });

    fireEvent.dragStart(alpha);
    fireEvent.dragOver(beta);
    fireEvent.drop(beta);

    await waitFor(() => expect(savedOrders).toEqual([["p2", "p1"]]));
    const refreshedRail = screen.getByRole("complementary", { name: "Sessions" });
    await waitFor(() =>
      expect(
        within(refreshedRail)
          .getAllByRole("button", { name: /^(Alpha|Beta)$/ })
          .map((button) => button.getAttribute("aria-label")),
      ).toEqual(["Beta", "Alpha"]),
    );
    expect(within(refreshedRail).getByRole("link", { name: /Alpha run/ }).draggable).toBe(false);
  });

  it("previews the new order while dragging, before anything is saved", async () => {
    const { savedOrders } = useReorderableProjects();
    renderAppAt("/projects");
    const rail = await screen.findByRole("complementary", { name: "Sessions" });
    const alpha = await within(rail).findByRole("button", { name: "Alpha" });
    const beta = within(rail).getByRole("button", { name: "Beta" });

    fireEvent.dragStart(alpha);
    fireEvent.dragOver(beta);

    /* The list itself is the ghost: it reflows under the cursor. Before this
     * the only feedback was a ring on the target, which never said whether the
     * dragged project would land above or below it. */
    await waitFor(() =>
      expect(
        within(rail)
          .getAllByRole("button", { name: /^(Alpha|Beta)$/ })
          .map((button) => button.getAttribute("aria-label")),
      ).toEqual(["Beta", "Alpha"]),
    );
    expect(savedOrders).toEqual([]);
  });

  it("accepts a drop anywhere in the target project, not just its header", async () => {
    const { savedOrders } = useReorderableProjects();
    renderAppAt("/projects");
    const rail = await screen.findByRole("complementary", { name: "Sessions" });
    const beta = await within(rail).findByRole("button", { name: "Beta" });
    /* An expanded project's runs sit below its header inside the same block.
     * Only the header used to carry the handlers, so a release over this row
     * was refused by the browser and the drag silently did nothing. */
    const alphaRun = await within(rail).findByRole("link", { name: /Alpha run/ });

    fireEvent.dragStart(beta);
    fireEvent.dragOver(alphaRun);
    fireEvent.drop(alphaRun);

    await waitFor(() => expect(savedOrders).toEqual([["p2", "p1"]]));
  });
});

describe("Sessions with no project", () => {
  const standalone = {
    session_id: "solo_1",
    project_id: "unfiled-sessions",
    title: "One-off delivery check",
    status: "complete",
    created_at: "2026-07-22T10:00:00Z",
    updated_at: "2026-07-22T10:00:00Z",
    dataset_names: ["orders"],
    artifact_count: 2,
    report_status: "final",
    chat_message_count: 0,
  };

  it("lists them under their own group below the projects", async () => {
    server.use(
      http.get("/api/v1/projects/unfiled-sessions/sessions", () =>
        HttpResponse.json({ items: [standalone], next_cursor: null }),
      ),
    );
    renderAppAt("/projects");

    const rail = await screen.findByRole("complementary", { name: "Sessions" });
    expect(
      await within(rail).findByText("One-off delivery check"),
    ).toBeInTheDocument();

    /* Below every project, which is the whole point of the placement: a
     * standalone session is not a peer of a project's contents. */
    const groups = within(rail).getAllByRole("button", { expanded: true });
    const labels = groups.map((node) => node.getAttribute("aria-label"));
    expect(labels.indexOf("Recent")).toBe(labels.length - 1);
  });

  it("is reachable from the search that claims to cover every session", async () => {
    server.use(
      http.get("/api/v1/projects/unfiled-sessions/sessions", ({ request }) => {
        const q = new URL(request.url).searchParams.get("q")?.toLowerCase() ?? "";
        const match = !q || standalone.title.toLowerCase().includes(q);
        return HttpResponse.json({
          items: match ? [standalone] : [],
          next_cursor: null,
        });
      }),
    );
    renderAppAt("/projects");
    await screen.findByText("Demo run");

    fireEvent.click(screen.getByRole("button", { name: "Search sessions" }));
    const dialog = await screen.findByRole("dialog", { name: "Session search" });
    fireEvent.change(
      within(dialog).getByRole("searchbox", { name: "Search sessions" }),
      { target: { value: "delivery" } },
    );

    expect(
      await within(dialog).findByText("One-off delivery check"),
    ).toBeInTheDocument();
  });

  it("hides the group entirely when there are none", async () => {
    renderAppAt("/projects");
    await screen.findByText("Demo run");

    const rail = screen.getByRole("complementary", { name: "Sessions" });
    expect(within(rail).queryByText("Recent")).not.toBeInTheDocument();
  });
});
