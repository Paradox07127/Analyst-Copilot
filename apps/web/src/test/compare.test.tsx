import { describe, expect, it } from "vitest";
import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import type { SessionDetail, SessionPage } from "../api/client";
import { server } from "./msw/server";
import { renderAppAt, renderAppWithRouterAt } from "./render";
import { compareView } from "./msw/handlers";

const PAGE_PATH = "/projects/p1/compare?left=r1";

function twoRuns() {
  return http.get("/api/v1/projects/:projectId/sessions", ({ params }) =>
    HttpResponse.json({
      items: [
        {
          session_id: "r1",
          project_id: String(params["projectId"]),
          title: "Baseline run",
          status: "complete",
          created_at: "2026-07-20T10:00:00Z",
          updated_at: "2026-07-20T10:00:00Z",
          dataset_names: ["sample"],
          artifact_count: 12,
          report_status: "final",
          chat_message_count: 0,
        },
        {
          session_id: "r2",
          project_id: String(params["projectId"]),
          title: "Cleaned run",
          status: "complete",
          created_at: "2026-07-22T10:00:00Z",
          updated_at: "2026-07-22T10:00:00Z",
          dataset_names: ["sample", "extra"],
          artifact_count: 15,
          report_status: "draft",
          chat_message_count: 0,
        },
      ],
      next_cursor: null,
    } satisfies SessionPage),
  );
}

function metricRow(label: string): HTMLElement {
  return screen.getByRole("row", { name: new RegExp(`^${label}`) });
}

describe("Compare page", () => {
  it("asks for a right-hand run before comparing anything", async () => {
    server.use(twoRuns());
    renderAppAt(PAGE_PATH);

    await screen.findByRole("heading", { name: "Compare" });
    expect(
      screen.getByText("Pick a session to compare against"),
    ).toBeInTheDocument();
    /* The current run must never be offered as its own counterpart. */
    const picker = screen.getByRole("combobox", { name: /Compare against/ });
    await within(picker).findByRole("option", { name: /Cleaned run · r2/ });
    expect(within(picker).queryByRole("option", { name: /r1/ })).toBeNull();
  });

  it("renders both sides, deltas, and the artifact/dataset differences from ?right=", async () => {
    server.use(twoRuns());
    renderAppAt(`${PAGE_PATH}&right=r2`);

    await screen.findByRole("heading", { name: "Headline metrics" });
    const header = screen.getByRole("region", { name: "Compared sessions" });
    expect(within(header).getByText("Baseline run")).toBeInTheDocument();
    expect(within(header).getByText("Cleaned run")).toBeInTheDocument();
    expect(screen.getByText("partially controlled")).toBeInTheDocument();
    expect(screen.getByText("Session lineage")).toBeInTheDocument();
    expect(screen.getByText("direct parent")).toBeInTheDocument();

    /* An improvement on a metric where lower is better is coloured "ok". */
    const critical = metricRow("Critical issues");
    const criticalDelta = within(critical).getByText("−2");
    expect(criticalDelta).toHaveClass("text-status-ok");

    /* A neutral metric must be readable but never coloured as good/bad. */
    const rows = metricRow("Total rows");
    const rowsDelta = within(rows).getByText("−5");
    expect(rowsDelta).toHaveClass("text-status-neutral");

    /* No delta at all when both sides agree. */
    const charts = metricRow("Charts");
    expect(within(charts).getAllByText("—")).toHaveLength(1);

    expect(screen.getByText("ModelCard")).toBeInTheDocument();
    expect(screen.getByText("0 → 1")).toBeInTheDocument();
    /* Artifact counts carry no declared direction, so their deltas stay
     * neutral — an extra chart is neither an improvement nor a regression. */
    expect(screen.getByText("+1")).toHaveClass("text-status-neutral");
    /* Unchanged artifact types are filtered out of the differences list. */
    expect(screen.queryByText("ChartSpec")).not.toBeInTheDocument();

    expect(screen.getByText("In both sessions")).toBeInTheDocument();
    expect(screen.getByText("extra.csv")).toBeInTheDocument();
  });

  it("labels failed-side gaps unavailable instead of rendering a false zero", async () => {
    const failed = compareView("r1", "r2");
    failed.right.status = "failed";
    const critical = failed.metrics?.find((row) => row.key === "critical");
    if (!critical) throw new Error("critical fixture row missing");
    critical.right = {
      state: "unavailable",
      reason: "quality producer failed before writing output",
    };
    critical.delta = null;
    critical.verdict = "unknown";
    failed.artifact_deltas = [
      {
        type: "ModelCard",
        left: { state: "value", value: 0 },
        right: {
          state: "unavailable",
          reason: "model producer did not complete",
        },
        delta: null,
      },
    ];
    server.use(
      twoRuns(),
      http.get("/api/v1/compare", () => HttpResponse.json(failed)),
    );

    renderAppAt(`${PAGE_PATH}&right=r2`);

    const row = await screen.findByRole("row", { name: /Critical issues/ });
    expect(within(row).getByText("Unavailable").closest("[title]")).toHaveAttribute(
      "title",
      "quality producer failed before writing output",
    );
    expect(within(row).queryByText("0")).not.toBeInTheDocument();
    expect(screen.getByText("0 → Unavailable")).toHaveAttribute(
      "title",
      "model producer did not complete",
    );
  });

  /* Every column of every diff table used to be headed by its position —
   * "Left" and "Right" — and the properties table carried no header row at
   * all, so three unlabeled columns of values were the whole diff. Both runs
   * carry a title; the tables are headed with it. */
  it("heads the diff columns with the run titles, not their positions", async () => {
    server.use(twoRuns());
    renderAppAt(`${PAGE_PATH}&right=r2`);

    const metrics = await screen.findByRole("table", {
      name: "Headline metrics side by side",
    });
    expect(
      within(metrics).getByRole("columnheader", { name: "Baseline run" }),
    ).toBeInTheDocument();
    expect(
      within(metrics).getByRole("columnheader", { name: "Cleaned run" }),
    ).toBeInTheDocument();
    expect(
      within(metrics).queryByRole("columnheader", { name: "Left" }),
    ).not.toBeInTheDocument();

    const properties = screen.getByRole("table", {
      name: "Session properties side by side",
    });
    expect(
      within(properties).getByRole("columnheader", { name: "Baseline run" }),
    ).toBeInTheDocument();
    expect(
      within(properties).getByRole("columnheader", { name: "Cleaned run" }),
    ).toBeInTheDocument();

    /* Colour alone must not carry the verdict. */
    expect(
      within(metricRow("Critical issues")).getByText("improvement"),
    ).toBeInTheDocument();
  });

  it("switching the right-hand run refetches against the new pair", async () => {
    const requested: string[] = [];
    server.use(
      twoRuns(),
      http.get("/api/v1/compare", ({ request }) => {
        const url = new URL(request.url);
        const right = url.searchParams.get("right") ?? "";
        requested.push(right);
        return HttpResponse.json(
          compareView(url.searchParams.get("left") ?? "r1", right),
        );
      }),
    );

    const user = userEvent.setup();
    renderAppAt(PAGE_PATH);

    await screen.findByRole("heading", { name: "Compare" });
    const picker = screen.getByRole("combobox", { name: /Compare against/ });
    await within(picker).findByRole("option", { name: /Cleaned run · r2/ });
    await user.selectOptions(
      picker,
      "r2",
    );

    await screen.findByRole("heading", { name: "Headline metrics" });
    expect(requested).toEqual(["r2"]);
  });

  it("keeps an unlisted ?right= visible in the picker", async () => {
    /* Derived runs are not in the run list but are valid deep-link targets;
     * without an option for them the select just looks empty. */
    server.use(twoRuns());
    renderAppAt(`${PAGE_PATH}&right=ssess_20260704_replay`);

    await screen.findByRole("heading", { name: "Compare" });
    const picker = await screen.findByRole("combobox", {
      name: /Compare against/,
    });
    expect(picker).toHaveValue("ssess_20260704_replay");
    expect(
      within(picker).getByRole("option", {
        name: /ssess_20260704_replay · not in this list/,
      }),
    ).toBeInTheDocument();
  });

  it("surfaces a cross-project pair as a recoverable error", async () => {
    server.use(
      twoRuns(),
      http.get("/api/v1/compare", () =>
        HttpResponse.json(
          {
            error: {
              code: "compare_project_mismatch",
              message: "Sessions belong to different projects.",
            },
          },
          { status: 422 },
        ),
      ),
    );
    renderAppAt(`${PAGE_PATH}&right=r2`);

    expect(
      await screen.findByText("Request failed (compare_project_mismatch)"),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Sessions belong to different projects."),
    ).toBeInTheDocument();
  });

  it("persists a pinned baseline while an explicit URL left side wins", async () => {
    server.use(twoRuns());
    const user = userEvent.setup();
    const first = renderAppAt(`${PAGE_PATH}&right=r2`);

    await user.click(
      await screen.findByRole("button", { name: "Pin as baseline" }),
    );
    expect(
      screen.getByRole("button", { name: "Unpin baseline" }),
    ).toBePressed();
    first.unmount();

    const { router } = renderAppWithRouterAt(
      "/projects/p1/compare?right=r2",
    );
    await waitFor(() =>
      expect(
        new URLSearchParams(router.state.location.search).get("left"),
      ).toBe("r1"),
    );
    await waitFor(() =>
      expect(screen.getByRole("combobox", { name: "Baseline" })).toHaveValue(
        "r1",
      ),
    );

    await router.navigate("/projects/p1/compare?left=r2");
    await waitFor(() =>
      expect(screen.getByRole("combobox", { name: "Baseline" })).toHaveValue(
        "r2",
      ),
    );
  });

  it("swaps the complete pair atomically and browser back restores it", async () => {
    server.use(twoRuns());
    const user = userEvent.setup();
    const { router } = renderAppWithRouterAt(`${PAGE_PATH}&right=r2`);

    await user.click(await screen.findByRole("button", { name: "Swap sides" }));
    await waitFor(() => {
      const params = new URLSearchParams(router.state.location.search);
      expect(params.get("left")).toBe("r2");
      expect(params.get("right")).toBe("r1");
    });

    await router.navigate(-1);
    await waitFor(() => {
      const params = new URLSearchParams(router.state.location.search);
      expect(params.get("left")).toBe("r1");
      expect(params.get("right")).toBe("r2");
    });
  });

  it("keeps split sections independent and Inspector follows the active pane", async () => {
    let compareRequests = 0;
    server.use(
      twoRuns(),
      http.get("/api/v1/compare", () => {
        compareRequests += 1;
        return HttpResponse.json(compareView("r1", "r2"));
      }),
      http.get("/api/v1/sessions/:sessionId", ({ params }) =>
        HttpResponse.json({
          session_id: String(params["sessionId"]),
          project_id: "p1",
          title: params["sessionId"] === "r1" ? "Left run" : "Right run",
          status: "complete",
          created_at: "2026-07-20T10:00:00Z",
          updated_at: "2026-07-20T10:00:00Z",
          dataset_names: ["sample"],
          artifact_count: 2,
          report_status: "final",
          chat_message_count: 0,
          code_version: null,
          seed: null,
          source_session_id: null,
          artifact_type_counts: {},
          warnings: [],
        } satisfies SessionDetail),
      ),
    );
    const user = userEvent.setup();
    const { router } = renderAppWithRouterAt(
      `${PAGE_PATH}&right=r2&mode=split&leftSection=questions&rightSection=report`,
    );

    const leftPane = await screen.findByRole("region", {
      name: "Left session pane",
    });
    const rightPane = screen.getByRole("region", {
      name: "Right session pane",
    });
    expect(within(leftPane).getByRole("combobox", { name: "Left section" })).toHaveValue(
      "questions",
    );
    expect(
      within(rightPane).getByRole("combobox", { name: "Right section" }),
    ).toHaveValue("report");

    await user.selectOptions(
      within(leftPane).getByRole("combobox", { name: "Left section" }),
      "artifacts",
    );
    await waitFor(() => {
      const params = new URLSearchParams(router.state.location.search);
      expect(params.get("leftSection")).toBe("artifacts");
      expect(params.get("rightSection")).toBe("report");
    });

    fireEvent.pointerDown(rightPane);
    const inspector = screen.getByRole("complementary", {
      name: "Context Inspector",
    });
    await within(inspector).findByText("Right run");
    expect(within(inspector).getAllByText("Report").length).toBeGreaterThan(0);
    expect(compareRequests).toBe(0);
  });
});
