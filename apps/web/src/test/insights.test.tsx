import { describe, expect, it, vi, beforeEach } from "vitest";
import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { server } from "./msw/server";
import { queueDataOperation } from "./msw/handlers";
import { renderAppAt, renderAppWithRouterAt } from "./render";

/* vega-embed does real DOM measurement/canvas work that jsdom cannot do; the
 * mock records the spec each chart hands over so tests assert the contract. */
const embedCalls: Array<Record<string, unknown>> = [];
const embedOptions: Array<Record<string, unknown>> = [];
vi.mock("vega-embed", () => ({
  default: vi.fn(
    async (
      _el: HTMLElement,
      spec: Record<string, unknown>,
      opts: Record<string, unknown>,
    ) => {
      embedCalls.push(spec);
      embedOptions.push(opts);
      return { view: { finalize: vi.fn() } };
    },
  ),
}));

beforeEach(() => {
  embedCalls.length = 0;
  embedOptions.length = 0;
});

describe("Quality page", () => {
  it("restores shareable filters from the URL on a cold load", async () => {
    renderAppAt(
      "/projects/p1/sessions/r1/quality?dataset=sample&severity=critical&codes=empty_column",
    );

    expect(await screen.findByText("Column value is empty.")).toBeInTheDocument();
    expect(screen.getByLabelText("Dataset")).toHaveValue("sample");
    expect(screen.getByLabelText("Severity")).toHaveValue("critical");
    expect(screen.getByLabelText(/empty_column \(1\)/)).toBeChecked();
    expect(screen.getByLabelText(/high_missing \(1\)/)).not.toBeChecked();
    expect(
      screen.queryByText("Column name has 40% missing values."),
    ).not.toBeInTheDocument();
  });

  it("renders KPIs, dataset cards, and the issue table", async () => {
    renderAppAt("/projects/p1/sessions/r1/quality");

    expect(
      await screen.findByText("Column value is empty."),
    ).toBeInTheDocument();
    /* "Critical" appears as KPI label and issue-table badge. */
    expect(screen.getAllByText("Critical").length).toBeGreaterThanOrEqual(2);
    expect(screen.getByText("Critical: 1")).toBeInTheDocument();
    expect(screen.getByText("Info: 1")).toBeInTheDocument();
    expect(screen.getByText("empty_column")).toBeInTheDocument();
    expect(
      screen.getByText("Column id looks like an identifier."),
    ).toBeInTheDocument();
    expect(screen.getByText("Affected datasets")).toBeInTheDocument();
    expect(screen.getByText("Affected fields")).toBeInTheDocument();
    expect(
      screen.getByRole("heading", {
        level: 2,
        name: "Review 1 critical flag first",
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: "Inspect highest-priority table" }),
    ).toHaveAttribute("href", "/projects/p1/sessions/r1/table/sample");
    expect(
      screen.getByRole("link", { name: "Review cleaning options" }),
    ).toHaveAttribute("href", "/projects/p1/sessions/r1/cleaning");
  });

  it("filters issues by dataset id and severity", async () => {
    const user = userEvent.setup();
    renderAppAt("/projects/p1/sessions/r1/quality");
    await screen.findByText("Column value is empty.");

    /* Options label with the display name but carry the dataset id as value. */
    const datasetSelect = screen.getByLabelText<HTMLSelectElement>("Dataset");
    const optionValues = Array.from(datasetSelect.options).map((o) => o.value);
    expect(optionValues).toEqual(["", "sample", "other"]);

    await user.selectOptions(datasetSelect, "other");
    expect(screen.queryByText("Column value is empty.")).not.toBeInTheDocument();
    expect(
      screen.getByText("Column id looks like an identifier."),
    ).toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText("Severity"), "critical");
    expect(
      screen.getByText("No issues match the selected filters"),
    ).toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText("Dataset"), "");
    expect(screen.getByText("Column value is empty.")).toBeInTheDocument();
    expect(
      screen.queryByText("Column id looks like an identifier."),
    ).not.toBeInTheDocument();
  });

  it("preserves multiple filter fields in the URL and restores the resulting view", async () => {
    const user = userEvent.setup();
    const { router, view } = renderAppWithRouterAt(
      "/projects/p1/sessions/r1/quality",
    );
    await screen.findByText("Column value is empty.");

    await user.selectOptions(screen.getByLabelText("Dataset"), "other");
    await user.selectOptions(screen.getByLabelText("Severity"), "info");
    const savedLocation =
      router.state.location.pathname + router.state.location.search;
    const params = new URLSearchParams(router.state.location.search);
    expect(params.get("dataset")).toBe("other");
    expect(params.get("severity")).toBe("info");

    view.unmount();
    renderAppAt(savedLocation);
    expect(await screen.findByLabelText("Dataset")).toHaveValue("other");
    expect(screen.getByLabelText("Severity")).toHaveValue("info");
    expect(screen.getByText("Column id looks like an identifier.")).toBeInTheDocument();
  });

  it("filters issues by code type, with a per-option issue count", async () => {
    const user = userEvent.setup();
    renderAppAt("/projects/p1/sessions/r1/quality");
    await screen.findByText("Column value is empty.");

    // Default: every code checked.
    expect(screen.getByLabelText(/empty_column \(1\)/)).toBeChecked();
    expect(screen.getByLabelText(/high_missing \(1\)/)).toBeChecked();
    expect(screen.getByLabelText(/likely_id \(1\)/)).toBeChecked();

    await user.click(screen.getByLabelText(/empty_column \(1\)/));
    expect(screen.queryByText("Column value is empty.")).not.toBeInTheDocument();
    expect(
      screen.getByText("Column name has 40% missing values."),
    ).toBeInTheDocument();

    // Unchecking every code falls back to "no filter" — same quirk as the
    // Empty selection means no filter (`!selected || row.code in selected`).
    await user.click(screen.getByLabelText(/high_missing \(1\)/));
    await user.click(screen.getByLabelText(/likely_id \(1\)/));
    expect(screen.getByText("Column value is empty.")).toBeInTheDocument();
    expect(
      screen.getByText("Column id looks like an identifier."),
    ).toBeInTheDocument();
  });

  /* The page leads with severity, worst first, and every row reaches its
   * table in one hop — the two things the flat issue table could not do. */
  it("orders the issue groups by severity and links each row to its table", async () => {
    renderAppAt("/projects/p1/sessions/r1/quality");
    await screen.findByText("Column value is empty.");

    const groups = screen
      .getAllByRole("heading", { level: 3 })
      .map((heading) => heading.textContent);
    expect(groups).toEqual(["Critical", "Warning", "Info"]);

    expect(
      screen.getAllByRole("link", { name: "Inspect rows" })[0],
    ).toHaveAttribute("href", "/projects/p1/sessions/r1/table/sample");
  });

  it("scopes the issue list to a dataset picked from the ranked list", async () => {
    const user = userEvent.setup();
    renderAppAt("/projects/p1/sessions/r1/quality");
    await screen.findByText("Column value is empty.");

    await user.click(screen.getByRole("button", { name: /other\.csv/ }));
    expect(screen.queryByText("Column value is empty.")).not.toBeInTheDocument();
    expect(
      screen.getByText("Column id looks like an identifier."),
    ).toBeInTheDocument();
  });

  it("does not claim a clear session when summary counts lack issue details", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/quality", () =>
        HttpResponse.json({
          session_id: "r1",
          critical: 2,
          warn: 0,
          info: 0,
          datasets: [],
          issues: [],
        }),
      ),
    );

    renderAppAt("/projects/p1/sessions/r1/quality");

    expect(
      await screen.findByText("Issue details were not returned"),
    ).toBeInTheDocument();
    expect(screen.getByText(/reports 2 flags/)).toBeInTheDocument();
    expect(
      screen.queryByText("No quality flags recorded"),
    ).not.toBeInTheDocument();
  });

  it("states an evidence-limited empty result without implying cleaning ran", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/quality", () =>
        HttpResponse.json({
          session_id: "r1",
          critical: 0,
          warn: 0,
          info: 0,
          datasets: [],
          issues: [],
        }),
      ),
    );

    renderAppAt("/projects/p1/sessions/r1/quality");

    expect(
      await screen.findByText("No quality flags recorded"),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/did not record critical, warning, or informational/),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("link", { name: "Review cleaning options" }),
    ).not.toBeInTheDocument();
  });
});

describe("Profiles & Charts page", () => {
  const chartsUrl = "/projects/p1/sessions/r1/profiles?view=charts";

  it("restores dataset, search, type, and sort context from the URL", async () => {
    renderAppAt(
      "/projects/p1/sessions/r1/profiles?dataset=sample&q=value&kind=numeric&sort=name",
    );

    expect(await screen.findByDisplayValue("value")).toBeInTheDocument();
    expect(screen.getByLabelText("Sort")).toHaveValue("name");
    expect(
      screen.getByRole("button", { name: /numeric/i }),
    ).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText("1 of 3 columns")).toBeInTheDocument();
    expect(screen.getAllByText("value").length).toBeGreaterThan(0);
  });

  it("opens on field evidence and keeps charts in a separate task", async () => {
    renderAppAt("/projects/p1/sessions/r1/profiles");

    expect(
      await screen.findByRole("heading", { name: "Field profiles" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Field profiles" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    expect(screen.getByRole("tab", { name: "Charts" })).toHaveAttribute(
      "aria-selected",
      "false",
    );
    expect((await screen.findAllByText("sample.csv")).length).toBeGreaterThan(0);
    expect(screen.getByText("categorical: 1")).toBeInTheDocument();
    /* Field rows with formatted percents; null unique% renders empty. */
    expect(screen.getByText("40.0%")).toBeInTheDocument();
    expect(screen.queryByText("Value by name")).not.toBeInTheDocument();
  });

  it("restores the charts task from the URL", async () => {
    renderAppAt(chartsUrl);

    expect(await screen.findByText("Value by name")).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Charts" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    expect(screen.getByText("Value by name")).toBeInTheDocument();
    expect(screen.getByText("Value over id")).toBeInTheDocument();
  });

  it("hands each chart spec to vega-embed with the theme config", async () => {
    renderAppAt(chartsUrl);
    await screen.findByText("Value by name");

    await waitFor(() => expect(embedCalls).toHaveLength(2));
    const marks = embedCalls.map((spec) => spec["mark"]).sort();
    expect(marks).toEqual(["bar", "line"]);
    for (const spec of embedCalls) {
      /* Card header owns the title; the in-spec duplicate must be stripped. */
      expect(spec["title"]).toBeUndefined();
      expect(spec["width"]).toBe("container");
      const config = spec["config"] as Record<string, unknown>;
      expect(config["background"]).toBe("transparent");
      expect(config).toHaveProperty("range");
    }
    expect(embedCalls[0]).toHaveProperty("encoding");
    expect(embedCalls[0]).toHaveProperty("data");
  });

  it("embeds with only the local export action, a text-only tooltip, and an external-fetch-rejecting loader", async () => {
    renderAppAt(chartsUrl);
    await screen.findByText("Value by name");

    await waitFor(() => expect(embedOptions).toHaveLength(2));
    for (const opts of embedOptions) {
      /* Guard: the action menu may only offer the local PNG/SVG download.
       * "Open in Vega Editor" ships the whole spec to vega.github.io, and
       * source/compiled re-render the untrusted spec in a new window. */
      expect(opts["actions"]).toEqual({
        export: true,
        source: false,
        compiled: false,
        editor: false,
      });
      const actions = opts["actions"] as Record<string, unknown>;
      expect(actions["editor"]).toBe(false);
      expect(actions["source"]).toBe(false);
      /* Guard: the tooltip must never be plain `true` — vega-tooltip's stock
       * formatter renders a datum `image` field as <img>, fetching out around
       * the loader. Only our text-node formatter may be installed. */
      const tooltip = opts["tooltip"] as {
        formatTooltip?: (value: unknown) => string;
      };
      expect(typeof tooltip?.formatTooltip).toBe("function");
      const host = document.createElement("div");
      host.innerHTML = tooltip.formatTooltip!({
        image: "https://attacker.invalid/tracker.png",
      });
      expect(host.querySelector("img")).toBeNull();
      expect(host.textContent).toContain(
        "https://attacker.invalid/tracker.png",
      );
      const loader = opts["loader"] as Record<string, () => Promise<unknown>>;
      expect(loader).toBeDefined();
      for (const entry of ["load", "sanitize", "http", "file"] as const) {
        await expect(loader[entry]!()).rejects.toThrow(
          "external data loading disabled",
        );
      }
    }
  });

  it("draws a distribution sparkline beside the numeric column", async () => {
    renderAppAt("/projects/p1/sessions/r1/profiles");

    expect(
      await screen.findByRole("img", { name: "value distribution" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("img", { name: "name top values" }),
    ).toBeInTheDocument();
  });

  it("summarizes numeric and text fields with type-relevant evidence", async () => {
    renderAppAt("/projects/p1/sessions/r1/profiles");

    expect(await screen.findByText("0.5–9.5")).toBeInTheDocument();
    expect(screen.getByText("120 values · 5–7 chars")).toBeInTheDocument();
    expect(screen.getByText("100.0% distinct")).toBeInTheDocument();
  });

  it("opens the zoom modal with the plain-language caption", async () => {
    const user = userEvent.setup();
    renderAppAt(chartsUrl);
    await screen.findByText("Value by name");

    await user.click(screen.getAllByRole("button", { name: "Zoom" })[0]!);
    const dialog = await screen.findByRole("dialog", { name: "Value by name" });
    expect(dialog).toBeInTheDocument();
    expect(
      await within(dialog).findByText("Plain reading of chart_1."),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Close" }));
    expect(
      screen.queryByRole("dialog", { name: "Value by name" }),
    ).not.toBeInTheDocument();
  });

  it("opens a URL-backed split view whose left selection filters the right chart", async () => {
    const user = userEvent.setup();
    const { router, view } = renderAppWithRouterAt(
      chartsUrl,
    );
    await screen.findByText("Value by name");

    await user.click(
      screen.getByRole("button", { name: "Open linked split view" }),
    );
    const split = await screen.findByRole("region", {
      name: "Linked chart split view",
    });
    expect(within(split).getByLabelText("Left chart")).toHaveValue("chart_1");
    expect(within(split).getByLabelText("Right chart")).toHaveValue("chart_2");
    expect(router.state.location.search).toContain(
      "split=chart_1%2Cchart_2",
    );
    expect(
      within(split).getByText(/Click a mark in the left chart to filter/),
    ).toBeInTheDocument();

    await waitFor(() => {
      expect(
        embedCalls.some((spec) => Array.isArray(spec["hconcat"])),
      ).toBe(true);
    });
    const linked = embedCalls.find((spec) => Array.isArray(spec["hconcat"]))!;
    const children = linked["hconcat"] as Record<string, unknown>[];
    expect(children).toHaveLength(2);
    const left = children[0]!;
    const right = children[1]!;
    expect(left["params"]).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          name: "profile_split_selection",
          select: expect.objectContaining({
            type: "point",
            fields: ["value"],
            on: "click",
            clear: "dblclick",
          }),
        }),
      ]),
    );
    expect(right["transform"]).toEqual(
      expect.arrayContaining([
        {
          filter: {
            param: "profile_split_selection",
            empty: true,
          },
        },
      ]),
    );

    await user.click(within(split).getByRole("button", { name: "Zoom left" }));
    expect(
      await screen.findByRole("dialog", { name: "Value by name" }),
    ).toBeInTheDocument();

    view.unmount();
    renderAppAt(
      "/projects/p1/sessions/r1/profiles?view=charts&split=chart_1%2Cchart_2",
    );
    const restored = await screen.findByRole("region", {
      name: "Linked chart split view",
    });
    expect(within(restored).getByLabelText("Left chart")).toHaveValue("chart_1");
    expect(within(restored).getByLabelText("Right chart")).toHaveValue(
      "chart_2",
    );
  });

  it("restores a split URL whose second chart is on a later listing page", async () => {
    const requestedCursors: Array<string | null> = [];
    server.use(
      http.get("/api/v1/sessions/:sessionId/charts", ({ request }) => {
        const cursor = new URL(request.url).searchParams.get("cursor");
        requestedCursors.push(cursor);
        return HttpResponse.json(
          cursor === "page-2"
            ? {
                items: [
                  {
                    artifact_id: "chart_2",
                    title: "Value over id",
                    dataset_id: "sample",
                    dataset_name: "sample.csv",
                    mark: "line",
                    fields: ["id", "value"],
                    description: "Demo line chart.",
                  },
                ],
                next_cursor: null,
              }
            : {
                items: [
                  {
                    artifact_id: "chart_1",
                    title: "Value by name",
                    dataset_id: "sample",
                    dataset_name: "sample.csv",
                    mark: "bar",
                    fields: ["name", "value"],
                    description: "Demo bar chart.",
                  },
                ],
                next_cursor: "page-2",
              },
        );
      }),
    );

    renderAppAt(
      "/projects/p1/sessions/r1/profiles?view=charts&split=chart_1%2Cchart_2",
    );
    const split = await screen.findByRole("region", {
      name: "Linked chart split view",
    });
    expect(within(split).getByLabelText("Left chart")).toHaveValue("chart_1");
    expect(within(split).getByLabelText("Right chart")).toHaveValue("chart_2");
    expect(requestedCursors).toEqual([null, "page-2"]);
  });
});

/* Nine datasets used to render nine stacked column tables. Selection replaces
 * that: one table is mounted at a time and its columns are searchable. */
describe("Profiles dataset selection", () => {
  const field = (column: string, dtype: string, semantic: string) => ({
    column,
    dtype,
    semantic_type: semantic,
    missing_percent: 0,
    unique_percent: 100,
    sample_values: `${column}-1`,
  });
  const TWO_PROFILES = {
    session_id: "r1",
    datasets: [
      {
        dataset_id: "sample",
        name: "sample.csv",
        rows: 250,
        columns: 2,
        semantic_type_counts: { numeric: 1, categorical: 1 },
        fields: [
          field("alpha", "int64", "numeric"),
          field("beta", "string", "categorical"),
        ],
      },
      {
        dataset_id: "other",
        name: "other.csv",
        rows: 10,
        columns: 1,
        semantic_type_counts: { numeric: 1 },
        fields: [field("gamma", "float64", "numeric")],
      },
    ],
  };

  beforeEach(() => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/profiles", () =>
        HttpResponse.json(TWO_PROFILES),
      ),
    );
  });

  it("mounts one dataset's columns at a time and switches on selection", async () => {
    const user = userEvent.setup();
    renderAppAt("/projects/p1/sessions/r1/profiles");

    expect(await screen.findByText("alpha")).toBeInTheDocument();
    expect(screen.queryByText("gamma")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /other\.csv/ }));
    expect(await screen.findByText("gamma")).toBeInTheDocument();
    expect(screen.queryByText("alpha")).not.toBeInTheDocument();
  });

  it("writes the selected dataset and field controls to one restorable URL", async () => {
    const user = userEvent.setup();
    const { router, view } = renderAppWithRouterAt(
      "/projects/p1/sessions/r1/profiles",
    );
    await screen.findByText("alpha");

    await user.click(screen.getByRole("button", { name: /other\.csv/ }));
    await user.type(screen.getByLabelText("Find column"), "gam");
    await user.selectOptions(screen.getByLabelText("Sort"), "name");
    await user.click(screen.getByRole("button", { name: "numeric 1" }));
    const savedLocation =
      router.state.location.pathname + router.state.location.search;
    const params = new URLSearchParams(router.state.location.search);
    expect(Object.fromEntries(params)).toEqual({
      dataset: "other",
      kind: "numeric",
      q: "gam",
      sort: "name",
    });

    view.unmount();
    renderAppAt(savedLocation);
    expect(await screen.findByDisplayValue("gam")).toBeInTheDocument();
    expect(screen.getByLabelText("Sort")).toHaveValue("name");
    expect(screen.getByRole("button", { name: "numeric 1" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(screen.getByText("gamma")).toBeInTheDocument();
  });

  it("finds a column by name inside the selected dataset", async () => {
    renderAppAt("/projects/p1/sessions/r1/profiles");
    await screen.findByText("alpha");

    fireEvent.change(screen.getByLabelText("Find column"), {
      target: { value: "bet" },
    });
    expect(await screen.findByText("beta")).toBeInTheDocument();
    expect(screen.queryByText("alpha")).not.toBeInTheDocument();
  });

  it("filters the column list to one type", async () => {
    const user = userEvent.setup();
    renderAppAt("/projects/p1/sessions/r1/profiles");
    await screen.findByText("alpha");

    await user.click(screen.getByRole("button", { name: "text 1" }));
    expect(screen.queryByText("alpha")).not.toBeInTheDocument();
    expect(screen.getByText("beta")).toBeInTheDocument();
  });
});

/* Chart grouping: ported from _render_chart_artifacts in
 * Chart gallery — group by dataset_id, sort groups by
 * dataset display name, one bordered container + "Dataset: {name}" header
 * per group, 2-column grid only when the group has more than one chart. */
describe("Charts grouped by dataset", () => {
  const chartsUrl = "/projects/p1/sessions/r1/profiles?view=charts";
  const GROUPED_CHARTS = [
    {
      artifact_id: "b_chart_1",
      title: "B Chart One",
      dataset_id: "ds_1",
      dataset_name: "Bravo",
      mark: "bar",
      fields: ["x", "y"],
      description: "",
    },
    {
      artifact_id: "a_chart_1",
      title: "A Chart One",
      dataset_id: "ds_2",
      dataset_name: "Alpha",
      mark: "bar",
      fields: ["x", "y"],
      description: "",
    },
    {
      artifact_id: "a_chart_2",
      title: "A Chart Two",
      dataset_id: "ds_2",
      dataset_name: "Alpha",
      mark: "line",
      fields: ["x", "y"],
      description: "",
    },
  ];

  beforeEach(() => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/charts", () =>
        HttpResponse.json({ items: GROUPED_CHARTS, next_cursor: null }),
      ),
    );
  });

  it("groups same-dataset charts together, keeps different datasets apart, titles each group, and sorts groups by dataset display name", async () => {
    renderAppAt(chartsUrl);

    /* dataset_id order is ds_1 (Bravo) then ds_2 (Alpha), but Alpha sorts
     * first: the sort key is the display name, not id or encounter order. */
    const groupHeadings = await screen.findAllByText(/^Dataset: /);
    expect(groupHeadings.map((el) => el.textContent)).toEqual([
      "Dataset: Alpha",
      "Dataset: Bravo",
    ]);

    const alphaGroup = groupHeadings[0]!.closest("div")!;
    const bravoGroup = groupHeadings[1]!.closest("div")!;

    expect(within(alphaGroup).getByText("A Chart One")).toBeInTheDocument();
    expect(within(alphaGroup).getByText("A Chart Two")).toBeInTheDocument();
    expect(
      within(alphaGroup).queryByText("B Chart One"),
    ).not.toBeInTheDocument();
    expect(within(bravoGroup).getByText("B Chart One")).toBeInTheDocument();
    expect(within(bravoGroup).queryByText(/^A Chart/)).not.toBeInTheDocument();
  });

  it("grids a multi-chart group into 2 columns but leaves a single-chart group without a grid", async () => {
    renderAppAt(chartsUrl);

    const groupHeadings = await screen.findAllByText(/^Dataset: /);
    const alphaGroup = groupHeadings[0]!.closest("div")!; // 2 charts
    const bravoGroup = groupHeadings[1]!.closest("div")!; // 1 chart

    expect(alphaGroup.querySelector(".grid")).not.toBeNull();
    expect(bravoGroup.querySelector(".grid")).toBeNull();
  });

  it("drops the redundant per-card dataset name now that the group header carries it", async () => {
    renderAppAt(chartsUrl);
    await screen.findByText("Dataset: Alpha");

    expect(screen.queryByText("Alpha")).not.toBeInTheDocument();
    expect(screen.queryByText("Bravo")).not.toBeInTheDocument();
  });
});

/* Custom chart builder: ported from
 * Custom chart builder. The
 * default fixture dataset "sample" has columns id (int64), name (string),
 * value (float64) — see handlers.ts `sampleColumns`. */
describe("Custom chart builder", () => {
  const chartsUrl = "/projects/p1/sessions/r1/profiles?view=charts";

  async function openBuilder(user: ReturnType<typeof userEvent.setup>) {
    const toggle = await screen.findByRole("button", { name: "Open builder" });
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    await user.click(toggle);
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    await screen.findByRole("button", { name: "Build chart" });
  }

  it("sends y_column: null and forces drop_outliers: false when Y is Row count", async () => {
    const user = userEvent.setup();
    let sentBody: Record<string, unknown> | null = null;
    server.use(
      http.post(
        "/api/v1/sessions/:sessionId/charts/custom",
        async ({ request, params }) => {
          sentBody = (await request.json()) as Record<string, unknown>;
          return queueDataOperation(
            String(params["sessionId"]),
            "job_chart_request",
            {
            session_id: String(params["sessionId"]),
            dataset_id: String(sentBody["dataset_id"]),
            chart_type: String(sentBody["chart_type"]),
            aggregate: "count",
            row_count: 3,
            source_row_count: 3,
            truncated: false,
            row_limit: 5000,
            spec: { mark: "bar", encoding: {}, data: { values: [] } },
            },
          );
        },
      ),
    );

    renderAppAt(chartsUrl);
    await openBuilder(user);

    expect(screen.getByLabelText<HTMLSelectElement>("Y column")).toHaveValue(
      "Row count",
    );
    await user.click(screen.getByRole("button", { name: "Build chart" }));

    await waitFor(() => expect(sentBody).not.toBeNull());
    expect(sentBody!["y_column"]).toBeNull();
    expect(sentBody!["drop_outliers"]).toBe(false);
  });

  it("disables the outlier checkbox only while Y is Row count", async () => {
    const user = userEvent.setup();
    renderAppAt(chartsUrl);
    await openBuilder(user);

    expect(
      screen.getByLabelText("Exclude IQR outliers from Y"),
    ).toBeDisabled();

    await user.selectOptions(screen.getByLabelText("Y column"), "value");
    expect(
      screen.getByLabelText("Exclude IQR outliers from Y"),
    ).toBeEnabled();

    await user.selectOptions(screen.getByLabelText("Y column"), "Row count");
    expect(
      screen.getByLabelText("Exclude IQR outliers from Y"),
    ).toBeDisabled();
  });

  it("restricts the X column to numeric columns when chart type is histogram", async () => {
    const user = userEvent.setup();
    renderAppAt(chartsUrl);
    await openBuilder(user);

    await user.selectOptions(screen.getByLabelText("Chart type"), "histogram");
    const xSelect = screen.getByLabelText<HTMLSelectElement>("X column");
    const optionValues = Array.from(xSelect.options).map((o) => o.value);
    expect(optionValues).toEqual(["id", "value"]); // "name" (string) excluded
  });

  /* _dataset_display_labels only disambiguates collisions, so a unique name
   * must stay clean while duplicates gain an id suffix. */
  it("suffixes only same-named datasets in the picker", async () => {
    const base = {
      project_id: "p1",
      original_uri: "upload://sales.csv",
      format: "csv",
      content_hash: "aa",
      byte_size: 10,
      row_count: 10,
      schema: [{ name: "amount", dtype: "float64" }],
      ingest_status: "ready",
    };
    server.use(
      http.get("/api/v1/sessions/:sessionId/datasets", () =>
        HttpResponse.json([
          { ...base, dataset_id: "ds_aaaaaa111111", display_name: "sales.csv" },
          { ...base, dataset_id: "ds_bbbbbb222222", display_name: "sales.csv" },
          { ...base, dataset_id: "ds_cccccc333333", display_name: "refunds.csv" },
        ]),
      ),
    );
    const user = userEvent.setup();
    renderAppAt(chartsUrl);
    await openBuilder(user);

    const picker = await screen.findByRole("combobox", { name: "Dataset" });
    const labels = within(picker)
      .getAllByRole("option")
      .map((option) => option.textContent);
    expect(labels).toEqual([
      "sales.csv (111111)",
      "sales.csv (222222)",
      "refunds.csv",
    ]);
  });

  it("blocks histogram when no numeric column exists", async () => {
    const user = userEvent.setup();
    server.use(
      http.get("/api/v1/sessions/:sessionId/datasets", () =>
        HttpResponse.json([
          {
            dataset_id: "text_only",
            project_id: "p1",
            display_name: "text_only.csv",
            original_uri: "upload://text_only.csv",
            format: "csv",
            content_hash: "aa",
            byte_size: 10,
            row_count: 10,
            schema: [{ name: "label", dtype: "string" }],
            ingest_status: "ready",
          },
        ]),
      ),
      http.get(
        "/api/v1/sessions/:sessionId/datasets/:datasetId/schema",
        ({ params }) =>
          HttpResponse.json({
            dataset_id: String(params["datasetId"]),
            session_id: String(params["sessionId"]),
            columns: [{ name: "label", dtype: "string" }],
            source: "duckdb",
          }),
      ),
    );

    renderAppAt(chartsUrl);
    await openBuilder(user);
    await user.selectOptions(screen.getByLabelText("Chart type"), "histogram");

    expect(
      await screen.findByText("Histogram needs at least one numeric column."),
    ).toBeInTheDocument();
    expect(screen.queryByLabelText("X column")).not.toBeInTheDocument();
  });

  it("defaults aggregation per default_custom_agg: numeric Y sums, non-numeric Y counts", async () => {
    const user = userEvent.setup();
    renderAppAt(chartsUrl);
    await openBuilder(user);

    await user.selectOptions(screen.getByLabelText("Y column"), "value");
    expect(screen.getByLabelText<HTMLSelectElement>("Aggregation")).toHaveValue(
      "sum",
    );

    await user.selectOptions(screen.getByLabelText("Y column"), "name");
    expect(screen.getByLabelText<HTMLSelectElement>("Aggregation")).toHaveValue(
      "count",
    );

    await user.selectOptions(screen.getByLabelText("Y column"), "Row count");
    expect(screen.getByLabelText<HTMLSelectElement>("Aggregation")).toHaveValue(
      "count",
    );
  });

  it("shows a truncation hint when the response is truncated", async () => {
    const user = userEvent.setup();
    server.use(
      http.post("/api/v1/sessions/:sessionId/charts/custom", ({ params }) =>
        queueDataOperation(String(params["sessionId"]), "job_chart_truncated", {
          session_id: String(params["sessionId"]),
          dataset_id: "sample",
          chart_type: "bar",
          aggregate: "count",
          row_count: 500,
          source_row_count: 12000,
          truncated: true,
          row_limit: 5000,
          spec: { mark: "bar", encoding: {}, data: { values: [] } },
        }),
      ),
    );

    renderAppAt(chartsUrl);
    await openBuilder(user);
    await user.click(screen.getByRole("button", { name: "Build chart" }));

    expect(
      await screen.findByText(/^Chart preview is truncated:/i),
    ).toBeInTheDocument();
    expect(screen.getByText(/500/)).toBeInTheDocument();
    expect(screen.getByText(/12,000/)).toBeInTheDocument();
  });

  it("shows the empty state instead of a blank chart when row_count is 0", async () => {
    const user = userEvent.setup();
    server.use(
      http.post("/api/v1/sessions/:sessionId/charts/custom", ({ params }) =>
        queueDataOperation(String(params["sessionId"]), "job_chart_empty", {
          session_id: String(params["sessionId"]),
          dataset_id: "sample",
          chart_type: "bar",
          aggregate: "count",
          row_count: 0,
          source_row_count: 0,
          truncated: false,
          row_limit: 5000,
          spec: { mark: "bar", encoding: {}, data: { values: [] } },
        }),
      ),
    );

    renderAppAt(chartsUrl);
    await openBuilder(user);
    await user.click(screen.getByRole("button", { name: "Build chart" }));

    expect(
      await screen.findByText("No rows remain after the selected chart filters."),
    ).toBeInTheDocument();
  });

  it("renders the built chart through VegaChart", async () => {
    const user = userEvent.setup();
    renderAppAt(chartsUrl);
    await screen.findByText("Value by name");
    await waitFor(() => expect(embedCalls.length).toBeGreaterThanOrEqual(2));
    embedCalls.length = 0;
    embedOptions.length = 0;

    await openBuilder(user);
    await user.click(screen.getByRole("button", { name: "Build chart" }));

    await waitFor(() => expect(embedCalls).toHaveLength(1));
    expect(embedCalls[0]!["mark"]).toBe("bar");
  });
});
