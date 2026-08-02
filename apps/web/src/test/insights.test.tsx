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
    expect(
      screen.getByRole("button", { name: "Critical", pressed: true }),
    ).toBeInTheDocument();
    expect(screen.getByLabelText(/empty_column \(1\)/)).toBeChecked();
    expect(screen.getByLabelText(/high_missing \(1\)/)).not.toBeChecked();
    expect(
      screen.queryByText("Column name has 40% missing values."),
    ).not.toBeInTheDocument();
  });

  it("renders a metric strip, dataset scope, and the issue queue", async () => {
    renderAppAt("/projects/p1/sessions/r1/quality");

    expect(
      await screen.findByText("Column value is empty."),
    ).toBeInTheDocument();
    /* "Critical" appears as metric label and issue-group heading. */
    expect(screen.getAllByText("Critical").length).toBeGreaterThanOrEqual(2);
    expect(screen.getAllByText("Critical")[0]?.parentElement).toHaveTextContent("1");
    expect(screen.getAllByText("Info")[0]?.parentElement).toHaveTextContent("1");
    expect(screen.getAllByText("empty_column").length).toBeGreaterThanOrEqual(1);
    expect(
      screen.getByText("Column id looks like an identifier."),
    ).toBeInTheDocument();
    expect(screen.getByText("Affected datasets")).toBeInTheDocument();
    expect(screen.getByText("Affected fields")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Issue queue" })).toBeInTheDocument();
    expect(screen.getAllByRole("link", { name: "Inspect rows" })[0]).toHaveAttribute(
      "href",
      "/projects/p1/sessions/r1/table/sample",
    );
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

    await user.click(screen.getByRole("button", { name: "Critical" }));
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
    await user.click(screen.getByRole("button", { name: "Info" }));
    const savedLocation =
      router.state.location.pathname + router.state.location.search;
    const params = new URLSearchParams(router.state.location.search);
    expect(params.get("dataset")).toBe("other");
    expect(params.get("severity")).toBe("info");

    view.unmount();
    renderAppAt(savedLocation);
    expect(await screen.findByLabelText("Dataset")).toHaveValue("other");
    expect(
      screen.getByRole("button", { name: "Info", pressed: true }),
    ).toBeInTheDocument();
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

    // Empty is a real filter state, not an accidental reset to every code.
    await user.click(screen.getByLabelText(/high_missing \(1\)/));
    await user.click(screen.getByLabelText(/likely_id \(1\)/));
    expect(
      screen.getByText("No issues match the selected filters"),
    ).toBeInTheDocument();
  });

  it("closes the issue-type menu when focus moves outside it", async () => {
    const user = userEvent.setup();
    renderAppAt("/projects/p1/sessions/r1/quality");
    await screen.findByText("Column value is empty.");

    const trigger = screen.getByText(/Issue types/, { selector: "summary" });
    const menu = trigger?.closest("details");
    expect(trigger).not.toBeNull();
    expect(menu).not.toBeNull();

    await user.click(trigger!);
    expect(menu).toHaveAttribute("open");

    await user.click(screen.getByRole("heading", { name: "Issue queue" }));
    expect(menu).not.toHaveAttribute("open");
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
      screen.getByRole("heading", { level: 3, name: "Critical" }).closest("details"),
    ).toHaveAttribute("open");
    expect(
      screen.getByRole("heading", { level: 3, name: "Warning" }).closest("details"),
    ).not.toHaveAttribute("open");

    expect(
      screen.getAllByRole("link", { name: "Inspect rows" })[0],
    ).toHaveAttribute("href", "/projects/p1/sessions/r1/table/sample");
  });

  it("scopes the issue list from the shared dataset selector", async () => {
    const user = userEvent.setup();
    renderAppAt("/projects/p1/sessions/r1/quality");
    await screen.findByText("Column value is empty.");

    await user.selectOptions(screen.getByLabelText("Dataset"), "other");
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

describe("Profiles and Charts pages", () => {
  const chartsUrl = "/projects/p1/sessions/r1/charts";

  it("restores dataset, search, type, and sort context from the URL", async () => {
    renderAppAt(
      "/projects/p1/sessions/r1/profiles?dataset=sample&q=value&kind=numeric&sort=name-asc",
    );

    expect(await screen.findByDisplayValue("value")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Column" }).closest("th"),
    ).toHaveAttribute("aria-sort", "ascending");
    expect(
      screen.getByRole("button", { name: /numeric/i }),
    ).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText("1 / 3")).toBeInTheDocument();
    expect(screen.getAllByText("value").length).toBeGreaterThan(0);
  });

  it("opens the merged workspace directly on field evidence", async () => {
    renderAppAt("/projects/p1/sessions/r1/profiles");

    expect(
      await screen.findByRole("heading", { name: "Profiles & charts" }),
    ).toBeInTheDocument();
    expect(await screen.findByLabelText("Dataset")).toHaveValue("sample");
    expect(
      screen.getByRole("button", { name: "Profile", pressed: true }),
    ).toBeInTheDocument();
    expect(screen.queryByText("Field mix")).not.toBeInTheDocument();
    expect(screen.queryByText("Tables profiled")).not.toBeInTheDocument();
    /* Field rows with formatted percents; null unique% renders empty. */
    expect(screen.getByText("40.0%")).toBeInTheDocument();
    expect(screen.queryByText("Value by name")).not.toBeInTheDocument();
  });

  it("redirects the legacy Charts route into the merged chart view", async () => {
    const { router } = renderAppWithRouterAt(chartsUrl);

    expect(await screen.findByRole("heading", { level: 1, name: "Profiles & charts" })).toBeInTheDocument();
    expect(router.state.location.pathname).toBe("/projects/p1/sessions/r1/profiles");
    expect(
      await screen.findByRole("button", { name: "Charts", pressed: true }),
    ).toBeInTheDocument();
    expect(await screen.findByText("Value by name")).toBeInTheDocument();
    expect(screen.getByText("Value by name")).toBeInTheDocument();
    expect(screen.getByText("Value over id")).toBeInTheDocument();
  });

  it("keeps a legacy charts switch URL in the merged workspace", async () => {
    const { router } = renderAppWithRouterAt(
      "/projects/p1/sessions/r1/profiles?view=charts&split=chart_1%2Cchart_2",
    );

    expect(
      await screen.findByRole("heading", { level: 1, name: "Profiles & charts" }),
    ).toBeInTheDocument();
    expect(router.state.location.pathname).toBe(
      "/projects/p1/sessions/r1/profiles",
    );
    expect(router.state.location.search).toContain("split=chart_1%2Cchart_2");
    expect(router.state.location.search).toContain("view=charts");
  });

  it("folds single-field diagnostics into a disclosure instead of dropping them", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/charts", () =>
        HttpResponse.json({
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
            {
              artifact_id: "distribution_value",
              title: "Distribution of value",
              dataset_id: "sample",
              dataset_name: "sample.csv",
              mark: "bar",
              fields: ["value"],
              description: "Single-field diagnostic.",
            },
          ],
          next_cursor: null,
        }),
      ),
    );

    renderAppAt(chartsUrl);

    expect(await screen.findByText("Value by name")).toBeInTheDocument();
    /* The gallery still counts only the analytical chart, but the diagnostic is
     * on the page behind a collapsed disclosure rather than nowhere at all. */
    expect(screen.getByText("1 analytical chart")).toBeInTheDocument();
    const summary = screen.getByText(
      /Field diagnostics: 1 distribution or top-value chart/,
    );
    expect(screen.getByText("Distribution of value")).not.toBeVisible();

    await userEvent.click(summary);

    expect(screen.getByText("Distribution of value")).toBeVisible();
    expect(screen.getByRole("link", { name: "Review missingness" })).toHaveAttribute(
      "href",
      "/projects/p1/sessions/r1/quality?dataset=sample",
    );
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
      "/projects/p1/sessions/r1/charts?split=chart_1%2Cchart_2",
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
      "/projects/p1/sessions/r1/charts?split=chart_1%2Cchart_2",
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

    await user.selectOptions(screen.getByLabelText("Dataset"), "other");
    expect(await screen.findByText("gamma")).toBeInTheDocument();
    expect(screen.queryByText("alpha")).not.toBeInTheDocument();
  });

  it("writes the selected dataset and field controls to one restorable URL", async () => {
    const user = userEvent.setup();
    const { router, view } = renderAppWithRouterAt(
      "/projects/p1/sessions/r1/profiles",
    );
    await screen.findByText("alpha");

    await user.selectOptions(screen.getByLabelText("Dataset"), "other");
    await user.click(screen.getByRole("button", { name: "Find column" }));
    await user.type(screen.getByLabelText("Find column"), "gam");
    await user.click(screen.getByRole("button", { name: "Column" }));
    await user.click(screen.getByRole("button", { name: "numeric 1" }));
    const savedLocation =
      router.state.location.pathname + router.state.location.search;
    const params = new URLSearchParams(router.state.location.search);
    expect(Object.fromEntries(params)).toEqual({
      dataset: "other",
      kind: "numeric",
      q: "gam",
      sort: "name-asc",
    });

    view.unmount();
    renderAppAt(savedLocation);
    expect(await screen.findByDisplayValue("gam")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Column" }).closest("th"),
    ).toHaveAttribute("aria-sort", "ascending");
    expect(screen.getByRole("button", { name: "numeric 1" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(screen.getByText("gamma")).toBeInTheDocument();
  });

  it("finds a column by name inside the selected dataset", async () => {
    renderAppAt("/projects/p1/sessions/r1/profiles");
    await screen.findByText("alpha");

    fireEvent.click(screen.getByRole("button", { name: "Find column" }));
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

/* Chart gallery: datasets are sorted by display name and selected one at a
 * time, avoiding a long stack of unrelated chart groups. */
describe("Charts grouped by dataset", () => {
  const chartsUrl = "/projects/p1/sessions/r1/charts";
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
      http.get("/api/v1/sessions/:sessionId/profiles", () =>
        HttpResponse.json({
          session_id: "r1",
          datasets: [
            {
              dataset_id: "ds_2",
              name: "Alpha",
              rows: 20,
              columns: 2,
              semantic_type_counts: { numeric: 2 },
              fields: [],
            },
            {
              dataset_id: "ds_1",
              name: "Bravo",
              rows: 10,
              columns: 2,
              semantic_type_counts: { numeric: 2 },
              fields: [],
            },
          ],
        }),
      ),
    );
  });

  it("uses the shared dataset selector and initially shows one dataset", async () => {
    renderAppAt(chartsUrl);

    const dataset = await screen.findByLabelText("Dataset");
    expect(dataset).toHaveValue("ds_2");
    expect(within(dataset).getByRole("option", { name: "Alpha" })).toBeInTheDocument();
    expect(within(dataset).getByRole("option", { name: "Bravo" })).toBeInTheDocument();
    expect(screen.getByText("A Chart One")).toBeInTheDocument();
    expect(screen.getByText("A Chart Two")).toBeInTheDocument();
    expect(screen.queryByText("B Chart One")).not.toBeInTheDocument();
  });

  it("shows six chart cards at a time and exposes Load more", async () => {
    const requestedLimits: string[] = [];
    const manyCharts = Array.from({ length: 7 }, (_, index) => ({
      artifact_id: `a_chart_${index + 1}`,
      title: `A Chart ${index + 1}`,
      dataset_id: "ds_2",
      dataset_name: "Alpha",
      mark: "bar",
      fields: ["x", "y"],
      description: "",
    }));
    server.use(
      http.get("/api/v1/sessions/:sessionId/charts", ({ request }) => {
        requestedLimits.push(new URL(request.url).searchParams.get("limit") ?? "");
        return HttpResponse.json({ items: manyCharts, next_cursor: null });
      }),
    );

    renderAppAt(chartsUrl);

    await screen.findByText("A Chart 1");
    expect(requestedLimits).toEqual(["100"]);
    expect(screen.getByText("A Chart 6")).toBeInTheDocument();
    expect(screen.queryByText("A Chart 7")).not.toBeInTheDocument();
    await userEvent.setup().click(
      screen.getByRole("button", { name: "Load more charts" }),
    );
    expect(screen.getByText("A Chart 7")).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Load more charts" }),
    ).not.toBeInTheDocument();
  });

  it("switches the visible chart group when another dataset is selected", async () => {
    const user = userEvent.setup();
    renderAppAt(chartsUrl);

    await user.selectOptions(await screen.findByLabelText("Dataset"), "ds_1");
    expect(await screen.findByText("B Chart One")).toBeInTheDocument();
    expect(screen.queryByText("A Chart One")).not.toBeInTheDocument();
  });

  it("drops redundant dataset headings from chart cards", async () => {
    renderAppAt(chartsUrl);
    expect(await screen.findByLabelText("Dataset")).toHaveValue("ds_2");

    expect(screen.queryByRole("heading", { name: "Alpha" })).not.toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Bravo" })).not.toBeInTheDocument();
  });
});

/* Custom chart builder: ported from
 * Custom chart builder. The
 * default fixture dataset "sample" has columns id (int64), name (string),
 * value (float64) — see handlers.ts `sampleColumns`. */
describe("Custom chart builder", () => {
  const chartsUrl = "/projects/p1/sessions/r1/charts";

  async function openBuilder(user: ReturnType<typeof userEvent.setup>) {
    const toggle = await screen.findByRole("button", { name: "Build a custom chart" });
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

  it("names X as the fenced axis for a histogram and keeps the switch usable", async () => {
    const user = userEvent.setup();
    renderAppAt(chartsUrl);
    await openBuilder(user);

    await user.selectOptions(screen.getByLabelText("Chart type"), "histogram");
    /* The server fences the histogram's X column, so a Row-count Y must not
     * disable the switch and the copy must not promise a Y fence. */
    expect(
      screen.queryByLabelText("Exclude IQR outliers from Y"),
    ).not.toBeInTheDocument();
    expect(screen.getByLabelText("Exclude IQR outliers from X")).toBeEnabled();
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

  it("reuses the workspace dataset instead of rendering a second picker", async () => {
    const user = userEvent.setup();
    renderAppAt(chartsUrl);
    await openBuilder(user);

    expect(screen.getAllByRole("combobox", { name: "Dataset" })).toHaveLength(1);
    const selectedName = screen.getAllByText("sample.csv").find(
      (node) => node.closest("p")?.textContent?.includes("Building from"),
    );
    expect(selectedName?.closest("p")).toHaveTextContent(
      "Building from sample.csv, the dataset selected above.",
    );
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

  it("says groups were dropped, not rows, when an aggregate is byte-capped", async () => {
    const user = userEvent.setup();
    server.use(
      http.post("/api/v1/sessions/:sessionId/charts/custom", ({ params }) =>
        queueDataOperation(String(params["sessionId"]), "job_chart_series", {
          session_id: String(params["sessionId"]),
          dataset_id: "sample",
          chart_type: "bar",
          aggregate: "sum",
          /* An aggregated chart counts every observation, so the row wording
           * would read "showing 12,000 of 12,000 rows". */
          row_count: 12000,
          source_row_count: 12000,
          truncated: true,
          series_truncated: true,
          row_limit: 5000,
          spec: { mark: "bar", encoding: {}, data: { values: [] } },
        }),
      ),
    );

    renderAppAt(chartsUrl);
    await openBuilder(user);
    await user.click(screen.getByRole("button", { name: "Build chart" }));

    expect(
      await screen.findByText(/some groups were dropped to fit the size limit/i),
    ).toBeInTheDocument();
    expect(screen.queryByText(/12,000 of 12,000 rows/)).not.toBeInTheDocument();
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
