import { afterEach, describe, expect, it } from "vitest";
import { Handler } from "vega-tooltip";
import { compile } from "vega-lite";
import {
  formatTooltipAsText,
  tooltipOptions,
  vegaThemeConfig,
} from "../features/insights/VegaChart";
import {
  buildLinkedChartSpec,
  LINKED_SELECTION_NAME,
  sharedChartField,
} from "../features/insights/linked-chart";

/* A datum crafted the way a hostile chart artifact would be: `image` is the
 * field vega-tooltip's stock formatter turns into <img src>, `title` becomes
 * <h2>, and the remaining values carry markup and a javascript: URL. */
const HOSTILE_DATUM = {
  image: "https://attacker.invalid/tracker.png",
  title: "<h2>injected heading</h2>",
  label: '<img src=x onerror="globalThis.__pwned = true">',
  link: 'javascript:globalThis.__pwned = true"',
  count: "10,316",
};

/* Mirrors vega-tooltip's own call path: Handler.tooltipHandler assigns
 * formatTooltip's return value to the tooltip element's innerHTML. */
function renderTooltip(value: unknown): HTMLElement {
  const handler = new Handler(tooltipOptions());
  const call = handler.call as (
    h: unknown,
    e: MouseEvent,
    i: unknown,
    v: unknown,
  ) => void;
  call(null, new MouseEvent("mousemove", { clientX: 20, clientY: 20 }), null, value);
  const el = document.getElementById("vg-tooltip-element");
  if (!el) throw new Error("tooltip element was not created");
  return el;
}

afterEach(() => {
  document.getElementById("vg-tooltip-element")?.remove();
  delete (globalThis as Record<string, unknown>)["__pwned"];
});

describe("VegaChart tooltip", () => {
  it("renders hostile datum fields as text, never as elements", () => {
    const el = renderTooltip(HOSTILE_DATUM);

    expect(el.querySelector("img")).toBeNull();
    expect(el.querySelector("h2")).toBeNull();
    expect(el.querySelector("[onerror]")).toBeNull();
    expect(el.querySelector("a")).toBeNull();
    expect((globalThis as Record<string, unknown>)["__pwned"]).toBeUndefined();

    /* Only the table scaffold we build ourselves may appear. */
    const tags = new Set(
      [...el.querySelectorAll("*")].map((node) => node.tagName),
    );
    expect([...tags].sort()).toEqual(["TABLE", "TBODY", "TD", "TR"]);

    /* The values still reach the user — as text. */
    expect(el.textContent).toContain("https://attacker.invalid/tracker.png");
    expect(el.textContent).toContain("<h2>injected heading</h2>");
    expect(el.textContent).toContain('<img src=x onerror="globalThis.__pwned');
    expect(el.textContent).toContain("10,316");
  });

  it("keys every datum field and keeps number formatting readable", () => {
    const el = renderTooltip({ bin: "904.01–4302.57", count: "10,316" });

    const cells = [...el.querySelectorAll("td")].map((td) => [
      td.className,
      td.textContent,
    ]);
    expect(cells).toEqual([
      ["key", "bin"],
      ["value", "904.01–4302.57"],
      ["key", "count"],
      ["value", "10,316"],
    ]);
  });

  it("escapes markup in scalar tooltip values", () => {
    const html = formatTooltipAsText("<img src=x onerror=alert(1)>");
    const host = document.createElement("div");
    host.innerHTML = html;

    expect(host.querySelector("img")).toBeNull();
    expect(host.textContent).toContain("<img src=x onerror=alert(1)>");
  });
});

describe("vegaThemeConfig", () => {
  it("turns marks into tooltip sources with formatted numbers", () => {
    const config = vegaThemeConfig();

    /* Backend specs carry no tooltip encoding, so without this the handler
     * would never fire. */
    expect(config["mark"]).toMatchObject({ tooltip: true });
    expect(config["tooltipFormat"]).toEqual({ numberFormat: "," });
  });

  it("thins discrete x labels only", () => {
    const config = vegaThemeConfig();

    expect(config["axisXDiscrete"]).toEqual({
      labelOverlap: "greedy",
      labelSeparation: 2,
      labelLimit: 120,
    });
    /* Scoping matters: a plain axis/axisX rule would also rotate the
     * correlation heatmap's discrete y labels and the scatter plots' numeric
     * x labels. */
    expect(config["axisX"]).toBeUndefined();
    expect(config["axis"]).not.toHaveProperty("labelOverlap");
  });

  /* The correlation heatmap puts a discrete scale on both axes, so a plain
   * `axis`/`axisX` rule would thin the y column names too. Compiling the real
   * shape is the only thing that actually pins the scoping down. */
  it("leaves a discrete y axis alone when compiling a heatmap", () => {
    const compiled = compile(
      {
        mark: "rect",
        data: { values: [{ column_a: "a", column_b: "b", pearson: 0.4 }] },
        encoding: {
          x: { field: "column_a", type: "nominal" },
          y: { field: "column_b", type: "nominal" },
          color: { field: "pearson", type: "quantitative" },
        },
      },
      { config: vegaThemeConfig() },
    ).spec;

    const overlapFor = (scale: string) =>
      (compiled.axes ?? [])
        .filter((axis) => axis.scale === scale)
        .map((axis) => axis.labelOverlap);

    expect(overlapFor("x")).toContain("greedy");
    expect(overlapFor("y")).not.toContain("greedy");
  });
});

describe("linked chart split spec", () => {
  it("rejects derived aggregate fields and cross-dataset name collisions", () => {
    const summary = (
      artifact_id: string,
      dataset_id: string,
      fields: string[],
    ) => ({
      artifact_id,
      dataset_id,
      dataset_name: dataset_id,
      title: artifact_id,
      mark: "bar",
      fields,
      description: "",
    });

    expect(
      sharedChartField(
        summary("left", "d1", ["category", "count"]),
        summary("right", "d1", ["segment", "count"]),
      ),
    ).toBeNull();
    expect(
      sharedChartField(
        summary("left", "d1", ["category"]),
        summary("right", "d2", ["category"]),
      ),
    ).toBeNull();
    expect(
      sharedChartField(
        summary("left", "d1", ["category"]),
        summary("right", "d1", ["category", "amount"]),
      ),
    ).toBe("category");
    // Missingness heatmap vs correlation heatmap: both carry matrix axis
    // fields column_a/column_b with unrelated semantics — never linkable.
    expect(
      sharedChartField(
        summary("left", "d1", ["column_a", "column_b", "association"]),
        summary("right", "d1", ["column_a", "column_b", "pearson"]),
      ),
    ).toBeNull();
  });

  it("compiles a real cross-view selection predicate", () => {
    const linked = buildLinkedChartSpec(
      {
        mark: "bar",
        data: { values: [{ category: "A", amount: 3 }] },
        encoding: {
          x: { field: "category", type: "nominal" },
          y: { field: "amount", type: "quantitative" },
        },
      },
      {
        mark: "point",
        data: { values: [{ amount: 3, score: 9 }] },
        encoding: {
          x: { field: "amount", type: "quantitative" },
          y: { field: "score", type: "quantitative" },
        },
      },
      "amount",
    );

    expect(() =>
      compile(linked as unknown as Parameters<typeof compile>[0]),
    ).not.toThrow();
    const children = linked["hconcat"] as Record<string, unknown>[];
    expect(children).toHaveLength(2);
    const left = children[0]!;
    const right = children[1]!;
    expect(left["params"]).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ name: LINKED_SELECTION_NAME }),
      ]),
    );
    expect(right["transform"]).toEqual(
      expect.arrayContaining([
        { filter: { param: LINKED_SELECTION_NAME, empty: true } },
      ]),
    );
  });
});
