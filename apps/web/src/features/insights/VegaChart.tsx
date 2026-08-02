/* Vega-lite renderer. vega-embed is dynamically imported so the vega stack
 * ships as its own lazy chunk (§13.2); theme colors are read from the design
 * tokens at render time so light/dark stays consistent without hardcoded hex. */

import { useEffect, useRef, useState } from "react";
import { getEffectiveTheme } from "../../app/theme";

type EmbedResult = { view: { finalize(): void } };

/* Threat model: chart specs are artifact payloads and therefore untrusted; a
 * crafted spec (data.url, external images/links) must not make the browser
 * fetch out. Every loader entry point rejects, so only inline "values" data —
 * which never touches the loader — can render. */
const rejectExternal = () =>
  Promise.reject(new Error("external data loading disabled"));
const noExternalLoader = {
  load: rejectExternal,
  sanitize: rejectExternal,
  http: rejectExternal,
  file: rejectExternal,
};

/* Second half of that threat model: vega evaluates expression strings in the
 * spec. The server's get_chart rejects them, but ChatPage renders a raw
 * artifact payload from the generic artifact endpoint, which never passes
 * through that gate — so the check also lives at the render seam, where every
 * caller crosses. Key set mirrors insight_service._EXPRESSION_KEYS and
 * _STRING_EXPRESSION_KEYS; `filter` is only an expression in its string form,
 * a field predicate is data. Data values are keyed by column name, so scanning
 * them would reject a dataset with a column called "expr" without adding
 * safety — vega evaluates expressions in the spec, not in the data. */
const EXPRESSION_KEYS = new Set(["expr", "labelExpr", "signal", "calculate"]);
const STRING_EXPRESSION_KEYS = new Set(["filter"]);

export function containsVegaExpression(node: unknown, inData = false): boolean {
  if (Array.isArray(node)) {
    return node.some((item) => containsVegaExpression(item, inData));
  }
  if (typeof node !== "object" || node === null) return false;
  for (const [key, value] of Object.entries(node)) {
    if (inData) continue;
    if (EXPRESSION_KEYS.has(key)) return true;
    if (STRING_EXPRESSION_KEYS.has(key) && typeof value === "string") return true;
    if (key === "condition") {
      const conditions = Array.isArray(value) ? value : [value];
      for (const condition of conditions) {
        if (
          typeof condition === "object" &&
          condition !== null &&
          typeof (condition as Record<string, unknown>)["test"] === "string"
        ) {
          return true;
        }
      }
    }
    if (containsVegaExpression(value, key === "data")) return true;
  }
  return false;
}

/* Only the local PNG/SVG download stays on. "Open in Vega Editor" POSTs the
 * whole spec — data included — to vega.github.io, and source/compiled open
 * new windows rendering the same untrusted spec, so all three stay off. */
export const CHART_ACTIONS = {
  export: true,
  source: false,
  compiled: false,
  editor: false,
} as const;

/* vega-tooltip assigns formatTooltip's return value straight to innerHTML, and
 * its stock formatter emits `<img src=...>` for a datum `image` field, which
 * fetches out without ever reaching noExternalLoader. Every cell here is filled
 * through textContent, so the serializer escapes datum values and no element or
 * attribute can originate from the spec. */
export function formatTooltipAsText(value: unknown): string {
  const table = document.createElement("table");
  const rows: [string, unknown][] =
    typeof value === "object" && value !== null && !Array.isArray(value)
      ? Object.entries(value as Record<string, unknown>)
      : [["value", value]];
  for (const [key, raw] of rows) {
    if (raw === undefined) continue;
    const row = table.insertRow();
    const keyCell = row.insertCell();
    keyCell.className = "key";
    keyCell.textContent = key;
    const valueCell = row.insertCell();
    valueCell.className = "value";
    valueCell.textContent = String(raw);
  }
  return table.outerHTML;
}

/* Numbers arrive pre-formatted: vega-lite compiles the tooltip encoder as
 * format(datum[...], config.tooltipFormat.numberFormat), so String() is enough. */
export function tooltipOptions(): Record<string, unknown> {
  return { formatTooltip: formatTooltipAsText, theme: getEffectiveTheme() };
}

/* Without an explicit height charts fall back to vega's default 200px, which on
 * the 30-bin histograms left the vertical label gutter as tall as the plot. */
const CHART_HEIGHT = 240;

function token(name: string): string {
  return getComputedStyle(document.documentElement)
    .getPropertyValue(name)
    .trim();
}

export function vegaThemeConfig(): Record<string, unknown> {
  const text = token("--color-text");
  const border = token("--color-border");
  const neutral = token("--color-status-neutral");
  return {
    background: "transparent",
    font: token("--font-sans"),
    /* Single-series marks ignore the category range; without this they fall
     * back to vega's default blue instead of the brand token. tooltip:true is
     * what makes marks emit a tooltip at all — the backend specs carry no
     * tooltip encoding, so the handler alone would never fire. */
    mark: { color: token("--chart-1"), tooltip: true },
    tooltipFormat: { numberFormat: "," },
    axis: {
      labelColor: neutral,
      titleColor: text,
      gridColor: border,
      domainColor: border,
      tickColor: border,
    },
    /* Band scales leave labelOverlap off, so all 30 histogram bin labels
     * ("904.01–4302.57") render at once and collide into an unreadable stripe.
     * Thinning alone fixes that; vega's own vertical angle is kept because any
     * shallower angle widens each label enough that greedy also drops most of
     * the 10 categories on the top-values bars, where the names are the data.
     * Scoped to axisXDiscrete rather than axis/axisX so the correlation
     * heatmap's discrete y axis and the scatter plots' quantitative x axis
     * keep vega's defaults. */
    axisXDiscrete: {
      labelOverlap: "greedy",
      labelSeparation: 2,
      /* Caps the label gutter: vertical labels are as tall as they are long. */
      labelLimit: 120,
    },
    legend: { labelColor: neutral, titleColor: text },
    title: { color: text },
    range: {
      category: [1, 2, 3, 4, 5, 6, 7].map((i) => token(`--chart-${i}`)),
    },
  };
}

export function VegaChart({
  spec,
  label,
}: {
  spec: Record<string, unknown>;
  label: string;
}) {
  const container = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let disposed = false;
    let result: EmbedResult | null = null;
    (async () => {
      try {
        if (containsVegaExpression(spec)) {
          throw new Error("spec rejected by the expression safety check");
        }
        const { default: embed } = await import("vega-embed");
        if (disposed || !container.current) return;
        const embedSpec = {
          ...spec,
          width: "container",
          height: CHART_HEIGHT,
          config: vegaThemeConfig(),
        };
        const embedded = (await embed(
          container.current,
          embedSpec as unknown as Parameters<typeof embed>[1],
          {
            actions: { ...CHART_ACTIONS },
            renderer: "svg",
            loader: noExternalLoader,
            tooltip: tooltipOptions(),
          },
        )) as EmbedResult;
        /* Unmounting mid-embed used to leave the resolved view unfinalized:
         * cleanup had already run against a null `result`. */
        if (disposed) {
          embedded.view.finalize();
          return;
        }
        result = embedded;
      } catch (cause) {
        if (!disposed) {
          setError(cause instanceof Error ? cause.message : "Chart failed to render");
        }
      }
    })();
    return () => {
      disposed = true;
      result?.view.finalize();
    };
  }, [spec]);

  if (error) {
    return (
      <p role="alert" className="text-sm text-status-critical">
        Chart failed to render: {error}
      </p>
    );
  }
  return <div ref={container} role="img" aria-label={label} className="w-full" />;
}
