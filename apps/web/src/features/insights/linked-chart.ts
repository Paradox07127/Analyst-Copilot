import type { ChartSummary } from "../../api/client";

export const LINKED_SELECTION_NAME = "profile_split_selection";
const DERIVED_CHART_FIELDS = new Set([
  "count",
  "column_a",
  "column_b",
  "bin",
  "bin_start",
  "bin_end",
  "bin_label",
  "period",
  "missing",
  "missing_count",
  "missing_percent",
  "association",
  "pearson",
  "spearman",
]);

export function sharedChartField(
  left: ChartSummary,
  right: ChartSummary,
): string | null {
  if (left.dataset_id !== right.dataset_id) return null;
  const rightFields = new Set(
    (right.fields ?? []).filter((field) => !DERIVED_CHART_FIELDS.has(field)),
  );
  return (
    (left.fields ?? []).find(
      (field) => !DERIVED_CHART_FIELDS.has(field) && rightFields.has(field),
    ) ?? null
  );
}

function concatChild(spec: Record<string, unknown>): Record<string, unknown> {
  const {
    $schema: _schema,
    config: _config,
    autosize: _autosize,
    width: _width,
    height: _height,
    ...child
  } = spec;
  return child;
}

/** Build one Vega-Lite multi-view spec so the selection signal is genuinely
 * shared. Two separately embedded charts cannot see each other's signals. */
export function buildLinkedChartSpec(
  leftSpec: Record<string, unknown>,
  rightSpec: Record<string, unknown>,
  field: string,
): Record<string, unknown> {
  const left = concatChild(leftSpec);
  const right = concatChild(rightSpec);
  const leftEncoding =
    typeof left["encoding"] === "object" && left["encoding"] !== null
      ? (left["encoding"] as Record<string, unknown>)
      : {};
  const rightTransforms = Array.isArray(right["transform"])
    ? right["transform"]
    : [];

  return {
    $schema: "https://vega.github.io/schema/vega-lite/v6.json",
    hconcat: [
      {
        ...left,
        params: [
          ...(Array.isArray(left["params"]) ? left["params"] : []),
          {
            name: LINKED_SELECTION_NAME,
            select: {
              type: "point",
              fields: [field],
              on: "click",
              clear: "dblclick",
            },
          },
        ],
        encoding: {
          ...leftEncoding,
          ...(leftEncoding["opacity"] === undefined
            ? {
                opacity: {
                  condition: {
                    param: LINKED_SELECTION_NAME,
                    empty: true,
                    value: 1,
                  },
                  value: 0.25,
                },
              }
            : {}),
        },
      },
      {
        ...right,
        transform: [
          ...rightTransforms,
          {
            filter: {
              param: LINKED_SELECTION_NAME,
              empty: true,
            },
          },
        ],
      },
    ],
    resolve: { scale: { color: "independent" } },
  };
}
