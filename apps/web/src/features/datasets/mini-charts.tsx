/* Column-type vocabulary and the micro-charts that go beside a column name.
 *
 * Hand-rolled SVG rather than a vega-lite micro-spec: every vega-embed call
 * builds a dataflow graph and a View that has to be finalized, and a profile
 * table mounts one chart per visible row. */

import type { ColumnDistribution } from "../../api/client";

export type ColumnKind =
  | "numeric"
  | "text"
  | "temporal"
  | "boolean"
  | "id"
  | "other";

/* Ordered widest-to-narrowest so a composition bar always reads left to right
 * in the same order regardless of which kinds a table happens to contain. */
export const COLUMN_KINDS: readonly ColumnKind[] = [
  "numeric",
  "text",
  "temporal",
  "boolean",
  "id",
  "other",
];

export const KIND_LABEL: Record<ColumnKind, string> = {
  numeric: "numeric",
  text: "text",
  temporal: "date/time",
  boolean: "boolean",
  id: "identifier",
  other: "other",
};

const KIND_GLYPH: Record<ColumnKind, string> = {
  numeric: "#",
  text: "A",
  temporal: "T",
  boolean: "B",
  id: "ID",
  other: "?",
};

/* Field kinds are categorical data, but this is a compact operational card:
 * restrict the ramp to cool and neutral hues so none of its segments reads as
 * a red error, amber warning, or green success state. */
const KIND_TINT: Record<ColumnKind, string> = {
  numeric: "bg-chart-7",
  text: "bg-chart-1",
  temporal: "bg-chart-3",
  boolean: "bg-status-llm/65",
  id: "bg-status-neutral/75",
  other: "bg-status-neutral/45",
};

/* The profiler's semantic_type outranks the dtype: a CSV date or amount column
 * reads back as `object`, and letting the dtype answer first filed every parsed
 * column under "text", contradicting the semantic type shown beside it. */
export function classifyColumn(dtype: string, semanticType = ""): ColumnKind {
  const semantic = semanticType.toLowerCase();
  if (semantic === "id" || semantic === "identifier") return "id";
  if (semantic === "boolean") return "boolean";
  if (semantic === "datetime" || semantic === "temporal") return "temporal";
  if (semantic === "numeric" || semantic === "continuous") return "numeric";
  if (semantic === "categorical" || semantic === "text") return "text";
  const type = dtype.toLowerCase();
  if (/bool/.test(type)) return "boolean";
  if (/datetime|timestamp|date|time|period/.test(type)) return "temporal";
  if (/^(u?int|float|double|decimal|numeric|number)/.test(type)) return "numeric";
  if (/str|object|categor|utf8|char|text/.test(type)) return "text";
  return "other";
}

/** Decorative: the dtype and semantic type sit in the next cell, so the glyph
 *  would only repeat them to a screen reader. */
export function TypeIcon({ kind }: { kind: ColumnKind }) {
  return (
    <span
      aria-hidden="true"
      title={KIND_LABEL[kind]}
      className="inline-flex h-4 w-5 shrink-0 items-center justify-center rounded-xs bg-code-bg font-mono text-[10px] leading-none text-status-neutral"
    >
      {KIND_GLYPH[kind]}
    </span>
  );
}

const SPARK_WIDTH = 88;
const SPARK_HEIGHT = 20;

function NumericSpark({ dist }: { dist: ColumnDistribution }) {
  const counts = dist.counts ?? [];
  if (counts.length === 0) return <MissingSpark />;
  const peak = Math.max(...counts, 1);
  const gap = counts.length > 40 ? 0 : 1;
  const barWidth = (SPARK_WIDTH - gap * (counts.length - 1)) / counts.length;
  return (
    <svg
      viewBox={`0 0 ${SPARK_WIDTH} ${SPARK_HEIGHT}`}
      width={SPARK_WIDTH}
      height={SPARK_HEIGHT}
      role="img"
      aria-label={`${dist.name} distribution`}
      className="text-primary"
      preserveAspectRatio="none"
    >
      {counts.map((count, index) => {
        const height = Math.max(1, (count / peak) * SPARK_HEIGHT);
        return (
          <rect
            key={index}
            x={index * (barWidth + gap)}
            y={SPARK_HEIGHT - height}
            width={barWidth}
            height={height}
            fill="currentColor"
            opacity={0.8}
          />
        );
      })}
    </svg>
  );
}

function CategorySpark({ dist }: { dist: ColumnDistribution }) {
  const top = dist.top ?? [];
  if (top.length === 0) return <MissingSpark />;
  const rows = top.slice(0, 4);
  const peak = Math.max(...rows.map((row) => row.count), 1);
  const rowHeight = SPARK_HEIGHT / rows.length;
  return (
    <svg
      viewBox={`0 0 ${SPARK_WIDTH} ${SPARK_HEIGHT}`}
      width={SPARK_WIDTH}
      height={SPARK_HEIGHT}
      role="img"
      aria-label={`${dist.name} top values`}
      className="text-primary"
      preserveAspectRatio="none"
    >
      {rows.map((row, index) => (
        <rect
          key={index}
          x={0}
          y={index * rowHeight}
          width={Math.max(1, (row.count / peak) * SPARK_WIDTH)}
          height={Math.max(1, rowHeight - 1)}
          fill="currentColor"
          opacity={0.8}
        />
      ))}
    </svg>
  );
}

/* Reserves the column's width and draws nothing. It used to draw a flat rule,
 * which in a shape column reads as "this distribution is flat" — the opposite
 * of "there is no distribution for this column". */
function MissingSpark() {
  return <span aria-hidden="true" className="inline-block w-8 align-middle" />;
}

/** Mini histogram for numeric columns, top-value bars for categorical ones —
 *  Rill's split, which is what makes a wide table skimmable by shape. */
export function DistributionSpark({
  dist,
}: {
  dist: ColumnDistribution | undefined;
}) {
  if (!dist) return <MissingSpark />;
  if (dist.kind === "numeric") return <NumericSpark dist={dist} />;
  if (dist.kind === "categorical") return <CategorySpark dist={dist} />;
  return <MissingSpark />;
}

/** Null rate as its own track, so a column that is 100% empty is visible
 *  without reading the number beside it. */
export function MissingBar({
  percent,
  width = "w-10",
}: {
  percent: number | null | undefined;
  width?: string;
}) {
  const value = Math.min(Math.max(percent ?? 0, 0), 100);
  const tone =
    value >= 100
      ? "bg-status-critical"
      : value > 0
        ? "bg-status-warn"
        : "bg-transparent";
  return (
    <span
      aria-hidden="true"
      className={`inline-block h-1.5 ${width} shrink-0 overflow-hidden rounded-sm bg-track align-middle`}
    >
      <span
        className={`block h-full rounded-sm ${tone}`}
        style={{ width: `${value.toFixed(1)}%` }}
      />
    </span>
  );
}

export type KindCount = { kind: ColumnKind; count: number };

export function countColumnKinds(
  columns: readonly { dtype: string; semantic_type?: string }[],
): KindCount[] {
  const counts = new Map<ColumnKind, number>();
  for (const column of columns) {
    const kind = classifyColumn(column.dtype, column.semantic_type ?? "");
    counts.set(kind, (counts.get(kind) ?? 0) + 1);
  }
  return COLUMN_KINDS.filter((kind) => (counts.get(kind) ?? 0) > 0).map(
    (kind) => ({ kind, count: counts.get(kind) ?? 0 }),
  );
}

/** One bar showing what a table is made of. Replaces the six raw column-name
 *  chips that told you nothing about the other 40 columns. */
export function KindCompositionBar({ counts }: { counts: KindCount[] }) {
  const total = counts.reduce((sum, entry) => sum + entry.count, 0);
  if (total === 0) return null;
  return (
    <span
      role="img"
      aria-label={`Column types: ${kindCompositionText(counts)}`}
      className="flex h-1.5 w-full overflow-hidden rounded-sm bg-track"
    >
      {counts.map((entry) => (
        <span
          key={entry.kind}
          title={`${entry.count} ${KIND_LABEL[entry.kind]}`}
          className={KIND_TINT[entry.kind]}
          style={{ width: `${((entry.count / total) * 100).toFixed(2)}%` }}
        />
      ))}
    </span>
  );
}

export function kindCompositionText(counts: KindCount[]): string {
  return counts
    .map((entry) => `${entry.count} ${KIND_LABEL[entry.kind]}`)
    .join(" · ");
}
