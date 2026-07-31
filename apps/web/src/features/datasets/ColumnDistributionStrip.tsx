import { useState } from "react";
import type { ColumnDistribution } from "../../api/client";

function roundToSignificant(value: number, digits: number): number {
  if (value === 0) return 0;
  const magnitude = Math.floor(Math.log10(Math.abs(value))) + 1;
  const factor = 10 ** (digits - magnitude);
  return Math.round(value * factor) / factor;
}

function fmtDistNumber(value: number): string {
  if (Number.isInteger(value) && Math.abs(value) < 1e15) {
    return value.toLocaleString("en-US");
  }
  return roundToSignificant(value, 3).toLocaleString("en-US", {
    maximumFractionDigits: 20,
  });
}

function formatShare(count: number, total: number): string {
  if (total <= 0) return "0%";
  return `${((count / total) * 100).toFixed(1)}%`;
}

/* This floats over the first row instead of consuming a fixed line in every
 * header, so detailed hover feedback does not make the table taller. */
function HeaderHint({ children }: { children: string | null }) {
  if (!children) return null;

  return (
    <span
      aria-live="polite"
      aria-atomic="true"
      className="pointer-events-none absolute left-0 top-[calc(100%+4px)] z-40 w-max max-w-64 rounded-sm border border-border bg-text px-2 py-1 text-[10px] leading-tight text-bg shadow-md"
    >
      {children}
    </span>
  );
}

/* Every header reads the same way: one bar per bucket standing on a shared
 * baseline, height is the count. Numeric bins run low to high left to right,
 * so the bar outline is the distribution's shape. */
function VerticalBars({
  buckets,
  ariaLabel,
  testId,
}: {
  buckets: { key: string; label: string; count: number }[];
  ariaLabel: string;
  testId: string;
}) {
  const [activeIndex, setActiveIndex] = useState<number | null>(null);
  const peak = Math.max(...buckets.map((bucket) => bucket.count), 1);
  const total = buckets.reduce((sum, bucket) => sum + bucket.count, 0);
  const active = activeIndex === null ? null : buckets[activeIndex] ?? null;

  if (buckets.length === 0) return <MissingChart />;

  return (
    <div className="relative mt-1">
      <div
        aria-label={ariaLabel}
        className="flex h-7 items-end gap-px border-b border-table-border"
        role="group"
      >
        {buckets.map((bucket, index) => {
          const description = `${bucket.label}: ${bucket.count.toLocaleString("en-US")} records, ${formatShare(bucket.count, total)}`;
          const isActive = activeIndex === index;
          return (
            <button
              key={bucket.key}
              type="button"
              data-testid={testId}
              aria-label={description}
              title={description}
              onMouseEnter={() => setActiveIndex(index)}
              onMouseLeave={() => setActiveIndex(null)}
              onFocus={() => setActiveIndex(index)}
              onBlur={() => setActiveIndex(null)}
              className={`min-h-px flex-1 rounded-t-sm transition-opacity focus-visible:outline-offset-1 ${
                activeIndex !== null && !isActive
                  ? "bg-chart-1/30 opacity-45"
                  : "bg-chart-1/85 hover:bg-chart-1"
              }`}
              style={{ height: `${Math.max(4, (bucket.count / peak) * 100).toFixed(1)}%` }}
            />
          );
        })}
      </div>
      <HeaderHint
        children={
          active === null
            ? null
            : `${active.label} · ${active.count.toLocaleString("en-US")} records (${formatShare(active.count, total)})`
        }
      />
    </div>
  );
}

/* Bars stand side by side across a column that is 224px wide, so a header can
 * carry more bins than it could rows — but not the 60 Freedman-Diaconis will
 * ask for on a large table. Adjacent bins merge; Profiles keeps full detail. */
const MAX_HEADER_BARS = 24;

export function mergeBinsForHeader(
  counts: number[],
  edges: number[],
  maxBars = MAX_HEADER_BARS,
): { start: number; end: number; count: number }[] {
  if (counts.length === 0) return [];
  const groupSize = Math.ceil(counts.length / maxBars);
  const merged: { start: number; end: number; count: number }[] = [];
  for (let index = 0; index < counts.length; index += groupSize) {
    const slice = counts.slice(index, index + groupSize);
    merged.push({
      start: edges[index] ?? 0,
      end: edges[Math.min(index + slice.length, edges.length - 1)] ?? 0,
      count: slice.reduce((sum, value) => sum + value, 0),
    });
  }
  return merged;
}

function NumericHeaderChart({ dist }: { dist: ColumnDistribution }) {
  const merged = mergeBinsForHeader(dist.counts ?? [], dist.bin_edges ?? []);
  return (
    <VerticalBars
      ariaLabel="Column distribution"
      testId="header-distribution-bin"
      buckets={merged.map((bin, index) => ({
        key: String(index),
        label: `${fmtDistNumber(bin.start)} – ${fmtDistNumber(bin.end)}`,
        count: bin.count,
      }))}
    />
  );
}

function CategoricalHeaderChart({ dist }: { dist: ColumnDistribution }) {
  const rows = [
    ...(dist.top ?? []),
    ...(dist.other_count && dist.other_count > 0
      ? [{ value: "other", count: dist.other_count }]
      : []),
  ].slice(0, 6);
  return (
    <VerticalBars
      ariaLabel="Top values distribution"
      testId="header-distribution-category"
      buckets={rows.map((row) => ({
        key: row.value,
        label: row.value,
        count: row.count,
      }))}
    />
  );
}

function MissingChart() {
  return <span className="mt-2 block h-px w-full bg-track" />;
}

/** Interactive, header-sized distribution chart. It intentionally owns no
 * column label: the enclosing table header is that label, so a wide preview
 * never repeats names in a second card grid. */
export function HeaderDistributionChart({
  dist,
}: {
  dist: ColumnDistribution | undefined;
}) {
  if (!dist || dist.kind === "empty") return <MissingChart />;
  if (dist.kind === "numeric") return <NumericHeaderChart dist={dist} />;
  return <CategoricalHeaderChart dist={dist} />;
}
