import { useEffect, useMemo, useRef, useState } from "react";
import { Link, Navigate, useLocation, useParams } from "react-router";
import { useVirtualizer } from "@tanstack/react-virtual";
import type {
  ChartSummary,
  ColumnDistribution,
  DatasetProfileSummary,
} from "../../api/client";
import {
  useChart,
  useCharts,
  useDatasetDistributions,
  useProfiles,
  useQuality,
} from "../../api/hooks";
import {
  EmptyState,
  ErrorState,
  LoadingSkeleton,
} from "../../components/async-states";
import {
  DataWorkspacePage,
  DatasetScopeBar,
  SegmentedControl,
} from "../../components/data-workspace";
import {
  Card,
  Hint,
  Marquee,
  MetricStrip,
  MetricTile,
  formatCompact,
  formatPercent,
} from "../../components/ui";
import { sessionSectionPath } from "../../app/paths";
import {
  DistributionSpark,
  KIND_LABEL,
  MissingBar,
  TypeIcon,
  classifyColumn,
  type ColumnKind,
} from "../datasets/mini-charts";
import { CustomChartBuilder } from "./CustomChartBuilder";
import { useInView } from "./useInView";
import { VegaChart } from "./VegaChart";
import { useDialogFocus } from "../../components/use-dialog-focus";
import { useRouteSearchParam } from "../../app/route-state";
import {
  buildLinkedChartSpec,
  sharedChartField,
} from "./linked-chart";

type FieldRow = NonNullable<DatasetProfileSummary["fields"]>[number];
type SortKey =
  | "table"
  | "missing-asc"
  | "missing-desc"
  | "name-asc"
  | "name-desc"
  | "type-asc"
  | "type-desc"
  | "unique-asc"
  | "unique-desc";

/* Field rows arrive as 0-100, not as a ratio. */
function percentText(value: number | null | undefined): string {
  return value == null ? "—" : formatPercent(value / 100);
}

function missingRate(profile: DatasetProfileSummary): number {
  const fields = profile.fields ?? [];
  if (fields.length === 0) return 0;
  const total = fields.reduce(
    (sum, field) => sum + (field.missing_percent ?? 0),
    0,
  );
  return total / fields.length;
}

function emptyColumnCount(profile: DatasetProfileSummary): number {
  return (profile.fields ?? []).filter(
    (field) => (field.missing_percent ?? 0) >= 100,
  ).length;
}

const FIELD_ROW_HEIGHT = 40;

function FieldTable({
  fields,
  distributions,
  distributionsPending,
  sort,
  onSortChange,
}: {
  fields: FieldRow[];
  distributions: Map<string, ColumnDistribution>;
  distributionsPending: boolean;
  sort: SortKey;
  onSortChange: (key: "name" | "missing" | "type" | "unique") => void;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const virtualizer = useVirtualizer({
    count: fields.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => FIELD_ROW_HEIGHT,
    overscan: 12,
    /* jsdom reports zero-size rects; a non-zero initial rect keeps rows
     * renderable in tests until a real ResizeObserver measurement lands. */
    initialRect: { width: 800, height: 640 },
  });
  const virtualRows = virtualizer.getVirtualItems();
  const paddingTop = virtualRows[0]?.start ?? 0;
  const paddingBottom =
    virtualizer.getTotalSize() - (virtualRows[virtualRows.length - 1]?.end ?? 0);

  function fieldSummary(
    field: FieldRow,
    distribution: ColumnDistribution | undefined,
  ): string {
    const kind = classifyColumn(field.dtype, field.semantic_type);
    if (kind === "numeric" && distribution?.min != null && distribution.max != null) {
      return `${formatCompact(distribution.min)}–${formatCompact(distribution.max)}`;
    }
    if (kind === "text" && distribution) {
      const parts = [];
      if (distribution.unique_count != null) {
        parts.push(`${formatCompact(distribution.unique_count)} values`);
      }
      if (distribution.len_min != null && distribution.len_max != null) {
        parts.push(
          distribution.len_min === distribution.len_max
            ? `${distribution.len_min} chars`
            : `${distribution.len_min}–${distribution.len_max} chars`,
        );
      }
      if (parts.length > 0) return parts.join(" · ");
    }
    if (kind === "id") {
      return field.unique_percent == null
        ? "Identifier"
        : `${percentText(field.unique_percent)} distinct`;
    }
    if (kind === "boolean") {
      return distribution?.top
        ?.slice(0, 2)
        .map((item) => `${String(item.value)} ${formatCompact(item.count)}`)
        .join(" · ") || "True / false";
    }
    if (kind === "temporal") return field.sample_values || "Date / time values";
    return field.sample_values || "No sample";
  }

  return (
    <div
      ref={scrollRef}
      className="max-h-[32rem] min-h-0 overflow-auto rounded-base border border-table-border"
    >
      <table className="w-full min-w-[20rem] border-collapse text-xs sm:text-sm">
        <thead className="sticky top-0 z-10 bg-table-header-bg text-left">
          <tr>
            <th scope="col" aria-sort={sort.startsWith("name") ? (sort === "name-asc" ? "ascending" : "descending") : "none"} className="px-2 py-2 font-medium sm:px-3">
              <button type="button" onClick={() => onSortChange("name")} className="inline-flex items-center gap-1 hover:text-primary">
                Column <span aria-hidden="true" className="text-status-neutral">{sort === "name-asc" ? "↑" : sort === "name-desc" ? "↓" : "↕"}</span>
              </button>
            </th>
            <th
              scope="col"
              aria-sort={sort.startsWith("type") ? (sort === "type-asc" ? "ascending" : "descending") : "none"}
              className="hidden px-3 py-2 font-medium @4xl/data-page:table-cell"
            >
              <button type="button" onClick={() => onSortChange("type")} className="inline-flex items-center gap-1 hover:text-primary">
                Type <span aria-hidden="true" className="text-status-neutral">{sort === "type-asc" ? "↑" : sort === "type-desc" ? "↓" : "↕"}</span>
              </button>
            </th>
            <th scope="col" className="px-2 py-2 font-medium sm:px-3">
              Shape
            </th>
            <th
              scope="col"
              aria-sort={sort.startsWith("missing") ? (sort === "missing-asc" ? "ascending" : "descending") : "none"}
              className="px-2 py-2 text-right font-medium sm:px-3"
            >
              <button type="button" onClick={() => onSortChange("missing")} className="ml-auto inline-flex items-center gap-1 hover:text-primary">
                Missing <span aria-hidden="true" className="text-status-neutral">{sort === "missing-asc" ? "↑" : sort === "missing-desc" ? "↓" : "↕"}</span>
              </button>
            </th>
            <th
              scope="col"
              aria-sort={sort.startsWith("unique") ? (sort === "unique-asc" ? "ascending" : "descending") : "none"}
              className="hidden px-3 py-2 text-right font-medium @4xl/data-page:table-cell"
            >
              <button type="button" onClick={() => onSortChange("unique")} className="ml-auto inline-flex items-center gap-1 hover:text-primary">
                Unique <span aria-hidden="true" className="text-status-neutral">{sort === "unique-asc" ? "↑" : sort === "unique-desc" ? "↓" : "↕"}</span>
              </button>
            </th>
            <th
              scope="col"
              className="hidden px-3 py-2 font-medium @4xl/data-page:table-cell"
            >
              Samples
            </th>
          </tr>
        </thead>
        <tbody>
          {paddingTop > 0 && (
            <tr aria-hidden>
              <td colSpan={6} style={{ height: paddingTop }} />
            </tr>
          )}
          {virtualRows.map((virtualRow) => {
            const field = fields[virtualRow.index];
            if (!field) return null;
            const kind = classifyColumn(field.dtype, field.semantic_type);
            const distribution = distributions.get(field.column);
            const summary = fieldSummary(field, distribution);
            return (
              <tr key={field.column} className="border-t border-hairline">
                <td className="px-2 py-2 sm:px-3">
                  <span className="flex min-w-24 items-center gap-1.5">
                    <TypeIcon kind={kind} />
                    <span className="min-w-0">
                      <Marquee className="block font-mono text-xs" title={field.column}>
                        {field.column}
                      </Marquee>
                      <Marquee className="block font-mono text-[10px] text-status-neutral @4xl/data-page:hidden">
                        {field.dtype}
                      </Marquee>
                    </span>
                  </span>
                </td>
                <td className="hidden px-3 py-2 whitespace-nowrap @4xl/data-page:table-cell">
                  <span className="font-mono text-xs">{field.dtype}</span>
                  <span className="ml-1.5 text-xs text-status-neutral">
                    {field.semantic_type}
                  </span>
                </td>
                <td className="px-2 py-2 sm:px-3">
                  <span className="flex min-w-20 items-center gap-2">
                    <span className="hidden sm:inline-flex">
                      {distributionsPending ? (
                        <span className="skeleton inline-block h-2 w-16 rounded-sm align-middle" />
                      ) : (
                        <DistributionSpark dist={distribution} />
                      )}
                    </span>
                    <Marquee className="max-w-32 text-xs text-status-neutral" title={summary}>
                      {summary}
                    </Marquee>
                  </span>
                </td>
                <td className="px-2 py-2 text-right whitespace-nowrap sm:px-3">
                  <span className="inline-flex w-28 items-center justify-end gap-1.5">
                    <span className="hidden sm:inline-flex">
                      <MissingBar percent={field.missing_percent} width="w-16" />
                    </span>
                    <span className="tabular w-[4.5ch] text-right">
                      {percentText(field.missing_percent)}
                    </span>
                  </span>
                </td>
                <td className="tabular hidden px-3 py-2 text-right whitespace-nowrap @4xl/data-page:table-cell">
                  {percentText(field.unique_percent)}
                </td>
                <td className="hidden max-w-40 px-3 py-2 text-status-neutral @4xl/data-page:table-cell">
                  <Marquee title={field.sample_values}>
                    {field.sample_values}
                  </Marquee>
                </td>
              </tr>
            );
          })}
          {paddingBottom > 0 && (
            <tr aria-hidden>
              <td colSpan={6} style={{ height: paddingBottom }} />
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

function FieldPanel({
  sessionId,
  profile,
  panelId,
}: {
  sessionId: string;
  profile: DatasetProfileSummary;
  panelId: string;
}) {
  const [query, setQuery] = useRouteSearchParam("q");
  const [searchOpen, setSearchOpen] = useState(Boolean(query));
  const searchInputRef = useRef<HTMLInputElement>(null);
  const [kindParam, setKindParam] = useRouteSearchParam("kind", "all");
  const kindFilter: ColumnKind | "all" =
    kindParam === "all" ||
    ["numeric", "temporal", "boolean", "text", "id", "other"].includes(
      kindParam,
    )
      ? (kindParam as ColumnKind | "all")
      : "all";
  const setKindFilter = (kind: ColumnKind | "all") => setKindParam(kind);
  const [sortParam, setSortParam] = useRouteSearchParam("sort", "table");
  const sort: SortKey = ["table", "missing-asc", "missing-desc", "name-asc", "name-desc", "type-asc", "type-desc", "unique-asc", "unique-desc"].includes(sortParam)
    ? (sortParam as SortKey)
    : "table";
  const toggleSort = (key: "name" | "missing" | "type" | "unique") => {
    const ascending = `${key}-asc` as SortKey;
    const descending = `${key}-desc` as SortKey;
    setSortParam(sort === ascending ? descending : sort === descending ? "table" : ascending);
  };

  const distributions = useDatasetDistributions(sessionId, profile.dataset_id);
  const distByColumn = useMemo(() => {
    const map = new Map<string, ColumnDistribution>();
    for (const column of distributions.data?.columns ?? []) {
      map.set(column.name, column);
    }
    return map;
  }, [distributions.data]);

  const fields = profile.fields ?? [];
  const kindCounts = useMemo(() => {
    const counts = new Map<ColumnKind, number>();
    for (const field of fields) {
      const kind = classifyColumn(field.dtype, field.semantic_type);
      counts.set(kind, (counts.get(kind) ?? 0) + 1);
    }
    return counts;
  }, [fields]);

  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const filtered = fields.filter((field) => {
      if (
        kindFilter !== "all" &&
        classifyColumn(field.dtype, field.semantic_type) !== kindFilter
      ) {
        return false;
      }
      if (!needle) return true;
      return (
        field.column.toLowerCase().includes(needle) ||
        field.dtype.toLowerCase().includes(needle) ||
        field.semantic_type.toLowerCase().includes(needle)
      );
    });
    if (sort === "missing-asc" || sort === "missing-desc") {
      const direction = sort === "missing-asc" ? 1 : -1;
      return [...filtered].sort(
        (a, b) => direction * ((a.missing_percent ?? 0) - (b.missing_percent ?? 0)),
      );
    }
    if (sort === "name-asc" || sort === "name-desc") {
      const direction = sort === "name-asc" ? 1 : -1;
      return [...filtered].sort((a, b) => direction * a.column.localeCompare(b.column));
    }
    if (sort === "type-asc" || sort === "type-desc") {
      const direction = sort === "type-asc" ? 1 : -1;
      return [...filtered].sort(
        (a, b) =>
          direction * (
            classifyColumn(a.dtype, a.semantic_type).localeCompare(
              classifyColumn(b.dtype, b.semantic_type),
            ) || a.dtype.localeCompare(b.dtype) || a.column.localeCompare(b.column)
          ),
      );
    }
    if (sort === "unique-asc" || sort === "unique-desc") {
      const direction = sort === "unique-asc" ? 1 : -1;
      return [...filtered].sort(
        (a, b) => direction * ((a.unique_percent ?? -1) - (b.unique_percent ?? -1)),
      );
    }
    return filtered;
  }, [fields, kindFilter, query, sort]);

  return (
    <div id={panelId} className="flex min-w-0 flex-col gap-3">
      <div className="flex flex-wrap gap-1.5">
        <KindChip
          label="All"
          count={fields.length}
          active={kindFilter === "all"}
          onClick={() => setKindFilter("all")}
        />
        {[...kindCounts.entries()].map(([kind, count]) => (
          <KindChip
            key={kind}
            kind={kind}
            label={KIND_LABEL[kind]}
            count={count}
            active={kindFilter === kind}
            onClick={() => setKindFilter(kindFilter === kind ? "all" : kind)}
          />
        ))}
        <div className="ml-auto flex items-center gap-2">
          {searchOpen && (
            <input
              ref={searchInputRef}
              aria-label="Find column"
              type="search"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Find column…"
              className="min-w-40 rounded-base border border-border bg-bg px-2 py-1 text-sm"
            />
          )}
          <button
            type="button"
            aria-label={searchOpen ? "Close column search" : "Find column"}
            title={searchOpen ? "Close column search" : "Find column"}
            onClick={() => {
              if (searchOpen) {
                setSearchOpen(false);
                setQuery("");
              } else {
                setSearchOpen(true);
                requestAnimationFrame(() => searchInputRef.current?.focus());
              }
            }}
            className="inline-flex size-7 items-center justify-center rounded-base border border-border text-status-neutral hover:bg-surface hover:text-text"
          >
            {searchOpen ? "×" : "⌕"}
          </button>
          <Hint label="Distribution">
            A mini histogram for numeric columns and top-value bars for categorical ones. The bar beside Missing is the column&apos;s null rate.
          </Hint>
          <span className="tabular text-xs text-status-neutral">
            {visible.length} / {fields.length}
          </span>
        </div>
      </div>

      {distributions.isError && (
        <p className="text-xs text-status-neutral">
          Distributions are unavailable for this table; the rest of the profile
          is unaffected.
        </p>
      )}

      {visible.length === 0 ? (
        <EmptyState
          title="No columns match"
          description="Clear the search or the type filter to see every column again."
        />
      ) : (
        <FieldTable
          fields={visible}
          distributions={distByColumn}
          distributionsPending={distributions.isPending}
          sort={sort}
          onSortChange={toggleSort}
        />
      )}

      {distributions.data?.sampled && (
        <p className="text-xs text-status-neutral">
          Distributions are computed from a random sample of{" "}
          {distributions.data.sample_rows.toLocaleString("en-US")} of{" "}
          {distributions.data.row_count.toLocaleString("en-US")} rows.
        </p>
      )}
    </div>
  );
}

function KindChip({
  kind,
  label,
  count,
  active,
  onClick,
}: {
  kind?: ColumnKind;
  label: string;
  count: number;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      aria-pressed={active}
      onClick={onClick}
      className={`inline-flex items-center gap-1.5 rounded-sm border px-1.5 py-0.5 text-xs ${
        active
          ? "border-primary/45 bg-primary/10 text-primary"
          : "border-border text-status-neutral hover:border-primary/40"
      }`}
    >
      {kind && <TypeIcon kind={kind} />}
      <span>{label}</span>
      <span className="tabular">{count}</span>
    </button>
  );
}

/* The listing is metadata-only; the vega-lite spec is fetched per chart. The
 * card title stays in the card header, so the in-spec title is dropped to
 * avoid rendering twice. */
function chartSpecWithoutTitle(
  spec: Record<string, unknown>,
): Record<string, unknown> {
  const { title: _title, ...rest } = spec;
  return rest;
}

type ChartDatasetGroup = {
  datasetId: string;
  datasetName: string;
  analyticalCharts: ChartSummary[];
  diagnosticCharts: ChartSummary[];
};

/* Chart gallery: group by
 * dataset_id, sort groups by dataset display name (plain string comparison,
 * matching Python's default `sorted()`), keep charts in listing order within
 * a group. */
function groupChartsByDataset(charts: ChartSummary[]): ChartDatasetGroup[] {
  const groups = new Map<string, ChartDatasetGroup>();
  for (const chart of charts) {
    const group = groups.get(chart.dataset_id);
    if (group) {
      if (isProfileDiagnostic(chart)) group.diagnosticCharts.push(chart);
      else group.analyticalCharts.push(chart);
    } else {
      groups.set(chart.dataset_id, {
        datasetId: chart.dataset_id,
        datasetName: chart.dataset_name,
        analyticalCharts: isProfileDiagnostic(chart) ? [] : [chart],
        diagnosticCharts: isProfileDiagnostic(chart) ? [chart] : [],
      });
    }
  }
  return [...groups.values()].sort((a, b) =>
    a.datasetName < b.datasetName ? -1 : a.datasetName > b.datasetName ? 1 : 0,
  );
}

function isProfileDiagnostic(chart: ChartSummary): boolean {
  const title = chart.title.trim().toLowerCase();
  return (
    title.startsWith("distribution of ") ||
    title.startsWith("top values in ") ||
    title.startsWith("missing values by column")
  );
}

function ChartCard({
  chart,
  sessionId,
  onZoom,
}: {
  chart: ChartSummary;
  sessionId: string;
  onZoom: () => void;
}) {
  /* Spec fetches are deferred until the card scrolls into view, so a long
   * chart list doesn't fan out one detail request per card on page load. */
  const { ref, inView } = useInView<HTMLElement>();
  const detail = useChart(sessionId, chart.artifact_id, inView);
  const spec = useMemo(
    () => (detail.data ? chartSpecWithoutTitle(detail.data.spec ?? {}) : null),
    [detail.data],
  );

  return (
    <article
      ref={ref}
      className="flex flex-col gap-1.5 rounded-base border border-border bg-bg p-4"
    >
      <header className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <h3 className="min-w-0 text-sm font-semibold"><Marquee>{chart.title}</Marquee></h3>
        </div>
        <button
          type="button"
          onClick={onZoom}
          className="shrink-0 rounded-base border border-border px-2 py-1 text-xs hover:bg-surface"
        >
          Zoom
        </button>
      </header>
      {detail.data?.plain_language ?? chart.description ? (
        <p className="text-xs text-status-neutral">
          {detail.data?.plain_language ?? chart.description}
        </p>
      ) : null}
      {detail.isPending && <LoadingSkeleton lines={3} label="Loading chart" />}
      {detail.isError && (
        <ErrorState error={detail.error} onRetry={() => detail.refetch()} />
      )}
      {spec && <VegaChart spec={spec} label={chart.title} />}
    </article>
  );
}

function ChartGroupSection({
  group,
  sessionId,
  onZoom,
}: {
  group: ChartDatasetGroup;
  sessionId: string;
  onZoom: (chart: ChartSummary) => void;
}) {
  return (
    <section className="flex min-w-0 flex-col gap-3">
      <header className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <h3 className="text-sm font-semibold">Analytical charts</h3>
        <span className="tabular text-xs text-status-neutral">
          {group.analyticalCharts.length} analytical chart{group.analyticalCharts.length === 1 ? "" : "s"}
        </span>
      </header>
      {group.analyticalCharts.length > 1 ? (
        <div className="grid gap-3 @4xl/data-page:grid-cols-2">
          {group.analyticalCharts.map((chart) => (
            <ChartCard
              key={chart.artifact_id}
              chart={chart}
              sessionId={sessionId}
              onZoom={() => onZoom(chart)}
            />
          ))}
        </div>
      ) : (
        <ChartCard
          chart={group.analyticalCharts[0]!}
          sessionId={sessionId}
          onZoom={() => onZoom(group.analyticalCharts[0]!)}
        />
      )}
    </section>
  );
}

function diagnosticChartSummary(charts: ChartSummary[]): string {
  const missingness = charts.filter((chart) =>
    chart.title.trim().toLowerCase().startsWith("missing values by column"),
  ).length;
  const distributions = charts.length - missingness;
  const parts = [
    distributions > 0
      ? `${distributions} distribution or top-value chart${distributions === 1 ? "" : "s"}`
      : "",
    missingness > 0
      ? `${missingness} missingness chart${missingness === 1 ? "" : "s"}`
      : "",
  ].filter(Boolean);
  return parts.join(" and ");
}

function missingnessSummary(
  issues: Array<{ code: string; column?: string | null; message: string }>,
): string | null {
  const relevant = issues.filter((issue) => /missing|empty|null/i.test(issue.code));
  if (relevant.length === 0) return null;
  const fields = [...new Set(relevant.map((issue) => issue.column).filter(Boolean))];
  if (fields.length === 0) return `${relevant.length} missingness flag${relevant.length === 1 ? "" : "s"}`;
  const shown = fields.slice(0, 3).join(", ");
  const remainder = fields.length - 3;
  return `Missingness flags: ${shown}${remainder > 0 ? ` +${remainder} more` : ""}`;
}

function LinkedChartSplit({
  left,
  right,
  charts,
  sessionId,
  onPairChange,
  onClose,
  onZoom,
}: {
  left: ChartSummary;
  right: ChartSummary;
  charts: ChartSummary[];
  sessionId: string;
  onPairChange: (leftId: string, rightId: string) => void;
  onClose: () => void;
  onZoom: (chart: ChartSummary) => void;
}) {
  const leftDetail = useChart(sessionId, left.artifact_id, true);
  const rightDetail = useChart(sessionId, right.artifact_id, true);
  const field = sharedChartField(left, right);
  const spec = useMemo(() => {
    if (!field || !leftDetail.data || !rightDetail.data) return null;
    return buildLinkedChartSpec(
      chartSpecWithoutTitle(leftDetail.data.spec ?? {}),
      chartSpecWithoutTitle(rightDetail.data.spec ?? {}),
      field,
    );
  }, [field, leftDetail.data, rightDetail.data]);

  const compatibleWith = (chart: ChartSummary) =>
    charts.filter(
      (candidate) =>
        candidate.artifact_id !== chart.artifact_id &&
        sharedChartField(chart, candidate) !== null,
    );

  const changeLeft = (leftId: string) => {
    const nextLeft = charts.find((chart) => chart.artifact_id === leftId);
    if (!nextLeft) return;
    const nextRight =
      compatibleWith(nextLeft).find(
        (chart) => chart.artifact_id === right.artifact_id,
      ) ?? compatibleWith(nextLeft)[0];
    if (nextRight) onPairChange(nextLeft.artifact_id, nextRight.artifact_id);
  };

  const changeRight = (rightId: string) => {
    const nextRight = charts.find((chart) => chart.artifact_id === rightId);
    if (nextRight && sharedChartField(left, nextRight)) {
      onPairChange(left.artifact_id, nextRight.artifact_id);
    }
  };

  return (
    <Card
      tone="quiet"
      className="flex flex-col gap-3 p-3"
      role="region"
      aria-label="Linked chart split view"
    >
      <header className="flex flex-wrap items-end gap-3">
        <label className="flex min-w-48 flex-1 flex-col gap-1 text-xs">
          <span className="text-status-neutral">Left chart</span>
          <select
            value={left.artifact_id}
            onChange={(event) => changeLeft(event.target.value)}
            className="rounded-base border border-border bg-bg px-2 py-1.5 text-sm"
          >
            {charts
              .filter((chart) => compatibleWith(chart).length > 0)
              .map((chart) => (
                <option key={chart.artifact_id} value={chart.artifact_id}>
                  {chart.title}
                </option>
              ))}
          </select>
        </label>
        <label className="flex min-w-48 flex-1 flex-col gap-1 text-xs">
          <span className="text-status-neutral">Right chart</span>
          <select
            value={right.artifact_id}
            onChange={(event) => changeRight(event.target.value)}
            className="rounded-base border border-border bg-bg px-2 py-1.5 text-sm"
          >
            {compatibleWith(left).map((chart) => (
              <option key={chart.artifact_id} value={chart.artifact_id}>
                {chart.title}
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          onClick={() => onZoom(left)}
          className="rounded-base border border-border px-2 py-1.5 text-xs hover:bg-surface"
        >
          Zoom left
        </button>
        <button
          type="button"
          onClick={() => onZoom(right)}
          className="rounded-base border border-border px-2 py-1.5 text-xs hover:bg-surface"
        >
          Zoom right
        </button>
        <button
          type="button"
          onClick={onClose}
          className="rounded-base border border-border px-2 py-1.5 text-xs hover:bg-surface"
        >
          Close split view
        </button>
      </header>
      <p className="text-xs text-status-neutral">
        Click a mark in the left chart to filter the right chart by{" "}
        <span className="font-mono">{field}</span>. Double-click the left chart
        to clear the selection.
      </p>
      {(leftDetail.isPending || rightDetail.isPending) && (
        <LoadingSkeleton lines={5} label="Loading linked charts" />
      )}
      {leftDetail.isError && (
        <ErrorState
          error={leftDetail.error}
          onRetry={() => leftDetail.refetch()}
        />
      )}
      {rightDetail.isError && (
        <ErrorState
          error={rightDetail.error}
          onRetry={() => rightDetail.refetch()}
        />
      )}
      {spec && (
        <VegaChart
          spec={spec}
          label={`${left.title} linked to ${right.title}`}
        />
      )}
    </Card>
  );
}

function ChartZoomModal({
  chart,
  sessionId,
  onClose,
}: {
  chart: ChartSummary;
  sessionId: string;
  onClose: () => void;
}) {
  /* Zoom always fetches (enabled), independent of card visibility. */
  const detail = useChart(sessionId, chart.artifact_id, true);
  const spec = useMemo(
    () => (detail.data ? chartSpecWithoutTitle(detail.data.spec ?? {}) : null),
    [detail.data],
  );

  const { dialogRef, onKeyDown } = useDialogFocus(onClose);

  return (
    <div
      className="animate-fade fixed inset-0 z-50 flex items-center justify-center bg-scrim-strong p-6"
      onClick={onClose}
    >
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-label={chart.title}
        onKeyDown={onKeyDown}
        className="animate-enter flex max-h-full w-full max-w-4xl flex-col gap-2 overflow-auto rounded-base border border-border bg-bg p-5 shadow-overlay"
        onClick={(event) => event.stopPropagation()}
      >
        <header className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <h2 className="text-base font-semibold">{chart.title}</h2>
            <p className="text-xs text-status-neutral">{chart.dataset_name}</p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="shrink-0 rounded-base border border-border px-2 py-1 text-sm hover:bg-surface"
          >
            Close
          </button>
        </header>
        {detail.data?.plain_language && (
          <p className="text-sm text-status-neutral">
            {detail.data.plain_language}
          </p>
        )}
        {detail.isPending && <LoadingSkeleton lines={5} label="Loading chart" />}
        {spec && <VegaChart spec={spec} label={chart.title} />}
      </div>
    </div>
  );
}

function DatasetInsightsWorkspace({
  sessionId,
  projectId,
}: {
  sessionId: string;
  projectId: string;
}) {
  const profiles = useProfiles(sessionId);
  const charts = useCharts(sessionId);
  const quality = useQuality(sessionId);
  const [zoomed, setZoomed] = useState<ChartSummary | null>(null);
  const [datasetParam, setDatasetParam] = useRouteSearchParam("dataset");
  const [viewParam, setViewParam] = useRouteSearchParam("view");
  const [splitParam, setSplitParam] = useRouteSearchParam("split");
  const profileList = profiles.data?.datasets ?? [];
  const chartItems = charts.data?.pages.flatMap((page) => page.items) ?? [];
  const chartGroups = groupChartsByDataset(chartItems);
  const defaultDatasetId =
    profileList.find(
      (profile) =>
        chartGroups.find((group) => group.datasetId === profile.dataset_id)
          ?.analyticalCharts.length,
    )?.dataset_id ?? profileList[0]?.dataset_id ?? "";
  const selectedDatasetId = profileList.some(
    (profile) => profile.dataset_id === datasetParam,
  )
    ? datasetParam
    : defaultDatasetId;
  const selectedProfile =
    profileList.find((profile) => profile.dataset_id === selectedDatasetId) ?? null;
  const selectedGroup =
    chartGroups.find((group) => group.datasetId === selectedDatasetId) ?? {
      datasetId: selectedDatasetId,
      datasetName: selectedProfile?.name ?? "Selected dataset",
      analyticalCharts: [],
      diagnosticCharts: [],
    };
  const activeView = viewParam === "charts" ? "charts" : "profiles";
  const selectedMissingness = missingnessSummary(
    (quality.data?.issues ?? []).filter(
      (issue) => issue.dataset_id === selectedDatasetId,
    ),
  );
  const galleryItems = chartItems.filter((chart) => !isProfileDiagnostic(chart));
  const [leftId = "", rightId = ""] = splitParam.split(",", 2);
  const splitLeft = chartItems.find((chart) => chart.artifact_id === leftId);
  const splitRight = chartItems.find((chart) => chart.artifact_id === rightId);

  /* Restore a shared split deep link even when either chart lives beyond the
   * first cursor page. */
  useEffect(() => {
    if (
      leftId &&
      rightId &&
      (!splitLeft || !splitRight) &&
      charts.hasNextPage &&
      !charts.isFetchingNextPage
    ) {
      void charts.fetchNextPage();
    }
  }, [
    charts.fetchNextPage,
    charts.hasNextPage,
    charts.isFetchingNextPage,
    leftId,
    rightId,
    splitLeft,
    splitRight,
  ]);

  const splitPair =
    splitLeft &&
    splitRight &&
    splitLeft.artifact_id !== splitRight.artifact_id &&
    sharedChartField(splitLeft, splitRight)
      ? ([splitLeft, splitRight] as const)
      : null;
  const initialSplitPair = (() => {
    for (const left of selectedGroup.analyticalCharts) {
      const right = galleryItems.find(
        (candidate) =>
          candidate.artifact_id !== left.artifact_id &&
          sharedChartField(left, candidate) !== null,
      );
      if (right) return [left, right] as const;
    }
    return null;
  })();

  if (profiles.isPending) return <LoadingSkeleton lines={6} label="Loading datasets" />;
  if (profiles.isError) return <ErrorState error={profiles.error} onRetry={() => profiles.refetch()} />;
  if (profileList.length === 0) {
    return <EmptyState title="No dataset profiles" description="Run an analysis to inspect datasets in this session." />;
  }
  const resolvedProfile = selectedProfile ?? profileList[0]!;

  return (
    <section className="flex min-w-0 flex-col gap-4">
      <section aria-label="Dataset overview" className="flex min-w-0 flex-col gap-3">
        <DatasetScopeBar
          value={selectedDatasetId}
          onChange={setDatasetParam}
          options={profileList.map((profile) => ({
            value: profile.dataset_id,
            label: profile.name,
          }))}
        >
          <SegmentedControl
            label="Dataset view"
            value={activeView}
            onChange={(value) => setViewParam(value === "profiles" ? "" : value)}
            options={[
              { value: "profiles", label: "Profile" },
              { value: "charts", label: "Charts" },
            ]}
          />
        </DatasetScopeBar>
        <MetricStrip>
          <MetricTile label="Rows" value={resolvedProfile.rows.toLocaleString()} />
          <MetricTile label="Columns" value={resolvedProfile.columns} />
          <MetricTile label="Null rate" value={formatPercent(missingRate(resolvedProfile) / 100, 0)} />
          <MetricTile label="Empty columns" value={emptyColumnCount(resolvedProfile)} />
          <MetricTile label="Profiled fields" value={resolvedProfile.fields?.length ?? resolvedProfile.columns} />
          <MetricTile label="Analytical charts" value={selectedGroup.analyticalCharts.length} />
        </MetricStrip>
      </section>

      {activeView === "profiles" ? (
        <section role="tabpanel" aria-label="Profile" className="min-w-0">
          <FieldPanel
            key={resolvedProfile.dataset_id}
            sessionId={sessionId}
            profile={resolvedProfile}
            panelId="dataset-fields"
          />
        </section>
      ) : (
        <section role="tabpanel" aria-label="Charts" className="flex min-w-0 flex-col gap-4">
          <CustomChartBuilder
            sessionId={sessionId}
            projectId={projectId}
            datasetId={selectedDatasetId}
          />
          {charts.data && initialSplitPair && !splitPair && (
            <button
              type="button"
              onClick={() =>
                setSplitParam(
                  `${initialSplitPair[0].artifact_id},${initialSplitPair[1].artifact_id}`,
                )
              }
              className="self-start rounded-base border border-border px-3 py-1.5 text-sm font-medium hover:bg-surface"
            >
              Open linked split view
            </button>
          )}
          {splitPair && (
            <LinkedChartSplit
              left={splitPair[0]}
              right={splitPair[1]}
              charts={galleryItems}
              sessionId={sessionId}
              onPairChange={(left, right) => setSplitParam(`${left},${right}`)}
              onClose={() => setSplitParam("")}
              onZoom={setZoomed}
            />
          )}
          {selectedGroup.diagnosticCharts.length > 0 && (
            <Card tone="quiet" className="flex flex-wrap items-center gap-x-3 gap-y-2 px-3 py-2">
              <p className="min-w-0 flex-1 text-sm text-status-neutral">
                {diagnosticChartSummary(selectedGroup.diagnosticCharts)} are available in Profile and Quality instead of repeated here.{selectedMissingness ? ` ${selectedMissingness}.` : ""}
              </p>
              <Link to={`${sessionSectionPath(projectId, sessionId, "quality")}?dataset=${encodeURIComponent(selectedDatasetId)}`} className="text-sm font-medium text-primary hover:underline">Review missingness</Link>
            </Card>
          )}
          {charts.isPending && <LoadingSkeleton lines={4} label="Loading charts" />}
          {charts.isError && (
            <ErrorState error={charts.error} onRetry={() => charts.refetch()} />
          )}
          {charts.data && selectedGroup.analyticalCharts.length > 0 ? (
            <ChartGroupSection group={selectedGroup} sessionId={sessionId} onZoom={setZoomed} />
          ) : charts.data ? (
            <EmptyState
              title={charts.hasNextPage ? "No analytical charts loaded for this dataset yet" : "No analytical charts for this dataset"}
              description={charts.hasNextPage ? "Load the next chart batch to continue checking this dataset." : "Its single-field evidence is available in Profile and Quality."}
            />
          ) : null}
          {charts.hasNextPage && (
            <button
              type="button"
              onClick={() => charts.fetchNextPage()}
              disabled={charts.isFetchingNextPage}
              className="self-start rounded-base border border-border px-3 py-1.5 text-sm font-medium hover:bg-surface disabled:opacity-50"
            >
              {charts.isFetchingNextPage ? "Loading charts…" : "Load more charts"}
            </button>
          )}
        </section>
      )}
      {zoomed && <ChartZoomModal chart={zoomed} sessionId={sessionId} onClose={() => setZoomed(null)} />}
    </section>
  );
}

export function Component() {
  const { projectId = "", sessionId = "" } = useParams();
  return (
    <DataWorkspacePage
      title="Profiles & charts"
      description="Choose a dataset once, then move between its field evidence and analytical charts."
    >
      <DatasetInsightsWorkspace sessionId={sessionId} projectId={projectId} />
    </DataWorkspacePage>
  );
}

export function ChartsComponent() {
  const { projectId = "", sessionId = "" } = useParams();
  const location = useLocation();
  const search = new URLSearchParams(location.search);
  search.set("view", "charts");
  return (
    <Navigate
      replace
      to={`${sessionSectionPath(projectId, sessionId, "profiles")}?${search.toString()}`}
    />
  );
}
