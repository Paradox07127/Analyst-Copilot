import { useEffect, useMemo, useRef, useState } from "react";
import { useParams } from "react-router";
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
} from "../../api/hooks";
import {
  EmptyState,
  ErrorState,
  LoadingSkeleton,
} from "../../components/async-states";
import {
  Badge,
  Card,
  Hint,
  Marquee,
  MetricStrip,
  MetricTile,
  SectionHeader,
  formatCompact,
  formatPercent,
} from "../../components/ui";
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
type SortKey = "table" | "missing" | "name";

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

function ProfilesSummary({ profiles }: { profiles: DatasetProfileSummary[] }) {
  const columns = profiles.reduce(
    (sum, profile) => sum + (profile.fields ?? []).length,
    0,
  );
  const withNulls = profiles.reduce(
    (sum, profile) =>
      sum +
      (profile.fields ?? []).filter((field) => (field.missing_percent ?? 0) > 0)
        .length,
    0,
  );
  const empty = profiles.reduce(
    (sum, profile) => sum + emptyColumnCount(profile),
    0,
  );
  const rows = profiles.reduce((sum, profile) => sum + profile.rows, 0);

  return (
    <MetricStrip>
      <MetricTile label="Tables profiled" value={profiles.length} />
      <MetricTile
        label="Rows"
        value={formatCompact(rows)}
        hint={rows.toLocaleString()}
      />
      <MetricTile label="Columns" value={columns} />
      <MetricTile
        label="With nulls"
        value={withNulls}
        tone="warn"
        emphasis={withNulls > 0}
      />
      <MetricTile
        label="Entirely empty"
        value={empty}
        tone="critical"
        emphasis={empty > 0}
      />
    </MetricStrip>
  );
}

/* Nine datasets used to render nine full tables stacked vertically. The rail
 * is the summary level: pick one table, read one table. */
function DatasetRail({
  profiles,
  selectedId,
  onSelect,
  panelId,
}: {
  profiles: DatasetProfileSummary[];
  selectedId: string;
  onSelect: (datasetId: string) => void;
  panelId: string;
}) {
  return (
    <ul className="flex max-h-[32rem] flex-col gap-1 overflow-y-auto lg:pr-1">
      {profiles.map((profile) => {
        const selected = profile.dataset_id === selectedId;
        const rate = missingRate(profile);
        const empty = emptyColumnCount(profile);
        return (
          <li key={profile.dataset_id}>
            <button
              type="button"
              aria-pressed={selected}
              aria-controls={panelId}
              onClick={() => onSelect(profile.dataset_id)}
              className={`flex w-full flex-col gap-1 rounded-base border-l-2 px-2 py-1.5 text-left hover:bg-bg ${
                selected
                  ? "border-primary bg-bg"
                  : "border-transparent"
              }`}
            >
              <span className="flex items-center gap-1.5">
                <Marquee className="min-w-0 flex-1 text-sm font-medium" title={profile.name}>
                  {profile.name}
                </Marquee>
                {empty > 0 && (
                  <Badge tone="critical" title={`${empty} entirely empty columns`}>
                    {empty} empty
                  </Badge>
                )}
              </span>
              <span className="tabular text-xs text-status-neutral">
                {formatCompact(profile.rows)} rows · {profile.columns} cols
              </span>
              <span className="flex items-center gap-1.5">
                <MissingBar percent={rate} width="w-16" />
                <span className="tabular text-xs text-status-neutral">
                  {formatPercent(rate / 100, 0)} null
                </span>
              </span>
            </button>
          </li>
        );
      })}
    </ul>
  );
}

const FIELD_ROW_HEIGHT = 40;

function FieldTable({
  fields,
  distributions,
  distributionsPending,
}: {
  fields: FieldRow[];
  distributions: Map<string, ColumnDistribution>;
  distributionsPending: boolean;
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
            <th scope="col" className="px-2 py-2 font-medium sm:px-3">
              Column
            </th>
            <th
              scope="col"
              className="hidden px-3 py-2 font-medium md:table-cell"
            >
              Type
            </th>
            <th scope="col" className="px-2 py-2 font-medium sm:px-3">
              Shape
            </th>
            <th
              scope="col"
              className="px-2 py-2 text-right font-medium sm:px-3"
            >
              Missing
            </th>
            <th
              scope="col"
              className="hidden px-3 py-2 text-right font-medium lg:table-cell"
            >
              Unique
            </th>
            <th
              scope="col"
              className="hidden px-3 py-2 font-medium xl:table-cell"
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
                      <Marquee className="block font-mono text-[10px] text-status-neutral md:hidden">
                        {field.dtype}
                      </Marquee>
                    </span>
                  </span>
                </td>
                <td className="hidden px-3 py-2 whitespace-nowrap md:table-cell">
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
                  <span className="inline-flex items-center gap-1.5">
                    <span className="hidden sm:inline-flex">
                      <MissingBar percent={field.missing_percent} />
                    </span>
                    <span className="tabular">
                      {percentText(field.missing_percent)}
                    </span>
                  </span>
                </td>
                <td className="tabular hidden px-3 py-2 text-right whitespace-nowrap lg:table-cell">
                  {percentText(field.unique_percent)}
                </td>
                <td className="hidden max-w-40 px-3 py-2 text-status-neutral xl:table-cell">
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
  const sort: SortKey = ["table", "missing", "name"].includes(sortParam)
    ? (sortParam as SortKey)
    : "table";
  const setSort = (next: SortKey) => setSortParam(next);

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
    if (sort === "missing") {
      return [...filtered].sort(
        (a, b) => (b.missing_percent ?? 0) - (a.missing_percent ?? 0),
      );
    }
    if (sort === "name") {
      return [...filtered].sort((a, b) => a.column.localeCompare(b.column));
    }
    return filtered;
  }, [fields, kindFilter, query, sort]);

  const searchId = `${panelId}-search`;
  const sortId = `${panelId}-sort`;

  return (
    <div id={panelId} className="flex min-w-0 flex-col gap-3">
      <header className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <h3 className="min-w-0 text-sm font-semibold"><Marquee>{profile.name}</Marquee></h3>
        <span className="tabular text-xs text-status-neutral">
          {profile.rows.toLocaleString()} rows · {profile.columns} columns
        </span>
      </header>

      <div className="flex flex-wrap gap-1.5">
        {Object.entries(profile.semantic_type_counts ?? {}).map(
          ([type, count]) => (
            <Badge key={type}>
              {type}: {count}
            </Badge>
          ),
        )}
      </div>

      <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
        <label htmlFor={searchId} className="text-sm text-status-neutral">
          Find column
        </label>
        <input
          id={searchId}
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="name, dtype or semantic type…"
          className="min-w-40 rounded-base border border-border bg-bg px-2 py-1 text-sm"
        />
        <label htmlFor={sortId} className="text-sm text-status-neutral">
          Sort
        </label>
        <select
          id={sortId}
          value={sort}
          onChange={(event) => setSort(event.target.value as SortKey)}
          className="rounded-base border border-border bg-bg px-2 py-1 text-sm"
        >
          <option value="table">Table order</option>
          <option value="missing">Most missing</option>
          <option value="name">Name A–Z</option>
        </select>
        <Hint label="Distribution">
          A mini histogram for numeric columns and top-value bars for
          categorical ones, computed from the table (sampled above the row cap).
          The bar beside Missing is the column&apos;s null rate.
        </Hint>
        <span className="tabular ml-auto text-xs text-status-neutral">
          {visible.length} of {fields.length} columns
        </span>
      </div>

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

function FieldProfiles({
  sessionId,
  profiles,
}: {
  sessionId: string;
  profiles: DatasetProfileSummary[];
}) {
  const [datasetParam, setSelectedId] = useRouteSearchParam("dataset");
  const selectedId = profiles.some(
    (profile) => profile.dataset_id === datasetParam,
  )
    ? datasetParam
    : (profiles[0]?.dataset_id ?? "");

  const selected =
    profiles.find((profile) => profile.dataset_id === selectedId) ??
    profiles[0]!;
  const panelId = "profile-fields";

  return (
    <>
      <ProfilesSummary profiles={profiles} />
      <Card
        tone="quiet"
        className="grid gap-4 p-3 lg:grid-cols-[minmax(11rem,15rem)_minmax(0,1fr)]"
      >
        <label className="flex min-w-0 flex-col gap-1 text-xs lg:hidden">
          <span className="text-status-neutral">Profile dataset</span>
          <select
            value={selected.dataset_id}
            onChange={(event) => setSelectedId(event.target.value)}
            className="w-full rounded-base border border-border bg-bg px-2 py-1.5 text-sm"
          >
            {profiles.map((profile) => (
              <option key={profile.dataset_id} value={profile.dataset_id}>
                {profile.name}
              </option>
            ))}
          </select>
        </label>
        <div className="hidden lg:block">
          <DatasetRail
            profiles={profiles}
            selectedId={selected.dataset_id}
            onSelect={setSelectedId}
            panelId={panelId}
          />
        </div>
        <FieldPanel
          key={selected.dataset_id}
          sessionId={sessionId}
          profile={selected}
          panelId={panelId}
        />
      </Card>
    </>
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
  charts: ChartSummary[];
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
      group.charts.push(chart);
    } else {
      groups.set(chart.dataset_id, {
        datasetId: chart.dataset_id,
        datasetName: chart.dataset_name,
        charts: [chart],
      });
    }
  }
  return [...groups.values()].sort((a, b) =>
    a.datasetName < b.datasetName ? -1 : a.datasetName > b.datasetName ? 1 : 0,
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
    <Card tone="quiet" className="flex flex-col gap-3 p-3">
      <h3 className="text-sm font-semibold">Dataset: {group.datasetName}</h3>
      {group.charts.length > 1 ? (
        <div className="grid gap-3 xl:grid-cols-2">
          {group.charts.map((chart) => (
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
          chart={group.charts[0]!}
          sessionId={sessionId}
          onZoom={() => onZoom(group.charts[0]!)}
        />
      )}
    </Card>
  );
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

function FieldProfilesWorkspace({ sessionId }: { sessionId: string }) {
  const profiles = useProfiles(sessionId);
  const profileList = profiles.data?.datasets ?? [];

  return (
    <section
      id="profiles-fields-panel"
      role="tabpanel"
      aria-labelledby="profiles-fields-tab"
      className="flex min-w-0 flex-col gap-3"
    >
      <SectionHeader
        title="Field profiles"
        description="Choose one table, then search or filter its columns by the kind of evidence you need."
      />
      {profiles.isPending && (
        <LoadingSkeleton lines={4} label="Loading profiles" />
      )}
      {profiles.isError && (
        <ErrorState error={profiles.error} onRetry={() => profiles.refetch()} />
      )}
      {profiles.data &&
        (profileList.length === 0 ? (
          <EmptyState
            title="No dataset profiles"
            description="Run an analysis to profile the datasets in this session."
          />
        ) : (
          <FieldProfiles sessionId={sessionId} profiles={profileList} />
        ))}
    </section>
  );
}

function ChartsWorkspace({
  sessionId,
  projectId,
}: {
  sessionId: string;
  projectId: string;
}) {
  const charts = useCharts(sessionId);
  const [zoomed, setZoomed] = useState<ChartSummary | null>(null);
  const [splitParam, setSplitParam] = useRouteSearchParam("split");
  const chartItems = charts.data?.pages.flatMap((page) => page.items) ?? [];
  const chartGroups = groupChartsByDataset(chartItems);
  const [leftId = "", rightId = ""] = splitParam.split(",", 2);
  const splitLeft = chartItems.find((chart) => chart.artifact_id === leftId);
  const splitRight = chartItems.find((chart) => chart.artifact_id === rightId);
  /* A shared split URL may point at charts beyond the first listing page.
   * Follow the cursor until both ids are found (or the listing ends), so a
   * cold deep link restores the requested pair instead of silently offering a
   * different first-page pair. */
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
    for (const left of chartItems) {
      const right = chartItems.find(
        (candidate) =>
          candidate.artifact_id !== left.artifact_id &&
          sharedChartField(left, candidate) !== null,
      );
      if (right) return [left, right] as const;
    }
    return null;
  })();

  return (
    <section
      id="profiles-charts-panel"
      role="tabpanel"
      aria-labelledby="profiles-charts-tab"
      className="flex min-w-0 flex-col gap-4"
    >
      <SectionHeader
        title="Charts"
        description="Review charts already produced by the analysis, or open the builder for a focused one-off view."
      />

      <section aria-labelledby="custom-chart-heading">
        <CustomChartBuilder sessionId={sessionId} projectId={projectId} />
      </section>

      <section
        aria-label="Generated charts"
        className="flex flex-col gap-3"
      >
        <SectionHeader
          title="Generated charts"
          description={
            charts.data
              ? `${chartItems.length} loaded across ${chartGroups.length} dataset${chartGroups.length === 1 ? "" : "s"}.`
              : "Charts produced and persisted by this analysis."
          }
        />
        {charts.data && initialSplitPair && !splitPair && (
          <button
            type="button"
            onClick={() =>
              setSplitParam(
                `${initialSplitPair[0].artifact_id},${initialSplitPair[1].artifact_id}`,
              )
            }
            className="self-start rounded-base border border-border px-3 py-1.5 text-sm hover:bg-surface"
          >
            Open linked split view
          </button>
        )}
        {splitPair && (
          <LinkedChartSplit
            left={splitPair[0]}
            right={splitPair[1]}
            charts={chartItems}
            sessionId={sessionId}
            onPairChange={(left, right) => setSplitParam(`${left},${right}`)}
            onClose={() => setSplitParam("")}
            onZoom={setZoomed}
          />
        )}
        {charts.isPending && (
          <LoadingSkeleton lines={4} label="Loading charts" />
        )}
        {charts.isError && (
          <ErrorState error={charts.error} onRetry={() => charts.refetch()} />
        )}
        {charts.data &&
          (chartItems.length === 0 ? (
            <EmptyState
              title="No charts"
              description="No chart specs were generated for this session."
            />
          ) : (
            <>
              <div className="flex flex-col gap-3">
                {chartGroups.map((group) => (
                  <ChartGroupSection
                    key={group.datasetId}
                    group={group}
                    sessionId={sessionId}
                    onZoom={setZoomed}
                  />
                ))}
              </div>
              {charts.hasNextPage && (
                <button
                  type="button"
                  onClick={() => charts.fetchNextPage()}
                  disabled={charts.isFetchingNextPage}
                  className="self-start rounded-base border border-border px-3 py-1.5 text-sm hover:bg-surface disabled:opacity-60"
                >
                  {charts.isFetchingNextPage ? "Loading…" : "Load more"}
                </button>
              )}
            </>
          ))}
      </section>

      {zoomed && (
        <ChartZoomModal
          chart={zoomed}
          sessionId={sessionId}
          onClose={() => setZoomed(null)}
        />
      )}
    </section>
  );
}

type ProfilesView = "fields" | "charts";

export function Component() {
  const { projectId = "", sessionId = "" } = useParams();
  const [viewParam, setViewParam] = useRouteSearchParam("view", "fields");
  const view: ProfilesView = viewParam === "charts" ? "charts" : "fields";

  return (
    <div className="mx-auto flex w-[95%] max-w-data min-w-0 flex-col gap-4 p-6">
      <SectionHeader
        level={1}
        title="Profiles & Charts"
        description="Inspect column-level evidence or switch to the charts produced from this session."
      />

      <div
        role="tablist"
        aria-label="Profiles and charts tasks"
        className="grid grid-cols-2 gap-1 rounded-base border border-border bg-surface p-1 sm:flex sm:w-fit"
      >
        <button
          id="profiles-fields-tab"
          type="button"
          role="tab"
          aria-selected={view === "fields"}
          aria-controls="profiles-fields-panel"
          onClick={() => setViewParam("fields")}
          className={`rounded-base px-3 py-1.5 text-sm font-medium ${
            view === "fields"
              ? "bg-bg text-text shadow-sm"
              : "text-status-neutral hover:text-text"
          }`}
        >
          Field profiles
        </button>
        <button
          id="profiles-charts-tab"
          type="button"
          role="tab"
          aria-selected={view === "charts"}
          aria-controls="profiles-charts-panel"
          onClick={() => setViewParam("charts")}
          className={`rounded-base px-3 py-1.5 text-sm font-medium ${
            view === "charts"
              ? "bg-bg text-text shadow-sm"
              : "text-status-neutral hover:text-text"
          }`}
        >
          Charts
        </button>
      </div>

      {view === "fields" ? (
        <FieldProfilesWorkspace sessionId={sessionId} />
      ) : (
        <ChartsWorkspace sessionId={sessionId} projectId={projectId} />
      )}
    </div>
  );
}
