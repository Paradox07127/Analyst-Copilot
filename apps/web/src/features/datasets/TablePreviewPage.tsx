import { useEffect, useId, useMemo, useRef, useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router";
import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  useReactTable,
} from "@tanstack/react-table";
import { useVirtualizer } from "@tanstack/react-virtual";
import {
  type ColumnDistribution,
  type DatasetHandle,
  type DatasetPreview,
  type DatasetSchema,
} from "../../api/client";
import { tablePath } from "../../app/paths";
import {
  PREVIEW_PAGE_SIZE,
  useDatasetPreview,
  useDatasetDistributions,
  useDatasetSchema,
  useDatasets,
} from "../../api/hooks";
import {
  EmptyState,
  ErrorState,
  LoadingSkeleton,
  PartialState,
} from "../../components/async-states";
import { Marquee, SectionHeader, formatCompact } from "../../components/ui";
import { HeaderDistributionChart } from "./ColumnDistributionStrip";
import { TypeIcon, classifyColumn } from "./mini-charts";

type PreviewRow = unknown[];
type SortDirection = "asc" | "desc";
interface SortState {
  columnId: string;
  direction: SortDirection;
}

const DEFAULT_VISIBLE_COLUMNS = 24;
const INDEX_COLUMN_WIDTH = 56;
const DATA_COLUMN_WIDTH = 224;

const columnHelper = createColumnHelper<PreviewRow>();

function formatCell(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

export function filterPreviewRows(
  rows: PreviewRow[],
  query: string,
): PreviewRow[] {
  const q = query.trim().toLowerCase();
  if (!q) return rows;
  return rows.filter((row) =>
    row.some((cell) => formatCell(cell).toLowerCase().includes(q)),
  );
}

/* Numeric cells compare by value, everything else by string; null/undefined
 * always sort last regardless of direction, matching st.dataframe's toolbar. */
export function sortPreviewRows(
  rows: PreviewRow[],
  columnIndex: number,
  direction: SortDirection,
): PreviewRow[] {
  return rows
    .map((row, index) => ({ row, index }))
    .sort((x, y) => {
      const a = x.row[columnIndex];
      const b = y.row[columnIndex];
      const aEmpty = a === null || a === undefined;
      const bEmpty = b === null || b === undefined;
      if (aEmpty || bEmpty) {
        if (aEmpty && bEmpty) return x.index - y.index;
        return aEmpty ? 1 : -1;
      }
      const cmp =
        typeof a === "number" && typeof b === "number"
          ? a - b
          : String(a).localeCompare(String(b));
      return direction === "asc" ? cmp : -cmp;
    })
    .map(({ row }) => row);
}

function PreviewTable({
  preview,
  schema,
  distributions,
  offset,
  query,
  sort,
  onSortChange,
  displayedRows,
  showAllColumns,
}: {
  preview: DatasetPreview;
  schema: DatasetSchema | undefined;
  distributions: ColumnDistribution[] | undefined;
  offset: number;
  query: string;
  sort: SortState | null;
  onSortChange: (sort: SortState | null) => void;
  displayedRows: PreviewRow[];
  showAllColumns: boolean;
}) {
  const dtypeByName = useMemo(
    () => new Map(schema?.columns.map((col) => [col.name, col.dtype])),
    [schema],
  );
  /* Right-aligned tabular figures only line up if the column really is
   * numeric; the preview payload is untyped `unknown[]`, so the dtype from
   * the schema is the only reliable signal. */
  const numericColumns = useMemo(
    () =>
      new Set(
        (schema?.columns ?? [])
          .filter((col) => classifyColumn(col.dtype) === "numeric")
          .map((col) => col.name),
      ),
    [schema],
  );

  const columnIndexById = useMemo(
    () => new Map(preview.columns.map((name, index) => [name, index])),
    [preview.columns],
  );
  const distributionByName = useMemo(
    () => new Map(distributions?.map((distribution) => [distribution.name, distribution])),
    [distributions],
  );

  const visibleColumnNames = useMemo(
    () =>
      showAllColumns
        ? preview.columns
        : preview.columns.slice(0, DEFAULT_VISIBLE_COLUMNS),
    [preview.columns, showAllColumns],
  );
  const columns = useMemo(
    () =>
      visibleColumnNames.map((name) => {
        const index = columnIndexById.get(name) ?? 0;
        return columnHelper.accessor((row) => row[index], {
          id: name,
          header: name,
        });
      }),
    [visibleColumnNames, columnIndexById],
  );

  const table = useReactTable({
    data: displayedRows,
    columns,
    getCoreRowModel: getCoreRowModel(),
  });

  const scrollRef = useRef<HTMLDivElement>(null);
  // Rows highlight through CSS group-hover; a column has no CSS hover scope,
  // so the hovered column id is tracked here to draw the other crosshair arm.
  const [hoveredColumn, setHoveredColumn] = useState<string | null>(null);
  const rows = table.getRowModel().rows;
  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => 46,
    overscan: 10,
    /* jsdom reports zero-size rects; a non-zero initial rect keeps rows
     * renderable in tests until a real ResizeObserver measurement lands. */
    initialRect: { width: 800, height: 600 },
  });
  const virtualRows = virtualizer.getVirtualItems();
  const totalSize = virtualizer.getTotalSize();
  const paddingTop = virtualRows[0]?.start ?? 0;
  const paddingBottom =
    totalSize - (virtualRows[virtualRows.length - 1]?.end ?? 0);

  const toggleSort = (columnId: string) => {
    if (sort?.columnId !== columnId) {
      onSortChange({ columnId, direction: "asc" });
    } else if (sort.direction === "asc") {
      onSortChange({ columnId, direction: "desc" });
    } else {
      onSortChange(null);
    }
  };

  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col gap-2">
      <div className="flex items-center justify-between gap-3 px-0.5 text-xs text-status-neutral">
        <span>
          Showing {visibleColumnNames.length} of {preview.columns.length} columns
        </span>
        <span>Scroll horizontally to explore columns</span>
      </div>

      {displayedRows.length === 0 ? (
        <EmptyState
          title={query ? "No rows match your search" : "No rows on this page"}
          description={
            query
              ? `No rows among the loaded ${preview.rows.length.toLocaleString()} match “${query}”.`
              : "This page of the preview has no rows."
          }
        />
      ) : (
        <div
          ref={scrollRef}
          className="min-h-64 flex-1 overflow-auto rounded-base border border-table-border bg-bg"
        >
          <table
            className="min-w-max border-collapse text-sm"
            style={{
              minWidth: INDEX_COLUMN_WIDTH + columns.length * DATA_COLUMN_WIDTH,
            }}
          >
            <thead className="sticky top-0 z-10 bg-table-header-bg">
              {table.getHeaderGroups().map((headerGroup) => (
                <tr key={headerGroup.id}>
                  <th
                    scope="col"
                    className="sticky left-0 z-30 w-14 min-w-14 border-b border-r border-table-border bg-table-header-bg px-2 py-2 text-right font-mono text-xs text-status-neutral"
                  >
                    #
                  </th>
                  {headerGroup.headers.map((header) => {
                    const activeSort =
                      sort !== null && sort.columnId === header.column.id
                        ? sort
                        : null;
                    const ariaSort: "ascending" | "descending" | "none" =
                      activeSort === null
                        ? "none"
                        : activeSort.direction === "asc"
                          ? "ascending"
                          : "descending";
                    const dtype = dtypeByName.get(header.column.id) ?? "";
                    const distribution = distributionByName.get(header.column.id);
                    return (
                      <th
                        key={header.id}
                        scope="col"
                        aria-sort={ariaSort}
                        className="w-56 min-w-56 border-b border-r border-table-border px-3 py-2 text-left align-top"
                      >
                        <span className="flex items-start gap-1.5">
                          <TypeIcon kind={classifyColumn(dtype)} />
                          <button
                            type="button"
                            onClick={() => toggleSort(header.column.id)}
                            className="flex min-w-0 flex-1 items-center gap-1 text-left font-semibold hover:text-primary"
                          >
                            <Marquee>
                              {flexRender(
                                header.column.columnDef.header,
                                header.getContext(),
                              )}
                            </Marquee>
                            {activeSort && (
                              <span aria-hidden="true" className="shrink-0">
                                {activeSort.direction === "asc" ? "▲" : "▼"}
                              </span>
                            )}
                          </button>
                        </span>
                        <span className="mt-1 block font-mono text-xs font-normal text-status-neutral">
                          {dtype || "type unavailable"}
                        </span>
                        <HeaderDistributionChart dist={distribution} />
                      </th>
                    );
                  })}
                </tr>
              ))}
            </thead>
            <tbody>
              {paddingTop > 0 && (
                <tr aria-hidden>
                  <td colSpan={columns.length + 1} style={{ height: paddingTop }} />
                </tr>
              )}
              {virtualRows.map((virtualRow) => {
                const row = rows[virtualRow.index];
                if (!row) return null;
                /* Banding follows the absolute row number, not the virtual
                 * window index, so the stripes do not swap as you scroll. */
                const banded = (offset + virtualRow.index) % 2 === 1;
                return (
                  <tr
                    key={row.id}
                    className={`group border-b border-hairline ${banded ? "bg-surface/40" : ""}`}
                    onMouseLeave={() => setHoveredColumn(null)}
                  >
                    <td
                      className={`sticky left-0 z-10 w-14 min-w-14 border-r border-table-border px-2 py-1.5 text-right font-mono text-xs text-status-neutral group-hover:text-primary ${
                        banded ? "bg-table-header-bg" : "bg-bg"
                      }`}
                    >
                      {offset + virtualRow.index + 1}
                    </td>
                    {row.getVisibleCells().map((cell) => {
                      const text = formatCell(cell.getValue());
                      const numeric = numericColumns.has(cell.column.id);
                      const columnHovered = hoveredColumn === cell.column.id;
                      return (
                        <td
                          key={cell.id}
                          title={text}
                          onMouseEnter={() => setHoveredColumn(cell.column.id)}
                          /* Brand tint, not another shade of surface: on a
                           * dark theme surface-on-surface is invisible, and an
                           * invisible crosshair helps nobody track a column. */
                          className={`w-56 min-w-56 max-w-56 border-r border-table-border px-3 py-1.5 align-top font-mono text-xs group-hover:bg-primary/12 ${
                            numeric ? "tabular text-right" : ""
                          } ${columnHovered ? "bg-primary/8" : ""}`}
                        >
                          <span className="block line-clamp-2 break-all leading-5">
                            {text}
                          </span>
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
              {paddingBottom > 0 && (
                <tr aria-hidden>
                  <td
                    colSpan={columns.length + 1}
                    style={{ height: paddingBottom }}
                  />
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function PreviewControls({
  projectId,
  sessionId,
  datasetId,
  datasets,
  preview,
  query,
  searchOpen,
  showAllColumns,
  onSearchOpenChange,
  onQueryChange,
  onShowAllColumnsChange,
}: {
  projectId: string;
  sessionId: string;
  datasetId: string;
  datasets: DatasetHandle[];
  preview: DatasetPreview | undefined;
  query: string;
  searchOpen: boolean;
  showAllColumns: boolean;
  onSearchOpenChange: (open: boolean) => void;
  onQueryChange: (query: string) => void;
  onShowAllColumnsChange: (show: boolean) => void;
}) {
  const navigate = useNavigate();
  const selectId = useId();

  return (
    <div className="flex min-w-0 flex-wrap items-end gap-2">
      <label className="flex min-w-48 flex-1 flex-col gap-1 text-xs font-medium text-status-neutral" htmlFor={selectId}>
          Table
          <select
            id={selectId}
            value={datasetId}
            onChange={(event) =>
              navigate(tablePath(projectId, sessionId, event.target.value))
            }
            className="min-w-0 max-w-full rounded-base border border-border bg-bg px-2.5 py-1.5 text-sm text-text"
          >
            {datasets.length === 0 && (
              <option value={datasetId}>Current table</option>
            )}
            {datasets.map((item) => (
              <option key={item.dataset_id} value={item.dataset_id}>
                {item.display_name || "Untitled table"}
              </option>
            ))}
          </select>
      </label>
      {preview && preview.columns.length > DEFAULT_VISIBLE_COLUMNS && (
        <button
          type="button"
          onClick={() => onShowAllColumnsChange(!showAllColumns)}
          className="rounded-base border border-border bg-bg px-3 py-1.5 text-sm font-medium hover:border-primary hover:text-primary"
        >
          {showAllColumns
            ? `Show first ${DEFAULT_VISIBLE_COLUMNS} columns`
            : `Show all ${preview.columns.length} columns`}
        </button>
      )}
      <div className="ml-auto flex min-w-0 items-center">
        {searchOpen && preview && (
          <>
            <label htmlFor="table-preview-search" className="sr-only">
              Find in loaded rows
            </label>
            <input
              id="table-preview-search"
              type="search"
              value={query}
              onChange={(event) => onQueryChange(event.target.value)}
              placeholder="Search loaded rows"
              className="w-56 rounded-l-base border border-border bg-bg px-3 py-1.5 text-sm transition-[width] duration-base focus:relative focus:z-10"
            />
          </>
        )}
        <button
          type="button"
          aria-label="Search loaded rows"
          aria-expanded={searchOpen}
          onClick={() => {
            if (searchOpen) onQueryChange("");
            onSearchOpenChange(!searchOpen);
          }}
          disabled={!preview}
          title={searchOpen ? "Close search" : "Search loaded rows"}
          className={`inline-flex size-9 items-center justify-center border border-border bg-bg text-status-neutral hover:border-primary hover:text-primary disabled:opacity-50 ${
            searchOpen ? "rounded-r-base border-l-0" : "rounded-base"
          }`}
        >
          <svg aria-hidden viewBox="0 0 24 24" className="size-4 fill-none stroke-current stroke-2">
            <circle cx="11" cy="11" r="6" />
            <path d="m16 16 4 4" />
          </svg>
        </button>
      </div>
    </div>
  );
}

export function Component() {
  const { projectId = "", sessionId = "", datasetId = "" } = useParams();
  const [searchParams, setSearchParams] = useSearchParams();
  const offsetParam = Number.parseInt(searchParams.get("offset") ?? "0", 10);
  const offset =
    Number.isFinite(offsetParam) && offsetParam > 0 ? offsetParam : 0;
  const query = searchParams.get("q") ?? "";
  const sortColumn = searchParams.get("sort");
  const sortDirection = searchParams.get("dir");
  const sort: SortState | null =
    sortColumn && (sortDirection === "asc" || sortDirection === "desc")
      ? { columnId: sortColumn, direction: sortDirection }
      : null;
  const [searchOpen, setSearchOpen] = useState(Boolean(query));
  const [showAllColumns, setShowAllColumns] = useState(false);

  const schema = useDatasetSchema(sessionId, datasetId);
  const preview = useDatasetPreview(sessionId, datasetId, offset);
  const distributions = useDatasetDistributions(
    sessionId,
    datasetId,
    Boolean(preview.data),
  );
  const datasets = useDatasets(sessionId);
  const dataset = datasets.data?.find((item) => item.dataset_id === datasetId);
  const displayedRows = useMemo(() => {
    if (!preview.data) return [];
    const searched = filterPreviewRows(preview.data.rows, query);
    const columnIndex = sort
      ? preview.data.columns.indexOf(sort.columnId)
      : -1;
    return sort && columnIndex >= 0
      ? sortPreviewRows(searched, columnIndex, sort.direction)
      : searched;
  }, [preview.data, query, sort]);

  useEffect(() => {
    setSearchOpen(Boolean(query));
    setShowAllColumns(false);
  }, [datasetId]);

  const goToOffset = (next: number) => {
    setSearchParams(
      (params) => {
        if (next <= 0) params.delete("offset");
        else params.set("offset", String(next));
        return params;
      },
      { replace: true },
    );
  };

  const setQuery = (next: string) => {
    setSearchParams(
      (params) => {
        if (next.trim()) params.set("q", next);
        else params.delete("q");
        return params;
      },
      { replace: true },
    );
  };

  const setSort = (next: SortState | null) => {
    setSearchParams(
      (params) => {
        if (next) {
          params.set("sort", next.columnId);
          params.set("dir", next.direction);
        } else {
          params.delete("sort");
          params.delete("dir");
        }
        return params;
      },
      { replace: true },
    );
  };

  return (
    <div className="mx-auto flex h-full w-[95%] max-w-data min-w-0 flex-col gap-2 overflow-hidden px-6 pt-5 pb-2">
      <SectionHeader
        level={1}
        title="Table Preview"
        actions={
          dataset && (
            <span
              className="tabular text-sm text-status-neutral"
              title={
                dataset.row_count == null
                  ? undefined
                  : `${dataset.row_count.toLocaleString()} rows`
              }
            >
              {dataset.row_count == null
                ? "rows unknown"
                : `${formatCompact(dataset.row_count)} rows`}
              {" · "}
              {(dataset.schema ?? []).length} cols
            </span>
          )
        }
      />
      <PreviewControls
        projectId={projectId}
        sessionId={sessionId}
        datasetId={datasetId}
        datasets={datasets.data ?? []}
        preview={preview.data}
        query={query}
        searchOpen={searchOpen}
        showAllColumns={showAllColumns}
        onSearchOpenChange={setSearchOpen}
        onQueryChange={setQuery}
        onShowAllColumnsChange={setShowAllColumns}
      />

      {preview.isPending && (
        <LoadingSkeleton lines={5} label="Loading preview" />
      )}
      {preview.isError && (
        <ErrorState error={preview.error} onRetry={() => preview.refetch()} />
      )}
      {schema.isError && preview.data && (
        <PartialState error={schema.error} onRetry={() => schema.refetch()} />
      )}
      {distributions.isError && preview.data && (
        <PartialState
          error={distributions.error}
          onRetry={() => distributions.refetch()}
        />
      )}
      {datasets.isError && preview.data && (
        <PartialState
          error={datasets.error}
          onRetry={() => datasets.refetch()}
        />
      )}
      {preview.data &&
        (preview.data.rows.length === 0 && offset === 0 ? (
          <EmptyState
            title="There is no data to preview"
            description="This table contains no rows."
          />
        ) : (
          <>
            <PreviewTable
              key={`${sessionId}:${datasetId}`}
              preview={preview.data}
              schema={schema.data}
              distributions={distributions.data?.columns}
              offset={offset}
              query={query}
              sort={sort}
              onSortChange={setSort}
              displayedRows={displayedRows}
              showAllColumns={showAllColumns}
            />
            <footer className="relative z-20 flex shrink-0 items-center gap-3 bg-bg">
              <button
                type="button"
                onClick={() => goToOffset(offset - PREVIEW_PAGE_SIZE)}
                disabled={offset === 0 || preview.isPlaceholderData}
                className="rounded-base border border-border px-3 py-1 text-sm hover:bg-surface disabled:opacity-50"
              >
                Prev
              </button>
              <button
                type="button"
                onClick={() => goToOffset(offset + PREVIEW_PAGE_SIZE)}
                disabled={!preview.data.has_more || preview.isPlaceholderData}
                className="rounded-base border border-border px-3 py-1 text-sm hover:bg-surface disabled:opacity-50"
              >
                Next
              </button>
              <span className="tabular text-sm text-status-neutral">
                Rows {preview.data.rows.length === 0 ? 0 : offset + 1}–
                {offset + preview.data.rows.length}
                {dataset?.row_count != null
                  ? ` of ${dataset.row_count.toLocaleString()}`
                  : preview.data.has_more
                    ? " · more available"
                    : ""}
              </span>
            </footer>
          </>
        ))}
    </div>
  );
}
