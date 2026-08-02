/* Ad-hoc chart builder. Shaping/aggregation semantics live server-side in
 * chart_builder.py; this component only assembles the request and renders
 * the response. */

import { useEffect, useMemo, useState } from "react";
import {
  type CustomChartRequest,
  type DatasetColumn,
} from "../../api/client";
import { useBuildCustomChart, useDatasets, useDatasetSchema } from "../../api/hooks";
import {
  EmptyState,
  ErrorState,
  LoadingSkeleton,
  formatUnknownError,
} from "../../components/async-states";
import { VegaChart } from "./VegaChart";

const CHART_TYPES = ["bar", "line", "point", "area", "histogram"] as const;
const AGGREGATES = ["none", "count", "mean", "sum", "median"] as const;
const ROW_COUNT_Y = "Row count";
const NO_COLOR = "None";

type ChartType = (typeof CHART_TYPES)[number];
type Aggregate = (typeof AGGREGATES)[number];

function isNumericDtype(dtype: string): boolean {
  return /^(u?int|float)/i.test(dtype);
}

/* Mirrors chart_builder.default_custom_agg: numeric Y sums, everything else counts. */
function defaultAggregate(yColumn: string, numericColumns: Set<string>): Aggregate {
  return yColumn !== ROW_COUNT_Y && numericColumns.has(yColumn) ? "sum" : "count";
}

function errorText(error: unknown): string {
  return formatUnknownError(error, "The chart request failed.");
}

/* Both the 5,000-row cap and a 2 MB inline-byte budget can truncate a
 * response, so the copy reports what came back instead of naming a cause. */
function truncationNotice(rowCount: number, sourceRowCount: number): string {
  return `Chart preview is truncated: showing ${rowCount.toLocaleString()} of ${sourceRowCount.toLocaleString()} rows.`;
}

const selectClass =
  "min-w-0 w-full rounded-base border border-border bg-bg px-2 py-1 text-sm sm:w-auto";

function ChartControls({
  sessionId,
  projectId,
  datasetId,
  columns,
}: {
  sessionId: string;
  projectId: string;
  datasetId: string;
  columns: DatasetColumn[];
}) {
  const columnNames = useMemo(() => columns.map((c) => c.name), [columns]);
  const numericColumns = useMemo(
    () =>
      new Set(columns.filter((c) => isNumericDtype(c.dtype)).map((c) => c.name)),
    [columns],
  );

  const [chartType, setChartType] = useState<ChartType>("bar");
  const xOptions = useMemo(
    () =>
      chartType === "histogram"
        ? columnNames.filter((name) => numericColumns.has(name))
        : columnNames,
    [chartType, columnNames, numericColumns],
  );

  const [xColumn, setXColumn] = useState(xOptions[0] ?? "");
  useEffect(() => {
    if (xOptions.length > 0 && !xOptions.includes(xColumn)) {
      setXColumn(xOptions[0]!);
    }
  }, [xOptions, xColumn]);

  const [yColumn, setYColumn] = useState(ROW_COUNT_Y);
  const [colorColumn, setColorColumn] = useState(NO_COLOR);
  const [aggregate, setAggregate] = useState<Aggregate>(() =>
    defaultAggregate(ROW_COUNT_Y, numericColumns),
  );
  const [dropMissing, setDropMissing] = useState(true);
  const [dropOutliers, setDropOutliers] = useState(false);

  const buildChart = useBuildCustomChart(sessionId, projectId);

  function handleYChange(next: string) {
    setYColumn(next);
    setAggregate(defaultAggregate(next, numericColumns));
    if (next === ROW_COUNT_Y) setDropOutliers(false);
  }

  function handleBuild() {
    const body: CustomChartRequest = {
      dataset_id: datasetId,
      chart_type: chartType,
      x_column: xColumn,
      y_column: yColumn === ROW_COUNT_Y ? null : yColumn,
      color_column: colorColumn === NO_COLOR ? null : colorColumn,
      aggregate,
      drop_missing: dropMissing,
      // Never send outlier-dropping against a null Y: the API 422s on that combination.
      drop_outliers: yColumn === ROW_COUNT_Y ? false : dropOutliers,
    };
    buildChart.mutate(body, {
      onSuccess: (view) => setAggregate(view.aggregate as Aggregate),
    });
  }

  const outliersDisabled = yColumn === ROW_COUNT_Y;
  const histogramBlocked = chartType === "histogram" && xOptions.length === 0;

  return (
    <div className="flex flex-col gap-3">
      <label className="flex flex-col gap-1 text-sm sm:flex-row sm:items-center sm:gap-2">
        <span className="text-status-neutral">Chart type</span>
        <select
          value={chartType}
          onChange={(event) => setChartType(event.target.value as ChartType)}
          className={selectClass}
        >
          {CHART_TYPES.map((type) => (
            <option key={type} value={type}>
              {type}
            </option>
          ))}
        </select>
      </label>

      {histogramBlocked ? (
        <p className="text-sm text-status-neutral">
          Histogram needs at least one numeric column.
        </p>
      ) : (
        <>
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
            <label className="flex min-w-0 flex-col gap-1 text-sm">
              <span className="text-status-neutral">X column</span>
              <select
                value={xColumn}
                onChange={(event) => setXColumn(event.target.value)}
                className={selectClass}
              >
                {xOptions.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex min-w-0 flex-col gap-1 text-sm">
              <span className="text-status-neutral">Y column</span>
              <select
                value={yColumn}
                onChange={(event) => handleYChange(event.target.value)}
                className={selectClass}
              >
                <option value={ROW_COUNT_Y}>{ROW_COUNT_Y}</option>
                {columnNames.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex min-w-0 flex-col gap-1 text-sm">
              <span className="text-status-neutral">Color/group</span>
              <select
                value={colorColumn}
                onChange={(event) => setColorColumn(event.target.value)}
                className={selectClass}
              >
                <option value={NO_COLOR}>{NO_COLOR}</option>
                {columnNames.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <label className="flex max-w-sm flex-col gap-1 text-sm">
            <span className="text-status-neutral">Aggregation</span>
            <select
              value={aggregate}
              onChange={(event) => setAggregate(event.target.value as Aggregate)}
              className={selectClass}
            >
              {AGGREGATES.map((agg) => (
                <option key={agg} value={agg}>
                  {agg}
                </option>
              ))}
            </select>
          </label>

          <div className="flex flex-wrap gap-4">
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={dropMissing}
                onChange={(event) => setDropMissing(event.target.checked)}
              />
              <span>Drop missing rows for selected columns</span>
            </label>
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={dropOutliers}
                disabled={outliersDisabled}
                onChange={(event) => setDropOutliers(event.target.checked)}
              />
              <span>Exclude IQR outliers from Y</span>
            </label>
          </div>

          <button
            type="button"
            onClick={handleBuild}
            disabled={buildChart.isPending}
            className="self-start rounded-base border border-border px-4 py-2 text-sm font-medium hover:bg-surface disabled:opacity-50"
          >
            {buildChart.isPending ? "Building…" : "Build chart"}
          </button>

          {buildChart.isError && (
            <p role="alert" className="text-sm text-status-critical">
              {errorText(buildChart.error)}
            </p>
          )}

          {buildChart.data &&
            (buildChart.data.row_count === 0 ? (
              <EmptyState title="No rows remain after the selected chart filters." />
            ) : (
              <div className="flex flex-col gap-2">
                {buildChart.data.truncated && (
                  <p className="text-xs text-status-neutral">
                    {truncationNotice(
                      buildChart.data.row_count,
                      buildChart.data.source_row_count,
                    )}
                  </p>
                )}
                <VegaChart
                  spec={buildChart.data.spec}
                  label={`Custom ${buildChart.data.chart_type} chart`}
                />
              </div>
            ))}
        </>
      )}
    </div>
  );
}

function DatasetGate({
  sessionId,
  projectId,
  datasetId,
}: {
  sessionId: string;
  projectId: string;
  datasetId: string;
}) {
  const schema = useDatasetSchema(sessionId, datasetId);

  if (schema.isPending) {
    return <LoadingSkeleton lines={3} label="Loading columns" />;
  }
  if (schema.isError) {
    return <ErrorState error={schema.error} onRetry={() => schema.refetch()} />;
  }
  if (schema.data.columns.length === 0) {
    return (
      <p className="text-sm text-status-neutral">
        The selected dataset has no rows or columns to chart.
      </p>
    );
  }
  return (
    <ChartControls
      sessionId={sessionId}
      projectId={projectId}
      datasetId={datasetId}
      columns={schema.data.columns}
    />
  );
}

/* Dataset labels: only same-named datasets
 * get an id suffix, so the common case never shows raw ids. */
function datasetOptionLabel(
  dataset: { dataset_id: string; display_name: string },
  all: readonly { display_name: string }[],
): string {
  const duplicated =
    all.filter((other) => other.display_name === dataset.display_name).length > 1;
  return duplicated
    ? `${dataset.display_name} (${dataset.dataset_id.slice(-6)})`
    : dataset.display_name;
}

function CustomChartForm({
  sessionId,
  projectId,
  datasetId: controlledDatasetId,
}: {
  sessionId: string;
  projectId: string;
  datasetId?: string;
}) {
  const datasets = useDatasets(sessionId);
  const [localDatasetId, setLocalDatasetId] = useState("");
  const options = datasets.data ?? [];
  const datasetId = controlledDatasetId ?? localDatasetId;

  useEffect(() => {
    if (!controlledDatasetId && !localDatasetId && options.length > 0) {
      setLocalDatasetId(options[0]!.dataset_id);
    }
  }, [controlledDatasetId, localDatasetId, options]);

  if (datasets.isPending) {
    return <LoadingSkeleton lines={3} label="Loading datasets" />;
  }
  if (datasets.isError) {
    return <ErrorState error={datasets.error} onRetry={() => datasets.refetch()} />;
  }
  if (options.length === 0) return null;

  const selected = options.find((d) => d.dataset_id === datasetId) ?? options[0]!;

  return (
    <div className="flex flex-col gap-3 border-t border-hairline px-4 py-3">
      {controlledDatasetId ? (
        <p className="text-xs text-status-neutral">
          Building from <span className="font-medium text-text">{selected.display_name}</span>, the dataset selected above.
        </p>
      ) : (
        <label className="flex min-w-0 flex-col gap-1 text-sm sm:max-w-sm">
          <span className="text-status-neutral">Dataset</span>
          <select
            value={selected.dataset_id}
            onChange={(event) => setLocalDatasetId(event.target.value)}
            className={selectClass}
          >
            {options.map((dataset) => (
              <option key={dataset.dataset_id} value={dataset.dataset_id}>
                {datasetOptionLabel(dataset, options)}
              </option>
            ))}
          </select>
        </label>
      )}
      {selected.row_count === 0 ? (
        <p className="text-sm text-status-neutral">
          The selected dataset has no rows or columns to chart.
        </p>
      ) : (
        <DatasetGate
          key={selected.dataset_id}
          sessionId={sessionId}
          projectId={projectId}
          datasetId={selected.dataset_id}
        />
      )}
    </div>
  );
}

export function CustomChartBuilder({
  sessionId,
  projectId,
  datasetId,
}: {
  sessionId: string;
  projectId: string;
  datasetId?: string;
}) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="overflow-hidden rounded-base border border-border bg-bg">
      <div className="flex flex-wrap items-center justify-between gap-3 p-4">
        <div className="min-w-0">
          <h3 id="custom-chart-heading" className="text-sm font-semibold">
            Build a custom chart
          </h3>
          <p className="mt-0.5 text-xs text-status-neutral">
            Choose fields and aggregation only when the generated charts do not
            answer the question.
          </p>
        </div>
        <button
          type="button"
          aria-expanded={expanded}
          aria-controls="custom-chart-form"
          onClick={() => setExpanded((current) => !current)}
          className="shrink-0 rounded-base border border-border px-3 py-1.5 text-sm font-medium hover:bg-surface"
        >
          {expanded ? "Close builder" : "Open builder"}
        </button>
      </div>
      {expanded && (
        <div id="custom-chart-form">
          <CustomChartForm
            sessionId={sessionId}
            projectId={projectId}
            datasetId={datasetId}
          />
        </div>
      )}
    </div>
  );
}
