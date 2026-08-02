/* Ad-hoc chart builder. Shaping/aggregation semantics live server-side in
 * chart_builder.py; this component only assembles the request and renders
 * the response. */

import { useEffect, useMemo, useState } from "react";
import {
  type CustomChartRequest,
  type DatasetColumn,
} from "../../api/client";
import { useBuildCustomChart, useDatasetSchema } from "../../api/hooks";
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

  /* The server fences a histogram on X and everything else on Y, so the switch
   * has to name the axis it will actually cut. */
  const outliersDisabled = chartType !== "histogram" && yColumn === ROW_COUNT_Y;
  const outliersLabel =
    chartType === "histogram"
      ? "Exclude IQR outliers from X"
      : "Exclude IQR outliers from Y";

  function handleYChange(next: string) {
    setYColumn(next);
    setAggregate(defaultAggregate(next, numericColumns));
    if (next === ROW_COUNT_Y && chartType !== "histogram") setDropOutliers(false);
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
      // Never send outlier-dropping against a null Y: the API 422s on that
      // combination, except for a histogram, which fences its own X.
      drop_outliers: outliersDisabled ? false : dropOutliers,
    };
    buildChart.mutate(body, {
      onSuccess: (view) => setAggregate(view.aggregate as Aggregate),
    });
  }

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
              <span>{outliersLabel}</span>
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

function CustomChartForm({
  sessionId,
  projectId,
  datasetId,
  datasetName,
  rowCount,
}: {
  sessionId: string;
  projectId: string;
  datasetId: string;
  datasetName: string;
  rowCount: number;
}) {
  return (
    <div className="flex flex-col gap-3 border-t border-hairline px-4 py-3">
      <p className="text-xs text-status-neutral">
        Building from <span className="font-medium text-text">{datasetName}</span>, the dataset selected above.
      </p>
      {rowCount === 0 ? (
        <p className="text-sm text-status-neutral">
          The selected dataset has no rows or columns to chart.
        </p>
      ) : (
        <DatasetGate
          key={datasetId}
          sessionId={sessionId}
          projectId={projectId}
          datasetId={datasetId}
        />
      )}
    </div>
  );
}

export function CustomChartBuilder({
  sessionId,
  projectId,
  datasetId,
  datasetName,
  rowCount,
}: {
  sessionId: string;
  projectId: string;
  datasetId: string;
  datasetName: string;
  rowCount: number;
}) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="flex flex-col gap-3">
      <button
        type="button"
        aria-expanded={expanded}
        aria-controls="custom-chart-form"
        onClick={() => setExpanded((current) => !current)}
        className="w-full rounded-base border border-border px-3 py-2 text-sm font-medium hover:bg-surface"
      >
        {expanded ? "Close builder" : "Build a custom chart"}
      </button>
      {expanded && (
        <div id="custom-chart-form" className="overflow-hidden rounded-base border border-border bg-bg">
          <CustomChartForm
            sessionId={sessionId}
            projectId={projectId}
            datasetId={datasetId}
            datasetName={datasetName}
            rowCount={rowCount}
          />
        </div>
      )}
    </div>
  );
}
