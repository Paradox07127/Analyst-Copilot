/* Before-cleaning snapshot: raw profiles, raw charts, and raw data previews.
 * Raw pre-cleaning profiles (profiles/
 * charts/previews part only — the cleaning log lives in CleaningLogSection). */

import { useMemo } from "react";
import type {
  CleaningRawView,
  DatasetProfileSummary,
  RawChartView,
  RawDataPreviewView,
} from "../../api/client";
import { EmptyState } from "../../components/async-states";
import { Card, Disclosure, Marquee, SectionHeader } from "../../components/ui";
import { VegaChart } from "../insights/VegaChart";
import { SimpleTable } from "./SimpleTable";

function formatPercent(value: number | null | undefined): string {
  return value == null ? "—" : `${value.toFixed(1)}%`;
}

/* insights/ProfilesPage.tsx's ProfileCard isn't exported, so this reproduces
 * its column layout locally for the raw (pre-cleaning) profile cards. The
 * field table is collapsed: it is one row per column, and this whole section
 * is background evidence sitting under the recipe the page is actually for. */
function RawProfileCard({ profile }: { profile: DatasetProfileSummary }) {
  const fields = profile.fields ?? [];
  return (
    <Card as="article" className="flex flex-col gap-2 p-4">
      <header className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="min-w-0 text-base font-semibold"><Marquee>{profile.name}</Marquee></h3>
        <span className="tabular text-xs text-status-neutral">
          {profile.rows.toLocaleString()} rows · {profile.columns} cols
        </span>
      </header>
      <Disclosure summary={`Columns (${fields.length})`} meta={profile.dataset_id}>
        <div className="overflow-x-auto rounded-base border border-border">
          <table
            aria-label={`Raw profile: ${profile.name}`}
            className="w-full text-sm"
          >
            <thead className="bg-table-header-bg text-left">
              <tr>
                <th scope="col" className="px-3 py-2 font-medium">
                  Column
                </th>
                <th scope="col" className="px-3 py-2 font-medium">
                  Dtype
                </th>
                <th scope="col" className="px-3 py-2 font-medium">
                  Semantic
                </th>
                <th scope="col" className="px-3 py-2 text-right font-medium">
                  Missing
                </th>
                <th scope="col" className="px-3 py-2 text-right font-medium">
                  Unique
                </th>
                <th scope="col" className="px-3 py-2 font-medium">
                  Samples
                </th>
              </tr>
            </thead>
            <tbody>
              {fields.map((field) => (
                <tr key={field.column} className="border-t border-table-border">
                  <th
                    scope="row"
                    className="px-3 py-2 text-left font-mono text-xs font-normal"
                  >
                    {field.column}
                  </th>
                  <td className="px-3 py-2 font-mono text-xs">{field.dtype}</td>
                  <td className="px-3 py-2">{field.semantic_type}</td>
                  <td className="tabular px-3 py-2 text-right">
                    {formatPercent(field.missing_percent)}
                  </td>
                  <td className="tabular px-3 py-2 text-right">
                    {formatPercent(field.unique_percent)}
                  </td>
                  <td className="max-w-40 px-3 py-2 text-status-neutral">
                    <Marquee title={field.sample_values}>
                      {field.sample_values}
                    </Marquee>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Disclosure>
    </Card>
  );
}

/* The title lives in the card header, so it's dropped from the spec before
 * handing it to VegaChart (same rule as ProfilesPage's chartSpecWithoutTitle). */
function RawChartCard({ chart }: { chart: RawChartView }) {
  const spec = useMemo(() => {
    const { title: _title, ...rest } = chart.spec ?? {};
    return rest;
  }, [chart.spec]);
  return (
    <Card as="article" className="flex flex-col gap-1.5 p-4">
      <header>
        <h3 className="min-w-0 text-sm font-semibold"><Marquee>{chart.title}</Marquee></h3>
        <p className="min-w-0 text-xs text-status-neutral">
          <Marquee>{chart.dataset_name}</Marquee>
        </p>
      </header>
      {(chart.plain_language ?? chart.description) && (
        <p className="text-xs text-status-neutral">
          {chart.plain_language ?? chart.description}
        </p>
      )}
      <VegaChart spec={spec} label={chart.title} />
    </Card>
  );
}

function RawPreviewCard({ preview }: { preview: RawDataPreviewView }) {
  const columnNames = preview.column_names ?? [];
  const rows = preview.rows_preview ?? [];
  return (
    <Card as="article" className="flex flex-col gap-2 p-4">
      <header className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="min-w-0 text-base font-semibold"><Marquee>{preview.name}</Marquee></h3>
        <span className="tabular text-xs text-status-neutral">
          {preview.rows.toLocaleString()} rows · {preview.columns} columns
          before cleaning
        </span>
      </header>
      {rows.length > 0 ? (
        <SimpleTable
          ariaLabel={`Raw data preview: ${preview.name}`}
          columns={columnNames.map((name) => ({ key: name, label: name }))}
          rows={rows}
        />
      ) : (
        <p className="text-sm text-status-neutral">No preview rows.</p>
      )}
    </Card>
  );
}

export function RawPreCleaningSection({ raw }: { raw: CleaningRawView }) {
  const profiles = raw.profiles ?? [];
  const charts = raw.charts ?? [];
  const previews = raw.previews ?? [];

  return (
    <>
      <section className="flex flex-col gap-3">
        <SectionHeader
          level={2}
          title="Raw profiles"
          description="Column-by-column shape of each table as it arrived, before any automatic cleaning."
        />
        {profiles.length === 0 ? (
          <EmptyState title="No raw-before-cleaning profile was recorded for this session." />
        ) : (
          profiles.map((profile) => (
            <RawProfileCard key={profile.dataset_id} profile={profile} />
          ))
        )}
      </section>

      <section className="flex flex-col gap-3">
        <SectionHeader
          level={2}
          title="Raw charts"
          description="Distributions as they looked before cleaning."
        />
        {charts.length === 0 ? (
          <EmptyState title="No raw-before-cleaning chart specs were recorded for this session." />
        ) : (
          <div className="grid gap-3 lg:grid-cols-2">
            {charts.map((chart) => (
              <RawChartCard key={chart.artifact_id} chart={chart} />
            ))}
          </div>
        )}
      </section>

      <section className="flex flex-col gap-3">
        <SectionHeader
          level={2}
          title="Raw data preview"
          description="The first rows exactly as they arrived."
        />
        {previews.length === 0 ? (
          <EmptyState title="No raw-before-cleaning data preview was recorded for this session." />
        ) : (
          previews.map((preview) => (
            <RawPreviewCard key={preview.artifact_id} preview={preview} />
          ))
        )}
      </section>
    </>
  );
}
