import type { ReactNode } from "react";
import { SectionHeader } from "./ui";

/** Shared frame for the evidence-heavy pages under Understand the data.
 *
 * The outer node is the query container. Layout decisions below this point are
 * therefore based on the pane's real width (normal, split, or Inspector-open),
 * not on the browser viewport. */
export function DataWorkspacePage({
  title,
  description,
  actions,
  children,
  fill = false,
  gap = "normal",
}: {
  title: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  fill?: boolean;
  gap?: "compact" | "normal";
}) {
  return (
    <div
      className={`@container/data-page mx-auto w-[95%] max-w-data min-w-0 ${
        fill ? "h-full" : ""
      }`}
    >
      <div
        className={`flex min-w-0 flex-col px-3 pt-4 @2xl/data-page:px-4 @5xl/data-page:px-6 @5xl/data-page:pt-6 ${
          /* Top padding is the same responsive scale on every page under this
           * frame. fill only changes the bottom: a virtualized table needs a
           * bounded, non-scrolling page and a tight pb-2 instead of matching
           * the top's pb-6. Splitting the old py- shorthand into separate
           * top/bottom utilities kept fill's bottom compact without also
           * compacting the top, which had made Table Preview sit 8px higher
           * than Data Map, Quality, Profiles, Cleanup, Relationships and
           * Knowledge. */
          fill ? "h-full overflow-hidden pb-2" : "pb-4 @5xl/data-page:pb-6"
        } ${gap === "compact" ? "gap-2" : "gap-4"}`}
      >
        <SectionHeader
          level={1}
          title={title}
          description={description}
          actions={actions}
        />
        {children}
      </div>
    </div>
  );
}

export interface DatasetScopeOption {
  value: string;
  label: string;
}

/** One dataset selector contract across Preview, Quality, Profiles and Cleanup.
 * Page-specific filters follow in `children`, but dataset scope always comes
 * first and always stores a dataset id in the URL/state. */
export function DatasetScopeBar({
  value,
  onChange,
  options,
  allLabel,
  children,
  label = "Dataset",
}: {
  value: string;
  onChange: (value: string) => void;
  options: DatasetScopeOption[];
  allLabel?: string;
  children?: ReactNode;
  label?: string;
}) {
  return (
    <div className="flex min-w-0 flex-wrap items-end gap-x-4 gap-y-3 border-b border-hairline pb-3">
      <label className="flex min-w-[15rem] flex-1 flex-col gap-1 @3xl/data-page:max-w-xl">
        <span className="text-sm text-status-neutral">{label}</span>
        <select
          aria-label={label}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          className="min-w-0 rounded-base border border-border bg-bg px-3 py-2 text-sm font-medium"
        >
          {allLabel !== undefined && <option value="">{allLabel}</option>}
          {options.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </label>
      {children}
    </div>
  );
}

export interface SegmentOption {
  value: string;
  label: ReactNode;
  disabled?: boolean;
}

/** Immediate, mutually exclusive view/filter control. This is intentionally a
 * pressed-button group rather than an incomplete ARIA tab pattern. */
export function SegmentedControl({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: SegmentOption[];
  onChange: (value: string) => void;
}) {
  return (
    <div
      role="group"
      aria-label={label}
      className="inline-flex max-w-full overflow-x-auto rounded-base border border-border p-0.5"
    >
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          aria-pressed={value === option.value}
          disabled={option.disabled}
          onClick={() => onChange(option.value)}
          className={`shrink-0 rounded-sm px-2.5 py-1.5 text-sm font-medium disabled:opacity-40 ${
            value === option.value
              ? "bg-surface text-text shadow-sm"
              : "text-status-neutral hover:bg-surface/60 hover:text-text"
          }`}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}
