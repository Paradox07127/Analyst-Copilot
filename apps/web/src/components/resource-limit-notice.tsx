/* A run the resource gate stopped, explained in the terms the user can act on.
 *
 * "Limited" is not "failed": the analysis never started, nothing broke, and
 * every way out is a choice the user can make. The tone is therefore a caution,
 * not an alert, and the card always ends with something to do. */

import { Link } from "react-router";
import type { ResourceLimitGuidance } from "../api/client";
import { Card } from "./ui";

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 MB";
  if (bytes < 1024 ** 2) return `${Math.round(bytes / 1024)} KB`;
  if (bytes < 1024 ** 3) return `${Math.round(bytes / 1024 ** 2)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
}

function formatRows(rows: number): string {
  return rows.toLocaleString("en-US");
}

const OVER_MEMORY = "estimated_working_set_exceeded";
const OVER_ROWS = "row_count_exceeded";

/** What was measured against what, in one sentence per binding limit. */
function reasonSentences(limit: ResourceLimitGuidance): string[] {
  const codes = limit.reason_codes ?? [];
  const sentences: string[] = [];
  if (codes.includes(OVER_MEMORY)) {
    sentences.push(
      `These ${limit.dataset_count} tables need about ${formatBytes(limit.estimated_working_set_bytes)} of memory to analyse together, ` +
        `and one analysis is allowed ${formatBytes(limit.max_working_set_bytes)} — ${formatBytes(limit.over_budget_bytes)} over.`,
    );
    if (limit.largest_dataset_name) {
      sentences.push(
        `${limit.largest_dataset_name} is the largest table at about ${formatBytes(limit.largest_dataset_bytes)} once loaded.`,
      );
    }
  }
  if (codes.includes(OVER_ROWS) && limit.largest_dataset_name) {
    sentences.push(
      `${limit.largest_dataset_name} has ${formatRows(limit.largest_dataset_rows)} rows, ` +
        `and one table is allowed ${formatRows(limit.max_rows_per_dataset)}.`,
    );
  }
  if (sentences.length === 0) {
    sentences.push(
      `This data is larger than one analysis is currently allowed to hold.`,
    );
  }
  return sentences;
}

/** Only what would actually work for this stop, never a generic checklist. */
function waysOut(limit: ResourceLimitGuidance): string[] {
  const codes = limit.reason_codes ?? [];
  const ways: string[] = [];
  if (limit.disabling_precleaning_would_fit) {
    ways.push(
      "Start the analysis again with “Clean the data first” switched off — these tables fit without it.",
    );
  }
  if (codes.includes(OVER_MEMORY)) {
    ways.push(
      `Raise the memory limit to ${formatBytes(limit.suggested_max_working_set_bytes)} in Settings, then start the analysis again. ` +
        "Only go past what this machine actually has free — the analysis will run out of memory for real, not stop politely.",
    );
  }
  if (codes.includes(OVER_ROWS)) {
    ways.push(
      `Raise the largest-table limit past ${formatRows(limit.largest_dataset_rows)} rows in Settings, or analyse a smaller extract of that table.`,
    );
  }
  ways.push("Or run fewer tables at a time and analyse the rest separately.");
  return ways;
}

export function ResourceLimitNotice({
  limit,
  className = "",
}: {
  limit: ResourceLimitGuidance;
  className?: string;
}) {
  return (
    <Card
      role="status"
      aria-label="Why this analysis stopped"
      className={`flex flex-col gap-2 border-status-warn/40 p-4 ${className}`}
    >
      <p className="text-sm font-semibold text-status-warn">
        This analysis stopped before it read the data
      </p>
      {reasonSentences(limit).map((sentence) => (
        <p key={sentence} className="text-sm text-status-neutral">
          {sentence}
        </p>
      ))}
      <ul className="flex list-disc flex-col gap-1 pl-5 text-sm">
        {waysOut(limit).map((way) => (
          <li key={way}>{way}</li>
        ))}
      </ul>
      <Link
        to="/settings?section=analysis"
        className="self-start text-sm font-medium text-primary hover:underline"
      >
        Open analysis settings
      </Link>
    </Card>
  );
}

/* The same explanation in the job strip, where a card would not fit. The
 * numbers stay; the ways out are one line so the strip keeps its shape. */
export function ResourceLimitLine({ limit }: { limit: ResourceLimitGuidance }) {
  const [first] = reasonSentences(limit);
  return (
    <div role="status" className="flex flex-col gap-1 text-xs leading-5">
      <p className="text-status-warn">{first}</p>
      <p className="text-status-neutral">{waysOut(limit)[0]}</p>
    </div>
  );
}
