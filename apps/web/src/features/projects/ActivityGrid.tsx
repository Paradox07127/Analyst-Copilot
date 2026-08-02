/* Sessions with activity per UTC day, newest column last. Home switches this
 * between seven, thirty and one hundred eighty days. */

import type { CSSProperties } from "react";
import type { UsageDay } from "../../api/client";

const WEEKDAYS = 7;
const GRID_GAP_PX = 4;
const GRID_TARGET_WIDTH_PX = 412;
const MIN_CELL_PX = 12;
const MAX_CELL_PX = 20;

/* Four steps, not a continuous ramp: with a handful of runs a day, a smooth
 * scale makes 1 and 2 indistinguishable. */
function levelClass(sessions: number, busiest: number): string {
  if (sessions === 0) return "bg-track";
  if (busiest <= 1) return "bg-primary";
  const ratio = sessions / busiest;
  if (ratio > 0.66) return "bg-primary";
  if (ratio > 0.33) return "bg-primary/60";
  return "bg-primary/30";
}

/* timeZone: "UTC" is load-bearing. The buckets are UTC calendar days; rendering
 * one in the browser's zone labels 2026-07-28 as "Jul 27" anywhere west of
 * UTC, so the cell would name a different day than the one it counts. */
function label(day: UsageDay): string {
  const when = new Date(`${day.date}T00:00:00Z`).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    timeZone: "UTC",
  });
  return `${day.sessions} active session${day.sessions === 1 ? "" : "s"} on ${when}`;
}

export function ActivityGrid({ days }: { days: UsageDay[] }) {
  const busiest = days.reduce((max, day) => Math.max(max, day.sessions), 0);
  /* Column-major so each column is a week and reading left to right is
   * reading forward in time. */
  const weeks: UsageDay[][] = [];
  for (let index = 0; index < days.length; index += WEEKDAYS) {
    weeks.push(days.slice(index, index + WEEKDAYS));
  }
  const isSingleWeek = days.length <= WEEKDAYS;
  /* Keep the selected range honest — seven days should not render 173 blank
   * dates — while allocating a comfortable tile size for short windows. Once
   * the view grows, cells shrink just enough to keep the half-year grid within
   * the same visual measure rather than turning 30d into five giant columns. */
  const weekCount = Math.max(1, weeks.length);
  const cellSize = Math.max(
    MIN_CELL_PX,
    Math.min(
      MAX_CELL_PX,
      Math.floor(
        (GRID_TARGET_WIDTH_PX - (weekCount - 1) * GRID_GAP_PX) / weekCount,
      ),
    ),
  );
  const gridStyle = {
    "--activity-cell-size": `${cellSize}px`,
  } as CSSProperties;
  return (
    <div className="flex flex-col gap-2">
      <div
        data-activity-grid
        data-activity-layout="selected-window"
        data-activity-window={days.length}
        data-activity-cell-size={cellSize}
        role="img"
        aria-label={`Active sessions per day over the last ${days.length} days; busiest day had ${busiest}`}
        style={gridStyle}
        className={`grid w-fit gap-1 pb-1 ${
          isSingleWeek
            ? "grid-cols-[repeat(7,var(--activity-cell-size))]"
            : "grid-flow-col grid-rows-7 auto-cols-[var(--activity-cell-size)]"
        }`}
      >
        {isSingleWeek
          ? days.map((day) => (
              <span
                key={day.date}
                title={label(day)}
                className={`size-[var(--activity-cell-size)] rounded-[2px] ${levelClass(day.sessions, busiest)}`}
              />
            ))
          : weeks.flatMap((week) =>
              week.map((day) => (
                <span
                  key={day.date}
                  title={label(day)}
                  className={`size-[var(--activity-cell-size)] rounded-[2px] ${levelClass(day.sessions, busiest)}`}
                />
              )),
            )}
      </div>
    </div>
  );
}
