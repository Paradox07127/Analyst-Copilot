/* Sessions with activity per UTC day, newest column last. Home switches this
 * between seven, thirty and one hundred eighty days. */

import type { UsageDay } from "../../api/client";

const WEEKDAYS = 7;

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
  return (
    <div className="flex flex-col gap-2">
      <div
        data-activity-grid
        data-activity-layout="fixed-180"
        role="img"
        aria-label={`Active sessions per day over the last ${days.length} days; busiest day had ${busiest}`}
        className="grid w-full grid-flow-col auto-cols-fr gap-[3px] pb-1"
      >
        {weeks.map((week) => (
          <div key={week[0]?.date} className="flex min-w-0 flex-col gap-[3px]">
            {week.map((day) => (
              <span
                key={day.date}
                title={label(day)}
                className={`aspect-square w-full rounded-[2px] ${levelClass(day.sessions, busiest)}`}
              />
            ))}
          </div>
        ))}
      </div>
      <div className="flex items-center gap-1.5 text-xs text-status-neutral">
        <span>Less</span>
        <span className="size-2 rounded-[2px] bg-track" />
        <span className="size-2 rounded-[2px] bg-primary/30" />
        <span className="size-2 rounded-[2px] bg-primary/60" />
        <span className="size-2 rounded-[2px] bg-primary" />
        <span>More</span>
      </div>
    </div>
  );
}
