/* A report arrives as one uninterrupted scroll with no idea how long it is or
 * what is in it. Section chips give both in one strip, without hiding
 * anything: the report itself is unchanged below. */

import { Card, formatCompact } from "../../components/ui";
import type { ReportOutline } from "./report-outline";

export function ReportContents({
  outline,
  hasEvidence,
}: {
  outline: ReportOutline;
  hasEvidence: boolean;
}) {
  if (outline.headings.length < 2) return null;
  const sections = outline.headings.filter((heading) => heading.level === 2);

  const jump = (id: string) => {
    const target = document.getElementById(id);
    target?.scrollIntoView?.({ block: "start" });
  };

  return (
    <Card tone="quiet" className="flex flex-col gap-2 p-3">
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <h2 className="text-sm font-semibold">Contents</h2>
        <span className="tabular text-xs text-status-neutral">
          {`${sections.length} sections · ≈${formatCompact(outline.words)} words`}
        </span>
      </div>
      <nav aria-label="Report sections" className="flex flex-wrap gap-1.5">
        {outline.headings.map((heading) => (
          <button
            key={heading.id}
            type="button"
            onClick={() => jump(heading.id)}
            className={
              heading.level === 2
                ? "rounded-base border border-border px-2 py-0.5 text-xs hover:border-primary hover:text-primary"
                : "rounded-base px-2 py-0.5 text-xs text-status-neutral hover:text-primary"
            }
          >
            {heading.text}
          </button>
        ))}
      </nav>
      {hasEvidence && (
        <p className="text-xs text-status-neutral">
          Evidence ids in this report are buttons. One opens the artifact
          beside the text, so checking a figure does not cost you your place.
        </p>
      )}
    </Card>
  );
}
