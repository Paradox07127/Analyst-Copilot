/* The four cleaning-transparency tables, mirroring _render_cleaning_log in
 * Cleaning log. recipe_count (not summary.length) distinguishes
 * "no recipe at all" from "recipe ran, nothing was deleted". */

import type { ReactNode } from "react";
import type { CleaningLogView } from "../../api/client";
import { SectionHeader } from "../../components/ui";
import { SimpleTable } from "./SimpleTable";

function LogTable({
  title,
  description,
  children,
}: {
  title: string;
  description: string;
  children: ReactNode;
}) {
  return (
    <div className="flex flex-col gap-2">
      <SectionHeader level={3} title={title} description={description} />
      {children}
    </div>
  );
}

export function CleaningLogSection({ log }: { log: CleaningLogView }) {
  if (log.recipe_count === 0) {
    return (
      <p className="text-sm text-status-neutral">
        No cleaning recipe was recorded for this session.
      </p>
    );
  }

  const summaryRows = log.summary ?? [];
  const deletedRows = log.deleted_data ?? [];
  const guardrailRows = log.protection_triggers ?? [];
  const suggestionRows = log.suggestions ?? [];

  return (
    <div className="flex flex-col gap-4">
      {summaryRows.length > 0 && (
        <LogTable
          title="Summary"
          description="Shape of each table before and after the recipe ran."
        >
          <SimpleTable
            ariaLabel="Summary"
            columns={[
              { key: "dataset", label: "Dataset" },
              { key: "recipe_id", label: "Recipe" },
              { key: "rows_before", label: "Rows before" },
              { key: "rows_after", label: "Rows after" },
              { key: "rows_removed", label: "Rows removed" },
              { key: "columns_before", label: "Columns before" },
              { key: "columns_after", label: "Columns after" },
              { key: "columns_removed", label: "Columns removed" },
              { key: "delete_steps", label: "Delete steps" },
              { key: "protection_triggers", label: "Protection triggers" },
              { key: "requires_approval", label: "Requires approval" },
            ]}
            rows={summaryRows}
          />
        </LogTable>
      )}

      <LogTable
        title="Deleted data"
        description="Every row and column the recipe removed, and why."
      >
        {deletedRows.length > 0 ? (
          <SimpleTable
            ariaLabel="Deleted data"
            columns={[
              { key: "dataset", label: "Dataset" },
              { key: "operation", label: "Operation" },
              { key: "column", label: "Column" },
              { key: "rows_deleted", label: "Rows deleted" },
              { key: "columns_deleted", label: "Columns deleted" },
              { key: "reason", label: "Reason" },
              { key: "details", label: "Details" },
            ]}
            rows={deletedRows}
          />
        ) : (
          <p className="text-sm text-status-neutral">
            No rows or columns were deleted.
          </p>
        )}
      </LogTable>

      {guardrailRows.length > 0 && (
        <LogTable
          title="Protection triggers"
          description="Guardrails that stopped a step from deleting more than its threshold allows."
        >
          <SimpleTable
            ariaLabel="Protection triggers"
            columns={[
              { key: "dataset", label: "Dataset" },
              { key: "code", label: "Code" },
              { key: "reason", label: "Reason" },
              { key: "thresholds", label: "Thresholds" },
            ]}
            rows={guardrailRows}
          />
        </LogTable>
      )}

      {suggestionRows.length > 0 && (
        <LogTable
          title="Suggestions"
          description="Cleaning the profiler recommends but did not perform."
        >
          <SimpleTable
            ariaLabel="Suggestions"
            columns={[{ key: "suggestion", label: "Suggestion" }]}
            rows={suggestionRows}
          />
        </LogTable>
      )}
    </div>
  );
}
