/* Small generic table shared by the cleaning-log and raw-preview sections
 * below; styled to match the table in insights/ProfilesPage.tsx. */

function formatCell(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "number") return value.toLocaleString();
  return String(value);
}

/* Right-alignment is derived rather than declared per call site: the log
 * tables carry eleven columns each and nobody was going to keep a hand-written
 * alignment list in step with them. */
function isNumericColumn<T>(rows: T[], key: keyof T): boolean {
  let seen = false;
  for (const row of rows) {
    const value = row[key];
    if (value === null || value === undefined || value === "") continue;
    if (typeof value !== "number") return false;
    seen = true;
  }
  return seen;
}

export interface SimpleTableColumn<T> {
  key: keyof T;
  label: string;
}

export function SimpleTable<T>({
  ariaLabel,
  columns,
  rows,
}: {
  ariaLabel: string;
  columns: SimpleTableColumn<T>[];
  rows: T[];
}) {
  const numeric = new Set(
    columns.filter((col) => isNumericColumn(rows, col.key)).map((col) => col.key),
  );
  return (
    <div className="overflow-x-auto rounded-base border border-border">
      <table aria-label={ariaLabel} className="w-full text-sm">
        <thead className="bg-table-header-bg text-left">
          <tr>
            {columns.map((col) => (
              <th
                key={String(col.key)}
                scope="col"
                className={`px-3 py-2 font-medium whitespace-nowrap ${
                  numeric.has(col.key) ? "text-right" : ""
                }`}
              >
                {col.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index} className="border-t border-table-border">
              {columns.map((col) => (
                <td
                  key={String(col.key)}
                  className={`px-3 py-2 ${
                    numeric.has(col.key) ? "tabular text-right" : ""
                  }`}
                >
                  {formatCell(row[col.key])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
