/* CSV downloads are assembled client-side from data already on screen — the
 * exports say so in their labels; nothing here re-queries the server. */

import { saveBlob } from "../api/client";

function escapeCsvCell(value: string): string {
  return /[",\r\n]/.test(value) ? `"${value.replace(/"/g, '""')}"` : value;
}

export function csvContent(columns: string[], rows: string[][]): string {
  return [columns, ...rows]
    .map((row) => row.map(escapeCsvCell).join(","))
    .join("\r\n");
}

export function saveCsv(
  filename: string,
  columns: string[],
  rows: string[][],
): void {
  saveBlob(
    new Blob([csvContent(columns, rows)], { type: "text/csv;charset=utf-8" }),
    filename,
  );
}
