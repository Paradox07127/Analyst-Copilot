/* A session can exist without a project. It is still stored under a project id
 * so the project-scoped filesystem, quotas and APIs keep working unchanged —
 * that bucket is an implementation detail, never a project the user manages.
 *
 * The id is declared once per language: here and in
 * eda_platform/src/eda_platform/core/ids.py (INTERNAL_PROJECT_IDS).
 * src/test/unfiled.test.ts reads that file and fails if the two drift. */
export const UNFILED_PROJECT_ID = "unfiled-sessions";

/** Heading over these sessions in the rail. Deliberately not a project name:
 *  they sit under it because they belong to nothing, not because they belong
 *  to a folder called this. */
export const UNFILED_RAIL_HEADING = "Recent";

/** Used where a *project name* would go — a session row on Home, a search
 *  result. "Recent" would read as a project there. */
export const UNFILED_ROW_LABEL = "No project";

export function isUnfiled(projectId: string): boolean {
  return projectId === UNFILED_PROJECT_ID;
}

/** Project label for a session row, collapsing the storage bucket to the
 *  user-facing name. */
export function projectLabel(projectId: string): string {
  return isUnfiled(projectId) ? UNFILED_ROW_LABEL : projectId;
}
