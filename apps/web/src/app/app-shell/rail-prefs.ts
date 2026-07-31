/* Session-rail layout preferences, stored the way theme.ts stores appearance:
 * a validated synchronous localStorage read, so the very first render already
 * has the stored state and a collapsed rail never flashes open first. */

const RAIL_KEY = "eda.layout.rail-collapsed";
const PROJECTS_KEY = "eda.layout.rail-collapsed-projects";

export function getRailCollapsed(): boolean {
  return window.localStorage.getItem(RAIL_KEY) === "true";
}

export function setRailCollapsed(collapsed: boolean): void {
  window.localStorage.setItem(RAIL_KEY, String(collapsed));
}

export function getCollapsedProjects(): Set<string> {
  try {
    const raw = window.localStorage.getItem(PROJECTS_KEY);
    const parsed: unknown = raw ? JSON.parse(raw) : [];
    if (!Array.isArray(parsed)) return new Set();
    return new Set(parsed.filter((id): id is string => typeof id === "string"));
  } catch {
    return new Set();
  }
}

export function setCollapsedProjects(ids: ReadonlySet<string>): void {
  window.localStorage.setItem(PROJECTS_KEY, JSON.stringify([...ids]));
}
