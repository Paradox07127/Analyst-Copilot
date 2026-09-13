/* Page-to-page pre-fill only, never the source of truth. The goal typed on the
 * new-session screen is carried here until the Explore form persists it through
 * prepare/start; the run id is a fallback while the server-side listing
 * (GET /sessions/{id}/explorations) loads. Losing either is harmless. */

const GOAL_KEY_PREFIX = "eda.exploration.goal.v1.";
const RUN_KEY_PREFIX = "eda.exploration.run.v1.";

function read(key: string): string {
  try {
    return window.localStorage.getItem(key)?.trim() ?? "";
  } catch {
    return "";
  }
}

function write(key: string, value: string): void {
  try {
    if (value) window.localStorage.setItem(key, value);
    else window.localStorage.removeItem(key);
  } catch {
    // A carried-over goal is a convenience; Explore still accepts a typed one.
  }
}

export function readExplorationGoal(sessionId: string): string {
  return sessionId ? read(`${GOAL_KEY_PREFIX}${sessionId}`) : "";
}

export function writeExplorationGoal(sessionId: string, goal: string): void {
  if (sessionId) write(`${GOAL_KEY_PREFIX}${sessionId}`, goal.trim());
}

export function readLastExplorationId(sessionId: string): string {
  return sessionId ? read(`${RUN_KEY_PREFIX}${sessionId}`) : "";
}

export function writeLastExplorationId(
  sessionId: string,
  explorationId: string,
): void {
  if (sessionId) write(`${RUN_KEY_PREFIX}${sessionId}`, explorationId);
}
