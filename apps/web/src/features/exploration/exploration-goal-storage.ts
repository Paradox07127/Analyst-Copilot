/* The deep-dive goal is typed on the new-session screen and consumed on the
 * Explore screen, and the server cannot carry it between them: `business_context`
 * is a request-only field, and no endpoint lists a session's explorations. The
 * started run id is kept for the same reason — without it a run in flight is
 * only reachable through browser history. */

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
