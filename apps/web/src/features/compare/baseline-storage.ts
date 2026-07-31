const BASELINE_KEY_PREFIX = "eda.compare.baseline.v1.";

export function baselineStorageKey(projectId: string): string {
  return `${BASELINE_KEY_PREFIX}${projectId}`;
}

export function readBaseline(projectId: string): string {
  if (!projectId) return "";
  try {
    return window.localStorage.getItem(baselineStorageKey(projectId))?.trim() ?? "";
  } catch {
    return "";
  }
}

export function writeBaseline(projectId: string, sessionId: string): void {
  if (!projectId || !sessionId) return;
  try {
    window.localStorage.setItem(baselineStorageKey(projectId), sessionId);
  } catch {
    // Baseline persistence is a preference; Compare remains usable without it.
  }
}

export function clearBaseline(projectId: string): void {
  if (!projectId) return;
  try {
    window.localStorage.removeItem(baselineStorageKey(projectId));
  } catch {
    // Baseline persistence is a preference; Compare remains usable without it.
  }
}

