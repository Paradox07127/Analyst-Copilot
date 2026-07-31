import { api, ApiError, type ClientFailureRequest } from "./client";

type FailureCode = ClientFailureRequest["error_code"];
export type FailureOperation = ClientFailureRequest["operation"];

const SESSION_ID = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$/;
const keys = new WeakMap<object, string>();
const emitted = new Set<string>();

function runFromLocation(): string | null {
  const match = window.location.pathname.match(/\/sessions\/([^/]+)/);
  if (!match?.[1]) return null;
  try {
    const sessionId = decodeURIComponent(match[1]);
    return SESSION_ID.test(sessionId) ? sessionId : null;
  } catch {
    return null;
  }
}

function failureCode(error: unknown): FailureCode {
  if (!(error instanceof ApiError)) {
    return error instanceof TypeError ? "network_error" : "client_error";
  }
  if (error.status === 403) return "access_forbidden";
  if (error.status === 404) return "not_found";
  if (error.status === 409) return "conflict";
  if (error.status === 422) return "validation_error";
  if (error.status === 429) return "rate_limited";
  if (error.status >= 500) return "server_error";
  return "http_error";
}

function dedupeKey(error: unknown): string {
  if (typeof error === "object" && error !== null) {
    const existing = keys.get(error);
    if (existing) return existing;
    const created = crypto.randomUUID();
    keys.set(error, created);
    return created;
  }
  return crypto.randomUUID();
}

/** Best-effort handled-error telemetry. The request type has no message,
 * stack, URL, request body, or arbitrary context field by design. */
export function reportHandledClientFailure(
  error: unknown,
  operation: FailureOperation,
  explicitSessionId?: string,
): void {
  try {
    const sessionId = explicitSessionId && SESSION_ID.test(explicitSessionId)
      ? explicitSessionId
      : runFromLocation();
    if (!sessionId) return;
    const dedupe_key = dedupeKey(error);
    if (emitted.has(dedupe_key)) return;
    emitted.add(dedupe_key);
    void api.recordClientFailure(sessionId, {
      error_code: failureCode(error),
      operation,
      dedupe_key,
    }).catch(() => {
      // Observability must never replace or obscure the handled product error.
    });
  } catch {
    // Synchronous browser/API failures are best-effort for the same reason.
  }
}

export function resetClientFailureReporterForTests(): void {
  emitted.clear();
}
