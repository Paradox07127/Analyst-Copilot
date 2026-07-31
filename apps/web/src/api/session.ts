/* Session id for server-side Settings (§6.0). It is an opaque correlation key,
 * not a secret and not a credential — the API key it unlocks stays on the
 * server. sessionStorage on purpose: a closed tab drops the key server-side on
 * TTL expiry, and nothing about the LLM config ever lands in localStorage. */

export const SESSION_HEADER = "X-EDA-Session";

const STORAGE_KEY = "eda.session";

let cached: string | null = null;

function newSessionId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `s_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;
}

export function sessionId(): string {
  if (cached) return cached;
  let stored: string | null = null;
  try {
    stored = window.sessionStorage.getItem(STORAGE_KEY);
  } catch {
    /* storage disabled (private mode, sandboxed iframe): fall back to memory */
  }
  cached = stored ?? newSessionId();
  if (!stored) {
    try {
      window.sessionStorage.setItem(STORAGE_KEY, cached);
    } catch {
      /* same as above — the in-memory value still keeps one tab consistent */
    }
  }
  return cached;
}
