/* Session settings live in one server process's memory, so a rebuild drops the
 * selection back to the env defaults and the provider and model have to be
 * picked again by hand. The tab usually outlives the restart, so remembering
 * the selection for the tab's lifetime is enough to stop the re-picking.
 *
 * sessionStorage, alongside the session id in `api/session.ts` and for its
 * reason: nothing about the LLM config lands in localStorage. A closed tab
 * therefore forgets, which is the cost of not leaving the chosen provider,
 * model and endpoint on a shared machine.
 *
 * The API key is not here and could not be. It never reaches the browser: the
 * service keeps it in-process and `SettingsView` carries only whether one is
 * set and its last 4 characters. Switching provider re-seeds it from the
 * server's own env file, so remembering the selection is enough. */

import type { SettingsPatch, SettingsView } from "../../api/client";

const KEY = "eda.settings.selection.v1";
/* localStorage, unlike the selection above, and for the same reason stated
 * there in reverse: a memory ceiling is a property of the machine, not a
 * credential. It carries nothing private, and re-typing it after every restart
 * is exactly the friction that made a stopped run look unfixable. */
const RESOURCE_LIMITS_KEY = "eda.settings.resource-limits.v1";

export type SettingsSelection = {
  provider: string;
  model: string;
  report_model: string;
  base_url: string;
  analysis_depth: number;
  payload_policy: string;
};

function isSelection(value: unknown): value is SettingsSelection {
  if (typeof value !== "object" || value === null) return false;
  const item = value as Record<string, unknown>;
  return (
    typeof item["provider"] === "string" &&
    item["provider"].length > 0 &&
    typeof item["model"] === "string" &&
    typeof item["report_model"] === "string" &&
    typeof item["base_url"] === "string" &&
    typeof item["analysis_depth"] === "number" &&
    typeof item["payload_policy"] === "string"
  );
}

export function readSettingsSelection(): SettingsSelection | null {
  try {
    const raw = window.sessionStorage.getItem(KEY);
    if (!raw) return null;
    const parsed: unknown = JSON.parse(raw);
    return isSelection(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

export function writeSettingsSelection(view: SettingsView): void {
  try {
    const selection: SettingsSelection = {
      provider: view.provider,
      model: view.model,
      report_model: view.report_model ?? "",
      base_url: view.base_url ?? "",
      analysis_depth: view.analysis_depth ?? 0,
      payload_policy: view.payload_policy ?? "",
    };
    window.sessionStorage.setItem(KEY, JSON.stringify(selection));
  } catch {
    // Remembering the selection is a convenience; Settings still accepts one.
  }
}

export function clearSettingsSelection(): void {
  try {
    window.sessionStorage.removeItem(KEY);
  } catch {
    // As above.
  }
}

export type ResourceLimits = {
  max_working_set_bytes: number;
  max_rows_per_dataset: number;
};

function isPositiveInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value) && value > 0;
}

export function readResourceLimits(): ResourceLimits | null {
  try {
    const raw = window.localStorage.getItem(RESOURCE_LIMITS_KEY);
    if (!raw) return null;
    const parsed: unknown = JSON.parse(raw) as unknown;
    if (typeof parsed !== "object" || parsed === null) return null;
    const item = parsed as Record<string, unknown>;
    if (
      !isPositiveInteger(item["max_working_set_bytes"]) ||
      !isPositiveInteger(item["max_rows_per_dataset"])
    ) {
      return null;
    }
    return {
      max_working_set_bytes: item["max_working_set_bytes"],
      max_rows_per_dataset: item["max_rows_per_dataset"],
    };
  } catch {
    return null;
  }
}

export function writeResourceLimits(view: SettingsView): void {
  try {
    const limits: ResourceLimits = {
      max_working_set_bytes: view.max_working_set_bytes,
      max_rows_per_dataset: view.max_rows_per_dataset,
    };
    window.localStorage.setItem(RESOURCE_LIMITS_KEY, JSON.stringify(limits));
  } catch {
    // As above.
  }
}

export function clearResourceLimits(): void {
  try {
    window.localStorage.removeItem(RESOURCE_LIMITS_KEY);
  } catch {
    // As above.
  }
}

/** The patch that would restore what this browser remembers, or null when
 * nothing differs.
 *
 * Sending the whole selection in one patch matters: the server re-seeds model,
 * base_url and the API key whenever the provider changes, and the rest of the
 * patch is applied on top of that fresh state. The two stores are independent —
 * remembered limits must be restored even in a tab that never picked a model.
 */
export function restorePatch(
  view: SettingsView,
  selection: SettingsSelection | null,
  limits: ResourceLimits | null = null,
): SettingsPatch | null {
  const sameSelection =
    selection === null ||
    (view.provider === selection.provider &&
      view.model === selection.model &&
      (view.report_model ?? "") === selection.report_model &&
      (view.base_url ?? "") === selection.base_url &&
      (view.analysis_depth ?? 0) === selection.analysis_depth &&
      (view.payload_policy ?? "") === selection.payload_policy);
  const sameLimits =
    limits === null ||
    (view.max_working_set_bytes === limits.max_working_set_bytes &&
      view.max_rows_per_dataset === limits.max_rows_per_dataset);
  if (sameSelection && sameLimits) return null;
  return {
    ...(selection === null
      ? {}
      : {
          provider: selection.provider,
          model: selection.model,
          report_model: selection.report_model,
          base_url: selection.base_url,
          analysis_depth: selection.analysis_depth,
          payload_policy: selection.payload_policy,
        }),
    ...(limits === null ? {} : limits),
  };
}
