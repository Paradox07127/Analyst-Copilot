/* A rebuild restarts the API, whose session settings are a dict in memory, so
 * the provider and model fell back to the server env and had to be picked by
 * hand again. The selection is remembered for the tab's lifetime; the API key
 * is not, and cannot be — it never reaches the browser. */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { server } from "./msw/server";
import { defaultSettings } from "./msw/handlers";
import { renderAppAt } from "./render";
import { resetSettingsRestoreForTest } from "../api/hooks";
import {
  readSettingsSelection,
  restorePatch,
  writeSettingsSelection,
} from "../features/settings/settings-preference-storage";
import type { SettingsView } from "../api/client";

const STORAGE_KEY = "eda.settings.selection.v1";

function storeSelection(overrides: Partial<Record<string, unknown>> = {}) {
  window.sessionStorage.setItem(
    STORAGE_KEY,
    JSON.stringify({
      provider: "deepseek",
      model: "deepseek-v4-flash",
      report_model: "deepseek-v4",
      base_url: "https://api.deepseek.com",
      analysis_depth: 1,
      payload_policy: "schema+aggregates",
      ...overrides,
    }),
  );
}

/** Serve settings that look like a freshly restarted server: env defaults. */
function serveEnvDefaults(overrides: Record<string, unknown> = {}) {
  server.use(
    http.get("/api/v1/settings", () =>
      HttpResponse.json({
        ...defaultSettings(),
        provider: "offline",
        model: "offline-deterministic",
        source: "env",
        ...overrides,
      }),
    ),
  );
}

beforeEach(() => {
  resetSettingsRestoreForTest();
  window.sessionStorage.clear();
});

afterEach(() => {
  window.sessionStorage.clear();
});

describe("the remembered selection", () => {
  it("is re-applied when the server comes back on env defaults", async () => {
    const patches: unknown[] = [];
    serveEnvDefaults();
    server.use(
      http.put("/api/v1/settings", async ({ request }) => {
        const body = await request.json();
        patches.push(body);
        return HttpResponse.json({
          ...defaultSettings(),
          provider: "deepseek",
          model: "deepseek-v4-flash",
          source: "session",
        });
      }),
    );
    storeSelection();

    renderAppAt("/settings");

    await waitFor(() => expect(patches).toHaveLength(1));
    expect(patches[0]).toMatchObject({
      provider: "deepseek",
      model: "deepseek-v4-flash",
      report_model: "deepseek-v4",
    });
  });

  it("is not re-applied over a selection made this session", async () => {
    /* `source: "session"` means someone already chose deliberately; replacing
     * that with a remembered one would undo a click the user just made. */
    const patches: unknown[] = [];
    serveEnvDefaults({ source: "session", provider: "openai" });
    server.use(
      http.put("/api/v1/settings", async ({ request }) => {
        patches.push(await request.json());
        return HttpResponse.json(defaultSettings());
      }),
    );
    storeSelection();

    renderAppAt("/settings");

    await screen.findByRole("heading", { name: /settings/i });
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(patches).toHaveLength(0);
  });

  it("is dropped when the server refuses it", async () => {
    /* A remembered provider can stop being usable — a key removed from the
     * server env, a model retired. Re-sending it on every load would be a
     * failure the user cannot clear. */
    serveEnvDefaults();
    server.use(
      http.put("/api/v1/settings", () =>
        HttpResponse.json({ detail: "unknown model" }, { status: 422 }),
      ),
    );
    storeSelection();

    renderAppAt("/settings");

    await waitFor(() =>
      expect(window.sessionStorage.getItem(STORAGE_KEY)).toBeNull(),
    );
  });

  it("never holds an API key", () => {
    const view = {
      ...defaultSettings(),
      api_key_set: true,
      api_key_last4: "9911",
    } as unknown as SettingsView;
    writeSettingsSelection(view);
    const raw = window.sessionStorage.getItem(STORAGE_KEY) ?? "";
    expect(raw).not.toContain("9911");
    expect(raw).not.toContain("api_key");
    expect(Object.keys(readSettingsSelection() ?? {})).toEqual([
      "provider",
      "model",
      "report_model",
      "base_url",
      "analysis_depth",
      "payload_policy",
    ]);
  });

  it("sends nothing when the server already matches", () => {
    const view = {
      ...defaultSettings(),
      provider: "deepseek",
      model: "deepseek-v4-flash",
      report_model: "deepseek-v4",
      base_url: "https://api.deepseek.com",
      analysis_depth: 1,
      payload_policy: "schema+aggregates",
    } as unknown as SettingsView;
    storeSelection();
    expect(restorePatch(view, readSettingsSelection()!)).toBeNull();
  });

  it("ignores a malformed stored value", () => {
    window.sessionStorage.setItem(STORAGE_KEY, "{not json");
    expect(readSettingsSelection()).toBeNull();
    window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify({ provider: 7 }));
    expect(readSettingsSelection()).toBeNull();
  });
});

describe("sessionStorage being unavailable", () => {
  it("does not break the app", () => {
    const spy = vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("denied");
    });
    expect(readSettingsSelection()).toBeNull();
    spy.mockRestore();
  });
});
