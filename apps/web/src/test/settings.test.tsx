import { afterEach, describe, expect, it } from "vitest";
import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { server } from "./msw/server";
import { PROVIDERS, defaultSettings } from "./msw/handlers";
import { renderAppAt, renderAppWithRouterAt } from "./render";

const SECRET = "sk-live-supersecret-9911";

/** Serve one settings snapshot so the live/offline banner can be asserted for
 *  a state the default fixture never reaches. */
function useSettingsState(overrides: Record<string, unknown>) {
  server.use(
    http.get("/api/v1/settings", () =>
      HttpResponse.json({ ...defaultSettings(), ...overrides }),
    ),
  );
}

/* setup.ts resets data-theme between tests but not data-density (added for
 * the density preference below) — clear it locally so it cannot leak into
 * other tests in this file. */
afterEach(() => {
  delete document.documentElement.dataset["density"];
});

async function openSection(name: string) {
  const user = userEvent.setup();
  await user.click(await screen.findByRole("tab", { name }));
}

describe("Settings page", () => {
  it("renders the four sections and starts on Model & API", async () => {
    renderAppAt("/settings");
    expect(
      await screen.findByRole("heading", { name: "Settings" }),
    ).toBeInTheDocument();
    for (const section of [
      "Model & API",
      "Analysis behavior",
      "Appearance",
      "About",
    ]) {
      expect(screen.getByRole("tab", { name: section })).toBeInTheDocument();
    }
    expect(await screen.findByLabelText("Provider")).toHaveValue("offline");
  });

  it("restores the selected section from the URL and updates it on tab changes", async () => {
    const user = userEvent.setup();
    const { router } = renderAppWithRouterAt(
      "/settings?section=appearance",
    );

    expect(
      await screen.findByRole("tab", {
        name: "Appearance",
        selected: true,
      }),
    ).toBeInTheDocument();
    expect(
      await screen.findByRole("radio", { name: "Follow system" }),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: "About" }));
    await waitFor(() =>
      expect(router.state.location.search).toBe("?section=about"),
    );
  });

  it("lists every provider from the registry endpoint", async () => {
    renderAppAt("/settings");
    const select = await screen.findByLabelText("Provider");
    expect(
      Array.from(select.querySelectorAll("option")).map((o) => o.textContent),
    ).toEqual(["Offline (deterministic)", "DeepSeek"]);
  });

  it("switching provider re-seeds the model and shows its presets", async () => {
    renderAppAt("/settings");
    const select = await screen.findByLabelText("Provider");
    fireEvent.change(select, { target: { value: "deepseek" } });

    const preset = await screen.findByLabelText("Agent model");
    await waitFor(() =>
      expect(
        Array.from(preset.querySelectorAll("option")).map((o) => o.textContent),
      ).toEqual(["deepseek-v4-flash", "deepseek-v4-pro"]),
    );
    expect(screen.getByLabelText("Model")).toHaveValue("deepseek-v4-flash");
  });

  it("offers a provider that has no pre-verified model instead of disabling it", async () => {
    /* Twelve of eighteen providers were `disabled` on this count, including
     * every local one, whose model ids no catalog can enumerate. */
    server.use(
      http.get("/api/v1/settings/providers", () =>
        HttpResponse.json([
          ...PROVIDERS,
          {
            ...PROVIDERS[1],
            provider: "lm_studio",
            display_name: "LM Studio (local)",
            requires_api_key: false,
            preset_models: [],
            agent_model_count: 0,
          },
        ]),
      ),
    );
    renderAppAt("/settings");

    const select = await screen.findByLabelText("Provider");
    const local = within(select).getByRole("option", {
      name: /LM Studio/,
    }) as HTMLOptionElement;
    expect(local.disabled).toBe(false);
  });

  it("accepts a model id that is in no catalog", async () => {
    const user = userEvent.setup();
    renderAppAt("/settings");
    const select = await screen.findByLabelText("Provider");
    fireEvent.change(select, { target: { value: "deepseek" } });

    const typed = await screen.findByLabelText("Model id");
    await user.clear(typed);
    await user.type(typed, "my-finetune:latest");

    expect(typed).toHaveValue("my-finetune:latest");
  });

  it("says an unverified model gets probed rather than refusing it", async () => {
    useSettingsState({
      provider: "deepseek",
      model: "some-new-model",
      model_verified: false,
      status_state: "ready",
      is_ready_for_live_calls: true,
      api_key_set: true,
    });
    renderAppAt("/settings");

    expect(await screen.findByText(/probes this model/i)).toBeInTheDocument();
  });

  it("the API key is a password field and never rendered back in clear text", async () => {
    renderAppAt("/settings");
    fireEvent.change(await screen.findByLabelText("Provider"), {
      target: { value: "deepseek" },
    });

    const keyInput = await screen.findByLabelText("API key");
    expect(keyInput).toHaveAttribute("type", "password");
    fireEvent.change(keyInput, { target: { value: SECRET } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));

    expect(await screen.findByText("Configured ••••9911")).toBeInTheDocument();
    // The saved key must not survive in the input, nor appear anywhere on screen.
    await waitFor(() => expect(keyInput).toHaveValue(""));
    expect(document.body.textContent).not.toContain(SECRET);
  });

  it("sends the key in the request body and only once", async () => {
    const bodies: unknown[] = [];
    server.use(
      http.put("/api/v1/settings", async ({ request }) => {
        const body = await request.json();
        bodies.push(body);
        return HttpResponse.json({
          ...(await (await fetch("/api/v1/settings")).json()),
          api_key_set: true,
          api_key_last4: "9911",
          provider: "deepseek",
          source: "session",
        });
      }),
    );
    renderAppAt("/settings");
    fireEvent.change(await screen.findByLabelText("Provider"), {
      target: { value: "deepseek" },
    });
    await waitFor(() => expect(bodies).toHaveLength(1));
    fireEvent.change(await screen.findByLabelText("API key"), {
      target: { value: SECRET },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => expect(bodies).toHaveLength(2));
    expect(JSON.stringify(bodies[0])).not.toContain(SECRET);
    expect(JSON.stringify(bodies[1])).toContain(SECRET);
  });

  it("tests the connection and reports the result", async () => {
    renderAppAt("/settings");
    await screen.findByLabelText("Provider");
    fireEvent.click(screen.getByRole("button", { name: "Test connection" }));
    expect(
      await screen.findByText(/Connected · Provider responded\./),
    ).toBeInTheDocument();
  });

  it("surfaces a failed connection test without leaking the key", async () => {
    server.use(
      http.post("/api/v1/settings/test", () =>
        HttpResponse.json({
          ok: false,
          provider: "deepseek",
          model: "deepseek-v4-flash",
          elapsed_ms: 12,
          message: "LLM provider returned HTTP 401: invalid key ***",
          error_code: "RuntimeError",
        }),
      ),
    );
    renderAppAt("/settings");
    await screen.findByLabelText("Provider");
    fireEvent.click(screen.getByRole("button", { name: "Test connection" }));
    expect(await screen.findByText(/Failed · /)).toBeInTheDocument();
    expect(document.body.textContent).not.toContain(SECRET);
  });

  it("changes the payload policy from Analysis behavior", async () => {
    renderAppAt("/settings");
    await openSection("Analysis behavior");

    const sampleOption = await screen.findByRole("radio", {
      name: /Schema \+ aggregates \+ sample rows/,
    });
    expect(
      screen.getByRole("radio", { name: /Schema \+ aggregates$/ }),
    ).toBeChecked();
    fireEvent.click(sampleOption);
    await waitFor(() => expect(sampleOption).toBeChecked());
  });

  it("offers system/light/dark and stores an explicit choice", async () => {
    renderAppAt("/settings");
    await openSection("Appearance");

    expect(await screen.findByRole("radio", { name: "Follow system" })).toBeChecked();
    fireEvent.click(screen.getByRole("radio", { name: "Dark" }));
    expect(document.documentElement.dataset["theme"]).toBe("dark");
    expect(window.localStorage.getItem("eda.theme")).toBe("dark");

    fireEvent.click(screen.getByRole("radio", { name: "Follow system" }));
    expect(window.localStorage.getItem("eda.theme")).toBeNull();
    expect(document.documentElement.dataset["theme"]).toBeUndefined();
  });

  it("offers comfortable/compact density and applies it immediately", async () => {
    renderAppAt("/settings");
    await openSection("Appearance");

    expect(
      await screen.findByRole("radio", { name: "Comfortable" }),
    ).toBeChecked();
    expect(document.documentElement.dataset["density"]).toBeUndefined();

    fireEvent.click(screen.getByRole("radio", { name: "Compact" }));
    expect(document.documentElement.dataset["density"]).toBe("compact");
    expect(window.localStorage.getItem("eda.density")).toBe("compact");

    fireEvent.click(screen.getByRole("radio", { name: "Comfortable" }));
    expect(document.documentElement.dataset["density"]).toBeUndefined();
    expect(window.localStorage.getItem("eda.density")).toBe("comfortable");
  });

  it("About shows the version and a relativized workspace, never a full path", async () => {
    renderAppAt("/settings");
    await openSection("About");
    expect(await screen.findByText("0.2.0")).toBeInTheDocument();
    expect(screen.getByText("default workspace")).toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/\/Users\/|\/home\//);
  });

  it("resets back to environment defaults", async () => {
    renderAppAt("/settings");
    fireEvent.change(await screen.findByLabelText("Provider"), {
      target: { value: "deepseek" },
    });
    await waitFor(() =>
      expect(screen.getByLabelText("Provider")).toHaveValue("deepseek"),
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Reset to environment defaults" }),
    );
    await waitFor(() =>
      expect(screen.getByLabelText("Provider")).toHaveValue("offline"),
    );
  });

  /* The one thing this screen must never be vague about: whether a run will
   * call something that charges money. */
  it("says offline costs nothing, in every section", async () => {
    const user = userEvent.setup();
    renderAppAt("/settings");

    expect(await screen.findByText("No API calls")).toBeInTheDocument();
    expect(
      screen.getByText("Offline — no model is called"),
    ).toBeInTheDocument();
    expect(screen.queryByText("Calls a paid model")).not.toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: "About" }));
    expect(screen.getByText("No API calls")).toBeInTheDocument();
  });

  it("warns that a ready live provider costs money", async () => {
    useSettingsState({
      provider: "deepseek",
      model: "deepseek-v4-pro",
      api_key_set: true,
      api_key_last4: "9911",
      is_ready_for_live_calls: true,
      status_state: "ready",
      status_message: "Configuration is ready for live LLM calls.",
      resolved_base_url: "https://api.deepseek.com",
    });
    renderAppAt("/settings");

    expect(await screen.findByText("Calls a paid model")).toBeInTheDocument();
    expect(screen.getByText("DeepSeek · deepseek-v4-pro")).toBeInTheDocument();
    expect(screen.queryByText("No API calls")).not.toBeInTheDocument();
    expect(document.body.textContent).not.toContain(SECRET);
  });

  it("names what a half-configured provider is still missing", async () => {
    useSettingsState({
      provider: "deepseek",
      model: "deepseek-v4-pro",
      is_ready_for_live_calls: false,
      status_state: "incomplete",
      status_message: "Missing required LLM setting: api_key.",
      missing_fields: ["api_key", "base_url"],
    });
    renderAppAt("/settings");

    expect(await screen.findByText("Not usable yet")).toBeInTheDocument();
    expect(
      screen.getByText("Still needs an API key, a base URL."),
    ).toBeInTheDocument();
    expect(screen.queryByText("No API calls")).not.toBeInTheDocument();
  });

  /* Provider saves on change, every other field waits for Save; nothing said
   * so, which is how a typed-but-unsent API key gets left on screen. */
  it("flags edits that Save has not sent yet", async () => {
    useSettingsState({
      provider: "deepseek",
      model: "deepseek-v4-pro",
      is_ready_for_live_calls: true,
      status_state: "ready",
    });
    renderAppAt("/settings");
    const model = await screen.findByLabelText("Model");
    expect(screen.queryByText("Unsaved changes")).not.toBeInTheDocument();

    fireEvent.change(model, { target: { value: "deepseek-v4-flash" } });
    expect(screen.getByText("Unsaved changes")).toBeInTheDocument();
  });

  it("sends the session header on every API call", async () => {
    let seen: string | null = null;
    server.use(
      http.get("/api/v1/settings", ({ request }) => {
        seen = request.headers.get("X-EDA-Session");
        return HttpResponse.error();
      }),
    );
    renderAppAt("/settings");
    await waitFor(() => expect(seen).toBeTruthy());
  });
});

/* The gear opens the same panel the /settings route renders, so the sections
 * are asserted here too — a dialog that silently lost one would still pass the
 * open/close assertions. */
describe("Settings dialog", () => {
  const gear = () =>
    within(screen.getByRole("banner")).getByRole("button", { name: "Settings" });

  it("opens from the top-bar gear without leaving the page", async () => {
    const user = userEvent.setup();
    renderAppAt("/projects");
    await screen.findByRole("heading", { name: "Overview" });

    await user.click(gear());
    const dialog = await screen.findByRole("dialog", { name: "Settings" });
    expect(dialog).toHaveAttribute("aria-modal", "true");
    expect(await within(dialog).findByLabelText("Provider")).toHaveValue(
      "offline",
    );
    for (const section of [
      "Model & API",
      "Analysis behavior",
      "Appearance",
      "About",
    ]) {
      expect(
        within(dialog).getByRole("tab", { name: section }),
      ).toBeInTheDocument();
    }
    /* Still on the project list: the gear must not navigate. */
    expect(
      screen.getByRole("heading", { name: "Overview" }),
    ).toBeInTheDocument();
  });

  /* Driven by the keyboard, not a click: userEvent's click leaves activeElement
   * on <body> under jsdom, which would make the focus-restore assertion pass
   * against a dialog that restored nothing. */
  it("closes on Escape and hands focus back to the gear", async () => {
    const user = userEvent.setup();
    renderAppAt("/projects");
    await screen.findByRole("heading", { name: "Overview" });

    const button = gear();
    button.focus();
    expect(button).toHaveFocus();
    await user.keyboard("{Enter}");

    const dialog = await screen.findByRole("dialog", { name: "Settings" });
    expect(dialog.contains(document.activeElement)).toBe(true);

    await user.keyboard("{Escape}");
    await waitFor(() =>
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument(),
    );
    expect(button).toHaveFocus();
  });

  it("closes from its own Close button", async () => {
    const user = userEvent.setup();
    renderAppAt("/projects");
    await screen.findByRole("heading", { name: "Overview" });

    await user.click(gear());
    const dialog = await screen.findByRole("dialog", { name: "Settings" });
    await user.click(within(dialog).getByRole("button", { name: "Close" }));
    await waitFor(() =>
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument(),
    );
  });

  it("edits made in the dialog hit the same settings endpoint", async () => {
    const user = userEvent.setup();
    renderAppAt("/projects");
    await screen.findByRole("heading", { name: "Overview" });
    await user.click(gear());

    const dialog = await screen.findByRole("dialog", { name: "Settings" });
    fireEvent.change(within(dialog).getByLabelText("Provider"), {
      target: { value: "deepseek" },
    });
    await waitFor(() =>
      expect(within(dialog).getByLabelText("Provider")).toHaveValue("deepseek"),
    );
  });
});

describe("Model catalog freshness", () => {
  it("says the list is a built-in snapshot and why, not just a list", async () => {
    renderAppAt("/settings");
    await screen.findByRole("heading", { name: "Settings" });

    /* The old page showed the shipped preset list with nothing distinguishing
     * it from what the provider actually serves, so a retired model stayed
     * selectable and looked current. */
    expect(await screen.findByText("Built-in snapshot")).toBeInTheDocument();
    expect(
      screen.getByText("Save an API key before refreshing models."),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Refresh models" }),
    ).toBeInTheDocument();
  });

  it("prices the selected model and labels the figure as a list price", async () => {
    renderAppAt("/settings");
    fireEvent.change(await screen.findByLabelText("Provider"), {
      target: { value: "deepseek" },
    });

    const line = await screen.findByText(/List price: in \$0.14 \/ out \$0.28 per 1M/);
    expect(line).toHaveTextContent("cache read $0.0028");
    expect(line).toHaveTextContent("not an invoice");
  });
});
