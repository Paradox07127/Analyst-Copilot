/* Settings (§6.0): four sections — Model & API,
 * Analysis behavior, Appearance, About. The API key is write-only: it is typed
 * here, sent once, and only ever read back as "configured ••••1234".
 *
 * Owns no heading and no page chrome so the /settings route and the top-bar
 * dialog can render the identical body. */

import { useEffect, useState } from "react";
import type { ProviderInfo, SettingsPatch, SettingsView } from "../../api/client";
import {
  useProviders,
  useResetSettings,
  useSandboxStatus,
  useModels,
  useRefreshModels,
  useSettings,
  useTestConnection,
  useUpdateSettings,
} from "../../api/hooks";
import { ErrorState, LoadingSkeleton } from "../../components/async-states";
import { Badge, Button, Card, Hint, SectionHeader } from "../../components/ui";
import { LiveStatusCard } from "./live-status";
import {
  clearTheme,
  getDensity,
  getEffectiveTheme,
  hasStoredTheme,
  setDensity,
  setTheme,
  type Density,
  type Theme,
} from "../../app/theme";

const SECTIONS = [
  { id: "model", label: "Model & API" },
  { id: "analysis", label: "Analysis behavior" },
  { id: "appearance", label: "Appearance" },
  { id: "about", label: "About" },
] as const;

export type SettingsSectionId = (typeof SECTIONS)[number]["id"];

const PAYLOAD_POLICIES = [
  {
    value: "schema_only",
    label: "Schema only",
    hint: "Cheapest and most private: column names and types only.",
  },
  {
    value: "schema+aggregates",
    label: "Schema + aggregates",
    hint: "Adds computed statistics. The default balance of quality and cost.",
  },
  {
    value: "schema+aggregates+sample",
    label: "Schema + aggregates + sample rows",
    hint: "Shares real data values with the provider and costs the most tokens.",
  },
] as const;

/* Thinking level = the analysis-loop depth. The copy has to say what raising
 * it authorizes, not how clever it makes the agent: Deep and Ultra both spend
 * more budget, and Ultra spends it without asking again per round. */
const THINKING_LEVELS = [
  {
    value: 0,
    label: "Standard",
    hint: "One pass per question. Nothing runs that you did not approve individually.",
  },
  {
    value: 1,
    label: "Deep",
    hint:
      "Adds deep investigation: after the approved method runs, the model may " +
      "plan up to 3 read-only follow-up probes inside the same approved scope. " +
      "The probes are listed on the plan before you approve it.",
  },
  {
    value: 2,
    label: "Ultra",
    hint:
      "Deep, plus authorization for the macro loop: once you start it, the agent " +
      "writes its own follow-up questions and executes them for up to the round " +
      "cap without asking again, spending model budget each round.",
  },
] as const;

/* Labelled, not raw: `json_schema` in a dropdown is a wire value, and the
 * choice it stands for ("refuse anything off-schema") is not guessable. */
const STRUCTURED_MODES = [
  {
    value: "auto",
    label: "Automatic",
    hint: "Picks the right mode per provider.",
  },
  {
    value: "json_schema",
    label: "Strict schema",
    hint: "Strict schema — OpenAI, Anthropic, Gemini.",
  },
  {
    value: "json_object",
    label: "Loose JSON",
    hint: "Loose JSON — most OpenAI-compatible servers.",
  },
] as const;

const THEME_OPTIONS = [
  { value: "system", label: "Follow system" },
  { value: "light", label: "Light" },
  { value: "dark", label: "Dark" },
] as const;

const DENSITY_OPTIONS = [
  { value: "comfortable", label: "Comfortable" },
  { value: "compact", label: "Compact" },
] as const;

function payloadPolicyLabel(value: string): string {
  return (
    PAYLOAD_POLICIES.find((policy) => policy.value === value)?.label ?? value
  );
}

const inputClass =
  "w-full rounded-base border border-border bg-bg px-2 py-1 text-sm disabled:opacity-50";
const numberClass = `${inputClass} tabular`;
const labelClass = "text-sm font-medium";
const hintClass = "text-xs text-status-neutral";

function Field({
  label,
  htmlFor,
  hint,
  children,
}: {
  label: string;
  htmlFor: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-1">
      <label className={labelClass} htmlFor={htmlFor}>
        {label}
      </label>
      {children}
      {hint && <p className={hintClass}>{hint}</p>}
    </div>
  );
}

/* Settings sections were a stack of bordered cards inside a bordered dialog —
 * three frames deep before any control. A titled region divided by hairlines
 * gives the same grouping without the nesting. */
function Section({
  title,
  description,
  children,
}: {
  title: React.ReactNode;
  description?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section className="flex flex-col gap-3 border-b border-hairline pb-5 last:border-b-0 last:pb-0">
      <SectionHeader level={3} title={title} description={description} />
      {children}
    </section>
  );
}

/* One choice in a radio group, as a row the whole of which is a hit target
 * rather than a 13px dot. `aria-labelledby` is explicit because the hint lives
 * inside the wrapping label: without it the radio's accessible name would be
 * the label *and* the whole explanation. */
function OptionRow({
  id,
  name,
  value,
  label,
  hint,
  checked,
  onSelect,
}: {
  id: string;
  name: string;
  value: string | number;
  label: string;
  hint?: string;
  checked: boolean;
  onSelect: () => void;
}) {
  return (
    <label
      className={`flex cursor-pointer items-start gap-2.5 rounded-base border px-3 py-2 text-sm transition-colors duration-150 ease-out-quart ${
        checked
          ? "border-primary/45 bg-primary/5"
          : "border-border hover:border-primary/25 hover:bg-surface"
      }`}
    >
      <input
        id={id}
        type="radio"
        name={name}
        className="mt-0.5"
        value={value}
        aria-labelledby={`${id}-label`}
        aria-describedby={hint ? `${id}-hint` : undefined}
        checked={checked}
        onChange={onSelect}
      />
      <span className="flex min-w-0 flex-col gap-0.5">
        <span
          id={`${id}-label`}
          className={checked ? "font-medium text-primary" : "font-medium"}
        >
          {label}
        </span>
        {hint && (
          <span id={`${id}-hint`} className={hintClass}>
            {hint}
          </span>
        )}
      </span>
    </label>
  );
}

function ModelApiSection({
  settings,
  providers,
}: {
  settings: SettingsView;
  providers: ProviderInfo[];
}) {
  const update = useUpdateSettings();
  const reset = useResetSettings();
  const test = useTestConnection();

  const [draft, setDraft] = useState(settings);
  const [apiKey, setApiKey] = useState("");

  /* Server response is the source of truth: a provider switch re-seeds model
   * and base_url server-side, so the form follows the value that came back. */
  useEffect(() => {
    setDraft(settings);
  }, [settings]);

  const spec = providers.find((item) => item.provider === draft.provider);
  const models = useModels(settings.provider, settings.version);
  const refreshModels = useRefreshModels(settings.provider, settings.version);
  /* Discovery answers for the SAVED connection, so a catalog fetched under a
   * different provider must not be shown against this draft. */
  const catalog = models.data?.provider === draft.provider ? models.data : undefined;
  const selectedPrice = catalog?.models?.find((item) => item.id === draft.model);
  const connectionDraftDirty =
    draft.base_url !== settings.base_url || apiKey.length > 0;
  const offline = draft.provider === "offline";
  const presets = catalog?.models?.map((item) => item.id) ?? spec?.preset_models ?? [];
  const modelIsPreset = presets.includes(draft.model);

  const save = (patch: SettingsPatch) => {
    update.mutate(patch, { onSuccess: () => setApiKey("") });
  };

  const onSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    save({
      provider: draft.provider,
      model: draft.model,
      base_url: draft.base_url,
      temperature: draft.temperature,
      max_tokens: draft.max_tokens,
      timeout_seconds: draft.timeout_seconds,
      structured_output_mode: draft.structured_output_mode,
      usd_per_1k_prompt: draft.usd_per_1k_prompt,
      usd_per_1k_completion: draft.usd_per_1k_completion,
      ...(apiKey ? { api_key: apiKey } : {}),
    });
  };

  /* Two save paths sit in one form — the provider select writes on change, the
   * rest waits for Save — so the difference has to be visible, not learned. */
  const dirty =
    draft.model !== settings.model ||
    draft.base_url !== settings.base_url ||
    draft.structured_output_mode !== settings.structured_output_mode ||
    draft.temperature !== settings.temperature ||
    draft.max_tokens !== settings.max_tokens ||
    draft.timeout_seconds !== settings.timeout_seconds ||
    draft.usd_per_1k_prompt !== settings.usd_per_1k_prompt ||
    draft.usd_per_1k_completion !== settings.usd_per_1k_completion ||
    apiKey.length > 0;

  return (
    <form onSubmit={onSubmit} className="flex flex-col gap-5">
      <Section
        title="Connection"
        description="Where a model call goes and what authorizes it. The provider applies the moment you pick it; every other field applies when you press Save."
      >
        <div className="grid gap-4 lg:grid-cols-2">
          <Field
            label="Provider"
            htmlFor="settings-provider"
            hint="Offline uses the deterministic fallback; live providers enable report and chat."
          >
            <select
              id="settings-provider"
              className={inputClass}
              value={draft.provider}
              onChange={(event) => save({ provider: event.target.value })}
            >
              {providers.map((item) => (
                <option key={item.provider} value={item.provider}>
                  {item.display_name}
                </option>
              ))}
            </select>
          </Field>

          <Field
            label="Model"
            htmlFor="settings-model"
            hint="Pick a listed model or type any id this endpoint serves."
          >
            {presets.length > 0 && (
              <select
                id="settings-model"
                aria-label="Agent model"
                className={inputClass}
                value={modelIsPreset ? draft.model : ""}
                disabled={offline}
                onChange={(event) => {
                  setDraft({ ...draft, model: event.target.value });
                }}
              >
                {!modelIsPreset && (
                  <option value="" disabled>
                    {draft.model ? `Custom: ${draft.model}` : "Choose a model…"}
                  </option>
                )}
                {presets.map((model) => (
                  <option key={model} value={model}>
                    {model}
                  </option>
                ))}
              </select>
            )}
            {/* A self-hosted model's id is whatever the operator named it, so
             * the list can only ever be a suggestion. */}
            {!offline && (
              <input
                aria-label="Model id"
                className={`${inputClass} ${presets.length > 0 ? "mt-2" : ""} font-mono`}
                value={draft.model}
                placeholder="model id"
                onChange={(event) =>
                  setDraft({ ...draft, model: event.target.value })
                }
              />
            )}
            <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
              {models.isPending ? (
                <span className="text-status-neutral">Loading model list…</span>
              ) : catalog ? (
                <>
                  <Badge tone={catalog.source === "live" ? "ok" : "warn"}>
                    {catalog.source === "live"
                      ? "Live from provider"
                      : "Built-in snapshot"}
                  </Badge>
                  <span className="text-status-neutral">
                    fetched {new Date(catalog.fetched_at).toLocaleString()}
                  </span>
                  {catalog.warning && (
                    <span className="text-status-warn">{catalog.warning}</span>
                  )}
                  {catalog.truncated && (
                    <span className="text-status-warn">
                      Partial provider page; some models are not listed.
                    </span>
                  )}
                </>
              ) : null}
              <Button
                size="sm"
                onClick={() => refreshModels.mutate()}
                disabled={offline || connectionDraftDirty || refreshModels.isPending}
              >
                {refreshModels.isPending ? "Refreshing…" : "Refresh models"}
              </Button>
            </div>
            {connectionDraftDirty && !offline && (
              <p className={hintClass}>
                Save the key and base URL first — discovery always uses the
                saved connection, not this draft.
              </p>
            )}
            {selectedPrice &&
              (selectedPrice.input_usd_per_1m != null ||
                selectedPrice.output_usd_per_1m != null) && (
                <p className={hintClass}>
                  {`List price: in $${selectedPrice.input_usd_per_1m ?? "?"} / out $${selectedPrice.output_usd_per_1m ?? "?"} per 1M`}
                  {selectedPrice.cache_read_usd_per_1m != null &&
                    ` · cache read $${selectedPrice.cache_read_usd_per_1m}`}
                  {selectedPrice.cache_write_usd_per_1m != null &&
                    ` · cache write $${selectedPrice.cache_write_usd_per_1m}`}
                  {catalog?.pricing_notice ? ` · ${catalog.pricing_notice}` : ""}
                </p>
              )}
            {!offline && draft.model === settings.model && (
              <p className={hintClass}>
                {settings.model_verified ? (
                  <>
                    Agent verified · tool calling
                    {selectedPrice?.parallel_tool_calling ? " · parallel tools" : ""}
                    {selectedPrice?.structured_output
                      ? ` · ${selectedPrice.structured_output}`
                      : ""}
                  </>
                ) : (
                  "Not pre-verified. The run probes this model for tool calling before it spends anything, and falls back to the deterministic path if it cannot."
                )}
              </p>
            )}
          </Field>

          <Field
            label={spec?.requires_base_url ? "Base URL (required)" : "Base URL"}
            htmlFor="settings-base-url"
            hint={
              spec?.default_base_url
                ? `Leave blank to use the provider default: ${spec.default_base_url}`
                : "Required for Azure and custom OpenAI-compatible endpoints."
            }
          >
            <input
              id="settings-base-url"
              className={`${inputClass} font-mono`}
              value={draft.base_url}
              disabled={offline}
              placeholder={spec?.default_base_url || "https://host/v1"}
              onChange={(event) =>
                setDraft({ ...draft, base_url: event.target.value })
              }
            />
          </Field>

          {!offline && (
            <Field
              label="API key"
              htmlFor="settings-api-key"
              hint={
                spec?.requires_api_key
                  ? "Held in server memory for this browser session only — never written to disk, logs or the page."
                  : "Local and compatible servers usually need no key."
              }
            >
              <div className="flex flex-wrap items-center gap-2">
                <input
                  id="settings-api-key"
                  type="password"
                  autoComplete="off"
                  className={`${inputClass} font-mono`}
                  value={apiKey}
                  placeholder={
                    settings.api_key_set
                      ? "Leave blank to keep the saved key"
                      : "Paste the provider key"
                  }
                  onChange={(event) => setApiKey(event.target.value)}
                />
                {settings.api_key_set && (
                  <span className="text-xs text-status-ok">
                    Configured ••••{settings.api_key_last4}
                  </span>
                )}
                {settings.api_key_set && (
                  <Button size="sm" onClick={() => save({ clear_api_key: true })}>
                    Clear key
                  </Button>
                )}
              </div>
            </Field>
          )}
        </div>
      </Section>

      <Section
        title="Request shape"
        description="How each call is framed. Ignored while the provider is Offline."
      >
        <Field
          label="Structured output mode"
          htmlFor="settings-structured"
          hint={
            STRUCTURED_MODES.find(
              (mode) => mode.value === draft.structured_output_mode,
            )?.hint
          }
        >
          <select
            id="settings-structured"
            className={inputClass}
            value={draft.structured_output_mode}
            disabled={offline}
            onChange={(event) =>
              setDraft({ ...draft, structured_output_mode: event.target.value })
            }
          >
            {STRUCTURED_MODES.map((mode) => (
              <option key={mode.value} value={mode.value}>
                {`${mode.label} (${mode.value})`}
              </option>
            ))}
          </select>
        </Field>

        <div className="grid gap-3 sm:grid-cols-3">
          <Field
            label="Temperature"
            htmlFor="settings-temperature"
            hint={
              selectedPrice?.temperature_policy === "omit"
                ? "This model rejects sampling temperature; the request adapter omits it."
                : "Sent to the provider when this model supports sampling temperature."
            }
          >
            <input
              id="settings-temperature"
              type="number"
              step="0.1"
              min={0}
              max={2}
              className={numberClass}
              value={draft.temperature}
              disabled={
                offline || selectedPrice?.temperature_policy === "omit"
              }
              onChange={(event) =>
                setDraft({ ...draft, temperature: Number(event.target.value) })
              }
            />
          </Field>
          <Field label="Max tokens" htmlFor="settings-max-tokens">
            <input
              id="settings-max-tokens"
              type="number"
              min={256}
              className={numberClass}
              value={draft.max_tokens}
              disabled={offline}
              onChange={(event) =>
                setDraft({ ...draft, max_tokens: Number(event.target.value) })
              }
            />
          </Field>
          <Field label="Timeout (seconds)" htmlFor="settings-timeout">
            <input
              id="settings-timeout"
              type="number"
              min={10}
              max={600}
              className={numberClass}
              value={draft.timeout_seconds}
              disabled={offline}
              onChange={(event) =>
                setDraft({
                  ...draft,
                  timeout_seconds: Number(event.target.value),
                })
              }
            />
          </Field>
        </div>
      </Section>

      <Section
        title={
          <span className="inline-flex items-center gap-1.5">
            Cost accounting
            <Hint label="Cost / 1k tokens">
              These prices only report what a session cost. Changing them changes
              the figure in the trace and the report header, never what the
              provider actually bills.
            </Hint>
          </span>
        }
      >
        <div className="grid gap-3 sm:grid-cols-2">
          <Field
            label="Cost / 1k prompt tokens (USD)"
            htmlFor="settings-usd-prompt"
            hint="0 falls back to the built-in pricing table for the selected model."
          >
            <input
              id="settings-usd-prompt"
              type="number"
              step="0.0001"
              min={0}
              max={1}
              className={numberClass}
              value={draft.usd_per_1k_prompt}
              disabled={offline}
              onChange={(event) =>
                setDraft({
                  ...draft,
                  usd_per_1k_prompt: Number(event.target.value),
                })
              }
            />
          </Field>
          <Field
            label="Cost / 1k completion tokens (USD)"
            htmlFor="settings-usd-completion"
            hint="Drives the estimated session cost shown in traces and the report header."
          >
            <input
              id="settings-usd-completion"
              type="number"
              step="0.0001"
              min={0}
              max={1}
              className={numberClass}
              value={draft.usd_per_1k_completion}
              disabled={offline}
              onChange={(event) =>
                setDraft({
                  ...draft,
                  usd_per_1k_completion: Number(event.target.value),
                })
              }
            />
          </Field>
        </div>
      </Section>

      {/* Pinned rather than trailing the third section: the two write paths in
       * this form (provider on change, everything else on Save) are only
       * distinguishable if Save is on screen while you edit. */}
      <div className="sticky bottom-0 -mx-1 flex flex-wrap items-center gap-2 border-t border-border bg-bg px-1 py-3">
        <Button type="submit" variant="primary" disabled={update.isPending}>
          {update.isPending ? "Saving…" : "Save"}
        </Button>
        <Button onClick={() => test.mutate()} disabled={test.isPending}>
          {test.isPending ? "Testing…" : "Test connection"}
        </Button>
        {dirty && <Badge tone="warn">Unsaved changes</Badge>}
        <Button
          variant="ghost"
          onClick={() => reset.mutate()}
          disabled={reset.isPending}
          className="ml-auto"
        >
          Reset to environment defaults
        </Button>
      </div>

      {update.isError && <ErrorState error={update.error} />}
      {test.data && (
        <p
          role="status"
          className={`text-sm ${test.data.ok ? "text-status-ok" : "text-status-critical"}`}
        >
          {test.data.ok ? "Connected" : "Failed"} · {test.data.message}
          {test.data.elapsed_ms ? ` (${test.data.elapsed_ms} ms)` : ""}
        </p>
      )}
      {test.isError && <ErrorState error={test.error} />}
    </form>
  );
}

function AnalysisSection({ settings }: { settings: SettingsView }) {
  const update = useUpdateSettings();
  /* Depth 3 exists server-side (a longer round cap) but is not offered here;
   * it reads as Ultra so the selection still round-trips. */
  const selectedDepth = settings.analysis_depth >= 2 ? 2 : settings.analysis_depth;
  const offline = settings.provider === "offline";
  return (
    <div className="flex flex-col gap-5">
      <Section
        title="Thinking level"
        description="How much work the agent is allowed to do per question, and how much of it runs without a further prompt. Applies to analyses started from now on."
      >
        <div role="radiogroup" aria-label="Thinking level" className="flex flex-col gap-2">
          {THINKING_LEVELS.map((level) => (
            <OptionRow
              key={level.value}
              id={`depth-${level.value}`}
              name="analysis-depth"
              value={level.value}
              label={level.label}
              hint={level.hint}
              checked={selectedDepth === level.value}
              onSelect={() => update.mutate({ analysis_depth: level.value })}
            />
          ))}
        </div>
        {settings.analysis_depth >= 2 && (
          <Card tone="warn" className="p-3 text-xs text-status-warn">
            Ultra is an authorization, not a quality setting. Starting the macro
            loop on the Questions page lets the agent run multiple further
            rounds of analysis on its own and consume budget for each one.
          </Card>
        )}
      </Section>

      <Section
        title="Payload policy"
        description={
          "How much of a dataset leaves this machine when the agent asks the model a question. Applies to sessions started from now on." +
          (offline
            ? " Nothing is sent while the provider is Offline, so this takes effect only once a live provider is configured."
            : "")
        }
      >
        <div role="radiogroup" aria-label="Payload policy" className="flex flex-col gap-2">
          {PAYLOAD_POLICIES.map((policy) => (
            <OptionRow
              key={policy.value}
              id={`policy-${policy.value}`}
              name="payload-policy"
              value={policy.value}
              label={policy.label}
              hint={policy.hint}
              checked={settings.payload_policy === policy.value}
              onSelect={() => update.mutate({ payload_policy: policy.value })}
            />
          ))}
        </div>
      </Section>
      {update.isError && <ErrorState error={update.error} />}
    </div>
  );
}

function AppearanceSection() {
  const [choice, setChoice] = useState<"system" | Theme>(() =>
    hasStoredTheme() ? getEffectiveTheme() : "system",
  );
  const [density, setDensityChoice] = useState<Density>(() => getDensity());

  const apply = (next: "system" | Theme) => {
    setChoice(next);
    if (next === "system") clearTheme();
    else setTheme(next);
  };

  const applyDensity = (next: Density) => {
    setDensityChoice(next);
    setDensity(next);
  };

  return (
    <div className="flex flex-col gap-5">
      <Section
        title="Theme"
        description="“Follow system” tracks your OS light/dark setting. Stored in this browser, not on the server."
      >
        <div role="radiogroup" aria-label="Theme" className="flex flex-wrap gap-2">
          {THEME_OPTIONS.map((option) => (
            <label
              key={option.value}
              className={`flex cursor-pointer items-center gap-2 rounded-base border px-3 py-1.5 text-sm transition-colors duration-150 ease-out-quart ${
                choice === option.value
                  ? "border-primary/45 bg-primary/5 font-medium text-primary"
                  : "border-border hover:border-primary/25 hover:bg-surface"
              }`}
            >
              <input
                type="radio"
                name="theme"
                value={option.value}
                checked={choice === option.value}
                onChange={() => apply(option.value)}
              />
              {option.label}
            </label>
          ))}
        </div>
      </Section>

      <Section
        title="UI density"
        description="Compact tightens paddings and spacing to fit more on screen. Applies immediately."
      >
        <div role="radiogroup" aria-label="UI density" className="flex flex-wrap gap-2">
          {DENSITY_OPTIONS.map((option) => (
            <label
              key={option.value}
              className={`flex cursor-pointer items-center gap-2 rounded-base border px-3 py-1.5 text-sm transition-colors duration-150 ease-out-quart ${
                density === option.value
                  ? "border-primary/45 bg-primary/5 font-medium text-primary"
                  : "border-border hover:border-primary/25 hover:bg-surface"
              }`}
            >
              <input
                type="radio"
                name="density"
                value={option.value}
                checked={density === option.value}
                onChange={() => applyDensity(option.value)}
              />
              {option.label}
            </label>
          ))}
        </div>
      </Section>
    </div>
  );
}

function SandboxRow() {
  const sandbox = useSandboxStatus();
  if (sandbox.isPending) return <dd className={hintClass}>Checking…</dd>;
  if (sandbox.isError || !sandbox.data) {
    return <dd className={hintClass}>Status unavailable</dd>;
  }
  const usable = sandbox.data.open_python_analysis_available;
  return (
    <dd className={usable ? "text-status-ok" : "text-status-warn"}>
      <span className="font-mono">{sandbox.data.backend}</span> ·{" "}
      {sandbox.data.message}
    </dd>
  );
}

function AboutSection({ settings }: { settings: SettingsView }) {
  const update = useUpdateSettings();
  return (
    <div className="flex flex-col gap-5">
      <Section
        title="This workspace"
        description="Everything on these four tabs is workspace-wide: it applies to every project and every session, not to the session you have open."
      >
        <dl className="grid grid-cols-[auto_1fr] gap-x-6 gap-y-2 text-sm">
          <dt className="text-status-neutral">Version</dt>
          <dd className="font-mono">{settings.about.app_version}</dd>
          <dt className="text-status-neutral">Workspace</dt>
          <dd className="font-mono">{settings.about.workspace_label}</dd>
          <dt className="text-status-neutral">Config source</dt>
          <dd>
            {settings.source === "session"
              ? "This browser session (overrides the environment)"
              : "Server environment"}
          </dd>
          <dt className="text-status-neutral">Code sandbox</dt>
          <SandboxRow />
        </dl>
      </Section>

      {/* Deliberately untitled: a "Developer inspector" heading above a
       * "Developer inspector" checkbox is the same words twice. */}
      <label className="flex w-fit cursor-pointer items-start gap-2.5 text-sm">
        <input
          type="checkbox"
          className="mt-0.5"
          checked={settings.dev_mode}
          onChange={(event) => update.mutate({ dev_mode: event.target.checked })}
        />
        <span className="flex flex-col gap-0.5">
          <span className="font-medium">Developer inspector</span>
          <span className={hintClass}>
            Trace capture always runs; this only controls whether developer
            panels are shown.
          </span>
        </span>
      </label>
      {update.isError && <ErrorState error={update.error} />}
    </div>
  );
}

export function SettingsPanel({
  section: controlledSection,
  onSectionChange,
}: {
  section?: SettingsSectionId;
  onSectionChange?: (section: SettingsSectionId) => void;
} = {}) {
  const [internalSection, setInternalSection] =
    useState<SettingsSectionId>("model");
  const section = controlledSection ?? internalSection;
  const setSection = (nextSection: SettingsSectionId) => {
    if (controlledSection === undefined) setInternalSection(nextSection);
    onSectionChange?.(nextSection);
  };
  const settings = useSettings();
  const providers = useProviders();

  const providerName =
    providers.data?.find((item) => item.provider === settings.data?.provider)
      ?.display_name ??
    settings.data?.provider ??
    "";

  return (
    <div className="flex flex-col gap-4">
      {/* Above the tabs, not inside Model & API: whether a run spends money is
       * the answer every section is read against. */}
      {settings.data && (
        <LiveStatusCard
          settings={settings.data}
          providerName={providerName}
          payloadLabel={payloadPolicyLabel(settings.data.payload_policy)}
          {...(section === "model" ? {} : { onFixConnection: () => setSection("model") })}
        />
      )}

      <div className="flex flex-col gap-5 lg:flex-row lg:gap-8">
        <nav
          aria-label="Settings sections"
          role="tablist"
          className="grid grid-cols-2 gap-0.5 sm:grid-cols-4 lg:flex lg:w-48 lg:shrink-0 lg:flex-col lg:border-r lg:border-hairline lg:pr-4"
        >
          {SECTIONS.map((item) => (
            <button
              key={item.id}
              id={`settings-tab-${item.id}`}
              type="button"
              role="tab"
              aria-selected={section === item.id}
              aria-controls={`settings-panel-${item.id}`}
              onClick={() => setSection(item.id)}
              /* Hover changes text only: a `bg-surface` hover next to a
               * `bg-primary/10` active state are both near-white tints, and
               * two tabs appeared selected at once. */
              className={`rounded-base px-2.5 py-1.5 text-left text-sm transition-colors duration-150 ease-out-quart ${
                section === item.id
                  ? "bg-primary/10 font-medium text-primary"
                  : "text-status-neutral hover:text-text"
              }`}
            >
              {item.label}
            </button>
          ))}
        </nav>

        <section
          id={`settings-panel-${section}`}
          role="tabpanel"
          aria-labelledby={`settings-tab-${section}`}
          tabIndex={0}
          className="min-w-0 flex-1 outline-none"
        >
          {settings.isPending && (
            <LoadingSkeleton lines={5} label="Loading settings" />
          )}
          {settings.isError && (
            <ErrorState error={settings.error} onRetry={() => settings.refetch()} />
          )}
          {settings.data && (
            <>
              {section === "model" &&
                (providers.data ? (
                  <ModelApiSection
                    settings={settings.data}
                    providers={providers.data}
                  />
                ) : (
                  <LoadingSkeleton lines={4} label="Loading providers" />
                ))}
              {section === "analysis" && (
                <AnalysisSection settings={settings.data} />
              )}
              {section === "appearance" && <AppearanceSection />}
              {section === "about" && <AboutSection settings={settings.data} />}
            </>
          )}
        </section>
      </div>
    </div>
  );
}
