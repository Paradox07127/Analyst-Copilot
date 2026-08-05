/* The question Settings has to answer before any other: will a run call a paid
 * model, and if it is meant to, is it usable yet. The server already computes
 * this (`is_ready_for_live_calls`, `status_state`, `missing_fields`); nothing
 * of it reached the panel before, so "provider: DeepSeek" with no key read the
 * same as a working live setup. */

import type { SettingsView } from "../../api/client";
import { Badge, Card, Dot, type Tone } from "../../components/ui";

export type LiveState = "offline" | "incomplete" | "ready";

/** Server truth first: `is_ready_for_live_calls` is the flag the session path
 *  itself checks, so the panel must not infer a friendlier answer. */
export function liveState(settings: SettingsView): LiveState {
  if (settings.provider === "offline") return "offline";
  return settings.is_ready_for_live_calls ? "ready" : "incomplete";
}

const MISSING_LABEL: Record<string, string> = {
  api_key: "an API key",
  base_url: "a base URL",
  model: "a model id",
  provider: "a provider",
};

const STATE_TONE: Record<LiveState, Tone> = {
  offline: "ok",
  incomplete: "critical",
  ready: "warn",
};

/* Warn, not ok, for the working live setup: the tone tracks "this spends
 * money", which is what the user is deciding, not "this is configured". */
const STATE_BADGE: Record<LiveState, string> = {
  offline: "No API calls",
  incomplete: "Not usable yet",
  ready: "Calls a paid model",
};

const STATE_BODY: Record<LiveState, string> = {
  offline:
    "Profiling, quality, statistics and skill replays still run. Nothing leaves this machine and nothing costs money; answers come from the deterministic fallback instead of a model.",
  incomplete:
    "A live provider is selected but the connection is not complete, so anything that needs a model fails instead of falling back.",
  ready:
    "Question discovery, chat answers and the report call this provider, so sessions started from now on may cost money.",
};

export function LiveStatusCard({
  settings,
  providerName,
  payloadLabel,
  onFixConnection,
}: {
  settings: SettingsView;
  providerName: string;
  payloadLabel: string;
  onFixConnection?: () => void;
}) {
  const state = liveState(settings);
  const missing = settings.missing_fields ?? [];

  return (
    <Card
      tone={state === "ready" ? "warn" : state === "incomplete" ? "critical" : "quiet"}
      className="flex flex-col gap-2 p-4"
    >
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        <Dot tone={STATE_TONE[state]} />
        <h2 className="text-base font-semibold">
          {state === "offline"
            ? "Offline — no model is called"
            : `${providerName} · ${settings.model || "no model set"}`}
        </h2>
        <Badge tone={STATE_TONE[state]}>{STATE_BADGE[state]}</Badge>
      </div>

      <p className="max-w-content text-sm text-status-neutral">
        {STATE_BODY[state]}
      </p>

      {state === "incomplete" && missing.length > 0 && (
        <p className="text-sm text-status-critical">
          {`Still needs ${missing
            .map((field) => MISSING_LABEL[field] ?? field)
            .join(", ")}.`}
        </p>
      )}

      <dl className="flex flex-wrap gap-x-6 gap-y-1 text-xs">
        <div className="flex gap-1.5">
          <dt className="text-status-neutral">Endpoint</dt>
          <dd className="font-mono">
            {settings.resolved_base_url || "provider default"}
          </dd>
        </div>
        <div className="flex gap-1.5">
          <dt className="text-status-neutral">Sent to the model</dt>
          <dd>{state === "offline" ? "nothing" : payloadLabel}</dd>
        </div>
        {/* The heading names one model. With a report override two are in
         * play, and a status card that hides the second one is wrong. */}
        {state !== "offline" && settings.report_model && (
          <div className="flex gap-1.5">
            <dt className="text-status-neutral">Report written by</dt>
            <dd className="font-mono">{settings.report_model}</dd>
          </div>
        )}
        <div className="flex gap-1.5">
          <dt className="text-status-neutral">Set in</dt>
          <dd>
            {settings.source === "session"
              ? "this browser session"
              : "the server environment"}
          </dd>
        </div>
      </dl>

      {/* The server's own sentence, kept verbatim: it is the string the session
       * path reports when it refuses, so the two must not drift apart. */}
      {settings.status_message && (
        <p role="status" className="text-xs text-status-neutral">
          {settings.status_message}
        </p>
      )}

      {onFixConnection && state !== "ready" && (
        <button
          type="button"
          onClick={onFixConnection}
          className="self-start rounded-base border border-border px-2 py-1 text-xs hover:bg-surface"
        >
          {state === "incomplete" ? "Finish the connection" : "Change provider"}
        </button>
      )}
    </Card>
  );
}

