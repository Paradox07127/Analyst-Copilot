/* One artifact-payload reader, shared by the Artifacts page and the Report
 * page's evidence inspector, so evidence looks the same wherever it is read. */

import { Card, Disclosure, formatCompact } from "../../components/ui";

const PREVIEW_CHARS = 90;

export function PayloadBlock({ payload }: { payload: unknown }) {
  return (
    <pre className="max-h-96 overflow-auto rounded-base bg-code-bg p-3 font-mono text-xs text-code-text">
      {JSON.stringify(payload, null, 2)}
    </pre>
  );
}

/** Shape of a value, not the value itself, when it is too big to read inline. */
function describe(value: unknown): string {
  if (value === null) return "null";
  if (Array.isArray(value))
    return `${formatCompact(value.length)} ${value.length === 1 ? "item" : "items"}`;
  if (typeof value === "object") {
    const size = Object.keys(value as object).length;
    return `${formatCompact(size)} ${size === 1 ? "field" : "fields"}`;
  }
  if (typeof value === "string")
    return value.length > PREVIEW_CHARS
      ? `${value.slice(0, PREVIEW_CHARS)}…`
      : value;
  if (typeof value === "number") return formatCompact(value);
  return String(value);
}

function isScalar(value: unknown): boolean {
  return value === null || typeof value !== "object";
}

/** Top-level keys first; the raw JSON stays one disclosure away. */
export function PayloadView({ payload }: { payload: unknown }) {
  const entries =
    payload !== null && typeof payload === "object" && !Array.isArray(payload)
      ? Object.entries(payload as Record<string, unknown>)
      : null;

  if (!entries || entries.length === 0) return <PayloadBlock payload={payload} />;

  return (
    <div className="flex flex-col gap-2">
      <dl className="flex flex-col">
        {entries.map(([key, value]) => (
          <div
            key={key}
            className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5 border-t border-hairline py-1 first:border-t-0"
          >
            <dt className="font-mono text-xs text-status-neutral">{key}</dt>
            <dd
              className={`min-w-0 flex-1 text-xs break-words ${
                isScalar(value) ? "tabular" : "text-status-neutral"
              }`}
            >
              {describe(value)}
            </dd>
          </div>
        ))}
      </dl>
      <Disclosure summary="Raw JSON" meta={`${entries.length} top-level keys`}>
        <PayloadBlock payload={payload} />
      </Disclosure>
    </div>
  );
}

/** Warnings ride along with every artifact and were previously never shown —
 *  an evidence store that hides its own caveats is not evidence. */
export function ArtifactWarnings({ warnings }: { warnings: string[] }) {
  if (warnings.length === 0) return null;
  return (
    <Card tone="warn" className="flex flex-col gap-1 p-2">
      <span className="text-xs font-medium text-status-warn">
        {warnings.length === 1 ? "1 warning" : `${warnings.length} warnings`}
      </span>
      <ul className="flex flex-col gap-0.5">
        {warnings.map((warning) => (
          <li key={warning} className="text-xs text-status-warn">
            {warning}
          </li>
        ))}
      </ul>
    </Card>
  );
}

export function formatCreatedAt(iso: string | null | undefined): string | null {
  if (!iso) return null;
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return null;
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

