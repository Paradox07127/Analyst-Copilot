/* Evidence opens beside the report instead of navigating to the Artifacts
 * page: checking a number should not cost you your place in the reading.
 * The API is run-scoped, so callers must pass the evidence's source run. */

import { Link } from "react-router";
import { useArtifact } from "../../api/hooks";
import { artifactPath } from "../../app/paths";
import { ErrorState, LoadingSkeleton } from "../../components/async-states";
import { Badge, Card } from "../../components/ui";
import {
  ArtifactWarnings,
  formatCreatedAt,
  PayloadView,
} from "../artifacts/artifact-payload";
import type { ClaimLedgerRow } from "./report-outline";

export function EvidenceInspector({
  projectId,
  sessionId,
  artifactId,
  citations,
  onClose,
}: {
  projectId: string;
  sessionId: string;
  artifactId: string;
  citations: ClaimLedgerRow[];
  onClose: () => void;
}) {
  const detail = useArtifact(sessionId, artifactId, true);
  const created = formatCreatedAt(detail.data?.created_at);
  const mismatchedRun = detail.data != null && detail.data.session_id !== sessionId;

  return (
    <aside
      aria-label="Evidence inspector"
      className="fixed inset-x-3 bottom-3 z-30 flex max-h-[min(70vh,38rem)] min-w-0 flex-col gap-3 overflow-auto rounded-base border border-primary bg-surface p-3 shadow-card lg:sticky lg:inset-auto lg:top-4 lg:max-h-[calc(100vh-5rem)] lg:w-80 lg:shrink-0"
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex min-w-0 flex-col gap-1">
          <span className="text-xs font-medium tracking-wide text-status-neutral uppercase">
            Evidence
          </span>
          <span className="font-mono text-sm break-all">{artifactId}</span>
          <span className="text-xs text-status-neutral">
            Source run <span className="font-mono break-all">{sessionId}</span>
          </span>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="shrink-0 rounded-base border border-border px-2 py-0.5 text-xs hover:bg-bg"
        >
          Close
        </button>
      </div>

      <div className="flex flex-wrap items-center gap-2 text-xs text-status-neutral">
        {detail.data && <Badge tone="neutral">{detail.data.type}</Badge>}
        {created && <span className="tabular">{created}</span>}
      </div>

      {mismatchedRun && (
        <Card tone="warn" className="p-2">
          <p className="text-xs text-status-warn">
            {`The store returned run ${detail.data?.session_id}, but this reference points to ${sessionId}. Check the evidence link before reusing the claim.`}
          </p>
        </Card>
      )}

      {citations.length > 0 && (
        <div className="flex flex-col gap-1">
          <span className="text-xs font-medium">
            {citations.length === 1
              ? "Cited by 1 claim"
              : `Cited by ${citations.length} claims`}
          </span>
          <ul className="flex flex-col gap-1">
            {citations.map((row) => (
              <li
                key={`${row.section}-${row.claim}`}
                className="text-xs text-status-neutral"
              >
                <span className="text-text">{row.section}</span>
                <span className="font-mono"> · {row.claim}</span>
                {row.coverage === "gap" && (
                  <span className="text-status-warn">
                    {" "}
                    · no verified figure
                  </span>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      {detail.isPending && <LoadingSkeleton lines={4} label="Loading evidence" />}
      {detail.isError && (
        <div className="flex flex-col gap-2">
          <ErrorState error={detail.error} onRetry={() => detail.refetch()} />
          <p className="text-xs text-status-neutral">
            The report cites this id, but the store cannot resolve it. It may
            belong to a project that was removed.
          </p>
        </div>
      )}

      {detail.data && (
        <>
          <ArtifactWarnings warnings={detail.data.warnings ?? []} />
          <PayloadView payload={detail.data.payload} />
          <Link
            to={artifactPath(projectId, detail.data.session_id, artifactId)}
            className="text-xs text-primary underline-offset-2 hover:underline"
          >
            Open in Artifacts
          </Link>
        </>
      )}
    </aside>
  );
}
