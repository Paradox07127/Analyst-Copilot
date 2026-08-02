/* Artifact browser — the evidence store every claim in the Report cites.
 * `?artifact=<id>` is a deep link other pages point evidence at: the row opens
 * and scrolls into view, and when the id is not on the pages loaded so far it
 * is fetched by id instead (GET /artifacts/{id}), which also covers evidence
 * that lives in another run. */

import { useEffect, useMemo, useRef, useState } from "react";
import { useParams, useSearchParams } from "react-router";
import type { ArtifactSummary } from "../../api/client";
import { useArtifact, useArtifacts } from "../../api/hooks";
import {
  EmptyState,
  ErrorState,
  LoadingSkeleton,
} from "../../components/async-states";
import {
  Badge,
  Card,
  Chevron,
  Marquee,
  SectionHeader,
} from "../../components/ui";
import {
  ArtifactWarnings,
  formatCreatedAt,
  PayloadView,
} from "./artifact-payload";

const ARTIFACT_PARAM = "artifact";
const TYPE_PARAM = "type";
const QUERY_PARAM = "q";

const ARTIFACT_TYPE_LABEL: Record<string, string> = {
  ChartSpec: "Chart",
  DatasetProfile: "Dataset profile",
  QualityIssueSet: "Quality issues",
  RelationshipValidationSet: "Relationship validation",
  SqlResult: "Query result",
};

function artifactTypeLabel(type: string): string {
  return (
    ARTIFACT_TYPE_LABEL[type] ??
    type
      .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
      .replaceAll("_", " ")
  );
}

function ArtifactItem({
  artifact,
  sessionId,
  linked = false,
}: {
  artifact: ArtifactSummary;
  sessionId: string;
  linked?: boolean;
}) {
  const [open, setOpen] = useState(linked);
  const detail = useArtifact(sessionId, artifact.artifact_id, open);
  const item = useRef<HTMLLIElement>(null);
  const created = formatCreatedAt(artifact.created_at);

  /* Re-runs when the deep link changes while the page stays mounted, so a
   * second evidence click still opens and reveals its row. */
  useEffect(() => {
    if (!linked) return;
    setOpen(true);
    item.current?.scrollIntoView?.({ block: "center" });
  }, [linked, artifact.artifact_id]);

  return (
    <li
      ref={item}
      className={`rounded-base border ${
        linked ? "border-primary bg-primary/5" : "border-border bg-bg"
      }`}
    >
      <button
        type="button"
        onClick={() => setOpen((current) => !current)}
        aria-expanded={open}
        className="flex w-full items-center gap-2 px-3 py-2 text-left hover:bg-surface"
      >
        <Chevron open={open} />
        <Marquee className="min-w-0 flex-1 font-mono text-sm">
          {artifact.artifact_id}
        </Marquee>
        {created && (
          <span className="tabular shrink-0 text-xs text-status-neutral">
            {created}
          </span>
        )}
        <Badge tone="neutral">{artifactTypeLabel(artifact.type)}</Badge>
      </button>
      {open && (
        <div className="flex flex-col gap-2 border-t border-hairline px-3 py-2">
          {detail.isPending && (
            <LoadingSkeleton lines={3} label="Loading artifact" />
          )}
          {detail.isError && (
            <ErrorState error={detail.error} onRetry={() => detail.refetch()} />
          )}
          {detail.data && (
            <>
              <ArtifactWarnings warnings={detail.data.warnings ?? []} />
              <PayloadView payload={detail.data.payload} />
            </>
          )}
        </div>
      )}
    </li>
  );
}

/* Shown when the linked id is not among the loaded rows: either it sits past
 * the current page, the type filter hides it, or it belongs to another run. */
function LinkedArtifactPanel({
  artifactId,
  sessionId,
  onDismiss,
}: {
  artifactId: string;
  sessionId: string;
  onDismiss: () => void;
}) {
  const detail = useArtifact(sessionId, artifactId, true);
  const otherRun = detail.data != null && detail.data.session_id !== sessionId;

  return (
    <section
      aria-label="Linked artifact"
      className="flex flex-col gap-2 rounded-base border border-primary bg-primary/5 p-3"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <Marquee className="font-mono text-sm">{artifactId}</Marquee>
        <div className="flex items-center gap-2">
          {detail.data && (
            <Badge tone="neutral">{artifactTypeLabel(detail.data.type)}</Badge>
          )}
          <button
            type="button"
            onClick={onDismiss}
            className="rounded-base border border-border px-2 py-0.5 text-xs hover:bg-surface"
          >
            Dismiss
          </button>
        </div>
      </div>
      {otherRun && (
        <p className="text-xs text-status-warn">
          {`This artifact belongs to session ${detail.data?.session_id}, not the one you are viewing.`}
        </p>
      )}
      {!otherRun && detail.data && (
        <p className="text-xs text-status-neutral">
          Not on the loaded page of this session — fetched by id.
        </p>
      )}
      {detail.isPending && (
        <LoadingSkeleton lines={3} label="Loading linked artifact" />
      )}
      {detail.isError && (
        <ErrorState error={detail.error} onRetry={() => detail.refetch()} />
      )}
      {detail.data && (
        <>
          <ArtifactWarnings warnings={detail.data.warnings ?? []} />
          <PayloadView payload={detail.data.payload} />
        </>
      )}
    </section>
  );
}

export function Component() {
  const { sessionId = "" } = useParams();
  const [searchParams, setSearchParams] = useSearchParams();
  const linkedId = searchParams.get(ARTIFACT_PARAM) ?? "";
  const type = searchParams.get(TYPE_PARAM) || undefined;
  const query = searchParams.get(QUERY_PARAM) ?? "";
  /* The unfiltered query stays mounted so the dropdown keeps offering every
   * type seen so far even while a server-side filter is active. */
  const allArtifacts = useArtifacts(sessionId, undefined);
  const filteredArtifacts = useArtifacts(sessionId, type);
  const artifacts = type ? filteredArtifacts : allArtifacts;

  const typeOptions = useMemo(() => {
    const types = new Set<string>();
    for (const page of allArtifacts.data?.pages ?? []) {
      for (const item of page.items) types.add(item.type);
    }
    return [...types].sort();
  }, [allArtifacts.data]);

  const loaded = artifacts.data?.pages.flatMap((page) => page.items) ?? [];
  const needle = query.trim().toLowerCase();
  const items = needle
    ? loaded.filter((item) =>
        item.artifact_id.toLowerCase().includes(needle),
      )
    : loaded;
  const linkedIsListed = loaded.some((item) => item.artifact_id === linkedId);

  const dismissLink = () =>
    setSearchParams(
      (current) => {
        const next = new URLSearchParams(current);
        next.delete(ARTIFACT_PARAM);
        return next;
      },
      { replace: true },
    );

  const updateFilter = (key: typeof TYPE_PARAM | typeof QUERY_PARAM, value: string) =>
    setSearchParams(
      (current) => {
        const next = new URLSearchParams(current);
        const normalized = value.trim();
        if (normalized) next.set(key, value);
        else next.delete(key);
        return next;
      },
      { replace: true },
    );

  return (
    <div className="mx-auto flex w-[95%] max-w-data flex-col gap-4 p-6">
      <SectionHeader
        level={1}
        title="Artifacts"
        description="Every figure the agent published points at one of these records. Open one to read exactly what it holds."
      />

      <Card tone="quiet" className="flex flex-wrap items-center gap-x-4 gap-y-2 px-3 py-2">
        <label className="flex items-center gap-2 text-sm">
          <span className="text-status-neutral">Type</span>
          <select
            value={type ?? ""}
            onChange={(event) => updateFilter(TYPE_PARAM, event.target.value)}
            className="rounded-base border border-border bg-bg px-2 py-1 text-sm"
          >
            <option value="">All types</option>
            {typeOptions.map((option) => (
              <option key={option} value={option}>
                {artifactTypeLabel(option)}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2 text-sm">
          <span className="text-status-neutral">Find id</span>
          <input
            type="search"
            value={query}
            onChange={(event) => updateFilter(QUERY_PARAM, event.target.value)}
            placeholder="paste a cited id"
            className="rounded-base border border-border bg-bg px-2 py-1 font-mono text-sm"
          />
        </label>
        <span className="tabular ml-auto text-xs text-status-neutral">
          {needle
            ? `${items.length} of ${loaded.length} loaded`
            : `${loaded.length} loaded${artifacts.hasNextPage ? "+" : ""}`}
        </span>
        <span className="w-full text-xs text-status-neutral">
          Type filters the session on the server. ID search checks only the
          records loaded below.
        </span>
      </Card>

      {linkedId && artifacts.data && !linkedIsListed && (
        <LinkedArtifactPanel
          artifactId={linkedId}
          sessionId={sessionId}
          onDismiss={dismissLink}
        />
      )}

      {artifacts.isPending && (
        <LoadingSkeleton lines={5} label="Loading artifacts" />
      )}
      {artifacts.isError && (
        <ErrorState
          error={artifacts.error}
          onRetry={() => artifacts.refetch()}
        />
      )}
      {artifacts.data &&
        (loaded.length === 0 ? (
          <EmptyState
            title="No artifacts"
            description={
              type
                ? "No artifacts of this type in this session."
                : "Run an analysis to produce artifacts for this session. Profiles, quality issue sets, charts and tables all land here, and the Report cites them by id."
            }
          />
        ) : items.length === 0 ? (
          <EmptyState
            title="No id matches"
            description="No loaded artifact id contains that text. Load more pages, or clear the type filter."
          />
        ) : (
          <>
            <ul className="flex flex-col gap-2">
              {items.map((artifact) => (
                <ArtifactItem
                  key={artifact.artifact_id}
                  artifact={artifact}
                  sessionId={sessionId}
                  linked={artifact.artifact_id === linkedId}
                />
              ))}
            </ul>
            {artifacts.hasNextPage && (
              <button
                type="button"
                onClick={() => artifacts.fetchNextPage()}
                disabled={artifacts.isFetchingNextPage}
                className="self-start rounded-base border border-border px-3 py-1.5 text-sm hover:bg-surface disabled:opacity-60"
              >
                {artifacts.isFetchingNextPage ? "Loading…" : "Load more"}
              </button>
            )}
          </>
        ))}
    </div>
  );
}
