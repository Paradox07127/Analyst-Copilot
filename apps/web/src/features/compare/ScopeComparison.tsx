import { Link } from "react-router";
import type {
  CompareScopeItem,
  CompareScopeName,
  CompareScopeRecord,
} from "../../api/client";
import { useCompareScope } from "../../api/hooks";
import {
  EmptyState,
  ErrorState,
  LoadingSkeleton,
} from "../../components/async-states";
import { Badge, Button, Card, type Tone } from "../../components/ui";

type Change = CompareScopeItem["change"];

const CHANGE_TONE: Record<Change, Tone> = {
  added: "ok",
  removed: "critical",
  changed: "warn",
  same: "neutral",
  unavailable: "critical",
};

function titleCase(value: string): string {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function changedField(item: CompareScopeItem, key: string): boolean {
  return (item.changed_fields ?? []).some(
    (field) => field === key || field === `payload.${key}`,
  );
}

function RecordSide({
  label,
  record,
  item,
  projectId,
  rootSessionId,
}: {
  label: string;
  record: CompareScopeRecord | null | undefined;
  item: CompareScopeItem;
  projectId: string;
  rootSessionId: string;
}) {
  if (!record) {
    return (
      <div className="min-w-0 rounded-base bg-surface/60 p-3">
        <p className="text-[10px] font-semibold tracking-wide text-status-neutral uppercase">
          {label}
        </p>
        <p className="mt-2 text-sm text-status-neutral">Not present</p>
      </div>
    );
  }

  const artifactLink =
    record.artifact_id && record.source_session_id === rootSessionId
      ? `/projects/${encodeURIComponent(projectId)}/sessions/${encodeURIComponent(record.source_session_id)}/artifacts?artifact=${encodeURIComponent(record.artifact_id)}`
      : null;
  return (
    <div className="min-w-0 rounded-base bg-surface/60 p-3">
      <p className="text-[10px] font-semibold tracking-wide text-status-neutral uppercase">
        {label}
      </p>
      <div className="mt-1 flex min-w-0 flex-col gap-2">
        <div>
          <p className="break-words text-sm font-medium">{record.title}</p>
          <p className="text-xs text-status-neutral">{record.kind}</p>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {record.status && <Badge variant="outline">{record.status}</Badge>}
          {(record.tags ?? []).map((tag) => (
            <Badge key={tag} variant="outline">
              {tag}
            </Badge>
          ))}
        </div>
        {record.summary && (
          <p className="text-sm text-status-neutral">{record.summary}</p>
        )}
        {(record.fields ?? []).length > 0 && (
          <dl className="grid grid-cols-[minmax(7rem,auto)_1fr] gap-x-3 gap-y-1.5 border-t border-hairline pt-2 text-xs">
            {(record.fields ?? []).map((field) => {
              const changed = changedField(item, field.key);
              return (
                <div key={field.key} className="contents">
                  <dt className={changed ? "font-medium text-status-warn" : "text-status-neutral"}>
                    {field.label}
                  </dt>
                  <dd
                    className={`min-w-0 break-words ${
                      field.value_kind === "code" ? "font-mono" : ""
                    } ${changed ? "font-medium" : ""}`}
                  >
                    {field.value}
                  </dd>
                </div>
              );
            })}
          </dl>
        )}
        <div className="flex flex-wrap items-center gap-2 text-xs text-status-neutral">
          <span className="font-mono">{record.source_session_id}</span>
          {artifactLink ? (
            <Link className="text-primary hover:underline" to={artifactLink}>
              Open artifact
            </Link>
          ) : record.artifact_id ? (
            <span className="font-mono">{record.artifact_id}</span>
          ) : null}
        </div>
      </div>
    </div>
  );
}

function ScopeItemCard({
  item,
  projectId,
  leftLabel,
  rightLabel,
  leftSessionId,
  rightSessionId,
}: {
  item: CompareScopeItem;
  projectId: string;
  leftLabel: string;
  rightLabel: string;
  leftSessionId: string;
  rightSessionId: string;
}) {
  return (
    <Card as="li" className="flex flex-col gap-3 p-3">
      <div className="flex flex-wrap items-center gap-2">
        <Badge tone={CHANGE_TONE[item.change]} caps>
          {item.change}
        </Badge>
        <Badge variant="outline">
          {item.match_status === "unmatched"
            ? "unmatched"
            : `${item.match_status} match`}
        </Badge>
        <span className="text-xs text-status-neutral">
          {item.reason} · {item.confidence} confidence
        </span>
      </div>
      <div className="grid gap-3 lg:grid-cols-2">
        <RecordSide
          label={leftLabel}
          record={item.left}
          item={item}
          projectId={projectId}
          rootSessionId={leftSessionId}
        />
        <RecordSide
          label={rightLabel}
          record={item.right}
          item={item}
          projectId={projectId}
          rootSessionId={rightSessionId}
        />
      </div>
      {(item.changed_fields ?? []).length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5 border-t border-hairline pt-2 text-xs text-status-neutral">
          <span>Changed:</span>
          {(item.changed_fields ?? []).map((field) => (
            <Badge key={field} tone="warn" variant="outline">
              {field.replaceAll("_", " ")}
            </Badge>
          ))}
        </div>
      )}
    </Card>
  );
}

export function ScopeComparison({
  scope,
  leftSessionId,
  rightSessionId,
  leftLabel,
  rightLabel,
  filter,
}: {
  scope: CompareScopeName;
  leftSessionId: string;
  rightSessionId: string;
  leftLabel: string;
  rightLabel: string;
  filter: "all" | "differences";
}) {
  const query = useCompareScope(scope, leftSessionId, rightSessionId, filter);
  if (query.isPending) {
    return <LoadingSkeleton lines={7} label={`Loading ${scope} comparison`} />;
  }
  if (query.isError) {
    return <ErrorState error={query.error} onRetry={() => void query.refetch()} />;
  }

  const first = query.data.pages[0];
  if (!first) {
    return (
      <EmptyState
        title={`No ${scope} comparison response`}
        description="The comparison request completed without a result page."
      />
    );
  }
  const items = query.data.pages.flatMap((page) => page.items ?? []);
  const counts = first.counts;
  const states = [
    { label: leftLabel, value: first.left_state },
    { label: rightLabel, value: first.right_state },
  ].filter((side) => side.value.state !== "value");

  return (
    <section aria-labelledby={`${scope}-comparison-title`} className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <h2 id={`${scope}-comparison-title`} className="text-base font-semibold">
          {titleCase(scope)} comparison
        </h2>
        {(Object.keys(counts) as Change[]).map((change) => (
          <Badge key={change} tone={CHANGE_TONE[change]}>
            {counts[change]} {change}
          </Badge>
        ))}
      </div>
      <p className="text-xs text-status-neutral">
        Deterministic matching over each session result family. Missing and unavailable data are never treated as zero.
      </p>
      {states.map((side) => (
        <Card key={side.label} tone={side.value.state === "unavailable" ? "critical" : "quiet"} className="p-3">
          <p className="text-sm font-medium">
            {side.label} · {side.value.state}
          </p>
          {side.value.reason && (
            <p className="mt-1 text-xs text-status-neutral">{side.value.reason}</p>
          )}
        </Card>
      ))}
      {(first.warnings ?? []).length > 0 && (
        <Card tone="warn" className="p-3">
          <p className="text-sm font-medium">Comparison warnings</p>
          <ul className="mt-1 list-inside list-disc text-xs text-status-neutral">
            {(first.warnings ?? []).map((warning) => (
              <li key={warning}>{warning}</li>
            ))}
          </ul>
        </Card>
      )}
      {items.length === 0 ? (
        <EmptyState
          title={filter === "differences" ? `No ${scope} differences` : `No ${scope} records to compare`}
          description={
            filter === "differences"
              ? "Every deterministically matched record is the same."
              : "Neither side produced a comparable record for this scope."
          }
        />
      ) : (
        <ol className="flex flex-col gap-3">
          {items.map((item) => (
            <ScopeItemCard
              key={item.match_key}
              item={item}
              projectId={first.project_id}
              leftLabel={leftLabel}
              rightLabel={rightLabel}
              leftSessionId={leftSessionId}
              rightSessionId={rightSessionId}
            />
          ))}
        </ol>
      )}
      {query.hasNextPage && (
        <Button
          className="self-center"
          disabled={query.isFetchingNextPage}
          onClick={() => void query.fetchNextPage()}
        >
          {query.isFetchingNextPage ? "Loading…" : "Load more differences"}
        </Button>
      )}
    </section>
  );
}
