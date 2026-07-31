/* Relationships canvas (§10.1): dataset nodes plus the run's relationship
 * edges in three states — candidate (scored only), validated (DuckDB verified)
 * and confirmed (promoted to a join). Selecting an edge opens the Inspector
 * with its evidence and the Validate / Confirm / Revoke actions. Validation
 * reads both source tables, so it runs as a background job the activity
 * drawer tracks — and so does discovery, which most runs defer and therefore
 * have to trigger from here.
 *
 * The canvas draws one edge per *table pair* of the filtered set, not one per
 * candidate: a run with 9 tables and 288 column-level candidates is an
 * unreadable hairball at candidate granularity, and the pair is the unit a
 * user actually navigates by. */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Background,
  Controls,
  MarkerType,
  MiniMap,
  ReactFlow,
  useNodesState,
  type Edge,
  type Node,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import "./relationships-flow.css";
import {
  api,
  type RelationshipDiscoveryStarted,
  type RelationshipEdge,
  type RelationshipGraphView,
  type RelationshipNode,
  type RelationshipValidationPrepared,
  type RelationshipValidationStarted,
} from "../../api/client";
import { queryKeys, useRelationships } from "../../api/hooks";
import { approvalGuidanceText } from "../../api/stale-approval";
import { useJobActivity } from "../../app/job-activity";
import { artifactPath, sessionSectionPath } from "../../app/paths";
import {
  parseCsvParam,
  serializeCsvParam,
  useRouteSearchParam,
  useSetRouteSearchParams,
} from "../../app/route-state";
import {
  EmptyState,
  ErrorState,
  LoadingSkeleton,
} from "../../components/async-states";
import {
  Badge,
  Card,
  Disclosure,
  Dot,
  Hint,
  Marquee,
  MetricStrip,
  MetricTile,
  SectionHeader,
  formatCompact,
  formatPercent,
  type Tone,
} from "../../components/ui";
import { useDialogFocus } from "../../components/use-dialog-focus";

const NODE_WIDTH = 196;
const KEYBOARD_NODE_STEP = 24;
/* Above this many candidates in scope, pair groups open on demand instead of
 * all at once — an 80-row wall of columns is the thing the grouping fixes. */
const AUTO_EXPAND_LIMIT = 12;

type EdgeState = "candidate" | "validated" | "confirmed";

/* Semantic tokens, not raw colors: the canvas follows the active theme. */
const EDGE_STROKE: Record<EdgeState, string> = {
  candidate: "var(--color-status-neutral)",
  validated: "var(--color-status-info)",
  confirmed: "var(--color-status-ok)",
};

const STATE_TONE: Record<EdgeState, Tone> = {
  candidate: "neutral",
  validated: "info",
  confirmed: "ok",
};

const STATE_RANK: Record<EdgeState, number> = {
  candidate: 0,
  validated: 1,
  confirmed: 2,
};

const EDGE_LEGEND: { state: EdgeState; label: string; hint: string }[] = [
  { state: "candidate", label: "Candidate", hint: "scored only — dashed" },
  { state: "validated", label: "Validated", hint: "DuckDB verified" },
  { state: "confirmed", label: "Confirmed", hint: "usable as a join" },
];

function edgeState(edge: RelationshipEdge): EdgeState {
  return edge.state === "confirmed" || edge.state === "validated"
    ? edge.state
    : "candidate";
}

function percent(value: number | null | undefined): string {
  return value == null ? "—" : formatPercent(value);
}

function columnList(columns: string[] | undefined): string {
  return (columns ?? []).join(", ") || "—";
}

/* Screen-reader and test-facing identity of a row: the dotted path string is
 * unreadable on screen but the pair still needs one flat name. */
function edgeName(edge: RelationshipEdge): string {
  return `${edge.left_dataset}.${columnList(edge.left_columns)} → ${
    edge.right_dataset
  }.${columnList(edge.right_columns)}, ${edge.confidence} confidence, ${edgeState(
    edge,
  )}`;
}

/* Nodes on a circle rather than a grid: with 9 tables a grid runs edges
 * straight through the boxes between them. */
function layoutPositions(count: number): { x: number; y: number }[] {
  if (count <= 2) {
    return Array.from({ length: count }, (_, index) => ({
      x: index * (NODE_WIDTH + 140),
      y: 0,
    }));
  }
  const radius = Math.max(240, (count * (NODE_WIDTH + 70)) / (2 * Math.PI));
  return Array.from({ length: count }, (_, index) => {
    const angle = (2 * Math.PI * index) / count - Math.PI / 2;
    return {
      x: Math.round(radius * Math.cos(angle)),
      y: Math.round(radius * Math.sin(angle)),
    };
  });
}

function nodeLabel(node: RelationshipNode) {
  return (
    <div className="flex flex-col gap-0.5 text-left">
      <Marquee className="block text-sm font-semibold" title={node.name}>
        {node.name}
      </Marquee>
      <span className="tabular block text-xs text-status-neutral">
        {node.row_count != null
          ? `${formatCompact(node.row_count)} rows`
          : "rows unknown"}
        {" · "}
        {node.column_count} cols
      </span>
    </div>
  );
}

function toNodes(nodes: RelationshipNode[], connected: Set<string>): Node[] {
  const positions = layoutPositions(nodes.length);
  return nodes.map((node, index) => ({
    id: node.dataset_id,
    position: positions[index] ?? { x: 0, y: 0 },
    data: { label: nodeLabel(node) },
    style: { width: NODE_WIDTH },
    className: connected.has(node.dataset_id) ? "" : "rel-node-muted",
  }));
}

type Pair = {
  key: string;
  leftId: string;
  rightId: string;
  leftName: string;
  rightName: string;
  edges: RelationshipEdge[];
  best: number;
  state: EdgeState;
};

function groupByPair(edges: RelationshipEdge[]): Pair[] {
  const pairs = new Map<string, Pair>();
  for (const edge of edges) {
    const key = `${edge.left_dataset_id}→${edge.right_dataset_id}`;
    const state = edgeState(edge);
    const existing = pairs.get(key);
    if (existing) {
      existing.edges.push(edge);
      existing.best = Math.max(existing.best, edge.ensemble_score);
      if (STATE_RANK[state] > STATE_RANK[existing.state]) existing.state = state;
    } else {
      pairs.set(key, {
        key,
        leftId: edge.left_dataset_id,
        rightId: edge.right_dataset_id,
        leftName: edge.left_dataset,
        rightName: edge.right_dataset,
        edges: [edge],
        best: edge.ensemble_score,
        state,
      });
    }
  }
  const list = [...pairs.values()];
  for (const pair of list) {
    pair.edges.sort((left, right) => right.ensemble_score - left.ensemble_score);
  }
  return list.sort((left, right) => right.best - left.best);
}

function toGraphEdges(pairs: Pair[], focusKey: string | null): Edge[] {
  return pairs.map((pair) => {
    const focused = pair.key === focusKey;
    const only = pair.edges.length === 1 ? pair.edges[0] : undefined;
    return {
      id: pair.key,
      source: pair.leftId,
      target: pair.rightId,
      label: only
        ? columnList(only.left_columns)
        : `${pair.edges.length} candidates`,
      markerEnd: { type: MarkerType.ArrowClosed, color: EDGE_STROKE[pair.state] },
      style: {
        stroke: focused ? "var(--color-primary)" : EDGE_STROKE[pair.state],
        strokeWidth: focused ? 3 : pair.state === "candidate" ? 1.5 : 2,
        strokeDasharray: pair.state === "candidate" ? "6 4" : undefined,
      },
      labelStyle: { fill: "var(--color-text)", fontSize: 11 },
      labelBgStyle: { fill: "var(--color-surface)" },
    };
  });
}

function Legend() {
  return (
    <ul className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-status-neutral">
      {EDGE_LEGEND.map((item) => (
        <li key={item.state} className="flex items-center gap-2">
          <span
            aria-hidden
            className="inline-block h-0 w-8 border-t-2"
            style={{
              borderColor: EDGE_STROKE[item.state],
              borderTopStyle: item.state === "candidate" ? "dashed" : "solid",
            }}
          />
          <span className="text-text">{item.label}</span>
          <span>{item.hint}</span>
        </li>
      ))}
    </ul>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <dt className="text-xs text-status-neutral">{label}</dt>
      <dd className="tabular font-mono text-xs">{value}</dd>
    </div>
  );
}

/** Table + column on each side with the direction, replacing the unreadable
 *  `left.csv.col -> right.csv.col` path string. */
function RelationshipPair({ edge }: { edge: RelationshipEdge }) {
  const left = columnList(edge.left_columns);
  const right = columnList(edge.right_columns);
  return (
    <span className="flex min-w-0 flex-col gap-0.5">
      <span className="flex min-w-0 items-baseline gap-1.5">
        <Marquee className="min-w-0 flex-1 font-mono text-sm">
          {left}
        </Marquee>
        <span aria-hidden className="shrink-0 text-status-neutral">
          →
        </span>
        <Marquee className="min-w-0 flex-1 font-mono text-sm">
          {right}
        </Marquee>
      </span>
      <span className="flex min-w-0 items-baseline gap-1.5 text-xs text-status-neutral">
        <Marquee className="min-w-0 flex-1" title={edge.left_dataset}>
          {edge.left_dataset}
        </Marquee>
        <span aria-hidden className="shrink-0">
          →
        </span>
        <Marquee className="min-w-0 flex-1" title={edge.right_dataset}>
          {edge.right_dataset}
        </Marquee>
      </span>
    </span>
  );
}

function ReviewProgress({ state }: { state: EdgeState }) {
  const activeIndex = STATE_RANK[state];
  const stages = [
    { label: "Candidate", detail: "Review signals" },
    { label: "Validated", detail: "Check full tables" },
    { label: "Available join", detail: "Adopt project-wide" },
  ];

  return (
    <ol
      aria-label="Relationship review progress"
      className="grid grid-cols-3 gap-1"
    >
      {stages.map((stage, index) => {
        const complete = index < activeIndex;
        const active = index === activeIndex;
        return (
          <li
            key={stage.label}
            className={`min-w-0 rounded-base border px-2 py-1.5 ${
              active
                ? "border-primary bg-primary/10"
                : complete
                  ? "border-status-ok/40 bg-status-ok/10"
                  : "border-border bg-bg"
            }`}
          >
            <Marquee
              className={`text-xs font-medium ${
                active
                  ? "text-primary"
                  : complete
                    ? "text-status-ok"
                    : "text-text"
              }`}
            >
              {index + 1}. {stage.label}
            </Marquee>
            <Marquee className="block text-[10px] text-status-neutral">
              {stage.detail}
            </Marquee>
          </li>
        );
      })}
    </ol>
  );
}

function staleApprovalHint(error: unknown): string | null {
  return approvalGuidanceText(error, {
    approval_expired: "The approval expired. Prepare the validation again.",
    approval_consumed:
      "This approval was already used — check the activity drawer.",
    job_conflict:
      "The retry key was already used by another job. Prepare again.",
  });
}

/* Candidate filters: a min-score slider (0-1, step 0.01,
 * default 0) plus a confidence multiselect (high/medium/low, default
 * high+medium). The filter now drives the canvas as well as the list. */
const CONFIDENCE_LEVELS = ["high", "medium", "low"] as const;

function CandidateFilters({
  minimumScore,
  onMinimumScoreChange,
  confidenceLevels,
  onToggleConfidence,
}: {
  minimumScore: number;
  onMinimumScoreChange: (value: number) => void;
  confidenceLevels: Set<string>;
  onToggleConfidence: (level: string) => void;
}) {
  return (
    <div className="flex flex-col gap-2 text-xs">
      <label className="flex flex-col gap-1">
        <span className="text-status-neutral">
          Minimum score: {minimumScore.toFixed(2)}
        </span>
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={minimumScore}
          onChange={(event) => onMinimumScoreChange(Number(event.target.value))}
        />
      </label>
      <fieldset className="flex flex-wrap items-center gap-3">
        <legend className="text-status-neutral">Confidence</legend>
        {CONFIDENCE_LEVELS.map((level) => (
          <label key={level} className="flex items-center gap-1">
            <input
              type="checkbox"
              checked={confidenceLevels.has(level)}
              onChange={() => onToggleConfidence(level)}
            />
            {level}
          </label>
        ))}
      </fieldset>
    </div>
  );
}

/** What the filter is hiding, in the same terms the filter uses — otherwise
 *  "Showing 9 of 288" leaves the other 279 unaccounted for. */
function ScopeReadout({
  edges,
  visibleCount,
  minimumScore,
  confidenceLevels,
  pairsInScope,
  pairsTotal,
}: {
  edges: RelationshipEdge[];
  visibleCount: number;
  minimumScore: number;
  confidenceLevels: Set<string>;
  pairsInScope: number;
  pairsTotal: number;
}) {
  const belowScore = edges.filter(
    (edge) => edge.ensemble_score < minimumScore,
  ).length;
  const offConfidence = edges.filter(
    (edge) =>
      edge.ensemble_score >= minimumScore && !confidenceLevels.has(edge.confidence),
  ).length;
  const tally = CONFIDENCE_LEVELS.map(
    (level) => `${edges.filter((edge) => edge.confidence === level).length} ${level}`,
  ).join(" · ");

  return (
    <div className="flex flex-col gap-1">
      <p className="text-xs text-status-neutral">
        Showing {visibleCount} of {edges.length} candidates.
      </p>
      {belowScore + offConfidence > 0 && (
        <p className="text-xs text-status-neutral">
          Hidden: {belowScore} below score {minimumScore.toFixed(2)} ·{" "}
          {offConfidence} outside the confidence filter.
        </p>
      )}
      <p className="text-xs text-status-neutral">
        Scored set: {tally}. Across {pairsInScope} of {pairsTotal} table pair(s).
      </p>
    </div>
  );
}

function EdgeRow({
  edge,
  selected,
  onSelect,
}: {
  edge: RelationshipEdge;
  selected: boolean;
  onSelect: (id: string) => void;
}) {
  const state = edgeState(edge);
  return (
    <li>
      <button
        type="button"
        onClick={() => onSelect(edge.relationship_id)}
        aria-current={selected}
        aria-label={edgeName(edge)}
        className={`flex w-full flex-col gap-1.5 rounded-base border px-2 py-1.5 text-left hover:bg-bg ${
          selected ? "border-primary" : "border-transparent"
        }`}
      >
        <RelationshipPair edge={edge} />
        <span className="flex items-center gap-2">
          <Badge tone={STATE_TONE[state]} caps>
            {state}
          </Badge>
          <span className="text-xs text-status-neutral">
            {edge.confidence} confidence
          </span>
          <span className="tabular ml-auto text-xs">
            {edge.ensemble_score.toFixed(2)}
          </span>
        </span>
      </button>
    </li>
  );
}

/* Two levels only (NN/g): table pair -> its column candidates. The Inspector
 * is a pane, not a third level. */
function PairGroup({
  pair,
  expanded,
  selectedId,
  onSelect,
}: {
  pair: Pair;
  expanded: boolean;
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  return (
    <Disclosure
      defaultOpen={expanded}
      summary={
        <span className="flex min-w-0 items-center gap-2">
          <Dot tone={STATE_TONE[pair.state]} />
          <Marquee className="min-w-0 text-xs">
            {pair.leftName} → {pair.rightName}
          </Marquee>
        </span>
      }
      meta={`${pair.edges.length} · best ${pair.best.toFixed(2)}`}
    >
      <ul className="flex flex-col gap-1 border-l border-hairline pl-2">
        {pair.edges.map((edge) => (
          <EdgeRow
            key={edge.relationship_id}
            edge={edge}
            selected={edge.relationship_id === selectedId}
            onSelect={onSelect}
          />
        ))}
      </ul>
    </Disclosure>
  );
}

function EdgeInspector({
  projectId,
  sessionId,
  edge,
  seedsVersion,
  onClose,
}: {
  projectId: string;
  sessionId: string;
  edge: RelationshipEdge;
  seedsVersion: number;
  onClose: () => void;
}) {
  const { startTracking } = useJobActivity();
  const queryClient = useQueryClient();
  /* One idempotency key per prepared approval: Confirm retries replay the same
   * key (and job), while a fresh prepare binds a fresh key. */
  const [validateKey, setValidateKey] = useState("");

  const invalidate = useCallback(
    () =>
      void queryClient.invalidateQueries({
        queryKey: queryKeys.relationships(sessionId),
      }),
    [queryClient, sessionId],
  );

  const prepare = useMutation({
    mutationFn: () =>
      api.prepareRelationshipValidation(sessionId, edge.relationship_id),
    onSuccess: () => {
      validate.reset();
      setValidateKey(crypto.randomUUID());
    },
  });

  const validate = useMutation({
    mutationFn: (prepared: RelationshipValidationPrepared) =>
      api.validateRelationship(
        sessionId,
        edge.relationship_id,
        {
          action_hash: prepared.action_hash,
          approval_token: prepared.approval_token,
        },
        validateKey,
      ),
    onSuccess: (started: RelationshipValidationStarted) => {
      prepare.reset();
      startTracking({
        jobId: started.job.job_id,
        sessionId: started.execution_session_id,
        sourceSessionId: sessionId,
        projectId,
        eventsUrl: started.job.events_url,
      });
      invalidate();
    },
  });

  const confirm = useMutation({
    mutationFn: () =>
      api.confirmRelationship(sessionId, edge.relationship_id, seedsVersion),
    onSuccess: invalidate,
  });

  const revoke = useMutation({
    mutationFn: () =>
      api.revokeRelationship(sessionId, edge.relationship_id, seedsVersion),
    onSuccess: invalidate,
  });

  const staleHint = staleApprovalHint(validate.error);
  const validationDialog = useDialogFocus(() => prepare.reset());
  const state = edgeState(edge);

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-start justify-between gap-2">
        <div className="flex min-w-0 flex-col gap-1.5">
          <RelationshipPair edge={edge} />
          <span className="flex flex-wrap items-center gap-2">
            <Badge tone={STATE_TONE[state]} caps>
              {state}
            </Badge>
            {edge.join_status && (
              <Badge>{`join: ${edge.join_status}`}</Badge>
            )}
          </span>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="shrink-0 rounded-base border border-border px-2 py-0.5 text-xs hover:bg-bg"
        >
          Back to candidates
        </button>
      </div>

      <ReviewProgress state={state} />

      <Card tone="quiet" className="flex flex-col gap-1 p-2">
        <dl className="flex flex-col gap-1">
          <Row label="Left columns" value={columnList(edge.left_columns)} />
          <Row label="Right columns" value={columnList(edge.right_columns)} />
          <Row
            label="Confidence"
            value={`${edge.confidence} (${edge.ensemble_score.toFixed(3)})`}
          />
          <Row
            label="Overlap left→right"
            value={percent(edge.overlap_left_in_right)}
          />
          <Row
            label="Overlap right→left"
            value={percent(edge.overlap_right_in_left)}
          />
          <Row label="Right unique rate" value={percent(edge.right_unique_rate)} />
          <Row label="Cardinality" value={edge.cardinality ?? "not validated"} />
          <Row
            label="Join row multiplier"
            value={
              edge.join_row_multiplier == null
                ? "—"
                : `×${edge.join_row_multiplier.toFixed(2)}`
            }
          />
          <Row label="Orphan rate left" value={percent(edge.orphan_rate_left)} />
          <Row label="Orphan rate right" value={percent(edge.orphan_rate_right)} />
          <Row label="Validation freshness" value={edge.freshness ?? "—"} />
          {(edge.signals_sampled || edge.validation_sampled) && (
            <Row label="Sampled" value="yes — figures are estimates" />
          )}
        </dl>
      </Card>

      {(edge.warnings ?? []).length > 0 && (
        <Card tone="warn" className="p-2">
          <ul className="flex flex-col gap-1 text-xs text-status-warn">
            {(edge.warnings ?? []).map((warning) => (
              <li key={warning}>{warning}</li>
            ))}
          </ul>
        </Card>
      )}

      <div className="flex flex-col gap-1 text-xs">
        <span className="text-status-neutral">Evidence</span>
        {edge.candidate_artifact_id && (
          <Link
            to={artifactPath(projectId, sessionId, edge.candidate_artifact_id)}
            className="font-mono text-primary underline-offset-2 hover:underline"
          >
            {edge.candidate_artifact_id}
          </Link>
        )}
        {edge.validation_artifact_id && (
          <Link
            to={artifactPath(projectId, sessionId, edge.validation_artifact_id)}
            className="font-mono text-primary underline-offset-2 hover:underline"
          >
            {edge.validation_artifact_id}
          </Link>
        )}
        {!edge.candidate_artifact_id && !edge.validation_artifact_id && (
          <span className="text-status-neutral">No evidence artifact yet.</span>
        )}
      </div>

      {edge.verification_sql && (
        <details className="rounded-base border border-border p-2 text-xs">
          <summary className="cursor-pointer text-status-neutral">
            Verification SQL
          </summary>
          <pre className="mt-2 overflow-x-auto rounded-base bg-code-bg p-2 font-mono text-[11px]">
            {edge.verification_sql}
          </pre>
        </details>
      )}

      {prepare.data && !staleHint ? (
        <div
          ref={validationDialog.dialogRef}
          role="alertdialog"
          aria-label="Confirm relationship validation"
          onKeyDown={validationDialog.onKeyDown}
          className="flex flex-col gap-2 rounded-base border border-status-warn/50 p-3 text-sm"
        >
          <p className="font-medium">Validate this relationship?</p>
          <p className="text-xs text-status-neutral">
            Both source tables are read in full to measure cardinality and
            orphan rates. No model call — the work runs as a background job on a
            derived session, and the result lands on this session's graph.
          </p>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => validate.mutate(prepare.data)}
              disabled={validate.isPending}
              className="rounded-base bg-primary px-3 py-1.5 text-sm font-medium text-bg hover:opacity-90 disabled:opacity-50"
            >
              {validate.isPending ? "Starting…" : "Confirm & validate"}
            </button>
            <button
              type="button"
              onClick={() => prepare.reset()}
              disabled={validate.isPending}
              className="rounded-base border border-border px-3 py-1.5 text-sm hover:bg-bg"
            >
              Cancel
            </button>
          </div>
        </div>
      ) : (
        <div className="flex flex-col gap-2">
          {state === "candidate" &&
            (edge.can_validate ? (
              <button
                type="button"
                onClick={() => prepare.mutate()}
                disabled={prepare.isPending}
                className="self-start rounded-base bg-primary px-3 py-1.5 text-sm font-medium text-bg hover:opacity-90 disabled:opacity-50"
              >
                {prepare.isPending
                  ? "Preparing…"
                  : "Validate against full tables"}
              </button>
            ) : (
              <p className="text-xs text-status-neutral">
                This candidate cannot be validated because one of its source
                tables is no longer available.
              </p>
            ))}
          {state === "validated" &&
            (edge.can_confirm ? (
              <>
                <button
                  type="button"
                  onClick={() => confirm.mutate()}
                  disabled={confirm.isPending}
                  className="self-start rounded-base bg-primary px-3 py-1.5 text-sm font-medium text-bg hover:opacity-90 disabled:opacity-50"
                >
                  {confirm.isPending ? "Adding…" : "Use as project join"}
                </button>
                <p className="text-xs text-status-neutral">
                  This makes the verified join available to cross-table
                  analysis throughout the project.
                </p>
              </>
            ) : (
              <p className="text-xs text-status-neutral">
                This validation cannot be adopted. Only fresh, non
                many-to-many joins are eligible.
              </p>
            ))}
          {state === "confirmed" &&
            (edge.can_revoke ? (
              <button
                type="button"
                onClick={() => revoke.mutate()}
                disabled={revoke.isPending}
                className="self-start rounded-base border border-border px-3 py-1.5 text-sm hover:bg-bg disabled:opacity-50"
              >
                {revoke.isPending ? "Removing…" : "Remove from project joins"}
              </button>
            ) : (
              <p className="text-xs text-status-neutral">
                This relationship is available to cross-table analysis. Review
                the project-wide join policy on Knowledge.
              </p>
            ))}
        </div>
      )}

      {staleHint && (
        <div
          role="alert"
          className="flex flex-col gap-2 rounded-base border border-status-warn/50 p-3 text-sm"
        >
          <p className="text-status-warn">{staleHint}</p>
          <button
            type="button"
            onClick={() => prepare.mutate()}
            className="self-start rounded-base border border-border px-2 py-1 text-sm hover:bg-bg"
          >
            Prepare again
          </button>
        </div>
      )}

      {prepare.isError && (
        <ErrorState error={prepare.error} onRetry={() => prepare.mutate()} />
      )}
      {validate.isError && !staleHint && <ErrorState error={validate.error} />}
      {confirm.isError && <ErrorState error={confirm.error} />}
      {revoke.isError && <ErrorState error={revoke.error} />}
    </div>
  );
}

/* Discovery reads every source CSV, so it is a job like validation, not a
 * request. The drawer refreshes this exact source run graph when it settles. */
function DiscoverAction({
  projectId,
  sessionId,
  canDiscover,
  rerun,
}: {
  projectId: string;
  sessionId: string;
  canDiscover: boolean;
  rerun: boolean;
}) {
  const { startTracking } = useJobActivity();
  /* One key per attempt: a retry after a failed request replays the same job
   * instead of queueing a second scan of the same tables. */
  const [key, setKey] = useState(() => crypto.randomUUID());

  const discover = useMutation({
    mutationFn: () => api.discoverRelationships(sessionId, key),
    onSuccess: (started: RelationshipDiscoveryStarted) => {
      startTracking({
        jobId: started.job.job_id,
        sessionId: started.execution_session_id,
        sourceSessionId: sessionId,
        projectId,
        eventsUrl: started.job.events_url,
      });
      setKey(crypto.randomUUID());
    },
  });

  const label = rerun ? "Re-run discovery" : "Discover relationships";
  return (
    <div className="flex flex-col gap-2">
      <button
        type="button"
        onClick={() => discover.mutate()}
        disabled={!canDiscover || discover.isPending}
        title={
          canDiscover
            ? undefined
            : "Discovery needs at least two readable source tables in this session."
        }
        className={
          rerun
            ? "self-start rounded-base border border-border px-3 py-1.5 text-sm hover:bg-bg disabled:opacity-50"
            : "self-start rounded-base bg-primary px-3 py-1.5 text-sm font-medium text-bg hover:opacity-90 disabled:opacity-50"
        }
      >
        {discover.isPending ? "Starting…" : label}
      </button>
      {/* Discovery is not read-only: the driver proposes join-whitelist entries
       * and auto-confirms high-confidence id joins, which is what gates cross-
       * table SQL later. Same behaviour as the legacy UI, but say so up front. */}
      <p className="max-w-prose text-xs text-status-neutral">
        Discovery scans every source table and adds join candidates to this
        project&apos;s whitelist. High-confidence id joins are auto-confirmed and
        become usable by cross-table analysis.{" "}
        <Link
          to={
            sessionSectionPath(projectId, sessionId, "semantic") + "?view=joins"
          }
          className="text-primary underline-offset-2 hover:underline"
        >
          Review the project join policy in Knowledge.
        </Link>
      </p>
      {discover.isError && (
        <ErrorState error={discover.error} onRetry={() => discover.mutate()} />
      )}
    </div>
  );
}

function RelationshipGraph({
  nodes: nodeData,
  pairs,
  focusKey,
  onFocusPair,
}: {
  nodes: RelationshipNode[];
  pairs: Pair[];
  focusKey: string | null;
  onFocusPair: (key: string) => void;
}) {
  const connected = useMemo(() => {
    const ids = new Set<string>();
    for (const pair of pairs) {
      ids.add(pair.leftId);
      ids.add(pair.rightId);
    }
    return ids;
  }, [pairs]);
  const initialNodes = useMemo(
    () => toNodes(nodeData, connected),
    [nodeData, connected],
  );
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const edges = useMemo(() => toGraphEdges(pairs, focusKey), [pairs, focusKey]);
  const [grabbedNodeId, setGrabbedNodeId] = useState<string | null>(null);
  const [keyboardSnapshot, setKeyboardSnapshot] = useState<Node[] | null>(null);
  const [undoNodes, setUndoNodes] = useState<Node[] | null>(null);
  const [announcement, setAnnouncement] = useState("");
  const pointerSnapshot = useRef<Node[] | null>(null);

  useEffect(() => {
    setNodes(initialNodes);
    setGrabbedNodeId(null);
    setKeyboardSnapshot(null);
    setUndoNodes(null);
  }, [initialNodes, setNodes]);

  const cloneNodes = useCallback(
    (rows: Node[]) =>
      rows.map((node) => ({ ...node, position: { ...node.position } })),
    [],
  );

  const handleCanvasKeyDown = (
    event: React.KeyboardEvent<HTMLDivElement>,
  ) => {
    const target = event.target as Element;
    const edgeElement = target.closest<SVGGElement>(".react-flow__edge");
    if (
      edgeElement &&
      (event.key === "Enter" || event.key === " ")
    ) {
      const edgeId = edgeElement.dataset.id;
      if (edgeId) {
        event.preventDefault();
        event.stopPropagation();
        onFocusPair(edgeId);
        setAnnouncement("Relationship selected. Details are available in the list.");
      }
      return;
    }

    const nodeElement = target.closest<HTMLDivElement>(".react-flow__node");
    const nodeId = nodeElement?.dataset.id;
    if (!nodeId) return;
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      event.stopPropagation();
      if (grabbedNodeId === nodeId) {
        if (keyboardSnapshot) {
          setUndoNodes(cloneNodes(keyboardSnapshot));
        }
        setGrabbedNodeId(null);
        setKeyboardSnapshot(null);
        setAnnouncement("Dataset position saved. Undo is available.");
      } else {
        setGrabbedNodeId(nodeId);
        setKeyboardSnapshot(cloneNodes(nodes));
        setAnnouncement(
          "Dataset grabbed. Use arrow keys to move it, Enter to confirm, Escape to cancel.",
        );
      }
      return;
    }
    if (event.key === "Escape" && grabbedNodeId === nodeId) {
      event.preventDefault();
      event.stopPropagation();
      if (keyboardSnapshot) setNodes(cloneNodes(keyboardSnapshot));
      setGrabbedNodeId(null);
      setKeyboardSnapshot(null);
      setAnnouncement("Dataset move cancelled.");
      return;
    }
    if (grabbedNodeId !== nodeId) return;
    const delta: Record<string, { x: number; y: number }> = {
      ArrowUp: { x: 0, y: -KEYBOARD_NODE_STEP },
      ArrowDown: { x: 0, y: KEYBOARD_NODE_STEP },
      ArrowLeft: { x: -KEYBOARD_NODE_STEP, y: 0 },
      ArrowRight: { x: KEYBOARD_NODE_STEP, y: 0 },
    };
    const movement = delta[event.key];
    if (!movement) return;
    event.preventDefault();
    event.stopPropagation();
    setNodes((current) =>
      current.map((node) =>
        node.id === nodeId
          ? {
              ...node,
              position: {
                x: node.position.x + movement.x,
                y: node.position.y + movement.y,
              },
            }
          : node,
      ),
    );
    setAnnouncement(
      `Dataset moved ${event.key.replace("Arrow", "").toLowerCase()}. ` +
        "Press Enter to confirm or Escape to cancel.",
    );
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <p id="relationship-graph-help" className="sr-only">
        Focus a dataset and press Enter to grab it. Arrow keys move it; Enter
        confirms; Escape cancels. Focus a relationship and press Enter to scope
        the details list.
      </p>
      <ReactFlow
        className="relationships-flow"
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgeClick={(_event, edge) => onFocusPair(edge.id)}
        onKeyDown={handleCanvasKeyDown}
        onNodeDragStart={() => {
          pointerSnapshot.current = cloneNodes(nodes);
        }}
        onNodeDragStop={() => {
          if (pointerSnapshot.current) {
            setUndoNodes(pointerSnapshot.current);
            pointerSnapshot.current = null;
            setAnnouncement("Dataset position saved. Undo is available.");
          }
        }}
        fitView
        fitViewOptions={{ maxZoom: 1, padding: 0.15 }}
        nodesConnectable={false}
        nodesFocusable
        edgesFocusable
        /* The delegated handler above owns grab/move/commit/cancel and its
         * announcements. React Flow's built-in arrow movement otherwise
         * competes for the same key event and can skip our live-region update. */
        disableKeyboardA11y
        aria-describedby="relationship-graph-help"
        proOptions={{ hideAttribution: false }}
      >
        <Background gap={24} />
        <MiniMap pannable zoomable />
        <Controls />
      </ReactFlow>
      <div className="flex min-h-8 items-center gap-2 border-t border-hairline px-3 py-1">
        {undoNodes && (
          <button
            type="button"
            onClick={() => {
              setNodes(cloneNodes(undoNodes));
              setUndoNodes(null);
              setAnnouncement("Last dataset move undone.");
            }}
            className="rounded-base border border-border px-2 py-1 text-xs hover:bg-surface"
          >
            Undo graph move
          </button>
        )}
        <span aria-live="polite" className="sr-only">
          {announcement}
        </span>
      </div>
    </div>
  );
}

function GraphPane({
  nodes,
  pairs,
  pairsTotal,
  focusKey,
  onFocusPair,
}: {
  nodes: RelationshipNode[];
  pairs: Pair[];
  pairsTotal: number;
  focusKey: string | null;
  onFocusPair: (key: string) => void;
}) {
  return (
    <Card className="relationship-graph-pane flex min-h-80 min-w-0 flex-1 flex-col overflow-hidden">
      <RelationshipGraph
        nodes={nodes}
        pairs={pairs}
        focusKey={focusKey}
        onFocusPair={onFocusPair}
      />
      <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-2 border-t border-hairline px-3 py-2">
        <Legend />
        <p className="text-xs text-status-neutral">
          One line per table pair · {pairs.length} of {pairsTotal} pair(s) match
          the filter. Click a line, or focus it and press Enter, to scope the
          list.
        </p>
      </div>
    </Card>
  );
}

/* Keyed by sessionId by the caller: filters and selection belong to one run, and a
 * run the user never filtered must open at the defaults
 * (filter widgets are keyed by session_id). */
function Workbench({
  projectId,
  sessionId,
  graph,
}: {
  projectId: string;
  sessionId: string;
  graph: RelationshipGraphView;
}) {
  const nodes = graph.nodes ?? [];
  const edges = graph.edges ?? [];
  const [scoreParam, setScoreParam] = useRouteSearchParam("score");
  const parsedScore = Number(scoreParam);
  const minimumScore = Number.isFinite(parsedScore)
    ? Math.min(1, Math.max(0, parsedScore))
    : 0;
  const setMinimumScore = (score: number) =>
    setScoreParam(score === 0 ? "" : String(score));
  const [confidenceParam, setConfidenceParam] = useRouteSearchParam(
    "confidence",
    "high,medium",
  );
  const confidenceLevels = new Set(
    confidenceParam === "none"
      ? []
      : parseCsvParam(confidenceParam).filter((level) =>
          CONFIDENCE_LEVELS.includes(
            level as (typeof CONFIDENCE_LEVELS)[number],
          ),
        ),
  );
  const [focusKeyParam] = useRouteSearchParam("pair");
  const focusKey = focusKeyParam || null;
  const [selectedIdParam, setSelectedId] = useRouteSearchParam("edge");
  const selectedId = selectedIdParam || null;
  const setRouteSearchParams = useSetRouteSearchParams();

  const toggleConfidence = (level: string) => {
    const next = new Set(confidenceLevels);
    if (next.has(level)) next.delete(level);
    else next.add(level);
    setConfidenceParam(
      next.size === 0 ? "none" : serializeCsvParam(next),
    );
  };

  const visibleEdges = useMemo(
    () =>
      edges.filter(
        (edge) =>
          edge.ensemble_score >= minimumScore &&
          confidenceLevels.has(edge.confidence),
      ),
    [edges, minimumScore, confidenceLevels],
  );
  const pairs = useMemo(() => groupByPair(visibleEdges), [visibleEdges]);
  const pairsTotal = useMemo(() => groupByPair(edges).length, [edges]);
  const focused = pairs.find((pair) => pair.key === focusKey) ?? null;
  const listedPairs = focused ? [focused] : pairs;
  const expandGroups =
    listedPairs.length === 1 || visibleEdges.length <= AUTO_EXPAND_LIMIT;

  const selected =
    edges.find((edge) => edge.relationship_id === selectedId) ?? null;
  const discovered = graph.discovered;
  /* Mirrors the server's own gate, so the button is only offered when the
   * POST would actually be accepted. */
  const canDiscover = nodes.filter((node) => node.source_available).length >= 2;
  const noHighConfidence =
    edges.length > 0 && !edges.some((edge) => edge.confidence === "high");

  const focusPair = (key: string) => {
    const pair = pairs.find((item) => item.key === key);
    setRouteSearchParams({
      pair: key,
      edge:
        pair && pair.edges.length === 1
          ? (pair.edges[0]?.relationship_id ?? "")
          : "",
    });
  };

  /* Picking a candidate also focuses its pair, so closing the Inspector lands
   * back on that pair rather than on a wall of collapsed groups. */
  const selectEdge = (id: string) => {
    const pair = pairs.find((item) =>
      item.edges.some((edge) => edge.relationship_id === id),
    );
    setRouteSearchParams({
      pair: pair?.key ?? focusKey ?? "",
      edge: id,
    });
  };

  return (
    <div className="relationship-workbench min-h-0 flex-1 overflow-auto">
      <div className="relationship-workbench-layout grid min-h-full min-w-0 gap-4">
        <aside
          aria-label="Relationship inspector"
          className="relationship-review flex min-h-0 min-w-0 flex-col gap-3 overflow-auto rounded-base border border-border bg-surface p-3"
        >
          {/* Medium candidates on their own
           * are hypotheses, and the default filter lets them through. */}
          {noHighConfidence && (
            <Card tone="warn" className="p-2">
              <p role="alert" className="text-xs text-status-warn">
                No high-confidence relationship was found. Medium-confidence
                candidates are hypotheses only: validate them against the full
                tables before using a join.
              </p>
            </Card>
          )}
          {!discovered ? (
            <>
              <EmptyState
                title="Relationship discovery has not run"
                description="This session deferred cross-table discovery to keep the main analysis fast. Discovery reads every source table, so it runs as a background job."
              />
              <DiscoverAction
                projectId={projectId}
                sessionId={sessionId}
                canDiscover={canDiscover}
                rerun={false}
              />
            </>
          ) : edges.length === 0 ? (
            <>
              <EmptyState
                title="No relationship candidates"
                description="Discovery scored no candidate pair for these datasets. Re-run discovery after adding or replacing a table."
              />
              <DiscoverAction
                projectId={projectId}
                sessionId={sessionId}
                canDiscover={canDiscover}
                rerun
              />
            </>
          ) : (
            <>
              {!selected && (
                <>
                  <SectionHeader
                    level={2}
                    title="Review candidates"
                    description="Choose a scored column pair, inspect its evidence, then validate it before making it available as a join."
                  />
                  <Disclosure
                    defaultOpen
                    summary="Filter candidates"
                    meta={`${visibleEdges.length} of ${edges.length}`}
                  >
                    <div className="flex flex-col gap-2">
                      <CandidateFilters
                        minimumScore={minimumScore}
                        onMinimumScoreChange={setMinimumScore}
                        confidenceLevels={confidenceLevels}
                        onToggleConfidence={toggleConfidence}
                      />
                      <ScopeReadout
                        edges={edges}
                        visibleCount={visibleEdges.length}
                        minimumScore={minimumScore}
                        confidenceLevels={confidenceLevels}
                        pairsInScope={pairs.length}
                        pairsTotal={pairsTotal}
                      />
                    </div>
                  </Disclosure>
                </>
              )}
              {focused && (
                <div className="flex items-center gap-2">
                  <Badge tone="brand">
                    {focused.leftName} → {focused.rightName}
                  </Badge>
                  <button
                    type="button"
                    onClick={() =>
                      setRouteSearchParams({ pair: "", edge: "" })
                    }
                    className="text-xs text-primary underline-offset-2 hover:underline"
                  >
                    Show all pairs
                  </button>
                </div>
              )}
              {selected ? (
                <EdgeInspector
                  projectId={projectId}
                  sessionId={sessionId}
                  edge={selected}
                  seedsVersion={graph.seeds_version}
                  onClose={() => setSelectedId("")}
                />
              ) : visibleEdges.length === 0 ? (
                <p className="text-xs text-status-neutral">
                  No candidates match the current filters.
                </p>
              ) : (
                <div className="flex flex-col gap-1">
                  {listedPairs.map((pair) => (
                    <PairGroup
                      key={`${pair.key}-${expandGroups}`}
                      pair={pair}
                      expanded={expandGroups}
                      selectedId={selectedId}
                      onSelect={selectEdge}
                    />
                  ))}
                </div>
              )}
              {!selected && (
                <div className="mt-auto border-t border-hairline pt-3">
                  <Disclosure
                    summary="Discovery controls"
                    meta="Full-table scan"
                  >
                    <DiscoverAction
                      projectId={projectId}
                      sessionId={sessionId}
                      canDiscover={canDiscover}
                      rerun
                    />
                  </Disclosure>
                </div>
              )}
            </>
          )}
        </aside>
        <GraphPane
          nodes={nodes}
          pairs={pairs}
          pairsTotal={pairsTotal}
          focusKey={focusKey}
          onFocusPair={focusPair}
        />
      </div>
    </div>
  );
}

/** Turns the "bounded overlap budget" prose into figures: what was measured,
 *  what was skipped, and what never reached the artifact. */
function SearchCoverage({ graph }: { graph: RelationshipGraphView }) {
  const limited = graph.coverage_status === "limited";
  const truncated = graph.truncated_pairs;
  if (!limited && truncated === 0) return null;

  return (
    <div className="flex flex-col gap-2">
      <SectionHeader
        level={3}
        title={
          <span className="flex items-center gap-2">
            Search coverage
            <Badge tone="warn" caps>
              bounded
            </Badge>
            <Hint label="Bounded search">
              Overlap is measured pair by pair against the real tables, so the
              run works to a budget. A pair the budget skipped was never
              measured — that is not evidence the tables are unrelated.
            </Hint>
          </span>
        }
        description={
          limited
            ? "The overlap budget stopped before every table pair was measured. Deferred does not mean unrelated."
            : "Every pair was measured, but not every scored candidate reached the artifact this page reads."
        }
      />
      <MetricStrip>
        {limited && (
          <MetricTile
            label="Pairs evaluated"
            value={formatCompact(graph.overlap_pairs_evaluated)}
            hint="Overlap measured against both tables."
          />
        )}
        {limited && (
          <MetricTile
            label="Pairs deferred"
            value={formatCompact(graph.overlap_pairs_prefiltered)}
            tone="warn"
            emphasis
            hint="Skipped by the budget — never measured."
          />
        )}
        {truncated > 0 && (
          <MetricTile
            label="Omitted by size cap"
            value={formatCompact(truncated)}
            hint="Scored, then dropped from the displayed artifact by the output-size cap."
          />
        )}
      </MetricStrip>
    </div>
  );
}

export function Component() {
  const { projectId = "", sessionId = "" } = useParams();
  const graph = useRelationships(sessionId);

  if (graph.isPending) {
    return <LoadingSkeleton lines={4} label="Loading relationships" />;
  }
  if (graph.isError) {
    return (
      <div className="p-6">
        <ErrorState error={graph.error} onRetry={() => graph.refetch()} />
      </div>
    );
  }

  const nodes = graph.data.nodes ?? [];

  return (
    <div className="mx-auto flex w-[90%] max-w-data h-full flex-col gap-4 p-6">
      <header className="flex flex-col gap-3">
        <SectionHeader
          level={1}
          title="Relationships"
          description="Datasets stay separate until a join is confirmed. The canvas draws the table pairs that match the filter on the right; open a pair to pick a column candidate, see its evidence and promote it."
        />
        <SearchCoverage graph={graph.data} />
      </header>

      {nodes.length === 0 ? (
        <EmptyState
          title="No datasets in this session"
          description="Upload data and start a session to explore relationships."
        />
      ) : (
        <Workbench
          key={`workbench-${sessionId}`}
          projectId={projectId}
          sessionId={sessionId}
          graph={graph.data}
        />
      )}
    </div>
  );
}
