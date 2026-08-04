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
  BaseEdge,
  Background,
  Controls,
  MiniMap,
  ReactFlow,
  useInternalNode,
  useNodesState,
  type Edge,
  type EdgeProps,
  type Node,
  type ReactFlowInstance,
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
import {
  DataWorkspacePage,
  SegmentedControl,
} from "../../components/data-workspace";

const NODE_WIDTH = 196;
const KEYBOARD_NODE_STEP = 24;
const NEIGHBOR_LIMIT = 6;
/* Above this many candidates in scope, pair groups open on demand instead of
 * all at once — an 80-row wall of columns is the thing the grouping fixes. */
const AUTO_EXPAND_LIMIT = 12;

type MeasuredGraphNode = {
  measured: { width?: number; height?: number };
  internals: { positionAbsolute: { x: number; y: number } };
};

function pointOnNodeToward(
  node: MeasuredGraphNode,
  toward: MeasuredGraphNode,
) {
  const width = node.measured.width ?? NODE_WIDTH;
  const height = node.measured.height ?? 84;
  const towardWidth = toward.measured.width ?? NODE_WIDTH;
  const towardHeight = toward.measured.height ?? 84;
  const centreX = node.internals.positionAbsolute.x + width / 2;
  const centreY = node.internals.positionAbsolute.y + height / 2;
  const towardX = toward.internals.positionAbsolute.x + towardWidth / 2;
  const towardY = toward.internals.positionAbsolute.y + towardHeight / 2;
  const deltaX = towardX - centreX;
  const deltaY = towardY - centreY;
  const scale = Math.min(
    deltaX === 0 ? Number.POSITIVE_INFINITY : width / 2 / Math.abs(deltaX),
    deltaY === 0 ? Number.POSITIVE_INFINITY : height / 2 / Math.abs(deltaY),
  );
  return {
    x: centreX + deltaX * scale,
    y: centreY + deltaY * scale,
  };
}

function OverviewCurveEdge({
  id,
  source,
  target,
  sourceX,
  sourceY,
  targetX,
  targetY,
  style,
  label,
  labelStyle,
  labelBgStyle,
  markerStart,
  markerEnd,
  interactionWidth,
}: EdgeProps) {
  const sourceNode = useInternalNode(source);
  const targetNode = useInternalNode(target);
  const floatingSource =
    sourceNode && targetNode
      ? pointOnNodeToward(sourceNode, targetNode)
      : { x: sourceX, y: sourceY };
  const floatingTarget =
    sourceNode && targetNode
      ? pointOnNodeToward(targetNode, sourceNode)
      : { x: targetX, y: targetY };
  const deltaX = floatingTarget.x - floatingSource.x;
  const deltaY = floatingTarget.y - floatingSource.y;
  const distance = Math.max(1, Math.hypot(deltaX, deltaY));
  const bend = Math.min(96, Math.max(28, distance * 0.13));
  const normalX = -deltaY / distance;
  const normalY = deltaX / distance;
  const controlX =
    (floatingSource.x + floatingTarget.x) / 2 + normalX * bend;
  const controlY =
    (floatingSource.y + floatingTarget.y) / 2 + normalY * bend;
  const labelX =
    (floatingSource.x + 2 * controlX + floatingTarget.x) / 4;
  const labelY =
    (floatingSource.y + 2 * controlY + floatingTarget.y) / 4;

  return (
    <BaseEdge
      id={id}
      path={`M ${floatingSource.x},${floatingSource.y} Q ${controlX},${controlY} ${floatingTarget.x},${floatingTarget.y}`}
      label={label}
      labelX={labelX}
      labelY={labelY}
      labelStyle={labelStyle}
      labelBgStyle={labelBgStyle}
      markerStart={markerStart}
      markerEnd={markerEnd}
      interactionWidth={interactionWidth}
      style={style}
    />
  );
}

const RELATIONSHIP_EDGE_TYPES = { overviewCurve: OverviewCurveEdge };

type EdgeState = "candidate" | "validated" | "confirmed";
type RelationshipView = "overview" | "neighborhood" | "matrix" | "list";
type RelationshipKind =
  | "many_to_one"
  | "one_to_many"
  | "one_to_one"
  | "many_to_many"
  | "likely_key"
  | "shared_domain"
  | "partial_coverage"
  | "possible_match";
type TableRole = "focus" | "likely_fact" | "likely_lookup" | "bridge" | "hub" | "connected" | "isolated";

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

const KIND_LABEL: Record<RelationshipKind, string> = {
  many_to_one: "Many → one lookup",
  one_to_many: "One → many detail",
  one_to_one: "One ↔ one entity",
  many_to_many: "Many ↔ many risk",
  likely_key: "Likely key join",
  shared_domain: "Shared domain",
  partial_coverage: "Partial coverage",
  possible_match: "Possible match",
};

const KIND_SHORT: Record<RelationshipKind, string> = {
  many_to_one: "*:1",
  one_to_many: "1:*",
  one_to_one: "1:1",
  many_to_many: "*:* risk",
  likely_key: "key?",
  shared_domain: "domain",
  partial_coverage: "partial",
  possible_match: "match?",
};

const ROLE_LABEL: Record<TableRole, string> = {
  focus: "Focus",
  likely_fact: "Likely fact",
  likely_lookup: "Likely lookup",
  bridge: "Likely bridge",
  hub: "Hub",
  connected: "Connected",
  isolated: "Isolated",
};

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

function relationshipKind(edge: RelationshipEdge): RelationshipKind {
  if (
    edge.cardinality === "many_to_one" ||
    edge.cardinality === "one_to_many" ||
    edge.cardinality === "one_to_one" ||
    edge.cardinality === "many_to_many"
  ) {
    return edge.cardinality;
  }
  if (
    edge.overlap_left_in_right >= 0.6 &&
    edge.overlap_right_in_left >= 0.6 &&
    edge.right_unique_rate < 0.9
  ) {
    return "shared_domain";
  }
  if (
    Math.max(edge.overlap_left_in_right, edge.overlap_right_in_left) >= 0.6 &&
    Math.min(edge.overlap_left_in_right, edge.overlap_right_in_left) < 0.6
  ) {
    return "partial_coverage";
  }
  if (edge.overlap_left_in_right >= 0.6 && edge.right_unique_rate >= 0.9) {
    return "likely_key";
  }
  return "possible_match";
}

function relationshipReasons(edge: RelationshipEdge): string[] {
  const reasons: string[] = [];
  if (edge.name_similarity >= 0.95) reasons.push("Same column name");
  else if (edge.name_similarity >= 0.6) reasons.push("Similar column names");
  if (edge.overlap_left_in_right >= 0.95) reasons.push("Source values strongly covered");
  else if (edge.overlap_left_in_right >= 0.6) reasons.push("Source values partly covered");
  if (edge.right_unique_rate >= 0.99) reasons.push("Target is nearly unique");
  else if (edge.right_unique_rate >= 0.9) reasons.push("Target is mostly unique");
  if (edge.verified) reasons.push("Checked against full tables");
  if (edge.signals_sampled || edge.validation_sampled) reasons.push("Estimated from a sample");
  return reasons.length > 0 ? reasons : ["Compatible key-like fields"];
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

function circularCrossingScore(order: RelationshipNode[], pairs: Pair[]) {
  const position = new Map(
    order.map((node, index) => [node.dataset_id, index]),
  );
  const graphPairs = pairs.filter((pair) => pair.leftId !== pair.rightId);
  let crossings = 0;
  let span = 0;

  for (const pair of graphPairs) {
    const left = position.get(pair.leftId);
    const right = position.get(pair.rightId);
    if (left == null || right == null) continue;
    const distance = Math.abs(left - right);
    span += Math.min(distance, order.length - distance);
  }

  for (let leftIndex = 0; leftIndex < graphPairs.length; leftIndex += 1) {
    const leftPair = graphPairs[leftIndex]!;
    const a = position.get(leftPair.leftId);
    const b = position.get(leftPair.rightId);
    if (a == null || b == null) continue;
    for (
      let rightIndex = leftIndex + 1;
      rightIndex < graphPairs.length;
      rightIndex += 1
    ) {
      const rightPair = graphPairs[rightIndex]!;
      if (
        leftPair.leftId === rightPair.leftId ||
        leftPair.leftId === rightPair.rightId ||
        leftPair.rightId === rightPair.leftId ||
        leftPair.rightId === rightPair.rightId
      ) {
        continue;
      }
      const c = position.get(rightPair.leftId);
      const d = position.get(rightPair.rightId);
      if (c == null || d == null) continue;
      const between = (point: number, start: number, end: number) =>
        start < end
          ? point > start && point < end
          : point > start || point < end;
      if (between(c, a, b) !== between(d, a, b)) crossings += 1;
    }
  }

  /* Crossings dominate the objective. The smaller span term chooses the more
   * compact of two orders with the same crossing count. */
  return crossings * 10_000 + span;
}

function crossingReducedCircularOrder(
  nodes: RelationshipNode[],
  pairs: Pair[],
) {
  const degree = new Map(nodes.map((node) => [node.dataset_id, 0]));
  const adjacency = new Map(
    nodes.map((node) => [node.dataset_id, new Map<string, number>()]),
  );
  for (const pair of pairs) {
    if (pair.leftId === pair.rightId) continue;
    degree.set(pair.leftId, (degree.get(pair.leftId) ?? 0) + 1);
    degree.set(pair.rightId, (degree.get(pair.rightId) ?? 0) + 1);
    adjacency.get(pair.leftId)?.set(pair.rightId, pair.best);
    adjacency.get(pair.rightId)?.set(pair.leftId, pair.best);
  }

  const remaining = new Set(nodes.map((node) => node.dataset_id));
  const byId = new Map(nodes.map((node) => [node.dataset_id, node]));
  const first = [...nodes].sort(
    (left, right) =>
      (degree.get(right.dataset_id) ?? 0) -
        (degree.get(left.dataset_id) ?? 0) ||
      left.name.localeCompare(right.name),
  )[0];
  if (!first) return [];
  const order = [first];
  remaining.delete(first.dataset_id);

  /* Seed the circle by walking strongest adjacent tables. This gives the swap
   * pass a stable, graph-aware starting point instead of filename order. */
  while (remaining.size > 0) {
    const previous = order[order.length - 1]!;
    const nextId = [...remaining].sort((leftId, rightId) => {
      const leftWeight = adjacency.get(previous.dataset_id)?.get(leftId) ?? -1;
      const rightWeight = adjacency.get(previous.dataset_id)?.get(rightId) ?? -1;
      return (
        rightWeight - leftWeight ||
        (degree.get(rightId) ?? 0) - (degree.get(leftId) ?? 0) ||
        (byId.get(leftId)?.name ?? leftId).localeCompare(
          byId.get(rightId)?.name ?? rightId,
        )
      );
    })[0]!;
    order.push(byId.get(nextId)!);
    remaining.delete(nextId);
  }

  /* An exhaustive swap pass is cheap for the overview sizes this page is
   * designed for and directly optimises the visual problem: chord crossings. */
  if (order.length <= 24) {
    let bestScore = circularCrossingScore(order, pairs);
    let improved = true;
    while (improved) {
      improved = false;
      let bestSwap: [number, number] | null = null;
      for (let left = 0; left < order.length - 1; left += 1) {
        for (let right = left + 1; right < order.length; right += 1) {
          const candidate = [...order];
          [candidate[left], candidate[right]] = [candidate[right]!, candidate[left]!];
          const score = circularCrossingScore(candidate, pairs);
          if (score < bestScore) {
            bestScore = score;
            bestSwap = [left, right];
          }
        }
      }
      if (bestSwap) {
        [order[bestSwap[0]], order[bestSwap[1]]] = [
          order[bestSwap[1]]!,
          order[bestSwap[0]]!,
        ];
        improved = true;
      }
    }
  }
  return order;
}

/* Overview uses a graph-aware circle: an optimised circular order reduces
 * chord crossings, keeps every table equally reachable, and avoids the dense
 * horizontal traffic created by a hub-centred grid. */
function overviewPositions(nodes: RelationshipNode[], pairs: Pair[]) {
  if (nodes.length <= 1) return [{ x: 80, y: 120 }];
  const ordered = crossingReducedCircularOrder(nodes, pairs);
  const radius = Math.max(
    360,
    (nodes.length * (NODE_WIDTH + 34)) / (2 * Math.PI),
  );
  const centreX = radius + NODE_WIDTH / 2 + 44;
  const centreY = radius + 74;
  const positions = new Map(
    ordered.map((node, index) => {
      const angle = (2 * Math.PI * index) / ordered.length - Math.PI / 2;
      return [
        node.dataset_id,
        {
          x: Math.round(centreX + radius * Math.cos(angle) - NODE_WIDTH / 2),
          y: Math.round(centreY + radius * Math.sin(angle) - 42),
        },
      ];
    }),
  );
  return nodes.map((node) => positions.get(node.dataset_id)!);
}

function layoutPositions(
  nodes: RelationshipNode[],
  focusId: string | null,
  pairs: Pair[],
  overview: boolean,
): { x: number; y: number }[] {
  if (overview) return overviewPositions(nodes, pairs);
  const count = nodes.length;
  if (focusId && count > 1) {
    let neighborIndex = 0;
    return nodes.map((node) => {
      if (node.dataset_id === focusId) {
        return { x: 0, y: Math.max(0, ((count - 2) * 132) / 2) };
      }
      const position = { x: NODE_WIDTH + 190, y: neighborIndex * 132 };
      neighborIndex += 1;
      return position;
    });
  }
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

function nodeLabel(node: RelationshipNode, role: TableRole) {
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
      <span className="mt-1 block text-[10px] font-medium uppercase tracking-wide text-status-neutral">
        {ROLE_LABEL[role]}
        {!["focus", "connected", "isolated"].includes(role) ? " · inferred" : ""}
      </span>
    </div>
  );
}

function toNodes(
  nodes: RelationshipNode[],
  connected: Set<string>,
  focusId: string | null,
  roles: Map<string, TableRole>,
  pairs: Pair[],
  overview: boolean,
): Node[] {
  const positions = layoutPositions(nodes, focusId, pairs, overview);
  return nodes.map((node, index) => ({
    id: node.dataset_id,
    position: positions[index] ?? { x: 0, y: 0 },
    data: { label: nodeLabel(node, node.dataset_id === focusId ? "focus" : (roles.get(node.dataset_id) ?? "isolated")) },
    style: { width: NODE_WIDTH },
    className: `${connected.has(node.dataset_id) ? "" : "rel-node-muted"} ${node.dataset_id === focusId ? "rel-node-focus" : ""}`.trim(),
  }));
}

type Pair = {
  identity: string;
  key: string;
  legacyKeys: string[];
  leftId: string;
  rightId: string;
  leftName: string;
  rightName: string;
  edges: RelationshipEdge[];
  best: number;
  state: EdgeState;
  representative: RelationshipEdge;
  kind: RelationshipKind;
};

function groupByPair(edges: RelationshipEdge[]): Pair[] {
  const pairs = new Map<string, Pair>();
  for (const edge of edges) {
    const identity = [edge.left_dataset_id, edge.right_dataset_id].sort().join("↔");
    const key = `${edge.left_dataset_id}→${edge.right_dataset_id}`;
    const state = edgeState(edge);
    const existing = pairs.get(identity);
    if (existing) {
      existing.edges.push(edge);
      if (!existing.legacyKeys.includes(key)) existing.legacyKeys.push(key);
      existing.best = Math.max(existing.best, edge.ensemble_score);
      if (
        STATE_RANK[state] > STATE_RANK[existing.state] ||
        (STATE_RANK[state] === STATE_RANK[existing.state] &&
          edge.ensemble_score > existing.representative.ensemble_score)
      ) {
        existing.state = state;
        existing.representative = edge;
        existing.kind = relationshipKind(edge);
        existing.leftId = edge.left_dataset_id;
        existing.rightId = edge.right_dataset_id;
        existing.leftName = edge.left_dataset;
        existing.rightName = edge.right_dataset;
      }
    } else {
      pairs.set(identity, {
        identity,
        key,
        legacyKeys: [key],
        leftId: edge.left_dataset_id,
        rightId: edge.right_dataset_id,
        leftName: edge.left_dataset,
        rightName: edge.right_dataset,
        edges: [edge],
        best: edge.ensemble_score,
        state,
        representative: edge,
        kind: relationshipKind(edge),
      });
    }
  }
  const list = [...pairs.values()];
  for (const pair of list) {
    pair.edges.sort((left, right) => right.ensemble_score - left.ensemble_score);
  }
  return list.sort((left, right) => right.best - left.best);
}

function inferTableRoles(
  nodes: RelationshipNode[],
  pairs: Pair[],
): Map<string, TableRole> {
  const stats = new Map<
    string,
    { degree: number; factSignals: number; lookupSignals: number }
  >();
  for (const node of nodes) {
    stats.set(node.dataset_id, { degree: 0, factSignals: 0, lookupSignals: 0 });
  }
  for (const pair of pairs) {
    const left = stats.get(pair.leftId);
    const right = stats.get(pair.rightId);
    if (left) left.degree += 1;
    if (right) right.degree += 1;
    if (pair.kind === "many_to_one" || pair.kind === "likely_key") {
      if (left) left.factSignals += 1;
      if (right) right.lookupSignals += 1;
    } else if (pair.kind === "one_to_many") {
      if (left) left.lookupSignals += 1;
      if (right) right.factSignals += 1;
    }
  }
  return new Map(
    nodes.map((node) => {
      const stat = stats.get(node.dataset_id)!;
      let role: TableRole = "connected";
      if (stat.degree === 0) role = "isolated";
      else if (stat.degree >= 4) role = "hub";
      else if (stat.factSignals >= 2 && stat.lookupSignals >= 2) role = "bridge";
      else if (stat.factSignals >= 2) role = "likely_fact";
      else if (stat.lookupSignals >= 2) role = "likely_lookup";
      return [node.dataset_id, role];
    }),
  );
}

function toGraphEdges(
  pairs: Pair[],
  focusKey: string | null,
  overview: boolean,
  focusTableId: string | null,
): Edge[] {
  return pairs.filter((pair) => !overview || pair.leftId !== pair.rightId).map((pair) => {
    const focused =
      pair.key === focusKey || pair.legacyKeys.includes(focusKey ?? "");
    const incident =
      pair.leftId === focusTableId || pair.rightId === focusTableId;
    const only = pair.edges.length === 1 ? pair.edges[0] : undefined;
    return {
      id: pair.key,
      source: pair.leftId,
      target: pair.rightId,
      /* A small quadratic bend keeps the overview fluid without the large
       * hooks produced by handle-directed Bezier routing. */
      type: overview ? "overviewCurve" : undefined,
      label:
        overview && !focused
          ? undefined
          : `${KIND_SHORT[pair.kind]} · ${only ? columnList(only.left_columns) : `${pair.edges.length} candidates`}`,
      ariaLabel: `${pair.leftName} and ${pair.rightName}: ${KIND_LABEL[pair.kind]}, ${pair.edges.length} candidates`,
      style: {
        stroke: focused ? "var(--color-primary)" : EDGE_STROKE[pair.state],
        strokeWidth: focused
          ? 3
          : overview && incident
            ? 2.5
            : pair.state === "candidate"
              ? 1.5
              : 2,
        strokeDasharray: pair.state === "candidate" ? "6 4" : undefined,
        opacity: overview && focusTableId && !incident && !focused ? 0.16 : 1,
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
const RELATIONSHIP_STATES: EdgeState[] = [
  "candidate",
  "validated",
  "confirmed",
];

function CandidateFilters({
  minimumScore,
  onMinimumScoreChange,
  confidenceLevels,
  onToggleConfidence,
  states,
  onToggleState,
}: {
  minimumScore: number;
  onMinimumScoreChange: (value: number) => void;
  confidenceLevels: Set<string>;
  onToggleConfidence: (level: string) => void;
  states: Set<EdgeState>;
  onToggleState: (state: EdgeState) => void;
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
      <fieldset className="flex flex-wrap items-center gap-3">
        <legend className="text-status-neutral">Status</legend>
        {RELATIONSHIP_STATES.map((state) => (
          <label key={state} className="flex items-center gap-1">
            <input
              type="checkbox"
              checked={states.has(state)}
              onChange={() => onToggleState(state)}
            />
            {state[0]?.toUpperCase()}{state.slice(1)}
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
  states,
  pairsInScope,
  pairsTotal,
}: {
  edges: RelationshipEdge[];
  visibleCount: number;
  minimumScore: number;
  confidenceLevels: Set<string>;
  states: Set<EdgeState>;
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
  const offState = edges.filter(
    (edge) =>
      edge.ensemble_score >= minimumScore &&
      confidenceLevels.has(edge.confidence) &&
      !states.has(edgeState(edge)),
  ).length;
  const tally = CONFIDENCE_LEVELS.map(
    (level) => `${edges.filter((edge) => edge.confidence === level).length} ${level}`,
  ).join(" · ");

  return (
    <div className="flex flex-col gap-1">
      <p className="text-xs text-status-neutral">
        Showing {visibleCount} of {edges.length} candidates.
      </p>
      {belowScore + offConfidence + offState > 0 && (
        <p className="text-xs text-status-neutral">
          Hidden: {belowScore} below score {minimumScore.toFixed(2)} ·{" "}
          {offConfidence} outside the confidence filter
          {offState > 0 ? ` · ${offState} outside the status filter.` : "."}
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
            {pair.leftName} ↔ {pair.rightName}
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

/* Shown in place of the workbench until discovery has run. Teaches what the
 * step costs and produces, rather than rendering an empty grid of every table
 * pair and hiding the only available action underneath it. */
function PreDiscoveryPanel({
  projectId,
  sessionId,
  datasetCount,
  canDiscover,
}: {
  projectId: string;
  sessionId: string;
  datasetCount: number;
  /* Readable source tables, not table count: discovery re-reads every source
   * file, so a session whose uploads are gone cannot run it. */
  canDiscover: boolean;
}) {
  const pairCount = (datasetCount * (datasetCount - 1)) / 2;
  return (
    <Card className="flex max-w-3xl flex-col gap-3 p-5">
      <SectionHeader
        level={2}
        title="Relationship discovery has not run"
        description="Nothing on this page is empty because the tables are unrelated — the session has not looked yet. It deferred cross-table discovery to keep the main analysis fast."
      />
      <p className="text-sm">
        Running it reads all {datasetCount} source tables and scores the{" "}
        <span className="tabular">{pairCount}</span> possible table pairs for
        columns that could join. It runs as a background job; you can keep
        working while it does.
      </p>
      <DiscoverAction
        projectId={projectId}
        sessionId={sessionId}
        canDiscover={canDiscover}
        rerun={false}
      />
    </Card>
  );
}

function RelationshipGraph({
  nodes: nodeData,
  pairs,
  focusKey,
  focusTableId,
  roles,
  overview,
  onFocusPair,
  onFocusTable,
}: {
  nodes: RelationshipNode[];
  pairs: Pair[];
  focusKey: string | null;
  focusTableId: string | null;
  roles: Map<string, TableRole>;
  overview: boolean;
  onFocusPair: (key: string) => void;
  onFocusTable: (datasetId: string) => void;
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
    () => toNodes(nodeData, connected, focusTableId, roles, pairs, overview),
    [nodeData, connected, focusTableId, roles, pairs, overview],
  );
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const edges = useMemo(
    () => toGraphEdges(pairs, focusKey, overview, focusTableId),
    [pairs, focusKey, overview, focusTableId],
  );
  const [grabbedNodeId, setGrabbedNodeId] = useState<string | null>(null);
  const [keyboardSnapshot, setKeyboardSnapshot] = useState<Node[] | null>(null);
  const [undoNodes, setUndoNodes] = useState<Node[] | null>(null);
  const [announcement, setAnnouncement] = useState("");
  const pointerSnapshot = useRef<Node[] | null>(null);
  const flowInstance = useRef<ReactFlowInstance<Node, Edge> | null>(null);

  useEffect(() => {
    setNodes(initialNodes);
    setGrabbedNodeId(null);
    setKeyboardSnapshot(null);
    setUndoNodes(null);
    const frame = requestAnimationFrame(() => {
      void flowInstance.current?.fitView({ maxZoom: 1, padding: 0.15 });
    });
    return () => cancelAnimationFrame(frame);
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
        edgeTypes={RELATIONSHIP_EDGE_TYPES}
        onNodesChange={onNodesChange}
        onInit={(instance) => {
          flowInstance.current = instance;
          requestAnimationFrame(() => {
            void instance.fitView({ maxZoom: 1, padding: 0.15 });
          });
        }}
        onEdgeClick={(_event, edge) => onFocusPair(edge.id)}
        onNodeClick={(_event, node) => onFocusTable(node.id)}
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
        minZoom={overview ? 0.2 : 0.5}
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
  focusTableId,
  roles,
  overview,
  onFocusPair,
  onFocusTable,
}: {
  nodes: RelationshipNode[];
  pairs: Pair[];
  pairsTotal: number;
  focusKey: string | null;
  focusTableId: string | null;
  roles: Map<string, TableRole>;
  overview: boolean;
  onFocusPair: (key: string) => void;
  onFocusTable: (datasetId: string) => void;
}) {
  return (
    <Card
      className={`relationship-graph-pane flex min-h-80 min-w-0 flex-1 flex-col overflow-hidden ${overview ? "relationship-overview-pane" : ""}`}
    >
      <RelationshipGraph
        nodes={nodes}
        pairs={pairs}
        focusKey={focusKey}
        focusTableId={focusTableId}
        roles={roles}
        overview={overview}
        onFocusPair={onFocusPair}
        onFocusTable={onFocusTable}
      />
      <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-2 border-t border-hairline px-3 py-2">
        <Legend />
        <p className="text-xs text-status-neutral">
          One line per table pair · {pairs.length} of {pairsTotal} pair(s) match
          the filter.{" "}
          {overview
            ? "Same-table field matches stay in List. Edge labels appear on selection. "
            : ""}
          Click a line, or focus it and press Enter, to scope the list.
        </p>
      </div>
    </Card>
  );
}

function PairSummary({ pair }: { pair: Pair }) {
  const edge = pair.representative;
  const reasons = relationshipReasons(edge);
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <Badge tone={STATE_TONE[pair.state]} caps>{pair.state}</Badge>
        <Badge tone={pair.kind === "many_to_many" ? "warn" : "neutral"}>
          {KIND_LABEL[pair.kind]}
        </Badge>
        <span className="tabular ml-auto text-xs text-status-neutral">
          best {pair.best.toFixed(2)}
        </span>
      </div>
      <RelationshipPair edge={edge} />
      <Card tone="quiet" className="p-2">
        <dl className="flex flex-col gap-1.5">
          <Row label="Source values matched" value={percent(edge.overlap_left_in_right)} />
          <Row label="Reverse value overlap" value={percent(edge.overlap_right_in_left)} />
          <Row label="Target key uniqueness" value={percent(edge.right_unique_rate)} />
          <Row label="Cardinality" value={edge.cardinality ?? "not validated"} />
          <Row
            label="Expected row multiplier"
            value={edge.join_row_multiplier == null ? "—" : `×${edge.join_row_multiplier.toFixed(2)}`}
          />
        </dl>
      </Card>
      <div className="flex flex-col gap-1.5">
        <span className="text-xs font-medium text-status-neutral">Why this pair surfaced</span>
        <ul className="flex flex-wrap gap-1.5">
          {reasons.map((reason) => (
            <li key={reason} className="rounded-sm border border-border bg-bg px-2 py-1 text-xs">
              {reason}
            </li>
          ))}
        </ul>
      </div>
      <p className="text-xs text-status-neutral">
        {pair.edges.length} directional column candidate{pair.edges.length === 1 ? "" : "s"} are grouped into this single table pair.
      </p>
    </div>
  );
}

function RelationshipMatrix({
  nodes,
  pairs,
  focusTableId,
  onFocusPair,
  onFocusTable,
}: {
  nodes: RelationshipNode[];
  pairs: Pair[];
  focusTableId: string | null;
  onFocusPair: (key: string) => void;
  onFocusTable: (datasetId: string) => void;
}) {
  const pairByIdentity = new Map(pairs.map((pair) => [pair.identity, pair]));
  const ordered = [...nodes].sort((left, right) => left.name.localeCompare(right.name));
  return (
    <Card className="relationship-matrix-pane min-w-0 overflow-auto">
      <table className="relationship-matrix w-full min-w-[48rem] border-collapse text-xs" aria-label="Table relationship matrix">
        <thead className="sticky top-0 z-10 bg-table-header-bg">
          <tr>
            <th className="sticky left-0 z-20 min-w-44 border-b border-r border-table-border bg-table-header-bg px-3 py-2 text-left font-medium">
              Table pair
            </th>
            {ordered.map((node) => (
              <th key={node.dataset_id} className="min-w-24 max-w-32 border-b border-table-border px-2 py-2 text-left font-medium">
                <button
                  type="button"
                  onClick={() => onFocusTable(node.dataset_id)}
                  title={node.name}
                  className={`max-w-28 truncate hover:text-primary ${node.dataset_id === focusTableId ? "text-primary" : ""}`}
                >
                  {node.name}
                </button>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {ordered.map((left, rowIndex) => (
            <tr key={left.dataset_id} className={rowIndex % 2 === 1 ? "bg-code-bg/35" : ""}>
              <th className="sticky left-0 z-10 border-r border-t border-table-border bg-bg px-3 py-2 text-left font-medium">
                <button
                  type="button"
                  onClick={() => onFocusTable(left.dataset_id)}
                  title={left.name}
                  className={`max-w-40 truncate hover:text-primary ${left.dataset_id === focusTableId ? "text-primary" : ""}`}
                >
                  {left.name}
                </button>
              </th>
              {ordered.map((right) => {
                if (left.dataset_id === right.dataset_id) {
                  return <td key={right.dataset_id} className="border-t border-table-border bg-surface/40 text-center text-status-neutral">—</td>;
                }
                const identity = [left.dataset_id, right.dataset_id].sort().join("↔");
                const pair = pairByIdentity.get(identity);
                return (
                  <td key={right.dataset_id} className="border-t border-table-border p-1 text-center">
                    {pair ? (
                      <button
                        type="button"
                        onClick={() => onFocusPair(pair.key)}
                        aria-label={`${left.name} and ${right.name}: ${KIND_LABEL[pair.kind]}, ${pair.edges.length} candidates`}
                        title={`${KIND_LABEL[pair.kind]} · ${pair.edges.length} candidates · best ${pair.best.toFixed(2)}`}
                        className="inline-flex min-h-8 w-full items-center justify-center gap-1 rounded-sm px-1 hover:bg-surface focus-visible:outline-2 focus-visible:outline-primary"
                      >
                        <Dot tone={STATE_TONE[pair.state]} />
                        <span className="font-mono text-[10px]">{KIND_SHORT[pair.kind]}</span>
                        {pair.edges.length > 1 && <span className="tabular text-[10px] text-status-neutral">{pair.edges.length}</span>}
                      </button>
                    ) : (
                      <span className="text-status-neutral/40">·</span>
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  );
}

function RelationshipPairList({
  pairs,
  onFocusPair,
}: {
  pairs: Pair[];
  onFocusPair: (key: string) => void;
}) {
  return (
    <Card className="min-w-0 overflow-hidden">
      <div className="grid grid-cols-[minmax(0,1fr)_auto_auto] gap-3 border-b border-table-border bg-table-header-bg px-3 py-2 text-xs font-medium text-status-neutral">
        <span>Table pair</span><span>Relationship</span><span>Evidence</span>
      </div>
      <ul className="flex flex-col">
        {pairs.map((pair, index) => (
          <li key={pair.identity} className={index % 2 === 1 ? "bg-code-bg/35" : ""}>
            <button
              type="button"
              onClick={() => onFocusPair(pair.key)}
              className="grid w-full grid-cols-[minmax(0,1fr)_auto_auto] items-center gap-3 border-t border-hairline px-3 py-2.5 text-left first:border-t-0 hover:bg-surface"
            >
              <span className="flex min-w-0 items-center gap-2">
                <Dot tone={STATE_TONE[pair.state]} />
                <Marquee className="text-sm font-medium">{pair.leftName} ↔ {pair.rightName}</Marquee>
              </span>
              <Badge tone={pair.kind === "many_to_many" ? "warn" : "neutral"}>{KIND_LABEL[pair.kind]}</Badge>
              <span className="tabular text-xs text-status-neutral">{pair.edges.length} · {pair.best.toFixed(2)}</span>
            </button>
          </li>
        ))}
      </ul>
    </Card>
  );
}

type PairPanel = "summary" | "candidates" | "validation";

function PairWorkspace({
  pair,
  panel,
  onPanelChange,
  selectedId,
  onSelect,
}: {
  pair: Pair;
  panel: PairPanel;
  onPanelChange: (panel: PairPanel) => void;
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="flex flex-col gap-3">
      <SegmentedControl
        label="Pair details"
        value={panel}
        options={[
          { value: "summary", label: "Summary" },
          { value: "candidates", label: `Candidates · ${pair.edges.length}` },
          { value: "validation", label: "Validate" },
        ]}
        onChange={(value) => onPanelChange(value as PairPanel)}
      />
      {panel === "summary" && <PairSummary pair={pair} />}
      {panel === "candidates" && (
        <PairGroup
          pair={pair}
          expanded
          selectedId={selectedId}
          onSelect={onSelect}
        />
      )}
      {panel === "validation" && (
        <div className="flex flex-col gap-2">
          <Card tone="quiet" className="p-2.5">
            <p className="text-xs text-status-neutral">
              Select a column candidate to inspect its evidence and run a
              full-table validation. Confirmed joins remain available to the
              project until revoked.
            </p>
          </Card>
          {pair.edges.map((edge) => (
            <EdgeRow
              key={edge.relationship_id}
              edge={edge}
              selected={edge.relationship_id === selectedId}
              onSelect={() => onSelect(edge.relationship_id)}
            />
          ))}
        </div>
      )}
    </div>
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
  const [stateParam, setStateParam] = useRouteSearchParam(
    "state",
    "candidate,validated,confirmed",
  );
  const states = new Set<EdgeState>(
    stateParam === "none"
      ? []
      : parseCsvParam(stateParam).filter((state): state is EdgeState =>
          RELATIONSHIP_STATES.includes(state as EdgeState),
        ),
  );
  const [focusKeyParam] = useRouteSearchParam("pair");
  const focusKey = focusKeyParam || null;
  const [selectedIdParam, setSelectedId] = useRouteSearchParam("edge");
  const selectedId = selectedIdParam || null;
  const [viewParam, setViewParam] = useRouteSearchParam("view");
  const [tableParam] = useRouteSearchParam("table");
  const [scopeParam, setScopeParam] = useRouteSearchParam("scope");
  const [panelParam, setPanelParam] = useRouteSearchParam("detail");
  const setRouteSearchParams = useSetRouteSearchParams();

  const toggleConfidence = (level: string) => {
    const next = new Set(confidenceLevels);
    if (next.has(level)) next.delete(level);
    else next.add(level);
    setConfidenceParam(
      next.size === 0 ? "none" : serializeCsvParam(next),
    );
  };
  const toggleState = (state: EdgeState) => {
    const next = new Set(states);
    if (next.has(state)) next.delete(state);
    else next.add(state);
    setStateParam(
      next.size === 0 || next.size === RELATIONSHIP_STATES.length
        ? next.size === 0
          ? "none"
          : ""
        : serializeCsvParam(next),
    );
  };

  const visibleEdges = useMemo(
    () =>
      edges.filter(
        (edge) =>
          edge.ensemble_score >= minimumScore &&
          confidenceLevels.has(edge.confidence) &&
          states.has(edgeState(edge)),
      ),
    [edges, minimumScore, confidenceLevels, states],
  );
  const allPairs = useMemo(() => groupByPair(edges), [edges]);
  const pairs = useMemo(() => groupByPair(visibleEdges), [visibleEdges]);
  const pairsTotal = allPairs.length;
  const overviewPairs = useMemo(
    () => pairs.filter((pair) => pair.leftId !== pair.rightId),
    [pairs],
  );
  const overviewPairsTotal = useMemo(
    () => allPairs.filter((pair) => pair.leftId !== pair.rightId).length,
    [allPairs],
  );
  const rolePairs = useMemo(
    () =>
      groupByPair(
        edges.filter(
          (edge) => edge.confidence !== "low" || edgeState(edge) !== "candidate",
        ),
      ),
    [edges],
  );
  const roles = useMemo(() => inferTableRoles(nodes, rolePairs), [nodes, rolePairs]);
  const defaultFocusTableId = useMemo(() => {
    if (nodes.length === 0) return null;
    const degree = new Map(nodes.map((node) => [node.dataset_id, 0]));
    for (const pair of allPairs) {
      degree.set(pair.leftId, (degree.get(pair.leftId) ?? 0) + 1);
      degree.set(pair.rightId, (degree.get(pair.rightId) ?? 0) + 1);
    }
    return [...nodes].sort(
      (left, right) =>
        (degree.get(right.dataset_id) ?? 0) -
          (degree.get(left.dataset_id) ?? 0) ||
        left.name.localeCompare(right.name),
    )[0]?.dataset_id ?? null;
  }, [nodes, allPairs]);
  const focusTableId = nodes.some((node) => node.dataset_id === tableParam)
    ? tableParam
    : defaultFocusTableId;
  const view: RelationshipView =
    viewParam === "overview" ||
    viewParam === "neighborhood" ||
    viewParam === "matrix" ||
    viewParam === "list"
      ? viewParam
      : nodes.length >= 9
        ? "matrix"
        : "neighborhood";
  const panel: PairPanel =
    panelParam === "candidates" || panelParam === "validation"
      ? panelParam
      : "summary";
  const overviewFocusTableId = nodes.some(
    (node) => node.dataset_id === tableParam,
  )
    ? tableParam
    : null;
  const focused =
    pairs.find(
      (pair) => pair.key === focusKey || pair.legacyKeys.includes(focusKey ?? ""),
    ) ?? null;
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

  const focusPairs = focusTableId
    ? pairs.filter(
        (pair) => pair.leftId === focusTableId || pair.rightId === focusTableId,
      )
    : [];
  const neighborhoodPairs = (() => {
    if (scopeParam === "all") return focusPairs;
    if (!focused || !focusPairs.some((pair) => pair.identity === focused.identity)) {
      return focusPairs.slice(0, NEIGHBOR_LIMIT);
    }
    return [
      focused,
      ...focusPairs.filter((pair) => pair.identity !== focused.identity),
    ].slice(0, NEIGHBOR_LIMIT);
  })();
  const neighborhoodNodeIds = new Set<string>(
    focusTableId
      ? [
          focusTableId,
          ...neighborhoodPairs.flatMap((pair) => [pair.leftId, pair.rightId]),
        ]
      : [],
  );
  const neighborhoodNodes = nodes.filter((node) =>
    neighborhoodNodeIds.has(node.dataset_id),
  );

  const focusPair = (key: string) => {
    const pair = pairs.find(
      (item) => item.key === key || item.legacyKeys.includes(key),
    );
    const next: Record<string, string> = {
      pair: pair?.key ?? key,
      edge:
        pair && pair.edges.length === 1
          ? (pair.edges[0]?.relationship_id ?? "")
          : "",
      detail: "",
    };
    if (
      pair &&
      focusTableId !== pair.leftId &&
      focusTableId !== pair.rightId
    ) {
      next.table = pair.leftId === defaultFocusTableId ? "" : pair.leftId;
      next.scope = "";
    }
    setRouteSearchParams(next);
  };

  const focusTable = (datasetId: string) => {
    setRouteSearchParams({
      table:
        view === "overview"
          ? datasetId
          : datasetId === defaultFocusTableId
            ? ""
            : datasetId,
      scope: "",
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
      <Card className="relationship-toolbar mb-3 flex flex-wrap items-center gap-3 p-2.5">
        <label className="flex min-w-0 items-center gap-2 text-xs font-medium text-status-neutral">
          Focus table
          <select
            aria-label="Focus table"
            value={view === "overview" ? overviewFocusTableId ?? "" : focusTableId ?? ""}
            onChange={(event) => focusTable(event.target.value)}
            className="min-w-0 max-w-64 rounded-base border border-border bg-bg px-2.5 py-1.5 text-sm text-text"
          >
            {view === "overview" && <option value="">All tables</option>}
            {[...nodes]
              .sort((left, right) => left.name.localeCompare(right.name))
              .map((node) => (
                <option key={node.dataset_id} value={node.dataset_id}>
                  {node.name} · {ROLE_LABEL[roles.get(node.dataset_id) ?? "isolated"]}
                </option>
              ))}
          </select>
        </label>
        <SegmentedControl
          label="Relationship view"
          value={view}
          options={[
            { value: "overview", label: "Overview" },
            { value: "neighborhood", label: "Neighborhood" },
            { value: "matrix", label: "Matrix" },
            { value: "list", label: "List" },
          ]}
          onChange={(value) =>
            setViewParam(value === (nodes.length >= 9 ? "matrix" : "neighborhood") ? "" : value)
          }
        />
        <span className="ml-auto text-xs text-status-neutral">
          {view === "overview" ? overviewPairs.length : pairs.length} of{" "}
          {view === "overview" ? overviewPairsTotal : pairsTotal} table pairs in
          scope
        </span>
        {view === "neighborhood" && focusPairs.length > NEIGHBOR_LIMIT && (
          <button
            type="button"
            onClick={() => setScopeParam(scopeParam === "all" ? "" : "all")}
            className="rounded-base border border-border px-2.5 py-1.5 text-xs hover:bg-surface"
          >
            {scopeParam === "all"
              ? `Show strongest ${NEIGHBOR_LIMIT}`
              : `Show all ${focusPairs.length} related tables`}
          </button>
        )}
      </Card>
      <div className="relationship-workbench-layout grid min-h-full min-w-0 gap-4">
        <main className="relationship-main min-h-0 min-w-0">
          {view === "overview" ? (
            <GraphPane
              nodes={nodes}
              pairs={overviewPairs}
              pairsTotal={overviewPairsTotal}
              focusKey={focusKey}
              focusTableId={overviewFocusTableId}
              roles={roles}
              overview
              onFocusPair={focusPair}
              onFocusTable={focusTable}
            />
          ) : view === "neighborhood" ? (
            <GraphPane
              nodes={neighborhoodNodes}
              pairs={neighborhoodPairs}
              pairsTotal={focusPairs.length}
              focusKey={focusKey}
              focusTableId={focusTableId}
              roles={roles}
              overview={false}
              onFocusPair={focusPair}
              onFocusTable={focusTable}
            />
          ) : view === "matrix" ? (
            <RelationshipMatrix
              nodes={nodes}
              pairs={pairs}
              focusTableId={focusTableId}
              onFocusPair={focusPair}
              onFocusTable={focusTable}
            />
          ) : (
            <RelationshipPairList pairs={pairs} onFocusPair={focusPair} />
          )}
        </main>
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
                        states={states}
                        onToggleState={toggleState}
                      />
                      <ScopeReadout
                        edges={edges}
                        visibleCount={visibleEdges.length}
                        minimumScore={minimumScore}
                        confidenceLevels={confidenceLevels}
                        states={states}
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
                    {focused.leftName} ↔ {focused.rightName}
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
              ) : focused ? (
                <PairWorkspace
                  pair={focused}
                  panel={panel}
                  onPanelChange={(next) =>
                    setPanelParam(next === "summary" ? "" : next)
                  }
                  selectedId={selectedId}
                  onSelect={selectEdge}
                />
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
    return (
      <DataWorkspacePage
        title="Relationships"
        description="Discover and validate how tables connect. A candidate never becomes usable as a join until it is confirmed."
      >
        <LoadingSkeleton lines={4} label="Loading relationships" />
      </DataWorkspacePage>
    );
  }
  if (graph.isError) {
    return (
      <DataWorkspacePage
        title="Relationships"
        description="Discover and validate how tables connect. A candidate never becomes usable as a join until it is confirmed."
      >
        <ErrorState error={graph.error} onRetry={() => graph.refetch()} />
      </DataWorkspacePage>
    );
  }

  const nodes = graph.data.nodes ?? [];
  const edges = graph.data.edges ?? [];
  const pairCount = groupByPair(edges).length;
  const confirmedJoinCount = edges.filter(
    (edge) => edgeState(edge) === "confirmed",
  ).length;

  return (
    <DataWorkspacePage
      title="Relationships"
      description="Discover and validate how tables connect. A candidate never becomes usable as a join until it is confirmed."
    >
      {graph.data.discovered && (
        /* Before discovery runs these read "0 pairs, 0 candidates, 0 joins",
         * which states a finding the session never looked for. The pre-run
         * state is carried by the panel below instead. */
        <MetricStrip>
          <MetricTile label="Datasets" value={formatCompact(nodes.length)} />
          <MetricTile
            label="Dataset pairs"
            value={formatCompact(pairCount)}
            hint="Unique source-to-target table pairs with at least one candidate."
          />
          <MetricTile
            label="Column candidates"
            value={formatCompact(edges.length)}
            hint="Scored column-level relationships across all dataset pairs."
          />
          <MetricTile
            label="Confirmed joins"
            value={formatCompact(confirmedJoinCount)}
            tone={confirmedJoinCount > 0 ? "ok" : "neutral"}
            hint="Column relationships promoted for use as joins."
          />
          <MetricTile label="Discovery" value="Complete" tone="ok" />
        </MetricStrip>
      )}

      <SearchCoverage graph={graph.data} />

      {nodes.length === 0 ? (
        <EmptyState
          title="No datasets in this session"
          description="Upload data and start a session to explore relationships."
        />
      ) : !graph.data.discovered ? (
        /* The whole workbench is scaffolding for data that does not exist yet:
         * a focus picker listing every table as "Isolated", and a 12x12 matrix
         * of empty cells, with the one useful control — Discover — below all
         * 144 of them. */
        <PreDiscoveryPanel
          projectId={projectId}
          sessionId={sessionId}
          datasetCount={nodes.length}
          canDiscover={
            nodes.filter((node) => node.source_available).length >= 2
          }
        />
      ) : (
        <Workbench
          key={`workbench-${sessionId}`}
          projectId={projectId}
          sessionId={sessionId}
          graph={graph.data}
        />
      )}
    </DataWorkspacePage>
  );
}
