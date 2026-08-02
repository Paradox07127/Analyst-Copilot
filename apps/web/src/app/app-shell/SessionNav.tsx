import { useEffect, useId, useRef, useState } from "react";
import { Link, useLocation, useParams } from "react-router";
import { useDatasets, useEdaHandoff, useSessionDetail } from "../../api/hooks";
import { readBaseline } from "../../features/compare/baseline-storage";
import {
  sessionBasePath,
  sessionSectionPath,
  tablePath,
} from "../paths";
import { NavIcon, type IconName } from "./nav-icons";

interface NavPage {
  label: string;
  icon: IconName;
  to: string;
  activePath?: string;
}

interface NavGroup {
  title: string;
  pages: NavPage[];
}

type StageState = "ready" | "running" | "waiting";

const FINISHED_SESSION_STATUSES = new Set([
  "complete",
  "completed",
  "failed",
  "cancelled",
]);

/* Three stages of work, in the order a session moves through them. Board sits
 * with the investigation pages because it is the kanban over them; Trace & cost
 * and Artifacts sit together because both answer "where did this number come
 * from". */
function buildNavGroups(
  projectId: string,
  sessionId: string,
  firstDatasetId?: string,
): NavGroup[] {
  const at = (section: string) => sessionSectionPath(projectId, sessionId, section);
  const pinned = readBaseline(projectId);
  /* Compare belongs to the current run's workbench, not a project-level
   * standalone surface. Keeping the session path preserves the top context,
   * section navigation and Inspector while the query still names both sides
   * of the comparison. */
  const compareParams = new URLSearchParams(
    pinned && pinned !== sessionId
      ? { left: pinned, right: sessionId }
      : { left: sessionId },
  );
  const comparePath = `${at("compare")}?${compareParams.toString()}`;

  return [
    {
      title: "Understand the data",
      pages: [
        { label: "Data map", icon: "dashboard", to: at("data-map") },
        ...(firstDatasetId
          ? [
              {
                label: "Table preview",
                icon: "table" as const,
                to: tablePath(projectId, sessionId, firstDatasetId),
                /* The control opens the first table, while every table route
                 * belongs to this one section. */
                activePath: `${sessionBasePath(projectId, sessionId)}/table`,
              },
            ]
          : []),
        { label: "Quality", icon: "rule", to: at("quality") },
        { label: "Profiles & charts", icon: "profile", to: at("profiles") },
        {
          label: "Cleanup",
          icon: "cleaning",
          to: at("cleaning"),
        },
        { label: "Relationships", icon: "hub", to: at("relationships") },
        { label: "Knowledge", icon: "book", to: at("semantic") },
      ],
    },
    {
      title: "Investigate with the agent",
      pages: [
        { label: "Questions", icon: "quiz", to: at("questions") },
        { label: "Deep analysis", icon: "analytics", to: at("deep-analysis") },
        { label: "Findings", icon: "factCheck", to: at("findings") },
        {
          label: "Compare",
          icon: "compare",
          /* A pinned baseline is the left side and the run you came from
           * becomes the variant. Linking the current run as `left` made the
           * pin unreachable: ComparePage only consults it when `left` is
           * absent, so priority 1 of the entry rules could never fire. */
          to: comparePath,
        },
        { label: "Chat", icon: "chat", to: at("chat") },
        { label: "Skills", icon: "sparkle", to: at("skills") },
        { label: "Report", icon: "description", to: at("report") },
        { label: "Board", icon: "board", to: at("board") },
      ],
    },
    {
      title: "Trust & trace",
      pages: [
        { label: "Trace & cost", icon: "timeline", to: at("trace") },
        { label: "Artifacts", icon: "box", to: at("artifacts") },
      ],
    },
  ];
}

/* A handoff is more trustworthy than a session's coarse lifecycle status:
 * auto-EDA publishes it when the data surfaces are safe to browse, while the
 * agent can still be drafting questions and the report afterwards. */
function usePipelineReadiness(sessionId: string, hasDatasets: boolean): {
  data: StageState;
  agent: StageState;
} {
  const session = useSessionDetail(sessionId);
  const edaHandoff = useEdaHandoff(sessionId);
  const status = session.data?.status?.toLowerCase();
  const finished = status ? FINISHED_SESSION_STATUSES.has(status) : false;
  const completed = status === "complete" || status === "completed";
  /* A failed or cancelled run can still leave a complete, browseable dataset
   * workspace behind. The data itself is stronger evidence than the coarse
   * session status, so do not lock users out of partial EDA results. */
  const dataReady = Boolean(edaHandoff.data) || completed || hasDatasets;
  const active = Boolean(status && !finished);

  useEffect(() => {
    if (!active) return;
    const timer = window.setInterval(() => {
      void session.refetch();
      void edaHandoff.refetch();
    }, 2000);
    return () => window.clearInterval(timer);
  }, [active, edaHandoff.refetch, session.refetch]);

  return {
    data: dataReady ? "ready" : active ? "running" : "waiting",
    agent: completed ? "ready" : dataReady && active ? "running" : "waiting",
  };
}

/* Header of the centre column, not of the window: it sits inside the main
 * panel so the session rail and the Inspector keep their full height, and the
 * rail stays a session switcher rather than competing with 17 page links. */
const barClass =
  "relative z-30 flex min-w-0 shrink-0 flex-col gap-1 overflow-visible border-b border-border bg-bg px-3 py-1.5 sm:px-4";

const itemClass = ({ isActive }: { isActive: boolean }) =>
  `flex items-center gap-1.5 rounded-base px-2 py-1 text-sm transition-colors duration-150 ease-out-quart ${
    isActive
      ? "bg-surface font-medium text-primary"
      : "text-status-neutral hover:bg-surface hover:text-text"
  }`;

function matchesPath(pathname: string, to: string): boolean {
  const base = to.split("?")[0] ?? "";
  return pathname === base || pathname.startsWith(`${base}/`);
}

function pageMatches(pathname: string, page: NavPage): boolean {
  return matchesPath(pathname, page.activePath ?? page.to);
}

function unavailableReason(
  groupTitle: string,
  readiness: { data: StageState; agent: StageState },
): string | undefined {
  if (groupTitle === "Understand the data" && readiness.data !== "ready") {
    return "EDA is still preparing this workspace.";
  }
  if (groupTitle !== "Investigate with the agent" || readiness.agent === "ready") {
    return undefined;
  }
  return readiness.agent === "waiting"
    ? "Agent work begins after EDA publishes the data workspace."
    : "The agent is still preparing this analysis.";
}

/* The stage menu is intentionally on-demand. It is navigation rather than a
 * progress meter, so a compact current-stage trigger carries orientation at
 * rest and the full three-way choice only takes space while needed. */
function StagePickerButton({
  group,
  index,
  selected,
  current,
  panelId,
  onSelect,
}: {
  group: NavGroup;
  index: number;
  selected: boolean;
  current: boolean;
  panelId: string;
  onSelect: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onSelect}
      aria-current={current ? "true" : undefined}
      aria-controls={panelId}
      aria-label={`Show ${group.title} pages`}
      className={`flex items-center gap-1.5 rounded-base border px-2.5 py-1.5 text-sm transition-colors duration-150 ease-out-quart ${
        selected
          ? "border-primary/30 bg-primary/10 font-semibold text-primary"
          : "border-transparent text-status-neutral hover:bg-surface hover:text-text"
      }`}
    >
      <span className="inline-flex size-5 shrink-0 items-center justify-center rounded-full border border-current/25 font-mono text-[11px]">
        {index + 1}
      </span>
      <span>{group.title}</span>
      {/* Only when the two diverge: looking ahead at another stage must not
       * make the bar forget which stage the open page belongs to. */}
      {current && !selected && (
        <span aria-hidden className="size-1.5 rounded-full bg-primary" />
      )}
    </button>
  );
}

function SessionNavGroups({
  projectId,
  sessionId,
}: {
  projectId: string;
  sessionId: string;
}) {
  const datasets = useDatasets(sessionId);
  const readiness = usePipelineReadiness(
    sessionId,
    Boolean(datasets.data?.length),
  );
  const groups = buildNavGroups(
    projectId,
    sessionId,
    datasets.data?.[0]?.dataset_id,
  );
  const { pathname } = useLocation();
  const panelId = useId();
  const pickerId = useId();
  const navRef = useRef<HTMLElement>(null);
  const [stagePickerOpen, setStagePickerOpen] = useState(false);
  /* Stamped with the route it was chosen on, and compared during render rather
   * than cleared by an effect: an effect runs after commit, so the first render
   * on a new route would still show the stage picked on the previous one. */
  const [browsing, setBrowsing] = useState<{ pathname: string; title: string } | null>(
    null,
  );

  const currentGroup =
    groups.find((group) => group.pages.some((page) => pageMatches(pathname, page))) ??
    groups[0]!;
  const lookahead = browsing?.pathname === pathname ? browsing.title : null;
  const selected =
    groups.find((group) => group.title === lookahead) ?? currentGroup;

  useEffect(() => {
    setStagePickerOpen(false);
  }, [pathname]);

  useEffect(() => {
    const activeLink = navRef.current?.querySelector<HTMLAnchorElement>(
      'a[aria-current="page"]',
    );
    activeLink?.scrollIntoView?.({ block: "nearest", inline: "center" });
  }, [pathname, selected.title]);

  useEffect(() => {
    if (!stagePickerOpen) return;
    const closeOnOutsideClick = (event: PointerEvent) => {
      if (!navRef.current?.contains(event.target as Node)) {
        setStagePickerOpen(false);
      }
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setStagePickerOpen(false);
    };
    document.addEventListener("pointerdown", closeOnOutsideClick);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("pointerdown", closeOnOutsideClick);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [stagePickerOpen]);

  return (
    <nav ref={navRef} aria-label="Session sections" className={barClass}>
      <div className="flex min-w-0 items-center gap-2">
        <button
          type="button"
          aria-expanded={stagePickerOpen}
          aria-controls={pickerId}
          aria-current={selected.title === currentGroup.title ? "true" : undefined}
          onClick={() => setStagePickerOpen((open) => !open)}
          className="inline-flex shrink-0 items-center gap-1.5 rounded-base px-2 py-1.5 text-[15px] font-semibold text-text transition-colors duration-150 ease-out-quart hover:bg-surface hover:text-primary focus-visible:outline-offset-1"
        >
          {selected.title}
          <svg
            aria-hidden="true"
            viewBox="0 0 16 16"
            width="14"
            height="14"
            className={`shrink-0 transition-transform duration-150 ${stagePickerOpen ? "rotate-180" : ""}`}
          >
            <path d="m4 6 4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>
        <span aria-hidden className="h-4 w-px shrink-0 bg-border" />
        <ul
          id={panelId}
          aria-label={selected.title}
          className="flex min-w-0 flex-1 items-center gap-x-0.5 overflow-x-auto"
        >
          {selected.pages.map((page) => {
            const active = pageMatches(pathname, page);
            const reason = unavailableReason(selected.title, readiness);
            return (
              <li key={page.to} className="shrink-0">
                {reason ? (
                  <span
                    aria-disabled="true"
                    title={reason}
                    className="flex cursor-not-allowed items-center gap-1.5 rounded-base px-2 py-1 text-sm text-status-neutral/50"
                  >
                    <NavIcon name={page.icon} />
                    {page.label}
                  </span>
                ) : (
                  <Link
                    to={page.to}
                    aria-current={active ? "page" : undefined}
                    className={itemClass({ isActive: active })}
                  >
                    <NavIcon name={page.icon} />
                    {page.label}
                  </Link>
                )}
              </li>
            );
          })}
        </ul>
      </div>
      {stagePickerOpen && (
        <div
          id={pickerId}
          aria-label="Choose a work stage"
          className="animate-enter absolute left-3 top-[calc(100%+0.25rem)] z-40 flex max-w-[calc(100%-1.5rem)] flex-wrap items-center gap-1 rounded-base border border-border bg-surface p-2 shadow-overlay"
        >
          {groups.map((group, index) => (
            <StagePickerButton
              key={group.title}
              group={group}
              index={index}
              selected={group.title === selected.title}
              current={group.title === currentGroup.title}
              panelId={panelId}
              onSelect={() => {
                setBrowsing({ pathname, title: group.title });
                setStagePickerOpen(false);
              }}
            />
          ))}
        </div>
      )}
    </nav>
  );
}

/* The section links only exist once a session is loaded: every one of them
 * interpolates a session id, so before that they would all 404. */
export function SessionNav({
  projectId: projectIdProp,
  sessionId: sessionIdProp,
}: {
  projectId?: string;
  sessionId?: string;
} = {}) {
  const route = useParams();
  const projectId = projectIdProp ?? route.projectId;
  const sessionId = sessionIdProp ?? route.sessionId;
  if (!projectId || !sessionId) return null;
  return <SessionNavGroups projectId={projectId} sessionId={sessionId} />;
}
