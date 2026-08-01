import { useId, useState } from "react";
import { Link, useLocation, useParams } from "react-router";
import { useDatasets } from "../../api/hooks";
import { readBaseline } from "../../features/compare/baseline-storage";
import {
  projectComparePath,
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

/* Three stages of work, in the order a session moves through them. Board sits
 * with the investigation pages because it is the kanban over them; Trace & cost
 * and Artifacts sit together because both answer "where did this number come
 * from". */
function useNavGroups(projectId: string, sessionId: string): NavGroup[] {
  const datasets = useDatasets(sessionId);
  const firstDatasetId = datasets.data?.[0]?.dataset_id;
  const at = (section: string) => sessionSectionPath(projectId, sessionId, section);
  const pinned = readBaseline(projectId);
  const comparePath =
    pinned && pinned !== sessionId
      ? projectComparePath(projectId, pinned, sessionId)
      : projectComparePath(projectId, sessionId);

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
        { label: "Profiles & charts", icon: "chart", to: at("profiles") },
        {
          label: "Cleaning info and raw data",
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

/* Header of the centre column, not of the window: it sits inside the main
 * panel so the session rail and the Inspector keep their full height, and the
 * rail stays a session switcher rather than competing with 17 page links. */
const barClass =
  "flex min-w-0 shrink-0 flex-col gap-1 border-b border-border bg-bg px-3 pt-2 pb-1.5 sm:px-4";

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

/* Deliberately no step numbers: the three stages are an order of work, not a
 * progress bar, and a numbered chip claims a completion state nothing here
 * measures. The chevrons carry the sequence; type weight carries the state. */
function StageButton({
  group,
  selected,
  current,
  panelId,
  onSelect,
}: {
  group: NavGroup;
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
      className={`flex items-center gap-1.5 rounded-base px-1.5 py-0.5 text-xs transition-colors duration-150 ease-out-quart ${
        selected
          ? "font-semibold text-primary"
          : "text-status-neutral hover:text-text"
      }`}
    >
      {group.title}
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
  const groups = useNavGroups(projectId, sessionId);
  const { pathname } = useLocation();
  const panelId = useId();
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

  return (
    <nav aria-label="Session sections" className={barClass}>
      <div className="flex min-w-0 flex-wrap items-center gap-x-1 gap-y-0.5">
        {groups.map((group, index) => (
          <div key={group.title} className="flex items-center gap-1">
            {index > 0 && (
              <span aria-hidden className="text-xs text-status-neutral/40">
                ›
              </span>
            )}
            <StageButton
              group={group}
              selected={group.title === selected.title}
              current={group.title === currentGroup.title}
              panelId={panelId}
              onSelect={() => setBrowsing({ pathname, title: group.title })}
            />
          </div>
        ))}
      </div>
      <ul
        id={panelId}
        aria-label={selected.title}
        className="flex min-w-0 flex-wrap items-center gap-x-0.5 gap-y-0.5"
      >
        {selected.pages.map((page) => {
          const active = pageMatches(pathname, page);
          return (
            <li key={page.to}>
              <Link
                to={page.to}
                aria-current={active ? "page" : undefined}
                className={itemClass({ isActive: active })}
              >
                <NavIcon name={page.icon} />
                {page.label}
              </Link>
            </li>
          );
        })}
      </ul>
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
