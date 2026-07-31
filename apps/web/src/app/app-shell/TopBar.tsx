import { useEffect, useState } from "react";
import { Link, useLocation, useParams } from "react-router";
import { useSessionDetail, useSessionMetrics } from "../../api/hooks";
import {
  Badge,
  Button,
  Dot,
  Hint,
  IconButton,
  Marquee,
  type Tone,
} from "../../components/ui";
import { LlmStatusDot, useLlmStatus } from "./LlmStatus";
import { NavIcon } from "./nav-icons";
import {
  getEffectiveTheme,
  hasStoredTheme,
  setTheme,
  type Theme,
} from "../theme";

function statusTone(status: string | undefined): Tone {
  switch (status) {
    case "completed":
      return "ok";
    case "running":
      return "warn";
    case "failed":
    case "cancelled":
      return "critical";
    default:
      return "neutral";
  }
}

interface TopBarProps {
  inspectorCollapsed: boolean;
  onToggleInspector: () => void;
  onOpenSessions: () => void;
  onToggleSessions: () => void;
  sessionsCollapsed: boolean;
  onOpenSettings: () => void;
  showInspector: boolean;
}

export function TopBar({
  inspectorCollapsed,
  onToggleInspector,
  onOpenSessions,
  onToggleSessions,
  sessionsCollapsed,
  onOpenSettings,
  showInspector,
}: TopBarProps) {
  const { projectId, sessionId } = useParams();
  const { pathname } = useLocation();
  const atHome = pathname === "/projects";
  const llm = useLlmStatus();
  const [theme, setThemeState] = useState<Theme>(() => getEffectiveTheme());
  // Status and cost were hardcoded placeholders; they now read the same
  // sources as the Data Map and Trace pages so the header cannot disagree.
  const runDetail = useSessionDetail(sessionId ?? "");
  const metrics = useSessionMetrics(sessionId ?? "");

  /* Without a stored choice the CSS follows the OS via prefers-color-scheme;
   * keep the button's label in sync when the OS theme changes. */
  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => {
      if (!hasStoredTheme()) setThemeState(getEffectiveTheme());
    };
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  const toggleTheme = () => {
    const next: Theme = theme === "dark" ? "light" : "dark";
    setTheme(next);
    setThemeState(next);
  };

  const status = sessionId ? runDetail.data?.status : undefined;
  /* The run id was the headline here, which meant the bar read
   * "run_1785039866191_k1det2" while the run's actual name sat unused in the
   * same payload. Title leads; the id stays visible but subordinate, because
   * it is what you paste into a bug report. */
  const runTitle = runDetail.data?.title;

  return (
    /* Named because a page's own <header> also maps to the banner role, and
     * an unnamed query then matches two landmarks. */
    <header
      aria-label="Workbench"
      className="flex min-h-12 shrink-0 items-center gap-2 border-b border-border bg-surface px-3 py-2 sm:h-12 sm:gap-3 sm:px-4 sm:py-0"
    >
      {/* Wrapper, not `className="hidden md:…"` on the control: IconButton's
        * base sets inline-flex and Tailwind resolves that conflict by CSS
        * order, so the hidden never applied and the button stayed clickable on
        * mobile — where it wrote a rail-collapsed preference for a rail that
        * is not even rendered. */}
      <span className="hidden md:inline-flex">
        <IconButton
          label={sessionsCollapsed ? "Expand sessions" : "Collapse sessions"}
          onClick={onToggleSessions}
        >
          <RailGlyph collapsed={sessionsCollapsed} />
        </IconButton>
      </span>
      <nav
        aria-label="Context"
        className="hidden min-w-0 items-baseline gap-1.5 text-sm md:flex"
      >
        {atHome ? (
          <span className="font-medium">Home</span>
        ) : (
          <>
            <Link
              to="/projects"
              className="shrink-0 text-status-neutral hover:text-primary hover:underline"
            >
              {projectId ?? "—"}
            </Link>
            <span aria-hidden className="shrink-0 text-status-neutral/60">
              /
            </span>
            {sessionId ? (
              <>
                <Marquee className="font-medium" title={runTitle ?? sessionId}>
                  {runTitle ?? "Untitled session"}
                </Marquee>
                <span
                  className="shrink-0 font-mono text-xs text-status-neutral"
                  title="Session id"
                >
                  {sessionId}
                </span>
              </>
            ) : (
              <span className="text-status-neutral">no session open</span>
            )}
          </>
        )}
      </nav>
      <div className="ml-auto flex shrink-0 items-center gap-1 text-sm sm:gap-2">
        <Button size="sm" onClick={onOpenSessions} className="md:hidden">
          Sessions
        </Button>
        {sessionId && (
          <span
            className="hidden items-center gap-1.5 sm:flex"
            title="Session status"
          >
            <Dot
              tone={statusTone(status)}
              motion={status === "running" ? "working" : undefined}
            />
            <span className="text-sm">{status ?? "…"}</span>
          </span>
        )}
        {sessionId && (
          <span className="hidden items-center gap-1 lg:flex">
            <Badge tone="neutral" variant="outline">
              <span className="tabular font-mono">
                {typeof metrics.data?.est_cost_usd === "number"
                  ? `$${metrics.data.est_cost_usd.toFixed(4)}`
                  : "$—"}
              </span>
            </Badge>
            <Hint label="Estimated cost">
              Model spend for this session, priced from the token counts in its
              trace. It is an estimate: your provider's invoice is the
              authority.
            </Hint>
          </span>
        )}
        {/* The rail footer used to own this, which meant collapsing the rail
         * hid the one signal that says whether a session will call a paid model. */}
        {llm && (
          /* The dot never hides — it is the only signal for "will this session
           * call a paid model", and it costs 8px. Only its label drops. */
          <span className="flex items-center gap-1.5 text-xs text-status-neutral">
            <LlmStatusDot status={llm} />
            <Marquee className="hidden max-w-40 font-mono lg:inline">
              {llm.label}
            </Marquee>
          </span>
        )}
        <IconButton
          label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
          onClick={toggleTheme}
        >
          <ThemeGlyph theme={theme} />
        </IconButton>
        {showInspector && (
          <span className="hidden md:inline-flex">
            <IconButton
              label="Inspector"
              onClick={onToggleInspector}
              aria-pressed={!inspectorCollapsed}
            >
              <InspectorGlyph />
            </IconButton>
          </span>
        )}
        <IconButton label="Settings" onClick={onOpenSettings}>
          <NavIcon name="settings" />
        </IconButton>
      </div>
    </header>
  );
}

function InspectorGlyph() {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3.5" y="4" width="17" height="16" rx="2" />
      <path d="M15 4v16" />
    </svg>
  );
}

function RailGlyph({ collapsed }: { collapsed: boolean }) {
  return <svg aria-hidden="true" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round"><rect x="3.5" y="4" width="17" height="16" rx="2" /><path d="M9 4v16" />{collapsed ? <path d="m13 9 3 3-3 3" /> : <path d="m16 9-3 3 3 3" />}</svg>;
}

function ThemeGlyph({ theme }: { theme: Theme }) {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 24 24"
      width="16"
      height="16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.75"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {theme === "dark" ? (
        <path d="M20 14.5A8.5 8.5 0 1 1 9.5 4a7 7 0 0 0 10.5 10.5z" />
      ) : (
        <>
          <circle cx="12" cy="12" r="4" />
          <path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M19.1 4.9l-1.4 1.4M6.3 17.7l-1.4 1.4" />
        </>
      )}
    </svg>
  );
}
