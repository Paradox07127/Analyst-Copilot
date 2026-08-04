import { useState } from "react";
import { Link, useSearchParams } from "react-router";
import {
  type ProjectSummary,
  type UsageRecentSession,
  type WorkspaceUsageView,
} from "../../api/client";
import {
  useProjects,
  useWorkspaceUsage,
} from "../../api/hooks";
import {
  ErrorState,
  LoadingSkeleton,
} from "../../components/async-states";
import {
  Button,
  Card,
  Dot,
  Marquee,
  SectionHeader,
  buttonClass,
  formatCompact,
  type Tone,
} from "../../components/ui";
import { ActivityGrid } from "./ActivityGrid";
import { projectLabel } from "../../app/unfiled";
import { newProjectSessionPath, newSessionPath, sessionBasePath } from "../../app/paths";
import { DeleteProjectDialog } from "./DeleteProjectDialog";

const DEFAULT_USAGE_WINDOW_DAYS = 180;
const MONTH_USAGE_WINDOW_DAYS = 30;
const WEEK_USAGE_WINDOW_DAYS = 7;
const RECENT_SESSION_LIMIT = 6;
const PROJECT_LIMIT = 2;
type UsagePeriod = "week" | "month" | "half-year";

function statusTone(status: string): Tone {
  const s = status.toLowerCase();
  if (["complete", "completed", "succeeded", "success", "ready"].includes(s))
    return "ok";
  if (["running", "in_progress", "active", "pending", "queued", "connecting"].includes(s)) return "warn";
  if (["failed", "error", "cancelled"].includes(s)) return "critical";
  return "neutral";
}

function statusLabel(status: string): string {
  const normalized = status.toLowerCase();
  if (["complete", "completed", "succeeded", "success", "ready"].includes(normalized)) {
    return "Completed";
  }
  if (["running", "in_progress", "active"].includes(normalized)) return "Running";
  if (["pending", "queued", "connecting"].includes(normalized)) return "Queued";
  if (["failed", "error"].includes(normalized)) return "Failed";
  if (normalized === "cancelled") return "Cancelled";
  return status || "Unknown";
}

/* An empty project needs one line, not a card: eight of them saying "No
 * sessions yet" in identical boxes is what pushed the projects that actually
 * have work in them below the fold. */
function EmptyProjectRow({ project }: { project: ProjectSummary }) {
  const [deleting, setDeleting] = useState(false);

  return (
    <li className="group/row flex items-center gap-3 rounded-base px-3 py-2 transition-colors duration-150 ease-out-quart hover:bg-surface">
      <h3 className="min-w-0 flex-1 text-sm font-medium"><Marquee>{project.name}</Marquee></h3>
      <span className="shrink-0 text-xs text-status-neutral">empty</span>
      <Link
        to={newProjectSessionPath(project.project_id)}
        aria-label={`New session in ${project.name}`}
        className={buttonClass({ size: "sm", variant: "ghost" })}
      >
        New session
      </Link>
      <Button
        variant="danger"
        size="sm"
        onClick={() => setDeleting(true)}
        aria-label={`Delete project ${project.name}`}
        className="sm:opacity-0 sm:focus-visible:opacity-100 sm:group-hover/row:opacity-100"
      >
        Delete
      </Button>
      {deleting && (
        <DeleteProjectDialog project={project} onClose={() => setDeleting(false)} />
      )}
    </li>
  );
}

/* Recent work already answers "what changed last". Projects are therefore a
 * compact destination list instead of a second set of session cards. */
function ProjectRow({ project }: { project: ProjectSummary }) {
  const [deleting, setDeleting] = useState(false);

  return (
    <li className="group/project flex min-w-0 items-center gap-2 rounded-base px-2 py-1.5 hover:bg-surface">
      <span aria-hidden className="text-status-neutral">▱</span>
      <Marquee className="min-w-0 flex-1 text-sm font-medium">{project.name}</Marquee>
      <span className="tabular shrink-0 text-xs text-status-neutral">
        {project.session_count} session{project.session_count === 1 ? "" : "s"}
      </span>
      <Link
        to={newProjectSessionPath(project.project_id)}
        aria-label={`New session in ${project.name}`}
        className={buttonClass({ size: "sm", variant: "ghost" })}
      >
        New session
      </Link>
      <Button
        variant="danger"
        size="sm"
        onClick={() => setDeleting(true)}
        aria-label={`Delete project ${project.name}`}
        className="sm:opacity-0 sm:focus-visible:opacity-100 sm:group-hover/project:opacity-100"
      >
        Delete
      </Button>
      {deleting && (
        <DeleteProjectDialog
          project={project}
          onClose={() => setDeleting(false)}
        />
      )}
    </li>
  );
}

/* Compact workspace figures for quick scanning. Detailed metric methodology
 * belongs in Usage rather than competing with the Home launch surface. */
function formatBytes(bytes: number): string {
  if (bytes <= 0) return "0 B";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
}

function DashboardMetric({
  label,
  compactLabel,
  value,
}: {
  label: string;
  compactLabel?: string;
  value: string | number;
}) {
  return (
    <div className="flex min-w-0 flex-col rounded-md bg-surface px-1.5 py-1.5 xl:px-2 xl:py-1">
      <Marquee
        title={compactLabel ?? label}
        className="text-xs leading-tight text-status-neutral"
      >
        {label}
      </Marquee>
      <span className="tabular whitespace-nowrap text-lg leading-tight font-semibold">
        {value}
      </span>
    </div>
  );
}

function UsageDashboard({
  usage,
  activityDays,
  windowDays,
  period,
  onPeriodChange,
}: {
  usage: WorkspaceUsageView;
  activityDays: NonNullable<WorkspaceUsageView["daily"]>;
  windowDays: number;
  period: UsagePeriod;
  onPeriodChange: (period: UsagePeriod) => void;
}) {
  const days = usage.daily ?? [];
  const activeDays = days.filter((day) => day.sessions > 0).length;

  return (
    <Card
      tone="quiet"
      className="flex h-full min-w-0 flex-col gap-3 p-4 sm:p-5 xl:p-4 2xl:p-5"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h1 className="text-xl font-semibold" id="workspace-overview">
          Overview
        </h1>
        <UsagePeriodToggle period={period} onChange={onPeriodChange} />
      </div>

      <div
        data-dashboard-metrics
        className="grid grid-cols-2 gap-x-3 gap-y-2 md:grid-cols-4"
      >
        <DashboardMetric
          label="Active days"
          compactLabel="Days"
          value={activeDays}
        />
        <DashboardMetric
          label="Sessions"
          compactLabel="Sessions"
          value={usage.session_count}
        />
        <DashboardMetric
          label="Rows profiled"
          compactLabel="Rows"
          value={formatCompact(usage.profiled_rows ?? 0)}
        />
        <DashboardMetric
          label="Data stored"
          compactLabel="Data"
          value={formatBytes(usage.data_bytes ?? 0)}
        />
        <DashboardMetric
          label="Total tokens"
          compactLabel="Tokens"
          value={formatCompact(usage.total_tokens)}
        />
        <DashboardMetric
          label="Est. cost"
          compactLabel="Cost"
          value={`$${usage.est_cost_usd.toFixed(4)}`}
        />
        <DashboardMetric
          label="LLM calls"
          compactLabel="Calls"
          value={formatCompact(usage.llm_calls)}
        />
        <DashboardMetric
          label="Artifacts"
          compactLabel="Artifacts"
          value={formatCompact(usage.artifact_count)}
        />
      </div>

      <section className="flex min-w-0 flex-col gap-2 border-t border-border pt-3">
        <h2 className="text-sm font-semibold">
          Activity · {windowDays} days
        </h2>
        <ActivityGrid days={activityDays} />
      </section>

      {(usage.truncated_sessions ?? 0) > 0 && (
        <p className="text-xs text-status-warn">
          These figures are partial: {usage.truncated_sessions} session
          {usage.truncated_sessions === 1 ? "" : "s"} beyond the per-project
          scan limit are not counted.
        </p>
      )}
    </Card>
  );
}

/* Flat and cross-project, which is the one view the rail cannot give: the rail
 * groups by project and answers "go to a session I know about", this answers
 * "what was I last doing". */
function formatRecentDate(session: UsageRecentSession): string {
  const timestamp = session.updated_at ?? session.created_at;
  if (!timestamp) return "";
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}

function RecentSessions({ sessions }: { sessions: UsageRecentSession[] }) {
  if (sessions.length === 0) return null;
  return (
    <section className="flex min-w-0 flex-col gap-2">
      <SectionHeader
        title="Recent work"
        description="Continue from the sessions you touched most recently."
      />
      <ul className="flex flex-col">
        {sessions.slice(0, RECENT_SESSION_LIMIT).map((session) => (
          <li key={`${session.project_id}-${session.session_id}`}>
            <Link
              to={sessionBasePath(session.project_id, session.session_id)}
              className="flex min-w-0 items-center gap-2.5 rounded-base px-3 py-2 transition-colors duration-150 ease-out-quart hover:bg-surface"
            >
              <Dot
                tone={statusTone(session.status)}
                motion={
                  statusTone(session.status) === "warn" ? "working" : undefined
                }
              />
              <Marquee className="min-w-0 flex-1 text-sm">
                {session.title ?? session.session_id}
              </Marquee>
              <Marquee
                className="max-w-[52%] shrink-0 text-xs text-status-neutral"
                title={`${statusLabel(session.status)} · ${projectLabel(session.project_id)}`}
              >
                {statusLabel(session.status)} · {projectLabel(session.project_id)}
                {formatRecentDate(session) ? ` · ${formatRecentDate(session)}` : ""}
              </Marquee>
            </Link>
          </li>
        ))}
      </ul>
    </section>
  );
}

function UsagePeriodToggle({
  period,
  onChange,
}: {
  period: UsagePeriod;
  onChange: (period: UsagePeriod) => void;
}) {
  return (
    <div
      role="group"
      aria-label="Workspace activity period"
      className="inline-flex rounded-lg bg-surface p-0.5"
    >
      {(["half-year", "month", "week"] as const).map((value) => (
        <button
          key={value}
          type="button"
          aria-pressed={period === value}
          onClick={() => onChange(value)}
          className={`rounded-md px-2 py-1 text-[11px] font-medium transition-colors ${
            period === value
              ? "bg-bg text-text shadow-sm"
              : "text-status-neutral hover:text-text"
          }`}
        >
          {value === "half-year" ? "180d" : value === "month" ? "30d" : "7d"}
        </button>
      ))}
    </div>
  );
}

/* A compact final action keeps Home scan-friendly and sends every launch
 * through the one authoritative composer on New session. */
function QuickStart() {
  return (
    <Card tone="quiet" className="flex flex-wrap items-center gap-x-6 gap-y-3 px-4 py-3.5 shadow-overlay sm:px-5">
      <div className="min-w-0 flex-1">
        <h2 className="text-base font-semibold">Start an analysis</h2>
        <p className="text-sm text-status-neutral">
          Choose a project, select existing data or upload CSV files, then review the run before it starts.
        </p>
      </div>
      <Link
        to={newSessionPath()}
        className={buttonClass({ variant: "primary" })}
      >
        New session
      </Link>
    </Card>
  );
}

export function Component() {
  const [searchParams, setSearchParams] = useSearchParams();
  const requestedRange = searchParams.get("range");
  const period: UsagePeriod =
    requestedRange === "week"
      ? "week"
      : requestedRange === "month"
        ? "month"
        : "half-year";
  const usageWindowDays =
    period === "week"
      ? WEEK_USAGE_WINDOW_DAYS
      : period === "month"
        ? MONTH_USAGE_WINDOW_DAYS
        : DEFAULT_USAGE_WINDOW_DAYS;
  const projects = useProjects();
  const usage = useWorkspaceUsage(usageWindowDays);
  const historyUsage = useWorkspaceUsage(DEFAULT_USAGE_WINDOW_DAYS);
  const empty = projects.data?.length === 0;
  const active = (projects.data ?? []).filter((p) => p.session_count > 0);
  const idle = (projects.data ?? []).filter((p) => p.session_count === 0);
  const visibleActive = active.slice(0, PROJECT_LIMIT);
  const visibleIdle = idle.slice(0, Math.max(0, PROJECT_LIMIT - visibleActive.length));
  const hiddenProjectCount = active.length + idle.length - visibleActive.length - visibleIdle.length;

  const changePeriod = (nextPeriod: UsagePeriod) => {
    const next = new URLSearchParams(searchParams);
    if (nextPeriod === "half-year") next.delete("range");
    else next.set("range", nextPeriod);
    setSearchParams(next, { replace: true });
  };

  return (
    <div className="mx-auto flex min-h-full w-[95%] max-w-data flex-col gap-4 py-4 sm:py-5 lg:py-6">
      <div
        data-home-layout="workspace"
        className="grid min-w-0 gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] lg:items-stretch"
      >
        <div
          data-home-column="overview"
          className="flex min-w-0 flex-col"
        >
          {(usage.isPending || historyUsage.isPending) && (
            <LoadingSkeleton lines={6} label="Loading workspace overview" />
          )}
          {(usage.isError || historyUsage.isError) && (
            <ErrorState
              error={usage.error ?? historyUsage.error}
              onRetry={() => {
                void usage.refetch();
                void historyUsage.refetch();
              }}
            />
          )}
          {usage.data && historyUsage.data && (
            <UsageDashboard
              usage={usage.data}
              activityDays={usage.data.daily ?? []}
              windowDays={usageWindowDays}
              period={period}
              onPeriodChange={changePeriod}
            />
          )}
        </div>

        <aside
          aria-label="Recent projects and sessions"
          data-home-column="recent"
          className="min-w-0"
        >
          <Card tone="quiet" className="flex h-full min-w-0 flex-col divide-y divide-hairline p-0">
            {/* Both halves are `flex-1` inside a fixed-height card, so a recent
              * list longer than its share used to paint over the Projects
              * heading below it (measured 318px of content in a 260px box).
              * Scrolling keeps each half inside its own bounds. */}
            <section className="min-h-0 flex-1 overflow-auto p-3.5 sm:p-4">
              {historyUsage.isPending && (
                <LoadingSkeleton lines={3} label="Loading recent work" />
              )}
              {historyUsage.data && (
                <RecentSessions sessions={historyUsage.data.recent ?? []} />
              )}
              {historyUsage.data && (historyUsage.data.recent ?? []).length === 0 && (
                <div className="flex flex-col gap-1">
                  <h2 className="text-base font-semibold">Recent work</h2>
                  <p className="text-sm text-status-neutral">
                    Completed and in-progress sessions will appear here.
                  </p>
                </div>
              )}
            </section>

            <section className="min-h-0 flex-1 overflow-auto p-3.5 sm:p-4">
              {projects.isPending && (
                <LoadingSkeleton lines={3} label="Loading projects" />
              )}
              {projects.isError && (
                <ErrorState error={projects.error} onRetry={() => projects.refetch()} />
              )}
              {projects.data && (empty ? (
                <div className="flex flex-col gap-1">
                  <h2 className="text-base font-semibold">Projects</h2>
                  <p className="text-sm text-status-neutral">
                    No projects yet. Create one while starting a new session.
                  </p>
                </div>
              ) : (
                <div className="flex flex-col gap-2">
                  <SectionHeader title="Projects" level={2} />
                  <ul className="flex flex-col">
                    {visibleActive.map((project) => (
                      <ProjectRow key={project.project_id} project={project} />
                    ))}
                    {visibleIdle.map((project) => (
                      <EmptyProjectRow key={project.project_id} project={project} />
                    ))}
                  </ul>
                  {hiddenProjectCount > 0 && (
                    <p className="text-xs text-status-neutral">
                      + {hiddenProjectCount} more project{hiddenProjectCount === 1 ? "" : "s"}
                    </p>
                  )}
                </div>
              ))}
            </section>
          </Card>
        </aside>
      </div>

      <div data-home-launch className="sticky bottom-3 z-10 mt-auto pt-2">
        <QuickStart />
      </div>
    </div>
  );
}
