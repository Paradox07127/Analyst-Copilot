import { useState, type FormEvent } from "react";
import { Link, useNavigate, useSearchParams } from "react-router";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  api,
  ApiError,
  type ProjectSummary,
  type UsageRecentSession,
  type WorkspaceUsageView,
} from "../../api/client";
import {
  queryKeys,
  useProjects,
  useSessions,
  useWorkspaceUsage,
} from "../../api/hooks";
import {
  EmptyState,
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
import { NewSessionPanel } from "../launchpad/LaunchpadPage";
import { projectLabel } from "../../app/unfiled";
import { newProjectSessionPath, newSessionPath, sessionBasePath } from "../../app/paths";
import { DeleteProjectDialog } from "./DeleteProjectDialog";

const MAX_PROJECT_ID_LENGTH = 64;
const MAX_PROJECT_NAME_LENGTH = 200;
const DEFAULT_USAGE_WINDOW_DAYS = 180;
const MONTH_USAGE_WINDOW_DAYS = 30;
const WEEK_USAGE_WINDOW_DAYS = 7;
type UsagePeriod = "week" | "month" | "half-year";

/* Mirrors the server's project_id rules (run_service._validated_project_id):
 * single path segment, starts alphanumeric, spaces allowed. */
function deriveProjectId(name: string): string {
  return name
    .replace(/[^A-Za-z0-9 _.-]/g, "")
    .replace(/\s+/g, " ")
    .replace(/^[^A-Za-z0-9]+/, "")
    .trim()
    .slice(0, MAX_PROJECT_ID_LENGTH);
}

function NewProjectForm({ onCancel }: { onCancel: () => void }) {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [name, setName] = useState("");
  const [editedId, setEditedId] = useState<string | null>(null);
  const projectId = (editedId ?? deriveProjectId(name)).trim();

  const create = useMutation({
    mutationFn: (idempotencyKey: string) =>
      api.createProject(
        { project_id: projectId, name: name.trim() || projectId },
        idempotencyKey,
      ),
    onSuccess: (project: ProjectSummary) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.projects });
      navigate(newProjectSessionPath(project.project_id));
    },
  });

  const onSubmit = (event: FormEvent) => {
    event.preventDefault();
    /* Minted per submit, not per call: a retry of this same submit must reuse
     * the key so the server replays instead of creating twice. */
    if (projectId) create.mutate(crypto.randomUUID());
  };

  const error = create.error instanceof ApiError ? create.error : null;
  const conflict = error?.code === "project_conflict";

  return (
    <form
      onSubmit={onSubmit}
      className="flex max-w-content flex-col gap-4 rounded-xl border border-border bg-bg p-5"
    >
      <SectionHeader
        title="New project"
        description="A project is a folder of sessions that share uploads and settings. Creating one opens its Launchpad, where you add data."
      />

      {/* Name leads; the id is derived and only edited when the derived one is
       * wrong, so it sits at caption weight below rather than as an equal. */}
      <div className="flex flex-col gap-1">
        <label className="text-sm font-medium" htmlFor="new-project-name">
          Name
        </label>
        <input
          id="new-project-name"
          autoFocus
          value={name}
          onChange={(event) => setName(event.target.value)}
          maxLength={MAX_PROJECT_NAME_LENGTH}
          placeholder="Brazilian E-Commerce"
          className="rounded-base border border-border bg-bg px-2.5 py-1.5 text-sm"
        />
      </div>

      <div className="flex flex-col gap-1">
        <label className="text-xs text-status-neutral" htmlFor="new-project-id">
          Project id
        </label>
        <input
          id="new-project-id"
          value={projectId}
          onChange={(event) => setEditedId(event.target.value)}
          maxLength={MAX_PROJECT_ID_LENGTH}
          aria-describedby="new-project-id-hint"
          className="rounded-base border border-border bg-bg px-2.5 py-1.5 font-mono text-sm"
        />
        <p id="new-project-id-hint" className="text-xs text-status-neutral">
          The workspace folder name, derived from the name above. Letters,
          digits, spaces, “_”, “.” and “-”; must start with a letter or digit.
        </p>
      </div>

      <div className="flex items-center gap-2">
        <Button
          type="submit"
          variant="primary"
          disabled={!projectId || create.isPending}
        >
          {create.isPending ? "Creating…" : "Create project"}
        </Button>
        <Button onClick={onCancel}>Cancel</Button>
      </div>

      {error && (
        <div
          role="alert"
          className={`flex flex-col gap-1 rounded-base border p-3 text-sm ${
            conflict ? "border-status-warn/50" : "border-status-critical/40"
          }`}
        >
          <p
            className={`font-medium ${
              conflict ? "text-status-warn" : "text-status-critical"
            }`}
          >
            {conflict
              ? "That id clashes with an existing project."
              : "Could not create the project."}
          </p>
          <p className="text-status-neutral">{error.message}</p>
        </div>
      )}
      {create.isError && !error && (
        <div
          role="alert"
          className="rounded-base border border-status-critical/40 p-3 text-sm text-status-critical"
        >
          {create.error instanceof Error
            ? create.error.message
            : "Could not create the project."}
        </div>
      )}
    </form>
  );
}

function statusTone(status: string): Tone {
  const s = status.toLowerCase();
  if (["complete", "completed", "succeeded", "success", "ready"].includes(s))
    return "ok";
  if (["running", "in_progress", "active", "pending"].includes(s)) return "warn";
  if (["failed", "error", "cancelled"].includes(s)) return "critical";
  return "neutral";
}

function LatestSessionLink({ project }: { project: ProjectSummary }) {
  const sessions = useSessions(project.project_id);
  const latest = sessions.data?.pages[0]?.items[0];

  if (sessions.isPending) {
    return <span className="text-xs text-status-neutral">Loading sessions…</span>;
  }
  if (!latest) {
    return (
      <span className="text-xs text-status-neutral">
        No sessions yet — start the first one below.
      </span>
    );
  }
  const when = latest.updated_at ?? latest.created_at;
  const whenLabel = when
    ? new Date(when).toLocaleDateString(undefined, {
        month: "short",
        day: "numeric",
      })
    : null;

  /* The card answered "which project" but not "is anything happening in it",
   * which is the question you open this page with. */
  return (
    <div className="flex min-w-0 flex-col gap-1">
      <Link
        to={sessionBasePath(latest.project_id, latest.session_id)}
        className="min-w-0 text-sm text-link hover:underline"
      >
        <Marquee>Latest: {latest.title ?? latest.session_id}</Marquee>
      </Link>
      <span className="flex items-center gap-1.5 text-xs text-status-neutral">
        <Dot
          tone={statusTone(latest.status)}
          motion={statusTone(latest.status) === "warn" ? "working" : undefined}
        />
        {[latest.status, whenLabel].filter(Boolean).join(" · ")}
      </span>
    </div>
  );
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

/* Project management lives here, not only behind the rail's hover-revealed "…".
 * A destructive action that is only reachable by hovering the right row of a
 * sidebar is not reachable. */
function ProjectCard({ project }: { project: ProjectSummary }) {
  const [deleting, setDeleting] = useState(false);

  return (
    <article className="group/card flex flex-col gap-3 rounded-xl border border-border bg-bg p-4 transition-colors duration-150 ease-out-quart hover:border-primary/35">
      <div className="flex items-baseline justify-between gap-2">
        <h2 className="min-w-0 text-base font-semibold"><Marquee>{project.name}</Marquee></h2>
        <span className="tabular shrink-0 text-xs text-status-neutral">
          {project.session_count} session{project.session_count === 1 ? "" : "s"}
        </span>
      </div>
      <LatestSessionLink project={project} />
      <div className="mt-auto flex items-center gap-2 border-t border-border pt-3">
        <Link
          to={newProjectSessionPath(project.project_id)}
          aria-label={`New session in ${project.name}`}
          className={buttonClass({ size: "sm" })}
        >
          New session
        </Link>
        <Button
          variant="danger"
          size="sm"
          onClick={() => setDeleting(true)}
          aria-label={`Delete project ${project.name}`}
          className="ml-auto sm:opacity-0 sm:focus-visible:opacity-100 sm:group-hover/card:opacity-100"
        >
          Delete
        </Button>
      </div>
      {deleting && (
        <DeleteProjectDialog
          project={project}
          onClose={() => setDeleting(false)}
        />
      )}
    </article>
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
    <div className="flex min-w-0 flex-col rounded-md bg-surface px-1.5 py-1 xl:px-1 xl:py-0.5">
      <Marquee className="text-[11px] leading-tight text-status-neutral xl:hidden 2xl:block">
        {label}
      </Marquee>
      <span className="hidden truncate text-[10px] leading-tight text-status-neutral xl:block 2xl:hidden">
        {compactLabel ?? label}
      </span>
      <span className="tabular whitespace-nowrap text-base leading-tight font-semibold xl:text-sm 2xl:text-base">
        {value}
      </span>
    </div>
  );
}

function UsageDashboard({
  usage,
  activityDays,
  period,
  onPeriodChange,
}: {
  usage: WorkspaceUsageView;
  activityDays: NonNullable<WorkspaceUsageView["daily"]>;
  period: UsagePeriod;
  onPeriodChange: (period: UsagePeriod) => void;
}) {
  const days = usage.daily ?? [];
  const activeDays = days.filter((day) => day.sessions > 0).length;

  return (
    <Card
      tone="quiet"
      className="flex min-w-0 flex-col gap-2.5 p-3 sm:p-4 xl:p-3 2xl:p-4"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h1 className="text-lg font-semibold" id="workspace-overview">
          Overview
        </h1>
        <UsagePeriodToggle period={period} onChange={onPeriodChange} />
      </div>

      <div
        data-dashboard-metrics
        className="grid grid-cols-2 gap-x-1 gap-y-0.5 md:grid-cols-4"
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

      <section className="flex min-w-0 flex-col gap-1.5 border-t border-border pt-2">
        <h2 className="text-xs font-medium">
          Activity · {DEFAULT_USAGE_WINDOW_DAYS} days
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
        {sessions.map((session) => (
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
              <Marquee className="max-w-[42%] shrink-0 text-xs text-status-neutral">
                {projectLabel(session.project_id)}
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

/* The launch surface itself, not a link to it: dropping a CSV is the first thing
 * a returning user does here, and it used to cost a navigation to reach a page
 * that then asked for the same project this one already lists. */
function QuickStart({ onNewProject }: { onNewProject: () => void }) {
  return (
    <Card tone="quiet" className="flex flex-col gap-3 px-4 py-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-base font-semibold">Start an analysis</h2>
        <div className="flex shrink-0 flex-wrap items-center gap-2">
          <Link
            to={newSessionPath()}
            className={buttonClass({ variant: "ghost", size: "sm" })}
          >
            Open full page
          </Link>
          <Button size="sm" onClick={onNewProject}>
            New project
          </Button>
        </div>
      </div>
      <NewSessionPanel />
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
  const [creating, setCreating] = useState(() => searchParams.get("new") === "1");
  const active = (projects.data ?? []).filter((p) => p.session_count > 0);
  const idle = (projects.data ?? []).filter((p) => p.session_count === 0);

  const changePeriod = (nextPeriod: UsagePeriod) => {
    const next = new URLSearchParams(searchParams);
    if (nextPeriod === "half-year") next.delete("range");
    else next.set("range", nextPeriod);
    setSearchParams(next, { replace: true });
  };

  return (
    <div className="mx-auto flex w-[90%] max-w-data flex-col gap-4 py-4 sm:py-5 lg:py-6">
      <div
        data-home-layout="workspace"
        className="grid min-w-0 gap-4 xl:grid-cols-5 xl:items-start"
      >
        <div
          data-home-column="overview"
          className="flex min-w-0 flex-col gap-4 xl:col-span-2"
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
              activityDays={historyUsage.data.daily ?? []}
              period={period}
              onPeriodChange={changePeriod}
            />
          )}
        </div>

        <div
          data-home-column="actions"
          className="flex min-w-0 flex-col gap-4 xl:col-span-3"
        >
          <QuickStart onNewProject={() => setCreating(true)} />

          {creating && <NewProjectForm onCancel={() => setCreating(false)} />}
        </div>
      </div>

      <aside
        aria-label="Recent projects and sessions"
        data-home-column="recent"
        className="flex w-full min-w-0 flex-col gap-4"
      >
        <div className="min-w-0 rounded-xl border border-border bg-bg p-4">
          {historyUsage.isPending && (
            <LoadingSkeleton lines={3} label="Loading recent work" />
          )}
          {historyUsage.data && (
            <RecentSessions sessions={historyUsage.data.recent ?? []} />
          )}
          {historyUsage.data &&
            (historyUsage.data.recent ?? []).length === 0 && (
              <div className="flex flex-col gap-1">
                <h2 className="text-base font-semibold">Recent work</h2>
                <p className="text-sm text-status-neutral">
                  Completed and in-progress sessions will appear here.
                </p>
              </div>
            )}
        </div>

        <div className="min-w-0">
          {projects.isPending && (
            <LoadingSkeleton lines={4} label="Loading projects" />
          )}
          {projects.isError && (
            <ErrorState
              error={projects.error}
              onRetry={() => projects.refetch()}
            />
          )}
          {projects.data &&
            (empty ? (
              <EmptyState
                title="No projects yet"
                description="Projects keep related analyses and shared uploads together. You can create one from Start an analysis or continue using standalone analyses."
              />
            ) : (
              <section className="flex flex-col gap-3">
                <SectionHeader title="Projects" level={2} />
                {active.length > 0 && (
                  <div className="grid items-stretch gap-3 2xl:grid-cols-2">
                    {active.map((project) => (
                      <ProjectCard
                        key={project.project_id}
                        project={project}
                      />
                    ))}
                  </div>
                )}
                {idle.length > 0 && (
                  <ul className="flex flex-col">
                    {idle.map((project) => (
                      <EmptyProjectRow
                        key={project.project_id}
                        project={project}
                      />
                    ))}
                  </ul>
                )}
              </section>
            ))}
        </div>
      </aside>
    </div>
  );
}
