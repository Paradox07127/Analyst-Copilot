import {
  Suspense,
  lazy,
  useContext,
  useMemo,
  type ReactNode,
} from "react";
import {
  Navigate,
  Outlet,
  Route,
  Routes,
  UNSAFE_NavigationContext,
  UNSAFE_RouteContext,
  createPath,
  type Navigator,
  type To,
} from "react-router";
import {
  Panel,
  PanelGroup,
  PanelResizeHandle,
} from "react-resizable-panels";
import { SessionNav } from "./SessionNav";
import {
  EMPTY_WORKSPACE_PATH,
  workspacePathContext,
  type WorkspaceSide,
} from "../workspace-split";

const DataMapPage = lazy(() =>
  import("../../features/datasets/DataMapPage").then((module) => ({
    default: module.Component,
  })),
);
const TablePreviewPage = lazy(() =>
  import("../../features/datasets/TablePreviewPage").then((module) => ({
    default: module.Component,
  })),
);
const QualityPage = lazy(() =>
  import("../../features/insights/QualityPage").then((module) => ({
    default: module.Component,
  })),
);
const ProfilesPage = lazy(() =>
  import("../../features/insights/ProfilesPage").then((module) => ({
    default: module.Component,
  })),
);
const RelationshipsPage = lazy(() =>
  import("../../features/relationships/RelationshipsPage").then((module) => ({
    default: module.Component,
  })),
);
const QuestionsPage = lazy(() =>
  import("../../features/questions/QuestionsPage").then((module) => ({
    default: module.Component,
  })),
);
const FindingsPage = lazy(() =>
  import("../../features/findings/FindingsPage").then((module) => ({
    default: module.Component,
  })),
);
const KnowledgePage = lazy(() =>
  import("../../features/semantic/KnowledgePage").then((module) => ({
    default: module.Component,
  })),
);
const CleaningPage = lazy(() =>
  import("../../features/cleaning/CleaningPage").then((module) => ({
    default: module.Component,
  })),
);
const DeepAnalysisPage = lazy(() =>
  import("../../features/analysis/DeepAnalysisPage").then((module) => ({
    default: module.Component,
  })),
);
const TracePage = lazy(() =>
  import("../../features/trace/TracePage").then((module) => ({
    default: module.Component,
  })),
);
const ReportPage = lazy(() =>
  import("../../features/reports/ReportPage").then((module) => ({
    default: module.Component,
  })),
);
const ArtifactsPage = lazy(() =>
  import("../../features/artifacts/ArtifactsPage").then((module) => ({
    default: module.Component,
  })),
);
const SkillsPage = lazy(() =>
  import("../../features/skills/SkillsPage").then((module) => ({
    default: module.Component,
  })),
);
const ChatPage = lazy(() =>
  import("../../features/chat/ChatPage").then((module) => ({
    default: module.Component,
  })),
);
const BoardPage = lazy(() =>
  import("../../features/board/BoardPage").then((module) => ({
    default: module.Component,
  })),
);
const ComparePage = lazy(() =>
  import("../../features/compare/ComparePage").then((module) => ({
    default: module.Component,
  })),
);

function LoadingPane() {
  return (
    <div role="status" className="p-5 text-sm text-status-neutral">
      Loading pane…
    </div>
  );
}

function SessionIndex() {
  return <Navigate to="data-map" replace />;
}

function EmptyPane() {
  return (
    <div className="grid h-full min-h-80 place-items-center p-6 text-center">
      <div className="max-w-xs rounded-lg border border-dashed border-primary/45 bg-surface/70 px-6 py-8">
        <h1 className="text-base font-semibold">Empty workspace pane</h1>
        <p className="mt-1 text-sm text-status-neutral">
          Drag another session from the session rail and drop it on this side.
        </p>
      </div>
    </div>
  );
}

function PaneRouteLayout({
  projectId,
  sessionId,
}: {
  projectId: string;
  sessionId: string;
}) {
  return (
    <div className="flex h-full min-h-0 min-w-0 flex-col">
      <SessionNav projectId={projectId} sessionId={sessionId} />
      <main className="min-h-0 min-w-0 flex-1 overflow-auto">
        <Suspense fallback={<LoadingPane />}>
          <Outlet />
        </Suspense>
      </main>
    </div>
  );
}

function PaneNavigationScope({
  path,
  onNavigate,
  children,
}: {
  path: string;
  onNavigate: (to: string, replace: boolean) => void;
  children: ReactNode;
}) {
  const parent = useContext(UNSAFE_NavigationContext);
  const navigator = useMemo<Navigator>(
    () => ({
      createHref: (to: To) => parent.navigator.createHref(to),
      encodeLocation: parent.navigator.encodeLocation
        ? (to: To) => parent.navigator.encodeLocation!(to)
        : undefined,
      go: (delta: number) => parent.navigator.go(delta),
      push: (to: To) =>
        onNavigate(typeof to === "string" ? to : createPath(to), false),
      replace: (to: To) =>
        onNavigate(typeof to === "string" ? to : createPath(to), true),
    }),
    [onNavigate, parent.navigator],
  );
  const navigation = useMemo(
    () => ({ ...parent, navigator }),
    [navigator, parent],
  );
  /* Resetting RouteContext makes this a small client-side router rather than a
   * descendant of the outer /split data route. Links and useNavigate then use
   * the pane navigator and cannot accidentally replace the other pane. */
  const routeContext = useMemo(
    () => ({ outlet: null, matches: [], isDataRoute: false }),
    [],
  );

  return (
    <UNSAFE_RouteContext.Provider value={routeContext}>
      <UNSAFE_NavigationContext.Provider value={navigation}>
        <Routes location={path}>{children}</Routes>
      </UNSAFE_NavigationContext.Provider>
    </UNSAFE_RouteContext.Provider>
  );
}

function PaneRoutes({
  path,
  onNavigate,
}: {
  path: string;
  onNavigate: (to: string, replace: boolean) => void;
}) {
  const context = workspacePathContext(path);
  return (
    <PaneNavigationScope path={path} onNavigate={onNavigate}>
      <Route
        element={
          <PaneRouteLayout
            projectId={context?.projectId ?? ""}
            sessionId={context?.sessionId ?? ""}
          />
        }
      >
        <Route path={EMPTY_WORKSPACE_PATH.slice(1)} element={<EmptyPane />} />
        <Route path="projects/:projectId/compare" element={<ComparePage />} />
        <Route path="projects/:projectId/sessions/:sessionId">
          <Route index element={<SessionIndex />} />
          <Route path="data-map" element={<DataMapPage />} />
          <Route path="table/:datasetId" element={<TablePreviewPage />} />
          <Route path="quality" element={<QualityPage />} />
          <Route path="profiles" element={<ProfilesPage />} />
          <Route path="relationships" element={<RelationshipsPage />} />
          <Route path="questions" element={<QuestionsPage />} />
          <Route path="findings" element={<FindingsPage />} />
          <Route path="semantic" element={<KnowledgePage />} />
          <Route path="cleaning" element={<CleaningPage />} />
          <Route path="deep-analysis" element={<DeepAnalysisPage />} />
          <Route path="trace" element={<TracePage />} />
          <Route path="report" element={<ReportPage />} />
          <Route path="artifacts" element={<ArtifactsPage />} />
          <Route path="skills" element={<SkillsPage />} />
          <Route path="chat" element={<ChatPage />} />
          <Route path="board" element={<BoardPage />} />
        </Route>
      </Route>
    </PaneNavigationScope>
  );
}

function PaneHeader({
  side,
  path,
  active,
  onClose,
}: {
  side: WorkspaceSide;
  path: string;
  active: boolean;
  onClose: () => void;
}) {
  const context = workspacePathContext(path);
  return (
    <header
      className={`flex h-9 shrink-0 items-center gap-2 border-b px-3 text-xs ${
        active
          ? "border-b-primary bg-primary/6 text-text"
          : "border-border bg-surface text-status-neutral"
      }`}
    >
      <span
        aria-hidden
        className={`size-1.5 rounded-full ${active ? "bg-primary" : "bg-status-neutral/45"}`}
      />
      <span className="font-semibold uppercase tracking-wide">{side}</span>
      <span className="min-w-0 truncate font-mono">
        {context?.sessionId || "Compare"}
      </span>
      <button
        type="button"
        aria-label={`Close ${side} pane`}
        onClick={onClose}
        className="ml-auto inline-flex size-6 items-center justify-center rounded-sm text-sm hover:bg-primary/10 hover:text-primary"
      >
        ×
      </button>
    </header>
  );
}

function WorkspacePane({
  side,
  path,
  active,
  onFocus,
  onClose,
  onNavigate,
}: {
  side: WorkspaceSide;
  path: string;
  active: boolean;
  onFocus: () => void;
  onClose: () => void;
  onNavigate: (to: string, replace: boolean) => void;
}) {
  return (
    <section
      aria-label={`${side === "left" ? "Left" : "Right"} workspace pane`}
      onPointerDown={onFocus}
      onFocusCapture={onFocus}
      className={`flex h-full min-h-0 min-w-0 flex-col bg-bg ${
        active ? "ring-1 ring-inset ring-primary/35" : ""
      }`}
    >
      <PaneHeader side={side} path={path} active={active} onClose={onClose} />
      <div className="min-h-0 min-w-0 flex-1">
        <PaneRoutes path={path} onNavigate={onNavigate} />
      </div>
    </section>
  );
}

export function SplitWorkspace({
  left,
  right,
  active,
  compact = false,
  onFocus,
  onClose,
  onNavigate,
}: {
  left: string;
  right: string;
  active: WorkspaceSide;
  compact?: boolean;
  onFocus: (side: WorkspaceSide) => void;
  onClose: (side: WorkspaceSide) => void;
  onNavigate: (side: WorkspaceSide, to: string, replace: boolean) => void;
}) {
  const pane = (side: WorkspaceSide, path: string) => (
    <WorkspacePane
      side={side}
      path={path}
      active={active === side}
      onFocus={() => onFocus(side)}
      onClose={() => onClose(side)}
      onNavigate={(to, replace) => onNavigate(side, to, replace)}
    />
  );

  if (compact) {
    return (
      <div aria-label="Split workspace" className="flex h-full min-h-0 flex-col">
        <nav
          aria-label="Workspace panes"
          className="grid shrink-0 grid-cols-2 border-b border-border bg-surface p-1"
        >
          {(["left", "right"] as const).map((side) => (
            <button
              key={side}
              type="button"
              aria-pressed={active === side}
              onClick={() => onFocus(side)}
              className={`rounded-sm px-2 py-1 text-xs font-medium capitalize ${
                active === side
                  ? "bg-primary text-bg"
                  : "text-status-neutral hover:bg-primary/10 hover:text-text"
              }`}
            >
              {side} pane
            </button>
          ))}
        </nav>
        <div className="min-h-0 flex-1">
          {active === "left" ? pane("left", left) : pane("right", right)}
        </div>
      </div>
    );
  }

  return (
    <div aria-label="Split workspace" className="h-full min-h-0 min-w-0">
      <PanelGroup
        direction="horizontal"
        autoSaveId="eda.layout.workspace-split"
        className="min-h-0 min-w-0"
      >
        <Panel id="workspace-left" defaultSize={50} minSize={25} order={1}>
          {pane("left", left)}
        </Panel>
        <PanelResizeHandle
          aria-label="Resize workspace panes"
          className="w-1 bg-border transition-colors hover:bg-primary data-[resize-handle-active]:bg-primary"
        />
        <Panel id="workspace-right" defaultSize={50} minSize={25} order={2}>
          {pane("right", right)}
        </Panel>
      </PanelGroup>
    </div>
  );
}
