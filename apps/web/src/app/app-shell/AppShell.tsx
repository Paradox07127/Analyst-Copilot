import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Outlet, useLocation, useNavigate } from "react-router";
import {
  DndContext,
  DragOverlay,
  KeyboardSensor,
  MeasuringStrategy,
  MouseSensor,
  TouchSensor,
  pointerWithin,
  rectIntersection,
  useDroppable,
  useSensor,
  useSensors,
  type CollisionDetection,
  type DragEndEvent,
  type DragStartEvent,
} from "@dnd-kit/core";
import {
  Panel,
  PanelGroup,
  PanelResizeHandle,
  type ImperativePanelHandle,
} from "react-resizable-panels";
import { TopBar } from "./TopBar";
import { SessionNav } from "./SessionNav";
import { CollapsedSessionRail, SessionRail } from "./SessionRail";
import { Inspector } from "./Inspector";
import { ActivityCenter } from "./ActivityCenter";
import { JobActivityProvider } from "../job-activity";
import { getRailCollapsed, setRailCollapsed } from "./rail-prefs";
import { SettingsDialog } from "../../features/settings/SettingsDialog";
import { useDialogFocus } from "../../components/use-dialog-focus";
import { SettingsDialogProvider } from "../settings-dialog";
import {
  WorkspaceFocusProvider,
  useWorkspaceFocus,
  type WorkspacePaneContext,
} from "../workspace-focus";
import {
  readSessionDragData,
  SessionDragPreview,
  SessionDragProvider,
  useSessionDrag,
} from "../session-drag";
import {
  EMPTY_WORKSPACE_PATH,
  normalizeWorkspacePath,
  readSplitWorkspace,
  sessionWorkspacePath,
  splitWorkspacePath,
  workspacePathContext,
  type SplitWorkspaceState,
  type SessionDragPayload,
  type WorkspaceSide,
} from "../workspace-split";
import { readCompareRouteState } from "../../features/compare/compare-route-state";
import { SplitWorkspace } from "./SplitWorkspace";

function MobileSessionRailDialog({ onClose }: { onClose: () => void }) {
  const { dialogRef, onKeyDown } = useDialogFocus(onClose);

  return (
    <div
      className="animate-fade fixed inset-0 z-50 flex bg-scrim"
      onClick={onClose}
    >
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-label="Mobile sessions"
        onKeyDown={onKeyDown}
        onClick={(event) => event.stopPropagation()}
        className="animate-enter h-full w-full max-w-sm bg-rail-bg shadow-overlay"
      >
        <SessionRail />
      </div>
    </div>
  );
}

function WorkspaceFocusSync({
  split,
  pathname,
  search,
  onNavigatePane,
}: {
  split: SplitWorkspaceState | null;
  pathname: string;
  search: string;
  onNavigatePane: (side: WorkspaceSide, to: string, replace: boolean) => void;
}) {
  const workspace = useWorkspaceFocus();
  const left = useMemo<WorkspacePaneContext | null>(() => {
    if (split) {
      const context = workspacePathContext(split.left);
      if (!context) return null;
      return {
        ...context,
        onSectionChange: (section) => {
          if (!context.sessionId) return;
          onNavigatePane(
            "left",
            `/projects/${encodeURIComponent(context.projectId)}/sessions/${encodeURIComponent(context.sessionId)}/${section}`,
            false,
          );
        },
      };
    }
    const match = pathname.match(/^\/projects\/([^/]+)\/compare$/);
    if (!match) return null;
    const state = readCompareRouteState(new URLSearchParams(search));
    return {
      projectId: decodeURIComponent(match[1] ?? ""),
      sessionId: state.left,
      section: state.scope,
    };
  }, [onNavigatePane, pathname, search, split]);
  const right = useMemo<WorkspacePaneContext | null>(() => {
    if (split) {
      const context = workspacePathContext(split.right);
      if (!context) return null;
      return {
        ...context,
        onSectionChange: (section) => {
          if (!context.sessionId) return;
          onNavigatePane(
            "right",
            `/projects/${encodeURIComponent(context.projectId)}/sessions/${encodeURIComponent(context.sessionId)}/${section}`,
            false,
          );
        },
      };
    }
    const match = pathname.match(/^\/projects\/([^/]+)\/compare$/);
    if (!match) return null;
    const state = readCompareRouteState(new URLSearchParams(search));
    return {
      projectId: decodeURIComponent(match[1] ?? ""),
      sessionId: state.right,
      section: state.scope,
    };
  }, [onNavigatePane, pathname, search, split]);

  useEffect(() => {
    if (split) {
      workspace.configure("split", left, right);
      workspace.focusPane(split.active);
    } else if (left || right) {
      workspace.configure("compare", left, right);
    } else {
      workspace.reset();
    }
  }, [
    left,
    right,
    split,
    workspace.configure,
    workspace.focusPane,
    workspace.reset,
  ]);
  return null;
}

const SPLIT_DROP_PREFIX = "split-pane:";

function splitDropId(side: WorkspaceSide): string {
  return `${SPLIT_DROP_PREFIX}${side}`;
}

function sideFromDropId(value: unknown): WorkspaceSide | null {
  if (value === splitDropId("left")) return "left";
  if (value === splitDropId("right")) return "right";
  return null;
}

/* Pointer location is authoritative for mouse/touch. rectIntersection is the
 * fallback that lets dnd-kit's keyboard sensor move between the two panes. */
const splitCollisionDetection: CollisionDetection = (args) => {
  const pointerHits = pointerWithin(args);
  return pointerHits.length > 0 ? pointerHits : rectIntersection(args);
};

function SplitDropTarget({
  side,
  session,
}: {
  side: WorkspaceSide;
  session: SessionDragPayload | null;
}) {
  const { isOver, setNodeRef } = useDroppable({
    id: splitDropId(side),
    disabled: !session,
  });

  return (
    <div
      ref={setNodeRef}
      role={session ? "button" : undefined}
      tabIndex={session ? -1 : undefined}
      data-split-drop-side={side}
      data-drop-active={isOver ? "true" : "false"}
      aria-label={session ? `Drop ${session.label} in ${side} pane` : undefined}
      className={`grid place-items-center rounded-lg border-2 transition-all duration-150 ${
        isOver
          ? "border-dashed border-primary bg-primary/20 text-primary shadow-overlay backdrop-blur-[2px]"
          : "border-transparent"
      }`}
    >
      {isOver && session && (
        <span className="flex flex-col items-center gap-2 rounded-base bg-bg/90 px-4 py-3 text-center shadow-card">
          <span className="text-sm font-semibold">Drop in {side} pane</span>
          <span className="max-w-48 truncate text-xs">{session.label}</span>
        </span>
      )}
    </div>
  );
}

function SplitDropOverlay() {
  const { draggingSession } = useSessionDrag();

  return (
    <div
      aria-label={draggingSession ? "Choose split side" : undefined}
      aria-hidden={draggingSession ? undefined : true}
      className={`absolute inset-2 z-40 grid grid-cols-2 ${
        draggingSession
          ? "cursor-grabbing"
          : "invisible pointer-events-none"
      }`}
    >
      {(["left", "right"] as const).map((side) => (
        <SplitDropTarget
          key={side}
          side={side}
          session={draggingSession}
        />
      ))}
    </div>
  );
}

export function AppShell() {
  return (
    <JobActivityProvider>
      <WorkspaceFocusProvider>
        <SessionDragProvider>
          <AppShellLayout />
        </SessionDragProvider>
      </WorkspaceFocusProvider>
    </JobActivityProvider>
  );
}

function AppShellLayout() {
  const location = useLocation();
  const { pathname, search, hash } = location;
  const navigate = useNavigate();
  const workspace = useWorkspaceFocus();
  const { beginSessionDrag, draggingSession, endSessionDrag } = useSessionDrag();
  const sensors = useSensors(
    useSensor(MouseSensor, { activationConstraint: { distance: 6 } }),
    useSensor(TouchSensor, {
      activationConstraint: { delay: 180, tolerance: 8 },
    }),
    useSensor(KeyboardSensor),
  );
  const split = useMemo(
    () => readSplitWorkspace(pathname, search),
    [pathname, search],
  );

  useEffect(() => {
    if (pathname === "/split" && !split) navigate("/projects", { replace: true });
  }, [navigate, pathname, split]);
  const inspectorRef = useRef<ImperativePanelHandle>(null);
  const [inspectorCollapsed, setInspectorCollapsed] = useState(false);
  /* Read synchronously so a stored collapse is already applied on first paint. */
  const [railCollapsed, setRailCollapsedState] = useState(getRailCollapsed);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [mobileRailOpen, setMobileRailOpen] = useState(false);
  const [narrowLayout, setNarrowLayout] = useState(
    () => window.matchMedia("(max-width: 767px)").matches,
  );

  useEffect(() => {
    const query = window.matchMedia("(max-width: 767px)");
    const update = () => setNarrowLayout(query.matches);
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, []);

  useEffect(() => {
    setMobileRailOpen(false);
  }, [pathname]);

  /* The Inspector holds whatever context the route has: a session's evidence,
   * or a project's stored data. Off both it renders a heading and one "open a
   * session" sentence while holding ~22% of the window, which is what made
   * Home feel cramped. Keyed on entering/leaving that context rather than run
   * on every render, so a manual expand still sticks while you stay put. */
  const hasSession = split
    ? Boolean(
        workspacePathContext(split.left)?.sessionId ||
          workspacePathContext(split.right)?.sessionId,
      )
    : /\/sessions\/[^/]+/.test(pathname) || /\/projects\/[^/]+/.test(pathname);
  const splitMode = Boolean(split);
  useEffect(() => {
    /* The panel API asserts ("Panel size not found") until the group has been
     * measured, which on first mount happens after this effect — and an
     * uncaught assert there takes the whole shell down to the error boundary.
     * Retry once on the next frame, then give up: the wrong panel width is a
     * far smaller problem than a blank app. */
    let frame = 0;
    const apply = (retry: boolean) => {
      const panel = inspectorRef.current;
      if (!panel) return;
      try {
        /* A split already divides the editor width in half. Start it with the
         * Inspector tucked away so data tables and reports remain usable; the
         * top-bar control can still reopen it, and this effect does not rerun
         * merely because one pane changes section. */
        if (splitMode) panel.collapse();
        else if (hasSession) panel.expand();
        else panel.collapse();
      } catch {
        if (retry) frame = requestAnimationFrame(() => apply(false));
      }
    };
    apply(true);
    return () => cancelAnimationFrame(frame);
  }, [hasSession, splitMode]);

  const toggleInspector = () => {
    const panel = inspectorRef.current;
    if (!panel) return;
    if (panel.isCollapsed()) panel.expand();
    else panel.collapse();
  };

  const toggleRail = () => {
    const next = !railCollapsed;
    setRailCollapsedState(next);
    setRailCollapsed(next);
  };

  const openSettings = () => setSettingsOpen(true);

  const navigatePane = useCallback(
    (side: WorkspaceSide, to: string, replace: boolean) => {
      if (!split) return;
      const normalized = normalizeWorkspacePath(to);
      if (!normalized) {
        navigate(to, { replace });
        return;
      }
      navigate(
        splitWorkspacePath(
          side === "left" ? normalized : split.left,
          side === "right" ? normalized : split.right,
          side,
        ),
        { replace },
      );
    },
    [navigate, split],
  );

  const focusPane = useCallback(
    (side: WorkspaceSide) => {
      workspace.focusPane(side);
      if (!split || split.active === side) return;
      navigate(splitWorkspacePath(split.left, split.right, side), {
        replace: true,
      });
    },
    [navigate, split, workspace],
  );

  const closePane = useCallback(
    (side: WorkspaceSide) => {
      if (!split) return;
      const remaining = side === "left" ? split.right : split.left;
      navigate(remaining === EMPTY_WORKSPACE_PATH ? "/projects" : remaining);
    },
    [navigate, split],
  );

  const dropSession = useCallback(
    (side: WorkspaceSide, session: SessionDragPayload) => {
      const dropped = sessionWorkspacePath(session);
      if (split) {
        navigate(
          splitWorkspacePath(
            side === "left" ? dropped : split.left,
            side === "right" ? dropped : split.right,
            side,
          ),
        );
      } else {
        const current = normalizeWorkspacePath(`${pathname}${search}${hash}`);
        if (current) {
          navigate(
            splitWorkspacePath(
              side === "left" ? dropped : current,
              side === "right" ? dropped : current,
              side,
            ),
          );
        } else {
          navigate(
            splitWorkspacePath(
              side === "left" ? dropped : EMPTY_WORKSPACE_PATH,
              side === "right" ? dropped : EMPTY_WORKSPACE_PATH,
              side,
            ),
          );
        }
      }
    },
    [hash, navigate, pathname, search, split],
  );

  const onSessionDragStart = useCallback(
    (event: DragStartEvent) => {
      const session = readSessionDragData(event.active.data.current);
      if (session) beginSessionDrag(session);
    },
    [beginSessionDrag],
  );

  const onSessionDragEnd = useCallback(
    (event: DragEndEvent) => {
      const session = readSessionDragData(event.active.data.current);
      const side = sideFromDropId(event.over?.id);
      endSessionDrag();
      if (session && side) dropSession(side, session);
    },
    [dropSession, endSessionDrag],
  );

  const currentSessionContext = useMemo(
    () =>
      pathname.includes("/sessions/")
        ? workspacePathContext(`${pathname}${search}${hash}`)
        : null,
    [hash, pathname, search],
  );

  const workspaceView = split ? (
    <SplitWorkspace
      left={split.left}
      right={split.right}
      active={split.active}
      compact={narrowLayout}
      onFocus={focusPane}
      onClose={closePane}
      onNavigate={navigatePane}
    />
  ) : (
    <div className="flex h-full min-h-0 min-w-0 flex-col">
      <SessionNav
        projectId={currentSessionContext?.projectId}
        sessionId={currentSessionContext?.sessionId}
      />
      <main className="min-h-0 min-w-0 flex-1 overflow-auto">
        <Outlet />
      </main>
    </div>
  );

  return (
    <DndContext
      sensors={sensors}
      collisionDetection={splitCollisionDetection}
      measuring={{ droppable: { strategy: MeasuringStrategy.Always } }}
      onDragStart={onSessionDragStart}
      onDragCancel={endSessionDrag}
      onDragEnd={onSessionDragEnd}
    >
      <SettingsDialogProvider value={openSettings}>
      <WorkspaceFocusSync
        split={split}
        pathname={pathname}
        search={search}
        onNavigatePane={navigatePane}
      />
      <div className="flex h-[100dvh] max-h-[100dvh] flex-col overflow-hidden bg-bg text-text">
        <TopBar
          inspectorCollapsed={inspectorCollapsed}
          onToggleInspector={toggleInspector}
          onOpenSessions={() => setMobileRailOpen(true)}
          onToggleSessions={toggleRail}
          sessionsCollapsed={railCollapsed}
          onOpenSettings={openSettings}
          showInspector={hasSession}
        />
        <div className="flex min-h-0 min-w-0 flex-1">
          {narrowLayout ? (
            <div className="relative flex min-h-0 min-w-0 flex-1 flex-col">
              {workspaceView}
              <SplitDropOverlay />
            </div>
          ) : (
            <>
              {railCollapsed && <CollapsedSessionRail />}
              <PanelGroup
                direction="horizontal"
                autoSaveId="eda.layout.workbench"
                className="min-h-0 min-w-0 flex-1"
              >
                {!railCollapsed && (
                  <>
                    {/* 18/12 put the rail at 142px on an 800px window, which
                      * was narrower than the project names it lists. The three
                      * defaults must still total 100: react-resizable-panels
                      * warns and renormalizes otherwise, so the numbers here
                      * would not be the ones rendered. */}
                    <Panel
                      id="rail"
                      defaultSize={22}
                      minSize={16}
                      maxSize={32}
                      order={1}
                    >
                      <SessionRail />
                    </Panel>
                    <PanelResizeHandle className="group/resize relative w-2 cursor-col-resize bg-transparent after:absolute after:inset-y-0 after:left-1/2 after:w-px after:-translate-x-1/2 after:bg-border after:transition-colors after:content-[''] hover:after:bg-primary active:after:bg-primary" />
                  </>
                )}
                <Panel id="main" defaultSize={56} minSize={30} order={2}>
                  <div className="relative h-full min-h-0 min-w-0">
                    {workspaceView}
                    <SplitDropOverlay />
                  </div>
                </Panel>
                {hasSession && (
                  <>
                    <PanelResizeHandle className="group/resize relative w-2 cursor-col-resize bg-transparent after:absolute after:inset-y-0 after:left-1/2 after:w-px after:-translate-x-1/2 after:bg-border after:transition-colors after:content-[''] hover:after:bg-primary active:after:bg-primary" />
                    <Panel
                      id="inspector"
                      ref={inspectorRef}
                      collapsible
                      collapsedSize={0}
                      defaultSize={22}
                      minSize={14}
                      maxSize={40}
                      order={3}
                      onCollapse={() => setInspectorCollapsed(true)}
                      onExpand={() => setInspectorCollapsed(false)}
                    >
                      <Inspector />
                    </Panel>
                  </>
                )}
              </PanelGroup>
            </>
          )}
        </div>
        <ActivityCenter />
        {mobileRailOpen && (
          <MobileSessionRailDialog onClose={() => setMobileRailOpen(false)} />
        )}
        {settingsOpen && (
          <SettingsDialog onClose={() => setSettingsOpen(false)} />
        )}
        </div>
      </SettingsDialogProvider>
      {typeof document !== "undefined" &&
        createPortal(
          <DragOverlay dropAnimation={null} zIndex={80}>
            {draggingSession ? (
              <SessionDragPreview session={draggingSession} />
            ) : null}
          </DragOverlay>,
          document.body,
        )}
    </DndContext>
  );
}
