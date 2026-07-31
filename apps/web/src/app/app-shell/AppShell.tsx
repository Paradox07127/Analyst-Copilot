import { useEffect, useRef, useState } from "react";
import { Outlet, useLocation } from "react-router";
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
import { WorkspaceFocusProvider } from "../workspace-focus";

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

export function AppShell() {
  const { pathname } = useLocation();
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
  const hasSession =
    /\/sessions\/[^/]+/.test(pathname) || /\/projects\/[^/]+/.test(pathname);
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
        if (hasSession) panel.expand();
        else panel.collapse();
      } catch {
        if (retry) frame = requestAnimationFrame(() => apply(false));
      }
    };
    apply(true);
    return () => cancelAnimationFrame(frame);
  }, [hasSession]);

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

  return (
    <JobActivityProvider>
      <SettingsDialogProvider value={openSettings}>
        <WorkspaceFocusProvider>
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
            <div className="flex min-h-0 min-w-0 flex-1 flex-col">
              <SessionNav />
              <main className="min-h-0 min-w-0 flex-1 overflow-auto">
                <Outlet />
              </main>
            </div>
          ) : (
            <>
              {railCollapsed && (
                <CollapsedSessionRail />
              )}
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
                    <PanelResizeHandle className="w-1 bg-border transition-colors hover:bg-primary" />
                  </>
                )}
                <Panel id="main" defaultSize={56} minSize={30} order={2}>
                  <div className="flex h-full min-h-0 min-w-0 flex-col">
                    <SessionNav />
                    <main className="min-h-0 min-w-0 flex-1 overflow-auto">
                      <Outlet />
                    </main>
                  </div>
                </Panel>
                {hasSession && (
                  <>
                    <PanelResizeHandle className="w-1 bg-border transition-colors hover:bg-primary" />
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
        </WorkspaceFocusProvider>
      </SettingsDialogProvider>
    </JobActivityProvider>
  );
}
