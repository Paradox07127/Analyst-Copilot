import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";

export type WorkspacePane = "left" | "right";
export type WorkspaceMode = "single" | "compare" | "split";

export interface WorkspacePaneContext {
  projectId: string;
  sessionId: string;
  section: string;
  selectedEntity?: { type: string; id: string; label?: string } | null;
  onSectionChange?: (section: string) => void;
}

interface WorkspaceFocusState {
  mode: WorkspaceMode;
  activePane: WorkspacePane;
  left: WorkspacePaneContext | null;
  right: WorkspacePaneContext | null;
}

interface WorkspaceFocusValue extends WorkspaceFocusState {
  activeContext: WorkspacePaneContext | null;
  configure: (
    mode: WorkspaceMode,
    left: WorkspacePaneContext | null,
    right: WorkspacePaneContext | null,
  ) => void;
  focusPane: (pane: WorkspacePane) => void;
  reset: () => void;
}

const INITIAL_STATE: WorkspaceFocusState = {
  mode: "single",
  activePane: "left",
  left: null,
  right: null,
};

const WorkspaceFocusContext = createContext<WorkspaceFocusValue | null>(null);

export function WorkspaceFocusProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState(INITIAL_STATE);

  const configure = useCallback(
    (
      mode: WorkspaceMode,
      left: WorkspacePaneContext | null,
      right: WorkspacePaneContext | null,
    ) => {
      setState((current) => ({
        mode,
        activePane:
          current.activePane === "right" && right ? "right" : "left",
        left,
        right,
      }));
    },
    [],
  );

  const focusPane = useCallback((activePane: WorkspacePane) => {
    setState((current) => ({ ...current, activePane }));
  }, []);

  const reset = useCallback(() => setState(INITIAL_STATE), []);
  const activeContext =
    state.activePane === "right" ? state.right : state.left;
  const value = useMemo(
    () => ({ ...state, activeContext, configure, focusPane, reset }),
    [activeContext, configure, focusPane, reset, state],
  );

  return (
    <WorkspaceFocusContext.Provider value={value}>
      {children}
    </WorkspaceFocusContext.Provider>
  );
}

export function useWorkspaceFocus(): WorkspaceFocusValue {
  const context = useContext(WorkspaceFocusContext);
  if (!context) {
    throw new Error("useWorkspaceFocus must be used inside WorkspaceFocusProvider");
  }
  return context;
}
