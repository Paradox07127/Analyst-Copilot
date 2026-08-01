import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import type { SessionDragPayload } from "./workspace-split";

export const SESSION_DRAG_TYPE = "session";

export interface SessionDragData {
  type: typeof SESSION_DRAG_TYPE;
  session: SessionDragPayload;
}

export function sessionDragId(
  session: SessionDragPayload,
  source: "rail" | "search" = "rail",
): string {
  return `session:${source}:${encodeURIComponent(session.projectId)}:${encodeURIComponent(session.sessionId)}`;
}

export function sessionDragData(session: SessionDragPayload): SessionDragData {
  return { type: SESSION_DRAG_TYPE, session };
}

export function readSessionDragData(value: unknown): SessionDragPayload | null {
  if (!value || typeof value !== "object") return null;
  const candidate = value as Partial<SessionDragData>;
  const session = candidate.session;
  if (
    candidate.type !== SESSION_DRAG_TYPE ||
    !session ||
    typeof session.projectId !== "string" ||
    typeof session.sessionId !== "string" ||
    typeof session.label !== "string"
  ) {
    return null;
  }
  return session;
}

interface SessionDragValue {
  draggingSession: SessionDragPayload | null;
  beginSessionDrag: (session: SessionDragPayload) => void;
  endSessionDrag: () => void;
}

const SessionDragContext = createContext<SessionDragValue | null>(null);

export function SessionDragProvider({ children }: { children: ReactNode }) {
  const [draggingSession, setDraggingSession] =
    useState<SessionDragPayload | null>(null);
  const beginSessionDrag = useCallback(
    (session: SessionDragPayload) => setDraggingSession(session),
    [],
  );
  const endSessionDrag = useCallback(() => setDraggingSession(null), []);
  const value = useMemo(
    () => ({ draggingSession, beginSessionDrag, endSessionDrag }),
    [beginSessionDrag, draggingSession, endSessionDrag],
  );
  return (
    <SessionDragContext.Provider value={value}>
      {children}
    </SessionDragContext.Provider>
  );
}

export function useSessionDrag(): SessionDragValue {
  const value = useContext(SessionDragContext);
  if (!value) {
    throw new Error("useSessionDrag must be used inside SessionDragProvider");
  }
  return value;
}

/* The browser never snapshots a session row for dragging. This deliberately
 * small overlay is rendered in a body portal by AppShell, so a Marquee's
 * translated, full-width text layer can never expand the drag preview. */
export function SessionDragPreview({
  session,
}: {
  session: SessionDragPayload;
}) {
  return (
    <div
      data-testid="session-drag-preview"
      aria-hidden="true"
      className="pointer-events-none flex w-[224px] max-w-[calc(100vw-32px)] items-center gap-2 overflow-hidden rounded-base border border-primary bg-bg px-3 py-2 text-sm shadow-overlay"
    >
      <span className="size-2 shrink-0 rounded-full bg-primary" />
      <span className="min-w-0 flex-1 truncate font-medium">
        {session.label}
      </span>
    </div>
  );
}
