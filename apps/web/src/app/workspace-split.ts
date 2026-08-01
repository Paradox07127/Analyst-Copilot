import { sessionSectionPath } from "./paths";

export const EMPTY_WORKSPACE_PATH = "/workspace/empty";

export type WorkspaceSide = "left" | "right";

export interface SessionDragPayload {
  projectId: string;
  sessionId: string;
  label: string;
}

export interface SplitWorkspaceState {
  left: string;
  right: string;
  active: WorkspaceSide;
}

export interface WorkspacePathContext {
  projectId: string;
  sessionId: string;
  section: string;
}

const SPLIT_PATH = "/split";

function decodePathSegment(value: string): string {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

/** Only workbench pages may be rendered inside a pane. This also prevents a
 * malformed shared URL from recursively embedding the split workspace. */
export function normalizeWorkspacePath(value: string | null): string | null {
  if (!value || !value.startsWith("/") || value.startsWith("//")) return null;
  try {
    const url = new URL(value, "https://workspace.local");
    const supported =
      url.pathname === EMPTY_WORKSPACE_PATH ||
      /^\/projects\/[^/]+\/sessions\/[^/]+(?:\/.*)?$/.test(url.pathname) ||
      /^\/projects\/[^/]+\/compare$/.test(url.pathname);
    if (!supported) return null;
    return `${url.pathname}${url.search}${url.hash}`;
  } catch {
    return null;
  }
}

export function readSplitWorkspace(
  pathname: string,
  search: string,
): SplitWorkspaceState | null {
  if (pathname !== SPLIT_PATH) return null;
  const params = new URLSearchParams(search);
  const left = normalizeWorkspacePath(params.get("left"));
  const right = normalizeWorkspacePath(params.get("right"));
  if (!left || !right) return null;
  return {
    left,
    right,
    active: params.get("active") === "right" ? "right" : "left",
  };
}

export function splitWorkspacePath(
  left: string,
  right: string,
  active: WorkspaceSide,
): string {
  const normalizedLeft = normalizeWorkspacePath(left);
  const normalizedRight = normalizeWorkspacePath(right);
  if (!normalizedLeft || !normalizedRight) {
    throw new Error("Split panes require two supported workspace paths");
  }
  const params = new URLSearchParams({
    left: normalizedLeft,
    right: normalizedRight,
    active,
  });
  return `${SPLIT_PATH}?${params.toString()}`;
}

export function sessionWorkspacePath(payload: SessionDragPayload): string {
  return sessionSectionPath(payload.projectId, payload.sessionId, "data-map");
}

export function workspacePathContext(path: string): WorkspacePathContext | null {
  const normalized = normalizeWorkspacePath(path);
  if (!normalized) return null;
  const url = new URL(normalized, "https://workspace.local");
  const segments = url.pathname.split("/").filter(Boolean);
  if (segments[0] !== "projects") return null;
  const projectId = decodePathSegment(segments[1] ?? "");
  if (segments[2] !== "sessions") {
    if (segments[2] !== "compare") return null;
    const compare = new URLSearchParams(url.search);
    return {
      projectId,
      sessionId: compare.get("left") ?? "",
      section: compare.get("scope") ?? "overview",
    };
  }
  const sessionId = decodePathSegment(segments[3] ?? "");
  const routeSection = segments[4] ?? "data-map";
  return {
    projectId,
    sessionId,
    section: routeSection === "table" ? "table" : routeSection,
  };
}
