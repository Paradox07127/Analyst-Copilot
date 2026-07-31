/* Central route construction. Route params arrive decoded from useParams;
 * interpolating them raw breaks on ids containing `#`, `/`, or `?`. */

export function sessionBasePath(projectId: string, sessionId: string): string {
  return `/projects/${encodeURIComponent(projectId)}/sessions/${encodeURIComponent(sessionId)}`;
}

export function sessionSectionPath(
  projectId: string,
  sessionId: string,
  section: string,
): string {
  return `${sessionBasePath(projectId, sessionId)}/${section}`;
}

export function projectComparePath(
  projectId: string,
  leftSessionId?: string,
  rightSessionId?: string,
): string {
  const path = `/projects/${encodeURIComponent(projectId)}/compare`;
  const query = new URLSearchParams();
  if (leftSessionId) query.set("left", leftSessionId);
  if (rightSessionId && rightSessionId !== leftSessionId) {
    query.set("right", rightSessionId);
  }
  const search = query.toString();
  return search ? `${path}?${search}` : path;
}

/* Deep link into the Artifacts page: `?artifact=` opens that row and scrolls
 * to it, and falls back to a by-id fetch when it is not on the loaded page. */
export function artifactPath(
  projectId: string,
  sessionId: string,
  artifactId: string,
): string {
  const query = new URLSearchParams({ artifact: artifactId });
  return `${sessionSectionPath(projectId, sessionId, "artifacts")}?${query.toString()}`;
}

export function newProjectSessionPath(projectId: string): string {
  return `/projects/${encodeURIComponent(projectId)}/new-session`;
}

/** A standalone task. Its storage container is intentionally not exposed as a
 * project in the UI; see LaunchpadPage for the implementation boundary. */
export function newSessionPath(): string {
  return "/new-session";
}

export function tablePath(
  projectId: string,
  sessionId: string,
  datasetId: string,
): string {
  return `${sessionBasePath(projectId, sessionId)}/table/${encodeURIComponent(datasetId)}`;
}
