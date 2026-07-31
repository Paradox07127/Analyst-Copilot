import { useEffect, useRef } from "react";
import {
  createBrowserRouter,
  redirect,
  type RouteObject,
  useRouteError,
} from "react-router";
import { AppShell } from "./app-shell/AppShell";
import { projectComparePath, sessionSectionPath } from "./paths";

function RouteLoadingFallback() {
  return (
    <div role="status" className="p-6 text-sm text-status-neutral">
      Loading page…
    </div>
  );
}

const CHUNK_RELOAD_ERROR_KEY = "eda.chunk-reload-error";

export function isDynamicImportFailure(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error);
  return /dynamically imported module|importing a module script failed/i.test(message);
}

function RouteErrorFallback() {
  const error = useRouteError();
  const message = error instanceof Error ? error.message : String(error);
  const reloadAttempted = useRef(false);
  const canRecover =
    typeof window !== "undefined" &&
    isDynamicImportFailure(error) &&
    window.sessionStorage.getItem(CHUNK_RELOAD_ERROR_KEY) !== message;

  useEffect(() => {
    if (!canRecover || reloadAttempted.current) return;
    reloadAttempted.current = true;
    window.sessionStorage.setItem(CHUNK_RELOAD_ERROR_KEY, message);
    window.location.reload();
  }, [canRecover, message]);

  return (
    <main className="flex min-h-dvh items-center justify-center bg-bg p-6">
      <section className="max-w-md rounded-base border border-border bg-surface p-5">
        <h1 className="text-lg font-semibold">
          {canRecover ? "Updating page…" : "This page could not be loaded"}
        </h1>
        <p className="mt-1 text-sm text-status-neutral">
          {canRecover
            ? "Loading the latest version of the workbench."
            : "Refresh the page to load the latest version and try again."}
        </p>
        {!canRecover && (
          <button
            type="button"
            onClick={() => window.location.reload()}
            className="mt-3 rounded-base bg-primary px-3 py-1.5 text-sm font-medium text-bg hover:opacity-90"
          >
            Refresh page
          </button>
        )}
      </section>
    </main>
  );
}

export const routes: RouteObject[] = [
  {
    path: "/",
    Component: AppShell,
    HydrateFallback: RouteLoadingFallback,
    errorElement: <RouteErrorFallback />,
    children: [
      { index: true, loader: () => redirect("/projects") },
      {
        path: "projects",
        lazy: () => import("../features/projects/ProjectListPage"),
      },
      {
        path: "new-session",
        lazy: () => import("../features/launchpad/LaunchpadPage"),
      },
      {
        path: "settings",
        lazy: () => import("../features/settings/SettingsPage"),
      },
      {
        path: "projects/:projectId/new-session",
        lazy: () => import("../features/launchpad/LaunchpadPage"),
      },
      {
        path: "projects/:projectId/compare",
        lazy: () => import("../features/compare/ComparePage"),
      },
      {
        path: "projects/:projectId/sessions/:sessionId",
        children: [
          {
            index: true,
            loader: ({ params }) =>
              redirect(
                sessionSectionPath(params.projectId ?? "", params.sessionId ?? "", "data-map"),
              ),
          },
          {
            path: "data-map",
            lazy: () => import("../features/datasets/DataMapPage"),
          },
          {
            path: "table/:datasetId",
            lazy: () => import("../features/datasets/TablePreviewPage"),
          },
          {
            path: "quality",
            lazy: () => import("../features/insights/QualityPage"),
          },
          {
            path: "profiles",
            lazy: () => import("../features/insights/ProfilesPage"),
          },
          {
            path: "relationships",
            lazy: () => import("../features/relationships/RelationshipsPage"),
          },
          {
            path: "questions",
            lazy: () => import("../features/questions/QuestionsPage"),
          },
          {
            path: "findings",
            lazy: () => import("../features/findings/FindingsPage"),
          },
          {
            path: "semantic",
            lazy: () => import("../features/semantic/KnowledgePage"),
          },
          {
            path: "cleaning",
            lazy: () => import("../features/cleaning/CleaningPage"),
          },
          {
            path: "deep-analysis",
            lazy: () => import("../features/analysis/DeepAnalysisPage"),
          },
          {
            path: "trace",
            lazy: () => import("../features/trace/TracePage"),
          },
          {
            path: "report",
            lazy: () => import("../features/reports/ReportPage"),
          },
          {
            path: "artifacts",
            lazy: () => import("../features/artifacts/ArtifactsPage"),
          },
          {
            path: "compare",
            loader: ({ params, request }) => {
              const query = new URL(request.url).searchParams;
              query.set("left", params.sessionId ?? "");
              return redirect(
                `${projectComparePath(params.projectId ?? "")}?${query.toString()}`,
              );
            },
          },
          {
            path: "skills",
            lazy: () => import("../features/skills/SkillsPage"),
          },
          {
            path: "chat",
            lazy: () => import("../features/chat/ChatPage"),
          },
          {
            path: "board",
            lazy: () => import("../features/board/BoardPage"),
          },
        ],
      },
    ],
  },
];

export function createAppRouter() {
  return createBrowserRouter(routes);
}
