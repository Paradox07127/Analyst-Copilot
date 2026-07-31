import { QueryClient } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import { createMemoryRouter, RouterProvider } from "react-router";
import { AppProviders } from "../app/providers";
import { routes } from "../app/router";

/* Fresh client per render: no cache bleed between tests, no retry delays. */
function testQueryClient() {
  return new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
}

export function renderAppAt(path: string) {
  const router = createMemoryRouter(routes, { initialEntries: [path] });
  return render(
    <AppProviders client={testQueryClient()}>
      <RouterProvider router={router} />
    </AppProviders>,
  );
}

export function renderAppWithRouterAt(path: string) {
  const router = createMemoryRouter(routes, { initialEntries: [path] });
  const view = render(
    <AppProviders client={testQueryClient()}>
      <RouterProvider router={router} />
    </AppProviders>,
  );
  return { router, view };
}
