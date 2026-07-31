/* Case ②: a table deep link with an offset must survive both a cold load
 * (SPA fallback on an unknown server path) and a browser refresh. */

import { expect, test, type Page } from "@playwright/test";
import { FIXTURE_ROW_COUNT, readSeed } from "./support/seed";

const PAGE_SIZE = 100;
const TABLE_FIRST_PAINT_BUDGET_MS = 1_000;

const RUN_ROUTES = [
  ["data-map", "Data Map"],
  ["quality", "Quality"],
  ["profiles", "Profiles & Charts"],
  ["relationships", "Relationships"],
  ["questions", "Questions"],
  ["findings", "Findings"],
  ["semantic", "Knowledge"],
  ["cleaning", "Cleaning"],
  ["deep-analysis", "Deep analysis"],
  ["trace", "Trace & cost"],
  ["report", "Report"],
  ["artifacts", "Artifacts"],
  ["compare", "Compare"],
  ["skills", "Skills"],
  ["chat", "Chat"],
  ["board", "Investigation board"],
] as const;

/* The header is row 0; virtualiser spacers are aria-hidden and not rows. */
function firstDataCell(page: Page) {
  return page.getByRole("row").nth(1).getByRole("cell").first();
}

test("deep link to a table page with an offset survives a refresh", async ({
  page,
}) => {
  const { projectId, sessionId, datasetId } = readSeed();
  const url = `/projects/${projectId}/sessions/${sessionId}/table/${datasetId}?offset=${PAGE_SIZE}`;

  await page.goto(url);
  await expect(page.getByRole("heading", { name: "Table Preview" })).toBeVisible();

  const footer = page.getByText(`Rows ${PAGE_SIZE + 1}–${FIXTURE_ROW_COUNT}`);
  await expect(footer).toBeVisible();
  /* Row 101 of the fixture: proof the offset reached the server, not just the URL. */
  await expect(firstDataCell(page)).toHaveText("C0101");
  await expect(page.getByRole("button", { name: "Next" })).toBeDisabled();

  await page.reload();

  await expect(page).toHaveURL(new RegExp(`offset=${PAGE_SIZE}$`));
  await expect(footer).toBeVisible();
  await expect(firstDataCell(page)).toHaveText("C0101");

  /* Paging back rewrites the URL, and the fresh URL is itself deep-linkable. */
  await page.getByRole("button", { name: "Prev" }).click();
  await expect(page).not.toHaveURL(/offset=/);
  await expect(firstDataCell(page)).toHaveText("C0001");
  await expect(page.getByText(`Rows 1–${PAGE_SIZE}`)).toBeVisible();
});

test("every run page supports a cold deep link and hard refresh", async ({
  page,
}) => {
  const { projectId, sessionId } = readSeed();

  for (const [route, heading] of RUN_ROUTES) {
    const response = await page.goto(`/projects/${projectId}/sessions/${sessionId}/${route}`);
    expect(response?.status(), `${route} cold-load status`).toBe(200);
    await expect(
      page.getByRole("main").getByRole("heading", { name: heading, exact: true }),
    ).toBeVisible();

    await page.reload();
    await expect(
      page.getByRole("main").getByRole("heading", { name: heading, exact: true }),
    ).toBeVisible();
    await expect(page.getByText(/Application error|Something went wrong|Page not found/i))
      .toHaveCount(0);
  }
});

test("table meaningful first paint stays below the one-second budget", async ({
  page,
}) => {
  const { projectId, sessionId, datasetId } = readSeed();
  const startedAt = Date.now();
  const response = await page.goto(
    `/projects/${projectId}/sessions/${sessionId}/table/${datasetId}?offset=${PAGE_SIZE}`,
  );

  expect(response?.status()).toBe(200);
  await expect(page.getByRole("heading", { name: "Table Preview" })).toBeVisible();
  await expect(page.getByText(`Rows ${PAGE_SIZE + 1}–${FIXTURE_ROW_COUNT}`))
    .toBeVisible();
  expect(Date.now() - startedAt).toBeLessThan(TABLE_FIRST_PAINT_BUDGET_MS);

  const fcp = await page.evaluate(() =>
    performance.getEntriesByName("first-contentful-paint")[0]?.startTime,
  );
  expect(fcp, "browser must expose a first-contentful-paint entry").toBeDefined();
  expect(fcp!).toBeLessThan(TABLE_FIRST_PAINT_BUDGET_MS);
});
