/* Case ①: upload a CSV in the Launchpad, start an offline analysis, watch the
 * Activity panel stepper, and check the Data Map against the API's numbers
 * for the run that was just produced. */

import { expect, test } from "@playwright/test";
import { FIXTURE_CSV, FIXTURE_ROW_COUNT, PROJECT_ID } from "./support/seed";

function kpiValue(page: import("@playwright/test").Page, label: string) {
  return page
    .getByRole("main")
    .getByText(label, { exact: true })
    .locator("xpath=..");
}

test("upload → offline run → Activity progress → Data Map shows the run's numbers", async ({
  page,
  request,
}) => {
  await page.goto(`/projects/${PROJECT_ID}/new-session`);
  await expect(page.getByRole("heading", { name: "New run" })).toBeVisible();

  await page.setInputFiles("#launchpad-files", FIXTURE_CSV);
  await expect(page.getByText(/^Ready · ds_/)).toBeVisible();

  await page.selectOption("#launchpad-llm", "offline");
  await page
    .getByLabel("Business context")
    .fill("Playwright end-to-end smoke run.");
  await page.getByRole("button", { name: "Run analysis" }).click();

  /* Launchpad hands the job to Activity and routes to the run. */
  await expect(page).toHaveURL(/\/sessions\/sess_[^/]+\/data-map$/);
  const sessionId = new URL(page.url()).pathname.split("/sessions/")[1].split("/")[0];

  await page.getByRole("button", { name: "Open activity" }).click();
  const drawer = page.getByRole("dialog", { name: "Activity" });
  await expect(drawer.getByLabel("Pipeline phases")).toBeVisible();
  await expect(drawer.getByLabel("Reading data: done")).toBeVisible({
    timeout: 90_000,
  });
  await expect(drawer.getByText("Completed", { exact: true })).toBeVisible({
    timeout: 90_000,
  });
  /* Progress and the raw trace are separate sections in the same panel. */
  await drawer.getByRole("tab", { name: /Event log/ }).click();
  await expect(drawer.getByLabel("Job event log")).toContainText("job.completed");

  /* Ground truth for the assertions below comes from the API, so the test
   * fails if the page renders stale or invented numbers. */
  const detail = await (await request.get(`/api/v1/sessions/${sessionId}`)).json();
  expect(detail.status).toBe("completed");
  expect(detail.artifact_count).toBeGreaterThan(0);

  await expect(kpiValue(page, "Datasets")).toContainText("1");
  await expect(kpiValue(page, "Artifacts")).toContainText(
    String(detail.artifact_count),
  );
  await expect(kpiValue(page, "Report")).toContainText(detail.report_status);
  await expect(
    page.getByRole("heading", { name: "e2e_sales.csv" }),
  ).toBeVisible();
  await expect(page.getByRole("main")).toContainText(
    `${FIXTURE_ROW_COUNT} rows · 4 cols`,
  );
});
