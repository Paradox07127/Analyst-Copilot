/* Case ③: the report page renders the markdown the API returns for a run
 * whose pipeline generated one. */

import { expect, test } from "@playwright/test";
import { readSeed } from "./support/seed";

test("report page renders the generated report", async ({ page, request }) => {
  const { projectId, sessionId } = readSeed();

  const report = await (
    await request.get(`/api/v1/sessions/${sessionId}/report`)
  ).json();
  expect(report.status).not.toBe("none");
  expect(report.markdown.length).toBeGreaterThan(0);

  await page.goto(`/projects/${projectId}/sessions/${sessionId}/report`);

  const main = page.getByRole("main");
  await expect(page.getByRole("heading", { name: "Report", exact: true })).toBeVisible();
  await expect(main.getByText(report.status, { exact: true })).toBeVisible();

  /* The markdown is rendered, not dumped: its h1 becomes a real heading. */
  const firstHeading = report.markdown
    .split("\n")
    .find((line: string) => line.startsWith("# "))!
    .slice(2)
    .trim();
  await expect(
    main.getByRole("heading", { name: firstHeading, level: 1 }),
  ).toBeVisible();
  await expect(main.getByRole("heading", { name: "Executive Summary" })).toBeVisible();
});
