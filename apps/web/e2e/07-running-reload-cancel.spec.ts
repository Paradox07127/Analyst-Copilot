/* A deliberately non-trivial CSV keeps the offline worker active long enough
 * to prove that browser reload recovery and cancellation work on the real
 * stack. No timing stub or network mock is involved. */

import { expect, test } from "@playwright/test";
import { PROJECT_ID } from "./support/seed";

function slowCsv(rowCount = 250_000): Buffer {
  const rows = ["id,segment,region,amount,units,discount,score,event_date"];
  for (let index = 1; index <= rowCount; index += 1) {
    rows.push(
      `${index},segment_${index % 17},region_${index % 9},${(index % 10_000) / 10},`
      + `${index % 101},${index % 7},${index % 1_000},2026-07-${String((index % 28) + 1).padStart(2, "0")}`,
    );
  }
  return Buffer.from(rows.join("\n"));
}

test("a running job survives reload and can be cancelled from Activity", async ({
  page,
  request,
}) => {
  await page.goto(`/projects/${PROJECT_ID}/new-session`);
  await page.setInputFiles("#launchpad-files", {
    name: "reload_cancel.csv",
    mimeType: "text/csv",
    buffer: slowCsv(),
  });
  await expect(page.getByText(/^Ready · ds_/)).toBeVisible({ timeout: 60_000 });
  await page.selectOption("#launchpad-llm", "offline");
  await page.getByRole("button", { name: "Run analysis" }).click();

  await expect(page).toHaveURL(/\/sessions\/sess_[^/]+\/data-map$/);
  const sessionId = new URL(page.url()).pathname.split("/sessions/")[1].split("/")[0];
  await page.getByRole("button", { name: "Open activity" }).click();
  const drawer = page.getByRole("dialog", { name: "Activity" });
  const tracking = drawer.getByText(/Tracking job_[^ ]+ · session run_/);
  await expect(tracking).toBeVisible();
  const trackingText = await tracking.textContent();
  const jobId = trackingText?.match(/Tracking (job_[^ ]+)/)?.[1];
  expect(jobId).toBeDefined();
  await expect(drawer.getByRole("button", { name: "Cancel job" })).toBeVisible();

  await page.reload();

  await expect(page).toHaveURL(new RegExp(`/sessions/${sessionId}/data-map$`));
  await expect(drawer.getByText(new RegExp(`Tracking ${jobId}`))).toBeVisible();
  await drawer.getByRole("button", { name: "Cancel job" }).click();
  await page.getByRole("button", { name: "Confirm cancel" }).click();
  await expect(drawer.getByText("Cancelled", { exact: true })).toBeVisible({
    timeout: 60_000,
  });

  const job = await (await request.get(`/api/v1/jobs/${jobId}`)).json();
  expect(job.status).toBe("cancelled");
  const run = await (await request.get(`/api/v1/sessions/${sessionId}`)).json();
  expect(run.status).toBe("cancelled");
});
