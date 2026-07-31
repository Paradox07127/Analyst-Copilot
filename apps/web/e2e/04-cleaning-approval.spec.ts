/* Case ④: the cleaning approval is one-shot. Preview registers a pending
 * approval, the first apply consumes it, and a second apply of the same
 * approval fails asynchronously with approval_consumed and a guided prompt.
 *
 * The first apply is issued out of band because a successful apply navigates
 * away from the page; both applies still use the real durable-operation
 * endpoint and worker. */

import { randomUUID } from "node:crypto";
import { expect, test, type APIRequestContext } from "@playwright/test";
import {
  FIXTURE_ROWS_AFTER_DEDUPE,
  FIXTURE_ROW_COUNT,
  readSeed,
} from "./support/seed";
import { waitForRunCompleted } from "./support/session";

async function waitForOperationResult<T>(
  request: APIRequestContext,
  jobId: string,
  resultPath: string,
): Promise<T> {
  await expect.poll(
    async () => (await request.get(`/api/v1/jobs/${jobId}`)).json()
      .then((job) => job.status),
    { timeout: 60_000, message: `${jobId} must complete` },
  ).toBe("completed");
  const response = await request.get(`/api/v1/jobs/${jobId}/${resultPath}`);
  expect(response.status(), await response.text()).toBe(200);
  return response.json() as Promise<T>;
}

test("cleaning approval cannot be applied twice", async ({ page, request }) => {
  const { projectId, sessionId } = readSeed();
  await page.goto(`/projects/${projectId}/sessions/${sessionId}/cleaning`);
  await expect(page.getByRole("heading", { name: "Cleaning" })).toBeVisible();

  const previewResponse = page.waitForResponse(
    (response) =>
      response.url().includes("/cleaning/preview") &&
      response.request().method() === "POST",
  );
  await page.getByRole("button", { name: "Preview cleaning" }).click();
  const previewStarted = await (await previewResponse).json();
  const approval = await waitForOperationResult<{
    action_hash: string;
    approval_token: string;
  }>(
    request,
    previewStarted.job.job_id,
    "cleaning-preview-result",
  );

  const previewCard = page.getByRole("region", { name: "Cleaning preview" });
  await expect(previewCard).toContainText(
    `Rows ${FIXTURE_ROW_COUNT} → ${FIXTURE_ROWS_AFTER_DEDUPE}`,
  );
  await expect(previewCard).toContainText("Remove exact duplicate rows.");

  const firstApply = await request.post(
    `/api/v1/sessions/${sessionId}/cleaning/apply`,
    {
      headers: { "Idempotency-Key": randomUUID() },
      data: {
        action_hash: approval.action_hash,
        approval_token: approval.approval_token,
        llm: "offline",
      },
    },
  );
  expect(firstApply.status(), await firstApply.text()).toBe(202);
  const applyStarted = await firstApply.json();
  const applied = await waitForOperationResult<{ new_session_id: string }>(
    request,
    applyStarted.job.job_id,
    "cleaning-apply-result",
  );
  expect(applied.new_session_id).toBeTruthy();
  await waitForRunCompleted(request, applied.new_session_id);

  /* Second apply — same approval, different idempotency key, via the UI. */
  await page.getByRole("button", { name: "Apply & analyze" }).click();
  const dialog = page.getByRole("alertdialog", { name: "Confirm cleaning" });
  await expect(dialog).toBeVisible();

  const replayResponse = page.waitForResponse(
    (response) =>
      response.url().includes("/cleaning/apply") &&
      response.request().method() === "POST",
  );
  await dialog.getByRole("button", { name: "Confirm apply" }).click();
  const replay = await replayResponse;
  expect(replay.status()).toBe(202);

  await expect(page.getByText("This preview was already applied.")).toBeVisible();
  await expect(
    page.getByRole("button", {
      name: "Preview again (creates a new cleaned version)",
    }),
  ).toBeVisible();
  /* Refused, so the page stayed put instead of routing to a second fork. */
  await expect(page).toHaveURL(new RegExp(`/sessions/${sessionId}/cleaning$`));
});
