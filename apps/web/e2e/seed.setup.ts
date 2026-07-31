/* Seeds one finished offline run through the real API so the specs that need
 * existing analysis output (deep link, report, cleaning, board) do not each
 * pay for a full pipeline. The UI upload→run path is covered by 01-run.spec. */

import { readFileSync } from "node:fs";
import { expect, test as setup } from "@playwright/test";
import {
  FIXTURE_CSV,
  FIXTURE_ROW_COUNT,
  PROJECT_ID,
  writeSeed,
} from "./support/seed";
import { waitForRunCompleted } from "./support/session";

setup("seed a completed offline run", async ({ request }) => {
  const upload = await request.post(`/api/v1/projects/${PROJECT_ID}/uploads`, {
    multipart: {
      file: {
        name: "e2e_sales.csv",
        mimeType: "text/csv",
        buffer: readFileSync(FIXTURE_CSV),
      },
    },
  });
  expect(upload.status(), await upload.text()).toBe(201);
  const uploaded = await upload.json();
  expect(uploaded.status).toBe("completed");
  const datasetId: string = uploaded.dataset.dataset_id;

  const sessionId = `run_e2e_seed_${Date.now()}`;
  const job = await request.post(`/api/v1/sessions/${sessionId}/jobs`, {
    headers: { "Idempotency-Key": `seed-${sessionId}` },
    data: {
      kind: "auto_eda",
      project_id: PROJECT_ID,
      datasets: [datasetId],
      business_context: "E2E seed run.",
      generate_report: true,
      llm: "offline",
    },
  });
  expect(job.status(), await job.text()).toBe(201);

  const detail = await waitForRunCompleted(request, sessionId);
  expect(detail.artifact_count).toBeGreaterThan(0);
  expect(detail.dataset_names).toContain("e2e_sales.csv");

  const datasets = await request.get(`/api/v1/sessions/${sessionId}/datasets`);
  expect(datasets.ok()).toBeTruthy();
  const handles = await datasets.json();
  expect(handles[0].row_count).toBe(FIXTURE_ROW_COUNT);

  writeSeed({ projectId: PROJECT_ID, sessionId, datasetId });
});
