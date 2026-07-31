import { expect, type APIRequestContext } from "@playwright/test";

interface SessionDetail {
  status: string;
  artifact_count: number;
  report_status: string;
  dataset_names: string[];
}

/** Polls the real run index until the worker process finishes the pipeline. */
export async function waitForRunCompleted(
  request: APIRequestContext,
  sessionId: string,
  timeoutMs = 90_000,
): Promise<SessionDetail> {
  const deadline = Date.now() + timeoutMs;
  let last = "unknown";
  while (Date.now() < deadline) {
    const response = await request.get(`/api/v1/sessions/${sessionId}`);
    if (response.ok()) {
      const detail = (await response.json()) as SessionDetail;
      last = detail.status;
      if (detail.status === "completed") return detail;
      expect(detail.status, "run must not fail").not.toBe("failed");
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new Error(`run ${sessionId} did not complete in ${timeoutMs}ms (last status: ${last})`);
}
