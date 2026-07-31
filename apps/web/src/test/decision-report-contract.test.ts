import { describe, expect, it } from "vitest";
import type { operations } from "../api/generated/schema";

type DecisionReportResponses =
  operations["get_decision_report_api_v1_sessions__session_id__decision_report_get"]["responses"];
type HasReadFailureContract =
  404 extends keyof DecisionReportResponses
    ? 500 extends keyof DecisionReportResponses
      ? 503 extends keyof DecisionReportResponses
        ? true
        : false
      : false
    : false;

describe("generated decision-report read contract", () => {
  it("retains missing, corrupt, and retryable response variants", () => {
    const generatedContractHasReadFailures: HasReadFailureContract = true;
    expect(generatedContractHasReadFailures).toBe(true);
  });
});
