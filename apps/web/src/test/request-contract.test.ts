import { afterEach, describe, expect, expectTypeOf, it, vi } from "vitest";
import {
  api,
  type BoardUpdateRequest,
  type CleaningApplyRequest,
  type InvestigationDecisionPrepareRequest,
  type InvestigationPlanRequest,
  type ProposalAcceptRequest,
  type QuestionCardEdit,
  type SeedImportRequest,
  type SemanticSeedsUpdateRequest,
  type SkillReplayPrepareRequest,
} from "../api/client";

describe("generated request contracts", () => {
  it("keeps the eleven historically drifted request shapes generated", () => {
    const cleaning: CleaningApplyRequest = {
      action_hash: "hash",
      approval_token: "token",
      llm: "env",
    };
    const semantic: SemanticSeedsUpdateRequest = {
      expected_version: 1,
      field_meanings: [],
      metric_definitions: null,
      entity_notes: null,
      verified_answers: null,
    };
    const proposal: ProposalAcceptRequest = {
      dataset: "orders",
      column: "amount",
      expected_version: 3,
      meaning: null,
      unit: null,
    };
    const seed: SeedImportRequest = {
      dataset_ids: ["orders"],
      name: "Imported seed",
    };
    const replay: SkillReplayPrepareRequest = { dataset_ids: ["orders"] };
    const board: BoardUpdateRequest = { expected_version: 2 };
    const plan: InvestigationPlanRequest = {
      question_ids: ["question_1"],
      deep: false,
    };
    const decision: InvestigationDecisionPrepareRequest = {
      decision: "approved",
      reason: "",
    };
    const edit: QuestionCardEdit = {
      expected_version: 1,
      question_en: null,
      business_decision: null,
      risks: null,
      data_requirements: null,
    };

    expect({
      cleaning,
      semantic,
      proposal,
      seed,
      replay,
      board,
      plan,
      decision,
      edit,
    }).toBeTruthy();
    expectTypeOf(api.prepareQuestion).toBeCallableWith("run", "question");
    /* The key is mandatory: an omitted one silently disables replay. */
    expectTypeOf(api.generateReport).toBeCallableWith("run", undefined, "key");
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("omits optional request bodies instead of serializing a shadow shape", async () => {
    const requests: Array<{ input: RequestInfo | URL; init?: RequestInit }> = [];
    const fetchMock = vi.fn(
      async (input: RequestInfo | URL, init?: RequestInit) => {
        requests.push({ input, init });
        return new Response("{}", {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    await api.prepareQuestion("run", "question");
    await api.generateReport("run", undefined, "idempotency-key");

    expect(requests).toHaveLength(2);
    for (const { init } of requests) {
      expect(init?.body).toBeUndefined();
      expect(new Headers(init?.headers).has("Content-Type")).toBe(false);
    }
    expect(
      new Headers(requests[1]?.init?.headers).get("Idempotency-Key"),
    ).toBe("idempotency-key");
  });

  it("adds the remote CSRF signal to mutations without changing reads", async () => {
    const requests: RequestInit[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
        requests.push(init ?? {});
        return new Response("{}", {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }),
    );

    await api.createUpload(
      "demo",
      new File(["a,b\n1,2\n"], "data.csv", { type: "text/csv" }),
    );
    await api.listProjects();

    expect(new Headers(requests[0]?.headers).get("X-EDA-CSRF")).toBe("1");
    expect(new Headers(requests[1]?.headers).has("X-EDA-CSRF")).toBe(false);
  });
});
