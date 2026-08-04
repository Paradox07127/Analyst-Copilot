import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";
import type {
  ExplorationPreparedDto,
  ExplorationViewDto,
} from "../api/client";
import {
  ExplorationReport,
  ExplorationRunPanel,
} from "../features/exploration/ExplorationView";
import {
  EXPLORATION_REPORT_SECTION_ORDER,
  buildExplorationReportGroups,
  coverageGaps,
  type ExplorationRunView,
} from "../features/exploration/exploration-model";
import { explorationRunFromDto } from "../features/exploration/exploration-adapter";
import { splitQualifiers } from "../features/reports/ReportBody";
import { FakeEventSource } from "./fake-event-source";
import { renderAppWithRouterAt } from "./render";
import { server } from "./msw/server";

function runView(overrides: Partial<ExplorationRunView> = {}): ExplorationRunView {
  return {
    explorationId: "expl_1",
    goal: "Find durable revenue patterns",
    tier: "standard",
    status: "stopped",
    stopReason: "no_new_information",
    currentHypothesis: {
      hypothesisId: "hyp_current",
      statement: "Revenue differs by region",
      whySelected: "Region is a mandatory unswept comparison.",
      status: "running",
    },
    currentEvidence: [
      {
        receiptId: "rcpt_region",
        toolName: "run_stat_test",
        summary: "North and South revenue comparison",
        factIds: ["fact_north", "fact_south"],
        facts: [
          { factId: "fact_north", name: "north_mean", value: 118.4, unit: "usd" },
          { factId: "fact_south", name: "south_mean", value: 92.1, unit: "usd" },
        ],
        statistics: {
          testName: "welch_t",
          outcome: "supports",
          testStatistic: 6.12,
          pValue: 0.0000000123,
          adjustedPValue: 0.0000000492,
          effectSize: 0.1165,
          ciLow: null,
          ciHigh: null,
          sampleSize: 1086,
        },
      },
    ],
    insights: [
      {
        insightId: "ins_supported",
        hypothesisId: "hyp_supported",
        statement: "Region is associated with revenue differences.",
        family: "Diagnostic",
        status: "reinforced",
        trustLevel: "supported",
        evidenceLane: "confirmatory",
        proof: [
          {
            receiptId: "rcpt_region",
            factIds: ["fact_north", "fact_south"],
            comparison: "supports",
          },
        ],
        limitations: ["Natural holdout is small."],
      },
      {
        insightId: "ins_refuted",
        hypothesisId: "hyp_refuted",
        statement: "The spike repeats every Monday.",
        family: "Exploratory",
        status: "refuted",
        trustLevel: "refuted",
        evidenceLane: "exploratory",
        proof: [
          {
            receiptId: "rcpt_dates",
            factIds: ["fact_monday"],
            comparison: "contradicts",
          },
        ],
        limitations: [],
      },
      {
        insightId: "ins_inconclusive",
        hypothesisId: "hyp_inconclusive",
        statement: "Missingness depends on channel.",
        family: "Diagnostic",
        status: "inconclusive",
        trustLevel: "unsupported",
        evidenceLane: "exploratory",
        proof: [
          {
            receiptId: "rcpt_missing",
            factIds: ["fact_missing"],
            comparison: "supports",
          },
        ],
        limitations: ["Sample is sparse."],
      },
    ],
    limitations: ["Single frozen data snapshot."],
    coverageTargets: [
      "spike_day",
      "region_difference",
      "missingness_mechanism",
      "spike_day",
    ],
    coverageCompleted: ["region_difference"],
    budget: {
      base: {
        modelRequests: 10,
        successfulToolCalls: 12,
        rounds: 4,
        costUsd: 5,
      },
      amendments: [
        {
          amendmentId: "amend_1",
          reason: "Approved one bounded follow-up.",
          increase: {
            modelRequests: 2,
            successfulToolCalls: 3,
            rounds: 1,
            costUsd: 2,
          },
        },
      ],
      used: {
        modelRequests: 5,
        successfulToolCalls: 4,
        rounds: 2,
        costUsd: 1.25,
      },
    },
    ...overrides,
  };
}

describe("E5 exploration report contract", () => {
  it("computes not-explored coverage as a deterministic set difference", () => {
    expect(
      coverageGaps(
        ["spike_day", "region_difference", "missingness_mechanism", "spike_day"],
        ["region_difference", "not_a_target"],
      ),
    ).toEqual(["missingness_mechanism", "spike_day"]);

    const groups = buildExplorationReportGroups(runView());
    expect(groups.coverageGaps).toEqual(["missingness_mechanism", "spike_day"]);
    expect(groups.limitations).toEqual([
      "Natural holdout is small.",
      "Sample is sparse.",
      "Single frozen data snapshot.",
    ]);
  });

  it("always renders the exact six report sections in fixed order", () => {
    const { container } = render(<ExplorationReport run={runView()} />);
    const sections = [...container.querySelectorAll("[data-section-id]")];
    expect(sections.map((section) => section.getAttribute("data-section-id"))).toEqual(
      EXPLORATION_REPORT_SECTION_ORDER,
    );
    expect(sections).toHaveLength(6);
    expect(within(sections[1] as HTMLElement).getByText(/spike repeats/)).toBeInTheDocument();
    expect(within(sections[4] as HTMLElement).getByText("spike_day")).toBeInTheDocument();
    expect(within(sections[5] as HTMLElement).getByText("no_new_information")).toBeInTheDocument();
  });

  it("visually separates evidence lanes without calling either one established fact", () => {
    render(<ExplorationReport run={runView()} />);
    const confirmatory = screen.getByText("Confirmatory evidence");
    const exploratory = screen.getAllByText("Exploratory")[0];
    expect(confirmatory.className).toContain("text-status-info");
    expect(exploratory?.className).toContain("text-status-warn");
    expect(screen.getByText(/neither label is a claim of certainty/i)).toBeInTheDocument();
    expect(document.body).not.toHaveTextContent(/Confirmed insights|Validated insights/);
    expect(
      splitQualifiers(
        "[Confirmatory evidence — not a claim of certainty] supported claim",
      ),
    ).toEqual({
      labels: ["Confirmatory evidence — not a claim of certainty"],
      rest: "supported claim",
    });
  });
});

describe("E5 exploration run component", () => {
  it("shows the adjudicating numbers, not just a receipt id", () => {
    render(<ExplorationRunPanel run={runView()} />);

    // A tiny p-value must survive formatting; fixed decimals would show 0.00.
    expect(screen.getByText("4.92e-8")).toBeInTheDocument();
    expect(screen.getByText("0.1165")).toBeInTheDocument();
    expect(screen.getByText("1086")).toBeInTheDocument();
    expect(screen.getByText("welch_t")).toBeInTheDocument();
    expect(screen.getAllByText("supports").length).toBeGreaterThan(0);
    expect(screen.getAllByText("rcpt_region").length).toBeGreaterThan(0);
  });

  it("distinguishes paused resumable state from stopped terminal state", () => {
    const paused = runView({ status: "paused", stopReason: null });
    const view = render(<ExplorationRunPanel run={paused} />);
    expect(screen.getByText("Paused · resumable")).toBeInTheDocument();
    expect(screen.getByText(/Paused is not a terminal stop/)).toBeInTheDocument();
    expect(screen.queryByText(/Stop reason:/)).not.toBeInTheDocument();

    view.rerender(<ExplorationRunPanel run={runView()} />);
    expect(screen.getByText("Stopped · terminal")).toBeInTheDocument();
    expect(screen.getByText("No new information")).toBeInTheDocument();
  });

  it("raises an alert when a run stopped because something went wrong", () => {
    // A failed run used to render the same quiet grey line as a healthy one,
    // so the seed-8 provider outage was indistinguishable from convergence.
    const failed = runView({ status: "stopped", stopReason: "failed" });
    const view = render(<ExplorationRunPanel run={failed} />);
    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent("Failed");
    expect(alert).toHaveTextContent(/evidence.*kept|already committed/i);

    view.rerender(
      <ExplorationRunPanel
        run={runView({ status: "stopped", stopReason: "state_witness_changed" })}
      />,
    );
    expect(screen.getByRole("alert")).toHaveTextContent(/data changed/i);
  });

  it("flags an incomplete run without calling it an error", () => {
    render(
      <ExplorationRunPanel
        run={runView({ status: "stopped", stopReason: "budget_exhausted" })}
      />,
    );
    const notice = screen.getByRole("status", { name: /stop/i });
    expect(notice).toHaveTextContent("Budget exhausted");
    expect(notice).toHaveTextContent(/incomplete/i);
  });

  it("keeps a healthy stop quiet", () => {
    render(<ExplorationRunPanel run={runView()} />);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.getByText("No new information")).toBeInTheDocument();
  });

  it("shows the current hypothesis, committed evidence, base plus amended budget, cost, and proof", () => {
    render(<ExplorationRunPanel run={runView()} />);
    expect(screen.getByText("Revenue differs by region")).toBeInTheDocument();
    expect(screen.getByText(/Region is a mandatory unswept comparison/)).toBeInTheDocument();
    expect(screen.getAllByText("rcpt_region")).toHaveLength(2);
    expect(screen.getByText("7 left")).toBeInTheDocument();
    expect(screen.getByText("11 left")).toBeInTheDocument();
    expect(screen.getByText("3 left")).toBeInTheDocument();
    expect(screen.getByText("$5.75 left")).toBeInTheDocument();
    expect(screen.getByText(/base 10 \+ 2 amended · 5 used/)).toBeInTheDocument();
    expect(screen.getByText("amend_1")).toBeInTheDocument();
    expect(screen.getByText("fact_north")).toBeInTheDocument();
    expect(screen.getByText("Cost $1.25 / $7.00 cap")).toBeInTheDocument();
  });

  it("is a read-only projection with no unsupported run mutation controls", () => {
    render(<ExplorationRunPanel run={runView({ status: "paused", stopReason: null })} />);
    expect(
      screen.queryByRole("button", { name: /pause|resume|cancel|write|apply/i }),
    ).not.toBeInTheDocument();
  });
});

const baseBudget = {
  llm: {
    max_requests: 10,
    max_input_tokens: 10000,
    max_output_tokens: 5000,
    max_total_tokens: 15000,
    max_cost_usd: "5.00",
    max_wall_seconds: 300,
    protected_requests: 0,
    protected_input_tokens: 0,
    protected_output_tokens: 0,
    protected_total_tokens: 0,
    protected_cost_usd: "0",
    unknown_usage_policy: "reject" as const,
  },
  max_successful_tool_calls: 12,
  max_tool_calls_by_kind: { sql_query: 8 },
  max_rows_scanned: 100000,
  max_result_cells: 10000,
  idle_timeout_seconds: 60,
  max_rounds: 4,
};

function explorationDto(
  patch: Partial<ExplorationViewDto> = {},
): ExplorationViewDto {
  return {
    exploration_id: "expl_1",
    session_id: "r1",
    project_id: "p1",
    goal: "Investigate regional revenue",
    thinking_level: "standard",
    status: "running",
    stop_reason: null,
    last_seq: 0,
    policy_fingerprint: "policy_12345678",
    effective_policy_fingerprint: "policy_12345678",
    data_state_witness: "witness_12345678",
    amendment_ids: [],
    current_hypothesis: {
      hypothesis_id: "hyp_region",
      statement: "Revenue differs by region",
      why_selected: "Mandatory diagnostic coverage",
      status: "running",
      priority: 0.9,
    },
    current_evidence: [{
      receipt_id: "rcpt_region",
      tool_name: "sql_query",
      summary: "Read-only regional aggregation",
      fact_ids: ["fact_north"],
      facts: [{ fact_id: "fact_north", name: "north_mean", value: 118.4, unit: "usd" }],
      statistics: null,
    }],
    insights: [{
      insight_id: "ins_region",
      hypothesis_id: "hyp_region",
      statement: "North is associated with higher observed revenue.",
      family: "Diagnostic",
      status: "new",
      trust_level: "supported",
      evidence_lane: "confirmatory",
      proof: [{
        receipt_id: "rcpt_region",
        fact_ids: ["fact_north"],
        comparison: "supports",
        evidence_independence_key: "holdout_1",
      }],
      limitations: ["Small holdout"],
    }],
    limitations: ["One frozen snapshot"],
    coverage_targets: ["Diagnostic", "Exploratory"],
    coverage_completed: ["Diagnostic"],
    coverage_unexplored: ["Exploratory"],
    report: { available: false, artifact_ref: null },
    budget: {
      base: baseBudget,
      max_llm_requests: 10,
      remaining_llm_requests: 8,
      max_successful_tool_calls: 12,
      remaining_successful_tool_calls: 9,
      max_rows_scanned: 100000,
      rows_scanned: 200,
      max_result_cells: 10000,
      result_cells: 20,
      max_rounds: 4,
      remaining_rounds: 3,
      max_cost_usd: "5.00",
      cost_usd: "1.25",
      remaining_cost_usd: "3.75",
      llm_requests_used: 2,
      successful_tool_calls_used: 3,
      rounds_used: 1,
      amendments: [],
    },
    job: {
      job_id: "job_expl_1",
      execution_session_id: "explsess_1",
      status: "running",
    },
    events_url: "/api/v1/sessions/r1/explorations/expl_1/events",
    ...patch,
  };
}

const preparedDto: ExplorationPreparedDto = {
  exploration_id: "expl_1",
  session_id: "r1",
  project_id: "p1",
  policy: {
    mode: "open",
    goal: null,
    dataset_scope: ["sample"],
    thinking_level: "standard",
    coverage_targets: ["Diagnostic", "Exploratory"],
    budget: baseBudget,
    scoring_policy_version: "score_v1",
    statistical_policy_version: "stats_v1",
    tool_capability_digest: "tools_12345678",
    policy_fingerprint: "policy_12345678",
  },
  data_state_witness: "witness_12345678",
  cost_range: {
    minimum_usd: "0",
    maximum_usd: "5.00",
    basis: "policy_hard_cap",
    exact: false,
  },
  action_hash: "action_12345678",
  approval_token: "approval_12345678",
  expires_at: "2026-08-02T16:00:00Z",
  release_certificate_digest: "release_12345678",
};

describe("E5 exploration API workflow", () => {
  it("keeps the navigation entry fail-closed without a trusted release capability", async () => {
    renderAppWithRouterAt("/projects/p1/sessions/r1/questions");
    await screen.findByRole("heading", { name: "Questions" });
    expect(screen.queryByRole("link", { name: "Explore" })).not.toBeInTheDocument();
  });

  it("shows the navigation entry only when the server advertises exploration availability", async () => {
    server.use(
      http.get("/api/v1/system/capabilities", () => HttpResponse.json({
        pdf_export_available: true,
        pdf_export_hint: "",
        exploration_available: true,
      })),
    );
    renderAppWithRouterAt("/projects/p1/sessions/r1/questions");
    expect(await screen.findByRole("link", { name: "Explore" })).toHaveAttribute(
      "href",
      "/projects/p1/sessions/r1/explorations",
    );
  });

  it("maps every typed snake_case product field without replacing coverage difference", () => {
    const dto = explorationDto({ coverage_unexplored: ["incorrect-server-value"] });
    const run = explorationRunFromDto(dto);
    expect(run.currentHypothesis?.statement).toBe("Revenue differs by region");
    expect(run.currentEvidence[0]?.receiptId).toBe("rcpt_region");
    expect(run.insights[0]?.evidenceLane).toBe("confirmatory");
    expect(run.budget.used.costUsd).toBe(1.25);
    expect(run.report).toEqual({ available: false, artifactRef: null });
    expect(run.coverageProjectionConsistent).toBe(false);
    expect(buildExplorationReportGroups(run).coverageGaps).toEqual(["Exploratory"]);
  });

  it("prepares one-time read-only approval, starts, resumes SSE from Last-Event-ID, and keeps paused resumable", async () => {
    const user = userEvent.setup();
    let current = explorationDto();
    let prepareBody: unknown;
    let startBody: unknown;
    let resumeBody: unknown;
    let pauseBody: unknown;
    let extendBody: unknown;
    let extendKey: string | null = null;
    let cancelBody: unknown;
    server.use(
      http.post("/api/v1/sessions/:sessionId/explorations/prepare", async ({ request }) => {
        prepareBody = await request.json();
        return HttpResponse.json(preparedDto);
      }),
      http.post("/api/v1/sessions/:sessionId/explorations", async ({ request }) => {
        startBody = await request.json();
        return HttpResponse.json({ exploration: current, job: current.job }, { status: 201 });
      }),
      http.get("/api/v1/sessions/:sessionId/explorations/:explorationId", () =>
        HttpResponse.json(current),
      ),
      http.post("/api/v1/sessions/:sessionId/explorations/:explorationId/pause", async ({ request }) => {
        pauseBody = await request.json();
        current = explorationDto({ status: "pause_requested", last_seq: 1 });
        return HttpResponse.json(current);
      }),
      http.post("/api/v1/sessions/:sessionId/explorations/:explorationId/resume", async ({ request }) => {
        resumeBody = await request.json();
        current = explorationDto({ status: "running", last_seq: 3 });
        return HttpResponse.json({ exploration: current, job: current.job }, { status: 201 });
      }),
      http.post("/api/v1/sessions/:sessionId/explorations/:explorationId/extend-budget", async ({ request }) => {
        extendBody = await request.json();
        extendKey = request.headers.get("Idempotency-Key");
        return HttpResponse.json({
          exploration: current,
          amendment: {
            amendment_id: "xamend_1",
            previous_effective_fingerprint: "policy_12345678",
            increase: { max_requests: 2 },
            reason: "One bounded follow-up",
            approved_by: "system:e4b-api",
            created_at: "2026-08-02T12:00:00Z",
          },
          effective_policy_fingerprint: "policy_amended_1",
        });
      }),
      http.post("/api/v1/sessions/:sessionId/explorations/:explorationId/cancel", async ({ request }) => {
        cancelBody = await request.json();
        current = explorationDto({
          status: "stopped",
          stop_reason: "cancelled",
          last_seq: 4,
        });
        return HttpResponse.json(current);
      }),
    );

    const { router } = renderAppWithRouterAt("/projects/p1/sessions/r1/explorations");
    await screen.findByRole("heading", { name: "Read-only exploration" });
    await user.click(screen.getByRole("button", { name: "Review authorization" }));
    expect(await screen.findByRole("heading", { name: "One-time read-only authorization" })).toBeInTheDocument();
    expect(screen.getByText("$0.00–$5.00")).toBeInTheDocument();
    expect(prepareBody).toEqual({
      mode: "open",
      goal: null,
      dataset_ids: ["sample"],
      thinking_level: "standard",
    });

    await user.click(screen.getByRole("button", { name: "Authorize and start" }));
    await waitFor(() => expect(router.state.location.pathname).toBe("/projects/p1/sessions/r1/explorations/expl_1"));
    expect(startBody).toEqual({
      action_hash: "action_12345678",
      approval_token: "approval_12345678",
    });
    await screen.findByText("Revenue differs by region");
    expect(FakeEventSource.latest().url).toBe(
      "/api/v1/sessions/r1/explorations/expl_1/events?last_event_id=expl_1%3A0",
    );

    await user.click(screen.getByRole("button", { name: "Pause" }));
    expect(pauseBody).toEqual({});
    expect(await screen.findByRole("button", { name: "Pause requested…" })).toBeDisabled();

    current = explorationDto({ status: "paused", last_seq: 2 });
    FakeEventSource.latest().emit("paused", {
      event_id: "expl_1:2",
      exploration_id: "expl_1",
      seq: 2,
      type: "paused",
      occurred_at: "2026-08-02T12:00:00Z",
      data: {},
    });
    expect(await screen.findByRole("button", { name: "Resume" })).toBeInTheDocument();
    expect(screen.getByText("Paused · resumable")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Resume" }));
    await waitFor(() => expect(resumeBody).toEqual({}));
    expect(await screen.findByRole("button", { name: "Pause" })).toBeInTheDocument();

    await user.click(screen.getByText("Extend hard caps"));
    await user.clear(screen.getByLabelText("Model requests increase"));
    await user.type(screen.getByLabelText("Model requests increase"), "2");
    await user.type(screen.getByLabelText("Reason"), "One bounded follow-up");
    await user.click(screen.getByRole("button", { name: "Approve additive increase" }));
    await waitFor(() => expect(extendBody).toEqual({
      increase: {
        max_requests: 2,
        max_successful_tool_calls: 0,
        max_rounds: 0,
        max_cost_usd: 0,
      },
      reason: "One bounded follow-up",
    }));
    expect(extendKey).toMatch(/^[0-9a-f-]{36}$/);

    await user.click(screen.getByRole("button", { name: "Cancel…" }));
    await user.click(screen.getByRole("button", { name: "Confirm terminal cancel" }));
    await waitFor(() => expect(cancelBody).toEqual({}));
    expect(await screen.findByText("Stopped · terminal")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Resume" })).not.toBeInTheDocument();
  });

  it("sets the first SSE cursor after an initially-undefined GET resolves", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/explorations/:explorationId", () =>
        HttpResponse.json(explorationDto({ last_seq: 7 })),
      ),
    );
    renderAppWithRouterAt("/projects/p1/sessions/r1/explorations/expl_1");
    await screen.findByText("Revenue differs by region");
    await waitFor(() => expect(FakeEventSource.instances).toHaveLength(1));
    expect(FakeEventSource.latest().url).toBe(
      "/api/v1/sessions/r1/explorations/expl_1/events?last_event_id=expl_1%3A7",
    );
  });

  it("renders a terminal stopped route with typed cost, stop reason, lanes, coverage, and no resume control", async () => {
    const stopped = explorationDto({
      status: "stopped",
      stop_reason: "budget_exhausted",
      report: { available: true, artifact_ref: "report_expl_1" },
    });
    server.use(
      http.get("/api/v1/sessions/:sessionId/explorations/:explorationId", () => HttpResponse.json(stopped)),
      http.get(
        "/api/v1/sessions/:sessionId/explorations/:explorationId/report",
        () => HttpResponse.text("# Exploration report\n\n- exploration_id: expl_1"),
      ),
    );
    renderAppWithRouterAt("/projects/p1/sessions/r1/explorations/expl_1");
    expect(await screen.findByText("Stopped · terminal")).toBeInTheDocument();
    expect(screen.getAllByText("budget_exhausted").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Confirmatory evidence").length).toBeGreaterThan(0);
    expect(screen.getByText("Exploratory")).toBeInTheDocument();
    expect(screen.getAllByText("Cost $1.25 / $5.00 cap")).toHaveLength(2);
    // The report is served as markdown by the run's own endpoint; it has no
    // artifact id, and the old link pointed at one that never existed.
    expect(
      await screen.findByText(/- exploration_id: expl_1/),
    ).toBeInTheDocument();
    expect(screen.queryByText("Open report artifact")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Resume" })).not.toBeInTheDocument();
    expect(document.querySelectorAll("[data-section-id]")).toHaveLength(6);
    expect(FakeEventSource.instances).toHaveLength(0);
  });
});
