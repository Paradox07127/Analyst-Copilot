import { describe, expect, it, vi, beforeEach } from "vitest";
import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { server } from "./msw/server";
import {
  queueDataOperation,
  queueFailedDataOperation,
} from "./msw/handlers";
import { renderAppAt, renderAppWithRouterAt } from "./render";

/* vega-embed does real DOM measurement/canvas work jsdom cannot do; the mock
 * just records the spec so the raw-chart test can assert VegaChart was used
 * (same pattern as insights.test.tsx). */
const embedCalls: Array<Record<string, unknown>> = [];
vi.mock("vega-embed", () => ({
  default: vi.fn(async (_el: HTMLElement, spec: Record<string, unknown>) => {
    embedCalls.push(spec);
    return { view: { finalize: vi.fn() } };
  }),
}));

beforeEach(() => {
  embedCalls.length = 0;
});

const PAGE_PATH = "/projects/p1/sessions/r1/cleaning";

describe("Cleaning page", () => {
  it("keeps dataset scope in the URL and blocks an empty recipe", async () => {
    const user = userEvent.setup();
    const { router } = renderAppWithRouterAt(PAGE_PATH);

    const dataset = await screen.findByLabelText("Dataset");
    expect(dataset).toHaveValue("sample");
    await waitFor(() =>
      expect(new URLSearchParams(router.state.location.search).get("dataset")).toBe(
        "sample",
      ),
    );

    await user.click(screen.getByLabelText(/Trim whitespace/));
    await user.click(screen.getByLabelText(/Drop duplicate rows/));
    expect(screen.getByText(/0 selected/)).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Preview cleaning" }),
    ).toBeDisabled();
    expect(
      screen.getByText("Select at least one operation to preview."),
    ).toBeInTheDocument();
  });

  it("previews a recipe and renders the diff, operations, and lossy badges", async () => {
    let previewBody: Record<string, unknown> | null = null;
    server.use(
      http.post(
        "/api/v1/sessions/:sessionId/cleaning/preview",
        async ({ request, params }) => {
          previewBody = (await request.json()) as Record<string, unknown>;
          return queueDataOperation(
            String(params["sessionId"]),
            "job_preview_override",
            {
            session_id: String(params["sessionId"]),
            dataset_id: "sample",
            action_hash: "b".repeat(64),
            approval_token: "d".repeat(32),
            expires_at: "2026-07-25T12:00:00Z",
            operations: [
              {
                transform_id: "dedupe",
                type: "drop_duplicate_rows",
                target_column: null,
                description: "Remove exact duplicate rows.",
                lossy: false,
              },
              {
                transform_id: "drop_missing",
                type: "drop_missing_rows",
                target_column: null,
                description: "Drop every row that contains a missing value.",
                lossy: true,
              },
            ],
            preview: {
              dataset_id: "sample",
              recipe_id: "api_sample",
              source_version: 1,
              target_version: 2,
              row_count_before: 250,
              row_count_after: 240,
              rows_dropped: 10,
              rows_edited: 4,
              cells_changed: 4,
              column_changes: [],
              warnings: ["constant_column:flag"],
            },
            },
          );
        },
      ),
    );

    const user = userEvent.setup();
    renderAppAt(PAGE_PATH);

    await screen.findByRole("button", { name: "Preview cleaning" });
    expect(screen.getByText(/2 selected · 1 can delete rows/)).toBeInTheDocument();
    await user.click(
      screen.getByLabelText(/Drop rows with missing values/),
    );
    expect(screen.getByText(/3 selected · 2 can delete rows/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Preview cleaning" }));

    expect(
      await screen.findByText(/Rows 250 → 240 · 10 dropped · 4 edited/),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Remove exact duplicate rows."),
    ).toBeInTheDocument();
    expect(screen.getAllByText("Lossy").length).toBeGreaterThan(0);
    expect(screen.getByText(/constant_column:flag/)).toBeInTheDocument();
    expect(
      screen.getByText(/If applied, this becomes version v2/),
    ).toBeInTheDocument();
    expect(previewBody).toMatchObject({
      dataset_id: "sample",
      trim_whitespace: true,
      drop_duplicate_rows: true,
      drop_missing_rows: true,
      drop_outlier_rows: false,
    });
  });

  /* The authorization stops are drawn before the user commits, so the review
   * action cannot read as the point of no return. */
  it("shows the whole preview → confirm → new run sequence up front", async () => {
    const user = userEvent.setup();
    renderAppAt(PAGE_PATH);

    await screen.findByRole("button", { name: "Preview cleaning" });
    const stages = screen.getByRole("list", { name: "Cleanup steps" });
    expect(within(stages).getAllByRole("listitem")).toHaveLength(4);
    expect(within(stages).getByText("Confirm apply")).toBeInTheDocument();
    expect(
      within(stages).getByText("Derived run"),
    ).toBeInTheDocument();
    expect(within(stages).getByText("Suggested recipe")).toHaveAttribute(
      "aria-current",
      "step",
    );

    await user.click(screen.getByRole("button", { name: "Preview cleaning" }));
    await screen.findByRole("button", { name: "Review and confirm" });
    expect(within(stages).getByText("Preview changes")).toHaveAttribute(
      "aria-current",
      "step",
    );

    await user.click(
      screen.getByRole("button", { name: "Review and confirm" }),
    );
    expect(within(stages).getByText("Confirm apply")).toHaveAttribute(
      "aria-current",
      "step",
    );
    expect(
      screen.getByText(/This session and every earlier table version remain unchanged/),
    ).toBeInTheDocument();
  });

  it("applies after confirmation, tracks the fork job, and navigates to the new run", async () => {
    let idempotencyKey: string | null = null;
    let applyBody: Record<string, unknown> | null = null;
    server.use(
      http.post(
        "/api/v1/sessions/:sessionId/cleaning/apply",
        async ({ request, params }) => {
          idempotencyKey = request.headers.get("Idempotency-Key");
          applyBody = (await request.json()) as Record<string, unknown>;
          return queueDataOperation(
            String(params["sessionId"]),
            "job_apply_override",
            {
              session_id: String(params["sessionId"]),
              new_session_id: "run_cleaned_1",
              dataset_id: "sample",
              target_version: 2,
              job: {
                job_id: "job_clean_1",
                session_id: "run_cleaned_1",
                status: "queued",
                events_url: "/api/v1/jobs/job_clean_1/events",
              },
            },
          );
        },
      ),
    );

    const user = userEvent.setup();
    renderAppAt(PAGE_PATH);

    await screen.findByRole("button", { name: "Preview cleaning" });
    await user.click(screen.getByRole("button", { name: "Preview cleaning" }));
    await user.click(
      await screen.findByRole("button", { name: "Review and confirm" }),
    );
    // Two-step confirmation before the destructive-ish action fires.
    await user.click(
      await screen.findByRole("button", { name: "Apply and start run" }),
    );

    expect(
      await screen.findByRole("heading", { name: "Data Map" }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Tracking job_clean_1 · session run_cleaned_1/),
    ).toBeInTheDocument();
    expect(idempotencyKey).toMatch(/^[0-9a-f-]{36}$/);
    expect(applyBody).toMatchObject({
      action_hash: "a".repeat(64),
      approval_token: "c".repeat(32),
    });
  });

  it("reuses the idempotency key on Confirm retries and rotates it per preview", async () => {
    const idempotencyKeys: (string | null)[] = [];
    server.use(
      http.post("/api/v1/sessions/:sessionId/cleaning/apply", ({ request }) => {
        idempotencyKeys.push(request.headers.get("Idempotency-Key"));
        return queueFailedDataOperation(
          "r1",
          "job_apply_failure",
          "internal_error",
          "boom",
        );
      }),
    );

    const user = userEvent.setup();
    renderAppAt(PAGE_PATH);

    await screen.findByRole("button", { name: "Preview cleaning" });
    await user.click(screen.getByRole("button", { name: "Preview cleaning" }));
    await user.click(
      await screen.findByRole("button", { name: "Review and confirm" }),
    );
    await user.click(
      await screen.findByRole("button", { name: "Apply and start run" }),
    );
    await screen.findByText("boom");
    // Retry with the same previewed approval: the key must not change.
    await user.click(
      screen.getByRole("button", { name: "Apply and start run" }),
    );
    await screen.findByText("boom");
    expect(idempotencyKeys).toHaveLength(2);
    expect(idempotencyKeys[0]).toMatch(/^[0-9a-f-]{36}$/);
    expect(idempotencyKeys[1]).toBe(idempotencyKeys[0]);

    // A fresh preview clears the stale error and binds a fresh key.
    await user.click(screen.getByRole("button", { name: "Preview cleaning" }));
    await user.click(
      await screen.findByRole("button", { name: "Review and confirm" }),
    );
    expect(screen.queryByText("boom")).not.toBeInTheDocument();
    await user.click(
      await screen.findByRole("button", { name: "Apply and start run" }),
    );
    await screen.findByText("boom");
    expect(idempotencyKeys).toHaveLength(3);
    expect(idempotencyKeys[2]).toMatch(/^[0-9a-f-]{36}$/);
    expect(idempotencyKeys[2]).not.toBe(idempotencyKeys[0]);
  });

  async function applyAndFailWith(status: number, code: string) {
    server.use(
      http.post("/api/v1/sessions/:sessionId/cleaning/apply", ({ params }) =>
        queueFailedDataOperation(
          String(params["sessionId"]),
          `job_apply_${status}_${code}`,
          code,
          "stale approval",
        ),
      ),
    );

    const user = userEvent.setup();
    renderAppAt(PAGE_PATH);

    await screen.findByRole("button", { name: "Preview cleaning" });
    await user.click(screen.getByRole("button", { name: "Preview cleaning" }));
    await user.click(
      await screen.findByRole("button", { name: "Review and confirm" }),
    );
    await user.click(
      await screen.findByRole("button", { name: "Apply and start run" }),
    );
  }

  it("guides the user back to preview on 410 approval_expired", async () => {
    await applyAndFailWith(410, "approval_expired");

    expect(
      await screen.findByText("The preview approval expired."),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Run the preview again to request a fresh approval/),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Re-run preview" }),
    ).toBeInTheDocument();
  });

  it("explains an already-applied preview on 409 approval_consumed", async () => {
    await applyAndFailWith(409, "approval_consumed");

    expect(
      await screen.findByText("This preview was already applied."),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/cleaned version and analysis session already exist/),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", {
        name: "Preview again (creates a new cleaned version)",
      }),
    ).toBeInTheDocument();
    /* The ambiguous old CTA must be gone from the consumed branch. */
    expect(
      screen.queryByRole("button", { name: "Re-run preview" }),
    ).not.toBeInTheDocument();
  });

  it("shows the empty state when the run has no datasets", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/datasets", () => HttpResponse.json([])),
    );
    renderAppAt(PAGE_PATH);
    expect(
      await screen.findByText("No datasets in this session"),
    ).toBeInTheDocument();
  });

  it("renders the raw pre-cleaning profiles, chart, and data preview", async () => {
    const user = userEvent.setup();
    renderAppAt(PAGE_PATH);

    const rawAudit = await screen.findByRole("button", {
      name: /Raw snapshot before automatic cleaning/,
    });
    expect(rawAudit).toHaveAttribute("aria-expanded", "false");
    expect(
      screen.queryByRole("heading", { name: "Raw profiles" }),
    ).not.toBeInTheDocument();
    await user.click(rawAudit);

    expect(
      await screen.findByRole("heading", { name: "Raw profiles" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "sample.csv" }),
    ).toBeInTheDocument();

    expect(
      screen.getByRole("heading", { name: "Raw charts" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Raw value distribution")).toBeInTheDocument();
    expect(
      screen.getByRole("img", { name: "Raw value distribution" }),
    ).toBeInTheDocument();
    expect(embedCalls).toHaveLength(1);
    expect(embedCalls[0]?.["mark"]).toBe("bar");

    expect(
      screen.getByRole("heading", { name: "Raw data preview" }),
    ).toBeInTheDocument();
    const previewTable = screen.getByRole("table", {
      name: "Raw data preview: sample",
    });
    expect(within(previewTable).getByText("notes")).toBeInTheDocument();
    expect(within(previewTable).getByText("row-1")).toBeInTheDocument();
    expect(within(previewTable).getByText("row-2")).toBeInTheDocument();
    // Header row + 2 data rows.
    expect(within(previewTable).getAllByRole("row")).toHaveLength(3);
  });

  it("renders the four cleaning log tables with matching titles and rows", async () => {
    const user = userEvent.setup();
    renderAppAt(PAGE_PATH);

    const logAudit = await screen.findByRole("button", {
      name: /Automatic cleaning log/,
    });
    expect(logAudit).toHaveAttribute("aria-expanded", "false");
    await user.click(logAudit);

    const summaryTable = screen.getByRole("table", { name: "Summary" });
    expect(within(summaryTable).getAllByRole("row")).toHaveLength(2);
    expect(within(summaryTable).getByText("recipe_1")).toBeInTheDocument();
    expect(within(summaryTable).getByText("244")).toBeInTheDocument();

    const deletedTable = screen.getByRole("table", { name: "Deleted data" });
    expect(within(deletedTable).getAllByRole("row")).toHaveLength(3);
    expect(
      within(deletedTable).getByText("drop_duplicate_rows"),
    ).toBeInTheDocument();
    expect(within(deletedTable).getByText("notes")).toBeInTheDocument();

    const guardrailTable = screen.getByRole("table", {
      name: "Protection triggers",
    });
    expect(within(guardrailTable).getAllByRole("row")).toHaveLength(2);
    expect(within(guardrailTable).getByText("row_loss_ratio")).toBeInTheDocument();

    const suggestionTable = screen.getByRole("table", { name: "Suggestions" });
    expect(within(suggestionTable).getAllByRole("row")).toHaveLength(2);
    expect(
      within(suggestionTable).getByText("Consider normalising the region column."),
    ).toBeInTheDocument();
  });

  it("shows the pre-cleaning-disabled empty state and skips the raw/log tables", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/cleaning/raw", ({ params }) =>
        HttpResponse.json({
          session_id: String(params["sessionId"]),
          precleaning_recorded: false,
          profiles: [],
          charts: [],
          previews: [],
        }),
      ),
    );
    renderAppAt(PAGE_PATH);

    expect(
      await screen.findByText(/did not enable automatic pre-cleaning on Launchpad/),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("heading", { name: "Raw profiles" }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /Automatic cleaning log/ }),
    ).not.toBeInTheDocument();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
  });

  it("distinguishes 'no recipe at all' from 'recipe ran, nothing deleted'", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/cleaning/log", ({ params }) =>
        HttpResponse.json({
          session_id: String(params["sessionId"]),
          recipe_count: 0,
          summary: [],
          deleted_data: [],
          protection_triggers: [],
          suggestions: [],
        }),
      ),
    );
    const user = userEvent.setup();
    renderAppAt(PAGE_PATH);

    await user.click(
      await screen.findByRole("button", { name: /Automatic cleaning log/ }),
    );

    expect(
      await screen.findByText("No cleaning recipe was recorded for this session."),
    ).toBeInTheDocument();
    expect(
      screen.queryByText("No rows or columns were deleted."),
    ).not.toBeInTheDocument();
  });

  it("shows 'no rows or columns were deleted' when a recipe ran but deleted nothing", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/cleaning/log", ({ params }) =>
        HttpResponse.json({
          session_id: String(params["sessionId"]),
          recipe_count: 1,
          summary: [
            {
              dataset: "sample",
              recipe_id: "recipe_1",
              rows_before: 250,
              rows_after: 250,
              rows_removed: 0,
              columns_before: 4,
              columns_after: 4,
              columns_removed: 0,
              delete_steps: 0,
              protection_triggers: 0,
              requires_approval: false,
            },
          ],
          deleted_data: [],
          protection_triggers: [],
          suggestions: [],
        }),
      ),
    );
    const user = userEvent.setup();
    renderAppAt(PAGE_PATH);

    await user.click(
      await screen.findByRole("button", { name: /Automatic cleaning log/ }),
    );

    expect(
      await screen.findByText("No rows or columns were deleted."),
    ).toBeInTheDocument();
    expect(
      screen.queryByText("No cleaning recipe was recorded for this session."),
    ).not.toBeInTheDocument();
  });
});
