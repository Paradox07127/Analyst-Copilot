import { describe, expect, it } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { server } from "./msw/server";
import { sampleRows } from "./msw/handlers";
import { renderAppAt } from "./render";

function runSummary(sessionId: string, title: string) {
  return {
    session_id: sessionId,
    project_id: "p1",
    title,
    status: "complete",
    created_at: "2026-07-20T10:00:00Z",
    updated_at: "2026-07-21T10:00:00Z",
    dataset_names: ["sample"],
    artifact_count: 1,
    report_status: "final",
    chat_message_count: 0,
  };
}

describe("Session Rail with real API", () => {
  it("lists project sessions and loads the next cursor page", async () => {
    server.use(
      /* The wildcard below also answers for the no-project bucket; without
       * this the rail's standalone group mirrors the staged sessions. */
      http.get("/api/v1/projects/unfiled-sessions/sessions", () =>
        HttpResponse.json({ items: [], next_cursor: null }),
      ),
      http.get("/api/v1/projects/:projectId/sessions", ({ request }) => {
        const cursor = new URL(request.url).searchParams.get("cursor");
        if (cursor === "c2") {
          return HttpResponse.json({
            items: [runSummary("r3", "Third run")],
            next_cursor: null,
          });
        }
        return HttpResponse.json({
          items: [runSummary("r1", "First run"), runSummary("r2", "Second run")],
          next_cursor: "c2",
        });
      }),
    );

    const user = userEvent.setup();
    renderAppAt("/projects");

    expect(await screen.findByText("First run")).toBeInTheDocument();
    expect(screen.getByText("Second run")).toBeInTheDocument();
    expect(screen.queryByText("Third run")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Load more" }));
    expect(await screen.findByText("Third run")).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Load more" }),
    ).not.toBeInTheDocument();
  });

  it("marks running runs with a badge and links runs to their deep link", async () => {
    server.use(
      /* The wildcard below also answers for the no-project bucket; without
       * this the rail's standalone group mirrors the staged sessions. */
      http.get("/api/v1/projects/unfiled-sessions/sessions", () =>
        HttpResponse.json({ items: [], next_cursor: null }),
      ),
      http.get("/api/v1/projects/:projectId/sessions", () =>
        HttpResponse.json({
          items: [{ ...runSummary("r9", "Live run"), status: "running" }],
          next_cursor: null,
        }),
      ),
    );

    renderAppAt("/projects");
    expect(await screen.findByText("Running")).toBeInTheDocument();
    // ^ anchors to the SessionRail item; the project card link is "Latest: …".
    expect(screen.getByRole("link", { name: /^Live run/ })).toHaveAttribute(
      "href",
      "/projects/p1/sessions/r9",
    );
  });
});

describe("Data Map with real API", () => {
  it("renders SessionDetail KPIs and dataset cards", async () => {
    renderAppAt("/projects/p1/sessions/r1/data-map");

    // The dataset card proves the independent session and datasets requests
    // both settled; metric labels also occur in the Inspector.
    expect(await screen.findByText("sample.csv")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /Table inventory/ })).toHaveTextContent(
      "1 table",
    );
    expect(
      screen.queryByText(
        "Scan shape, field mix and readiness here; open a focused workspace for row or issue details.",
      ),
    ).not.toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: "Preview sample.csv" }),
    ).toHaveAttribute("href", "/projects/p1/sessions/r1/table/sample");
    expect(screen.getByText("250 rows")).toBeInTheDocument();
    expect(screen.getByText("3 columns")).toBeInTheDocument();
    expect(screen.getByText("CSV")).toBeInTheDocument();
    expect(screen.getByText("Columns")).toBeInTheDocument();
    const cleanedTile = screen.getByText("Cleaned tables").parentElement;
    expect(cleanedTile).toHaveTextContent("0");
    expect(screen.getByText("Stored size")).toBeInTheDocument();
    expect(screen.getAllByText("1 KB")).toHaveLength(2);
    expect(screen.queryByText("Original upload")).not.toBeInTheDocument();
    expect(screen.getByText("2 numeric · 1 text")).toBeInTheDocument();
    expect(
      screen.getByRole("img", { name: "Column types: 2 numeric · 1 text" }),
    ).toBeInTheDocument();
    expect(screen.queryByText(/Input snapshot/)).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Columns (3)" })).not.toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "Open relationships" })).not.toBeInTheDocument();
  });

  it("surfaces the EdaHandoff readiness gate and PII count on dataset cards", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/artifacts", ({ request }) => {
        const url = new URL(request.url);
        if (url.searchParams.get("type") !== "EdaHandoff") {
          return HttpResponse.json({ items: [], next_cursor: null });
        }
        return HttpResponse.json({
          items: [
            { artifact_id: "handoff_1", type: "EdaHandoff", created_at: null },
          ],
          next_cursor: null,
        });
      }),
      http.get("/api/v1/sessions/:sessionId/artifacts/handoff_1", () =>
        HttpResponse.json({
          artifact_id: "handoff_1",
          type: "EdaHandoff",
          project_id: "p1",
          session_id: "r1",
          created_at: "2026-07-31T12:00:00Z",
          payload: {
            datasets: [
              {
                dataset_id: "sample",
                analysis_ready: false,
                quality: { material_codes: ["empty_dataset", "id_not_unique"] },
                pii_columns: { customer_email: "email" },
              },
            ],
          },
          warnings: [],
        }),
      ),
    );

    renderAppAt("/projects/p1/sessions/r1/data-map");

    expect(await screen.findByText("limited for analysis")).toBeInTheDocument();
    expect(screen.getByText("1 PII column")).toBeInTheDocument();
    expect(screen.getByText("limited for analysis")).toHaveAttribute(
      "title",
      expect.stringContaining("empty_dataset, id_not_unique"),
    );
  });

  it("marks cleaned datasets as edited instead of calling them original uploads", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/datasets", () =>
        HttpResponse.json([
          {
            dataset_id: "cleaned",
            project_id: "p1",
            display_name: "cleaned-sample.csv",
            original_uri: "projects/p1/cleaned/cleaned-sample.csv",
            format: "csv",
            content_hash: "deadbeef",
            byte_size: 1024,
            row_count: 250,
            schema: [
              { name: "id", dtype: "int64" },
              { name: "name", dtype: "string" },
            ],
            ingest_status: "ready",
          },
        ]),
      ),
    );

    renderAppAt("/projects/p1/sessions/r1/data-map");
    expect(await screen.findByText("cleaned-sample.csv")).toBeInTheDocument();
    expect(screen.getByText("edited")).toBeInTheDocument();
    const cleanedTile = screen.getByText("Cleaned tables").parentElement;
    expect(cleanedTile).toHaveTextContent("1");
    expect(screen.queryByText("Original upload")).not.toBeInTheDocument();
    expect(screen.getByText("1 numeric · 1 text")).toBeInTheDocument();
  });

  it("keeps unknown row counts unknown instead of summing them as zero", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/datasets", () =>
        HttpResponse.json([
          {
            dataset_id: "sample",
            project_id: "p1",
            display_name: "sample.csv",
            original_uri: "upload://sample.csv",
            format: "csv",
            content_hash: "deadbeef",
            byte_size: 1024,
            row_count: null,
            schema: [],
            ingest_status: "ready",
          },
        ]),
      ),
    );

    renderAppAt("/projects/p1/sessions/r1/data-map");
    const tile = (await screen.findByText("Rows")).parentElement;
    expect(tile).toHaveTextContent("—");
    expect(tile).toHaveTextContent("Available after every table is profiled");
  });

  it("shows selected inputs while a new session is still producing dataset profiles", async () => {
    window.localStorage.setItem(
      "eda.activity.job",
      JSON.stringify({
        jobId: "job_live",
        sessionId: "r1",
        sourceSessionId: "r1",
        projectId: "p1",
        eventsUrl: "/api/v1/jobs/job_live/events",
        inputDatasets: [{ datasetId: "ds_orders", displayName: "orders.csv" }],
      }),
    );
    server.use(
      http.get("/api/v1/sessions/:sessionId", ({ params }) =>
        HttpResponse.json({
          ...runSummary(String(params.sessionId), "Live run"),
          status: "running",
          warnings: [],
        }),
      ),
      http.get("/api/v1/sessions/:sessionId/datasets", () =>
        HttpResponse.json([]),
      ),
      http.get("/api/v1/sessions/:sessionId/quality", ({ params }) =>
        HttpResponse.json({
          session_id: String(params.sessionId),
          critical: 0,
          warn: 0,
          info: 0,
          datasets: [],
          issues: [],
        }),
      ),
    );

    renderAppAt("/projects/p1/sessions/r1/data-map");
    expect(await screen.findByText("Preparing data")).toBeInTheDocument();
    expect(screen.getByText("orders.csv")).toBeInTheDocument();
    expect(screen.queryByText("No datasets in this session")).not.toBeInTheDocument();
  });

  it("does not mark quality as clean while its session is still running", async () => {
    window.localStorage.setItem(
      "eda.activity.job",
      JSON.stringify({
        jobId: "job_live",
        sessionId: "r1",
        sourceSessionId: "r1",
        projectId: "p1",
        eventsUrl: "/api/v1/jobs/job_live/events",
      }),
    );
    server.use(
      http.get("/api/v1/sessions/:sessionId", ({ params }) =>
        HttpResponse.json({
          ...runSummary(String(params.sessionId), "Live run"),
          status: "running",
          warnings: [],
        }),
      ),
      http.get("/api/v1/sessions/:sessionId/quality", ({ params }) =>
        HttpResponse.json({
          session_id: String(params.sessionId),
          critical: 0,
          warn: 0,
          info: 0,
          datasets: [],
          issues: [],
        }),
      ),
    );

    renderAppAt("/projects/p1/sessions/r1/data-map");
    expect(await screen.findByText("quality pending")).toBeInTheDocument();
    expect(screen.queryByText("no issues")).not.toBeInTheDocument();
  });

  it("shows a typed error with retry when SessionDetail fails", async () => {
    let failures = 0;
    server.use(
      http.get("/api/v1/sessions/:sessionId", () => {
        failures += 1;
        return HttpResponse.json(
          { error: { code: "session_not_found", message: "Session r1 is missing" } },
          { status: 404 },
        );
      }),
    );

    const user = userEvent.setup();
    renderAppAt("/projects/p1/sessions/r1/data-map");

    expect(
      await screen.findByText("Request failed (session_not_found)"),
    ).toBeInTheDocument();
    expect(screen.getByText("Session r1 is missing")).toBeInTheDocument();

    server.resetHandlers();
    await user.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByText("sample.csv")).toBeInTheDocument();
    expect(failures).toBe(1);
  });

  it("keeps successful datasets visible when the parallel quality request fails", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/quality", () =>
        HttpResponse.json(
          {
            error: {
              code: "quality_unavailable",
              message: "Quality scoring is temporarily unavailable",
            },
          },
          { status: 503 },
        ),
      ),
    );

    renderAppAt("/projects/p1/sessions/r1/data-map");

    expect(await screen.findByText("sample.csv")).toBeInTheDocument();
    const partial = await screen.findByRole("status", {
      name: "Partial data",
    });
    expect(partial).toHaveTextContent("Some data could not be loaded");
    expect(partial).toHaveTextContent(
      "Quality scoring is temporarily unavailable",
    );
    expect(screen.getByText("quality unavailable")).toBeInTheDocument();
    expect(screen.queryByText("no issues")).not.toBeInTheDocument();
    expect(
      screen.queryByText("No datasets in this session"),
    ).not.toBeInTheDocument();
  });

  it("renders quality 403 as forbidden without claiming the dataset has no issues", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/quality", () =>
        HttpResponse.json(
          {
            error: {
              code: "quality_forbidden",
              message: "Quality evidence is restricted",
            },
          },
          { status: 403 },
        ),
      ),
    );

    renderAppAt("/projects/p1/sessions/r1/data-map");

    expect(await screen.findByText("sample.csv")).toBeInTheDocument();
    expect(
      await screen.findByRole("alert", { name: "Access forbidden" }),
    ).toHaveTextContent("Quality evidence is restricted");
    expect(screen.getByText("quality unavailable")).toBeInTheDocument();
    expect(screen.queryByText("no issues")).not.toBeInTheDocument();
  });
});

describe("Table Preview with real API", () => {
  it("renders schema dtypes and pages with Prev/Next", async () => {
    const user = userEvent.setup();
    renderAppAt("/projects/p1/sessions/r1/table/sample");

    expect(await screen.findByText("row-0")).toBeInTheDocument();
    expect(screen.getByText("int64")).toBeInTheDocument();
    expect(
      screen.getByText("Loaded page · 3 of 3 columns"),
    ).toBeInTheDocument();
    expect(screen.getByText(/Rows 1–100/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Prev" })).toBeDisabled();

    await user.click(screen.getByRole("button", { name: "Next" }));
    expect(await screen.findByText("row-100")).toBeInTheDocument();
    expect(screen.getByText(/Rows 101–200/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Prev" })).toBeEnabled();

    await user.click(screen.getByRole("button", { name: "Next" }));
    // 250 total rows: the last page has 50 rows and no further page.
    expect(await screen.findByText("row-200")).toBeInTheDocument();
    expect(screen.getByText(/Rows 201–250/)).toBeInTheDocument();
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Next" })).toBeDisabled(),
    );

    await user.click(screen.getByRole("button", { name: "Prev" }));
    expect(await screen.findByText("row-100")).toBeInTheDocument();
  });

  it("shows a typed error state when the preview fails", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/datasets/:datasetId/preview", () =>
        HttpResponse.json(
          { error: { code: "dataset_not_found", message: "No such dataset" } },
          { status: 404 },
        ),
      ),
    );

    renderAppAt("/projects/p1/sessions/r1/table/nope");
    expect(
      await screen.findByText("Request failed (dataset_not_found)"),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });

  it("keeps preview rows visible and reports schema failure as partial data", async () => {
    server.use(
      http.get(
        "/api/v1/sessions/:sessionId/datasets/:datasetId/schema",
        () =>
          HttpResponse.json(
            {
              error: {
                code: "schema_unavailable",
                message: "Schema metadata is temporarily unavailable",
              },
            },
            { status: 503 },
          ),
      ),
    );

    renderAppAt("/projects/p1/sessions/r1/table/sample");

    expect(await screen.findByText("row-0")).toBeInTheDocument();
    expect(
      await screen.findByRole("status", { name: "Partial data" }),
    ).toHaveTextContent("Schema metadata is temporarily unavailable");
    expect(screen.queryByText("There is no data to preview")).not.toBeInTheDocument();
  });

  it("renders preview 403 as forbidden rather than an empty table", async () => {
    server.use(
      http.get(
        "/api/v1/sessions/:sessionId/datasets/:datasetId/preview",
        () =>
          HttpResponse.json(
            {
              error: {
                code: "preview_forbidden",
                message: "Table rows are restricted",
              },
            },
            { status: 403 },
          ),
      ),
    );

    renderAppAt("/projects/p1/sessions/r1/table/sample");

    expect(
      await screen.findByRole("alert", { name: "Access forbidden" }),
    ).toHaveTextContent("Table rows are restricted");
    expect(screen.queryByText("There is no data to preview")).not.toBeInTheDocument();
  });

  it("keeps sampleRows fixtures consistent with pagination math", () => {
    expect(sampleRows(200, 100)).toHaveLength(50);
    expect(sampleRows(0, 100)[0]?.[1]).toBe("row-0");
  });
});
