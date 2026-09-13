/* T13: the memory ceiling is editable and remembered, and a run the resource
 * gate stopped says why and what to do — instead of a bare "limited" badge. */

import { describe, expect, it } from "vitest";
import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import type { SettingsPatch } from "../api/client";
import { server } from "./msw/server";
import { defaultSettings } from "./msw/handlers";
import { FakeEventSource } from "./fake-event-source";
import { renderAppAt } from "./render";

const GIB = 1024 ** 3;

const STOP = {
  schema_version: 1,
  status: "limited",
  reason_codes: ["estimated_working_set_exceeded"],
  dataset_count: 9,
  estimated_working_set_bytes: 2284 * 1024 ** 2,
  max_working_set_bytes: 2 * GIB,
  over_budget_bytes: 236 * 1024 ** 2,
  largest_dataset_name: "olist_geolocation_dataset.csv",
  largest_dataset_bytes: 232 * 1024 ** 2,
  largest_dataset_rows: 1_000_163,
  max_rows_per_dataset: 10_000_000,
  precleaning_enabled: true,
  working_set_without_precleaning_bytes: 1820 * 1024 ** 2,
  disabling_precleaning_would_fit: true,
  suggested_max_working_set_bytes: 2560 * 1024 ** 2,
};

function limitedSessionDetail() {
  server.use(
    http.get("/api/v1/sessions/:sessionId", ({ params }) =>
      HttpResponse.json({
        session_id: String(params["sessionId"]),
        project_id: "p1",
        title: "Olist nine tables",
        status: "limited",
        created_at: "2026-08-25T10:00:00Z",
        updated_at: "2026-08-25T10:00:01Z",
        dataset_names: [],
        artifact_count: 1,
        report_status: "not_generated",
        chat_message_count: 0,
        code_version: "abc123",
        seed: 42,
        source_session_id: null,
        artifact_type_counts: {},
        warnings: [],
        resource_limit: STOP,
      }),
    ),
    http.get("/api/v1/sessions/:sessionId/datasets", () =>
      HttpResponse.json([]),
    ),
  );
}

async function openAnalysisSettings() {
  const user = userEvent.setup();
  renderAppAt("/settings?section=analysis");
  await screen.findByRole("heading", { name: "Settings" });
  return user;
}

describe("Resource limits in Settings", () => {
  it("shows the shipped ceiling in units a person uses, not bytes", async () => {
    await openAnalysisSettings();
    expect(
      await screen.findByLabelText("Memory for one analysis (GB)"),
    ).toHaveValue(2);
    expect(screen.getByLabelText("Largest table (million rows)")).toHaveValue(
      10,
    );
    expect(
      screen.getByText(/Default 2 GB\. Raise it only up to what this machine/),
    ).toBeInTheDocument();
  });

  it("saves a raised ceiling as bytes and remembers it beyond this tab", async () => {
    const patches: SettingsPatch[] = [];
    server.use(
      http.put("/api/v1/settings", async ({ request }) => {
        const patch = (await request.json()) as SettingsPatch;
        patches.push(patch);
        return HttpResponse.json({
          ...defaultSettings(),
          max_working_set_bytes: patch.max_working_set_bytes ?? 2 * GIB,
          max_rows_per_dataset: patch.max_rows_per_dataset ?? 10_000_000,
          source: "session",
        });
      }),
    );
    const user = await openAnalysisSettings();
    const memory = await screen.findByLabelText("Memory for one analysis (GB)");

    fireEvent.change(memory, { target: { value: "6" } });
    expect(memory).toHaveValue(6);
    await user.click(screen.getByRole("button", { name: "Save limits" }));

    await waitFor(() => expect(patches).toHaveLength(1));
    expect(patches[0]?.max_working_set_bytes).toBe(6 * GIB);
    expect(patches[0]?.max_rows_per_dataset).toBe(10_000_000);
    /* sessionStorage dies with the tab; a machine's memory ceiling is not a
     * credential and must survive a restart, or the way out of a stopped run
     * has to be re-typed every time. */
    expect(
      JSON.parse(
        window.localStorage.getItem("eda.settings.resource-limits.v1") ?? "{}",
      ),
    ).toEqual({
      max_working_set_bytes: 6 * GIB,
      max_rows_per_dataset: 10_000_000,
    });
  });

  it("refuses to send a ceiling outside the range the server accepts", async () => {
    let writes = 0;
    server.use(
      http.put("/api/v1/settings", () => {
        writes += 1;
        return HttpResponse.json(defaultSettings());
      }),
    );
    await openAnalysisSettings();
    const memory = await screen.findByLabelText("Memory for one analysis (GB)");

    fireEvent.change(memory, { target: { value: "512" } });

    expect(screen.getByRole("button", { name: "Save limits" })).toBeDisabled();
    expect(
      screen.getByText(/Memory must be between 0\.5 and 64 GB/),
    ).toBeInTheDocument();
    expect(writes).toBe(0);
  });

  it("puts a remembered ceiling back after the server forgets it", async () => {
    window.localStorage.setItem(
      "eda.settings.resource-limits.v1",
      JSON.stringify({
        max_working_set_bytes: 8 * GIB,
        max_rows_per_dataset: 10_000_000,
      }),
    );
    const patches: SettingsPatch[] = [];
    server.use(
      http.put("/api/v1/settings", async ({ request }) => {
        const patch = (await request.json()) as SettingsPatch;
        patches.push(patch);
        return HttpResponse.json({
          ...defaultSettings(),
          max_working_set_bytes: patch.max_working_set_bytes ?? 2 * GIB,
          source: "session",
        });
      }),
    );

    await openAnalysisSettings();
    await waitFor(() => expect(patches).toHaveLength(1));
    expect(patches[0]?.max_working_set_bytes).toBe(8 * GIB);
  });
});

describe("A run the resource gate stopped", () => {
  it("explains the stop and the ways out on the session's data page", async () => {
    limitedSessionDetail();
    renderAppAt("/projects/p1/sessions/r_limited/data-map");

    const notice = await screen.findByRole("status", {
      name: "Why this analysis stopped",
    });
    expect(
      within(notice).getByText(/These 9 tables need about 2\.2 GB of memory/),
    ).toBeInTheDocument();
    expect(
      within(notice).getByText(/one analysis is allowed 2\.0 GB/),
    ).toBeInTheDocument();
    expect(
      within(notice).getByText(
        /olist_geolocation_dataset\.csv is the largest table at about 232 MB/,
      ),
    ).toBeInTheDocument();

    // All three ways out, and the clean one only because it was computed to fit.
    expect(
      within(notice).getByText(/“Clean the data first” switched off/),
    ).toBeInTheDocument();
    expect(
      within(notice).getByText(/Raise the memory limit to 2\.5 GB/),
    ).toBeInTheDocument();
    expect(
      within(notice).getByText(/run fewer tables at a time/),
    ).toBeInTheDocument();
    expect(
      within(notice).getByRole("link", { name: "Open analysis settings" }),
    ).toHaveAttribute("href", "/settings?section=analysis");

    /* The generic "upload data to see datasets" empty state would send the
     * user to fix a problem they do not have. */
    expect(
      screen.queryByText("No datasets in this session"),
    ).not.toBeInTheDocument();
  });

  it("does not offer turning the clean off when that would not help", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId", ({ params }) =>
        HttpResponse.json({
          session_id: String(params["sessionId"]),
          project_id: "p1",
          title: "Too big either way",
          status: "limited",
          created_at: "2026-08-25T10:00:00Z",
          updated_at: "2026-08-25T10:00:01Z",
          dataset_names: [],
          artifact_count: 1,
          report_status: "not_generated",
          chat_message_count: 0,
          artifact_type_counts: {},
          warnings: [],
          resource_limit: {
            ...STOP,
            disabling_precleaning_would_fit: false,
            working_set_without_precleaning_bytes: 3 * GIB,
          },
        }),
      ),
      http.get("/api/v1/sessions/:sessionId/datasets", () =>
        HttpResponse.json([]),
      ),
    );
    renderAppAt("/projects/p1/sessions/r_limited/data-map");

    const notice = await screen.findByRole("status", {
      name: "Why this analysis stopped",
    });
    expect(
      within(notice).queryByText(/“Clean the data first” switched off/),
    ).not.toBeInTheDocument();
    expect(
      within(notice).getByText(/Raise the memory limit to 2\.5 GB/),
    ).toBeInTheDocument();
  });

  it("explains a row-cap stop in rows, not in memory it never ran out of", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId", ({ params }) =>
        HttpResponse.json({
          session_id: String(params["sessionId"]),
          project_id: "p1",
          title: "One very long table",
          status: "limited",
          created_at: "2026-08-25T10:00:00Z",
          updated_at: "2026-08-25T10:00:01Z",
          dataset_names: [],
          artifact_count: 1,
          report_status: "not_generated",
          chat_message_count: 0,
          artifact_type_counts: {},
          warnings: [],
          resource_limit: {
            ...STOP,
            reason_codes: ["row_count_exceeded"],
            dataset_count: 1,
            largest_dataset_name: "events.csv",
            largest_dataset_rows: 42_000_000,
            precleaning_enabled: false,
            disabling_precleaning_would_fit: false,
          },
        }),
      ),
      http.get("/api/v1/sessions/:sessionId/datasets", () =>
        HttpResponse.json([]),
      ),
    );
    renderAppAt("/projects/p1/sessions/r_limited/data-map");

    const notice = await screen.findByRole("status", {
      name: "Why this analysis stopped",
    });
    expect(
      within(notice).getByText(
        /events\.csv has 42,000,000 rows, and one table is allowed 10,000,000/,
      ),
    ).toBeInTheDocument();
    expect(
      within(notice).getByText(
        /Raise the largest-table limit past 42,000,000 rows/,
      ),
    ).toBeInTheDocument();
    expect(
      within(notice).queryByText(/These 1 tables need about/),
    ).not.toBeInTheDocument();
  });

  it("shows the reason on the live job strip, not just a limited badge", async () => {
    const user = userEvent.setup();
    renderAppAt("/projects/p1/new-session");
    await screen.findByRole("heading", { name: "New session" });
    await user.upload(
      screen.getByLabelText("Data files (.csv)"),
      new File(["id\n1\n"], "orders.csv", { type: "text/csv" }),
    );
    await screen.findByRole("checkbox", { name: "Exclude orders.csv" });
    await user.click(screen.getByRole("button", { name: "Run analysis" }));
    await screen.findByRole("heading", { name: "Data Map" });
    await user.click(screen.getByRole("button", { name: "Open activity" }));
    const source = FakeEventSource.latest();
    const drawer = screen.getByRole("dialog", { name: "Activity" });

    act(() =>
      source.emit("job.started", {
        event_id: 1,
        job_id: "job_1",
        session_id: "r_new",
        type: "job.started",
        name: "job_1",
        timestamp: "2026-08-25T10:00:00Z",
        summary: {},
      }),
    );
    act(() =>
      source.emit("job.completed", {
        event_id: 2,
        job_id: "job_1",
        session_id: "r_new",
        type: "job.completed",
        name: "job_1",
        timestamp: "2026-08-25T10:00:01Z",
        summary: { session_status: "limited", resource_limit: STOP },
      }),
    );

    expect(within(drawer).getByText("Resource limited")).toBeInTheDocument();
    expect(
      within(drawer).getByText(/These 9 tables need about 2\.2 GB of memory/),
    ).toBeInTheDocument();
    expect(
      within(drawer).getByText(/“Clean the data first” switched off/),
    ).toBeInTheDocument();
  });
});
