import { describe, expect, it } from "vitest";
import { act, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { server } from "./msw/server";
import { jobStatus } from "./msw/handlers";
import { FakeEventSource } from "./fake-event-source";
import { renderAppAt } from "./render";

let eventId = 0;
function frame(type: string, name: string) {
  eventId += 1;
  return {
    event_id: eventId,
    job_id: "job_1",
    session_id: "r_new",
    type,
    name,
    timestamp: "2026-07-25T10:00:02Z",
    summary: {},
  };
}

/* Runs the Launchpad flow, opens the Activity panel, then returns its SSE
 * double. */
async function launchTrackedJob() {
  const user = userEvent.setup();
  renderAppAt("/projects/p1/new-session");
  await screen.findByRole("heading", { name: "New session" });
  await user.upload(
    screen.getByLabelText("Data files (.csv)"),
    new File(["id\n1\n"], "orders.csv", { type: "text/csv" }),
  );
  await screen.findByText(/Ready · ds_orders/);
  await user.click(screen.getByRole("button", { name: "Run analysis" }));
  await screen.findByRole("heading", { name: "Data Map" });
  await user.click(screen.getByRole("button", { name: "Open activity" }));
  return { user, source: FakeEventSource.latest() };
}

describe("Activity center with job SSE", () => {
  it("restores a legacy localStorage job with sessionId as its source fallback", async () => {
    let reportReads = 0;
    server.use(
      http.get("/api/v1/jobs/:jobId", ({ params }) =>
        HttpResponse.json(
          jobStatus(String(params["jobId"]), {
            kind: "report_generate",
            session_id: "r1",
          }),
        ),
      ),
      http.get("/api/v1/sessions/:sessionId/report", ({ params }) => {
        reportReads += 1;
        return HttpResponse.json({
          session_id: String(params["sessionId"]),
          status: "validated",
          markdown: "# Restored report",
          generated_at: "2026-07-25T10:00:00Z",
        });
      }),
    );
    window.localStorage.setItem(
      "eda.activity.job",
      JSON.stringify({
        jobId: "job_legacy",
        sessionId: "r1",
        projectId: "p1",
        eventsUrl: "/api/v1/jobs/job_legacy/events",
      }),
    );
    window.localStorage.setItem("eda.layout.drawer-open", "true");

    renderAppAt("/projects/p1/sessions/r1/report");
    expect(await screen.findByText("Restored report")).toBeInTheDocument();
    await waitFor(() => expect(reportReads).toBe(1));

    const source = FakeEventSource.latest();
    act(() =>
      source.emit(
        "job.completed",
        {
          ...frame("job.completed", "job_legacy"),
          job_id: "job_legacy",
          session_id: "r1",
        },
      ),
    );

    await waitFor(() => expect(reportReads).toBe(2));
    expect(
      JSON.parse(window.localStorage.getItem("eda.activity.jobs") ?? "[]"),
    ).toHaveLength(1);
  });

  it("keeps multiple background runs monitored and lets the user inspect each one", async () => {
    server.use(
      http.get("/api/v1/jobs/:jobId", ({ params }) => {
        const jobId = String(params["jobId"]);
        return HttpResponse.json(
          jobStatus(jobId, {
            kind: jobId === "job_a" ? "auto_eda" : "report_generate",
            session_id: jobId === "job_a" ? "r_a" : "r_b",
          }),
        );
      }),
    );
    window.localStorage.setItem(
      "eda.activity.jobs",
      JSON.stringify([
        {
          jobId: "job_a",
          sessionId: "r_a",
          sourceSessionId: "r_a",
          projectId: "p1",
          eventsUrl: "/api/v1/jobs/job_a/events",
        },
        {
          jobId: "job_b",
          sessionId: "r_b",
          sourceSessionId: "r_b",
          projectId: "p1",
          eventsUrl: "/api/v1/jobs/job_b/events",
        },
      ]),
    );
    window.localStorage.setItem("eda.activity.selected-job", "job_a");
    window.localStorage.setItem("eda.layout.activity-open", "true");

    const user = userEvent.setup();
    renderAppAt("/projects/p1/sessions/r1/data-map");
    const drawer = await screen.findByRole("dialog", { name: "Activity" });
    expect(within(drawer).getByText("Agent activity")).toBeInTheDocument();
    expect(within(drawer).getByRole("tab", { name: "Runs" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    expect(
      within(drawer).getByRole("button", { name: "View run job_a" }),
    ).toBeInTheDocument();
    expect(
      within(drawer).getByRole("button", { name: "View run job_b" }),
    ).toBeInTheDocument();

    await waitFor(() => expect(FakeEventSource.instances).toHaveLength(2));
    const sourceA = FakeEventSource.instances.find((source) =>
      source.url.includes("job_a"),
    );
    const sourceB = FakeEventSource.instances.find((source) =>
      source.url.includes("job_b"),
    );
    expect(sourceA).toBeDefined();
    expect(sourceB).toBeDefined();

    act(() => {
      sourceA!.emit("job.started", {
        ...frame("job.started", "job_a"),
        job_id: "job_a",
      });
      sourceB!.emit("job.started", {
        ...frame("job.started", "job_b"),
        job_id: "job_b",
      });
    });
    expect(await within(drawer).findByText("2 active")).toBeInTheDocument();

    /* A non-selected run still settles because every inbox entry owns a live
     * observer, rather than only the row the user happens to be viewing. */
    act(() =>
      sourceA!.emit("job.completed", {
        ...frame("job.completed", "job_a"),
        job_id: "job_a",
      }),
    );
    expect(await screen.findByText("Analysis completed.")).toBeInTheDocument();
    expect(sourceA!.readyState).toBe(FakeEventSource.CLOSED);
    expect(sourceB!.readyState).toBe(FakeEventSource.OPEN);

    await user.click(
      within(drawer).getByRole("button", { name: "View run job_b" }),
    );
    expect(within(drawer).getByText(/Selected job_b/)).toBeInTheDocument();
    await user.click(within(drawer).getByRole("tab", { name: /Event log/ }));
    act(() =>
      sourceB!.emit("tool_completed", {
        ...frame("tool_completed", "render_report"),
        job_id: "job_b",
      }),
    );
    expect(within(drawer).getByRole("log")).toHaveTextContent("render_report");

    await user.click(within(drawer).getByRole("tab", { name: "Runs" }));
    /* job_a settled above, so it now sits in the collapsed history. */
    await user.click(
      within(drawer).getByRole("button", { name: "Show 1 finished run" }),
    );
    await user.click(
      within(
        within(drawer)
          .getByRole("button", { name: "View run job_a" })
          .closest("li")!,
      ).getByRole("button", { name: "Dismiss" }),
    );
    expect(
      within(drawer).queryByRole("button", { name: "View run job_a" }),
    ).not.toBeInTheDocument();
    expect(
      JSON.parse(window.localStorage.getItem("eda.activity.jobs") ?? "[]"),
    ).toHaveLength(1);
  });

  it("shows the tracked run's non-zero quality summary", async () => {
    server.use(
      http.get("/api/v1/sessions/:sessionId/metrics", ({ params }) =>
        HttpResponse.json({
          session_id: String(params["sessionId"]),
          source: "aggregated",
          question_failed: 2,
          coverage_limited: true,
          steps: [],
        }),
      ),
    );

    await launchTrackedJob();

    const drawer = screen.getByRole("dialog", { name: "Activity" });
    expect(
      await within(drawer).findByText("2 non-zero quality signals"),
    ).toBeInTheDocument();
  });

  it("advances the stepper from step events and toasts on completion", async () => {
    const { source } = await launchTrackedJob();

    act(() => source.emit("job.queued", frame("job.queued", "job_1")));
    expect(screen.getByText("Queued")).toBeInTheDocument();

    act(() => source.emit("job.started", frame("job.started", "job_1")));
    expect(screen.getByText("Running")).toBeInTheDocument();

    act(() =>
      source.emit("step_started", frame("step_started", "profile_dataset")),
    );
    expect(screen.getByLabelText("Reading data: active")).toBeInTheDocument();
    expect(screen.getByLabelText("Checking quality: pending")).toBeInTheDocument();

    act(() =>
      source.emit("step_completed", frame("step_completed", "profile_dataset")),
    );
    expect(screen.getByLabelText("Reading data: done")).toBeInTheDocument();
    /* Between two steps nothing is running, and the strip now says so. It used
     * to promote the next un-done stage to active, which is what let it claim
     * a stage was running for minutes at a time. */
    expect(screen.getByLabelText("Checking quality: pending")).toBeInTheDocument();

    act(() => source.emit("job.completed", frame("job.completed", "job_1")));
    expect(screen.getByText("Completed")).toBeInTheDocument();
    expect(await screen.findByText("Analysis completed.")).toBeInTheDocument();
    // Terminal frame ends the stream; the client must not auto-reconnect.
    expect(source.readyState).toBe(FakeEventSource.CLOSED);
  });

  it("uses separate Activity and Event log sections", async () => {
    const { user, source } = await launchTrackedJob();
    const drawer = screen.getByRole("dialog", { name: "Activity" });

    expect(
      within(drawer).getByRole("tab", { name: "Activity" }),
    ).toHaveAttribute("aria-selected", "true");
    expect(within(drawer).queryByRole("log")).toBeNull();

    act(() => source.emit("job.started", frame("job.started", "job_1")));
    await user.click(within(drawer).getByRole("tab", { name: /Event log/ }));

    expect(
      within(drawer).getByRole("tab", { name: /Event log/ }),
    ).toHaveAttribute("aria-selected", "true");
    expect(within(drawer).getByRole("log")).toHaveTextContent("job.started");
  });

  it("shows auto-EDA phase completion in the launcher ring", async () => {
    const { source } = await launchTrackedJob();
    act(() => source.emit("job.started", frame("job.started", "job_1")));
    act(() =>
      source.emit("step_started", frame("step_started", "profile_dataset")),
    );
    act(() =>
      source.emit("step_completed", frame("step_completed", "profile_dataset")),
    );

    expect(
      screen.getByRole("button", {
        name: "Close activity from floating button",
      }),
    ).toHaveTextContent("17% complete");
  });

  it("keeps tracking and settles a job while the panel is closed", async () => {
    const { user, source } = await launchTrackedJob();
    await user.click(
      within(screen.getByRole("dialog", { name: "Activity" })).getByRole(
        "button",
        { name: "Close activity" },
      ),
    );
    expect(
      screen.queryByRole("dialog", { name: "Activity" }),
    ).not.toBeInTheDocument();

    act(() => source.emit("job.completed", frame("job.completed", "job_1")));

    expect(await screen.findByText("Analysis completed.")).toBeInTheDocument();
    expect(source.readyState).toBe(FakeEventSource.CLOSED);
  });

  /* The stage list matches the pipeline stepper, which had already
   * drifted from drivers/auto_eda.py. Two of the three longest-running steps
   * (discover_questions, export_agentic_report) were missing from it, so while
   * either ran the stepper fell through to "first stage not yet done" and
   * advertised a stage that was not running — on the Olist run that is 270s of
   * a 426s run spent pointing at the wrong step. */
  it("names the running stage for steps the old stage list omitted", async () => {
    const { source } = await launchTrackedJob();
    act(() => source.emit("job.started", frame("job.started", "job_1")));

    for (const step of [
      "profile_dataset",
      "scan_quality",
      "build_quality_context",
      "create_chart_specs",
      "create_analysis_tables",
      "build_value_map",
    ]) {
      act(() => source.emit("step_started", frame("step_started", step)));
      act(() => source.emit("step_completed", frame("step_completed", step)));
    }

    act(() =>
      source.emit("step_started", frame("step_started", "discover_questions")),
    );

    /* Exactly one stage may claim to be running, and it must be the one the
     * backend actually started. Before the fix this found "Stats: active" —
     * a step that does not run at all on a dataset with no test targets. */
    const active = screen
      .getAllByRole("listitem")
      .map((node) => node.getAttribute("aria-label") ?? "")
      .filter((label) => label.endsWith(": active"));

    expect(active).toEqual(["Finding questions: active"]);
  });

  /* profile_dataset re-emits step_started once per dataset with an `index` in
   * its summary — nine times on the real nine-table run. The strip used to
   * render one undifferentiated "Reading data" for all nine, so a 20s step
   * looked identical to a stalled one. */
  it("counts items within a step from the summary index", async () => {
    const { source } = await launchTrackedJob();
    act(() => source.emit("job.started", frame("job.started", "job_1")));

    /* Only `index` — that is all core/kernel.py puts on step_started. An
     * earlier version derived the total from the highest index seen, which
     * rendered every update as "N of N", i.e. permanently 100%. */
    const started = (index: number) => ({
      ...frame("step_started", "profile_dataset"),
      summary: { index },
    });

    act(() => source.emit("step_started", started(0)));
    expect(screen.getByText("item 1")).toBeInTheDocument();

    act(() => source.emit("step_started", started(3)));
    expect(screen.getByText("item 4")).toBeInTheDocument();
    expect(
      screen.getByLabelText("Reading data: active, item 4"),
    ).toBeInTheDocument();
    expect(screen.queryByText(/4 of 4/)).toBeNull();

    /* Within the drawer, the plain-language line is the only live region: the
     * strip and the raw log must not also announce or a screen reader gets
     * three streams. Scoped, because LoadingSkeleton is a role="status" too
     * and the page behind the drawer may still be loading. */
    const drawer = screen.getByRole("dialog", { name: "Activity" });
    expect(within(drawer).getAllByRole("status")).toHaveLength(1);
  });

  /* A killed worker leaves step_started with no terminator, so activeStage
   * survived into the terminal job and the strip kept an amber segment with an
   * elapsed counter climbing forever while the header said Failed. */
  it("does not leave a phase running after the job has failed", async () => {
    const { source } = await launchTrackedJob();
    act(() => source.emit("job.started", frame("job.started", "job_1")));
    act(() =>
      source.emit("step_started", frame("step_started", "export_agentic_report")),
    );
    expect(screen.getByLabelText("Writing report: active")).toBeInTheDocument();

    act(() => source.emit("job.failed", frame("job.failed", "job_1")));

    expect(screen.getByLabelText("Writing report: failed")).toBeInTheDocument();
    const drawer = screen.getByRole("dialog", { name: "Activity" });
    expect(
      within(drawer)
        .getAllByRole("listitem")
        .map((n) => n.getAttribute("aria-label") ?? "")
        .filter((l) => l.endsWith(": active")),
    ).toEqual([]);
  });

  it("cancels a running job after confirmation", async () => {
    let cancelCalled = false;
    server.use(
      http.post("/api/v1/jobs/:jobId/cancel", ({ params }) => {
        cancelCalled = true;
        return HttpResponse.json(
          jobStatus(String(params["jobId"]), { cancel_requested: true }),
        );
      }),
    );

    const { user, source } = await launchTrackedJob();
    act(() => source.emit("job.started", frame("job.started", "job_1")));

    /* findBy: the button renders once the job-status fetch reports a
     * cancellable kind. */
    await user.click(await screen.findByRole("button", { name: "Cancel job" }));
    expect(screen.getByText("Cancel this job?")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Confirm cancel" }));
    expect(cancelCalled).toBe(true);

    act(() =>
      source.emit(
        "job.cancel_requested",
        frame("job.cancel_requested", "job_1"),
      ),
    );
    expect(screen.getByText(/cancel requested/)).toBeInTheDocument();

    act(() => source.emit("job.cancelled", frame("job.cancelled", "job_1")));
    expect(screen.getByText("Cancelled")).toBeInTheDocument();
    expect(await screen.findByText("Analysis cancelled.")).toBeInTheDocument();
  });

  /* Opening a finished run's table queues a column scan. That is a page-level
   * read, not a task the user started, and Activity used to file it next to
   * full analyses — complete with a dop_ run id nobody could act on. */
  it("keeps a table preview's column scan out of the runs inbox", async () => {
    window.localStorage.setItem("eda.layout.activity-open", "true");
    renderAppAt("/projects/p1/sessions/r1/table/sample");

    const valueHeader = await screen.findByRole("columnheader", {
      name: "value float64",
    });
    expect(
      await within(valueHeader).findAllByTestId("header-distribution-bin"),
    ).toHaveLength(10);

    const drawer = screen.getByRole("dialog", { name: "Activity" });
    expect(
      within(drawer).getByText("No background runs", { selector: "p" }),
    ).toBeInTheDocument();
    expect(
      JSON.parse(window.localStorage.getItem("eda.activity.jobs") ?? "[]"),
    ).toHaveLength(0);
  });

  it("files a settled run under a collapsed history instead of the live list", async () => {
    server.use(
      http.get("/api/v1/jobs/:jobId", ({ params }) =>
        HttpResponse.json(
          jobStatus(String(params["jobId"]), { kind: "auto_eda" }),
        ),
      ),
    );
    window.localStorage.setItem(
      "eda.activity.jobs",
      JSON.stringify(
        ["job_a", "job_b"].map((jobId) => ({
          jobId,
          sessionId: `r_${jobId}`,
          sourceSessionId: `r_${jobId}`,
          projectId: "p1",
          eventsUrl: `/api/v1/jobs/${jobId}/events`,
        })),
      ),
    );
    window.localStorage.setItem("eda.layout.activity-open", "true");

    const user = userEvent.setup();
    renderAppAt("/projects/p1/sessions/r1/data-map");
    const drawer = await screen.findByRole("dialog", { name: "Activity" });
    await waitFor(() => expect(FakeEventSource.instances).toHaveLength(2));
    const sourceA = FakeEventSource.instances.find((source) =>
      source.url.includes("job_a"),
    )!;
    const sourceB = FakeEventSource.instances.find((source) =>
      source.url.includes("job_b"),
    )!;

    act(() => {
      sourceA.emit("job.started", { ...frame("job.started", "job_a"), job_id: "job_a" });
      sourceB.emit("job.started", { ...frame("job.started", "job_b"), job_id: "job_b" });
    });
    expect(
      within(drawer).getByRole("button", { name: "View run job_b" }),
    ).toBeInTheDocument();

    act(() =>
      sourceB.emit("job.completed", {
        ...frame("job.completed", "job_b"),
        job_id: "job_b",
      }),
    );

    await waitFor(() =>
      expect(
        within(drawer).queryByRole("button", { name: "View run job_b" }),
      ).toBeNull(),
    );
    expect(
      within(drawer).getByRole("button", { name: "View run job_a" }),
    ).toBeInTheDocument();

    await user.click(
      within(drawer).getByRole("button", { name: "Show 1 finished run" }),
    );
    expect(
      within(drawer).getByRole("button", { name: "View run job_b" }),
    ).toBeInTheDocument();

    /* A failed run is terminal too, but folding it away would hide the one
     * thing the user has to act on — and contradict the launcher badge. */
    act(() =>
      sourceA.emit("job.failed", {
        ...frame("job.failed", "job_a"),
        job_id: "job_a",
      }),
    );
    await waitFor(() =>
      expect(within(drawer).getByText("Failed")).toBeInTheDocument(),
    );
    /* Still 1 — the history did not absorb job_a. */
    expect(
      within(drawer).getByRole("button", { name: "Hide 1 finished run" }),
    ).toBeInTheDocument();
    expect(
      within(
        within(drawer).getByRole("list", { name: "Tracked runs" }),
      ).getByRole("button", { name: "View run job_a" }),
    ).toBeInTheDocument();
  });

  /* The badge used to accumulate every event ever streamed, so a finished
   * analysis left "99+" pinned to the launcher for the rest of the session. */
  it("counts runs still in flight on the launcher, not events already seen", async () => {
    const { source } = await launchTrackedJob();

    act(() => source.emit("job.started", frame("job.started", "job_1")));
    expect(screen.getByTestId("activity-run-count")).toHaveTextContent("1");

    act(() =>
      source.emit("step_started", frame("step_started", "profile_dataset")),
    );
    act(() =>
      source.emit("step_completed", frame("step_completed", "profile_dataset")),
    );
    expect(screen.getByTestId("activity-run-count")).toHaveTextContent("1");

    act(() => source.emit("job.completed", frame("job.completed", "job_1")));
    expect(screen.queryByTestId("activity-run-count")).toBeNull();
  });

  it("hides the cancel button for a question_exec job", async () => {
    server.use(
      http.get("/api/v1/jobs/:jobId", ({ params }) =>
        HttpResponse.json(
          jobStatus(String(params["jobId"]), { kind: "question_exec" }),
        ),
      ),
    );

    const { source } = await launchTrackedJob();
    act(() => source.emit("job.started", frame("job.started", "job_1")));

    expect(
      await screen.findByText("Cannot be cancelled mid-run."),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Cancel job" }),
    ).not.toBeInTheDocument();
  });
});
