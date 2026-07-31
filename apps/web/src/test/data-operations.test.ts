import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  ApiError,
  api,
  type DataOperationStarted,
  type JobStatus,
} from "../api/client";
import {
  runDataOperation,
  dataOperationStorageKey,
} from "../api/data-operations";
import { fetchDatasetDistributions } from "../api/hooks";
import { columnDistributionsView } from "./msw/handlers";

const started: DataOperationStarted = {
  session_id: "source",
  execution_session_id: "dop_1",
  job: {
    job_id: "job_data_1",
    session_id: "dop_1",
    status: "queued",
    events_url: "/api/v1/jobs/job_data_1/events",
  },
};

describe("recoverable data operations", () => {
  beforeEach(() => {
    window.localStorage.clear();
    vi.restoreAllMocks();
  });

  it("persists the queued job and fetches its typed result after settlement", async () => {
    vi.spyOn(api, "getJob").mockResolvedValue({
      job_id: "job_data_1",
      session_id: "dop_1",
      project_id: "p1",
      kind: "custom_chart",
      status: "completed",
      cancel_requested: false,
      created_at: "2026-07-27T00:00:00Z",
      started_at: "2026-07-27T00:00:01Z",
      finished_at: "2026-07-27T00:00:02Z",
      error_code: null,
      error_message: null,
      events_url: started.job.events_url,
    });
    const start = vi.fn().mockResolvedValue(started);
    const getResult = vi.fn().mockResolvedValue({ value: 42 });
    const onStarted = vi.fn();
    const storageKey = dataOperationStorageKey("custom-chart", "source");

    await expect(
      runDataOperation(storageKey, start, getResult, onStarted),
    ).resolves.toEqual({ value: 42 });
    expect(start).toHaveBeenCalledOnce();
    expect(onStarted).toHaveBeenCalledWith(started);
    expect(getResult).toHaveBeenCalledWith("job_data_1", undefined);
    expect(window.localStorage.getItem(storageKey)).toBeNull();
  });

  it("reaches the cleanup path when the started callback throws", async () => {
    // A throwing onStarted must not skip the cleanup path and leave the queued
    // job running with nothing watching it.
    vi.spyOn(api, "getJob").mockResolvedValue({
      job_id: "job_data_1",
      session_id: "dop_1",
      project_id: "p1",
      kind: "custom_chart",
      status: "completed",
      cancel_requested: false,
      created_at: "2026-07-27T00:00:00Z",
      started_at: "2026-07-27T00:00:01Z",
      finished_at: "2026-07-27T00:00:02Z",
      error_code: null,
      error_message: null,
      events_url: started.job.events_url,
    });
    const start = vi.fn().mockResolvedValue(started);
    const getResult = vi.fn();
    const onStarted = vi.fn(() => {
      throw new DOMException("quota", "QuotaExceededError");
    });
    const storageKey = dataOperationStorageKey("custom-chart", "throwing");

    await expect(
      runDataOperation(storageKey, start, getResult, onStarted),
    ).rejects.toThrow("quota");
    // Proof the failure went through the catch rather than escaping the
    // function before any recovery could run.
    expect(api.getJob).toHaveBeenCalledWith("job_data_1");
  });

  it("resumes a persisted job without starting a duplicate", async () => {
    const storageKey = dataOperationStorageKey("dataset-distributions", "source");
    window.localStorage.setItem(storageKey, JSON.stringify(started));
    vi.spyOn(api, "getJob").mockResolvedValue({
      job_id: "job_data_1",
      session_id: "dop_1",
      project_id: "p1",
      kind: "dataset_distributions",
      status: "completed",
      cancel_requested: false,
      created_at: "2026-07-27T00:00:00Z",
      started_at: "2026-07-27T00:00:01Z",
      finished_at: "2026-07-27T00:00:02Z",
      error_code: null,
      error_message: null,
      events_url: started.job.events_url,
    });
    const start = vi.fn();

    await runDataOperation(
      storageKey,
      start,
      async () => ({ recovered: true }),
      vi.fn(),
    );
    expect(start).not.toHaveBeenCalled();
  });

  it("rejects a malformed start response before polling an unreachable job", async () => {
    const getJob = vi.spyOn(api, "getJob");
    await expect(
      runDataOperation(
        dataOperationStorageKey("custom-chart", "source"),
        async () => ({ value: 42 }) as unknown as Promise<DataOperationStarted>,
        async () => ({ unreachable: true }),
        vi.fn(),
      ),
    ).rejects.toMatchObject({ code: "data_operation_contract_invalid" });
    expect(getJob).not.toHaveBeenCalled();
  });
});

describe("dataset distribution scans", () => {
  const scanStarted: DataOperationStarted = {
    session_id: "s1",
    execution_session_id: "dop_1",
    job: {
      job_id: "job_dist_1",
      session_id: "dop_1",
      status: "queued",
      events_url: "/api/v1/jobs/job_dist_1/events",
    },
  };

  function jobRow(status: JobStatus["status"]): JobStatus {
    return {
      job_id: "job_dist_1",
      session_id: "dop_1",
      project_id: "p1",
      kind: "dataset_distributions",
      status,
      cancel_requested: false,
      created_at: "2026-07-30T00:00:00Z",
      started_at: "2026-07-30T00:00:01Z",
      finished_at: "2026-07-30T00:00:02Z",
      error_code: status === "failed" ? "scan_failed" : null,
      error_message: status === "failed" ? "worker died" : null,
      events_url: scanStarted.job.events_url,
    };
  }

  beforeEach(() => {
    window.localStorage.clear();
    vi.restoreAllMocks();
  });

  /* A dataset is immutable inside its run, so the second visit must land on the
   * key the first visit used: the server then replays the finished job instead
   * of scanning the whole table again. */
  it("scans under one stable key so revisiting a dataset replays the finished job", async () => {
    const start = vi
      .spyOn(api, "startDatasetDistributions")
      .mockResolvedValue(scanStarted);
    vi.spyOn(api, "getJob").mockResolvedValue(jobRow("completed"));
    vi.spyOn(api, "getDatasetDistributionsResult").mockResolvedValue(
      columnDistributionsView("s1", "d1"),
    );

    await fetchDatasetDistributions("s1", "d1");
    await fetchDatasetDistributions("s1", "d1");

    expect(start.mock.calls.map((call) => call[2])).toEqual([
      "dataset-distributions:s1:d1",
      "dataset-distributions:s1:d1",
    ]);
  });

  /* Without this the stable key is a trap: one cancelled scan would pin the
   * dataset to a failed job forever, and the column headers could never load. */
  it("retries under a one-off key when the replayed scan had already failed", async () => {
    const start = vi
      .spyOn(api, "startDatasetDistributions")
      .mockResolvedValue(scanStarted);
    vi.spyOn(api, "getJob")
      .mockResolvedValueOnce(jobRow("failed"))
      .mockResolvedValueOnce(jobRow("failed"))
      .mockResolvedValue(jobRow("completed"));
    vi.spyOn(api, "getDatasetDistributionsResult").mockResolvedValue(
      columnDistributionsView("s1", "d1"),
    );

    await expect(fetchDatasetDistributions("s1", "d1")).resolves.toMatchObject({
      dataset_id: "d1",
    });
    expect(start).toHaveBeenCalledTimes(2);
    expect(start.mock.calls[0]?.[2]).toBe("dataset-distributions:s1:d1");
    expect(start.mock.calls[1]?.[2]).not.toBe("dataset-distributions:s1:d1");
  });

  /* The server rejects a second active job on the same run+kind lane with a
   * 409 too. Nothing was queued under the stable key there, so retrying can
   * only lose the same race again. */
  it("does not retry when the start request itself was rejected", async () => {
    const start = vi
      .spyOn(api, "startDatasetDistributions")
      .mockRejectedValue(new ApiError(409, "job_conflict", "lane is busy"));

    await expect(fetchDatasetDistributions("s1", "d1")).rejects.toMatchObject({
      code: "job_conflict",
    });
    expect(start).toHaveBeenCalledOnce();
  });

  it("gives up instead of rescanning when the caller aborted", async () => {
    const controller = new AbortController();
    const start = vi
      .spyOn(api, "startDatasetDistributions")
      .mockResolvedValue(scanStarted);
    vi.spyOn(api, "getJob").mockImplementation(async () => {
      controller.abort(new DOMException("aborted", "AbortError"));
      throw new DOMException("aborted", "AbortError");
    });

    await expect(
      fetchDatasetDistributions("s1", "d1", controller.signal),
    ).rejects.toThrow("aborted");
    expect(start).toHaveBeenCalledOnce();
  });
});
