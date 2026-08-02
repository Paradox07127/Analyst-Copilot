/* Cleaning slice (§7.5): pick options → server preview + pending approval →
 * confirm → apply consumes the action_hash and forks an auto_eda job on the
 * cleaned version, which the activity drawer then tracks. */

import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router";
import { useMutation } from "@tanstack/react-query";
import {
  api,
  type CleaningApplied,
  type CleaningPreviewResult,
} from "../../api/client";
import { useCleaningLog, useCleaningRaw, useDatasets } from "../../api/hooks";
import {
  approvalGuidance,
  type ApprovalGuidance,
} from "../../api/stale-approval";
import {
  dataOperationStorageKey,
  operationActivity,
  runDataOperation,
} from "../../api/data-operations";
import { useJobActivity } from "../../app/job-activity";
import { sessionSectionPath } from "../../app/paths";
import { useRouteSearchParam } from "../../app/route-state";
import {
  EmptyState,
  ErrorState,
  LoadingSkeleton,
} from "../../components/async-states";
import {
  Badge,
  Button,
  Card,
  Disclosure,
  SectionHeader,
  StepChain,
} from "../../components/ui";
import { useDialogFocus } from "../../components/use-dialog-focus";
import {
  DataWorkspacePage,
  DatasetScopeBar,
} from "../../components/data-workspace";
import { CleaningLogSection } from "./CleaningLogSection";
import { RawPreCleaningSection } from "./RawPreCleaningSection";
import { CLEANING_STAGES } from "./StageChain";

interface CleaningOptions {
  trim_whitespace: boolean;
  drop_duplicate_rows: boolean;
  drop_missing_rows: boolean;
  drop_outlier_rows: boolean;
}

const DEFAULT_OPTIONS: CleaningOptions = {
  trim_whitespace: true,
  drop_duplicate_rows: true,
  drop_missing_rows: false,
  drop_outlier_rows: false,
};

const OPTION_FIELDS: {
  key: keyof CleaningOptions;
  label: string;
  hint: string;
  lossy: boolean;
  deletesRows: boolean;
}[] = [
  {
    key: "trim_whitespace",
    label: "Trim whitespace",
    hint: "Strip leading/trailing spaces in text columns.",
    lossy: false,
    deletesRows: false,
  },
  {
    key: "drop_duplicate_rows",
    label: "Drop duplicate rows",
    hint: "Delete rows that are exact duplicates.",
    /* The server classifies dedupe as lossy (repeated observations can be
     * legitimate data); the chip must match the approval it will require. */
    lossy: true,
    deletesRows: true,
  },
  {
    key: "drop_missing_rows",
    label: "Drop rows with missing values",
    hint: "Delete every row that contains a missing cell.",
    lossy: true,
    deletesRows: true,
  },
  {
    key: "drop_outlier_rows",
    label: "Drop IQR outlier rows",
    hint: "Delete rows outside the IQR fences of numeric columns.",
    lossy: true,
    deletesRows: true,
  },
];

/* Approval lifecycle errors get a guided recovery path, not a raw alert. */
function staleApprovalGuidance(error: unknown): ApprovalGuidance | null {
  return approvalGuidance(error, {
    approval_expired: {
      message: "The preview approval expired.",
      hint: "Run the preview again to request a fresh approval, then apply.",
      cta: "Re-run preview",
    },
    approval_consumed: {
      message: "This preview was already applied.",
      hint:
        "Its cleaned version and analysis session already exist — find the session in " +
        "the activity drawer or the session list. Previewing again prepares " +
        "another, separate cleaned version.",
      cta: "Preview again (creates a new cleaned version)",
    },
  });
}

function LossyBadge() {
  return (
    <Badge tone="warn" caps title="Deletes rows; the original version keeps them">
      Lossy
    </Badge>
  );
}

function PreviewCard({ result }: { result: CleaningPreviewResult }) {
  const preview = result.preview;
  const operations = result.operations ?? [];
  const columnChanges = preview.column_changes ?? [];
  const warnings = preview.warnings ?? [];
  return (
    <Card
      as="section"
      aria-label="Cleanup preview"
      tone="brand"
      className="flex flex-col gap-3 p-4"
    >
      <SectionHeader
        level={2}
        title="Previewed changes"
        description="Nothing has been written yet. This is what applying the recipe would produce."
        actions={
          <Badge tone="info" variant="outline">
            Read-only
          </Badge>
        }
      />
      {/* One line, one sentence: the e2e reads "Rows N → M" off this node. */}
      <p className="tabular text-sm">
        Rows {preview.row_count_before.toLocaleString()} →{" "}
        {preview.row_count_after.toLocaleString()} ·{" "}
        {preview.rows_dropped ?? 0} dropped · {preview.rows_edited ?? 0} edited
      </p>
      <p className="text-sm text-status-neutral">
        If applied, this becomes version v{preview.target_version} of the table.
        The current version stays exactly as it is.
      </p>
      <ul className="flex flex-col gap-1">
        {operations.map((op) => (
          <li
            key={op.transform_id}
            className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5 text-sm"
          >
            <span>{op.description || op.target_column || op.transform_id}</span>
            {op.lossy && <LossyBadge />}
            <span className="font-mono text-xs text-status-neutral">
              {op.type}
            </span>
          </li>
        ))}
      </ul>
      {columnChanges.length > 0 && (
        <p className="text-xs text-status-neutral">
          Changed columns:{" "}
          {columnChanges
            .map((change) => `${change.column} (${change.changed_rows} rows)`)
            .join(", ")}
        </p>
      )}
      {warnings.length > 0 && (
        <p className="text-xs text-status-warn">
          Warnings: {warnings.join(", ")}
        </p>
      )}
    </Card>
  );
}

const PAGE_DESCRIPTION =
  "Review a suggested recipe, preview its impact, then explicitly authorize a cleaned copy and a new analysis run. The current data is never overwritten.";

export function Component() {
  const { projectId = "", sessionId = "" } = useParams();
  const navigate = useNavigate();
  const { startTracking } = useJobActivity();

  const datasets = useDatasets(sessionId);
  const cleaningRaw = useCleaningRaw(sessionId);
  const cleaningLog = useCleaningLog(sessionId);
  const [datasetParam, setDatasetParam] = useRouteSearchParam("dataset");
  const [options, setOptions] = useState<CleaningOptions>(DEFAULT_OPTIONS);
  const [confirming, setConfirming] = useState(false);
  /* One idempotency key per previewed approval: Confirm retries replay the
   * same key (and job), while a fresh preview binds a fresh key. */
  const [applyKey, setApplyKey] = useState("");

  const selectedDatasetId =
    datasets.data?.find((dataset) => dataset.dataset_id === datasetParam)
      ?.dataset_id ??
    datasets.data?.[0]?.dataset_id ??
    "";

  useEffect(() => {
    if (selectedDatasetId && selectedDatasetId !== datasetParam) {
      setDatasetParam(selectedDatasetId);
    }
  }, [datasetParam, selectedDatasetId, setDatasetParam]);

  const preview = useMutation({
    mutationFn: () =>
      runDataOperation(
        dataOperationStorageKey("cleaning-preview", sessionId, selectedDatasetId),
        () =>
          api.previewCleaning(
            sessionId,
            { dataset_id: selectedDatasetId, ...options },
            crypto.randomUUID(),
          ),
        (jobId, signal) => api.getCleaningPreviewResult(jobId, signal),
        (started) => startTracking(operationActivity(started, projectId)),
      ),
    onSuccess: () => {
      setConfirming(false);
      /* A new approval supersedes any stale apply result or error state. */
      apply.reset();
      setApplyKey(crypto.randomUUID());
    },
  });

  const apply = useMutation({
    mutationFn: (approval: { actionHash: string; approvalToken: string }) =>
      runDataOperation(
        dataOperationStorageKey("cleaning-apply", sessionId),
        () =>
          api.applyCleaning(
            sessionId,
            {
              action_hash: approval.actionHash,
              approval_token: approval.approvalToken,
              llm: "env",
            },
            applyKey,
          ),
        (jobId, signal) => api.getCleaningApplyResult(jobId, signal),
        (started) => startTracking(operationActivity(started, projectId)),
      ),
    onSuccess: (applied: CleaningApplied) => {
      startTracking({
        jobId: applied.job.job_id,
        sessionId: applied.new_session_id,
        sourceSessionId: sessionId,
        projectId,
        eventsUrl: applied.job.events_url,
      });
      navigate(sessionSectionPath(projectId, applied.new_session_id, "data-map"));
    },
  });

  /* Any form change invalidates the previewed approval on the client side. */
  const invalidatePreview = () => {
    preview.reset();
    apply.reset();
    setConfirming(false);
  };

  const staleGuidance = staleApprovalGuidance(apply.error);
  const cleaningDialog = useDialogFocus(() => setConfirming(false), confirming);
  const stage = apply.isPending
    ? 3
    : confirming
      ? 2
      : preview.data
        ? 1
        : 0;
  const selectedCount = OPTION_FIELDS.filter(
    (field) => options[field.key],
  ).length;
  const selectedRowDeleteCount = OPTION_FIELDS.filter(
    (field) => field.deletesRows && options[field.key],
  ).length;
  const rawEvidenceCount = cleaningRaw.data
    ? (cleaningRaw.data.profiles?.length ?? 0) +
      (cleaningRaw.data.charts?.length ?? 0) +
      (cleaningRaw.data.previews?.length ?? 0)
    : 0;
  const canPreview = Boolean(selectedDatasetId && selectedCount > 0);

  if (datasets.isPending) {
    return (
      <DataWorkspacePage title="Cleanup" description={PAGE_DESCRIPTION}>
        <LoadingSkeleton lines={4} label="Loading datasets" />
      </DataWorkspacePage>
    );
  }
  if (datasets.isError) {
    return (
      <DataWorkspacePage title="Cleanup" description={PAGE_DESCRIPTION}>
        <ErrorState error={datasets.error} onRetry={() => datasets.refetch()} />
      </DataWorkspacePage>
    );
  }
  if (datasets.data.length === 0) {
    return (
      <DataWorkspacePage title="Cleanup" description={PAGE_DESCRIPTION}>
        <EmptyState
          title="No datasets in this session"
          description="This run has no ingested table to preview. Start another analysis with at least one CSV, then return to Cleanup."
        />
      </DataWorkspacePage>
    );
  }

  return (
    <DataWorkspacePage title="Cleanup" description={PAGE_DESCRIPTION}>
      <DatasetScopeBar
        value={selectedDatasetId}
        onChange={(value) => {
          setDatasetParam(value);
          invalidatePreview();
        }}
        options={datasets.data.map((dataset) => ({
          value: dataset.dataset_id,
          label: dataset.display_name,
        }))}
      />

      <StepChain
        label="Cleanup steps"
        steps={CLEANING_STAGES}
        current={stage}
      />

      <Card className="flex flex-col gap-4 p-4">
        <SectionHeader
          level={2}
          title="Suggested recipe"
          description="Review the starting operations for one table. You can change them before requesting a read-only preview."
          actions={
            <Badge tone="brand" variant="outline">
              Not applied
            </Badge>
          }
        />

        <fieldset className="flex flex-col gap-2">
          <legend className="pb-1 text-sm font-medium">
            Operations
            <span className="ml-2 font-normal text-status-neutral">
              {selectedCount} selected
              {selectedRowDeleteCount > 0
                ? ` · ${selectedRowDeleteCount} can delete rows`
                : " · no row-deleting operation selected"}
            </span>
          </legend>
          <div className="grid gap-2 sm:grid-cols-2">
            {OPTION_FIELDS.map((field) => (
              <div
                key={field.key}
                className="flex min-w-0 flex-col gap-0.5 rounded-base border border-border px-3 py-2"
              >
                {/* Hint is described-by rather than inside the label so the
                 * control's accessible name stays the operation itself. */}
                <label
                  className="flex items-center gap-2 text-sm"
                  htmlFor={`cleaning-${field.key}`}
                >
                  <input
                    id={`cleaning-${field.key}`}
                    type="checkbox"
                    checked={options[field.key]}
                    aria-describedby={`cleaning-${field.key}-hint`}
                    onChange={(event) => {
                      setOptions((current) => ({
                        ...current,
                        [field.key]: event.target.checked,
                      }));
                      invalidatePreview();
                    }}
                  />
                  <span>{field.label}</span>
                  {field.lossy && <LossyBadge />}
                </label>
                <span
                  id={`cleaning-${field.key}-hint`}
                  className="pl-6 text-xs text-status-neutral"
                >
                  {field.hint}
                </span>
              </div>
            ))}
          </div>
        </fieldset>

        <div className="flex flex-col items-start gap-2 sm:flex-row sm:items-center sm:gap-3">
          <Button
            variant={preview.data ? "secondary" : "primary"}
            onClick={() => preview.mutate()}
            disabled={!canPreview || preview.isPending}
          >
            {preview.isPending ? "Previewing…" : "Preview cleaning"}
          </Button>
          <span className="text-xs text-status-neutral">
            {selectedCount === 0
              ? "Select at least one operation to preview."
              : "Reads the current version only. No table version or analysis run is created."}
          </span>
        </div>
      </Card>

      {preview.isError && (
        <ErrorState error={preview.error} onRetry={() => preview.mutate()} />
      )}

      {preview.data && (
        <>
          <PreviewCard result={preview.data} />

          {staleGuidance ? (
            <Card
              tone="warn"
              role="alert"
              className="flex flex-col gap-2 p-3 text-sm"
            >
              <p className="font-medium text-status-warn">
                {staleGuidance.message}
              </p>
              <p className="text-status-neutral">{staleGuidance.hint}</p>
              <Button
                onClick={() => preview.mutate()}
                className="self-start"
              >
                {staleGuidance.cta}
              </Button>
            </Card>
          ) : confirming ? (
            <Card
              ref={cleaningDialog.dialogRef}
              onKeyDown={cleaningDialog.onKeyDown}
              tone="warn"
              role="alertdialog"
              aria-label="Confirm cleaning"
              className="flex flex-col gap-2 p-3 text-sm"
            >
              <p className="font-medium">
                Create the cleaned version and start a derived run?
              </p>
              <p className="text-status-neutral">
                This authorizes version v
                {preview.data.preview.target_version} as a new copy and starts
                analysis on that copy. This session and every earlier table
                version remain unchanged.
              </p>
              <div className="flex flex-wrap items-center gap-2">
                <Button
                  variant="primary"
                  onClick={() =>
                    apply.mutate({
                      actionHash: preview.data.action_hash,
                      approvalToken: preview.data.approval_token,
                    })
                  }
                  disabled={apply.isPending}
                >
                  {apply.isPending ? "Applying…" : "Apply and start run"}
                </Button>
                <Button
                  onClick={() => setConfirming(false)}
                  disabled={apply.isPending}
                >
                  Cancel
                </Button>
              </div>
            </Card>
          ) : (
            <div className="flex flex-col items-start gap-2 sm:flex-row sm:items-center sm:gap-3">
              <Button
                variant="primary"
                onClick={() => setConfirming(true)}
              >
                Review and confirm
              </Button>
              <span className="text-xs text-status-neutral">
                Opens the final authorization step. It does not apply yet.
              </span>
            </div>
          )}
        </>
      )}

      {apply.isError && !staleGuidance && (
        <Card
          tone="critical"
          role="alert"
          className="p-3 text-sm text-status-critical"
        >
          {apply.error instanceof Error
            ? apply.error.message
            : "Failed to apply the cleaning recipe."}
        </Card>
      )}

      <hr className="border-border" />

      {/* This is historical evidence from Launchpad's optional pre-cleaning,
       * not part of the unapproved recipe above. Keep it available without
       * letting a large raw audit compete with the current authorization task. */}
      <Card as="section" tone="quiet" className="flex flex-col gap-3 p-4">
        <SectionHeader
          level={2}
          title="Automatic pre-cleaning audit"
          description="Historical evidence from when this run started. Reviewing it does not change the suggested recipe above."
          actions={
            cleaningRaw.isPending ? (
              <Badge tone="neutral">Checking</Badge>
            ) : cleaningRaw.isError ? (
              <Badge tone="critical">Unavailable</Badge>
            ) : cleaningRaw.data?.precleaning_recorded ? (
              <Badge tone="ok">Recorded</Badge>
            ) : (
              <Badge tone="neutral">Not recorded</Badge>
            )
          }
        />

        {cleaningRaw.isPending && (
          <LoadingSkeleton lines={2} label="Loading pre-cleaning audit status" />
        )}
        {cleaningRaw.isError && (
          <ErrorState
            error={cleaningRaw.error}
            onRetry={() => cleaningRaw.refetch()}
          />
        )}
        {cleaningRaw.data && !cleaningRaw.data.precleaning_recorded && (
          <p className="text-sm text-status-neutral">
            This run did not enable automatic pre-cleaning on Launchpad, so no
            before-cleaning snapshot or automatic recipe log was recorded.
          </p>
        )}
        {cleaningRaw.data?.precleaning_recorded && (
          <div className="flex flex-col gap-2">
            <Disclosure
              summary="Raw snapshot before automatic cleaning"
              meta={`${rawEvidenceCount} recorded item${rawEvidenceCount === 1 ? "" : "s"}`}
            >
              <RawPreCleaningSection raw={cleaningRaw.data} />
            </Disclosure>
            <Disclosure
              summary="Automatic cleaning log"
              meta={
                cleaningLog.isPending
                  ? "Loading"
                  : cleaningLog.isError
                    ? "Unavailable"
                    : `${cleaningLog.data?.recipe_count ?? 0} recipe${cleaningLog.data?.recipe_count === 1 ? "" : "s"}`
              }
            >
              <section
                aria-label="Automatic cleaning log details"
                className="flex flex-col gap-3"
              >
                <p className="text-sm text-status-neutral">
                  Operations that were already performed when this run started,
                  including deleted data and protection triggers.
                </p>
                {cleaningLog.isPending && (
                  <LoadingSkeleton lines={3} label="Loading cleaning log" />
                )}
                {cleaningLog.isError && (
                  <ErrorState
                    error={cleaningLog.error}
                    onRetry={() => cleaningLog.refetch()}
                  />
                )}
                {cleaningLog.data && (
                  <CleaningLogSection log={cleaningLog.data} />
                )}
              </section>
            </Disclosure>
          </div>
        )}
      </Card>
    </DataWorkspacePage>
  );
}
