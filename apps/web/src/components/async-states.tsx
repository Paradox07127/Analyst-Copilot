import { useEffect } from "react";
import { useParams } from "react-router";
import { ApiError } from "../api/client";
import {
  reportHandledClientFailure,
  type FailureOperation,
} from "../api/client-failures";

/** Human-readable text for unknown/Error/ApiError values. */
export function formatUnknownError(
  error: unknown,
  fallback = "Something went wrong",
): string {
  if (error instanceof Error) return error.message;
  if (typeof error === "string" && error.trim()) return error;
  return fallback;
}

export function LoadingSkeleton({
  lines = 3,
  label = "Loading",
}: {
  lines?: number;
  label?: string;
}) {
  return (
    <div role="status" aria-label={label} className="flex flex-col gap-2 p-4">
      {Array.from({ length: lines }, (_, i) => (
        <div
          key={i}
          className="skeleton h-4 rounded-base"
          style={{ width: `${100 - i * 12}%` }}
        />
      ))}
      <span className="sr-only">{label}…</span>
    </div>
  );
}

export function EmptyState({
  title,
  description,
}: {
  title: string;
  description?: string;
}) {
  return (
    /* No shadow: these render as leaf cards on the page background, where the
     * elevation token measured 9/255 against a plain border — invisible, and
     * against the rule that elevation is reserved for overlays. */
    <div className="flex flex-col gap-1 rounded-base border border-border p-4">
      <p className="text-sm font-medium">{title}</p>
      {description && (
        <p className="text-sm text-status-neutral">{description}</p>
      )}
    </div>
  );
}

export function ErrorState({
  error,
  onRetry,
  operation = "render",
}: {
  error: unknown;
  onRetry?: () => void;
  operation?: FailureOperation;
}) {
  const { sessionId } = useParams();
  useEffect(() => {
    reportHandledClientFailure(error, operation, sessionId);
  }, [error, operation, sessionId]);
  const apiError = error instanceof ApiError ? error : null;
  if (apiError?.status === 403) {
    return <ForbiddenState error={apiError} onRetry={onRetry} />;
  }
  const message =
    apiError?.message ??
    (error instanceof Error ? error.message : "Something went wrong");

  return (
    <div
      role="alert"
      className="flex flex-col gap-2 rounded-base border border-status-critical/40 p-4"
    >
      <p className="text-sm font-medium text-status-critical">
        {apiError ? `Request failed (${apiError.code})` : "Request failed"}
      </p>
      <p className="text-sm text-status-neutral">{message}</p>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="self-start rounded-base border border-border px-3 py-1.5 text-sm hover:bg-surface"
        >
          Retry
        </button>
      )}
    </div>
  );
}

export function ForbiddenState({
  error,
  onRetry,
}: {
  error?: unknown;
  onRetry?: () => void;
}) {
  const message =
    error instanceof Error
      ? error.message
      : "You do not have permission to view this resource.";

  return (
    <div
      role="alert"
      aria-label="Access forbidden"
      className="flex flex-col gap-2 rounded-base border border-status-critical/40 p-4"
    >
      <p className="text-sm font-medium text-status-critical">
        Access forbidden
      </p>
      <p className="text-sm text-status-neutral">{message}</p>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="self-start rounded-base border border-border px-3 py-1.5 text-sm hover:bg-surface"
        >
          Retry
        </button>
      )}
    </div>
  );
}

export function PartialState({
  error,
  onRetry,
}: {
  error: unknown;
  onRetry?: () => void;
}) {
  const apiError = error instanceof ApiError ? error : null;
  if (apiError?.status === 403) {
    return <ForbiddenState error={apiError} onRetry={onRetry} />;
  }
  const message =
    apiError?.message ??
    (error instanceof Error
      ? error.message
      : "One part of this page could not be loaded.");

  return (
    <div
      role="status"
      aria-label="Partial data"
      className="flex flex-col gap-2 rounded-base border border-status-warn/40 p-4"
    >
      <p className="text-sm font-medium text-status-warn">
        Some data could not be loaded
      </p>
      <p className="text-sm text-status-neutral">{message}</p>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="self-start rounded-base border border-border px-3 py-1.5 text-sm hover:bg-surface"
        >
          Retry
        </button>
      )}
    </div>
  );
}
