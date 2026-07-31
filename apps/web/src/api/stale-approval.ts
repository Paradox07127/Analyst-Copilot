/* Map approval/job conflict API codes to domain-specific recovery copy.
 * Call sites keep their own wording; this only removes the shared
 * instanceof/code-switch boilerplate. */

import { ApiError } from "./client";

export type ApprovalGuidance = {
  message: string;
  hint: string;
  cta?: string;
};

/** Structured guidance (message + hint, optional CTA). */
export function approvalGuidance(
  error: unknown,
  byCode: Partial<Record<string, ApprovalGuidance>>,
): ApprovalGuidance | null {
  if (!(error instanceof ApiError)) return null;
  return byCode[error.code] ?? null;
}

/** Single-string guidance (used where the UI has one alert line). */
export function approvalGuidanceText(
  error: unknown,
  byCode: Partial<Record<string, string>>,
): string | null {
  if (!(error instanceof ApiError)) return null;
  return byCode[error.code] ?? null;
}
