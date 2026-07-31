/* Stop vocabulary for the preview -> apply flow. The chain component itself
 * lives in components/ui.tsx; Questions walks the same kind of staged flow. */

/** Showing the confirm and derived-run stops before review is what stops the
 *  review action reading as irreversible: it opens a dialog, then forks. */
export const CLEANING_STAGES = [
  "Suggested recipe",
  "Preview changes",
  "Confirm apply",
  "Derived run",
] as const;
