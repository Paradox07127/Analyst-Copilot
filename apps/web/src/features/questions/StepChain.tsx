/* Step vocabularies for the question routes. The chain component itself lives
 * in components/ui.tsx — Cleaning walks the same kind of staged flow and had
 * duplicated it. */

/** The one-question route: prepare returns an approval token and the confirm
 *  card, execute consumes it. Three user-visible stops, not one button. */
export const RUN_ONE_STEPS = [
  "Approve",
  "Review what will run",
  "Execute",
] as const;

/** Drafting a card from free text — a job too, so it gets the same review stop. */
export const DRAFT_STEPS = ["Write it", "Review", "Draft the card"] as const;
