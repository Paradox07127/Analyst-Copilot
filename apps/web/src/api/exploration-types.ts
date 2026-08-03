/* Exploration aliases over the checked-in OpenAPI contract.  Keep the few
 * product-friendly names here, but never hand-copy backend DTO fields. */

import type { components } from "./generated/schema";

type Schemas = components["schemas"];

export type ExplorationPrepareRequest = Schemas["ExplorationPrepareRequest"];
export type ExplorationPreparedDto = Schemas["ExplorationPrepared"];
export type ExplorationStartedDto = Schemas["ExplorationStarted"];
export type ExplorationViewDto = Schemas["ExplorationView"];
export type ExplorationBudgetExtendedDto =
  Schemas["ExplorationBudgetExtended"];
export type ExplorationTierDto = ExplorationViewDto["thinking_level"];

/* StreamingResponse payloads are not represented in OpenAPI. This is the one
 * hand-authored transport envelope; its nested data remains opaque. */
export interface ExplorationEventDto {
  event_id: string;
  exploration_id: string;
  seq: number;
  type: string;
  occurred_at: string;
  data: Record<string, unknown>;
}

/* Pydantic supplies zero defaults for omitted additive dimensions.  The
 * generated input model currently marks those defaulted properties required,
 * so express the actual PATCH-like wire contract explicitly as Partial. */
export type ExplorationBudgetIncrease = Partial<
  Schemas["BudgetCapIncrease-Input"]
>;
