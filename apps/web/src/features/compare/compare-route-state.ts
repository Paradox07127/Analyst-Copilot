export const COMPARE_SCOPES = [
  "overview",
  "questions",
  "analysis",
  "findings",
  "report",
  "artifacts",
  "execution",
] as const;

export const SPLIT_SECTIONS = [
  "overview",
  "questions",
  "deep-analysis",
  "findings",
  "report",
  "artifacts",
  "trace",
  "chat",
] as const;

export type CompareMode = "compare" | "split";
export type CompareScope = (typeof COMPARE_SCOPES)[number];
export type CompareFilter = "all" | "differences";
export type SplitSection = (typeof SPLIT_SECTIONS)[number];

export interface CompareRouteState {
  left: string;
  right: string;
  mode: CompareMode;
  scope: CompareScope;
  filter: CompareFilter;
  leftSection: SplitSection;
  rightSection: SplitSection;
}

const DEFAULT_STATE: CompareRouteState = {
  left: "",
  right: "",
  mode: "compare",
  scope: "overview",
  filter: "all",
  leftSection: "questions",
  rightSection: "report",
};

function oneOf<T extends string>(
  value: string | null,
  values: readonly T[],
  fallback: T,
): T {
  return value && values.includes(value as T) ? (value as T) : fallback;
}

export function readCompareRouteState(params: URLSearchParams): CompareRouteState {
  return {
    left: params.get("left") ?? "",
    right: params.get("right") ?? "",
    mode: oneOf(params.get("mode"), ["compare", "split"], DEFAULT_STATE.mode),
    scope: oneOf(params.get("scope"), COMPARE_SCOPES, DEFAULT_STATE.scope),
    filter: oneOf(
      params.get("filter"),
      ["all", "differences"],
      DEFAULT_STATE.filter,
    ),
    leftSection: oneOf(
      params.get("leftSection"),
      SPLIT_SECTIONS,
      DEFAULT_STATE.leftSection,
    ),
    rightSection: oneOf(
      params.get("rightSection"),
      SPLIT_SECTIONS,
      DEFAULT_STATE.rightSection,
    ),
  };
}

export function writeCompareRouteState(
  state: CompareRouteState,
  source?: URLSearchParams,
): URLSearchParams {
  const params = new URLSearchParams(source);
  if (state.left) params.set("left", state.left);
  else params.delete("left");
  if (state.right) params.set("right", state.right);
  else params.delete("right");
  params.set("mode", state.mode);
  params.set("scope", state.scope);
  params.set("filter", state.filter);
  params.set("leftSection", state.leftSection);
  params.set("rightSection", state.rightSection);
  return params;
}

export function swapCompareRouteState(state: CompareRouteState): CompareRouteState {
  return {
    ...state,
    left: state.right,
    right: state.left,
    leftSection: state.rightSection,
    rightSection: state.leftSection,
  };
}
