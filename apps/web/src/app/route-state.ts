import { useCallback } from "react";
import { useSearchParams } from "react-router";

/**
 * Small route-state primitive for shareable page context. Default values are
 * omitted so links stay compact; updates preserve every other page parameter.
 */
export function useRouteSearchParam(
  key: string,
  defaultValue = "",
): readonly [string, (value: string) => void] {
  const [searchParams, setSearchParams] = useSearchParams();
  const value = searchParams.get(key) ?? defaultValue;
  const setValue = useCallback(
    (nextValue: string) => {
      setSearchParams(
        (current) => {
          const next = new URLSearchParams(current);
          if (!nextValue || nextValue === defaultValue) next.delete(key);
          else next.set(key, nextValue);
          return next;
        },
        { replace: true },
      );
    },
    [defaultValue, key, setSearchParams],
  );
  return [value, setValue] as const;
}

/** Atomically updates related route fields so one interaction cannot clobber
 * another search-param update from the same render. */
export function useSetRouteSearchParams() {
  const [, setSearchParams] = useSearchParams();
  return useCallback(
    (updates: Record<string, string>) => {
      setSearchParams(
        (current) => {
          const next = new URLSearchParams(current);
          for (const [key, value] of Object.entries(updates)) {
            if (value) next.set(key, value);
            else next.delete(key);
          }
          return next;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );
}

export function parseCsvParam(value: string): string[] {
  return [...new Set(value.split(",").map((item) => item.trim()).filter(Boolean))];
}

export function serializeCsvParam(values: Iterable<string>): string {
  return [...new Set(values)].sort().join(",");
}
