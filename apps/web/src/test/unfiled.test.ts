import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { UNFILED_PROJECT_ID, isUnfiled, projectLabel } from "../app/unfiled";

/* The bucket id is a string literal on both sides of the API. Nothing in the
 * generated schema carries a *value*, so the only way to stop the two drifting
 * is to read the Python source — the same trick page-layout.test.tsx uses. */
const idsPy = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../../../../eda_platform/src/eda_platform/core/ids.py",
);

describe("unfiled bucket id", () => {
  it("matches INTERNAL_PROJECT_IDS in core/ids.py", () => {
    const source = readFileSync(idsPy, "utf8");
    const pythonId = source.match(
      /UNFILED_PROJECT_ID\s*=\s*"([^"]+)"/,
    )?.[1];
    const declared = source.match(
      /INTERNAL_PROJECT_IDS\s*=\s*frozenset\(\{([^}]*)\}\)/,
    )?.[1];
    expect(pythonId, "UNFILED_PROJECT_ID not found in core/ids.py").toBe(
      UNFILED_PROJECT_ID,
    );
    expect(declared, "INTERNAL_PROJECT_IDS not found in core/ids.py").toBeDefined();
    expect(declared).toMatch(/\bUNFILED_PROJECT_ID\b/);
  });

  it("labels the bucket rather than leaking its id", () => {
    expect(isUnfiled(UNFILED_PROJECT_ID)).toBe(true);
    expect(projectLabel(UNFILED_PROJECT_ID)).not.toContain("unfiled-sessions");
    expect(projectLabel("customer-churn")).toBe("customer-churn");
  });
});
