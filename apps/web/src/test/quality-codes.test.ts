import { describe, expect, it } from "vitest";
import {
  qualityCodeLabel,
  qualityCodeMeaning,
  qualityCodeTitle,
} from "../api/quality-codes";

describe("quality code vocabulary", () => {
  it("names a known condition instead of echoing the scanner enum", () => {
    expect(qualityCodeLabel("id_not_unique")).toBe("Ids repeat");
    expect(qualityCodeLabel("empty_column")).toBe("Entirely missing");
    expect(qualityCodeMeaning("duplicate_rows")).toMatch(/counts and totals/);
  });

  it("shows an unmapped code rather than hiding it", () => {
    // A scanner code added upstream must stay visible, not vanish.
    expect(qualityCodeLabel("brand_new_check")).toBe("brand new check");
    expect(qualityCodeMeaning("brand_new_check")).toBeUndefined();
    expect(qualityCodeTitle("brand_new_check")).toBe("brand_new_check");
  });

  it("keeps the code reachable in the title where only the label renders", () => {
    expect(qualityCodeTitle("high_missing")).toContain("high_missing");
    expect(qualityCodeTitle("high_missing")).toContain("Missing often enough");
  });
});
