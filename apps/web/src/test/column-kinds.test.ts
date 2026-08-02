import { describe, expect, it } from "vitest";
import { classifyColumn } from "../features/datasets/mini-charts";

/* CSV columns arrive as object/string dtypes whatever they hold, so the
 * profiler's semantic_type is the only signal that separates a date column from
 * free text. Letting the dtype answer first put every parsed column in "text". */
describe("classifyColumn", () => {
  it("prefers the semantic type over a string dtype", () => {
    expect(classifyColumn("object", "datetime")).toBe("temporal");
    expect(classifyColumn("object", "numeric")).toBe("numeric");
    expect(classifyColumn("object", "boolean")).toBe("boolean");
    expect(classifyColumn("object", "id")).toBe("id");
    expect(classifyColumn("string", "categorical")).toBe("text");
  });

  it("still classifies by dtype when no semantic type is known", () => {
    expect(classifyColumn("object")).toBe("text");
    expect(classifyColumn("datetime64[ns]")).toBe("temporal");
    expect(classifyColumn("int64")).toBe("numeric");
    expect(classifyColumn("bool")).toBe("boolean");
  });

  it("falls back to the dtype for an unknown semantic type", () => {
    expect(classifyColumn("float64", "unknown")).toBe("numeric");
    expect(classifyColumn("object", "unknown")).toBe("text");
  });
});
