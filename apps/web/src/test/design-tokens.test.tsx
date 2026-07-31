import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

/* The token layer is plain CSS that jsdom never evaluates, so these assert on
 * the stylesheet source. Runtime resolution is covered by the Playwright pass. */

const stylesDir = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../styles",
);
const read = (file: string) =>
  readFileSync(path.join(stylesDir, file), "utf8").replace(
    /\/\*[\s\S]*?\*\//g,
    "",
  );

const tokensCss = read("design-tokens.css");
const indexCss = read("index.css");
const srcDir = path.resolve(stylesDir, "..");

/** Body of the first block with this header, braces balanced. */
function block(css: string, header: string): string {
  const at = css.indexOf(header);
  if (at === -1) throw new Error(`block not found: ${header}`);
  const open = css.indexOf("{", at + header.length - 1);
  let depth = 0;
  for (let i = open; i < css.length; i += 1) {
    if (css[i] === "{") depth += 1;
    else if (css[i] === "}") {
      depth -= 1;
      if (depth === 0) return css.slice(open + 1, i);
    }
  }
  throw new Error(`unbalanced block: ${header}`);
}

function decls(body: string): Map<string, string> {
  const out = new Map<string, string>();
  for (const raw of body.split(";")) {
    const line = raw.trim();
    if (!line.startsWith("--")) continue;
    const colon = line.indexOf(":");
    out.set(
      line.slice(0, colon).trim(),
      line.slice(colon + 1).replace(/\s+/g, " ").trim(),
    );
  }
  return out;
}

function must(map: Map<string, string>, key: string): string {
  const value = map.get(key);
  if (value === undefined) throw new Error(`missing declaration: ${key}`);
  return value;
}

const rootTheme = decls(block(indexCss, "@theme {"));
const inlineTheme = decls(block(indexCss, "@theme inline {"));
const light = decls(block(tokensCss, ":root {"));
const compact = decls(block(tokensCss, ':root[data-density="compact"] {'));
const darkMedia = decls(block(tokensCss, ':root:not([data-theme="light"]) {'));
const darkAttr = decls(block(tokensCss, ':root[data-theme="dark"] {'));

const rem = (value: string) => Number.parseFloat(value) * 16;

describe("Type scale", () => {
  it("matches Anthropic's published heading tokens", () => {
    // font-heading-lg/xl/2xl/3xl = 20/24/28/36px at 1.25/1.25/1.1/1.
    expect(rem(must(rootTheme, "--text-lg"))).toBe(20);
    expect(must(rootTheme, "--text-lg--line-height")).toBe("1.25");
    expect(rem(must(rootTheme, "--text-xl"))).toBe(24);
    expect(must(rootTheme, "--text-xl--line-height")).toBe("1.25");
    expect(rem(must(rootTheme, "--text-2xl"))).toBe(28);
    expect(must(rootTheme, "--text-2xl--line-height")).toBe("1.1");
    expect(rem(must(rootTheme, "--text-3xl"))).toBe(36);
    expect(must(rootTheme, "--text-3xl--line-height")).toBe("1");
  });

  it("sets body and caption above the old 14/12px baseline", () => {
    expect(rem(must(rootTheme, "--text-sm"))).toBe(15);
    expect(rem(must(rootTheme, "--text-xs"))).toBe(13);
    expect(rem(must(rootTheme, "--text-base"))).toBe(16);
  });

  it("keeps body leading at Anthropic's absolute 22.4px", () => {
    const leading =
      rem(must(rootTheme, "--text-sm")) *
      Number.parseFloat(must(rootTheme, "--text-sm--line-height"));
    expect(leading).toBeCloseTo(22.4, 0);
  });

  it("pairs every size step with an explicit line height", () => {
    const sizes = [...rootTheme.keys()].filter((k) =>
      /^--text-[a-z0-9]+$/.test(k),
    );
    expect(sizes.length).toBeGreaterThanOrEqual(7);
    for (const size of sizes) {
      expect(rootTheme.has(`${size}--line-height`)).toBe(true);
    }
  });
});

describe("Density", () => {
  it("keeps compact at 80% of the comfortable unit", () => {
    const comfortable = Number.parseFloat(must(rootTheme, "--spacing"));
    const tightened = Number.parseFloat(must(compact, "--spacing"));

    expect(must(rootTheme, "--spacing")).toBe("0.275rem");
    expect(must(compact, "--spacing")).toBe("0.22rem");
    expect(tightened).toBeLessThan(comfortable);
    expect(tightened / comfortable).toBeCloseTo(0.8, 2);
  });

  it("leaves --spacing out of @theme inline, which would bake it in", () => {
    // `inline` resolves values at build time; density overrides at runtime.
    expect(inlineTheme.has("--spacing")).toBe(false);
    expect([...inlineTheme.keys()].filter((k) => k.startsWith("--text-"))).toEqual(
      [],
    );
  });
});

describe("Shape and elevation", () => {
  it("adds the Anthropic radius ladder without moving --radius-base", () => {
    expect(rem(must(light, "--radius-base"))).toBe(8);
    expect(rem(must(rootTheme, "--radius-xs"))).toBe(4);
    expect(rem(must(rootTheme, "--radius-sm"))).toBe(6);
    expect(rem(must(rootTheme, "--radius-md"))).toBe(8);
    expect(rem(must(rootTheme, "--radius-lg"))).toBe(10);
    expect(rem(must(rootTheme, "--radius-xl"))).toBe(12);
  });

  it("exposes the elevation tokens as live vars, not baked values", () => {
    for (const token of ["--shadow-card", "--shadow-panel", "--shadow-overlay"]) {
      expect(must(light, token)).toContain("rgb(0 0 0 /");
      expect(must(inlineTheme, token)).toBe(`var(${token})`);
    }
  });

  it("constrains the reading column", () => {
    expect(rem(must(rootTheme, "--container-content"))).toBe(768);
  });
});

describe("Dark theme", () => {
  it("re-declares every elevation token, at a heavier alpha", () => {
    const shadows = [...light.keys()].filter((k) => k.startsWith("--shadow-"));
    expect(shadows.length).toBeGreaterThan(0);
    for (const token of shadows) {
      // A 5%-black shadow is invisible on the dark surface.
      expect(darkAttr.get(token)).toBeDefined();
      expect(darkAttr.get(token)).not.toBe(light.get(token));
    }
  });

  it("keeps the media-query and attribute blocks identical", () => {
    const sorted = (m: Map<string, string>) =>
      [...m.entries()].sort(([a], [b]) => a.localeCompare(b));
    expect(sorted(darkAttr)).toEqual(sorted(darkMedia));
  });
});

/* WCAG relative luminance, sRGB. */
function luminance(hex: string): number {
  const h = hex.replace("#", "");
  const channel = (pair: string) => {
    const c = Number.parseInt(pair, 16) / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  };
  return (
    0.2126 * channel(h.slice(0, 2)) +
    0.7152 * channel(h.slice(2, 4)) +
    0.0722 * channel(h.slice(4, 6))
  );
}

function contrast(a: string, b: string): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x) as [
    number,
    number,
  ];
  return (hi + 0.05) / (lo + 0.05);
}

/* The categorical ramp shipped light-only, so dark mode drew #115e59 on the
 * #1a1917 page at 2.32:1. 3:1 is the WCAG 1.4.11 floor for non-text graphics. */
describe("Chart ramp is legible in both themes", () => {
  const CHART_KEYS = ["--chart-1", "--chart-2", "--chart-3", "--chart-4",
    "--chart-5", "--chart-6", "--chart-7"];

  it("declares the full ramp in both dark blocks", () => {
    for (const key of CHART_KEYS) {
      expect(must(darkAttr, key)).toMatch(/^#[0-9a-f]{6}$/i);
      expect(must(darkMedia, key)).toBe(must(darkAttr, key));
    }
  });

  it.each([
    ["light", () => light, "--color-bg"],
    ["dark", () => darkAttr, "--color-bg"],
  ])("clears 3:1 against the %s page background", (_theme, block, bgKey) => {
    const scope = block();
    const bg = must(scope, bgKey);
    for (const key of CHART_KEYS) {
      expect(contrast(must(scope, key), bg), `${key} on ${bg}`).toBeGreaterThanOrEqual(3);
    }
  });

  it("clears 3:1 against the card surface too, where most charts sit", () => {
    for (const scope of [light, darkAttr]) {
      const surface = must(scope, "--color-surface");
      for (const key of CHART_KEYS) {
        expect(
          contrast(must(scope, key), surface),
          `${key} on ${surface}`,
        ).toBeGreaterThanOrEqual(3);
      }
    }
  });
});

describe("Colour tokens are untouched", () => {
  it("still carries the workbench brand values", () => {
    expect(must(light, "--color-primary")).toBe("#115e59");
    expect(must(light, "--color-bg")).toBe("#fbfaf8");
    expect(must(light, "--color-surface")).toBe("#f3f1ec");
    expect(must(light, "--color-text")).toBe("#1c1b19");
    expect(must(light, "--color-border")).toBe("#e5e1d8");
    expect(must(darkAttr, "--color-primary")).toBe("#2fa79b");
    expect(must(darkAttr, "--color-bg")).toBe("#1a1917");
  });
});

describe("Design-system boundary guards", () => {
  it("routes every production scrim through a live design token", () => {
    expect(must(light, "--color-scrim")).toBe("rgb(0 0 0 / 0.4)");
    expect(must(light, "--color-scrim-strong")).toBe("rgb(0 0 0 / 0.5)");
    expect(must(inlineTheme, "--color-scrim")).toBe(
      "var(--color-scrim)",
    );
    expect(must(inlineTheme, "--color-scrim-strong")).toBe(
      "var(--color-scrim-strong)",
    );
  });

  it("uses React Flow's public attribution variable, not its DOM class", () => {
    const flowCss = readFileSync(
      path.join(srcDir, "features/relationships/relationships-flow.css"),
      "utf8",
    );
    expect(flowCss).toContain("--xy-attribution-background-color:");
    expect(flowCss).not.toContain(".react-flow__attribution");
  });
});
