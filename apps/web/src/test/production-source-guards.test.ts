import {
  mkdtempSync,
  mkdirSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import path from "node:path";
import { tmpdir } from "node:os";
import { afterEach, describe, expect, it } from "vitest";
import {
  findAttributionSelectorBypasses,
  findScrimBypasses,
  findUnreachableProductionFiles,
} from "./production-source-guards";

const srcDir = path.resolve(import.meta.dirname, "..");
const fixtureRoots: string[] = [];

function fixture(files: Record<string, string>): string {
  const root = mkdtempSync(path.join(tmpdir(), "eda-production-guard-"));
  fixtureRoots.push(root);
  for (const [relative, source] of Object.entries(files)) {
    const target = path.join(root, relative);
    mkdirSync(path.dirname(target), { recursive: true });
    writeFileSync(target, source, "utf8");
  }
  return root;
}

afterEach(() => {
  for (const root of fixtureRoots.splice(0)) {
    rmSync(root, { recursive: true, force: true });
  }
});

describe("production source reachability", () => {
  it("has no unreachable production TypeScript or CSS from main.tsx", () => {
    expect(findUnreachableProductionFiles(srcDir, "main.tsx")).toEqual([]);
  });

  it("follows static imports, dynamic imports, and CSS @imports", () => {
    const root = fixture({
      "main.tsx": `
        import { App } from "./App";
        import("./lazy/Panel");
        void App;
      `,
      "App.tsx": `import "./styles/root.css"; export const App = null;`,
      "lazy/Panel.tsx": `export const Panel = null;`,
      "styles/root.css": `@import "./theme.css";`,
      "styles/theme.css": `:root { --color-bg: white; }`,
      "orphan.ts": `export const orphan = true;`,
      "styles/orphan.css": `.orphan { display: none; }`,
      "types.d.ts": `declare const ambient: string;`,
      "ignored.test.tsx": `export const testOnly = true;`,
      "test/helper.ts": `export const helper = true;`,
    });

    expect(findUnreachableProductionFiles(root, "main.tsx")).toEqual([
      "orphan.ts",
      "styles/orphan.css",
    ]);
  });
});

describe("production scrim boundary", () => {
  it("has no overlay that bypasses the two design-token scrims", () => {
    expect(findScrimBypasses(srcDir)).toEqual([]);
  });

  it("finds utility, arbitrary utility, inline-style, and CSS bypasses", () => {
    const root = fixture({
      "main.tsx": `
        export function Modal() {
          return <>
            <div className="fixed inset-0 bg-black/40" role="dialog" />
            <div className="modal-backdrop fixed inset-0 bg-[#000]/50" />
            <div className="overlay fixed inset-0 bg-[rgb(0_0_0_/_0.35)]" />
            <div
              aria-modal="true"
              style={{ position: "fixed", inset: 0, backgroundColor: "rgba(0,0,0,.45)" }}
            />
            <div
              data-overlay="true"
              style={{ position: "fixed", inset: 0, background: "black" }}
            />
          </>;
        }
      `,
      "overlay.css": `
        .modal-backdrop { position: fixed; inset: 0; background: #0008; }
      `,
      "other-tokens.css": `
        :root { --modal-scrim: rgba(0, 0, 0, 0.6); }
      `,
    });

    expect(findScrimBypasses(root)).toEqual([
      "main.tsx:bg-black/40",
      "main.tsx:bg-[#000]/50",
      "main.tsx:bg-[rgb(0_0_0_/_0.35)]",
      "main.tsx:backgroundColor: \"rgba(0,0,0,.45)\"",
      "main.tsx:background: \"black\"",
      "other-tokens.css::root:--modal-scrim: rgba(0, 0, 0, 0.6)",
      "overlay.css:.modal-backdrop:background: #0008",
    ]);
  });

  it("does not misclassify an intentional black content surface", () => {
    const root = fixture({
      "main.tsx": `
        export const Swatch = () => <div className="bg-black text-white">Black</div>;
      `,
      "content.css": `.ink-swatch { background-color: rgb(0 0 0); }`,
      "design-tokens.css": `
        :root {
          --color-scrim: rgb(0 0 0 / 0.4);
          --color-scrim-strong: rgb(0 0 0 / 0.5);
        }
      `,
    });

    expect(findScrimBypasses(root)).toEqual([]);
  });
});

describe("React Flow attribution boundary", () => {
  it("has no undocumented attribution selector in production CSS or TSX", () => {
    expect(findAttributionSelectorBypasses(srcDir)).toEqual([]);
  });

  it("scans every production CSS and TSX file", () => {
    const root = fixture({
      "main.tsx": `
        export const selector = ".react-flow__attribution";
      `,
      "nested/flow.css": `
        .relationships-flow .react-flow__attribution { color: gray; }
      `,
    });

    expect(findAttributionSelectorBypasses(root)).toEqual([
      "main.tsx:.react-flow__attribution",
      "nested/flow.css:.react-flow__attribution",
    ]);
  });
});
