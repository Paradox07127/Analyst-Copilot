import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it, vi } from "vitest";
import { screen } from "@testing-library/react";
import { renderAppAt } from "./render";

vi.mock("vega-embed", () => ({
  default: vi.fn(async () => ({ view: { finalize: vi.fn() } })),
}));

const featuresDir = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../features",
);
const source = (file: string) =>
  readFileSync(path.join(featuresDir, file), "utf8");
const sharedDataWorkspace = readFileSync(
  path.resolve(featuresDir, "../components/data-workspace.tsx"),
  "utf8",
);


/* One width for every page.
 *
 * This replaced a three-bucket split (reading column / data column / full
 * bleed) that produced a 576px jump — 768px on Questions against 1344px on
 * Trace — every time you moved between sections. The buckets were also not
 * honest: `FULL_WIDTH` had contained a page capped at max-w-data, and two more
 * pages sat at Tailwind's max-w-4xl with one at an arbitrary max-w-[72rem].
 *
 * 95% gives a visible gutter at any panel width; max-w-data stops the shell
 * before a wide monitor turns running text into 180-character lines. Prose is
 * bounded separately and closer to the text: report-markdown.css caps the
 * report body at 72ch, SectionHeader descriptions carry max-w-content, and the
 * chat bubble sets its own measure. */
const PAGE_ROOTS: string[] = [
  "projects/ProjectListPage.tsx",
  "launchpad/LaunchpadPage.tsx",
  "settings/SettingsPage.tsx",
  "datasets/DataMapPage.tsx",
  "datasets/TablePreviewPage.tsx",
  "insights/QualityPage.tsx",
  "insights/ProfilesPage.tsx",
  "relationships/RelationshipsPage.tsx",
  "questions/QuestionsPage.tsx",
  "findings/FindingsPage.tsx",
  "semantic/KnowledgePage.tsx",
  "cleaning/CleaningPage.tsx",
  "analysis/DeepAnalysisPage.tsx",
  "trace/TracePage.tsx",
  "reports/ReportPage.tsx",
  "artifacts/ArtifactsPage.tsx",
  "compare/ComparePage.tsx",
  "skills/SkillsPage.tsx",
  "chat/ChatPage.tsx",
  "board/BoardPage.tsx",
];

/* A page root is the first className of a `return (` inside the route's
 * `Component`, which is the only thing the width rule governs. Matching any
 * padded flex container instead swept up inner forms and banners — one earlier
 * codemod widened four of them by exactly that mistake. */
function pageRoots(file: string): string[] {
  const src = source(file);
  const start = src.search(/export function Component\(/);
  if (start < 0) throw new Error(`${file}: no exported Component`);
  /* Bounded to Component's own body: it is not the last function in every
   * file, and an unbounded slice picked up returns from the helpers below it. */
  const rest = src.slice(start);
  const next = rest.slice(1).search(/\n(?:export )?function /);
  const body = next < 0 ? rest : rest.slice(0, next + 1);
  return [...body.matchAll(/\n {2}return \(/g)]
    .map((m) => body.slice(m.index, m.index + 500).match(/className="([^"]*)"/))
    .map((m) => m?.[1])
    .filter((cls): cls is string => cls !== undefined);
}

describe("page width adoption", () => {
  it.each(PAGE_ROOTS)("%s uses the shared page width", (file) => {
    const src = source(file);
    if (src.includes("<DataWorkspacePage")) {
      expect(sharedDataWorkspace).toContain("mx-auto");
      expect(sharedDataWorkspace).toContain("w-[95%]");
      expect(sharedDataWorkspace).toContain("max-w-data");
      return;
    }
    const roots = pageRoots(file);
    expect(roots.length, `${file}: no page root found`).toBeGreaterThan(0);
    for (const cls of roots) {
      expect(cls, file).toContain("mx-auto");
      expect(cls, file).toContain("w-[95%]");
      expect(cls, file).toContain("max-w-data");
    }
  });

  /* The point of the rework: one width token, no Tailwind defaults and no
   * arbitrary values, so "aligned" is checkable rather than asserted. */
  it("no page root carries a competing width", () => {
    for (const file of PAGE_ROOTS) {
      if (source(file).includes("<DataWorkspacePage")) continue;
      for (const cls of pageRoots(file)) {
        expect(cls, file).not.toMatch(/max-w-content/);
        expect(cls, file).not.toMatch(/max-w-\[/);
        expect(cls, file).not.toMatch(
          /max-w-(?:xs|sm|md|lg|xl|\dxl|prose|screen-\w+)\b/,
        );
      }
    }
  });

  /* Running text does not inherit the shell's width. */
  it("keeps a reading measure on prose the shell no longer bounds", () => {
    expect(
      readFileSync(
        path.resolve(featuresDir, "reports/report-markdown.css"),
        "utf8",
      ),
    ).toMatch(/max-width:\s*7\dch/);
    expect(source("chat/ChatPage.tsx")).toContain(
      'className="max-w-content whitespace-pre-wrap"',
    );
  });
});


/* Elevation is reserved for things that float over a filled backdrop. Cards
 * share the page background, so a shadow there measured 9/255 max delta
 * against the plain border — invisible. Overlays sit on a scrim and measured
 * 30/255, so they take the token. */
describe("elevation is reserved for overlays", () => {
  it.each([
    ["insights/ProfilesPage.tsx", 'role="dialog"'],
    ["settings/SettingsDialog.tsx", 'role="dialog"'],
    ["board/BoardPage.tsx", "DragOverlay"],
  ])("%s gives its overlay the token shadow", (file, marker) => {
    const src = source(file);
    expect(src).toContain(marker);
    expect(src).toContain("shadow-overlay");
    /* raw Tailwind shadows don't switch alpha for dark mode */
    expect(src).not.toContain("shadow-lg");
  });

  it("leaf cards stay border-only", () => {
    for (const file of [
      "datasets/DataMapPage.tsx",
      "insights/ProfilesPage.tsx",
      "projects/ProjectListPage.tsx",
    ]) {
      expect(source(file)).not.toMatch(/shadow-(card|panel)/);
    }
  });
});

/* jsdom does no layout, so the guard is on the width budget rather than a
 * measured box. Playwright at 1440px put the five fixed columns of the widest
 * profile table at 579px inside a 767px card, leaving 188px for Samples; the
 * old max-w-64 asked for 282px and clipped the column at the card edge. */
const SAMPLES_BUDGET_PX = 188;

function spacingRem(): number {
  const css = readFileSync(
    path.resolve(
      path.dirname(fileURLToPath(import.meta.url)),
      "../styles/index.css",
    ),
    "utf8",
  );
  const match = css.match(/--spacing:\s*([\d.]+)rem/);
  if (!match) throw new Error("--spacing not found in index.css");
  return Number(match[1]);
}

describe("Profiles field table fits its card", () => {
  it("samples column caps within the leftover width budget", async () => {
    renderAppAt("/projects/p1/sessions/r1/profiles");

    /* The cap belongs to the cell, while the text now sits inside a Marquee
     * span within it — so resolve up to the <td> rather than reading the class
     * off whatever element happens to hold the string. */
    const cell = (await screen.findByText("1, 2, 3")).closest("td");
    const cap = cell?.className.match(/max-w-(\d+)/)?.[1];
    expect(cap, "samples cell must carry a max-w-* cap").toBeDefined();

    const capPx = Number(cap) * spacingRem() * 16;
    expect(capPx).toBeLessThanOrEqual(SAMPLES_BUDGET_PX);
  });

  it("keeps the truncated samples reachable", async () => {
    renderAppAt("/projects/p1/sessions/r1/profiles");

    const cell = await screen.findByText("1, 2, 3");
    expect(cell).toHaveAttribute("title", "1, 2, 3");
  });
});
