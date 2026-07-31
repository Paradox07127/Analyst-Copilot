/* Stable visual baselines for the core read-only surfaces. These run against
 * the production bundle and real seeded API; no request is mocked. */

import { expect, test, type Page } from "@playwright/test";
import { readSeed } from "./support/seed";

async function useTheme(page: Page, theme: "light" | "dark"): Promise<void> {
  await page.addInitScript((value) => {
    window.localStorage.setItem("eda.theme", value);
    window.localStorage.setItem("eda.density", "comfortable");
  }, theme);
}

test.describe("core visual regression", () => {
  test.use({ viewport: { width: 1440, height: 1_000 } });

  for (const theme of ["light", "dark"] as const) {
    test(`Data Map is stable in ${theme} theme`, async ({ page }) => {
      const { projectId, sessionId } = readSeed();
      await useTheme(page, theme);
      await page.goto(`/projects/${projectId}/sessions/${sessionId}/data-map`);
      await expect(page.getByRole("heading", { name: "Data Map" })).toBeVisible();

      await expect(page.getByRole("main")).toHaveScreenshot(
        `data-map-${theme}.png`,
        { animations: "disabled", caret: "hide" },
      );
    });
  }

  test("Table Preview is stable at a deep-linked offset", async ({ page }) => {
    const { projectId, sessionId, datasetId } = readSeed();
    await useTheme(page, "light");
    await page.goto(
      `/projects/${projectId}/sessions/${sessionId}/table/${datasetId}?offset=100`,
    );
    await expect(page.getByText("Rows 101–122")).toBeVisible();

    await expect(page.getByRole("main")).toHaveScreenshot(
      "table-preview-offset.png",
      { animations: "disabled", caret: "hide" },
    );
  });

  test("Report is stable in dark theme", async ({ page }) => {
    const { projectId, sessionId } = readSeed();
    await useTheme(page, "dark");
    await page.goto(`/projects/${projectId}/sessions/${sessionId}/report`);
    await expect(page.getByRole("heading", { name: "Executive Summary" }))
      .toBeVisible();

    await expect(page.getByRole("main")).toHaveScreenshot("report-dark.png", {
      animations: "disabled",
      caret: "hide",
      /* Seed time changes on every suite run; the rest of the report must not. */
      mask: [page.getByText(/^Generated /)],
    });
  });
});
