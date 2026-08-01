import { expect, test } from "@playwright/test";
import { readSeed } from "./support/seed";

test("session split drag uses one bounded preview and highlights only its destination", async ({
  page,
}) => {
  const { projectId, sessionId } = readSeed();
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));

  await page.goto(`/projects/${projectId}/sessions/${sessionId}/data-map`);
  const rail = page.getByRole("complementary", { name: "Sessions" });
  const session = rail
    .locator(
      `a[href="/projects/${encodeURIComponent(projectId)}/sessions/${encodeURIComponent(sessionId)}"]`,
    )
    .first();
  await expect(session).toBeVisible();
  await expect(session).toHaveAttribute("draggable", "false");

  /* Recreate the pathological case from the report: the row owns a very wide
   * transformed marquee layer. A custom overlay must stay bounded regardless
   * of the source DOM's composited width. */
  await session.locator(".transition-transform").last().evaluate((node) => {
    node.textContent = "very-long-dataset-name.csv, ".repeat(250);
  });
  await session.evaluate((node) => {
    (window as typeof window & { __sessionNativeDragStarts?: number })
      .__sessionNativeDragStarts = 0;
    node.addEventListener("dragstart", () => {
      const state = window as typeof window & {
        __sessionNativeDragStarts?: number;
      };
      state.__sessionNativeDragStarts = (state.__sessionNativeDragStarts ?? 0) + 1;
    });
  });

  const sourceBox = await session.boundingBox();
  expect(sourceBox).not.toBeNull();
  await page.mouse.move(sourceBox!.x + 18, sourceBox!.y + 18);
  await page.mouse.down();
  await page.mouse.move(sourceBox!.x + 32, sourceBox!.y + 30, { steps: 3 });

  const preview = page.getByTestId("session-drag-preview");
  await expect(preview).toBeVisible();
  const previewBox = await preview.boundingBox();
  expect(previewBox).not.toBeNull();
  expect(previewBox!.width).toBeLessThanOrEqual(225);
  expect(previewBox!.height).toBeLessThanOrEqual(64);
  await expect(preview.locator(".transition-transform")).toHaveCount(0);

  const left = page.locator('[data-split-drop-side="left"]');
  const right = page.locator('[data-split-drop-side="right"]');
  const rightBox = await right.boundingBox();
  expect(rightBox).not.toBeNull();
  await page.mouse.move(
    rightBox!.x + rightBox!.width / 2,
    rightBox!.y + rightBox!.height / 2,
    { steps: 5 },
  );
  await expect(right).toHaveAttribute("data-drop-active", "true");
  await expect(left).toHaveAttribute("data-drop-active", "false");
  await expect(page.locator('[data-drop-active="true"]')).toHaveCount(1);

  await page.mouse.up();
  await expect(page).toHaveURL(/\/split\?.*active=right/);
  await expect(preview).toHaveCount(0);
  expect(
    await page.evaluate(
      () =>
        (window as typeof window & { __sessionNativeDragStarts?: number })
          .__sessionNativeDragStarts ?? 0,
    ),
  ).toBe(0);
  expect(pageErrors).toEqual([]);
});
