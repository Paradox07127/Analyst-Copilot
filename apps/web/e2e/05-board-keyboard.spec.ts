/* Case ⑤ (§5.4): every drag must have a keyboard alternative. Tab to a drag
 * handle, Enter to grab, arrows to move, Enter to commit — and the order the
 * server stored must survive a reload. */

import { expect, test, type Locator, type Page } from "@playwright/test";
import { readSeed } from "./support/seed";

const HANDLE_PREFIX = "Move card: ";

async function cardOrder(column: Locator): Promise<string[]> {
  const labels = await column.getByRole("button").evaluateAll((nodes) =>
    nodes.map((node) => node.getAttribute("aria-label") ?? ""),
  );
  return labels
    .filter((label) => label.startsWith(HANDLE_PREFIX))
    .map((label) => label.slice(HANDLE_PREFIX.length));
}

/** Proves the handle is reachable by keyboard alone, not just focusable. */
async function tabTo(page: Page, handle: Locator): Promise<void> {
  await page.evaluate(() => (document.activeElement as HTMLElement)?.blur());
  for (let i = 0; i < 80; i += 1) {
    await page.keyboard.press("Tab");
    if (await handle.evaluate((node) => node === document.activeElement)) return;
  }
  throw new Error("drag handle was never reached by Tab");
}

test("board cards can be reordered with the keyboard and the order persists", async ({
  page,
}) => {
  const { projectId, sessionId } = readSeed();
  await page.goto(`/projects/${projectId}/sessions/${sessionId}/board`);
  await expect(
    page.getByRole("heading", { name: "Investigation board" }),
  ).toBeVisible();

  const leads = page.getByRole("region", { name: "Leads" });
  const picker = page.getByLabel("Add card from");
  await expect(picker.locator("option")).not.toHaveCount(1);
  const values = await picker
    .locator("option")
    .evaluateAll((nodes) =>
      nodes.map((node) => (node as HTMLOptionElement).value).filter(Boolean),
    );
  expect(values.length).toBeGreaterThanOrEqual(2);

  const before = await cardOrder(leads);
  for (const value of values.slice(0, 2)) {
    const saved = page.waitForResponse(
      (response) =>
        response.url().includes("/boards/") &&
        response.request().method() === "PUT",
    );
    await picker.selectOption(value);
    await page.getByRole("button", { name: "Add card" }).click();
    expect((await saved).status()).toBe(200);
  }
  const added = (await cardOrder(leads)).slice(before.length);
  expect(added).toHaveLength(2);
  const [first, second] = added;

  const handle = page.getByRole("button", {
    name: `${HANDLE_PREFIX}${first}`,
    exact: true,
  });
  await tabTo(page, handle);
  await page.keyboard.press("Enter");
  await expect(handle).toHaveAttribute("aria-pressed", "true");

  await page.keyboard.press("ArrowDown");
  expect(await cardOrder(leads)).toEqual([
    ...before,
    second,
    first,
  ]);

  const committed = page.waitForResponse(
    (response) =>
      response.url().includes("/boards/") &&
      response.request().method() === "PUT",
  );
  await page.keyboard.press("Enter");
  expect((await committed).status()).toBe(200);
  await expect(handle).toHaveAttribute("aria-pressed", "false");

  await page.reload();
  await expect(
    page.getByRole("heading", { name: "Investigation board" }),
  ).toBeVisible();
  expect(await cardOrder(page.getByRole("region", { name: "Leads" }))).toEqual([
    ...before,
    second,
    first,
  ]);
});
