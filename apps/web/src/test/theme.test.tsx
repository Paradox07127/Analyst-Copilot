import { afterEach, describe, expect, it } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderAppAt } from "./render";
import { getTimeTheme, initDensity } from "../app/theme";

/* setup.ts resets data-theme between tests but not data-density. */
afterEach(() => {
  delete document.documentElement.dataset["density"];
});

describe("Theme switching", () => {
  it("stamps data-theme on <html> and persists the choice", async () => {
    const user = userEvent.setup();
    renderAppAt("/projects");
    await screen.findByRole("heading", { name: "Overview" });

    /* The toggle is named for what it does, not for what is currently set —
     * so its name is the opposite of the active theme. */
    const initial = getTimeTheme();
    const firstTarget = initial === "light" ? "dark" : "light";
    await user.click(
      screen.getByRole("button", { name: `Switch to ${firstTarget} theme` }),
    );
    expect(document.documentElement.dataset["theme"]).toBe(firstTarget);
    expect(window.localStorage.getItem("eda.theme")).toBe(firstTarget);

    const secondTarget = firstTarget === "light" ? "dark" : "light";
    await user.click(
      screen.getByRole("button", { name: `Switch to ${secondTarget} theme` }),
    );
    expect(document.documentElement.dataset["theme"]).toBe(secondTarget);
    expect(window.localStorage.getItem("eda.theme")).toBe(secondTarget);
  });

  it("follows local time while no explicit choice is stored", async () => {
    expect(getTimeTheme(new Date(2026, 7, 1, 6, 59))).toBe("dark");
    expect(getTimeTheme(new Date(2026, 7, 1, 7))).toBe("light");
    expect(getTimeTheme(new Date(2026, 7, 1, 18, 59))).toBe("light");
    expect(getTimeTheme(new Date(2026, 7, 1, 19))).toBe("dark");
  });
});

describe("Density boot-time restore", () => {
  it("re-applies a stored compact preference before first paint", () => {
    expect(document.documentElement.dataset["density"]).toBeUndefined();

    window.localStorage.setItem("eda.density", "compact");
    initDensity();
    expect(document.documentElement.dataset["density"]).toBe("compact");
  });

  it("stays comfortable (no attribute) with nothing stored", () => {
    initDensity();
    expect(document.documentElement.dataset["density"]).toBeUndefined();
  });
});
