import { afterEach, describe, expect, it } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderAppAt } from "./render";
import { initDensity } from "../app/theme";

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
    // matchMedia stub reports light; first toggle goes dark.
    await user.click(
      screen.getByRole("button", { name: "Switch to dark theme" }),
    );
    expect(document.documentElement.dataset["theme"]).toBe("dark");
    expect(window.localStorage.getItem("eda.theme")).toBe("dark");

    await user.click(
      screen.getByRole("button", { name: "Switch to light theme" }),
    );
    expect(document.documentElement.dataset["theme"]).toBe("light");
    expect(window.localStorage.getItem("eda.theme")).toBe("light");
  });

  it("follows OS theme changes while no explicit choice is stored", async () => {
    let matches = false;
    const listeners = new Set<(ev: MediaQueryListEvent) => void>();
    const original = window.matchMedia;
    window.matchMedia = (query: string) =>
      ({
        get matches() {
          return matches;
        },
        media: query,
        onchange: null,
        addEventListener: (_: string, cb: (ev: MediaQueryListEvent) => void) =>
          listeners.add(cb),
        removeEventListener: (
          _: string,
          cb: (ev: MediaQueryListEvent) => void,
        ) => listeners.delete(cb),
        addListener: () => {},
        removeListener: () => {},
        dispatchEvent: () => false,
      }) as MediaQueryList;

    try {
      const { act } = await import("@testing-library/react");
      renderAppAt("/projects");
      await screen.findByRole("button", { name: "Switch to dark theme" });

      await act(async () => {
        matches = true;
        listeners.forEach((cb) => cb({ matches } as MediaQueryListEvent));
      });
      expect(
        screen.getByRole("button", { name: "Switch to light theme" }),
      ).toBeInTheDocument();
      // An explicit choice must stop the OS from overriding it.
      expect(window.localStorage.getItem("eda.theme")).toBeNull();
    } finally {
      window.matchMedia = original;
    }
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
