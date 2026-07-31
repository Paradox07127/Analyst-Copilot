export type Theme = "light" | "dark";
export type Density = "comfortable" | "compact";

const STORAGE_KEY = "eda.theme";
const DENSITY_STORAGE_KEY = "eda.density";

function readStoredTheme(): Theme | null {
  const stored = window.localStorage.getItem(STORAGE_KEY);
  return stored === "light" || stored === "dark" ? stored : null;
}

function readStoredDensity(): Density | null {
  const stored = window.localStorage.getItem(DENSITY_STORAGE_KEY);
  return stored === "comfortable" || stored === "compact" ? stored : null;
}

/* Comfortable is the default and needs no attribute (CSS is comfortable by
 * default); only compact stamps data-density, mirroring how clearTheme()
 * below removes data-theme rather than writing "light" explicitly. */
export function getDensity(): Density {
  return readStoredDensity() ?? "comfortable";
}

export function setDensity(density: Density): void {
  window.localStorage.setItem(DENSITY_STORAGE_KEY, density);
  if (density === "compact") {
    document.documentElement.dataset["density"] = "compact";
  } else {
    delete document.documentElement.dataset["density"];
  }
}

/* Re-apply a stored compact choice on boot, before first paint. */
export function initDensity(): void {
  if (getDensity() === "compact") {
    document.documentElement.dataset["density"] = "compact";
  }
}

export function hasStoredTheme(): boolean {
  return readStoredTheme() !== null;
}

export function getEffectiveTheme(): Theme {
  const stored = readStoredTheme();
  if (stored) return stored;
  return window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

export function setTheme(theme: Theme): void {
  window.localStorage.setItem(STORAGE_KEY, theme);
  document.documentElement.dataset["theme"] = theme;
}

/* Drop the explicit choice and hand control back to prefers-color-scheme. */
export function clearTheme(): void {
  window.localStorage.removeItem(STORAGE_KEY);
  delete document.documentElement.dataset["theme"];
}

/* Re-apply an explicit choice on boot; without one, the system preference
 * (prefers-color-scheme media query in design-tokens.css) governs.
 * initDensity() piggybacks here since main.tsx only calls initTheme() before
 * the first render — this keeps both boot-time attributes applied pre-paint
 * from the one existing call site. */
export function initTheme(): void {
  const stored = readStoredTheme();
  if (stored) document.documentElement.dataset["theme"] = stored;
  initDensity();
}
