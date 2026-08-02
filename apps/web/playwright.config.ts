/* Browser E2E against the real stack (§15.2): `scripts/serve.py` serves the
 * production build and the API from one origin, so these specs exercise the
 * same SPA-fallback + /api path the shipped app uses. Nothing is mocked. */

import { execFileSync } from "node:child_process";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, devices } from "@playwright/test";
import { PROJECT_ID } from "./e2e/support/seed";

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const EXTERNAL_BASE_URL = process.env.E2E_EXTERNAL_BASE_URL;
const EXTERNAL_HOST_HEADER = process.env.E2E_EXTERNAL_HOST_HEADER;

/* Runner and workers each load this file, so the picked port and workspace are
 * published through the environment the workers inherit — recomputing them per
 * process would point the specs at a server that does not exist.
 *
 * A throwaway port keeps a stray dev server on :8000 from being mistaken for
 * the app under test, and a throwaway workspace keeps these specs (which
 * upload files, start jobs and write board state) off the real one. */
const firstLoad = !EXTERNAL_BASE_URL && !process.env.E2E_PORT;
process.env.E2E_PORT ??= String(4300 + Math.floor(Math.random() * 600));
process.env.E2E_WORKSPACE ??= mkdtempSync(path.join(tmpdir(), "eda-e2e-"));

const PORT = Number(process.env.E2E_PORT);
const WORKSPACE = process.env.E2E_WORKSPACE;

/* Register the shared E2E project before the server or any spec starts. The
 * product create flow is covered separately; direct setup keeps this seed deterministic. */
if (firstLoad) {
  execFileSync(
    "uv",
    [
      "run",
      "python",
      "-c",
      "import sys;from eda_platform.core.store import ArtifactStore;"
      + "ArtifactStore(sys.argv[1]).ensure_project(sys.argv[2], 'E2E project')",
      WORKSPACE,
      PROJECT_ID,
    ],
    { cwd: REPO_ROOT, stdio: "inherit" },
  );
}

export default defineConfig({
  testDir: "./e2e",
  outputDir: "./e2e/.output",
  /* One worker: the specs share one workspace and one job runner. */
  workers: 1,
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  timeout: 120_000,
  expect: { timeout: 15_000 },
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : "list",
  use: {
    baseURL: EXTERNAL_BASE_URL ?? `http://127.0.0.1:${PORT}`,
    extraHTTPHeaders: EXTERNAL_HOST_HEADER
      ? { Host: EXTERNAL_HOST_HEADER }
      : undefined,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [
    { name: "seed", testMatch: /seed\.setup\.ts/ },
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
      dependencies: ["seed"],
    },
    {
      name: "firefox",
      testMatch: /08-session-split-drag\.spec\.ts/,
      use: { ...devices["Desktop Firefox"] },
      dependencies: ["seed"],
    },
    {
      name: "webkit",
      testMatch: /08-session-split-drag\.spec\.ts/,
      use: { ...devices["Desktop Safari"] },
      dependencies: ["seed"],
    },
  ],
  webServer: EXTERNAL_BASE_URL
    ? undefined
    : {
        command: `uv run python scripts/serve.py --port ${PORT} --workspace ${WORKSPACE}`,
        cwd: REPO_ROOT,
        url: `http://127.0.0.1:${PORT}/api/v1/projects`,
        reuseExistingServer: false,
        timeout: 120_000,
        stdout: "pipe",
        stderr: "pipe",
        env: {
          EDA_WORKSPACE: WORKSPACE,
          /* Offline provider only: E2E must never spend money or need a key. */
          EDA_LLM_PROVIDER: "offline",
          EDA_LLM_API_KEY: "",
        },
      },
});
