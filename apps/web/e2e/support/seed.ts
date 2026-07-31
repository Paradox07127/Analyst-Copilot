/* Ids produced by the seed project and consumed by the specs that need a
 * finished run. Side-effect free: the Playwright config imports it too. */

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

/* apps/web is an ESM package, so Playwright loads these as ES modules and
 * `__dirname` does not exist. */
const HERE = path.dirname(fileURLToPath(import.meta.url));

export const PROJECT_ID = "e2e";
export const FIXTURE_CSV = path.resolve(HERE, "../fixtures/e2e_sales.csv");

/** Rows in the fixture, excluding the header and including 2 exact duplicates. */
export const FIXTURE_ROW_COUNT = 122;
export const FIXTURE_ROWS_AFTER_DEDUPE = 120;

const SEED_FILE = path.resolve(HERE, "../.state/seed.json");

export interface Seed {
  projectId: string;
  sessionId: string;
  datasetId: string;
}

export function writeSeed(seed: Seed): void {
  mkdirSync(path.dirname(SEED_FILE), { recursive: true });
  writeFileSync(SEED_FILE, JSON.stringify(seed, null, 2), "utf-8");
}

export function readSeed(): Seed {
  return JSON.parse(readFileSync(SEED_FILE, "utf-8")) as Seed;
}
