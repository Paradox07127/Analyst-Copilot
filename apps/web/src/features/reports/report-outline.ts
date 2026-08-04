/* Reads the structure the exporter itself writes — headings, the
 * `Status: … · Gate: …` line, the Claim Ledger table — so the reader chrome
 * can navigate and cite. The prose is never rewritten: number formatting in
 * the body is the exporter's job, not this file's. */

const STATUS_LINE =
  /^Status:\s*([A-Za-z][\w-]*)(?:\s*·\s*Gate:\s*([A-Za-z][\w-]*))?\s*$/;
const HEADING = /^(#{2,3})\s+(.+?)\s*#*$/;
const FENCE = /^\s*(?:```|~~~)/;
/* exporter.py opens each Data Map / analysis line with `- \`<artifact id>\``.
 * Anywhere else a code span is at least as likely to be a column name, and
 * `order_id` has the same shape as an artifact id. */
const LEADING_ID = /^-\s+`([A-Za-z][A-Za-z0-9]*_[A-Za-z0-9][\w-]*)`/;
const ARTIFACT_ID = /^[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9][\w-]*$/;
const SEPARATOR_CELL = /^:?-{2,}:?$/;
const WORDISH = /[A-Za-z0-9]/;

export type ReportHeading = { id: string; text: string; level: 2 | 3 };

export type ClaimLedgerRow = {
  section: string;
  claim: string;
  evidence: string[];
  coverage: string;
};

/* Headings from here down are the machine record, not the narrative: a chart
 * inventory with one entry per generated chart, the claim ledger, the
 * per-figure verification trace, and the validator's own notes. They are the
 * reason the report is defensible and they are also ~half its words, so the
 * page keeps them behind one disclosure instead of in the reading path.
 * schemas/reports.py:318 fixes this section order. */
const REFERENCE_HEADINGS = new Set([
  "Appendix: Charts and Technical Summary",
  "Claim Ledger",
  "Evidence detail",
  "Validator Findings",
  "Audit Notes",
]);

/* schemas/reports.py:10-11. The exporter prints one of these under every
 * section heading that carries claims, so a 11-section report repeats the same
 * sentence up to 11 times. The badges on each claim already say what tier of
 * evidence backs it. */
const SECTION_PREAMBLES = new Set([
  "Validated evidence-backed findings are listed below.",
  "Analysis focus questions from this run are listed below.",
]);

export type ReportOutline = {
  /** Markdown with the hoisted status line removed. */
  body: string;
  /** `body` up to the first reference heading — the part meant to be read. */
  narrative: string;
  /** The reference half, or "" when the report has no such sections. */
  reference: string;
  /** Headings inside `reference`, for the disclosure's summary line. */
  referenceHeadings: ReportHeading[];
  /** Words in `narrative` only — what the reading estimate should count. */
  narrativeWords: number;
  status: string | null;
  gate: string | null;
  headings: ReportHeading[];
  words: number;
  ledger: ClaimLedgerRow[];
  /** Ids the report printed as evidence — the only tokens made clickable. */
  inspectableIds: Set<string>;
};

export function isArtifactIdShape(token: string): boolean {
  return token.length <= 128 && ARTIFACT_ID.test(token);
}

/** Stable across the table of contents and the rendered heading, so a jump
 *  target never depends on render order. */
export function headingId(text: string): string {
  const slug = text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return `report-${slug || "section"}`;
}

function splitRow(line: string): string[] {
  return line
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((cell) => cell.trim());
}

export function readReportOutline(markdown: string): ReportOutline {
  const kept: string[] = [];
  const narrativeLines: string[] = [];
  const referenceLines: string[] = [];
  const headings: ReportHeading[] = [];
  const referenceHeadings: ReportHeading[] = [];
  const ledger: ClaimLedgerRow[] = [];
  const inspectableIds = new Set<string>();
  const seenHeadings = new Set<string>();
  let status: string | null = null;
  let gate: string | null = null;
  let fenced = false;
  let inLedger = false;
  let inReference = false;
  let words = 0;
  let narrativeWords = 0;

  for (const line of markdown.split("\n")) {
    const emit = (text: string) => {
      kept.push(text);
      (inReference ? referenceLines : narrativeLines).push(text);
    };

    if (FENCE.test(line)) {
      fenced = !fenced;
      emit(line);
      continue;
    }
    if (fenced) {
      emit(line);
      continue;
    }

    if (status === null) {
      const match = STATUS_LINE.exec(line.trim());
      if (match) {
        status = match[1] ?? null;
        gate = match[2] ?? null;
        /* Dropped from the body because the page header shows the same pair
         * as badges a few lines above it. */
        continue;
      }
    }

    if (SECTION_PREAMBLES.has(line.trim())) continue;

    const heading = HEADING.exec(line);
    if (heading) {
      const text = (heading[2] ?? "").trim();
      const id = headingId(text);
      /* Latching: once the report reaches its reference half it stays there,
       * so a stray narrative-looking heading after the ledger cannot pull the
       * evidence tables back into the reading path. */
      inReference = inReference || REFERENCE_HEADINGS.has(text);
      const entry: ReportHeading = {
        id,
        text,
        level: heading[1]?.length === 2 ? 2 : 3,
      };
      if (!seenHeadings.has(id)) {
        seenHeadings.add(id);
        headings.push(entry);
        if (inReference) referenceHeadings.push(entry);
      }
      inLedger = text === "Claim Ledger";
    } else if (/^\s*#/.test(line)) {
      inLedger = false;
    }

    if (inLedger && line.trim().startsWith("|")) {
      const cells = splitRow(line);
      const isSeparator = cells.every((cell) => SEPARATOR_CELL.test(cell));
      if (cells.length >= 4 && !isSeparator && cells[0] !== "Section") {
        const evidence = (cells[2] ?? "")
          .split(",")
          .map((token) => token.trim())
          .filter(Boolean);
        for (const token of evidence) {
          if (isArtifactIdShape(token)) inspectableIds.add(token);
        }
        ledger.push({
          section: cells[0] ?? "",
          claim: cells[1] ?? "",
          evidence,
          coverage: cells[3] ?? "",
        });
      }
    }

    const leading = LEADING_ID.exec(line.trim());
    if (leading?.[1]) inspectableIds.add(leading[1]);

    const lineWords = line
      .split(/\s+/)
      .filter((token) => WORDISH.test(token)).length;
    words += lineWords;
    if (!inReference) narrativeWords += lineWords;
    emit(line);
  }

  return {
    body: kept.join("\n"),
    narrative: narrativeLines.join("\n"),
    reference: referenceLines.join("\n").trim() ? referenceLines.join("\n") : "",
    referenceHeadings,
    narrativeWords,
    status,
    gate,
    headings,
    words,
    ledger,
    inspectableIds,
  };
}

/** Claim ids and sections that cite one artifact, for the inspector's header. */
export function citationsFor(
  ledger: ClaimLedgerRow[],
  artifactId: string,
): ClaimLedgerRow[] {
  return ledger.filter((row) => row.evidence.includes(artifactId));
}
