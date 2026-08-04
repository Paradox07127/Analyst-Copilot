/* Display vocabulary for the profiler's issue codes (tools/quality.py).
 *
 * The codes are a backend enum. They belong in URLs (`?codes=empty_column`),
 * in filter values and in artifact payloads — but they were also the primary
 * label wherever a flag was shown, so the Data map inspector read
 * "duplicate_rows · empty_column · high_missing · id_not_unique" and the
 * reader had to already know the vocabulary to use the page.
 *
 * `label` is a short noun phrase naming the condition; `meaning` says what it
 * costs the analysis, matching the sentences the report generator writes
 * (tools/quality_context.py:_LIMITATION_BY_CODE). An unmapped code falls back
 * to a de-underscored form rather than being hidden — a new scanner code
 * shows up honestly instead of vanishing. */

interface QualityCodeCopy {
  label: string;
  meaning: string;
}

const QUALITY_CODES: Record<string, QualityCodeCopy> = {
  empty_column: {
    label: "Entirely missing",
    meaning: "Every row is blank, so nothing can rest on this column.",
  },
  empty_dataset: {
    label: "Empty table",
    meaning: "The table has no rows to analyse.",
  },
  constant_column: {
    label: "No variation",
    meaning:
      "One value in every row, so it cannot explain any difference between rows.",
  },
  high_missing: {
    label: "Often missing",
    meaning:
      "Missing often enough that results using it describe the rows that have it, not the table as a whole.",
  },
  outlier_detected: {
    label: "Extreme values",
    meaning: "Means and totals over this column move with a few far-out rows.",
  },
  duplicate_rows: {
    label: "Duplicate rows",
    meaning: "Rows repeat, so counts and totals may be inflated.",
  },
  id_not_unique: {
    label: "Ids repeat",
    meaning:
      "Does not identify a row on its own — joining or counting on it multiplies rows.",
  },
  id_missing: {
    label: "Ids blank",
    meaning: "Some rows have no id, so they cannot be joined or traced.",
  },
  likely_id_column: {
    label: "Looks like an id",
    meaning: "Averages and correlations over an identifier carry no meaning.",
  },
  high_cardinality_category: {
    label: "Nearly unique values",
    meaning: "Grouping by it produces mostly single-row groups.",
  },
  numeric_parse_failure: {
    label: "Non-numeric values",
    meaning: "Some values are not numbers and drop out of any calculation.",
  },
  date_parse_failure: {
    label: "Unreadable dates",
    meaning: "Some values are not readable dates and drop out of time views.",
  },
  mixed_type_string: {
    label: "Mixed formats",
    meaning:
      "Value formats differ, so a normalization rule is needed before comparing or aggregating.",
  },
  non_finite_numeric: {
    label: "Infinite values",
    meaning: "Infinite or undefined values distort any statistic over it.",
  },
  surrounding_whitespace: {
    label: "Stray spaces",
    meaning:
      "Values differ only by leading or trailing spaces, so grouping splits one category in two.",
  },
};

export function qualityCodeLabel(code: string): string {
  return QUALITY_CODES[code]?.label ?? code.replaceAll("_", " ");
}

export function qualityCodeMeaning(code: string): string | undefined {
  return QUALITY_CODES[code]?.meaning;
}

/** Label plus the code itself, for a `title` where only the label is visible. */
export function qualityCodeTitle(code: string): string {
  const meaning = qualityCodeMeaning(code);
  return meaning ? `${meaning} (${code})` : code;
}
