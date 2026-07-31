from __future__ import annotations

import math
import re
from typing import Any, NamedTuple

from eda_platform.core.ids import make_artifact_id
from eda_platform.core.provenance import env_digest
from eda_platform.schemas.artifacts import (
    AnalysisTable,
    Artifact,
    ArtifactType,
    DatasetProfile,
    QualityIssueSet,
)
from eda_platform.schemas.charts import ChartSpec
from eda_platform.schemas.decision_report import DecisionReport
from eda_platform.schemas.quality_context import QualityContext, QualityContextSet
from eda_platform.schemas.reports import (
    ReportBundle,
    ReportClaim,
    ReportSection,
    merge_duplicate_sections,
)
from eda_platform.tools.evidence import PayloadPolicy
from eda_platform.tools.evidence_display import evidence_display_context, evidence_lines

# Sections whose empty-state body is replaced by a deterministic synthesis
# rendered from quality/chart artifacts (m4-plan §6 Day-0 patches).
_LIMITATIONS_SECTION = "Limitations and Risks"
_APPENDIX_SECTION = "Appendix: Charts and Technical Summary"
# Fallback body injected upstream (agents/reporting.py) for claimless sections.
_EMPTY_SECTION_BODY = "No validated conclusion is available for this section."
# Heading that separates the human narrative from the evidence/trust tables.
_CLAIM_LEDGER_HEADING = "### Claim Ledger"
# Prefix injected for Executive Summary claims (agents/reporting.py); pure
# duplication at render time, so it is stripped from the narrative only.
_SUMMARY_PREFIX = "Summary: "
# Max column names to enumerate before collapsing to a "… and N more" tail.
_MAX_LIMITATION_COLUMNS = 8
# Quality codes that flag a column as high-risk for downstream analysis.
_HIGH_RISK_QUALITY_CODES = {
    "high_missing",
    "empty_column",
    "outlier_detected",
    "mixed_type_string",
    "high_cardinality_category",
    "date_parse_failure",
}

# Aggregate repeated quality-context lines at render time.
QUALITY_AGGREGATE_THRESHOLD = 3
# Aggregate issue codes that cross the report-wide threshold.
CROSS_DATASET_AGGREGATE_THRESHOLD = 6
# High-risk column footers collapse the same way once this many datasets are
# affected (one line per dataset otherwise).
HIGH_RISK_FOOTER_DATASET_THRESHOLD = 3
# Per-code aggregate phrasings ({scope} = "12 of 31 columns" or "12 columns").
_QUALITY_CODE_AGGREGATE_PHRASES = {
    "outlier_detected": "Outliers were detected in {scope}; interpret numeric extremes with care",
    "high_missing": (
        "High missing rates affect {scope}; conclusions involving them may not generalize"
    ),
    "constant_column": "{scope} hold a single constant value and carry no analytical signal",
    "likely_id_column": "{scope} look like identifier columns with limited analytical meaning",
    "mixed_type_string": (
        "Mixed value types were observed in {scope}; type coercion may distort aggregates"
    ),
}
_QUALITY_AGGREGATE_FALLBACK_PHRASE = (
    "The observed {code} condition affects {scope}; interpret related results with care"
)
_QUALITY_AGGREGATE_SUFFIX = " (full list on the Quality page)."

# Internal artifact-id hygiene: narrative prose and limitations must not leak
# run-internal ids; full provenance stays in the Claim Ledger and the Appendix.
_INTERNAL_ID_TOKEN_PATTERN = re.compile(
    r"`?\b(?:qualityctx|sql|ds|chart|qexec)_[0-9a-fA-F]{6,}\b`?"
)
_EVIDENCE_PARENTHETICAL_PATTERN = re.compile(r"\s*\((?:evidence|source)s?:[^)]*\)")

# Number formatting (pure render-time functions; artifact values are untouched).
THOUSANDS_MIN_ABS = 10_000
# Fraction digits scale with magnitude rather than being fixed, so a duration
# renders "12.5 days" instead of "12.4973 days" and a share renders "8.11%"
# instead of "8.1129%". A fixed count is only ever right for one magnitude band.
SIGNIFICANT_DIGITS = 3
# Ceiling for the small-number tail. Without it a p-value of 1.2e-5 would round
# to "0" and assert something the data does not say.
MAX_FRACTION_DIGITS = 10
# Grouped alternative first: an already thousands-separated number is one
# token, or "1,020" would re-split into 1 and 020 and render as "1,20".
_NUMBER_TOKEN_PATTERN = re.compile(
    r"(?<![\w.-])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?%?(?!\d)"
    r"|(?<![\w.-])-?\d+(?:\.\d+)?%?"
)
# Double-quoted spans are verbatim citations and backticked spans are code /
# column names: both are preserved byte-identically by the text formatter.
_PROTECTED_SPAN_PATTERN = re.compile(r'"[^"]*"|`[^`]*`')
# Analysis-table fields whose 0..1 values render as percentages.
_RATIO_FIELD_PATTERN = re.compile(
    r"(?:^|_)(?:rate|ratio|share|pct|percent|proportion|fraction)(?:_|$)", re.IGNORECASE
)

# Deduplicate claims with matching numbers, columns, and evidence semantics.
_DEDUP_EXEMPT_SECTIONS = {"Executive Summary"}
_CROSS_REFERENCE_TEMPLATE = 'See "{section}" for the full statement of this finding.'
_COLUMN_TOKEN_PATTERN = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")


def fraction_digits_for(number: float) -> int:
    """Decimal places that keep SIGNIFICANT_DIGITS of a non-integer.

    Counts are never rounded — whole numbers take the integer branch above and
    "96,476 rows" stays exact. This only bounds the fractional tail.

    Shortening must never turn a non-integer into an integer: a rate of 99.9987%
    rendered as "100%" claims a perfection the data denies, a correlation of
    0.99991 shown as "1" reads as a deterministic identity, and "1,500.5 exceeds
    the 1,500 threshold" collapses into a sentence contradicting itself. So the
    tail is widened until the rendered value is still distinguishable from the
    whole number it would otherwise become.
    """
    if not math.isfinite(number):
        return 0
    magnitude = math.floor(math.log10(abs(number)))
    digits = max(0, min(SIGNIFICANT_DIGITS - 1 - magnitude, MAX_FRACTION_DIGITS))
    while digits < MAX_FRACTION_DIGITS and float(round(number, digits)).is_integer():
        digits += 1
    return digits


def format_number(
    value: float | int,
    *,
    ratio_as_percent: bool = False,
    force_thousands: bool = False,
) -> str:
    """Format one number for display: thousands separators, magnitude-scaled decimals.

    ``force_thousands`` keeps separators below THOUSANDS_MIN_ABS for tokens the
    source text already grouped (e.g. "3,095" must not degrade to "3095").
    """
    number = float(value)
    if ratio_as_percent and 0.0 <= number <= 1.0:
        rendered = f"{number * 100:.2f}".rstrip("0").rstrip(".")
        return f"{rendered or '0'}%"
    if number.is_integer():
        integer = int(number)
        if force_thousands or abs(integer) >= THOUSANDS_MIN_ABS:
            return f"{integer:,}"
        return str(integer)
    digits = fraction_digits_for(number)
    if force_thousands or abs(number) >= THOUSANDS_MIN_ABS:
        rendered = f"{number:,.{digits}f}"
    else:
        rendered = f"{number:.{digits}f}"
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def format_numbers_in_text(text: str) -> str:
    """Re-render number tokens in prose (render-only; citations/code untouched)."""

    def _format_segment(segment: str) -> str:
        def _replace(match: re.Match[str]) -> str:
            token = match.group(0)
            is_percent = token.endswith("%")
            raw = token.removesuffix("%")
            grouped = "," in raw
            try:
                value = float(raw.replace(",", ""))
            except ValueError:
                return token
            rendered = format_number(value, force_thousands=grouped)
            return f"{rendered}%" if is_percent else rendered

        return _NUMBER_TOKEN_PATTERN.sub(_replace, segment)

    pieces: list[str] = []
    last = 0
    for match in _PROTECTED_SPAN_PATTERN.finditer(text):
        pieces.append(_format_segment(text[last : match.start()]))
        pieces.append(match.group(0))
        last = match.end()
    pieces.append(_format_segment(text[last:]))
    return "".join(pieces)


def scrub_internal_ids(text: str) -> str:
    """Remove run-internal artifact ids from narrative prose (render-only)."""
    cleaned = _EVIDENCE_PARENTHETICAL_PATTERN.sub("", text)
    cleaned = _INTERNAL_ID_TOKEN_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r" +([,.;:!?)])", r"\1", cleaned)
    return cleaned.strip()


def decision_report_to_markdown(report: DecisionReport) -> str:
    """Render an evidence-bounded SCQA decision report as markdown."""
    lines = [f"# {report.title}", ""]
    for title, body in (
        ("Situation", report.scqa.situation),
        ("Complication", report.scqa.complication),
        ("Question", report.scqa.question),
        ("Answer", report.scqa.answer),
    ):
        lines.extend([f"## {title}", "", body, ""])

    for section in report.sections:
        lines.extend([f"## {section.title}", "", section.body, ""])

    lines.extend(["## Limitations", ""])
    # DI10 W3: limitations render without run-internal evidence ids (the
    # persisted DecisionReport payload keeps the original strings).
    lines.extend(
        f"- {scrub_internal_ids(limitation)}" for limitation in report.limitations
    )
    if not report.limitations:
        lines.append("- No additional limitations were recorded.")
    lines.extend(["", "## Investigation Gaps", ""])
    lines.extend(f"- {gap}" for gap in report.investigation_gaps)
    if not report.investigation_gaps:
        lines.append("- No investigation gaps were recorded.")
    lines.extend(["", f"Report readiness: `{report.report_readiness}`", ""])
    return "\n".join(lines).rstrip() + "\n"


def export_markdown_report(
    artifacts: list[Artifact],
    *,
    project_id: str,
    session_id: str,
) -> Artifact:
    profiles = [
        (artifact, DatasetProfile.model_validate(artifact.payload))
        for artifact in artifacts
        if artifact.type is ArtifactType.DATASET_PROFILE
    ]
    quality_sets = [
        (artifact, QualityIssueSet.model_validate(artifact.payload))
        for artifact in artifacts
        if artifact.type is ArtifactType.QUALITY_ISSUE_SET
    ]
    charts = [
        (artifact, ChartSpec.model_validate(artifact.payload))
        for artifact in artifacts
        if artifact.type is ArtifactType.CHART_SPEC
    ]
    analysis_tables = [
        (artifact, AnalysisTable.model_validate(artifact.payload))
        for artifact in artifacts
        if artifact.type is ArtifactType.TABLE
    ]

    lines = ["# EDA Run Report", ""]
    lines.extend(_data_map_lines(profiles))
    lines.extend(_quality_lines(quality_sets))
    lines.extend(_analysis_lines(analysis_tables))
    lines.extend(_chart_lines(charts))
    lines.extend(_suggestion_lines(profiles, quality_sets))
    lines.extend(_limitation_lines())
    lines.extend(["", "## Artifact Index"])
    for artifact in artifacts:
        lines.append(f"- `{artifact.id}` ({artifact.type.value})")
    lines.extend(_provenance_footer_lines())
    markdown = "\n".join(lines) + "\n"
    payload = {"markdown": markdown}
    report_identity = {"session_id": session_id, "artifact_ids": [a.id for a in artifacts]}
    return Artifact(
        id=make_artifact_id("report", report_identity),
        type=ArtifactType.MARKDOWN_REPORT,
        project_id=project_id,
        session_id=session_id,
        parents=[artifact.id for artifact in artifacts],
        payload=payload,
    )


def _data_map_lines(profiles: list[tuple[Artifact, DatasetProfile]]) -> list[str]:
    lines = ["## Data Map"]
    if not profiles:
        return [*lines, "- No dataset profiles were generated.", ""]
    for artifact, profile in profiles:
        lines.append(
            f"- `{artifact.id}` `{profile.name}`: {profile.rows} rows, "
            f"{profile.columns} columns, {profile.duplicate_rows} duplicate rows."
        )
        type_counts = ", ".join(
            f"{semantic_type}: {count}"
            for semantic_type, count in sorted(profile.semantic_type_counts.items())
        )
        if type_counts:
            lines.append(f"  - Semantic types: {type_counts}.")
        if profile.primary_key_candidates:
            candidates = ", ".join(profile.primary_key_candidates)
            lines.append(f"  - PK candidates: {candidates}.")
    lines.append("")
    return lines


def _quality_lines(quality_sets: list[tuple[Artifact, QualityIssueSet]]) -> list[str]:
    lines = ["## Quality Risks"]
    if not quality_sets:
        return [*lines, "- No quality scan artifacts were generated.", ""]
    for artifact, issue_set in quality_sets:
        if not issue_set.issues:
            lines.append(f"- `{artifact.id}`: no quality issues.")
            continue
        for issue in issue_set.issues:
            column = f" `{issue.column}`" if issue.column else ""
            lines.append(
                f"- `{artifact.id}` **{issue.severity}** `{issue.code}`{column}: "
                f"{issue.message} Recommendation: {issue.recommendation}"
            )
    lines.append("")
    return lines


def _chart_lines(charts: list[tuple[Artifact, ChartSpec]]) -> list[str]:
    lines = ["## Generated Charts"]
    if not charts:
        return [*lines, "- No chart specs were generated.", ""]
    for artifact, chart in charts:
        lines.append(f"- `{artifact.id}` {chart.title}: {chart.description}")
        if artifact.plain_language:
            # Plain-language read-me so each chart is self-explaining in the report.
            lines.append(f"  - {artifact.plain_language}")
    lines.append("")
    return lines


def _provenance_footer_lines() -> list[str]:
    """A report footer that records the reproducible-environment fingerprint."""
    return ["", "---", "", f"Environment digest: `{env_digest()}`", ""]


def _analysis_lines(tables: list[tuple[Artifact, AnalysisTable]]) -> list[str]:
    lines = ["## Deterministic Analyses"]
    if not tables:
        return [*lines, "- No deterministic analysis tables were generated.", ""]
    for artifact, table in tables:
        lines.append(f"- `{artifact.id}` {table.title}: {table.description}")
        preview = table.rows[:3]
        for row in preview:
            lines.append(f"  - {_format_analysis_row(row)}")
    lines.append("")
    return lines


def _format_analysis_row(row: dict[str, Any]) -> str:
    """Render one preview row with display formatting (DI10 W3, render-only)."""
    rendered = {
        key: (
            format_number(value, ratio_as_percent=bool(_RATIO_FIELD_PATTERN.search(key)))
            if isinstance(value, int | float) and not isinstance(value, bool)
            else value
        )
        for key, value in row.items()
    }
    return str(rendered)


def _suggestion_lines(
    profiles: list[tuple[Artifact, DatasetProfile]],
    quality_sets: list[tuple[Artifact, QualityIssueSet]],
) -> list[str]:
    lines = ["## Suggested Next Analyses"]
    has_datetime = any(
        column.semantic_type == "datetime"
        for _, profile in profiles
        for column in profile.columns_detail
    )
    has_numeric = any(
        column.semantic_type == "numeric"
        for _, profile in profiles
        for column in profile.columns_detail
    )
    has_quality_risk = any(
        issue.code != "no_high_missing"
        for _, q in quality_sets
        for issue in q.issues
    )

    if has_datetime and has_numeric:
        lines.append("- Review time trends for numeric measures.")
    if has_numeric:
        lines.append("- Compare numeric measures across categorical segments.")
    if has_quality_risk:
        lines.append("- Resolve quality warnings before making high-stakes conclusions.")
    if len(lines) == 1:
        lines.append("- Continue exploratory profiling with additional business context.")
    lines.append("")
    return lines


def _limitation_lines() -> list[str]:
    return [
        "## Limitations",
        "- M1 reports are deterministic and do not include LLM-written interpretation.",
        "- Chart data is based on profile-safe summaries and samples.",
        "- Cleaning recommendations are not applied to the source data in M1.",
        "",
    ]


def _neutralize_markdown_inline(text: str) -> str:
    """Neutralize markdown syntax in untrusted text so it cannot fabricate
    report structure (links, code spans, headings, line breaks)."""
    collapsed = " ".join(text.split())
    for char in ("\\", "`", "[", "]"):
        collapsed = collapsed.replace(char, f"\\{char}")
    return collapsed


def render_status_line(bundle: ReportBundle) -> str:
    """Status plus semantic-gate verdict, so "validated" cannot mask a
    degraded or rejected gate outcome (A3)."""
    line = f"Status: {bundle.status.value}"
    if bundle.audit is not None and bundle.audit.gate_verdict != "pass":
        line += f" · Gate: {bundle.audit.gate_verdict}"
    return line


def report_bundle_to_markdown(
    bundle: ReportBundle,
    *,
    artifacts: list[Artifact] | None = None,
    payload_policy: PayloadPolicy = "schema+aggregates",
) -> str:
    """Render a report bundle to markdown.

    ``payload_policy`` must be the policy the bundle's claims were validated
    under, so evidence-detail lines agree with persisted numeric statuses.
    """
    context = section_render_context(artifacts)
    # Render each title once even if assembly forwarded duplicate-titled sections.
    sections = merge_duplicate_sections(bundle.sections)
    # Render one full claim per finding and cross-reference later sections.
    dispositions = _duplicate_claim_dispositions(sections)

    lines = [
        "# EDA Agent Report",
        "",
        render_status_line(bundle),
        "",
    ]
    for section in sections:
        lines.extend([f"## {section.title}", ""])
        body = display_section_body(section, context)
        if body:
            lines.extend([body, ""])
        if section.title == _LIMITATIONS_SECTION and context.execution_disclosures:
            # DI10-W5 red line: any execution that ran over an auto-confirmed
            # join must disclose it here, whatever else the section holds.
            lines.extend([*context.execution_disclosures, ""])
        if section.focus_items:
            # F4: structured focus entries, one list item per executed question.
            lines.extend(
                f'- Analysis focus: "{_neutralize_markdown_inline(item.question)}" '
                f"(outcome: {item.outcome})"
                for item in section.focus_items
            )
            lines.append("")
        rendered_claim_lines = 0
        for claim in section.claims:
            disposition = dispositions.get(id(claim))
            if disposition is not None:
                mode, primary_section = disposition
                if mode == "drop":
                    continue
                if mode == "crossref":
                    lines.append(
                        f"- {_CROSS_REFERENCE_TEMPLATE.format(section=primary_section)}"
                    )
                    rendered_claim_lines += 1
                    continue
            # Narrative shows human-readable claim text only; ids and inline
            # evidence live in the ### Claim Ledger table below (P0-1).
            lines.append(f"- {display_claim_text(claim, section.title)}")
            rendered_claim_lines += 1
        if rendered_claim_lines:
            lines.append("")

    lines.extend([_CLAIM_LEDGER_HEADING, ""])
    claim_rows = [
        (section.title, claim)
        for section in sections
        for claim in section.claims
    ]
    if claim_rows:
        lines.append("| Section | Claim | Evidence | Coverage |")
        lines.append("|---|---|---|---|")
        for section_title, claim in claim_rows:
            evidence = ", ".join(_claim_evidence_ids(claim))
            coverage = "gap" if claim.quantitative_coverage_gap else "ok"
            lines.append(
                f"| {section_title} | {claim.id or claim.text} | {evidence} | {coverage} |"
            )
        lines.extend(["", "#### Evidence detail", ""])
        evidence_pack, sql_results = (
            evidence_display_context(artifacts, payload_policy=payload_policy)
            if artifacts
            else (None, {})
        )
        for _section_title, claim in claim_rows:
            lines.append(f"- {claim.id or claim.text}")
            lines.extend(
                f"  - {evidence_line}"
                for evidence_line in evidence_lines(claim, evidence_pack, sql_results)
            )
    else:
        lines.append("- No claims were generated.")
    lines.append("")

    if bundle.audit:
        lines.extend(["### Validator Findings", ""])
        if bundle.audit.findings:
            for finding in bundle.audit.findings:
                lines.append(f"- **{finding.severity.value}** `{finding.code}`: {finding.message}")
        else:
            lines.append("- No validator findings.")
        if bundle.audit.quantitative_coverage_gap_count > 0:
            lines.append(
                "- Quantitative coverage gaps: "
                f"{bundle.audit.quantitative_coverage_gap_count}"
            )
        if bundle.audit.semantic_notes:
            lines.extend(["", "### Audit Notes", ""])
            for note in bundle.audit.semantic_notes:
                lines.append(f"- {note}")
    lines.extend(_provenance_footer_lines())
    return "\n".join(lines).rstrip() + "\n"


def narrative_markdown(markdown: str) -> str:
    """Return only the narrative part of a rendered report markdown."""
    index = markdown.find(_CLAIM_LEDGER_HEADING)
    if index == -1:
        return markdown
    return markdown[:index].rstrip() + "\n"


def _claim_display_text(claim: ReportClaim) -> str:
    """Human-readable claim text for the narrative body."""
    return claim.text.removeprefix(_SUMMARY_PREFIX)


def display_claim_text(claim: ReportClaim, section_title: str) -> str:
    """DI10 W3 narrative pipeline: strip prefix, scrub ids, format numbers.
    Shared by the markdown and HTML exporters (P0 parity fix)."""
    text = _claim_display_text(claim)
    if claim.numeric_rollup == "unverified":
        text = f"[Unverified figures] {text}"
    # F6 strength tiers: strong carries no prefix; legacy "verified" /
    # "low_relevance" (pre-F6 bundles) keep their old rendering.
    if claim.confidence_label == "exploratory":
        text = f"[Exploratory — hypothesis-generating] {text}"
    elif claim.confidence_label == "indicative":
        text = f"[Indicative] {text}"
    elif claim.confidence_label == "low_relevance":
        text = f"[Low relevance] {text}"
    if section_title == _APPENDIX_SECTION:
        return text
    return format_numbers_in_text(scrub_internal_ids(text))


def _claim_number_signature(text: str) -> frozenset[str]:
    """Normalized set of number tokens outside citations/code spans."""
    unquoted = _PROTECTED_SPAN_PATTERN.sub(" ", text)
    tokens: set[str] = set()
    for match in _NUMBER_TOKEN_PATTERN.finditer(unquoted):
        raw = match.group(0).removesuffix("%").replace(",", "")
        try:
            tokens.add(repr(float(raw)))
        except ValueError:
            continue
    return frozenset(tokens)


def _claim_column_signature(claim: ReportClaim) -> frozenset[str]:
    """Referenced columns plus snake_case column-like tokens in the text."""
    tokens = {column.lower() for column in claim.referenced_columns}
    tokens.update(_COLUMN_TOKEN_PATTERN.findall(claim.text.lower()))
    return frozenset(tokens)


def _claim_evidence_signature(claim: ReportClaim) -> frozenset[tuple[str, str]]:
    """Evidence kind + locator identifies the asserted metric semantics."""
    return frozenset((evidence.kind, evidence.locator) for evidence in claim.evidence)


def _duplicate_claim_dispositions(
    sections: list[ReportSection],
) -> dict[int, tuple[str, str]]:
    """Detect same-finding claims across the narrative (DI10 W3)."""
    entries: list[
        tuple[
            str,
            ReportClaim,
            frozenset[str],
            frozenset[str],
            frozenset[tuple[str, str]],
        ]
    ] = []
    for section in sections:
        if section.title in _DEDUP_EXEMPT_SECTIONS:
            continue
        for claim in section.claims:
            numbers = _claim_number_signature(claim.text)
            if not numbers:
                continue
            entries.append(
                (
                    section.title,
                    claim,
                    numbers,
                    _claim_column_signature(claim),
                    _claim_evidence_signature(claim),
                )
            )

    groups: dict[
        frozenset[str],
        list[
            tuple[
                str,
                ReportClaim,
                frozenset[str],
                frozenset[str],
                frozenset[tuple[str, str]],
            ]
        ],
    ]
    groups = {}
    for entry in entries:
        groups.setdefault(entry[2], []).append(entry)

    dispositions: dict[int, tuple[str, str]] = {}
    for group in groups.values():
        if len(group) < 2:
            continue
        for cluster in _related_clusters(group):
            if len(cluster) < 2:
                continue
            primary = max(cluster, key=lambda item: len(item[1].text))
            primary_section = primary[0]
            crossref_sections: set[str] = set()
            for item in cluster:
                if item is primary:
                    dispositions[id(item[1])] = ("primary", primary_section)
                elif item[0] == primary_section or item[0] in crossref_sections:
                    dispositions[id(item[1])] = ("drop", primary_section)
                else:
                    crossref_sections.add(item[0])
                    dispositions[id(item[1])] = ("crossref", primary_section)
    return dispositions


def _related_clusters(
    group: list[
        tuple[
            str,
            ReportClaim,
            frozenset[str],
            frozenset[str],
            frozenset[tuple[str, str]],
        ]
    ],
) -> list[
    list[
        tuple[
            str,
            ReportClaim,
            frozenset[str],
            frozenset[str],
            frozenset[tuple[str, str]],
        ]
    ]
]:
    """Connected components over column and evidence compatibility."""
    parents = list(range(len(group)))

    def _find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for left in range(len(group)):
        for right in range(left + 1, len(group)):
            columns_left, columns_right = group[left][3], group[right][3]
            evidence_overlap = group[left][4] & group[right][4]
            columns_compatible = (
                not columns_left or not columns_right or bool(columns_left & columns_right)
            )
            if columns_compatible and evidence_overlap:
                parents[_find(left)] = _find(right)

    clusters: dict[
        int,
        list[
            tuple[
                str,
                ReportClaim,
                frozenset[str],
                frozenset[str],
                frozenset[tuple[str, str]],
            ]
        ],
    ] = {}
    for index, entry in enumerate(group):
        clusters.setdefault(_find(index), []).append(entry)
    return list(clusters.values())


def _column_totals(artifacts: list[Artifact]) -> dict[str, int]:
    """Map dataset_id -> profiled column count (for aggregate scope phrasing)."""
    totals: dict[str, int] = {}
    for artifact in artifacts:
        if artifact.type is ArtifactType.DATASET_PROFILE:
            profile = DatasetProfile.model_validate(artifact.payload)
            totals[profile.dataset_id] = profile.columns
    return totals


def _dataset_names(artifacts: list[Artifact]) -> dict[str, str]:
    """Map dataset_id -> human dataset name from profile artifacts."""
    names: dict[str, str] = {}
    for artifact in artifacts:
        if artifact.type is ArtifactType.DATASET_PROFILE:
            profile = DatasetProfile.model_validate(artifact.payload)
            names[profile.dataset_id] = profile.name
    return names


def _quality_sets(artifacts: list[Artifact]) -> list[tuple[Artifact, QualityIssueSet]]:
    return [
        (artifact, QualityIssueSet.model_validate(artifact.payload))
        for artifact in artifacts
        if artifact.type is ArtifactType.QUALITY_ISSUE_SET
    ]


def _charts(artifacts: list[Artifact]) -> list[tuple[Artifact, ChartSpec]]:
    return [
        (artifact, ChartSpec.model_validate(artifact.payload))
        for artifact in artifacts
        if artifact.type is ArtifactType.CHART_SPEC
    ]


def _quality_context_sets(
    artifacts: list[Artifact],
) -> list[tuple[Artifact, QualityContextSet]]:
    return [
        (artifact, QualityContextSet.model_validate(artifact.payload))
        for artifact in artifacts
        if artifact.type is ArtifactType.QUALITY_CONTEXT_SET
    ]


def _claim_evidence_ids(claim: ReportClaim) -> list[str]:
    """Deduplicate an evidence ref list for display, preserving first-seen order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for evidence in claim.evidence:
        ref = evidence.artifact_id or evidence.locator
        if ref in seen:
            continue
        seen.add(ref)
        ordered.append(ref)
    return ordered


def _execution_disclosures(artifacts: list[Artifact]) -> list[str]:
    """Collect report disclosures from question execution results."""
    disclosures: set[str] = set()
    for artifact in artifacts:
        if artifact.type is not ArtifactType.QUESTION_EXECUTION_RESULT:
            continue
        limitations = artifact.payload.get("limitations")
        if isinstance(limitations, list):
            disclosures.update(
                line for line in limitations if isinstance(line, str) and line.strip()
            )
    return sorted(disclosures)


class SectionRenderContext(NamedTuple):
    """Artifact-derived inputs both exporters need to render section bodies."""

    quality_sets: list[tuple[Artifact, QualityIssueSet]]
    quality_context_sets: list[tuple[Artifact, QualityContextSet]]
    charts: list[tuple[Artifact, ChartSpec]]
    dataset_names: dict[str, str]
    column_totals: dict[str, int]
    execution_disclosures: list[str]


def section_render_context(artifacts: list[Artifact] | None) -> SectionRenderContext:
    items = artifacts or []
    return SectionRenderContext(
        quality_sets=_quality_sets(items),
        quality_context_sets=_quality_context_sets(items),
        charts=_charts(items),
        dataset_names=_dataset_names(items),
        column_totals=_column_totals(items),
        execution_disclosures=_execution_disclosures(items),
    )


def display_section_body(section: ReportSection, context: SectionRenderContext) -> str:
    """Synthesized display body with id scrubbing and number formatting applied
    (appendix exempt); shared by the markdown and HTML exporters so the two
    renderings cannot diverge (P0 parity fix)."""
    body = _section_body(
        section.title,
        section,
        quality_sets=context.quality_sets,
        quality_context_sets=context.quality_context_sets,
        charts=context.charts,
        dataset_names=context.dataset_names,
        column_totals=context.column_totals,
    )
    if body and section.title != _APPENDIX_SECTION:
        body = format_numbers_in_text(scrub_internal_ids(body))
    return body


def _section_body(
    title: str,
    section: ReportSection,
    *,
    quality_sets: list[tuple[Artifact, QualityIssueSet]],
    quality_context_sets: list[tuple[Artifact, QualityContextSet]] | None = None,
    charts: list[tuple[Artifact, ChartSpec]],
    dataset_names: dict[str, str] | None = None,
    column_totals: dict[str, int] | None = None,
) -> str:
    """Return the body to render, synthesizing empty-state sections when possible."""
    if section.claims or section.body != _EMPTY_SECTION_BODY:
        return section.body
    if title == _LIMITATIONS_SECTION:
        synthesized = _limitations_body(
            quality_sets,
            dataset_names or {},
            quality_context_sets or [],
            column_totals or {},
        )
        return synthesized if synthesized else section.body
    if title == _APPENDIX_SECTION:
        synthesized = _appendix_body(charts)
        return synthesized if synthesized else section.body
    return section.body


def _limitations_body(
    quality_sets: list[tuple[Artifact, QualityIssueSet]],
    dataset_names: dict[str, str],
    quality_context_sets: list[tuple[Artifact, QualityContextSet]] | None = None,
    column_totals: dict[str, int] | None = None,
) -> str:
    """Synthesize the Limitations section from quality artifacts (m4-plan §6 patch 1)."""
    lines: list[str] = []
    seen: set[str] = set()
    column_totals = column_totals or {}
    # Identify issue codes that qualify for report-wide aggregation.
    global_by_code: dict[str, set[tuple[str, str]]] = {}
    for _artifact, context_set in quality_context_sets or []:
        for context in context_set.contexts:
            if context.severity == "critical" or not context.column:
                continue
            global_by_code.setdefault(context.issue_code, set()).add(
                (context_set.dataset_id, context.column)
            )
    globally_aggregated: set[str] = set()
    for code, members in sorted(global_by_code.items()):
        if len(members) <= CROSS_DATASET_AGGREGATE_THRESHOLD:
            continue
        globally_aggregated.add(code)
        dataset_count = len({dataset_id for dataset_id, _column in members})
        scope = f"{len(members)} columns across {dataset_count} datasets"
        phrase = _QUALITY_CODE_AGGREGATE_PHRASES.get(
            code, _QUALITY_AGGREGATE_FALLBACK_PHRASE
        ).format(scope=scope, code=code)
        lines.append(f"- {phrase}{_QUALITY_AGGREGATE_SUFFIX}")
    # Place dataset quality context before deterministic scan results.
    for _artifact, context_set in quality_context_sets or []:
        name = dataset_names.get(context_set.dataset_id, context_set.dataset_name)
        total = column_totals.get(context_set.dataset_id, 0)
        for line in _quality_context_lines(
            context_set,
            name=name,
            column_total=total,
            exclude_codes=globally_aggregated,
        ):
            if line in seen:
                continue
            seen.add(line)
            lines.append(line)
    # High-risk footers: with many affected datasets, one line per dataset is
    # noise — collapse into a single report-wide count (details on Quality page).
    high_risk_by_dataset = {
        issue_set.dataset_id: sorted(
            {
                issue.column
                for issue in issue_set.issues
                if issue.column and issue.code in _HIGH_RISK_QUALITY_CODES
            }
        )
        for _artifact, issue_set in quality_sets
    }
    high_risk_datasets = {k: v for k, v in high_risk_by_dataset.items() if v}
    collapse_high_risk = len(high_risk_datasets) > HIGH_RISK_FOOTER_DATASET_THRESHOLD
    if collapse_high_risk:
        total_columns = sum(len(columns) for columns in high_risk_datasets.values())
        lines.append(
            f"- High-risk columns were flagged in {len(high_risk_datasets)} datasets "
            f"({total_columns} columns); treat their statistics with care "
            "(full list on the Quality page)."
        )
    for _artifact, issue_set in quality_sets:
        name = dataset_names.get(issue_set.dataset_id, issue_set.dataset_id)
        empty_columns = _issue_columns(issue_set, "empty_column")
        constant_columns = _issue_columns(issue_set, "constant_column")
        high_risk_columns = high_risk_by_dataset.get(issue_set.dataset_id, [])
        sampling_notes = [
            issue.message
            for issue in issue_set.issues
            if "sampl" in issue.message.lower() or "sampl" in issue.code.lower()
        ]
        candidates: list[str] = []
        if high_risk_columns and not collapse_high_risk:
            candidates.append(
                f"- {name}: high-risk columns for analysis: {_join_columns(high_risk_columns)}."
            )
        if empty_columns:
            candidates.append(
                f"- {name}: columns that are 100% missing: {_join_columns(empty_columns)}."
            )
        if constant_columns:
            candidates.append(
                f"- {name}: constant columns (single value across rows): "
                f"{_join_columns(constant_columns)}."
            )
        for note in sampling_notes:
            candidates.append(f"- {name}: sampling note: {note}")
        for line in candidates:
            if line in seen:
                continue
            seen.add(line)
            lines.append(line)
    return "\n".join(lines)


def _quality_context_lines(
    context_set: QualityContextSet,
    *,
    name: str,
    column_total: int,
    exclude_codes: set[str] | None = None,
) -> list[str]:
    """Render one dataset's quality contexts with same-code aggregation (DI10 W3)."""
    exclude_codes = exclude_codes or set()
    itemized: list[QualityContext] = []
    by_code: dict[str, list[QualityContext]] = {}
    for context in context_set.contexts:
        # Critical conditions and dataset-level (columnless) conditions are
        # never aggregated away.
        if context.severity == "critical" or not context.column:
            itemized.append(context)
        elif context.issue_code in exclude_codes:
            continue
        else:
            by_code.setdefault(context.issue_code, []).append(context)

    lines: list[str] = []
    for code, group in by_code.items():
        columns = list(dict.fromkeys(context.column for context in group if context.column))
        if len(columns) > QUALITY_AGGREGATE_THRESHOLD:
            scope = (
                f"{len(columns)} of {column_total} columns"
                if column_total
                else f"{len(columns)} columns"
            )
            phrase = _QUALITY_CODE_AGGREGATE_PHRASES.get(
                code, _QUALITY_AGGREGATE_FALLBACK_PHRASE
            ).format(scope=scope, code=code)
            lines.append(f"- {name}: {phrase}{_QUALITY_AGGREGATE_SUFFIX}")
        else:
            itemized.extend(group)
    lines.extend(
        f"- {name}: {scrub_internal_ids(context.report_limitation)}" for context in itemized
    )
    return lines


def _appendix_body(charts: list[tuple[Artifact, ChartSpec]]) -> str:
    """Render a deterministic chart inventory for the Appendix (m4-plan §6 patch 2)."""
    if not charts:
        return ""
    lines = ["Chart inventory:"]
    for artifact, chart in charts:
        lines.append(
            f"- `{artifact.id}` {chart.title} ({chart.mark} chart, dataset {chart.dataset_id})."
        )
        if artifact.plain_language:
            lines.append(f"  - {artifact.plain_language}")
    return "\n".join(lines)


def _issue_columns(issue_set: QualityIssueSet, code: str) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for issue in issue_set.issues:
        if issue.code == code and issue.column and issue.column not in seen:
            seen.add(issue.column)
            ordered.append(issue.column)
    return ordered


def _join_columns(columns: list[str]) -> str:
    """Enumerate up to 8 column names, collapsing the rest to a counted tail."""
    shown = ", ".join(f"`{column}`" for column in columns[:_MAX_LIMITATION_COLUMNS])
    remaining = len(columns) - _MAX_LIMITATION_COLUMNS
    if remaining > 0:
        return f"{shown} … and {remaining} more (see the Quality page)"
    return shown
