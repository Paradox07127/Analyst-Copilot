from __future__ import annotations

from html import escape

import vl_convert as vlc

from eda_platform.core.provenance import env_digest
from eda_platform.schemas.artifacts import Artifact, SqlResult
from eda_platform.schemas.charts import ChartSpec
from eda_platform.schemas.reports import ReportBundle, merge_duplicate_sections
from eda_platform.tools.evidence import EvidencePack, PayloadPolicy
from eda_platform.tools.evidence_display import evidence_display_context, evidence_lines
from eda_platform.tools.exporter import (
    _LIMITATIONS_SECTION,
    display_claim_text,
    display_narrative,
    display_section_body,
    render_status_line,
    section_render_context,
)


def export_report_html(
    bundle: ReportBundle,
    charts: list[ChartSpec] | None = None,
    *,
    artifacts: list[Artifact] | None = None,
    payload_policy: PayloadPolicy = "schema+aggregates",
) -> str:
    """Render the self-contained HTML report.

    ``payload_policy`` must be the policy the bundle's claims were validated
    under, so evidence-detail lines agree with persisted numeric statuses.
    """
    lang = "en"
    parts = [
        "<!doctype html>",
        f'<html lang="{lang}">',
        "<head>",
        '<meta charset="utf-8">',
        "<title>EDA Agent Report</title>",
        "<style>",
        _stylesheet(),
        "</style>",
        "</head>",
        "<body>",
        "<main>",
        "<h1>EDA Agent Report</h1>",
        f'<p class="status">{escape(render_status_line(bundle))}</p>',
    ]
    # Section bodies come from the same synthesis/scrub/format pipeline as the
    # markdown exporter (display_section_body), so the two cannot diverge.
    context = section_render_context(artifacts)
    # Merge duplicate sections and keep evidence in the claim ledger.
    for section in merge_duplicate_sections(bundle.sections):
        parts.append("<section>")
        parts.append(f"<h2>{escape(section.title)}</h2>")
        body = display_section_body(section, context)
        if body:
            parts.extend(_body_html(body))
        if section.narrative:
            parts.extend(_body_html(display_narrative(section.narrative)))
        if section.title == _LIMITATIONS_SECTION and context.execution_disclosures:
            parts.extend(_body_html("\n".join(context.execution_disclosures)))
        if section.focus_items:
            # F4: one list item per executed question, never a single blob.
            parts.append('<ul class="focus-items">')
            for item in section.focus_items:
                reason = (
                    f" &mdash; {escape(item.reason)}" if item.reason else ""
                )
                parts.append(
                    '<li class="focus-item">'
                    f'Analysis focus: "{escape(item.question)}" '
                    f"(outcome: {escape(item.outcome)}){reason}</li>"
                )
            parts.append("</ul>")
        if section.claims:
            parts.append("<ul>")
            for claim in section.claims:
                # Same prefix/scrub/format pipeline as the markdown narrative.
                parts.append(f"<li>{escape(display_claim_text(claim, section.title))}</li>")
            parts.append("</ul>")
        parts.append("</section>")
    parts.extend(_chart_gallery(charts or []))
    evidence_pack, sql_results = (
        evidence_display_context(artifacts, payload_policy=payload_policy)
        if artifacts
        else (None, {})
    )
    parts.extend(_claim_ledger(bundle, evidence_pack, sql_results))
    parts.extend(_audit_sections(bundle))
    parts.extend(_provenance_footer())
    parts.extend(["</main>", "</body>", "</html>"])
    return "\n".join(parts) + "\n"


def _body_html(body: str) -> list[str]:
    """Render a synthesized section body: markdown-style "- " lines become list
    items, other lines paragraphs."""
    parts: list[str] = []
    bullets: list[str] = []

    def _flush() -> None:
        if bullets:
            parts.append("<ul>")
            parts.extend(f"<li>{escape(item)}</li>" for item in bullets)
            parts.append("</ul>")
            bullets.clear()

    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("- "):
            bullets.append(line[2:])
        else:
            _flush()
            parts.append(f"<p>{escape(line)}</p>")
    _flush()
    return parts


def _provenance_footer() -> list[str]:
    """Add the reproducible-environment fingerprint to the report footer."""
    return [
        '<footer class="provenance">',
        f"Environment digest: <code>{escape(env_digest())}</code>",
        "</footer>",
    ]


def _chart_gallery(charts: list[ChartSpec]) -> list[str]:
    if not charts:
        return []
    parts = ['<section class="charts">', "<h2>Charts</h2>"]
    for chart in charts:
        svg = _render_chart_svg(chart)
        parts.append('<figure class="chart">')
        parts.append(f"<figcaption>{escape(chart.title)}</figcaption>")
        parts.append(svg if svg is not None else "<p>(chart could not be rendered)</p>")
        if chart.description:
            # Plain-language read-me for the chart, so each figure is self-explaining.
            parts.append(f'<p class="plain-language">{escape(chart.description)}</p>')
        parts.append("</figure>")
    parts.append("</section>")
    return parts


def _render_chart_svg(chart: ChartSpec) -> str | None:
    try:
        svg = vlc.vegalite_to_svg(chart.to_vegalite())
    except (RuntimeError, ValueError):
        return None
    return svg.replace(
        "<svg ",
        f'<svg role="img" aria-label="{escape(chart.title)}" ',
        1,
    )


def _claim_ledger(
    bundle: ReportBundle,
    evidence_pack: EvidencePack | None,
    sql_results: dict[str, SqlResult],
) -> list[str]:
    rows = [
        (section.title, claim)
        for section in merge_duplicate_sections(bundle.sections)
        for claim in section.claims
    ]
    parts = ['<section class="ledger">', "<h2>Claim Ledger</h2>"]
    if not rows:
        parts.append("<p>No claims were generated.</p>")
        parts.append("</section>")
        return parts
    parts.extend(
        [
            "<table>",
            "<thead><tr><th>Section</th><th>Claim</th><th>Evidence</th>"
            "<th>Numeric</th><th>Coverage</th></tr></thead>",
            "<tbody>",
        ]
    )
    for section_title, claim in rows:
        evidence = ", ".join(
            escape(evidence.artifact_id or evidence.locator) for evidence in claim.evidence
        )
        coverage = "gap" if claim.quantitative_coverage_gap else "ok"
        parts.append(
            "<tr>"
            f"<td>{escape(section_title)}</td>"
            f"<td>{escape(claim.id or claim.text)}</td>"
            f"<td>{evidence}</td>"
            f"<td>{escape(claim.numeric_rollup)}</td>"
            f"<td>{coverage}</td>"
            "</tr>"
        )
        detail = "".join(
            f"<li>{escape(line)}</li>"
            for line in evidence_lines(claim, evidence_pack, sql_results)
        )
        parts.append(
            '<tr class="evidence-detail"><td colspan="5">'
            f'<ul class="evidence-lines">{detail}</ul></td></tr>'
        )
    parts.extend(["</tbody>", "</table>", "</section>"])
    return parts


def _audit_sections(bundle: ReportBundle) -> list[str]:
    if not bundle.audit:
        return []
    parts = ['<section class="audit">', "<h2>Validator Findings</h2>"]
    if bundle.audit.findings:
        parts.append("<ul>")
        for finding in bundle.audit.findings:
            parts.append(
                "<li>"
                f"<strong>{escape(finding.severity.value)}</strong> "
                f"{escape(finding.code)}: {escape(finding.message)}"
                "</li>"
            )
        parts.append("</ul>")
    else:
        parts.append("<p>No validator findings.</p>")
    if bundle.audit.numeric_unverified_claim_count > 0:
        parts.append(
            "<p>Unverified numeric claims: "
            f"{bundle.audit.numeric_unverified_claim_count}</p>"
        )
    if bundle.audit.quantitative_coverage_gap_count > 0:
        parts.append(
            "<p>Quantitative coverage gaps: "
            f"{bundle.audit.quantitative_coverage_gap_count}</p>"
        )
    if bundle.audit.semantic_notes:
        parts.append("<h2>Audit Notes</h2>")
        parts.append("<ul>")
        for note in bundle.audit.semantic_notes:
            parts.append(f"<li>{escape(note)}</li>")
        parts.append("</ul>")
    parts.append("</section>")
    return parts


def _stylesheet() -> str:
    return """
body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: #172026; background: #f7f8fa; }
main { max-width: 1100px; margin: 0 auto; padding: 32px; background: #ffffff; min-height: 100vh; }
h1 { margin: 0 0 8px; font-size: 32px; }
h2 { margin-top: 28px; border-bottom: 1px solid #d8dee4; padding-bottom: 6px; font-size: 22px; }
p, li, td, th { font-size: 14px; line-height: 1.55; }
.status { color: #425466; margin-top: 0; }
.chart { margin: 16px 0; }
.chart svg { max-width: 100%; height: auto; border: 1px solid #e2e8f0; background: #fff; }
.chart figcaption { font-weight: 600; margin-bottom: 6px; }
.plain-language { color: #52616b; font-size: 13px; margin-top: 6px; }
table { width: 100%; border-collapse: collapse; margin-top: 8px; }
th, td { border: 1px solid #d8dee4; padding: 8px; text-align: left; vertical-align: top; }
th { background: #edf2f7; }
.evidence-detail td { border-top: none; background: #fbfcfd; padding-top: 4px; }
.evidence-lines { margin: 0; padding-left: 12px; list-style: none; }
.evidence-lines li { color: #52616b; font-size: 12px; line-height: 1.5; }
.provenance { margin-top: 32px; padding-top: 12px; border-top: 1px solid #d8dee4;
  color: #52616b; font-size: 12px; }
.provenance code { font-size: 12px; color: #425466; }
""".strip()
