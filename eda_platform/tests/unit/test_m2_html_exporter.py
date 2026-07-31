from __future__ import annotations

from eda_platform.schemas.artifacts import Artifact, ArtifactType, EvidenceRef
from eda_platform.schemas.quality_context import QualityContext, QualityContextSet
from eda_platform.schemas.reports import (
    ReportAudit,
    ReportBundle,
    ReportClaim,
    ReportSection,
    ReportStatus,
)
from eda_platform.tools.exporter import report_bundle_to_markdown
from eda_platform.tools.html_exporter import export_report_html


def test_report_bundle_to_markdown_renders_sections_claims_and_audit() -> None:
    bundle = _bundle()

    markdown = report_bundle_to_markdown(bundle)

    assert "## Executive Summary" in markdown
    assert "## Appendix: Charts and Technical Summary" in markdown
    assert "### Claim Ledger" in markdown
    assert "claim_1" in markdown
    assert "table_1" in markdown
    assert "Status: validated" in markdown


def test_export_report_html_is_self_contained_and_escapes_user_content() -> None:
    bundle = _bundle()
    bundle.sections[0].claims[0].text = "Revenue <script>alert(1)</script> is 120."

    html = export_report_html(bundle)

    assert "<!doctype html>" in html.lower()
    assert "Executive Summary" in html
    assert "Claim Ledger" in html
    assert "table_1" in html
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "https://" not in html
    assert "http://" not in html


def test_export_report_html_narrative_drops_ids_but_keeps_ledger() -> None:
    bundle = _bundle()
    bundle.sections[0].claims[0].text = "Summary: Revenue is 120."

    html = export_report_html(bundle)

    narrative, _, ledger = html.partition("Claim Ledger")
    # Narrative shows human claim text only: no id prefix, no inline evidence div.
    assert "<strong>claim_1</strong>" not in narrative
    assert 'class="evidence"' not in narrative
    assert "Revenue is 120." in narrative
    assert "Summary: Revenue is 120." not in narrative
    # Evidence id still lives in the Claim Ledger table.
    assert "table_1" in ledger


def test_export_report_html_marks_unverified_claims_like_markdown() -> None:
    # The HTML narrative must carry the same unverified prefix as the markdown
    # exporter, and the ledger must expose the numeric verification state.
    bundle = _bundle()
    bundle.sections[0].claims[0].numeric_rollup = "unverified"

    html = export_report_html(bundle)
    narrative, _, ledger = html.partition("Claim Ledger")

    assert "[Unverified figures] Revenue is 120." in narrative
    assert "<th>Numeric</th>" in ledger
    assert "<td>unverified</td>" in ledger


def test_export_report_html_renders_unverified_claim_count_in_audit() -> None:
    bundle = _bundle()
    bundle.audit = ReportAudit(
        status=ReportStatus.VALIDATED,
        numeric_unverified_claim_count=2,
    )

    html = export_report_html(bundle)

    assert "Unverified numeric claims: 2" in html

    bundle.audit.numeric_unverified_claim_count = 0
    assert "Unverified numeric claims" not in export_report_html(bundle)


def test_exports_surface_coverage_gap_in_ledger_and_audit() -> None:
    # F3 pass-through: the gap flag and the audit gap count must be visible in
    # both the HTML and the markdown exports.
    bundle = _bundle()
    bundle.sections[0].claims[0].quantitative_coverage_gap = True
    bundle.audit = ReportAudit(
        status=ReportStatus.VALIDATED,
        quantitative_coverage_gap_count=1,
    )

    html = export_report_html(bundle)
    _, _, html_ledger = html.partition("Claim Ledger")
    assert "<th>Coverage</th>" in html_ledger
    assert "<td>gap</td>" in html_ledger
    assert "Quantitative coverage gaps: 1" in html

    markdown = report_bundle_to_markdown(bundle)
    assert "| Section | Claim | Evidence | Coverage |" in markdown
    assert "| gap |" in markdown
    assert "Quantitative coverage gaps: 1" in markdown


def test_exports_show_ok_coverage_and_hide_zero_gap_count() -> None:
    bundle = _bundle()
    bundle.audit = ReportAudit(status=ReportStatus.VALIDATED)

    html = export_report_html(bundle)
    assert "<td>ok</td>" in html
    assert "Quantitative coverage gaps" not in html

    markdown = report_bundle_to_markdown(bundle)
    assert "| ok |" in markdown
    assert "Quantitative coverage gaps" not in markdown


def test_export_report_html_dedupes_duplicate_titled_sections() -> None:
    bundle = ReportBundle.empty(project_id="project_demo", session_id="run_demo")
    focus = next(s for s in bundle.sections if s.title == "Selected Analysis Focus")
    focus.claims.append(
        ReportClaim(
            id="a",
            text="First focus.",
            evidence=[EvidenceRef(kind="artifact", artifact_id="x", locator="l")],
        )
    )
    bundle.sections.append(
        ReportSection(
            title="Selected Analysis Focus",
            claims=[
                ReportClaim(
                    id="b",
                    text="Second focus.",
                    evidence=[EvidenceRef(kind="artifact", artifact_id="y", locator="l")],
                )
            ],
        )
    )

    html = export_report_html(bundle)

    assert html.count("<h2>Selected Analysis Focus</h2>") == 1
    assert "First focus." in html
    assert "Second focus." in html


def test_limitations_section_threads_quality_context_report_limitations() -> None:
    # Hand-merge check (DI sprint 2): QualityContextSet artifacts contribute
    # their ``report_limitation`` prose to the synthesized Limitations
    # section, alongside — not instead of — the existing QualityIssueSet
    # synthesis (main's claim-ledger dedup/merge behaviour is untouched).
    bundle = _bundle_with_empty_limitations_section()
    context_artifact = Artifact(
        id="qctx_1",
        type=ArtifactType.QUALITY_CONTEXT_SET,
        project_id="project_demo",
        session_id="run_demo",
        payload=QualityContextSet(
            dataset_id="ds_sales",
            dataset_name="sales.csv",
            contexts=[
                QualityContext(
                    context_id="ctx_1",
                    dataset_id="ds_sales",
                    dataset_name="sales.csv",
                    issue_code="high_missing",
                    severity="warn",
                    column="discount",
                    observation="`discount` is missing in 42% of rows.",
                    report_limitation=(
                        "Discount-related conclusions may not generalize: 42% of rows "
                        "lack a discount value."
                    ),
                )
            ],
        ).model_dump(mode="json"),
    )

    markdown = report_bundle_to_markdown(bundle, artifacts=[context_artifact])
    limitations = _section_text(markdown, "## Limitations and Risks")

    assert "No validated conclusion is available" not in limitations
    assert "sales.csv: Discount-related conclusions may not generalize" in limitations
    # DI10 W3: internal evidence ids no longer render in the human-readable
    # limitations prose (provenance stays on the Quality page / artifacts).
    assert "evidence:" not in limitations
    assert "qctx_1" not in limitations


def test_limitations_section_dedupes_identical_quality_context_lines() -> None:
    # If the same QualityContextSet artifact is collected twice upstream (e.g.
    # forwarded by more than one investigation record), its report_limitation
    # line must not be duplicated in the rendered Limitations section.
    bundle = _bundle_with_empty_limitations_section()
    context_artifact = Artifact(
        id="qctx_1",
        type=ArtifactType.QUALITY_CONTEXT_SET,
        project_id="project_demo",
        session_id="run_demo",
        payload=QualityContextSet(
            dataset_id="ds_sales",
            dataset_name="sales.csv",
            contexts=[
                QualityContext(
                    context_id="ctx_1",
                    dataset_id="ds_sales",
                    dataset_name="sales.csv",
                    issue_code="high_missing",
                    severity="warn",
                    column="discount",
                    observation="`discount` is missing in 42% of rows.",
                    report_limitation="Discount-related conclusions may not generalize.",
                )
            ],
        ).model_dump(mode="json"),
    )

    markdown = report_bundle_to_markdown(
        bundle, artifacts=[context_artifact, context_artifact]
    )
    limitations = _section_text(markdown, "## Limitations and Risks")

    assert limitations.count("Discount-related conclusions may not generalize") == 1


def test_export_report_html_renders_synthesized_sections_like_markdown() -> None:
    # P0 parity: HTML must consume the same synthesized/scrubbed/formatted
    # section bodies as the markdown exporter (appendix keeps its ids in both).
    bundle = _bundle_with_empty_limitations_section()
    findings = next(s for s in bundle.sections if s.title == "Business Findings")
    findings.body = (
        "Fraud shares differ across 96476 rows (evidence: ds_b2bed9f7eb14)."
    )
    appendix = next(s for s in bundle.sections if s.title.startswith("Appendix"))
    appendix.body = "Chart inventory:\n- `chart_abc123def456` Fraud rate by class."
    context_artifact = Artifact(
        id="qctx_1",
        type=ArtifactType.QUALITY_CONTEXT_SET,
        project_id="project_demo",
        session_id="run_demo",
        payload=QualityContextSet(
            dataset_id="ds_sales",
            dataset_name="sales.csv",
            contexts=[
                QualityContext(
                    context_id="ctx_1",
                    dataset_id="ds_sales",
                    dataset_name="sales.csv",
                    issue_code="high_missing",
                    severity="warn",
                    column="discount",
                    observation="`discount` is missing in 42% of rows.",
                    report_limitation=(
                        "Discount-related conclusions may not generalize: 42% of "
                        "rows lack a discount value."
                    ),
                )
            ],
        ).model_dump(mode="json"),
    )

    html = export_report_html(bundle, artifacts=[context_artifact])
    markdown = report_bundle_to_markdown(bundle, artifacts=[context_artifact])

    # Body-less structured Limitations entries render in HTML, not the fallback.
    limitations_html = _html_section_text(html, "Limitations and Risks")
    assert "Discount-related conclusions may not generalize" in limitations_html
    assert "No validated conclusion is available" not in limitations_html
    # Internal ids are scrubbed and numbers formatted outside the appendix.
    narrative_html, _, _ = html.partition("Claim Ledger")
    assert "ds_b2bed9f7eb14" not in narrative_html
    assert "96,476" in narrative_html
    # The appendix keeps its artifact ids in both exporters.
    assert "chart_abc123def456" in html
    assert "chart_abc123def456" in markdown
    # Both exporters render the same set of sections.
    for section in bundle.sections:
        assert f"## {section.title}" in markdown
        assert f"<h2>{section.title}</h2>" in html


def test_export_report_html_scrubs_internal_ids_from_claim_text() -> None:
    bundle = _bundle()
    bundle.sections[0].claims[0].text = (
        "The fraud rate is 0.5 (evidence: ds_fef572615a7e)."
    )

    html = export_report_html(bundle)
    narrative, _, _ = html.partition("Claim Ledger")

    assert "ds_fef572615a7e" not in narrative
    assert "The fraud rate is 0.5" in narrative


def _html_section_text(html: str, title: str) -> str:
    _before, _, rest = html.partition(f"<h2>{title}</h2>")
    body, _, _after = rest.partition("<h2>")
    return body


def _bundle_with_empty_limitations_section() -> ReportBundle:
    bundle = ReportBundle.empty(project_id="project_demo", session_id="run_demo")
    for section in bundle.sections:
        section.body = "No validated conclusion is available for this section."
    return bundle


def _section_text(markdown: str, heading: str) -> str:
    _before, _, rest = markdown.partition(heading)
    body, _, _after = rest.partition("\n## ")
    return body


def _bundle() -> ReportBundle:
    bundle = ReportBundle.empty(project_id="project_demo", session_id="run_demo")
    bundle.status = ReportStatus.VALIDATED
    for section in bundle.sections:
        section.body = f"{section.title} body."
    bundle.sections[0].claims.append(
        ReportClaim(
            id="claim_1",
            text="Revenue is 120.",
            evidence=[
                EvidenceRef(
                    kind="stat",
                    artifact_id="table_1",
                    locator="rows[0]",
                    value=120,
                )
            ],
            referenced_datasets=["sales.csv"],
            referenced_columns=["revenue"],
        )
    )
    return bundle
