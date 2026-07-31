from __future__ import annotations

from eda_platform.schemas.artifacts import EvidenceRef
from eda_platform.schemas.reports import (
    ReportBundle,
    ReportClaim,
    ReportFocusItem,
    ReportSection,
    ReportStatus,
    merge_duplicate_sections,
    required_report_sections,
)


def test_report_bundle_requires_fixed_m2_sections() -> None:
    names = required_report_sections()

    assert names == [
        "Executive Summary",
        "Dataset Overview",
        "File-by-File EDA Summary",
        "Data Quality Findings",
        "Key EDA Insights",
        "Selected Analysis Focus",
        "Agent-Performed Analysis",
        "Business Findings",
        "Business Recommendations",
        "Limitations and Risks",
        "Appendix: Charts and Technical Summary",
    ]

    bundle = ReportBundle.empty(
        project_id="project_demo",
        session_id="run_demo",
    )

    assert bundle.status is ReportStatus.DRAFT
    assert [section.title for section in bundle.sections] == names
    assert all(section.claims == [] for section in bundle.sections)


def test_report_claim_keeps_evidence_and_numeric_values_typed() -> None:
    claim = ReportClaim(
        text="Revenue is 120.5 in the sample.",
        evidence=[
            EvidenceRef(
                kind="stat",
                artifact_id="table_abc",
                locator="rows[0].revenue",
                value=120.5,
            )
        ],
        referenced_datasets=["sales.csv"],
        referenced_columns=["revenue"],
    )

    assert claim.evidence[0].artifact_id == "table_abc"
    assert claim.evidence[0].value == 120.5
    assert claim.referenced_datasets == ["sales.csv"]
    assert claim.referenced_columns == ["revenue"]


def test_merge_collapses_duplicate_titled_sections() -> None:
    sections = [
        ReportSection(
            title="Selected Analysis Focus",
            body="First body.",
            claims=[ReportClaim(id="a", text="First focus.")],
        ),
        ReportSection(
            title="Selected Analysis Focus",
            body="Second body (dropped).",
            claims=[
                ReportClaim(id="a", text="Duplicate id, dropped."),
                ReportClaim(id="b", text="Second focus."),
            ],
        ),
    ]

    merged = merge_duplicate_sections(sections)

    titles = [section.title for section in merged]
    assert titles == ["Selected Analysis Focus"]
    focus = merged[0]
    # First occurrence's body wins; claims merge, deduped by id, order preserved.
    assert focus.body == "First body."
    assert [claim.id for claim in focus.claims] == ["a", "b"]
    assert focus.claims[0].text == "First focus."


def test_merge_duplicate_sections_keeps_empty_id_claims() -> None:
    sections = [
        ReportSection(title="X", claims=[ReportClaim(text="unlabeled one")]),
        ReportSection(title="X", claims=[ReportClaim(text="unlabeled two")]),
    ]

    merged = merge_duplicate_sections(sections)

    assert len(merged) == 1
    # Empty-id claims cannot be deduped, so both are kept.
    assert [claim.text for claim in merged[0].claims] == ["unlabeled one", "unlabeled two"]


def test_merge_duplicate_sections_dedupes_claims_within_a_single_section() -> None:
    # Bundles persisted by older code carry the same injected claim twice inside
    # ONE section (append-only repair rounds) — the id filter must apply there too.
    sections = [
        ReportSection(
            title="Selected Analysis Focus",
            claims=[
                ReportClaim(id="qfocus_1", text="Focus one."),
                ReportClaim(id="qfocus_2", text="Focus two."),
                ReportClaim(id="qfocus_1", text="Focus one (older duplicate)."),
                ReportClaim(id="qfocus_2", text="Focus two (older duplicate)."),
            ],
        ),
    ]

    merged = merge_duplicate_sections(sections)

    assert len(merged) == 1
    assert [claim.id for claim in merged[0].claims] == ["qfocus_1", "qfocus_2"]
    assert merged[0].claims[0].text == "Focus one."


def test_merge_duplicate_sections_merges_and_dedupes_focus_items() -> None:
    # F4: structured focus entries survive the merge, deduped by question_id.
    sections = [
        ReportSection(
            title="Selected Analysis Focus",
            focus_items=[
                ReportFocusItem(question="Q one?", outcome="answered", question_id="q1"),
                ReportFocusItem(question="Q one (dup)?", outcome="answered", question_id="q1"),
            ],
        ),
        ReportSection(
            title="Selected Analysis Focus",
            focus_items=[
                ReportFocusItem(question="Q one?", outcome="answered", question_id="q1"),
                ReportFocusItem(question="Q two?", outcome="failed", question_id="q2"),
            ],
        ),
    ]

    merged = merge_duplicate_sections(sections)

    assert len(merged) == 1
    assert [item.question_id for item in merged[0].focus_items] == ["q1", "q2"]
    assert merged[0].focus_items[1].outcome == "failed"
