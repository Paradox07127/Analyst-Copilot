"""The narrative layer may connect validated claims; it may not add facts.

A report whose prose is generated has to be trustworthy in exactly one way:
every figure in it already survived the numeric gate as part of a claim. The
narrator therefore reorders and joins; a paragraph carrying any other number is
discarded, and the section falls back to its bullets.
"""

from __future__ import annotations

from typing import Any

import pytest

from eda_platform.agents.narrator import borrowed_numbers_only, narrate_report
from eda_platform.schemas.artifacts import EvidenceRef
from eda_platform.schemas.reports import ReportBundle, ReportClaim


def _claim(claim_id: str, text: str) -> ReportClaim:
    return ReportClaim(
        id=claim_id,
        text=text,
        evidence=[
            EvidenceRef(kind="sql", artifact_id=f"sql_{claim_id}", locator="rows[0]")
        ],
        confidence="high",
    )


_CLAIMS = [
    _claim("c1", "Late deliveries reached 8.2% of orders in Q4."),
    _claim("c2", "Average delivery time was 12.5 days."),
]


class _FakeLLM:
    """Returns a canned narrative; records the payload it was handed."""

    def __init__(self, text: str, cited: list[str]) -> None:
        self._text = text
        self._cited = cited
        self.payloads: list[dict[str, Any]] = []

    def structured(self, *, task: str, schema: type, payload: dict) -> Any:
        self.payloads.append(payload)
        return schema.model_validate(
            {"text": self._text, "cited_claim_ids": self._cited}
        )


def _bundle() -> ReportBundle:
    bundle = ReportBundle.empty(project_id="p", session_id="s")
    for section in bundle.sections:
        if section.title == "Business Findings":
            section.claims.extend(_CLAIMS)
    return bundle


# --------------------------------------------------------------------------- #
# 1. The number guard
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "prose",
    [
        "Late deliveries reached 8.2% of orders while delivery took 12.5 days.",
        "Delivery ran to 12.5 days; 8.2% of orders arrived late.",
        "Both findings point the same way, with no figure of their own.",
    ],
)
def test_prose_reusing_only_claim_figures_is_accepted(prose: str) -> None:
    assert borrowed_numbers_only(prose, _CLAIMS)


@pytest.mark.parametrize(
    "prose",
    [
        # An average the model computed instead of quoting.
        "Late deliveries reached 8.2%, costing roughly 340 orders a week.",
        # A plausible rounding is still a new number.
        "Delivery took about 13 days on average.",
        # A rate nobody measured.
        "Late deliveries reached 8.2% of orders, up 3.1 points year on year.",
    ],
)
def test_prose_introducing_a_new_figure_is_rejected(prose: str) -> None:
    assert not borrowed_numbers_only(prose, _CLAIMS)


def test_a_figure_inside_a_quoted_column_name_is_not_a_new_number() -> None:
    # Column and table names travel in backticks; the gate reads prose, not ids.
    assert borrowed_numbers_only("The `top_90_pct` column drives this.", _CLAIMS)


# --------------------------------------------------------------------------- #
# 2. Narration end to end
# --------------------------------------------------------------------------- #
def test_a_clean_narrative_is_stored_with_its_citations() -> None:
    bundle = _bundle()
    llm = _FakeLLM(
        "Late deliveries reached 8.2% of orders and delivery took 12.5 days.",
        ["c1", "c2"],
    )
    narrated = narrate_report(bundle, llm=llm).written

    section = next(s for s in bundle.sections if s.title == "Business Findings")
    assert narrated == 1
    assert section.narrative.startswith("Late deliveries reached 8.2%")
    # Citations are appended by us from the claims' own evidence, never authored
    # by the model.
    assert "`sql_c1`" in section.narrative
    assert "`sql_c2`" in section.narrative


def test_a_narrative_with_an_invented_number_is_discarded() -> None:
    bundle = _bundle()
    llm = _FakeLLM("Late deliveries cost roughly 340 orders a week.", ["c1"])

    assert narrate_report(bundle, llm=llm).written == 0
    section = next(s for s in bundle.sections if s.title == "Business Findings")
    assert section.narrative == ""


def test_a_rejection_is_counted_so_silence_has_a_cause() -> None:
    """Without this, "no prose anywhere" reads the same as "never ran".

    A run whose guard rejects everything and a run whose model was unreachable
    both produce a bullet-only report; only the counts tell them apart, and
    that is the first question to ask of a live run.
    """
    bundle = _bundle()
    llm = _FakeLLM("Late deliveries cost roughly 340 orders a week.", ["c1"])
    result = narrate_report(bundle, llm=llm)

    assert result.written == 0
    assert result.rejected == 1
    # Sections with fewer than two claims were never sent, so they are neither.
    assert result.skipped > 0


def test_a_citation_naming_an_unknown_claim_is_discarded() -> None:
    bundle = _bundle()
    llm = _FakeLLM("Both findings point the same way.", ["c1", "c99"])

    assert narrate_report(bundle, llm=llm).written == 0


def test_a_single_claim_section_is_not_narrated() -> None:
    # One bullet needs no connective prose; narrating it only restates it.
    bundle = ReportBundle.empty(project_id="p", session_id="s")
    next(s for s in bundle.sections if s.title == "Business Findings").claims.append(
        _CLAIMS[0]
    )
    llm = _FakeLLM("Late deliveries reached 8.2% of orders.", ["c1"])

    assert narrate_report(bundle, llm=llm).written == 0
    assert llm.payloads == []


def test_the_model_is_shown_claims_and_told_not_to_compute() -> None:
    bundle = _bundle()
    llm = _FakeLLM("Both findings point the same way.", ["c1", "c2"])
    narrate_report(bundle, llm=llm)

    payload = llm.payloads[0]
    assert [claim["id"] for claim in payload["claims"]] == ["c1", "c2"]
    assert "8.2%" in payload["claims"][0]["text"]
    instructions = payload["instructions"].lower()
    assert "do not" in instructions and "number" in instructions


def test_an_offline_client_narrates_nothing() -> None:
    bundle = _bundle()
    assert narrate_report(bundle, llm=None).written == 0
    section = next(s for s in bundle.sections if s.title == "Business Findings")
    assert section.narrative == ""


def test_section_narrative_renders_above_the_bullets() -> None:
    from eda_platform.tools.exporter import report_bundle_to_markdown

    bundle = _bundle()
    section = next(s for s in bundle.sections if s.title == "Business Findings")
    section.narrative = "Both findings point the same way. (evidence: `sql_c1`)"
    markdown = report_bundle_to_markdown(bundle)

    body = markdown.split("## Business Findings")[1].split("## ")[0]
    assert body.index("Both findings point the same way") < body.index("Late deliveries")


def test_a_narrative_citation_survives_the_id_scrub_that_strips_claim_prose() -> None:
    """The one deliberate exception to "no internal ids in the narrative".

    exporter.py strips artifact ids from claim text because a claim that names
    its own evidence inline reads as hash soup. A narrative citation is the
    opposite case: it is the only handle the reader has for jumping from a
    sentence to the evidence under it, and the frontend turns it into a button.
    """
    from eda_platform.tools.exporter import report_bundle_to_markdown

    bundle = _bundle()
    section = next(s for s in bundle.sections if s.title == "Business Findings")
    # A claim naming its evidence inline is still scrubbed.
    section.claims.append(
        _claim("c3", "Returns rose (evidence: `sql_c3`) across every region.")
    )
    section.narrative = "The two move together. (evidence: `sql_c1`, `sql_c2`)"
    markdown = report_bundle_to_markdown(bundle)
    body = markdown.split("## Business Findings")[1].split("## ")[0]

    assert "`sql_c1`" in body and "`sql_c2`" in body
    assert "(evidence: `sql_c3`)" not in body
    assert "Returns rose" in body


def test_model_prose_cannot_fabricate_report_structure() -> None:
    # The narrative is model output rendered into a markdown document whose
    # headings drive the table of contents and the narrative/reference split.
    # A "#" or a "[" from the model must stay punctuation.
    bundle = _bundle()
    llm = _FakeLLM(
        "# Executive Summary\nSee [the ledger](http://evil) and `sql_forged`.",
        ["c1", "c2"],
    )
    narrate_report(bundle, llm=llm)
    section = next(s for s in bundle.sections if s.title == "Business Findings")

    assert not section.narrative.startswith("#")
    assert "[the ledger]" not in section.narrative
    assert "`sql_forged`" not in section.narrative
    # Our own citation is still a real code span the app can turn into a button.
    assert "(evidence: `sql_c1`, `sql_c2`)" in section.narrative


def test_a_narrative_renders_its_numbers_like_the_bullets_below_it() -> None:
    """2026-08-05 credit-card run: 0.08822511669373306 above 0.0882.

    Claim text is stored raw and formatted at render; the narrative was written
    from that raw text and then rendered as-is, so a business reader met a
    seventeen-digit float directly above the same figure written properly.
    The trailing evidence ids sit in code spans and must survive untouched.
    """
    from eda_platform.schemas.reports import ReportBundle
    from eda_platform.tools.exporter import report_bundle_to_markdown

    bundle = ReportBundle.empty(project_id="project_demo", session_id="run_demo")
    section = next(
        item for item in bundle.sections if item.title == "Executive Summary"
    )
    section.narrative = (
        "The t-test reported a p-value of 0.08822511669373306 over 568630 "
        "observations. (evidence: `sql_0cf104a021c8`)"
    )
    markdown = report_bundle_to_markdown(bundle)

    assert "0.08822511669373306" not in markdown
    assert "0.0882" in markdown
    assert "568,630" in markdown
    assert "`sql_0cf104a021c8`" in markdown


def test_html_and_markdown_narratives_agree() -> None:
    from eda_platform.schemas.reports import ReportBundle
    from eda_platform.tools.html_exporter import export_report_html

    bundle = ReportBundle.empty(project_id="project_demo", session_id="run_demo")
    section = next(
        item for item in bundle.sections if item.title == "Executive Summary"
    )
    section.narrative = "An HHI of 0.5000008438758905 across 2 groups."

    html = export_report_html(bundle)
    assert "0.5000008438758905" not in html
    assert "0.5" in html
