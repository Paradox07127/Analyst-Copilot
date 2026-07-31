"""A1: thousands-separated tokens survive format_numbers_in_text.
A3: exporters disclose the semantic-gate verdict next to the bundle status."""

from __future__ import annotations

import pytest

from eda_platform.schemas.reports import (
    ReportAudit,
    ReportBundle,
    ReportStatus,
)
from eda_platform.tools.exporter import (
    format_numbers_in_text,
    report_bundle_to_markdown,
)
from eda_platform.tools.html_exporter import export_report_html

# --------------------------------------------------------------------------- #
# A1 — thousands-separated numbers are single tokens, never re-split
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        "Across 1,000 rows",
        "sum 1,020",
        "revenue 1,000,000 usd",
        "sellers 3,095",
        "geolocation (1,000,163)",
        "growth of 1,020%",
    ],
)
def test_grouped_numbers_pass_through_unchanged(text: str) -> None:
    assert format_numbers_in_text(text) == text


def test_bare_numbers_still_format_normally() -> None:
    assert format_numbers_in_text("30950 rows") == "30,950 rows"
    # Below THOUSANDS_MIN_ABS bare integers stay ungrouped (frozen behavior:
    # years like 2018 must not become 2,018).
    assert format_numbers_in_text("3095 rows") == "3095 rows"
    assert format_numbers_in_text("2018 orders") == "2018 orders"


def test_decimals_scale_with_magnitude() -> None:
    """Decimals follow SIGNIFICANT_DIGITS, not a fixed 4 places.

    This test previously asserted "rate 8.1129%" survived verbatim, which is
    the reading complaint it was meant to guard against: four decimals is only
    right for one magnitude band. Counts stay exact (see the integer branch).
    """
    assert format_numbers_in_text("rate 8.1129%") == "rate 8.11%"
    assert format_numbers_in_text("takes 12.497345 days") == "takes 12.5 days"
    # Small magnitudes keep precision instead of collapsing to 0.
    assert format_numbers_in_text("corr 0.0233") == "corr 0.0233"
    assert format_numbers_in_text("p 0.0000123") == "p 0.0000123"
    # A grouped decimal keeps its separators; only the tail is bounded.
    assert format_numbers_in_text("average 1,234.5 units") == "average 1,234.5 units"


def test_rounding_never_turns_a_non_integer_into_a_whole_number() -> None:
    """Shortening may not manufacture a stronger claim than the data supports.

    Rendering 99.9987% as "100%" asserts a perfection the data denies, 0.99991
    as "1" reads as a deterministic identity, and collapsing 1,500.5 to 1,500
    made one sentence say 1,500 exceeds 1,500.
    """
    assert (
        format_numbers_in_text("on_time_rate is 99.9987%") == "on_time_rate is 99.999%"
    )
    assert format_numbers_in_text("correlation is 0.99991") == "correlation is 0.9999"
    assert (
        format_numbers_in_text("Median 1,500.5 exceeds the 1,500 threshold")
        == "Median 1,500.5 exceeds the 1,500 threshold"
    )
    # And the grouping boundary stays consistent within one sentence.
    assert (
        format_numbers_in_text("9999.99 usd vs 10000 usd")
        == "9999.99 usd vs 10,000 usd"
    )


def test_protected_spans_keep_grouped_numbers_verbatim() -> None:
    text = 'cited "value 1,020" and `col_3,095` stay verbatim'
    assert format_numbers_in_text(text) == text


# --------------------------------------------------------------------------- #
# A3 — gate verdict rendered next to Status
# --------------------------------------------------------------------------- #


def _validated_bundle(audit: ReportAudit | None) -> ReportBundle:
    bundle = ReportBundle.empty(project_id="project_demo", session_id="run_demo")
    bundle.status = ReportStatus.VALIDATED
    bundle.audit = audit
    return bundle


def test_markdown_status_line_discloses_degraded_gate() -> None:
    bundle = _validated_bundle(
        ReportAudit(status=ReportStatus.VALIDATED, gate_verdict="degraded")
    )
    markdown = report_bundle_to_markdown(bundle)
    assert "Status: validated · Gate: degraded" in markdown


def test_markdown_status_line_omits_gate_when_pass_or_absent() -> None:
    passing = _validated_bundle(
        ReportAudit(status=ReportStatus.VALIDATED, gate_verdict="pass")
    )
    assert "Gate:" not in report_bundle_to_markdown(passing)

    no_audit = _validated_bundle(None)
    assert "Gate:" not in report_bundle_to_markdown(no_audit)


def test_html_status_line_discloses_degraded_gate() -> None:
    bundle = _validated_bundle(
        ReportAudit(status=ReportStatus.VALIDATED, gate_verdict="degraded")
    )
    html = export_report_html(bundle)
    assert "Status: validated · Gate: degraded" in html

    passing = _validated_bundle(
        ReportAudit(status=ReportStatus.VALIDATED, gate_verdict="pass")
    )
    assert "Gate:" not in export_report_html(passing)
