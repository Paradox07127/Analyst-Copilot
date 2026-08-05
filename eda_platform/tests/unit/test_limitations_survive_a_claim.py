"""One surviving claim must not delete the data-quality inventory.

2026-08-04, comparing the two FIFA runs: run 1's Limitations section carried a
24-line quality inventory and run 2's carried a single sentence about a 38-day
window. The datasets were byte-identical. The difference was that run 2 had one
LLM claim survive into that section, and the synthesis only ran for a section
with no claims at all -- so 11 high-missing columns, 57 outlier columns and 3
entirely empty columns silently left the report.

This is under-reporting, which is worse than the verbosity the aggregation
rules were written to fix.
"""

from __future__ import annotations

from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.quality_context import QualityContext, QualityContextSet
from eda_platform.schemas.reports import ReportBundle, ReportClaim, ReportSection
from eda_platform.tools.exporter import report_bundle_to_markdown

_SECTIONS = (
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
)

_EMPTY_BODY = "No validated conclusion is available for this section."


def _context(index: int, *, column: str) -> QualityContext:
    return QualityContext(
        context_id=f"ctx_{index}",
        dataset_id="ds_x",
        dataset_name="x.csv",
        issue_code="high_missing",
        severity="warn",
        column=column,
        observation=f"{column} shows the high_missing condition.",
        report_limitation=(
            f"Interpretation involving {column} should account for the observed "
            "high_missing condition; its business cause remains unconfirmed."
        ),
    )


def _context_artifact() -> Artifact:
    return Artifact(
        id="qualityctx_240544bab75d",
        type=ArtifactType.QUALITY_CONTEXT_SET,
        project_id="project_demo",
        session_id="run_demo",
        payload=QualityContextSet(
            dataset_id="ds_x",
            dataset_name="x.csv",
            contexts=[_context(index, column=f"c{index}") for index in range(1, 4)],
        ).model_dump(mode="json"),
    )


def _bundle(*, limitations_claim: bool) -> ReportBundle:
    sections = []
    for title in _SECTIONS:
        claims = []
        body = _EMPTY_BODY
        if title == "Limitations and Risks" and limitations_claim:
            claims = [
                ReportClaim(
                    id="c_feature_time_span",
                    text=(
                        "The match_prediction_features date coverage spans 38 days, "
                        "limiting conclusions about longer-term form trends."
                    ),
                )
            ]
            body = "Validated evidence-backed findings are listed below."
        sections.append(ReportSection(title=title, body=body, claims=claims))
    return ReportBundle(
        project_id="project_demo",
        session_id="run_demo",
        sections=sections,
        status="validated",
    )


def _limitations(markdown: str) -> str:
    body = markdown.split("## Limitations and Risks", 1)[1]
    return body.split("\n## ", 1)[0]


def test_the_quality_inventory_survives_a_surviving_claim() -> None:
    section = _limitations(
        report_bundle_to_markdown(
            _bundle(limitations_claim=True), artifacts=[_context_artifact()]
        )
    )
    for index in range(1, 4):
        assert f"Interpretation involving c{index}" in section, section
    assert "38 days" in section, section


def test_a_claimless_section_is_unchanged() -> None:
    """The path that already worked must keep working."""
    section = _limitations(
        report_bundle_to_markdown(
            _bundle(limitations_claim=False), artifacts=[_context_artifact()]
        )
    )
    for index in range(1, 4):
        assert f"Interpretation involving c{index}" in section, section
    assert _EMPTY_BODY not in section, section
