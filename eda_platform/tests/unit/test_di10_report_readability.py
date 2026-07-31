"""DI10 W3 report readability.

Covers: limitations same-code aggregation (threshold, correct column count,
critical never hidden), Executive Summary selection rules (shape claims banned,
score-ordered business findings, legacy fallback), same-finding render dedup
(one full statement + cross-references, ledger lossless, Executive Summary
exempt), internal-id scrubbing of the rendered narrative, the pure number
formatters, and validator-triggered forced evidence interleave on
numeric_mismatch. All LLM calls are mocked.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, TypeVar, cast

from pydantic import BaseModel

from eda_platform.agents.reporting import (
    _apply_executive_summary_fallback,
    _forced_interleave_note,
    generate_agentic_report,
)
from eda_platform.schemas.artifacts import Artifact, ArtifactType, EvidenceRef, SqlResult
from eda_platform.schemas.quality_context import QualityContext, QualityContextSet
from eda_platform.schemas.questions import (
    FindingScore,
    QuestionExecutionResult,
    QuestionFinding,
)
from eda_platform.schemas.reports import (
    EvidenceGrant,
    EvidenceRejection,
    EvidenceRequest,
    GrantValue,
    InterleaveExchange,
    ReportBundle,
    ReportClaim,
    ReportPlanClaim,
    ReportPlanDraft,
)
from eda_platform.tools.exporter import (
    _format_analysis_row,
    format_number,
    format_numbers_in_text,
    narrative_markdown,
    report_bundle_to_markdown,
    scrub_internal_ids,
)
from eda_platform.tools.loader import load_csv
from eda_platform.tools.profiler import profile_dataset

T = TypeVar("T", bound=BaseModel)

_INTERNAL_ID_RE = re.compile(r"\b(?:qualityctx|sql|ds|chart)_[0-9a-f]{6,}\b")
_EMPTY_BODY = "No validated conclusion is available for this section."


class ScriptedPlanLLM:
    """Returns pre-built ReportPlanDraft objects and records payloads."""

    def __init__(self, plans: list[ReportPlanDraft]) -> None:
        self.plans = plans
        self.payloads: list[dict[str, Any]] = []
        self.calls = 0

    def structured(self, *, task: str, schema: type[T], payload: dict[str, Any]) -> T:
        self.payloads.append(payload)
        plan = self.plans[min(self.calls, len(self.plans) - 1)]
        self.calls += 1
        return cast(T, plan)

    def text(self, *, task: str, payload: dict[str, Any]) -> str:
        return "scripted"

    def last_usage(self) -> None:
        return None


def _empty_bundle() -> ReportBundle:
    bundle = ReportBundle.empty(project_id="project_demo", session_id="run_demo")
    for section in bundle.sections:
        section.body = _EMPTY_BODY
    return bundle


def _section(bundle: ReportBundle, title: str):
    return next(section for section in bundle.sections if section.title == title)


def _section_text(markdown: str, heading: str) -> str:
    start = markdown.index(heading)
    end = markdown.find("\n## ", start + len(heading))
    return markdown[start : end if end != -1 else len(markdown)]


def _quality_context(
    index: int,
    *,
    code: str,
    severity: str = "warn",
    column: str | None = None,
) -> QualityContext:
    return QualityContext(
        context_id=f"ctx_{code}_{index}",
        dataset_id="ds_x",
        dataset_name="x.csv",
        issue_code=code,
        severity=severity,  # type: ignore[arg-type]
        column=column,
        observation=f"{column or 'dataset'} shows the {code} condition.",
        report_limitation=(
            f"Interpretation involving {column or 'the dataset'} should account for "
            f"the observed {code} condition; its business cause remains unconfirmed."
        ),
    )


def _context_artifact(contexts: list[QualityContext]) -> Artifact:
    return Artifact(
        id="qualityctx_240544bab75d",
        type=ArtifactType.QUALITY_CONTEXT_SET,
        project_id="project_demo",
        session_id="run_demo",
        payload=QualityContextSet(
            dataset_id="ds_x", dataset_name="x.csv", contexts=contexts
        ).model_dump(mode="json"),
    )


def _profile_artifact(tmp_path: Path) -> Artifact:
    csv_path = tmp_path / "x.csv"
    csv_path.write_text(
        "c1,c2,c3,c4,c5,c6,c7,c8\n1,2,3,4,5,6,7,8\n9,10,11,12,13,14,15,16\n",
        encoding="utf-8",
    )
    loaded = load_csv(csv_path, dataset_id="ds_x")
    return profile_dataset(loaded, project_id="project_demo", session_id="run_demo")


# --------------------------------------------------------------------------- #
# 1. Limitations aggregation
# --------------------------------------------------------------------------- #


def test_limitations_aggregate_more_than_three_same_code_warnings(tmp_path: Path) -> None:
    contexts = [
        _quality_context(i, code="outlier_detected", column=f"c{i}") for i in range(1, 6)
    ]
    markdown = report_bundle_to_markdown(
        _empty_bundle(), artifacts=[_profile_artifact(tmp_path), _context_artifact(contexts)]
    )
    limitations = _section_text(markdown, "## Limitations and Risks")

    # One aggregate sentence with the affected column count and the profile total.
    assert "Outliers were detected in 5 of 8 columns" in limitations
    assert limitations.count("Outliers were detected") == 1
    # The per-column dump is gone.
    assert "Interpretation involving c1" not in limitations
    assert "outlier_detected condition; its business cause" not in limitations
    # No internal evidence ids in the limitations prose.
    assert "evidence:" not in limitations
    assert not _INTERNAL_ID_RE.search(limitations)


def test_limitations_keep_at_most_three_same_code_warnings_itemized() -> None:
    contexts = [
        _quality_context(i, code="outlier_detected", column=f"c{i}") for i in range(1, 4)
    ]
    markdown = report_bundle_to_markdown(
        _empty_bundle(), artifacts=[_context_artifact(contexts)]
    )
    limitations = _section_text(markdown, "## Limitations and Risks")

    assert "Outliers were detected in" not in limitations
    for index in range(1, 4):
        assert f"Interpretation involving c{index}" in limitations


def test_limitations_aggregation_never_hides_critical_conditions() -> None:
    contexts = [
        *(_quality_context(i, code="high_missing", column=f"c{i}") for i in range(1, 6)),
        _quality_context(9, code="high_missing", severity="critical", column="c9"),
    ]
    markdown = report_bundle_to_markdown(
        _empty_bundle(), artifacts=[_context_artifact(contexts)]
    )
    limitations = _section_text(markdown, "## Limitations and Risks")

    # The 5 warn conditions aggregate; the critical one keeps its own line.
    assert "High missing rates affect 5 columns" in limitations
    assert "Interpretation involving c9" in limitations


# --------------------------------------------------------------------------- #
# 2. Executive Summary selection
# --------------------------------------------------------------------------- #


def _question(question_id: str, text: str, final: float) -> QuestionExecutionResult:
    return QuestionExecutionResult(
        question_id=question_id,
        question=f"Question {question_id}?",
        origin="template",
        status="succeeded",
        findings=[
            QuestionFinding(
                text=text,
                evidence=[EvidenceRef(kind="table", artifact_id="t1", locator="rows")],
                score=FindingScore(impact=1.0, significance=final, final=final),
            )
        ],
    )


def test_executive_summary_bans_shape_claims_and_orders_by_score() -> None:
    bundle = _empty_bundle()
    _section(bundle, "Dataset Overview").claims.extend(
        ReportClaim(
            id=f"dataset_overview_{index}",
            text=f"file_{index}.csv has {index * 100} rows and {index} columns.",
            evidence=[EvidenceRef(kind="stat", artifact_id="p1", locator="rows")],
        )
        for index in range(1, 4)
    )
    _section(bundle, "Agent-Performed Analysis").claims.append(
        ReportClaim(
            id="qfind_q2_0",
            text="Average freight value is 20.5 in the order items table.",
            evidence=[EvidenceRef(kind="table", artifact_id="t2", locator="rows")],
        )
    )
    _section(bundle, "Business Findings").claims.append(
        ReportClaim(
            id="qbiz_q1_0",
            text="473 orders were delivered to the carrier after the estimated date.",
            evidence=[EvidenceRef(kind="table", artifact_id="t1", locator="rows")],
        )
    )
    questions = [
        _question("q1", "473 orders were late.", 0.9),
        _question("q2", "Average freight value is 20.5.", 0.4),
    ]

    injected = _apply_executive_summary_fallback(bundle, questions)

    summary_texts = [claim.text for claim in _section(bundle, "Executive Summary").claims]
    assert injected == len(summary_texts) == 2
    # No row/column shape claim enters the summary.
    assert not any("rows and" in text for text in summary_texts)
    # Business claims enter ordered by FindingScore.final (0.9 before 0.4).
    assert "473 orders" in summary_texts[0]
    assert "20.5" in summary_texts[1]


def test_executive_summary_falls_back_when_no_qualified_claim_exists() -> None:
    bundle = _empty_bundle()
    _section(bundle, "Dataset Overview").claims.append(
        ReportClaim(
            id="dataset_overview_only",
            text="x.csv has 71 rows and 2 columns.",
            evidence=[EvidenceRef(kind="stat", artifact_id="p1", locator="rows")],
        )
    )

    injected = _apply_executive_summary_fallback(bundle, [])

    # Legacy behavior: with nothing qualified, the summary mirrors what exists.
    assert injected == 1
    assert "71 rows" in _section(bundle, "Executive Summary").claims[0].text


def test_executive_summary_guarantees_a_numeric_business_finding() -> None:
    bundle = _empty_bundle()
    _section(bundle, "Key EDA Insights").claims.extend(
        [
            ReportClaim(
                id="insight_wordy",
                text="Review scores are generally positive across regions.",
                evidence=[EvidenceRef(kind="artifact", artifact_id="a1", locator="x")],
            ),
            ReportClaim(
                id="insight_numeric",
                text="Late deliveries reached 473 orders in the observed period.",
                evidence=[EvidenceRef(kind="table", artifact_id="t1", locator="rows")],
            ),
        ]
    )

    _apply_executive_summary_fallback(bundle, [])

    summary_texts = [claim.text for claim in _section(bundle, "Executive Summary").claims]
    assert any("473" in text for text in summary_texts)


def test_executive_summary_prefers_question_findings_over_generic_stat_diagnostic() -> None:
    bundle = _empty_bundle()
    _section(bundle, "Agent-Performed Analysis").claims.extend(
        [
            ReportClaim(
                id="stat_test_available",
                text="one_way_anova ran with p-value 0.",
                evidence=[EvidenceRef(kind="stat", artifact_id="s1", locator="p_value")],
            ),
            ReportClaim(
                id="qfind_q1_0",
                text="Total GMV is 13,591,643.7.",
                evidence=[EvidenceRef(kind="sql", artifact_id="q1", locator="gmv")],
            ),
        ]
    )
    questions = [_question("q1", "What is GMV?", 0.9)]

    _apply_executive_summary_fallback(bundle, questions)

    summary_texts = [claim.text for claim in _section(bundle, "Executive Summary").claims]
    assert any("GMV" in text for text in summary_texts)
    assert not any("one_way_anova" in text for text in summary_texts)


# --------------------------------------------------------------------------- #
# 3. Same-finding render dedup
# --------------------------------------------------------------------------- #


def _late_claim(claim_id: str, text: str) -> ReportClaim:
    return ReportClaim(
        id=claim_id,
        text=text,
        evidence=[EvidenceRef(kind="table", artifact_id="t1", locator="rows")],
        referenced_columns=[],
    )


def test_same_finding_four_places_renders_one_full_statement() -> None:
    bundle = _empty_bundle()
    # Mirrors the real Olist run: two near-identical statements in
    # Agent-Performed Analysis, one in Business Findings (the most complete),
    # and one plain-prose restatement in Business Recommendations.
    _section(bundle, "Agent-Performed Analysis").claims.extend(
        [
            _late_claim(
                "qfind_q1_0",
                "473 orders had order_delivered_carrier_date after the estimate.",
            ),
            _late_claim("qfind_q1_1", "The returned value is 473."),
        ]
    )
    _section(bundle, "Business Findings").claims.append(
        _late_claim(
            "qbiz_q1_0",
            "The analysis identified 473 late orders whose "
            "order_delivered_carrier_date fell after the estimated delivery date.",
        )
    )
    _section(bundle, "Business Recommendations").claims.append(
        _late_claim(
            "rec_1", "Investigate the carrier stage, as 473 orders were delayed."
        )
    )

    markdown = report_bundle_to_markdown(bundle)
    narrative = narrative_markdown(markdown)

    # Exactly one full statement of the finding survives in the narrative.
    assert narrative.count("473") == 1
    assert "The analysis identified 473 late orders" in narrative
    # Other sections carry at most a short cross-reference.
    assert 'See "Business Findings" for the full statement' in narrative
    # The ledger keeps every claim (lossless traceability).
    ledger = markdown[markdown.index("### Claim Ledger") :]
    for claim_id in ("qfind_q1_0", "qfind_q1_1", "qbiz_q1_0", "rec_1"):
        assert claim_id in ledger


def test_executive_summary_is_exempt_from_render_dedup() -> None:
    bundle = _empty_bundle()
    _section(bundle, "Business Findings").claims.append(
        _late_claim("qbiz_q1_0", "473 orders were late at the carrier_stage handoff.")
    )
    _section(bundle, "Executive Summary").claims.append(
        _late_claim(
            "exec_summary_qbiz_q1_0",
            "Summary: 473 orders were late at the carrier_stage handoff.",
        )
    )

    narrative = narrative_markdown(report_bundle_to_markdown(bundle))

    # The summary repeats the finding on purpose; both renders survive.
    assert narrative.count("473") == 2


def test_same_number_different_evidence_metrics_are_not_deduplicated() -> None:
    bundle = _empty_bundle()
    _section(bundle, "File-by-File EDA Summary").claims.append(
        ReportClaim(
            id="dataset_column_count",
            text="translation.csv has 2 columns.",
            evidence=[
                EvidenceRef(kind="stat", artifact_id="profile_1", locator="columns", value=2)
            ],
            referenced_datasets=["translation.csv"],
        )
    )
    _section(bundle, "Data Quality Findings").claims.append(
        ReportClaim(
            id="quality_issue_count",
            text="translation.csv quality scan found 2 issues.",
            evidence=[
                EvidenceRef(
                    kind="artifact",
                    artifact_id="quality_1",
                    locator="issues",
                    value=2,
                )
            ],
            referenced_datasets=["translation.csv"],
        )
    )

    narrative = narrative_markdown(report_bundle_to_markdown(bundle))

    assert "translation.csv has 2 columns." in narrative
    assert "translation.csv quality scan found 2 issues." in narrative
    assert "full statement of this finding" not in narrative


# --------------------------------------------------------------------------- #
# 4. Internal-id scrubbing
# --------------------------------------------------------------------------- #


def test_rendered_narrative_contains_no_internal_ids() -> None:
    contexts = [
        _quality_context(i, code="outlier_detected", column=f"c{i}") for i in range(1, 6)
    ]
    bundle = _empty_bundle()
    _section(bundle, "Business Findings").claims.append(
        ReportClaim(
            id="qbiz_q1_0",
            text=(
                "Late orders reached 473 (evidence: `sql_938fabcdef12`) according "
                "to chart_7b66aa01cd23 over ds_78bfcc45de67."
            ),
            evidence=[EvidenceRef(kind="table", artifact_id="sql_938fabcdef12", locator="rows")],
        )
    )

    markdown = report_bundle_to_markdown(bundle, artifacts=[_context_artifact(contexts)])
    narrative = narrative_markdown(markdown)
    # The Appendix is the technical index and legitimately keeps artifact ids.
    body = narrative.split("## Appendix", 1)[0]

    assert not _INTERNAL_ID_RE.search(body)
    assert "Late orders reached 473" in body
    # Full provenance is preserved in the ledger.
    assert "sql_938fabcdef12" in markdown[markdown.index("### Claim Ledger") :]


def test_scrub_internal_ids_pure_function() -> None:
    assert (
        scrub_internal_ids("Total late 473 (evidence: `qualityctx_240544bab75d`).")
        == "Total late 473."
    )
    assert "sql_938fabcdef12" not in scrub_internal_ids("See sql_938fabcdef12 for detail.")
    # Non-hex suffixes (real column names) are never scrubbed.
    assert scrub_internal_ids("Column ds_total_sales matters.") == (
        "Column ds_total_sales matters."
    )


# --------------------------------------------------------------------------- #
# 5. Number formatting
# --------------------------------------------------------------------------- #


def test_format_number_branches() -> None:
    assert format_number(1_000_163) == "1,000,163"
    assert format_number(2018) == "2018"  # years stay ungrouped
    assert format_number(88.34961925095455) == "88.3"
    assert format_number(0.5) == "0.5"
    # Not "1,234,568": shortening never manufactures a whole number.
    assert format_number(1234567.89) == "1,234,567.9"
    assert format_number(0.0812, ratio_as_percent=True) == "8.12%"
    assert format_number(0.5, ratio_as_percent=True) == "50%"
    assert format_number(1.0, ratio_as_percent=True) == "100%"
    assert format_number(1.5, ratio_as_percent=True) == "1.5"  # out of 0..1 range


def test_format_numbers_in_text_preserves_citations_and_code() -> None:
    text = (
        'Correlation is 0.9536211199 over 1000163 rows; "quoted 1000163" and '
        "`col_88.34961925095455` stay verbatim."
    )
    formatted = format_numbers_in_text(text)
    assert "0.954 over 1,000,163 rows" in formatted
    assert '"quoted 1000163"' in formatted
    assert "`col_88.34961925095455`" in formatted


def test_analysis_row_renders_ratio_fields_as_percent() -> None:
    rendered = _format_analysis_row(
        {"late_rate": 0.0811, "orders": 12345, "city": "SP", "year": 2018}
    )
    assert "'late_rate': '8.11%'" in rendered
    assert "'orders': '12,345'" in rendered
    assert "'city': 'SP'" in rendered
    assert "'year': '2018'" in rendered


# --------------------------------------------------------------------------- #
# 6. Forced evidence interleave on numeric_mismatch
# --------------------------------------------------------------------------- #


def test_numeric_mismatch_triggers_forced_interleave_with_granted_values(
    tmp_path: Path,
) -> None:
    sql_artifact = Artifact(
        id="sql_late_orders",
        type=ArtifactType.SQL_RESULT,
        project_id="project_demo",
        session_id="run_demo",
        payload=SqlResult(
            sql="SELECT count(*) AS late_rows FROM orders",
            columns=["late_rows"],
            dtypes={"late_rows": "bigint"},
            rows_preview=[{"late_rows": 473}],
            row_count=1,
        ).model_dump(mode="json"),
    )
    artifacts = [_profile_artifact(tmp_path), sql_artifact]

    wrong_plan = ReportPlanDraft(
        claims=[
            ReportPlanClaim(
                section_title="Business Findings",
                id="late",
                text="The returned late_rows is 999.",
                evidence=[
                    EvidenceRef(
                        kind="table",
                        artifact_id=sql_artifact.id,
                        locator="rows[0].late_rows",
                    )
                ],
            )
        ],
    )
    fixed_plan = ReportPlanDraft(
        claims=[
            ReportPlanClaim(
                section_title="Business Findings",
                id="late",
                text="The returned late_rows is 473.",
                evidence=[
                    EvidenceRef(
                        kind="table",
                        artifact_id=sql_artifact.id,
                        locator="rows[0].late_rows",
                    )
                ],
            )
        ],
    )
    llm = ScriptedPlanLLM([wrong_plan, fixed_plan])

    result = generate_agentic_report(
        artifacts,
        project_id="project_demo",
        session_id="run_demo",
        business_context="Delivery analysis",
        llm=llm,
    )

    # The retry payload carries the deterministically granted value.
    assert llm.calls == 2
    repair_payload = llm.payloads[1]
    granted = repair_payload["granted_repair_evidence"]
    assert granted, "forced interleave must inject resolved evidence on retry"
    assert granted[0]["artifact_id"] == sql_artifact.id
    assert any(item["value"] == 473.0 for item in granted[0]["values"])
    assert "granted_repair_instructions" in repair_payload
    # Metering: the forced channel is counted and disclosed.
    assert result.forced_evidence_requests == 1
    assert any(
        "Forced evidence interleave" in note for note in result.audit.semantic_notes
    )
    transcript = result.interleave_transcript
    assert transcript is not None
    assert any(exchange.section == "numeric_repair" for exchange in transcript.exchanges)
    # The corrected claim survived the hard gate.
    claims = [
        claim
        for section in result.bundle.sections
        if section.title == "Business Findings"
        for claim in section.claims
    ]
    assert any("473" in claim.text for claim in claims)


def test_clean_first_attempt_never_fires_forced_interleave(tmp_path: Path) -> None:
    profile = _profile_artifact(tmp_path)
    plan = ReportPlanDraft(
        claims=[
            ReportPlanClaim(
                section_title="Dataset Overview",
                id="rows",
                text="The dataset has 2 rows.",
                evidence=[
                    EvidenceRef(kind="stat", artifact_id=profile.id, locator="rows", value=2)
                ],
                referenced_datasets=["x.csv"],
            )
        ],
    )
    llm = ScriptedPlanLLM([plan])

    result = generate_agentic_report(
        [profile],
        project_id="project_demo",
        session_id="run_demo",
        business_context="Overview",
        llm=llm,
    )

    assert llm.calls == 1
    assert result.forced_evidence_requests == 0
    assert all("granted_repair_evidence" not in payload for payload in llm.payloads)


def test_forced_interleave_note_counts_only_granted_requests() -> None:
    # 5 granted + 1 rejected must read "5 auto-resolved" (and disclose the
    # rejection), never "6 auto-resolved": rejected requests resolved nothing.
    def _exchange(index: int, *, granted: bool) -> InterleaveExchange:
        request = EvidenceRequest(artifact_id=f"sql_{index:06x}", locator="rows[0].n")
        if granted:
            grant = EvidenceGrant(
                artifact_id=request.artifact_id,
                artifact_type="sql_result",
                locator=request.locator,
                values=[GrantValue(value=float(index))],
            )
            return InterleaveExchange(section="numeric_repair", request=request, grant=grant)
        rejection = EvidenceRejection(
            artifact_id=request.artifact_id,
            locator=request.locator,
            reason_code="unresolvable_locator",
            message="No such locator.",
        )
        return InterleaveExchange(
            section="numeric_repair", request=request, rejection=rejection
        )

    exchanges = [_exchange(index, granted=True) for index in range(5)]
    exchanges.append(_exchange(5, granted=False))

    note = _forced_interleave_note(exchanges)

    assert note is not None
    assert "5 evidence request(s) auto-resolved" in note
    assert "6" not in note
    assert "1" in note and "rejected" in note
    assert _forced_interleave_note([]) is None


def test_limitations_cross_dataset_tier_collapses_small_groups() -> None:
    # DI10 integration tier: one code spread thin across many datasets (2 columns
    # each, under the per-dataset threshold) must still collapse into ONE
    # report-wide sentence once the cross-dataset total exceeds the tier
    # threshold (live-Olist regression: 14 likely_id_column lines from 9 tables).
    artifacts = []
    for d in range(1, 6):
        contexts = [
            QualityContext(
                context_id=f"ctx_id_{d}_{i}",
                dataset_id=f"ds_{d}",
                dataset_name=f"t{d}.csv",
                issue_code="likely_id_column",
                severity="warn",
                column=f"col_{i}",
                observation="looks like an id.",
                report_limitation=(
                    f"Interpretation involving col_{i} should account for the "
                    "observed likely_id_column condition; its business cause "
                    "remains unconfirmed."
                ),
            )
            for i in range(1, 3)
        ]
        artifacts.append(
            Artifact(
                id=f"qualityctx_{d:012x}",
                type=ArtifactType.QUALITY_CONTEXT_SET,
                project_id="project_demo",
                session_id="run_demo",
                payload=QualityContextSet(
                    dataset_id=f"ds_{d}", dataset_name=f"t{d}.csv", contexts=contexts
                ).model_dump(mode="json"),
            )
        )
    markdown = report_bundle_to_markdown(_empty_bundle(), artifacts=artifacts)
    limitations = _section_text(markdown, "## Limitations and Risks")

    # 10 columns across 5 datasets > CROSS_DATASET_AGGREGATE_THRESHOLD (6).
    assert "10 columns across 5 datasets" in limitations
    assert limitations.count("look like identifier columns") == 1
    assert "Interpretation involving col_1" not in limitations
