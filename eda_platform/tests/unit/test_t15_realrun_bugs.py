"""T15: the five defects the 2026-08-26 Compare run exposed.

Session sess_1787771303025_kojblh, reproduced from its own payloads:

a. Two questions whose SQL returned 2591 and 73 rows were discarded because the
   method contract was not proven, and the report told the reader nothing had
   been computed.
b. Two causal-worded claims were deleted, emptying Key EDA Insights and
   Business Recommendations, after an LLM repair round returned them verbatim.
c. A correct quality claim was labelled unverified because the locator grammar
   the plan model writes did not parse.
d. A share was restated against a denominator the evidence never gave.
e. A question's finding and its interpretation rendered as two bullets stating
   the same number, and the deduplicator left a pointer bullet behind.
"""

from __future__ import annotations

from eda_platform.core.claim_language import rewrite_causal_language
from eda_platform.core.ids import make_artifact_id
from eda_platform.drivers.question_exec import _successful_qexec_artifact
from eda_platform.schemas.artifacts import (
    Artifact,
    ArtifactType,
    EvidenceRef,
    SqlResult,
)
from eda_platform.schemas.questions import (
    QuestionCandidate,
    QuestionExecutionResult,
    QuestionFinding,
    QuestionScore,
)
from eda_platform.schemas.reports import (
    ReportAudit,
    ReportBundle,
    ReportClaim,
    ReportSection,
    ReportStatus,
)
from eda_platform.tools.evidence import (
    EvidenceArtifactSummary,
    EvidencePack,
    EvidenceQualityIssue,
)
from eda_platform.tools.exporter import report_bundle_to_markdown
from eda_platform.tools.report_validator import (
    _quality_issue_numbers,
    _repair_mode_for_code,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _sql_artifact(rows: list[dict[str, object]], *, sql: str = "select 1") -> Artifact:
    columns = list(rows[0]) if rows else []
    payload = SqlResult(
        sql=sql,
        columns=columns,
        dtypes=dict.fromkeys(columns, "DOUBLE"),
        rows_preview=rows,
        row_count=len(rows),
    ).model_dump(mode="json")
    return Artifact(
        id=make_artifact_id("sql", payload),
        type=ArtifactType.SQL_RESULT,
        project_id="project_t15",
        session_id="run_t15",
        payload=payload,
    )


def _candidate(question: str, *, analysis_mode: str) -> QuestionCandidate:
    return QuestionCandidate(
        question_id="q_t15",
        question_en=question,
        origin="llm",
        template_id=None,
        analysis_mode=analysis_mode,
        target_datasets=["olist_order_items_dataset.csv"],
        exploratory=True,
        score=QuestionScore(
            data_availability=1.0,
            statistical_signal=0.5,
            quality_risk=0.0,
            join_risk=0.0,
            deterministic_score=0.6,
        ),
    )


# --------------------------------------------------------------------------- #
# a. A proven-method failure must degrade, not discard
# --------------------------------------------------------------------------- #
# The real run's freight question: SQL returned 2591 rows (sql_bdcc30f97497),
# the anomaly contract was unproven, and the answer was thrown away.
_FREIGHT_ROWS = [
    {"order_id": "o1", "freight_to_price_ratio": 12.4, "freight_value": 45.0},
    {"order_id": "o2", "freight_to_price_ratio": 9.8, "freight_value": 31.0},
]


def test_anomaly_question_publishes_its_sql_result_when_the_method_is_unproven() -> None:
    candidate = _candidate(
        "Which order-item records have unusually high freight cost relative to item price?",
        analysis_mode="anomaly",
    )
    sql_artifact = _sql_artifact(_FREIGHT_ROWS)

    artifact = _successful_qexec_artifact(
        candidate,
        sql_artifact=sql_artifact,
        project_id="project_t15",
        session_id="run_t15",
        parent_ids=[sql_artifact.id],
        plan_summary="ratio screen in SQL",
    )
    result = QuestionExecutionResult.model_validate(artifact.payload)

    assert result.outcome == "answered"
    assert result.findings, "the computed rows must reach the report"
    assert result.sql_result_artifact_id == sql_artifact.id
    assert result.contract_status == "unverified"
    # The reader must be told what is missing, not that nothing was computed.
    assert result.limitations
    disclosure = " ".join(result.limitations).lower()
    assert "outlier screen" in disclosure
    assert "method_contract_failed" not in disclosure
    assert "contract" not in disclosure


def test_the_degradation_reaches_the_reader_in_limitations() -> None:
    candidate = _candidate(
        "Which order-item records have unusually high freight cost relative to item price?",
        analysis_mode="anomaly",
    )
    sql_artifact = _sql_artifact(_FREIGHT_ROWS)
    qexec = _successful_qexec_artifact(
        candidate,
        sql_artifact=sql_artifact,
        project_id="project_t15",
        session_id="run_t15",
        parent_ids=[sql_artifact.id],
        plan_summary="ratio screen in SQL",
    )

    markdown = report_bundle_to_markdown(
        ReportBundle.empty(project_id="project_t15", session_id="run_t15"),
        artifacts=[qexec],
    )

    limitations = markdown.split("## Limitations and Risks")[1]
    assert "asks for an outlier screen" in limitations
    assert "unusually high freight cost" in limitations


def test_segmentation_question_publishes_its_sql_result_when_the_method_is_unproven() -> None:
    candidate = _candidate(
        "How do product content and physical attributes vary by product category?",
        analysis_mode="segmentation",
    )
    sql_artifact = _sql_artifact(
        [
            {"product_category_name": "bed_bath_table", "avg_photos_qty": 1.7},
            {"product_category_name": "health_beauty", "avg_photos_qty": 2.1},
        ]
    )

    artifact = _successful_qexec_artifact(
        candidate,
        sql_artifact=sql_artifact,
        project_id="project_t15",
        session_id="run_t15",
        parent_ids=[sql_artifact.id],
        plan_summary="per-category aggregate",
    )
    result = QuestionExecutionResult.model_validate(artifact.payload)

    assert result.outcome == "answered"
    assert result.contract_status == "unverified"
    assert result.findings


def test_a_causal_question_still_refuses_an_observational_proxy() -> None:
    """The degrade path must not reopen the causal-proxy hole it was built to close."""
    candidate = _candidate(
        "Did the free-shipping change cause the drop in late deliveries?",
        analysis_mode="causal_experiment",
    )
    sql_artifact = _sql_artifact([{"late_rate_before": 0.09, "late_rate_after": 0.07}])

    artifact = _successful_qexec_artifact(
        candidate,
        sql_artifact=sql_artifact,
        project_id="project_t15",
        session_id="run_t15",
        parent_ids=[sql_artifact.id],
        plan_summary="before/after rates",
    )
    result = QuestionExecutionResult.model_validate(artifact.payload)

    assert result.outcome == "abstained"
    assert result.abstention_code == "method_contract_failed"


def test_an_empty_degraded_result_still_abstains() -> None:
    """Degrading must not bypass the checks that reject an unusable result."""
    candidate = _candidate(
        "Which order-item records have unusually high freight cost?",
        analysis_mode="anomaly",
    )
    sql_artifact = _sql_artifact([])

    artifact = _successful_qexec_artifact(
        candidate,
        sql_artifact=sql_artifact,
        project_id="project_t15",
        session_id="run_t15",
        parent_ids=[sql_artifact.id],
        plan_summary="empty screen",
    )
    result = QuestionExecutionResult.model_validate(artifact.payload)

    assert result.outcome == "abstained"
    assert result.abstention_code == "empty_query_result"


# --------------------------------------------------------------------------- #
# b. Causal wording is rewritten, not deleted
# --------------------------------------------------------------------------- #
def test_causal_overclaim_is_repaired_deterministically_not_by_an_llm_round() -> None:
    # The 2026-08-26 run burned two m2 rounds on c3/c7, got byte-identical text
    # back, then deleted both, emptying two sections.
    assert _repair_mode_for_code("causal_overclaim", "c3") == "deterministic"
    assert _repair_mode_for_code("causal_overclaim", "qfind_q_1_0") == "deterministic"


def test_causal_phrases_are_rewritten_into_association_wording() -> None:
    rewritten = rewrite_causal_language(
        "Late deliveries are driven by freight cost, which leads to lower review scores."
    )

    assert rewritten is not None
    assert "driven by" not in rewritten
    assert "leads to" not in rewritten
    assert "associated with" in rewritten
    # The rewrite may not invent or drop figures.
    assert "freight cost" in rewritten and "review scores" in rewritten


def test_a_rewrite_that_cannot_remove_the_causal_wording_returns_nothing() -> None:
    assert rewrite_causal_language("Revenue rose 4% in Q3.") is None


def test_a_causal_claim_survives_the_repair_round_instead_of_emptying_its_section() -> None:
    from eda_platform.agents.reporting import _apply_deterministic_repairs
    from eda_platform.tools.report_validator import validate_report_bundle

    pack = _quality_pack()
    bundle = ReportBundle(
        project_id="project_t15",
        session_id="run_t15",
        status=ReportStatus.VALIDATED,
        sections=[
            ReportSection(
                title="Key EDA Insights",
                body="",
                claims=[
                    ReportClaim(
                        id="c3",
                        text="Missing review text is driven by unengaged buyers.",
                        evidence=[
                            EvidenceRef(
                                kind="artifact",
                                artifact_id="quality_cf2381c92f30",
                                locator="issues",
                            )
                        ],
                    )
                ],
            )
        ],
        audit=ReportAudit(status=ReportStatus.VALIDATED),
    )

    audit = validate_report_bundle(bundle, pack)
    assert any(finding.code == "causal_overclaim" for finding in audit.findings)

    repairs = _apply_deterministic_repairs(bundle, audit, evidence_pack=pack)
    regated = validate_report_bundle(bundle, pack)

    assert repairs >= 1
    assert not any(finding.code == "causal_overclaim" for finding in regated.findings)
    section = bundle.sections[0]
    assert len(section.claims) == 1
    assert "driven by" not in section.claims[0].text
    assert "associated with" in section.claims[0].text


# --------------------------------------------------------------------------- #
# c. The plan model's quality locator grammar must resolve
# --------------------------------------------------------------------------- #
def _quality_pack() -> EvidencePack:
    return EvidencePack(
        payload_policy="schema+aggregates",
        artifact_index={
            "quality_cf2381c92f30": EvidenceArtifactSummary(
                artifact_id="quality_cf2381c92f30",
                artifact_type=ArtifactType.QUALITY_ISSUE_SET.value,
                title="Quality issues",
                dataset_id="ds_reviews",
            )
        },
        quality_issues=[
            EvidenceQualityIssue(
                artifact_id="quality_cf2381c92f30",
                dataset_id="ds_reviews",
                severity="warn",
                code="high_missing",
                column="review_comment_title",
                message="review_comment_title is 88.34% missing.",
                recommendation="Treat findings on it as limited.",
                metric_value=88.34,
                metric_unit="percent",
            ),
            EvidenceQualityIssue(
                artifact_id="quality_cf2381c92f30",
                dataset_id="ds_reviews",
                severity="warn",
                code="duplicate_rows",
                column=None,
                message="824 potential duplicate rows.",
                recommendation="Deduplicate before analysis.",
                affected_count=824,
            ),
        ],
    )


def test_the_bracket_predicate_quality_locator_resolves() -> None:
    pack = _quality_pack()

    values = _quality_issue_numbers(
        EvidenceRef(
            kind="artifact",
            artifact_id="quality_cf2381c92f30",
            locator=(
                "quality_issues[code=high_missing,column=review_comment_title].metric_value"
            ),
        ),
        pack,
    )

    assert [value for value, *_ in values] == [88.34]


def test_the_bracket_predicate_locator_without_a_column_resolves() -> None:
    pack = _quality_pack()

    values = _quality_issue_numbers(
        EvidenceRef(
            kind="artifact",
            artifact_id="quality_cf2381c92f30",
            locator="quality_issues[code=duplicate_rows].affected_count",
        ),
        pack,
    )

    assert [value for value, *_ in values] == [824.0]


def test_a_bracket_predicate_locator_cannot_borrow_another_columns_figure() -> None:
    pack = _quality_pack()

    values = _quality_issue_numbers(
        EvidenceRef(
            kind="artifact",
            artifact_id="quality_cf2381c92f30",
            locator=(
                "quality_issues[code=high_missing,column=review_comment_message].metric_value"
            ),
        ),
        pack,
    )

    assert values == []


def test_claim_qualifier_prefixes_read_as_plain_language() -> None:
    bundle = ReportBundle(
        project_id="project_t15",
        session_id="run_t15",
        status=ReportStatus.VALIDATED,
        sections=[
            ReportSection(
                title="Data Quality Findings",
                body="Validated evidence-backed findings are listed below.",
                claims=[
                    ReportClaim(
                        id="c2",
                        text="review_comment_title is 88.34% missing.",
                        evidence=[
                            EvidenceRef(
                                kind="artifact",
                                artifact_id="quality_1",
                                locator="quality_issue:high_missing:review_comment_title",
                            )
                        ],
                        numeric_rollup="unverified",
                        confidence_label="exploratory",
                    )
                ],
            )
        ],
        audit=ReportAudit(status=ReportStatus.VALIDATED),
    )

    markdown = report_bundle_to_markdown(bundle)

    assert "[Unverified figures]" not in markdown
    assert "[Exploratory — hypothesis-generating]" not in markdown
    assert "[Figures not re-checked]" in markdown
    assert "[A lead, not a conclusion]" in markdown


# --------------------------------------------------------------------------- #
# d. A share must keep the denominator its evidence names
# --------------------------------------------------------------------------- #
def test_the_plan_prompt_forbids_restating_a_share_against_a_new_denominator() -> None:
    # The report said "freight represents 14.2% of order-item GMV of
    # 13,591,643.7" while the query divided by price + freight (16.6% of GMV).
    # The numeric gate verifies 14.2 and never reads the "of ..." phrase.
    from typing import Any, cast

    from eda_platform.agents.reporting import _request_plan
    from eda_platform.schemas.reports import ReportPlanDraft

    class _CapturingLLM:
        def __init__(self) -> None:
            self.payloads: list[dict[str, Any]] = []

        def structured(self, *, task: str, schema: type, payload: dict) -> Any:
            self.payloads.append(payload)
            return ReportPlanDraft()

        def text(self, *, task: str, payload: dict) -> str:
            return ""

        def last_usage(self) -> None:
            return None

    llm = _CapturingLLM()
    _request_plan(
        EvidencePack(payload_policy="schema+aggregates"),
        business_context="Revenue analysis",
        llm=cast(Any, llm),
        prior_error=None,
        prior_findings=[],
        prior_bundle=None,
        question_results=[_question_with_interpretation()],
    )

    instructions = llm.payloads[0]["instructions"].lower()
    assert "denominator" in instructions
    assert "not a share of gmv" in instructions


# --------------------------------------------------------------------------- #
# e. One bullet per question, and no pointer bullets
# --------------------------------------------------------------------------- #
def _question_with_interpretation() -> QuestionExecutionResult:
    return QuestionExecutionResult(
        question_id="q_4aa1c9aa7d",
        question="What is the average order value in Olist Order Items Dataset?",
        origin="llm",
        findings=[
            QuestionFinding(
                text=(
                    "What is the average order value (price per distinct order_id) in "
                    "Olist Order Items Dataset? Across 98666 orders the average order "
                    "value is 137.7541."
                ),
                evidence=[
                    EvidenceRef(
                        kind="sql",
                        artifact_id="sql_101a09707ee6",
                        locator="rows_preview[0].avg_order_value",
                    )
                ],
            )
        ],
        status="succeeded",
        outcome="answered",
        interpretation=(
            "Across 98666 orders, the average order value is 137.754. This provides "
            "the observed baseline price per distinct order_id for the dataset."
        ),
        interpretation_status="validated",
    )


def test_a_questions_finding_and_interpretation_render_as_one_bullet() -> None:
    from eda_platform.agents.reporting import _inject_question_claims

    bundle = ReportBundle(
        project_id="project_t15",
        session_id="run_t15",
        status=ReportStatus.VALIDATED,
        sections=[
            ReportSection(title="Selected Analysis Focus", body=""),
            ReportSection(title="Agent-Performed Analysis", body=""),
        ],
        audit=ReportAudit(status=ReportStatus.VALIDATED),
    )

    _inject_question_claims(bundle, [_question_with_interpretation()])

    analysis = next(s for s in bundle.sections if s.title == "Agent-Performed Analysis")
    assert len(analysis.claims) == 1
    text = analysis.claims[0].text
    # The interpretation's restatement of the finding's own figures is dropped;
    # what it adds is kept.
    assert text.count("137.75") == 1
    assert "observed baseline price per distinct order_id" in text


def test_a_restatement_in_scientific_notation_is_still_recognised() -> None:
    """The finding builder renders 13591643.7 as 1.35916e+07 (real run, q_0495f29239)."""
    from eda_platform.agents.reporting import _interpretation_addition

    addition = _interpretation_addition(
        "Total GMV over 112650 rows is 13591643.7. This provides the baseline.",
        "What is the total GMV? Total GMV over 112650 rows is 1.35916e+07.",
    )

    assert addition == "This provides the baseline."


def test_deduplication_drops_a_repeat_instead_of_leaving_a_pointer() -> None:
    evidence = [
        EvidenceRef(
            kind="sql",
            artifact_id="sql_b57ecbb1b97b",
            locator="rows_preview[0].hhi",
        )
    ]
    bundle = ReportBundle(
        project_id="project_t15",
        session_id="run_t15",
        status=ReportStatus.VALIDATED,
        sections=[
            ReportSection(
                title="Dataset Overview",
                body="",
                claims=[
                    ReportClaim(
                        id="c1",
                        text=(
                            "Payment value is concentrated across 5 payment-type groups: "
                            "the HHI is 0.6467 and the top group share is 0.7834."
                        ),
                        evidence=evidence,
                    )
                ],
            ),
            ReportSection(
                title="Business Findings",
                body="",
                claims=[
                    ReportClaim(
                        id="c5",
                        text=(
                            "Payment value is concentrated across 5 groups, HHI 0.6467, "
                            "top share 0.7834."
                        ),
                        evidence=evidence,
                    )
                ],
            ),
        ],
        audit=ReportAudit(status=ReportStatus.VALIDATED),
    )

    markdown = report_bundle_to_markdown(bundle)

    assert "for the full statement of this finding" not in markdown
    assert "See \"Dataset Overview\"" not in markdown
