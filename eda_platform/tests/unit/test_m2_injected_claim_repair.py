"""Platform-injected claims must not buy LLM repair rounds.

2026-08-20, Compare run sess_1787201833042_9n7rig: `report_validation` fired
three times with byte-identical results — the same 5 critical findings, 4 of
them on `qfind_q_*` claims (3 numeric_mismatch, 1 invalid_evidence_locator).
Those claims are built by `_inject_question_claims` from executed question
findings; the plan LLM never wrote their text and cannot rewrite it, so both
repair rounds were dead weight (a full m2 call each: ~$0.02 on gpt-5.6-luna,
~90s on deepseek) and the hard gate pruned the claims anyway.

Why those qfind claims failed numeric_mismatch there — di4's problem, not this
one: the interpretation text carried numbers the evidence pool never held.
`qfind_q_4f3e36d6d4_0` asserted 2709, 50 and 1894 against a pool of
{1.97, 2.51, 3.19, 9, 24, 47} resolved from `rows_preview[*].item_count` and
`rows_preview[*].avg_freight_to_price_ratio` — totals and counts derived
during interpretation rather than read off the cited locator. Fixing di4 is
out of scope; this file only stops the platform from paying an LLM to rewrite
text the LLM did not write.
"""

from __future__ import annotations

from typing import Any, TypeVar, cast

from pydantic import BaseModel

from eda_platform.agents.reporting import _generate_with_repair
from eda_platform.schemas.artifacts import EvidenceRef
from eda_platform.schemas.questions import QuestionExecutionResult, QuestionFinding
from eda_platform.schemas.reports import (
    ReportBundle,
    ReportClaim,
    ReportPlanClaim,
    ReportPlanDraft,
    ReportSeverity,
)
from eda_platform.tools.evidence import (
    EvidenceAnalysisTable,
    EvidenceArtifactSummary,
    EvidenceDataset,
    EvidencePack,
)
from eda_platform.tools.report_validator import validate_report_bundle

T = TypeVar("T", bound=BaseModel)

_ANALYSIS_SECTION = "Agent-Performed Analysis"
_BAD_NUMBER_TEXT = "Freight averaged 2709 per order."


def _pack() -> EvidencePack:
    return EvidencePack(
        payload_policy="schema+aggregates",
        artifact_index={
            "table_1": EvidenceArtifactSummary(
                artifact_id="table_1",
                artifact_type="Table",
                title="Numeric summary",
                dataset_id="ds_sales",
            )
        },
        datasets=[
            EvidenceDataset(
                artifact_id="prof_1",
                dataset_id="ds_sales",
                name="sales.csv",
                row_count=10,
                column_count=2,
                columns=["region", "revenue"],
                dtypes={"region": "object", "revenue": "float64"},
            )
        ],
        analysis_tables=[
            EvidenceAnalysisTable(
                artifact_id="table_1",
                dataset_id="ds_sales",
                title="Numeric summary",
                kind="aggregation",
                description="Revenue summary",
                rows=[{"revenue": 120}],
            )
        ],
    )


def _evidence() -> list[EvidenceRef]:
    return [EvidenceRef(kind="stat", artifact_id="table_1", locator="rows[0]", value=120)]


def _bundle_with_claim(claim: ReportClaim, *, section_title: str) -> ReportBundle:
    bundle = ReportBundle.empty(project_id="p", session_id="r")
    section = next(s for s in bundle.sections if s.title == section_title)
    section.claims.append(claim)
    for section in bundle.sections:
        section.body = section.structural_body()
    return bundle


def _numeric_finding(audit):
    return next(f for f in audit.findings if f.code == "numeric_mismatch")


def test_injected_claim_numeric_mismatch_routes_to_prune() -> None:
    audit = validate_report_bundle(
        _bundle_with_claim(
            ReportClaim(
                id="qfind_q_4f3e36d6d4_0",
                text=_BAD_NUMBER_TEXT,
                evidence=_evidence(),
            ),
            section_title=_ANALYSIS_SECTION,
        ),
        _pack(),
    )

    finding = _numeric_finding(audit)
    assert finding.severity is ReportSeverity.CRITICAL
    assert finding.repair_mode == "prune"


def test_llm_authored_claim_numeric_mismatch_still_routes_to_llm() -> None:
    audit = validate_report_bundle(
        _bundle_with_claim(
            ReportClaim(id="claim_3", text=_BAD_NUMBER_TEXT, evidence=_evidence()),
            section_title="Business Findings",
        ),
        _pack(),
    )

    assert _numeric_finding(audit).repair_mode == "llm"


class _Settings:
    def __init__(self) -> None:
        self.max_tokens = 4_000


class _RecordingLLM:
    """Counts plan calls; every attempt returns the same draft."""

    def __init__(self, draft: ReportPlanDraft) -> None:
        self.settings = _Settings()
        self.draft = draft
        self.plan_calls: list[str] = []

    def structured(self, *, task: str, schema: type[T], payload: dict[str, Any]) -> T:
        self.plan_calls.append(task)
        return cast(T, self.draft.model_copy(deep=True))

    def text(self, *, task: str, payload: dict[str, Any]) -> str:
        return "fake"

    def last_usage(self) -> None:
        return None


def _run(llm: _RecordingLLM, question_results: list[QuestionExecutionResult]):
    return _generate_with_repair(
        _pack(),
        project_id="p",
        session_id="r",
        business_context="Freight analysis",
        llm=llm,
        question_results=question_results,
        sql_results={},
        llm_calls=[],
        llm_events=[],
        validation_events=[],
    )


def _question_results() -> list[QuestionExecutionResult]:
    return [
        QuestionExecutionResult(
            question_id="q_4f3e36d6d4",
            question="What does freight cost per order?",
            origin="template",
            status="succeeded",
            outcome="answered",
            findings=[QuestionFinding(text=_BAD_NUMBER_TEXT, evidence=_evidence())],
        )
    ]


def test_all_injected_criticals_spend_a_single_plan_call() -> None:
    llm = _RecordingLLM(ReportPlanDraft(claims=[]))

    bundle, audit, used_fallback = _run(llm, _question_results())

    assert used_fallback is False
    # The draft attempt only; no repair round, because nothing the plan LLM
    # authored is broken.
    assert llm.plan_calls == ["m2_report_claim_plan"]
    # ...and the injected claim is still gone from the published bundle.
    claim_ids = [claim.id for section in bundle.sections for claim in section.claims]
    assert "qfind_q_4f3e36d6d4_0" not in claim_ids
    assert not audit.has_critical_findings


def test_llm_authored_critical_still_spends_a_repair_round() -> None:
    llm = _RecordingLLM(
        ReportPlanDraft(
            claims=[
                ReportPlanClaim(
                    id="claim_1",
                    section_title="Business Findings",
                    text=_BAD_NUMBER_TEXT,
                    evidence=_evidence(),
                )
            ]
        )
    )

    _run(llm, [])

    # Draft + one repair round; the third is skipped only because this fake
    # returns an identical plan (the no-progress breaker).
    assert len(llm.plan_calls) == 2
