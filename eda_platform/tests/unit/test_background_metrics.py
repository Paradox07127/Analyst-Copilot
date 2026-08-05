"""Three registry metrics describe the data; they must not headline the report.

`concentration_hhi`, `missing_hotspots` and `time_coverage` carry no domain
precondition, so they resolve on any dataset. They were taken into the first
analysis slots unconditionally, ahead of every LLM question, and their
`business_decision` is one shared registry sentence -- "Address material quality
issues before relying on affected results." -- including for the HHI question,
which is not about quality at all.

The 2026-08-04 FIFA run 2 opened with "venue capacity is concentrated across the
two knockout groups (HHI 0.568)" as its headline finding. On a binary `is_*`
split the lower bound is 0.5, so 0.568 is barely off even, and the split is
really the 72-vs-32 match count. Meanwhile eight LLM questions with business
relevance 0.77-0.91 competed for the remaining slots and two never ran.

They cost no LLM call (their SQL is a registry template), so they keep running.
They stop spending analysis slots and stop being read as analysis.
"""

from __future__ import annotations

from typing import Any

from eda_platform.agents.reporting import (
    _apply_business_findings_fallback,
    _apply_dataset_overview_fallback,
    _inject_question_claims,
)
from eda_platform.schemas.questions import (
    QuestionCandidate,
    QuestionCandidateSet,
    QuestionExecutionResult,
    QuestionFinding,
    QuestionScore,
)
from eda_platform.schemas.reports import ReportBundle, ReportSection
from eda_platform.tools.domain_metrics import metric_definition
from eda_platform.tools.question_discovery import (
    auto_execution_composition,
    select_auto_execution_set,
)

_BACKGROUND_METRICS = ("concentration_hhi", "missing_hotspots", "time_coverage")

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


def _score(**overrides: float) -> QuestionScore:
    values: dict[str, Any] = {
        "data_availability": 1.0,
        "statistical_signal": 0.6,
        "quality_risk": 0.0,
        "join_risk": 0.0,
        "deterministic_score": 0.6,
    }
    values.update(overrides)
    return QuestionScore.model_validate(values)


def _metric(metric_id: str) -> QuestionCandidate:
    return QuestionCandidate(
        question_id=f"q_{metric_id}",
        question_en=f"Question for {metric_id}?",
        origin="template",
        template_id="domain_metric",
        metric_id=metric_id,
        target_datasets=["orders.csv"],
        sql_template="select 1",
        score=_score(deterministic_score=0.84),
    )


def _exploratory(index: int) -> QuestionCandidate:
    return QuestionCandidate(
        question_id=f"q_llm_{index}",
        question_en=f"LLM question {index}?",
        origin="llm",
        target_datasets=["orders.csv"],
        exploratory=True,
        score=_score(deterministic_score=0.775, llm_business_relevance=0.85),
    )


def _bundle() -> ReportBundle:
    return ReportBundle(
        project_id="project_demo",
        session_id="run_demo",
        sections=[ReportSection(title=title, body="") for title in _SECTIONS],
        status="validated",
    )


def _section(bundle: ReportBundle, title: str) -> ReportSection:
    return next(section for section in bundle.sections if section.title == title)


def _result(metric_id: str | None, *, question_id: str) -> QuestionExecutionResult:
    return QuestionExecutionResult(
        question_id=question_id,
        question=f"Question for {metric_id or question_id}?",
        origin="template" if metric_id else "llm",
        metric_id=metric_id,
        findings=[
            QuestionFinding(
                text=f"Finding from {metric_id or question_id}.", evidence=[]
            )
        ],
        status="succeeded",
        outcome="answered",
    )


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


def test_the_three_domain_agnostic_metrics_declare_a_background_section() -> None:
    for metric_id in _BACKGROUND_METRICS:
        definition = metric_definition(metric_id)
        assert definition is not None
        assert definition.background_section in _SECTIONS, metric_id


def test_a_domain_metric_is_not_background() -> None:
    """The e-commerce metrics answer business questions and keep their slot."""
    for metric_id in ("gmv", "aov", "late_delivery_rate"):
        definition = metric_definition(metric_id)
        assert definition is not None
        assert definition.background_section is None, metric_id


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


def test_background_metrics_do_not_spend_analysis_slots() -> None:
    candidates = [_metric(metric_id) for metric_id in _BACKGROUND_METRICS]
    candidates += [_exploratory(index) for index in range(12)]

    selected = select_auto_execution_set(QuestionCandidateSet(candidates=candidates))

    chosen = {candidate.question_id for candidate in selected}
    for metric_id in _BACKGROUND_METRICS:
        assert f"q_{metric_id}" in chosen, metric_id
    analysis = [
        candidate for candidate in selected if candidate.metric_id not in _BACKGROUND_METRICS
    ]
    assert len(analysis) == 10, [candidate.question_id for candidate in analysis]


def test_a_business_metric_still_competes_for_a_slot() -> None:
    candidates = [_metric("gmv"), _metric("aov")]
    candidates += [_exploratory(index) for index in range(12)]

    selected = select_auto_execution_set(QuestionCandidateSet(candidates=candidates))

    assert len(selected) == 10
    assert "q_gmv" in {candidate.question_id for candidate in selected}


def test_the_composition_counts_background_separately() -> None:
    """Telemetry must not report a background metric as an executed analysis."""
    candidates = [_metric("time_coverage"), _metric("gmv"), _exploratory(0)]
    composition = auto_execution_composition(
        select_auto_execution_set(QuestionCandidateSet(candidates=candidates))
    )
    assert composition["n_background"] == 1
    assert composition["n_domain_metric"] == 1
    assert (
        composition["n_background"]
        + composition["n_domain_metric"]
        + composition["n_exploratory"]
        + composition["n_template"]
        == composition["selected_count"]
    )


# --------------------------------------------------------------------------- #
# Report routing
# --------------------------------------------------------------------------- #


def test_a_background_finding_lands_in_its_declared_section() -> None:
    bundle = _bundle()
    _inject_question_claims(
        bundle,
        [
            _result("time_coverage", question_id="q_time"),
            _result("missing_hotspots", question_id="q_missing"),
            _result(None, question_id="q_llm"),
        ],
    )
    assert [claim.text for claim in _section(bundle, "Agent-Performed Analysis").claims] == [
        "Finding from q_llm."
    ]
    assert any(
        "time_coverage" in claim.text for claim in _section(bundle, "Dataset Overview").claims
    )
    assert any(
        "missing_hotspots" in claim.text
        for claim in _section(bundle, "Data Quality Findings").claims
    )


def test_a_background_finding_is_not_an_analysis_focus_item() -> None:
    bundle = _bundle()
    _inject_question_claims(
        bundle,
        [
            _result("concentration_hhi", question_id="q_hhi"),
            _result(None, question_id="q_llm"),
        ],
    )
    focus = _section(bundle, "Selected Analysis Focus").focus_items
    assert [item.question_id for item in focus] == ["q_llm"]


def test_a_background_claim_cannot_be_promoted_to_the_executive_summary() -> None:
    """The id prefix is what the summary's picker filters on."""
    bundle = _bundle()
    _inject_question_claims(bundle, [_result("concentration_hhi", question_id="q_hhi")])
    background = [
        claim
        for section in bundle.sections
        for claim in section.claims
        if "concentration_hhi" in claim.text
    ]
    assert background
    for claim in background:
        assert not (claim.id or "").startswith(("qfind_", "qbiz_")), claim.id


def test_the_business_findings_fallback_is_the_other_door() -> None:
    """Offline FIFA run, 2026-08-05: routing the injection was not enough.

    `_inject_question_claims` filed time coverage under Dataset Overview as
    intended, and then the Business Findings fallback injected the same finding
    again as `qbiz_`, which the summary picker does select. The report opened
    with a date range, and the dedup pass left Business Findings holding two
    "See Dataset Overview" stubs and nothing else.
    """
    bundle = _bundle()
    results = [
        _result("time_coverage", question_id="q_time"),
        _result("concentration_hhi", question_id="q_hhi"),
    ]
    _inject_question_claims(bundle, results)
    injected = _apply_business_findings_fallback(bundle, results)

    assert injected == 0
    assert _section(bundle, "Business Findings").claims == []


def test_a_background_metric_is_excluded_on_its_own_grounds() -> None:
    """Two independent exclusions, so neither may be mistaken for the other.

    Nothing here reaches Agent-Performed Analysis, so the already-published rule
    cannot fire; what keeps time coverage out is that it is background.
    """
    bundle = _bundle()
    results = [
        _result("time_coverage", question_id="q_time"),
        _result(None, question_id="q_llm"),
    ]

    assert _apply_business_findings_fallback(bundle, results) == 1
    texts = [claim.text for claim in _section(bundle, "Business Findings").claims]
    assert texts == ["Finding from q_llm."]


def test_dataset_overview_still_gets_its_profile_claims() -> None:
    """A background claim is not an authored summary, so it must not stand in.

    The overview fallback fills a section nobody wrote; treating the metric's
    own line as that writing would drop the row and column counts.
    """
    from eda_platform.tools.evidence import EvidenceDataset, EvidencePack

    pack = EvidencePack(
        payload_policy="schema+aggregates",
        datasets=[
            EvidenceDataset(
                artifact_id="prof_1",
                dataset_id="ds_x",
                name="player_stats.csv",
                row_count=1248,
                column_count=21,
                columns=["player_id"],
                dtypes={"player_id": "int64"},
            )
        ],
    )
    bundle = _bundle()
    _inject_question_claims(bundle, [_result("time_coverage", question_id="q_time")])

    assert _apply_dataset_overview_fallback(bundle, pack) == 1
    overview = _section(bundle, "Dataset Overview").claims
    assert any((claim.id or "").startswith("dataset_overview_") for claim in overview)
    assert any((claim.id or "").startswith("qbg_") for claim in overview)


def test_the_fallback_does_not_inject_what_the_report_already_states() -> None:
    """Business Findings was a list of pointers and nothing else.

    The fallback copies executed findings in when no business claim was
    authored, but `_inject_question_claims` has already published those same
    findings verbatim under Agent-Performed Analysis. The exporter's dedup then
    replaces every copy with `See "Agent-Performed Analysis" ...`, so across
    three live runs the section never held anything but stubs.
    """
    bundle = _bundle()
    results = [_result(None, question_id="q_llm")]
    _inject_question_claims(bundle, results)

    assert _apply_business_findings_fallback(bundle, results) == 0
    assert _section(bundle, "Business Findings").claims == []


def test_the_fallback_still_fires_for_a_finding_the_report_lacks() -> None:
    """Its purpose survives: a finding not already published is injected."""
    bundle = _bundle()
    published = _result(None, question_id="q_llm")
    _inject_question_claims(bundle, [published])
    extra = _result(None, question_id="q_other")

    assert _apply_business_findings_fallback(bundle, [published, extra]) == 1
    texts = [claim.text for claim in _section(bundle, "Business Findings").claims]
    assert texts == ["Finding from q_other."]
