from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar, cast

import pytest
from pydantic import BaseModel

from eda_platform.agents.reporting import (
    AgenticReportResult,
    ReportValidationTraceEvent,
    _apply_executive_summary_fallback,
    _bundle_from_plan,
    generate_agentic_report,
)
from eda_platform.core.kernel import SessionContext
from eda_platform.core.llm import LLMResultMetadata, LLMUsage
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers import auto_eda
from eda_platform.schemas.artifacts import (
    Artifact,
    ArtifactType,
    EvidenceRef,
    SqlResult,
)
from eda_platform.schemas.questions import (
    QuestionExecutionResult,
    QuestionFinding,
)
from eda_platform.schemas.reports import (
    ReportAudit,
    ReportBundle,
    ReportFocusItem,
    ReportPlanClaim,
    ReportPlanDraft,
    ReportStatus,
)
from eda_platform.tools.analysis import create_analysis_tables
from eda_platform.tools.chart_specs import create_chart_specs
from eda_platform.tools.evidence import EvidencePack
from eda_platform.tools.exporter import narrative_markdown, report_bundle_to_markdown
from eda_platform.tools.html_exporter import export_report_html
from eda_platform.tools.loader import load_csv
from eda_platform.tools.profiler import profile_dataset
from eda_platform.tools.quality import scan_quality

T = TypeVar("T", bound=BaseModel)

_SQL_RESULT_ID = "sql_qexec_revenue"


class FakeReportPlanLLM:
    def __init__(self, plan: ReportPlanDraft | list[ReportPlanDraft]) -> None:
        self.plans = plan if isinstance(plan, list) else [plan]
        self.payloads: list[dict[str, Any]] = []
        self.schemas: list[str] = []
        self.call_count = 0

    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        self.payloads.append({"task": task, "payload": payload})
        self.schemas.append(schema.__name__)
        index = min(self.call_count, len(self.plans) - 1)
        self.call_count += 1
        return cast(T, self.plans[index])

    def text(self, *, task: str, payload: dict) -> str:
        return "fake"

    def last_usage(self) -> None:
        return None


class _MutableLLMSettings:
    def __init__(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens


class TruncatingThenValidReportLLM:
    def __init__(self, plan: ReportPlanDraft, *, max_tokens: int = 6000) -> None:
        self.plan = plan
        self.settings = _MutableLLMSettings(max_tokens=max_tokens)
        self.payloads: list[dict[str, Any]] = []
        self.max_tokens_seen: list[int] = []
        self.call_count = 0
        self._last: LLMResultMetadata | None = None

    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        self.payloads.append({"task": task, "payload": payload})
        self.max_tokens_seen.append(self.settings.max_tokens)
        if self.call_count == 0:
            self.call_count += 1
            self._last = LLMResultMetadata(
                provider="fake",
                model="fake",
                usage=LLMUsage(
                    prompt_tokens=100,
                    completion_tokens=self.settings.max_tokens,
                    total_tokens=100 + self.settings.max_tokens,
                ),
            )
            return schema.model_validate_json('{"claims":[{"text":"cut')
        self.call_count += 1
        self._last = LLMResultMetadata(
            provider="fake",
            model="fake",
            usage=LLMUsage(prompt_tokens=50, completion_tokens=10, total_tokens=60),
        )
        return cast(T, self.plan)

    def text(self, *, task: str, payload: dict) -> str:
        return "fake"

    def last_usage(self) -> LLMResultMetadata | None:
        return self._last


def _base_artifacts(tmp_path: Path) -> list[Artifact]:
    csv_path = tmp_path / "sales.csv"
    csv_path.write_text(
        "order_id,order_date,revenue,region\n"
        "1,2026-01-01,10,East\n"
        "2,2026-01-02,,West\n"
        "3,2026-01-03,30,East\n",
        encoding="utf-8",
    )
    loaded = load_csv(csv_path, dataset_id="ds_sales")
    profile = profile_dataset(loaded, project_id="project_demo", session_id="run_demo")
    quality = scan_quality(profile, project_id="project_demo", session_id="run_demo")
    charts = create_chart_specs(loaded, profile, project_id="project_demo", session_id="run_demo")
    tables = create_analysis_tables(loaded, profile, project_id="project_demo", session_id="run_demo")
    return [profile, quality, *charts, *tables]


def _sql_result_artifact(east_revenue: float = 40.0) -> Artifact:
    result = SqlResult(
        sql="select region, sum(revenue) as total from sales group by region",
        columns=["region", "total"],
        dtypes={"region": "varchar", "total": "double"},
        rows_preview=[
            {"region": "East", "total": east_revenue},
            {"region": "West", "total": 0.0},
        ],
        row_count=2,
    )
    return Artifact(
        id=_SQL_RESULT_ID,
        type=ArtifactType.SQL_RESULT,
        project_id="project_demo",
        session_id="run_demo",
        payload=result.model_dump(mode="json"),
    )


def _qexec_artifacts(
    *, finding_value: float = 40.0, exploratory: bool = False
) -> list[Artifact]:
    succeeded = QuestionExecutionResult(
        question_id="q_revenue_by_region",
        question="Which region drives the most revenue across the sales file?",
        origin="llm" if exploratory else "template",
        plan_summary="Group revenue by region.",
        sql="select region, sum(revenue) as total from sales group by region",
        sql_result_artifact_id=_SQL_RESULT_ID,
        findings=[
            QuestionFinding(
                text=f"East region total revenue is {int(finding_value)}.",
                evidence=[
                    EvidenceRef(
                        kind="table",
                        artifact_id=_SQL_RESULT_ID,
                        locator="rows[0].total",
                        value=finding_value,
                    )
                ],
                exploratory=exploratory,
            ),
            QuestionFinding(
                text="West region total revenue is 0.",
                evidence=[
                    EvidenceRef(
                        kind="table",
                        artifact_id=_SQL_RESULT_ID,
                        locator="rows[1].total",
                        value=0,
                    )
                ],
                exploratory=exploratory,
            ),
        ],
        status="succeeded",
        exploratory=exploratory,
    )
    failed = QuestionExecutionResult(
        question_id="q_failed_join",
        question="Do orders join cleanly to a customers table?",
        origin="llm",
        status="failed",
        error="No customers table available.",
    )
    return [
        _artifact_from(succeeded, "qexec_q_revenue_by_region"),
        _artifact_from(failed, "qexec_q_failed_join"),
    ]


def _artifact_from(result: QuestionExecutionResult, artifact_id: str) -> Artifact:
    return Artifact(
        id=artifact_id,
        type=ArtifactType.QUESTION_EXECUTION_RESULT,
        project_id="project_demo",
        session_id="run_demo",
        payload=result.model_dump(mode="json"),
    )


def _minimal_plan(profile_id: str) -> ReportPlanDraft:
    return ReportPlanDraft(
        claims=[
            ReportPlanClaim(
                section_title="Dataset Overview",
                id="rows",
                text="The dataset has 3 rows.",
                evidence=[
                    EvidenceRef(kind="stat", artifact_id=profile_id, locator="rows", value=3)
                ],
                referenced_datasets=["sales.csv"],
            )
        ],
    )


def _section(result: Any, title: str):
    return next(section for section in result.bundle.sections if section.title == title)


def _section_of(bundle: Any, title: str):
    return next(section for section in bundle.sections if section.title == title)


def test_qexec_artifacts_fill_focus_and_analysis_sections(tmp_path: Path) -> None:
    base = _base_artifacts(tmp_path)
    profile = base[0]
    artifacts = [*base, _sql_result_artifact(), *_qexec_artifacts()]

    result = generate_agentic_report(
        artifacts,
        project_id="project_demo",
        session_id="run_demo",
        business_context="Revenue analysis",
        llm=FakeReportPlanLLM(_minimal_plan(profile.id)),
    )

    # F4: executed questions become structured focus items, never claims.
    focus = _section(result, "Selected Analysis Focus")
    assert focus.claims == []
    assert len(focus.focus_items) == 2
    assert all(item.question and item.outcome for item in focus.focus_items)
    assert {item.outcome for item in focus.focus_items} == {"answered", "failed"}
    assert {item.question_id for item in focus.focus_items} == {
        "q_revenue_by_region",
        "q_failed_join",
    }
    all_claims = [claim for section in result.bundle.sections for claim in section.claims]
    assert not any((claim.id or "").startswith("qfocus_") for claim in all_claims)
    # No claim may cite an artifact id that does not exist on disk.
    known_ids = {artifact.id for artifact in artifacts}
    dangling = [
        ref.artifact_id
        for claim in all_claims
        for ref in claim.evidence
        if ref.artifact_id and ref.artifact_id not in known_ids
    ]
    assert dangling == []

    # Findings become analysis claims carrying their own sql-result evidence.
    analysis = _section(result, "Agent-Performed Analysis")
    analysis_ids = {claim.id for claim in analysis.claims}
    assert analysis_ids == {
        "qfind_q_revenue_by_region_0",
        "qfind_q_revenue_by_region_1",
    }
    for claim in analysis.claims:
        assert claim.evidence
        assert claim.evidence[0].artifact_id == _SQL_RESULT_ID

    assert result.bundle.status is ReportStatus.VALIDATED
    assert result.audit.status is ReportStatus.VALIDATED
    assert result.audit.findings == []


def test_exploratory_qexec_claims_labeled_by_evidence_strength(tmp_path: Path) -> None:
    # F6: the tier comes from the evidence, not from the question's
    # exploratory origin — sql-result-backed findings are indicative and the
    # label stays visible in the rendered report.
    base = _base_artifacts(tmp_path)
    profile = base[0]
    artifacts = [
        *base,
        _sql_result_artifact(),
        *_qexec_artifacts(exploratory=True),
    ]

    result = generate_agentic_report(
        artifacts,
        project_id="project_demo",
        session_id="run_demo",
        business_context="Revenue analysis",
        llm=FakeReportPlanLLM(_minimal_plan(profile.id)),
    )
    analysis = _section(result, "Agent-Performed Analysis")

    assert analysis.claims
    assert all(claim.confidence_label == "indicative" for claim in analysis.claims)
    markdown = report_bundle_to_markdown(result.bundle, artifacts=artifacts)
    assert "[Indicative]" in markdown


def test_llm_focus_claims_are_dropped_with_trace(
    tmp_path: Path,
) -> None:
    # F4: the app owns Selected Analysis Focus. An LLM claim targeting it is
    # dropped (not merged, not moved) and the drop is recorded in the
    # validation trace; finding claims still dedupe against injection.
    base = _base_artifacts(tmp_path)
    profile = base[0]
    artifacts = [*base, _sql_result_artifact(), *_qexec_artifacts()]
    plan = ReportPlanDraft(
        claims=[
            ReportPlanClaim(
                section_title="Dataset Overview",
                id="rows",
                text="The dataset has 3 rows.",
                evidence=[
                    EvidenceRef(kind="stat", artifact_id=profile.id, locator="rows", value=3)
                ],
                referenced_datasets=["sales.csv"],
            ),
            ReportPlanClaim(
                section_title="Selected Analysis Focus",
                id="qfocus_q_revenue_by_region",
                text=(
                    'Analysis focus: "Which region drives the most revenue across '
                    'the sales file?" (outcome: answered)'
                ),
                evidence=[
                        EvidenceRef(
                            kind="artifact",
                            artifact_id="qexec_q_revenue_by_region",
                            locator="rows",
                        )
                ],
            ),
            ReportPlanClaim(
                section_title="Agent-Performed Analysis",
                id="qfind_q_revenue_by_region_0",
                text="East region total revenue is 40.",
                evidence=[
                    EvidenceRef(
                        kind="table",
                        artifact_id=_SQL_RESULT_ID,
                        locator="rows[0].total",
                        value=40.0,
                    )
                ],
            ),
        ],
    )

    result = generate_agentic_report(
        artifacts,
        project_id="project_demo",
        session_id="run_demo",
        business_context="Revenue analysis",
        llm=FakeReportPlanLLM(plan),
    )
    focus = _section(result, "Selected Analysis Focus")
    analysis = _section(result, "Agent-Performed Analysis")

    assert focus.claims == []
    assert len(focus.focus_items) == 2
    assert [claim.id for claim in analysis.claims].count("qfind_q_revenue_by_region_0") == 1
    # The drop is traced, not silent.
    assert any(
        event.dropped_focus_claim_count == 1 for event in result.validation_events
    )
    assert any(
        "Selected Analysis Focus" in note for note in result.audit.semantic_notes
    )
    assert result.bundle.status is ReportStatus.VALIDATED


def test_focus_items_render_as_lists_in_markdown_and_html(tmp_path: Path) -> None:
    # F4 R3: each focus item renders as its own list entry in both exports.
    base = _base_artifacts(tmp_path)
    profile = base[0]
    artifacts = [*base, _sql_result_artifact(), *_qexec_artifacts()]

    result = generate_agentic_report(
        artifacts,
        project_id="project_demo",
        session_id="run_demo",
        business_context="Revenue analysis",
        llm=FakeReportPlanLLM(_minimal_plan(profile.id)),
    )

    markdown = report_bundle_to_markdown(result.bundle, artifacts=artifacts)
    narrative = narrative_markdown(markdown)
    assert (
        '- Analysis focus: "Which region drives the most revenue across the '
        'sales file?" (outcome: answered)' in narrative
    )
    assert (
        '- Analysis focus: "Do orders join cleanly to a customers table?" '
        "(outcome: failed)" in narrative
    )

    html = export_report_html(result.bundle)
    assert html.count('<li class="focus-item">') == 2
    assert "(outcome: answered)" in html
    assert "(outcome: failed)" in html


def test_markdown_focus_question_markdown_syntax_is_neutralized() -> None:
    # F4 fix 1: question text is untrusted; markdown control syntax must not
    # fabricate report structure (links, headings, code spans, line breaks).
    bundle = ReportBundle.empty(project_id="project_demo", session_id="run_demo")
    focus = _section_of(bundle, "Selected Analysis Focus")
    focus.focus_items.append(
        ReportFocusItem(
            question="[click](http://evil/x)\n# Fake Heading\nwith `code`",
            outcome="failed",
            question_id="q_evil",
        )
    )

    markdown = report_bundle_to_markdown(bundle)

    assert "[click](http://evil/x)" not in markdown
    assert "`code`" not in markdown
    focus_line = next(line for line in markdown.splitlines() if "evil" in line)
    assert focus_line.startswith('- Analysis focus: "')
    assert "(outcome: failed)" in focus_line
    assert not any(line.startswith("# Fake") for line in markdown.splitlines())


def test_export_step_trace_summary_includes_dropped_focus_claim_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # F4 fix 2: the intake drop count must survive into the persisted trace.
    fake_report = AgenticReportResult(
        bundle=ReportBundle.empty(project_id="project_demo", session_id="run_demo"),
        audit=ReportAudit(status=ReportStatus.VALIDATED),
        evidence_pack=EvidencePack(payload_policy="schema+aggregates"),
        validation_events=[
            ReportValidationTraceEvent(
                attempt=1,
                status="validated",
                finding_count=0,
                critical_count=0,
                dropped_focus_claim_count=2,
            )
        ],
    )
    monkeypatch.setattr(auto_eda, "generate_agentic_report", lambda *a, **k: fake_report)
    store = ArtifactStore(tmp_path)
    ctx = SessionContext(project_id="project_demo", session_id="run_demo", store=store)
    step = auto_eda.ExportAgenticReportStep(
        [],
        business_context="Revenue analysis",
        llm=FakeReportPlanLLM(ReportPlanDraft(claims=[])),
        payload_policy="schema+aggregates",
    )

    step.run(ctx)

    events = store.list_trace_events(project_id="project_demo", session_id="run_demo")
    validation = next(e for e in events if e.event_type == "report_validation")
    assert validation.summary["dropped_focus_claim_count"] == 2


def test_failed_focus_items_are_not_labeled_validated(tmp_path: Path) -> None:
    # F4 fix 3: a focus list holding failed questions must not sit under a
    # "Validated evidence-backed findings" banner.
    base = _base_artifacts(tmp_path)
    profile = base[0]
    failed = QuestionExecutionResult(
        question_id="q_failed_join",
        question="Do orders join cleanly to a customers table?",
        origin="llm",
        status="failed",
        error="No customers table available.",
    )
    artifacts = [*base, _artifact_from(failed, "qexec_failed_only")]

    result = generate_agentic_report(
        artifacts,
        project_id="project_demo",
        session_id="run_demo",
        business_context="Revenue analysis",
        llm=FakeReportPlanLLM(_minimal_plan(profile.id)),
    )

    focus = _section(result, "Selected Analysis Focus")
    assert focus.claims == []
    assert {item.outcome for item in focus.focus_items} == {"failed"}
    assert "Validated" not in focus.body
    assert "Analysis focus" in focus.body
    markdown = report_bundle_to_markdown(result.bundle, artifacts=artifacts)
    assert focus.body in markdown


def test_focus_claim_title_case_variants_are_dropped_not_appendixed() -> None:
    # F4 fix 4: case variants of the app-owned section title are dropped at
    # plan intake instead of slipping into the Appendix fallback.
    draft = ReportPlanDraft(
        claims=[
            ReportPlanClaim(
                section_title="SELECTED ANALYSIS FOCUS",
                id="qfocus_upper",
                text="Focus claim with a shouted title.",
            ),
            ReportPlanClaim(
                section_title="selected analysis focus",
                id="qfocus_lower",
                text="Focus claim with a lowercase title.",
            ),
        ],
    )

    bundle, dropped = _bundle_from_plan(draft, project_id="project_demo", session_id="run_demo")

    assert dropped == ["qfocus_upper", "qfocus_lower"]
    appendix = _section_of(bundle, "Appendix: Charts and Technical Summary")
    assert appendix.claims == []
    assert _section_of(bundle, "Selected Analysis Focus").claims == []


def test_executive_summary_fallback_handles_empty_sources(tmp_path: Path) -> None:
    # F4: with the qfocus filter gone, an all-empty source pool must inject
    # nothing and not crash.
    base = _base_artifacts(tmp_path)
    profile = base[0]
    artifacts = [*base, _sql_result_artifact(), *_qexec_artifacts()]

    result = generate_agentic_report(
        artifacts,
        project_id="project_demo",
        session_id="run_demo",
        business_context="Revenue analysis",
        llm=FakeReportPlanLLM(_minimal_plan(profile.id)),
    )
    bundle = result.bundle.model_copy(deep=True)
    for section in bundle.sections:
        section.claims = []
    injected = _apply_executive_summary_fallback(bundle, [])
    assert injected == 0
    assert _section_of(bundle, "Executive Summary").claims == []


def test_qexec_digest_reaches_llm_manifest(tmp_path: Path) -> None:
    base = _base_artifacts(tmp_path)
    profile = base[0]
    artifacts = [*base, _sql_result_artifact(), *_qexec_artifacts()]
    llm = FakeReportPlanLLM(_minimal_plan(profile.id))

    generate_agentic_report(
        artifacts,
        project_id="project_demo",
        session_id="run_demo",
        business_context="Revenue analysis",
        llm=llm,
    )

    payload = llm.payloads[0]["payload"]
    assert "question_results" in payload
    digest = payload["question_results"]
    assert {entry["question_id"] for entry in digest} == {
        "q_revenue_by_region",
        "q_failed_join",
    }
    revenue_entry = next(e for e in digest if e["question_id"] == "q_revenue_by_region")
    assert revenue_entry["sql_result_artifact_id"] == _SQL_RESULT_ID
    evidence = revenue_entry["findings"][0]["evidence"][0]
    assert evidence["artifact_id"] == _SQL_RESULT_ID
    # Optional exact-unit provenance must not add null keys/tokens to the
    # overwhelmingly common unseeded path.
    assert "unit_label" not in evidence
    assert "unit_reference" not in evidence


def test_tampered_finding_value_is_pruned_by_validator(tmp_path: Path) -> None:
    base = _base_artifacts(tmp_path)
    profile = base[0]
    # sql-result says East total is 40 but the finding claims 999.
    tampered = QuestionExecutionResult(
        question_id="q_tampered",
        question="Which region leads revenue?",
        origin="template",
        sql_result_artifact_id=_SQL_RESULT_ID,
        findings=[
            QuestionFinding(
                text="East region total revenue is 999.",
                evidence=[
                    EvidenceRef(
                        kind="table",
                        artifact_id=_SQL_RESULT_ID,
                        locator="rows[0].total",
                    )
                ],
            )
        ],
        status="succeeded",
    )
    artifacts = [
        *base,
        _sql_result_artifact(east_revenue=40.0),
        _artifact_from(tampered, "qexec_q_tampered"),
    ]

    result = generate_agentic_report(
        artifacts,
        project_id="project_demo",
        session_id="run_demo",
        business_context="Revenue analysis",
        llm=FakeReportPlanLLM(_minimal_plan(profile.id)),
    )

    analysis = _section(result, "Agent-Performed Analysis")
    business = _section(result, "Business Findings")
    tampered_ids = {
        claim.id
        for section in result.bundle.sections
        for claim in section.claims
        if "999" in claim.text
    }
    # The unsupported 999 claim survives nowhere in the report.
    assert tampered_ids == set()
    assert all("999" not in claim.text for claim in analysis.claims)
    assert all("999" not in claim.text for claim in business.claims)
    assert result.bundle.status is ReportStatus.VALIDATED
    assert result.audit.status is ReportStatus.VALIDATED


def test_business_findings_fallback_injects_when_llm_returns_no_business_claims(
    tmp_path: Path,
) -> None:
    base = _base_artifacts(tmp_path)
    profile = base[0]
    artifacts = [*base, _sql_result_artifact(), *_qexec_artifacts()]
    # LLM proposes only a Dataset Overview claim -> no Business Findings claims.
    result = generate_agentic_report(
        artifacts,
        project_id="project_demo",
        session_id="run_demo",
        business_context="Revenue analysis",
        llm=FakeReportPlanLLM(_minimal_plan(profile.id)),
    )

    business = _section(result, "Business Findings")
    assert business.claims
    assert len(business.claims) <= 3
    for claim in business.claims:
        assert claim.evidence
        assert claim.id.startswith("qbiz_")
    assert any(
        "Injected" in note and "Business Findings" in note
        for note in result.audit.semantic_notes
    )
    assert result.bundle.status is ReportStatus.VALIDATED


def test_business_findings_fallback_skipped_when_llm_supplies_business_claim(
    tmp_path: Path,
) -> None:
    base = _base_artifacts(tmp_path)
    artifacts = [*base, _sql_result_artifact(), *_qexec_artifacts()]
    plan = ReportPlanDraft(
        claims=[
            ReportPlanClaim(
                section_title="Business Findings",
                id="llm_business",
                text="East region total revenue is 40.",
                evidence=[
                    EvidenceRef(
                        kind="table",
                        artifact_id=_SQL_RESULT_ID,
                        locator="rows[0].total",
                        value=40,
                    )
                ],
            )
        ],
    )

    result = generate_agentic_report(
        artifacts,
        project_id="project_demo",
        session_id="run_demo",
        business_context="Revenue analysis",
        llm=FakeReportPlanLLM(plan),
    )

    business = _section(result, "Business Findings")
    assert [claim.id for claim in business.claims] == ["llm_business"]
    assert not any("Business Findings" in note for note in result.audit.semantic_notes)


def test_truncated_report_attempt_raises_budget_and_shortens_retry(
    tmp_path: Path,
) -> None:
    base = _base_artifacts(tmp_path)
    profile = base[0]
    llm = TruncatingThenValidReportLLM(_minimal_plan(profile.id), max_tokens=6000)

    result = generate_agentic_report(
        base,
        project_id="project_demo",
        session_id="run_demo",
        business_context="Revenue analysis",
        llm=llm,
    )

    assert result.bundle.status is ReportStatus.VALIDATED
    assert llm.max_tokens_seen == [6000, 9000]
    assert llm.payloads[0]["payload"] != llm.payloads[1]["payload"]
    assert llm.payloads[1]["payload"]["max_claims"] < llm.payloads[0]["payload"][
        "max_claims"
    ]
    assert "shorter" in llm.payloads[1]["payload"]["instructions"].lower()
    assert "truncat" in result.llm_events[0].error_type.lower()


def test_dataset_overview_gets_deterministic_claim_when_profiles_exist(
    tmp_path: Path,
) -> None:
    base = _base_artifacts(tmp_path)
    empty_plan = ReportPlanDraft(claims=[])

    result = generate_agentic_report(
        base,
        project_id="project_demo",
        session_id="run_demo",
        business_context="Revenue analysis",
        llm=FakeReportPlanLLM(empty_plan),
    )

    overview = _section(result, "Dataset Overview")
    assert overview.claims
    assert "sales.csv" in overview.claims[0].text
    assert "3 rows" in overview.claims[0].text
    assert "4 columns" in overview.claims[0].text
    assert overview.body != "No validated conclusion is available for this section."
    assert result.bundle.status is ReportStatus.VALIDATED


def test_executive_summary_gets_deterministic_claim_from_surviving_claims(
    tmp_path: Path,
) -> None:
    base = _base_artifacts(tmp_path)
    profile = base[0]

    result = generate_agentic_report(
        base,
        project_id="project_demo",
        session_id="run_demo",
        business_context="Revenue analysis",
        llm=FakeReportPlanLLM(_minimal_plan(profile.id)),
    )

    summary = _section(result, "Executive Summary")
    assert summary.claims
    assert summary.claims[0].evidence
    assert summary.body != "No validated conclusion is available for this section."
    assert result.bundle.status is ReportStatus.VALIDATED


def test_zero_qexec_artifacts_leaves_report_behavior_unchanged(tmp_path: Path) -> None:
    base = _base_artifacts(tmp_path)
    profile = base[0]
    plan = _minimal_plan(profile.id)

    baseline = generate_agentic_report(
        base,
        project_id="project_demo",
        session_id="run_demo",
        business_context="Revenue analysis",
        llm=FakeReportPlanLLM(plan),
    )

    # No qexec/sql-result artifacts -> business sections carry no injected claims.
    focus = _section(baseline, "Selected Analysis Focus")
    analysis = _section(baseline, "Agent-Performed Analysis")
    business = _section(baseline, "Business Findings")
    assert focus.claims == []
    assert analysis.claims == []
    assert business.claims == []
    assert baseline.bundle.status is ReportStatus.VALIDATED
    # LLM manifest carries no question digest when there are no executed questions.
    assert "question_results" not in baseline.evidence_pack.artifact_index
    llm = FakeReportPlanLLM(plan)
    generate_agentic_report(
        base,
        project_id="project_demo",
        session_id="run_demo",
        business_context="Revenue analysis",
        llm=llm,
    )
    assert "question_results" not in llm.payloads[0]["payload"]
    assert not any("Business Findings" in note for note in baseline.audit.semantic_notes)
