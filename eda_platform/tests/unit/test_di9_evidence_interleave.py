"""DI9 H9-A: write-time evidence interleaving (EvidFuse layer).

Covers: typed request resolution per artifact type (positive and negative),
teaching-style rejections, the per-section / whole-report budgets, the
decision-report gate over interleave-granted numbers, the zero-regression
fallback when the LLM is absent or crashes, transcript persistence, and the
agentic report plan channel. All LLM calls are mocked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar, cast

import pytest
from pydantic import BaseModel

from eda_platform.agents.evidence_interleave import (
    EvidenceInterleaveSession,
    InMemoryEvidenceResolver,
    StoreEvidenceResolver,
)
from eda_platform.agents.reporting import generate_agentic_report
from eda_platform.core.llm import OfflineLLMClient
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.decision_report import create_decision_report
from eda_platform.schemas.artifacts import (
    Artifact,
    ArtifactType,
    DatasetProfile,
    EvidenceRef,
    SqlResult,
)
from eda_platform.schemas.decision_report import DecisionReport
from eda_platform.schemas.investigations import ValidatedFinding
from eda_platform.schemas.quality_context import QualityContext
from eda_platform.schemas.questions import QuestionFinding
from eda_platform.schemas.reports import (
    EvidenceGrant,
    EvidenceRejection,
    EvidenceRequest,
    InterleaveTranscript,
    ReportPlanClaim,
    ReportPlanDraft,
)
from eda_platform.schemas.stats import StatTestResult
from eda_platform.schemas.synthesis import SynthesisBrief, SynthesisStoryBeat
from eda_platform.tools.loader import load_csv
from eda_platform.tools.profiler import profile_dataset

T = TypeVar("T", bound=BaseModel)

_PROJECT = "project_di9"
_RUN = "run_di9"


class ScriptedLLM:
    """Returns scripted dict responses validated against the requested schema."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.payloads: list[dict[str, Any]] = []
        self.calls = 0

    def structured(self, *, task: str, schema: type[T], payload: dict[str, Any]) -> T:
        self.payloads.append(payload)
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return schema.model_validate(response)

    def text(self, *, task: str, payload: dict[str, Any]) -> str:
        return "scripted"

    def last_usage(self) -> None:
        return None


class CrashingLLM:
    def structured(self, *, task: str, schema: type[T], payload: dict[str, Any]) -> T:
        raise RuntimeError("LLM transport failure")

    def text(self, *, task: str, payload: dict[str, Any]) -> str:
        return ""

    def last_usage(self) -> None:
        return None


class ScriptedPlanLLM:
    """Agentic-report fake: returns prebuilt ReportPlanDraft objects in order."""

    def __init__(self, drafts: list[ReportPlanDraft]) -> None:
        self.drafts = drafts
        self.payloads: list[dict[str, Any]] = []
        self.calls = 0

    def structured(self, *, task: str, schema: type[T], payload: dict[str, Any]) -> T:
        self.payloads.append(payload)
        draft = self.drafts[min(self.calls, len(self.drafts) - 1)]
        self.calls += 1
        return cast(T, draft)

    def text(self, *, task: str, payload: dict[str, Any]) -> str:
        return "scripted"

    def last_usage(self) -> None:
        return None


def _sql_artifact() -> Artifact:
    payload = SqlResult(
        sql="select late_rate from kpis",
        columns=["late_rate", "orders"],
        dtypes={"late_rate": "double", "orders": "bigint"},
        rows_preview=[{"late_rate": 12.5, "orders": 480}],
        row_count=1,
    ).model_dump(mode="json")
    return Artifact(
        id="sqlres_di9",
        type=ArtifactType.SQL_RESULT,
        project_id=_PROJECT,
        session_id=_RUN,
        payload=payload,
    )


def _stat_artifact() -> Artifact:
    payload = StatTestResult(
        dataset_id="ds_orders",
        test_type="independent_t_test",
        statistic=2.4,
        p_value=0.03,
        effect_size=0.5,
        sample_size=500,
    ).model_dump(mode="json")
    return Artifact(
        id="stat_di9",
        type=ArtifactType.STAT_TEST_RESULT,
        project_id=_PROJECT,
        session_id=_RUN,
        payload=payload,
    )


def _profile_artifact() -> Artifact:
    payload = DatasetProfile(
        dataset_id="ds_orders",
        name="orders.csv",
        rows=500,
        columns=2,
        column_names=["order_id", "amount"],
        dtypes={"order_id": "int64", "amount": "float64"},
        missing_values={"order_id": 0, "amount": 40},
        missing_percent={"order_id": 0.0, "amount": 8.0},
        numeric_columns=["amount"],
        categorical_columns=[],
    ).model_dump(mode="json")
    return Artifact(
        id="profile_di9",
        type=ArtifactType.DATASET_PROFILE,
        project_id=_PROJECT,
        session_id=_RUN,
        payload=payload,
    )


def _chart_artifact() -> Artifact:
    return Artifact(
        id="chart_di9",
        type=ArtifactType.CHART_SPEC,
        project_id=_PROJECT,
        session_id=_RUN,
        payload={"title": "Late rate by month"},
    )


def _validated_finding() -> ValidatedFinding:
    return ValidatedFinding(
        finding_id="finding_orders",
        investigation_id="inv_orders",
        question_id="q_orders",
        question="How do order values vary by channel?",
        quality_context=[
            QualityContext(
                context_id="context_orders",
                dataset_id="ds_orders",
                dataset_name="orders.csv",
                issue_code="high_missing",
                severity="warn",
                column="amount",
                observation="Amount completeness was reviewed.",
                report_limitation="Amount coverage may narrow the interpretation.",
            )
        ],
        claim_class="observed",
        findings=[
            QuestionFinding(
                text="The observed average order value is 125.5.",
                evidence=[
                    EvidenceRef(
                        kind="sql",
                        artifact_id="sqlres_di9",
                        locator="rows[0].orders",
                        value=125.5,
                    )
                ],
            )
        ],
        evidence_support="high",
        analytical_reliability="high",
        decision_readiness="medium",
        report_eligible=True,
        report_readiness="eligible_with_limitations",
        report_readiness_reason="Validated with disclosed data conditions.",
        source_artifact_ids=["sqlres_di9", "stat_di9", "profile_di9"],
    )


def _finding_artifact() -> Artifact:
    return Artifact(
        id="vf_di9",
        type=ArtifactType.VALIDATED_FINDING,
        project_id=_PROJECT,
        session_id=_RUN,
        payload=_validated_finding().model_dump(mode="json"),
    )


def _all_artifacts() -> list[Artifact]:
    return [_sql_artifact(), _stat_artifact(), _profile_artifact(), _chart_artifact(),
            _finding_artifact()]


def _seed_store(tmp_path: Path) -> tuple[ArtifactStore, str]:
    store = ArtifactStore(tmp_path / "workspace")
    store.ensure_project(_PROJECT, "DI9")
    store.start_session(_PROJECT, _RUN)
    for artifact in _all_artifacts():
        store.save_artifact(artifact)
    brief = SynthesisBrief(
        brief_id="brief_di9",
        project_id=_PROJECT,
        selected_finding_artifact_ids=["vf_di9"],
        decision_context="Which evidence should guide the fulfilment decision?",
        headline="The observed average order value is 125.5.",
        storyline=[
            SynthesisStoryBeat(
                title="Evidence",
                body="Validated findings are ready for review.",
                finding_artifact_ids=["vf_di9"],
            )
        ],
        report_eligible=True,
        report_readiness="eligible_with_limitations",
    )
    brief_artifact = Artifact(
        id="sbrief_di9",
        type=ArtifactType.SYNTHESIS_BRIEF,
        project_id=_PROJECT,
        session_id=_RUN,
        parents=["vf_di9"],
        payload=brief.model_dump(mode="json"),
    )
    store.save_artifact(brief_artifact)
    return store, brief_artifact.id


def _session(
    artifacts: list[Artifact] | None = None,
    *,
    per_section_limit: int = 2,
    total_limit: int = 8,
) -> EvidenceInterleaveSession:
    return EvidenceInterleaveSession(
        InMemoryEvidenceResolver(artifacts if artifacts is not None else _all_artifacts()),
        per_section_limit=per_section_limit,
        total_limit=total_limit,
    )


# --------------------------------------------------------------------------- #
# Typed resolution per artifact type — positive and negative cases.
# --------------------------------------------------------------------------- #


def test_sql_result_locator_grants_cell_values() -> None:
    session = _session()
    outcome = session.request(
        EvidenceRequest(artifact_id="sqlres_di9", locator="rows[0].late_rate"),
        section="answer",
    )
    assert isinstance(outcome, EvidenceGrant)
    assert outcome.artifact_type == "SqlResult"
    assert [(item.value, item.unit) for item in outcome.values] == [(12.5, "raw")]

    column = session.request(
        EvidenceRequest(artifact_id="sqlres_di9", locator="orders"), section="answer"
    )
    assert isinstance(column, EvidenceGrant)
    assert [(item.value, item.unit) for item in column.values] == [(480.0, "raw")]


def test_stat_test_locator_grants_field_and_rejects_unknown_field() -> None:
    session = _session()
    granted = session.request(
        EvidenceRequest(artifact_id="stat_di9", locator="p_value"), section="answer"
    )
    assert isinstance(granted, EvidenceGrant)
    assert [(item.value, item.unit) for item in granted.values] == [(0.03, "raw")]

    rejected = session.request(
        EvidenceRequest(artifact_id="stat_di9", locator="pvalue"), section="answer"
    )
    assert isinstance(rejected, EvidenceRejection)
    assert rejected.reason_code == "unresolvable_locator"
    # Teaching-style: names the valid locators and the usable catalog.
    assert "p_value" in rejected.message
    assert rejected.available_artifacts


def test_dataset_profile_missing_percent_grants_percent_unit() -> None:
    session = _session()
    granted = session.request(
        EvidenceRequest(artifact_id="profile_di9", locator="missing_percent.amount"),
        section="situation",
    )
    assert isinstance(granted, EvidenceGrant)
    assert [(item.value, item.unit) for item in granted.values] == [(8.0, "percent")]

    rejected = session.request(
        EvidenceRequest(artifact_id="profile_di9", locator="mean.amount"),
        section="situation",
    )
    assert isinstance(rejected, EvidenceRejection)
    assert rejected.reason_code == "unresolvable_locator"
    assert "missing_percent.<column>" in rejected.message


def test_validated_finding_grants_claim_texts_and_rejects_bad_index() -> None:
    session = _session()
    granted = session.request(
        EvidenceRequest(artifact_id="vf_di9", locator=""), section="answer"
    )
    assert isinstance(granted, EvidenceGrant)
    assert granted.texts == ["The observed average order value is 125.5."]
    assert [(item.value, item.unit) for item in granted.values] == [(125.5, "raw")]

    rejected = session.request(
        EvidenceRequest(artifact_id="vf_di9", locator="findings[9]"), section="answer"
    )
    assert isinstance(rejected, EvidenceRejection)
    assert rejected.reason_code == "unresolvable_locator"


def test_unknown_artifact_and_unsupported_type_are_teaching_rejections() -> None:
    session = _session()
    unknown = session.request(
        EvidenceRequest(artifact_id="missing_artifact", locator="rows"), section="answer"
    )
    assert isinstance(unknown, EvidenceRejection)
    assert unknown.reason_code == "unknown_artifact"
    assert any("sqlres_di9" in entry for entry in unknown.available_artifacts)

    unsupported = session.request(
        EvidenceRequest(artifact_id="chart_di9", locator="title"), section="answer"
    )
    assert isinstance(unsupported, EvidenceRejection)
    assert unsupported.reason_code == "unsupported_type"
    assert "SqlResult" in unsupported.message


def test_store_resolver_hides_other_projects(tmp_path: Path) -> None:
    store, _ = _seed_store(tmp_path)
    foreign = Artifact(
        id="sqlres_foreign",
        type=ArtifactType.SQL_RESULT,
        project_id="another_project",
        session_id="another_run",
        payload=_sql_artifact().payload,
    )
    store.ensure_project("another_project", "Other")
    store.start_session("another_project", "another_run")
    store.save_artifact(foreign)

    resolver = StoreEvidenceResolver(
        store, project_id=_PROJECT, catalog_artifact_ids=["sqlres_di9"]
    )
    session = EvidenceInterleaveSession(resolver)
    outcome = session.request(
        EvidenceRequest(artifact_id="sqlres_foreign", locator="rows"), section="answer"
    )
    assert isinstance(outcome, EvidenceRejection)
    assert outcome.reason_code == "unknown_artifact"


# --------------------------------------------------------------------------- #
# Bounded loop: per-section and whole-report budgets.
# --------------------------------------------------------------------------- #


def test_per_section_budget_rejects_third_request() -> None:
    session = _session()
    request = EvidenceRequest(artifact_id="sqlres_di9", locator="rows")
    assert isinstance(session.request(request, section="answer"), EvidenceGrant)
    assert isinstance(session.request(request, section="answer"), EvidenceGrant)
    third = session.request(request, section="answer")
    assert isinstance(third, EvidenceRejection)
    assert third.reason_code == "budget_exhausted"
    # Another section still has budget.
    assert isinstance(session.request(request, section="situation"), EvidenceGrant)


def test_total_budget_caps_the_whole_report() -> None:
    session = _session(per_section_limit=8, total_limit=8)
    request = EvidenceRequest(artifact_id="sqlres_di9", locator="rows")
    for index in range(8):
        outcome = session.request(request, section=f"section_{index}")
        assert isinstance(outcome, EvidenceGrant)
    assert session.remaining_total == 0
    ninth = session.request(request, section="section_extra")
    assert isinstance(ninth, EvidenceRejection)
    assert ninth.reason_code == "budget_exhausted"
    transcript = session.transcript
    assert transcript.granted_count == 8
    assert transcript.rejected_count == 1


def test_failed_resolution_still_consumes_budget() -> None:
    session = _session()
    bad = EvidenceRequest(artifact_id="missing", locator="rows")
    assert isinstance(session.request(bad, section="answer"), EvidenceRejection)
    assert isinstance(session.request(bad, section="answer"), EvidenceRejection)
    third = session.request(bad, section="answer")
    assert isinstance(third, EvidenceRejection)
    assert third.reason_code == "budget_exhausted"


# --------------------------------------------------------------------------- #
# Decision report: interleaved SCQA writing under the existing gates.
# --------------------------------------------------------------------------- #


def test_interleaved_answer_passes_gate_only_with_granted_number(tmp_path: Path) -> None:
    store, brief_id = _seed_store(tmp_path)
    llm = ScriptedLLM(
        [
            {
                "evidence_requests": [
                    {
                        "artifact_id": "sqlres_di9",
                        "locator": "rows[0].late_rate",
                        "section": "answer",
                    }
                ]
            },
            {
                "situation": "The fulfilment analysis frames the current decision.",
                "complication": "Coverage conditions narrow interpretation.",
                "answer": (
                    "The observed average order value is 125.5 and the granted "
                    "late rate metric is 12.5."
                ),
            },
        ]
    )
    report_id = create_decision_report(
        store, project_id=_PROJECT, brief_artifact_id=brief_id, llm=llm
    )
    report = DecisionReport.model_validate(store.get_artifact(report_id).payload)

    assert llm.calls == 2
    assert report.narrative_status == "llm_refined"
    assert "12.5" in report.scqa.answer
    # The interleave channel surfaced the grant to the second round.
    assert "granted_evidence" in llm.payloads[1]
    # Provenance: the requested artifact is recorded on the report.
    assert report.granted_evidence_artifact_ids == ["sqlres_di9"]
    assert report.interleave_transcript_artifact_id is not None


def test_non_granted_number_falls_back_to_deterministic(tmp_path: Path) -> None:
    store, brief_id = _seed_store(tmp_path)
    llm = ScriptedLLM(
        [
            {
                "evidence_requests": [
                    {
                        "artifact_id": "sqlres_di9",
                        "locator": "rows[0].late_rate",
                        "section": "answer",
                    }
                ]
            },
            {
                "situation": "The fulfilment analysis frames the current decision.",
                "complication": "Coverage conditions narrow interpretation.",
                "answer": "A fabricated late rate of 47 drives the decision.",
            },
        ]
    )
    report_id = create_decision_report(
        store, project_id=_PROJECT, brief_artifact_id=brief_id, llm=llm
    )
    report = DecisionReport.model_validate(store.get_artifact(report_id).payload)

    assert report.narrative_status == "deterministic"
    assert "47" not in report.scqa.answer


def test_transcript_artifact_is_persisted_with_trace_events(tmp_path: Path) -> None:
    store, brief_id = _seed_store(tmp_path)
    llm = ScriptedLLM(
        [
            {
                "evidence_requests": [
                    {
                        "artifact_id": "sqlres_di9",
                        "locator": "rows[0].late_rate",
                        "section": "answer",
                    },
                    {"artifact_id": "nope", "locator": "rows", "section": "answer"},
                ]
            },
            {
                "situation": "The fulfilment analysis frames the current decision.",
                "complication": "Coverage conditions narrow interpretation.",
                "answer": "Use the validated observations in the decision review.",
            },
        ]
    )
    report_id = create_decision_report(
        store, project_id=_PROJECT, brief_artifact_id=brief_id, llm=llm
    )
    report_artifact = store.get_artifact(report_id)
    report = DecisionReport.model_validate(report_artifact.payload)

    assert report.interleave_transcript_artifact_id is not None
    transcript_artifact = store.get_artifact(report.interleave_transcript_artifact_id)
    assert transcript_artifact.type is ArtifactType.EVIDENCE_INTERLEAVE_TRANSCRIPT
    assert report.interleave_transcript_artifact_id in report_artifact.parents
    transcript = InterleaveTranscript.model_validate(transcript_artifact.payload)
    assert transcript.granted_count == 1
    assert transcript.rejected_count == 1
    assert transcript.exchanges[0].grant is not None
    assert transcript.exchanges[1].rejection is not None
    assert transcript.exchanges[1].rejection.reason_code == "unknown_artifact"

    event_types = {
        event.event_type for event in store.list_trace_events(project_id=_PROJECT, session_id=_RUN)
    }
    assert "evidence_interleave_request" in event_types
    assert "evidence_interleave_granted" in event_types
    assert "evidence_interleave_rejected" in event_types


def test_transcript_save_failure_rolls_back_refined_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, brief_id = _seed_store(tmp_path)
    llm = ScriptedLLM(
        [
            {
                "evidence_requests": [
                    {
                        "artifact_id": "sqlres_di9",
                        "locator": "rows[0].late_rate",
                        "section": "answer",
                    }
                ]
            },
            {
                "situation": "The fulfilment analysis frames the current decision.",
                "complication": "Coverage conditions narrow interpretation.",
                "answer": "The granted late rate metric is 12.5.",
            },
        ]
    )
    original_save = store.save_artifact

    def fail_transcript(artifact: Artifact) -> None:
        if artifact.type is ArtifactType.EVIDENCE_INTERLEAVE_TRANSCRIPT:
            raise OSError("transcript storage unavailable")
        original_save(artifact)

    monkeypatch.setattr(store, "save_artifact", fail_transcript)

    report_id = create_decision_report(
        store, project_id=_PROJECT, brief_artifact_id=brief_id, llm=llm
    )
    report = DecisionReport.model_validate(store.get_artifact(report_id).payload)

    assert report.narrative_status == "deterministic"
    assert "12.5" not in report.scqa.answer
    assert report.interleave_transcript_artifact_id is None
    assert report.granted_evidence_artifact_ids == []


def test_llm_absent_path_is_unchanged_snapshot(tmp_path: Path) -> None:
    store, brief_id = _seed_store(tmp_path)
    baseline_id = create_decision_report(
        store, project_id=_PROJECT, brief_artifact_id=brief_id, llm=None
    )
    baseline = store.get_artifact(baseline_id)

    # An offline client never attempts a rewrite, so it must stay byte-identical
    # (the artifact id is content-derived, so id equality is a strict snapshot).
    offline_id = create_decision_report(
        store, project_id=_PROJECT, brief_artifact_id=brief_id, llm=OfflineLLMClient()
    )
    assert offline_id == baseline_id

    # A crashing live client falls back to the same narrative CONTENT, but the
    # fallback is disclosed rather than silent (2026-07-22 audit).
    crashed_id = create_decision_report(
        store, project_id=_PROJECT, brief_artifact_id=brief_id, llm=CrashingLLM()
    )
    crashed = DecisionReport.model_validate(store.get_artifact(crashed_id).payload)
    baseline_report = DecisionReport.model_validate(baseline.payload)
    assert crashed.scqa == baseline_report.scqa
    assert crashed.narrative_status == "deterministic"
    assert crashed.narrative_fallback_reason == "rewrite_call_failed:RuntimeError"
    assert baseline_report.narrative_fallback_reason == ""

    report = DecisionReport.model_validate(baseline.payload)
    assert report.narrative_status == "deterministic"
    assert report.interleave_transcript_artifact_id is None
    assert report.granted_evidence_artifact_ids == []
    assert "orders.csv" in report.scqa.situation
    # No transcript artifact was persisted for any of the three sessions.
    persisted_types = {
        artifact.type for artifact in store.list_artifacts(project_id=_PROJECT, session_id=_RUN)
    }
    assert ArtifactType.EVIDENCE_INTERLEAVE_TRANSCRIPT not in persisted_types


# --------------------------------------------------------------------------- #
# Agentic report: bounded evidence channel around the claim plan.
# --------------------------------------------------------------------------- #


def _report_artifacts(tmp_path: Path) -> list[Artifact]:
    csv_path = tmp_path / "sales.csv"
    csv_path.write_text(
        "order_id,order_date,revenue,region\n"
        "1,2026-01-01,10,East\n"
        "2,2026-01-02,20,West\n"
        "3,2026-01-03,30,East\n",
        encoding="utf-8",
    )
    loaded = load_csv(csv_path, dataset_id="ds_sales")
    profile = profile_dataset(loaded, project_id="project_demo", session_id="run_demo")
    return [profile]


def test_plan_interleave_attaches_grants_and_keeps_validator_flow(tmp_path: Path) -> None:
    artifacts = _report_artifacts(tmp_path)
    profile = artifacts[0]
    request_draft = ReportPlanDraft(
        evidence_requests=[EvidenceRequest(artifact_id=profile.id, locator="rows")]
    )
    plan_draft = ReportPlanDraft(
        claims=[
            ReportPlanClaim(
                section_title="Dataset Overview",
                id="rows",
                text="The dataset has 3 rows.",
                evidence=[
                    EvidenceRef(kind="stat", artifact_id=profile.id, locator="rows", value=3)
                ],
                referenced_datasets=["sales.csv"],
            )
        ],
    )
    llm = ScriptedPlanLLM([request_draft, plan_draft])

    result = generate_agentic_report(
        artifacts,
        project_id="project_demo",
        session_id="run_demo",
        business_context="Revenue analysis",
        llm=llm,
    )

    assert llm.calls == 2
    assert "evidence_interleave" in llm.payloads[0]
    requested = llm.payloads[1]["requested_evidence"]
    assert len(requested) == 1
    assert requested[0]["artifact_id"] == profile.id
    assert any(item["value"] == 3.0 for item in requested[0]["values"])
    assert result.interleave_transcript is not None
    assert result.interleave_transcript.granted_count == 1
    assert result.bundle.sections[1].claims[0].id == "rows"
    assert not result.used_fallback


def test_plan_interleave_budget_is_capped_at_four(tmp_path: Path) -> None:
    artifacts = _report_artifacts(tmp_path)
    profile = artifacts[0]
    request_draft = ReportPlanDraft(
        evidence_requests=[
            EvidenceRequest(artifact_id=profile.id, locator="rows") for _ in range(6)
        ]
    )
    plan_draft = ReportPlanDraft(claims=[])
    llm = ScriptedPlanLLM([request_draft, plan_draft])

    result = generate_agentic_report(
        artifacts,
        project_id="project_demo",
        session_id="run_demo",
        business_context="Revenue analysis",
        llm=llm,
    )

    transcript = result.interleave_transcript
    assert transcript is not None
    assert transcript.granted_count == 4
    assert transcript.rejected_count == 2
    assert all(
        exchange.rejection.reason_code == "budget_exhausted"
        for exchange in transcript.exchanges
        if exchange.rejection is not None
    )


def test_plan_without_requests_needs_single_call(tmp_path: Path) -> None:
    artifacts = _report_artifacts(tmp_path)
    profile = artifacts[0]
    plan_draft = ReportPlanDraft(
        claims=[
            ReportPlanClaim(
                section_title="Dataset Overview",
                id="rows",
                text="The dataset has 3 rows.",
                evidence=[
                    EvidenceRef(kind="stat", artifact_id=profile.id, locator="rows", value=3)
                ],
                referenced_datasets=["sales.csv"],
            )
        ],
    )
    llm = ScriptedPlanLLM([plan_draft])

    result = generate_agentic_report(
        artifacts,
        project_id="project_demo",
        session_id="run_demo",
        business_context="Revenue analysis",
        llm=llm,
    )

    assert llm.calls == 1
    assert result.interleave_transcript is None


def test_offline_llm_keeps_deterministic_fallback_without_interleave(tmp_path: Path) -> None:
    artifacts = _report_artifacts(tmp_path)
    result = generate_agentic_report(
        artifacts,
        project_id="project_demo",
        session_id="run_demo",
        business_context="Revenue analysis",
        llm=OfflineLLMClient(),
    )
    assert result.used_fallback
    assert result.interleave_transcript is None
