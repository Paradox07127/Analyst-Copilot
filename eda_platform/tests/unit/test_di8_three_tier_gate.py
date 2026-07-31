"""DI8-D three-tier gate: hard gate binary, semantic soft gate three-tier.

Covers the sprint-8 red lines:

- A correctness (hard-gate) failure is always ``rejected`` — it can never be
  downgraded into a degraded publish.
- Semantic failures degrade with *structured* confidence labels (machine-
  readable fields, not disclaimer prose) and the claim keeps publishing.
- Insight ranking is impact x significance: identifier/sequence-column
  findings score impact 0 and sink to the bottom instead of being deleted.
- The decision-report executive answer follows the MetaInsight
  "commonality + exception" skeleton, deterministically (no LLM required).

All LLM interactions are mocked or absent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar, cast

from pydantic import BaseModel

from eda_platform.agents.reporting import generate_agentic_report
from eda_platform.core.column_roles import (
    ColumnRole,
    ColumnRoleName,
    ColumnRoleSet,
    column_role_set_artifact,
)
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.decision_report import create_decision_report
from eda_platform.schemas.artifacts import Artifact, ArtifactType, EvidenceRef
from eda_platform.schemas.decision_report import DecisionReport
from eda_platform.schemas.investigations import ValidatedFinding
from eda_platform.schemas.questions import FindingScore, QuestionFinding
from eda_platform.schemas.reports import (
    ReportAudit,
    ReportBundle,
    ReportClaim,
    ReportPlanClaim,
    ReportPlanDraft,
    ReportSeverity,
    ReportStatus,
)
from eda_platform.schemas.stats import StatTestResult
from eda_platform.schemas.synthesis import SynthesisBrief, SynthesisStoryBeat
from eda_platform.tools.evidence import (
    EvidenceAnalysisTable,
    EvidenceArtifactSummary,
    EvidenceDataset,
    EvidencePack,
)
from eda_platform.tools.loader import load_csv
from eda_platform.tools.method_findings import stat_findings
from eda_platform.tools.profiler import profile_dataset
from eda_platform.tools.report_validator import (
    apply_semantic_gate,
    validate_report_bundle,
)

T = TypeVar("T", bound=BaseModel)


class FakeReportPlanLLM:
    def __init__(self, plan: ReportPlanDraft) -> None:
        self.plan = plan

    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        return cast(T, self.plan)

    def text(self, *, task: str, payload: dict) -> str:
        return "fake"

    def last_usage(self) -> None:
        return None


def _roles(dataset: str = "sales.csv") -> ColumnRoleSet:
    return ColumnRoleSet(
        dataset=dataset,
        roles=[
            ColumnRole(
                column="order_item_id",
                role=ColumnRoleName.SEQUENCE,
                confidence=0.95,
                provenance="inferred",
                verified_by=["sequence_strict_1n_within_group:order_id"],
            ),
            ColumnRole(
                column="revenue",
                role=ColumnRoleName.MEASURE,
                confidence=0.85,
                provenance="inferred",
                verified_by=["measure_numeric_dtype"],
            ),
        ],
    )


def _pack() -> EvidencePack:
    return EvidencePack(
        payload_policy="schema+aggregates",
        artifact_index={
            "table_1": EvidenceArtifactSummary(
                artifact_id="table_1",
                artifact_type="Table",
                title="Numeric summary",
                dataset_id="ds_sales",
            ),
        },
        datasets=[
            EvidenceDataset(
                artifact_id="prof_1",
                dataset_id="ds_sales",
                name="sales.csv",
                row_count=10,
                column_count=3,
                columns=["region", "revenue", "order_item_id"],
                dtypes={
                    "region": "object",
                    "revenue": "float64",
                    "order_item_id": "int64",
                },
            )
        ],
        # F1: the numeric pool only holds resolved payloads, so table_1 must
        # actually resolve for a wrong number to fail (inline values no longer count).
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


def _bundle_with_claim(claim: ReportClaim) -> ReportBundle:
    bundle = ReportBundle.empty(project_id="project_demo", session_id="run_demo")
    bundle.sections[0].claims.append(claim)
    for section in bundle.sections:
        section.body = section.structural_body()
    return bundle


def _valid_claim(**overrides: Any) -> ReportClaim:
    payload: dict[str, Any] = {
        "id": "claim_1",
        "text": "Revenue is 120.",
        "evidence": [
            EvidenceRef(kind="stat", artifact_id="table_1", locator="rows[0]", value=120)
        ],
        "referenced_datasets": ["sales.csv"],
        "referenced_columns": ["revenue"],
    }
    payload.update(overrides)
    return ReportClaim(**payload)


# --------------------------------------------------------------------------- #
# Hard gate stays binary — a correctness failure can never publish degraded.
# --------------------------------------------------------------------------- #
def test_hard_gate_failure_stays_rejected_and_is_never_degraded() -> None:
    claim = _valid_claim(text="Revenue is 999.")
    bundle = _bundle_with_claim(claim)
    audit = validate_report_bundle(bundle, _pack())
    assert audit.has_critical_findings  # numeric_mismatch

    outcome = apply_semantic_gate(bundle, audit, role_sets=[_roles()])

    assert outcome.verdict == "rejected"
    assert audit.gate_verdict == "rejected"
    assert audit.status is ReportStatus.NEEDS_REVISION
    # The claim is NOT relabeled into a degraded publish.
    assert claim.gate_verdict == "pass"
    assert claim.confidence_label == "verified"
    assert outcome.degraded_claim_count == 0
    assert audit.degraded_claim_count == 0


# --------------------------------------------------------------------------- #
# Semantic soft gate: degrade with structured labels, never prune.
# --------------------------------------------------------------------------- #
def test_identifier_column_claim_degrades_with_structured_label() -> None:
    claim = _valid_claim(
        text="Order item sequence values average 120.",
        referenced_columns=["order_item_id"],
    )
    bundle = _bundle_with_claim(claim)
    pack = _pack()
    audit = validate_report_bundle(bundle, pack)
    assert not audit.has_critical_findings

    outcome = apply_semantic_gate(bundle, audit, evidence_pack=pack, role_sets=[_roles()])

    # F6: the low_impact flag stays on the per-claim soft axis; the bundle
    # verdict is the strong ratio (1/1 Table-backed claim -> pass).
    assert outcome.verdict == "pass"
    assert audit.gate_verdict == "pass"
    assert claim.confidence_label == "strong"
    assert claim.gate_verdict == "degraded"
    assert "low_impact_column:order_item_id" in claim.gate_flags
    # The claim still publishes: it is present and the status is untouched.
    assert bundle.sections[0].claims == [claim]
    assert audit.status is ReportStatus.VALIDATED
    assert audit.degraded_claim_count == 1
    # Disclosure lands as a WARN (never critical) validator finding.
    warn = [f for f in audit.findings if f.severity is ReportSeverity.WARN]
    assert len(warn) == 1
    assert warn[0].code == "semantic_degraded"
    assert not audit.has_critical_findings


def test_low_confidence_claim_is_tiered_by_evidence_not_self_confidence() -> None:
    # F6: the model's self-reported confidence no longer drives the label —
    # a "low" claim backed by a full-table Table artifact is strong, and the
    # exploratory_unvalidated flag chain is gone.
    claim = _valid_claim(confidence="low")
    bundle = _bundle_with_claim(claim)
    pack = _pack()
    audit = validate_report_bundle(bundle, pack)

    apply_semantic_gate(bundle, audit, evidence_pack=pack, role_sets=[_roles()])

    assert claim.confidence_label == "strong"
    assert claim.gate_verdict == "pass"
    assert claim.gate_flags == []


def test_missing_role_set_defaults_to_full_weight_and_passes() -> None:
    claim = _valid_claim(referenced_columns=["order_item_id"])
    bundle = _bundle_with_claim(claim)
    pack = _pack()
    audit = validate_report_bundle(bundle, pack)

    outcome = apply_semantic_gate(bundle, audit, evidence_pack=pack, role_sets=None)

    assert outcome.verdict == "pass"
    assert claim.confidence_label == "strong"
    assert claim.gate_verdict == "pass"
    assert audit.gate_verdict == "pass"


# --------------------------------------------------------------------------- #
# Insight scoring = impact x significance (three observable components).
# --------------------------------------------------------------------------- #
def _stat_result(value_column: str) -> StatTestResult:
    return StatTestResult(
        dataset_id="sales.csv",
        test_type="one_way_anova",
        group_column="region",
        value_column=value_column,
        statistic=8.1,
        p_value=0.002,
        effect_size=0.2,
        sample_size=120,
    )


def test_stat_finding_scores_identifier_column_with_zero_impact() -> None:
    roles = _roles()

    sunk = stat_findings(_stat_result("order_item_id"), "artifact", role_set=roles)[0]
    kept = stat_findings(_stat_result("revenue"), "artifact", role_set=roles)[0]

    assert sunk.score is not None and kept.score is not None
    assert sunk.score.impact == 0.0
    assert sunk.score.final == 0.0
    assert kept.score.impact == 1.0
    assert kept.score.significance == 0.998
    assert kept.score.final == 0.998
    # All three components are observable on the finding structure.
    assert set(sunk.score.model_dump()) == {"impact", "significance", "final"}


def test_stat_finding_without_role_set_keeps_full_impact() -> None:
    finding = stat_findings(_stat_result("order_item_id"), "artifact")[0]
    assert finding.score is not None
    assert finding.score.impact == 1.0


# --------------------------------------------------------------------------- #
# Decision report: ranking + commonality/exception skeleton (no LLM).
# --------------------------------------------------------------------------- #
def _validated_finding(
    *,
    finding_id: str,
    question: str,
    text: str,
    value: float,
    score: FindingScore | None,
    claim_class: str = "observed",
) -> ValidatedFinding:
    return ValidatedFinding(
        finding_id=finding_id,
        investigation_id=f"inv_{finding_id}",
        question_id=f"q_{finding_id}",
        question=question,
        claim_class=cast(Any, claim_class),
        findings=[
            QuestionFinding(
                text=text,
                evidence=[
                    EvidenceRef(
                        kind="stat",
                        artifact_id=f"stat_{finding_id}",
                        locator="value",
                        value=value,
                    )
                ],
                score=score,
            )
        ],
        evidence_support="high",
        analytical_reliability="high",
        decision_readiness="medium",
        report_eligible=True,
        report_readiness="eligible_with_limitations",
        report_readiness_reason="Validated with disclosed data conditions.",
    )


def _seed_decision_store(
    tmp_path: Path, findings: list[ValidatedFinding]
) -> tuple[ArtifactStore, str, list[str]]:
    store = ArtifactStore(tmp_path / "workspace")
    project_id = "project_di8"
    store.ensure_project(project_id, "DI8")
    store.start_session(project_id, "finding_run")
    finding_ids: list[str] = []
    for index, finding in enumerate(findings, start=1):
        artifact = Artifact(
            id=f"vf_{index}",
            type=ArtifactType.VALIDATED_FINDING,
            project_id=project_id,
            session_id="finding_run",
            payload=finding.model_dump(mode="json"),
        )
        store.save_artifact(artifact)
        finding_ids.append(artifact.id)
    brief = SynthesisBrief(
        brief_id="brief_di8",
        project_id=project_id,
        selected_finding_artifact_ids=finding_ids,
        decision_context="Which evidence should guide the decision?",
        headline=findings[0].findings[0].text,
        storyline=[
            SynthesisStoryBeat(
                title="Evidence",
                body="Validated findings are ready for review.",
                finding_artifact_ids=finding_ids,
            )
        ],
        report_eligible=True,
        report_readiness="eligible",
    )
    store.start_session(project_id, "synthesis_run")
    brief_artifact = Artifact(
        id="sbrief_di8",
        type=ArtifactType.SYNTHESIS_BRIEF,
        project_id=project_id,
        session_id="synthesis_run",
        parents=finding_ids,
        payload=brief.model_dump(mode="json"),
    )
    store.save_artifact(brief_artifact)
    return store, brief_artifact.id, finding_ids


def test_decision_report_ranking_sinks_identifier_finding(tmp_path: Path) -> None:
    sequence_finding = _validated_finding(
        finding_id="seq",
        question="Does order_item_id differ by region?",
        text="The observed sequence-column mean is 3.2.",
        value=3.2,
        score=FindingScore(impact=0.0, significance=0.99, final=0.0),
    )
    measure_finding = _validated_finding(
        finding_id="rev",
        question="How does revenue vary by channel?",
        text="The observed average revenue is 125.5.",
        value=125.5,
        score=FindingScore(impact=1.0, significance=0.95, final=0.95),
    )
    # Selection order puts the impact-0 finding FIRST; ranking must invert it.
    store, brief_id, finding_ids = _seed_decision_store(
        tmp_path, [sequence_finding, measure_finding]
    )

    report_id = create_decision_report(
        store, project_id="project_di8", brief_artifact_id=brief_id
    )
    report = DecisionReport.model_validate(store.get_artifact(report_id).payload)

    assert [section.title for section in report.sections] == [
        "How does revenue vary by channel?",
        "Does order_item_id differ by region?",
    ]
    assert report.source_finding_artifact_ids == [finding_ids[1], finding_ids[0]]
    # The impact-0 finding is degraded in rank, not deleted.
    assert any("3.2" in section.body for section in report.sections)


def test_decision_report_meta_insight_commonality_and_exception(
    tmp_path: Path,
) -> None:
    common_a = _validated_finding(
        finding_id="a",
        question="How does revenue vary by channel?",
        text="The observed average revenue is 125.5.",
        value=125.5,
        score=FindingScore(impact=1.0, significance=0.9, final=0.9),
    )
    common_b = _validated_finding(
        finding_id="b",
        question="What share of orders are returned?",
        text="The observed return share is 8.",
        value=8,
        score=FindingScore(impact=1.0, significance=0.8, final=0.8),
    )
    exception = _validated_finding(
        finding_id="x",
        question="Are there anomalous payments?",
        text="Anomaly screening found 20 flagged observations.",
        value=20,
        score=FindingScore(impact=1.0, significance=0.7, final=0.7),
    )
    store, brief_id, finding_ids = _seed_decision_store(
        tmp_path, [common_a, common_b, exception]
    )

    # No LLM at all: the deterministic skeleton must still be produced.
    report_id = create_decision_report(
        store, project_id="project_di8", brief_artifact_id=brief_id
    )
    report = DecisionReport.model_validate(store.get_artifact(report_id).payload)

    assert report.narrative_status == "deterministic"
    assert report.meta_insight is not None
    assert report.meta_insight.commonality_statements == [
        "The observed average revenue is 125.5.",
        "The observed return share is 8.",
    ]
    assert report.meta_insight.commonality_finding_artifact_ids == [
        finding_ids[0],
        finding_ids[1],
    ]
    assert report.meta_insight.exception_statements == [
        "Anomaly screening found 20 flagged observations."
    ]
    assert report.meta_insight.exception_finding_artifact_ids == [finding_ids[2]]
    # The executive answer is organized as shared pattern + exceptions.
    assert "Shared pattern across the validated findings:" in report.scqa.answer
    assert "Exceptions that run against the shared pattern:" in report.scqa.answer
    assert "Anomaly screening found 20 flagged observations." in report.scqa.answer


# --------------------------------------------------------------------------- #
# End-to-end report integration: role artifacts flow into the semantic gate.
# --------------------------------------------------------------------------- #
def test_generate_agentic_report_applies_role_labels_and_gate_rollup(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "sales.csv"
    csv_path.write_text(
        "order_id,order_item_id,revenue,region\n"
        "1,1,10,East\n"
        "2,1,20,West\n"
        "3,1,30,East\n",
        encoding="utf-8",
    )
    loaded = load_csv(csv_path, dataset_id="ds_sales")
    profile = profile_dataset(loaded, project_id="project_demo", session_id="run_demo")
    role_artifact = column_role_set_artifact(
        _roles(),
        project_id="project_demo",
        session_id="run_demo",
    )
    plan = ReportPlanDraft(
        claims=[
            ReportPlanClaim(
                section_title="Key EDA Insights",
                id="seq_insight",
                text="The dataset has 3 rows.",
                evidence=[
                    EvidenceRef(kind="stat", artifact_id=profile.id, locator="rows", value=3)
                ],
                referenced_datasets=["sales.csv"],
                referenced_columns=["order_item_id"],
            )
        ],
    )

    result = generate_agentic_report(
        [profile, role_artifact],
        project_id="project_demo",
        session_id="run_demo",
        business_context="Revenue analysis",
        llm=FakeReportPlanLLM(plan),
    )

    assert result.bundle.status is ReportStatus.VALIDATED
    # F6: every claim is profile-backed (strong), so the ratio verdict passes;
    # the identifier-column finding still degrades the claim on its own axis.
    assert result.audit.gate_verdict == "pass"
    assert result.audit.degraded_claim_count >= 1
    labeled = [
        claim
        for section in result.bundle.sections
        for claim in section.claims
        if claim.id == "seq_insight"
    ]
    assert len(labeled) == 1
    assert labeled[0].confidence_label == "strong"
    assert labeled[0].gate_verdict == "degraded"
    assert "low_impact_column:order_item_id" in labeled[0].gate_flags
    assert any("Semantic soft gate" in note for note in result.audit.semantic_notes)
    assert any("Evidence strength" in note for note in result.audit.semantic_notes)


def test_audit_defaults_keep_older_payloads_valid() -> None:
    audit = ReportAudit(status=ReportStatus.VALIDATED)
    assert audit.gate_verdict == "pass"
    assert audit.degraded_claim_count == 0
    assert audit.time_boundary_truncations == 0
