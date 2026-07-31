from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

import eda_platform.drivers.investigation_orchestrator as investigation_driver
from eda_platform.core.claim_language import contains_causal_phrase
from eda_platform.core.kernel import SessionCancelled
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.auto_eda import AutoEDAResult, run_auto_eda
from eda_platform.drivers.investigation_orchestrator import (
    approve_plan,
    create_investigation_plans,
    execute_investigation_plans,
    reject_plan,
)
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.investigations import (
    InvestigationPlan,
    InvestigationRecord,
    ValidatedFinding,
)
from eda_platform.schemas.quality_context import QualityContext, QualityContextSet
from eda_platform.schemas.questions import QuestionCandidate, QuestionCandidateSet, QuestionFinding

GOLDEN_DATA = Path(__file__).parents[1] / "golden" / "data"


def _source(tmp_path: Path) -> AutoEDAResult:
    return run_auto_eda(
        [GOLDEN_DATA / "ecommerce_orders.csv"],
        workspace=tmp_path / "workspace",
        project_id="project_demo",
        session_id="source_run",
    )


def _candidate_set(source: AutoEDAResult) -> QuestionCandidateSet:
    artifact = next(
        item
        for item in source.artifacts
        if item.type is ArtifactType.QUESTION_CANDIDATE_SET
    )
    return QuestionCandidateSet.model_validate(artifact.payload)


def _candidate_set_artifact(store: ArtifactStore, source: AutoEDAResult) -> Artifact:
    return next(
        store.get_artifact(item.id)
        for item in source.artifacts
        if item.type is ArtifactType.QUESTION_CANDIDATE_SET
    )


def _first_template_candidate(candidate_set: QuestionCandidateSet) -> QuestionCandidate:
    return next(
        item
        for item in candidate_set.candidates
        if item.origin == "template" and item.sql_template is not None
    )


def _plan_artifact(planned) -> Artifact:  # noqa: ANN001
    return next(
        item
        for item in planned.artifacts
        if item.type is ArtifactType.INVESTIGATION_PLAN
    )


def test_approved_plan_emits_validated_finding(tmp_path: Path) -> None:
    source = _source(tmp_path)
    candidate = _first_template_candidate(_candidate_set(source))
    planned = create_investigation_plans(
        project_id=source.project_id,
        source_session_id=source.session_id,
        question_ids=[candidate.question_id],
        workspace=source.workspace,
        session_id="plan_run",
    )
    plan_artifact = _plan_artifact(planned)
    plan = InvestigationPlan.model_validate(plan_artifact.payload)
    assert plan.status == "planned"
    assert plan.execution_ready is True
    assert plan.user_approval_required is True
    assert plan.candidate_fingerprint
    assert "read_only_sql" in plan.allowed_tools

    approve_plan(
        project_id=source.project_id,
        plan_session_id=planned.session_id,
        plan_id=plan_artifact.id,
        workspace=source.workspace,
    )
    completed = execute_investigation_plans(
        project_id=source.project_id,
        plan_session_id=planned.session_id,
        plan_ids=[plan_artifact.id],
        workspace=source.workspace,
    )
    finding_artifact = next(
        item for item in completed.artifacts if item.type is ArtifactType.VALIDATED_FINDING
    )
    finding = ValidatedFinding.model_validate(finding_artifact.payload)
    assert finding.report_eligible is True
    assert finding.evidence_support in {"high", "medium"}
    record = next(
        InvestigationRecord.model_validate(item.payload)
        for item in completed.artifacts
        if item.type is ArtifactType.INVESTIGATION_RECORD
    )
    assert record.status == "validated"
    assert record.finding_artifact_id == finding_artifact.id


def test_execution_without_approval_is_refused(tmp_path: Path) -> None:
    source = _source(tmp_path)
    candidate = _first_template_candidate(_candidate_set(source))
    planned = create_investigation_plans(
        project_id=source.project_id,
        source_session_id=source.session_id,
        question_ids=[candidate.question_id],
        workspace=source.workspace,
        session_id="plan_run",
    )
    plan_artifact = _plan_artifact(planned)
    with pytest.raises(ValueError, match="matching approval artifact"):
        execute_investigation_plans(
            project_id=source.project_id,
            plan_session_id=planned.session_id,
            plan_ids=[plan_artifact.id],
            workspace=source.workspace,
        )


def test_investigation_cancellation_stops_before_next_plan_and_terminal_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    candidates = _candidate_set(source).candidates[:2]
    assert len(candidates) == 2
    planned = create_investigation_plans(
        project_id=source.project_id,
        source_session_id=source.session_id,
        question_ids=[candidate.question_id for candidate in candidates],
        workspace=source.workspace,
        session_id="plan_cancelled",
    )
    plan_artifacts = [
        artifact
        for artifact in planned.artifacts
        if artifact.type is ArtifactType.INVESTIGATION_PLAN
    ]
    assert len(plan_artifacts) == 2
    for artifact in plan_artifacts:
        approve_plan(
            project_id=source.project_id,
            plan_session_id=planned.session_id,
            plan_id=artifact.id,
            workspace=source.workspace,
        )
    entered: list[str] = []

    def fake_execute(
        plan: InvestigationPlan,
        **kwargs: object,
    ) -> tuple[list[Artifact], list[Artifact]]:
        entered.append(plan.investigation_id)
        return [], []

    monkeypatch.setattr(investigation_driver, "_execute_by_method", fake_execute)

    with pytest.raises(SessionCancelled):
        execute_investigation_plans(
            project_id=source.project_id,
            plan_session_id=planned.session_id,
            plan_ids=[artifact.id for artifact in plan_artifacts],
            workspace=source.workspace,
            cancel_check=lambda: bool(entered),
        )

    assert len(entered) == 1
    store = ArtifactStore(source.workspace)
    artifacts = store.list_artifacts(
        project_id=source.project_id,
        session_id=planned.session_id,
    )
    assert not any(artifact.type is ArtifactType.VALIDATED_FINDING for artifact in artifacts)
    records = [
        InvestigationRecord.model_validate(artifact.payload)
        for artifact in artifacts
        if artifact.type is ArtifactType.INVESTIGATION_RECORD
    ]
    entered_records = [
        record for record in records if record.investigation_id == entered[0]
    ]
    assert len(entered_records) == 1
    assert entered_records[0].reason_code == "executing"
    assert store.get_session_status(planned.session_id) == "awaiting_approval"


def test_approval_fingerprint_mismatch_is_refused(tmp_path: Path) -> None:
    source = _source(tmp_path)
    candidate = _first_template_candidate(_candidate_set(source))
    planned = create_investigation_plans(
        project_id=source.project_id,
        source_session_id=source.session_id,
        question_ids=[candidate.question_id],
        workspace=source.workspace,
        session_id="plan_run",
    )
    plan_artifact = _plan_artifact(planned)
    approve_plan(
        project_id=source.project_id,
        plan_session_id=planned.session_id,
        plan_id=plan_artifact.id,
        workspace=source.workspace,
    )
    # A material edit to the approved plan invalidates the prior approval.
    store = ArtifactStore(source.workspace)
    plan = InvestigationPlan.model_validate(plan_artifact.payload)
    plan.status_reason = plan.status_reason + " (revised after approval)"
    plan_artifact.payload = plan.model_dump(mode="json")
    store.save_artifact(plan_artifact)
    with pytest.raises(ValueError, match="does not match the current plan"):
        execute_investigation_plans(
            project_id=source.project_id,
            plan_session_id=planned.session_id,
            plan_ids=[plan_artifact.id],
            workspace=source.workspace,
        )


def test_stale_candidate_fingerprint_is_rejected(tmp_path: Path) -> None:
    source = _source(tmp_path)
    candidate = _first_template_candidate(_candidate_set(source))
    planned = create_investigation_plans(
        project_id=source.project_id,
        source_session_id=source.session_id,
        question_ids=[candidate.question_id],
        workspace=source.workspace,
        session_id="plan_run",
    )
    plan_artifact = _plan_artifact(planned)
    approve_plan(
        project_id=source.project_id,
        plan_session_id=planned.session_id,
        plan_id=plan_artifact.id,
        workspace=source.workspace,
    )
    # The live Question Card is edited after approval; execution must fail closed.
    store = ArtifactStore(source.workspace)
    candidate_set_artifact = _candidate_set_artifact(store, source)
    candidate_set = QuestionCandidateSet.model_validate(candidate_set_artifact.payload)
    for item in candidate_set.candidates:
        if item.question_id == candidate.question_id:
            item.card_version = item.card_version + 1
    candidate_set_artifact.payload = candidate_set.model_dump(mode="json")
    store.save_artifact(candidate_set_artifact)

    completed = execute_investigation_plans(
        project_id=source.project_id,
        plan_session_id=planned.session_id,
        plan_ids=[plan_artifact.id],
        workspace=source.workspace,
    )
    assert not any(
        item.type is ArtifactType.VALIDATED_FINDING for item in completed.artifacts
    )
    record = next(
        InvestigationRecord.model_validate(item.payload)
        for item in completed.artifacts
        if item.type is ArtifactType.INVESTIGATION_RECORD
    )
    assert record.status == "rejected"
    assert record.reason_code == "stale_candidate"


def test_double_execution_is_refused(tmp_path: Path) -> None:
    source = _source(tmp_path)
    candidate = _first_template_candidate(_candidate_set(source))
    planned = create_investigation_plans(
        project_id=source.project_id,
        source_session_id=source.session_id,
        question_ids=[candidate.question_id],
        workspace=source.workspace,
        session_id="plan_run",
    )
    plan_artifact = _plan_artifact(planned)
    approve_plan(
        project_id=source.project_id,
        plan_session_id=planned.session_id,
        plan_id=plan_artifact.id,
        workspace=source.workspace,
    )
    execute_investigation_plans(
        project_id=source.project_id,
        plan_session_id=planned.session_id,
        plan_ids=[plan_artifact.id],
        workspace=source.workspace,
    )
    # A second run must not re-execute: the plan is skipped with its reason.
    again = execute_investigation_plans(
        project_id=source.project_id,
        plan_session_id=planned.session_id,
        plan_ids=[plan_artifact.id],
        workspace=source.workspace,
    )
    assert again.artifacts == []
    assert len(again.skipped) == 1
    assert "already has an outcome" in again.skipped[0].reason


def test_blocked_plan_is_skipped_while_ready_plan_executes(tmp_path: Path) -> None:
    source = _source(tmp_path)
    candidates = [
        item
        for item in _candidate_set(source).candidates
        if item.origin == "template" and item.sql_template is not None
    ]
    planned = create_investigation_plans(
        project_id=source.project_id,
        source_session_id=source.session_id,
        question_ids=[candidate.question_id for candidate in candidates],
        workspace=source.workspace,
        session_id="plan_run",
    )
    plan_artifacts = [
        item
        for item in planned.artifacts
        if item.type is ArtifactType.INVESTIGATION_PLAN
        and InvestigationPlan.model_validate(item.payload).status == "planned"
    ][:2]
    assert len(plan_artifacts) == 2
    # Force one plan into the not-execution-ready state (e.g. an unimplemented
    # method family) before approval, so the approval fingerprint still matches.
    store = ArtifactStore(source.workspace)
    blocked_artifact = plan_artifacts[1]
    blocked_plan = InvestigationPlan.model_validate(blocked_artifact.payload)
    blocked_plan.execution_ready = False
    blocked_artifact.payload = blocked_plan.model_dump(mode="json")
    store.save_artifact(blocked_artifact)
    for plan_artifact in plan_artifacts:
        approve_plan(
            project_id=source.project_id,
            plan_session_id=planned.session_id,
            plan_id=plan_artifact.id,
            workspace=source.workspace,
        )

    completed = execute_investigation_plans(
        project_id=source.project_id,
        plan_session_id=planned.session_id,
        plan_ids=[artifact.id for artifact in plan_artifacts],
        workspace=source.workspace,
    )
    assert [skip.plan_id for skip in completed.skipped] == [blocked_artifact.id]
    assert "not execution-ready" in completed.skipped[0].reason
    ready_plan = InvestigationPlan.model_validate(plan_artifacts[0].payload)
    records = [
        InvestigationRecord.model_validate(item.payload)
        for item in completed.artifacts
        if item.type is ArtifactType.INVESTIGATION_RECORD
    ]
    assert any(
        record.investigation_id == ready_plan.investigation_id and record.status == "validated"
        for record in records
    )
    # The blocked plan produced no outcome artifacts at all.
    assert all(record.investigation_id != blocked_plan.investigation_id for record in records)


def test_out_of_scope_dataset_fails_the_scope_gate(tmp_path: Path) -> None:
    source = _source(tmp_path)
    candidate = _first_template_candidate(_candidate_set(source))
    planned = create_investigation_plans(
        project_id=source.project_id,
        source_session_id=source.session_id,
        question_ids=[candidate.question_id],
        workspace=source.workspace,
        session_id="plan_run",
    )
    plan_artifact = _plan_artifact(planned)
    # The approved plan no longer covers the Card's dataset; execution must not
    # widen scope back to the source datasets.
    store = ArtifactStore(source.workspace)
    plan = InvestigationPlan.model_validate(plan_artifact.payload)
    plan.target_datasets = ["dataset_outside_the_approved_scope.csv"]
    plan_artifact.payload = plan.model_dump(mode="json")
    store.save_artifact(plan_artifact)
    approve_plan(
        project_id=source.project_id,
        plan_session_id=planned.session_id,
        plan_id=plan_artifact.id,
        workspace=source.workspace,
    )
    completed = execute_investigation_plans(
        project_id=source.project_id,
        plan_session_id=planned.session_id,
        plan_ids=[plan_artifact.id],
        workspace=source.workspace,
    )
    assert not any(
        item.type is ArtifactType.VALIDATED_FINDING for item in completed.artifacts
    )
    record = next(
        InvestigationRecord.model_validate(item.payload)
        for item in completed.artifacts
        if item.type is ArtifactType.INVESTIGATION_RECORD
    )
    assert record.status == "rejected"
    assert record.reason_code == "scope_violation"
    assert any(
        gate.name == "scope" and gate.status == "failed"
        for gate in record.validation_gates
    )


def test_reject_plan_records_a_rejection_and_blocks_execution(tmp_path: Path) -> None:
    source = _source(tmp_path)
    candidate = _first_template_candidate(_candidate_set(source))
    planned = create_investigation_plans(
        project_id=source.project_id,
        source_session_id=source.session_id,
        question_ids=[candidate.question_id],
        workspace=source.workspace,
        session_id="plan_run",
    )
    plan_artifact = _plan_artifact(planned)
    artifacts = reject_plan(
        project_id=source.project_id,
        plan_session_id=planned.session_id,
        plan_id=plan_artifact.id,
        workspace=source.workspace,
        reason="Not worth the analyst time this cycle.",
    )
    assert any(item.type is ArtifactType.INVESTIGATION_APPROVAL for item in artifacts)
    record = next(
        InvestigationRecord.model_validate(item.payload)
        for item in artifacts
        if item.type is ArtifactType.INVESTIGATION_RECORD
    )
    assert record.status == "rejected"
    completed = execute_investigation_plans(
        project_id=source.project_id,
        plan_session_id=planned.session_id,
        plan_ids=[plan_artifact.id],
        workspace=source.workspace,
    )
    assert completed.artifacts == []
    assert len(completed.skipped) == 1
    assert "already has an outcome" in completed.skipped[0].reason


def test_quality_context_marks_a_finding_reportable_with_limitations(tmp_path: Path) -> None:
    source = _source(tmp_path)
    store = ArtifactStore(source.workspace)
    context = _missing_amount_context()
    context_set = QualityContextSet(
        dataset_id=context.dataset_id,
        dataset_name="ecommerce_orders.csv",
        contexts=[context],
    )
    context_payload = context_set.model_dump(mode="json")
    context_artifact = Artifact(
        id="qualityctx_investigation_test",
        type=ArtifactType.QUALITY_CONTEXT_SET,
        project_id=source.project_id,
        session_id=source.session_id,
        payload=context_payload,
    )
    store.save_artifact(context_artifact)

    candidate_set_artifact = _candidate_set_artifact(store, source)
    candidate_set = QuestionCandidateSet.model_validate(candidate_set_artifact.payload)
    candidate = _first_template_candidate(candidate_set)
    for item in candidate_set.candidates:
        if item.question_id == candidate.question_id:
            item.quality_context_artifact_ids = [context_artifact.id]
    candidate_set_artifact.payload = candidate_set.model_dump(mode="json")
    store.save_artifact(candidate_set_artifact)

    planned = create_investigation_plans(
        project_id=source.project_id,
        source_session_id=source.session_id,
        question_ids=[candidate.question_id],
        workspace=source.workspace,
        session_id="plan_run",
    )
    plan_artifact = _plan_artifact(planned)
    approve_plan(
        project_id=source.project_id,
        plan_session_id=planned.session_id,
        plan_id=plan_artifact.id,
        workspace=source.workspace,
    )
    completed = execute_investigation_plans(
        project_id=source.project_id,
        plan_session_id=planned.session_id,
        plan_ids=[plan_artifact.id],
        workspace=source.workspace,
    )
    finding = next(
        ValidatedFinding.model_validate(item.payload)
        for item in completed.artifacts
        if item.type is ArtifactType.VALIDATED_FINDING
    )
    assert finding.report_eligible is True
    assert finding.report_readiness == "eligible_with_limitations"
    assert "data conditions" in finding.report_readiness_reason
    assert context.report_limitation in finding.limitations


def test_causal_phrasing_is_detected_across_paraphrases() -> None:
    for text in (
        "Higher spend causes higher revenue.",
        "The gap was caused by seasonality.",
        "Revenue is driven by the discount rate.",
        "The change led to a lower refund rate.",
        "Losses rose because of the outage.",
        "The delay is due to the missing labels.",
        "This shows the effect of the new policy.",
        "Cutting price results in more orders.",
            "A price reduction caused sales to increase.",
    ):
        assert contains_causal_phrase(text) is True
    assert contains_causal_phrase(
        "Average amount increased from 100 to 200 across the returned periods."
    ) is False


def test_contradictory_report_eligibility_raises() -> None:
    with pytest.raises(ValidationError):
        ValidatedFinding(
            finding_id="finding_x",
            investigation_id="inv_x",
            question_id="q_x",
            question="Any question?",
            claim_class="observed",
            findings=[QuestionFinding(text="A descriptive statement.")],
            evidence_support="low",
            analytical_reliability="low",
            decision_readiness="low",
            report_eligible=True,
            report_readiness="not_eligible",
            report_readiness_reason="Contradictory eligibility must not be silently normalized.",
        )


def _missing_amount_context() -> QualityContext:
    return QualityContext(
        context_id="qctx_test_missing_amount",
        dataset_id="ds_orders",
        dataset_name="ecommerce_orders.csv",
        issue_code="high_missing",
        severity="warn",
        column="amount",
        observation="Column amount has 40.00% missing values.",
        pattern_facts=["4 of 10 rows are missing amount."],
        analysis_impacts=["Analyses using amount may exclude a non-random subset of rows."],
        open_questions=["Does missingness in amount differ by group?"],
        validation_steps=["Compare missing and non-missing rows before reporting."],
        report_limitation=(
            "Interpretation involving amount should account for the observed high_missing "
            "condition; its business cause remains unconfirmed."
        ),
        source_artifact_ids=["profile_test", "quality_test"],
    )
