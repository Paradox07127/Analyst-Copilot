from __future__ import annotations

from datetime import UTC, datetime

from eda_platform.core.finding_freshness import (
    assess_decision_report_freshness,
    assess_finding_freshness,
)
from eda_platform.core.publication_fingerprint import (
    DECISION_REPORT_POLICY_VERSION,
    decision_report_input_fingerprint,
)
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType, DatasetProfile, EvidenceRef
from eda_platform.schemas.investigations import InvestigationPlan, ValidatedFinding
from eda_platform.schemas.questions import QuestionFinding
from eda_platform.schemas.sessions import SessionManifest

PROJECT = "project_di6"
DATASET_NAME = "orders.csv"


def _manifest(store: ArtifactStore, session_id: str, day: int) -> None:
    store.start_session(PROJECT, session_id)
    store.write_manifest(
        SessionManifest(
            session_id=session_id,
            project_id=PROJECT,
            input_hashes={},
            code_version="test",
            created_at=datetime(2026, 7, day, tzinfo=UTC),
        )
    )


def _profile(dataset_id: str) -> DatasetProfile:
    return DatasetProfile(
        dataset_id=dataset_id,
        name=DATASET_NAME,
        rows=2,
        columns=1,
        column_names=["amount"],
        dtypes={"amount": "int64"},
        missing_values={"amount": 0},
        missing_percent={"amount": 0.0},
        numeric_columns=["amount"],
        categorical_columns=[],
    )


def _save_profile(store: ArtifactStore, session_id: str, dataset_id: str) -> None:
    store.save_artifact(
        Artifact(
            id=f"profile_{dataset_id}",
            type=ArtifactType.DATASET_PROFILE,
            project_id=PROJECT,
            session_id=session_id,
            payload=_profile(dataset_id).model_dump(),
        )
    )


def _plan(investigation_id: str = "inv_1") -> InvestigationPlan:
    return InvestigationPlan(
        investigation_id=investigation_id,
        source_session_id="run_source",
        question_id="q_1",
        card_version=1,
        candidate_fingerprint="fingerprint",
        question="What was average order value?",
        target_datasets=[DATASET_NAME],
        method_family="descriptive",
        method_recipe="average amount",
        allowed_tools=["sql"],
        feasibility="ready",
        status="planned",
        status_reason="Ready to execute.",
    )


def _finding(
    *, investigation_id: str = "inv_1", evidence_id: str = "evidence_1"
) -> ValidatedFinding:
    return ValidatedFinding(
        finding_id="finding_1",
        investigation_id=investigation_id,
        question_id="q_1",
        question="What was average order value?",
        value_hypothesis="This could increase profit by millions.",
        decision_action="Raise prices.",
        claim_class="observed",
        findings=[
            QuestionFinding(
                text="Average order value was $42.",
                evidence=[
                    EvidenceRef(
                        kind="table",
                        artifact_id=evidence_id,
                        locator="rows[0].average",
                        value=42,
                    )
                ],
            )
        ],
        evidence_support="high",
        analytical_reliability="high",
        decision_readiness="medium",
        limitations=["Refunded orders were not identified."],
        report_eligible=True,
        report_readiness="eligible_with_limitations",
        report_readiness_reason="The descriptive claim is supported.",
        source_artifact_ids=[evidence_id, "profile_ds_old"],
    )


def _fixture(
    tmp_path, *, include_plan: bool = True, evidence_id: str = "evidence_1"
) -> ArtifactStore:
    store = ArtifactStore(tmp_path)
    store.ensure_project(PROJECT, "DI6")
    _manifest(store, "run_source", 1)
    _save_profile(store, "run_source", "ds_old")
    _manifest(store, "run_plan", 2)
    if include_plan:
        store.save_artifact(
            Artifact(
                id="plan_1",
                type=ArtifactType.INVESTIGATION_PLAN,
                project_id=PROJECT,
                session_id="run_plan",
                payload=_plan().model_dump(),
            )
        )
    _manifest(store, "run_finding", 3)
    if evidence_id == "evidence_1":
        store.save_artifact(
            Artifact(
                id=evidence_id,
                type=ArtifactType.TABLE,
                project_id=PROJECT,
                session_id="run_finding",
                payload={"rows": [{"average": 42}]},
            )
        )
    store.save_artifact(
        Artifact(
            id="finding_artifact_1",
            type=ArtifactType.VALIDATED_FINDING,
            project_id=PROJECT,
            session_id="run_finding",
            payload=_finding(evidence_id=evidence_id).model_dump(),
        )
    )
    return store


def _save_decision_report(
    store: ArtifactStore,
    *,
    finding_id: str = "finding_artifact_1",
    include_fingerprint: bool = True,
    include_parent: bool = True,
) -> None:
    finding_artifact = store.get_artifact(finding_id) if include_fingerprint else None
    store.save_artifact(
        Artifact(
            id="decision_report_1",
            type=ArtifactType.DECISION_REPORT,
            project_id=PROJECT,
            session_id="run_finding",
            parents=[finding_id] if include_parent else [],
            payload={
                "report_id": "report_1",
                "brief_id": "brief_1",
                "project_id": PROJECT,
                "title": "Decision Report",
                "scqa": {
                    "situation": "Orders were reviewed.",
                    "complication": "Evidence is bounded.",
                    "question": "What should change?",
                    "answer": "Review the validated result.",
                },
                "sections": [{"title": "Finding", "body": "AOV was $42."}],
                "report_readiness": "eligible",
                "source_finding_artifact_ids": [finding_id],
                "publication_input_fingerprint": (
                    decision_report_input_fingerprint([finding_artifact])
                    if finding_artifact is not None
                    else None
                ),
                "report_policy_version": (
                    DECISION_REPORT_POLICY_VERSION if include_fingerprint else None
                ),
            },
        )
    )


def test_fresh_when_current_dataset_ids_and_evidence_match(tmp_path) -> None:
    freshness = assess_finding_freshness(_fixture(tmp_path), PROJECT, "finding_artifact_1")

    assert freshness.status == "fresh"
    assert freshness.checked_dataset_names == [DATASET_NAME]


def test_stale_after_new_upload_changes_dataset_id(tmp_path) -> None:
    store = _fixture(tmp_path)
    _manifest(store, "run_reupload", 4)
    _save_profile(store, "run_reupload", "ds_new")

    freshness = assess_finding_freshness(store, PROJECT, "finding_artifact_1")

    assert freshness.status == "stale"
    assert any(DATASET_NAME in reason and "ds_new" in reason for reason in freshness.reasons)


def test_unverifiable_when_claim_evidence_artifact_is_missing(tmp_path) -> None:
    store = _fixture(tmp_path, evidence_id="missing_evidence")

    freshness = assess_finding_freshness(store, PROJECT, "finding_artifact_1")

    assert freshness.status == "unverifiable"
    assert any("missing_evidence" in reason for reason in freshness.reasons)


def test_orphan_finding_is_unverifiable(tmp_path) -> None:
    store = _fixture(tmp_path, include_plan=False)

    freshness = assess_finding_freshness(store, PROJECT, "finding_artifact_1")

    assert freshness.status == "unverifiable"
    assert any("InvestigationPlan" in reason for reason in freshness.reasons)


def test_decision_report_is_fresh_when_lineage_and_fingerprint_match(tmp_path) -> None:
    store = _fixture(tmp_path)
    _save_decision_report(store)

    freshness = assess_decision_report_freshness(store, PROJECT, "decision_report_1")

    assert freshness.status == "fresh"
    assert freshness.finding_statuses == {"finding_artifact_1": "fresh"}


def test_decision_report_is_stale_after_dataset_changes(tmp_path) -> None:
    store = _fixture(tmp_path)
    _save_decision_report(store)
    _manifest(store, "run_reupload", 4)
    _save_profile(store, "run_reupload", "ds_new")

    freshness = assess_decision_report_freshness(store, PROJECT, "decision_report_1")

    assert freshness.status == "stale"
    assert any(DATASET_NAME in reason for reason in freshness.reasons)


def test_legacy_decision_report_is_unverifiable_not_fresh(tmp_path) -> None:
    store = _fixture(tmp_path)
    _save_decision_report(store, include_fingerprint=False)

    freshness = assess_decision_report_freshness(store, PROJECT, "decision_report_1")

    assert freshness.status == "unverifiable"
    assert any("legacy" in reason for reason in freshness.reasons)


def test_decision_report_parent_mismatch_is_unverifiable(tmp_path) -> None:
    store = _fixture(tmp_path)
    _save_decision_report(store, include_parent=False)

    freshness = assess_decision_report_freshness(store, PROJECT, "decision_report_1")

    assert freshness.status == "unverifiable"
    assert any("lineage" in reason for reason in freshness.reasons)


def test_decision_report_detects_same_id_payload_upsert(tmp_path) -> None:
    store = _fixture(tmp_path)
    _save_decision_report(store)
    finding_artifact = store.get_artifact("finding_artifact_1")
    changed_payload = dict(finding_artifact.payload)
    changed_payload["decision_action"] = "A changed hypothesis context."
    store.save_artifact(finding_artifact.model_copy(update={"payload": changed_payload}))

    freshness = assess_decision_report_freshness(store, PROJECT, "decision_report_1")

    assert freshness.status == "stale"
    assert any("fingerprint" in reason for reason in freshness.reasons)
