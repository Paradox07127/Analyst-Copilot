from __future__ import annotations

from datetime import UTC, datetime

import pytest
from semantic_test_helpers import confirm_promotion, load_seeds

from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.knowledge_promotion import build_promotion_candidate
from eda_platform.schemas.artifacts import Artifact, ArtifactType, DatasetProfile, EvidenceRef
from eda_platform.schemas.investigations import InvestigationPlan, ValidatedFinding
from eda_platform.schemas.questions import QuestionFinding
from eda_platform.schemas.sessions import SessionManifest

PROJECT = "project_promotion"
QUESTION = "What changed?"


def _start(store: ArtifactStore, session_id: str, day: int) -> None:
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
        name="sales.csv",
        rows=10,
        columns=1,
        column_names=["sales"],
        dtypes={"sales": "float64"},
        missing_values={"sales": 0},
        missing_percent={"sales": 0.0},
        numeric_columns=["sales"],
        categorical_columns=[],
    )


def _finding(first_claim: str = "Sales increased 12%.") -> ValidatedFinding:
    evidence = EvidenceRef(kind="table", artifact_id="evidence_sales", locator="rows[0]")
    return ValidatedFinding(
        finding_id="finding_sales",
        investigation_id="inv_sales",
        question_id="q_sales",
        question=QUESTION,
        value_hypothesis="HYPOTHESIS: this will double future profit.",
        decision_action="DECISION: hire a larger sales team.",
        interpretation="INTERPRETATION: growth will continue indefinitely.",
        interpretation_status="validated",
        claim_class="observed",
        findings=[
            QuestionFinding(text=first_claim, evidence=[evidence]),
            QuestionFinding(text="The West region led growth.", evidence=[evidence]),
        ],
        evidence_support="high",
        analytical_reliability="high",
        decision_readiness="medium",
        limitations=["Only two quarters were compared.", "Seasonality was not modeled."],
        report_eligible=True,
        report_readiness="eligible_with_limitations",
        report_readiness_reason="Claims are supported with a time-window limitation.",
        source_artifact_ids=["evidence_sales", "profile_sales_old"],
    )


def _fixture(tmp_path) -> ArtifactStore:
    store = ArtifactStore(tmp_path)
    store.ensure_project(PROJECT, "Promotion")
    _start(store, "source", 1)
    store.save_artifact(
        Artifact(
            id="profile_sales_old",
            type=ArtifactType.DATASET_PROFILE,
            project_id=PROJECT,
            session_id="source",
            payload=_profile("ds_sales_old").model_dump(),
        )
    )
    _start(store, "plan", 2)
    plan = InvestigationPlan(
        investigation_id="inv_sales",
        source_session_id="source",
        question_id="q_sales",
        card_version=1,
        candidate_fingerprint="fingerprint",
        question=QUESTION,
        target_datasets=["sales.csv"],
        method_family="descriptive",
        method_recipe="compare quarters",
        allowed_tools=["sql"],
        feasibility="ready",
        status="planned",
        status_reason="Ready.",
    )
    store.save_artifact(
        Artifact(
            id="plan_sales",
            type=ArtifactType.INVESTIGATION_PLAN,
            project_id=PROJECT,
            session_id="plan",
            payload=plan.model_dump(),
        )
    )
    _start(store, "finding", 3)
    store.save_artifact(
        Artifact(
            id="evidence_sales",
            type=ArtifactType.TABLE,
            project_id=PROJECT,
            session_id="finding",
            payload={"rows": [{"growth": 0.12}]},
        )
    )
    store.save_artifact(
        Artifact(
            id="finding_sales_artifact",
            type=ArtifactType.VALIDATED_FINDING,
            project_id=PROJECT,
            session_id="finding",
            payload=_finding().model_dump(),
        )
    )
    return store


def test_candidate_answer_uses_only_claim_sentences(tmp_path) -> None:
    candidate = build_promotion_candidate(_fixture(tmp_path), PROJECT, "finding_sales_artifact")

    assert candidate.answer == "Sales increased 12%. The West region led growth."
    assert "HYPOTHESIS" not in candidate.answer
    assert "DECISION" not in candidate.answer
    assert "INTERPRETATION" not in candidate.answer
    assert "evidence_sales" in (candidate.evidence_note or "")
    assert "Only two quarters" in (candidate.evidence_note or "")
    assert "Seasonality" not in (candidate.evidence_note or "")


def test_stale_finding_is_refused(tmp_path) -> None:
    store = _fixture(tmp_path)
    _start(store, "reupload", 4)
    store.save_artifact(
        Artifact(
            id="profile_sales_new",
            type=ArtifactType.DATASET_PROFILE,
            project_id=PROJECT,
            session_id="reupload",
            payload=_profile("ds_sales_new").model_dump(),
        )
    )

    with pytest.raises(ValueError, match="stale.*cannot be promoted"):
        confirm_promotion(store, PROJECT, "finding_sales_artifact")


def test_confirmation_roundtrips_through_semantic_seeds(tmp_path) -> None:
    store = _fixture(tmp_path)

    path = confirm_promotion(store, PROJECT, "finding_sales_artifact")
    seeds = load_seeds(store.project_dir(PROJECT))

    assert path.exists()
    assert len(seeds.verified_answers) == 1
    assert seeds.verified_answers[0].question == QUESTION
    assert seeds.verified_answers[0].answer.startswith("Sales increased 12%")


def test_repromotion_replaces_same_question_instead_of_duplicating(tmp_path) -> None:
    store = _fixture(tmp_path)
    confirm_promotion(store, PROJECT, "finding_sales_artifact")
    updated = _finding(first_claim="Sales increased 15%.")
    store.save_artifact(
        Artifact(
            id="finding_sales_artifact",
            type=ArtifactType.VALIDATED_FINDING,
            project_id=PROJECT,
            session_id="finding",
            payload=updated.model_dump(),
        )
    )

    confirm_promotion(store, PROJECT, "finding_sales_artifact")
    answers = load_seeds(store.project_dir(PROJECT)).verified_answers

    assert len(answers) == 1
    assert answers[0].answer.startswith("Sales increased 15%")
