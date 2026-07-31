from pathlib import Path

from eda_platform.core.decision_coverage import (
    assess_decision_coverage,
    persist_decision_coverage,
)
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.investigations import InvestigationRecord, ValidatedFinding
from eda_platform.schemas.questions import (
    OpportunityFeasibility,
    QuestionCandidate,
    QuestionCandidateSet,
    QuestionScore,
)

_PROJECT = "project-coverage"


def _store(tmp_path: Path, *session_ids: str) -> ArtifactStore:
    store = ArtifactStore(tmp_path / "workspace")
    store.ensure_project(_PROJECT, "Coverage project")
    for session_id in session_ids:
        store.start_session(_PROJECT, session_id)
    return store


def _candidate(question_id: str, question: str, score: float) -> QuestionCandidate:
    return QuestionCandidate(
        question_id=question_id,
        question_en=question,
        origin="template",
        target_datasets=["orders"],
        score=QuestionScore(
            data_availability=1.0,
            statistical_signal=score,
            quality_risk=0.0,
            join_risk=0.0,
            deterministic_score=score,
        ),
        feasibility=OpportunityFeasibility(status="ready"),
    )


def _save_candidates(
    store: ArtifactStore,
    session_id: str,
    candidates: list[QuestionCandidate],
    *,
    artifact_id: str | None = None,
) -> None:
    payload = QuestionCandidateSet(candidates=candidates).model_dump(mode="json")
    store.save_artifact(
        Artifact(
            id=artifact_id or f"candidates-{session_id}",
            type=ArtifactType.QUESTION_CANDIDATE_SET,
            project_id=_PROJECT,
            session_id=session_id,
            payload=payload,
        )
    )


def _save_finding(
    store: ArtifactStore,
    session_id: str,
    question_id: str,
    *,
    report_eligible: bool = True,
) -> None:
    finding = ValidatedFinding(
        finding_id=f"finding-{question_id}-{session_id}",
        investigation_id=f"investigation-{question_id}",
        question_id=question_id,
        question=f"Question {question_id}",
        claim_class="observed",
        evidence_support="high",
        analytical_reliability="high",
        decision_readiness="high",
        report_eligible=report_eligible,
        report_readiness="eligible" if report_eligible else "not_eligible",
        report_readiness_reason="Deterministic test verdict.",
    )
    store.save_artifact(
        Artifact(
            id=f"finding-{question_id}-{session_id}",
            type=ArtifactType.VALIDATED_FINDING,
            project_id=_PROJECT,
            session_id=session_id,
            payload=finding.model_dump(mode="json"),
        )
    )


def _save_record(store: ArtifactStore, session_id: str, question_id: str) -> None:
    record = InvestigationRecord(
        record_id=f"record-{question_id}-{session_id}",
        investigation_id=f"investigation-{question_id}",
        question_id=question_id,
        status="inconclusive",
        reason_code="insufficient_signal",
        reason="The investigation reached a terminal outcome.",
        next_action="Review the available evidence.",
    )
    store.save_artifact(
        Artifact(
            id=f"record-{question_id}-{session_id}",
            type=ArtifactType.INVESTIGATION_RECORD,
            project_id=_PROJECT,
            session_id=session_id,
            payload=record.model_dump(mode="json"),
        )
    )


def test_coverage_ready_when_all_top_cards_are_terminal_and_finding_exists(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path, "run-1")
    _save_candidates(
        store,
        "run-1",
        [
            _candidate("q1", "Which segment is growing?", 0.9),
            _candidate("q2", "Why did margin fall?", 0.8),
        ],
    )
    _save_finding(store, "run-1", "q1")
    _save_record(store, "run-1", "q2")

    coverage = assess_decision_coverage(store, _PROJECT)

    assert coverage.top_cards_total == 2
    assert coverage.top_cards_terminal == 2
    assert coverage.validated_findings == 1
    assert coverage.coverage_ready is True
    assert coverage.uninvestigated_high_value == []
    assert coverage.gaps == []


def test_internal_derived_runs_are_skipped(tmp_path: Path) -> None:
    """Macro-loop helper runs (``__internal`` marker) must not shape coverage."""
    store = _store(tmp_path, "run-1", "run-1_macro_r1__internal")
    _save_candidates(
        store,
        "run-1",
        [_candidate("q1", "Which segment is growing?", 0.8)],
    )
    _save_finding(store, "run-1", "q1")
    # A higher-scored, uninvestigated card that only exists on the derived run.
    _save_candidates(
        store,
        "run-1_macro_r1__internal",
        [_candidate("q_internal", "Why does the follow-up funnel exist?", 0.95)],
    )

    coverage = assess_decision_coverage(store, _PROJECT)

    assert coverage.top_cards_total == 1
    assert coverage.uninvestigated_high_value == []
    assert coverage.coverage_ready is True


def test_uninvestigated_high_value_card_blocks_readiness(tmp_path: Path) -> None:
    store = _store(tmp_path, "run-1")
    _save_candidates(
        store,
        "run-1",
        [
            _candidate("q1", "Which segment is growing?", 0.9),
            _candidate("q2", "Why did margin fall?", 0.8),
        ],
    )
    _save_finding(store, "run-1", "q2")

    coverage = assess_decision_coverage(store, _PROJECT)

    assert coverage.coverage_ready is False
    assert coverage.uninvestigated_high_value == ["Which segment is growing?"]
    assert coverage.gaps == [
        "1 of the 2 highest-value cards has not been investigated yet."
    ]


def test_findings_not_eligible_are_counted(tmp_path: Path) -> None:
    store = _store(tmp_path, "run-1")
    _save_candidates(store, "run-1", [_candidate("q1", "Question one?", 0.9)])
    _save_finding(store, "run-1", "q1", report_eligible=False)
    _save_finding(store, "run-1", "q-other", report_eligible=True)

    coverage = assess_decision_coverage(store, _PROJECT)

    assert coverage.validated_findings == 2
    assert coverage.findings_not_eligible == 1
    assert coverage.coverage_ready is True


def test_candidates_are_deduplicated_across_runs_by_best_score(tmp_path: Path) -> None:
    store = _store(tmp_path, "run-1", "run-2")
    _save_candidates(
        store,
        "run-1",
        [_candidate("shared", "Older wording", 0.4), _candidate("q2", "Second card", 0.8)],
    )
    _save_candidates(
        store,
        "run-2",
        [_candidate("shared", "Highest-score wording", 0.95)],
    )

    coverage = assess_decision_coverage(store, _PROJECT)

    assert coverage.top_cards_total == 2
    assert coverage.uninvestigated_high_value == [
        "Highest-score wording",
        "Second card",
    ]


def test_assessment_is_deterministic(tmp_path: Path) -> None:
    store = _store(tmp_path, "run-1")
    _save_candidates(store, "run-1", [_candidate("q1", "Question one?", 0.9)])

    first = assess_decision_coverage(store, _PROJECT)
    second = assess_decision_coverage(store, _PROJECT)

    assert first == second
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_persist_decision_coverage_is_idempotent_per_run(tmp_path: Path) -> None:
    store = _store(tmp_path, "run-1")
    _save_candidates(store, "run-1", [_candidate("q1", "Question one?", 0.9)])

    first_id = persist_decision_coverage(store, _PROJECT, "run-1")
    second_id = persist_decision_coverage(store, _PROJECT, "run-1")

    assert first_id == second_id
    artifact = store.get_artifact(first_id)
    assert artifact.type is ArtifactType.DECISION_COVERAGE
    assert artifact.payload == assess_decision_coverage(store, _PROJECT).model_dump(
        mode="json"
    )
    artifacts = store.list_artifacts(project_id=_PROJECT, session_id="run-1")
    assert sum(a.type is ArtifactType.DECISION_COVERAGE for a in artifacts) == 1
