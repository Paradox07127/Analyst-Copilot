from __future__ import annotations

from pathlib import Path

import pytest

from eda_platform.api.main import create_app
from eda_platform.application.services.compare_service import CompareProjectMismatchError
from eda_platform.application.services.session_service import InvalidCursorError
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import AnalysisTable, Artifact, ArtifactType, EvidenceRef
from eda_platform.schemas.investigations import InvestigationRecord, ValidatedFinding
from eda_platform.schemas.questions import (
    OpportunityFeasibility,
    QuestionCandidate,
    QuestionCandidateSet,
    QuestionFinding,
    QuestionScore,
)
from eda_platform.schemas.sessions import TraceEvent

PROJECT = "compare-scopes"
LEFT = "scope-left"
RIGHT = "scope-right"


def _candidate(question_id: str, text: str, priority: float) -> QuestionCandidate:
    return QuestionCandidate(
        question_id=question_id,
        question_en=text,
        origin="template",
        target_datasets=["orders"],
        score=QuestionScore(
            data_availability=1.0,
            statistical_signal=priority,
            quality_risk=0.0,
            join_risk=0.0,
            deterministic_score=priority,
        ),
        feasibility=OpportunityFeasibility(status="ready"),
        business_decision="Choose the operating response.",
    )


def _save(
    store: ArtifactStore,
    session_id: str,
    artifact_id: str,
    artifact_type: ArtifactType,
    payload: dict,
) -> None:
    store.save_artifact(
        Artifact(
            id=artifact_id,
            type=artifact_type,
            project_id=PROJECT,
            session_id=session_id,
            payload=payload,
        )
    )


def _finding(session_id: str, *, text: str) -> Artifact:
    finding = ValidatedFinding(
        finding_id=f"finding-{session_id}",
        investigation_id="investigation-shared",
        question_id="question-shared",
        question="What changed in revenue?",
        claim_class="observed",
        findings=[
            QuestionFinding(
                text=text,
                evidence=[
                    EvidenceRef(
                        kind="table",
                        artifact_id=f"table-{session_id}",
                        locator="rows[0].revenue",
                    )
                ],
            )
        ],
        evidence_support="high",
        analytical_reliability="high",
        decision_readiness="high",
        report_eligible=True,
        report_readiness="eligible",
        report_readiness_reason="Validated evidence.",
    )
    return Artifact(
        id=f"validated-{session_id}",
        type=ArtifactType.VALIDATED_FINDING,
        project_id=PROJECT,
        session_id=session_id,
        payload=finding.model_dump(mode="json"),
    )


@pytest.fixture
def scope_service(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    store.ensure_project(PROJECT, name="Compare scopes")
    for session_id in (LEFT, RIGHT):
        store.start_session(PROJECT, session_id)

    left_candidates = QuestionCandidateSet(
        candidates=[
            _candidate("shared", "What changed in revenue?", 0.9),
            _candidate("left-only", "Which issue exists only in baseline?", 0.5),
        ]
    )
    right_candidates = QuestionCandidateSet(
        candidates=[
            _candidate("shared", "What changed in revenue?", 0.7),
            _candidate("right-only", "Which issue exists only in variant?", 0.6),
        ]
    )
    _save(
        store,
        LEFT,
        "questions-left",
        ArtifactType.QUESTION_CANDIDATE_SET,
        left_candidates.model_dump(mode="json"),
    )
    _save(
        store,
        RIGHT,
        "questions-right",
        ArtifactType.QUESTION_CANDIDATE_SET,
        right_candidates.model_dump(mode="json"),
    )

    for session_id, value in ((LEFT, 10), (RIGHT, 12)):
        table = AnalysisTable(
            dataset_id="orders",
            title="Revenue summary",
            kind="numeric_summary",
            description="Revenue by period.",
            rows=[{"metric": "revenue", "value": value}],
        )
        _save(
            store,
            session_id,
            f"table-{session_id}",
            ArtifactType.TABLE,
            table.model_dump(mode="json"),
        )
        finding = _finding(session_id, text=f"Revenue was {value} units.")
        store.save_artifact(finding)
        _save(
            store,
            session_id,
            f"record-{session_id}",
            ArtifactType.INVESTIGATION_RECORD,
            InvestigationRecord(
                record_id=f"record-{session_id}",
                investigation_id="investigation-shared",
                question_id="question-shared",
                status="validated",
                reason_code="validated",
                reason="Evidence passed validation.",
                next_action="Include in report.",
                finding_artifact_id=finding.id,
            ).model_dump(mode="json"),
        )
        _save(
            store,
            session_id,
            f"report-{session_id}",
            ArtifactType.MARKDOWN_REPORT,
            {
                "markdown": (
                    "# Executive summary\n"
                    f"Revenue was {value} units.\n\n"
                    + ("## Variant note\nNew section." if session_id == RIGHT else "")
                )
            },
        )
        store.append_trace(
            PROJECT,
            TraceEvent(
                session_id=session_id,
                event_type="tool_completed",
                name="sql_runner",
                summary={
                    "status": "completed",
                    "tool_name": "sql_runner",
                    "row_count": value,
                },
            ),
        )
        store.mark_session_status(PROJECT, session_id, "completed")

    app = create_app(tmp_path)
    return app.state.compare_scope_service


@pytest.mark.parametrize(
    "scope",
    ["questions", "analysis", "findings", "report", "artifacts", "execution"],
)
def test_every_scope_returns_typed_real_differences(scope_service, scope: str) -> None:
    view = scope_service.compare_scope(scope, LEFT, RIGHT, limit=100)

    assert view.scope == scope
    assert view.left.session_id == LEFT
    assert view.right.session_id == RIGHT
    assert view.left_state.state == "value"
    assert view.right_state.state == "value"
    assert view.items
    assert view.counts.changed + view.counts.added + view.counts.removed > 0
    assert all(item.matcher_version for item in view.items)
    assert all(item.left is not None or item.right is not None for item in view.items)


def test_questions_swap_reverses_added_and_removed(scope_service) -> None:
    forward = scope_service.compare_scope("questions", LEFT, RIGHT, limit=100)
    backward = scope_service.compare_scope("questions", RIGHT, LEFT, limit=100)

    assert forward.counts.added == backward.counts.removed == 1
    assert forward.counts.removed == backward.counts.added == 1


def test_scope_filter_and_cursor_are_pair_bound(scope_service) -> None:
    first = scope_service.compare_scope(
        "artifacts",
        LEFT,
        RIGHT,
        filter_mode="differences",
        limit=1,
    )
    assert len(first.items) == 1
    assert first.items[0].change != "same"
    assert first.next_cursor

    second = scope_service.compare_scope(
        "artifacts",
        LEFT,
        RIGHT,
        filter_mode="differences",
        limit=1,
        cursor=first.next_cursor,
    )
    assert second.items
    assert second.items[0].match_key != first.items[0].match_key

    with pytest.raises(InvalidCursorError):
        scope_service.compare_scope(
            "artifacts",
            RIGHT,
            LEFT,
            filter_mode="differences",
            limit=1,
            cursor=first.next_cursor,
        )


def test_scope_comparison_refuses_cross_project_pair(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    store.ensure_project("one", name="One")
    store.ensure_project("two", name="Two")
    store.start_session("one", "left")
    store.start_session("two", "right")
    app = create_app(tmp_path)

    with pytest.raises(CompareProjectMismatchError):
        app.state.compare_scope_service.compare_scope("questions", "left", "right")
