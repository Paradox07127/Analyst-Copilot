from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import pytest
from pydantic import BaseModel

from eda_platform.agents.investigation_loop import run_bounded_loop
from eda_platform.core.budget import SessionBudgetExceeded
from eda_platform.core.loop_journal import (
    JsonlLoopJournal,
    LoopTransitionError,
    make_loop_call_id,
    make_loop_step_id,
)
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.investigation_orchestrator import _initialize_loop_journal
from eda_platform.schemas.investigations import InvestigationPlan
from eda_platform.tools.loader import load_csv
from eda_platform.tools.sql_runner import SqlCatalog, build_catalog

T = TypeVar("T", bound=BaseModel)


class ScriptedLLM:
    def __init__(self, decisions: list[dict[str, str]]) -> None:
        self.decisions = list(decisions)
        self.structured_calls = 0

    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        self.structured_calls += 1
        if not self.decisions:
            raise AssertionError("unexpected provider call")
        return schema(**self.decisions.pop(0))  # type: ignore[return-value]

    def text(self, *, task: str, payload: dict) -> str:
        return ""

    def last_usage(self) -> None:
        return None


class FailOnceBeforeProbeCompletion(JsonlLoopJournal):
    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.failed = False

    def append_new(self, event_type, **fields):  # noqa: ANN001, ANN201
        if event_type == "probe_completed" and not self.failed:
            self.failed = True
            raise RuntimeError("fault after artifact commit")
        return super().append_new(event_type, **fields)


class PreflightBudgetLLM(ScriptedLLM):
    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        self.structured_calls += 1
        raise SessionBudgetExceeded(
            "requests",
            limit=0,
            attempted=1,
            stage="reservation",
        )


def _plan() -> InvestigationPlan:
    return InvestigationPlan(
        investigation_id="inv_resume",
        source_session_id="source_run",
        question_id="question_resume",
        card_version=1,
        candidate_fingerprint="candidate-v1",
        question="What is the maximum amount?",
        target_datasets=["tiny.csv"],
        method_family="descriptive",
        method_recipe="Read-only aggregate.",
        allowed_tools=["read_only_sql", "llm_probe_planner"],
        feasibility="ready",
        status="planned",
        status_reason="Ready.",
    )


def _catalog(tmp_path: Path) -> SqlCatalog:
    csv_path = tmp_path / "tiny.csv"
    csv_path.write_text("amount\n10\n20\n30\n", encoding="utf-8")
    return build_catalog([load_csv(csv_path, dataset_id="ds_tiny")])


def _journal(tmp_path: Path, *, journal_type=JsonlLoopJournal):  # noqa: ANN001, ANN202
    journal = journal_type(tmp_path / "investigations" / "inv_resume" / "loop.jsonl")
    journal.initialize(
        investigation_id="inv_resume",
        source_session_id="source_run",
        question_id="question_resume",
        plan_fingerprint="plan-v1",
        policy_fingerprint="policy-v1",
        code_fingerprint="code-v1",
        max_steps=3,
        llm_call_cap=8,
    )
    return journal


def _run(
    llm: ScriptedLLM,
    catalog: SqlCatalog,
    journal: JsonlLoopJournal,
):
    return run_bounded_loop(
        llm,
        plan=_plan(),
        primary_findings=[],
        query_engine=catalog.engine,
        catalog=catalog,
        max_steps=3,
        llm_call_cap=8,
        journal=journal,
    )


def test_terminal_result_replays_without_another_provider_call(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    journal = _journal(tmp_path)
    llm = ScriptedLLM(
        [
            {
                "action": "probe",
                "purpose": "maximum amount",
                "sql": "SELECT max(amount) AS max_amount FROM tiny",
            },
            {
                "action": "conclude",
                "rationale": "The bounded follow-up is complete.",
            },
        ]
    )

    first = _run(llm, catalog, journal)
    replay_llm = ScriptedLLM([])
    replay = _run(replay_llm, catalog, JsonlLoopJournal(journal.path))

    assert replay == first
    assert replay_llm.structured_calls == 0
    assert [event.event_type for event in journal.events()] == [
        "loop_started",
        "attempt_started",
        "decision_call_started",
        "decision_call_completed",
        "probe_started",
        "artifact_committed",
        "probe_completed",
        "decision_call_started",
        "decision_call_completed",
        "loop_concluded",
    ]


def test_preflight_budget_exit_is_typed_and_replays_without_provider_call(
    tmp_path: Path,
) -> None:
    catalog = _catalog(tmp_path)
    journal = _journal(tmp_path)
    first_llm = PreflightBudgetLLM([])

    first = _run(first_llm, catalog, journal)
    replay_llm = ScriptedLLM([])
    replay = _run(replay_llm, catalog, JsonlLoopJournal(journal.path))

    assert first.exit_reason == "budget_exhausted"
    assert replay == first
    assert first_llm.structured_calls == 1
    assert replay_llm.structured_calls == 0
    state = journal.rebuild()
    assert state is not None
    assert state.status == "budget_exhausted"
    assert state.final_draft_ref == "terminal-result.json"


def test_pending_provider_call_becomes_uncertain_without_repayment(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    journal = _journal(tmp_path)
    journal.append_new(
        "decision_call_started",
        iteration=0,
        call_id=make_loop_call_id("inv_resume", 0),
    )
    llm = ScriptedLLM([])

    with pytest.raises(LoopTransitionError, match="uncertain"):
        _run(llm, catalog, journal)

    state = journal.rebuild()
    assert state is not None
    assert state.status == "uncertain"
    assert state.llm_calls_settled == 0
    assert llm.structured_calls == 0


def test_completed_decision_is_reused_instead_of_repaid(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    journal = _journal(tmp_path)
    call_id = make_loop_call_id("inv_resume", 0)
    journal.append_new("decision_call_started", iteration=0, call_id=call_id)
    journal.append_new(
        "decision_call_completed",
        iteration=0,
        call_id=call_id,
        step_id=make_loop_step_id("decision", call_id),
        response_hash="response-v1",
        typed_decision={
            "action": "probe",
            "purpose": "maximum amount",
            "sql": "SELECT max(amount) AS max_amount FROM tiny",
            "rationale": "",
        },
    )
    llm = ScriptedLLM(
        [
            {
                "action": "conclude",
                "rationale": "The bounded follow-up is complete.",
            }
        ]
    )

    result = _run(llm, catalog, journal)

    assert result.exit_reason == "concluded"
    assert result.llm_calls_used == 2
    assert llm.structured_calls == 1


def test_committed_probe_artifact_recovers_after_completion_fault(
    tmp_path: Path,
) -> None:
    catalog = _catalog(tmp_path)
    executed_sql: list[str] = []
    execute_select = catalog.engine.execute_select

    def spy_execute(sql: str):  # noqa: ANN202
        executed_sql.append(sql)
        return execute_select(sql)

    catalog.engine.execute_select = spy_execute  # type: ignore[method-assign]
    faulting = _journal(tmp_path, journal_type=FailOnceBeforeProbeCompletion)
    probe_llm = ScriptedLLM(
        [
            {
                "action": "probe",
                "purpose": "maximum amount",
                "sql": "SELECT max(amount) AS max_amount FROM tiny",
            }
        ]
    )

    with pytest.raises(RuntimeError, match="fault after artifact commit"):
        _run(probe_llm, catalog, faulting)
    executions_before_resume = len(executed_sql)
    state = faulting.rebuild()
    assert state is not None
    assert state.pending_probe_id is not None
    assert state.step_artifact_refs

    conclude_llm = ScriptedLLM(
        [
            {
                "action": "conclude",
                "rationale": "The bounded follow-up is complete.",
            }
        ]
    )
    result = _run(conclude_llm, catalog, JsonlLoopJournal(faulting.path))

    assert result.exit_reason == "concluded"
    assert len(executed_sql) == executions_before_resume
    assert conclude_llm.structured_calls == 1
    assert any(step.result_artifact_id for step in result.steps)


def test_orchestrator_uses_stable_run_workspace_journal_path(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "workspace")
    store.ensure_project("project", "Project")
    store.start_session("project", "plan_run")

    first = _initialize_loop_journal(
        store,
        project_id="project",
        session_id="plan_run",
        plan=_plan(),
    )
    second = _initialize_loop_journal(
        store,
        project_id="project",
        session_id="plan_run",
        plan=_plan(),
    )

    expected = (
        store.session_dir("project", "plan_run")
        / "investigations"
        / "inv_resume"
        / "loop.journal.jsonl"
    )
    assert first.path == second.path == expected
    assert len(first.events()) == 1
