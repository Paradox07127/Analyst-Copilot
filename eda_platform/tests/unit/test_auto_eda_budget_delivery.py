from pathlib import Path
from typing import Any

import pytest

from eda_platform.core.budget import BudgetExceeded, BudgetUsageUncertain, SessionBudgetExceeded
from eda_platform.core.cancellation import (
    CancellationCause,
    CancellationError,
    CancellationSnapshot,
    KillFenceState,
)
from eda_platform.core.graph_execution import GraphEffectUncertain
from eda_platform.core.llm import is_offline_client
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers import auto_eda
from eda_platform.schemas.artifacts import ArtifactType

DATA = Path(__file__).parents[1] / "golden" / "data" / "ecommerce_orders.csv"


@pytest.mark.parametrize("typed", [False, True])
def test_budget_exhaustion_delivers_completed_evidence_and_unfinished_questions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, typed: bool,
) -> None:
    original_execute = auto_eda.execute_question_candidate
    original_select = auto_eda.select_auto_execution_set
    original_report = auto_eda.generate_agentic_report
    calls: list[str] = []
    persisted: list[str] = []

    def select(candidates: Any) -> list[Any]:
        first = original_select(candidates)[0]
        return [first.model_copy(update={"question_id": f"q-{index}"}) for index in range(3)]

    def execute(candidate: Any, **kwargs: Any) -> Any:
        calls.append(candidate.question_id)
        if len(calls) == 2:
            store = ArtifactStore(tmp_path)
            persisted.extend(a.payload["question_id"] for a in store.list_artifacts(
                project_id="project", session_id="run",
            ) if a.type is ArtifactType.QUESTION_EXECUTION_RESULT)
            if typed:
                raise SessionBudgetExceeded("requests", limit=37, attempted=38,
                                            call_id="rejected", stage="reservation")
            raise BudgetExceeded("analysis token budget exhausted")
        return original_execute(candidate, **kwargs)

    def report(artifacts: Any, **kwargs: Any) -> Any:
        assert is_offline_client(kwargs["llm"])
        assert kwargs["narrator_llm"] is None
        answers = [a for a in artifacts if a.type is ArtifactType.QUESTION_EXECUTION_RESULT]
        assert len(answers) == 3
        return original_report(artifacts, **kwargs)

    def title(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("budget exhaustion must not attempt cosmetic model work")

    monkeypatch.setattr(auto_eda, "select_auto_execution_set", select)
    monkeypatch.setattr(auto_eda, "execute_question_candidate", execute)
    monkeypatch.setattr(auto_eda, "generate_agentic_report", report)
    monkeypatch.setattr(auto_eda, "_llm_session_title", title)
    result = auto_eda.run_auto_eda(
        [DATA], workspace=tmp_path, project_id="project", session_id="run",
    )
    assert calls == ["q-0", "q-1"]
    assert persisted == ["q-0"]
    results = {a.payload["question_id"]: a.payload for a in result.artifacts
               if a.type is ArtifactType.QUESTION_EXECUTION_RESULT}
    assert results["q-0"]["status"] == "succeeded"
    for key in ("q-1", "q-2"):
        assert results[key]["outcome"] == "abstained"
        assert results[key]["abstention_code"] == "budget_exhausted"
    store = ArtifactStore(tmp_path)
    row = store.get_session_index_row("run")
    assert row is not None and row["status"] == "limited"
    assert (store.session_dir("project", "run") / "report/report.md").read_text()
    assert any(a.type is ArtifactType.AGENT_HANDOFF for a in result.artifacts)
    events = store.list_trace_events(project_id="project", session_id="run")
    assert any(e.event_type == "report_degraded" and e.summary.get("degraded") for e in events)
    # Reopening the same graph must preserve the limitation and avoid paid work.
    auto_eda.run_auto_eda([DATA], workspace=tmp_path, project_id="project", session_id="run")
    assert calls == ["q-0", "q-1"]
    row = store.get_session_index_row("run")
    assert row is not None and row["status"] == "limited"


@pytest.mark.parametrize("error", [
    BudgetUsageUncertain("unknown", stage="settlement", missing=("cost_usd",)),
    GraphEffectUncertain("provider outcome unknown"),
    SessionBudgetExceeded("wall_seconds", limit=1, attempted=2),
    CancellationError(CancellationSnapshot(
        CancellationCause.CANCEL_REQUESTED, "cancelled", None, 0, KillFenceState.ELIGIBLE,
    )),
])
def test_unsafe_or_cancelled_outcome_never_becomes_budget_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception,
) -> None:
    def execute(*args: Any, **kwargs: Any) -> Any:
        raise error

    def report(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("unsafe outcome must not publish a degraded report")

    monkeypatch.setattr(auto_eda, "execute_question_candidate", execute)
    monkeypatch.setattr(auto_eda, "generate_agentic_report", report)
    with pytest.raises(type(error)):
        auto_eda.run_auto_eda([DATA], workspace=tmp_path, project_id="project", session_id="run")


def test_report_budget_exhaustion_finishes_without_provider_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_report = auto_eda.generate_agentic_report
    calls = []

    def report(artifacts: Any, **kwargs: Any) -> Any:
        calls.append(kwargs["persistence"].execution_id)
        if len(calls) == 1:
            raise SessionBudgetExceeded("requests", limit=40, attempted=41,
                                        call_id="report", stage="reservation")
        assert is_offline_client(kwargs["llm"])
        assert kwargs["narrator_llm"] is None
        return original_report(artifacts, **kwargs)

    def title(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("budget-limited delivery must skip cosmetic model work")

    monkeypatch.setattr(auto_eda, "generate_agentic_report", report)
    monkeypatch.setattr(auto_eda, "_llm_session_title", title)
    auto_eda.run_auto_eda([DATA], workspace=tmp_path, project_id="project", session_id="run")
    store = ArtifactStore(tmp_path)
    row = store.get_session_index_row("run")
    assert row is not None and row["status"] == "limited"
    assert (store.session_dir("project", "run") / "report/report.md").read_text()
    assert len(calls) == 2
    assert calls[0] != calls[1]
    auto_eda.run_auto_eda([DATA], workspace=tmp_path, project_id="project", session_id="run")
    row = store.get_session_index_row("run")
    assert row is not None and row["status"] == "limited"
    assert len(calls) == 2


@pytest.mark.parametrize("allowed_calls", [0, 1])
def test_early_model_allowance_exhaustion_delivers_fixed_eda_and_template_questions(
    tmp_path: Path, allowed_calls: int,
) -> None:
    from eda_platform.core.budget import SessionBudgetPolicy
    from eda_platform.core.llm import LLMResultMetadata, LLMUsage

    class Provider:
        calls: list[str]

        def __init__(self) -> None:
            self.calls = []

        def structured(self, *, task: str, schema: Any, payload: dict) -> Any:
            self.calls.append(task)
            assert task == "di8_semantic_bootstrap"
            return schema.model_validate({"entity": "orders", "columns": []})

        def text(self, **kwargs: Any) -> str:
            raise AssertionError("no model calls after early budget exhaustion")

        def last_usage(self) -> LLMResultMetadata:
            return LLMResultMetadata(provider="test", model="test", usage=LLMUsage(total_tokens=1))

    provider = Provider()
    policy = SessionBudgetPolicy(max_requests=allowed_calls)
    for _ in range(2):
        result = auto_eda.run_auto_eda(
            [DATA], workspace=tmp_path, project_id="project", session_id="run",
            llm=provider, budget_policy=policy,
        )
        assert result.report_markdown
        assert any(a.type is ArtifactType.DATASET_PROFILE for a in result.artifacts)
        assert any(a.type is ArtifactType.QUESTION_EXECUTION_RESULT for a in result.artifacts)
        store = ArtifactStore(tmp_path)
        row = store.get_session_index_row("run")
        assert row is not None and row["status"] == "limited"
        events = store.list_trace_events(project_id="project", session_id="run")
        assert any(e.event_type == "budget_rejected" for e in events)
    assert len(provider.calls) == allowed_calls
