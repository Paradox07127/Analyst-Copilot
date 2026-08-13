"""The m2 claim-plan budget must scale with run width.

2026-08-12 deepseek run (11 questions, 9 tables): all three plan attempts
truncated at the 12,000-token completion cap with max_claims=12, burning 53%
of the run's tokens for zero LLM claims and dropping the report to the
deterministic fallback. The first attempt now shrinks its claim budget one
per question past the wide-run threshold instead of discovering the ceiling
by failing.
"""

from __future__ import annotations

from eda_platform.agents.reporting import (
    _COMPLETION_BUDGET_CEILING,
    _plan_claim_budget,
    _raise_completion_budget,
    _set_completion_budget,
)


def test_narrow_runs_keep_the_original_budgets() -> None:
    assert _plan_claim_budget(1, truncation_retry=False) == 12
    assert _plan_claim_budget(6, truncation_retry=False) == 12
    assert _plan_claim_budget(6, truncation_retry=True) == 8


def test_wide_run_first_attempt_shrinks_per_question() -> None:
    # The 2026-08-12 failing shape: 11 questions -> 7 first, 4 on retry.
    assert _plan_claim_budget(11, truncation_retry=False) == 7
    assert _plan_claim_budget(11, truncation_retry=True) == 4


def test_budgets_have_floors() -> None:
    assert _plan_claim_budget(30, truncation_retry=False) == 6
    assert _plan_claim_budget(30, truncation_retry=True) == 4


class _Settings:
    def __init__(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens


class _Client:
    def __init__(self, max_tokens: int) -> None:
        self.settings = _Settings(max_tokens)


def test_raise_completion_budget_stops_at_the_ceiling() -> None:
    client = _Client(_COMPLETION_BUDGET_CEILING)
    assert _raise_completion_budget(client, _COMPLETION_BUDGET_CEILING) is False

    client = _Client(8_000)
    assert _raise_completion_budget(client, 8_000) is True
    assert client.settings.max_tokens == _COMPLETION_BUDGET_CEILING


def test_set_completion_budget_writes_through() -> None:
    client = _Client(4_000)
    assert _set_completion_budget(client, _COMPLETION_BUDGET_CEILING) is True
    assert client.settings.max_tokens == _COMPLETION_BUDGET_CEILING


def test_wide_run_cap_raise_is_restored_after_plan_generation(tmp_path) -> None:
    """Codex acceptance finding (2026-08-12): the pre-raised 12k cap leaked
    into the narration and session-title calls on the same client, inflating
    their worst-case ledger reservations. The cap must be restored once plan
    generation is done."""
    from pathlib import Path
    from typing import Any, TypeVar, cast

    from pydantic import BaseModel

    from eda_platform.agents.reporting import generate_agentic_report
    from eda_platform.schemas.artifacts import Artifact, ArtifactType
    from eda_platform.schemas.questions import QuestionExecutionResult, QuestionFinding
    from eda_platform.schemas.reports import ReportPlanDraft
    from eda_platform.tools.loader import load_csv
    from eda_platform.tools.profiler import profile_dataset

    T = TypeVar("T", bound=BaseModel)

    csv_path = Path(tmp_path) / "sales.csv"
    csv_path.write_text("region,revenue\nEast,10\nWest,20\n", encoding="utf-8")
    profile = profile_dataset(
        load_csv(csv_path, dataset_id="ds_sales"),
        project_id="p",
        session_id="r",
    )
    questions = [
        Artifact(
            id=f"qexec_{index}",
            type=ArtifactType.QUESTION_EXECUTION_RESULT,
            project_id="p",
            session_id="r",
            parents=[profile.id],
            payload=QuestionExecutionResult(
                question_id=f"q_{index}",
                question=f"Question {index}?",
                origin="template",
                status="succeeded",
                outcome="answered",
                findings=[
                    QuestionFinding(
                        text=f"Finding {index}.",
                        evidence=[],
                    )
                ],
            ).model_dump(mode="json"),
        )
        for index in range(7)  # > _WIDE_RUN_QUESTIONS
    ]

    class CapAwareLLM:
        def __init__(self) -> None:
            self.settings = _Settings(4_000)
            self.caps_seen: list[int] = []

        def structured(self, *, task: str, schema: type[T], payload: dict[str, Any]) -> T:
            self.caps_seen.append(self.settings.max_tokens)
            return cast(T, ReportPlanDraft(claims=[]))

        def text(self, *, task: str, payload: dict[str, Any]) -> str:
            return "fake"

        def last_usage(self) -> None:
            return None

    llm = CapAwareLLM()
    generate_agentic_report(
        [profile, *questions],
        project_id="p",
        session_id="r",
        business_context="Revenue analysis",
        llm=llm,
    )

    # Wide run: the plan call itself ran at the ceiling...
    assert llm.caps_seen and llm.caps_seen[0] == _COMPLETION_BUDGET_CEILING
    # ...and the client left the function at its configured cap.
    assert llm.settings.max_tokens == 4_000
