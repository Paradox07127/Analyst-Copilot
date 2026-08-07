"""A planner rejection an automatic run can act on must cost a rewrite, not a question.

Hallucinated columns and unbindable SQL always got one round of model feedback
(`agents/planner.build_plan`). Two other dead ends sat outside that loop and
killed the question outright. Across the stored LLM runs they cost ten answers:

- the join scope guard (7): three FIFA questions on a CTE self-join
  (2026-08-04) and one credit-card question on `CROSS JOIN LATERAL`
  (2026-08-05) among them. Both were fixed by making `sql_base_tables`
  smarter, which does nothing for the next SQL shape it misreads.
- `needs_approval` (3): the planner is told to raise it for costly work, and
  an automatic run has nobody to approve it.

Not everything belongs in the loop. `required_relations` comes off the question
card and cannot change between attempts, so re-asking buys the same answer for
the price of a call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar, cast

import pytest
from pydantic import BaseModel

from eda_platform.drivers.question_exec import execute_question_candidate
from eda_platform.schemas.artifacts import ArtifactType
from eda_platform.schemas.plans import AnalysisPlan
from eda_platform.schemas.questions import QuestionCandidate, QuestionScore
from eda_platform.tools.loader import LoadedDataset, load_csv

T = TypeVar("T", bound=BaseModel)


class ScriptedPlanLLM:
    """One scripted plan per `m3_build_plan` call, in order; nothing else answered.

    `calls` counts planner calls only, which is what these tests measure. The
    downstream interpretation call degrades to a fallback on RuntimeError, so
    refusing it leaves the execution path intact.
    """

    def __init__(self, plans: list[AnalysisPlan]) -> None:
        self.plans = plans
        self.calls: list[dict[str, Any]] = []

    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        if task != "m3_build_plan":
            raise RuntimeError(f"this double answers only the planner, not {task}")
        self.calls.append({"task": task, "payload": payload})
        if not self.plans:
            raise AssertionError("planner asked for more plans than the script holds")
        return cast(T, self.plans.pop(0))

    def text(self, *, task: str, payload: dict) -> str:
        return "fake"

    def last_usage(self) -> None:
        return None


def _plan(sql: str, *, dataset_names: list[str], columns: list[str]) -> AnalysisPlan:
    return AnalysisPlan(
        question="Which venues host the most matches?",
        dataset_names=dataset_names,
        columns=columns,
        filters=[],
        sql=sql,
        method="group and count",
        rationale="Counts matches per venue.",
        needs_approval=False,
        estimated_scan="small",
    )


_CROSS_TABLE_PLAN = _plan(
    "select v.city, count(*) as match_count "
    "from matches m join venues v on m.venue_id = v.venue_id group by v.city",
    dataset_names=["matches", "venues"],
    columns=["city", "venue_id"],
)

_SINGLE_TABLE_PLAN = _plan(
    "select venue_id, count(*) as match_count from matches group by venue_id",
    dataset_names=["matches"],
    columns=["venue_id"],
)


@pytest.fixture
def datasets(tmp_path: Path) -> list[LoadedDataset]:
    matches = tmp_path / "matches.csv"
    matches.write_text(
        "match_id,venue_id\n" + "".join(f"M{index},V{index % 3}\n" for index in range(12)),
        encoding="utf-8",
    )
    venues = tmp_path / "venues.csv"
    venues.write_text("venue_id,city\nV0,Doha\nV1,Lusail\nV2,Al Khor\n", encoding="utf-8")
    return [
        load_csv(matches, dataset_id="ds_matches"),
        load_csv(venues, dataset_id="ds_venues"),
    ]


def _candidate(**overrides: Any) -> QuestionCandidate:
    values: dict[str, Any] = {
        "question_id": "q_llm_join",
        "question_en": "Which venues host the most matches?",
        "origin": "llm",
        "target_datasets": ["matches.csv", "venues.csv"],
        "sql_template": None,
        "exploratory": True,
        "score": QuestionScore(
            data_availability=1.0,
            statistical_signal=0.5,
            quality_risk=0.0,
            join_risk=0.0,
            deterministic_score=0.6,
        ),
    }
    values.update(overrides)
    return QuestionCandidate.model_validate(values)


def _execute(candidate: QuestionCandidate, llm: ScriptedPlanLLM, datasets: list[LoadedDataset]):
    return execute_question_candidate(
        candidate,
        datasets=datasets,
        project_id="project_demo",
        session_id="run_demo",
        parent_ids=[],
        llm=llm,  # type: ignore[arg-type] -- protocol-compatible test double
    )


def _qexec(artifacts: list[Any]) -> dict[str, Any]:
    return next(
        artifact.payload
        for artifact in artifacts
        if artifact.type is ArtifactType.QUESTION_EXECUTION_RESULT
    )


def test_an_undeclared_join_is_sent_back_for_one_rewrite(
    datasets: list[LoadedDataset],
) -> None:
    llm = ScriptedPlanLLM([_CROSS_TABLE_PLAN, _SINGLE_TABLE_PLAN])

    payload = _qexec(_execute(_candidate(), llm, datasets))

    assert payload["status"] == "succeeded"
    assert payload["sql"] == _SINGLE_TABLE_PLAN.sql
    assert len(llm.calls) == 2
    assert "required_relations" in llm.calls[1]["payload"]["previous_error"]


def test_a_rewrite_that_still_joins_is_still_refused(
    datasets: list[LoadedDataset],
) -> None:
    """The retry is a repair channel, not a way around the whitelist."""
    llm = ScriptedPlanLLM([_CROSS_TABLE_PLAN, _CROSS_TABLE_PLAN])

    payload = _qexec(_execute(_candidate(), llm, datasets))

    assert payload["status"] == "failed"
    assert len(llm.calls) == 2


def test_an_unconfirmed_relation_never_reaches_the_planner(
    datasets: list[LoadedDataset],
) -> None:
    """A card-level fact cannot be repaired, so it must not cost an LLM call."""
    llm = ScriptedPlanLLM([_SINGLE_TABLE_PLAN])

    payload = _qexec(
        _execute(_candidate(required_relations=["matches__venues"]), llm, datasets)
    )

    assert payload["status"] == "failed"
    assert "confirmed join" in payload["error"]
    assert llm.calls == []


_NEEDS_APPROVAL_PLAN = _plan(
    "select match_id, venue_id from matches",
    dataset_names=["matches"],
    columns=["match_id", "venue_id"],
).model_copy(update={"needs_approval": True, "estimated_scan": "large"})


def test_a_plan_needing_approval_is_asked_once_for_a_cheaper_one(
    datasets: list[LoadedDataset],
) -> None:
    llm = ScriptedPlanLLM([_NEEDS_APPROVAL_PLAN, _SINGLE_TABLE_PLAN])

    payload = _qexec(_execute(_candidate(), llm, datasets))

    assert payload["status"] == "succeeded"
    assert payload["sql"] == _SINGLE_TABLE_PLAN.sql
    assert len(llm.calls) == 2
    assert "needs_approval" in llm.calls[1]["payload"]["previous_error"]


def test_a_planner_that_insists_on_approval_still_abstains(
    datasets: list[LoadedDataset],
) -> None:
    """The retry is one chance to fit the budget, not a way to run unapproved work."""
    llm = ScriptedPlanLLM([_NEEDS_APPROVAL_PLAN, _NEEDS_APPROVAL_PLAN])

    payload = _qexec(_execute(_candidate(), llm, datasets))

    assert payload["outcome"] == "awaiting_approval"
    assert payload["abstention_code"] == "approval_required"
    assert len(llm.calls) == 2


class _PayloadCapturingLLM(ScriptedPlanLLM):
    """Same script, but keeps the instructions block of every planner payload."""

    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        result = super().structured(task=task, schema=schema, payload=payload)
        self.instructions = str(payload["instructions"])
        return result


def test_the_automatic_route_does_not_invite_an_approval_it_cannot_grant(
    datasets: list[LoadedDataset],
) -> None:
    """Three of twenty golden questions died on `needs_approval` (2026-08-06 eval).

    Asking again for an approval-free plan did not help, because the sentence
    that produced the flag was still in the payload of the retry. An automatic
    run has nobody to approve, so it must not ask for something it will then
    refuse.
    """
    llm = _PayloadCapturingLLM([_SINGLE_TABLE_PLAN])

    _execute(_candidate(), llm, datasets)

    assert "needs_approval=true" not in llm.instructions
    assert "approval" in llm.instructions.lower(), "say why, do not just go quiet"


def test_the_interactive_route_still_asks_for_approval_on_costly_work() -> None:
    """Chat registers the flag with ApprovalService and surfaces a pending frame;
    the model is its only source, so the instruction has to stay there."""
    from eda_platform.agents.planner import build_plan

    llm = _PayloadCapturingLLM([_SINGLE_TABLE_PLAN])
    build_plan("total matches", llm=llm, catalog_columns={"matches": {"match_id", "venue_id"}})

    assert "needs_approval=true" in llm.instructions
