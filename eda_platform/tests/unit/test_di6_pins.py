"""DI sprint-6 (DI6-B): semantic-definition pins ("语义定义钉子").

Community research flags the top silent-failure class as a metric definition being
re-interpreted run-to-run. The fix is to pin the user-confirmed definitions and
consult them before answering. These tests cover:

- ``pinned_context_block`` assembly: deterministic ordering, optional
  formula/caveats/unit/aliases, empty seeds -> "", and the truncation note;
- injection into the *question proposal* prompt when seeds carry definitions,
  with a hard "these are established facts" instruction, and no injection (byte-
  identical payload) when seeds are empty/absent;
- injection into the *Level-1 interpretation* prompt under the same conditions,
  never widening the admissible-number set (the validator gate is untouched).
"""

from __future__ import annotations

from typing import Any, TypeVar, cast

from pydantic import BaseModel

from eda_platform.agents.interpretation import interpret_findings
from eda_platform.agents.question_agent import propose_llm_question_candidates
from eda_platform.core.semantic import (
    FieldMeaning,
    MetricDefinition,
    SemanticSeeds,
    pinned_context_block,
)
from eda_platform.schemas.artifacts import EvidenceRef
from eda_platform.schemas.questions import QuestionFinding

T = TypeVar("T", bound=BaseModel)


# --------------------------------------------------------------------------- #
# Spy clients
# --------------------------------------------------------------------------- #
class SpyQuestionLLM:
    """Records the payload of every ``structured`` call, returns one proposal."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        self.calls.append({"task": task, "payload": payload})
        return schema.model_validate(
            {
                "questions": [
                    {
                        "question_en": "Which region has the most revenue?",
                        "target_datasets": ["sales.csv"],
                        "llm_business_relevance": 0.8,
                        "llm_actionability": 0.7,
                    }
                ]
            }
        )

    def text(self, *, task: str, payload: dict) -> str:
        return "fake"

    def last_usage(self) -> None:
        return None


class SpyInterpretationLLM:
    """Records payloads and returns a fixed, gate-passing interpretation."""

    def __init__(self, interpretation: str) -> None:
        self._interpretation = interpretation
        self.calls: list[dict[str, Any]] = []

    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        self.calls.append({"task": task, "payload": payload})
        return cast(T, schema(interpretation=self._interpretation))  # type: ignore[call-arg]

    def text(self, *, task: str, payload: dict) -> str:
        return self._interpretation

    def last_usage(self) -> None:
        return None


def _profile_artifact(tmp_path: Any) -> Any:
    from eda_platform.tools.loader import load_csv
    from eda_platform.tools.profiler import profile_dataset

    csv_path = tmp_path / "sales.csv"
    csv_path.write_text("region,revenue\nEast,10\nWest,20\n", encoding="utf-8")
    loaded = load_csv(csv_path, dataset_id="ds_sales")
    return profile_dataset(loaded, project_id="project_demo", session_id="run_demo")


def _seeds_with_definitions() -> SemanticSeeds:
    return SemanticSeeds(
        metric_definitions=[
            MetricDefinition(
                name="Revenue",
                definition="Net booked revenue after refunds.",
                formula="sum(amount) - sum(refund)",
                caveats="Excludes tax.",
            ),
            MetricDefinition(
                name="Active User",
                definition="A user with at least one session in the period.",
            ),
        ],
        field_meanings=[
            FieldMeaning(
                dataset="sales.csv",
                column="revenue",
                meaning="Per-order net revenue in USD.",
                unit="USD",
                aliases=["rev", "net_rev"],
            ),
        ],
    )


def _findings() -> list[QuestionFinding]:
    return [
        QuestionFinding(
            text="East leads with revenue 20, ahead of West at 10.",
            evidence=[
                EvidenceRef(
                    kind="sql",
                    artifact_id="sql_a",
                    locator="rows_preview[0].revenue",
                    value=20,
                ),
                EvidenceRef(
                    kind="sql",
                    artifact_id="sql_a",
                    locator="rows_preview[1].revenue",
                    value=10,
                ),
            ],
        )
    ]


# --------------------------------------------------------------------------- #
# Block assembly
# --------------------------------------------------------------------------- #
def test_empty_seeds_produce_empty_block() -> None:
    assert pinned_context_block(SemanticSeeds()) == ""


def test_block_orders_deterministically_and_includes_optional_fields() -> None:
    block = pinned_context_block(_seeds_with_definitions())

    # Metrics precede field meanings, each under its own header.
    assert block.index("Metrics:") < block.index("Field meanings:")
    # Metrics are sorted case-insensitively by name: "Active User" before "Revenue".
    assert block.index("Active User") < block.index("Revenue")
    # Optional metric attributes are rendered when present, omitted when absent.
    assert "Formula: sum(amount) - sum(refund)." in block
    assert "Caveats: Excludes tax." in block
    assert "Formula:" not in block.split("Revenue")[0]  # Active User has no formula
    # Field meaning carries dataset.column plus unit and aliases.
    assert "- sales.csv.revenue: Per-order net revenue in USD." in block
    assert "unit: USD" in block
    assert "aka rev, net_rev" in block


def test_block_is_byte_stable_across_calls() -> None:
    seeds = _seeds_with_definitions()
    assert pinned_context_block(seeds) == pinned_context_block(seeds)


def test_block_truncates_with_omission_note_and_stays_within_budget() -> None:
    seeds = SemanticSeeds(
        metric_definitions=[
            MetricDefinition(name=f"Metric {index:02d}", definition="X" * 40)
            for index in range(40)
        ]
    )
    block = pinned_context_block(seeds, max_chars=300)

    assert len(block) <= 300
    assert "more definitions omitted)" in block
    # The very first (sorted) metric survives; later ones are dropped.
    assert "Metric 00" in block
    assert "Metric 39" not in block
    # The note names how many were omitted (a positive count).
    note = block.rsplit("… (", maxsplit=1)[1]
    omitted = int(note.split(" ", maxsplit=1)[0])
    assert omitted > 0


def test_single_oversized_seed_is_truncated_into_budget() -> None:
    """Regression: the first entry used to be included unconditionally, so one
    huge seed blew straight past max_chars."""
    seeds = SemanticSeeds(
        field_meanings=[
            FieldMeaning(dataset="orders.csv", column="notes", meaning="X" * 40_000)
        ]
    )
    block = pinned_context_block(seeds, max_chars=800)

    assert len(block) <= 800
    assert block.startswith("Field meanings:")
    assert block.rstrip().endswith("…")

    # Same guarantee when a second entry would add the omission note.
    seeds.field_meanings.append(
        FieldMeaning(dataset="orders.csv", column="amount", meaning="Small.")
    )
    block_with_note = pinned_context_block(seeds, max_chars=800)
    assert len(block_with_note) <= 800
    assert "more definitions omitted)" in block_with_note


# --------------------------------------------------------------------------- #
# Injection into the question-proposal prompt
# --------------------------------------------------------------------------- #
def test_question_prompt_injects_pinned_definitions_with_hard_instruction(
    tmp_path: Any,
) -> None:
    profile = _profile_artifact(tmp_path)
    llm = SpyQuestionLLM()

    result = propose_llm_question_candidates(
        [profile], llm=llm, seeds=_seeds_with_definitions(), max_questions=1
    )

    assert result.error is None
    payload = llm.calls[0]["payload"]
    assert "pinned_definitions" in payload
    pinned = payload["pinned_definitions"]
    assert "established" in pinned and "never redefine" in pinned
    assert "Revenue" in pinned and "Active User" in pinned
    assert "sales.csv.revenue" in pinned


def test_question_prompt_omits_pins_when_seeds_absent(tmp_path: Any) -> None:
    profile = _profile_artifact(tmp_path)
    llm = SpyQuestionLLM()

    propose_llm_question_candidates([profile], llm=llm, max_questions=1)

    assert "pinned_definitions" not in llm.calls[0]["payload"]


def test_question_prompt_omits_pins_when_seeds_have_no_definitions(tmp_path: Any) -> None:
    profile = _profile_artifact(tmp_path)
    llm = SpyQuestionLLM()

    propose_llm_question_candidates(
        [profile], llm=llm, seeds=SemanticSeeds(), max_questions=1
    )

    assert "pinned_definitions" not in llm.calls[0]["payload"]


# --------------------------------------------------------------------------- #
# Injection into the interpretation prompt
# --------------------------------------------------------------------------- #
def test_interpretation_prompt_injects_pinned_definitions() -> None:
    llm = SpyInterpretationLLM("East leads with revenue 20, ahead of West at 10.")

    result = interpret_findings(
        llm,
        question="Which region leads on revenue?",
        findings=_findings(),
        seeds=_seeds_with_definitions(),
    )

    assert result.status == "validated"
    payload = llm.calls[0]["payload"]
    assert "pinned_definitions" in payload
    pinned = payload["pinned_definitions"]
    assert "established" in pinned and "never redefine" in pinned
    assert "Revenue" in pinned
    # The pin is context only: it must not appear in allowed_numbers.
    assert "pinned_definitions" not in payload["allowed_numbers"]


def test_interpretation_prompt_omits_pins_when_seeds_absent() -> None:
    llm = SpyInterpretationLLM("East leads with revenue 20, ahead of West at 10.")

    result = interpret_findings(
        llm, question="Which region leads on revenue?", findings=_findings()
    )

    assert result.status == "validated"
    assert "pinned_definitions" not in llm.calls[0]["payload"]


def test_interpretation_prompt_omits_pins_when_seeds_empty() -> None:
    llm = SpyInterpretationLLM("East leads with revenue 20, ahead of West at 10.")

    interpret_findings(
        llm,
        question="Which region leads on revenue?",
        findings=_findings(),
        seeds=SemanticSeeds(),
    )

    assert "pinned_definitions" not in llm.calls[0]["payload"]
