"""The exit gate for free-text agent answers, and the self-approval hole it closes.

The bounded tool loop replaced a single planned SQL with a model-chosen tool
combination, which removed the one deterministic checkpoint the old path had:
`di4_l1_interpretation`'s number gate. Until this gate exists, an agent answer
lands verbatim in the fact layer with only a system-prompt instruction behind it.

These tests drive the gate adversarially. The number pool is built solely from
the typed parse of persisted tool payloads, so an answer can never widen its own
admissible set, and a model/prediction assertion without a ModelCard is refused
outright. The mutation test clears the term family and asserts the model claim
then passes, which is the only proof that the guard has teeth.
"""

from __future__ import annotations

import pytest

from eda_platform.agents import interpretation
from eda_platform.agents.interpretation import validate_agent_answer
from eda_platform.schemas.artifacts import (
    Artifact,
    ArtifactType,
    EvidenceRef,
    SqlResult,
)
from eda_platform.schemas.model_card import ModelCard
from eda_platform.schemas.questions import QuestionFinding

PROJECT = "proj"
SESSION = "qsess_gate"


def _sql_artifact(
    artifact_id: str,
    *,
    rows: list[dict[str, object]],
    units: dict[str, str] | None = None,
) -> Artifact:
    columns = list(rows[0]) if rows else []
    return Artifact(
        id=artifact_id,
        type=ArtifactType.SQL_RESULT,
        project_id=PROJECT,
        session_id=SESSION,
        payload=SqlResult(
            sql="SELECT amount, n FROM creditcard",
            columns=columns,
            dtypes=dict.fromkeys(columns, "float64"),
            units=units or {},
            rows_preview=rows,
            row_count=len(rows),
        ).model_dump(mode="json"),
    )


def _model_card_artifact(artifact_id: str) -> Artifact:
    return Artifact(
        id=artifact_id,
        type=ArtifactType.MODEL_CARD,
        project_id=PROJECT,
        session_id=SESSION,
        payload=ModelCard(
            dataset_id="ds_1",
            task_type="classification",
            target_column="Class",
            feature_columns=["V1", "V2"],
            split_strategy="random_stratified",
            train_rows=800,
            test_rows=200,
            model_type="logistic_regression",
            metrics={"roc_auc": 0.91},
        ).model_dump(mode="json"),
    )


def test_a_model_claim_without_a_model_card_is_refused() -> None:
    """The observed failure: no model was built, yet the answer narrates one."""
    evidence = [_sql_artifact("sql_1", rows=[{"amount": 17982.1, "n": 50}])]

    admitted, reason = validate_agent_answer(
        "The model produced 50 example transactions with amount 17982.1.",
        evidence,
    )

    assert admitted is False
    assert "model" in reason


def test_the_same_model_claim_passes_once_a_model_card_is_present() -> None:
    """The entity gate keys on evidence, not on wording."""
    evidence = [
        _sql_artifact("sql_1", rows=[{"amount": 17982.1, "n": 50}]),
        _model_card_artifact("mc_1"),
    ]

    admitted, reason = validate_agent_answer(
        "The model scored the 50 transactions; the largest amount is 17982.1.",
        evidence,
    )

    assert admitted is True, reason


def test_clearing_the_model_term_family_admits_the_fabricated_model_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation check: with an empty term family the guard must go silent."""
    monkeypatch.setattr(interpretation, "MODEL_ASSERTION_TERMS", ())
    evidence = [_sql_artifact("sql_1", rows=[{"amount": 17982.1, "n": 50}])]

    admitted, _reason = validate_agent_answer(
        "The model produced 50 example transactions with amount 17982.1.",
        evidence,
    )

    assert admitted is True


def test_a_fabricated_number_is_refused() -> None:
    evidence = [_sql_artifact("sql_1", rows=[{"amount": 17982.1, "n": 50}])]

    admitted, reason = validate_agent_answer(
        "The largest amount is 24039.93 across 50 rows.",
        evidence,
    )

    assert admitted is False
    assert "24039.93" in reason


def test_a_traceable_answer_is_admitted() -> None:
    evidence = [_sql_artifact("sql_1", rows=[{"amount": 17982.1, "n": 50}])]

    admitted, reason = validate_agent_answer(
        "The largest amount is 17982.1 across 50 rows.",
        evidence,
    )

    assert admitted is True, reason


def test_an_answer_without_any_evidence_artifact_is_refused() -> None:
    admitted, reason = validate_agent_answer("The largest amount is 17982.1.", [])

    assert admitted is False
    assert reason


def test_causal_language_is_refused() -> None:
    evidence = [_sql_artifact("sql_1", rows=[{"amount": 17982.1, "n": 50}])]

    admitted, reason = validate_agent_answer(
        "The 17982.1 amount is caused by the 50 duplicated rows.",
        evidence,
    )

    assert admitted is False
    assert "causal" in reason


def test_an_answer_cannot_widen_its_own_number_pool() -> None:
    """The gate reads persisted payloads only; the answer text is never a source."""
    evidence = [_sql_artifact("sql_1", rows=[{"amount": 17982.1, "n": 50}])]

    admitted, _reason = validate_agent_answer(
        "Exactly 777.7 transactions exceeded 17982.1.",
        evidence,
    )

    assert admitted is False


def test_finding_text_numbers_do_not_enter_the_interpretation_pool() -> None:
    """`question_en` is model-authored and leads every finding text.

    Admitting its digits let a model approve its own interpretation by writing
    the number into the question first.
    """
    finding = QuestionFinding(
        text="How many of the 424242 transactions are fraudulent? 12 rows returned.",
        evidence=[
            EvidenceRef(
                kind="sql",
                artifact_id="sql_1",
                locator="rows[0].n",
                value=12,
                unit="raw",
            )
        ],
    )

    pool = [value for value, _is_percent in interpretation._allowed_numbers([finding])]

    assert 12 in pool
    assert 424242 not in pool
