"""R3 review fixes: threshold tokens must name the statistic they bound, and the
claim gate and the report-body gate must share one causal-phrase table."""

from __future__ import annotations

import pytest

from eda_platform.core.claim_language import (
    CAUSAL_PHRASES,
    REPORT_BODY_CAUSAL_TERMS,
    implies_causation,
)
from eda_platform.schemas.artifacts import EvidenceRef
from eda_platform.schemas.reports import ReportClaim
from eda_platform.tools import report_validator as rv
from eda_platform.tools.evidence import (
    EvidenceArtifactSummary,
    EvidencePack,
    EvidenceStatTest,
)


def _pack() -> EvidencePack:
    return EvidencePack(
        payload_policy="schema+aggregates",
        artifact_index={
            "stat_1": EvidenceArtifactSummary(
                artifact_id="stat_1",
                artifact_type="StatTestResult",
                title="t-test",
                dataset_id="ds_1",
            ),
        },
        stat_tests=[
            EvidenceStatTest(
                artifact_id="stat_1",
                dataset_id="ds_1",
                test_type="t_test",
                statistic=5.83,
                p_value=0.003,
                effect_size=0.42,
                sample_size=200,
            )
        ],
    )


def _token(text: str) -> rv._NumericToken:
    (token,) = rv._numeric_tokens_from_text(text)
    return token


def _statuses(text: str, locator: str) -> list[tuple[float, str]]:
    claim = ReportClaim(
        text=text,
        evidence=[EvidenceRef(kind="stat", artifact_id="stat_1", locator=locator)],
    )
    return [
        (status.number, status.status)
        for status in rv._numeric_token_statuses(
            claim, evidence_pack=_pack(), numeric_tolerance=0.01, sql_results={}
        )
    ]


# --- F1: threshold tokens are bound to the statistic they name ----------------


@pytest.mark.parametrize(
    "text",
    [
        "The difference is significant at p < 0.05.",
        "The difference is significant (p<0.001).",
        "p-value < 0.0001 for the group comparison.",
        "The difference is significant (p < 1e-10).",
    ],
)
def test_p_value_bounds_keep_their_threshold_subject(text: str) -> None:
    token = _token(text)
    assert token.threshold_op is not None
    assert token.threshold_subject == "p_value"


@pytest.mark.parametrize(
    ("text", "subject"),
    [
        ("The test statistic > 3.84 for the chi-square test.", "test_statistic"),
        ("Cohen's d > 0.8 between the groups.", "effect_size"),
        ("The effect size > 0.5 in every cohort.", "effect_size"),
    ],
)
def test_other_statistic_names_bind_to_their_own_field(text: str, subject: str) -> None:
    assert _token(text).threshold_subject == subject


@pytest.mark.parametrize(
    "text",
    [
        "Weekly enterprise churn stayed < 0.01 in every cohort.",
        "Total fraud losses this quarter were < 1000000 USD.",
        "Model lift over baseline was > 0.1 on every segment.",
        "Records processed: < 1470.",
        "Groups compared: < 9999.",
    ],
)
def test_business_quantities_get_no_threshold_subject(text: str) -> None:
    token = _token(text)
    assert token.threshold_op is not None
    assert token.threshold_subject is None


def test_a_p_bound_outside_the_unit_interval_is_not_a_p_value() -> None:
    # "p < 1000000" is not a p-value assertion; binding it would reopen the
    # laundering path the subject check exists to close.
    assert _token("Quarterly losses p < 1000000 USD.").threshold_subject is None


def test_stat_locators_declare_which_statistic_they_are_eligible_for() -> None:
    pack = _pack()
    for locator, subject in (
        ("p_value", "p_value"),
        ("statistic", "test_statistic"),
        ("effect_size", "effect_size"),
    ):
        values = rv._resolve_evidence_numbers(
            EvidenceRef(kind="stat", artifact_id="stat_1", locator=locator), pack, {}
        )
        assert [entry[3] for entry in values] == [subject]
    sample = rv._resolve_evidence_numbers(
        EvidenceRef(kind="stat", artifact_id="stat_1", locator="sample_size"), pack, {}
    )
    assert [entry[3] for entry in sample] == [None]


def test_a_business_bound_cannot_borrow_the_p_value_pool() -> None:
    assert _statuses(
        "Weekly enterprise churn stayed < 0.01 in every cohort.", "p_value"
    ) == [(0.01, "failed")]


def test_an_effect_size_bound_cannot_be_satisfied_by_the_p_value() -> None:
    # 0.003 satisfies "> 0.001" numerically, but the only cited locator is the
    # p-value, so an effect-size bound must not borrow it.
    assert _statuses("Cohen's d > 0.001 between the groups.", "p_value") == [
        (0.001, "failed")
    ]


def test_a_p_bound_still_verifies_against_the_p_value_pool() -> None:
    assert _statuses("The difference is significant (p < 0.05).", "p_value") == [
        (0.05, "number_verified")
    ]


def test_an_effect_size_bound_verifies_against_the_effect_size_pool() -> None:
    assert _statuses("Cohen's d > 0.2 between the groups.", "effect_size") == [
        (0.2, "number_verified")
    ]


# --- F4: one causal table, shared by both gates -------------------------------


def test_the_two_causal_gates_read_the_same_table() -> None:
    assert REPORT_BODY_CAUSAL_TERMS == CAUSAL_PHRASES


@pytest.mark.parametrize(
    "text",
    [
        "Discounts cause the 42 returns.",
        "Discounts drove the 42 returns.",
        "Discounts are responsible for the 42 returns.",
        "The 42 returns are attributable to discounts.",
        "The discount triggered the 42 returns.",
        "The 42 returns are a consequence of the discount.",
    ],
)
def test_bare_verb_forms_are_on_the_causal_table(text: str) -> None:
    assert implies_causation(text)


def test_the_causal_table_is_ascii_only() -> None:
    # The old comment advertised Chinese coverage the table never had.
    assert all(phrase.isascii() for phrase in CAUSAL_PHRASES)
