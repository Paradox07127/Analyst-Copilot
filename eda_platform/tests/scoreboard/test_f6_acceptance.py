"""F6 acceptance on the frozen corpus: evidence-strength tiers + ratio verdict.

Per-run strong/indicative/exploratory counts and the strong-ratio verdict are
pinned to the first measurement in baseline.json; any drift is an explicit
decision, not noise.

Pre-registered control (adjudicated by the lead model, NOT scored here): the
previous audit's subjectively weakest three runs are brazilian_e_commerce,
stack_overflow_2018 and world_cup. The final strong_ratio_cut is chosen
against this distribution by the lead model; the 0.60 constant in
report_validator.py is a placeholder until then.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from corpus import CorpusRun, load_corpus

from eda_platform.tools.report_validator import strong_ratio_verdict
from scoreboard import compute_scoreboard

BASELINE_PATH = Path(__file__).resolve().parent / "baseline.json"


@pytest.fixture(scope="module")
def board() -> dict:
    return compute_scoreboard()


@pytest.fixture(scope="module")
def corpus() -> list[CorpusRun]:
    return load_corpus()


@pytest.fixture(scope="module")
def baseline() -> dict:
    return json.loads(BASELINE_PATH.read_text())["evidence_strength"]


def test_strength_distribution_and_verdicts_pinned(board: dict, baseline: dict) -> None:
    assert board["evidence_strength"]["by_run"] == baseline["by_run"]
    assert board["evidence_strength"]["strong_ratio_cut"] == baseline["strong_ratio_cut"]


def test_verdict_is_exactly_the_strong_ratio_rule(board: dict, baseline: dict) -> None:
    for slug, row in board["evidence_strength"]["by_run"].items():
        total = row["strong"] + row["indicative"] + row["exploratory"]
        expected = strong_ratio_verdict(
            row["strong"], total, cut=baseline["strong_ratio_cut"]
        )
        assert row["verdict"] == expected, slug


def test_denominator_excludes_only_legacy_qfocus_claims(
    board: dict, corpus: list[CorpusRun]
) -> None:
    # F4 exclusion: qfocus_* claims are the pre-F4 catalog form (their
    # synthetic evidence chain was deleted); everything else is counted.
    for run in corpus:
        claims = [
            claim
            for section in run.bundle.sections
            for claim in section.claims
            if not (claim.id or "").startswith("qfocus_")
        ]
        row = board["evidence_strength"]["by_run"][run.slug]
        assert row["strong"] + row["indicative"] + row["exploratory"] == len(claims), (
            run.slug
        )
