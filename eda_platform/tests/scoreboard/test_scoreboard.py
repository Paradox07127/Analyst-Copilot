"""Scoreboard guard: fixed-corpus metrics must never regress below baseline.

Totals are pinned exactly (a drift means the harness or corpus changed, which
must be an explicit decision); detection counts are ratchets (>= baseline),
independently for both detection scopes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scoreboard import compute_scoreboard

BASELINE_PATH = Path(__file__).resolve().parent / "baseline.json"

# hard_gate_detection: a finding attributes the exact mutated token.
# not_verified_disposition: hard detection OR the token was never published
# as verified (clean state unverified).
SCOPES = ("hard_gate_detection", "not_verified_disposition")


@pytest.fixture(scope="module")
def board() -> dict:
    return compute_scoreboard()


@pytest.fixture(scope="module")
def baseline() -> dict:
    return json.loads(BASELINE_PATH.read_text())


def test_mutation_totals_pinned(board: dict, baseline: dict) -> None:
    for scope in SCOPES:
        assert board[scope]["mutation_overall"][1] == baseline[scope]["mutation_overall"][1]
        for name, (_, total) in baseline[scope]["mutation_by_class"].items():
            assert board[scope]["mutation_by_class"][name][1] == total, (scope, name)
        assert board[scope]["count_plus1"][1] == baseline[scope]["count_plus1"][1]
    assert board["resolvable_refs"][1] == baseline["resolvable_refs"][1]


def test_mutation_detection_never_regresses(board: dict, baseline: dict) -> None:
    for scope in SCOPES:
        assert board[scope]["mutation_overall"][0] >= baseline[scope]["mutation_overall"][0]
        for name, (caught, _) in baseline[scope]["mutation_by_class"].items():
            assert board[scope]["mutation_by_class"][name][0] >= caught, (scope, name)
        assert board[scope]["count_plus1"][0] >= baseline[scope]["count_plus1"][0]


def test_disposition_scope_contains_hard_gate_scope(board: dict) -> None:
    # Disposition credits everything hard detection credits, plus the
    # unverified band; per class it can never be below the hard count.
    hard = board["hard_gate_detection"]
    disposition = board["not_verified_disposition"]
    assert disposition["mutation_overall"][0] >= hard["mutation_overall"][0]
    for name, (caught, _) in hard["mutation_by_class"].items():
        assert disposition["mutation_by_class"][name][0] >= caught, name
    assert disposition["count_plus1"][0] >= hard["count_plus1"][0]


def test_resolvable_refs_never_regress(board: dict, baseline: dict) -> None:
    assert board["resolvable_refs"][0] >= baseline["resolvable_refs"][0]


def test_clean_replay_stays_clean(board: dict, baseline: dict) -> None:
    # Legit published numbers (incl. the §5.5 in-window legitimate renderings,
    # 43-45 depending on the counting rule) must never be killed by a
    # tolerance change: zero mismatches on unmutated claims.
    assert board["clean_claim_mismatches"] == baseline["clean_claim_mismatches"] == 0


def test_clean_unverified_tokens_pinned(board: dict, baseline: dict) -> None:
    # The unverified band must not grow silently: any change is an explicit
    # baseline decision, not drift.
    assert board["clean_unverified_tokens"] == baseline["clean_unverified_tokens"]


def test_no_collateral_only_detections(board: dict, baseline: dict) -> None:
    # A mutation must be caught by flagging the mutated token itself; flagging
    # a different (legit) number in the same claim is a binding bug, not a catch.
    assert board["collateral_only_detections"] == baseline["collateral_only_detections"] == 0


def test_threshold_injection_no_eligible_blocked(board: dict, baseline: dict) -> None:
    # Republishing a clean verified token as "< {2x}" must not stay verified
    # when the own-unit pool has no threshold-eligible (stat) values. The one
    # allowed pass is world_cup claim_1 "20" -> "< 40": 40 is another true
    # value in the same claim pool, the claim-to-cell binding class deferred
    # in analysis-v3 §10 (same collision as its x2 mutation miss).
    blocked, total = board["threshold_injection"]["no_eligible_blocked"]
    base_blocked, base_total = baseline["threshold_injection"]["no_eligible_blocked"]
    assert total == base_total
    assert blocked >= base_blocked


def test_threshold_injection_eligible_passes_pinned(board: dict, baseline: dict) -> None:
    # Inequality phrasing over stat p_value/statistic/effect_size is legal
    # design, not a wash: measured pass-through pinned exactly.
    assert (
        board["threshold_injection"]["eligible_passed_by_design"]
        == baseline["threshold_injection"]["eligible_passed_by_design"]
    )


def test_corpus_content_frozen(baseline: dict) -> None:
    import hashlib

    root = Path(__file__).resolve().parent.parent / "fixtures" / "scoreboard_corpus"
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    assert digest.hexdigest() == baseline["corpus_sha256"]
