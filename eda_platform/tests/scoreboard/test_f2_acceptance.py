"""F2 acceptance on the frozen corpus: value-tiered numeric matching.

Red/green probes (analysis-v3 §5.2): near-miss counts that the old global
±1% window accepted must now fail; the controls (display rounding 0.20 and
the two inequality p-value claims) must stay number_verified — they are the
truths the old 0.01 absolute floor happened to protect.
"""

from __future__ import annotations

import json

import pytest
from corpus import CorpusRun, load_corpus

from eda_platform.schemas.artifacts import EvidenceRef
from eda_platform.schemas.reports import ReportClaim
from eda_platform.tools import report_validator as rv
from scoreboard import MANIFEST_PATH


@pytest.fixture(scope="module")
def corpus() -> list[CorpusRun]:
    return load_corpus()


def _run(corpus: list[CorpusRun], slug: str) -> CorpusRun:
    return next(run for run in corpus if run.slug == slug)


def _claim(run: CorpusRun, claim_id: str) -> ReportClaim:
    return next(
        claim
        for section in run.bundle.sections
        for claim in section.claims
        if claim.id == claim_id
    )


def _statuses(run: CorpusRun, claim: ReportClaim) -> list[tuple[float, str]]:
    return [
        (status.number, status.status)
        for status in rv._numeric_token_statuses(
            claim,
            evidence_pack=run.pack,
            numeric_tolerance=0.01,
            sql_results=run.sql_results,
        )
    ]


def _status_of(run: CorpusRun, claim: ReportClaim, number: float) -> str:
    return next(status for value, status in _statuses(run, claim) if value == number)


# --- §5.2 probes: near-miss counts must fail (red on the ±1% window) ---


@pytest.mark.parametrize("wrong", [1465, 1484])
def test_hr_off_count_fails_against_profile(corpus: list[CorpusRun], wrong: int) -> None:
    # True HR profile: 1470 rows, 35 columns. 1465/1484 sat inside the old
    # ±14.7 window; exact count policy must reject them.
    run = _run(corpus, "hr_attrition")
    profile = next(d for d in run.pack.datasets if d.row_count == 1470)
    claim = ReportClaim(
        text=f"The dataset contains {wrong} employee records with 35 attributes.",
        evidence=[
            EvidenceRef(kind="artifact", artifact_id=profile.artifact_id, locator="summary")
        ],
    )
    assert _status_of(run, claim, wrong) == "failed"
    assert _status_of(run, claim, 35) == "number_verified"


def test_credit_card_row_count_2193_off_fails(corpus: list[CorpusRun]) -> None:
    # True 284807 rows; 287000 sat inside the old ±2848 window.
    run = _run(corpus, "credit_card")
    claim = _claim(run, "dataset_overview_ds_78bf2a42205a")
    assert "284807" in claim.text
    mutated = claim.model_copy(update={"text": claim.text.replace("284807", "287000")})
    assert _status_of(run, mutated, 287000) == "failed"
    details = rv._numeric_mismatch_details(
        mutated,
        evidence_pack=run.pack,
        numeric_tolerance=0.01,
        sql_results=run.sql_results,
    )
    assert any(detail.number == 287000 for detail in details)


# --- Controls: legitimate roundings/inequalities must stay verified ---


def test_world_cup_display_rounding_stays_verified(corpus: list[CorpusRun]) -> None:
    # claim_7 renders r=0.20 from 0.196566...: rounded(2) must accept it
    # without the removed 0.01 absolute floor.
    run = _run(corpus, "world_cup")
    claim = _claim(run, "claim_7")
    assert _status_of(run, claim, 0.20) == "number_verified"


@pytest.mark.parametrize(
    ("slug", "claim_id", "bound"),
    [("hr_attrition", "c8", 0.0001), ("default_071054", "c5", 0.001)],
)
def test_p_value_inequalities_stay_verified(
    corpus: list[CorpusRun], slug: str, claim_id: str, bound: float
) -> None:
    # "p < 0.0001" / "p<0.001" are inequality assertions: the threshold tier
    # must verify them (evidence 1.38e-8 / 0.0 satisfies the bound); without
    # it, removing the absolute floor would kill these true claims.
    run = _run(corpus, slug)
    claim = _claim(run, claim_id)
    assert _status_of(run, claim, bound) == "number_verified"


def test_sci_notation_bound_is_parsed_and_checked(corpus: list[CorpusRun]) -> None:
    # "p < 1e-10" against truth 1.38e-8: the exponent is part of the token.
    # Pre-fix the tokenizer cut it to "1" and the bound trivially passed.
    run = _run(corpus, "hr_attrition")
    claim = _claim(run, "c8")
    assert "p < 0.0001" in claim.text
    mutated = claim.model_copy(update={"text": claim.text.replace("0.0001", "1e-10")})
    assert _statuses(run, mutated) == [(1e-10, "failed")]


# --- Derivation-consistency guard: validator policy vs manifest label ---

# Measured final states of the 4 label tokens (name/code numbers). F4 removed
# the synthetic focus pool, so the three qfocus title numbers in the frozen
# (legacy-form) bundles no longer self-verify: their qexec_* refs resolve
# nothing and the tokens land in the unverified band, with an empty hit set.
# rag_enabled=0 still verifies exactly from a persisted int cell.
EXPECTED_LABEL_STATES = {
    ("default_071054", "qfocus_q_3ef726b2e0", 0): ("unverified", ()),
    ("default_071054", "qfocus_q_b662582ffb", 0): ("unverified", ()),
    ("genai_llm_usage", "qfocus_q_6dbe9ebef1", 0): ("unverified", ()),
    ("genai_llm_usage", "c8", 0): ("number_verified", ("exact",)),
}


def test_policy_derivation_matches_manifest(corpus: list[CorpusRun]) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    labels = {
        (e["run"], e["claim_id"], e["token_index"]): e["policy"]
        for e in manifest["entries"]
    }
    seen_labels: dict[tuple[str, str, int], tuple[str, tuple[str, ...]]] = {}
    verified = {"count": 0, "rounded": 0, "threshold": 0}
    for run in corpus:
        for section in run.bundle.sections:
            for claim in section.claims:
                tokens = rv._numeric_tokens_from_text(claim.text)
                if not tokens:
                    continue
                statuses = rv._numeric_token_statuses(
                    claim,
                    evidence_pack=run.pack,
                    numeric_tolerance=0.01,
                    sql_results=run.sql_results,
                )
                values, _ = rv._numeric_evidence_values_with_sources(
                    claim, run.pack, run.sql_results
                )
                for index, token in enumerate(tokens):
                    key = (run.slug, claim.id, index)
                    policy = labels.get(key)
                    if policy is None:
                        continue
                    status = statuses[index].status
                    pool = [
                        (value, value_policy)
                        for value, unit, value_policy, _eligible in values
                        if (unit == "percent") == token.is_percent
                    ]
                    hits = tuple(
                        sorted(
                            {
                                value_policy
                                for value, value_policy in pool
                                if rv._value_supports_token(token, value, value_policy)
                            }
                        )
                    )
                    if policy == "label":
                        seen_labels[key] = (status, hits)
                        continue
                    if policy == "threshold":
                        assert token.threshold_op is not None, key
                        assert status == "number_verified", key
                        verified["threshold"] += 1
                        continue
                    if status != "number_verified":
                        # Zero-resolution claims (QualityIssue prose numbers)
                        # stay unverified; anything else must verify.
                        assert status == "unverified", (key, status)
                        continue
                    verified[policy] += 1
                    if policy == "count":
                        # units=="count" is producer-authoritative: even a JSON
                        # float count cell (late_rows 7827.0) must hit exact.
                        assert "exact" in hits, (key, hits)
                    else:  # rounded
                        assert "rounded" in hits, (key, hits)
    assert seen_labels == EXPECTED_LABEL_STATES
    # Verified-token counts pinned: silent shrinkage of any tier is a corpus
    # or gate change, never drift.
    assert verified == {"count": 49, "rounded": 57, "threshold": 2}
