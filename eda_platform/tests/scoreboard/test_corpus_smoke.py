"""Smoke test: the frozen scoreboard corpus loads and matches audited checksums."""

from __future__ import annotations

import pytest
from corpus import CorpusRun, load_corpus

from eda_platform.tools.report_validator import _NUMBER_PATTERN

EXPECTED_SLUGS = [
    "brazilian_e_commerce",
    "credit_card",
    "default_064005",
    "default_071054",
    "genai_llm_usage",
    "hr_attrition",
    "stack_overflow_2018",
    "world_cup",
]

# Frozen baseline from the 2026-07-23 validation-gate audit.
EXPECTED_TOTAL_CLAIMS = 114
EXPECTED_TOTAL_EVIDENCE_REFS = 234
EXPECTED_TOTAL_NUMBER_TOKENS = 124


@pytest.fixture(scope="module")
def corpus() -> list[CorpusRun]:
    return load_corpus()


def _claims(run: CorpusRun):
    return [claim for section in run.bundle.sections for claim in section.claims]


def test_all_runs_load(corpus: list[CorpusRun]) -> None:
    assert [run.slug for run in corpus] == EXPECTED_SLUGS


def test_corpus_checksums(corpus: list[CorpusRun]) -> None:
    claims = [claim for run in corpus for claim in _claims(run)]
    assert len(claims) == EXPECTED_TOTAL_CLAIMS
    assert sum(len(claim.evidence) for claim in claims) == EXPECTED_TOTAL_EVIDENCE_REFS
    number_tokens = sum(
        len(list(_NUMBER_PATTERN.finditer(claim.text))) for claim in claims
    )
    assert number_tokens == EXPECTED_TOTAL_NUMBER_TOKENS


@pytest.mark.parametrize("slug", EXPECTED_SLUGS)
def test_run_has_claims_and_sections(corpus: list[CorpusRun], slug: str) -> None:
    run = next(r for r in corpus if r.slug == slug)
    assert len(_claims(run)) > 0
    assert run.bundle.sections
