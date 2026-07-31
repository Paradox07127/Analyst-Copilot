"""F3 acceptance on the frozen corpus: quantitative coverage gaps.

Clean replay through the production entry (validate_report_bundle): every
quantitative-section claim with zero verified numbers is a coverage gap.
Totals and per-run distribution are pinned to the first measurement in
baseline.json; any drift is an explicit decision, not noise.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from corpus import CorpusRun, load_corpus

from eda_platform.schemas.reports import ReportAudit
from eda_platform.tools.report_validator import (
    QUANTITATIVE_SECTIONS,
    validate_report_bundle,
)

BASELINE_PATH = Path(__file__).resolve().parent / "baseline.json"


@pytest.fixture(scope="module")
def corpus() -> list[CorpusRun]:
    return load_corpus()


@pytest.fixture(scope="module")
def audits(corpus: list[CorpusRun]) -> dict[str, ReportAudit]:
    return {
        run.slug: validate_report_bundle(
            run.bundle, run.pack, sql_results=run.sql_results
        )
        for run in corpus
    }


@pytest.fixture(scope="module")
def baseline() -> dict:
    return json.loads(BASELINE_PATH.read_text())["quantitative_coverage_gaps"]


def test_gap_total_and_per_run_distribution_pinned(
    audits: dict[str, ReportAudit], baseline: dict
) -> None:
    by_run = {slug: audit.quantitative_coverage_gap_count for slug, audit in audits.items()}
    assert by_run == baseline["by_run"]
    assert sum(by_run.values()) == baseline["total"]


def test_section_51_sanitized_texts_are_flagged(corpus: list[CorpusRun]) -> None:
    # The confirmed draft-vs-published deletions (analysis-v3 §5.1) must be
    # visible: the published numberless versions carry the gap flag.
    expected = {
        ("stack_overflow_2018", "claim_3"): "show very high missing rates",
        ("default_071054", "c3"): "a high number of IQR outliers",
        # 2026-07-23 cross-review adjudication: all three §5.1 pairs must be
        # covered, so Executive Summary / Dataset Overview joined the
        # quantitative sections.
        ("hr_attrition", "c1"): "employee records with attributes",
    }
    flagged = {
        (run.slug, claim.id): claim.text
        for run in corpus
        for section in run.bundle.sections
        for claim in section.claims
        if claim.quantitative_coverage_gap
    }
    for key, snippet in expected.items():
        assert key in flagged
        assert snippet in flagged[key]


def test_gaps_only_in_quantitative_sections_and_zero_verified(
    corpus: list[CorpusRun],
) -> None:
    for run in corpus:
        for section in run.bundle.sections:
            for claim in section.claims:
                if not claim.quantitative_coverage_gap:
                    continue
                assert section.title in QUANTITATIVE_SECTIONS, (run.slug, claim.id)
                assert not any(
                    status.status == "number_verified"
                    for status in claim.numeric_statuses
                ), (run.slug, claim.id)


def test_gaps_produce_no_findings_and_no_verdict_change(
    audits: dict[str, ReportAudit],
) -> None:
    # Disclosure only: the gap count must not surface as findings or flip the
    # (pre-semantic-gate) verdict on the clean corpus replay.
    for slug, audit in audits.items():
        assert [f for f in audit.findings if "coverage" in f.code] == [], slug
        assert audit.gate_verdict == "pass", slug
