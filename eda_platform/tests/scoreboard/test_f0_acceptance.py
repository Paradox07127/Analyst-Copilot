"""F0 acceptance: injecting a known-wrong number into an archived bundle must
produce a finding that locates the claim, the number, and the evidence pool."""

from __future__ import annotations

import pytest
from corpus import CorpusRun, load_corpus

from eda_platform.tools.report_validator import _NUMBER_PATTERN, validate_report_bundle


@pytest.fixture(scope="module")
def hr_run() -> CorpusRun:
    return next(run for run in load_corpus() if run.slug == "hr_attrition")


def test_injected_wrong_number_is_fully_located(hr_run: CorpusRun) -> None:
    bundle = hr_run.bundle.model_copy(deep=True)
    target = next(
        claim
        for section in bundle.sections
        for claim in section.claims
        if claim.id == "c9"
    )
    match = _NUMBER_PATTERN.search(target.text)
    assert match is not None and match.group(0) == "237"
    target.text = target.text[: match.start()] + "9999" + target.text[match.end() :]

    audit = validate_report_bundle(bundle, hr_run.pack, sql_results=hr_run.sql_results)
    numeric = [f for f in audit.findings if f.code == "numeric_mismatch"]
    assert len(numeric) == 1
    finding = numeric[0]
    assert finding.claim_id == "c9"
    detail = next(d for d in finding.numeric_details if d.number == 9999)
    assert detail.reason == "outside_tolerance"
    assert 237 in detail.evidence_values
    assert any(source.artifact_id and source.resolved for source in detail.sources)
    assert any(source.locator for source in detail.sources if source.resolved)


def test_clean_bundle_has_no_numeric_findings(hr_run: CorpusRun) -> None:
    audit = validate_report_bundle(
        hr_run.bundle, hr_run.pack, sql_results=hr_run.sql_results
    )
    assert [f for f in audit.findings if f.code == "numeric_mismatch"] == []
