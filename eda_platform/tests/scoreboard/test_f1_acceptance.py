"""F1 acceptance on the frozen corpus: type dispatch raises resolvable refs,
inline self-certification is gone, and the clean replay stays finding-free
through the production entry point (not a private wrapper)."""

from __future__ import annotations

import pytest
from corpus import CorpusRun, load_corpus

from eda_platform.tools import report_validator as rv
from eda_platform.tools.report_validator import validate_report_bundle


@pytest.fixture(scope="module")
def corpus() -> list[CorpusRun]:
    return load_corpus()


def test_clean_replay_has_no_numeric_mismatch_via_production_gate(
    corpus: list[CorpusRun],
) -> None:
    for run in corpus:
        bundle = run.bundle.model_copy(deep=True)
        audit = validate_report_bundle(bundle, run.pack, sql_results=run.sql_results)
        numeric = [f for f in audit.findings if f.code == "numeric_mismatch"]
        assert numeric == [], (run.slug, [(f.claim_id, f.message) for f in numeric])


def test_world_cup_profile_claim_is_number_verified(corpus: list[CorpusRun]) -> None:
    # R3: claim_1 cites profile rows/columns via kind="artifact"; type dispatch
    # must verify it from the persisted profile instead of the inline values.
    run = next(r for r in corpus if r.slug == "world_cup")
    bundle = run.bundle.model_copy(deep=True)
    validate_report_bundle(bundle, run.pack, sql_results=run.sql_results)
    claim = next(
        c for section in bundle.sections for c in section.claims if c.id == "claim_1"
    )
    assert claim.numeric_statuses
    assert all(s.status == "number_verified" for s in claim.numeric_statuses)
    assert claim.numeric_rollup == "number_verified"


def test_resolvable_refs_reach_type_dispatch_floor(corpus: list[CorpusRun]) -> None:
    resolved = 0
    total = 0
    for run in corpus:
        for section in run.bundle.sections:
            for claim in section.claims:
                for evidence in claim.evidence:
                    total += 1
                    if rv._resolve_evidence_numbers(evidence, run.pack, run.sql_results):
                        resolved += 1
    assert total == 234
    assert resolved >= 138


def test_no_source_contributes_values_without_resolution(
    corpus: list[CorpusRun],
) -> None:
    # Invariant: inline fallback fully removed -> an unresolved ref contributes
    # zero values everywhere in the corpus.
    for run in corpus:
        for section in run.bundle.sections:
            for claim in section.claims:
                _, sources = rv._numeric_evidence_values_with_sources(
                    claim, run.pack, run.sql_results
                )
                for source in sources:
                    assert source.resolved or source.value_count == 0, (
                        run.slug,
                        claim.id,
                        source,
                    )
