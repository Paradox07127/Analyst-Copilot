"""Guards on the expected_policy manifest: coverage, key uniqueness, and the
frozen policy distribution. A drift here silently reshapes every per-class
scoreboard metric, so it must be an explicit, reviewed change."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from corpus import CorpusRun, load_corpus

from eda_platform.tools.report_validator import _NUMBER_PATTERN
from scoreboard import MANIFEST_PATH

BASELINE_PATH = Path(__file__).resolve().parent / "baseline.json"


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


@pytest.fixture(scope="module")
def corpus() -> list[CorpusRun]:
    return load_corpus()


def test_manifest_covers_every_corpus_token(manifest: dict, corpus: list[CorpusRun]) -> None:
    corpus_tokens: dict[tuple[str, str, int], str] = {}
    for run in corpus:
        for section in run.bundle.sections:
            for claim in section.claims:
                for index, match in enumerate(_NUMBER_PATTERN.finditer(claim.text)):
                    corpus_tokens[(run.slug, claim.id, index)] = match.group(0)
    manifest_tokens = {
        (e["run"], e["claim_id"], e["token_index"]): e["token"]
        for e in manifest["entries"]
    }
    assert manifest_tokens == corpus_tokens


def test_manifest_keys_unique(manifest: dict) -> None:
    keys = [(e["run"], e["claim_id"], e["token_index"]) for e in manifest["entries"]]
    assert len(keys) == len(set(keys))


def test_manifest_policy_distribution_frozen(manifest: dict) -> None:
    baseline = json.loads(BASELINE_PATH.read_text())
    distribution = Counter(e["policy"] for e in manifest["entries"])
    assert dict(distribution) == baseline["manifest_policy_distribution"]


def test_rounded_entries_declare_decimals(manifest: dict) -> None:
    for entry in manifest["entries"]:
        if entry["policy"] == "rounded":
            assert entry["decimals"] is not None, entry["claim_id"]
