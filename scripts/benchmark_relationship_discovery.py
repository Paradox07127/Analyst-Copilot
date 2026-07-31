"""Offline relationship-discovery latency/coverage benchmark.

No model calls and no workspace writes. Example:

    uv run python scripts/benchmark_relationship_discovery.py data/*.csv \
      --expected-edge 'orders.csv.customer_id -> customers.csv.customer_id'
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from eda_platform.core.query import DuckDBQueryEngine
from eda_platform.tools.loader import load_csv
from eda_platform.tools.relationship_discovery import (
    discover_relationship_candidates,
    eager_validation_candidates,
    validate_relationships,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--expected-edge", action="append", default=[])
    parser.add_argument("--max-overlap-checks", type=int, default=4)
    args = parser.parse_args()

    datasets = [load_csv(path, dataset_id=f"bench_{index}") for index, path in enumerate(args.csv)]
    engine = DuckDBQueryEngine()
    for dataset in datasets:
        engine.register_frame(dataset.record.dataset_id, dataset.frame)

    started = perf_counter()
    candidates = discover_relationship_candidates(
        datasets,
        engine,
        max_overlap_checks_per_dataset_pair=args.max_overlap_checks,
    )
    discovered_at = perf_counter()
    validations = validate_relationships(eager_validation_candidates(candidates), engine)
    finished = perf_counter()

    useful_labels = {
        candidate.pair.label()
        for candidate in candidates.candidates
        if candidate.confidence in {"high", "medium"}
    }
    expected = set(args.expected_edge)
    found = expected & useful_labels
    output = {
        "dataset_count": len(datasets),
        "discovery_seconds": round(discovered_at - started, 6),
        "validation_seconds": round(finished - discovered_at, 6),
        "total_seconds": round(finished - started, 6),
        "overlap_pairs_evaluated": candidates.overlap_pairs_evaluated,
        "overlap_pairs_prefiltered": candidates.overlap_pairs_prefiltered,
        "coverage_status": candidates.coverage_status,
        "candidate_count": len(candidates.candidates),
        "high_medium_count": len(useful_labels),
        "high_medium_labels": sorted(useful_labels),
        "full_validation_count": len(validations.validations),
        "candidate_payload_bytes": len(candidates.model_dump_json().encode("utf-8")),
        "expected_edge_count": len(expected),
        "expected_edges_found": sorted(found),
        "expected_recall": round(len(found) / len(expected), 6) if expected else None,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
