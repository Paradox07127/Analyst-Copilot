#!/usr/bin/env python
"""J3 demo: multi-table -> relationship discovery -> ER diagram -> questions.

Runs 100% offline (deterministic template route, no LLM key needed) on the
golden e-commerce tables and walks the J3 journey end to end:

1. candidate generation with ensemble signals (name/type/overlap/uniqueness),
2. DuckDB validation of every >=medium candidate (join multiplier, orphan
   rates, cardinality) — the CU004 duplicate-key trap and the CU999 orphan
   are called out with warnings instead of being silently adopted,
3. ER diagram as Graphviz DOT (the UI renders the same payload),
4. question discovery (template route) with score breakdown,
5. auto-selection and execution of the top questions, with evidence-backed
   findings.

Usage:
    .venv/bin/python scripts/demo_j3.py
    .venv/bin/python scripts/demo_j3.py --files a.csv b.csv --dot-out er.dot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "eda_platform" / "src"))

from eda_platform.core.query import DuckDBQueryEngine  # noqa: E402
from eda_platform.drivers.question_exec import (  # noqa: E402
    execute_question_candidate,
    select_auto_execution_candidates,
)
from eda_platform.schemas.artifacts import ArtifactType  # noqa: E402
from eda_platform.schemas.questions import QuestionExecutionResult  # noqa: E402
from eda_platform.tools.analysis import create_analysis_tables  # noqa: E402
from eda_platform.tools.er_diagram import build_er_diagram  # noqa: E402
from eda_platform.tools.loader import load_csv  # noqa: E402
from eda_platform.tools.profiler import profile_dataset  # noqa: E402
from eda_platform.tools.quality import scan_quality  # noqa: E402
from eda_platform.tools.question_discovery import discover_question_candidates  # noqa: E402
from eda_platform.tools.relationship_discovery import (  # noqa: E402
    discover_relationship_candidates,
    validate_relationships,
)

GOLDEN_DATA = REPO_ROOT / "eda_platform" / "tests" / "golden" / "data"
DEFAULT_FILES = [
    GOLDEN_DATA / "ecommerce_orders.csv",
    GOLDEN_DATA / "ecommerce_customers.csv",
    GOLDEN_DATA / "ecommerce_products.csv",
    GOLDEN_DATA / "ecommerce_marketing.csv",
]
PROJECT_ID = "demo_j3"
RUN_ID = "demo_j3_run"


def _heading(title: str) -> None:
    print()
    print(f"=== {title} " + "=" * max(0, 66 - len(title)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--files",
        nargs="+",
        type=Path,
        default=DEFAULT_FILES,
        help="CSV files (default: golden e-commerce multi-table set)",
    )
    parser.add_argument(
        "--dot-out",
        type=Path,
        default=None,
        help="optional path to write the ER diagram DOT source",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="how many question candidates to list (default 10)",
    )
    args = parser.parse_args()

    for file_path in args.files:
        if not file_path.exists():
            print(f"error: input file not found: {file_path}", file=sys.stderr)
            return 2
    if len(args.files) < 2:
        print("error: J3 needs at least two tables", file=sys.stderr)
        return 2

    datasets = [
        load_csv(path, dataset_id=f"ds_{path.stem}") for path in args.files
    ]
    engine = DuckDBQueryEngine()
    for loaded in datasets:
        engine.register_frame(loaded.record.dataset_id, loaded.frame)

    _heading("1. Relationship candidates (deterministic ensemble)")
    candidates = discover_relationship_candidates(datasets, engine)
    by_confidence = {"high": 0, "medium": 0, "low": 0}
    for candidate in candidates.candidates:
        by_confidence[candidate.confidence] += 1
    print(
        f"{len(candidates.candidates)} candidates "
        f"(high={by_confidence['high']} medium={by_confidence['medium']} "
        f"low={by_confidence['low']}; truncated_pairs={candidates.truncated_pairs})"
    )
    for candidate in candidates.candidates:
        if candidate.confidence == "low":
            continue
        signals = candidate.signals
        print(
            f"- [{candidate.confidence:>6}] {candidate.pair.label()}\n"
            f"    score={candidate.ensemble_score:.3f} "
            f"overlap={signals.overlap_left_in_right:.2f} "
            f"right_unique={signals.right_unique_rate:.2f} "
            f"auto_adopted={candidate.auto_adopted}"
        )

    _heading("2. DuckDB validation (join trap / orphan / cardinality)")
    validations = validate_relationships(candidates, engine)
    for validation in validations.validations:
        print(
            f"- {validation.pair.label()}\n"
            f"    multiplier={validation.join_row_multiplier} "
            f"orphan_left={validation.orphan_rate_left} "
            f"orphan_right={validation.orphan_rate_right} "
            f"cardinality={validation.cardinality}"
        )
        for warning in validation.warnings:
            print(f"    warning: {warning}")

    _heading("3. ER diagram (Graphviz DOT, rendered by the Relationships page)")
    diagram = build_er_diagram(candidates, validations)
    print(diagram.dot_source)
    if args.dot_out is not None:
        args.dot_out.write_text(diagram.dot_source, encoding="utf-8")
        print(f"(DOT written to {args.dot_out})")

    _heading("4. Question discovery (template route, offline)")
    profile_artifacts = [
        profile_dataset(loaded, project_id=PROJECT_ID, session_id=RUN_ID) for loaded in datasets
    ]
    quality_artifacts = [
        scan_quality(profile, project_id=PROJECT_ID, session_id=RUN_ID)
        for profile in profile_artifacts
    ]
    analysis_artifacts = [
        artifact
        for loaded, profile in zip(datasets, profile_artifacts, strict=True)
        for artifact in create_analysis_tables(
            loaded, profile, project_id=PROJECT_ID, session_id=RUN_ID
        )
    ]
    question_set = discover_question_candidates(
        datasets,
        profile_artifacts=profile_artifacts,
        quality_artifacts=quality_artifacts,
        analysis_artifacts=analysis_artifacts,
        relationship_candidates=candidates,
        relationship_validations=validations,
    )
    print(
        f"{len(question_set.candidates)} candidates "
        f"(trivial_dropped={question_set.trivial_dropped} "
        f"dedup_dropped={question_set.dedup_dropped})"
    )
    for candidate in question_set.candidates[: args.top]:
        score = candidate.score
        print(
            f"- [{candidate.template_id or candidate.origin}] {candidate.question_en}\n"
            f"    det={score.deterministic_score:.2f} avail={score.data_availability:.2f} "
            f"signal={score.statistical_signal:.2f} quality_risk={score.quality_risk:.2f} "
            f"join_risk={score.join_risk:.2f}"
        )

    _heading("5. Auto-selected top questions -> execution -> findings")
    selected = select_auto_execution_candidates(
        question_set, relationship_candidates=candidates, limit=3
    )
    if not selected:
        print("(no candidates cleared the deterministic auto-execution gate)")
    for candidate in selected:
        artifacts = execute_question_candidate(
            candidate,
            datasets=datasets,
            project_id=PROJECT_ID,
            session_id=RUN_ID,
            parent_ids=[],
        )
        execution = next(
            QuestionExecutionResult.model_validate(artifact.payload)
            for artifact in artifacts
            if artifact.type is ArtifactType.QUESTION_EXECUTION_RESULT
        )
        print(f"- [{execution.status}] {execution.question}")
        if execution.sql:
            print(f"    sql: {execution.sql}")
        for finding in execution.findings:
            print(f"    finding: {finding.text}")

    _heading("Done")
    print(
        "J3 demo complete: "
        f"{by_confidence['high']} auto-adopted joins, "
        f"{by_confidence['medium']} sent to HITL review, "
        f"{len(selected)} questions executed with evidence."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
