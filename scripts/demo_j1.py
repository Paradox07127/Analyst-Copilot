#!/usr/bin/env python
"""J1 demo: multi-file upload -> fully automatic EDA -> evidence-backed report.

Runs 100% offline (deterministic fallback, no LLM key needed) on the repo's
golden e-commerce tables, exercising the whole M4 Run-A pipeline: profile ->
quality -> charts -> analysis tables -> relationship discovery (with the
seeded CU004 duplicate-key trap and CU999 orphan) -> question discovery ->
auto-execution of the top questions -> report export (markdown + HTML).

Usage:
    .venv/bin/python scripts/demo_j1.py
    .venv/bin/python scripts/demo_j1.py --files a.csv b.csv

The workspace defaults to a fresh temp directory and is kept so you can open
the exported report; the path is printed at the end.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "eda_platform" / "src"))

from eda_platform.core.config import (  # noqa: E402
    WorkspaceConfigError,
    require_absolute_workspace,
)
from eda_platform.core.llm import OfflineLLMClient  # noqa: E402
from eda_platform.drivers.auto_eda import run_auto_eda  # noqa: E402
from eda_platform.schemas.artifacts import ArtifactType  # noqa: E402
from eda_platform.schemas.questions import QuestionExecutionResult  # noqa: E402
from eda_platform.schemas.relations import RelationshipValidationSet  # noqa: E402
from eda_platform.schemas.sessions import TraceEvent  # noqa: E402

GOLDEN_DATA = REPO_ROOT / "eda_platform" / "tests" / "golden" / "data"
DEFAULT_FILES = [
    GOLDEN_DATA / "ecommerce_orders.csv",
    GOLDEN_DATA / "ecommerce_customers.csv",
    GOLDEN_DATA / "ecommerce_products.csv",
    GOLDEN_DATA / "ecommerce_marketing.csv",
]


def _print_trace(event: TraceEvent) -> None:
    print(f"  [trace] {event.event_type:<24} {event.name}")


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
        help="CSV files to analyse (default: golden e-commerce multi-table set)",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="workspace directory (default: fresh temp dir, kept after the run)",
    )
    parser.add_argument(
        "--context",
        default="E-commerce demo: orders, customers, products and marketing spend.",
        help="business context string fed to the report",
    )
    args = parser.parse_args()

    for file_path in args.files:
        if not file_path.exists():
            print(f"error: input file not found: {file_path}", file=sys.stderr)
            return 2
    try:
        workspace = (
            require_absolute_workspace(args.workspace)
            if args.workspace is not None
            else Path(tempfile.mkdtemp(prefix="eda_demo_j1_")).resolve()
        )
    except WorkspaceConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    _heading("J1: upload -> auto analysis -> report (offline deterministic)")
    print(f"inputs   : {[str(path) for path in args.files]}")
    print(f"workspace: {workspace}")
    print()

    result = run_auto_eda(
        args.files,
        workspace=workspace,
        project_id="demo_j1",
        business_context=args.context,
        llm=OfflineLLMClient(),
        on_trace_event=_print_trace,
    )

    _heading("Datasets")
    for loaded in result.loaded_datasets:
        frame = loaded.frame
        print(
            f"- {loaded.record.name:<28} rows={len(frame):<4} cols={len(frame.columns):<3} "
            f"encoding={loaded.record.encoding}"
        )

    _heading("Artifacts (evidence chain)")
    counts = Counter(artifact.type.value for artifact in result.artifacts)
    for type_name, count in sorted(counts.items()):
        print(f"- {type_name:<28} x{count}")

    _heading("Relationships (DuckDB-verified)")
    validation_artifact = next(
        (a for a in result.artifacts if a.type is ArtifactType.RELATIONSHIP_VALIDATION_SET),
        None,
    )
    if validation_artifact is None:
        print("(single table: relationship discovery skipped)")
    else:
        validations = RelationshipValidationSet.model_validate(validation_artifact.payload)
        for validation in validations.validations:
            print(
                f"- {validation.pair.label():<70} multiplier={validation.join_row_multiplier} "
                f"orphanL={validation.orphan_rate_left} card={validation.cardinality}"
            )
            for warning in validation.warnings:
                print(f"    warning: {warning}")

    _heading("Auto-executed questions (top-3 with evidence)")
    executions = [
        QuestionExecutionResult.model_validate(artifact.payload)
        for artifact in result.artifacts
        if artifact.type is ArtifactType.QUESTION_EXECUTION_RESULT
    ]
    if not executions:
        print("(no auto-executed questions)")
    for execution in executions:
        print(f"- [{execution.status}] {execution.question}")
        for finding in execution.findings[:2]:
            print(f"    finding: {finding.text}")

    _heading("Report")
    report_dir = workspace / "projects" / "demo_j1" / "runs" / result.session_id / "report"
    print(f"session_id  : {result.session_id}")
    print(f"markdown: {report_dir / 'report.md'}")
    print(f"html    : {report_dir / 'report.html'}")
    print()
    excerpt = result.report_markdown.splitlines()[:30]
    for line in excerpt:
        print(f"  {line}")
    print("  ...")

    _heading("Done")
    print(f"J1 demo complete: {len(result.artifacts)} artifacts, workspace kept at {workspace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
