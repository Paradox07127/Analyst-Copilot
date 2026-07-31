#!/usr/bin/env python
"""Offline benchmark skeleton: golden e-commerce tables end-to-end, N times.

Runs the fully deterministic auto-EDA pipeline (no LLM key needed) on the
repo's golden e-commerce multi-table set, then prints a metrics table built
from each run's persisted SessionMetrics rollup (``core.session_metrics.summarize_session``):
duration, artifact counts by type, findings count, failures count — plus
mean/min/max when ``--repeat`` > 1.

Exit status is nonzero when any run raises.

Usage:
    .venv/bin/python scripts/benchmark_offline.py
    .venv/bin/python scripts/benchmark_offline.py --repeat 3
    .venv/bin/python scripts/benchmark_offline.py --workspace /tmp/bench_ws
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "eda_platform" / "src"))

from eda_platform.core.config import (  # noqa: E402
    WorkspaceConfigError,
    require_absolute_workspace,
)
from eda_platform.core.llm import OfflineLLMClient  # noqa: E402
from eda_platform.core.session_metrics import summarize_session  # noqa: E402
from eda_platform.core.store import ArtifactStore  # noqa: E402
from eda_platform.drivers.auto_eda import run_auto_eda  # noqa: E402
from eda_platform.drivers.investigation_orchestrator import (  # noqa: E402
    approve_plan,
    create_investigation_plans,
    execute_investigation_plans,
)
from eda_platform.schemas.artifacts import ArtifactType  # noqa: E402
from eda_platform.schemas.questions import QuestionCandidateSet  # noqa: E402
from eda_platform.schemas.session_metrics import SessionMetrics  # noqa: E402

GOLDEN_DATA = REPO_ROOT / "eda_platform" / "tests" / "golden" / "data"
DEFAULT_FILES = [
    GOLDEN_DATA / "ecommerce_orders.csv",
    GOLDEN_DATA / "ecommerce_customers.csv",
    GOLDEN_DATA / "ecommerce_products.csv",
    GOLDEN_DATA / "ecommerce_marketing.csv",
]


@dataclass(frozen=True)
class InvestigationMetrics:
    duration_seconds: float
    artifact_counts: dict[str, int]

    @property
    def artifact_total(self) -> int:
        return sum(self.artifact_counts.values())

    @property
    def finding_count(self) -> int:
        return self.artifact_counts.get(ArtifactType.VALIDATED_FINDING.value, 0)


def _heading(title: str) -> None:
    print()
    print(f"=== {title} " + "=" * max(0, 66 - len(title)))


def _run_investigation(result, store: ArtifactStore) -> InvestigationMetrics:  # noqa: ANN001
    candidate_artifact = next(
        artifact
        for artifact in result.artifacts
        if artifact.type is ArtifactType.QUESTION_CANDIDATE_SET
    )
    candidate_set = QuestionCandidateSet.model_validate(candidate_artifact.payload)
    candidate = next(
        candidate
        for candidate in candidate_set.candidates
        if candidate.feasibility is not None
        and candidate.feasibility.status == "ready"
        and candidate.sql_template is not None
    )

    started = perf_counter()
    planned = create_investigation_plans(
        project_id=result.project_id,
        source_session_id=result.session_id,
        question_ids=[candidate.question_id],
        workspace=result.workspace,
    )
    plan_artifact = next(
        artifact
        for artifact in planned.artifacts
        if artifact.type is ArtifactType.INVESTIGATION_PLAN
    )
    approve_plan(
        project_id=result.project_id,
        plan_session_id=planned.session_id,
        plan_id=plan_artifact.id,
        workspace=result.workspace,
        reason="Offline benchmark investigation path.",
    )
    completed = execute_investigation_plans(
        project_id=result.project_id,
        plan_session_id=planned.session_id,
        plan_ids=[plan_artifact.id],
        workspace=result.workspace,
    )
    if not any(
        artifact.type is ArtifactType.VALIDATED_FINDING
        for artifact in completed.artifacts
    ):
        raise AssertionError("Investigation did not produce a ValidatedFinding artifact")
    elapsed = perf_counter() - started
    artifacts = store.list_artifacts(
        project_id=result.project_id,
        session_id=planned.session_id,
    )
    return InvestigationMetrics(
        duration_seconds=elapsed,
        artifact_counts=dict(Counter(artifact.type.value for artifact in artifacts)),
    )


def _run_once(
    index: int,
    files: list[Path],
    workspace: Path,
    *,
    with_investigation: bool,
    dataset_workers: int,
) -> tuple[SessionMetrics, InvestigationMetrics | None]:
    """One end-to-end offline run in its own subdirectory (no checkpoint reuse)."""
    run_workspace = workspace / f"run_{index:02d}"
    result = run_auto_eda(
        files,
        workspace=run_workspace,
        project_id="benchmark_offline",
        business_context=(
            "E-commerce benchmark: orders, customers, products and marketing spend."
        ),
        llm=OfflineLLMClient(),
        dataset_workers=dataset_workers,
    )
    store = ArtifactStore(run_workspace)
    metrics = summarize_session(store, result.project_id, result.session_id)
    investigation = _run_investigation(result, store) if with_investigation else None
    return metrics, investigation


def _print_run_table(
    all_metrics: list[SessionMetrics],
    errors: list[str],
    investigation_metrics: list[InvestigationMetrics] | None = None,
) -> None:
    _heading("Per-run metrics")
    header = (
        f"{'run':<5} {'duration_s':>10} {'artifacts':>9} {'findings':>8} "
        f"{'failures':>8} {'llm':>4} {'tool':>5} {'tokens':>7}"
    )
    if investigation_metrics is not None:
        header += f" {'inv_s':>8} {'inv_art':>8} {'inv_find':>8}"
    print(header)
    print("-" * len(header))
    for index, metrics in enumerate(all_metrics, start=1):
        row = (
            f"{index:<5} {metrics.duration_seconds:>10.2f} "
            f"{sum(metrics.artifact_counts.values()):>9} "
            f"{metrics.findings_count:>8} {metrics.failures_count:>8} "
            f"{metrics.llm_calls:>4} {metrics.tool_calls:>5} {metrics.total_tokens:>7}"
        )
        if investigation_metrics is not None:
            investigation = investigation_metrics[index - 1]
            row += (
                f" {investigation.duration_seconds:>8.2f} "
                f"{investigation.artifact_total:>8} {investigation.finding_count:>8}"
            )
        print(row)
    for message in errors:
        print(f"FAIL  {message}")


def _print_artifact_breakdown(all_metrics: list[SessionMetrics]) -> None:
    _heading("Artifact counts by type (per run)")
    type_names = sorted({name for m in all_metrics for name in m.artifact_counts})
    for name in type_names:
        counts = [m.artifact_counts.get(name, 0) for m in all_metrics]
        print(f"- {name:<30} " + " ".join(f"{count:>4}" for count in counts))


def _print_aggregates(all_metrics: list[SessionMetrics]) -> None:
    _heading("Aggregates (mean / min / max)")
    rows: list[tuple[str, list[float]]] = [
        ("duration_seconds", [m.duration_seconds for m in all_metrics]),
        (
            "artifact_total",
            [float(sum(m.artifact_counts.values())) for m in all_metrics],
        ),
        ("findings_count", [float(m.findings_count) for m in all_metrics]),
        ("failures_count", [float(m.failures_count) for m in all_metrics]),
    ]
    for label, values in rows:
        print(
            f"- {label:<18} mean={mean(values):.2f} "
            f"min={min(values):.2f} max={max(values):.2f}"
        )


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
        "--repeat",
        type=int,
        default=1,
        help="number of end-to-end runs (default: 1)",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="workspace directory (default: fresh temp dir, kept after the runs)",
    )
    parser.add_argument(
        "--with-investigation",
        action="store_true",
        help="also plan, approve, and execute one ready template investigation",
    )
    parser.add_argument(
        "--dataset-workers",
        type=int,
        choices=(1, 2),
        default=1,
        help="independent dataset compute workers (default: 1; benchmark before choosing 2)",
    )
    args = parser.parse_args()

    if args.repeat < 1:
        print("error: --repeat must be >= 1", file=sys.stderr)
        return 2
    for file_path in args.files:
        if not file_path.exists():
            print(f"error: input file not found: {file_path}", file=sys.stderr)
            return 2
    try:
        workspace = (
            require_absolute_workspace(args.workspace)
            if args.workspace is not None
            else Path(tempfile.mkdtemp(prefix="eda_bench_")).resolve()
        )
    except WorkspaceConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    _heading("Offline benchmark: golden data end-to-end")
    print(f"inputs   : {[str(path) for path in args.files]}")
    print(f"workspace: {workspace}")
    print(f"repeat   : {args.repeat}")
    print(f"workers  : {args.dataset_workers}")

    all_metrics: list[SessionMetrics] = []
    investigation_metrics: list[InvestigationMetrics] = []
    errors: list[str] = []
    for index in range(1, args.repeat + 1):
        print(f"\n--- run {index}/{args.repeat} ---")
        try:
            metrics, investigation = _run_once(
                index,
                list(args.files),
                workspace,
                with_investigation=args.with_investigation,
                dataset_workers=args.dataset_workers,
            )
        except Exception as exc:  # noqa: BLE001 - a failed run must not stop the batch
            errors.append(f"run {index}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            continue
        all_metrics.append(metrics)
        if investigation is not None:
            investigation_metrics.append(investigation)
        print(
            f"run {index} ok: duration={metrics.duration_seconds:.2f}s "
            f"artifacts={sum(metrics.artifact_counts.values())} "
            f"findings={metrics.findings_count} failures={metrics.failures_count}"
        )

    if all_metrics:
        _print_run_table(
            all_metrics,
            errors,
            investigation_metrics if args.with_investigation else None,
        )
        _print_artifact_breakdown(all_metrics)
        if len(all_metrics) > 1:
            _print_aggregates(all_metrics)

    _heading("Done")
    if errors:
        print(f"benchmark FAILED: {len(errors)} of {args.repeat} run(s) errored")
        return 1
    print(f"benchmark complete: {len(all_metrics)} run(s), workspace kept at {workspace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
