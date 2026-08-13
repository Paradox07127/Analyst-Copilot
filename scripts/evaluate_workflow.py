#!/usr/bin/env python
"""Run or replay deterministic whole-workflow quality/cost evaluations.

Examples:
    uv run python scripts/evaluate_workflow.py \
      --case eda_platform/tests/evals/workflow_quality/cases/semantic_guardrails.json \
      --input-dir eda_platform/tests/evals/workflow_quality/data --repeat 3

    uv run python scripts/evaluate_workflow.py \
      --case eda_platform/tests/evals/workflow_quality/cases/olist.json \
      --session-dir /path/to/projects/project/sessions/session_id
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "eda_platform" / "src"))

from eda_platform.core.config import (  # noqa: E402
    WorkspaceConfigError,
    require_absolute_workspace,
)
from eda_platform.drivers.workflow_eval import run_fresh_workflow_eval_case  # noqa: E402
from eda_platform.schemas.artifacts import Artifact, ArtifactType  # noqa: E402
from eda_platform.schemas.sessions import TraceEvent  # noqa: E402
from eda_platform.schemas.workflow_eval import (  # noqa: E402
    WorkflowEvalComparison,
    WorkflowEvalFailureNode,
    WorkflowEvalSpec,
    WorkflowEvalSuiteResult,
    WorkflowEvalTrial,
)
from eda_platform.tools.workflow_eval import (  # noqa: E402
    aggregate_workflow_eval_trials,
    build_workflow_eval_trial,
    compare_workflow_evaluations,
    grade_workflow_quality,
    verify_workflow_eval_trial_sources,
)


def _load_spec(path: Path) -> WorkflowEvalSpec:
    return WorkflowEvalSpec.model_validate_json(path.read_text(encoding="utf-8"))


def _load_run_artifacts(session_dir: Path) -> list[Artifact]:
    artifact_dir = session_dir / "artifacts"
    if not artifact_dir.is_dir():
        raise ValueError(f"Run directory has no artifacts folder: {session_dir}")
    return [
        Artifact.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(artifact_dir.glob("*.json"))
    ]


def _load_trace_events(session_dir: Path) -> list[TraceEvent]:
    trace_path = session_dir / "trace.jsonl"
    if not trace_path.is_file():
        return []
    events: list[TraceEvent] = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(TraceEvent.model_validate_json(line))
    return events


def _persisted_trial(artifacts: list[Artifact]) -> WorkflowEvalTrial:
    candidates = [
        artifact for artifact in artifacts if artifact.type is ArtifactType.WORKFLOW_EVAL_TRIAL
    ]
    if not candidates:
        raise ValueError("Fresh workflow eval did not persist its canonical trial.")
    artifact = max(candidates, key=lambda item: item.created_at)
    trial = WorkflowEvalTrial.model_validate(artifact.payload)
    failures = verify_workflow_eval_trial_sources(trial, artifacts)
    if failures:
        return trial.model_copy(
            update={
                "status": "failed",
                "failure_nodes": [*trial.failure_nodes, *failures],
            }
        )
    return trial


def _print_summary(result: WorkflowEvalSuiteResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    print(f"{status} workflow eval: {result.case_name}")
    print(
        f"runs={len(result.quality_results)} stability={result.stability_rate:.3f} "
        f"duration_mean={result.duration_mean_seconds:.3f}s "
        f"duration_p95={result.duration_p95_seconds:.3f}s "
        f"tokens_mean={result.tokens_mean:.1f}"
    )
    for index, run in enumerate(result.quality_results, start=1):
        answer_metrics = (
            f"{run.answer_precision:.3f}/{run.answer_recall:.3f}"
            if run.expected_answer_count or run.answered_count
            else "n/a"
        )
        abstention_metrics = (
            f"{run.abstention_precision:.3f}/{run.abstention_recall:.3f}"
            if run.expected_abstention_count or run.abstained_count
            else "n/a"
        )
        print(
            f"run[{index}] {'pass' if run.passed else 'fail'} "
            f"answer_p/r={answer_metrics} "
            f"abstain_p/r={abstention_metrics} "
            f"report={run.report_dataset_coverage:.3f} "
            f"quality={run.quality_dataset_coverage:.3f} "
            f"escape={run.semantic_escape_rate:.3f} "
            f"tokens={run.total_tokens}"
        )
    for failure in result.gate_failures:
        print(f"- {failure}")


def _print_comparison(comparison: WorkflowEvalComparison) -> None:
    print(f"{'PASS' if comparison.passed else 'FAIL'} baseline comparison")
    for name, delta in sorted(comparison.metric_deltas.items()):
        print(f"delta.{name}={delta:.6f}")
    for failure in comparison.gate_failures:
        print(f"- {failure}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True, help="workflow eval case JSON")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--session-dir",
        type=Path,
        nargs="+",
        help="one or more existing run directories to replay",
    )
    source.add_argument(
        "--input-dir",
        type=Path,
        help="directory containing the case's input_files; executes fresh offline runs",
    )
    parser.add_argument("--repeat", type=int, default=1, help="fresh-run repeat count")
    parser.add_argument("--workspace", type=Path, default=None, help="fresh-run workspace")
    parser.add_argument("--output", type=Path, default=None, help="write suite JSON here")
    parser.add_argument("--baseline", type=Path, default=None, help="baseline suite JSON")
    parser.add_argument(
        "--comparison-output", type=Path, default=None, help="write comparison JSON here"
    )
    args = parser.parse_args()

    if args.repeat < 1:
        parser.error("--repeat must be >= 1")
    spec = _load_spec(args.case)
    if args.session_dir:
        artifact_runs = [_load_run_artifacts(path) for path in args.session_dir]
        trials: list[WorkflowEvalTrial] = []
        for repetition, (path, artifacts) in enumerate(
            zip(args.session_dir, artifact_runs, strict=True), start=1
        ):
            events = _load_trace_events(path)
            trial = build_workflow_eval_trial(
                spec,
                artifacts,
                events=events,
                repetition=repetition,
                trace_ref=str((path / "trace.jsonl").resolve()) if events else None,
            )
            if not events:
                trial = trial.model_copy(
                    update={
                        "status": "inconclusive",
                        "failure_nodes": [
                            *trial.failure_nodes,
                            WorkflowEvalFailureNode(
                                node_id="__trace__",
                                code="trace_unavailable",
                                message="Replay cannot be release-eligible without durable trace.",
                            ),
                        ],
                    }
                )
            trials.append(trial)
    else:
        try:
            workspace = (
                require_absolute_workspace(args.workspace)
                if args.workspace is not None
                else Path(tempfile.mkdtemp(prefix="workflow_eval_")).resolve()
            )
        except WorkspaceConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        artifact_runs = run_fresh_workflow_eval_case(
            spec,
            input_dir=args.input_dir,
            workspace=workspace,
            repeat=args.repeat,
        )
        trials = [_persisted_trial(artifacts) for artifacts in artifact_runs]
        print(f"workspace={workspace}")
    results = [grade_workflow_quality(artifacts, spec) for artifacts in artifact_runs]
    suite = aggregate_workflow_eval_trials(spec, results=results, trials=trials)
    _print_summary(suite)
    comparison: WorkflowEvalComparison | None = None
    if args.baseline is not None:
        baseline = WorkflowEvalSuiteResult.model_validate_json(
            args.baseline.read_text(encoding="utf-8")
        )
        comparison = compare_workflow_evaluations(spec, baseline=baseline, current=suite)
        _print_comparison(comparison)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(suite.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"output={args.output}")
    if args.comparison_output is not None:
        if comparison is None:
            parser.error("--comparison-output requires --baseline")
        args.comparison_output.parent.mkdir(parents=True, exist_ok=True)
        args.comparison_output.write_text(
            json.dumps(comparison.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"comparison_output={args.comparison_output}")
    return 0 if suite.passed and (comparison is None or comparison.passed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
