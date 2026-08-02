"""Per-item JSON + bucketed summary I/O skeleton for Eval-0.

Layout borrows VeriGraph's judge.py shape (doc §10.3): one JSON file per
item×model×tier×seed under results/<bucket>/, plus a bucket-aggregated
summary. Judges are the deterministic checkers in ``checkers.py``.
"""

from __future__ import annotations

import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .checkers import CHECKER_VERSION


# Scored metrics where a larger number is a worse result. Everything else is
# treated as higher-is-better, so a metric added later defaults to that.
LOWER_IS_BETTER_METRICS = frozenset(
    {
        "absent_pattern_violations",
        "canary_leak_count",
        "forbidden_call_count",
        "duplicate_rate",
        "fabricated_receipt_rate",
    }
)


def metric_direction(name: str) -> str:
    return "lower_is_better" if name in LOWER_IS_BETTER_METRICS else "higher_is_better"


class RunUsage(BaseModel):
    """Efficiency record per R7: requests/tokens/cost, tool calls, scan
    volume, wall clock. p50/p95 live in the summary, not per run."""

    llm_requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float | None = None
    tool_calls: int = 0
    rows_scanned: int = 0
    cells_scanned: int = 0
    wall_clock_seconds: float = 0.0


class ItemResult(BaseModel):
    item_id: str
    bucket: str  # planted | negative | injection | external
    suite: str  # capability | regression
    model: str
    tier: str  # quick | standard | deep
    seed: int
    status: Literal["scored", "not_run", "error"]
    passed: bool | None = None
    scores: dict[str, float] = Field(default_factory=dict)
    violations: list[str] = Field(default_factory=list)
    usage: RunUsage = Field(default_factory=RunUsage)
    checker_version: str = CHECKER_VERSION
    capability_catalog_version: str = ""
    error: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class SuiteItem(BaseModel):
    item_id: str
    bucket: str
    dataset: str = ""
    ground_truth: str = ""
    task: str = ""
    status: str = "ready"  # ready | blocked_no_license | deferred
    notes: str = ""


class Suite(BaseModel):
    suite: Literal["capability", "regression"]
    items: list[SuiteItem]


def load_suite(path: Path) -> Suite:
    return Suite.model_validate_json(Path(path).read_text(encoding="utf-8"))


def item_result_path(results_dir: Path, result: ItemResult) -> Path:
    name = f"{result.item_id}__{result.model}__{result.tier}__seed{result.seed}.json"
    return Path(results_dir) / result.bucket / name


def write_item_result(results_dir: Path, result: ItemResult) -> Path:
    path = item_result_path(results_dir, result)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_item_results(results_dir: Path) -> list[ItemResult]:
    root = Path(results_dir)
    return [
        ItemResult.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(root.glob("*/*.json"))
    ]


def summarize(results: list[ItemResult]) -> dict:
    """Aggregate per bucket: metric mean/worst/p50/p95, pass rate, and per-item
    pass@1 (any seed passed) plus worst-seed verdict — per R7 stability rules,
    a single good demo run must never stand in for the distribution."""
    buckets: dict[str, dict] = {}
    for bucket_name in sorted({r.bucket for r in results}):
        bucket_results = [r for r in results if r.bucket == bucket_name]
        scored = [r for r in bucket_results if r.status == "scored"]
        metric_names = sorted({name for r in scored for name in r.scores})
        metrics = {}
        for name in metric_names:
            values = [r.scores[name] for r in scored if name in r.scores]
            direction = metric_direction(name)
            lower_is_better = direction == "lower_is_better"
            metrics[name] = {
                "direction": direction,
                "mean": round(statistics.fmean(values), 6),
                "worst": max(values) if lower_is_better else min(values),
                "best": min(values) if lower_is_better else max(values),
                "p50": round(_quantile(values, 0.50), 6),
                "p95": round(_quantile(values, 0.95), 6),
                "n": len(values),
            }
        items: dict[str, dict] = {}
        for item_id in sorted({r.item_id for r in bucket_results}):
            runs = [r for r in bucket_results if r.item_id == item_id]
            verdicts = [r.passed for r in runs if r.passed is not None]
            items[item_id] = {
                "seeds": sorted(r.seed for r in runs),
                "n_runs": len(runs),
                "pass_at_1": any(verdicts) if verdicts else None,
                "worst_seed_passed": all(verdicts) if verdicts else None,
            }
        verdicts = [r.passed for r in scored if r.passed is not None]
        wall_clocks = [r.usage.wall_clock_seconds for r in scored]
        buckets[bucket_name] = {
            "n_items": len(items),
            "n_runs": len(bucket_results),
            "n_scored": len(scored),
            "pass_rate": round(sum(verdicts) / len(verdicts), 6) if verdicts else None,
            "metrics": metrics,
            "wall_clock_p50": round(_quantile(wall_clocks, 0.50), 3) if wall_clocks else None,
            "wall_clock_p95": round(_quantile(wall_clocks, 0.95), 3) if wall_clocks else None,
            "total_cost_usd": round(
                sum(r.usage.estimated_cost_usd or 0.0 for r in bucket_results), 6
            ),
            "total_tokens": sum(r.usage.total_tokens for r in bucket_results),
            "items": items,
        }
    return {
        "checker_version": CHECKER_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "n_results": len(results),
        "buckets": buckets,
    }


def write_summary(results_dir: Path, summary: dict) -> Path:
    path = Path(results_dir) / "summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return float(ordered[index])
