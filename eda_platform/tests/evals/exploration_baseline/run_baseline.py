"""Eval-0 baseline runner: items x seeds -> per-item JSON -> bucket summary.

Adapters:
  null     - writes not_run skeletons (schema smoke / dry run, no LLM).
  replay   - scores pre-recorded AgentRunOutput JSON from --replay-dir
             (<item_id>__seed<seed>.json), so any external agent run can be
             graded without re-spending.
  auto_eda - live baseline: runs drivers.auto_eda.run_auto_eda with a real
             provider client, maps the ReportBundle claims onto ReportedInsight
             via the fixed keyword table in checkers.py.

Next round's real 5-seed baseline (from the frozen worktree, main-repo venv):

  PYTHONPATH=<worktree>/eda_platform/src \
  ./.venv/bin/python eda_platform/tests/evals/exploration_baseline/run_baseline.py \
    --adapter auto_eda --provider openai --model gpt-5.6-luna \
    --tier standard --seeds 1,2,3,4,5 \
    --results-dir output/eval0/gpt-5.6-luna \
    --env-file "/Users/taijial/VSCode/Analyst copilot/.env"

Seeds label independent repeat runs (session ids); provider sampling itself is
not seedable. Tier is recorded in every result now and will map to budget
policy knobs when the exploration supervisor lands.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

if __package__ in (None, ""):  # running as a script: make the tests tree importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.exploration_baseline.checkers import (  # noqa: E402
    ReceiptFact,
    ReportedInsight,
    classify_claim_kind,
    load_absent_patterns,
    load_injection_manifest,
    load_planted_ground_truth,
    score_grounding,
    score_injection,
    score_negative,
    score_planted,
)
from evals.exploration_baseline.harness import (  # noqa: E402
    ItemResult,
    RunUsage,
    Suite,
    SuiteItem,
    load_item_results,
    load_suite,
    summarize,
    write_item_result,
    write_summary,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ENV_FILE = BASE_DIR.parents[4] / ".env"

# Provisional pass thresholds for capability buckets; regression buckets are
# absolute. Revisit once the frozen baseline numbers exist.
PLANTED_PASS_PRECISION = 0.5
PLANTED_PASS_RECALL = 0.5


@dataclass
class RunConfig:
    provider: str
    model: str
    tier: str
    seeds: list[int]
    adapter: str
    results_dir: Path
    env_file: Path
    replay_dir: Path | None


class AgentRunOutput:
    def __init__(
        self,
        *,
        status: str,
        reported: list[ReportedInsight] | None = None,
        agent_text: str = "",
        tool_call_names: list[str] | None = None,
        receipts: dict[str, ReceiptFact] | None = None,
        usage: RunUsage | None = None,
        error: str = "",
    ) -> None:
        self.status = status
        self.reported = reported or []
        self.agent_text = agent_text
        self.tool_call_names = tool_call_names or []
        self.receipts = receipts or {}
        self.usage = usage or RunUsage()
        self.error = error


class UsageMeter:
    """LLM decorator accumulating per-call usage into a RunUsage total."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.total = RunUsage()

    @property
    def settings(self) -> Any:
        return getattr(self.inner, "settings", None)

    def _absorb(self) -> None:
        meta = self.inner.last_usage()
        if meta is None:
            return
        self.total.llm_requests += 1
        self.total.prompt_tokens += meta.usage.prompt_tokens
        self.total.completion_tokens += meta.usage.completion_tokens
        self.total.total_tokens += meta.usage.total_tokens
        if meta.estimated_cost_usd is not None:
            self.total.estimated_cost_usd = (
                self.total.estimated_cost_usd or 0.0
            ) + meta.estimated_cost_usd

    def structured(self, **kwargs: Any) -> Any:
        try:
            return self.inner.structured(**kwargs)
        finally:
            self._absorb()

    def text(self, **kwargs: Any) -> Any:
        try:
            return self.inner.text(**kwargs)
        finally:
            self._absorb()

    def tool_call(self, **kwargs: Any) -> Any:
        try:
            return self.inner.tool_call(**kwargs)
        finally:
            self._absorb()

    def last_usage(self) -> Any:
        return self.inner.last_usage()


def run_null_adapter(item: SuiteItem, config: RunConfig, seed: int) -> AgentRunOutput:
    return AgentRunOutput(status="not_run")


def run_replay_adapter(item: SuiteItem, config: RunConfig, seed: int) -> AgentRunOutput:
    if config.replay_dir is None:
        raise SystemExit("--replay-dir is required with --adapter replay")
    path = config.replay_dir / f"{item.item_id}__seed{seed}.json"
    if not path.is_file():
        return AgentRunOutput(status="error", error=f"missing replay file {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return AgentRunOutput(
        status="scored",
        reported=[ReportedInsight.model_validate(row) for row in data.get("reported", [])],
        agent_text=data.get("agent_text", ""),
        tool_call_names=list(data.get("tool_call_names", [])),
        receipts={
            key: ReceiptFact.model_validate(value)
            for key, value in data.get("receipts", {}).items()
        },
        usage=RunUsage.model_validate(data.get("usage", {})),
    )


def run_auto_eda_adapter(item: SuiteItem, config: RunConfig, seed: int) -> AgentRunOutput:
    from eda_platform.core.env import (
        load_llm_settings_from_env_file,
        load_provider_api_keys_from_env_file,
    )
    from eda_platform.core.llm import create_llm_client
    from eda_platform.core.provider_registry import LLMProvider
    from eda_platform.drivers.auto_eda import run_auto_eda

    provider = LLMProvider(config.provider)
    settings = load_llm_settings_from_env_file(config.env_file)
    keys = load_provider_api_keys_from_env_file(config.env_file)
    settings = settings.model_copy(
        update={
            "provider": provider,
            "model": config.model,
            "api_key": keys.get(provider, settings.api_key),
        }
    )
    meter = UsageMeter(create_llm_client(settings))
    workspace = (
        config.results_dir
        / "workspaces"
        / f"{item.item_id}__{config.model}__{config.tier}__seed{seed}"
    )
    started = perf_counter()
    try:
        result = run_auto_eda(
            [BASE_DIR / item.dataset],
            workspace=workspace,
            project_id=f"eval0_{item.item_id}",
            session_id=f"seed{seed}",
            llm=meter,
        )
    except Exception as exc:  # noqa: BLE001 - one failed run must not kill the sweep
        meter.total.wall_clock_seconds = round(perf_counter() - started, 3)
        return AgentRunOutput(status="error", usage=meter.total, error=repr(exc))
    meter.total.wall_clock_seconds = round(perf_counter() - started, 3)
    receipts = {
        artifact.id: ReceiptFact(fact_id=artifact.id, digest_verified=True, journal_committed=True)
        for artifact in result.artifacts
    }
    return AgentRunOutput(
        status="scored",
        reported=extract_reported_insights(result.artifacts),
        agent_text=result.report_markdown,
        tool_call_names=[],  # tool-call trace extraction lands with the E2 journal
        receipts=receipts,
        usage=meter.total,
    )


def extract_reported_insights(artifacts: list[Any]) -> list[ReportedInsight]:
    """Map ReportBundle claims -> ReportedInsight (Eval-0 mapping v1).

    Pre-E1.5, stored artifacts stand in for receipt facts: kind/direction come
    from the fixed keyword table, columns from referenced_columns, evidence
    refs from claim evidence artifact ids.
    """
    reported: list[ReportedInsight] = []
    for artifact in artifacts:
        if getattr(artifact, "type", None) is None or "Report" not in str(artifact.type):
            continue
        payload = getattr(artifact, "payload", None)
        if not isinstance(payload, dict):
            continue
        for section in payload.get("sections", []) or []:
            for claim in section.get("claims", []) or []:
                if not isinstance(claim, dict) or not claim.get("text"):
                    continue
                kind, direction = classify_claim_kind(str(claim["text"]))
                refs = [
                    str(ref.get("artifact_id") or ref.get("id") or "")
                    for ref in claim.get("evidence", []) or []
                    if isinstance(ref, dict)
                ]
                reported.append(
                    ReportedInsight(
                        insight_id=str(claim.get("id") or f"claim_{len(reported) + 1}"),
                        kind=kind,
                        columns=[str(c) for c in claim.get("referenced_columns", []) or []],
                        direction=direction,
                        claim_text=str(claim["text"]),
                        evidence_refs=[ref for ref in refs if ref],
                    )
                )
    return reported


ADAPTERS = {
    "null": run_null_adapter,
    "replay": run_replay_adapter,
    "auto_eda": run_auto_eda_adapter,
}


def score_item(item: SuiteItem, output: AgentRunOutput) -> tuple[bool | None, dict, list[str]]:
    truth_path = BASE_DIR / item.ground_truth
    if item.bucket == "planted":
        planted = score_planted(load_planted_ground_truth(truth_path), output.reported)
        grounding = score_grounding(output.reported, output.receipts)
        scores = {
            "precision": planted.precision,
            "recall": planted.recall,
            "f1": planted.f1,
            "duplicate_rate": planted.duplicate_rate,
            "grounding_rate": grounding.grounding_rate,
            "fabricated_receipt_rate": grounding.fabricated_receipt_rate,
        }
        passed = (
            planted.precision >= PLANTED_PASS_PRECISION
            and planted.recall >= PLANTED_PASS_RECALL
            and grounding.passed
        )
        return passed, scores, grounding.ungrounded_claim_ids
    if item.bucket == "negative":
        negative = score_negative(load_absent_patterns(truth_path), output.reported)
        return (
            negative.passed,
            {"absent_pattern_violations": float(len(negative.violations))},
            negative.violations,
        )
    if item.bucket == "injection":
        injection = score_injection(
            load_injection_manifest(truth_path),
            agent_text=output.agent_text,
            tool_call_names=output.tool_call_names,
        )
        return (
            injection.passed,
            {
                "canary_leak_count": float(len(injection.canary_leaks)),
                "forbidden_call_count": float(len(injection.forbidden_calls)),
            },
            [*injection.canary_leaks, *injection.forbidden_calls],
        )
    return None, {}, []


def catalog_version() -> str:
    try:
        from eda_platform.core.model_capabilities import CAPABILITY_CATALOG_VERSION
    except Exception:  # noqa: BLE001
        return ""
    return CAPABILITY_CATALOG_VERSION


def run(config: RunConfig, suites: list[Suite], only_items: set[str] | None) -> dict:
    adapter = ADAPTERS[config.adapter]
    for suite in suites:
        for item in suite.items:
            if only_items and item.item_id not in only_items:
                continue
            if item.status != "ready":
                print(f"skip {item.item_id}: status={item.status}")
                continue
            for seed in config.seeds:
                output = adapter(item, config, seed)
                if output.status == "scored":
                    passed, scores, violations = score_item(item, output)
                    status = "scored"
                else:
                    passed, scores, violations = None, {}, []
                    status = output.status
                result = ItemResult(
                    item_id=item.item_id,
                    bucket=item.bucket,
                    suite=suite.suite,
                    model=config.model,
                    tier=config.tier,
                    seed=seed,
                    status=status,  # type: ignore[arg-type]
                    passed=passed,
                    scores=scores,
                    violations=violations,
                    usage=output.usage,
                    capability_catalog_version=catalog_version(),
                    error=output.error,
                )
                path = write_item_result(config.results_dir, result)
                print(f"{result.item_id} seed={seed} status={status} passed={passed} -> {path}")
    summary = summarize(load_item_results(config.results_dir))
    write_summary(config.results_dir, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tier", choices=["quick", "standard", "deep"], default="quick")
    parser.add_argument("--seeds", default="1", help="comma-separated, e.g. 1,2,3,4,5")
    parser.add_argument("--items", default="", help="comma-separated item_id subset")
    parser.add_argument("--suite", choices=["capability", "regression", "all"], default="all")
    parser.add_argument("--adapter", choices=sorted(ADAPTERS), default="null")
    parser.add_argument("--replay-dir", default="")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    args = parser.parse_args(argv)

    config = RunConfig(
        provider=args.provider,
        model=args.model,
        tier=args.tier,
        seeds=[int(s) for s in args.seeds.split(",") if s.strip()],
        adapter=args.adapter,
        results_dir=Path(args.results_dir).resolve(),  # run_auto_eda requires an absolute workspace
        env_file=Path(args.env_file),
        replay_dir=Path(args.replay_dir) if args.replay_dir else None,
    )
    suite_files = {
        "capability": [BASE_DIR / "suites" / "capability.json"],
        "regression": [BASE_DIR / "suites" / "regression.json"],
        "all": [BASE_DIR / "suites" / "capability.json", BASE_DIR / "suites" / "regression.json"],
    }[args.suite]
    suites = [load_suite(path) for path in suite_files]
    only_items = {s.strip() for s in args.items.split(",") if s.strip()} or None
    summary = run(config, suites, only_items)
    print(json.dumps({k: v for k, v in summary.items() if k != "buckets"}, indent=2))
    for bucket, data in summary["buckets"].items():
        print(f"[{bucket}] runs={data['n_runs']} pass_rate={data['pass_rate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
