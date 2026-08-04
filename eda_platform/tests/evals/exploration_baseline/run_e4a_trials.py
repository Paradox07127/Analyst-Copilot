"""E4a real-provider trial runner over the planted retail fixture.

Per (tier, seed): compose the production-shaped shadow exploration (worker
composition minus the release-certificate gate, which cannot exist before
these trials), write the E4a evidence-root artifacts (run spec + deterministic
checker), verify the root with the issuer, and freeze its manifest digest into
an issuer-owned trial plan. No production certificate is issued and
TRUSTED_E4A_RELEASE_PUBLIC_KEYS is never touched.

Usage (real provider; spends money — run only with explicit authorization):

  PYTHONPATH=eda_platform/src uv run python \
    eda_platform/tests/evals/exploration_baseline/run_e4a_trials.py \
    --adapter real --provider openai --model gpt-5.6-terra \
    --tier quick --seeds 1,2,3,4,5 \
    --results-dir output/e4a/gpt-5.6-terra \
    --env-file .env

Offline harness validation (no network, no spend):

  ... --adapter scripted --tier quick --seeds 1 --results-dir <scratch>

Idempotent: completed trial ids in the summary are skipped, so a partial
sweep can be re-invoked. A crashed run resumes from its journal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

if __package__ in (None, ""):  # running as a script
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eda_platform.agents.data_tools import (
    DataToolContext,
    _witness_entries,
    build_data_tools,
)
from eda_platform.agents.exploration.candidates import (
    CandidateSeed,
    DatasetExplorationProfile,
    mandatory_probe_seeds,
)
from eda_platform.agents.exploration.scheduler import (
    AdmissionContext,
    CandidateSignals,
)
from eda_platform.agents.exploration.supervisor import (
    CandidateBatch,
    PhaseContext,
    SupervisorPhase,
    phase_step_id,
)
from eda_platform.core.env import (
    load_llm_settings_from_env_file,
    load_provider_api_keys_from_env_file,
)
from eda_platform.core.exploration_journal import (
    JsonlExplorationJournal,
    sealed_policy,
)
from eda_platform.core.exploration_profiles import (
    build_exploration_policy,
    build_read_only_exploration_toolset,
    exploration_budget_profile,
)
from eda_platform.core.exploration_release_gate import E4aEvidenceBindings
from eda_platform.core.exploration_report import render_exploration_report
from eda_platform.core.llm import (
    LLMResultMetadata,
    LLMSettings,
    LLMToolCall,
    LLMToolResponse,
    LLMUsage,
    create_llm_client,
)
from eda_platform.core.provider_registry import LLMProvider, pricing_per_1m
from eda_platform.core.stat_registry import StatTestRegistry
from eda_platform.drivers.exploration import (
    CallableWitnessPort,
    JsonExplorationWorkflowStateStore,
    JsonlShadowBudgetStore,
    JsonSupervisorRecoveryStore,
    exploration_tool_capability_digest,
    run_composed_shadow_exploration,
    shadow_run_root,
)
from eda_platform.drivers.exploration_evidence_issuer import (
    E4A_CHECKER_VERSION_V2,
    E4aCheckerResult,
    E4aEvidenceIssuerBindings,
    E4aEvidenceRunSpec,
    E4aExpectedStructure,
    E4aGroundTruthFixture,
    _llm_usage,
    _recompute_checker,
    verify_e4a_evidence_root,
)
from eda_platform.schemas.artifacts import ArtifactType
from eda_platform.schemas.datasets import DatasetRecord
from eda_platform.schemas.hypotheses import HypothesisPredicate
from eda_platform.schemas.exploration_budget import ExplorationBudgetPolicy
from eda_platform.schemas.receipts import data_state_witness_digest
from eda_platform.tools.loader import LoadedDataset
from eda_platform.tools.profiler import profile_dataset
from eda_platform.tools.quality import scan_quality
from eda_platform.tools.sql_runner import build_catalog
from eda_platform.worker.exploration import (
    DatasetToolUsageMeter,
    _business_value,
    _candidate_expected_cost,
    _candidate_multiplicity_risk,
    _durable_admission_state,
    _remaining_cost_fraction,
    _scheduler_policy,
    _stat_attempt_counts,
)

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "planted"
DATASET_ID = "planted_retail"
ITEM_ID = "planted_retail_v1"
CHECKER_VERSION = E4A_CHECKER_VERSION_V2
EVIDENCE_KEY_ID = "e4a-trial-evidence-v1"
_HARNESS_CHECKER_NAME = "e4a-checker-result.json"
# Trial-evidence signing key: deterministic and non-production by design; the
# production release path uses a separately provisioned key.
EVIDENCE_SIGNING_KEY = hashlib.sha256(b"e4a-trial-evidence-signing-v1").digest()

_TARGET_BY_PROBE_KIND = {
    "region_difference": ("region_difference_recall", "run_stat_test"),
    "missingness_mechanism": ("missingness_mechanism_recall", "diagnose_missingness"),
    "spike_day": ("spike_day_recall", "analyze_time_series"),
}


@dataclass(frozen=True)
class PlantedBundle:
    dataset: LoadedDataset
    profiles: tuple[DatasetExplorationProfile, ...]
    fixture: E4aGroundTruthFixture


def load_planted_bundle() -> PlantedBundle:
    csv_path = FIXTURE_DIR / "planted_retail.csv"
    content_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    dataset = LoadedDataset(
        record=DatasetRecord(
            dataset_id=DATASET_ID,
            name=csv_path.name,
            path=csv_path,
            content_hash=content_hash,
        ),
        frame=pd.read_csv(csv_path),
    )
    profiles = (
        DatasetExplorationProfile(
            dataset_id=DATASET_ID,
            region_dimensions=("region",),
            metric_columns=("revenue",),
            missing_value_columns=("satisfaction",),
            missingness_group_dimensions=("channel",),
            datetime_columns=("order_date",),
            spike_metric_columns=("revenue",),
        ),
    )
    # Expected structures inherit the mandatory-probe predicates verbatim so
    # the fixture is achievable exactly by the deterministic coverage probes.
    by_kind = {
        seed.proposal.probe_kind: seed for seed in mandatory_probe_seeds(profiles)
    }
    missing = set(_TARGET_BY_PROBE_KIND) - set(by_kind)
    if missing:
        raise RuntimeError(f"planted profile produced no mandatory probe for {missing}")
    structures = tuple(
        E4aExpectedStructure(
            structure_id=kind,
            target_metric=metric,  # type: ignore[arg-type]
            tool_names=(tool_name,),
            required_columns=by_kind[kind].proposal.columns,
            predicate=by_kind[kind].proposal.predicate,
        )
        for kind, (metric, tool_name) in sorted(_TARGET_BY_PROBE_KIND.items())
    )
    # ground_truth.json documents planted_trend_revenue (~0.8%/day growth on
    # revenue over order_date). No mandatory probe carries a trend predicate,
    # so this structure is reachable only through the checker-v2 semantic pass.
    structures = (
        *structures,
        E4aExpectedStructure(
            structure_id="planted_trend_revenue",
            target_metric="trend_recall",
            tool_names=("analyze_time_series",),
            required_columns=("order_date", "revenue"),
            predicate=HypothesisPredicate(
                metric="revenue", operator="greater_than", right_operand="order_date"
            ),
            # "has_trend" is not in the HypothesisPredicate operator vocabulary;
            # spike/differs phrasings over these columns are the reachable ones.
            alternate_operators=("has_spike", "differs"),
            required_fact_values=(("trend_direction", "increasing"),),
        ),
    )
    fixture = E4aGroundTruthFixture(item_id=ITEM_ID, expected_structures=structures)
    return PlantedBundle(dataset=dataset, profiles=profiles, fixture=fixture)


def code_fingerprint() -> str:
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).resolve().parents[4],
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).resolve().parents[4],
    ).stdout.strip()
    return f"git-{head}{'-dirty' if dirty else ''}"


class ScriptedProvider:
    """Offline provider: concluded generate batches (mandatory seeds carry the
    round-0 coverage) and one deterministic tool call per probe conversation."""

    settings = LLMSettings(
        provider=LLMProvider.OPENAI,
        model="scripted-v0",
        max_tokens=100,
        usd_per_1k_prompt=0.001,
        usd_per_1k_completion=0.001,
    )

    def __init__(self) -> None:
        import threading

        self.calls = 0
        self._calls_lock = threading.Lock()
        self._local = threading.local()

    def _record(self) -> None:
        with self._calls_lock:
            self.calls += 1
        self._local.value = LLMResultMetadata(
            provider="scripted",
            model="scripted-v0",
            usage=LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            estimated_cost_usd=0.0,
            usage_reported=True,
        )

    def last_usage(self) -> LLMResultMetadata | None:
        return getattr(self._local, "value", None)

    def structured(self, *, task: str, schema: type, payload: dict) -> Any:
        del task, payload
        self._record()
        return schema.model_validate(
            {"concluded": True, "conclusion_reason": "Scripted harness validation."}
        )

    def tool_call(
        self,
        *,
        task: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMToolResponse:
        del task, tools
        self._record()
        if any(message.get("role") == "tool" for message in messages):
            return LLMToolResponse(content="Probe complete.", finish_reason="stop")
        prompt = " ".join(str(message.get("content", "")) for message in messages)
        arguments: dict[str, Any]
        if '"operator":"has_spike"' in prompt.replace(" ", ""):
            name = "analyze_time_series"
            arguments = {
                "dataset_id": DATASET_ID,
                "time_column": "order_date",
                "value_column": "revenue",
            }
        elif '"operator":"associated_with"' in prompt.replace(" ", ""):
            name = "diagnose_missingness"
            arguments = {
                "dataset_id": DATASET_ID,
                "target_column": "satisfaction",
                "group_columns": ["channel"],
            }
        else:
            name = "run_stat_test"
            arguments = {
                "dataset_id": DATASET_ID,
                "test_type": "independent_t_test",
                "group_column": "region",
                "value_column": "revenue",
            }
        return LLMToolResponse(
            tool_calls=[
                LLMToolCall(
                    call_id=f"scripted-call-{self.calls}",
                    name=name,
                    arguments=arguments,
                )
            ],
            finish_reason="tool_calls",
        )


def seed_profile_artifacts(context: DataToolContext) -> None:
    """Seed profile/quality artifacts the exploration toolset cannot produce.

    profile_dataset and scan_quality are excluded from the exploration toolset
    (they emit no receipts), yet run_domain_metrics and recommend_cleaning
    look their artifacts up on context.artifacts — in seed 6, 23 of 40 tool
    failures were these missing-precondition errors.
    """
    for loaded in context.datasets:
        profile = profile_dataset(
            loaded,
            project_id=context.project_id,
            session_id=context.session_id,
        )
        quality = scan_quality(
            profile,
            project_id=context.project_id,
            session_id=context.session_id,
        )
        context.add_artifact(profile)
        context.add_artifact(quality)
    loaded_ids = {dataset.record.dataset_id for dataset in context.datasets}
    profiled = {
        artifact.payload.get("dataset_id")
        for artifact in context.artifacts
        if artifact.type is ArtifactType.DATASET_PROFILE
    }
    scanned = {
        artifact.payload.get("dataset_id")
        for artifact in context.artifacts
        if artifact.type is ArtifactType.QUALITY_ISSUE_SET
    }
    if not (loaded_ids <= profiled and loaded_ids <= scanned):
        raise RuntimeError(
            "profile artifact seeding failed to satisfy the run_domain_metrics/"
            "recommend_cleaning preconditions"
        )


def trial_run_id(tier: str, seed: int, budget_scale: int) -> str:
    """Calibration runs carry their scale, so they can never occupy or be
    mistaken for the production-tier trial of the same tier/seed."""
    suffix = "" if budget_scale == 1 else f"-x{budget_scale}"
    return f"e4a-{tier}-seed{seed}{suffix}"


def scaled_budget(tier: str, scale: int) -> ExplorationBudgetPolicy:
    """Multiply the LLM and tool-call caps for a capability probe.

    A tier profile that strangles the loop measures the cap, not the agent, so
    calibration runs relax it and the production tiers get sized from what the
    model actually consumes. Every dimension scales together, keeping the
    protected reserves valid, and the scaled policy is what the run spec
    records — the evidence root stays internally consistent either way.
    """
    budget = exploration_budget_profile(tier)  # type: ignore[arg-type]
    if scale == 1:
        return budget
    llm = budget.llm
    scaled_llm = llm.model_copy(
        update={
            name: value * scale
            for name, value in (
                ("max_requests", llm.max_requests),
                ("max_input_tokens", llm.max_input_tokens),
                ("max_output_tokens", llm.max_output_tokens),
                ("max_total_tokens", llm.max_total_tokens),
                ("max_wall_seconds", llm.max_wall_seconds),
                ("protected_requests", llm.protected_requests),
                ("protected_total_tokens", llm.protected_total_tokens),
                ("max_cost_usd", llm.max_cost_usd),
                ("protected_cost_usd", llm.protected_cost_usd),
            )
            if value is not None
        }
    )
    return budget.model_copy(
        update={
            "llm": scaled_llm,
            "max_successful_tool_calls": budget.max_successful_tool_calls * scale,
            "max_tool_calls_by_kind": {
                kind: cap * scale
                for kind, cap in budget.max_tool_calls_by_kind.items()
            },
        }
    )


def ledger_usage_report(root: Path) -> list[str]:
    """Per-call output usage, so a calibration run answers 'how many tokens?'."""
    path = root / "llm-budget.jsonl"
    if not path.exists():
        return []
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        event = json.loads(raw)
        if event.get("event_type") != "llm_usage":
            continue
        summary = event.get("summary", {})
        lines.append(
            "    {task}/{kind}: completion={completion} reasoning={reasoning} "
            "prompt={prompt} finish={finish} status={status}".format(
                task=summary.get("task"),
                kind=summary.get("kind"),
                completion=summary.get("completion_tokens"),
                reasoning=summary.get("reasoning_tokens"),
                prompt=summary.get("prompt_tokens"),
                finish=summary.get("finish_reason"),
                status=summary.get("status"),
            )
        )
    return lines


def build_real_client(
    provider_name: str,
    model: str,
    env_file: Path,
    tier: str,
    max_tokens: int | None,
    budget_scale: int = 1,
) -> Any:
    """Build the client with a per-call output cap the tier budget can afford.

    Every call reserves ``max_tokens`` of output worst-case, so an inherited
    interactive default (.env ships 12000, exactly the quick tier's whole
    output budget) exhausts the run on its second call.
    """
    provider = LLMProvider(provider_name)
    settings = load_llm_settings_from_env_file(env_file)
    keys = load_provider_api_keys_from_env_file(env_file)
    if provider not in keys:
        raise SystemExit(f"no API key for provider {provider.value} in {env_file}")
    prices = pricing_per_1m(provider, model)
    if prices is None:
        raise SystemExit(f"no listed pricing for {provider.value}/{model}")
    llm_budget = scaled_budget(tier, budget_scale).llm
    if llm_budget.max_output_tokens is None or llm_budget.max_requests is None:
        raise SystemExit(f"tier {tier} has no output-token or request cap to size against")
    if max_tokens is None:
        max_tokens = llm_budget.max_output_tokens // llm_budget.max_requests
    settings = settings.model_copy(
        update={
            "provider": provider,
            "model": model,
            "api_key": keys[provider],
            "max_tokens": max_tokens,
            "usd_per_1k_prompt": prices[0] / 1000.0,
            "usd_per_1k_completion": prices[1] / 1000.0,
        }
    )
    return create_llm_client(settings)


def run_trial(
    *,
    bundle: PlantedBundle,
    tier: str,
    seed: int,
    provider_client: Any,
    workspace: Path,
    fingerprint: str,
    budget_scale: int = 1,
    require_issuance: bool = True,
    max_batch_size: int | None = None,
    explore_until_exhausted: bool = False,
    probe_concurrency: int = 1,
) -> dict[str, Any]:
    exploration_id = trial_run_id(tier, seed, budget_scale)
    run_root = shadow_run_root(workspace, exploration_id)
    stat_registry = StatTestRegistry(run_root / "stat_registry.jsonl")
    context = DataToolContext(
        datasets=[bundle.dataset],
        catalog=build_catalog([bundle.dataset]),
        project_id="e4a-trials",
        session_id=exploration_id,
        store=None,
        payload_policy="schema+aggregates",
        stat_registry=stat_registry,
    )
    registered = {tool.name: tool for tool in build_data_tools(context)}
    seed_profile_artifacts(context)
    # Receipts witness the seeded profile artifact ids (_witness_entries), so
    # the run witness must be derived from the context after seeding.
    run_witness = data_state_witness_digest(_witness_entries(context))
    tools = build_read_only_exploration_toolset(registered)
    policy = build_exploration_policy(
        tier=tier,  # type: ignore[arg-type]
        dataset_scope=(DATASET_ID,),
        tool_capability_digest=exploration_tool_capability_digest(tools),
    )
    if budget_scale != 1:
        policy = sealed_policy(
            policy.model_copy(
                update={
                    "budget": scaled_budget(tier, budget_scale),
                    "policy_fingerprint": "",
                }
            )
        )
    coverage_targets = frozenset(
        seed_.coverage_key for seed_ in mandatory_probe_seeds(bundle.profiles)
    )
    dataset_columns = {
        DATASET_ID: frozenset(str(column) for column in bundle.dataset.frame.columns)
    }
    supported_methods = frozenset(
        {*(tool.name for tool in tools), "compare_groups"}
    )
    workflow_store = JsonExplorationWorkflowStateStore(run_root / "workflow-state.json")

    def admission(phase_context: PhaseContext) -> AdmissionContext:
        workflow_state = workflow_store.load()
        durable = _durable_admission_state(workflow_state, policy)
        return AdmissionContext(
            dataset_columns=dataset_columns,
            allowed_dataset_ids=frozenset(policy.dataset_scope),
            supported_method_families=supported_methods,
            historical_hypothesis_fingerprints=durable[0],
            answered_hypothesis_fingerprints=durable[1],
            executed_query_fingerprints=durable[2],
            remaining_cost=_remaining_cost_fraction(
                phase_context, policy.budget.llm.max_cost_usd
            ),
            family_quota_remaining=durable[3],
            unexplored_coverage_keys=(
                coverage_targets - workflow_state.coverage_completed
            ),
        )

    def signals(
        _phase_context: PhaseContext, seeds: tuple[CandidateSeed, ...]
    ) -> dict[str, CandidateSignals]:
        stat_counts = _stat_attempt_counts(stat_registry)
        workflow_state = workflow_store.load()
        return {
            seed_.hypothesis_id: CandidateSignals(
                business_value=_business_value(seed_, policy.goal),
                information_gain_proxy=(
                    0.9
                    if seed_.coverage_key not in workflow_state.coverage_completed
                    else 0.35
                ),
                expected_cost=_candidate_expected_cost(
                    seed_, [bundle.dataset], policy.budget
                ),
                multiplicity_risk=_candidate_multiplicity_risk(seed_, stat_counts),
                query_fingerprint=seed_.hypothesis_fingerprint[:16],
            )
            for seed_ in seeds
        }

    scheduler_policy = _scheduler_policy(policy)
    if max_batch_size is not None:
        # Capability probe: let every admitted candidate run instead of the
        # tier's batch cap, which otherwise gives the mandatory probes first
        # refusal and starves the model's own proposals.
        scheduler_policy = scheduler_policy.model_copy(
            update={"max_batch_size": max_batch_size}
        )
    journal = JsonlExplorationJournal(run_root / "journal.jsonl")
    prior = journal.rebuild()
    if prior is not None and prior.status == "stopped":
        raise RuntimeError(
            f"{run_root} already holds a stopped run "
            f"(stop_reason={prior.stop_reason}); a stopped journal cannot be "
            "re-run. Move or delete that directory to retry this trial."
        )
    journal.initialize(
        exploration_id=exploration_id,
        policy=policy,
        code_fingerprint=fingerprint,
        data_state_witness=run_witness,
    )
    started = perf_counter()
    result = run_composed_shadow_exploration(
        workspace=workspace,
        exploration_id=exploration_id,
        policy=policy,
        code_fingerprint=fingerprint,
        data_state_witness=run_witness,
        provider=provider_client,
        tools=tools,
        dataset_profiles=bundle.profiles,
        scheduler_policy=scheduler_policy,
        admission_context=admission,
        signals=signals,
        witness=CallableWitnessPort(lambda expected: expected == run_witness),
        usage_meter=DatasetToolUsageMeter([bundle.dataset]),
        stat_attempt_counts=lambda: _stat_attempt_counts(stat_registry),
        probe_concurrency=probe_concurrency,
        coverage_target_met=(
            (lambda _state: False)
            if explore_until_exhausted
            else (
                lambda state: bool(coverage_targets)
                and coverage_targets.issubset(state.coverage_completed)
            )
        ),
        dataset_columns=dataset_columns,
        supported_method_families=supported_methods,
    )
    wall_seconds = round(perf_counter() - started, 3)
    if result.result.error:
        # The supervisor's error only lives on the in-memory result; without
        # this a failed run is diagnosed from the journal alone, which never
        # records why an unclassified exception ended it.
        print(f"    supervisor error: {result.result.error}", flush=True)
    if result.result.status != "stopped":
        raise RuntimeError(
            f"trial {exploration_id} ended {result.result.status}; rerun to resume"
        )

    root = result.journal_path.parent
    spec = E4aEvidenceRunSpec(
        item_id=ITEM_ID,
        seed=seed,
        policy=policy,
        coverage_targets=tuple(sorted(coverage_targets)),
        checker_version=CHECKER_VERSION,
        ground_truth_digest=bundle.fixture.digest,
    )
    (root / "e4a-run-spec.json").write_text(
        spec.model_dump_json(indent=2), encoding="utf-8"
    )
    _write_checker_artifact(root, bundle.fixture, exploration_id)
    _ensure_report_artifact(root, spec)

    bindings = E4aEvidenceIssuerBindings(
        certificate=E4aEvidenceBindings(
            checker_version=CHECKER_VERSION,
            code_fingerprint=fingerprint,
            tool_capability_digest=policy.tool_capability_digest,
            evidence_key_id=EVIDENCE_KEY_ID,
        ),
        fixture=bundle.fixture,
    )
    record: dict[str, Any] = {
        "trial_id": exploration_id,
        "item_id": ITEM_ID,
        "tier": tier,
        "seed": seed,
        "stop_reason": str(result.result.stop_reason),
        "wall_clock_seconds": wall_seconds,
        "budget_scale": budget_scale,
        "code_fingerprint": fingerprint,
        "root": str(root),
        "attribution": _origin_attribution(root, exploration_id),
    }
    try:
        trial, manifest = verify_e4a_evidence_root(
            root,
            issuer_bindings=bindings,
            evidence_signing_key=EVIDENCE_SIGNING_KEY,
        )
    except ValueError as exc:
        # Issuance is the release gate, not the measurement. A capability trial
        # still yields its scores, usage and timings from the durable run, so a
        # refused root is recorded rather than thrown away -- the same root can
        # be re-verified later once every tool has a result contract.
        if require_issuance:
            raise RuntimeError(
                f"evidence root rejected (stop_reason={result.result.stop_reason}, "
                f"rounds_settled={result.result.rounds_settled}): {exc}"
            ) from exc
        record.update(_unverified_projection(root, exploration_id))
        record["issued"] = False
        record["issuance_error"] = f"{type(exc).__name__}: {exc}"
        return record
    record.update(
        {
            "trial_id": trial.trial_id,
            "status": trial.status,
            "scores": trial.scores,
            "usage": trial.usage.model_dump(mode="json"),
            "manifest_digest": manifest.root_digest,
            "provider": trial.provider,
            "model": trial.model,
            "issued": True,
        }
    )
    return record


def _origin_attribution(root: Path, exploration_id: str) -> dict[str, Any]:
    """Who earned each insight: the system's mandatory probes, or the model?

    The checker's recall can only ever credit mandatory probes (it matches on
    predicate identity with the fixture, whose predicates were copied from
    mandatory_probe_seeds), so this is the only place the model's own
    contribution is visible at all.
    """
    recovery = JsonSupervisorRecoveryStore(root / "phase-responses")
    terminal = JsonlExplorationJournal(root / "journal.jsonl").rebuild()
    origins: dict[str, str] = {}
    for round_index in range(0 if terminal is None else terminal.rounds_started):
        step_id = phase_step_id(exploration_id, round_index, SupervisorPhase.GENERATE)
        try:
            batch = recovery.load_required(step_id)
        except KeyError:
            continue
        if not isinstance(batch, CandidateBatch):
            continue
        for seed_ in batch.candidates:
            if isinstance(seed_, CandidateSeed):
                origins.setdefault(seed_.hypothesis_id, seed_.origin)
    workflow = JsonExplorationWorkflowStateStore(root / "workflow-state.json").load()
    proposed = Counter(origins.values())
    chosen = Counter(
        origins.get(decision.hypothesis_id, "unknown")
        for decision in workflow.decisions
        if decision.chosen
    )
    insighted = Counter(
        origins.get(insight.hypothesis_id, "unknown")
        for insight in workflow.insights.values()
    )
    return {
        "proposed": dict(proposed),
        "chosen": dict(chosen),
        "insights": dict(insighted),
    }


def _unverified_projection(root: Path, exploration_id: str) -> dict[str, Any]:
    """Scores and usage read straight from the run's own durable artifacts.

    Used when issuance is skipped or refused: the checker artifact was written
    by the same deterministic recomputation the issuer would run, and usage
    comes from the spend ledger, so the measurement is unaffected -- only its
    anti-forgery attestation is missing.
    """
    checker = E4aCheckerResult.model_validate_json(
        (root / "e4a-checker-result.json").read_text(encoding="utf-8")
    )
    ledger = JsonlShadowBudgetStore(root / "llm-budget.jsonl").events()
    provider, model, llm_requests, total_tokens, cost = _llm_usage(ledger)
    terminal = JsonlExplorationJournal(root / "journal.jsonl").rebuild()
    return {
        "status": "unverified",
        "scores": checker.scores,
        "usage": {
            "llm_requests": llm_requests,
            "total_tokens": total_tokens,
            "estimated_cost_usd": cost,
            "tool_calls": 0 if terminal is None else terminal.tool_calls_committed,
            "rows_scanned": 0 if terminal is None else terminal.rows_scanned,
            "cells_scanned": 0 if terminal is None else terminal.result_cells,
        },
        "provider": provider,
        "model": model,
    }


def _recovered_candidates(root: Path, exploration_id: str, rounds_started: int) -> dict[str, CandidateSeed]:
    recovery = JsonSupervisorRecoveryStore(root / "phase-responses")
    candidates: dict[str, CandidateSeed] = {}
    for round_index in range(rounds_started):
        step_id = phase_step_id(exploration_id, round_index, SupervisorPhase.GENERATE)
        batch = recovery.load_required(step_id)
        if not isinstance(batch, CandidateBatch):
            raise RuntimeError("recovered candidate batch has the wrong type")
        for seed_ in batch.candidates:
            if isinstance(seed_, CandidateSeed):
                candidates[seed_.hypothesis_id] = seed_
    return candidates


def _write_checker_artifact(
    root: Path, fixture: E4aGroundTruthFixture, exploration_id: str
) -> None:
    journal = JsonlExplorationJournal(root / "journal.jsonl")
    events = journal.events()
    terminal = journal.rebuild()
    if terminal is None or terminal.status != "stopped":
        raise RuntimeError("checker artifact requires a stopped journal")
    workflow = JsonExplorationWorkflowStateStore(root / "workflow-state.json").load()
    candidates = _recovered_candidates(root, exploration_id, terminal.rounds_started)
    checker = _recompute_checker(
        workflow=workflow,
        events=events,
        terminal=terminal,
        candidates=candidates,
        stop_reason=terminal.stop_reason,
        fixture=fixture,
        checker_version=CHECKER_VERSION,
    )
    (root / "e4a-checker-result.json").write_text(
        checker.model_dump_json(indent=2), encoding="utf-8"
    )


def rescore_run_root(run_root: Path) -> E4aCheckerResult:
    """Offline checker-v2 recomputation over an existing run root.

    No issuer verification and no network: candidates are rebuilt exactly as
    the normal scoring path does, and the result lands next to (never over)
    the original checker artifact.
    """
    run_root = run_root.resolve()
    bundle = load_planted_bundle()
    journal = JsonlExplorationJournal(run_root / "journal.jsonl")
    events = journal.events()
    terminal = journal.rebuild()
    if terminal is None or terminal.status != "stopped":
        raise SystemExit(f"{run_root} has no stopped journal to rescore")
    workflow = JsonExplorationWorkflowStateStore(run_root / "workflow-state.json").load()
    candidates = _recovered_candidates(
        run_root, terminal.exploration_id, terminal.rounds_started
    )
    checker = _recompute_checker(
        workflow=workflow,
        events=events,
        terminal=terminal,
        candidates=candidates,
        stop_reason=terminal.stop_reason,
        fixture=bundle.fixture,
        checker_version=CHECKER_VERSION,
    )
    (run_root / "e4a-checker-result-v2.json").write_text(
        checker.model_dump_json(indent=2), encoding="utf-8"
    )
    return checker


def _ensure_report_artifact(root: Path, spec: E4aEvidenceRunSpec) -> None:
    """Write the deterministic report when the run ended without one.

    A graceful stop from the empty-frontier path skips the finalizer, so the
    root lacks report.md (known product gap; the issuer recipe is reproduced
    here byte-for-byte and re-verified during root verification).
    """
    report_path = root / "report.md"
    if report_path.exists():
        return
    journal = JsonlExplorationJournal(root / "journal.jsonl")
    terminal = journal.rebuild()
    if terminal is None or terminal.status != "stopped":
        raise RuntimeError("report artifact requires a stopped journal")
    workflow = JsonExplorationWorkflowStateStore(root / "workflow-state.json").load()
    effective_budget = spec.policy.budget
    ledger_events = JsonlShadowBudgetStore(root / "llm-budget.jsonl").events()
    _provider, _model, llm_requests, total_tokens, cost = _llm_usage(ledger_events)
    budget_summary: dict[str, object] = {
        "llm_requests_used": llm_requests,
        "llm_total_tokens_used": total_tokens,
        "llm_cost_usd_used": cost,
        "successful_tool_calls": terminal.tool_calls_committed,
        "rows_scanned": terminal.rows_scanned,
        "result_cells": terminal.result_cells,
        "rounds_started": terminal.rounds_started,
        "max_llm_requests": effective_budget.llm.max_requests,
        "max_cost_usd": (
            None
            if effective_budget.llm.max_cost_usd is None
            else float(effective_budget.llm.max_cost_usd)
        ),
        "max_successful_tool_calls": effective_budget.max_successful_tool_calls,
        "max_rounds": effective_budget.max_rounds,
    }
    rendered = render_exploration_report(
        workflow,
        run_metadata={
            "exploration_id": terminal.exploration_id,
            "policy_fingerprint": terminal.effective_policy_fingerprint,
            "witness": terminal.data_state_witness,
        },
        coverage_targets=spec.coverage_targets,
        budget_summary=budget_summary,
        stop_reason=str(terminal.stop_reason),
    )
    report_path.write_text(rendered.markdown, encoding="utf-8")


def _load_summary(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _freeze_manifest(plan_path: Path, record: dict[str, Any]) -> None:
    if not record.get("issued"):
        # A refused root has no manifest digest to pin; it can be re-verified
        # and frozen later.
        return
    if record.get("budget_scale", 1) != 1:
        # The trial plan is the anti-cherry-pick admission device for release.
        # A run on a scaled (non-production) budget is a capability probe, never
        # a release candidate, so it stays out of the plan by construction.
        print(
            f"note {record['trial_id']}: budget_scale="
            f"{record['budget_scale']}, not frozen into the trial plan",
            flush=True,
        )
        return
    plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.exists() else []
    for entry in plan:
        if entry["trial_id"] == record["trial_id"]:
            if entry["manifest_digest"] != record["manifest_digest"]:
                raise RuntimeError(
                    f"trial {record['trial_id']} manifest changed after freezing"
                )
            return
    plan.append(
        {
            "role": "treatment",
            "trial_id": record["trial_id"],
            "tier": record["tier"],
            "seed": record["seed"],
            "manifest_digest": record["manifest_digest"],
        }
    )
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", choices=("real", "scripted"), default=None)
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--tier", choices=("quick", "standard", "deep"), default=None)
    parser.add_argument("--seeds", default=None, help="comma-separated, e.g. 1,2,3")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument(
        "--rescore",
        type=Path,
        default=None,
        help=(
            "recompute the v2 checker offline over an existing run root and "
            "write e4a-checker-result-v2.json into it; no issuer verify, no "
            "network, and no other flags required"
        ),
    )
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help=(
            "per-call output cap; defaults to max_output_tokens // max_requests. "
            "Reasoning models spend this budget on reasoning tokens, so raise it "
            "for a calibration run and read the printed usage."
        ),
    )
    parser.add_argument(
        "--require-issuance",
        action="store_true",
        help=(
            "fail a trial whose evidence root the issuer refuses. Off by "
            "default: issuance is the release gate, not the measurement, and a "
            "refused root still yields scores, usage and timings."
        ),
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=None,
        help=(
            "override the tier's per-round batch cap. The tier default gives "
            "mandatory probes first refusal, so the model's own proposals only "
            "get the leftover slots; raise this to run every admitted candidate."
        ),
    )
    parser.add_argument(
        "--explore-until-exhausted",
        action="store_true",
        help=(
            "do not stop when the mandatory coverage checklist completes. The "
            "default criterion ends the run the moment the deterministic probes "
            "finish, which can be round 0 -- the model never gets a second round."
        ),
    )
    parser.add_argument(
        "--probe-concurrency",
        type=int,
        default=1,
        help=(
            "run a round's chosen probe sessions in this many worker threads "
            "(speedup plan P4). 1 keeps the serial loop; raise only after "
            "checking the provider's account-level rate limits."
        ),
    )
    parser.add_argument(
        "--budget-scale",
        type=int,
        default=1,
        help=(
            "multiply the tier's LLM and tool-call caps for a capability probe. "
            "Recorded in the summary; the production tiers are unchanged."
        ),
    )
    args = parser.parse_args()
    if args.rescore is not None:
        checker = rescore_run_root(args.rescore)
        print(json.dumps(checker.scores, indent=2, sort_keys=True))
        return 0
    for required in ("adapter", "tier", "seeds", "results_dir"):
        if getattr(args, required) is None:
            raise SystemExit(
                f"--{required.replace('_', '-')} is required unless --rescore is given"
            )
    if args.budget_scale < 1:
        raise SystemExit("--budget-scale must be at least 1")

    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    workspace = results_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "trials-summary.jsonl"
    plan_path = results_dir / "trial-plan.json"

    bundle = load_planted_bundle()
    fingerprint = code_fingerprint()
    done = {row["trial_id"] for row in _load_summary(summary_path)}

    if args.adapter == "real":
        if args.env_file is None:
            raise SystemExit("--env-file is required for the real adapter")
        client_factory = lambda: build_real_client(  # noqa: E731
            args.provider,
            args.model,
            args.env_file,
            args.tier,
            args.max_tokens,
            args.budget_scale,
        )
    else:
        client_factory = ScriptedProvider

    failures = 0
    for seed in seeds:
        exploration_id = trial_run_id(args.tier, seed, args.budget_scale)
        if exploration_id in done:
            print(f"skip {exploration_id}: already recorded")
            continue
        print(f"run  {exploration_id} adapter={args.adapter} ...", flush=True)
        try:
            record = run_trial(
                bundle=bundle,
                tier=args.tier,
                seed=seed,
                provider_client=client_factory(),
                workspace=workspace,
                fingerprint=fingerprint,
                budget_scale=args.budget_scale,
                require_issuance=args.require_issuance,
                max_batch_size=args.max_batch_size,
                explore_until_exhausted=args.explore_until_exhausted,
                probe_concurrency=args.probe_concurrency,
            )
        except Exception as exc:  # noqa: BLE001 - one failure must not kill the sweep
            failures += 1
            traceback.print_exc()
            print(f"FAIL {exploration_id}: {type(exc).__name__}: {exc}", flush=True)
            for line in ledger_usage_report(shadow_run_root(workspace, exploration_id)):
                print(line, flush=True)
            continue
        record["adapter"] = args.adapter
        with summary_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        _freeze_manifest(plan_path, record)
        print(
            f"done {exploration_id}: stop={record['stop_reason']} "
            f"recall={record['scores'].get('recall')} "
            f"auc={record['scores'].get('auc_over_steps')} "
            f"first={record['scores'].get('first_improvement_step')} "
            f"cost=${record['usage'].get('estimated_cost_usd')} "
            f"wall={record['wall_clock_seconds']}s "
            f"issued={record.get('issued')} "
            f"insights={record.get('attribution', {}).get('insights')}",
            flush=True,
        )
        for line in ledger_usage_report(Path(record["root"])):
            print(line, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
