"""Production E4b adapter for the certified E4a composition root."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Callable, Mapping
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from eda_platform.agents.data_tools import DataToolContext, build_data_tools
from eda_platform.agents.exploration.candidates import (
    CandidateSeed,
    DatasetExplorationProfile,
    mandatory_probe_seeds,
)
from eda_platform.agents.exploration.executor import ToolUsageMeter
from eda_platform.agents.exploration.scheduler import (
    AdmissionContext,
    CandidateSignals,
    PriorityWeights,
    SchedulerPolicy,
    family_quotas_for_level,
)
from eda_platform.agents.exploration.supervisor import (
    CandidateBatch,
    PhaseContext,
    SupervisorPhase,
    phase_step_id,
)
from eda_platform.agents.exploration.workflow import (
    ColumnFact,
    DatasetFacts,
    ExplorationWorkflowState,
)
from eda_platform.agents.runtime import AgentTool, AgentToolResult
from eda_platform.application.services.approval_service import ApprovalService
from eda_platform.application.services.exploration_service import (
    APPROVAL_KIND_EXPLORATION_START,
    ExplorationRunMetadata,
    assert_budget_covered_by_certificate,
    assert_certificate_matches_runtime,
    assert_policy_covered_by_certificate,
    assert_policy_matches_runtime,
    resolve_configured_release_trust,
    resolve_exploration_source_snapshot,
)
from eda_platform.core.exploration_budget import ToolCallProjection, apply_budget_increase
from eda_platform.core.exploration_journal import (
    JsonlExplorationJournal,
    assert_policy_sealed,
)
from eda_platform.core.exploration_profiles import build_read_only_exploration_toolset
from eda_platform.core.exploration_shadow_store import (
    shadow_run_root,
    validate_shadow_run_path,
)
from eda_platform.core.llm import LLMClient, LLMToolCall
from eda_platform.core.session_loader import load_run
from eda_platform.core.stat_registry import StatTestRegistry, derive_family_id
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.exploration import (
    CallableWitnessPort,
    JsonExplorationWorkflowStateStore,
    JsonSupervisorRecoveryStore,
    exploration_tool_capability_digest,
    run_composed_shadow_exploration,
)
from eda_platform.schemas.artifacts import (
    Artifact,
    ArtifactType,
    ColumnProfile,
    DatasetProfile,
)
from eda_platform.schemas.exploration import (
    BudgetAmendedEvent,
    ExplorationPolicy,
    InsightFamily,
)
from eda_platform.schemas.exploration_budget import ExplorationBudgetPolicy
from eda_platform.schemas.receipts import load_verified_receipt
from eda_platform.tools.evidence import PayloadPolicy
from eda_platform.tools.loader import LoadedDataset
from eda_platform.tools.sql_runner import build_catalog

CancelCheck = Callable[[], object]
_FAILED_RESULT_CELL_PROJECTION = 10_000


class ExplorationWorkerParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_session_id: str
    exploration_id: str
    policy: ExplorationPolicy
    data_state_witness: str
    code_fingerprint: str
    release_certificate_digest: str
    provider: str
    payload_policy: PayloadPolicy
    operation: str


class DatasetToolUsageMeter(ToolUsageMeter):
    """Charge selected data scans and receipt-backed result materialization.

    Scan counts are deliberately conservative because pandas/DuckDB adapters do
    not expose physical row counters. A failed call keeps the full projection;
    a successful call may lower only result cells using its verified receipt.
    """

    def __init__(self, datasets: list[LoadedDataset]) -> None:
        self._sizes = {
            dataset.record.dataset_id: (
                len(dataset.frame),
                len(dataset.frame.columns),
            )
            for dataset in datasets
        }

    def project(
        self,
        *,
        call: LLMToolCall,
        tool: AgentTool,
        arguments: BaseModel,
    ) -> ToolCallProjection:
        del call
        payload = arguments.model_dump(mode="json")
        selected = self._selected_ids(payload)
        rows = sum(self._sizes[dataset_id][0] for dataset_id in selected)
        total_cells = sum(
            self._sizes[dataset_id][0] * self._sizes[dataset_id][1]
            for dataset_id in selected
        )
        return ToolCallProjection(
            kind=tool.name,
            rows_scanned=rows,
            result_cells=min(
                max(1, total_cells), _FAILED_RESULT_CELL_PROJECTION
            ),
        )

    def success(
        self,
        *,
        call: LLMToolCall,
        tool: AgentTool,
        arguments: BaseModel,
        result: AgentToolResult,
        projected: ToolCallProjection,
    ) -> ToolCallProjection:
        del call, tool, arguments
        artifact = result.receipt_artifact
        if not isinstance(artifact, Artifact):
            raise ValueError("successful exploration tools require a typed receipt artifact")
        receipt = load_verified_receipt(artifact.payload)
        cells = max(
            1,
            receipt.result_count,
            len(receipt.facts) + len(receipt.derivations),
        )
        return ToolCallProjection(
            kind=projected.kind,
            rows_scanned=projected.rows_scanned,
            result_cells=cells,
        )

    def failure(
        self,
        *,
        call: LLMToolCall,
        tool: AgentTool,
        arguments: BaseModel,
        error: BaseException,
        projected: ToolCallProjection,
    ) -> ToolCallProjection:
        del call, tool, arguments, error
        return projected

    def _selected_ids(self, payload: dict[str, Any]) -> tuple[str, ...]:
        candidates: list[str] = []
        for key in ("dataset_id", "left_dataset_id", "right_dataset_id"):
            value = payload.get(key)
            if isinstance(value, str):
                candidates.append(value)
        many = payload.get("target_dataset_ids")
        if isinstance(many, list):
            candidates.extend(str(item) for item in many)
        selected = tuple(dict.fromkeys(candidates)) or tuple(self._sizes)
        unknown = [dataset_id for dataset_id in selected if dataset_id not in self._sizes]
        if unknown:
            raise ValueError(f"tool usage scope contains unknown dataset: {unknown[0]}")
        return selected


def run_exploration_worker(
    store: ArtifactStore,
    workspace: Path | str,
    job: dict[str, Any],
    raw_params: dict[str, Any],
    *,
    llm: LLMClient,
    cancel_check: CancelCheck | None = None,
) -> None:
    """Revalidate every frozen identity and execute only the certified tool set."""
    _checkpoint(cancel_check)
    params = ExplorationWorkerParams.model_validate(raw_params)
    if params.operation not in {"start", "resume"}:
        raise ValueError("exploration worker operation must be start or resume")
    assert_policy_sealed(params.policy)

    release_trust = resolve_configured_release_trust()
    certificate = release_trust.certificate
    if certificate is None:
        raise RuntimeError("the E4a production release certificate is unavailable")
    assert_certificate_matches_runtime(certificate, release_trust.runtime_identity)
    if certificate.certificate_digest != params.release_certificate_digest:
        raise RuntimeError("exploration job release certificate digest changed")
    if params.provider.casefold() not in {
        provider.casefold() for provider in certificate.providers
    }:
        raise RuntimeError("exploration provider is not certified")
    if certificate.bindings.code_fingerprint != params.code_fingerprint:
        raise RuntimeError("exploration code fingerprint is not certified")
    assert_policy_covered_by_certificate(params.policy, certificate)
    assert_policy_matches_runtime(params.policy, release_trust.runtime_identity)

    metadata = _load_and_verify_metadata(store, params, certificate.certificate_digest)
    effective_budget = params.policy.budget
    for event in _journal_events(store.root, params.exploration_id):
        if isinstance(event, BudgetAmendedEvent):
            effective_budget = apply_budget_increase(effective_budget, event.increase)
    assert_budget_covered_by_certificate(effective_budget, certificate)
    _verify_consumed_approval(store, metadata)
    if str(job.get("project_id", "")) != metadata.project_id:
        raise ValueError("exploration job project does not match its metadata")
    if str(job.get("request_scope", "")) != metadata.source_session_id:
        raise ValueError("exploration job request scope must be the source session")

    actual_provider = getattr(getattr(llm, "settings", None), "provider", None)
    actual_provider_name = getattr(actual_provider, "value", str(actual_provider or ""))
    if actual_provider_name.casefold() != params.provider.casefold():
        raise RuntimeError("worker provider settings changed after approval")

    snapshot = resolve_exploration_source_snapshot(
        store,
        params.source_session_id,
        params.policy.dataset_scope,
    )
    if (
        snapshot.project_id != metadata.project_id
        or snapshot.data_state_witness != params.data_state_witness
    ):
        raise RuntimeError("exploration source witness changed before worker execution")
    loaded = load_run(
        metadata.project_id,
        metadata.source_session_id,
        workspace=store.root,
    )
    if not loaded.ok or not loaded.datasets_available or loaded.result is None:
        raise ValueError("exploration source datasets cannot be reloaded")
    selected = _selected_datasets(
        loaded.result.loaded_datasets, params.policy.dataset_scope
    )
    artifacts = _scoped_source_artifacts(
        loaded.result.artifacts, params.policy.dataset_scope
    )
    stat_registry = StatTestRegistry(
        shadow_run_root(store.root, params.exploration_id) / "stat_registry.jsonl"
    )
    context = DataToolContext(
        datasets=selected,
        catalog=build_catalog(selected),
        project_id=metadata.project_id,
        session_id=params.exploration_id,
        store=None,
        payload_policy=params.payload_policy,
        artifacts=artifacts,
        stat_registry=stat_registry,
    )
    registered = {tool.name: tool for tool in build_data_tools(context)}
    tools = build_read_only_exploration_toolset(registered)
    actual_tool_digest = exploration_tool_capability_digest(tools)
    if actual_tool_digest != certificate.bindings.tool_capability_digest:
        raise RuntimeError("worker read-only tool inventory is not certified")

    profiles = _dataset_profiles(artifacts, params.policy.dataset_scope)
    coverage_targets = frozenset(
        seed.coverage_key for seed in mandatory_probe_seeds(profiles)
    )
    dataset_columns = {
        dataset.record.dataset_id: frozenset(str(column) for column in dataset.frame.columns)
        for dataset in selected
    }
    supported_methods = frozenset(
        {
            *(tool.name for tool in tools),
            "compare_groups",
        }
    )
    run_root = shadow_run_root(store.root, params.exploration_id)
    workflow_store = JsonExplorationWorkflowStateStore(
        run_root / "workflow-state.json"
    )
    recovery_store = JsonSupervisorRecoveryStore(run_root / "phase-responses")
    journal = JsonlExplorationJournal(run_root / "journal.jsonl")

    def admission(phase_context: PhaseContext) -> AdmissionContext:
        workflow_state = workflow_store.load()
        durable = _durable_admission_state(workflow_state, params.policy)
        return AdmissionContext(
            dataset_columns=dataset_columns,
            allowed_dataset_ids=frozenset(params.policy.dataset_scope),
            supported_method_families=supported_methods,
            historical_hypothesis_fingerprints=durable[0],
            answered_hypothesis_fingerprints=durable[1],
            executed_query_fingerprints=durable[2],
            remaining_cost=_remaining_cost_fraction(
                phase_context, effective_budget.llm.max_cost_usd
            ),
            family_quota_remaining=durable[3],
            unexplored_coverage_keys=(
                coverage_targets - workflow_state.coverage_completed
            ),
        )

    def signals(
        _phase_context: PhaseContext, seeds: tuple[CandidateSeed, ...]
    ) -> Mapping[str, CandidateSignals]:
        workflow_state = workflow_store.load()
        stat_counts = _stat_attempt_counts(stat_registry)
        return {
            seed.hypothesis_id: CandidateSignals(
                business_value=_business_value(seed, params.policy.goal),
                information_gain_proxy=(
                    0.9
                    if seed.coverage_key not in workflow_state.coverage_completed
                    else 0.35
                ),
                expected_cost=_candidate_expected_cost(
                    seed, selected, effective_budget
                ),
                multiplicity_risk=_candidate_multiplicity_risk(seed, stat_counts),
                query_fingerprint=seed.hypothesis_fingerprint[:16],
            )
            for seed in seeds
        }

    result = run_composed_shadow_exploration(
        workspace=workspace,
        exploration_id=params.exploration_id,
        policy=params.policy,
        code_fingerprint=params.code_fingerprint,
        data_state_witness=params.data_state_witness,
        provider=llm,
        tools=tools,
        dataset_profiles=profiles,
        scheduler_policy=_scheduler_policy(params.policy),
        admission_context=admission,
        signals=signals,
        witness=CallableWitnessPort(
            lambda expected: _witness_matches(store, params, expected)
        ),
        usage_meter=DatasetToolUsageMeter(selected),
        stat_attempt_counts=lambda: _stat_attempt_counts(stat_registry),
        goal_satisfied=lambda state: (
            params.policy.mode == "goal_directed"
            and _goal_satisfied(
                state,
                goal=params.policy.goal,
                recovery=recovery_store,
                journal=journal,
            )
        ),
        coverage_target_met=lambda state: (
            bool(coverage_targets)
            and coverage_targets.issubset(state.coverage_completed)
        ),
        dataset_columns=dataset_columns,
        supported_method_families=supported_methods,
        dataset_facts=_dataset_facts(artifacts, params.policy.dataset_scope),
    )
    _checkpoint(cancel_check)
    if result.result.status not in {"paused", "stopped"}:
        raise RuntimeError("exploration composition returned an invalid status")


def _load_and_verify_metadata(
    store: ArtifactStore,
    params: ExplorationWorkerParams,
    certificate_digest: str,
) -> ExplorationRunMetadata:
    root = shadow_run_root(store.root, params.exploration_id)
    path = validate_shadow_run_path(
        store.root, params.exploration_id, root / "api-request.json"
    )
    metadata = ExplorationRunMetadata.model_validate_json(path.read_bytes())
    if (
        metadata.exploration_id != params.exploration_id
        or metadata.source_session_id != params.source_session_id
        or metadata.policy != params.policy
        or metadata.data_state_witness != params.data_state_witness
        or metadata.release_certificate_digest != certificate_digest
    ):
        raise ValueError("exploration worker params do not match immutable metadata")
    return metadata


def _verify_consumed_approval(
    store: ArtifactStore, metadata: ExplorationRunMetadata
) -> None:
    payload, _digest, status = ApprovalService(store).inspect_payload(
        metadata.approval_action_hash,
        session_id=metadata.source_session_id,
    )
    if status != "consumed":
        raise RuntimeError("exploration start approval was not consumed")
    if (
        str(payload.get("exploration_id", "")) != metadata.exploration_id
        or str(payload.get("policy_fingerprint", ""))
        != metadata.policy.policy_fingerprint
        or str(payload.get("release_certificate_digest", ""))
        != metadata.release_certificate_digest
        or str(payload.get("type", "")) != APPROVAL_KIND_EXPLORATION_START
    ):
        raise RuntimeError("consumed exploration approval does not match metadata")


def _selected_datasets(
    datasets: list[LoadedDataset], dataset_ids: tuple[str, ...]
) -> list[LoadedDataset]:
    by_id = {dataset.record.dataset_id: dataset for dataset in datasets}
    try:
        return [by_id[dataset_id] for dataset_id in dataset_ids]
    except KeyError as exc:
        raise ValueError("an approved exploration dataset is unavailable") from exc


def _scoped_source_artifacts(
    artifacts: list[Artifact], dataset_ids: tuple[str, ...]
) -> list[Artifact]:
    allowed = set(dataset_ids)
    return [
        artifact
        for artifact in artifacts
        if isinstance(artifact.payload.get("dataset_id"), str)
        and str(artifact.payload["dataset_id"]) in allowed
    ]


_MAX_EXAMPLE_VALUES = 5
_MAX_EXAMPLE_VALUE_CHARS = 40
_MAX_EXAMPLE_DISTINCT = 20


def _example_values(column: ColumnProfile) -> tuple[str, ...]:
    """Only for low-cardinality labels: an id or a free-text column leaks rows
    without telling the model anything it can use to pick a probe."""
    if column.semantic_type not in {"categorical", "boolean"}:
        return ()
    if column.unique_count > _MAX_EXAMPLE_DISTINCT:
        return ()
    values = []
    for raw in column.sample_values[:_MAX_EXAMPLE_VALUES]:
        text = str(raw)
        values.append(
            text
            if len(text) <= _MAX_EXAMPLE_VALUE_CHARS
            else text[: _MAX_EXAMPLE_VALUE_CHARS - 1] + "…"
        )
    return tuple(values)


def _dataset_facts(
    artifacts: list[Artifact], dataset_ids: tuple[str, ...]
) -> dict[str, DatasetFacts]:
    """Turn this session's existing EDA profiles into GENERATE prompt context."""
    facts: dict[str, DatasetFacts] = {}
    for profile in _profiles_by_id(artifacts).values():
        if profile.dataset_id not in dataset_ids:
            continue
        facts[profile.dataset_id] = DatasetFacts(
            dataset_id=profile.dataset_id,
            row_count=profile.rows,
            grain=profile.grain,
            columns=tuple(
                ColumnFact(
                    name=column.name,
                    role=column.semantic_type,
                    shape=column.distribution_kind,
                    missing_percent=column.missing_percent,
                    distinct_count=column.unique_count,
                    example_values=_example_values(column),
                )
                for column in profile.columns_detail
            ),
        )
    return facts


def _profiles_by_id(artifacts: list[Artifact]) -> dict[str, DatasetProfile]:
    by_id: dict[str, DatasetProfile] = {}
    for artifact in artifacts:
        if artifact.type is not ArtifactType.DATASET_PROFILE:
            continue
        profile = DatasetProfile.model_validate(artifact.payload)
        by_id[profile.dataset_id] = profile
    return by_id


def _dataset_profiles(
    artifacts: list[Artifact], dataset_ids: tuple[str, ...]
) -> tuple[DatasetExplorationProfile, ...]:
    by_id = _profiles_by_id(artifacts)
    profiles: list[DatasetExplorationProfile] = []
    for dataset_id in dataset_ids:
        profile = by_id.get(dataset_id)
        if profile is None:
            raise ValueError(f"dataset profile is unavailable: {dataset_id}")
        categorical = tuple(profile.categorical_columns)
        numeric = tuple(profile.numeric_columns)
        region = tuple(
            column
            for column in categorical
            if any(
                marker in column.casefold()
                for marker in ("region", "country", "state", "market", "territory")
            )
        )
        datetimes = tuple(
            column
            for column, dtype in profile.dtypes.items()
            if "date" in column.casefold()
            or "time" in column.casefold()
            or "datetime" in dtype.casefold()
        )
        profiles.append(
            DatasetExplorationProfile(
                dataset_id=dataset_id,
                region_dimensions=region,
                metric_columns=numeric,
                missing_value_columns=tuple(
                    column
                    for column, count in profile.missing_values.items()
                    if count > 0
                ),
                missingness_group_dimensions=categorical,
                datetime_columns=datetimes,
                spike_metric_columns=numeric,
            )
        )
    return tuple(profiles)


def _scheduler_policy(policy: ExplorationPolicy) -> SchedulerPolicy:
    batch_size = {"quick": 1, "standard": 3, "deep": 5}[policy.thinking_level]
    return SchedulerPolicy(
        scoring_policy_version=policy.scoring_policy_version,
        weights=PriorityWeights(
            business_value=0.30,
            information_gain_proxy=0.25,
            novelty=0.15,
            coverage_gap=0.15,
            feasibility=0.15,
            expected_cost=0.10,
            redundancy=0.10,
            multiplicity_risk=0.10,
        ),
        admission_priority=0.25,
        no_information_priority=0.25,
        max_batch_size=batch_size,
    )


def _durable_admission_state(
    state: ExplorationWorkflowState,
    policy: ExplorationPolicy,
) -> tuple[
    frozenset[str],
    frozenset[str],
    frozenset[str],
    Mapping[InsightFamily, int],
]:
    by_hypothesis = {
        decision.hypothesis_id: decision for decision in state.decisions
    }
    historical = frozenset(
        decision.hypothesis_fingerprint for decision in state.decisions
    )
    answered = frozenset(
        decision.hypothesis_fingerprint
        for insight in state.insights.values()
        if insight.status != "inconclusive"
        and insight.trust_level in {"supported", "refuted"}
        if (decision := by_hypothesis.get(insight.hypothesis_id)) is not None
    )
    executed = frozenset(
        decision.hypothesis_fingerprint[:16]
        for decision in state.decisions
        if decision.chosen
    )
    quotas = family_quotas_for_level(
        policy.thinking_level,
        policy.coverage_targets,
    )
    used = Counter(
        decision.family for decision in state.decisions if decision.chosen
    )
    remaining = {
        family: max(0, quota - used[family])
        for family, quota in quotas.items()
    }
    return historical, answered, executed, remaining


def _remaining_cost_fraction(
    context: PhaseContext,
    maximum_cost: Decimal | None,
) -> float:
    # None means "no cost cap" (schema-legal): that is zero budget pressure,
    # not zero remaining budget. Mapping it to 0.0 rejected every candidate.
    if maximum_cost is None:
        return 1.0
    if maximum_cost <= 0:
        return 0.0
    marker = "[exploration_soft_countdown] remaining="
    prefix, separator, tail = context.soft_countdown_context.partition(marker)
    if prefix or not separator:
        return 0.0
    encoded, separator, _instruction = tail.partition(". Prefer ")
    if not separator:
        return 0.0
    try:
        payload = json.loads(encoded)
        raw = payload.get("physical_llm_cost_usd")
        remaining = Decimal(str(raw))
    except (AttributeError, InvalidOperation, TypeError, ValueError):
        return 0.0
    if not remaining.is_finite() or remaining <= 0:
        return 0.0
    return float(min(Decimal("1"), remaining / maximum_cost))


def _candidate_expected_cost(
    seed: CandidateSeed,
    datasets: list[LoadedDataset],
    budget: ExplorationBudgetPolicy,
) -> float:
    method_cost = {
        "run_open_analysis": 0.28,
        "run_baseline_model": 0.24,
        "analyze_time_series": 0.20,
        "run_stat_test": 0.18,
        "compare_groups": 0.16,
        "diagnose_missingness": 0.14,
        "screen_anomalies": 0.14,
        "run_sql": 0.10,
    }.get(seed.proposal.method_family, 0.08)
    by_id = {dataset.record.dataset_id: dataset for dataset in datasets}
    scoped = [
        by_id[dataset_id]
        for dataset_id in seed.proposal.dataset_ids
        if dataset_id in by_id
    ]
    rows = sum(len(dataset.frame) for dataset in scoped)
    cells = sum(len(dataset.frame) * len(dataset.frame.columns) for dataset in scoped)
    max_rows = budget.max_rows_scanned
    max_cells = budget.max_result_cells
    row_share = 1.0 if not max_rows else min(1.0, rows / max_rows)
    cell_share = 1.0 if not max_cells else min(1.0, cells / max_cells)
    return min(1.0, method_cost + 0.20 * row_share + 0.10 * cell_share)


def _candidate_multiplicity_risk(
    seed: CandidateSeed,
    counts: Mapping[str, int],
) -> float:
    family_id = derive_family_id(
        dataset_id=seed.proposal.dataset_ids[0],
        columns=seed.proposal.columns,
    )
    attempts = counts.get(family_id, 0)
    return min(1.0, attempts / 10.0)


_GOAL_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "by",
        "for",
        "how",
        "in",
        "is",
        "of",
        "the",
        "to",
        "what",
        "why",
    }
)


def _goal_tokens(value: str) -> frozenset[str]:
    return frozenset(
        token
        for token in re.findall(r"[a-z0-9]+", value.casefold())
        if token not in _GOAL_STOP_WORDS and len(token) > 1
    )


def _business_value(seed: CandidateSeed, goal: str | None) -> float:
    if goal is None:
        return 1.0 if seed.mandatory else 0.6
    goal_tokens = _goal_tokens(goal)
    if not goal_tokens:
        return 0.0
    candidate_tokens = _candidate_goal_tokens(seed)
    return min(1.0, len(goal_tokens & candidate_tokens) / len(goal_tokens))


def _candidate_goal_tokens(seed: CandidateSeed) -> frozenset[str]:
    proposal = seed.proposal
    return _goal_tokens(
        " ".join(
            (
                proposal.statement,
                proposal.rationale,
                proposal.expected_evidence,
                proposal.method_family,
                proposal.probe_kind,
                *proposal.dataset_ids,
                *proposal.columns,
            )
        )
    )


def _goal_satisfied(
    state: ExplorationWorkflowState,
    *,
    goal: str | None,
    recovery: JsonSupervisorRecoveryStore,
    journal: JsonlExplorationJournal,
) -> bool:
    if goal is None:
        return False
    goal_tokens = _goal_tokens(goal)
    if not goal_tokens:
        return False
    journal_state = journal.rebuild()
    if journal_state is None:
        return False
    candidates: dict[str, CandidateSeed] = {}
    for round_index in range(journal_state.rounds_started):
        step_id = phase_step_id(
            journal_state.exploration_id,
            round_index,
            SupervisorPhase.GENERATE,
        )
        try:
            batch = recovery.load_required(step_id)
        except KeyError:
            continue
        if not isinstance(batch, CandidateBatch):
            raise ValueError("goal binding requires a durable candidate batch")
        candidates.update(
            {
                candidate.hypothesis_id: candidate
                for candidate in batch.candidates
                if isinstance(candidate, CandidateSeed)
            }
        )
    required_overlap = max(1, math.ceil(len(goal_tokens) / 2))
    for insight in state.insights.values():
        if insight.status == "inconclusive" or insight.trust_level not in {
            "supported",
            "refuted",
        }:
            continue
        candidate = candidates.get(insight.hypothesis_id)
        if candidate is None:
            continue
        if len(goal_tokens & _candidate_goal_tokens(candidate)) >= required_overlap:
            return True
    return False


def _journal_events(workspace: Path, exploration_id: str) -> tuple[object, ...]:
    path = shadow_run_root(workspace, exploration_id) / "journal.jsonl"
    return tuple(JsonlExplorationJournal(path).events())


def _stat_attempt_counts(registry: StatTestRegistry) -> Mapping[str, int]:
    counts: dict[str, int] = {}
    for attempt in registry.attempts():
        counts[attempt.family_id] = counts.get(attempt.family_id, 0) + 1
    return counts


def _witness_matches(
    store: ArtifactStore,
    params: ExplorationWorkerParams,
    expected: str,
) -> bool:
    try:
        current = resolve_exploration_source_snapshot(
            store, params.source_session_id, params.policy.dataset_scope
        )
    except Exception:
        return False
    return current.data_state_witness == expected


def _checkpoint(cancel_check: CancelCheck | None) -> None:
    if cancel_check is not None:
        cancel_check()
