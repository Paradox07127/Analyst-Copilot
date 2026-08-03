"""Privileged E4a certificate issuer over complete shadow evidence roots.

The public boundary accepts paths plus issuer-owned policy, fixture and signing
material. It never accepts a score, usage total, trial attestation, or evidence
public-key mapping from its caller. Every value projected into
``E4aTrialEvidence`` is rebuilt from the durable E4a run.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import BaseModel, ConfigDict, Field

from eda_platform.agents.data_tool_result_contracts import (
    verify_data_tool_result_contract,
)
from eda_platform.agents.data_tools import data_tool_argument_schema
from eda_platform.agents.exploration.branching import (
    bundle_hypotheses_from_events,
    derive_branch_constraints,
)
from eda_platform.agents.exploration.candidates import CandidateSeed
from eda_platform.agents.exploration.executor import (
    canonical_probe_fingerprint,
    durable_tool_result_digest,
    make_executor_llm_step_id,
    split_tool_sequence_index,
)
from eda_platform.agents.exploration.supervisor import (
    CandidateBatch,
    ReductionOutcome,
    reduction_outcome_digest,
)
from eda_platform.agents.exploration.workflow import (
    ExplorationWorkflowState,
    candidate_batch_digest,
    final_reduction_state_digest,
    scheduling_decision_digest,
)
from eda_platform.agents.receipts import adjudicate_receipt_hypothesis
from eda_platform.agents.runtime import (
    canonical_tool_arguments,
    canonical_tool_arguments_digest,
    canonical_tool_input_digest,
)
from eda_platform.agents.tool_context import (
    HypothesisExecutionBinding,
    ToolExecutionContext,
    make_logical_step_id,
)
from eda_platform.core.claim_gates import (
    GateReport,
    claim_bundle_digest,
    run_claim_gates,
)
from eda_platform.core.exploration_budget import apply_budget_increase
from eda_platform.core.exploration_journal import (
    JsonlExplorationJournal,
    assert_policy_sealed,
)
from eda_platform.core.exploration_release_gate import (
    E4aEvidenceBindings,
    E4aHardCaps,
    E4aReleaseCertificate,
    E4aTrialEvidence,
    E4aTrialUsage,
    _attest_e4a_verified_trial_evidence,
    _issue_e4a_release_certificate_from_verified_trials,
)
from eda_platform.core.exploration_report import render_exploration_report
from eda_platform.core.ids import stable_hash
from eda_platform.core.llm import LLMToolCall, LLMToolResponse
from eda_platform.core.llm_ledger import (
    BUDGET_SETTLED_EVENT,
    LLM_USAGE_EVENT,
    budget_policy_fingerprint,
    restore_run_budget_state,
)
from eda_platform.core.stat_registry import StatTestRegistry, derive_family_id
from eda_platform.drivers.exploration import (
    JsonExplorationWorkflowStateStore,
    JsonLlmResponseStore,
    JsonlShadowBudgetStore,
    JsonSupervisorRecoveryStore,
    JsonToolResultStore,
)
from eda_platform.schemas.claims import Claim, ClaimBundle, ClaimScope
from eda_platform.schemas.exploration import (
    BranchAbandonedEvent,
    BranchConstraint,
    BudgetAmendedEvent,
    ExplorationLoopEvent,
    ExplorationLoopState,
    ExplorationPolicy,
    ExplorationStartedEvent,
    GateVerdictEvent,
    LlmCallCompletedEvent,
    LlmCallRejectedEvent,
    LlmCallStartedEvent,
    LlmCallUncertainEvent,
    ReceiptCommittedEvent,
    ReductionCommittedEvent,
    RoundSettledEvent,
    RoundStartedEvent,
    ToolCallStartedEvent,
)
from eda_platform.schemas.exploration_shadow import ShadowExplorationProjection
from eda_platform.schemas.hypotheses import HypothesisPredicate
from eda_platform.schemas.insights import InsightRecord
from eda_platform.schemas.receipts import EvidenceReceipt, verify_receipt_digest
from eda_platform.schemas.sessions import TraceEvent

E4A_EVIDENCE_ROOT_SCHEMA_VERSION = 1
E4A_EVIDENCE_MANIFEST_VERSION = "e4a-evidence-root-v1"
_RUN_SPEC_NAME = "e4a-run-spec.json"
_CHECKER_RESULT_NAME = "e4a-checker-result.json"
_REQUIRED_ROOT_FILES = (
    _RUN_SPEC_NAME,
    _CHECKER_RESULT_NAME,
    "journal.jsonl",
    "journal.snapshot.json",
    "projection.json",
    "workflow-state.json",
    "report.md",
    "llm-budget.jsonl",
)
_ARTIFACT_DIRECTORIES = ("phase-responses", "llm-responses", "tool-results")
_OPTIONAL_ROOT_FILES = ("stat_registry.jsonl",)
_TARGET_METRICS = (
    "region_difference_recall",
    "missingness_mechanism_recall",
    "spike_day_recall",
)
# Checker v2 adds a semantic (non-predicate-identity) matching pass; older
# checker versions must keep recomputing byte-identically, so every v2
# behavior keys off this exact version string.
E4A_CHECKER_VERSION_V2 = "e4a-checker-v2"
_TARGET_METRICS_V2 = (*_TARGET_METRICS, "trend_recall")


@dataclass(frozen=True, slots=True)
class _VerifiedInvocation:
    call: LLMToolCall
    canonical_arguments: dict[str, object]
    arguments_digest: str


class E4aExpectedStructure(BaseModel):
    """One issuer-owned planted fact matched only through committed receipts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    structure_id: str = Field(min_length=1)
    target_metric: Literal[
        "region_difference_recall",
        "missingness_mechanism_recall",
        "spike_day_recall",
        "trend_recall",
    ]
    tool_names: tuple[str, ...] = Field(min_length=1)
    required_columns: tuple[str, ...] = Field(min_length=1)
    predicate: HypothesisPredicate
    # Checker-v2 semantic pass only: predicate operators accepted besides
    # exact predicate equality (empty = any operator), and fact values that
    # must hold verbatim on the supporting receipt.
    alternate_operators: tuple[str, ...] = ()
    required_fact_values: tuple[tuple[str, str], ...] = ()


class E4aGroundTruthFixture(BaseModel):
    """Frozen fixture content owned by the evaluator deployment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: str = Field(min_length=1)
    bucket: Literal["planted"] = "planted"
    expected_structures: tuple[E4aExpectedStructure, ...] = Field(min_length=1)

    @property
    def digest(self) -> str:
        return stable_hash(self.model_dump(mode="json"), length=64)


class E4aPlannedTrial(BaseModel):
    """Issuer-owned anti-relabel/cherry-pick identity for one required run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["baseline", "treatment"]
    trial_id: str = Field(min_length=1)
    tier: Literal["quick", "standard", "deep"]
    seed: int
    # The issuer freezes this after the planned run completes and before it
    # accepts the full set for release. Both arms are exact-root allowlists.
    manifest_digest: str = Field(min_length=64, max_length=64)


class E4aEvidenceIssuerBindings(BaseModel):
    """All non-path inputs controlled by the trusted evaluator/issuer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    certificate: E4aEvidenceBindings
    fixture: E4aGroundTruthFixture
    trial_plan: tuple[E4aPlannedTrial, ...] = ()


class E4aEvidenceRunSpec(BaseModel):
    """Unsigned run metadata; every security-relevant field is cross-checked."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = E4A_EVIDENCE_ROOT_SCHEMA_VERSION
    item_id: str = Field(min_length=1)
    bucket: Literal["planted"] = "planted"
    seed: int
    policy: ExplorationPolicy
    coverage_targets: tuple[str, ...] = ()
    checker_version: str = Field(min_length=1)
    ground_truth_digest: str = Field(min_length=64, max_length=64)


class E4aCheckerResult(BaseModel):
    """Deterministic checker artifact that must equal issuer recomputation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    checker_version: str = Field(min_length=1)
    evaluated_insight_ids: tuple[str, ...] = ()
    matched_structure_ids: tuple[str, ...] = ()
    unmatched_insight_ids: tuple[str, ...] = ()
    scores: dict[str, float]


class E4aVerifiedEvidenceManifest(BaseModel):
    """Canonical root over bytes and independently reconstructed projections."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_version: Literal["e4a-evidence-root-v1"] = E4A_EVIDENCE_MANIFEST_VERSION
    trial_id: str = Field(min_length=1)
    artifact_sha256: dict[str, str]
    journal_terminal_digest: str = Field(min_length=64, max_length=64)
    workflow_digest: str = Field(min_length=64, max_length=64)
    projection_digest: str = Field(min_length=64, max_length=64)
    report_sha256: str = Field(min_length=64, max_length=64)
    llm_ledger_digest: str = Field(min_length=64, max_length=64)
    tool_results_digest: str = Field(min_length=64, max_length=64)
    checker_result_digest: str = Field(min_length=64, max_length=64)
    ground_truth_digest: str = Field(min_length=64, max_length=64)

    @property
    def root_digest(self) -> str:
        return stable_hash(self.model_dump(mode="json"), length=64)


def issue_e4a_release_from_evidence_roots(
    *,
    baseline_roots: Sequence[Path | str],
    treatment_roots: Sequence[Path | str],
    hard_caps: E4aHardCaps,
    issuer_bindings: E4aEvidenceIssuerBindings,
    evidence_signing_key: bytes,
    release_signing_key: bytes,
    release_key_id: str,
    minimum_trials_per_tier: int = 5,
    issued_at: datetime | None = None,
) -> E4aReleaseCertificate:
    """Verify complete run roots, internally attest them, and issue a certificate.

    The evidence public key is derived from the issuer's private key and cannot
    be nominated by a workspace manifest. Raw score/usage objects are not part
    of this API.
    """
    if not baseline_roots or not treatment_roots:
        raise ValueError("baseline and treatment evidence roots must be non-empty")
    if not release_key_id.strip():
        raise ValueError("release_key_id must be non-empty")
    evidence_key_id = issuer_bindings.certificate.evidence_key_id
    baseline_verified = [
        _verify_and_project_root(
            Path(root), issuer_bindings=issuer_bindings, signing_key=evidence_signing_key
        )
        for root in baseline_roots
    ]
    treatment_verified = [
        _verify_and_project_root(
            Path(root), issuer_bindings=issuer_bindings, signing_key=evidence_signing_key
        )
        for root in treatment_roots
    ]
    _verify_trial_plan(
        baseline_verified,
        treatment_verified,
        issuer_bindings.trial_plan,
    )
    baseline = [trial for trial, _manifest in baseline_verified]
    treatment = [trial for trial, _manifest in treatment_verified]
    _require_unique_trials(baseline, treatment)
    evidence_public_key = (
        Ed25519PrivateKey.from_private_bytes(evidence_signing_key).public_key().public_bytes_raw()
    )
    return _issue_e4a_release_certificate_from_verified_trials(
        baseline=baseline,
        treatment=treatment,
        hard_caps=hard_caps,
        bindings=issuer_bindings.certificate,
        minimum_trials_per_tier=minimum_trials_per_tier,
        issued_at=issued_at,
        evidence_public_keys={evidence_key_id: evidence_public_key},
        release_signing_key=release_signing_key,
        release_key_id=release_key_id,
    )


def verify_e4a_evidence_root(
    root: Path | str,
    *,
    issuer_bindings: E4aEvidenceIssuerBindings,
    evidence_signing_key: bytes,
) -> tuple[E4aTrialEvidence, E4aVerifiedEvidenceManifest]:
    """Audit one root and expose its derived trial plus canonical manifest."""
    return _verify_and_project_root(
        Path(root),
        issuer_bindings=issuer_bindings,
        signing_key=evidence_signing_key,
    )


def _verify_and_project_root(
    root: Path,
    *,
    issuer_bindings: E4aEvidenceIssuerBindings,
    signing_key: bytes,
) -> tuple[E4aTrialEvidence, E4aVerifiedEvidenceManifest]:
    root = _validated_root(root)
    spec = E4aEvidenceRunSpec.model_validate_json(_read_regular(root / _RUN_SPEC_NAME))
    if (
        spec.item_id != issuer_bindings.fixture.item_id
        or spec.bucket != issuer_bindings.fixture.bucket
    ):
        raise ValueError("run spec does not select the issuer-owned planted fixture")
    if spec.checker_version != issuer_bindings.certificate.checker_version:
        raise ValueError("run spec checker version does not match issuer bindings")
    if spec.ground_truth_digest != issuer_bindings.fixture.digest:
        raise ValueError("run spec ground truth digest does not match issuer fixture")
    assert_policy_sealed(spec.policy)
    if spec.policy.tool_capability_digest != issuer_bindings.certificate.tool_capability_digest:
        raise ValueError("run policy tool capability digest does not match issuer bindings")

    journal = JsonlExplorationJournal(root / "journal.jsonl")
    events = journal.events()
    terminal = journal.rebuild()
    if not events or terminal is None or terminal.status != "stopped":
        raise ValueError("evidence journal must have a valid terminal state")
    trial_id = terminal.exploration_id
    started = events[0]
    if not isinstance(started, ExplorationStartedEvent):
        raise ValueError("evidence journal must begin with exploration_started")
    if started.code_fingerprint != issuer_bindings.certificate.code_fingerprint:
        raise ValueError("journal code fingerprint does not match issuer bindings")
    if started.policy_fingerprint != spec.policy.policy_fingerprint:
        raise ValueError("run spec policy does not match the journal")
    if started.budget != spec.policy.budget:
        raise ValueError("run spec budget does not match the journal")
    snapshot = journal.read_snapshot()
    if snapshot != terminal:
        raise ValueError("journal snapshot does not match the authoritative replay")

    projection = ShadowExplorationProjection.model_validate_json(
        _read_regular(root / "projection.json")
    )
    if (
        projection.exploration_id != trial_id
        or projection.last_seq != terminal.last_seq
        or projection.status != terminal.status
        or projection.stop_reason != terminal.stop_reason
        or projection.policy_fingerprint != terminal.effective_policy_fingerprint
        or projection.data_state_witness != terminal.data_state_witness
    ):
        raise ValueError("shadow projection does not match the journal terminal state")

    workflow = JsonExplorationWorkflowStateStore(root / "workflow-state.json").load()
    _verify_committed_receipts(workflow, terminal, events)
    candidates, candidates_by_round = _verify_scheduler_and_reductions(
        root, events, workflow
    )
    canonical_bundles = _rebuild_canonical_bundles(
        events,
        workflow,
        candidates_by_round=candidates_by_round,
    )
    invocations = _verify_production_invocations(
        root,
        events,
        workflow,
        candidates_by_round=candidates_by_round,
    )
    stat_attempt_counts = _verify_stat_registry(
        root, workflow.committed_receipts, invocations=invocations
    )
    _verify_workflow(
        workflow,
        terminal,
        canonical_bundles=canonical_bundles,
        stat_attempt_counts=stat_attempt_counts,
    )
    _verify_branch_abandonments(
        events,
        workflow,
        candidates_by_round=candidates_by_round,
    )
    if tuple(sorted(workflow.insights.values(), key=lambda item: item.insight_id)) != (
        projection.insight_records
    ):
        raise ValueError("projection insights do not match workflow state")

    effective_budget = spec.policy.budget
    accepted_policy_fingerprints = {budget_policy_fingerprint(effective_budget.llm.to_policy())}
    for event in events:
        if isinstance(event, BudgetAmendedEvent):
            effective_budget = apply_budget_increase(effective_budget, event.increase)
            accepted_policy_fingerprints.add(
                budget_policy_fingerprint(effective_budget.llm.to_policy())
            )
    ledger_events = JsonlShadowBudgetStore(root / "llm-budget.jsonl").events()
    llm_policy = effective_budget.llm.to_policy()
    # restore performs reservation/settlement consistency and unique-call checks.
    llm_state = restore_run_budget_state(
        llm_policy,
        ledger_events,
        run_started_at=events[0].occurred_at,
        accepted_policy_fingerprints=frozenset(accepted_policy_fingerprints),
    )
    _verify_llm_correspondence(events, ledger_events)
    provider, model, llm_requests, total_tokens, cost = _llm_usage(ledger_events)
    if llm_state.requests_used != llm_requests or llm_state.total_tokens_used != total_tokens:
        raise ValueError("LLM usage events disagree with the restored spend ledger")

    tool_digest = _verify_tool_results(
        root,
        events,
        workflow.committed_receipts,
        invocations=invocations,
    )
    _verify_completed_llm_bodies(root, events)
    usage = E4aTrialUsage(
        llm_requests=llm_requests,
        total_tokens=total_tokens,
        estimated_cost_usd=cost,
        wall_clock_seconds=_wall_clock_seconds(events, ledger_events),
        tool_calls=terminal.tool_calls_committed,
        rows_scanned=terminal.rows_scanned,
        cells_scanned=terminal.result_cells,
    )
    budget_summary: Mapping[str, object] = {
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
            "exploration_id": trial_id,
            "policy_fingerprint": terminal.effective_policy_fingerprint,
            "witness": terminal.data_state_witness,
        },
        coverage_targets=spec.coverage_targets,
        budget_summary=budget_summary,
        stop_reason=str(terminal.stop_reason),
    )
    report_bytes = _read_regular(root / "report.md")
    if report_bytes.decode("utf-8") != rendered.markdown:
        raise ValueError("report artifact does not match deterministic rerendering")

    checker = _recompute_checker(
        workflow=workflow,
        events=events,
        terminal=terminal,
        candidates=candidates,
        stop_reason=terminal.stop_reason,
        fixture=issuer_bindings.fixture,
        checker_version=issuer_bindings.certificate.checker_version,
    )
    persisted_checker = E4aCheckerResult.model_validate_json(
        _read_regular(root / _CHECKER_RESULT_NAME)
    )
    if persisted_checker != checker:
        raise ValueError("checker artifact does not match deterministic recomputation")

    artifact_sha256 = _artifact_digests(root)
    manifest = E4aVerifiedEvidenceManifest(
        trial_id=trial_id,
        artifact_sha256=artifact_sha256,
        journal_terminal_digest=stable_hash(terminal.model_dump(mode="json"), length=64),
        workflow_digest=stable_hash(_workflow_payload(workflow), length=64),
        projection_digest=stable_hash(projection.model_dump(mode="json"), length=64),
        report_sha256=hashlib.sha256(report_bytes).hexdigest(),
        llm_ledger_digest=stable_hash(
            [event.model_dump(mode="json") for event in ledger_events], length=64
        ),
        tool_results_digest=tool_digest,
        checker_result_digest=stable_hash(checker.model_dump(mode="json"), length=64),
        ground_truth_digest=issuer_bindings.fixture.digest,
    )
    raw_trial = E4aTrialEvidence(
        trial_id=trial_id,
        item_id=spec.item_id,
        bucket=spec.bucket,
        model=model,
        provider=provider,
        tier=spec.policy.thinking_level,
        seed=spec.seed,
        status="scored",
        passed=True,
        scores=checker.scores,
        usage=usage,
        checker_version=spec.checker_version,
        code_fingerprint=started.code_fingerprint,
        tool_capability_digest=spec.policy.tool_capability_digest,
    )
    signed = _attest_e4a_verified_trial_evidence(
        raw_trial,
        signing_key=signing_key,
        key_id=issuer_bindings.certificate.evidence_key_id,
        source_manifest_digest=manifest.root_digest,
    )
    return signed, manifest


def _verify_production_invocations(
    root: Path,
    events: Sequence[ExplorationLoopEvent],
    workflow: ExplorationWorkflowState,
    *,
    candidates_by_round: Mapping[int, Mapping[str, CandidateSeed]],
) -> dict[str, _VerifiedInvocation]:
    """Bind each committed receipt to one canonical production tool invocation."""
    responses: dict[str, LLMToolResponse] = {}
    for response_path in sorted((root / "llm-responses").glob("*.json")):
        raw = json.loads(_read_regular(response_path))
        logical_step_id = raw.get("logical_step_id")
        if not isinstance(logical_step_id, str) or not logical_step_id:
            raise ValueError("durable LLM response lacks its logical step id")
        response = LLMToolResponse.model_validate(raw.get("response"))
        if logical_step_id in responses:
            raise ValueError("durable LLM response step ids must be unique")
        responses[logical_step_id] = response

    started_by_step: dict[str, ToolCallStartedEvent] = {}
    committed_by_receipt: dict[str, ReceiptCommittedEvent] = {}
    receipt_round: dict[str, int] = {}
    active_round: int | None = None
    for event in events:
        if isinstance(event, RoundStartedEvent):
            active_round = event.round_index
        elif isinstance(event, ToolCallStartedEvent):
            if event.logical_step_id in started_by_step:
                raise ValueError("tool logical steps must start exactly once")
            started_by_step[event.logical_step_id] = event
        elif isinstance(event, ReceiptCommittedEvent):
            if active_round is None or event.receipt_id in committed_by_receipt:
                raise ValueError("receipt commit lacks one unique active round")
            committed_by_receipt[event.receipt_id] = event
            receipt_round[event.receipt_id] = active_round
        elif isinstance(event, RoundSettledEvent):
            active_round = None

    verified: dict[str, _VerifiedInvocation] = {}
    used_physical_calls: set[tuple[str, str]] = set()
    for receipt_id, receipt in workflow.committed_receipts.items():
        execution = receipt.execution
        commit = committed_by_receipt.get(receipt_id)
        round_index = receipt_round.get(receipt_id)
        if execution is None or commit is None or round_index is None:
            raise ValueError("committed receipt lacks its production execution identity")
        candidate = _candidate_for_execution(
            receipt,
            exploration_id=commit.exploration_id,
            round_index=round_index,
            candidates=candidates_by_round.get(round_index, {}),
        )
        step, action_index = split_tool_sequence_index(execution.sequence_index)
        llm_step_id = make_executor_llm_step_id(execution.run_id, step)
        response = responses.get(llm_step_id)
        if response is None or action_index > len(response.tool_calls):
            raise ValueError("receipt sequence does not select a durable provider tool call")
        call = response.tool_calls[action_index - 1]
        physical_key = (llm_step_id, call.call_id)
        if physical_key in used_physical_calls:
            raise ValueError("one physical provider tool call produced multiple receipts")
        used_physical_calls.add(physical_key)
        schema = data_tool_argument_schema(call.name)
        canonical_arguments = canonical_tool_arguments(schema, call.arguments)
        expected_logical_step = make_logical_step_id(
            execution.run_id,
            call.call_id,
            execution.sequence_index,
        )
        expected_tool_call_id = ToolExecutionContext(
            run_id=execution.run_id,
            provider_call_id=call.call_id,
            logical_step_id=expected_logical_step,
            attempt_epoch=execution.attempt_epoch,
            sequence_index=execution.sequence_index,
        ).call_identity()
        started = started_by_step.get(expected_logical_step)
        if (
            call.call_id != execution.provider_call_id
            or call.name != receipt.tool_name
            or execution.logical_step_id != expected_logical_step
            or commit.logical_step_id != expected_logical_step
            or execution.attempt_epoch != commit.attempt_epoch
            or receipt.tool_call_id != expected_tool_call_id
            or receipt.input_digest != canonical_tool_input_digest(schema, canonical_arguments)
            or started is None
            or started.attempt_epoch != execution.attempt_epoch
            or started.tool_kind != receipt.tool_name
            or started.input_fingerprint
            != canonical_probe_fingerprint(call.name, canonical_arguments)
        ):
            raise ValueError("receipt invocation identity fails canonical production binding")
        binding = HypothesisExecutionBinding(
            hypothesis_id=candidate.hypothesis_id,
            predicate=candidate.proposal.predicate,
            method_family=candidate.proposal.method_family,
            dataset_ids=candidate.proposal.dataset_ids,
            columns=candidate.proposal.columns,
        )
        adjudicated = adjudicate_receipt_hypothesis(receipt, binding)
        actual_statistics = receipt.statistics
        expected_statistics = adjudicated.statistics
        if (
            actual_statistics is None
            or expected_statistics is None
            or actual_statistics.hypothesis_id != expected_statistics.hypothesis_id
            or actual_statistics.hypothesis_outcome
            != expected_statistics.hypothesis_outcome
        ):
            raise ValueError("receipt typed adjudication fails candidate replay")
        verified[receipt_id] = _VerifiedInvocation(
            call=call,
            canonical_arguments=canonical_arguments,
            arguments_digest=canonical_tool_arguments_digest(
                schema, canonical_arguments
            ),
        )
    return verified


def _candidate_for_execution(
    receipt: EvidenceReceipt,
    *,
    exploration_id: str,
    round_index: int,
    candidates: Mapping[str, CandidateSeed],
) -> CandidateSeed:
    execution = receipt.execution
    if execution is None:
        raise ValueError("receipt lacks executor identity")
    prefix = f"{exploration_id}:round:{round_index}:hypothesis:"
    suffix = ":execute_probes"
    if not execution.run_id.startswith(prefix) or not execution.run_id.endswith(suffix):
        raise ValueError("receipt executor identity does not match its journal round")
    hypothesis_id = execution.run_id[len(prefix) : -len(suffix)]
    candidate = candidates.get(hypothesis_id)
    if candidate is None:
        raise ValueError("receipt executor identity lacks its recovered candidate")
    return candidate


def _verify_stat_registry(
    root: Path,
    committed_receipts: Mapping[str, EvidenceReceipt],
    *,
    invocations: Mapping[str, _VerifiedInvocation],
) -> dict[str, int]:
    path = root / "stat_registry.jsonl"
    statistical_receipts = {
        receipt.receipt_id: receipt
        for receipt in committed_receipts.values()
        if receipt.statistics is not None and receipt.statistics.p_value is not None
    }
    if not path.exists():
        if statistical_receipts:
            raise ValueError("statistical receipts require an authoritative registry")
        return {}
    if path.is_symlink() or not path.is_file():
        raise ValueError("statistical registry must be a regular file")
    attempts = StatTestRegistry(path).attempts()
    if any(attempt.status == "running" for attempt in attempts):
        raise ValueError("terminal evidence contains a running statistical attempt")
    counts: dict[str, int] = {}
    completed_by_receipt = {}
    for attempt in attempts:
        counts[attempt.family_id] = counts.get(attempt.family_id, 0) + 1
        if attempt.status == "completed":
            if attempt.receipt_id is None or attempt.receipt_id in completed_by_receipt:
                raise ValueError("statistical registry completion receipts must be unique")
            completed_by_receipt[attempt.receipt_id] = attempt
    if not set(completed_by_receipt).issubset(committed_receipts):
        raise ValueError("statistical registry cites an uncommitted receipt")
    for receipt_id, receipt in statistical_receipts.items():
        statistics = receipt.statistics
        assert statistics is not None
        family_id = statistics.statistical_family_id or statistics.hypothesis_id
        attempt = completed_by_receipt.get(receipt_id)
        invocation = invocations.get(receipt_id)
        requested_test_type = (
            None
            if invocation is None
            else invocation.canonical_arguments.get("test_type")
        )
        if (
            requested_test_type is None
            and invocation is not None
            and receipt.tool_name == "analyze_time_series"
        ):
            # The tool has no test_type argument; its registered test is fixed.
            requested_test_type = "ljung_box"
        expected_family_id = None
        if invocation is not None:
            dataset_id = invocation.canonical_arguments.get("dataset_id")
            columns = tuple(
                value
                for key in (
                    "group_column",
                    "value_column",
                    "category_column",
                    "pair_column",
                    "time_column",
                )
                if isinstance(
                    value := invocation.canonical_arguments.get(key), str
                )
            )
            if isinstance(dataset_id, str):
                expected_family_id = derive_family_id(
                    dataset_id=dataset_id, columns=columns
                )
        if (
            family_id is None
            or attempt is None
            or attempt.family_id != family_id
            or attempt.family_id != expected_family_id
            or attempt.sequence_index != statistics.sequence_index
            or attempt.requested_test_type
            != str(receipt.method.parameters.get("requested_test_type", statistics.test_name))
            or requested_test_type != attempt.requested_test_type
            or invocation is None
            or invocation.arguments_digest != attempt.arguments_digest
            or receipt.execution is None
            or attempt.logical_step_id != receipt.execution.logical_step_id
        ):
            raise ValueError("statistical receipt does not match its registry attempt")
    return counts


def _verify_workflow(
    workflow: ExplorationWorkflowState,
    terminal: ExplorationLoopState,
    *,
    canonical_bundles: Mapping[str, ClaimBundle],
    stat_attempt_counts: Mapping[str, int],
) -> None:
    committed_receipts = workflow.committed_receipts
    gate_reports = workflow.gate_reports
    admitted_bundles = workflow.admitted_bundles
    if set(gate_reports) != set(terminal.gate_verdicts):
        raise ValueError("workflow gate reports do not exactly match journal verdicts")
    if set(canonical_bundles) != set(gate_reports):
        raise ValueError("canonical reducer bundles do not exactly match journal verdicts")
    for bundle_id, report in gate_reports.items():
        verdict = terminal.gate_verdicts[bundle_id]
        if verdict != ("passed" if report.passed else "rejected"):
            raise ValueError("workflow gate report conflicts with journal verdict")
        if report.run_witness != terminal.data_state_witness:
            raise ValueError("workflow gate report witness does not match journal")
        canonical = canonical_bundles[bundle_id]
        recomputed = run_claim_gates(
            canonical,
            committed_receipts=committed_receipts,
            run_witness=terminal.data_state_witness,
            stat_attempt_counts=stat_attempt_counts,
        )
        if recomputed != report:
            raise ValueError("gate report fails canonical deterministic replay")
    for bundle_id, bundle in admitted_bundles.items():
        report = gate_reports.get(bundle_id)
        if (
            report is None
            or not report.passed
            or report.claim_bundle_digest != claim_bundle_digest(bundle)
        ):
            raise ValueError("admitted bundle lacks its exact passing gate report")
        if bundle != canonical_bundles[bundle_id]:
            raise ValueError("admitted bundle differs from canonical reducer output")
    passed_bundle_ids = {bundle_id for bundle_id, report in gate_reports.items() if report.passed}
    if set(admitted_bundles) != passed_bundle_ids:
        raise ValueError("admitted bundles must exactly equal all passed gate reports")


def _verify_committed_receipts(
    workflow: ExplorationWorkflowState,
    terminal: ExplorationLoopState,
    events: Sequence[ExplorationLoopEvent],
) -> None:
    committed_receipts = workflow.committed_receipts
    journal_receipts = set(terminal.step_receipt_refs.values())
    if not set(committed_receipts).issubset(journal_receipts):
        raise ValueError("workflow receipts do not exactly match journal commits")
    journal_only = journal_receipts - set(committed_receipts)
    if journal_only and journal_only != _trailing_unsettled_receipts(events):
        raise ValueError("workflow receipts do not exactly match journal commits")
    for receipt_id, receipt in committed_receipts.items():
        if (
            receipt_id != receipt.receipt_id
            or not verify_receipt_digest(receipt)
            or receipt.data_state_witness != terminal.data_state_witness
        ):
            raise ValueError("workflow contains an invalid committed receipt")


def _trailing_unsettled_receipts(
    events: Sequence[ExplorationLoopEvent],
) -> set[str]:
    """Receipts committed strictly after the last round_settled event.

    An interrupted run (e.g. budget_exhausted mid-round) journals these commits
    but never reaches the reduce that folds them into workflow state; they are
    the only journal-side surplus issuance tolerates."""
    trailing: set[str] = set()
    for event in events:
        if isinstance(event, RoundSettledEvent):
            trailing.clear()
        elif isinstance(event, ReceiptCommittedEvent):
            trailing.add(event.receipt_id)
    return trailing


def _verify_branch_abandonments(
    events: Sequence[ExplorationLoopEvent],
    workflow: ExplorationWorkflowState,
    *,
    candidates_by_round: Mapping[int, Mapping[str, CandidateSeed]],
) -> None:
    """Recompute every branch_abandoned constraint set at its event position.

    The reducer already enforces the structural branch rules during replay;
    this pass pins the semantic content: constraints must equal the shared
    deterministic derivation over receipts, gate reports and candidates known
    at that point (plan E6 gate 2)."""
    if not any(isinstance(event, BranchAbandonedEvent) for event in events):
        return
    candidates: dict[str, CandidateSeed] = {}
    receipts_so_far: dict[str, EvidenceReceipt] = {}
    reports_so_far: dict[str, GateReport] = {}
    prior: list[BranchConstraint] = []
    prefix: list[ExplorationLoopEvent] = []
    for event in events:
        prefix.append(event)
        if isinstance(event, RoundStartedEvent):
            for hypothesis_id, seed in candidates_by_round.get(
                event.round_index, {}
            ).items():
                candidates[hypothesis_id] = seed
        elif isinstance(event, ReceiptCommittedEvent):
            receipt = workflow.committed_receipts.get(event.receipt_id)
            if receipt is None:
                raise ValueError("journal receipt is missing from workflow state")
            receipts_so_far[event.receipt_id] = receipt
        elif isinstance(event, GateVerdictEvent):
            report = workflow.gate_reports.get(event.claim_bundle_id)
            if report is None:
                raise ValueError("journal gate verdict lacks its stored report")
            if report.passed != (event.verdict == "passed"):
                raise ValueError("stored gate report disagrees with the journal verdict")
            reports_so_far[event.claim_bundle_id] = report
        elif isinstance(event, BranchAbandonedEvent):
            expected = derive_branch_constraints(
                candidates=candidates,
                committed_receipts=receipts_so_far,
                gate_reports=reports_so_far,
                bundle_hypotheses=bundle_hypotheses_from_events(
                    prefix, receipts_so_far
                ),
                prior=tuple(prior),
            )
            if tuple(event.constraints) != expected:
                raise ValueError(
                    "branch abandonment constraints do not match deterministic "
                    "recomputation"
                )
            prior.extend(event.constraints)


def _rebuild_canonical_bundles(
    events: Sequence[ExplorationLoopEvent],
    workflow: ExplorationWorkflowState,
    *,
    candidates_by_round: Mapping[int, Mapping[str, CandidateSeed]],
) -> dict[str, ClaimBundle]:
    """Replay the reducer's bundle construction without trusting stored bundles.

    Receipt-to-hypothesis identity comes from the executor-minted ``run_id``;
    gate ordering comes from the append-only journal.  Stored claim ids, text,
    evidence references, and even the stored bundle hypothesis id are never
    inputs to this reconstruction.
    """
    prior_supporting: dict[str, tuple[str, ...]] = {}
    prior_contradicting: dict[str, tuple[str, ...]] = {}
    canonical: dict[str, ClaimBundle] = {}
    active_round: int | None = None
    current_by_hypothesis: dict[str, list[str]] = {}
    gate_order: list[str] = []
    gate_cursor = 0
    exploration_id: str | None = None

    for event in events:
        exploration_id = exploration_id or event.exploration_id
        if event.exploration_id != exploration_id:
            raise ValueError("journal events span multiple exploration ids")
        if isinstance(event, RoundStartedEvent):
            if active_round is not None:
                raise ValueError("canonical bundle replay found overlapping rounds")
            active_round = event.round_index
            current_by_hypothesis = {}
            gate_order = []
            gate_cursor = 0
            continue
        if isinstance(event, ReceiptCommittedEvent):
            if active_round is None:
                raise ValueError("receipt commit is outside a scheduler round")
            receipt = workflow.committed_receipts.get(event.receipt_id)
            if receipt is None or receipt.execution is None:
                raise ValueError("canonical bundle receipt lacks executor identity")
            round_candidates = candidates_by_round.get(active_round, {})
            candidate = _candidate_for_execution(
                receipt,
                exploration_id=exploration_id,
                round_index=active_round,
                candidates=round_candidates,
            )
            hypothesis_id = candidate.hypothesis_id
            statistics = receipt.statistics
            if (
                statistics is not None
                and statistics.hypothesis_id is not None
                and statistics.hypothesis_id != hypothesis_id
            ):
                raise ValueError("receipt adjudication conflicts with executor identity")
            if receipt.facts:
                if hypothesis_id not in current_by_hypothesis:
                    current_by_hypothesis[hypothesis_id] = []
                    gate_order.append(hypothesis_id)
                current_by_hypothesis[hypothesis_id].append(receipt.receipt_id)
            continue
        if isinstance(event, GateVerdictEvent):
            if active_round is None or gate_cursor >= len(gate_order):
                raise ValueError("gate verdict lacks its deterministic receipt group")
            hypothesis_id = gate_order[gate_cursor]
            gate_cursor += 1
            candidate = candidates_by_round[active_round][hypothesis_id]
            current_ids = tuple(current_by_hypothesis[hypothesis_id])
            cumulative_ids = tuple(
                dict.fromkeys(
                    (
                        *prior_supporting.get(hypothesis_id, ()),
                        *prior_contradicting.get(hypothesis_id, ()),
                        *current_ids,
                    )
                )
            )
            bundle = _canonical_claim_bundle(
                candidate,
                tuple(workflow.committed_receipts[item] for item in cumulative_ids),
            )
            if event.claim_bundle_id != bundle.claim_bundle_id:
                raise ValueError("journal gate id differs from canonical reducer bundle")
            prior_bundle = canonical.get(bundle.claim_bundle_id)
            if prior_bundle is not None and prior_bundle != bundle:
                raise ValueError("one canonical bundle id maps to conflicting bodies")
            canonical[bundle.claim_bundle_id] = bundle
            if event.verdict == "passed":
                supporting: list[str] = []
                contradicting: list[str] = []
                for item in current_ids:
                    statistics = workflow.committed_receipts[item].statistics
                    outcome = None if statistics is None else statistics.hypothesis_outcome
                    # Mirrors workflow ClaimGateReducerPort.reduce verbatim: an
                    # unadjudicated receipt joins neither prior side.
                    if outcome == "supports":
                        supporting.append(item)
                    elif outcome == "contradicts":
                        contradicting.append(item)
                prior_supporting[hypothesis_id] = tuple(
                    dict.fromkeys((*prior_supporting.get(hypothesis_id, ()), *supporting))
                )
                prior_contradicting[hypothesis_id] = tuple(
                    dict.fromkeys(
                        (*prior_contradicting.get(hypothesis_id, ()), *contradicting)
                    )
                )
            continue
        if isinstance(event, RoundSettledEvent):
            if active_round is not None and gate_cursor != len(gate_order):
                raise ValueError("scheduler round has receipt groups without gate verdicts")
            active_round = None
    if active_round is not None:
        raise ValueError("canonical bundle replay ended with an open round")
    return canonical


def _canonical_claim_bundle(
    candidate: CandidateSeed, receipts: Sequence[EvidenceReceipt]
) -> ClaimBundle:
    """Independent, fail-closed reconstruction of workflow ``_claim_bundle``."""
    claims = tuple(
        Claim(
            claim_id="clm_"
            + stable_hash(
                {"receipt_id": receipt.receipt_id, "fact_id": fact.fact_id}, length=20
            ),
            claim_type="absence" if fact.support_type == "absence" else "observation",
            claim_text=f"{fact.name}: {_canonical_fact_text(fact.value, fact.value_type)}",
            support_type=fact.support_type,
            evidence_fact_ids=(f"{receipt.receipt_id}:{fact.fact_id}",),
            statistics_receipt_ids=(
                (receipt.receipt_id,)
                if _has_canonical_confirmatory_statistics(receipt)
                else ()
            ),
            uncertainty=("; ".join(receipt.method.warnings) or None),
            limitations=tuple(receipt.method.warnings),
            scope=(
                _canonical_absence_scope(receipt)
                if fact.support_type == "absence"
                else None
            ),
        )
        for receipt in receipts
        for fact in receipt.facts
        if not (
            isinstance(fact.value, str)
            and any(character.isdigit() for character in fact.value)
        )
    )
    bundle_id = "cb_" + stable_hash(
        {
            "hypothesis_id": candidate.hypothesis_id,
            "receipts": [
                {
                    "receipt_id": receipt.receipt_id,
                    "facts": [fact.fact_id for fact in receipt.facts],
                }
                for receipt in receipts
            ],
        },
        length=24,
    )
    return ClaimBundle(
        claim_bundle_id=bundle_id,
        hypothesis_id=candidate.hypothesis_id,
        evidence_lane=(
            "confirmatory"
            if receipts
            and all(_has_canonical_confirmatory_statistics(receipt) for receipt in receipts)
            else "exploratory"
        ),
        claims=claims,
    )


def _canonical_absence_scope(receipt: EvidenceReceipt) -> ClaimScope:
    """Mirrors workflow ``_absence_scope``: the declared scope is the scanned one."""
    return ClaimScope(
        dataset_ids=receipt.scope.dataset_ids,
        columns=receipt.scope.columns,
        filters=receipt.scope.filters,
        time_range=receipt.scope.time_range,
    )


def _canonical_fact_text(value: object, value_type: str) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if value_type == "percent":
        return f"{value}%"
    return str(value)


def _has_canonical_confirmatory_statistics(receipt: EvidenceReceipt) -> bool:
    statistics = receipt.statistics
    return bool(
        statistics is not None
        and statistics.p_value is not None
        and statistics.test_statistic is not None
        and statistics.effect_size is not None
        and statistics.ci_low is not None
        and statistics.ci_high is not None
        and statistics.sample_size is not None
    )


def _candidate_identity(candidate: CandidateSeed) -> tuple[object, ...]:
    """Semantic identity of a candidate across rounds.

    ``sequence_index``, ``status``, ``origin``, ``mandatory`` and ``priority``
    are excluded: a mandatory probe replayed into a later round is the same
    hypothesis carrying a new batch position and control-plane state, not a
    conflicting body.
    """
    return (
        candidate.proposal,
        candidate.hypothesis_id,
        candidate.hypothesis_fingerprint,
        candidate.canonical_group_key,
        candidate.coverage_key,
    )


def _verify_scheduler_and_reductions(
    root: Path,
    events: Sequence[ExplorationLoopEvent],
    workflow: ExplorationWorkflowState,
) -> tuple[dict[str, CandidateSeed], dict[int, dict[str, CandidateSeed]]]:
    recovery = JsonSupervisorRecoveryStore(root / "phase-responses")
    response_digests = {
        event.step_id: event.response_digest
        for event in events
        if isinstance(event, LlmCallCompletedEvent)
    }
    active_round: int | None = None
    candidates: dict[str, CandidateSeed] = {}
    candidates_by_round: dict[int, dict[str, CandidateSeed]] = {}
    decision_cursor = 0
    last_reduction_event: ReductionCommittedEvent | None = None
    for event in events:
        if isinstance(event, RoundStartedEvent):
            active_round = event.round_index
            continue
        if not isinstance(event, ReductionCommittedEvent):
            continue
        last_reduction_event = event
        if active_round is None:
            raise ValueError("reduction commit has no active scheduler round")
        generate_step = f"{event.exploration_id}:round:{active_round}:generate"
        batch = recovery.load_required(generate_step)
        if not isinstance(batch, CandidateBatch):
            raise ValueError("scheduler recovery body is not a candidate batch")
        if candidate_batch_digest(batch) != response_digests.get(generate_step):
            raise ValueError("scheduler candidate batch fails its journal digest")
        round_candidates = {
            item.hypothesis_id: item for item in batch.candidates if isinstance(item, CandidateSeed)
        }
        if len(round_candidates) != len(
            [item for item in batch.candidates if isinstance(item, CandidateSeed)]
        ):
            raise ValueError("candidate batch contains duplicate hypothesis ids")
        if active_round in candidates_by_round:
            raise ValueError("one scheduler round maps to multiple reductions")
        candidates_by_round[active_round] = round_candidates
        for hypothesis_id, candidate in round_candidates.items():
            prior = candidates.get(hypothesis_id)
            if prior is not None and _candidate_identity(prior) != _candidate_identity(
                candidate
            ):
                raise ValueError("one hypothesis id maps to conflicting candidate bodies")
            candidates.setdefault(hypothesis_id, candidate)
        round_decisions = workflow.decisions[
            decision_cursor : decision_cursor + len(round_candidates)
        ]
        decision_cursor += len(round_candidates)
        if len(round_decisions) != len(round_candidates):
            raise ValueError("workflow decisions are missing a scheduler round")
        if {item.hypothesis_id for item in round_decisions} != set(round_candidates):
            raise ValueError("workflow decisions do not exactly cover the candidate batch")
        for decision in round_decisions:
            candidate = round_candidates[decision.hypothesis_id]
            if (
                decision.hypothesis_fingerprint != candidate.hypothesis_fingerprint
                or decision.family != candidate.proposal.family
            ):
                raise ValueError("workflow decision identity conflicts with its candidate")
        expected_frontier = "frontier_" + scheduling_decision_digest(round_decisions)
        if event.frontier_digest != expected_frontier:
            raise ValueError("scheduler decisions fail the journal frontier digest")
        reduce_step = f"{event.exploration_id}:round:{active_round}:reduce"
        reduction = recovery.load_required(reduce_step)
        if not isinstance(reduction, ReductionOutcome):
            raise ValueError("reduction recovery body has the wrong type")
        if (
            reduction.frontier.digest != event.frontier_digest
            or reduction.ledger_digest != event.ledger_digest
            or reduction_outcome_digest(reduction) != event.reduction_digest
        ):
            raise ValueError("reduction body fails its journal digests")
    if decision_cursor != len(workflow.decisions):
        raise ValueError("workflow contains decisions absent from candidate recovery bodies")
    if last_reduction_event is None:
        raise ValueError("evidence root contains no committed reduction")
    final_ledger_digest = final_reduction_state_digest(workflow)
    if last_reduction_event.ledger_digest != final_ledger_digest:
        raise ValueError("final reducer state fails its journal ledger digest")
    if any(
        insight.hypothesis_id not in candidates
        for insight in workflow.insights.values()
    ):
        raise ValueError("workflow insight lacks its recovered candidate")
    expected_coverage = {
        candidates[insight.hypothesis_id].coverage_key
        for insight in workflow.insights.values()
    }
    if workflow.coverage_completed != expected_coverage:
        raise ValueError("workflow coverage does not exactly match terminal insights")
    return candidates, candidates_by_round


def _verify_llm_correspondence(
    journal_events: Sequence[ExplorationLoopEvent], ledger_events: Sequence[TraceEvent]
) -> None:
    started = [event for event in journal_events if isinstance(event, LlmCallStartedEvent)]
    terminal = [
        event
        for event in journal_events
        if isinstance(
            event,
            LlmCallCompletedEvent | LlmCallRejectedEvent | LlmCallUncertainEvent,
        )
    ]
    started_ids = [event.call_id for event in started]
    terminal_ids = [event.call_id for event in terminal]
    if len(started_ids) != len(set(started_ids)) or set(started_ids) != set(terminal_ids):
        raise ValueError("journal LLM calls must have one unique terminal outcome")
    logical_ids = [
        event.summary.get("logical_call_id")
        for event in ledger_events
        if event.event_type == LLM_USAGE_EVENT
    ]
    if any(not isinstance(item, str) or not item for item in logical_ids):
        raise ValueError("physical LLM usage must identify its logical journal call")
    if len(logical_ids) != len(set(logical_ids)) or set(logical_ids) != set(started_ids):
        raise ValueError("physical LLM usage does not map one-to-one to journal calls")


def _llm_usage(events: Sequence[TraceEvent]) -> tuple[str, str, int, int, float]:
    """Aggregate provider spend across the run's ledger.

    A call with ``usage_known`` false (provider never confirmed its own usage)
    does not block issuance: it is billed the same conservative reservation
    that ``restore_run_budget_state``'s ``mark_uncertain`` path already
    consumed (llm_ledger.py:665-673), read here from that call's
    ``budget_settled`` event so the two totals cannot diverge. A call whose
    usage is known must still carry fully measured, provider-reported usage.
    """
    usage_events = [event for event in events if event.event_type == LLM_USAGE_EVENT]
    if not usage_events:
        raise ValueError("evidence root contains no measured provider usage")
    call_ids = [getattr(event, "call_id", None) for event in usage_events]
    if any(not call_id for call_id in call_ids) or len(set(call_ids)) != len(call_ids):
        raise ValueError("LLM ledger call ids must be present and unique")
    settled_by_call = {
        event.call_id: event.summary
        for event in events
        if event.event_type == BUDGET_SETTLED_EVENT and event.call_id
    }
    pairs: set[tuple[str, str]] = set()
    total_tokens = 0
    cost = 0.0
    for event in usage_events:
        summary = event.summary
        usage_known = summary.get("usage_known") is True
        if usage_known:
            if summary.get("provider_usage_reported") is not True:
                raise ValueError("every provider call must carry measured usage")
            provider = summary.get("provider")
            model = summary.get("model")
            if (
                not isinstance(provider, str)
                or not provider
                or not isinstance(model, str)
                or not model
            ):
                raise ValueError("every provider call must identify provider and model")
            pairs.add((provider.casefold(), model))
            tokens = summary.get("total_tokens")
            event_cost = summary.get("estimated_cost_usd")
        else:
            # call_id non-emptiness is enforced above for every usage event.
            settled = settled_by_call.get(event.call_id or "")
            if settled is None:
                raise ValueError("uncertain provider call has no settled budget reservation")
            provider = summary.get("provider")
            model = summary.get("model")
            if (
                isinstance(provider, str)
                and provider
                and isinstance(model, str)
                and model
            ):
                pairs.add((provider.casefold(), model))
            tokens = settled.get("total_tokens")
            event_cost = settled.get("estimated_cost_usd")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
            raise ValueError("provider total_tokens must be measured and non-negative")
        if (
            isinstance(event_cost, bool)
            or not isinstance(event_cost, int | float)
            or event_cost < 0
        ):
            raise ValueError("provider cost must be measured and non-negative")
        total_tokens += tokens
        cost += float(event_cost)
    if not pairs:
        raise ValueError("evidence root has no LLM call with an identified provider/model")
    if len(pairs) != 1:
        raise ValueError("one evidence root must use exactly one provider/model pair")
    provider, model = next(iter(pairs))
    return provider, model, len(usage_events), total_tokens, round(cost, 12)


def _verify_tool_results(
    root: Path,
    events: Sequence[object],
    committed_receipts: Mapping[str, EvidenceReceipt],
    *,
    invocations: Mapping[str, _VerifiedInvocation],
) -> str:
    store = JsonToolResultStore(root / "tool-results")
    committed_events = [event for event in events if isinstance(event, ReceiptCommittedEvent)]
    bodies: list[dict[str, object]] = []
    for event in committed_events:
        durable = store.load_required(event.logical_step_id)
        artifact = durable.result.receipt_artifact
        if artifact is None:
            raise ValueError("committed tool result has no receipt artifact")
        receipt = EvidenceReceipt.model_validate(artifact.payload)
        if receipt.receipt_id != event.receipt_id or receipt != committed_receipts.get(
            event.receipt_id
        ):
            raise ValueError("tool result receipt does not match workflow/journal")
        if not verify_receipt_digest(receipt):
            raise ValueError("tool result contains an invalid receipt digest")
        invocation = invocations.get(receipt.receipt_id)
        if invocation is None:
            raise ValueError("tool result lacks its verified provider invocation")
        verify_data_tool_result_contract(
            receipt,
            durable.result,
            invocation.canonical_arguments,
        )
        body_digest = durable_tool_result_digest(durable)
        if event.result_digest is None or event.result_digest != body_digest:
            raise ValueError("tool result body fails its journal result digest")
        if (
            durable.usage.rows_scanned != event.rows_scanned
            or durable.usage.result_cells != event.result_cells
        ):
            raise ValueError("tool result usage does not match journal settlement")
        bodies.append(
            {
                "logical_step_id": event.logical_step_id,
                "receipt": receipt.model_dump(mode="json"),
                "usage": {
                    "kind": durable.usage.kind,
                    "rows_scanned": durable.usage.rows_scanned,
                    "result_cells": durable.usage.result_cells,
                },
            }
        )
    if set(committed_receipts) != {event.receipt_id for event in committed_events}:
        raise ValueError("tool results do not cover every committed workflow receipt")
    return stable_hash(bodies, length=64)


def _verify_completed_llm_bodies(root: Path, events: Sequence[object]) -> None:
    phase = JsonSupervisorRecoveryStore(root / "phase-responses")
    executor = JsonLlmResponseStore(root / "llm-responses")
    for event in events:
        if not isinstance(event, LlmCallCompletedEvent):
            continue
        try:
            phase_body = phase.load_required(event.step_id)
        except KeyError:
            phase_digest = None
        else:
            phase_digest = (
                candidate_batch_digest(phase_body)
                if isinstance(phase_body, CandidateBatch)
                else phase.digest_if_present(event.step_id)
            )
        digests = (
            phase_digest,
            executor.digest_if_present(event.step_id),
        )
        if event.response_digest not in digests:
            raise ValueError("completed LLM response body does not match journal digest")


def _proof_metrics(
    workflow: ExplorationWorkflowState, terminal: ExplorationLoopState
) -> dict[str, float]:
    journal_receipts = set(terminal.step_receipt_refs.values())
    total_insights = len(workflow.insights)
    reachable_insights = 0
    total_edges = 0
    fabricated_edges = 0
    journal_edges = 0
    reachable_edges = 0
    for insight in workflow.insights.values():
        insight_reachable = True
        bundle = workflow.admitted_bundles.get(insight.claim_bundle_id)
        report = workflow.gate_reports.get(insight.claim_bundle_id)
        if (
            bundle is None
            or report is None
            or not report.passed
            or report.claim_bundle_digest != claim_bundle_digest(bundle)
        ):
            insight_reachable = False
        for proof in insight.proof:
            total_edges += 1
            receipt = workflow.committed_receipts.get(proof.receipt_id)
            if receipt is None:
                fabricated_edges += 1
                insight_reachable = False
                continue
            if proof.receipt_id in journal_receipts:
                journal_edges += 1
            else:
                insight_reachable = False
            expected_comparison = (
                "supports"
                if proof.receipt_id in insight.supporting_receipt_ids
                else "contradicts"
                if proof.receipt_id in insight.contradicting_receipt_ids
                else None
            )
            fact_ids = {fact.fact_id for fact in receipt.facts} | {
                derivation.derived_fact_id for derivation in receipt.derivations
            }
            edge_reachable = (
                expected_comparison == proof.comparison
                and set(proof.fact_ids).issubset(fact_ids)
                and verify_receipt_digest(receipt)
                and receipt.data_state_witness == terminal.data_state_witness
                and proof.receipt_id in journal_receipts
            )
            if edge_reachable:
                reachable_edges += 1
            else:
                insight_reachable = False
        if insight.proof and insight_reachable:
            reachable_insights += 1
    return {
        "grounding_rate": round(
            1.0 if total_insights == 0 else reachable_insights / total_insights, 6
        ),
        "fabricated_receipt_rate": round(
            0.0 if total_edges == 0 else fabricated_edges / total_edges, 6
        ),
        "proof_reachability_rate": round(
            1.0 if total_edges == 0 else reachable_edges / total_edges, 6
        ),
        "journal_provenance_rate": round(
            1.0 if total_edges == 0 else journal_edges / total_edges, 6
        ),
    }


def _search_dynamics_scores(
    *,
    matched_insight_rounds: Sequence[int],
    expected_count: int,
    rounds_started: int,
) -> dict[str, float]:
    """Search-dynamics metrics (plan §10.3.1): AUC-over-steps and first improvement step."""
    if rounds_started <= 0:
        auc = 0.0
    else:
        auc = sum(
            sum(1 for created in matched_insight_rounds if created <= r) / expected_count
            for r in range(rounds_started)
        ) / rounds_started
    return {
        "auc_over_steps": round(auc, 6),
        "first_improvement_step": (
            float(min(matched_insight_rounds)) if matched_insight_rounds else -1.0
        ),
    }


def _recompute_checker(
    *,
    workflow: ExplorationWorkflowState,
    events: Sequence[ExplorationLoopEvent],
    terminal: ExplorationLoopState,
    candidates: Mapping[str, CandidateSeed],
    stop_reason: object,
    fixture: E4aGroundTruthFixture,
    checker_version: str,
) -> E4aCheckerResult:
    semantic = checker_version == E4A_CHECKER_VERSION_V2
    insights = tuple(
        sorted(
            (item for item in workflow.insights.values() if item.status in {"new", "reinforced"}),
            key=lambda item: item.insight_id,
        )
    )
    receipts = workflow.committed_receipts
    proof_metrics = _proof_metrics(workflow, terminal)
    # Pass 1: exact predicate identity, one structure per insight and one
    # insight per structure (unchanged checker-v1 semantics).
    matched_by_insight: dict[str, str] = {}
    for insight in insights:
        for expected in fixture.expected_structures:
            if expected.structure_id in matched_by_insight.values():
                continue
            candidate = candidates.get(insight.hypothesis_id)
            if candidate is None or candidate.proposal.predicate != expected.predicate:
                continue
            for receipt_id in insight.supporting_receipt_ids:
                if _structure_supporting_receipt(insight, receipts[receipt_id], expected):
                    matched_by_insight[insight.insight_id] = expected.structure_id
                    break
            if insight.insight_id in matched_by_insight:
                break
    matched_pairs: set[tuple[str, str]] = {
        (structure_id, insight_id)
        for insight_id, structure_id in matched_by_insight.items()
    }
    # Pass 2 (v2 only): semantic matching for insights pass 1 left unmatched,
    # against every structure — replication of a matched structure is credit.
    if semantic:
        for insight in insights:
            if insight.insight_id in matched_by_insight:
                continue
            candidate = candidates.get(insight.hypothesis_id)
            if candidate is None:
                continue
            operator = candidate.proposal.predicate.operator
            for expected in fixture.expected_structures:
                if (
                    expected.alternate_operators
                    and operator not in expected.alternate_operators
                ):
                    continue
                for receipt_id in insight.supporting_receipt_ids:
                    receipt = receipts.get(receipt_id)
                    if (
                        receipt is None
                        or not _structure_supporting_receipt(insight, receipt, expected)
                        or not _required_fact_values_hold(receipt, expected)
                    ):
                        continue
                    matched_pairs.add((expected.structure_id, insight.insight_id))
                    break
    matched_structures = {structure_id for structure_id, _insight_id in matched_pairs}
    matched_insight_ids = {insight_id for _structure_id, insight_id in matched_pairs}
    expected_count = len(fixture.expected_structures)
    reported_count = len(insights)
    precision = len(matched_insight_ids) / reported_count if reported_count else 0.0
    recall = len(matched_structures) / expected_count
    no_information_rounds = 0
    for event in reversed(events):
        if isinstance(event, RoundSettledEvent) and not event.progress:
            no_information_rounds += 1
        elif isinstance(event, RoundSettledEvent):
            break
    fingerprints = [decision.hypothesis_fingerprint for decision in workflow.decisions]
    canonical_groups = {
        candidates[decision.hypothesis_id].canonical_group_key for decision in workflow.decisions
    }
    scores: dict[str, float] = {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "grounding_rate": proof_metrics["grounding_rate"],
        "fabricated_receipt_rate": proof_metrics["fabricated_receipt_rate"],
        "spam_fixture_input_count": float(len(fingerprints)),
        "spam_fixture_canonical_groups": float(len(canonical_groups)),
        "no_information_rounds": float(no_information_rounds),
        "no_information_stopped": float(stop_reason == "no_new_information"),
        "proof_reachability_rate": proof_metrics["proof_reachability_rate"],
        "journal_provenance_rate": proof_metrics["journal_provenance_rate"],
    }
    insight_rounds = {item.insight_id: item.created_round for item in insights}
    if semantic:
        # Structure-level dynamics: a structure counts from the earliest round
        # any insight (either pass) matched it.
        dynamics_rounds = tuple(
            min(
                insight_rounds[insight_id]
                for structure_id, insight_id in matched_pairs
                if structure_id == matched_structure_id
            )
            for matched_structure_id in matched_structures
        )
        scores["mandatory_probe_recall"] = round(
            len(set(matched_by_insight.values())) / expected_count, 6
        )
        scores.update(
            _checker_v2_scores(workflow, insights, candidates, matched_pairs)
        )
    else:
        dynamics_rounds = tuple(
            insight_rounds[insight_id] for insight_id in matched_by_insight
        )
    scores.update(
        _search_dynamics_scores(
            matched_insight_rounds=dynamics_rounds,
            expected_count=expected_count,
            rounds_started=terminal.rounds_started,
        )
    )
    for metric in _TARGET_METRICS_V2 if semantic else _TARGET_METRICS:
        expected_ids = {
            item.structure_id
            for item in fixture.expected_structures
            if item.target_metric == metric
        }
        scores[metric] = float(bool(expected_ids) and expected_ids.issubset(matched_structures))
    return E4aCheckerResult(
        checker_version=checker_version,
        evaluated_insight_ids=tuple(item.insight_id for item in insights),
        matched_structure_ids=tuple(sorted(matched_structures)),
        unmatched_insight_ids=tuple(
            item.insight_id for item in insights if item.insight_id not in matched_insight_ids
        ),
        scores=scores,
    )


def _structure_supporting_receipt(
    insight: InsightRecord,
    receipt: EvidenceReceipt,
    structure: E4aExpectedStructure,
) -> bool:
    """Receipt-level match conditions shared by both checker passes."""
    if receipt.tool_name not in structure.tool_names:
        return False
    if not set(structure.required_columns).issubset(receipt.scope.columns):
        return False
    if (
        receipt.statistics is None
        or receipt.statistics.hypothesis_outcome != "supports"
        or receipt.statistics.hypothesis_id != insight.hypothesis_id
    ):
        return False
    proof = next(
        (
            edge
            for edge in insight.proof
            if edge.receipt_id == receipt.receipt_id and edge.comparison == "supports"
        ),
        None,
    )
    fact_ids = {fact.fact_id for fact in receipt.facts} | {
        derivation.derived_fact_id for derivation in receipt.derivations
    }
    return proof is not None and set(proof.fact_ids).issubset(fact_ids)


def _required_fact_values_hold(
    receipt: EvidenceReceipt, structure: E4aExpectedStructure
) -> bool:
    return all(
        any(
            fact.fact_id == fact_id and str(fact.value) == value
            for fact in receipt.facts
        )
        for fact_id, value in structure.required_fact_values
    )


def _checker_v2_scores(
    workflow: ExplorationWorkflowState,
    insights: Sequence[InsightRecord],
    candidates: Mapping[str, CandidateSeed],
    matched_pairs: set[tuple[str, str]],
) -> dict[str, float]:
    """Origin-attribution metrics; agent-origin means any non-mandatory seed."""
    origin_by_insight = {
        item.insight_id: candidates[item.hypothesis_id].origin
        for item in insights
        if item.hypothesis_id in candidates
    }

    def _is_agent(insight_id: str) -> bool:
        origin = origin_by_insight.get(insight_id)
        return origin is not None and origin != "mandatory"

    structure_insights: dict[str, set[str]] = {}
    for structure_id, insight_id in matched_pairs:
        structure_insights.setdefault(structure_id, set()).add(insight_id)
    matched_insight_ids = {insight_id for _structure_id, insight_id in matched_pairs}
    agent_matched_rounds = [
        item.created_round
        for item in insights
        if item.insight_id in matched_insight_ids and _is_agent(item.insight_id)
    ]
    return {
        "agent_discovery_count": float(
            sum(
                1
                for insight_ids in structure_insights.values()
                if any(_is_agent(insight_id) for insight_id in insight_ids)
            )
        ),
        "independent_replication_count": float(
            sum(
                1
                for insight_ids in structure_insights.values()
                if any(
                    origin_by_insight.get(insight_id) == "mandatory"
                    for insight_id in insight_ids
                )
                and any(_is_agent(insight_id) for insight_id in insight_ids)
            )
        ),
        "agent_novel_supported_count": float(
            sum(
                1
                for item in insights
                if _is_agent(item.insight_id)
                and item.insight_id not in matched_insight_ids
            )
        ),
        "refuted_insight_count": float(
            sum(1 for item in workflow.insights.values() if item.status == "refuted")
        ),
        "agent_first_discovery_round": (
            float(min(agent_matched_rounds)) if agent_matched_rounds else -1.0
        ),
    }


def _require_unique_trials(
    baseline: Sequence[E4aTrialEvidence], treatment: Sequence[E4aTrialEvidence]
) -> None:
    all_ids = [item.trial_id for item in (*baseline, *treatment)]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("baseline and treatment evidence roots must have unique trial ids")
    tier_seeds = [(item.tier, item.seed) for item in treatment]
    if len(tier_seeds) != len(set(tier_seeds)):
        raise ValueError("treatment evidence roots must have unique tier/seed pairs")


def _verify_trial_plan(
    baseline: Sequence[tuple[E4aTrialEvidence, E4aVerifiedEvidenceManifest]],
    treatment: Sequence[tuple[E4aTrialEvidence, E4aVerifiedEvidenceManifest]],
    plan: Sequence[E4aPlannedTrial],
) -> None:
    if not plan:
        raise ValueError("production issuance requires an issuer-owned trial plan")
    indexed: dict[tuple[str, str], E4aPlannedTrial] = {}
    for item in plan:
        key = (item.role, item.trial_id)
        if key in indexed:
            raise ValueError("issuer trial plan contains duplicate role/trial ids")
        indexed[key] = item
    actual_keys = {
        *(("baseline", trial.trial_id) for trial, _manifest in baseline),
        *(("treatment", trial.trial_id) for trial, _manifest in treatment),
    }
    if actual_keys != set(indexed):
        raise ValueError("evidence roots do not exactly match the issuer-owned trial plan")
    for role, verified in (("baseline", baseline), ("treatment", treatment)):
        for trial, manifest in verified:
            planned = indexed[(role, trial.trial_id)]
            if (trial.tier, trial.seed) != (planned.tier, planned.seed):
                raise ValueError("evidence root tier/seed conflicts with the issuer trial plan")
            if manifest.root_digest != planned.manifest_digest:
                raise ValueError("evidence root manifest is not pinned by the issuer trial plan")


def _wall_clock_seconds(
    events: Sequence[ExplorationLoopEvent], ledger_events: Sequence[TraceEvent]
) -> float:
    starts = [item.occurred_at for item in events]
    starts.extend(item.started_at for item in ledger_events)
    finishes = [item.occurred_at for item in events]
    finishes.extend(item.finished_at or item.started_at for item in ledger_events)
    return round(max(0.0, (max(finishes) - min(starts)).total_seconds()), 6)


def _workflow_payload(workflow: ExplorationWorkflowState) -> dict[str, object]:
    return {
        "decisions": [
            {
                "hypothesis_id": item.hypothesis_id,
                "hypothesis_fingerprint": item.hypothesis_fingerprint,
                "family": item.family.value,
                "status": item.status,
                "admission_checks": [vars(check) for check in item.admission_checks],
                "priority_features": item.priority_features.model_dump(mode="json"),
                "priority": item.priority,
                "scoring_policy_version": item.scoring_policy_version,
                "quota_deferred": item.quota_deferred,
                "chosen": item.chosen,
            }
            for item in workflow.decisions
        ],
        "committed_receipts": [
            item.model_dump(mode="json")
            for item in sorted(
                workflow.committed_receipts.values(), key=lambda value: value.receipt_id
            )
        ],
        "gate_reports": [
            item.model_dump(mode="json")
            for item in sorted(
                workflow.gate_reports.values(), key=lambda value: value.claim_bundle_id
            )
        ],
        "admitted_bundles": [
            item.model_dump(mode="json")
            for item in sorted(
                workflow.admitted_bundles.values(), key=lambda value: value.claim_bundle_id
            )
        ],
        "insights": [
            item.model_dump(mode="json")
            for item in sorted(workflow.insights.values(), key=lambda value: value.insight_id)
        ],
        "coverage_completed": sorted(workflow.coverage_completed),
    }


def _artifact_digests(root: Path) -> dict[str, str]:
    allowed: list[Path] = [root / name for name in _REQUIRED_ROOT_FILES]
    allowed.extend(root / name for name in _OPTIONAL_ROOT_FILES if (root / name).exists())
    for name in _ARTIFACT_DIRECTORIES:
        directory = root / name
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"evidence artifact directory {name!r} is missing or unsafe")
        allowed.extend(sorted(directory.glob("*.json")))
    allowed_set = {path.resolve(strict=True) for path in allowed}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("evidence roots cannot contain symlinks")
        if (
            path.is_file()
            and not path.name.endswith(".lock")
            and path.resolve(strict=True) not in allowed_set
        ):
            raise ValueError(f"unexpected evidence artifact {path.relative_to(root)}")
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(_read_regular(path)).hexdigest()
        for path in sorted(allowed, key=lambda item: item.relative_to(root).as_posix())
    }


def _validated_root(root: Path) -> Path:
    if not root.is_absolute():
        raise ValueError("evidence roots must be absolute paths")
    if root.is_symlink() or not root.is_dir():
        raise ValueError("evidence root must be a real directory")
    resolved = root.resolve(strict=True)
    for name in _REQUIRED_ROOT_FILES:
        _read_regular(resolved / name)
    return resolved


def _read_regular(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"evidence artifact {path.name!r} must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(descriptor)
