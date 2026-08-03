"""Production E4a port composition over deterministic exploration components.

The model may propose hypotheses, but identity, scheduling, tool execution,
claim construction, gates, insight transitions and stopping signals remain
system-owned.  The classes here are concrete supervisor ports rather than test
doubles; the driver supplies only environment-specific LLM/tools/storage.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from eda_platform.agents.exploration.branching import (
    bundle_hypotheses_from_events,
    derive_branch_constraints,
)
from eda_platform.agents.exploration.candidates import (
    CandidateSeed,
    DatasetExplorationProfile,
    mandatory_probe_seeds,
    materialize_proposal_batch,
)
from eda_platform.agents.exploration.executor import (
    PROBE_LOCAL_ERROR_CODES,
    JsonlProbeJournalHooks,
    LlmResponseStore,
    NativeToolProvider,
    PhaseToolMap,
    ProbeExecutionResult,
    ProbeExecutor,
    ToolResultStore,
    ToolUsageMeter,
)
from eda_platform.agents.exploration.reducer import reduce_insight
from eda_platform.agents.exploration.scheduler import (
    AdmissionContext,
    CandidateSignals,
    SchedulerPolicy,
    SchedulingDecision,
    schedule_candidates,
)
from eda_platform.agents.exploration.supervisor import (
    CandidateBatch,
    CompletedStepRecoveryPort,
    FrontierItem,
    PhaseContext,
    ProbeOutcome,
    ProbeSelection,
    ReductionOutcome,
    ScoredFrontier,
    SupervisorBudgetExhausted,
    SupervisorCancelled,
    SupervisorInvariantError,
    SupervisorPhase,
    ValidationOutcome,
    phase_step_id,
)
from eda_platform.agents.runtime import AgentTool
from eda_platform.agents.tool_context import HypothesisExecutionBinding
from eda_platform.core.budget import BudgetExceeded
from eda_platform.core.cancellation import CancellationError
from eda_platform.core.claim_gates import GateReport, run_claim_gates
from eda_platform.core.exploration_budget import ToolCallLedger
from eda_platform.core.exploration_journal import JsonlExplorationJournal
from eda_platform.core.ids import make_artifact_id, stable_hash
from eda_platform.core.kernel import SessionCancelled
from eda_platform.core.llm import MalformedProviderResponseError
from eda_platform.core.llm_ledger import logical_llm_call
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.claims import Claim, ClaimBundle, ClaimScope
from eda_platform.schemas.exploration import BranchConstraint, LlmCallCompletedEvent
from eda_platform.schemas.hypotheses import HypothesisProposalBatch
from eda_platform.schemas.insights import InsightRecord, TransitionProposal
from eda_platform.schemas.receipts import EvidenceReceipt, verify_receipt_digest

# Read from the schema so the prompt and the contract cannot drift apart.
_MAX_PROPOSALS_PER_BATCH = next(
    constraint.max_length
    for constraint in HypothesisProposalBatch.model_fields["proposals"].metadata
    if hasattr(constraint, "max_length")
)
# One repair attempt. A model that breaks the contract twice in a row is not
# going to be talked into it by a third copy of the same error.
_PROPOSAL_SCHEMA_ATTEMPTS = 2


class StructuredHypothesisProvider(Protocol):
    def structured(
        self,
        *,
        task: str,
        schema: type[HypothesisProposalBatch],
        payload: dict[str, Any],
    ) -> HypothesisProposalBatch: ...


class ExplorationProvider(StructuredHypothesisProvider, NativeToolProvider, Protocol):
    """Provider surface required by the concrete E4a composition."""


@dataclass(slots=True)
class ExplorationWorkflowState:
    decisions: tuple[SchedulingDecision, ...] = ()
    committed_receipts: dict[str, EvidenceReceipt] = field(default_factory=dict)
    gate_reports: dict[str, GateReport] = field(default_factory=dict)
    admitted_bundles: dict[str, ClaimBundle] = field(default_factory=dict)
    insights: dict[str, InsightRecord] = field(default_factory=dict)
    coverage_completed: set[str] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class ExecutedProbeBatch:
    selection: ProbeSelection
    executions: tuple[ProbeExecutionResult, ...]
    receipts: tuple[EvidenceReceipt, ...]
    receipt_hypothesis_bindings: tuple[tuple[str, str], ...]


@dataclass(slots=True)
class JournaledCandidateGenerator:
    """One structured proposal call plus deterministic mandatory coverage seeds."""

    provider: StructuredHypothesisProvider
    journal: JsonlExplorationJournal
    recovery: CompletedStepRecoveryPort
    dataset_profiles: tuple[DatasetExplorationProfile, ...]
    goal: str | None = None
    task: str = "exploration_generate_hypotheses"
    dataset_columns: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    supported_method_families: tuple[str, ...] = ()
    coverage_completed: Callable[[], frozenset[str]] = lambda: frozenset()

    def generate(self, context: PhaseContext, *, logical_step_id: str) -> CandidateBatch:
        call_id = "llm_" + stable_hash(
            {"step_id": logical_step_id, "phase": context.phase.value}, length=24
        )
        journal_state = self.journal.rebuild()
        if journal_state is None:
            raise SupervisorInvariantError("candidate generation requires an initialized journal.")
        if call_id in journal_state.uncertain_call_ids:
            # The provider may have accepted the request before the process died.
            # The stable logical call must latch closed across recovery.
            raise SupervisorInvariantError(
                "candidate generation outcome is uncertain after recovery; "
                "refusing to resend the logical provider request."
            )
        preflight = getattr(self.provider, "preflight_structured", None)
        if callable(preflight):
            try:
                preflight(
                    task=self.task,
                    schema=HypothesisProposalBatch,
                    payload=self._payload(context),
                )
            except BudgetExceeded as exc:
                raise SupervisorBudgetExhausted(str(exc)) from exc
        batch = self._propose(context, call_id=call_id, logical_step_id=logical_step_id)
        first_index = context.round_index * 10_000 + 1
        generated = materialize_proposal_batch(
            batch,
            first_sequence_index=first_index,
        )
        # Replayed every round, not just round 0: a batch-limited tier can only
        # execute part of the mandatory coverage per round, and a probe dropped
        # from a later batch is coverage lost for the whole run.
        explored = self.coverage_completed()
        mandatory = tuple(
            seed
            for seed in mandatory_probe_seeds(
                self.dataset_profiles,
                first_sequence_index=first_index + len(generated),
            )
            if seed.coverage_key not in explored
        )
        result = CandidateBatch((*mandatory, *generated))
        self.recovery.remember(logical_step_id, result)
        self.journal.append_new(
            "llm_call_completed",
            call_id=call_id,
            step_id=logical_step_id,
            response_digest=candidate_batch_digest(result),
        )
        return result

    def _propose(
        self,
        context: PhaseContext,
        *,
        call_id: str,
        logical_step_id: str,
    ) -> HypothesisProposalBatch:
        """One proposal call, retried once when the answer breaks the schema.

        A schema violation means the provider answered and the answer is
        unusable -- the same class as unparseable tool arguments, not an unknown
        outcome. Two unusable answers conclude the round rather than discarding
        every round already paid for.

        The last attempt deliberately leaves its call pending: the caller
        settles it with ``llm_call_completed``, whose digest covers the durable
        candidate batch rather than the raw provider answer. Journaling a
        rejection there instead would clear the pending call and leave the
        generate step with no way to be marked durably complete.
        """
        rejections: list[str] = []
        last = _PROPOSAL_SCHEMA_ATTEMPTS - 1
        for attempt in range(_PROPOSAL_SCHEMA_ATTEMPTS):
            payload = self._payload(context, rejections=rejections)
            self.journal.append_new(
                "llm_call_started", call_id=call_id, step_id=logical_step_id
            )
            try:
                with logical_llm_call(call_id):
                    return self.provider.structured(
                        task=self.task,
                        schema=HypothesisProposalBatch,
                        payload=payload,
                    )
            except BudgetExceeded as exc:
                self.journal.append_new(
                    "llm_call_rejected",
                    call_id=call_id,
                    error=_safe_error(exc),
                )
                raise SupervisorBudgetExhausted(str(exc)) from exc
            except MalformedProviderResponseError as exc:
                error = _safe_error(exc)
                rejections.append(error)
                if attempt < last:
                    self.journal.append_new(
                        "llm_call_rejected",
                        call_id=call_id,
                        error=error,
                    )
            except Exception as exc:
                self.journal.append_new(
                    "llm_call_uncertain",
                    call_id=call_id,
                    error=_safe_error(exc),
                )
                raise
        return HypothesisProposalBatch(
            concluded=True,
            conclusion_reason=(
                f"The proposal call broke the batch contract on all "
                f"{_PROPOSAL_SCHEMA_ATTEMPTS} attempts "
                f"({' | '.join(rejections)}); concluding this round on the "
                "evidence already gathered."
            ),
        )

    def _payload(
        self,
        context: PhaseContext,
        *,
        rejections: Sequence[str] = (),
    ) -> dict[str, Any]:
        instruction = (
            "Propose a small, falsifiable batch of at most "
            f"{_MAX_PROPOSALS_PER_BATCH} hypotheses. Scores, priorities, coverage "
            "and identities are assigned by the system."
        )
        if self.goal is not None:
            instruction += " Every proposal must directly test the stated user goal."
        instruction += (
            " Every proposal must use the exact dataset ids, column names and "
            "method_family values listed here; anything else is rejected by the "
            "system before it can run."
        )
        return {
            "round_index": context.round_index,
            "data_state_witness": context.data_state_witness,
            "budget_context": context.soft_countdown_context,
            "goal": self.goal,
            "datasets": [
                {"dataset_id": dataset_id, "columns": list(columns)}
                for dataset_id, columns in sorted(self.dataset_columns.items())
            ],
            "method_families": sorted(self.supported_method_families),
            "instruction": instruction,
            "rejected_prior_attempts": list(rejections),
        }


@dataclass(slots=True)
class WorkflowBranchDeriver:
    """Live-side abandonment constraints; must equal the issuer's replay recompute."""

    journal: JsonlExplorationJournal
    recovery: CompletedStepRecoveryPort
    state: ExplorationWorkflowState

    def derive(self, context: PhaseContext) -> tuple[BranchConstraint, ...]:
        journal_state = self.journal.rebuild()
        if journal_state is None:
            raise SupervisorInvariantError(
                "branch derivation requires an initialized journal."
            )
        candidates: dict[str, CandidateSeed] = {}
        for round_index in range(journal_state.rounds_started):
            step_id = phase_step_id(
                context.exploration_id, round_index, SupervisorPhase.GENERATE
            )
            batch = self.recovery.load_required(step_id)
            if not isinstance(batch, CandidateBatch):
                raise SupervisorInvariantError(
                    "recovered candidate batch has the wrong type."
                )
            for seed in batch.candidates:
                if isinstance(seed, CandidateSeed):
                    candidates[seed.hypothesis_id] = seed
        return derive_branch_constraints(
            candidates=candidates,
            committed_receipts=self.state.committed_receipts,
            gate_reports=self.state.gate_reports,
            bundle_hypotheses=bundle_hypotheses_from_events(
                self.journal.events(), self.state.committed_receipts
            ),
            prior=tuple(journal_state.abandoned_constraints),
        )


@dataclass(frozen=True, slots=True)
class ExplorationWorkflowComponents:
    """Concrete, production-shaped supervisor ports built from environment seams."""

    state: ExplorationWorkflowState
    generator: JournaledCandidateGenerator
    scheduler: DeterministicSchedulerPort
    executor: SupervisorProbeExecutorPort
    validator: PassThroughValidatorPort
    reducer: ClaimGateReducerPort
    tool_ledger: ToolCallLedger
    branch_deriver: WorkflowBranchDeriver


def compose_exploration_workflow(
    *,
    provider: ExplorationProvider,
    journal: JsonlExplorationJournal,
    recovery: CompletedStepRecoveryPort,
    dataset_profiles: Sequence[DatasetExplorationProfile],
    tools: Sequence[AgentTool],
    tool_ledger: ToolCallLedger,
    scheduler_policy: SchedulerPolicy,
    admission_context: Callable[[PhaseContext], AdmissionContext],
    signals: Callable[[PhaseContext, tuple[CandidateSeed, ...]], Mapping[str, CandidateSignals]],
    llm_response_store: LlmResponseStore,
    tool_result_store: ToolResultStore,
    usage_meter: ToolUsageMeter | None = None,
    phase_tools: PhaseToolMap | None = None,
    goal_satisfied: Callable[[ExplorationWorkflowState], bool] = lambda _state: False,
    coverage_target_met: Callable[[ExplorationWorkflowState], bool] = lambda _state: False,
    max_probe_steps: int = 8,
    max_probe_tool_calls: int = 12,
    initial_state: ExplorationWorkflowState | None = None,
    persist_state: Callable[[ExplorationWorkflowState], None] = lambda _state: None,
    stat_attempt_counts: Callable[[], Mapping[str, int]] = lambda: {},
    goal: str | None = None,
    dataset_columns: Mapping[str, Iterable[str]] | None = None,
    supported_method_families: Iterable[str] = (),
) -> ExplorationWorkflowComponents:
    """Wire real E4a control components; only provider/tools/data signals are injected."""
    if not tools:
        raise ValueError("exploration workflow requires at least one read-only tool.")
    state = initial_state or ExplorationWorkflowState()
    probe_executor = ProbeExecutor(
        provider,
        tools,
        tool_ledger,
        phase_tools=(
            phase_tools or PhaseToolMap({"execute_probes": tuple(tool.name for tool in tools)})
        ),
        usage_meter=usage_meter,
        journal=JsonlProbeJournalHooks(journal),
        response_store=llm_response_store,
        tool_result_store=tool_result_store,
        max_steps=max_probe_steps,
        max_tool_calls=max_probe_tool_calls,
    )
    return ExplorationWorkflowComponents(
        state=state,
        generator=JournaledCandidateGenerator(
            provider=provider,
            journal=journal,
            recovery=recovery,
            dataset_profiles=tuple(dataset_profiles),
            goal=goal,
            dataset_columns={
                dataset_id: tuple(sorted(columns))
                for dataset_id, columns in (dataset_columns or {}).items()
            },
            supported_method_families=tuple(sorted(supported_method_families)),
            coverage_completed=lambda: frozenset(state.coverage_completed),
        ),
        scheduler=DeterministicSchedulerPort(
            policy=scheduler_policy,
            admission_context=admission_context,
            signals=signals,
            state=state,
            persist_state=persist_state,
            journal=journal,
        ),
        executor=SupervisorProbeExecutorPort(
            executor=probe_executor,
            journal=journal,
            state=state,
            receipt_decoder=artifact_receipt_decoder,
            persist_state=persist_state,
        ),
        validator=PassThroughValidatorPort(),
        reducer=ClaimGateReducerPort(
            journal=journal,
            state=state,
            goal_satisfied=goal_satisfied,
            coverage_target_met=coverage_target_met,
            persist_state=persist_state,
            stat_attempt_counts=stat_attempt_counts,
        ),
        tool_ledger=tool_ledger,
        branch_deriver=WorkflowBranchDeriver(
            journal=journal, recovery=recovery, state=state
        ),
    )


@dataclass(slots=True)
class DeterministicSchedulerPort:
    policy: SchedulerPolicy
    admission_context: Callable[[PhaseContext], AdmissionContext]
    signals: Callable[[PhaseContext, tuple[CandidateSeed, ...]], Mapping[str, CandidateSignals]]
    state: ExplorationWorkflowState
    persist_state: Callable[[ExplorationWorkflowState], None] = lambda _state: None
    journal: JsonlExplorationJournal | None = None

    def admit_and_score(self, context: PhaseContext, candidates: CandidateBatch) -> ScoredFrontier:
        typed = tuple(candidates.candidates)
        if not all(isinstance(candidate, CandidateSeed) for candidate in typed):
            raise SupervisorInvariantError("candidate batch contains a non-CandidateSeed.")
        seeds = tuple(candidate for candidate in typed if isinstance(candidate, CandidateSeed))
        context_value = self.admission_context(context)
        if self.journal is not None:
            journal_state = self.journal.rebuild()
            if journal_state is not None and journal_state.abandoned_constraints:
                # System-owned merge: abandonment facts always reach admission,
                # regardless of what the environment-specific builder supplies.
                context_value = replace(
                    context_value,
                    abandoned_constraints=(
                        *context_value.abandoned_constraints,
                        *journal_state.abandoned_constraints,
                    ),
                )
        result = schedule_candidates(
            seeds,
            signals=self.signals(context, seeds),
            context=context_value,
            policy=self.policy,
        )
        # Every round appends its whole batch, duplicates included: the round's
        # frontier digest covers all of its decisions, so replay reads them back
        # as one positional slice per round.
        self.state.decisions = (*self.state.decisions, *result.decisions)
        self.persist_state(self.state)
        by_id = {seed.hypothesis_id: seed for seed in seeds}
        decisions = {decision.hypothesis_id: decision for decision in result.decisions}
        items = tuple(
            FrontierItem(
                hypothesis_id=hypothesis_id,
                priority=decisions[hypothesis_id].priority,
                payload=replace(
                    by_id[hypothesis_id],
                    status="admitted",
                    priority=decisions[hypothesis_id].priority,
                ),
            )
            for hypothesis_id in result.chosen_hypothesis_ids
        )
        digest = scheduling_decision_digest(result.decisions)
        return ScoredFrontier(items=items, digest=f"frontier_{digest}")

    def select(self, context: PhaseContext, frontier: ScoredFrontier) -> ProbeSelection | None:
        del context
        return ProbeSelection(frontier.items) if frontier.items else None


def scheduling_decision_digest(decisions: Sequence[SchedulingDecision]) -> str:
    """Bind every replayable scheduler feature/check/choice to the frontier."""
    return stable_hash(
        [
            {
                "hypothesis_id": decision.hypothesis_id,
                "fingerprint": decision.hypothesis_fingerprint,
                "family": decision.family.value,
                "status": decision.status,
                "admission_checks": [
                    {
                        "name": check.name,
                        "passed": check.passed,
                        "detail_code": check.detail_code,
                    }
                    for check in decision.admission_checks
                ],
                "priority_features": decision.priority_features.model_dump(mode="json"),
                "priority": decision.priority,
                "scoring_policy_version": decision.scoring_policy_version,
                "quota_deferred": decision.quota_deferred,
                "chosen": decision.chosen,
            }
            for decision in decisions
        ],
        length=32,
    )


def _probe_outcome_is_usable(execution: ProbeExecutionResult) -> bool:
    """Whether the round may continue on what this probe already committed."""
    if execution.status in {"completed", "limit_reached"}:
        return True
    return (
        execution.status == "failed"
        and execution.error_code in PROBE_LOCAL_ERROR_CODES
    )


@dataclass(slots=True)
class SupervisorProbeExecutorPort:
    executor: ProbeExecutor
    journal: JsonlExplorationJournal
    state: ExplorationWorkflowState
    receipt_decoder: Callable[[object], EvidenceReceipt | None]
    persist_state: Callable[[ExplorationWorkflowState], None] = lambda _state: None

    def execute(self, context: PhaseContext, selection: ProbeSelection) -> ProbeOutcome:
        executions: list[ProbeExecutionResult] = []
        receipts: list[EvidenceReceipt] = []
        bindings: list[tuple[str, str]] = []
        for item in selection.items:
            if not isinstance(item.payload, CandidateSeed):
                raise SupervisorInvariantError("probe selection contains a non-candidate.")
            candidate = item.payload
            journal_state = self.journal.rebuild()
            if journal_state is None:
                raise SupervisorInvariantError("probe executor requires an initialized journal.")
            execution_run_id = (
                f"{context.exploration_id}:round:{context.round_index}:"
                f"hypothesis:{candidate.hypothesis_id}:execute_probes"
            )
            try:
                execution = self.executor.run(
                    phase="execute_probes",
                    system_prompt=(
                        "Execute only this falsifiable probe. Return a concise "
                        "summary after tools; claims are built and gated separately. "
                        + context.soft_countdown_context
                    ),
                    user_message=_candidate_prompt(candidate),
                    run_id=execution_run_id,
                    seen_probe_fingerprints=set(journal_state.completed_probe_fingerprints),
                    failure_history=journal_state.failure_history,
                    completed_step_ids=set(journal_state.completed_step_ids),
                    completed_response_digests=_completed_response_digests(self.journal),
                    blocked_llm_call_ids=set(journal_state.uncertain_call_ids),
                    hypothesis=HypothesisExecutionBinding(
                        hypothesis_id=candidate.hypothesis_id,
                        predicate=candidate.proposal.predicate,
                        method_family=candidate.proposal.method_family,
                        dataset_ids=candidate.proposal.dataset_ids,
                        columns=candidate.proposal.columns,
                    ),
                )
            except BudgetExceeded as exc:
                raise SupervisorBudgetExhausted(str(exc)) from exc
            except (SessionCancelled, CancellationError) as exc:
                # Cancellation is a decision, not a defect. Escaping as a
                # generic exception settled the run as `failed`.
                raise SupervisorCancelled(str(exc) or "probe cancelled") from exc
            # `limit_reached` means this probe used up its own conversation
            # step/tool budget — an expected outcome for an eager model, not a
            # control-plane fault. Its committed receipts are already journaled
            # and valid, so the round proceeds with the evidence it did gather.
            # `failed` with a probe-local code (truncation, content filter, an
            # unresponsive model) is the same shape. Every other `failed` is an
            # integrity fault and stays terminal.
            if not _probe_outcome_is_usable(execution):
                raise SupervisorInvariantError(
                    f"probe execution ended as {execution.status}: {execution.error or ''}"
                )
            executions.append(execution)
            execution_receipts = tuple(
                receipt
                for artifact in execution.artifacts
                if (receipt := self.receipt_decoder(artifact)) is not None
            )
            committed_state = self.journal.rebuild()
            if committed_state is None:  # pragma: no cover - checked above
                raise SupervisorInvariantError("exploration journal disappeared after probes.")
            committed_ids = set(committed_state.step_receipt_refs.values())
            for receipt in execution_receipts:
                if receipt.receipt_id not in committed_ids:
                    raise SupervisorInvariantError(
                        f"probe returned receipt {receipt.receipt_id!r} before journal commit."
                    )
                if not verify_receipt_digest(receipt):
                    raise SupervisorInvariantError(
                        f"probe returned receipt {receipt.receipt_id!r} with invalid digest."
                    )
                if receipt.data_state_witness != context.data_state_witness:
                    raise SupervisorInvariantError("probe receipt witness does not match the run.")
                if receipt.execution is None or receipt.execution.run_id != execution_run_id:
                    raise SupervisorInvariantError(
                        "probe receipt is not bound to its system-owned hypothesis run."
                    )
                self.state.committed_receipts[receipt.receipt_id] = receipt
                receipts.append(receipt)
                bindings.append((receipt.receipt_id, candidate.hypothesis_id))
        self.persist_state(self.state)
        return ProbeOutcome(
            ExecutedProbeBatch(
                selection,
                tuple(executions),
                tuple(receipts),
                tuple(bindings),
            )
        )


@dataclass(frozen=True, slots=True)
class PassThroughValidatorPort:
    def validate(self, context: PhaseContext, probes: ProbeOutcome) -> ValidationOutcome:
        del context
        if not isinstance(probes.payload, ExecutedProbeBatch):
            raise SupervisorInvariantError("validator requires an ExecutedProbeBatch.")
        return ValidationOutcome(probes.payload)


@dataclass(slots=True)
class ClaimGateReducerPort:
    journal: JsonlExplorationJournal
    state: ExplorationWorkflowState
    goal_satisfied: Callable[[ExplorationWorkflowState], bool] = lambda _state: False
    coverage_target_met: Callable[[ExplorationWorkflowState], bool] = lambda _state: False
    persist_state: Callable[[ExplorationWorkflowState], None] = lambda _state: None
    stat_attempt_counts: Callable[[], Mapping[str, int]] = lambda: {}

    def reduce_without_probes(
        self,
        context: PhaseContext,
        frontier: ScoredFrontier,
        *,
        logical_step_id: str,
    ) -> ReductionOutcome:
        """No probes ran, so no insight moves; only the round's decisions need
        binding to the journal through the frontier digest."""
        del context, logical_step_id
        return ReductionOutcome(
            transitions=(),
            frontier=frontier,
            ledger_digest=final_reduction_state_digest(self.state),
            goal_satisfied=self.goal_satisfied(self.state),
            coverage_target_met=self.coverage_target_met(self.state),
        )

    def reduce(
        self,
        context: PhaseContext,
        validated: ValidationOutcome,
        frontier: ScoredFrontier,
        *,
        logical_step_id: str,
    ) -> ReductionOutcome:
        del logical_step_id
        if not isinstance(validated.payload, ExecutedProbeBatch):
            raise SupervisorInvariantError("reducer requires an ExecutedProbeBatch.")
        candidates = {
            item.payload.hypothesis_id: item.payload
            for item in frontier.items
            if isinstance(item.payload, CandidateSeed)
        }
        transitions: list[str] = []
        admitted_before = set(self.state.admitted_bundles)
        journal_state = self.journal.rebuild()
        recorded_verdicts = {} if journal_state is None else dict(journal_state.gate_verdicts)
        bindings = dict(validated.payload.receipt_hypothesis_bindings)
        if len(bindings) != len(validated.payload.receipt_hypothesis_bindings):
            raise SupervisorInvariantError("one receipt has multiple hypothesis bindings.")
        receipts_by_hypothesis: dict[str, list[EvidenceReceipt]] = {}
        for receipt in validated.payload.receipts:
            hypothesis_id = bindings.get(receipt.receipt_id)
            if receipt.facts and hypothesis_id in candidates:
                receipts_by_hypothesis.setdefault(hypothesis_id, []).append(receipt)

        for hypothesis_id, current_receipts in receipts_by_hypothesis.items():
            candidate = candidates[hypothesis_id]
            insight_id = "ins_" + stable_hash(candidate.hypothesis_id, length=24)
            prior = self.state.insights.get(insight_id)
            prior_supporting = () if prior is None else prior.supporting_receipt_ids
            prior_contradicting = () if prior is None else prior.contradicting_receipt_ids
            new_supporting: list[str] = []
            new_contradicting: list[str] = []
            limitations: list[str] = []
            for receipt in current_receipts:
                statistics = receipt.statistics
                if (
                    statistics is not None
                    and statistics.hypothesis_outcome is not None
                    and statistics.hypothesis_id != candidate.hypothesis_id
                ):
                    raise SupervisorInvariantError(
                        "receipt hypothesis outcome conflicts with its execution binding."
                    )
                outcome = None if statistics is None else statistics.hypothesis_outcome
                # An unadjudicated receipt is a legitimate exploratory claim, so
                # it stays in the bundle; it is simply not evidence either way.
                if outcome == "supports":
                    new_supporting.append(receipt.receipt_id)
                elif outcome == "contradicts":
                    new_contradicting.append(receipt.receipt_id)
                limitations.extend(receipt.method.warnings)
                if statistics is None or statistics.hypothesis_outcome is None:
                    limitations.append(
                        "The probe produced evidence but did not deterministically "
                        "adjudicate the original hypothesis."
                    )

            supporting = _merge_ids(prior_supporting, new_supporting)
            contradicting = _merge_ids(prior_contradicting, new_contradicting)
            cumulative_receipts = tuple(
                self.state.committed_receipts[receipt_id]
                for receipt_id in _merge_ids(
                    (*prior_supporting, *prior_contradicting),
                    [receipt.receipt_id for receipt in current_receipts],
                )
            )
            bundle = _claim_bundle(candidate, cumulative_receipts)
            report = run_claim_gates(
                bundle,
                committed_receipts=self.state.committed_receipts,
                run_witness=context.data_state_witness,
                stat_attempt_counts=self.stat_attempt_counts(),
            )
            verdict = "passed" if report.passed else "rejected"
            # A crash between the verdict and reduction_committed replays this
            # whole phase; a duplicate verdict event would desynchronize the
            # issuer's positional gate cursor for good.
            if recorded_verdicts.get(bundle.claim_bundle_id) != verdict:
                self.journal.append_new(
                    "gate_verdict",
                    claim_bundle_id=bundle.claim_bundle_id,
                    verdict=verdict,
                )
                recorded_verdicts[bundle.claim_bundle_id] = verdict
            self.state.gate_reports[bundle.claim_bundle_id] = report
            if not report.passed:
                continue
            self.state.admitted_bundles[bundle.claim_bundle_id] = bundle
            proposal = TransitionProposal(
                hypothesis_id=candidate.hypothesis_id,
                insight_id=insight_id,
                family=candidate.proposal.family,
                claim_bundle_id=bundle.claim_bundle_id,
                supporting_receipt_ids=supporting,
                contradicting_receipt_ids=contradicting,
                limitations=tuple(dict.fromkeys(limitations)),
                proposed_status="new",
            )
            if (
                prior is not None
                and prior.claim_bundle_id == proposal.claim_bundle_id
                and set(proposal.supporting_receipt_ids).issubset(prior.supporting_receipt_ids)
                and set(proposal.contradicting_receipt_ids).issubset(
                    prior.contradicting_receipt_ids
                )
            ):
                # Exact replay after a crash between workflow-state persistence
                # and reduction_committed adopts the prior result. It must not
                # manufacture a second, supposedly independent reinforcement.
                insight = prior
            elif not new_supporting and not new_contradicting:
                # The round produced evidence but adjudicated nothing; the
                # bundle admission is the only ledger movement it earned.
                continue
            else:
                insight = reduce_insight(
                    proposal,
                    prior=prior,
                    committed_receipts=self.state.committed_receipts,
                    admitted_claim_bundles=self.state.admitted_bundles,
                    expected_witness=context.data_state_witness,
                    round_index=context.round_index,
                    require_typed_hypothesis_outcome=True,
                    statement=candidate.proposal.statement,
                    rationale=candidate.proposal.rationale,
                )
            self.state.insights[insight.insight_id] = insight
            transitions.append(insight.status)
            self.state.coverage_completed.add(candidate.coverage_key)
        self.persist_state(self.state)
        return ReductionOutcome(
            transitions=tuple(transitions),  # type: ignore[arg-type]
            frontier=frontier,
            ledger_digest=final_reduction_state_digest(self.state),
            goal_satisfied=self.goal_satisfied(self.state),
            coverage_target_met=self.coverage_target_met(self.state),
            admitted_bundle_count=len(set(self.state.admitted_bundles) - admitted_before),
        )


def artifact_receipt_decoder(value: object) -> EvidenceReceipt | None:
    if not isinstance(value, Artifact) or value.type is not ArtifactType.EVIDENCE_RECEIPT:
        return None
    receipt = EvidenceReceipt.model_validate(value.payload)
    valid_artifact_ids = {
        receipt.receipt_id,  # legacy/test adapters used the receipt id directly
        make_artifact_id("receipt", value.payload),
    }
    if value.id not in valid_artifact_ids:
        raise ValueError("receipt artifact id does not match its content-addressed payload.")
    return receipt


def candidate_batch_digest(batch: CandidateBatch) -> str:
    return stable_hash(
        [
            {
                "proposal": candidate.proposal.model_dump(mode="json"),
                "hypothesis_id": candidate.hypothesis_id,
                "hypothesis_fingerprint": candidate.hypothesis_fingerprint,
                "canonical_group_key": candidate.canonical_group_key,
                "coverage_key": candidate.coverage_key,
                "sequence_index": candidate.sequence_index,
                "status": candidate.status,
                "origin": candidate.origin,
                "mandatory": candidate.mandatory,
                "priority": candidate.priority,
            }
            for candidate in batch.candidates
            if isinstance(candidate, CandidateSeed)
        ],
        length=24,
    )


def final_reduction_state_digest(state: ExplorationWorkflowState) -> str:
    """Bind every reducer-owned terminal projection into one journal digest."""
    return "ledger_" + stable_hash(
        {
            "insights": [
                item.model_dump(mode="json")
                for item in sorted(state.insights.values(), key=lambda value: value.insight_id)
            ],
            "admitted_bundles": [
                item.model_dump(mode="json")
                for item in sorted(
                    state.admitted_bundles.values(),
                    key=lambda value: value.claim_bundle_id,
                )
            ],
            "gate_reports": [
                item.model_dump(mode="json")
                for item in sorted(
                    state.gate_reports.values(),
                    key=lambda value: value.claim_bundle_id,
                )
            ],
            "coverage_completed": sorted(state.coverage_completed),
        },
        length=32,
    )


def _completed_response_digests(
    journal: JsonlExplorationJournal,
) -> dict[str, str]:
    return {
        event.step_id: event.response_digest
        for event in journal.events()
        if isinstance(event, LlmCallCompletedEvent)
    }


def _candidate_prompt(value: object) -> str:
    if not isinstance(value, CandidateSeed):
        raise SupervisorInvariantError("frontier payload is not a CandidateSeed.")
    proposal = value.proposal
    return (
        f"hypothesis_id={value.hypothesis_id}; statement={proposal.statement}; "
        f"expected_evidence={proposal.expected_evidence}; "
        f"falsification={'; '.join(proposal.falsification_conditions)}; "
        f"predicate={proposal.predicate.model_dump_json()}; "
        f"datasets={','.join(proposal.dataset_ids)}; columns={','.join(proposal.columns)}"
    )


def _claim_bundle(candidate: CandidateSeed, receipts: Sequence[EvidenceReceipt]) -> ClaimBundle:
    claims = tuple(
        Claim(
            claim_id="clm_"
            + stable_hash({"receipt_id": receipt.receipt_id, "fact_id": fact.fact_id}, length=20),
            claim_type="absence" if fact.support_type == "absence" else "observation",
            claim_text=f"{fact.name}: {_fact_text(fact.value, fact.value_type)}",
            support_type=fact.support_type,
            evidence_fact_ids=(f"{receipt.receipt_id}:{fact.fact_id}",),
            statistics_receipt_ids=(
                (receipt.receipt_id,) if _has_confirmatory_statistics(receipt) else ()
            ),
            uncertainty=("; ".join(receipt.method.warnings) or None),
            limitations=tuple(receipt.method.warnings),
            scope=(
                _absence_scope(receipt) if fact.support_type == "absence" else None
            ),
        )
        for receipt in receipts
        for fact in receipt.facts
        if not (
            isinstance(fact.value, str) and any(character.isdigit() for character in fact.value)
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
            if receipts and all(_has_confirmatory_statistics(receipt) for receipt in receipts)
            else "exploratory"
        ),
        claims=claims,
    )


def _absence_scope(receipt: EvidenceReceipt) -> ClaimScope:
    """An absence claim may only declare what its own receipt actually scanned,
    so the scope is copied verbatim — the gate compares the two for identity."""
    return ClaimScope(
        dataset_ids=receipt.scope.dataset_ids,
        columns=receipt.scope.columns,
        filters=receipt.scope.filters,
        time_range=receipt.scope.time_range,
    )


def _merge_ids(existing: Sequence[str], incoming: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys((*existing, *incoming)))


def _fact_text(value: object, value_type: str) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if value_type == "percent":
        return f"{value}%"
    return str(value)


def _has_confirmatory_statistics(receipt: EvidenceReceipt) -> bool:
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


def _safe_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {' '.join(str(exc).split())}"[:800]
