"""E4a shadow exploration lifecycle and durable adapter wiring.

This driver is deliberately product-store blind.  Its only writable surface is
``<workspace>/exploration-eval/<exploration_id>``; callers inject analysis ports,
while this module owns the execution lock, recovery epoch, journal projection,
durable phase/LLM response bodies, and shadow projection.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from eda_platform.agents.exploration.candidates import (
    CandidateSeed,
    DatasetExplorationProfile,
    mandatory_probe_seeds,
)
from eda_platform.agents.exploration.executor import (
    DurableToolResult,
    ToolUsageMeter,
    durable_tool_result_digest,
    durable_tool_result_payload,
)
from eda_platform.agents.exploration.scheduler import (
    AdmissionCheck,
    AdmissionContext,
    CandidateSignals,
    PriorityFeatures,
    SchedulerPolicy,
    SchedulingDecision,
)
from eda_platform.agents.exploration.supervisor import (
    BudgetPort,
    CandidateBatch,
    CandidateGeneratorPort,
    CompletedStepRecoveryPort,
    ControlPort,
    ExplorationSupervisor,
    FinalizationOutcome,
    FinalizerPort,
    FrontierItem,
    PhaseContext,
    PhaseTransition,
    ProbeExecutorPort,
    ReducerPort,
    ReductionOutcome,
    SchedulerPort,
    ScoredFrontier,
    SupervisorBudgetExhausted,
    SupervisorConfig,
    SupervisorJournalState,
    SupervisorRunResult,
    SynthesisUnavailable,
    ValidatorPort,
    WitnessPort,
)
from eda_platform.agents.exploration.workflow import (
    ExplorationProvider,
    ExplorationWorkflowState,
    candidate_batch_digest,
    compose_exploration_workflow,
    scheduling_decision_digest,
)
from eda_platform.agents.runtime import AgentTool, AgentToolResult
from eda_platform.core.budget import SessionBudgetState
from eda_platform.core.claim_gates import GateReport, claim_bundle_digest
from eda_platform.core.config import require_absolute_workspace
from eda_platform.core.exploration_budget import (
    ToolCallLedger,
    ToolCallProjection,
    apply_budget_increase,
)
from eda_platform.core.exploration_journal import JsonlExplorationJournal, RecoveredToolCommit
from eda_platform.core.exploration_report import render_exploration_report
from eda_platform.core.exploration_shadow_store import (
    ShadowExplorationStore,
    shadow_run_root,
    validate_shadow_run_path,
)
from eda_platform.core.fs import BINARY_FLAG
from eda_platform.core.ids import stable_hash
from eda_platform.core.llm import LLMClient, LLMToolResponse
from eda_platform.core.llm_ledger import (
    LedgerLLMClient,
    budget_policy_fingerprint,
    restore_run_budget_state,
)
from eda_platform.schemas.artifacts import Artifact
from eda_platform.schemas.claims import ClaimBundle
from eda_platform.schemas.exploration import (
    BudgetAmendedEvent,
    ExplorationGracefulStopReason,
    ExplorationLoopState,
    ExplorationPolicy,
    InsightFamily,
    LlmCallCompletedEvent,
    ReceiptCommittedEvent,
    ReductionCommittedEvent,
    RoundStartedEvent,
)
from eda_platform.schemas.exploration_budget import ExplorationBudgetPolicy
from eda_platform.schemas.exploration_shadow import ShadowExplorationProjection
from eda_platform.schemas.hypotheses import HypothesisProposal
from eda_platform.schemas.insights import InsightRecord
from eda_platform.schemas.receipts import EvidenceReceipt, verify_receipt_digest
from eda_platform.schemas.sessions import TraceEvent


@dataclass(frozen=True, slots=True)
class ShadowProjectionData:
    """Evaluation-only material made visible in the shadow projection."""

    insight_records: tuple[InsightRecord, ...] = ()
    coverage_completed: tuple[str, ...] = ()
    coverage_unexplored: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ShadowExplorationRunResult:
    result: SupervisorRunResult
    journal_path: Path
    projection_path: Path


@dataclass(slots=True)
class JournalControlPort:
    """Optional side-effect-free checkpoint hook used for cancellation polling."""

    callback: Callable[[PhaseTransition], object] | None = None

    def checkpoint(self, transition: PhaseTransition) -> None:
        if self.callback is not None:
            self.callback(transition)


@dataclass(frozen=True, slots=True)
class CallableWitnessPort:
    callback: Callable[[str], bool]

    def recheck(self, expected_witness: str) -> bool:
        return bool(self.callback(expected_witness))


@dataclass(frozen=True, slots=True)
class JournalBudgetPort:
    """Journal-backed countdown plus an optional wall/idle/session hard check."""

    hard_check: Callable[[SupervisorJournalState], object] | None = None
    extra_remaining: Callable[[], Mapping[str, object]] | None = None

    def check_start_round(self, state: SupervisorJournalState) -> None:
        if self.hard_check is not None:
            self.hard_check(state)

    def remaining(self, state: SupervisorJournalState) -> Mapping[str, object]:
        values: dict[str, object] = {
            "llm_requests": state.remaining_llm_call_budget,
            "rounds": state.remaining_round_budget,
            "successful_tools": state.remaining_tool_call_budget,
        }
        if self.extra_remaining is not None:
            for key, value in self.extra_remaining().items():
                if key in values:
                    raise ValueError(f"duplicate budget countdown key {key!r}.")
                values[key] = value
        return values


class JsonlShadowBudgetStore:
    """Durable, shadow-local ledger events consumed by ``LedgerLLMClient``.

    A final unterminated record is ignored on restore.  The provider wrapper
    persists a reservation before network I/O, so an interrupted append fails
    the call before it can spend; an interrupted terminal append is restored as
    an uncertain reservation and charged conservatively.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def events(self) -> list[TraceEvent]:
        try:
            payload = _read_bytes_no_follow(self.path)
        except FileNotFoundError:
            return []
        complete = payload if payload.endswith(b"\n") else payload.rpartition(b"\n")[0]
        if not complete:
            return []
        events: list[TraceEvent] = []
        for line_number, line in enumerate(complete.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                events.append(TraceEvent.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(
                    f"invalid shadow budget event at line {line_number}: {exc}"
                ) from exc
        return events

    def append(self, event: TraceEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink():
            raise ValueError("shadow budget event path cannot be a symlink.")
        self._truncate_unterminated_tail()
        body = event.model_dump_json().encode("utf-8") + b"\n"
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | BINARY_FLAG
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.path, flags, 0o600)
        try:
            view = memoryview(body)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:  # pragma: no cover - defensive OS contract
                    raise OSError("failed to append the shadow budget event.")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(self.path.parent)

    def _truncate_unterminated_tail(self) -> None:
        flags = os.O_RDWR | os.O_CREAT | BINARY_FLAG
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.path, flags, 0o600)
        try:
            size = os.lseek(descriptor, 0, os.SEEK_END)
            if size == 0:
                return
            os.lseek(descriptor, size - 1, os.SEEK_SET)
            if os.read(descriptor, 1) == b"\n":
                return
            cursor = size
            truncate_at = 0
            while cursor > 0:
                start = max(0, cursor - 4096)
                os.lseek(descriptor, start, os.SEEK_SET)
                block = os.read(descriptor, cursor - start)
                newline = block.rfind(b"\n")
                if newline >= 0:
                    truncate_at = start + newline + 1
                    break
                cursor = start
            os.ftruncate(descriptor, truncate_at)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class JsonExplorationWorkflowStateStore:
    """Atomic typed snapshot for reducer facts; the journal remains lifecycle authority."""

    _KEYS = frozenset(
        {
            "schema_version",
            "decisions",
            "committed_receipts",
            "gate_reports",
            "admitted_bundles",
            "insights",
            "coverage_completed",
        }
    )

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def load(self) -> ExplorationWorkflowState:
        if self.path.is_symlink():
            raise ValueError("workflow-state path cannot be a symlink.")
        try:
            raw = json.loads(_read_bytes_no_follow(self.path))
        except FileNotFoundError:
            return ExplorationWorkflowState()
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid workflow-state snapshot: {exc}") from exc
        legacy_keys = self._KEYS - {"decisions"}
        if not isinstance(raw, dict) or set(raw) not in {self._KEYS, legacy_keys}:
            raise ValueError("workflow-state snapshot has an unsupported shape.")
        if raw.get("schema_version") not in {1, 2}:
            raise ValueError("workflow-state snapshot has an unsupported version.")
        receipts = _typed_items(raw.get("committed_receipts"), EvidenceReceipt)
        reports = _typed_items(raw.get("gate_reports"), GateReport)
        bundles = _typed_items(raw.get("admitted_bundles"), ClaimBundle)
        insights = _typed_items(raw.get("insights"), InsightRecord)
        coverage = raw.get("coverage_completed")
        if not isinstance(coverage, list) or not all(
            isinstance(item, str) and item for item in coverage
        ):
            raise ValueError("workflow-state coverage must be a non-empty string list.")
        state = ExplorationWorkflowState(
            decisions=tuple(
                _decode_scheduling_decision(item)
                for item in _required_list(raw.get("decisions", []), "decisions")
            ),
            committed_receipts={item.receipt_id: item for item in receipts},
            gate_reports={item.claim_bundle_id: item for item in reports},
            admitted_bundles={item.claim_bundle_id: item for item in bundles},
            insights={item.insight_id: item for item in insights},
            coverage_completed=set(coverage),
        )
        if len(state.committed_receipts) != len(receipts):
            raise ValueError("workflow-state receipt ids must be unique.")
        if len(state.gate_reports) != len(reports):
            raise ValueError("workflow-state gate report ids must be unique.")
        if len(state.admitted_bundles) != len(bundles):
            raise ValueError("workflow-state claim bundle ids must be unique.")
        if len(state.insights) != len(insights):
            raise ValueError("workflow-state insight ids must be unique.")
        return state

    def remember(self, state: ExplorationWorkflowState) -> None:
        _write_json_atomic(
            self.path,
            {
                "schema_version": 2,
                "decisions": [
                    _encode_scheduling_decision(item) for item in state.decisions
                ],
                "committed_receipts": [
                    item.model_dump(mode="json")
                    for item in sorted(
                        state.committed_receipts.values(),
                        key=lambda value: value.receipt_id,
                    )
                ],
                "gate_reports": [
                    item.model_dump(mode="json")
                    for item in sorted(
                        state.gate_reports.values(),
                        key=lambda value: value.claim_bundle_id,
                    )
                ],
                "admitted_bundles": [
                    item.model_dump(mode="json")
                    for item in sorted(
                        state.admitted_bundles.values(),
                        key=lambda value: value.claim_bundle_id,
                    )
                ],
                "insights": [
                    item.model_dump(mode="json")
                    for item in sorted(
                        state.insights.values(), key=lambda value: value.insight_id
                    )
                ],
                "coverage_completed": sorted(state.coverage_completed),
            },
        )


@dataclass(frozen=True, slots=True)
class ShadowBudgetRuntime:
    """Crash-restored LLM/tool ledgers plus the supervisor budget projection."""

    provider: LedgerLLMClient
    llm_state: SessionBudgetState
    tool_ledger: ToolCallLedger
    budget_port: JournalBudgetPort
    event_store: JsonlShadowBudgetStore
    effective_policy: ExplorationBudgetPolicy


@dataclass(frozen=True, slots=True)
class ShadowExecutionComponents:
    """Ports constructed after recovery while the run execution lock is held."""

    generator: CandidateGeneratorPort
    scheduler: SchedulerPort
    executor: ProbeExecutorPort
    validator: ValidatorPort
    reducer: ReducerPort
    finalizer: FinalizerPort
    budget: BudgetPort
    projection_data: Callable[[], ShadowProjectionData]


@dataclass(frozen=True, slots=True)
class DeterministicShadowFinalizer:
    """Render only gate-passed bundles into a shadow-local report artifact."""

    workspace: Path
    exploration_id: str
    state: ExplorationWorkflowState
    journal: JsonlExplorationJournal
    coverage_targets: tuple[str, ...]
    budget_summary: Callable[[], Mapping[str, object]]

    def synthesize(
        self,
        context: Any,
        reduction: ReductionOutcome,
        *,
        logical_step_id: str,
    ) -> FinalizationOutcome:
        del context, reduction, logical_step_id
        raise SynthesisUnavailable("deterministic shadow rendering is configured.")

    def render_deterministic(
        self,
        context: Any,
        reduction: ReductionOutcome,
    ) -> FinalizationOutcome:
        del reduction
        journal_state = self.journal.rebuild()
        if journal_state is None or journal_state.pending_terminal_reason is None:
            raise ValueError("deterministic finalization requires a durable stop decision.")
        rendered = render_exploration_report(
            self.state,
            run_metadata={
                "exploration_id": self.exploration_id,
                "policy_fingerprint": journal_state.effective_policy_fingerprint,
                "witness": str(context.data_state_witness),
            },
            coverage_targets=self.coverage_targets,
            budget_summary=self.budget_summary(),
            stop_reason=journal_state.pending_terminal_reason,
        )
        path = validate_shadow_run_path(
            self.workspace,
            self.exploration_id,
            shadow_run_root(self.workspace, self.exploration_id) / "report.md",
        )
        _write_text_atomic(path, rendered.markdown)
        return FinalizationOutcome(str(path.relative_to(self.workspace)))


def build_shadow_budget_runtime(
    *,
    exploration_id: str,
    provider: LLMClient,
    policy: ExplorationPolicy,
    journal: JsonlExplorationJournal,
    event_store: JsonlShadowBudgetStore,
) -> ShadowBudgetRuntime:
    """Restore every hard budget dimension and meter all future provider calls."""
    expected_event_path = journal.path.parent / "llm-budget.jsonl"
    if event_store.path != expected_event_path:
        raise ValueError(
            "shadow LLM budget events must use <run-root>/llm-budget.jsonl."
        )
    effective_budget = policy.budget
    accepted_fingerprints = {budget_policy_fingerprint(effective_budget.llm.to_policy())}
    journal_events = journal.events()
    for event in journal_events:
        if isinstance(event, BudgetAmendedEvent):
            effective_budget = apply_budget_increase(effective_budget, event.increase)
            accepted_fingerprints.add(
                budget_policy_fingerprint(effective_budget.llm.to_policy())
            )
    journal_state = journal.rebuild()
    tool_ledger = (
        ToolCallLedger(effective_budget)
        if journal_state is None
        else ToolCallLedger.restore_from_journal_state(effective_budget, journal_state)
    )
    run_started_at = journal_events[0].occurred_at if journal_events else None
    llm_state = restore_run_budget_state(
        effective_budget.llm.to_policy(),
        event_store.events(),
        run_started_at=run_started_at,
        accepted_policy_fingerprints=frozenset(accepted_fingerprints),
    )

    def hard_check(_state: SupervisorJournalState) -> None:
        try:
            llm_state.check_wall_time()
        except Exception as exc:
            raise SupervisorBudgetExhausted(str(exc)) from exc
        events = journal.events()
        if not events:
            return
        idle_seconds = max(
            0.0,
            (datetime.now(UTC) - events[-1].occurred_at).total_seconds(),
        )
        if idle_seconds > effective_budget.idle_timeout_seconds:
            raise SupervisorBudgetExhausted(
                "idle timeout exhausted: "
                f"{idle_seconds:.3f}s > {effective_budget.idle_timeout_seconds:.3f}s"
            )

    def remaining() -> Mapping[str, object]:
        values: dict[str, object] = {
            f"physical_llm_{dimension}": llm_state.remaining(dimension)
            for dimension in (
                "requests",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "cost_usd",
                "wall_seconds",
            )
        }
        values.update(
            {f"tool_{key}": value for key, value in tool_ledger.remaining().items()}
        )
        return values

    metered = LedgerLLMClient(
        provider,
        session_id=exploration_id,
        emit=event_store.append,
        budget=llm_state,
    )
    return ShadowBudgetRuntime(
        provider=metered,
        llm_state=llm_state,
        tool_ledger=tool_ledger,
        budget_port=JournalBudgetPort(hard_check=hard_check, extra_remaining=remaining),
        event_store=event_store,
        effective_policy=effective_budget,
    )


def exploration_tool_capability_digest(tools: Sequence[AgentTool]) -> str:
    """Bind approval to the exact, read-only tool schemas exposed to the model."""
    names = [tool.name for tool in tools]
    if not names or len(names) != len(set(names)):
        raise ValueError("exploration tools must be non-empty and uniquely named.")
    return "xpltools_" + stable_hash(
        [
            tool.provider_schema()
            for tool in sorted(tools, key=lambda item: item.name)
        ],
        length=32,
    )


def run_composed_shadow_exploration(
    *,
    workspace: Path | str,
    exploration_id: str,
    policy: ExplorationPolicy,
    code_fingerprint: str,
    data_state_witness: str,
    provider: LLMClient,
    tools: Sequence[AgentTool],
    dataset_profiles: Sequence[DatasetExplorationProfile],
    scheduler_policy: SchedulerPolicy,
    admission_context: Callable[[PhaseContext], AdmissionContext],
    signals: Callable[
        [PhaseContext, tuple[CandidateSeed, ...]], Mapping[str, CandidateSignals]
    ],
    witness: WitnessPort,
    usage_meter: ToolUsageMeter | None = None,
    stat_attempt_counts: Callable[[], Mapping[str, int]] = lambda: {},
    goal_satisfied: Callable[[ExplorationWorkflowState], bool] = lambda _state: False,
    coverage_target_met: Callable[
        [ExplorationWorkflowState], bool
    ] = lambda _state: False,
    admission_score_threshold: float | None = None,
) -> ShadowExplorationRunResult:
    """Official E4a composition root: real ports, durable ledgers, shadow-only sinks."""
    workspace_path = require_absolute_workspace(Path(workspace))
    run_root = shadow_run_root(workspace_path, exploration_id)
    tool_tuple = tuple(tools)
    unknown_tools = sorted(
        set(tool.name for tool in tool_tuple)
        - set(policy.budget.max_tool_calls_by_kind)
    )
    if unknown_tools:
        raise ValueError(
            "exploration tool inventory contains capabilities outside the sealed policy: "
            + ", ".join(unknown_tools)
        )
    actual_tool_digest = exploration_tool_capability_digest(tool_tuple)
    if actual_tool_digest != policy.tool_capability_digest:
        raise ValueError("sealed tool_capability_digest does not match the tool inventory.")
    if usage_meter is None:
        raise ValueError(
            "official exploration composition requires a trusted ToolUsageMeter; "
            "unknown row/cell usage cannot be recorded as zero."
        )

    journal = JsonlExplorationJournal(run_root / "journal.jsonl")
    recovery = JsonSupervisorRecoveryStore(run_root / "phase-responses")
    llm_response_store = JsonLlmResponseStore(run_root / "llm-responses")
    tool_result_store = JsonToolResultStore(run_root / "tool-results")
    workflow_state_store = JsonExplorationWorkflowStateStore(
        run_root / "workflow-state.json"
    )
    mandatory_coverage = frozenset(
        seed.coverage_key for seed in mandatory_probe_seeds(dataset_profiles)
    )

    def component_factory(
        recovered_journal_state: ExplorationLoopState,
    ) -> ShadowExecutionComponents:
        # This callback runs only after claim_recovery and while the executor
        # lock is held. Every durable projection therefore observes the latest
        # predecessor attempt rather than a snapshot loaded before serialization.
        initial_workflow_state = workflow_state_store.load()
        if (
            recovered_journal_state.last_seq == 0
            and (
                initial_workflow_state.decisions
                or initial_workflow_state.committed_receipts
                or initial_workflow_state.gate_reports
                or initial_workflow_state.admitted_bundles
                or initial_workflow_state.insights
                or initial_workflow_state.coverage_completed
            )
        ):
            raise ValueError("workflow-state exists without its authoritative journal.")
        journal_receipts = set(recovered_journal_state.step_receipt_refs.values())
        for step_id, receipt_id in recovered_journal_state.step_receipt_refs.items():
            expected_digest = recovered_journal_state.step_result_digests.get(step_id)
            recovered_tool = tool_result_store.recovered_commit_if_present(step_id)
            if (
                expected_digest is None
                or recovered_tool is None
                or recovered_tool.receipt_id != receipt_id
                or recovered_tool.result_digest != expected_digest
            ):
                raise ValueError(
                    "journal-committed tool result is missing or fails its body digest."
                )
        if not set(initial_workflow_state.committed_receipts).issubset(journal_receipts):
            raise ValueError("workflow-state cites receipts absent from the journal.")
        if not set(initial_workflow_state.gate_reports).issubset(
            recovered_journal_state.gate_verdicts
        ):
            raise ValueError("workflow-state cites gate verdicts absent from the journal.")
        for receipt_id, receipt in initial_workflow_state.committed_receipts.items():
            if (
                receipt_id != receipt.receipt_id
                or not verify_receipt_digest(receipt)
                or receipt.data_state_witness
                != recovered_journal_state.data_state_witness
            ):
                raise ValueError("workflow-state contains an invalid committed receipt.")
        for bundle_id, report in initial_workflow_state.gate_reports.items():
            expected_verdict = recovered_journal_state.gate_verdicts[bundle_id]
            if expected_verdict != ("passed" if report.passed else "rejected"):
                raise ValueError("workflow-state gate report conflicts with the journal.")
            if report.run_witness != recovered_journal_state.data_state_witness:
                raise ValueError("workflow-state gate report has a different witness.")
        for bundle_id, bundle in initial_workflow_state.admitted_bundles.items():
            report = initial_workflow_state.gate_reports.get(bundle_id)
            if (
                report is None
                or not report.passed
                or report.claim_bundle_digest != claim_bundle_digest(bundle)
            ):
                raise ValueError(
                    "workflow-state admitted bundle lacks its exact passed gate report."
                )
        active_round: int | None = None
        response_digests = _completed_response_digests(journal)
        for event in journal.events():
            if isinstance(event, RoundStartedEvent):
                active_round = event.round_index
                continue
            if not isinstance(event, ReductionCommittedEvent):
                continue
            if active_round is None:
                raise ValueError("reduction commit has no active scheduler round.")
            step_id = f"{exploration_id}:round:{active_round}:generate"
            batch = recovery.load_required(step_id)
            if not isinstance(batch, CandidateBatch):
                raise ValueError("scheduler decision recovery body is not a candidate batch.")
            expected_batch_digest = response_digests.get(step_id)
            if (
                expected_batch_digest is None
                or candidate_batch_digest(batch) != expected_batch_digest
            ):
                raise ValueError("scheduler candidate batch fails its journal digest.")
            candidate_ids = {
                candidate.hypothesis_id
                for candidate in batch.candidates
                if isinstance(candidate, CandidateSeed)
            }
            round_decisions = tuple(
                decision
                for decision in initial_workflow_state.decisions
                if decision.hypothesis_id in candidate_ids
            )
            expected_frontier_digest = (
                "frontier_" + scheduling_decision_digest(round_decisions)
            )
            if event.frontier_digest != expected_frontier_digest:
                raise ValueError(
                    "workflow-state scheduler decisions fail their journal frontier digest."
                )

        budget_runtime = build_shadow_budget_runtime(
            exploration_id=exploration_id,
            provider=provider,
            policy=policy,
            journal=journal,
            event_store=JsonlShadowBudgetStore(run_root / "llm-budget.jsonl"),
        )
        workflow = compose_exploration_workflow(
            provider=cast(ExplorationProvider, budget_runtime.provider),
            journal=journal,
            recovery=recovery,
            dataset_profiles=dataset_profiles,
            tools=tool_tuple,
            tool_ledger=budget_runtime.tool_ledger,
            scheduler_policy=scheduler_policy,
            admission_context=admission_context,
            signals=signals,
            llm_response_store=llm_response_store,
            tool_result_store=tool_result_store,
            usage_meter=usage_meter,
            goal_satisfied=goal_satisfied,
            coverage_target_met=coverage_target_met,
            initial_state=initial_workflow_state,
            persist_state=workflow_state_store.remember,
            stat_attempt_counts=stat_attempt_counts,
            goal=policy.goal,
        )

        def report_budget_summary() -> Mapping[str, object]:
            state = journal.rebuild()
            tools_used = budget_runtime.tool_ledger.snapshot()
            return {
                "llm_requests_used": budget_runtime.llm_state.requests_used,
                "llm_total_tokens_used": budget_runtime.llm_state.total_tokens_used,
                "llm_cost_usd_used": float(budget_runtime.llm_state.cost_usd_used),
                "successful_tool_calls": tools_used["successful_tool_calls"],
                "rows_scanned": tools_used["rows_scanned"],
                "result_cells": tools_used["result_cells"],
                "rounds_started": 0 if state is None else state.rounds_started,
                "max_llm_requests": budget_runtime.effective_policy.llm.max_requests,
                "max_cost_usd": (
                    None
                    if budget_runtime.effective_policy.llm.max_cost_usd is None
                    else float(budget_runtime.effective_policy.llm.max_cost_usd)
                ),
                "max_successful_tool_calls": (
                    budget_runtime.effective_policy.max_successful_tool_calls
                ),
                "max_rounds": budget_runtime.effective_policy.max_rounds,
            }

        finalizer = DeterministicShadowFinalizer(
            workspace=workspace_path,
            exploration_id=exploration_id,
            state=workflow.state,
            journal=journal,
            coverage_targets=tuple(sorted(mandatory_coverage)),
            budget_summary=report_budget_summary,
        )

        def projection_data() -> ShadowProjectionData:
            state = journal.rebuild()
            terminal_insights = (
                tuple(
                    sorted(
                        workflow.state.insights.values(),
                        key=lambda item: item.insight_id,
                    )
                )
                if state is not None and state.status == "stopped"
                else ()
            )
            return ShadowProjectionData(
                insight_records=terminal_insights,
                coverage_completed=tuple(sorted(workflow.state.coverage_completed)),
                coverage_unexplored=tuple(
                    sorted(mandatory_coverage - workflow.state.coverage_completed)
                ),
            )

        return ShadowExecutionComponents(
            generator=workflow.generator,
            scheduler=workflow.scheduler,
            executor=workflow.executor,
            validator=workflow.validator,
            reducer=workflow.reducer,
            finalizer=finalizer,
            budget=budget_runtime.budget_port,
            projection_data=projection_data,
        )

    threshold = (
        scheduler_policy.admission_priority
        if admission_score_threshold is None
        else admission_score_threshold
    )
    return run_shadow_exploration(
        workspace=workspace_path,
        exploration_id=exploration_id,
        policy=policy,
        code_fingerprint=code_fingerprint,
        data_state_witness=data_state_witness,
        config=SupervisorConfig(admission_score_threshold=threshold),
        witness=witness,
        journal=journal,
        recovery=recovery,
        llm_response_store=llm_response_store,
        tool_result_store=tool_result_store,
        component_factory=component_factory,
    )


@dataclass(frozen=True, slots=True)
class JsonlSupervisorJournalAdapter:
    """Project the generic JSONL journal into the supervisor's narrow protocol."""

    journal: JsonlExplorationJournal

    def snapshot(self) -> SupervisorJournalState:
        state = self.journal.rebuild()
        if state is None:
            raise RuntimeError("initialize the exploration journal before supervision.")
        return _supervisor_projection(self.journal, state)

    def start_round(self, round_index: int) -> SupervisorJournalState:
        self.journal.append_new("round_started", round_index=round_index)
        return self.snapshot()

    def commit_reduction(
        self, *, frontier_digest: str, ledger_digest: str, reduction_digest: str
    ) -> SupervisorJournalState:
        self.journal.append_new(
            "reduction_committed",
            frontier_digest=frontier_digest,
            ledger_digest=ledger_digest,
            reduction_digest=reduction_digest,
        )
        return self.snapshot()

    def settle_round(
        self,
        round_index: int,
        *,
        progress: bool,
        terminal_reason: ExplorationGracefulStopReason | None,
    ) -> SupervisorJournalState:
        state = self.journal.rebuild()
        if state is None:
            raise RuntimeError("exploration journal disappeared before settlement.")
        self.journal.append_new(
            "round_settled",
            round_index=round_index,
            progress=progress,
            terminal_reason=terminal_reason,
            terminal_has_reduction=bool(
                terminal_reason and state.current_round_reduction_committed
            ),
        )
        return self.snapshot()

    def mark_paused(self) -> SupervisorJournalState:
        state = self.snapshot()
        if state.status == "paused":
            return state
        self.journal.append_new("paused")
        return self.snapshot()

    def stop(
        self, reason: str, *, report_ref: str | None
    ) -> SupervisorJournalState:
        state = self.snapshot()
        if state.status == "stopped":
            if state.stop_reason == reason and state.final_report_ref == report_ref:
                return state
            raise RuntimeError("exploration already stopped with another outcome.")
        self.journal.append_new(
            "exploration_stopped",
            stop_reason=reason,
            final_report_ref=report_ref,
        )
        return self.snapshot()


@dataclass(frozen=True, slots=True)
class _ShadowFinalizerGuard:
    inner: FinalizerPort
    validate_report_ref: Callable[[str | None], str | None]

    def synthesize(
        self,
        context: Any,
        reduction: ReductionOutcome,
        *,
        logical_step_id: str,
    ) -> FinalizationOutcome:
        outcome = self.inner.synthesize(
            context, reduction, logical_step_id=logical_step_id
        )
        return FinalizationOutcome(self.validate_report_ref(outcome.report_ref))

    def render_deterministic(
        self, context: Any, reduction: ReductionOutcome
    ) -> FinalizationOutcome:
        outcome = self.inner.render_deterministic(context, reduction)
        return FinalizationOutcome(self.validate_report_ref(outcome.report_ref))


@dataclass(frozen=True, slots=True)
class _ShadowRecoveryGuard(CompletedStepRecoveryPort):
    inner: CompletedStepRecoveryPort
    validate_report_ref: Callable[[str | None], str | None]
    expected_response_digest: Callable[[str], str | None]

    def load_required(self, logical_step_id: str) -> object:
        result = self.inner.load_required(logical_step_id)
        if isinstance(result, CandidateBatch):
            expected = self.expected_response_digest(logical_step_id)
            if expected is None or candidate_batch_digest(result) != expected:
                raise ValueError(
                    "recovered candidate batch digest does not match the journal."
                )
        if isinstance(result, FinalizationOutcome):
            return FinalizationOutcome(self.validate_report_ref(result.report_ref))
        return result

    def remember(self, logical_step_id: str, result: object) -> None:
        if isinstance(result, FinalizationOutcome):
            result = FinalizationOutcome(self.validate_report_ref(result.report_ref))
        self.inner.remember(logical_step_id, result)


class JsonSupervisorRecoveryStore(CompletedStepRecoveryPort):
    """Immutable, atomic JSON bodies for supervisor paid/reduction steps."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def load_required(self, logical_step_id: str) -> object:
        path = self._path(logical_step_id)
        try:
            raw = json.loads(_read_bytes_no_follow(path))
        except FileNotFoundError as exc:
            raise KeyError(
                f"completed step {logical_step_id!r} has no durable body; "
                "refusing to repeat it."
            ) from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"invalid recovery body for {logical_step_id!r}: {exc}"
            ) from exc
        if raw.get("logical_step_id") != logical_step_id:
            raise ValueError("recovery body logical_step_id does not match its key.")
        return _decode_supervisor_result(raw.get("result"))

    def remember(self, logical_step_id: str, result: object) -> None:
        encoded = {
            "logical_step_id": logical_step_id,
            "result": _encode_supervisor_result(result),
        }
        path = self._path(logical_step_id)
        if path.exists():
            current = self.load_required(logical_step_id)
            if current != result:
                raise ValueError(
                    f"recovery body {logical_step_id!r} is immutable and cannot be replaced."
                )
            return
        _write_json_atomic(path, encoded)

    def digest_if_present(self, logical_step_id: str) -> str | None:
        try:
            result = self.load_required(logical_step_id)
        except KeyError:
            return None
        return stable_hash(_encode_supervisor_result(result), length=24)

    def _path(self, logical_step_id: str) -> Path:
        if not logical_step_id:
            raise ValueError("logical_step_id cannot be empty.")
        return self.root / f"{stable_hash(logical_step_id, length=32)}.json"


class JsonLlmResponseStore:
    """Durable executor response bodies; journal completion is written afterward."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def load_required(self, logical_step_id: str) -> LLMToolResponse:
        path = self._path(logical_step_id)
        try:
            raw = json.loads(_read_bytes_no_follow(path))
        except FileNotFoundError as exc:
            raise KeyError(
                f"completed LLM response {logical_step_id!r} is unavailable."
            ) from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"invalid LLM response body for {logical_step_id!r}: {exc}"
            ) from exc
        if raw.get("logical_step_id") != logical_step_id:
            raise ValueError("LLM response logical_step_id does not match its key.")
        return LLMToolResponse.model_validate(raw.get("response"))

    def remember(self, logical_step_id: str, response: LLMToolResponse) -> None:
        path = self._path(logical_step_id)
        if path.exists():
            if self.load_required(logical_step_id) != response:
                raise ValueError(
                    f"LLM response {logical_step_id!r} is immutable and cannot be replaced."
                )
            return
        _write_json_atomic(
            path,
            {
                "logical_step_id": logical_step_id,
                "response": response.model_dump(mode="json"),
            },
        )

    def digest_if_present(self, logical_step_id: str) -> str | None:
        try:
            response = self.load_required(logical_step_id)
        except KeyError:
            return None
        return stable_hash(response.model_dump(mode="json"), length=24)

    def _path(self, logical_step_id: str) -> Path:
        if not logical_step_id:
            raise ValueError("logical_step_id cannot be empty.")
        return self.root / f"{stable_hash(logical_step_id, length=32)}.json"


class JsonToolResultStore:
    """Durable JSON envelope for shadow tool observations and Artifact receipts."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def load_required(self, logical_step_id: str) -> DurableToolResult:
        path = self._path(logical_step_id)
        try:
            raw = json.loads(_read_bytes_no_follow(path))
        except FileNotFoundError as exc:
            raise KeyError(f"tool result {logical_step_id!r} is unavailable.") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid tool result {logical_step_id!r}: {exc}") from exc
        if raw.get("logical_step_id") != logical_step_id:
            raise ValueError("tool result logical_step_id does not match its key.")
        result_raw = raw.get("result")
        usage_raw = raw.get("usage")
        if not isinstance(result_raw, dict) or not isinstance(usage_raw, dict):
            raise ValueError("tool result body is incomplete.")
        artifacts_raw = result_raw.get("artifacts", [])
        if not isinstance(artifacts_raw, list):
            raise ValueError("tool result artifacts must be a list.")
        receipt_raw = result_raw.get("receipt_artifact")
        return DurableToolResult(
            result=AgentToolResult(
                content=cast(Any, result_raw.get("content", "")),
                artifacts=[Artifact.model_validate(item) for item in artifacts_raw],
                receipt_artifact=(
                    None if receipt_raw is None else Artifact.model_validate(receipt_raw)
                ),
            ),
            usage=ToolCallProjection(
                kind=str(usage_raw.get("kind", "")),
                rows_scanned=_required_non_negative_int(
                    usage_raw.get("rows_scanned"), "rows_scanned"
                ),
                result_cells=_required_non_negative_int(
                    usage_raw.get("result_cells"), "result_cells"
                ),
            ),
        )

    def remember(self, logical_step_id: str, result: DurableToolResult) -> None:
        encoded = {
            "logical_step_id": logical_step_id,
            **durable_tool_result_payload(result),
        }
        path = self._path(logical_step_id)
        if path.exists():
            if self.load_required(logical_step_id) != result:
                raise ValueError(
                    f"tool result {logical_step_id!r} is immutable and cannot be replaced."
                )
            return
        _write_json_atomic(path, encoded)

    def recovered_commit_if_present(
        self, logical_step_id: str
    ) -> RecoveredToolCommit | None:
        try:
            durable = self.load_required(logical_step_id)
        except KeyError:
            return None
        receipt = _require_artifact(durable.result.receipt_artifact)
        return RecoveredToolCommit(
            receipt_id=_receipt_payload_id(receipt),
            result_digest=durable_tool_result_digest(durable),
            rows_scanned=durable.usage.rows_scanned,
            result_cells=durable.usage.result_cells,
        )

    def _path(self, logical_step_id: str) -> Path:
        if not logical_step_id:
            raise ValueError("logical_step_id cannot be empty.")
        return self.root / f"{stable_hash(logical_step_id, length=32)}.json"


def run_shadow_exploration(
    *,
    workspace: Path | str,
    exploration_id: str,
    policy: ExplorationPolicy,
    code_fingerprint: str,
    data_state_witness: str,
    config: SupervisorConfig,
    generator: CandidateGeneratorPort | None = None,
    scheduler: SchedulerPort | None = None,
    executor: ProbeExecutorPort | None = None,
    validator: ValidatorPort | None = None,
    reducer: ReducerPort | None = None,
    finalizer: FinalizerPort | None = None,
    witness: WitnessPort,
    budget: BudgetPort | None = None,
    control: ControlPort | None = None,
    projection_data: Callable[[], ShadowProjectionData] | None = None,
    journal: JsonlExplorationJournal | None = None,
    recovery: JsonSupervisorRecoveryStore | None = None,
    llm_response_store: JsonLlmResponseStore | None = None,
    tool_result_store: JsonToolResultStore | None = None,
    after_recovery: Callable[[ExplorationLoopState], None] | None = None,
    component_factory: Callable[
        [ExplorationLoopState], ShadowExecutionComponents
    ]
    | None = None,
) -> ShadowExplorationRunResult:
    """Run or resume one E4a workflow and publish only a shadow projection."""
    workspace_path = require_absolute_workspace(Path(workspace))
    run_root = shadow_run_root(workspace_path, exploration_id)
    journal_path = validate_shadow_run_path(
        workspace_path, exploration_id, run_root / "journal.jsonl"
    )
    actual_journal = journal or JsonlExplorationJournal(journal_path)
    validated_journal_path = validate_shadow_run_path(
        workspace_path, exploration_id, actual_journal.path
    )
    if validated_journal_path != journal_path:
        raise ValueError("E4a journal must live inside its exploration-eval run directory.")
    actual_recovery = recovery or JsonSupervisorRecoveryStore(
        run_root / "phase-responses"
    )
    actual_llm_response_store = llm_response_store or JsonLlmResponseStore(
        run_root / "llm-responses"
    )
    actual_tool_result_store = tool_result_store or JsonToolResultStore(
        run_root / "tool-results"
    )
    actual_control = control or JournalControlPort()
    for root in (
        actual_recovery.root,
        actual_llm_response_store.root,
        actual_tool_result_store.root,
    ):
        validate_shadow_run_path(workspace_path, exploration_id, root)

    def validate_report_ref(report_ref: str | None) -> str | None:
        if report_ref is None:
            return None
        candidate = Path(report_ref)
        if not candidate.is_absolute():
            candidate = workspace_path / candidate
        validated = validate_shadow_run_path(
            workspace_path, exploration_id, candidate
        )
        return str(validated.relative_to(workspace_path))

    guarded_recovery = _ShadowRecoveryGuard(
        actual_recovery,
        validate_report_ref,
        lambda step_id: _completed_response_digests(actual_journal).get(step_id),
    )

    with actual_journal.execution_lock():
        state = actual_journal.initialize(
            exploration_id=exploration_id,
            policy=policy,
            code_fingerprint=code_fingerprint,
            data_state_witness=data_state_witness,
        )
        if state.status != "stopped":
            state = actual_journal.claim_recovery(
                completed_response_digest=lambda step_id: (
                    actual_llm_response_store.digest_if_present(step_id)
                    or actual_recovery.digest_if_present(step_id)
                ),
                completed_tool_result=(
                    actual_tool_result_store.recovered_commit_if_present
                ),
            )
        if after_recovery is not None:
            after_recovery(state)
        if component_factory is not None:
            if any(
                item is not None
                for item in (
                    generator,
                    scheduler,
                    executor,
                    validator,
                    reducer,
                    finalizer,
                    budget,
                    projection_data,
                )
            ):
                raise ValueError(
                    "component_factory cannot be mixed with prebuilt shadow ports."
                )
            components = component_factory(state)
            actual_generator = components.generator
            actual_scheduler = components.scheduler
            actual_executor = components.executor
            actual_validator = components.validator
            actual_reducer = components.reducer
            actual_finalizer = components.finalizer
            actual_budget = components.budget
            actual_projection_data = components.projection_data
        else:
            required = (generator, scheduler, executor, validator, reducer, finalizer)
            if any(item is None for item in required):
                raise ValueError(
                    "shadow execution requires every supervisor port or a component_factory."
                )
            actual_generator = cast(CandidateGeneratorPort, generator)
            actual_scheduler = cast(SchedulerPort, scheduler)
            actual_executor = cast(ProbeExecutorPort, executor)
            actual_validator = cast(ValidatorPort, validator)
            actual_reducer = cast(ReducerPort, reducer)
            actual_finalizer = cast(FinalizerPort, finalizer)
            actual_budget = budget or JournalBudgetPort()
            actual_projection_data = projection_data or (lambda: ShadowProjectionData())
        guarded_finalizer = _ShadowFinalizerGuard(actual_finalizer, validate_report_ref)
        adapter = JsonlSupervisorJournalAdapter(actual_journal)
        supervisor = ExplorationSupervisor(
            config=config,
            journal=adapter,
            witness=witness,
            budget=actual_budget,
            control=actual_control,
            generator=actual_generator,
            scheduler=actual_scheduler,
            executor=actual_executor,
            validator=actual_validator,
            reducer=actual_reducer,
            finalizer=guarded_finalizer,
            recovery=guarded_recovery,
        )
        result = supervisor.run()
        final_state = actual_journal.rebuild()
        if final_state is None:  # pragma: no cover - initialize guarantees this
            raise RuntimeError("exploration journal disappeared after supervision.")
        actual_journal.write_snapshot(final_state)
        payload = actual_projection_data()
        projection = ShadowExplorationProjection(
            exploration_id=exploration_id,
            last_seq=final_state.last_seq,
            status=final_state.status,
            stop_reason=final_state.stop_reason,
            policy_fingerprint=final_state.effective_policy_fingerprint,
            data_state_witness=final_state.data_state_witness,
            insight_records=payload.insight_records,
            coverage_completed=tuple(sorted(set(payload.coverage_completed))),
            coverage_unexplored=tuple(sorted(set(payload.coverage_unexplored))),
        )
        shadow = ShadowExplorationStore(workspace_path)
        existing = shadow.read(exploration_id)
        if existing is None or projection.last_seq > existing.last_seq:
            projection_path = shadow.project(projection)
        elif (
            projection.last_seq == existing.last_seq
            and projection.model_dump(exclude={"projected_at"})
            == existing.model_dump(exclude={"projected_at"})
        ):
            projection_path = shadow.path_for(exploration_id)
        else:
            raise RuntimeError("shadow projection is ahead of or conflicts with the journal.")
    return ShadowExplorationRunResult(
        result=result,
        journal_path=journal_path,
        projection_path=projection_path,
    )


def _supervisor_projection(
    journal: JsonlExplorationJournal,
    state: ExplorationLoopState,
) -> SupervisorJournalState:
    receipt_ids: set[str] = set()
    if state.current_round_index is not None:
        in_current_round = False
        for event in journal.events():
            if isinstance(event, RoundStartedEvent):
                in_current_round = event.round_index == state.current_round_index
                if in_current_round:
                    receipt_ids.clear()
            elif in_current_round and isinstance(event, ReceiptCommittedEvent):
                receipt_ids.add(event.receipt_id)
    return SupervisorJournalState(
        exploration_id=state.exploration_id,
        data_state_witness=state.data_state_witness,
        status=state.status,
        stop_reason=state.stop_reason,
        final_report_ref=state.final_report_ref,
        rounds_started=state.rounds_started,
        rounds_settled=state.rounds_settled,
        current_round_index=state.current_round_index,
        remaining_round_budget=state.remaining_round_budget,
        remaining_llm_call_budget=state.remaining_llm_call_budget,
        remaining_tool_call_budget=state.remaining_tool_call_budget,
        consecutive_no_progress=state.consecutive_no_progress,
        completed_step_ids=frozenset(state.completed_step_ids),
        completed_probe_fingerprints=frozenset(state.completed_probe_fingerprints),
        uncertain_call_ids=frozenset(state.uncertain_call_ids),
        step_receipt_refs=dict(state.step_receipt_refs),
        current_round_receipt_ids=frozenset(receipt_ids),
        current_round_reduction_committed=(
            state.current_round_reduction_committed
        ),
        pending_terminal_reason=state.pending_terminal_reason,
        pending_terminal_has_reduction=state.pending_terminal_has_reduction,
        frontier_digest=state.frontier_digest,
        ledger_digest=state.ledger_digest,
        reduction_digest=state.reduction_digest,
    )


def _completed_response_digests(
    journal: JsonlExplorationJournal,
) -> dict[str, str]:
    return {
        event.step_id: event.response_digest
        for event in journal.events()
        if isinstance(event, LlmCallCompletedEvent)
    }


def _encode_supervisor_result(result: object) -> dict[str, Any]:
    if isinstance(result, CandidateBatch):
        return {
            "kind": "candidate_batch",
            "candidates": [_encode_payload(item) for item in result.candidates],
        }
    if isinstance(result, ReductionOutcome):
        return {
            "kind": "reduction_outcome",
            "transitions": list(result.transitions),
            "frontier": _encode_frontier(result.frontier),
            "ledger_digest": result.ledger_digest,
            "goal_satisfied": result.goal_satisfied,
            "coverage_target_met": result.coverage_target_met,
        }
    if isinstance(result, FinalizationOutcome):
        return {"kind": "finalization_outcome", "report_ref": result.report_ref}
    raise TypeError(
        "supervisor recovery accepts CandidateBatch, ReductionOutcome, or "
        "FinalizationOutcome only."
    )


def _decode_supervisor_result(raw: object) -> object:
    if not isinstance(raw, dict):
        raise ValueError("supervisor recovery result must be an object.")
    kind = raw.get("kind")
    if kind == "candidate_batch":
        candidates = raw.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError("candidate recovery body must contain a list.")
        return CandidateBatch(tuple(_decode_payload(item) for item in candidates))
    if kind == "reduction_outcome":
        transitions = raw.get("transitions")
        if not isinstance(transitions, list) or not all(
            isinstance(item, str) for item in transitions
        ):
            raise ValueError("reduction transitions must be a string list.")
        return ReductionOutcome(
            transitions=cast(tuple, tuple(transitions)),
            frontier=_decode_frontier(raw.get("frontier")),
            ledger_digest=str(raw.get("ledger_digest", "")),
            goal_satisfied=bool(raw.get("goal_satisfied", False)),
            coverage_target_met=bool(raw.get("coverage_target_met", False)),
        )
    if kind == "finalization_outcome":
        report_ref = raw.get("report_ref")
        if report_ref is not None and not isinstance(report_ref, str):
            raise ValueError("finalization report_ref must be a string or null.")
        return FinalizationOutcome(report_ref)
    raise ValueError(f"unknown supervisor recovery result kind {kind!r}.")


def _encode_frontier(frontier: ScoredFrontier) -> dict[str, Any]:
    return {
        "digest": frontier.digest,
        "items": [
            {
                "hypothesis_id": item.hypothesis_id,
                "priority": item.priority,
                "payload": _encode_payload(item.payload),
            }
            for item in frontier.items
        ],
    }


def _decode_frontier(raw: object) -> ScoredFrontier:
    if not isinstance(raw, dict) or not isinstance(raw.get("items"), list):
        raise ValueError("frontier recovery body is invalid.")
    return ScoredFrontier(
        items=tuple(
            _decode_frontier_item(item)
            for item in raw["items"]
            if isinstance(item, dict)
        ),
        digest=str(raw.get("digest", "")),
    )


def _decode_frontier_item(raw: dict[object, object]) -> FrontierItem:
    priority = raw.get("priority")
    if isinstance(priority, bool) or not isinstance(priority, int | float):
        raise ValueError("frontier item priority must be numeric.")
    return FrontierItem(
        hypothesis_id=str(raw.get("hypothesis_id", "")),
        priority=float(priority),
        payload=_decode_payload(raw.get("payload")),
    )


def _encode_payload(value: object) -> object:
    if isinstance(value, CandidateSeed):
        return {
            "__kind__": "candidate_seed",
            "proposal": value.proposal.model_dump(mode="json"),
            "hypothesis_id": value.hypothesis_id,
            "hypothesis_fingerprint": value.hypothesis_fingerprint,
            "canonical_group_key": value.canonical_group_key,
            "coverage_key": value.coverage_key,
            "sequence_index": value.sequence_index,
            "status": value.status,
            "origin": value.origin,
            "mandatory": value.mandatory,
            "priority": value.priority,
        }
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, tuple | list):
        return [_encode_payload(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("recovery payload mappings require string keys.")
        return {key: _encode_payload(item) for key, item in value.items()}
    raise TypeError(f"unsupported supervisor recovery payload {type(value).__name__}.")


def _decode_payload(value: object) -> object:
    if isinstance(value, list):
        return tuple(_decode_payload(item) for item in value)
    if not isinstance(value, dict):
        return value
    if value.get("__kind__") != "candidate_seed":
        return {str(key): _decode_payload(item) for key, item in value.items()}
    return CandidateSeed(
        proposal=HypothesisProposal.model_validate(value.get("proposal")),
        hypothesis_id=str(value.get("hypothesis_id", "")),
        hypothesis_fingerprint=str(value.get("hypothesis_fingerprint", "")),
        canonical_group_key=str(value.get("canonical_group_key", "")),
        coverage_key=str(value.get("coverage_key", "")),
        sequence_index=int(value.get("sequence_index", 0)),
        status=cast(Any, value.get("status")),
        origin=cast(Any, value.get("origin")),
        mandatory=bool(value.get("mandatory", False)),
        priority=float(value.get("priority", 0.0)),
    )


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.stem}-",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        body = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.stem}-",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _required_non_negative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer.")
    return value


def _typed_items(value: object, model: type[Any]) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError("workflow-state typed collections must be lists.")
    return [model.model_validate(item) for item in value]


def _required_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"workflow-state {label} must be a list.")
    return value


def _encode_scheduling_decision(value: SchedulingDecision) -> dict[str, object]:
    return {
        "hypothesis_id": value.hypothesis_id,
        "hypothesis_fingerprint": value.hypothesis_fingerprint,
        "family": value.family.value,
        "status": value.status,
        "admission_checks": [
            {
                "name": check.name,
                "passed": check.passed,
                "detail_code": check.detail_code,
            }
            for check in value.admission_checks
        ],
        "priority_features": value.priority_features.model_dump(mode="json"),
        "priority": value.priority,
        "scoring_policy_version": value.scoring_policy_version,
        "quota_deferred": value.quota_deferred,
        "chosen": value.chosen,
    }


def _decode_scheduling_decision(raw: object) -> SchedulingDecision:
    if not isinstance(raw, dict):
        raise ValueError("workflow-state scheduling decision must be an object.")
    checks = _required_list(raw.get("admission_checks"), "admission_checks")
    return SchedulingDecision(
        hypothesis_id=str(raw.get("hypothesis_id", "")),
        hypothesis_fingerprint=str(raw.get("hypothesis_fingerprint", "")),
        family=InsightFamily(str(raw.get("family", ""))),
        status=cast(Any, raw.get("status")),
        admission_checks=tuple(
            AdmissionCheck(
                name=cast(Any, _required_mapping(item, "admission check").get("name")),
                passed=bool(_required_mapping(item, "admission check").get("passed")),
                detail_code=str(
                    _required_mapping(item, "admission check").get("detail_code", "")
                ),
            )
            for item in checks
        ),
        priority_features=PriorityFeatures.model_validate(raw.get("priority_features")),
        priority=float(raw.get("priority", 0.0)),
        scoring_policy_version=str(raw.get("scoring_policy_version", "")),
        quota_deferred=bool(raw.get("quota_deferred", False)),
        chosen=bool(raw.get("chosen", False)),
    )


def _required_mapping(value: object, label: str) -> dict[object, object]:
    if not isinstance(value, dict):
        raise ValueError(f"workflow-state {label} must be an object.")
    return value


def _require_artifact(value: object) -> Artifact:
    if not isinstance(value, Artifact):
        raise TypeError("durable shadow tool results accept Artifact objects only.")
    return value


def _receipt_payload_id(artifact: Artifact) -> str:
    value = artifact.payload.get("receipt_id")
    if not isinstance(value, str) or not value:
        raise ValueError("receipt artifact payload has no receipt_id.")
    return value


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_bytes_no_follow(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as handle:
        return handle.read()
