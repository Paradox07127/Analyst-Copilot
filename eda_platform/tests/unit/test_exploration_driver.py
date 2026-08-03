"""E4a driver composition, durable body stores, and shadow isolation."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from eda_platform.agents.exploration.candidates import candidate_seed
from eda_platform.agents.exploration.supervisor import (
    CandidateBatch,
    FinalizationOutcome,
    FrontierItem,
    PhaseContext,
    ProbeOutcome,
    ProbeSelection,
    ReductionOutcome,
    ScoredFrontier,
    SupervisorConfig,
    ValidationOutcome,
    reduction_outcome_digest,
)
from eda_platform.core.exploration_journal import JsonlExplorationJournal
from eda_platform.core.exploration_profiles import build_exploration_policy
from eda_platform.core.ids import stable_hash
from eda_platform.core.llm import LLMToolCall, LLMToolResponse
from eda_platform.drivers.exploration import (
    CallableWitnessPort,
    JsonLlmResponseStore,
    JsonlSupervisorJournalAdapter,
    JsonSupervisorRecoveryStore,
    ShadowProjectionData,
    run_shadow_exploration,
)
from eda_platform.schemas.exploration import InsightFamily
from eda_platform.schemas.hypotheses import HypothesisPredicate, HypothesisProposal
from eda_platform.schemas.insights import InsightProof, InsightRecord


def _policy():
    return build_exploration_policy(
        tier="quick",
        dataset_scope=("ds-1",),
        tool_capability_digest="tools-v1",
    )


def _proposal() -> HypothesisProposal:
    return HypothesisProposal(
        statement="Does revenue differ by region?",
        rationale="Test planted region structure.",
        expected_evidence="Group comparison with effect size.",
        falsification_conditions=("No material difference is observed.",),
        family=InsightFamily.DIAGNOSTIC,
        method_family="compare_groups",
        dataset_ids=("ds-1",),
        columns=("region", "revenue"),
        probe_kind="region_difference",
        predicate=HypothesisPredicate(
            metric="revenue", operator="differs", left_operand="region"
        ),
    )


def _insight() -> InsightRecord:
    return InsightRecord(
        insight_id="insight-1",
        hypothesis_id="hypothesis-1",
        family=InsightFamily.DIAGNOSTIC,
        status="new",
        trust_level="supported",
        claim_bundle_id="claim-bundle-1",
        supporting_receipt_ids=("receipt-1",),
        proof=(
            InsightProof(
                receipt_id="receipt-1",
                fact_ids=("fact-1",),
                comparison="supports",
            ),
        ),
        created_round=0,
        last_updated_round=0,
    )


def test_durable_response_stores_round_trip_and_are_immutable(tmp_path: Path) -> None:
    recovery = JsonSupervisorRecoveryStore(tmp_path / "phase")
    seed = candidate_seed(_proposal(), sequence_index=1, mandatory=True)
    candidate_batch = CandidateBatch((seed, {"source": "bootstrap"}))
    reduction = ReductionOutcome(
        transitions=("new",),
        frontier=ScoredFrontier(
            (FrontierItem(seed.hypothesis_id, 0.9, seed),),
            "frontier-1",
        ),
        ledger_digest="ledger-1",
        goal_satisfied=True,
    )
    finalization = FinalizationOutcome("exploration-eval/xpl/report.json")

    for step_id, value in (
        ("generate", candidate_batch),
        ("reduce", reduction),
        ("synthesize", finalization),
    ):
        recovery.remember(step_id, value)
        assert recovery.load_required(step_id) == value
        recovery.remember(step_id, value)

    with pytest.raises(ValueError, match="immutable"):
        recovery.remember("generate", CandidateBatch(({"changed": True},)))
    with pytest.raises(KeyError, match="refusing to repeat"):
        recovery.load_required("missing")

    responses = JsonLlmResponseStore(tmp_path / "llm")
    response = LLMToolResponse(
        content="",
        tool_calls=[LLMToolCall(call_id="call-1", name="run_sql", arguments={})],
        finish_reason="tool_calls",
    )
    responses.remember("llm-step", response)
    assert responses.load_required("llm-step") == response
    responses.remember("llm-step", response)
    with pytest.raises(ValueError, match="immutable"):
        responses.remember("llm-step", LLMToolResponse(content="changed"))


def test_uncommitted_reduction_body_can_be_replaced_by_a_recomputation(
    tmp_path: Path,
) -> None:
    """A crash between supervisor._reduce's remember() and commit_reduction()
    must not fail the resumed round: replay recomputes a different
    ReductionOutcome (admitted_bundle_count diffs against the baseline the
    first attempt already persisted), and that recomputation must be allowed
    to replace the stale, uncommitted body instead of raising."""
    journal = JsonlExplorationJournal(tmp_path / "journal.jsonl")
    journal.initialize(
        exploration_id="xpl-crash",
        policy=_policy(),
        code_fingerprint="code-v1",
        data_state_witness="witness-v1",
    )
    journal.claim_recovery()
    journal.append_new("round_started", round_index=0, branch_id=None)
    recovery = JsonSupervisorRecoveryStore(tmp_path / "phase", journal=journal)

    seed = candidate_seed(_proposal(), sequence_index=1, mandatory=True)
    frontier = ScoredFrontier(
        (FrontierItem(seed.hypothesis_id, 0.9, seed),), "frontier-1"
    )
    first_attempt = ReductionOutcome(
        transitions=("new",),
        frontier=frontier,
        ledger_digest="ledger-1",
        admitted_bundle_count=2,
    )
    recovery.remember("reduce-step", first_attempt)

    second_attempt = ReductionOutcome(
        transitions=("new",),
        frontier=frontier,
        ledger_digest="ledger-1",
        admitted_bundle_count=0,
    )
    recovery.remember("reduce-step", second_attempt)

    assert recovery.load_required("reduce-step") == second_attempt


def test_committed_reduction_body_stays_immutable(tmp_path: Path) -> None:
    """Control group: once the journal has actually committed a reduction
    outcome, a different body must still be rejected."""
    journal = JsonlExplorationJournal(tmp_path / "journal.jsonl")
    journal.initialize(
        exploration_id="xpl-committed",
        policy=_policy(),
        code_fingerprint="code-v1",
        data_state_witness="witness-v1",
    )
    journal.claim_recovery()
    journal.append_new("round_started", round_index=0, branch_id=None)
    recovery = JsonSupervisorRecoveryStore(tmp_path / "phase", journal=journal)

    seed = candidate_seed(_proposal(), sequence_index=1, mandatory=True)
    frontier = ScoredFrontier(
        (FrontierItem(seed.hypothesis_id, 0.9, seed),), "frontier-1"
    )
    committed = ReductionOutcome(
        transitions=("new",),
        frontier=frontier,
        ledger_digest="ledger-1",
        admitted_bundle_count=2,
    )
    recovery.remember("reduce-step", committed)
    journal.append_new(
        "reduction_committed",
        frontier_digest=frontier.digest,
        ledger_digest=committed.ledger_digest,
        reduction_digest=reduction_outcome_digest(committed),
    )

    different = ReductionOutcome(
        transitions=("new",),
        frontier=frontier,
        ledger_digest="ledger-1",
        admitted_bundle_count=0,
    )
    with pytest.raises(ValueError, match="immutable"):
        recovery.remember("reduce-step", different)


def test_durable_response_store_refuses_a_symlinked_body(tmp_path: Path) -> None:
    if not hasattr(os, "O_NOFOLLOW"):
        pytest.skip("requires POSIX O_NOFOLLOW")
    root = tmp_path / "llm"
    store = JsonLlmResponseStore(root)
    step_id = "llm-step"
    store.remember(step_id, LLMToolResponse(content="original"))
    body = root / f"{stable_hash(step_id, length=32)}.json"
    body.unlink()
    outside = tmp_path / "outside.json"
    outside.write_text("do not touch", encoding="utf-8")
    body.symlink_to(outside)

    with pytest.raises(ValueError, match="invalid LLM response body"):
        store.load_required(step_id)
    with pytest.raises(ValueError, match="invalid LLM response body"):
        store.remember(step_id, LLMToolResponse(content="changed"))

    assert outside.read_text(encoding="utf-8") == "do not touch"


def test_event_journal_refuses_a_symlinked_writer_lock(tmp_path: Path) -> None:
    if not hasattr(os, "O_NOFOLLOW"):
        pytest.skip("requires POSIX O_NOFOLLOW")
    journal = JsonlExplorationJournal(tmp_path / "journal.jsonl")
    outside = tmp_path / "outside.lock"
    outside.write_text("do not touch", encoding="utf-8")
    journal.lock_path.symlink_to(outside)

    with pytest.raises(OSError):
        journal.initialize(
            exploration_id="xpl-lock",
            policy=_policy(),
            code_fingerprint="code-v1",
            data_state_witness="witness-v1",
        )

    assert outside.read_text(encoding="utf-8") == "do not touch"


def test_jsonl_supervisor_adapter_projects_round_local_receipts_and_terminal_intent(
    tmp_path: Path,
) -> None:
    journal = JsonlExplorationJournal(tmp_path / "journal.jsonl")
    journal.initialize(
        exploration_id="xpl-adapter",
        policy=_policy(),
        code_fingerprint="code-v1",
        data_state_witness="witness-v1",
    )
    journal.claim_recovery()
    adapter = JsonlSupervisorJournalAdapter(journal)
    state = adapter.start_round(0)
    assert state.current_round_receipt_ids == frozenset()

    journal.append_new(
        "tool_call_started",
        logical_step_id="tool-step-1",
        input_fingerprint="input-1",
    )
    journal.append_new(
        "receipt_prepared",
        logical_step_id="tool-step-1",
        receipt_id="receipt-1",
    )
    journal.append_new(
        "receipt_committed",
        logical_step_id="tool-step-1",
        receipt_id="receipt-1",
    )
    assert adapter.snapshot().current_round_receipt_ids == frozenset({"receipt-1"})

    adapter.commit_reduction(
        frontier_digest="frontier-1",
        ledger_digest="ledger-1",
        reduction_digest="reduction-1",
    )
    settled = adapter.settle_round(
        0,
        progress=True,
        terminal_reason="completed",
    )
    assert settled.current_round_index is None
    assert settled.pending_terminal_reason == "completed"
    assert settled.pending_terminal_has_reduction is True
    assert settled.current_round_receipt_ids == frozenset()


def test_deterministic_render_survives_a_mid_round_budget_latch(
    tmp_path: Path,
) -> None:
    """Seed-6 regression: budget death inside an unsettled round left
    pending_terminal_reason unset, the renderer raised, and the certified
    report ref silently became None. The supervisor's latched reason on the
    phase context must be an accepted substitute."""
    from eda_platform.agents.exploration.supervisor import SupervisorPhase
    from eda_platform.agents.exploration.workflow import ExplorationWorkflowState
    from eda_platform.drivers.exploration import (
        DeterministicShadowFinalizer,
        shadow_run_root,
    )

    def _reduction_outcome() -> ReductionOutcome:
        seed = candidate_seed(_proposal(), sequence_index=1)
        return ReductionOutcome(
            transitions=(),
            frontier=ScoredFrontier(
                (FrontierItem(seed.hypothesis_id, 0.9, seed),), "frontier-1"
            ),
            ledger_digest="ledger-1",
            goal_satisfied=False,
        )

    workspace = tmp_path / "workspace"
    exploration_id = "xpl-budget-latch"
    run_root = shadow_run_root(workspace, exploration_id)
    run_root.mkdir(parents=True)
    journal = JsonlExplorationJournal(run_root / "journal.jsonl")
    journal.initialize(
        exploration_id=exploration_id,
        policy=_policy(),
        code_fingerprint="code-v1",
        data_state_witness="witness-v1",
    )
    journal.claim_recovery()
    adapter = JsonlSupervisorJournalAdapter(journal)
    adapter.start_round(0)  # started, never settled: no terminal reason
    assert adapter.snapshot().pending_terminal_reason is None

    finalizer = DeterministicShadowFinalizer(
        workspace=workspace,
        exploration_id=exploration_id,
        state=ExplorationWorkflowState(),
        journal=journal,
        coverage_targets=(),
        budget_summary=lambda: {"llm_requests_used": 1},
    )
    context = PhaseContext(
        exploration_id=exploration_id,
        round_index=0,
        phase=SupervisorPhase.SYNTHESIZE,
        data_state_witness="witness-v1",
        soft_countdown_context="",
        completed_step_ids=frozenset(),
        terminal_reason="budget_exhausted",
    )

    outcome = finalizer.render_deterministic(context, _reduction_outcome())

    assert outcome.report_ref is not None
    report = (workspace / outcome.report_ref).read_text(encoding="utf-8")
    assert "budget_exhausted" in report

    # Without either a settled terminal reason or a latched one, rendering
    # must still refuse: a report may not precede a durable stop decision.
    bare_context = context.for_phase(SupervisorPhase.SYNTHESIZE)
    assert bare_context.terminal_reason == "budget_exhausted"
    undecided = PhaseContext(
        exploration_id=exploration_id,
        round_index=0,
        phase=SupervisorPhase.SYNTHESIZE,
        data_state_witness="witness-v1",
        soft_countdown_context="",
        completed_step_ids=frozenset(),
    )
    with pytest.raises(ValueError, match="durable stop decision"):
        finalizer.render_deterministic(undecided, _reduction_outcome())


class _Generator:
    def __init__(
        self,
        journal: JsonlExplorationJournal,
        recovery: JsonSupervisorRecoveryStore,
    ) -> None:
        self.journal = journal
        self.recovery = recovery
        self.calls = 0

    def generate(self, context: PhaseContext, *, logical_step_id: str) -> CandidateBatch:
        self.calls += 1
        result = CandidateBatch((candidate_seed(_proposal(), sequence_index=1),))
        self.journal.append_new("llm_call_started", call_id="generate-call-1")
        self.recovery.remember(logical_step_id, result)
        self.journal.append_new(
            "llm_call_completed",
            call_id="generate-call-1",
            step_id=logical_step_id,
            response_digest="generate-response-1",
        )
        return result


class _Scheduler:
    def admit_and_score(
        self, context: PhaseContext, candidates: CandidateBatch
    ) -> ScoredFrontier:
        seed = candidates.candidates[0]
        assert hasattr(seed, "hypothesis_id")
        return ScoredFrontier(
            (FrontierItem(seed.hypothesis_id, 0.9, seed),),  # type: ignore[union-attr]
            "frontier-1",
        )

    def select(
        self, context: PhaseContext, frontier: ScoredFrontier
    ) -> ProbeSelection:
        return ProbeSelection(frontier.items)


class _Executor:
    def __init__(self, journal: JsonlExplorationJournal) -> None:
        self.journal = journal
        self.calls = 0

    def execute(self, context: PhaseContext, selection: ProbeSelection) -> ProbeOutcome:
        self.calls += 1
        self.journal.append_new(
            "tool_call_started",
            logical_step_id="probe-step-1",
            input_fingerprint="probe-input-1",
        )
        self.journal.append_new(
            "receipt_prepared",
            logical_step_id="probe-step-1",
            receipt_id="receipt-1",
        )
        self.journal.append_new(
            "receipt_committed",
            logical_step_id="probe-step-1",
            receipt_id="receipt-1",
        )
        return ProbeOutcome({"receipt_ids": ("receipt-1",)})


class _Validator:
    def validate(self, context: PhaseContext, probes: ProbeOutcome) -> ValidationOutcome:
        return ValidationOutcome(probes.payload)


class _Reducer:
    def __init__(self) -> None:
        self.calls = 0

    def reduce_without_probes(
        self,
        context: PhaseContext,
        frontier: ScoredFrontier,
        *,
        logical_step_id: str,
    ) -> ReductionOutcome:
        return ReductionOutcome(
            transitions=(),
            frontier=frontier,
            ledger_digest="ledger-empty",
        )

    def reduce(
        self,
        context: PhaseContext,
        validated: ValidationOutcome,
        frontier: ScoredFrontier,
        *,
        logical_step_id: str,
    ) -> ReductionOutcome:
        self.calls += 1
        return ReductionOutcome(
            transitions=("new",),
            frontier=frontier,
            ledger_digest="ledger-1",
            goal_satisfied=True,
        )


class _Finalizer:
    def __init__(
        self,
        journal: JsonlExplorationJournal,
        recovery: JsonSupervisorRecoveryStore,
    ) -> None:
        self.journal = journal
        self.recovery = recovery
        self.calls = 0

    def synthesize(
        self,
        context: PhaseContext,
        reduction: ReductionOutcome,
        *,
        logical_step_id: str,
    ) -> FinalizationOutcome:
        self.calls += 1
        outcome = FinalizationOutcome("exploration-eval/xpl-driver/report.json")
        self.journal.append_new("llm_call_started", call_id="synthesize-call-1")
        self.recovery.remember(logical_step_id, outcome)
        self.journal.append_new(
            "llm_call_completed",
            call_id="synthesize-call-1",
            step_id=logical_step_id,
            response_digest="synthesize-response-1",
        )
        return outcome

    def render_deterministic(
        self, context: PhaseContext, reduction: ReductionOutcome
    ) -> FinalizationOutcome:
        raise AssertionError("the scripted synthesis path should be available")


def test_shadow_driver_runs_full_phase_chain_and_is_idempotent_after_stop(
    tmp_path: Path,
) -> None:
    exploration_id = "xpl-driver"
    run_root = tmp_path / "exploration-eval" / exploration_id
    journal = JsonlExplorationJournal(run_root / "journal.jsonl")
    recovery = JsonSupervisorRecoveryStore(run_root / "phase-responses")
    generator = _Generator(journal, recovery)
    executor = _Executor(journal)
    reducer = _Reducer()
    finalizer = _Finalizer(journal, recovery)

    first = run_shadow_exploration(
        workspace=tmp_path,
        exploration_id=exploration_id,
        policy=_policy(),
        code_fingerprint="code-v1",
        data_state_witness="witness-v1",
        config=SupervisorConfig(admission_score_threshold=0.5),
        generator=generator,
        scheduler=_Scheduler(),
        executor=executor,
        validator=_Validator(),
        reducer=reducer,
        finalizer=finalizer,
        witness=CallableWitnessPort(lambda expected: expected == "witness-v1"),
        projection_data=lambda: ShadowProjectionData(
            insight_records=(_insight(),),
            coverage_completed=("region_difference",),
            coverage_unexplored=("spike_day", "missingness_mechanism"),
        ),
        journal=journal,
        recovery=recovery,
    )

    assert first.result.stop_reason == "completed"
    assert first.result.report_ref == "exploration-eval/xpl-driver/report.json"
    assert generator.calls == executor.calls == reducer.calls == finalizer.calls == 1
    projection = first.projection_path.read_text(encoding="utf-8")
    assert '"user_visible": false' in projection
    assert '"production_artifact_ids": []' in projection
    assert str(first.projection_path).startswith(str(run_root))

    second = run_shadow_exploration(
        workspace=tmp_path,
        exploration_id=exploration_id,
        policy=_policy(),
        code_fingerprint="code-v1",
        data_state_witness="witness-v1",
        config=SupervisorConfig(admission_score_threshold=0.5),
        generator=generator,
        scheduler=_Scheduler(),
        executor=executor,
        validator=_Validator(),
        reducer=reducer,
        finalizer=finalizer,
        witness=CallableWitnessPort(lambda expected: True),
        projection_data=lambda: ShadowProjectionData(
            insight_records=(_insight(),),
            coverage_completed=("region_difference",),
            coverage_unexplored=("spike_day", "missingness_mechanism"),
        ),
        journal=journal,
        recovery=recovery,
    )
    assert second.result.stop_reason == "completed"
    assert generator.calls == executor.calls == reducer.calls == finalizer.calls == 1
