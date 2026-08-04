"""Production-shaped E4a composition across real scheduler/executor/gates/reducer."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar, cast

import pandas as pd
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import BaseModel, ConfigDict

from eda_platform.agents.data_tools import DataToolContext, build_data_tools
from eda_platform.agents.exploration.candidates import (
    CandidateSeed,
    DatasetExplorationProfile,
    candidate_seed,
    mandatory_probe_seeds,
)
from eda_platform.agents.exploration.executor import (
    ProbeExecutionResult,
    durable_tool_result_digest,
)
from eda_platform.agents.exploration.scheduler import (
    AdmissionContext,
    CandidateSignals,
    PriorityWeights,
    SchedulerPolicy,
)
from eda_platform.agents.exploration.supervisor import (
    CandidateBatch,
    FrontierItem,
    PhaseContext,
    ProbeSelection,
    ReductionOutcome,
    SupervisorBudgetExhausted,
    SupervisorCancelled,
    SupervisorInvariantError,
    SupervisorPhase,
    reduction_outcome_digest,
)
from eda_platform.agents.exploration.workflow import (
    DeterministicSchedulerPort,
    ExecutedProbeBatch,
    ExplorationWorkflowState,
    JournaledCandidateGenerator,
    SupervisorProbeExecutorPort,
    artifact_receipt_decoder,
    candidate_batch_digest,
    final_reduction_state_digest,
    scheduling_decision_digest,
)
from eda_platform.agents.receipts import build_receipt
from eda_platform.agents.runtime import AgentTool, AgentToolResult
from eda_platform.agents.tool_context import current_execution_context
from eda_platform.core.budget import BudgetExceeded
from eda_platform.core.claim_gates import run_claim_gates
from eda_platform.core.exploration_budget import ToolCallProjection
from eda_platform.core.exploration_journal import (
    JsonlExplorationJournal,
    sealed_policy,
)
from eda_platform.core.exploration_profiles import build_exploration_policy
from eda_platform.core.exploration_release_gate import (
    E4aEvidenceBindings,
    E4aHardCaps,
    E4aTrialEvidence,
    attest_e4a_trial_evidence,
    issue_e4a_release_certificate,
    verify_e4a_trial_evidence,
)
from eda_platform.core.ids import make_artifact_id, stable_hash
from eda_platform.core.kernel import SessionCancelled
from eda_platform.core.llm import (
    LLMResultMetadata,
    LLMSettings,
    LLMToolCall,
    LLMToolResponse,
    LLMUsage,
)
from eda_platform.core.provider_registry import LLMProvider
from eda_platform.core.stat_registry import StatTestRegistry
from eda_platform.drivers.exploration import (
    CallableWitnessPort,
    JsonExplorationWorkflowStateStore,
    JsonlShadowBudgetStore,
    JsonSupervisorRecoveryStore,
    JsonToolResultStore,
    exploration_tool_capability_digest,
    run_composed_shadow_exploration,
)
from eda_platform.drivers.exploration_evidence_issuer import (
    E4aCheckerResult,
    E4aEvidenceIssuerBindings,
    E4aEvidenceRunSpec,
    E4aExpectedStructure,
    E4aGroundTruthFixture,
    E4aPlannedTrial,
    _artifact_digests,
    _candidate_identity,
    _search_dynamics_scores,
    _verify_branch_abandonments,
    _verify_trial_plan,
    issue_e4a_release_from_evidence_roots,
    verify_e4a_evidence_root,
)
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.claims import ClaimBundle
from eda_platform.schemas.datasets import DatasetRecord
from eda_platform.schemas.exploration import (
    BranchAbandonedEvent,
    InsightFamily,
    LlmCallStartedEvent,
    RoundSettledEvent,
    RoundStartedEvent,
)
from eda_platform.schemas.exploration_budget import (
    BudgetCapIncrease,
    ExplorationBranchPolicy,
)
from eda_platform.schemas.hypotheses import (
    HypothesisPredicate,
    HypothesisProposal,
    HypothesisProposalBatch,
)
from eda_platform.schemas.receipts import (
    EvidenceReceipt,
    ReceiptExecution,
    ReceiptFact,
    ReceiptMethod,
    ReceiptScope,
    ReceiptStatistics,
    data_state_witness_digest,
    receipt_content_digest,
    verify_receipt_digest,
)
from eda_platform.schemas.sessions import TraceEvent
from eda_platform.tools.loader import LoadedDataset
from eda_platform.tools.sql_runner import build_catalog

WITNESS = "dsw1_" + "a" * 32
T = TypeVar("T", bound=BaseModel)


class _NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Provider:
    settings = LLMSettings(
        provider=LLMProvider.OPENAI,
        model="gpt-5.6-terra",
        max_tokens=100,
        usd_per_1k_prompt=0.001,
        usd_per_1k_completion=0.001,
    )

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.structured_calls = 0
        self.tool_calls = 0
        self._last: LLMResultMetadata | None = None

    def structured(
        self,
        *,
        task: str,
        schema: type[T],
        payload: dict[str, Any],
    ) -> T:
        assert self.enabled
        assert task == "exploration_generate_hypotheses"
        assert payload["round_index"] == 0
        self.structured_calls += 1
        self._record_usage()
        return schema.model_validate(
            HypothesisProposalBatch(proposals=(_proposal(),)).model_dump(mode="json")
        )

    def text(self, *, task: str, payload: dict[str, Any]) -> str:
        raise AssertionError(f"unexpected text call {task}: {payload}")

    def tool_call(
        self,
        *,
        task: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMToolResponse:
        assert self.enabled
        assert task == "exploration_probe_loop"
        assert messages and [tool["name"] for tool in tools] == ["profile_slice"]
        self.tool_calls += 1
        self._record_usage()
        if self.tool_calls == 1:
            return LLMToolResponse(
                tool_calls=[
                    LLMToolCall(
                        call_id="provider-call-1",
                        name="profile_slice",
                        arguments={},
                    )
                ],
                finish_reason="tool_calls",
            )
        return LLMToolResponse(content="Probe complete.", finish_reason="stop")

    def last_usage(self) -> LLMResultMetadata | None:
        return self._last

    def _record_usage(self) -> None:
        self._last = LLMResultMetadata(
            provider="openai",
            model="gpt-5.6-terra",
            usage=LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            estimated_cost_usd=0.000015,
            usage_reported=True,
        )


class _UsageMeter:
    def __init__(self, kind: str = "profile_slice") -> None:
        self.kind = kind

    def project(self, **kwargs: Any) -> ToolCallProjection:
        # tool_kind is bound into the journal's tool_call_started event and the
        # issuer re-derives it from the receipt, so it must follow the call.
        call = kwargs.get("call")
        kind = getattr(call, "name", None) or self.kind
        return ToolCallProjection(kind=kind, rows_scanned=60, result_cells=20)

    def success(self, *, projected: ToolCallProjection, **_kwargs: Any) -> ToolCallProjection:
        return projected

    def failure(self, *, projected: ToolCallProjection, **_kwargs: Any) -> ToolCallProjection:
        return projected


class _DataProvider(_Provider):
    def __init__(
        self,
        proposal: HypothesisProposal,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        extra_calls: tuple[tuple[str, dict[str, Any]], ...] = (),
    ) -> None:
        super().__init__()
        self.proposal = proposal
        self.tool_name = tool_name
        self.arguments = arguments
        self.extra_calls = extra_calls

    def structured(
        self,
        *,
        task: str,
        schema: type[T],
        payload: dict[str, Any],
    ) -> T:
        assert task == "exploration_generate_hypotheses"
        assert payload["round_index"] == 0
        self.structured_calls += 1
        self._record_usage()
        return schema.model_validate(
            HypothesisProposalBatch(proposals=(self.proposal,)).model_dump(mode="json")
        )

    def tool_call(
        self,
        *,
        task: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMToolResponse:
        assert task == "exploration_probe_loop"
        assert messages and [tool["name"] for tool in tools] == [
            self.tool_name,
            *(name for name, _ in self.extra_calls),
        ]
        self.tool_calls += 1
        self._record_usage()
        if self.tool_calls == 1:
            return LLMToolResponse(
                tool_calls=[
                    LLMToolCall(
                        call_id="provider-data-call-1",
                        name=self.tool_name,
                        arguments=self.arguments,
                    ),
                    *(
                        LLMToolCall(
                            call_id=f"provider-data-call-{index + 2}",
                            name=name,
                            arguments=arguments,
                        )
                        for index, (name, arguments) in enumerate(self.extra_calls)
                    ),
                ],
                finish_reason="tool_calls",
            )
        return LLMToolResponse(content="Probe complete.", finish_reason="stop")


def _proposal() -> HypothesisProposal:
    return HypothesisProposal(
        statement="Does revenue differ by region?",
        rationale="Check the planted region structure.",
        expected_evidence="A deterministic regional observation.",
        falsification_conditions=("No regional difference is observed.",),
        family=InsightFamily.DIAGNOSTIC,
        method_family="profile_slice",
        dataset_ids=("ds-1",),
        columns=("region", "revenue"),
        probe_kind="region_difference",
        predicate=HypothesisPredicate(metric="revenue", operator="differs", left_operand="region"),
    )


def _tool() -> AgentTool:
    def execute(_arguments: BaseModel) -> AgentToolResult:
        execution = current_execution_context()
        assert execution is not None
        receipt = build_receipt(
            tool_call_id=execution.provider_call_id,
            tool_name="profile_slice",
            tool_version="1",
            arguments={},
            raw_output={"regional_difference": 42},
            artifact_ids=(),
            result_count=1,
            scope=ReceiptScope(
                dataset_ids=("ds-1",),
                columns=("region", "revenue"),
                scope_resolution="explicit",
            ),
            facts=(
                ReceiptFact(
                    fact_id="regional_difference",
                    name="regional difference",
                    value=42,
                    value_type="number",
                    unit="raw",
                ),
            ),
            method=ReceiptMethod(family="catalog_inspection"),
            statistics=ReceiptStatistics(
                hypothesis_id=candidate_seed(_proposal(), sequence_index=1).hypothesis_id,
                hypothesis_outcome="supports",
                test_name="independent_t_test",
                test_statistic=2.5,
                p_value=0.01,
                adjusted_p_value=0.01,
                effect_size=0.5,
                ci_low=0.1,
                ci_high=0.9,
                sample_size=20,
                sequence_index=1,
            ),
            data_state_witness=WITNESS,
            created_at="2026-08-02T00:00:00Z",
            execution=ReceiptExecution(
                run_id=execution.run_id,
                provider_call_id=execution.provider_call_id,
                logical_step_id=execution.logical_step_id,
                attempt_epoch=execution.attempt_epoch,
                sequence_index=execution.sequence_index,
            ),
        )
        artifact = Artifact(
            id=receipt.receipt_id,
            type=ArtifactType.EVIDENCE_RECEIPT,
            project_id="shadow-project",
            session_id="shadow-session",
            payload=receipt.model_dump(mode="json"),
        )
        return AgentToolResult(
            content={"regional_difference": 42},
            receipt_artifact=artifact,
        )

    return AgentTool(
        name="profile_slice",
        description="Inspect the immutable data catalog.",
        args_schema=_NoArgs,
        execute=execute,
    )


def _scheduler_policy() -> SchedulerPolicy:
    return SchedulerPolicy(
        scoring_policy_version="scheduler-v1",
        weights=PriorityWeights(
            business_value=1,
            information_gain_proxy=0,
            novelty=0,
            coverage_gap=0,
            feasibility=0,
            expected_cost=0,
            redundancy=0,
            multiplicity_risk=0,
        ),
        admission_priority=0.5,
        no_information_priority=0.5,
        max_batch_size=1,
    )


def test_composed_shadow_workflow_runs_and_recovers_without_reissuing(
    tmp_path: Path,
) -> None:
    tool = _tool()
    policy = build_exploration_policy(
        tier="quick",
        dataset_scope=("ds-1",),
        tool_capability_digest=exploration_tool_capability_digest((tool,)),
    )

    def admission(_context: Any) -> AdmissionContext:
        seed = candidate_seed(_proposal(), sequence_index=1)
        return AdmissionContext(
            dataset_columns={"ds-1": frozenset({"region", "revenue"})},
            allowed_dataset_ids=frozenset({"ds-1"}),
            supported_method_families=frozenset({"profile_slice"}),
            historical_hypothesis_fingerprints=frozenset(),
            answered_hypothesis_fingerprints=frozenset(),
            executed_query_fingerprints=frozenset(),
            remaining_cost=1,
            family_quota_remaining={InsightFamily.DIAGNOSTIC: 1},
            unexplored_coverage_keys=frozenset({seed.coverage_key}),
        )

    def signals(_context: Any, seeds: tuple[Any, ...]) -> dict[str, CandidateSignals]:
        return {seed.hypothesis_id: CandidateSignals(business_value=1) for seed in seeds}

    provider = _Provider()
    journal = JsonlExplorationJournal(
        tmp_path / "exploration-eval" / "xpl-composed" / "journal.jsonl"
    )
    journal.initialize(
        exploration_id="xpl-composed",
        policy=policy,
        code_fingerprint="code-v1",
        data_state_witness=WITNESS,
    )
    amended_state = journal.amend_budget(
        amendment_id="amend-report-fingerprint",
        increase=BudgetCapIncrease(max_rounds=1),
    )
    first = run_composed_shadow_exploration(
        workspace=tmp_path,
        exploration_id="xpl-composed",
        policy=policy,
        code_fingerprint="code-v1",
        data_state_witness=WITNESS,
        provider=provider,
        tools=(tool,),
        dataset_profiles=(),
        scheduler_policy=_scheduler_policy(),
        admission_context=admission,
        signals=signals,
        witness=CallableWitnessPort(lambda expected: expected == WITNESS),
        usage_meter=_UsageMeter(),
        stat_attempt_counts=lambda: {
            candidate_seed(_proposal(), sequence_index=1).hypothesis_id: 1
        },
        goal_satisfied=lambda state: bool(state.insights),
    )

    assert first.result.stop_reason == "completed", first.result.error
    assert provider.structured_calls == 1
    assert provider.tool_calls == 2
    assert first.result.report_ref == "exploration-eval/xpl-composed/report.md"
    report = (tmp_path / first.result.report_ref).read_text(encoding="utf-8")
    assert amended_state.effective_policy_fingerprint in report
    assert f"- policy_fingerprint: {policy.policy_fingerprint}" not in report
    assert "**Does revenue differ by region?**" in report
    assert "## Supported insights" in report
    assert "## Refuted hypotheses" in report
    assert "## Inconclusive questions" in report
    assert "## Data and method limitations" in report
    assert "## Coverage gaps / not explored" in report
    assert "## Cost and structured stop reason" in report
    assert "Confirmed findings" not in report
    projection = first.projection_path.read_text(encoding="utf-8")
    assert '"insight_records": [' in projection
    assert '"user_visible": false' in projection

    logical_call_ids = {
        event.call_id for event in journal.events() if isinstance(event, LlmCallStartedEvent)
    }
    physical_events = JsonlShadowBudgetStore(
        tmp_path / "exploration-eval" / "xpl-composed" / "llm-budget.jsonl"
    ).events()
    assert physical_events
    assert {event.summary.get("logical_call_id") for event in physical_events} == logical_call_ids

    disabled_provider = _Provider(enabled=False)
    second = run_composed_shadow_exploration(
        workspace=tmp_path,
        exploration_id="xpl-composed",
        policy=policy,
        code_fingerprint="code-v1",
        data_state_witness=WITNESS,
        provider=disabled_provider,
        tools=(tool,),
        dataset_profiles=(),
        scheduler_policy=_scheduler_policy(),
        admission_context=admission,
        signals=signals,
        witness=CallableWitnessPort(lambda expected: expected == WITNESS),
        usage_meter=_UsageMeter(),
        stat_attempt_counts=lambda: {
            candidate_seed(_proposal(), sequence_index=1).hypothesis_id: 1
        },
        goal_satisfied=lambda state: bool(state.insights),
    )
    assert second.result.stop_reason == "completed"
    assert disabled_provider.structured_calls == 0
    assert disabled_provider.tool_calls == 0

    workflow_state_path = tmp_path / "exploration-eval" / "xpl-composed" / "workflow-state.json"
    original_workflow_state = workflow_state_path.read_text(encoding="utf-8")
    tampered_workflow_state = json.loads(original_workflow_state)
    tampered_workflow_state["decisions"][0]["admission_checks"][0]["detail_code"] = "forged"
    workflow_state_path.write_text(json.dumps(tampered_workflow_state), encoding="utf-8")
    with pytest.raises(ValueError, match="scheduler decisions fail"):
        run_composed_shadow_exploration(
            workspace=tmp_path,
            exploration_id="xpl-composed",
            policy=policy,
            code_fingerprint="code-v1",
            data_state_witness=WITNESS,
            provider=disabled_provider,
            tools=(tool,),
            dataset_profiles=(),
            scheduler_policy=_scheduler_policy(),
            admission_context=admission,
            signals=signals,
            witness=CallableWitnessPort(lambda expected: expected == WITNESS),
            usage_meter=_UsageMeter(),
            stat_attempt_counts=lambda: {
                candidate_seed(_proposal(), sequence_index=1).hypothesis_id: 1
            },
            goal_satisfied=lambda state: bool(state.insights),
        )
    workflow_state_path.write_text(original_workflow_state, encoding="utf-8")

    tool_body = next(
        (tmp_path / "exploration-eval" / "xpl-composed" / "tool-results").glob("*.json")
    )
    tampered = json.loads(tool_body.read_text(encoding="utf-8"))
    tampered["usage"]["rows_scanned"] = 2
    tool_body.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="fails its body digest"):
        run_composed_shadow_exploration(
            workspace=tmp_path,
            exploration_id="xpl-composed",
            policy=policy,
            code_fingerprint="code-v1",
            data_state_witness=WITNESS,
            provider=disabled_provider,
            tools=(tool,),
            dataset_profiles=(),
            scheduler_policy=_scheduler_policy(),
            admission_context=admission,
            signals=signals,
            witness=CallableWitnessPort(lambda expected: expected == WITNESS),
            usage_meter=_UsageMeter(),
            stat_attempt_counts=lambda: {
                candidate_seed(_proposal(), sequence_index=1).hypothesis_id: 1
            },
            goal_satisfied=lambda state: bool(state.insights),
        )


def test_official_composition_deterministically_adjudicates_real_data_tools(
    tmp_path: Path,
) -> None:
    cases = (
        (
            "stat",
            pd.DataFrame(
                {
                    "region": ["east"] * 30 + ["west"] * 30,
                    "revenue": [100.0 + index / 10 for index in range(30)]
                    + [10.0 + index / 10 for index in range(30)],
                }
            ),
            HypothesisProposal(
                statement="Revenue differs by region.",
                rationale="Exercise the guarded group comparison.",
                expected_evidence="A registered test with effect and interval.",
                falsification_conditions=("No detectable group difference.",),
                family=InsightFamily.DIAGNOSTIC,
                method_family="run_stat_test",
                dataset_ids=("ds-real",),
                columns=("region", "revenue"),
                probe_kind="region_difference",
                predicate=HypothesisPredicate(
                    metric="revenue", operator="differs", left_operand="region"
                ),
            ),
            "run_stat_test",
            {
                "dataset_id": "ds-real",
                "test_type": "independent_t_test",
                "group_column": "region",
                "value_column": "revenue",
            },
            "supports",
        ),
        (
            "missingness",
            pd.DataFrame(
                {
                    "channel": ["phone"] * 30 + ["web"] * 30,
                    "score": [None] * 24
                    + [float(index) for index in range(6)]
                    + [None] * 3
                    + [float(index) for index in range(27)],
                }
            ),
            HypothesisProposal(
                statement="Score missingness differs materially by channel.",
                rationale="Exercise observed missing-rate ranges.",
                expected_evidence="At least a five percentage-point range.",
                falsification_conditions=("The range is below five points.",),
                family=InsightFamily.DIAGNOSTIC,
                method_family="diagnose_missingness",
                dataset_ids=("ds-real",),
                columns=("score", "channel"),
                probe_kind="missingness_mechanism",
                predicate=HypothesisPredicate(
                    metric="score",
                    operator="associated_with",
                    right_operand="channel",
                    threshold=5.0,
                ),
            ),
            "diagnose_missingness",
            {"dataset_id": "ds-real", "group_columns": ["channel"]},
            "supports",
        ),
        (
            "spike",
            pd.DataFrame(
                {
                    "order_date": pd.date_range("2026-01-01", periods=35, freq="D"),
                    "revenue": [10.0] * 17 + [1000.0] + [10.0] * 17,
                }
            ),
            HypothesisProposal(
                statement="Revenue contains a date-localized spike.",
                rationale="Exercise robust series spike detection.",
                expected_evidence="A dated robust deviation above 3.5.",
                falsification_conditions=("No robust date-localized spike.",),
                family=InsightFamily.EXPLORATORY,
                method_family="analyze_time_series",
                dataset_ids=("ds-real",),
                columns=("order_date", "revenue"),
                probe_kind="spike_day",
                predicate=HypothesisPredicate(
                    metric="revenue",
                    operator="has_spike",
                    right_operand="order_date",
                    threshold=3.5,
                ),
            ),
            "analyze_time_series",
            {
                "dataset_id": "ds-real",
                "time_column": "order_date",
                "value_column": "revenue",
                "freq": "D",
                "period": 7,
                "agg": "sum",
            },
            "supports",
        ),
    )

    for name, frame, proposal, tool_name, arguments, expected_outcome in cases:
        dataset = LoadedDataset(
            record=DatasetRecord(
                dataset_id="ds-real",
                name=f"{name}.csv",
                path=Path(f"/data/{name}.csv"),
                content_hash=f"hash-{name}",
            ),
            frame=frame,
        )
        data_context = DataToolContext(
            datasets=[dataset],
            catalog=build_catalog([dataset]),
            project_id="shadow-project",
            session_id=f"xpl-real-{name}",
            store=None,
            payload_policy="schema+aggregates",
        )
        tool = next(item for item in build_data_tools(data_context) if item.name == tool_name)
        provider = _DataProvider(
            proposal,
            tool_name=tool_name,
            arguments=arguments,
        )
        witness = data_state_witness_digest([("ds-real", None, dataset.record.content_hash)])
        policy = build_exploration_policy(
            tier="quick",
            dataset_scope=("ds-real",),
            tool_capability_digest=exploration_tool_capability_digest((tool,)),
        )
        seed = candidate_seed(proposal, sequence_index=1)

        def admission(
            _context: Any,
            *,
            _columns: tuple[str, ...] = tuple(str(column) for column in frame.columns),
            _method: str = proposal.method_family,
            _family: InsightFamily = proposal.family,
            _coverage_key: str = seed.coverage_key,
        ) -> AdmissionContext:
            return AdmissionContext(
                dataset_columns={"ds-real": frozenset(_columns)},
                allowed_dataset_ids=frozenset({"ds-real"}),
                supported_method_families=frozenset({_method}),
                historical_hypothesis_fingerprints=frozenset(),
                answered_hypothesis_fingerprints=frozenset(),
                executed_query_fingerprints=frozenset(),
                remaining_cost=1,
                family_quota_remaining={_family: 1},
                unexplored_coverage_keys=frozenset({_coverage_key}),
            )

        def signals(_context: Any, seeds: tuple[Any, ...]) -> dict[str, CandidateSignals]:
            return {item.hypothesis_id: CandidateSignals(business_value=1) for item in seeds}

        def stat_counts(*, _data_context: DataToolContext = data_context) -> dict[str, int]:
            counts: dict[str, int] = {}
            assert _data_context.stat_registry is not None
            for attempt in _data_context.stat_registry.attempts():
                counts[attempt.family_id] = counts.get(attempt.family_id, 0) + 1
            return counts

        result = run_composed_shadow_exploration(
            workspace=tmp_path / name,
            exploration_id=f"xpl-real-{name}",
            policy=policy,
            code_fingerprint="code-real-v1",
            data_state_witness=witness,
            provider=provider,
            tools=(tool,),
            dataset_profiles=(),
            scheduler_policy=_scheduler_policy(),
            admission_context=admission,
            signals=signals,
            witness=CallableWitnessPort(
                lambda expected, run_witness=witness: expected == run_witness
            ),
            usage_meter=_UsageMeter(tool_name),
            stat_attempt_counts=stat_counts,
            goal_satisfied=lambda state: bool(state.insights),
        )

        assert result.result.stop_reason == "completed", result.result.error
        state = JsonExplorationWorkflowStateStore(
            tmp_path / name / "exploration-eval" / f"xpl-real-{name}" / "workflow-state.json"
        ).load()
        assert len(state.committed_receipts) == 1
        receipt = next(iter(state.committed_receipts.values()))
        assert receipt.statistics is not None
        assert receipt.statistics.hypothesis_id == seed.hypothesis_id
        assert receipt.statistics.hypothesis_outcome == expected_outcome
        assert verify_receipt_digest(receipt)
        assert receipt.data_state_witness == witness

    stat_state = JsonExplorationWorkflowStateStore(
        tmp_path / "stat" / "exploration-eval" / "xpl-real-stat" / "workflow-state.json"
    ).load()
    assert any(insight.trust_level == "supported" for insight in stat_state.insights.values())


def test_shadow_budget_store_truncates_only_an_unterminated_tail(tmp_path: Path) -> None:
    path = tmp_path / "llm-budget.jsonl"
    store = JsonlShadowBudgetStore(path)
    first = TraceEvent(
        session_id="xpl",
        event_type="budget_reserved",
        name="generate",
        call_id="call-1",
        started_at=datetime(2026, 8, 2, tzinfo=UTC),
    )
    second = first.model_copy(update={"call_id": "call-2"})
    store.append(first)
    with path.open("ab") as handle:
        handle.write(b'{"interrupted":')

    assert store.events() == [first]
    store.append(second)
    assert store.events() == [first, second]


def _generate_context() -> PhaseContext:
    return PhaseContext(
        exploration_id="xpl-generate",
        round_index=0,
        phase=SupervisorPhase.GENERATE,
        data_state_witness=WITNESS,
        soft_countdown_context="remaining={}",
        completed_step_ids=frozenset(),
    )


def test_candidate_generator_never_resends_an_uncertain_logical_call(
    tmp_path: Path,
) -> None:
    policy = build_exploration_policy(
        tier="quick",
        dataset_scope=("ds-1",),
        tool_capability_digest="tools-v1",
    )
    path = tmp_path / "journal.jsonl"
    original = JsonlExplorationJournal(path)
    original.initialize(
        exploration_id="xpl-generate",
        policy=policy,
        code_fingerprint="code-v1",
        data_state_witness=WITNESS,
    )
    original.claim_recovery()
    step_id = "xpl-generate:round:0:generate"
    call_id = "llm_" + stable_hash(
        {"step_id": step_id, "phase": SupervisorPhase.GENERATE.value}, length=24
    )
    original.append_new("llm_call_started", call_id=call_id, step_id=step_id)

    recovered = JsonlExplorationJournal(path)
    state = recovered.claim_recovery()
    assert call_id in state.uncertain_call_ids
    provider = _Provider(enabled=False)
    generator = JournaledCandidateGenerator(
        provider=provider,
        journal=recovered,
        recovery=JsonSupervisorRecoveryStore(tmp_path / "responses"),
        dataset_profiles=(),
    )

    with pytest.raises(SupervisorInvariantError, match="refusing to resend"):
        generator.generate(_generate_context(), logical_step_id=step_id)

    assert provider.structured_calls == 0


def test_candidate_generator_preflight_budget_failure_writes_no_started_event(
    tmp_path: Path,
) -> None:
    class _PreflightBlockedProvider(_Provider):
        def preflight_structured(self, **_kwargs: Any) -> None:
            raise BudgetExceeded("request budget exhausted")

    policy = build_exploration_policy(
        tier="quick",
        dataset_scope=("ds-1",),
        tool_capability_digest="tools-v1",
    )
    journal = JsonlExplorationJournal(tmp_path / "journal.jsonl")
    journal.initialize(
        exploration_id="xpl-generate",
        policy=policy,
        code_fingerprint="code-v1",
        data_state_witness=WITNESS,
    )
    journal.claim_recovery()
    provider = _PreflightBlockedProvider()
    generator = JournaledCandidateGenerator(
        provider=provider,
        journal=journal,
        recovery=JsonSupervisorRecoveryStore(tmp_path / "responses"),
        dataset_profiles=(),
    )

    with pytest.raises(SupervisorBudgetExhausted, match="request budget exhausted"):
        generator.generate(_generate_context(), logical_step_id="xpl-generate:round:0:generate")

    assert provider.structured_calls == 0
    assert all(event.event_type != "llm_call_started" for event in journal.events())


_ROOT_EVIDENCE_PRIVATE = bytes.fromhex("31" * 32)


def _build_evidence_issuer_root(
    tmp_path: Path,
    *,
    with_probe_only_tool: bool = False,
) -> tuple[Path, E4aEvidenceIssuerBindings]:
    proposal = HypothesisProposal(
        statement="Revenue differs by region.",
        rationale="Exercise the production statistical comparison.",
        expected_evidence="A registered test with effect and interval.",
        falsification_conditions=("No detectable group difference.",),
        family=InsightFamily.DIAGNOSTIC,
        method_family="compare_groups",
        dataset_ids=("ds-1",),
        columns=("region", "revenue"),
        probe_kind="region_difference",
        predicate=HypothesisPredicate(
            metric="revenue", operator="differs", left_operand="region"
        ),
    )
    dataset = LoadedDataset(
        record=DatasetRecord(
            dataset_id="ds-1",
            name="evidence.csv",
            path=Path("/data/evidence.csv"),
            content_hash="evidence-hash-v1",
        ),
        frame=pd.DataFrame(
            {
                "region": ["east"] * 30 + ["west"] * 30,
                "revenue": [100.0 + index / 10 for index in range(30)]
                + [10.0 + index / 10 for index in range(30)],
            }
        ),
    )
    data_context = DataToolContext(
        datasets=[dataset],
        catalog=build_catalog([dataset]),
        project_id="shadow-project",
        session_id="xpl-evidence-root",
        store=None,
        payload_policy="schema+aggregates",
        stat_registry=StatTestRegistry(
            tmp_path
            / "exploration-eval"
            / "xpl-evidence-root"
            / "stat_registry.jsonl"
        ),
    )
    assert data_context.stat_registry is not None
    stat_registry = data_context.stat_registry
    built = {item.name: item for item in build_data_tools(data_context)}
    tool = built["run_stat_test"]
    # profile_slice has no durable result contract: as reconnaissance no insight
    # cites, it must stay issuable (option B).
    extra_calls: tuple[tuple[str, dict[str, Any]], ...] = (
        (("profile_slice", {"dataset_id": "ds-1", "columns": ["revenue"]}),)
        if with_probe_only_tool
        else ()
    )
    tools = (tool, *(built[name] for name, _ in extra_calls))
    run_witness = data_state_witness_digest(
        [("ds-1", None, dataset.record.content_hash)]
    )
    policy = build_exploration_policy(
        tier="quick",
        dataset_scope=("ds-1",),
        tool_capability_digest=exploration_tool_capability_digest(tools),
    )
    seed = candidate_seed(proposal, sequence_index=1)

    def admission(_context: Any) -> AdmissionContext:
        return AdmissionContext(
            dataset_columns={"ds-1": frozenset({"region", "revenue"})},
            allowed_dataset_ids=frozenset({"ds-1"}),
            supported_method_families=frozenset({"compare_groups"}),
            historical_hypothesis_fingerprints=frozenset(),
            answered_hypothesis_fingerprints=frozenset(),
            executed_query_fingerprints=frozenset(),
            remaining_cost=1,
            family_quota_remaining={InsightFamily.DIAGNOSTIC: 1},
            unexplored_coverage_keys=frozenset({seed.coverage_key}),
        )

    def signals(_context: Any, seeds: tuple[Any, ...]) -> dict[str, CandidateSignals]:
        return {item.hypothesis_id: CandidateSignals(business_value=1) for item in seeds}

    def stat_counts() -> dict[str, int]:
        counts: dict[str, int] = {}
        for attempt in stat_registry.attempts():
            counts[attempt.family_id] = counts.get(attempt.family_id, 0) + 1
        return counts

    journal = JsonlExplorationJournal(
        tmp_path / "exploration-eval" / "xpl-evidence-root" / "journal.jsonl"
    )
    journal.initialize(
        exploration_id="xpl-evidence-root",
        policy=policy,
        code_fingerprint="code-evidence-v1",
        data_state_witness=run_witness,
    )
    journal.amend_budget(
        amendment_id="issuer-chain-amendment",
        increase=BudgetCapIncrease(max_rounds=1),
    )
    result = run_composed_shadow_exploration(
        workspace=tmp_path,
        exploration_id="xpl-evidence-root",
        policy=policy,
        code_fingerprint="code-evidence-v1",
        data_state_witness=run_witness,
        provider=_DataProvider(
            proposal,
            tool_name="run_stat_test",
            arguments={
                "dataset_id": "ds-1",
                "test_type": "independent_t_test",
                "group_column": "region",
                "value_column": "revenue",
            },
            extra_calls=extra_calls,
        ),
        tools=tools,
        dataset_profiles=(),
        scheduler_policy=_scheduler_policy(),
        admission_context=admission,
        signals=signals,
        witness=CallableWitnessPort(lambda expected: expected == run_witness),
        usage_meter=_UsageMeter("run_stat_test"),
        stat_attempt_counts=stat_counts,
        goal_satisfied=lambda state: bool(state.insights),
    )
    assert result.result.stop_reason == "completed"
    root = result.journal_path.parent
    fixture = E4aGroundTruthFixture(
        item_id="planted_retail_v1",
        expected_structures=(
            E4aExpectedStructure(
                structure_id="region_difference",
                target_metric="region_difference_recall",
                tool_names=("run_stat_test",),
                required_columns=("region", "revenue"),
                predicate=proposal.predicate,
            ),
        ),
    )
    certificate_bindings = E4aEvidenceBindings(
        checker_version="root-checker-v1",
        code_fingerprint="code-evidence-v1",
        tool_capability_digest=policy.tool_capability_digest,
        evidence_key_id="root-evidence-key-v1",
    )
    issuer_bindings = E4aEvidenceIssuerBindings(
        certificate=certificate_bindings,
        fixture=fixture,
    )
    spec = E4aEvidenceRunSpec(
        item_id=fixture.item_id,
        seed=7,
        policy=policy,
        checker_version=certificate_bindings.checker_version,
        ground_truth_digest=fixture.digest,
    )
    (root / "e4a-run-spec.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    workflow = JsonExplorationWorkflowStateStore(root / "workflow-state.json").load()
    receipt = next(iter(workflow.committed_receipts.values()))
    assert receipt.statistics is not None and receipt.execution is not None
    journal_events = JsonlExplorationJournal(root / "journal.jsonl").events()
    no_information_rounds = 0
    for event in reversed(journal_events):
        if isinstance(event, RoundSettledEvent) and not event.progress:
            no_information_rounds += 1
        elif isinstance(event, RoundSettledEvent):
            break
    checker = E4aCheckerResult(
        checker_version=certificate_bindings.checker_version,
        evaluated_insight_ids=tuple(sorted(workflow.insights)),
        matched_structure_ids=("region_difference",),
        scores={
            "precision": 1.0,
            "recall": 1.0,
            "grounding_rate": 1.0,
            "fabricated_receipt_rate": 0.0,
            "spam_fixture_input_count": float(len(workflow.decisions)),
            "spam_fixture_canonical_groups": float(
                len({item.hypothesis_fingerprint for item in workflow.decisions})
            ),
            "no_information_rounds": float(no_information_rounds),
            "no_information_stopped": 0.0,
            "proof_reachability_rate": 1.0,
            "journal_provenance_rate": 1.0,
            "region_difference_recall": 1.0,
            "missingness_mechanism_recall": 0.0,
            "spike_day_recall": 0.0,
            "auc_over_steps": 1.0,
            "first_improvement_step": 0.0,
        },
    )
    (root / "e4a-checker-result.json").write_text(
        checker.model_dump_json(indent=2), encoding="utf-8"
    )
    return root, issuer_bindings


def _inject_trailing_unsettled_round(root: Path) -> str:
    """Append a round that never settles, in the real interruption shape.

    A budget latch mid-round (seed-6/seed-7) leaves ALL of: a generate
    recovery body, persisted scheduling decisions, committed orphan receipts,
    and a journal ahead of the last reduce. Probes only run on chosen
    candidates, so a trailing round with receipts but no decisions cannot
    occur — the fixture writes the full shape.
    """
    journal_path = root / "journal.jsonl"
    lines = journal_path.read_text(encoding="utf-8").splitlines()
    stopped = json.loads(lines[-1])
    assert stopped["event_type"] == "exploration_stopped"
    seq = int(stopped["seq"])
    exploration_id = stopped["exploration_id"]
    occurred = stopped["occurred_at"]
    epoch = stopped["attempt_epoch"]
    round_index = 1 + max(
        (
            int(json.loads(line)["round_index"])
            for line in lines
            if json.loads(line)["event_type"] == "round_started"
        ),
        default=-1,
    )
    orphan = build_receipt(
        tool_call_id="call-orphan-trailing",
        tool_name="profile_slice",
        tool_version="1",
        arguments={"dataset_id": "ds-1"},
        raw_output={"rows": []},
        artifact_ids=(),
        result_count=1,
        scope=ReceiptScope(
            dataset_ids=("ds-1",), columns=("region",), scope_resolution="explicit"
        ),
        facts=(
            ReceiptFact(
                fact_id="rows_in_slice",
                name="rows_in_slice",
                value=60,
                value_type="count",
                support_type="direct",
            ),
        ),
        method=ReceiptMethod(family="profile_slice"),
        data_state_witness=json.loads(lines[0])["data_state_witness"],
        created_at=occurred,
    )
    step_id = "step_" + "a" * 24
    body = {
        "logical_step_id": step_id,
        "result": {
            "content": "orphan slice",
            "artifacts": [],
            "receipt_artifact": Artifact(
                id=make_artifact_id("receipt", orphan.model_dump(mode="json")),
                type=ArtifactType.EVIDENCE_RECEIPT,
                payload=orphan.model_dump(mode="json"),
                created_at=occurred,
                project_id="shadow-project",
                session_id="xpl-evidence-root",
            ).model_dump(mode="json"),
        },
        "usage": {"kind": "profile_slice", "rows_scanned": 60, "result_cells": 20},
    }
    (root / "tool-results" / f"{stable_hash(step_id, length=32)}.json").write_text(
        json.dumps(body), encoding="utf-8"
    )

    def _event(kind: str, index: int, **fields: object) -> str:
        return json.dumps(
            {
                "schema_version": 1,
                "seq": seq + index,
                "exploration_id": exploration_id,
                "attempt_epoch": epoch,
                "occurred_at": occurred,
                "event_type": kind,
                **fields,
            }
        )

    body_digest = durable_tool_result_digest(
        JsonToolResultStore(root / "tool-results").load_required(step_id)
    )
    # The trailing round's generate completed and its admission decisions were
    # persisted before the latch — exactly what seed 7 left behind.
    trailing_proposal = HypothesisProposal(
        statement="Does revenue trend upward over the half year?",
        rationale="Trailing-round candidate the interruption stranded.",
        expected_evidence="A time-series diagnostic on daily revenue.",
        falsification_conditions=("No upward trend is detectable.",),
        family=InsightFamily.EXPLORATORY,
        method_family="analyze_time_series",
        dataset_ids=("ds-1",),
        columns=("region", "revenue"),
        probe_kind="trend",
        predicate=HypothesisPredicate(
            metric="revenue", operator="has_spike", left_operand="region"
        ),
    )
    trailing_seed = candidate_seed(trailing_proposal, sequence_index=90001)
    trailing_batch = CandidateBatch((trailing_seed,))
    generate_step = f"{exploration_id}:round:{round_index}:generate"
    JsonSupervisorRecoveryStore(root / "phase-responses").remember(
        generate_step, trailing_batch
    )
    raw_state = json.loads((root / "workflow-state.json").read_text(encoding="utf-8"))
    cloned = json.loads(json.dumps(raw_state["decisions"][-1]))
    cloned["hypothesis_id"] = trailing_seed.hypothesis_id
    cloned["hypothesis_fingerprint"] = trailing_seed.hypothesis_fingerprint
    cloned["family"] = trailing_seed.proposal.family.value
    raw_state["decisions"].append(cloned)
    (root / "workflow-state.json").write_text(json.dumps(raw_state), encoding="utf-8")
    # Billing settled before the latch: the ledger carries the trailing
    # generate's reservation/usage pair like any other call.
    ledger_path = root / "llm-budget.jsonl"
    ledger_lines = ledger_path.read_text(encoding="utf-8").splitlines()
    triple = [
        json.loads(line)
        for line in ledger_lines
        if json.loads(line)["event_type"]
        in ("budget_reserved", "llm_usage", "budget_settled")
    ][:3]
    assert [entry["event_type"] for entry in triple] == [
        "budget_reserved",
        "llm_usage",
        "budget_settled",
    ]
    trailing_tokens = 0
    trailing_cost = 0.0
    for entry in triple:
        entry = json.loads(json.dumps(entry))
        entry["call_id"] = "physical-trailing-generate"
        entry["summary"]["logical_call_id"] = "llm-trailing-generate"
        if entry["event_type"] == "llm_usage":
            trailing_tokens = entry["summary"].get("total_tokens", 0)
            trailing_cost = entry["summary"].get("estimated_cost_usd", 0.0)
        ledger_lines.append(json.dumps(entry))
    ledger_path.write_text("\n".join(ledger_lines) + "\n", encoding="utf-8")
    injected = [
        _event("round_started", 0, round_index=round_index),
        _event("llm_call_started", 1, call_id="llm-trailing-generate"),
        _event(
            "llm_call_completed",
            2,
            call_id="llm-trailing-generate",
            step_id=generate_step,
            response_digest=candidate_batch_digest(trailing_batch),
        ),
        _event(
            "tool_call_started",
            3,
            logical_step_id=step_id,
            tool_kind="profile_slice",
            input_fingerprint="orphan-input",
            projected_rows_scanned=60,
            projected_result_cells=20,
        ),
        _event(
            "receipt_prepared",
            4,
            logical_step_id=step_id,
            receipt_id=orphan.receipt_id,
            result_digest=body_digest,
        ),
        _event(
            "receipt_committed",
            5,
            logical_step_id=step_id,
            receipt_id=orphan.receipt_id,
            result_digest=body_digest,
            rows_scanned=60,
            result_cells=20,
        ),
    ]
    # A run that already latched "completed" cannot open another round, so the
    # last settled round becomes an ordinary one and the stop becomes the
    # budget latch that actually produces this shape.
    head = list(lines[:-1])
    for index in range(len(head) - 1, -1, -1):
        event = json.loads(head[index])
        if event["event_type"] == "round_settled":
            event["terminal_reason"] = None
            event["terminal_has_reduction"] = False
            head[index] = json.dumps(event)
            break
    stopped["seq"] = seq + len(injected)
    stopped["stop_reason"] = "budget_exhausted"
    journal_path.write_text(
        "\n".join([*head, *injected, json.dumps(stopped)]) + "\n",
        encoding="utf-8",
    )
    journal = JsonlExplorationJournal(journal_path)
    journal.write_snapshot(journal.rebuild())
    projection_path = root / "projection.json"
    projection = json.loads(projection_path.read_text(encoding="utf-8"))
    projection["last_seq"] = seq + len(injected)
    projection["stop_reason"] = "budget_exhausted"
    projection_path.write_text(json.dumps(projection), encoding="utf-8")
    checker_path = root / "e4a-checker-result.json"
    checker_raw = json.loads(checker_path.read_text(encoding="utf-8"))
    checker_raw["scores"]["spam_fixture_input_count"] = 2.0
    checker_raw["scores"]["spam_fixture_canonical_groups"] = 2.0
    checker_path.write_text(json.dumps(checker_raw, indent=2), encoding="utf-8")
    report_path = root / "report.md"
    report = report_path.read_text(encoding="utf-8")
    old_cost = next(
        line for line in report.splitlines() if line.startswith("- llm_cost_usd_used: ")
    )
    new_cost = "- llm_cost_usd_used: " + json.dumps(
        float(old_cost.removeprefix("- llm_cost_usd_used: ")) + trailing_cost
    )
    for old_line, new_line in (
        ("- stop_reason: completed", "- stop_reason: budget_exhausted"),
        ("- successful_tool_calls: 1", "- successful_tool_calls: 2"),
        ("- rows_scanned: 60", "- rows_scanned: 120"),
        ("- result_cells: 20", "- result_cells: 40"),
        ("- rounds_started: 1", f"- rounds_started: {round_index + 1}"),
        ("- llm_requests_used: 3", "- llm_requests_used: 4"),
        (
            "- llm_total_tokens_used: 45",
            f"- llm_total_tokens_used: {45 + trailing_tokens}",
        ),
        (old_cost, new_cost),
    ):
        assert old_line in report, old_line
        report = report.replace(old_line, new_line)
    report_path.write_text(report, encoding="utf-8")
    return orphan.receipt_id


def test_evidence_root_certifies_a_run_interrupted_mid_round(tmp_path: Path) -> None:
    """P0-1/P0-2 regression: relaxing _verify_committed_receipts alone left four
    other exact-set checks that reject the same interrupted root."""
    root, bindings = _build_evidence_issuer_root(tmp_path)
    orphan_id = _inject_trailing_unsettled_round(root)
    workflow = JsonExplorationWorkflowStateStore(root / "workflow-state.json").load()
    assert orphan_id not in workflow.committed_receipts

    trial, _manifest = verify_e4a_evidence_root(
        root,
        issuer_bindings=bindings,
        evidence_signing_key=_ROOT_EVIDENCE_PRIVATE,
    )

    # The orphan's spend is real and stays counted; its evidence is not certified.
    assert trial.usage.tool_calls == 2
    assert trial.usage.rows_scanned == 120


def test_trailing_decisions_not_covered_by_the_batch_still_reject(
    tmp_path: Path,
) -> None:
    """The seed-7 tolerance is not a blank cheque: a leftover decision that the
    trailing round's recovered candidate batch cannot account for is forgery."""
    root, bindings = _build_evidence_issuer_root(tmp_path)
    _inject_trailing_unsettled_round(root)
    state_path = root / "workflow-state.json"
    raw_state = json.loads(state_path.read_text(encoding="utf-8"))
    smuggled = json.loads(json.dumps(raw_state["decisions"][-1]))
    smuggled["hypothesis_id"] = "hyp_" + "d" * 24
    smuggled["hypothesis_fingerprint"] = "d" * 32
    raw_state["decisions"].append(smuggled)
    state_path.write_text(json.dumps(raw_state), encoding="utf-8")

    with pytest.raises(ValueError, match="do not exactly cover the candidate batch"):
        verify_e4a_evidence_root(
            root,
            issuer_bindings=bindings,
            evidence_signing_key=_ROOT_EVIDENCE_PRIVATE,
        )


def test_evidence_root_certifies_probe_only_reconnaissance_without_a_contract(
    tmp_path: Path,
) -> None:
    """Option B (user decision 2026-08-03): the tool allowlist has 11 entries and
    data_tool_result_contracts covers 3, so any real run using profile_slice was
    unissuable. Only receipts an insight rests on are release evidence."""
    root, bindings = _build_evidence_issuer_root(tmp_path, with_probe_only_tool=True)
    workflow = JsonExplorationWorkflowStateStore(root / "workflow-state.json").load()
    cited = {
        receipt_id
        for insight in workflow.insights.values()
        for receipt_id in (
            *insight.supporting_receipt_ids,
            *insight.contradicting_receipt_ids,
        )
    }
    by_tool = {
        receipt.tool_name: receipt for receipt in workflow.committed_receipts.values()
    }
    assert by_tool["run_stat_test"].receipt_id in cited
    # Committed, gated, in the claim bundle — but adjudicated by nothing, so it
    # is reconnaissance rather than release evidence.
    assert by_tool["profile_slice"].receipt_id not in cited

    trial, _manifest = verify_e4a_evidence_root(
        root,
        issuer_bindings=bindings,
        evidence_signing_key=_ROOT_EVIDENCE_PRIVATE,
    )
    assert trial.usage.tool_calls == 2


def _rewrite_first_workflow_receipt(root: Path, mutation: str) -> None:
    workflow_path = root / "workflow-state.json"
    raw = json.loads(workflow_path.read_text(encoding="utf-8"))
    receipt = raw["committed_receipts"][0]
    if mutation == "provider_call_id":
        receipt["execution"]["provider_call_id"] = "forged-provider-call"
    elif mutation == "logical_step_id":
        receipt["execution"]["logical_step_id"] = "step_" + "f" * 24
    elif mutation == "sequence_index":
        receipt["execution"]["sequence_index"] = 2
    elif mutation == "typed_outcome":
        receipt["statistics"]["hypothesis_outcome"] = "contradicts"
    else:  # pragma: no cover - test helper contract
        raise AssertionError(mutation)
    digest_payload = {key: value for key, value in receipt.items() if key != "content_digest"}
    receipt["content_digest"] = receipt_content_digest(digest_payload)
    workflow_path.write_text(json.dumps(raw), encoding="utf-8")


def _rewrite_provider_tool_call(root: Path, mutation: str) -> None:
    for response_path in sorted((root / "llm-responses").glob("*.json")):
        raw = json.loads(response_path.read_text(encoding="utf-8"))
        calls = raw["response"]["tool_calls"]
        if not calls:
            continue
        if mutation == "provider_arguments":
            calls[0]["arguments"]["value_column"] = "forged_revenue"
        elif mutation == "provider_name":
            calls[0]["name"] = "diagnose_missingness"
            calls[0]["arguments"] = {"dataset_id": "ds-1"}
        else:  # pragma: no cover - test helper contract
            raise AssertionError(mutation)
        response_path.write_text(json.dumps(raw), encoding="utf-8")
        return
    raise AssertionError("root contains no provider tool call")


def _rewrite_final_reduction_ledger(root: Path) -> None:
    workflow = JsonExplorationWorkflowStateStore(root / "workflow-state.json").load()
    ledger_digest = final_reduction_state_digest(workflow)
    reduction_path = next(
        path
        for path in (root / "phase-responses").glob("*.json")
        if json.loads(path.read_text(encoding="utf-8"))["logical_step_id"].endswith(
            ":reduce"
        )
    )
    reduction_raw = json.loads(reduction_path.read_text(encoding="utf-8"))
    logical_step_id = reduction_raw["logical_step_id"]
    reduction = JsonSupervisorRecoveryStore(root / "phase-responses").load_required(
        logical_step_id
    )
    assert isinstance(reduction, ReductionOutcome)
    rewritten_reduction = replace(reduction, ledger_digest=ledger_digest)
    reduction_raw["result"]["ledger_digest"] = ledger_digest
    reduction_path.write_text(json.dumps(reduction_raw), encoding="utf-8")
    rewritten_digest = reduction_outcome_digest(rewritten_reduction)

    journal_path = root / "journal.jsonl"
    rows = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()]
    event = next(row for row in rows if row["event_type"] == "reduction_committed")
    event["ledger_digest"] = ledger_digest
    event["reduction_digest"] = rewritten_digest
    journal_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    snapshot_path = root / "journal.snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["ledger_digest"] = ledger_digest
    snapshot["reduction_digest"] = rewritten_digest
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")


def _synchronize_receipt_contract_forgery(root: Path, mutation: str) -> None:
    workflow_path = root / "workflow-state.json"
    workflow_raw = json.loads(workflow_path.read_text(encoding="utf-8"))
    receipt_raw = workflow_raw["committed_receipts"][0]
    fabricated_claim_text: str | None = None
    forged_primary_id: str | None = None
    if mutation == "fact_value":
        fact = receipt_raw["facts"][0]
        fact["name"] = "fabricated_revenue"
        fact["value"] = 999999
        fabricated_claim_text = "fabricated_revenue: 999999"
    elif mutation == "statistics":
        receipt_raw["statistics"]["test_statistic"] = 999999
    elif mutation == "output_digest":
        receipt_raw["output_digest"] = "f" * 64
    elif mutation == "artifact_id":
        forged_primary_id = "stat_forged_payload"
        receipt_raw["artifact_ids"] = [forged_primary_id]
    else:  # pragma: no cover - test helper contract
        raise AssertionError(mutation)
    normalized_receipt = EvidenceReceipt.model_validate(receipt_raw)
    receipt_raw = normalized_receipt.model_dump(mode="json")
    receipt_raw["content_digest"] = receipt_content_digest(
        normalized_receipt.model_dump(mode="json", exclude={"content_digest"})
    )
    workflow_raw["committed_receipts"][0] = receipt_raw
    workflow_path.write_text(json.dumps(workflow_raw), encoding="utf-8")

    store = JsonExplorationWorkflowStateStore(workflow_path)
    workflow = store.load()
    bundle_id, bundle = next(iter(workflow.admitted_bundles.items()))
    if fabricated_claim_text is not None:
        claims = list(bundle.claims)
        claims[0] = claims[0].model_copy(update={"claim_text": fabricated_claim_text})
        bundle = bundle.model_copy(update={"claims": tuple(claims)})
        workflow.admitted_bundles[bundle_id] = bundle
    terminal = JsonlExplorationJournal(root / "journal.jsonl").rebuild()
    assert terminal is not None
    receipt = next(iter(workflow.committed_receipts.values()))
    assert receipt.statistics is not None
    family_id = receipt.statistics.statistical_family_id
    assert family_id is not None
    workflow.gate_reports[bundle_id] = run_claim_gates(
        bundle,
        committed_receipts=workflow.committed_receipts,
        run_witness=terminal.data_state_witness,
        stat_attempt_counts={family_id: 1},
    )
    store.remember(workflow)

    tool_path = next((root / "tool-results").glob("*.json"))
    tool_raw = json.loads(tool_path.read_text(encoding="utf-8"))
    receipt_artifact = tool_raw["result"]["receipt_artifact"]
    receipt_artifact["payload"] = receipt_raw
    receipt_artifact["id"] = make_artifact_id("receipt", receipt_raw)
    if forged_primary_id is not None:
        tool_raw["result"]["artifacts"][0]["id"] = forged_primary_id
        tool_raw["result"]["content"]["artifact_id"] = forged_primary_id
        receipt_artifact["parents"] = [forged_primary_id]
    tool_path.write_text(json.dumps(tool_raw), encoding="utf-8")
    logical_step_id = tool_raw["logical_step_id"]
    durable = JsonToolResultStore(root / "tool-results").load_required(logical_step_id)
    result_digest = durable_tool_result_digest(durable)

    journal_path = root / "journal.jsonl"
    rows = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if row["event_type"] in {"receipt_prepared", "receipt_committed"}:
            row["result_digest"] = result_digest
    journal_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    snapshot_path = root / "journal.snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["step_result_digests"][logical_step_id] = result_digest
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    _rewrite_final_reduction_ledger(root)
    # Claim texts are audit-plane only since the statement-led report format;
    # report.md does not render them, so the forged report needs no patching.


def test_evidence_root_rederives_usage_provider_checker_and_manifest(
    tmp_path: Path,
) -> None:
    root, bindings = _build_evidence_issuer_root(tmp_path)
    trial, manifest = verify_e4a_evidence_root(
        root,
        issuer_bindings=bindings,
        evidence_signing_key=_ROOT_EVIDENCE_PRIVATE,
    )
    assert (trial.provider, trial.model, trial.tier, trial.seed) == (
        "openai",
        "gpt-5.6-terra",
        "quick",
        7,
    )
    assert (trial.usage.llm_requests, trial.usage.total_tokens) == (3, 45)
    assert (
        trial.usage.tool_calls,
        trial.usage.rows_scanned,
        trial.usage.cells_scanned,
    ) == (1, 60, 20)
    assert trial.source_manifest_digest == manifest.root_digest
    public_key = (
        Ed25519PrivateKey.from_private_bytes(_ROOT_EVIDENCE_PRIVATE).public_key().public_bytes_raw()
    )
    assert verify_e4a_trial_evidence(trial, public_key)


def test_checker_scores_include_search_dynamics(tmp_path: Path) -> None:
    root, bindings = _build_evidence_issuer_root(tmp_path)
    trial, _ = verify_e4a_evidence_root(
        root,
        issuer_bindings=bindings,
        evidence_signing_key=_ROOT_EVIDENCE_PRIVATE,
    )
    assert trial.scores["auc_over_steps"] == 1.0
    assert trial.scores["first_improvement_step"] == 0.0


def test_manifest_digest_covers_numeric_changes(tmp_path: Path) -> None:
    """verify_e4a_evidence_root never reruns analysis tools, so the "did this
    number really come from this tool on this data" dimension is covered only
    by manifest.root_digest pinning every artifact byte."""
    root, bindings = _build_evidence_issuer_root(tmp_path)
    _trial, manifest = verify_e4a_evidence_root(
        root,
        issuer_bindings=bindings,
        evidence_signing_key=_ROOT_EVIDENCE_PRIVATE,
    )
    tool_path = next((root / "tool-results").glob("*.json"))
    tool_raw = json.loads(tool_path.read_text(encoding="utf-8"))
    tool_raw["result"]["artifacts"][0]["payload"]["p_value"] = 0.5
    tool_path.write_text(json.dumps(tool_raw), encoding="utf-8")
    mutated = manifest.model_copy(update={"artifact_sha256": _artifact_digests(root)})
    assert mutated.root_digest != manifest.root_digest


def test_trial_plan_rejects_unpinned_manifest_digest(tmp_path: Path) -> None:
    root, bindings = _build_evidence_issuer_root(tmp_path)
    trial, manifest = verify_e4a_evidence_root(
        root,
        issuer_bindings=bindings,
        evidence_signing_key=_ROOT_EVIDENCE_PRIVATE,
    )
    assert trial.tier == "quick"
    pinned = E4aPlannedTrial(
        role="baseline",
        trial_id=trial.trial_id,
        tier="quick",
        seed=trial.seed,
        manifest_digest=manifest.root_digest,
    )
    _verify_trial_plan([(trial, manifest)], [], (pinned,))
    unpinned = pinned.model_copy(update={"manifest_digest": "0" * 64})
    with pytest.raises(ValueError, match="not pinned"):
        _verify_trial_plan([(trial, manifest)], [], (unpinned,))


def test_search_dynamics_scores_formula() -> None:
    # R=3, expected=4, matches admitted at rounds 0 and 2:
    # recall_at_r = 0.25, 0.25, 0.5 -> auc = round(1/3, 6)
    assert _search_dynamics_scores(
        matched_insight_rounds=(0, 2), expected_count=4, rounds_started=3
    ) == {"auc_over_steps": 0.333333, "first_improvement_step": 0.0}
    assert _search_dynamics_scores(
        matched_insight_rounds=(), expected_count=4, rounds_started=3
    ) == {"auc_over_steps": 0.0, "first_improvement_step": -1.0}
    assert _search_dynamics_scores(
        matched_insight_rounds=(), expected_count=4, rounds_started=0
    ) == {"auc_over_steps": 0.0, "first_improvement_step": -1.0}


@pytest.mark.parametrize(
    "mutation",
    (
        "provider_call_id",
        "logical_step_id",
        "sequence_index",
        "provider_arguments",
        "provider_name",
        "typed_outcome",
        "stat_arguments_digest",
    ),
)
def test_real_data_tool_root_rejects_invocation_and_adjudication_tampering(
    tmp_path: Path, mutation: str
) -> None:
    root, bindings = _build_evidence_issuer_root(tmp_path / mutation)
    if mutation in {
        "provider_call_id",
        "logical_step_id",
        "sequence_index",
        "typed_outcome",
    }:
        _rewrite_first_workflow_receipt(root, mutation)
    elif mutation in {"provider_arguments", "provider_name"}:
        _rewrite_provider_tool_call(root, mutation)
    else:
        registry_path = root / "stat_registry.jsonl"
        rows = [
            json.loads(line)
            for line in registry_path.read_text(encoding="utf-8").splitlines()
        ]
        started = next(row for row in rows if row["event"] == "attempt_started")
        started["arguments_digest"] = "f" * 64
        registry_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    with pytest.raises(ValueError):
        verify_e4a_evidence_root(
            root,
            issuer_bindings=bindings,
            evidence_signing_key=_ROOT_EVIDENCE_PRIVATE,
        )


@pytest.mark.parametrize(
    "mutation",
    ("fact_value", "statistics", "output_digest", "artifact_id", "artifact_payload"),
)
def test_real_data_tool_root_rejects_receipt_output_contract_forgery(
    tmp_path: Path, mutation: str
) -> None:
    root, bindings = _build_evidence_issuer_root(tmp_path / mutation)
    if mutation == "artifact_payload":
        tool_path = next((root / "tool-results").glob("*.json"))
        tool_raw = json.loads(tool_path.read_text(encoding="utf-8"))
        tool_raw["result"]["artifacts"][0]["payload"]["p_value"] = 0.5
        tool_path.write_text(json.dumps(tool_raw), encoding="utf-8")
    else:
        _synchronize_receipt_contract_forgery(root, mutation)
    with pytest.raises(ValueError, match="durable result|content addressed"):
        verify_e4a_evidence_root(
            root,
            issuer_bindings=bindings,
            evidence_signing_key=_ROOT_EVIDENCE_PRIVATE,
        )


def test_evidence_root_rejects_tampered_checker_report_and_tool(tmp_path: Path) -> None:
    root, bindings = _build_evidence_issuer_root(tmp_path)
    checker_path = root / "e4a-checker-result.json"
    checker = json.loads(checker_path.read_text(encoding="utf-8"))
    checker["scores"]["precision"] = 0.25
    checker_path.write_text(json.dumps(checker), encoding="utf-8")
    with pytest.raises(ValueError, match="checker artifact"):
        verify_e4a_evidence_root(
            root,
            issuer_bindings=bindings,
            evidence_signing_key=_ROOT_EVIDENCE_PRIVATE,
        )

    checker["scores"]["precision"] = 1.0
    checker_path.write_text(json.dumps(checker), encoding="utf-8")
    report = root / "report.md"
    original_report = report.read_text(encoding="utf-8")
    report.write_text(original_report + "forged\n", encoding="utf-8")
    with pytest.raises(ValueError, match="rerendering"):
        verify_e4a_evidence_root(
            root,
            issuer_bindings=bindings,
            evidence_signing_key=_ROOT_EVIDENCE_PRIVATE,
        )

    report.write_text(original_report, encoding="utf-8")
    tool_body = next((root / "tool-results").glob("*.json"))
    raw = json.loads(tool_body.read_text(encoding="utf-8"))
    raw["usage"]["rows_scanned"] += 1
    tool_body.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="result digest"):
        verify_e4a_evidence_root(
            root,
            issuer_bindings=bindings,
            evidence_signing_key=_ROOT_EVIDENCE_PRIVATE,
        )


def test_evidence_root_rejects_self_consistent_fabricated_bundle(tmp_path: Path) -> None:
    root, bindings = _build_evidence_issuer_root(tmp_path)
    store = JsonExplorationWorkflowStateStore(root / "workflow-state.json")
    workflow = store.load()
    bundle_id, bundle = next(iter(workflow.admitted_bundles.items()))
    original_claim = bundle.claims[0]
    fabricated_text = "Fabricated but internally self-consistent conclusion"
    fabricated_claim = original_claim.model_copy(
        update={
            "claim_id": original_claim.claim_id + "_forged",
            "claim_text": fabricated_text,
        }
    )
    fabricated_bundle = ClaimBundle.model_validate(
        {
            **bundle.model_dump(mode="json"),
            "claims": [
                fabricated_claim.model_dump(mode="json"),
                *[item.model_dump(mode="json") for item in bundle.claims[1:]],
            ],
        }
    )
    terminal = JsonlExplorationJournal(root / "journal.jsonl").rebuild()
    assert terminal is not None
    receipt = next(iter(workflow.committed_receipts.values()))
    assert receipt.statistics is not None
    family_id = receipt.statistics.statistical_family_id or receipt.statistics.hypothesis_id
    assert family_id is not None
    fabricated_report = run_claim_gates(
        fabricated_bundle,
        committed_receipts=workflow.committed_receipts,
        run_witness=terminal.data_state_witness,
        stat_attempt_counts={family_id: 1},
    )
    assert fabricated_report.passed
    workflow.admitted_bundles[bundle_id] = fabricated_bundle
    workflow.gate_reports[bundle_id] = fabricated_report
    store.remember(workflow)
    # Simulate a hostile root that also rewrites the unsigned recovery body and
    # journal projection. Canonical reducer replay must still reject the claims.
    # (Claim texts no longer appear in report.md, so no report patch is needed
    # to keep the hostile root self-consistent.)
    _rewrite_final_reduction_ledger(root)

    with pytest.raises(ValueError, match="canonical deterministic replay"):
        verify_e4a_evidence_root(
            root,
            issuer_bindings=bindings,
            evidence_signing_key=_ROOT_EVIDENCE_PRIVATE,
        )


def test_self_reported_trial_and_recomputed_digest_cannot_forge_attestation(
    tmp_path: Path,
) -> None:
    root, bindings = _build_evidence_issuer_root(tmp_path)
    trial, manifest = verify_e4a_evidence_root(
        root,
        issuer_bindings=bindings,
        evidence_signing_key=_ROOT_EVIDENCE_PRIVATE,
    )
    public_key = (
        Ed25519PrivateKey.from_private_bytes(_ROOT_EVIDENCE_PRIVATE).public_key().public_bytes_raw()
    )
    forged = trial.model_copy(
        update={
            "scores": {**trial.scores, "precision": 0.123},
            "source_manifest_digest": "f" * 64,
        }
    )
    assert manifest.root_digest != forged.source_manifest_digest
    assert not verify_e4a_trial_evidence(forged, public_key)
    with pytest.raises(RuntimeError, match="verified evidence-root"):
        attest_e4a_trial_evidence(
            E4aTrialEvidence.model_validate(forged.model_dump(mode="json")),
            signing_key=_ROOT_EVIDENCE_PRIVATE,
            key_id=bindings.certificate.evidence_key_id,
            source_manifest_digest="f" * 64,
        )
    with pytest.raises(RuntimeError, match="verified evidence roots"):
        issue_e4a_release_certificate(baseline=[trial], treatment=[forged])


def test_evidence_root_binds_physical_calls_scheduler_and_fixture_predicate(
    tmp_path: Path,
) -> None:
    root, bindings = _build_evidence_issuer_root(tmp_path)
    ledger_path = root / "llm-budget.jsonl"
    rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    usage = next(row for row in rows if row["event_type"] == "llm_usage")
    usage["summary"]["logical_call_id"] = "forged-logical-call"
    ledger_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="one-to-one"):
        verify_e4a_evidence_root(
            root,
            issuer_bindings=bindings,
            evidence_signing_key=_ROOT_EVIDENCE_PRIVATE,
        )

    root, bindings = _build_evidence_issuer_root(tmp_path / "scheduler")
    workflow_path = root / "workflow-state.json"
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    workflow["decisions"][0]["priority"] = 0.25
    workflow_path.write_text(json.dumps(workflow), encoding="utf-8")
    with pytest.raises(ValueError, match="frontier digest"):
        verify_e4a_evidence_root(
            root,
            issuer_bindings=bindings,
            evidence_signing_key=_ROOT_EVIDENCE_PRIVATE,
        )

    root, bindings = _build_evidence_issuer_root(tmp_path / "predicate")
    expected = bindings.fixture.expected_structures[0]
    wrong_fixture = bindings.fixture.model_copy(
        update={
            "expected_structures": (
                expected.model_copy(
                    update={
                        "predicate": expected.predicate.model_copy(update={"operator": "less_than"})
                    }
                ),
            )
        }
    )
    run_spec_path = root / "e4a-run-spec.json"
    run_spec = E4aEvidenceRunSpec.model_validate_json(
        run_spec_path.read_text(encoding="utf-8")
    ).model_copy(update={"ground_truth_digest": wrong_fixture.digest})
    run_spec_path.write_text(run_spec.model_dump_json(), encoding="utf-8")
    wrong_bindings = bindings.model_copy(update={"fixture": wrong_fixture})
    with pytest.raises(ValueError, match="checker artifact"):
        verify_e4a_evidence_root(
            root,
            issuer_bindings=wrong_bindings,
            evidence_signing_key=_ROOT_EVIDENCE_PRIVATE,
        )


def test_release_issuer_rejects_unpinned_baseline_and_tier_seed_relabel(
    tmp_path: Path,
) -> None:
    root, bindings = _build_evidence_issuer_root(tmp_path)
    _trial, manifest = verify_e4a_evidence_root(
        root,
        issuer_bindings=bindings,
        evidence_signing_key=_ROOT_EVIDENCE_PRIVATE,
    )
    caps = E4aHardCaps(
        max_wall_seconds=60,
        max_llm_requests=10,
        max_total_tokens=1_000,
        max_cost_usd=1,
        max_tool_calls=10,
        max_rows_scanned=1_000,
        max_cells_scanned=1_000,
    )
    unpinned = bindings.model_copy(
        update={
            "trial_plan": (
                E4aPlannedTrial(
                    role="baseline",
                    trial_id="xpl-evidence-root",
                    tier="quick",
                    seed=7,
                    manifest_digest="f" * 64,
                ),
                E4aPlannedTrial(
                    role="treatment",
                    trial_id="xpl-evidence-root",
                    tier="quick",
                    seed=7,
                    manifest_digest=manifest.root_digest,
                ),
            )
        }
    )
    with pytest.raises(ValueError, match="not pinned"):
        issue_e4a_release_from_evidence_roots(
            baseline_roots=(root,),
            treatment_roots=(root,),
            hard_caps=caps,
            issuer_bindings=unpinned,
            evidence_signing_key=_ROOT_EVIDENCE_PRIVATE,
            release_signing_key=bytes.fromhex("32" * 32),
            release_key_id="release-v1",
        )

    relabelled = unpinned.model_copy(
        update={
            "trial_plan": (
                unpinned.trial_plan[0].model_copy(update={"manifest_digest": manifest.root_digest}),
                unpinned.trial_plan[1].model_copy(update={"seed": 99}),
            )
        }
    )
    with pytest.raises(ValueError, match="tier/seed"):
        issue_e4a_release_from_evidence_roots(
            baseline_roots=(root,),
            treatment_roots=(root,),
            hard_caps=caps,
            issuer_bindings=relabelled,
            evidence_signing_key=_ROOT_EVIDENCE_PRIVATE,
            release_signing_key=bytes.fromhex("32" * 32),
            release_key_id="release-v1",
        )


def test_evidence_root_rejects_cherry_picked_ledger_bundle_and_coverage(
    tmp_path: Path,
) -> None:
    root, bindings = _build_evidence_issuer_root(tmp_path)
    workflow_path = root / "workflow-state.json"
    projection_path = root / "projection.json"
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    projection = json.loads(projection_path.read_text(encoding="utf-8"))
    workflow["insights"] = []
    workflow["coverage_completed"] = []
    projection["insight_records"] = []
    projection["coverage_completed"] = []
    workflow_path.write_text(json.dumps(workflow), encoding="utf-8")
    projection_path.write_text(json.dumps(projection), encoding="utf-8")
    with pytest.raises(ValueError, match="final reducer state"):
        verify_e4a_evidence_root(
            root,
            issuer_bindings=bindings,
            evidence_signing_key=_ROOT_EVIDENCE_PRIVATE,
        )

    root, bindings = _build_evidence_issuer_root(tmp_path / "bundle")
    workflow_path = root / "workflow-state.json"
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    workflow["admitted_bundles"] = []
    workflow_path.write_text(json.dumps(workflow), encoding="utf-8")
    with pytest.raises(ValueError, match="final reducer state"):
        verify_e4a_evidence_root(
            root,
            issuer_bindings=bindings,
            evidence_signing_key=_ROOT_EVIDENCE_PRIVATE,
        )

    root, bindings = _build_evidence_issuer_root(tmp_path / "coverage")
    workflow_path = root / "workflow-state.json"
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    workflow["coverage_completed"] = ["forged-coverage-key"]
    workflow_path.write_text(json.dumps(workflow), encoding="utf-8")
    with pytest.raises(ValueError, match="final reducer state"):
        verify_e4a_evidence_root(
            root,
            issuer_bindings=bindings,
            evidence_signing_key=_ROOT_EVIDENCE_PRIVATE,
        )

    root, bindings = _build_evidence_issuer_root(tmp_path / "gate")
    workflow_path = root / "workflow-state.json"
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    workflow["gate_reports"][0]["health_score"] = 0.5
    workflow_path.write_text(json.dumps(workflow), encoding="utf-8")
    with pytest.raises(ValueError, match="final reducer state"):
        verify_e4a_evidence_root(
            root,
            issuer_bindings=bindings,
            evidence_signing_key=_ROOT_EVIDENCE_PRIVATE,
        )


# --------------------------------------------------------------- E6 branching


class _BranchProvider(_Provider):
    """Scripted per-round proposals; every odd probe call runs run_stat_test."""

    def __init__(
        self, proposals_by_round: dict[int, tuple[HypothesisProposal, ...]]
    ) -> None:
        super().__init__()
        self.proposals_by_round = proposals_by_round
        self.probe_calls = 0

    def structured(
        self,
        *,
        task: str,
        schema: type[T],
        payload: dict[str, Any],
    ) -> T:
        assert task == "exploration_generate_hypotheses"
        self.structured_calls += 1
        self._record_usage()
        proposals = self.proposals_by_round.get(payload["round_index"], ())
        batch = (
            HypothesisProposalBatch(proposals=proposals)
            if proposals
            else HypothesisProposalBatch(
                concluded=True, conclusion_reason="No further hypotheses this round."
            )
        )
        return schema.model_validate(batch.model_dump(mode="json"))

    def tool_call(
        self,
        *,
        task: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMToolResponse:
        assert task == "exploration_probe_loop"
        self.tool_calls += 1
        self._record_usage()
        self.probe_calls += 1
        if self.probe_calls % 2 == 1:
            # Distinct arguments (probe dedup) but the same dataset/columns,
            # so both attempts land in one multiple-comparisons family.
            test_type = (
                "independent_t_test" if self.probe_calls == 1 else "mann_whitney_u"
            )
            return LLMToolResponse(
                tool_calls=[
                    LLMToolCall(
                        call_id=f"provider-branch-call-{self.probe_calls}",
                        name="run_stat_test",
                        arguments={
                            "dataset_id": "ds-1",
                            "test_type": test_type,
                            "group_column": "region",
                            "value_column": "revenue",
                        },
                    )
                ],
                finish_reason="tool_calls",
            )
        return LLMToolResponse(content="Probe complete.", finish_reason="stop")


def _branch_proposal(operator: str, probe_kind: str) -> HypothesisProposal:
    return HypothesisProposal(
        statement=f"Regional revenue {operator} probe.",
        rationale="Branch-mode driver test.",
        expected_evidence="A deterministic statistical observation.",
        falsification_conditions=("The statistical test finds nothing.",),
        family=InsightFamily.DIAGNOSTIC,
        method_family="compare_groups",
        dataset_ids=("ds-1",),
        columns=("region", "revenue"),
        probe_kind=probe_kind,
        predicate=HypothesisPredicate(
            metric="revenue",
            operator=operator,  # type: ignore[arg-type]
            left_operand="region",
        ),
    )


def test_branch_mode_runs_end_to_end_and_issuer_recomputes_constraints(
    tmp_path: Path,
) -> None:
    refuted_proposal = _branch_proposal("absent", "absence_probe")
    branch_proposal = _branch_proposal("differs", "difference_probe")
    dataset = LoadedDataset(
        record=DatasetRecord(
            dataset_id="ds-1",
            name="branch.csv",
            path=Path("/data/branch.csv"),
            content_hash="branch-hash-v1",
        ),
        frame=pd.DataFrame(
            {
                "region": ["east"] * 30 + ["west"] * 30,
                "revenue": [100.0 + index / 10 for index in range(30)]
                + [10.0 + index / 10 for index in range(30)],
            }
        ),
    )
    data_context = DataToolContext(
        datasets=[dataset],
        catalog=build_catalog([dataset]),
        project_id="shadow-project",
        session_id="xpl-branch-mode",
        store=None,
        payload_policy="schema+aggregates",
        stat_registry=StatTestRegistry(
            tmp_path / "exploration-eval" / "xpl-branch-mode" / "stat_registry.jsonl"
        ),
    )
    assert data_context.stat_registry is not None
    stat_registry = data_context.stat_registry
    tool = next(
        item for item in build_data_tools(data_context) if item.name == "run_stat_test"
    )
    run_witness = data_state_witness_digest([("ds-1", None, dataset.record.content_hash)])
    base_policy = build_exploration_policy(
        tier="standard",
        dataset_scope=("ds-1",),
        tool_capability_digest=exploration_tool_capability_digest((tool,)),
    )
    policy = sealed_policy(
        base_policy.model_copy(
            update={
                "budget": base_policy.budget.model_copy(
                    update={
                        "branching": ExplorationBranchPolicy(
                            trigger_stagnant_rounds=2, max_branches=1
                        )
                    }
                ),
                "policy_fingerprint": "",
            }
        )
    )
    refuted_seed = candidate_seed(refuted_proposal, sequence_index=1)
    branch_seed = candidate_seed(branch_proposal, sequence_index=1)

    def admission(_context: Any) -> AdmissionContext:
        return AdmissionContext(
            dataset_columns={"ds-1": frozenset({"region", "revenue"})},
            allowed_dataset_ids=frozenset({"ds-1"}),
            supported_method_families=frozenset({"compare_groups"}),
            historical_hypothesis_fingerprints=frozenset(),
            answered_hypothesis_fingerprints=frozenset(),
            executed_query_fingerprints=frozenset(),
            remaining_cost=1,
            family_quota_remaining={InsightFamily.DIAGNOSTIC: 1},
            unexplored_coverage_keys=frozenset(
                {refuted_seed.coverage_key, branch_seed.coverage_key}
            ),
        )

    def signals(_context: Any, seeds: tuple[Any, ...]) -> dict[str, CandidateSignals]:
        return {item.hypothesis_id: CandidateSignals(business_value=1) for item in seeds}

    def stat_counts() -> dict[str, int]:
        counts: dict[str, int] = {}
        for attempt in stat_registry.attempts():
            counts[attempt.family_id] = counts.get(attempt.family_id, 0) + 1
        return counts

    journal = JsonlExplorationJournal(
        tmp_path / "exploration-eval" / "xpl-branch-mode" / "journal.jsonl"
    )
    journal.initialize(
        exploration_id="xpl-branch-mode",
        policy=policy,
        code_fingerprint="code-branch-v1",
        data_state_witness=run_witness,
    )
    result = run_composed_shadow_exploration(
        workspace=tmp_path,
        exploration_id="xpl-branch-mode",
        policy=policy,
        code_fingerprint="code-branch-v1",
        data_state_witness=run_witness,
        provider=_BranchProvider(
            {0: (refuted_proposal,), 3: (branch_proposal,), 4: (refuted_proposal,)}
        ),
        tools=(tool,),
        dataset_profiles=(),
        scheduler_policy=_scheduler_policy(),
        admission_context=admission,
        signals=signals,
        witness=CallableWitnessPort(lambda expected: expected == run_witness),
        usage_meter=_UsageMeter("run_stat_test"),
        stat_attempt_counts=stat_counts,
    )
    assert result.result.error is None, result.result.error
    assert result.result.stop_reason == "no_new_information"

    events = JsonlExplorationJournal(result.journal_path).events()
    round_branches = [
        event.branch_id for event in events if isinstance(event, RoundStartedEvent)
    ]
    assert round_branches[:3] == [None, None, None]
    assert len(round_branches) > 3
    assert all(branch_id == "br_1" for branch_id in round_branches[3:])
    abandonments = [
        event for event in events if isinstance(event, BranchAbandonedEvent)
    ]
    assert len(abandonments) == 1
    assert abandonments[0].branch_id == "main"
    assert abandonments[0].round_index == 2
    constraint_keys = {
        (item.reason, item.hypothesis_fingerprint)
        for item in abandonments[0].constraints
    }
    assert ("refuted", refuted_seed.hypothesis_fingerprint) in constraint_keys

    # Cross-branch multiplicity: the same test family ran once on the main
    # line and once inside br_1; the shared registry must count both.
    attempts = list(stat_registry.attempts())
    assert len(attempts) == 2
    assert len({item.family_id for item in attempts}) == 1

    # The refuted hypothesis is blocked on re-proposal after abandonment: its
    # round runs the generate call only, never a probe conversation.
    llm_calls_by_round: dict[int, int] = {}
    open_round: int | None = None
    for event in events:
        if isinstance(event, RoundStartedEvent):
            open_round = event.round_index
            llm_calls_by_round[open_round] = 0
        elif isinstance(event, LlmCallStartedEvent) and open_round is not None:
            llm_calls_by_round[open_round] += 1
        elif isinstance(event, RoundSettledEvent):
            open_round = None
    assert llm_calls_by_round[0] > 1  # main-line probe ran
    assert llm_calls_by_round[3] > 1  # branch probe ran
    assert llm_calls_by_round[4] == 1  # re-proposed refuted hypothesis blocked
    workflow = JsonExplorationWorkflowStateStore(
        result.journal_path.parent / "workflow-state.json"
    ).load()

    # Issuer-side recomputation accepts the honest journal and rejects tampering.
    candidates_by_round = {
        0: {refuted_seed.hypothesis_id: refuted_seed},
        3: {
            candidate_seed(branch_proposal, sequence_index=30001).hypothesis_id: (
                candidate_seed(branch_proposal, sequence_index=30001)
            )
        },
        4: {
            candidate_seed(refuted_proposal, sequence_index=40001).hypothesis_id: (
                candidate_seed(refuted_proposal, sequence_index=40001)
            )
        },
    }
    _verify_branch_abandonments(
        events, workflow, candidates_by_round=candidates_by_round
    )
    tampered_events = [
        (
            event.model_copy(update={"constraints": ()})
            if isinstance(event, BranchAbandonedEvent)
            else event
        )
        for event in events
    ]
    with pytest.raises(ValueError, match="branch abandonment constraints"):
        _verify_branch_abandonments(
            tampered_events, workflow, candidates_by_round=candidates_by_round
        )


# --- 2026-08-03 evidence-semantics repairs (R1/R5/P1a/P1c) --------------------

_REDUCER_SEED = candidate_seed(_proposal(), sequence_index=1)


def _reducer_receipt(
    tool_call_id: str,
    *,
    outcome: str | None = None,
    result_count: int = 1,
    facts: tuple[ReceiptFact, ...] | None = None,
    scope: ReceiptScope | None = None,
    method_family: str = "profile_slice",
    warnings: tuple[str, ...] = (),
) -> EvidenceReceipt:
    statistics = None
    if outcome is not None:
        statistics = ReceiptStatistics(
            hypothesis_id=_REDUCER_SEED.hypothesis_id,
            hypothesis_outcome=outcome,  # type: ignore[arg-type]
            test_name="welch_anova",
            test_statistic=12.5,
            p_value=0.0001,
            effect_size=0.4,
            ci_low=0.1,
            ci_high=0.9,
            sample_size=1086,
            sequence_index=1,
        )
    return build_receipt(
        tool_call_id=tool_call_id,
        tool_name=method_family,
        tool_version="1",
        arguments={"call": tool_call_id},
        raw_output={"rows": []},
        artifact_ids=(),
        result_count=result_count,
        scope=scope
        or ReceiptScope(
            dataset_ids=("ds-1",),
            columns=("region", "revenue"),
            scope_resolution="explicit",
        ),
        facts=(
            facts
            if facts is not None
            else (
                ReceiptFact(
                    fact_id="rows_in_slice",
                    name="rows_in_slice",
                    value=362,
                    value_type="count",
                    support_type="direct",
                ),
            )
        ),
        method=ReceiptMethod(family=method_family, warnings=warnings),
        statistics=statistics,
        data_state_witness=WITNESS,
        created_at="2026-08-02T00:00:00Z",
    )


_EMPTY_SLICE_RECEIPT = _reducer_receipt(
    "call_empty_slice",
    result_count=0,
    facts=(
        ReceiptFact(
            fact_id="empty_slice",
            name="empty_slice",
            value=None,
            value_type="null",
            support_type="absence",
        ),
    ),
    scope=ReceiptScope(
        dataset_ids=("ds-1",),
        columns=("region", "revenue"),
        scope_resolution="explicit",
        filters="region = 'East'",
    ),
    warnings=("The WHERE condition matched no rows.",),
)


def _reducer_port(
    tmp_path: Path, receipts: tuple[EvidenceReceipt, ...]
) -> tuple[JsonlExplorationJournal, Any, Any]:
    from eda_platform.agents.exploration.workflow import (
        ClaimGateReducerPort,
        ExplorationWorkflowState,
    )

    journal = JsonlExplorationJournal(tmp_path / "exploration.journal.jsonl")
    journal.initialize(
        exploration_id="xpl_reducer",
        policy=build_exploration_policy(
            tier="quick",
            dataset_scope=("ds-1",),
            tool_capability_digest="tools-v1",
        ),
        code_fingerprint="code-v1",
        data_state_witness=WITNESS,
    )
    journal.append_new("round_started", round_index=0)
    state = ExplorationWorkflowState()
    for receipt in receipts:
        state.committed_receipts[receipt.receipt_id] = receipt
    return (
        journal,
        state,
        ClaimGateReducerPort(
            journal=journal,
            state=state,
            stat_attempt_counts=lambda: {_REDUCER_SEED.hypothesis_id: 1},
        ),
    )


def _reduce_once(
    port: Any, receipts: tuple[EvidenceReceipt, ...]
) -> ReductionOutcome:
    from eda_platform.agents.exploration.supervisor import (
        FrontierItem,
        ProbeSelection,
        ScoredFrontier,
        ValidationOutcome,
    )
    from eda_platform.agents.exploration.workflow import ExecutedProbeBatch

    items = (
        FrontierItem(
            hypothesis_id=_REDUCER_SEED.hypothesis_id,
            priority=1.0,
            payload=_REDUCER_SEED,
        ),
    )
    frontier = ScoredFrontier(items=items, digest="frontier_reducer")
    validated = ValidationOutcome(
        ExecutedProbeBatch(
            ProbeSelection(items),
            (),
            receipts,
            tuple(
                (receipt.receipt_id, _REDUCER_SEED.hypothesis_id)
                for receipt in receipts
            ),
        )
    )
    context = PhaseContext(
        exploration_id="xpl_reducer",
        round_index=0,
        phase=SupervisorPhase.REDUCE,
        data_state_witness=WITNESS,
        soft_countdown_context="",
        completed_step_ids=frozenset(),
    )
    return port.reduce(context, validated, frontier, logical_step_id="step_reduce")


def test_unadjudicated_receipts_no_longer_veto_a_decisive_one(tmp_path: Path) -> None:
    """R1: reproduces the 2026-08-03 empty-frontier run — one typed `supports`
    receipt alongside two untyped profile slices for the same hypothesis."""
    decisive = _reducer_receipt(
        "call_missingness", outcome="supports", method_family="missingness_diagnostic"
    )
    silent_a = _reducer_receipt("call_slice_a")
    silent_b = _reducer_receipt("call_slice_b")
    receipts = (decisive, silent_a, silent_b)
    _journal, state, port = _reducer_port(tmp_path, receipts)

    outcome = _reduce_once(port, receipts)

    assert outcome.transitions == ("new",)
    insight = next(iter(state.insights.values()))
    assert insight.status == "new"
    assert insight.trust_level == "supported"
    assert insight.supporting_receipt_ids == (decisive.receipt_id,)
    assert insight.contradicting_receipt_ids == ()
    # The untyped slices stay in the gated bundle but never become evidence.
    bundle = state.admitted_bundles[insight.claim_bundle_id]
    assert set(bundle.referenced_receipt_ids()) == {
        receipt.receipt_id for receipt in receipts
    }


def test_an_absence_fact_carries_the_scanned_scope_into_its_claim(
    tmp_path: Path,
) -> None:
    """R5: an empty slice is a legitimate absence claim; the gate needs the
    scope the receipt actually scanned, which only the reducer can supply."""
    decisive = _reducer_receipt(
        "call_welch", outcome="supports", method_family="welch_anova"
    )
    receipts = (_EMPTY_SLICE_RECEIPT, decisive)
    _journal, state, port = _reducer_port(tmp_path, receipts)

    outcome = _reduce_once(port, receipts)

    report = next(iter(state.gate_reports.values()))
    assert report.passed, [
        violation.code
        for verdict in report.verdicts
        for violation in verdict.violations
    ]
    assert outcome.transitions == ("new",)
    bundle = next(iter(state.admitted_bundles.values()))
    absence = next(claim for claim in bundle.claims if claim.claim_type == "absence")
    assert absence.scope is not None
    assert absence.scope.dataset_ids == ("ds-1",)
    assert absence.scope.columns == ("region", "revenue")
    assert absence.scope.filters == "region = 'East'"


def test_a_gate_passing_round_reports_its_admitted_bundle_count(
    tmp_path: Path,
) -> None:
    """P1a: bundle admission is progress even when no insight moves."""
    decisive = _reducer_receipt(
        "call_welch_admitted", outcome="supports", method_family="welch_anova"
    )
    receipts = (decisive,)
    _journal, _state, port = _reducer_port(tmp_path, receipts)

    outcome = _reduce_once(port, receipts)

    assert outcome.admitted_bundle_count == 1


def test_replaying_a_reduction_does_not_append_a_second_gate_verdict(
    tmp_path: Path,
) -> None:
    """P1c: a crash between the gate verdict and the reduction commit replays
    the whole reduce phase; the journal must stay one verdict per bundle."""
    decisive = _reducer_receipt(
        "call_welch_replay", outcome="supports", method_family="welch_anova"
    )
    receipts = (decisive,)
    journal, _state, port = _reducer_port(tmp_path, receipts)

    _reduce_once(port, receipts)
    _reduce_once(port, receipts)

    verdicts = [
        event for event in journal.events() if event.event_type == "gate_verdict"
    ]
    assert len(verdicts) == 1


def _bound_receipt(
    base: EvidenceReceipt, *, exploration_id: str, round_index: int, step_id: str
) -> EvidenceReceipt:
    payload = base.model_dump()
    payload["execution"] = ReceiptExecution(
        run_id=(
            f"{exploration_id}:round:{round_index}:hypothesis:"
            f"{_REDUCER_SEED.hypothesis_id}:execute_probes"
        ),
        provider_call_id=f"call_{step_id}",
        logical_step_id=step_id,
    ).model_dump()
    payload.pop("content_digest")
    payload["content_digest"] = receipt_content_digest(payload)
    return EvidenceReceipt.model_validate(payload)


def test_the_issuer_mirrors_the_reducers_new_side_classification(
    tmp_path: Path,
) -> None:
    """The issuer rebuilds bundles from scratch; if its side classification or
    its absence scope drifts from the reducer's, every root is rejected. The
    scripted trials never produce untyped or absence receipts, so the mirror
    is pinned here instead."""
    from eda_platform.agents.exploration.workflow import (
        ExplorationWorkflowState,
        _claim_bundle,
    )
    from eda_platform.drivers.exploration_evidence_issuer import (
        _canonical_claim_bundle,
        _rebuild_canonical_bundles,
    )

    exploration_id = "xpl_mirror"
    typed_first = _bound_receipt(
        _reducer_receipt("call_typed_1", outcome="supports"),
        exploration_id=exploration_id,
        round_index=0,
        step_id="step_1",
    )
    untyped = _bound_receipt(
        _EMPTY_SLICE_RECEIPT,
        exploration_id=exploration_id,
        round_index=0,
        step_id="step_2",
    )
    typed_second = _bound_receipt(
        _reducer_receipt("call_typed_2", outcome="supports"),
        exploration_id=exploration_id,
        round_index=1,
        step_id="step_3",
    )
    # Body-level mirror: the absence claim's derived scope must match verbatim.
    assert _claim_bundle(
        _REDUCER_SEED, (typed_first, untyped)
    ) == _canonical_claim_bundle(_REDUCER_SEED, (typed_first, untyped))

    journal = JsonlExplorationJournal(tmp_path / "exploration.journal.jsonl")
    journal.initialize(
        exploration_id=exploration_id,
        policy=build_exploration_policy(
            tier="quick",
            dataset_scope=("ds-1",),
            tool_capability_digest="tools-v1",
        ),
        code_fingerprint="code-v1",
        data_state_witness=WITNESS,
    )
    workflow = ExplorationWorkflowState()
    rounds = ((0, (typed_first, untyped)), (1, (typed_second,)))
    expected: dict[str, ClaimBundle] = {}
    # Round 1 cumulates only what round 0 adjudicated: the untyped receipt must
    # drop out of the prior sides, or the round-1 bundle id changes.
    cumulative: list[EvidenceReceipt] = []
    for round_index, receipts in rounds:
        journal.append_new("round_started", round_index=round_index)
        for receipt in receipts:
            step = receipt.execution.logical_step_id  # type: ignore[union-attr]
            workflow.committed_receipts[receipt.receipt_id] = receipt
            journal.append_new(
                "tool_call_started", logical_step_id=step, input_fingerprint=f"fp-{step}"
            )
            journal.append_new(
                "receipt_prepared", logical_step_id=step, receipt_id=receipt.receipt_id
            )
            journal.append_new(
                "receipt_committed", logical_step_id=step, receipt_id=receipt.receipt_id
            )
        cumulative.extend(
            receipt
            for receipt in receipts
            if round_index == 0
            or (
                receipt.statistics is not None
                and receipt.statistics.hypothesis_outcome is not None
            )
        )
        if round_index == 1:
            cumulative = [
                receipt
                for receipt in cumulative
                if receipt.statistics is not None
                and receipt.statistics.hypothesis_outcome is not None
            ]
        bundle = _claim_bundle(_REDUCER_SEED, tuple(cumulative))
        expected[bundle.claim_bundle_id] = bundle
        journal.append_new(
            "gate_verdict", claim_bundle_id=bundle.claim_bundle_id, verdict="passed"
        )
        journal.append_new("round_settled", round_index=round_index, progress=True)

    candidates_by_round = {
        0: {_REDUCER_SEED.hypothesis_id: _REDUCER_SEED},
        1: {_REDUCER_SEED.hypothesis_id: _REDUCER_SEED},
    }
    assert (
        _rebuild_canonical_bundles(
            journal.events(), workflow, candidates_by_round=candidates_by_round
        )
        == expected
    )


def _round_context(round_index: int) -> PhaseContext:
    return PhaseContext(
        exploration_id="xpl-generate",
        round_index=round_index,
        phase=SupervisorPhase.ADMIT_AND_SCORE,
        data_state_witness=WITNESS,
        soft_countdown_context="remaining={}",
        completed_step_ids=frozenset(),
    )


def _repeat_admission_context(coverage_key: str) -> AdmissionContext:
    return AdmissionContext(
        dataset_columns={"ds-1": frozenset({"region", "revenue"})},
        allowed_dataset_ids=frozenset({"ds-1"}),
        supported_method_families=frozenset({"profile_slice"}),
        historical_hypothesis_fingerprints=frozenset(),
        answered_hypothesis_fingerprints=frozenset(),
        executed_query_fingerprints=frozenset(),
        remaining_cost=1.0,
        family_quota_remaining={InsightFamily.DIAGNOSTIC: 1},
        unexplored_coverage_keys=frozenset({coverage_key}),
    )


def test_a_repeated_hypothesis_keeps_its_own_per_round_decision() -> None:
    first_seed = candidate_seed(_proposal(), sequence_index=1, mandatory=True)
    replayed = candidate_seed(_proposal(), sequence_index=10_001, mandatory=True)
    state = ExplorationWorkflowState()
    port = DeterministicSchedulerPort(
        policy=_scheduler_policy(),
        admission_context=lambda _context: _repeat_admission_context(
            first_seed.coverage_key
        ),
        signals=lambda _context, seeds: {
            seed.hypothesis_id: CandidateSignals(business_value=1.0) for seed in seeds
        },
        state=state,
    )

    first = port.admit_and_score(_round_context(0), CandidateBatch((first_seed,)))
    second = port.admit_and_score(_round_context(1), CandidateBatch((replayed,)))

    # Both rounds scored the same hypothesis id, so both rounds own a decision;
    # each round's frontier digest must replay from its positional slice.
    assert len(state.decisions) == 2
    assert first.digest == "frontier_" + scheduling_decision_digest(state.decisions[0:1])
    assert second.digest == "frontier_" + scheduling_decision_digest(state.decisions[1:2])


def test_generate_payload_carries_the_dataset_schema_and_method_vocabulary(
    tmp_path: Path,
) -> None:
    class _CapturingProvider(_Provider):
        def __init__(self) -> None:
            super().__init__()
            self.payloads: list[dict[str, Any]] = []

        def structured(self, *, task: str, schema: type[T], payload: dict[str, Any]) -> T:
            self.payloads.append(payload)
            return super().structured(task=task, schema=schema, payload=payload)

    journal = JsonlExplorationJournal(tmp_path / "journal.jsonl")
    journal.initialize(
        exploration_id="xpl-generate",
        policy=build_exploration_policy(
            tier="quick",
            dataset_scope=("ds-1",),
            tool_capability_digest="tools-v1",
        ),
        code_fingerprint="code-v1",
        data_state_witness=WITNESS,
    )
    journal.claim_recovery()
    provider = _CapturingProvider()
    generator = JournaledCandidateGenerator(
        provider=provider,
        journal=journal,
        recovery=JsonSupervisorRecoveryStore(tmp_path / "responses"),
        dataset_profiles=(),
        dataset_columns={"ds-1": ("revenue", "region")},
        supported_method_families=("profile_slice", "compare_groups"),
    )

    generator.generate(
        _generate_context(), logical_step_id="xpl-generate:round:0:generate"
    )

    payload = provider.payloads[0]
    assert payload["datasets"] == [
        {"dataset_id": "ds-1", "columns": ["revenue", "region"]}
    ]
    assert payload["method_families"] == ["compare_groups", "profile_slice"]
    assert "exact" in payload["instruction"]


def test_the_issuer_reads_a_replayed_mandatory_probe_as_one_candidate() -> None:
    first = candidate_seed(_proposal(), sequence_index=1, mandatory=True)
    replayed = replace(
        candidate_seed(_proposal(), sequence_index=10_001, mandatory=True),
        status="admitted",
        priority=0.9,
    )

    assert first != replayed
    assert _candidate_identity(first) == _candidate_identity(replayed)


def test_the_issuer_reads_a_reworded_proposal_as_the_same_hypothesis() -> None:
    """luna seed 8: the model restated one hypothesis in round 6 with different
    prose. hypothesis_id covers only execution-relevant semantic fields, so both
    rounds carried the SAME id — but the issuer compared whole proposals and
    called one id with two bodies a forgery. Identity must be defined once."""
    original = candidate_seed(_proposal(), sequence_index=1)
    reworded_proposal = _proposal().model_copy(
        update={
            "statement": "Put a different way, does revenue differ by region?",
            "rationale": "Reworded in a later round.",
            "expected_evidence": "The same comparison, described differently.",
            "falsification_conditions": ("Worded differently too.",),
        }
    )
    reworded = candidate_seed(reworded_proposal, sequence_index=42)

    # The system already treats them as one hypothesis.
    assert reworded.hypothesis_id == original.hypothesis_id
    assert reworded.hypothesis_fingerprint == original.hypothesis_fingerprint
    assert _candidate_identity(original) == _candidate_identity(reworded)


def test_a_different_predicate_is_still_a_different_hypothesis() -> None:
    """Control: identity may not collapse to the id alone."""
    original = candidate_seed(_proposal(), sequence_index=1)
    other = candidate_seed(
        _proposal().model_copy(
            update={
                "predicate": HypothesisPredicate(
                    metric="units", operator="differs", left_operand="region"
                )
            }
        ),
        sequence_index=1,
    )
    assert _candidate_identity(original) != _candidate_identity(other)


def test_every_round_replays_the_mandatory_probes_coverage_still_misses(
    tmp_path: Path,
) -> None:
    profiles = (
        DatasetExplorationProfile(
            dataset_id="ds-1",
            region_dimensions=("region",),
            metric_columns=("revenue",),
            missing_value_columns=("satisfaction",),
            missingness_group_dimensions=("channel",),
            datetime_columns=("order_date",),
            spike_metric_columns=("revenue",),
        ),
    )
    all_seeds = mandatory_probe_seeds(profiles)
    assert len(all_seeds) == 3
    explored: set[str] = set()

    journal = JsonlExplorationJournal(tmp_path / "journal.jsonl")
    journal.initialize(
        exploration_id="xpl-generate",
        policy=build_exploration_policy(
            tier="quick",
            dataset_scope=("ds-1",),
            tool_capability_digest="tools-v1",
        ),
        code_fingerprint="code-v1",
        data_state_witness=WITNESS,
    )
    journal.claim_recovery()
    class _LaterRoundProvider(_Provider):
        def structured(self, *, task: str, schema: type[T], payload: dict[str, Any]) -> T:
            del task, payload
            self.structured_calls += 1
            self._record_usage()
            return schema.model_validate(
                {"concluded": True, "conclusion_reason": "Nothing further to add."}
            )

    generator = JournaledCandidateGenerator(
        provider=_LaterRoundProvider(),
        journal=journal,
        recovery=JsonSupervisorRecoveryStore(tmp_path / "responses"),
        dataset_profiles=profiles,
        coverage_completed=lambda: frozenset(explored),
    )

    explored.add(all_seeds[0].coverage_key)
    batch = generator.generate(
        replace(_generate_context(), round_index=2),
        logical_step_id="xpl-generate:round:2:generate",
    )

    replayed = [
        item
        for item in batch.candidates
        if isinstance(item, CandidateSeed) and item.mandatory
    ]
    assert {item.coverage_key for item in replayed} == {
        seed.coverage_key for seed in all_seeds[1:]
    }


def test_a_probe_that_uses_up_its_step_budget_is_not_an_invariant_violation(
    tmp_path: Path,
) -> None:
    """`limit_reached` means the probe conversation ran out of its own steps,
    not that the control plane is broken. Its committed receipts are valid and
    the round must go on; treating it as fatal killed a real deepseek trial in
    round 0 after 30 committed receipts (2026-08-03, seed 3)."""

    class _LimitReachedExecutor:
        def run(self, **_kwargs: Any) -> Any:
            return ProbeExecutionResult(status="limit_reached", answer="out of steps")

    journal = JsonlExplorationJournal(tmp_path / "xpl" / "journal.jsonl")
    journal.initialize(
        exploration_id="xpl-limit",
        policy=build_exploration_policy(
            tier="standard",
            dataset_scope=("ds-1",),
            tool_capability_digest="tools-v1",
        ),
        code_fingerprint="code-v1",
        data_state_witness=WITNESS,
    )
    port = SupervisorProbeExecutorPort(
        executor=cast(Any, _LimitReachedExecutor()),
        journal=journal,
        state=ExplorationWorkflowState(),
        receipt_decoder=artifact_receipt_decoder,
    )
    seed = candidate_seed(_proposal(), sequence_index=1)
    context = PhaseContext(
        exploration_id="xpl-limit",
        round_index=0,
        phase=SupervisorPhase.EXECUTE_PROBES,
        data_state_witness=WITNESS,
        soft_countdown_context="[exploration_soft_countdown] remaining={}. Prefer x",
        completed_step_ids=frozenset(),
    )
    outcome = port.execute(
        context, ProbeSelection((FrontierItem("h", 1.0, payload=seed),))
    )
    batch = outcome.payload
    assert isinstance(batch, ExecutedProbeBatch)
    assert batch.receipts == ()


def test_concurrent_probe_sessions_overlap_and_keep_result_order(
    tmp_path: Path,
) -> None:
    """Speedup plan P4: probe_concurrency > 1 runs a round's sessions in worker
    threads. Sessions must genuinely overlap, and executions must come back in
    selection order regardless of completion order."""
    import threading
    import time as _time

    class _SlowExecutor:
        def __init__(self) -> None:
            self.guard = threading.Lock()
            self.active = 0
            self.max_active = 0

        def run(self, **kwargs: Any) -> Any:
            with self.guard:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            # The first-submitted session is the slowest: order must still hold.
            run_id = kwargs["run_id"]
            _time.sleep(0.2 if "hyp-a" in run_id else 0.05)
            with self.guard:
                self.active -= 1
            return ProbeExecutionResult(status="completed", answer=run_id)

    journal = JsonlExplorationJournal(tmp_path / "xpl" / "journal.jsonl")
    journal.initialize(
        exploration_id="xpl-conc",
        policy=build_exploration_policy(
            tier="standard",
            dataset_scope=("ds-1",),
            tool_capability_digest="tools-v1",
        ),
        code_fingerprint="code-v1",
        data_state_witness=WITNESS,
    )
    slow = _SlowExecutor()
    port = SupervisorProbeExecutorPort(
        executor=cast(Any, slow),
        journal=journal,
        state=ExplorationWorkflowState(),
        receipt_decoder=artifact_receipt_decoder,
        probe_concurrency=3,
    )
    seeds = [
        candidate_seed(
            _proposal_with_statement(f"Question {name}?"), sequence_index=index + 1
        )
        for index, name in enumerate(("hyp-a", "hyp-b", "hyp-c"))
    ]
    items = tuple(
        FrontierItem(f"hyp-{name}", 1.0, payload=seed)
        for name, seed in zip(("a", "b", "c"), seeds, strict=True)
    )
    context = PhaseContext(
        exploration_id="xpl-conc",
        round_index=0,
        phase=SupervisorPhase.EXECUTE_PROBES,
        data_state_witness=WITNESS,
        soft_countdown_context="",
        completed_step_ids=frozenset(),
    )

    started = _time.perf_counter()
    outcome = port.execute(context, ProbeSelection(items))
    elapsed = _time.perf_counter() - started

    batch = outcome.payload
    assert isinstance(batch, ExecutedProbeBatch)
    assert slow.max_active >= 2, "probe sessions never overlapped"
    assert elapsed < 0.3, f"sessions ran serially: {elapsed:.3f}s"
    answers = [execution.answer for execution in batch.executions]
    assert answers == sorted(answers, key=lambda a: str(a))  # selection order
    assert [seed.hypothesis_id in str(a) for seed, a in zip(seeds, answers, strict=True)]


def _proposal_with_statement(statement: str) -> HypothesisProposal:
    proposal = _proposal()
    return proposal.model_copy(update={"statement": statement})


def test_a_failed_probe_is_still_an_invariant_violation(tmp_path: Path) -> None:
    """Control: `failed` covers integrity faults (recovered-digest mismatch and
    friends), so it must keep ending the run."""

    class _FailedExecutor:
        def run(self, **_kwargs: Any) -> Any:
            return ProbeExecutionResult(status="failed", error="digest mismatch")

    journal = JsonlExplorationJournal(tmp_path / "xpl" / "journal.jsonl")
    journal.initialize(
        exploration_id="xpl-failed",
        policy=build_exploration_policy(
            tier="standard",
            dataset_scope=("ds-1",),
            tool_capability_digest="tools-v1",
        ),
        code_fingerprint="code-v1",
        data_state_witness=WITNESS,
    )
    port = SupervisorProbeExecutorPort(
        executor=cast(Any, _FailedExecutor()),
        journal=journal,
        state=ExplorationWorkflowState(),
        receipt_decoder=artifact_receipt_decoder,
    )
    seed = candidate_seed(_proposal(), sequence_index=1)
    context = PhaseContext(
        exploration_id="xpl-failed",
        round_index=0,
        phase=SupervisorPhase.EXECUTE_PROBES,
        data_state_witness=WITNESS,
        soft_countdown_context="[exploration_soft_countdown] remaining={}. Prefer x",
        completed_step_ids=frozenset(),
    )
    with pytest.raises(SupervisorInvariantError, match="failed"):
        port.execute(context, ProbeSelection((FrontierItem("h", 1.0, payload=seed),)))


@pytest.mark.parametrize(
    "error_code",
    ["finish_reason_length", "content_filtered", "empty_response"],
)
def test_a_probe_that_the_model_cannot_finish_does_not_kill_the_round(
    tmp_path: Path,
    error_code: str,
) -> None:
    """Truncation, a content filter and an empty turn are all model/provider
    behaviour on one probe. The control plane is intact and the receipts this
    probe already committed stay valid, so the round must go on."""

    class _ProbeLocalFailureExecutor:
        def run(self, **_kwargs: Any) -> Any:
            return ProbeExecutionResult(
                status="failed",
                error="the model could not finish this probe",
                error_code=error_code,
            )

    journal = JsonlExplorationJournal(tmp_path / "xpl" / "journal.jsonl")
    journal.initialize(
        exploration_id="xpl-local",
        policy=build_exploration_policy(
            tier="standard",
            dataset_scope=("ds-1",),
            tool_capability_digest="tools-v1",
        ),
        code_fingerprint="code-v1",
        data_state_witness=WITNESS,
    )
    port = SupervisorProbeExecutorPort(
        executor=cast(Any, _ProbeLocalFailureExecutor()),
        journal=journal,
        state=ExplorationWorkflowState(),
        receipt_decoder=artifact_receipt_decoder,
    )
    seed = candidate_seed(_proposal(), sequence_index=1)
    context = PhaseContext(
        exploration_id="xpl-local",
        round_index=0,
        phase=SupervisorPhase.EXECUTE_PROBES,
        data_state_witness=WITNESS,
        soft_countdown_context="[exploration_soft_countdown] remaining={}. Prefer x",
        completed_step_ids=frozenset(),
    )

    outcome = port.execute(
        context, ProbeSelection((FrontierItem("h", 1.0, payload=seed),))
    )

    batch = outcome.payload
    assert isinstance(batch, ExecutedProbeBatch)
    assert batch.executions[0].error_code == error_code


def test_an_integrity_failure_is_still_an_invariant_violation(tmp_path: Path) -> None:
    """Control: recovery-time integrity checks say the control plane cannot be
    trusted, so they must keep ending the run."""

    class _IntegrityFailureExecutor:
        def run(self, **_kwargs: Any) -> Any:
            return ProbeExecutionResult(
                status="failed",
                error="completed LLM response digest does not match the journal.",
                error_code="completed_response_digest_mismatch",
            )

    journal = JsonlExplorationJournal(tmp_path / "xpl" / "journal.jsonl")
    journal.initialize(
        exploration_id="xpl-integrity",
        policy=build_exploration_policy(
            tier="standard",
            dataset_scope=("ds-1",),
            tool_capability_digest="tools-v1",
        ),
        code_fingerprint="code-v1",
        data_state_witness=WITNESS,
    )
    port = SupervisorProbeExecutorPort(
        executor=cast(Any, _IntegrityFailureExecutor()),
        journal=journal,
        state=ExplorationWorkflowState(),
        receipt_decoder=artifact_receipt_decoder,
    )
    seed = candidate_seed(_proposal(), sequence_index=1)
    context = PhaseContext(
        exploration_id="xpl-integrity",
        round_index=0,
        phase=SupervisorPhase.EXECUTE_PROBES,
        data_state_witness=WITNESS,
        soft_countdown_context="[exploration_soft_countdown] remaining={}. Prefer x",
        completed_step_ids=frozenset(),
    )

    with pytest.raises(SupervisorInvariantError, match="digest"):
        port.execute(context, ProbeSelection((FrontierItem("h", 1.0, payload=seed),)))


def test_cancellation_inside_a_probe_reaches_the_supervisor_as_cancellation(
    tmp_path: Path,
) -> None:
    """A cancelled probe used to escape as a generic exception and settle the
    run as `failed`, which reads as a platform defect in the trace."""

    class _CancellingExecutor:
        def run(self, **_kwargs: Any) -> Any:
            raise SessionCancelled("cancelled by user")

    journal = JsonlExplorationJournal(tmp_path / "xpl" / "journal.jsonl")
    journal.initialize(
        exploration_id="xpl-cancel",
        policy=build_exploration_policy(
            tier="standard",
            dataset_scope=("ds-1",),
            tool_capability_digest="tools-v1",
        ),
        code_fingerprint="code-v1",
        data_state_witness=WITNESS,
    )
    port = SupervisorProbeExecutorPort(
        executor=cast(Any, _CancellingExecutor()),
        journal=journal,
        state=ExplorationWorkflowState(),
        receipt_decoder=artifact_receipt_decoder,
    )
    seed = candidate_seed(_proposal(), sequence_index=1)
    context = PhaseContext(
        exploration_id="xpl-cancel",
        round_index=0,
        phase=SupervisorPhase.EXECUTE_PROBES,
        data_state_witness=WITNESS,
        soft_countdown_context="[exploration_soft_countdown] remaining={}. Prefer x",
        completed_step_ids=frozenset(),
    )

    with pytest.raises(SupervisorCancelled):
        port.execute(context, ProbeSelection((FrontierItem("h", 1.0, payload=seed),)))
