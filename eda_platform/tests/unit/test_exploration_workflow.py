"""Production-shaped E4a composition across real scheduler/executor/gates/reducer."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import BaseModel, ConfigDict

from eda_platform.agents.data_tools import DataToolContext, build_data_tools
from eda_platform.agents.exploration.candidates import candidate_seed
from eda_platform.agents.exploration.executor import durable_tool_result_digest
from eda_platform.agents.exploration.scheduler import (
    AdmissionContext,
    CandidateSignals,
    PriorityWeights,
    SchedulerPolicy,
)
from eda_platform.agents.exploration.supervisor import (
    PhaseContext,
    ReductionOutcome,
    SupervisorBudgetExhausted,
    SupervisorInvariantError,
    SupervisorPhase,
    reduction_outcome_digest,
)
from eda_platform.agents.exploration.workflow import (
    JournaledCandidateGenerator,
    final_reduction_state_digest,
)
from eda_platform.agents.receipts import build_receipt
from eda_platform.agents.runtime import AgentTool, AgentToolResult
from eda_platform.agents.tool_context import current_execution_context
from eda_platform.core.budget import BudgetExceeded
from eda_platform.core.claim_gates import run_claim_gates
from eda_platform.core.exploration_budget import ToolCallProjection
from eda_platform.core.exploration_journal import JsonlExplorationJournal
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
    issue_e4a_release_from_evidence_roots,
    verify_e4a_evidence_root,
)
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.claims import ClaimBundle
from eda_platform.schemas.datasets import DatasetRecord
from eda_platform.schemas.exploration import (
    InsightFamily,
    LlmCallStartedEvent,
    RoundSettledEvent,
)
from eda_platform.schemas.exploration_budget import BudgetCapIncrease
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
        assert messages and [tool["name"] for tool in tools] == ["inspect_data_catalog"]
        self.tool_calls += 1
        self._record_usage()
        if self.tool_calls == 1:
            return LLMToolResponse(
                tool_calls=[
                    LLMToolCall(
                        call_id="provider-call-1",
                        name="inspect_data_catalog",
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
    def __init__(self, kind: str = "inspect_data_catalog") -> None:
        self.kind = kind

    def project(self, **_kwargs: Any) -> ToolCallProjection:
        return ToolCallProjection(kind=self.kind, rows_scanned=60, result_cells=20)

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
    ) -> None:
        super().__init__()
        self.proposal = proposal
        self.tool_name = tool_name
        self.arguments = arguments

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
        assert messages and [tool["name"] for tool in tools] == [self.tool_name]
        self.tool_calls += 1
        self._record_usage()
        if self.tool_calls == 1:
            return LLMToolResponse(
                tool_calls=[
                    LLMToolCall(
                        call_id="provider-data-call-1",
                        name=self.tool_name,
                        arguments=self.arguments,
                    )
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
        method_family="inspect_data_catalog",
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
            tool_name="inspect_data_catalog",
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
        name="inspect_data_catalog",
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
            supported_method_families=frozenset({"inspect_data_catalog"}),
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
    assert "regional difference" in report
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
    tool = next(item for item in build_data_tools(data_context) if item.name == "run_stat_test")
    run_witness = data_state_witness_digest(
        [("ds-1", None, dataset.record.content_hash)]
    )
    policy = build_exploration_policy(
        tier="quick",
        dataset_scope=("ds-1",),
        tool_capability_digest=exploration_tool_capability_digest((tool,)),
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
        ),
        tools=(tool,),
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
        },
    )
    (root / "e4a-checker-result.json").write_text(
        checker.model_dump_json(indent=2), encoding="utf-8"
    )
    return root, issuer_bindings


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
    original_claim_text: str | None = None
    fabricated_claim_text: str | None = None
    forged_primary_id: str | None = None
    if mutation == "fact_value":
        fact = receipt_raw["facts"][0]
        fact["name"] = "fabricated_revenue"
        fact["value"] = 999999
        original_claim_text = "p_value: " + str(
            json.loads(workflow_path.read_text(encoding="utf-8"))["committed_receipts"][0][
                "facts"
            ][0]["value"]
        )
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

    if original_claim_text is not None and fabricated_claim_text is not None:
        report_path = root / "report.md"
        report = report_path.read_text(encoding="utf-8")
        assert original_claim_text in report
        report_path.write_text(
            report.replace(original_claim_text, fabricated_claim_text), encoding="utf-8"
        )


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
    _rewrite_final_reduction_ledger(root)

    report_path = root / "report.md"
    report_text = report_path.read_text(encoding="utf-8")
    assert original_claim.claim_text in report_text
    report_path.write_text(
        report_text.replace(original_claim.claim_text, fabricated_text),
        encoding="utf-8",
    )

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
