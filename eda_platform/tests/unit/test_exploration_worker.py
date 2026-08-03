"""E4b worker adapter and trusted resource-meter contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from exploration_test_helpers import (
    TEST_EVIDENCE_KEY_ID,
    release_certificate,
    runtime_identity,
)
from pydantic import BaseModel

from eda_platform.agents.exploration.candidates import candidate_seed
from eda_platform.agents.exploration.scheduler import (
    PriorityFeatures,
    SchedulingDecision,
)
from eda_platform.agents.exploration.supervisor import (
    CandidateBatch,
    SupervisorPhase,
    phase_step_id,
)
from eda_platform.agents.exploration.workflow import ExplorationWorkflowState
from eda_platform.agents.receipts import build_receipt
from eda_platform.agents.runtime import AgentTool, AgentToolResult
from eda_platform.application.services.exploration_service import (
    ExplorationRunMetadata,
    ExplorationSourceSnapshot,
)
from eda_platform.core.exploration_journal import JsonlExplorationJournal
from eda_platform.core.exploration_profiles import build_exploration_policy
from eda_platform.core.exploration_release_gate import E4aEvidenceBindings
from eda_platform.core.exploration_shadow_store import shadow_run_root
from eda_platform.core.llm import LLMSettings, LLMToolCall
from eda_platform.core.provider_registry import LLMProvider
from eda_platform.core.stat_registry import StatTestRegistry
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.exploration import (
    JsonExplorationWorkflowStateStore,
    JsonSupervisorRecoveryStore,
    exploration_tool_capability_digest,
)
from eda_platform.schemas.artifacts import Artifact, ArtifactType, DatasetProfile
from eda_platform.schemas.datasets import DatasetRecord
from eda_platform.schemas.exploration import InsightFamily
from eda_platform.schemas.hypotheses import HypothesisPredicate, HypothesisProposal
from eda_platform.schemas.insights import InsightProof, InsightRecord
from eda_platform.schemas.receipts import ReceiptMethod, ReceiptScope
from eda_platform.tools.loader import LoadedDataset
from eda_platform.worker import exploration as worker


class _ToolArgs(BaseModel):
    dataset_id: str


def _tool() -> AgentTool:
    return AgentTool(
        name="inspect_data_catalog",
        description="Read the approved catalog.",
        args_schema=_ToolArgs,
        execute=lambda _args: AgentToolResult(content={"ok": True}),
    )


def _dataset(tmp_path: Path) -> LoadedDataset:
    return LoadedDataset(
        record=DatasetRecord(
            dataset_id="ds_orders",
            name="Orders",
            path=tmp_path / "orders.csv",
            content_hash="sha256:orders",
        ),
        frame=pd.DataFrame(
            {
                "region": ["north", "south", "north"],
                "amount": [10.0, 20.0, 30.0],
            }
        ),
    )


def _profile_artifact() -> Artifact:
    profile = DatasetProfile(
        dataset_id="ds_orders",
        name="Orders",
        content_hash="sha256:orders",
        rows=3,
        columns=2,
        column_names=["region", "amount"],
        dtypes={"region": "object", "amount": "float64"},
        missing_values={"region": 0, "amount": 0},
        missing_percent={"region": 0.0, "amount": 0.0},
        numeric_columns=["amount"],
        categorical_columns=["region"],
    )
    return Artifact(
        id="art_profile",
        type=ArtifactType.DATASET_PROFILE,
        project_id="demo",
        session_id="run_source",
        payload=profile.model_dump(mode="json"),
    )


def _candidate(
    *,
    statement: str,
    family: InsightFamily,
    columns: tuple[str, ...],
    sequence_index: int,
):
    return candidate_seed(
        HypothesisProposal(
            statement=statement,
            rationale="Test a decision-relevant orders pattern.",
            expected_evidence="A typed comparison receipt.",
            falsification_conditions=("No material pattern is observed.",),
            family=family,
            method_family="compare_groups",
            dataset_ids=("ds_orders",),
            columns=columns,
            probe_kind="group_difference",
            predicate=HypothesisPredicate(
                metric=columns[-1],
                operator="differs",
                left_operand=columns[0],
            ),
        ),
        sequence_index=sequence_index,
    )


def _decision(candidate, *, chosen: bool = True) -> SchedulingDecision:
    return SchedulingDecision(
        hypothesis_id=candidate.hypothesis_id,
        hypothesis_fingerprint=candidate.hypothesis_fingerprint,
        family=candidate.proposal.family,
        status="admitted",
        admission_checks=(),
        priority_features=PriorityFeatures(
            business_value=1,
            information_gain_proxy=1,
            novelty=1,
            coverage_gap=1,
            feasibility=1,
            expected_cost=0.1,
            redundancy=0,
            multiplicity_risk=0,
        ),
        priority=1,
        scoring_policy_version="test-v1",
        quota_deferred=False,
        chosen=chosen,
    )


def test_dataset_usage_meter_never_projects_unknown_data_as_free(
    tmp_path: Path,
) -> None:
    meter = worker.DatasetToolUsageMeter([_dataset(tmp_path)])
    tool = _tool()
    call = LLMToolCall(
        call_id="call_1",
        name=tool.name,
        arguments={"dataset_id": "ds_orders"},
    )
    arguments = _ToolArgs(dataset_id="ds_orders")

    projected = meter.project(call=call, tool=tool, arguments=arguments)
    assert projected.rows_scanned == 3
    assert projected.result_cells == 6
    assert meter.failure(
        call=call,
        tool=tool,
        arguments=arguments,
        error=RuntimeError("failed after scan"),
        projected=projected,
    ) == projected

    receipt = build_receipt(
        tool_call_id="call_1",
        tool_name=tool.name,
        tool_version="1",
        arguments={"dataset_id": "ds_orders"},
        raw_output={"count": 2},
        artifact_ids=(),
        result_count=2,
        scope=ReceiptScope(
            dataset_ids=("ds_orders",),
            columns=("region", "amount"),
            scope_resolution="resolved",
        ),
        facts=(),
        method=ReceiptMethod(family="catalog"),
        data_state_witness="dsw1_test",
        created_at=datetime.now(UTC).isoformat(),
    )
    receipt_artifact = Artifact(
        id="art_receipt",
        type=ArtifactType.EVIDENCE_RECEIPT,
        project_id="demo",
        session_id="expl_test",
        payload=receipt.model_dump(mode="json"),
    )
    settled = meter.success(
        call=call,
        tool=tool,
        arguments=arguments,
        result=AgentToolResult(content={}, receipt_artifact=receipt_artifact),
        projected=projected,
    )
    assert settled.rows_scanned == 3
    assert settled.result_cells == 2

    with pytest.raises(ValueError, match="unknown dataset"):
        meter.project(
            call=call,
            tool=tool,
            arguments=_ToolArgs(dataset_id="ds_other"),
        )


def test_worker_invokes_official_composition_with_certified_tools_and_meter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ArtifactStore(tmp_path)
    store.ensure_project("demo", name="Demo")
    tool = _tool()
    tool_digest = exploration_tool_capability_digest((tool,))
    bindings = E4aEvidenceBindings(
        checker_version="checker-v1",
        code_fingerprint="code-v1",
        tool_capability_digest=tool_digest,
        evidence_key_id=TEST_EVIDENCE_KEY_ID,
    )
    certificate = release_certificate(bindings=bindings)
    policy = build_exploration_policy(
        tier="quick",
        dataset_scope=("ds_orders",),
        tool_capability_digest=tool_digest,
    )
    exploration_id = "expl_worker"
    witness = "dsw1_" + "a" * 32
    metadata = ExplorationRunMetadata(
        exploration_id=exploration_id,
        source_session_id="run_source",
        project_id="demo",
        policy=policy,
        data_state_witness=witness,
        release_certificate_digest=certificate.certificate_digest,
        approval_action_hash="approval_hash",
        created_at=datetime.now(UTC),
    )
    journal = JsonlExplorationJournal(
        shadow_run_root(tmp_path, exploration_id) / "journal.jsonl"
    )
    journal.initialize(
        exploration_id=exploration_id,
        policy=policy,
        code_fingerprint=bindings.code_fingerprint,
        data_state_witness=witness,
    )
    selected = _dataset(tmp_path)
    profile = _profile_artifact()
    captured: dict[str, Any] = {}
    registry = StatTestRegistry(
        shadow_run_root(tmp_path, exploration_id) / "stat_registry.jsonl"
    )
    registry.begin_attempt(
        family_id="fam_orders_amount",
        requested_test_type="welch_t",
        arguments_digest="args-v1",
        logical_step_id="step_failed_stat",
    )

    monkeypatch.setattr(worker, "load_configured_release_certificate", lambda: certificate)
    monkeypatch.setattr(
        worker,
        "TRUSTED_EXPLORATION_RUNTIME_IDENTITY",
        runtime_identity(bindings=bindings),
    )
    monkeypatch.setattr(worker, "_load_and_verify_metadata", lambda *_args: metadata)
    monkeypatch.setattr(worker, "_verify_consumed_approval", lambda *_args: None)
    monkeypatch.setattr(worker, "build_data_tools", lambda _context: [tool])
    monkeypatch.setattr(
        worker,
        "resolve_exploration_source_snapshot",
        lambda *_args: ExplorationSourceSnapshot(
            project_id="demo",
            dataset_ids=("ds_orders",),
            data_state_witness=witness,
        ),
    )
    monkeypatch.setattr(
        worker,
        "load_run",
        lambda *_args, **_kwargs: SimpleNamespace(
            ok=True,
            datasets_available=True,
            result=SimpleNamespace(
                loaded_datasets=[selected],
                artifacts=[profile],
            ),
        ),
    )

    def fake_composition(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(result=SimpleNamespace(status="paused"))

    monkeypatch.setattr(worker, "run_composed_shadow_exploration", fake_composition)
    llm = SimpleNamespace(settings=LLMSettings(provider=LLMProvider.OPENAI))
    params = {
        "source_session_id": "run_source",
        "exploration_id": exploration_id,
        "policy": policy.model_dump(mode="json"),
        "data_state_witness": witness,
        "code_fingerprint": bindings.code_fingerprint,
        "release_certificate_digest": certificate.certificate_digest,
        "provider": "openai",
        "payload_policy": "schema+aggregates",
        "operation": "start",
    }

    worker.run_exploration_worker(
        store,
        tmp_path,
        {"project_id": "demo", "request_scope": "run_source"},
        params,
        llm=llm,  # type: ignore[arg-type]
    )

    assert captured["tools"] == (tool,)
    assert isinstance(captured["usage_meter"], worker.DatasetToolUsageMeter)
    assert captured["stat_attempt_counts"]() == {"fam_orders_amount": 1}
    assert captured["policy"] == policy
    assert captured["code_fingerprint"] == bindings.code_fingerprint
    assert captured["data_state_witness"] == witness


def test_worker_refuses_to_enter_composition_without_release_certificate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool_digest = exploration_tool_capability_digest((_tool(),))
    policy = build_exploration_policy(
        tier="quick",
        dataset_scope=("ds_orders",),
        tool_capability_digest=tool_digest,
    )
    entered = False

    def fake_composition(**_kwargs: Any) -> SimpleNamespace:
        nonlocal entered
        entered = True
        return SimpleNamespace(result=SimpleNamespace(status="paused"))

    monkeypatch.setattr(worker, "load_configured_release_certificate", lambda: None)
    monkeypatch.setattr(worker, "run_composed_shadow_exploration", fake_composition)
    with pytest.raises(RuntimeError, match="certificate is unavailable"):
        worker.run_exploration_worker(
            ArtifactStore(tmp_path),
            tmp_path,
            {"project_id": "demo", "request_scope": "run_source"},
            {
                "source_session_id": "run_source",
                "exploration_id": "expl_closed",
                "policy": policy.model_dump(mode="json"),
                "data_state_witness": "dsw1_" + "a" * 32,
                "code_fingerprint": "code-v1",
                "release_certificate_digest": "cert_missing",
                "provider": "openai",
                "payload_policy": "schema+aggregates",
                "operation": "start",
            },
            llm=SimpleNamespace(  # type: ignore[arg-type]
                settings=LLMSettings(provider=LLMProvider.OPENAI)
            ),
        )
    assert entered is False


def test_restart_rebuilds_multi_round_scheduler_and_goal_state_from_disk(
    tmp_path: Path,
) -> None:
    first = _candidate(
        statement="Does order amount differ materially by region?",
        family=InsightFamily.DIAGNOSTIC,
        columns=("region", "amount"),
        sequence_index=1,
    )
    second = _candidate(
        statement="Do order counts differ by region?",
        family=InsightFamily.DESCRIPTIVE,
        columns=("region",),
        sequence_index=2,
    )
    insight = InsightRecord(
        insight_id="ins_goal",
        hypothesis_id=first.hypothesis_id,
        family=first.proposal.family,
        status="new",
        trust_level="supported",
        claim_bundle_id="bundle_goal",
        supporting_receipt_ids=("receipt_goal",),
        proof=(
            InsightProof(
                receipt_id="receipt_goal",
                fact_ids=("fact_goal",),
                comparison="supports",
            ),
        ),
        created_round=0,
        last_updated_round=0,
    )
    durable = ExplorationWorkflowState(
        decisions=(_decision(first), _decision(second)),
        insights={insight.insight_id: insight},
        coverage_completed={first.coverage_key},
    )
    state_path = tmp_path / "workflow-state.json"
    JsonExplorationWorkflowStateStore(state_path).remember(durable)
    restored = JsonExplorationWorkflowStateStore(state_path).load()
    policy = build_exploration_policy(
        tier="quick",
        dataset_scope=("ds_orders",),
        tool_capability_digest="tools-v1",
        mode="goal_directed",
        goal="Why does order amount vary by region?",
    )

    historical, answered, executed, quotas = worker._durable_admission_state(
        restored, policy
    )
    assert historical == {
        first.hypothesis_fingerprint,
        second.hypothesis_fingerprint,
    }
    assert answered == {first.hypothesis_fingerprint}
    assert executed == {
        first.hypothesis_fingerprint[:16],
        second.hypothesis_fingerprint[:16],
    }
    assert quotas[InsightFamily.DIAGNOSTIC] == 0
    assert quotas[InsightFamily.DESCRIPTIVE] == 0

    exploration_id = "expl_restart"
    journal = JsonlExplorationJournal(
        shadow_run_root(tmp_path, exploration_id) / "journal.jsonl"
    )
    journal.initialize(
        exploration_id=exploration_id,
        policy=policy,
        code_fingerprint="code-v1",
        data_state_witness="dsw1_" + "a" * 32,
    )
    journal.append_new("round_started", round_index=0)
    recovery = JsonSupervisorRecoveryStore(
        shadow_run_root(tmp_path, exploration_id) / "phase-responses"
    )
    recovery.remember(
        phase_step_id(exploration_id, 0, SupervisorPhase.GENERATE),
        CandidateBatch((first,)),
    )

    assert worker._goal_satisfied(
        restored,
        goal=policy.goal,
        recovery=recovery,
        journal=journal,
    )
    assert not worker._goal_satisfied(
        restored,
        goal="customer churn retention",
        recovery=recovery,
        journal=journal,
    )
    assert not worker._goal_satisfied(
        ExplorationWorkflowState(
            decisions=restored.decisions,
            insights={
                insight.insight_id: insight.model_copy(
                    update={"status": "inconclusive", "trust_level": "unsupported"}
                )
            },
        ),
        goal=policy.goal,
        recovery=recovery,
        journal=journal,
    )
