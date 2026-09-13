"""T4a: exploration takes over ValidatedFinding production — a gracefully
stopped exploration publishes one VALIDATED_FINDING per gate-passed insight,
and the Findings library lists them without an investigation record."""

from __future__ import annotations

from pathlib import Path

from eda_platform.agents.exploration.workflow import ExplorationWorkflowState
from eda_platform.agents.receipts import build_receipt
from eda_platform.application.services.finding_service import FindingService
from eda_platform.core.claim_gates import GateReport, claim_bundle_digest
from eda_platform.core.exploration_shadow_store import shadow_run_root
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.exploration import JsonExplorationWorkflowStateStore
from eda_platform.drivers.investigation_library import build_investigation_library
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.claims import Claim, ClaimBundle
from eda_platform.schemas.exploration import ExplorationLoopState
from eda_platform.schemas.insights import InsightProof, InsightRecord
from eda_platform.schemas.investigations import ValidatedFinding
from eda_platform.schemas.receipts import (
    ReceiptFact,
    ReceiptMethod,
    ReceiptScope,
    ReceiptStatistics,
)
from eda_platform.schemas.sessions import SessionManifest
from eda_platform.tools.loader import load_csv
from eda_platform.tools.profiler import profile_dataset
from eda_platform.worker.exploration import publish_exploration_outputs

PROJECT = "proj_xpl_vfind"
PRIMARY = "run_primary"
DERIVED = "explsess_derived"
XID = "xpl-vfind-1"
WITNESS = "dsw1_" + "c" * 60
CLAIM_TEXT = "Average revenue differs by region: East 40.0 vs West 12.5 (p-value 0.01)."
STATEMENT = "Revenue differs by region."
RATIONALE = "Planted regional structure."
LIMITATION = "Single-month sample only."


def _receipt():
    return build_receipt(
        tool_call_id="call-1",
        tool_name="run_stat_test",
        tool_version="1",
        arguments={"call": "call-1"},
        raw_output={"value": 1},
        artifact_ids=(),
        result_count=1,
        scope=ReceiptScope(
            dataset_ids=("ds_sales",),
            columns=("region", "revenue"),
            scope_resolution="explicit",
        ),
        facts=(
            ReceiptFact(
                fact_id="mean_east",
                name="mean_east",
                value=40.0,
                value_type="number",
                unit="raw",
            ),
            ReceiptFact(
                fact_id="mean_west",
                name="mean_west",
                value=12.5,
                value_type="number",
                unit="raw",
            ),
        ),
        method=ReceiptMethod(family="compare_groups"),
        statistics=ReceiptStatistics(
            hypothesis_id="hyp_1",
            hypothesis_outcome="supports",
            test_name="independent_t_test",
            test_statistic=2.5,
            p_value=0.01,
            effect_size=0.5,
            sample_size=20,
            sequence_index=1,
        ),
        data_state_witness=WITNESS,
        created_at="2026-08-03T00:00:00Z",
    )


def _workflow_state(receipt) -> ExplorationWorkflowState:
    bundle = ClaimBundle(
        claim_bundle_id="cb_1",
        hypothesis_id="hyp_1",
        evidence_lane="exploratory",
        claims=(
            Claim(
                claim_id="clm_1",
                claim_type="comparison",
                claim_text=CLAIM_TEXT,
                support_type="direct",
                evidence_fact_ids=(
                    f"{receipt.receipt_id}:mean_east",
                    f"{receipt.receipt_id}:mean_west",
                ),
                statistics_receipt_ids=(receipt.receipt_id,),
            ),
        ),
    )
    report = GateReport(
        claim_bundle_id="cb_1",
        claim_bundle_digest=claim_bundle_digest(bundle),
        run_witness=WITNESS,
        passed=True,
        verdicts=(),
        health_score=1.0,
    )
    insight = InsightRecord(
        insight_id="ins_1",
        hypothesis_id="hyp_1",
        family="Diagnostic",
        status="new",
        trust_level="supported",
        statement=STATEMENT,
        rationale=RATIONALE,
        claim_bundle_id="cb_1",
        supporting_receipt_ids=(receipt.receipt_id,),
        proof=(
            InsightProof(
                receipt_id=receipt.receipt_id,
                fact_ids=("mean_east", "mean_west"),
                comparison="supports",
            ),
        ),
        limitations=(LIMITATION,),
        created_round=0,
        last_updated_round=0,
    )
    return ExplorationWorkflowState(
        committed_receipts={receipt.receipt_id: receipt},
        gate_reports={"cb_1": report},
        admitted_bundles={"cb_1": bundle},
        insights={insight.insight_id: insight},
    )


def _loop_state() -> ExplorationLoopState:
    return ExplorationLoopState(
        exploration_id=XID,
        policy_fingerprint="pf-1",
        effective_policy_fingerprint="pf-1",
        code_fingerprint="code-v1",
        data_state_witness=WITNESS,
        attempt_epoch=0,
        status="stopped",
        stop_reason="completed",
        final_report_ref=f"exploration-eval/{XID}/report.md",
        max_successful_tool_calls=10,
        max_rounds=5,
        remaining_tool_call_budget=10,
        remaining_round_budget=5,
        last_seq=5,
    )


def _build_workspace(tmp_path: Path) -> ArtifactStore:
    workspace = tmp_path / "ws"
    store = ArtifactStore(workspace)
    store.ensure_project(PROJECT, name=PROJECT)
    store.start_session(PROJECT, PRIMARY)
    store.start_session(PROJECT, DERIVED)

    csv_path = tmp_path / "sales.csv"
    csv_path.write_text(
        "order_id,revenue,region\n1,10,East\n2,30,East\n3,12,West\n",
        encoding="utf-8",
    )
    loaded = load_csv(csv_path, dataset_id="ds_sales")
    store.save_artifact(profile_dataset(loaded, project_id=PROJECT, session_id=PRIMARY))
    store.write_manifest(
        SessionManifest(
            session_id=PRIMARY, project_id=PROJECT, input_hashes={}, code_version="test"
        )
    )
    store.write_manifest(
        SessionManifest(
            session_id=DERIVED,
            project_id=PROJECT,
            input_hashes={PRIMARY: "derived_job_lifecycle"},
            code_version="test",
            source_session_id=PRIMARY,
        )
    )

    receipt = _receipt()
    run_root = shadow_run_root(store.root, XID)
    JsonExplorationWorkflowStateStore(run_root / "workflow-state.json").remember(
        _workflow_state(receipt)
    )
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "report.md").write_text("# Exploration report\n", encoding="utf-8")
    return store


def _publish(store: ArtifactStore) -> list[Artifact]:
    return publish_exploration_outputs(
        store,
        project_id=PROJECT,
        session_id=DERIVED,
        source_session_id=PRIMARY,
        exploration_id=XID,
        goal="regional revenue drivers",
        dataset_scope=("ds_sales",),
        state=_loop_state(),
    )


def _validated_finding_artifacts(store: ArtifactStore) -> list[Artifact]:
    return [
        artifact
        for artifact in store.list_artifacts(project_id=PROJECT, session_id=DERIVED)
        if artifact.type is ArtifactType.VALIDATED_FINDING
    ]


def test_publish_produces_a_validated_finding_per_gate_passed_insight(
    tmp_path: Path,
) -> None:
    store = _build_workspace(tmp_path)
    _publish(store)

    artifacts = _validated_finding_artifacts(store)
    assert len(artifacts) == 1
    finding = ValidatedFinding.model_validate(artifacts[0].payload)

    assert finding.origin == "exploration"
    assert finding.investigation_id == XID
    assert finding.question_id == "hyp_1"
    assert finding.question == STATEMENT
    assert finding.value_hypothesis == RATIONALE
    assert finding.claim_class == "observed"
    assert [statement.text for statement in finding.findings] == [CLAIM_TEXT]
    assert LIMITATION in finding.limitations
    assert finding.report_eligible
    assert finding.report_readiness == "eligible_with_limitations"
    # No LLM interpretation ran through the report validator here.
    assert finding.interpretation_status == "absent"

    receipt_artifact = next(
        artifact
        for artifact in store.list_artifacts(project_id=PROJECT, session_id=DERIVED)
        if artifact.type is ArtifactType.EVIDENCE_RECEIPT
    )
    evidence = finding.findings[0].evidence
    assert [ref.artifact_id for ref in evidence] == [receipt_artifact.id]
    assert finding.source_artifact_ids == [receipt_artifact.id]
    assert finding.source_artifact_session_ids == {receipt_artifact.id: DERIVED}

    finding_set = next(
        artifact
        for artifact in store.list_artifacts(project_id=PROJECT, session_id=DERIVED)
        if artifact.type is ArtifactType.EXPLORATION_FINDING_SET
    )
    assert set(artifacts[0].parents) == {finding_set.id, receipt_artifact.id}


def test_finding_service_lists_exploration_findings(tmp_path: Path) -> None:
    store = _build_workspace(tmp_path)
    _publish(store)

    view = FindingService(store).list_findings(PRIMARY)
    assert len(view.findings) == 1
    summary = view.findings[0]
    assert summary.question == STATEMENT
    assert summary.statements[0].text == CLAIM_TEXT
    assert summary.statements[0].evidence, "claims must carry receipt evidence"
    assert summary.evidence_support == "high"
    assert summary.source_session_id == DERIVED
    assert LIMITATION in summary.limitations
    # No InvestigationPlan exists for an exploration, so freshness is honestly
    # unverifiable rather than silently fresh.
    assert summary.freshness.status == "unverifiable"
    # The library must not warn about the missing investigation record.
    assert not any("unverified ValidatedFinding" in warning for warning in view.warnings)


def test_republish_is_idempotent(tmp_path: Path) -> None:
    store = _build_workspace(tmp_path)
    _publish(store)
    before = store.list_artifacts(project_id=PROJECT, session_id=DERIVED)
    _publish(store)
    after = store.list_artifacts(project_id=PROJECT, session_id=DERIVED)
    assert len(after) == len(before)
    assert len(_validated_finding_artifacts(store)) == 1

    library = build_investigation_library(store, project_id=PROJECT)
    assert len(library.findings) == 1


def test_investigation_origin_findings_still_require_a_record(tmp_path: Path) -> None:
    """The record fence stays intact for the legacy producer's findings."""
    store = _build_workspace(tmp_path)
    _publish(store)
    artifact = _validated_finding_artifacts(store)[0]
    forged_payload = dict(artifact.payload)
    forged_payload["origin"] = "investigation"
    store.save_artifact(
        Artifact(
            id="forged_investigation_finding",
            type=ArtifactType.VALIDATED_FINDING,
            project_id=PROJECT,
            session_id=DERIVED,
            parents=[],
            payload=forged_payload,
        )
    )

    library = build_investigation_library(store, project_id=PROJECT)
    assert [item.artifact_id for item in library.findings] == [artifact.id]
    assert any("forged_investigation_finding" in warning for warning in library.warnings)
