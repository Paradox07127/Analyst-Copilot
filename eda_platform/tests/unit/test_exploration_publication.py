"""T1: a stopped exploration publishes its findings into the ArtifactStore and
the main session's regenerated report carries a gated deep-dive section."""

from __future__ import annotations

from pathlib import Path

from eda_platform.agents.exploration.workflow import ExplorationWorkflowState
from eda_platform.agents.receipts import build_receipt
from eda_platform.core.claim_gates import GateReport, claim_bundle_digest
from eda_platform.core.exploration_shadow_store import shadow_run_root
from eda_platform.core.ids import make_artifact_id
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.auto_eda import AutoEDAResult, generate_report_on_demand
from eda_platform.drivers.exploration import JsonExplorationWorkflowStateStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.claims import Claim, ClaimBundle
from eda_platform.schemas.exploration import (
    ExplorationFindingSet,
    ExplorationLoopState,
)
from eda_platform.schemas.insights import InsightProof, InsightRecord
from eda_platform.schemas.receipts import (
    ReceiptFact,
    ReceiptMethod,
    ReceiptScope,
    ReceiptStatistics,
    load_verified_receipt,
)
from eda_platform.schemas.reports import ReportBundle
from eda_platform.schemas.sessions import SessionManifest
from eda_platform.tools.loader import load_csv
from eda_platform.tools.profiler import profile_dataset
from eda_platform.worker.exploration import publish_exploration_outputs

PROJECT = "proj_xpl_publish"
PRIMARY = "run_primary"
DERIVED = "xplsess_derived"
XID = "xpl-publish-1"
WITNESS = "dsw1_" + "c" * 60
CLAIM_TEXT = "Average revenue differs by region: East 40.0 vs West 12.5 (p-value 0.01)."


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
        statement="Revenue differs by region.",
        rationale="Planted regional structure.",
        claim_bundle_id="cb_1",
        supporting_receipt_ids=(receipt.receipt_id,),
        proof=(
            InsightProof(
                receipt_id=receipt.receipt_id,
                fact_ids=("mean_east", "mean_west"),
                comparison="supports",
            ),
        ),
        created_round=0,
        last_updated_round=0,
    )
    return ExplorationWorkflowState(
        committed_receipts={receipt.receipt_id: receipt},
        gate_reports={"cb_1": report},
        admitted_bundles={"cb_1": bundle},
        insights={insight.insight_id: insight},
    )


def _loop_state(stop_reason: str = "completed") -> ExplorationLoopState:
    return ExplorationLoopState(
        exploration_id=XID,
        policy_fingerprint="pf-1",
        effective_policy_fingerprint="pf-1",
        code_fingerprint="code-v1",
        data_state_witness=WITNESS,
        attempt_epoch=0,
        status="stopped",
        stop_reason=stop_reason,
        final_report_ref=f"exploration-eval/{XID}/report.md",
        max_successful_tool_calls=10,
        max_rounds=5,
        remaining_tool_call_budget=10,
        remaining_round_budget=5,
        last_seq=5,
    )


def _build_workspace(tmp_path: Path) -> tuple[ArtifactStore, Artifact]:
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
    profile = profile_dataset(loaded, project_id=PROJECT, session_id=PRIMARY)
    store.save_artifact(profile)
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
    (run_root / "report.md").write_text(
        "# Exploration report\n\n- stop_reason: completed\n", encoding="utf-8"
    )
    return store, profile


def _publish(store: ArtifactStore, *, stop_reason: str = "completed") -> list[Artifact]:
    return publish_exploration_outputs(
        store,
        project_id=PROJECT,
        session_id=DERIVED,
        source_session_id=PRIMARY,
        exploration_id=XID,
        goal="regional revenue drivers",
        dataset_scope=("ds_sales",),
        state=_loop_state(stop_reason),
    )


def test_publish_writes_finding_set_receipts_and_report(tmp_path: Path) -> None:
    store, profile = _build_workspace(tmp_path)

    published = _publish(store)
    assert published, "a stopped exploration must publish artifacts"

    stored = store.list_artifacts(project_id=PROJECT, session_id=DERIVED)
    by_type: dict[ArtifactType, list[Artifact]] = {}
    for artifact in stored:
        by_type.setdefault(artifact.type, []).append(artifact)

    finding_sets = by_type.get(ArtifactType.EXPLORATION_FINDING_SET, [])
    assert len(finding_sets) == 1
    finding_set = ExplorationFindingSet.model_validate(finding_sets[0].payload)
    assert finding_set.exploration_id == XID
    assert finding_set.stop_reason == "completed"
    assert finding_set.findings, "gate-passed insights must be published"
    finding = finding_set.findings[0]
    assert finding.statement == "Revenue differs by region."
    assert finding.claims and finding.claims[0].text == CLAIM_TEXT
    assert finding.claims[0].receipt_artifact_ids, "claims must reference receipts"

    receipts = by_type.get(ArtifactType.EVIDENCE_RECEIPT, [])
    assert len(receipts) == 1
    receipt_artifact = receipts[0]
    assert receipt_artifact.id in finding.claims[0].receipt_artifact_ids
    load_verified_receipt(receipt_artifact.payload)
    # Collection closure in generate_report_on_demand walks parents back to the
    # source session; both artifacts must chain to a source profile.
    assert profile.id in receipt_artifact.parents
    assert set(finding_sets[0].parents) & {profile.id, receipt_artifact.id}

    reports = by_type.get(ArtifactType.MARKDOWN_REPORT, [])
    assert len(reports) == 1
    assert "# Exploration report" in str(reports[0].payload["markdown"])

    # Idempotent: a resume replaying the stopped run must not duplicate.
    _publish(store)
    again = store.list_artifacts(project_id=PROJECT, session_id=DERIVED)
    assert len(again) == len(stored)


def test_publish_skips_non_graceful_stops(tmp_path: Path) -> None:
    store, _profile = _build_workspace(tmp_path)
    assert _publish(store, stop_reason="failed") == []
    assert store.list_artifacts(project_id=PROJECT, session_id=DERIVED) == []


def _primary_result(store: ArtifactStore, profile: Artifact) -> AutoEDAResult:
    return AutoEDAResult(
        project_id=PROJECT,
        session_id=PRIMARY,
        business_context="",
        artifacts=[profile],
        report_markdown="",
        workspace=store.root,
        loaded_datasets=[],
    )


def test_regenerated_report_carries_gated_deep_dive_section(tmp_path: Path) -> None:
    store, profile = _build_workspace(tmp_path)
    _publish(store)

    generated = generate_report_on_demand(_primary_result(store, profile), llm=None)

    assert "Deep-Dive Exploration" in generated.report_markdown
    assert "Average revenue differs by region" in generated.report_markdown

    bundle_artifact = next(
        artifact
        for artifact in generated.artifacts
        if artifact.type is ArtifactType.REPORT_BUNDLE
    )
    bundle = ReportBundle.model_validate(bundle_artifact.payload)
    section = next(
        section for section in bundle.sections if section.title == "Deep-Dive Exploration"
    )
    assert section.claims, "the deep-dive section must carry surviving claims"
    for claim in section.claims:
        assert claim.id.startswith("xplf_")
        assert claim.evidence, "deep-dive claims must cite receipt artifacts"
        assert claim.numeric_rollup == "number_verified", (
            "deep-dive numbers must verify against receipt facts, got "
            f"{claim.numeric_rollup}"
        )


def test_deep_dive_claim_with_unsupported_number_is_pruned(tmp_path: Path) -> None:
    store, profile = _build_workspace(tmp_path)
    _publish(store)
    receipt_artifact = next(
        artifact
        for artifact in store.list_artifacts(project_id=PROJECT, session_id=DERIVED)
        if artifact.type is ArtifactType.EVIDENCE_RECEIPT
    )
    forged = ExplorationFindingSet(
        exploration_id="xpl-forged",
        source_session_id=PRIMARY,
        goal=None,
        stop_reason="completed",
        findings=(
            {
                "insight_id": "ins_forged",
                "statement": "Forged statement.",
                "status": "new",
                "trust_level": "supported",
                "claims": (
                    {
                        "text": "East revenue is 9999.",
                        "receipt_artifact_ids": (receipt_artifact.id,),
                    },
                ),
            },
        ),
    )
    payload = forged.model_dump(mode="json")
    store.save_artifact(
        Artifact(
            id=make_artifact_id("xplfindings", payload),
            type=ArtifactType.EXPLORATION_FINDING_SET,
            project_id=PROJECT,
            session_id=DERIVED,
            parents=[profile.id],
            payload=payload,
        )
    )

    generated = generate_report_on_demand(_primary_result(store, profile), llm=None)

    assert "9999" not in generated.report_markdown
    bundle_artifact = next(
        artifact
        for artifact in generated.artifacts
        if artifact.type is ArtifactType.REPORT_BUNDLE
    )
    bundle = ReportBundle.model_validate(bundle_artifact.payload)
    texts = [claim.text for section in bundle.sections for claim in section.claims]
    assert "East revenue is 9999." not in texts
    assert any("Average revenue differs by region" in text for text in texts)
