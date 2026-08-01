from __future__ import annotations

import random
from datetime import UTC, datetime
from pathlib import Path

import pytest

from eda_platform.core.llm import OfflineLLMClient
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.auto_eda import run_auto_eda
from eda_platform.schemas.artifacts import Artifact, ArtifactType, DatasetProfile
from eda_platform.schemas.handoff import AgentHandoffV3
from eda_platform.schemas.session_metrics import SessionMetrics
from eda_platform.tools.agent_handoff import create_agent_handoff_artifact


def _profile(dataset_id: str) -> Artifact:
    profile = DatasetProfile(
        dataset_id=dataset_id,
        name=f"{dataset_id}.csv",
        content_hash=f"hash_{dataset_id}",
        rows=10,
        columns=2,
        column_names=["id", "amount"],
        dtypes={"id": "int64", "amount": "float64"},
        missing_values={"id": 0, "amount": 0},
        missing_percent={"id": 0.0, "amount": 0.0},
        numeric_columns=["id", "amount"],
        categorical_columns=[],
        semantic_type_counts={"id": 1, "numeric": 1},
        primary_key_candidates=["id"],
        grain="One row per id.",
    )
    return Artifact(
        id=f"profile_{dataset_id}",
        type=ArtifactType.DATASET_PROFILE,
        project_id="p",
        session_id="s",
        payload=profile.model_dump(mode="json"),
    )


def test_large_chart_inventory_is_bounded_and_order_independent() -> None:
    artifacts = [_profile("one"), _profile("two")]
    artifacts.append(
        Artifact(
            id="quality_one",
            type=ArtifactType.QUALITY_ISSUE_SET,
            project_id="p",
            session_id="s",
            payload={
                "dataset_id": "one",
                "issues": [{"severity": "warn", "code": "duplicate_rows"}],
            },
        )
    )
    categories = ["quality", "time", "relationship", "comparison", "distribution"]
    for index in range(10_000):
        artifacts.append(
            Artifact(
                id=f"chart_{index:03d}",
                type=ArtifactType.CHART_SPEC,
                project_id="p",
                session_id="s",
                payload={
                    "dataset_id": "one",
                    "category": categories[index % len(categories)],
                    "title": f"Chart {index}",
                },
            )
        )
    external: list[Artifact] = []
    for index in range(5_000):
        external.append(
            Artifact(
                id=f"qexec_{index:04d}",
                type=ArtifactType.QUESTION_EXECUTION_RESULT,
                project_id="p",
                session_id="qsess_external",
                payload={
                    "question_id": f"q_{index:04d}",
                    "question": "Analyze alice@example.com " + "x" * 900,
                    "origin": "llm",
                    "status": "succeeded",
                    "outcome": "answered",
                    "limitations": ["alice@example.com requires review"],
                },
            )
        )
    fixed = datetime(2026, 7, 31, tzinfo=UTC)
    first = create_agent_handoff_artifact(
        artifacts,
        project_id="p",
        session_id="s",
        producer_version="test",
        execution_fingerprint="fingerprint",
        input_hashes={"one.csv": "1", "two.csv": "2"},
        generated_at=fixed,
        external_artifacts=external,
        fetch_session_id="s",
    )
    shuffled = list(artifacts)
    random.Random(42).shuffle(shuffled)
    shuffled_external = list(external)
    random.Random(43).shuffle(shuffled_external)
    second = create_agent_handoff_artifact(
        shuffled,
        project_id="p",
        session_id="s",
        producer_version="test",
        execution_fingerprint="fingerprint",
        input_hashes={"one.csv": "1", "two.csv": "2"},
        generated_at=fixed,
        external_artifacts=shuffled_external,
        fetch_session_id="s",
    )

    assert first.id == second.id
    assert first.payload == second.payload
    assert first.created_at == fixed
    handoff = AgentHandoffV3.model_validate(first.payload)
    assert handoff.run.artifact_counts["ChartSpec"] == 10_000
    assert handoff.run.referenced_external_artifact_count == 5_000
    assert handoff.run.source_inventory_count == len(artifacts)
    assert len(handoff.run.source_inventory_digest) == 64
    assert len(handoff.run.external_inventory_digest) == 64
    assert len([item for item in handoff.artifact_catalog if item.type == "ChartSpec"]) == 5
    assert len(handoff.artifact_catalog) <= 128
    assert len(handoff.question_results) == 32
    assert handoff.context_policy.omitted_question_result_count == 4_968
    assert handoff.context_policy.cataloged_artifact_count == len(handoff.artifact_catalog)
    assert handoff.context_policy.default_artifact_count == len(
        handoff.context_policy.default_artifact_ids
    )
    catalog_by_id = {entry.artifact_id: entry for entry in handoff.artifact_catalog}
    assert handoff.context_policy.default_artifact_bytes == sum(
        catalog_by_id[artifact_id].content_bytes
        for artifact_id in handoff.context_policy.default_artifact_ids
    )
    assert handoff.context_policy.default_artifact_estimated_tokens == sum(
        catalog_by_id[artifact_id].estimated_tokens
        for artifact_id in handoff.context_policy.default_artifact_ids
    )
    assert handoff.context_policy.initial_context_bytes == (
        handoff.context_policy.serialized_bytes + handoff.context_policy.default_artifact_bytes
    )
    assert handoff.context_policy.initial_context_estimated_tokens == (
        handoff.context_policy.estimated_tokens
        + handoff.context_policy.default_artifact_estimated_tokens
    )
    assert handoff.context_policy.initial_context_bytes <= 64 * 1024
    assert handoff.context_policy.initial_context_estimated_tokens <= 16_000
    assert handoff.context_policy.serialized_bytes == len(handoff.model_dump_json().encode("utf-8"))
    assert (
        handoff.context_policy.estimated_tokens
        == (handoff.context_policy.serialized_bytes + 3) // 4
    )
    assert len(first.parents) <= 128
    assert handoff.run.lineage_parents_truncated is True
    assert len(first.model_dump_json().encode("utf-8")) <= 128 * 1024
    assert "alice@example.com" not in first.model_dump_json()
    external_entry = next(
        entry
        for entry in handoff.artifact_catalog
        if entry.type is ArtifactType.QUESTION_EXECUTION_RESULT
    )
    assert external_entry.origin_session_id == "qsess_external"
    assert external_entry.fetch.startswith("/api/v1/sessions/qsess_external/artifacts/")
    assert handoff.datasets[0].readiness.status == "limited"
    assert handoff.readiness.cross_dataset_relationships.status == "deferred"
    assert handoff.readiness.cross_dataset_relationships.cross_table_claims_allowed is False
    with pytest.raises(ValueError):
        AgentHandoffV3.model_validate({**first.payload, "unexpected": True})


def test_auto_eda_returns_exact_final_store_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "orders.csv"
    source.write_text(
        "order_id,amount,segment\n1,10,a\n2,20,b\n3,30,a\n",
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    result = run_auto_eda(
        [source],
        workspace=workspace,
        project_id="p",
        session_id="s",
        llm=OfflineLLMClient(),
        generate_report=False,
    )
    store = ArtifactStore(workspace)
    stored = store.list_artifacts(project_id="p", session_id="s")

    assert [artifact.id for artifact in result.artifacts] == [artifact.id for artifact in stored]
    assert store.get_session_status("s") == "completed"
    handoffs = [artifact for artifact in stored if artifact.type is ArtifactType.AGENT_HANDOFF]
    assert len(handoffs) == 1
    handoff = AgentHandoffV3.model_validate(handoffs[0].payload)
    assert handoff.run.pipeline_artifact_count == len(stored) - 2
    assert handoff.run.persisted_source_artifact_count == len(stored) - 1
    assert handoff.run.referenced_external_artifact_count == 0
    assert handoff.run.source_inventory_count == len(stored) - 1
    assert handoff.run.artifact_count == len(stored)
    assert sum(handoff.run.artifact_counts.values()) == len(stored)
    assert handoff.run.artifact_counts["AgentHandoff"] == 1
    metrics_artifact = next(
        artifact for artifact in stored if artifact.type is ArtifactType.SESSION_METRICS
    )
    metrics = SessionMetrics.model_validate(metrics_artifact.payload)
    assert sum(metrics.artifact_counts.values()) == len(stored)
    assert metrics.artifact_counts["AgentHandoff"] == 1
    usage = metrics.resource_usage
    assert metrics.schema_version == 6
    assert usage.measurement_status == "verified"
    assert usage.preflight_status == "accepted"
    assert usage.processing_mode == "exact_in_memory"
    assert usage.inputs.analysis.dataset_count == 1
    assert usage.inputs.analysis.file_bytes == source.stat().st_size
    assert usage.inputs.analysis.rows == 3
    assert usage.inputs.analysis.columns == 3
    assert usage.inputs.analysis.frame_deep_bytes > 0
    assert usage.memory.peak_rss_bytes is not None
    assert usage.memory.working_set_budget_bytes == 2 << 30
    assert usage.artifacts.artifact_count == len(stored)
    assert usage.artifacts.agent_handoff_payload_bytes == (handoff.context_policy.serialized_bytes)
    assert usage.artifacts.default_context_bytes == (handoff.context_policy.initial_context_bytes)
    assert usage.artifacts.default_context_estimated_tokens == (
        handoff.context_policy.initial_context_estimated_tokens
    )
    assert handoff.report.status == "not_generated"


def test_agent_handoff_does_not_repeat_pii_values(tmp_path: Path) -> None:
    source = tmp_path / "contacts.csv"
    source.write_text(
        "customer_email,amount\nalice@example.com,10\nbob@example.com,20\n",
        encoding="utf-8",
    )
    result = run_auto_eda(
        [source],
        workspace=tmp_path / "workspace",
        project_id="p",
        session_id="pii",
        llm=OfflineLLMClient(),
        generate_report=False,
    )
    handoff = next(
        artifact for artifact in result.artifacts if artifact.type is ArtifactType.AGENT_HANDOFF
    )
    serialized = handoff.model_dump_json()
    assert "alice@example.com" not in serialized
    assert handoff.payload["datasets"][0]["pii_columns"] == {"customer_email": "email"}


def test_raw_lineage_and_parent_sensitivity_are_safe_by_default() -> None:
    profile_artifact = _profile("clean")
    profile_artifact.payload["pii_columns"] = {"email": "email"}
    legacy = Artifact(
        id="legacy",
        type=ArtifactType.EDA_HANDOFF,
        project_id="p",
        session_id="s",
        payload={"datasets": [{"dataset_id": "clean", "raw_dataset_id": "raw"}]},
    )
    raw_preview = Artifact(
        id="raw_preview",
        type=ArtifactType.RAW_DATA_PREVIEW,
        project_id="p",
        session_id="s",
        payload={"dataset_id": "raw", "rows": [["alice@example.com"]]},
    )
    derived_table = Artifact(
        id="derived_table",
        type=ArtifactType.TABLE,
        project_id="p",
        session_id="s",
        parents=[raw_preview.id],
        payload={"title": "Sensitive result", "rows": [{"email": "alice@example.com"}]},
    )
    question = Artifact(
        id="qexec_pii",
        type=ArtifactType.QUESTION_EXECUTION_RESULT,
        project_id="p",
        session_id="s",
        parents=[derived_table.id],
        payload={
            "question_id": "q_pii",
            "question": "Analyze alice@example.com",
            "origin": "llm",
            "status": "succeeded",
            "outcome": "answered",
            "limitations": ["alice@example.com must be reviewed"],
        },
    )

    handoff_artifact = create_agent_handoff_artifact(
        [profile_artifact, legacy, raw_preview, derived_table, question],
        project_id="p",
        session_id="s",
        producer_version="test",
        execution_fingerprint="fingerprint",
        input_hashes={},
    )
    handoff = AgentHandoffV3.model_validate(handoff_artifact.payload)
    catalog = {entry.artifact_id: entry for entry in handoff.artifact_catalog}

    assert "alice@example.com" not in handoff_artifact.model_dump_json()
    assert catalog[raw_preview.id].dataset_id == "clean"
    assert catalog[raw_preview.id].sensitivity == "pii_restricted"
    assert catalog[derived_table.id].sensitivity == "pii_restricted"
    assert raw_preview.id not in handoff.context_policy.default_artifact_ids
    assert derived_table.id not in handoff.context_policy.default_artifact_ids
    assert handoff.datasets[0].raw_dataset_id == "raw"
    assert handoff.datasets[0].artifact_ids["raw_preview"] == raw_preview.id
    assert handoff.question_results[0].limitation_count == 1


def test_report_uses_latest_audit_and_worst_status_gate() -> None:
    old = Artifact(
        id="audit_old",
        type=ArtifactType.REPORT_AUDIT,
        project_id="p",
        session_id="s",
        created_at=datetime(2026, 7, 30, tzinfo=UTC),
        payload={"status": "validated", "gate_verdict": "pass", "findings": []},
    )
    latest = Artifact(
        id="audit_latest",
        type=ArtifactType.REPORT_AUDIT,
        project_id="p",
        session_id="s",
        created_at=datetime(2026, 7, 31, tzinfo=UTC),
        payload={
            "status": "blocked_for_review",
            "gate_verdict": "pass",
            "findings": [],
        },
    )
    handoff_artifact = create_agent_handoff_artifact(
        [_profile("one"), old, latest],
        project_id="p",
        session_id="s",
        producer_version="test",
        execution_fingerprint="fingerprint",
        input_hashes={},
        generated_at=datetime(2026, 7, 31, tzinfo=UTC),
    )
    handoff = AgentHandoffV3.model_validate(handoff_artifact.payload)

    assert handoff.report.status == "failed"
    assert handoff.report.audit_artifact_id == latest.id
    assert handoff.readiness.report_gate.status == "fail"
    assert handoff.readiness.status == "blocked"
    audit_ids = {
        entry.artifact_id
        for entry in handoff.artifact_catalog
        if entry.type is ArtifactType.REPORT_AUDIT
    }
    assert audit_ids == {latest.id}


def test_limited_resource_preflight_defers_full_eda_and_blocks_report_action() -> None:
    preflight = Artifact(
        id="resource_preflight",
        type=ArtifactType.RESOURCE_PREFLIGHT,
        project_id="p",
        session_id="s",
        payload={
            "status": "limited",
            "compute_mode": "metadata_only",
            "reason_codes": ["working_set_limit"],
        },
    )
    handoff_artifact = create_agent_handoff_artifact(
        [preflight],
        project_id="p",
        session_id="s",
        producer_version="test",
        execution_fingerprint="fingerprint",
        input_hashes={},
    )
    handoff = AgentHandoffV3.model_validate(handoff_artifact.payload)

    assert handoff.readiness.status == "blocked"
    assert "resource_preflight_limited" in handoff.readiness.reasons
    assert handoff.capabilities.profiling == "deferred"
    assert handoff.capabilities.quality == "deferred"
    assert handoff.capabilities.visualization == "deferred"
    assert handoff.capabilities.statistics == "deferred"
    assert handoff.capabilities.questions == "deferred"
    assert handoff.capabilities.report == "deferred"
    catalog_entry = next(
        entry for entry in handoff.artifact_catalog if entry.artifact_id == preflight.id
    )
    assert (catalog_entry.stage, catalog_entry.role, catalog_entry.priority) == (
        "ingest",
        "gate",
        "critical",
    )
    assert catalog_entry.required is True
    actions = {action.action for action in handoff.next_actions}
    assert "partition_or_convert_dataset_and_rerun_auto_eda" in actions
    assert "generate_report_if_needed" not in actions
