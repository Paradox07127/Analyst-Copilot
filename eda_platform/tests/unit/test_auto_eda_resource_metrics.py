from __future__ import annotations

from pathlib import Path

import pytest

from eda_platform.core.llm import OfflineLLMClient
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.auto_eda import run_auto_eda
from eda_platform.schemas.artifacts import ArtifactType
from eda_platform.schemas.handoff import AgentHandoffV3
from eda_platform.schemas.resource_metrics import EdaResourcePolicy
from eda_platform.schemas.session_metrics import SessionMetrics
from eda_platform.tools.resource_preflight import EdaResourceLimitError


def _csv(tmp_path: Path) -> Path:
    source = tmp_path / "orders.csv"
    source.write_text(
        "order_id,amount,segment\n1,10,a\n2,20,b\n3,30,a\n",
        encoding="utf-8",
    )
    return source


def test_limited_preflight_publishes_only_three_metadata_artifacts(
    tmp_path: Path,
) -> None:
    source = _csv(tmp_path)
    workspace = tmp_path / "workspace"
    policy = EdaResourcePolicy(max_single_input_bytes=10)

    result = run_auto_eda(
        [source],
        workspace=workspace,
        project_id="p",
        session_id="limited",
        llm=OfflineLLMClient(),
        resource_policy=policy,
    )

    assert {artifact.type for artifact in result.artifacts} == {
        ArtifactType.RESOURCE_PREFLIGHT,
        ArtifactType.SESSION_METRICS,
        ArtifactType.AGENT_HANDOFF,
    }
    assert result.loaded_datasets == []
    handoff = AgentHandoffV3.model_validate(
        next(
            artifact.payload
            for artifact in result.artifacts
            if artifact.type is ArtifactType.AGENT_HANDOFF
        )
    )
    metrics = SessionMetrics.model_validate(
        next(
            artifact.payload
            for artifact in result.artifacts
            if artifact.type is ArtifactType.SESSION_METRICS
        )
    )
    assert handoff.readiness.status == "blocked"
    assert handoff.capabilities.profiling == "deferred"
    assert handoff.report.status == "not_generated"
    assert any(
        action.action == "partition_or_convert_dataset_and_rerun_auto_eda"
        and action.blocking
        for action in handoff.next_actions
    )
    assert not any(
        action.action == "generate_report_if_needed" for action in handoff.next_actions
    )
    assert metrics.resource_usage.preflight_status == "limited"
    assert metrics.resource_usage.processing_mode == "metadata_only"
    # A metadata-only run has no profile, no charts and no report; calling it
    # "completed" made a degraded run indistinguishable from a full one.
    assert ArtifactStore(workspace).get_session_status("limited") == "limited"


def test_verified_limited_keeps_the_limited_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: the verified-phase check treated every non-accepted status
    as a failure, so a user-selected `on_exceed="limited"` policy turned into a
    failed session AFTER all per-table work had already run."""
    import eda_platform.drivers.auto_eda as auto_eda_module

    real_verify = auto_eda_module._verify_resource_preflight

    def limited_verify(*args: object, **kwargs: object):
        decision = real_verify(*args, **kwargs)
        return decision.model_copy(
            update={"status": "limited", "reason_codes": ["row_count_exceeded"]}
        )

    monkeypatch.setattr(auto_eda_module, "_verify_resource_preflight", limited_verify)

    source = _csv(tmp_path)
    workspace = tmp_path / "workspace"
    result = run_auto_eda(
        [source],
        workspace=workspace,
        project_id="p",
        session_id="verified_limited",
        llm=OfflineLLMClient(),
        resource_policy=EdaResourcePolicy(on_exceed="limited"),
    )

    store = ArtifactStore(workspace)
    assert store.get_session_status("verified_limited") == "limited"
    # The work done before the check is kept, not thrown away.
    types = {
        artifact.type
        for artifact in store.list_artifacts(project_id="p", session_id="verified_limited")
    }
    assert ArtifactType.RESOURCE_PREFLIGHT in types
    assert result.report_markdown == ""
    # The dataset was fully loaded before the verified check, so the handoff
    # must carry the real input hashes, not the preliminary empty manifest.
    handoff = next(
        artifact
        for artifact in store.list_artifacts(project_id="p", session_id="verified_limited")
        if artifact.type is ArtifactType.AGENT_HANDOFF
    )
    manifest = store.read_manifest("p", "verified_limited")
    assert manifest is not None and manifest.input_hashes
    assert handoff.payload["run"]["input_hashes"] == manifest.input_hashes


def test_verified_rejected_still_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import eda_platform.drivers.auto_eda as auto_eda_module

    real_verify = auto_eda_module._verify_resource_preflight

    def rejected_verify(*args: object, **kwargs: object):
        decision = real_verify(*args, **kwargs)
        return decision.model_copy(
            update={"status": "rejected", "reason_codes": ["row_count_exceeded"]}
        )

    monkeypatch.setattr(auto_eda_module, "_verify_resource_preflight", rejected_verify)

    source = _csv(tmp_path)
    workspace = tmp_path / "workspace"
    with pytest.raises(EdaResourceLimitError):
        run_auto_eda(
            [source],
            workspace=workspace,
            project_id="p",
            session_id="verified_rejected",
            llm=OfflineLLMClient(),
            resource_policy=EdaResourcePolicy(on_exceed="reject"),
        )
    assert ArtifactStore(workspace).get_session_status("verified_rejected") == "failed"


def test_rejected_preflight_fails_without_publishing_handoff(tmp_path: Path) -> None:
    source = _csv(tmp_path)
    workspace = tmp_path / "workspace"
    policy = EdaResourcePolicy(max_single_input_bytes=10, on_exceed="reject")

    with pytest.raises(EdaResourceLimitError) as caught:
        run_auto_eda(
            [source],
            workspace=workspace,
            project_id="p",
            session_id="rejected",
            llm=OfflineLLMClient(),
            resource_policy=policy,
        )

    assert caught.value.error_code == "eda_resource_limit_exceeded"
    store = ArtifactStore(workspace)
    assert store.get_session_status("rejected") == "failed"
    assert [
        artifact.type
        for artifact in store.list_artifacts(project_id="p", session_id="rejected")
    ] == [ArtifactType.RESOURCE_PREFLIGHT]
