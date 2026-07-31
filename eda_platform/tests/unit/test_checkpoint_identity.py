from __future__ import annotations

from pathlib import Path

from eda_platform.core.kernel import SessionContext
from eda_platform.core.llm import LLMSettings, OpenAICompatibleLLMClient
from eda_platform.core.provider_registry import LLMProvider
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.auto_eda import ExportAgenticReportStep, run_auto_eda
from eda_platform.schemas.artifacts import ArtifactType


def _write_csv(path: Path, *, column: str, start: int) -> None:
    path.write_text(
        f"{column}\n" + "\n".join(str(start + offset) for offset in range(30)) + "\n",
        encoding="utf-8",
    )


def test_same_run_with_different_input_recomputes_checkpointed_profiles(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    _write_csv(first_path, column="old_value", start=1)
    _write_csv(second_path, column="new_value", start=100)
    workspace = tmp_path / "workspace"

    run_auto_eda(
        [first_path],
        workspace=workspace,
        project_id="demo",
        session_id="same_run",
        generate_report=False,
    )
    second = run_auto_eda(
        [second_path],
        workspace=workspace,
        project_id="demo",
        session_id="same_run",
        generate_report=False,
    )

    profiles = [
        artifact
        for artifact in second.artifacts
        if artifact.type is ArtifactType.DATASET_PROFILE
    ]
    assert [artifact.payload["name"] for artifact in profiles] == ["second.csv"]
    assert profiles[0].payload["column_names"] == ["new_value"]
    manifest = ArtifactStore(workspace).read_manifest("demo", "same_run")
    assert manifest is not None
    assert list(manifest.input_hashes) == ["second.csv"]

    store = ArtifactStore(workspace)
    hits_before = sum(
        event.event_type == "checkpoint_hit"
        for event in store.list_trace_events(project_id="demo", session_id="same_run")
    )
    third = run_auto_eda(
        [second_path],
        workspace=workspace,
        project_id="demo",
        session_id="same_run",
        generate_report=False,
    )
    new_events = store.list_trace_events(project_id="demo", session_id="same_run")
    assert sum(event.event_type == "checkpoint_hit" for event in new_events) > hits_before
    assert any(
        event.event_type == "checkpoint_hit" and event.name == "profile_dataset"
        for event in new_events
    )
    assert any(
        artifact.type is ArtifactType.DATASET_PROFILE
        and artifact.payload["name"] == "second.csv"
        for artifact in third.artifacts
    )


def test_report_cache_key_binds_full_effective_llm_configuration(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "workspace")
    store.ensure_project("demo", "Demo")
    ctx = SessionContext(project_id="demo", session_id="run", store=store)
    first = OpenAICompatibleLLMClient(
        LLMSettings(
            provider=LLMProvider.OPENAI,
            api_key="test-key",
            model="same-model",
            temperature=0.1,
            max_tokens=1000,
            base_url="https://first.invalid/v1",
            headers={"X-Route": "first"},
            structured_output_mode="json_schema",
        )
    )
    second = OpenAICompatibleLLMClient(
        LLMSettings(
            provider=LLMProvider.OPENAI,
            api_key="test-key",
            model="same-model",
            temperature=0.9,
            max_tokens=9000,
            base_url="https://second.invalid/v1",
            headers={"X-Route": "second"},
            structured_output_mode="json_object",
        )
    )

    first_key = ExportAgenticReportStep(
        [],
        business_context="",
        llm=first,
        payload_policy="schema+aggregates",
    ).cache_key(ctx)
    second_key = ExportAgenticReportStep(
        [],
        business_context="",
        llm=second,
        payload_policy="schema+aggregates",
    ).cache_key(ctx)

    assert first_key != second_key
