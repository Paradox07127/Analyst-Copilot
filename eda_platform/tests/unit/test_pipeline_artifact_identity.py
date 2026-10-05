from datetime import timedelta
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pytest

from eda_platform.agents.data_tool_result_contracts import verify_data_tool_result_contract
from eda_platform.agents.data_tools import (
    AnalyzeTimeSeriesArguments,
    DataToolContext,
    RunForecastArguments,
    build_data_tools,
)
from eda_platform.core.ids import make_artifact_id
from eda_platform.core.kernel import SessionContext, run_pipeline
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.question_exec import _valid_method_artifact
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.datasets import DatasetRecord
from eda_platform.schemas.receipts import EvidenceReceipt
from eda_platform.tools.loader import LoadedDataset
from eda_platform.tools.sql_runner import build_catalog


def context(tmp_path: Path) -> tuple[SessionContext, DataToolContext]:
    session = SessionContext(project_id="project", session_id="run", store=ArtifactStore(tmp_path))
    dataset = LoadedDataset(record=DatasetRecord(
        dataset_id="daily", name="daily.csv", path=tmp_path / "daily.csv", content_hash="daily",
    ), frame=pd.DataFrame({
        "date": list(pd.date_range("2024-01-01", periods=28, freq="D")) * 2,
        "value": [100.0 + index * 2 + (index % 7) * 3 for index in range(28)] * 2,
    }))
    tools = DataToolContext(
        datasets=[dataset], catalog=build_catalog([dataset]), project_id="project",
        session_id="run", store=session.store, payload_policy="schema+aggregates",
    )
    return session, tools


class Outputs:
    name: ClassVar[str] = "artifact_outputs"
    requires: ClassVar[tuple[ArtifactType, ...]] = ()
    produces: ClassVar[tuple[ArtifactType, ...]] = (
        ArtifactType.TABLE, ArtifactType.EVIDENCE_RECEIPT,
    )
    parallel_safe: ClassVar[bool] = True

    def __init__(self, artifacts: list[Artifact], key: str = "outputs") -> None:
        self.artifacts = artifacts
        self.key = key
        self.calls = 0

    def cache_key(self, ctx: SessionContext) -> str:
        return self.key

    def run(self, ctx: SessionContext) -> list[Artifact]:
        self.calls += 1
        return self.artifacts


@pytest.mark.parametrize("name,schema", [
    ("analyze_time_series", AnalyzeTimeSeriesArguments),
    ("run_forecast", RunForecastArguments),
])
def test_inferred_and_explicit_frequency_preserve_distinct_evidence_and_replay(
    tmp_path: Path, name: str,
    schema: type[AnalyzeTimeSeriesArguments] | type[RunForecastArguments],
) -> None:
    session, tools = context(tmp_path)
    tool = next(t for t in build_data_tools(tools) if t.name == name)
    primary = []
    for frequency in (None, "D"):
        arguments = schema(dataset_id="daily", time_column="date", value_column="value",
                           freq=frequency)
        result = tool.execute(arguments)
        assert result.receipt_artifact is not None
        receipt = EvidenceReceipt.model_validate(result.receipt_artifact.payload)
        verify_data_tool_result_contract(receipt, result, arguments.model_dump(mode="json"))
        artifact = result.artifacts[0]
        assert _valid_method_artifact(artifact)
        assert artifact.id == make_artifact_id("table", artifact.payload)
        stored = session.store.get_artifact(artifact.id, project_id="project", session_id="run")
        assert stored.payload["method_context"]["arguments"]["freq"] == frequency
        assert stored.payload["method_context"]["warnings"] == stored.warnings
        primary.append(artifact)
    assert primary[0].id != primary[1].id
    assert primary[0].warnings != primary[1].warnings
    assert ({k: v for k, v in primary[0].payload.items() if k != "method_context"}
            == {k: v for k, v in primary[1].payload.items() if k != "method_context"})
    step = Outputs(tools.artifacts)
    first = run_pipeline([step], session)
    resumed = run_pipeline([step], session)
    assert step.calls == 1
    assert {a.id for a in first.artifacts} == {a.id for a in resumed.artifacts}


@pytest.mark.parametrize("field,changed", [
    ("warnings", ["different disclosure"]), ("parents", ["other-source"]),
    ("payload", {"value": 2}), ("env_digest", "other-environment"),
])
@pytest.mark.parametrize("batch", [False, True])
def test_conflicting_outputs_are_rejected_before_any_batch_publication(
    tmp_path: Path, field: str, changed: object, batch: bool,
) -> None:
    session, _ = context(tmp_path)
    artifact = Artifact(id="table_duplicate", type=ArtifactType.TABLE, project_id="project",
                        session_id="run", payload={"value": 1})
    conflict = artifact.model_copy(update={field: changed})
    steps = ([Outputs([artifact], "a"), Outputs([conflict], "b")] if batch
             else [Outputs([artifact, conflict])])
    with pytest.raises(ValueError, match="Conflicting content for artifact"):
        run_pipeline(steps, session, max_workers=2 if batch else 1)
    assert session.store.list_artifacts(project_id="project", session_id="run") == []
    assert not any(event.event_type == "step_completed" for event in
                   session.store.list_trace_events(project_id="project", session_id="run"))


def test_equivalent_outputs_deduplicate_and_tool_replay_keeps_canonical_envelope(
    tmp_path: Path,
) -> None:
    session, tools = context(tmp_path)
    artifact = Artifact(id="table_duplicate", type=ArtifactType.TABLE, project_id="project",
                        session_id="run", payload={"value": 1})
    tools.add_artifact(artifact)
    later = artifact.model_copy(update={"created_at": artifact.created_at + timedelta(seconds=1),
                                       "env_digest": None})
    tools.add_artifact(later)
    stored = session.store.get_artifact(artifact.id, project_id="project", session_id="run")
    assert later.model_dump() == artifact.model_dump() == stored.model_dump()
    assert tools.artifact(artifact.id).model_dump() == tools.artifacts[0].model_dump()
    assert len(tools.artifacts) == 1
    result = run_pipeline([Outputs([artifact, later])], session)
    assert len(result.artifacts) == 1
    with pytest.raises(ValueError, match="Conflicting content"):
        tools.add_artifact(artifact.model_copy(update={"warnings": ["changed"]}))
    assert session.store.get_artifact(artifact.id, project_id="project",
                                      session_id="run").model_dump() == stored.model_dump()


def test_new_tool_context_cannot_overwrite_conflicting_durable_artifact(tmp_path: Path) -> None:
    session, first = context(tmp_path)
    artifact = Artifact(id="table_duplicate", type=ArtifactType.TABLE, project_id="project",
                        session_id="run", payload={"value": 1}, warnings=["original disclosure"])
    first.add_artifact(artifact)
    _, reopened = context(tmp_path)
    with pytest.raises(ValueError, match="Conflicting content"):
        reopened.add_artifact(artifact.model_copy(update={"warnings": ["different disclosure"]}))
    assert reopened.artifacts == []
    assert session.store.get_artifact(artifact.id, project_id="project",
                                      session_id="run").warnings == ["original disclosure"]


@pytest.mark.parametrize("equivalent", [False, True])
def test_immutable_artifact_competing_writers_are_fenced(
    tmp_path: Path, equivalent: bool,
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    session, _ = context(tmp_path)
    first = Artifact(id="race", type=ArtifactType.TABLE, project_id="project",
                     session_id="run", payload={"value": 1})
    second = first.model_copy(update={
        "created_at": first.created_at + timedelta(seconds=1),
        "warnings": [] if equivalent else ["different disclosure"],
    })
    barrier = Barrier(2)

    def write(artifact: Artifact) -> str:
        store = ArtifactStore(tmp_path)
        barrier.wait(timeout=5)
        try:
            store.save_artifacts([artifact], immutable=True)
            return "saved"
        except ValueError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(write, [first, second]))
    stored = session.store.get_artifact("race", project_id="project", session_id="run")
    if equivalent:
        assert outcomes == ["saved", "saved"]
        assert first.model_dump() == second.model_dump() == stored.model_dump()
    else:
        assert sorted(outcomes) == ["conflict", "saved"]
        winner = [first, second][outcomes.index("saved")]
        assert stored.model_dump() == winner.model_dump()


def test_immutable_batch_validates_every_existing_member_before_writing(tmp_path: Path) -> None:
    session, _ = context(tmp_path)
    old = Artifact(id="existing", type=ArtifactType.TABLE, project_id="project",
                   session_id="run", payload={"value": 1})
    session.store.save_artifact(old)
    new = old.model_copy(update={"id": "new"})
    conflict = old.model_copy(update={"warnings": ["changed"]})
    with pytest.raises(ValueError, match="Conflicting content"):
        session.store.save_artifacts([new, conflict], immutable=True)
    with pytest.raises(KeyError):
        session.store.get_artifact("new", project_id="project", session_id="run")
    assert session.store.get_artifact("existing", project_id="project",
                                      session_id="run").warnings == []


def test_default_mutable_repair_and_expected_reference_compare_and_swap(tmp_path: Path) -> None:
    from eda_platform.core.artifact_identity import artifact_content_digest

    session, _ = context(tmp_path)
    artifact = Artifact(id="existing", type=ArtifactType.TABLE, project_id="project",
                        session_id="run", payload={"value": 1})
    session.store.save_artifact(artifact)
    expected = artifact_content_digest(artifact)
    changed = artifact.model_copy(update={"payload": {"value": 2}})
    session.store.save_artifact(changed)
    with pytest.raises(ValueError, match="changed before commit"):
        session.store.save_artifacts([artifact.model_copy(update={"id": "new"})],
                                      expected_existing={"existing": expected})
    with pytest.raises(KeyError):
        session.store.get_artifact("new", project_id="project", session_id="run")
    path = session.store.artifact_path("project", "run", "existing")
    path.write_text("invalid artifact json")
    session.store.save_artifact(artifact)
    assert session.store.get_artifact("existing", project_id="project",
                                      session_id="run").payload == {"value": 1}
