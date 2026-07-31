from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from eda_platform.api.main import create_app
from eda_platform.application.distribution_view import reservoir_sample
from eda_platform.application.dto import ColumnDistributionsView, SettingsPatch
from eda_platform.application.job_results import (
    JobResultError,
    read_job_result,
    write_job_result,
)
from eda_platform.application.ports import JobCommand, JobRef
from eda_platform.application.services.data_operation_service import DataOperationService
from eda_platform.application.services.job_service import JobService
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.sessions import SessionManifest
from eda_platform.worker.runner import _write_data_operation_result

PROJECT = "data_ops"
RUN = "run_data_ops"


class _RecordingBackend:
    def __init__(self) -> None:
        self.commands: list[JobCommand] = []

    def enqueue(self, command: JobCommand) -> JobRef:
        self.commands.append(command)
        return JobRef(job_id=command.job_id)

    def cancel(self, job_id: str) -> None:
        return None

    def status(self, job_id: str) -> str:
        return "queued"


def _store(tmp_path: Path) -> ArtifactStore:
    store = ArtifactStore(tmp_path)
    store.ensure_project(PROJECT, name="Data operations")
    store.start_session(PROJECT, RUN)
    store.write_manifest(
        SessionManifest(
            session_id=RUN,
            project_id=PROJECT,
            input_hashes={"large.csv": "hash"},
            code_version="test",
        )
    )
    # The API start path must not touch this large input.
    large = store.project_dir(PROJECT) / "uploads" / "large.csv"
    large.parent.mkdir(parents=True, exist_ok=True)
    large.write_bytes(b"x" * (16 * 1024 * 1024))
    return store


def test_long_data_operation_api_starts_are_non_blocking(tmp_path: Path) -> None:
    store = _store(tmp_path)
    backend = _RecordingBackend()
    app = create_app(tmp_path)
    app.state.data_operation_service = DataOperationService(
        store,
        JobService(store, backend),
    )
    client = TestClient(app)

    requests = [
        (
            f"/api/v1/sessions/{RUN}/cleaning/preview",
            {
                "dataset_id": "ds_large",
                "trim_whitespace": True,
                "drop_duplicate_rows": True,
            },
        ),
        (
            f"/api/v1/sessions/{RUN}/cleaning/apply",
            {
                "action_hash": "12345678",
                "approval_token": "abcdefgh",
                "llm": "offline",
            },
        ),
        (
            f"/api/v1/sessions/{RUN}/datasets/ds_large/distributions",
            None,
        ),
        (
            f"/api/v1/sessions/{RUN}/charts/custom",
            {
                "dataset_id": "ds_large",
                "chart_type": "bar",
                "x_column": "x",
                "y_column": None,
                "color_column": None,
                "aggregate": "count",
                "drop_missing": True,
                "drop_outliers": False,
            },
        ),
    ]
    started_at = time.monotonic()
    for index, (path, body) in enumerate(requests):
        response = client.post(
            path,
            json=body,
            headers={"Idempotency-Key": f"data-op-{index}"},
        )
        assert response.status_code == 202
        assert response.json()["job"]["status"] == "queued"
    assert time.monotonic() - started_at < 1.0
    assert [command.kind for command in backend.commands] == [
        "cleaning_preview",
        "cleaning_apply",
        "dataset_distributions",
        "custom_chart",
    ]


def test_result_survives_service_reconstruction(tmp_path: Path) -> None:
    store = _store(tmp_path)
    backend = _RecordingBackend()
    started = DataOperationService(store, JobService(store, backend)).start(
        RUN,
        kind="dataset_distributions",
        params={"dataset_id": "ds_large"},
        idempotency_key="recover-distribution",
    )
    expected = ColumnDistributionsView(
        dataset_id="ds_large",
        session_id=RUN,
        row_count=2_000_000,
        sampled=True,
        sample_rows=10_000,
        sample_cap=10_000,
        bins=20,
        top_k=8,
    )
    write_job_result(
        store.root, PROJECT, RUN, started.job.job_id, expected.model_dump_json()
    )
    store.mark_job_status(started.job.job_id, "completed")

    recovered = DataOperationService(
        ArtifactStore(tmp_path),
        JobService(ArtifactStore(tmp_path), _RecordingBackend()),
    ).result(
        started.job.job_id,
        expected_kind="dataset_distributions",
        model=ColumnDistributionsView,
    )
    assert recovered == expected


def test_worker_files_the_result_under_the_source_run_not_the_job_run(
    tmp_path: Path,
) -> None:
    """The job's own run id is a throwaway ``dop_`` lifecycle run. Filing the
    result under it would put the document outside the source run's delete."""
    store = _store(tmp_path)
    started = DataOperationService(store, JobService(store, _RecordingBackend())).start(
        RUN,
        kind="dataset_distributions",
        params={"dataset_id": "ds_large"},
        idempotency_key="worker-writes-result",
    )
    expected = ColumnDistributionsView(
        dataset_id="ds_large",
        session_id=RUN,
        row_count=10,
        sampled=False,
        sample_rows=10,
        sample_cap=10_000,
        bins=20,
        top_k=8,
    )
    row = store.get_job(started.job.job_id)
    assert row is not None
    assert str(row["session_id"]).startswith("dop_")

    _write_data_operation_result(store, row, expected)

    results = tmp_path / "_job_results" / PROJECT / RUN
    assert [path.name for path in results.iterdir()] == [
        f"{started.job.job_id}.json"
    ]
    store.mark_job_status(started.job.job_id, "completed")
    assert (
        DataOperationService(store, JobService(store, _RecordingBackend())).result(
            started.job.job_id,
            expected_kind="dataset_distributions",
            model=ColumnDistributionsView,
        )
        == expected
    )


def test_cleaning_apply_passes_session_llm_settings_only_to_worker_env(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    backend = _RecordingBackend()

    DataOperationService(store, JobService(store, backend)).start(
        RUN,
        kind="cleaning_apply",
        params={
            "action_hash": "a" * 64,
            "approval_token": "b" * 32,
            "llm": "env",
        },
        idempotency_key="cleaning-session-settings",
        llm_env={"OPENAI_API_KEY": "session-only-key"},
    )

    command = backend.commands[-1]
    assert command.env == {"OPENAI_API_KEY": "session-only-key"}
    assert "session-only-key" not in command.params_json


def test_cleaning_apply_forwards_the_callers_session_settings_to_its_worker(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    backend = _RecordingBackend()
    app = create_app(tmp_path)
    app.state.data_operation_service = DataOperationService(
        store,
        JobService(store, backend),
    )
    app.state.settings_service.update_settings(
        SettingsPatch(provider="openai", api_key="session-only-key"),
        session_id="cleaning-session",
    )

    response = TestClient(app).post(
        f"/api/v1/sessions/{RUN}/cleaning/apply",
        json={
            "action_hash": "a" * 64,
            "approval_token": "b" * 32,
            "llm": "env",
        },
        headers={"X-EDA-Session": "cleaning-session"},
    )

    assert response.status_code == 202
    assert backend.commands[-1].env is not None
    assert backend.commands[-1].env["EDA_LLM_API_KEY"] == "session-only-key"
    assert "session-only-key" not in backend.commands[-1].params_json


@pytest.mark.parametrize("level", ["_job_results", f"_job_results/{PROJECT}"])
def test_job_result_directory_symlink_cannot_escape_workspace(
    level: str, tmp_path: Path
) -> None:
    """Any level of the tree, not just the leaf: a symlinked project directory
    would otherwise redirect every run under it past the containment check."""
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    outside.mkdir()
    link = workspace / level
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(JobResultError, match="symbolic link"):
        write_job_result(workspace, PROJECT, RUN, "job_escape", '{"ok":true}')
    assert list(outside.iterdir()) == []


def test_job_result_write_replaces_leaf_symlink_without_following_it(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    result_dir = workspace / "_job_results" / PROJECT / RUN
    result_dir.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text("secret", encoding="utf-8")
    (result_dir / "job_safe.json").symlink_to(outside)

    write_job_result(workspace, PROJECT, RUN, "job_safe", '{"ok":true}')

    assert outside.read_text(encoding="utf-8") == "secret"
    assert not (result_dir / "job_safe.json").is_symlink()
    assert read_job_result(workspace, PROJECT, RUN, "job_safe") == '{"ok":true}'


def test_distribution_scan_checks_cancellation_between_chunks() -> None:
    checks = 0

    def cancel() -> None:
        nonlocal checks
        checks += 1
        if checks == 2:
            raise RuntimeError("cancelled")

    chunks = [
        pd.DataFrame({"value": range(100)}),
        pd.DataFrame({"value": range(100, 200)}),
        pd.DataFrame({"value": range(200, 300)}),
    ]
    with pytest.raises(RuntimeError, match="cancelled"):
        reservoir_sample(chunks, cap=50, cancel_check=cancel)
    assert checks == 2
