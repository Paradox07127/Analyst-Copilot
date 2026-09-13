from __future__ import annotations

from pathlib import Path

import pytest

import eda_platform.core.session_loader as session_loader_module
from eda_platform.core.session_deletion import (
    SessionDeletionCoordinator,
    SessionDeletionNotFoundError,
)
from eda_platform.core.session_loader import load_run
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.auto_eda import AutoEDAResult, run_auto_eda

GOLDEN_DATA = Path(__file__).parents[1] / "golden" / "data"


def _run_ecommerce(tmp_path: Path) -> AutoEDAResult:
    """Two-dataset ecommerce run into an isolated workspace."""
    return run_auto_eda(
        [
            GOLDEN_DATA / "ecommerce_orders.csv",
            GOLDEN_DATA / "ecommerce_customers.csv",
        ],
        workspace=tmp_path / "ws",
        project_id="proj_load",
    )


def _upload_csv_paths(workspace: Path, project_id: str) -> list[Path]:
    uploads = workspace / "projects" / project_id / "uploads"
    return sorted(uploads.glob("*/v1/*.csv"))


def test_list_runs_and_load_run_roundtrip(tmp_path: Path) -> None:
    result = _run_ecommerce(tmp_path)
    workspace = tmp_path / "ws"
    store = ArtifactStore(workspace)

    runs = store.list_sessions("proj_load")
    assert len(runs) == 1
    info = runs[0]
    assert info.session_id == result.session_id
    assert set(info.dataset_names) == {
        "ecommerce_orders.csv",
        "ecommerce_customers.csv",
    }
    assert info.artifact_count > 0
    assert info.report_status is not None
    assert info.status == "completed"
    assert info.code_version is not None

    loaded = load_run("proj_load", result.session_id, workspace=workspace)
    assert loaded.ok
    assert loaded.datasets_available
    assert loaded.warnings == []
    assert loaded.result is not None
    assert loaded.result.artifacts
    assert loaded.result.report_markdown
    reloaded_names = {ds.record.name for ds in loaded.result.loaded_datasets}
    assert reloaded_names == {
        "ecommerce_orders.csv",
        "ecommerce_customers.csv",
    }


def test_load_run_survives_missing_source_csv(tmp_path: Path) -> None:
    result = _run_ecommerce(tmp_path)
    workspace = tmp_path / "ws"

    # Simulate a deleted upload: chat/relationships must degrade, not crash.
    uploads = _upload_csv_paths(workspace, "proj_load")
    assert uploads
    removed = uploads[0]
    removed.unlink()

    loaded = load_run("proj_load", result.session_id, workspace=workspace)
    assert loaded.ok  # view-only still usable
    assert not loaded.datasets_available
    assert loaded.result is not None
    assert loaded.result.artifacts
    assert loaded.result.report_markdown
    assert any(
        removed.name.rsplit(".", 1)[0] in warning or removed.name in warning
        for warning in loaded.warnings
    )
    # The warning is also surfaced on the reconstructed result.
    assert loaded.result.load_warnings == loaded.warnings


def test_load_run_reuses_manifest_content_hashes(tmp_path: Path, monkeypatch) -> None:
    result = _run_ecommerce(tmp_path)
    workspace = tmp_path / "ws"
    store = ArtifactStore(workspace)
    manifest = store.read_manifest("proj_load", result.session_id)
    assert manifest is not None

    seen_hashes: dict[str, str | None] = {}
    real_defer_csv = session_loader_module.defer_csv

    def recording_defer_csv(
        path, *, dataset_id=None, content_hash=None, frame_pool=None
    ):
        seen_hashes[Path(path).name] = content_hash
        return real_defer_csv(
            path,
            dataset_id=dataset_id,
            content_hash=content_hash,
            frame_pool=frame_pool,
        )

    monkeypatch.setattr(session_loader_module, "defer_csv", recording_defer_csv)

    loaded = load_run("proj_load", result.session_id, workspace=workspace)

    assert loaded.ok
    assert seen_hashes == manifest.input_hashes


def test_corrupt_artifact_json_is_skipped(tmp_path: Path) -> None:
    result = _run_ecommerce(tmp_path)
    workspace = tmp_path / "ws"
    store = ArtifactStore(workspace)

    artifacts_dir = (
        workspace / "projects" / "proj_load" / "sessions" / result.session_id / "artifacts"
    )
    # Baseline from disk: the store may hold meta-artifacts (e.g. SessionMetrics)
    # beyond the pipeline's returned artifact list.
    stored_files = sorted(artifacts_dir.glob("*.json"))
    victim = stored_files[0]
    victim.write_text("{bad", encoding="utf-8")

    # Session history is metadata-only and counts artifact files without parsing
    # every payload; corruption is detected when the session itself is opened.
    runs = store.list_sessions("proj_load")
    assert len(runs) == 1
    assert runs[0].artifact_count == len(stored_files)

    # load_run must not raise and skips the corrupt file with a warning.
    loaded = load_run("proj_load", result.session_id, workspace=workspace)
    assert loaded.ok
    assert loaded.result is not None
    assert len(loaded.result.artifacts) == len(stored_files) - 1
    assert any("unreadable artifact" in warning for warning in loaded.warnings)


def test_load_run_nonexistent_returns_not_ok(tmp_path: Path) -> None:
    _run_ecommerce(tmp_path)
    workspace = tmp_path / "ws"

    loaded = load_run("proj_load", "run_does_not_exist", workspace=workspace)
    assert not loaded.ok
    assert loaded.result is None
    assert not loaded.datasets_available
    assert loaded.warnings


def test_delete_run_removes_dir_and_index(tmp_path: Path) -> None:
    result = _run_ecommerce(tmp_path)
    workspace = tmp_path / "ws"
    store = ArtifactStore(workspace)

    session_dir = store.session_dir("proj_load", result.session_id)
    assert session_dir.exists()

    assert SessionDeletionCoordinator(store).delete(result.session_id).deleted
    assert not session_dir.exists()
    assert store.list_sessions("proj_load") == []
    # Deleting again finds nothing to remove.
    with pytest.raises(SessionDeletionNotFoundError):
        SessionDeletionCoordinator(store).delete(result.session_id)


def test_list_runs_survives_corrupt_manifest(tmp_path: Path) -> None:
    result = _run_ecommerce(tmp_path)
    workspace = tmp_path / "ws"
    store = ArtifactStore(workspace)

    manifest = store.session_dir("proj_load", result.session_id) / "manifest.json"
    manifest.write_text("{bad", encoding="utf-8")

    runs = store.list_sessions("proj_load")
    assert len(runs) == 1
    info = runs[0]
    assert info.session_id == result.session_id
    # Manifest-derived fields default gracefully; status still comes from SQLite.
    assert info.created_at is None
    assert info.code_version is None
    assert info.status == "completed"
    # Artifacts on disk are still scanned.
    assert info.artifact_count > 0


def test_delete_run_refuses_path_traversal(tmp_path: Path) -> None:
    """A session_id like "../uploads" must not let deletion escape outside the
    project's runs/ dir. Destructive ops are safe by construction."""
    workspace = tmp_path / "ws"
    store = ArtifactStore(workspace)
    (workspace / "projects" / "proj_load" / "sessions").mkdir(parents=True)
    victim = workspace / "projects" / "proj_load" / "uploads_keep"
    victim.mkdir(parents=True)
    (victim / "data.csv").write_text("keep me", encoding="utf-8")

    with pytest.raises(SessionDeletionNotFoundError):
        SessionDeletionCoordinator(store).delete("../uploads_keep")

    assert victim.exists()  # the escape target survived
    assert (victim / "data.csv").read_text(encoding="utf-8") == "keep me"


def test_load_derived_result_artifacts_collects_question_results(tmp_path: Path) -> None:
    """Chat context must see conclusions from derived question runs — and only
    conclusions: previews/receipts stay in their own session."""
    from datetime import UTC, datetime

    from eda_platform.core.session_loader import load_derived_result_artifacts
    from eda_platform.schemas.artifacts import Artifact, ArtifactType
    from eda_platform.schemas.sessions import SessionManifest

    workspace = tmp_path / "ws"
    store = ArtifactStore(workspace)
    store.ensure_project("proj_load", "Load")

    def manifest(session_id: str, source: str | None) -> SessionManifest:
        return SessionManifest(
            session_id=session_id,
            project_id="proj_load",
            input_hashes={},
            code_version="test",
            created_at=datetime.now(UTC),
            source_session_id=source,
        )

    def artifact(artifact_id: str, session_id: str, artifact_type: ArtifactType) -> Artifact:
        return Artifact(
            id=artifact_id,
            type=artifact_type,
            project_id="proj_load",
            session_id=session_id,
            payload={"question_id": "q1", "status": "succeeded", "findings": []},
        )

    store.start_session("proj_load", "run_src")
    store.write_manifest(manifest("run_src", None))
    store.start_session("proj_load", "qsess_run_src_1")
    store.write_manifest(manifest("qsess_run_src_1", "run_src"))
    store.start_session("proj_load", "run_other")
    store.write_manifest(manifest("run_other", None))

    store.save_artifact(
        artifact("qexec_1", "qsess_run_src_1", ArtifactType.QUESTION_EXECUTION_RESULT)
    )
    # Noise in the same derived run: not a result artifact, must not travel.
    store.save_artifact(artifact("sql_noise_1", "qsess_run_src_1", ArtifactType.SQL_RESULT))
    # A result artifact in an unrelated run: wrong lineage, must not travel.
    store.save_artifact(
        artifact("qexec_other", "run_other", ArtifactType.QUESTION_EXECUTION_RESULT)
    )

    collected = load_derived_result_artifacts(
        store, project_id="proj_load", session_id="run_src"
    )

    assert [item.id for item in collected] == ["qexec_1"]
