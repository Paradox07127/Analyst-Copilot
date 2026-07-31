"""Review round 3 (codex): upload path fencing, NUL names, invalid-CSV rollback."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from eda_platform.application.services.dataset_service import _locate_source
from eda_platform.application.services.session_service import ProjectNotFoundError
from eda_platform.application.services.upload_service import (
    UploadService,
    UploadValidationError,
    sanitize_upload_name,
    sweep_staging,
)
from eda_platform.core.query import TrustedFileQueryEngine
from eda_platform.core.store import ArtifactStore


def _service(tmp_path: Path, **kwargs) -> tuple[ArtifactStore, UploadService]:
    store = ArtifactStore(tmp_path)
    store.ensure_project("demo", name="Demo")
    engine = TrustedFileQueryEngine([tmp_path / "projects"])
    return store, UploadService(store, engine, **kwargs)


def test_nul_and_control_chars_are_sanitized() -> None:
    assert "\x00" not in sanitize_upload_name("bad\x00.csv")
    assert sanitize_upload_name("a\x1fb.csv") == "a_b.csv"


def test_path_segment_project_id_rejected_even_if_registered(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    # Poisoned registry row: project_id ".." exists in the DB.
    import sqlite3

    with sqlite3.connect(tmp_path / "state.sqlite") as conn:
        conn.execute("insert into projects(project_id, name, path) values('..','evil','..')")
    engine = TrustedFileQueryEngine([tmp_path / "projects"])
    service = UploadService(store, engine)
    with pytest.raises(ProjectNotFoundError):
        service.create_upload("..", "a.csv", io.BytesIO(b"x\n1\n"))


def test_unparsable_csv_is_rolled_back_not_promoted(tmp_path: Path) -> None:
    _, service = _service(tmp_path)
    # A non-UTF8 binary blob that DuckDB's CSV sniffer rejects.
    payload = bytes(range(256)) * 64
    with pytest.raises(UploadValidationError):
        service.create_upload("demo", "blob.csv", io.BytesIO(payload))
    uploads_dir = tmp_path / "projects" / "demo" / "uploads"
    leftovers = list(uploads_dir.rglob("*")) if uploads_dir.exists() else []
    assert leftovers == []
    assert list((tmp_path / "_staging").iterdir()) == []


def test_sweep_refuses_symlinked_staging_root(tmp_path: Path) -> None:
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep.txt").write_text("data")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "_staging").symlink_to(victim)

    removed = sweep_staging(workspace, ttl_seconds=0)

    assert removed == 0
    assert (victim / "keep.txt").exists()


def test_locate_source_ignores_absolute_recorded_name(tmp_path: Path) -> None:
    outside = tmp_path / "outside.csv"
    outside.write_text("a\n1\n")
    version_dir = tmp_path / "projects" / "demo" / "uploads" / "ds_x" / "v1"
    version_dir.mkdir(parents=True)
    assert _locate_source(version_dir, str(outside)) is None


# --- deletion over HTTP -----------------------------------------------------
def test_delete_upload_endpoint_removes_the_file(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from eda_platform.api.main import create_app

    workspace = tmp_path / "ws"
    store = ArtifactStore(workspace)
    store.ensure_project("demo", name="Demo")
    client = TestClient(create_app(workspace))

    created = client.post(
        "/api/v1/projects/demo/uploads",
        files={"file": ("sales.csv", b"id,amount\n1,2.5\n", "text/csv")},
    )
    assert created.status_code == 201
    dataset_id = created.json()["dataset"]["dataset_id"]
    uploaded = workspace / "projects" / "demo" / "uploads" / dataset_id
    assert uploaded.is_dir()

    deleted = client.delete(f"/api/v1/projects/demo/uploads/{dataset_id}")

    assert deleted.status_code == 200
    assert deleted.json() == {
        "project_id": "demo",
        "dataset_id": dataset_id,
        "deleted": True,
    }
    assert not uploaded.exists()
    # Second delete is a 404, not a silent success on an already-gone file.
    assert client.delete(f"/api/v1/projects/demo/uploads/{dataset_id}").status_code == 404


def test_delete_upload_cannot_escape_the_uploads_directory(tmp_path: Path) -> None:
    """`..` in the dataset segment must not reach the project directory —
    the same shape that once let a support-doc delete remove a whole project."""
    from fastapi.testclient import TestClient

    from eda_platform.api.main import create_app

    workspace = tmp_path / "ws"
    store = ArtifactStore(workspace)
    store.ensure_project("demo", name="Demo")
    client = TestClient(create_app(workspace))
    client.post(
        "/api/v1/projects/demo/uploads",
        files={"file": ("sales.csv", b"id,amount\n1,2.5\n", "text/csv")},
    )

    response = client.request(
        "DELETE", "/api/v1/projects/demo/uploads/..", follow_redirects=False
    )

    assert response.status_code != 200
    assert (workspace / "projects" / "demo").is_dir()
    assert list((workspace / "projects" / "demo" / "uploads").iterdir())
