"""Focused regressions for filesystem, partition, secret and Host boundaries."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from eda_platform.api.main import create_app
from eda_platform.application.backfill import backfill_sessions_index
from eda_platform.application.ports import JobCommand, JobRef
from eda_platform.application.services.job_service import (
    JobService,
    JobValidationError,
)
from eda_platform.application.services.session_fork_service import (
    SessionForkService,
    SessionForkValidationError,
)
from eda_platform.core.store import ArtifactStore
from eda_platform.core.support_docs import (
    save_support_doc,
    save_support_doc_extraction,
    support_docs_dir,
)
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.tools.cleaning import _next_free_version, _reserve_version_dir
from eda_platform.worker.runner import _sanitize_error


class _Backend:
    def __init__(self) -> None:
        self.commands: list[JobCommand] = []

    def enqueue(self, command: JobCommand) -> JobRef:
        self.commands.append(command)
        return JobRef(job_id=command.job_id)

    def cancel(self, job_id: str) -> None:
        return None

    def status(self, job_id: str) -> str:
        return "queued"


def _project_csv(store: ArtifactStore, project_id: str, dataset_id: str) -> Path:
    directory = store.project_dir(project_id) / "uploads" / dataset_id / "v1"
    directory.mkdir(parents=True)
    path = directory / "orders.csv"
    path.write_text("region,amount\nnorth,1\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("session_id", ["../escape", "run/escape", "run\\escape", ".", "bad id"])
def test_job_service_rejects_unsafe_session_id_segments(
    tmp_path: Path, session_id: str
) -> None:
    store = ArtifactStore(tmp_path)
    store.ensure_project("demo", name="Demo")
    _project_csv(store, "demo", "ds_orders")
    backend = _Backend()

    with pytest.raises(JobValidationError, match="Session ID"):
        JobService(store, backend).create_job(
            session_id,
            kind="auto_eda",
            project_id="demo",
            datasets=["ds_orders"],
        )

    assert backend.commands == []
    assert not (tmp_path / "escape").exists()


def test_store_rejects_session_and_project_symlink_escapes(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    store.ensure_project("demo", name="Demo")
    outside = tmp_path / "outside"
    outside.mkdir()
    sessions = store.project_dir("demo") / "sessions"
    sessions.mkdir()
    (sessions / "session_link").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="Session directory"):
        store.session_dir("demo", "session_link")

    linked_project = tmp_path / "projects" / "linked"
    linked_project.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="Project"):
        store.ensure_project("linked", name="Linked")
    assert not store.project_exists("linked")


def test_store_rejects_symlinked_projects_root(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    outside = tmp_path / "outside-projects"
    outside.mkdir()
    (tmp_path / "projects").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="projects root"):
        store.ensure_project("demo", name="Demo")

    assert list(outside.iterdir()) == []
    assert not store.project_exists("demo")


def test_job_dataset_refs_cannot_cross_project_boundaries(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    for project_id in ("alpha", "beta"):
        store.ensure_project(project_id, name=project_id)
    _project_csv(store, "beta", "ds_foreign")
    service = JobService(store, _Backend())

    with pytest.raises(JobValidationError, match="Unknown dataset reference"):
        service.create_job(
            "run_cross_project",
            kind="auto_eda",
            project_id="alpha",
            datasets=["projects/beta/uploads/ds_foreign/v1/orders.csv"],
        )


def test_job_dataset_refs_preserve_workspace_seed_compatibility(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path)
    store.ensure_project("alpha", name="alpha")
    seed = tmp_path / "seed" / "orders.csv"
    seed.parent.mkdir()
    seed.write_text("region,amount\nnorth,1\n", encoding="utf-8")
    backend = _Backend()

    JobService(store, backend).create_job(
        "run_seed",
        kind="auto_eda",
        project_id="alpha",
        datasets=["seed/orders.csv"],
    )

    assert len(backend.commands) == 1


def test_dataset_fork_accepts_only_source_run_dataset_ids(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    store.ensure_project("demo", name="Demo")
    _project_csv(store, "demo", "ds_source")
    _project_csv(store, "demo", "ds_other")
    store.start_session("demo", "run_source")
    store.save_artifact(
        Artifact(
            id="profile_source",
            type=ArtifactType.DATASET_PROFILE,
            project_id="demo",
            session_id="run_source",
            payload={"dataset_id": "ds_source", "name": "orders.csv"},
        )
    )
    service = SessionForkService(store, JobService(store, _Backend()))

    with pytest.raises(SessionForkValidationError, match="source run"):
        service.fork(
            "run_source",
            decision_kind="dataset",
            datasets=["ds_other"],
            llm="offline",
        )


def test_support_doc_atomic_replace_does_not_follow_leaf_symlink(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "projects" / "demo"
    docs = support_docs_dir(project_dir)
    docs.mkdir(parents=True)
    outside = tmp_path / "outside.md"
    outside.write_bytes(b"keep me")
    leaf = docs / "dictionary.md"
    leaf.symlink_to(outside)

    saved = save_support_doc(project_dir, "dictionary.md", b"safe content")

    assert saved == leaf
    assert outside.read_bytes() == b"keep me"
    assert not leaf.is_symlink()
    assert leaf.read_bytes() == b"safe content"


def test_support_doc_rejects_symlinked_docs_directory(tmp_path: Path) -> None:
    project_dir = tmp_path / "projects" / "demo"
    semantic = project_dir / "semantic"
    semantic.mkdir(parents=True)
    outside = tmp_path / "outside-docs"
    outside.mkdir()
    (semantic / "docs").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic link"):
        save_support_doc(project_dir, "dictionary.md", b"safe content")

    assert list(outside.iterdir()) == []


def test_support_doc_rejects_symlinked_extraction_directory(tmp_path: Path) -> None:
    project_dir = tmp_path / "projects" / "demo"
    semantic = project_dir / "semantic"
    semantic.mkdir(parents=True)
    outside = tmp_path / "outside-extractions"
    outside.mkdir()
    (semantic / "extracted").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic link"):
        save_support_doc_extraction(
            project_dir,
            "dictionary.pdf",
            "derived text",
            source_content=b"%PDF source",
        )

    assert list(outside.iterdir()) == []


def test_cleaning_version_parent_symlink_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "cleaned"
    output.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (output / "ds_orders").symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="symbolic link"):
        _next_free_version(output, "ds_orders", 2, "orders.csv")
    with pytest.raises(RuntimeError, match="symbolic link"):
        _reserve_version_dir(output, "ds_orders", 2, "orders.csv")
    assert list(outside.iterdir()) == []


def test_backfill_skips_project_and_run_symlinks(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    store.ensure_project("demo", name="Demo")
    outside_project = tmp_path / "outside-project"
    (outside_project / "sessions" / "run_foreign").mkdir(parents=True)
    (tmp_path / "projects" / "linked").symlink_to(
        outside_project, target_is_directory=True
    )
    outside_run = tmp_path / "outside-run"
    outside_run.mkdir()
    sessions = store.project_dir("demo") / "sessions"
    sessions.mkdir()
    (sessions / "session_link").symlink_to(outside_run, target_is_directory=True)

    assert backfill_sessions_index(tmp_path) == 0
    assert store.get_session_index_row("run_foreign") is None
    assert store.get_session_index_row("run_link") is None


def test_worker_error_redacts_active_keys_and_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-active-secret")
    raw = (
        "provider rejected sk-active-secret; "
        "Authorization: Bearer header-secret; bearer loose-secret"
    )

    cleaned = _sanitize_error(raw, str(tmp_path))

    assert "sk-active-secret" not in cleaned
    assert "header-secret" not in cleaned
    assert "loose-secret" not in cleaned
    assert cleaned.count("<redacted>") == 3


def test_api_rejects_non_loopback_host_header(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path))

    assert client.get("/api/v1/projects").status_code == 200
    rejected = client.get(
        "/api/v1/projects", headers={"Host": "attacker.example"}
    )
    assert rejected.status_code == 400
    assert "Invalid host header" in rejected.text
    assert (
        client.get(
            "/api/v1/projects", headers={"Host": "localhost:8000"}
        ).status_code
        == 200
    )
    # Starlette currently parses ``[::1]:8000`` as host ``[``, so keeping
    # ``[::1]`` in the allow-list would advertise support that does not work.
    assert (
        client.get(
            "/api/v1/projects", headers={"Host": "[::1]:8000"}
        ).status_code
        == 400
    )
