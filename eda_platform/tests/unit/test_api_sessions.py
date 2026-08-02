"""FastAPI Run/Session endpoints: happy paths, error envelope, isolation."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import eda_platform
from eda_platform.api.main import create_app
from eda_platform.core.store import ArtifactStore, ProjectOrderConflictError
from eda_platform.schemas.sessions import SessionManifest


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    store = ArtifactStore(tmp_path)
    store.ensure_project("demo", name="Demo")
    for day in (1, 2, 3):
        session_id = f"run_{day}"
        store.start_session("demo", session_id)
        store.write_manifest(
            SessionManifest(
                session_id=session_id,
                project_id="demo",
                input_hashes={"orders.csv": "abc"},
                code_version="v1",
                created_at=datetime(2026, 7, day, tzinfo=UTC),
                title=f"Run {day}",
            )
        )
        store.mark_session_status("demo", session_id, "completed")
    store.start_session("demo", "qsess__internal_1")
    store.mark_session_status("demo", "qsess__internal_1", "completed")
    for derived_session_id in ("qsess_20260704_batch", "ssess_20260704_replay"):
        store.start_session("demo", derived_session_id)
        store.write_manifest(
            SessionManifest(
                session_id=derived_session_id,
                project_id="demo",
                input_hashes={"orders.csv": "abc"},
                code_version="v1",
                created_at=datetime(2026, 7, 4, tzinfo=UTC),
                title=derived_session_id,
                source_session_id="run_3",
            )
        )
        store.mark_session_status("demo", derived_session_id, "completed")
    return tmp_path


@pytest.fixture
def client(workspace: Path) -> TestClient:
    return TestClient(create_app(workspace))


def test_list_projects(client: TestClient) -> None:
    response = client.get("/api/v1/projects")
    assert response.status_code == 200
    assert response.json() == [{"project_id": "demo", "name": "Demo", "session_count": 3}]


def test_list_runs_paginates_with_cursor(client: TestClient) -> None:
    first = client.get("/api/v1/projects/demo/sessions", params={"limit": 2})
    assert first.status_code == 200
    body = first.json()
    assert [run["session_id"] for run in body["items"]] == ["run_3", "run_2"]
    assert body["items"][0]["title"] == "Run 3"
    assert body["next_cursor"]

    second = client.get(
        "/api/v1/projects/demo/sessions",
        params={"limit": 2, "cursor": body["next_cursor"]},
    )
    assert second.status_code == 200
    assert [run["session_id"] for run in second.json()["items"]] == ["run_1"]
    assert second.json()["next_cursor"] is None


def test_internal_runs_hidden_from_api_list(client: TestClient) -> None:
    body = client.get("/api/v1/projects/demo/sessions").json()
    assert all("__internal" not in run["session_id"] for run in body["items"])
    assert len(body["items"]) == 3


def test_derived_runs_hidden_from_api_list_by_default(client: TestClient) -> None:
    """Question batches and skill replays are derived from another run; listing
    them alongside top-level runs buries the analyses a user started."""
    body = client.get("/api/v1/projects/demo/sessions").json()
    assert [run["session_id"] for run in body["items"]] == ["run_3", "run_2", "run_1"]


def test_derived_runs_listed_on_request(client: TestClient) -> None:
    body = client.get("/api/v1/projects/demo/sessions", params={"include_derived": "true"}).json()
    listed = [run["session_id"] for run in body["items"]]
    assert "qsess_20260704_batch" in listed
    assert "ssess_20260704_replay" in listed
    # The internal marker is a separate, unconditional exclusion.
    assert all("__internal" not in session_id for session_id in listed)


def test_derived_run_stays_reachable_by_direct_id(client: TestClient) -> None:
    response = client.get("/api/v1/sessions/ssess_20260704_replay")
    assert response.status_code == 200
    assert response.json()["source_session_id"] == "run_3"


def test_get_run_detail(client: TestClient) -> None:
    response = client.get("/api/v1/sessions/run_2")
    assert response.status_code == 200
    body = response.json()
    assert body["project_id"] == "demo"
    assert body["status"] == "completed"
    assert body["code_version"] == "v1"
    assert "artifact_type_counts" in body
    # SessionDetail is metadata-only: no payload/report/frame fields exist.
    assert "payload" not in body
    assert "report_markdown" not in body


def test_unknown_project_and_run_use_error_envelope(client: TestClient) -> None:
    missing_project = client.get("/api/v1/projects/nope/sessions")
    assert missing_project.status_code == 404
    assert missing_project.json() == {
        "error": {"code": "project_not_found", "message": "Project not found: nope"}
    }
    missing_run = client.get("/api/v1/sessions/run_missing")
    assert missing_run.status_code == 404
    assert missing_run.json()["error"]["code"] == "session_not_found"


def test_bad_cursor_and_bad_limit_are_typed_errors(client: TestClient) -> None:
    bad_cursor = client.get("/api/v1/projects/demo/sessions", params={"cursor": "@@"})
    assert bad_cursor.status_code == 400
    assert bad_cursor.json()["error"]["code"] == "invalid_cursor"
    bad_limit = client.get("/api/v1/projects/demo/sessions", params={"limit": 0})
    assert bad_limit.status_code == 422
    assert bad_limit.json()["error"]["code"] == "validation_error"


def test_create_project_registers_a_usable_project(client: TestClient, workspace: Path) -> None:
    response = client.post("/api/v1/projects", json={"project_id": "new_shop", "name": "New Shop"})
    assert response.status_code == 201
    assert response.json() == {"project_id": "new_shop", "name": "New Shop", "session_count": 0}
    assert (workspace / "projects" / "new_shop").is_dir()
    listed = client.get("/api/v1/projects").json()
    assert {"project_id": "new_shop", "name": "New Shop", "session_count": 0} in listed
    # A registered project is immediately usable by the project-scoped routes.
    assert client.get("/api/v1/projects/new_shop/sessions").status_code == 200


def test_create_project_defaults_name_to_id(client: TestClient) -> None:
    response = client.post("/api/v1/projects", json={"project_id": "nameless"})
    assert response.status_code == 201
    assert response.json()["name"] == "nameless"


def test_create_project_allows_spaces_in_id(client: TestClient) -> None:
    response = client.post("/api/v1/projects", json={"project_id": "Brazilian E-Commerce"})
    assert response.status_code == 201
    assert response.json()["project_id"] == "Brazilian E-Commerce"


def test_fresh_app_provisions_hidden_unfiled_bucket(client: TestClient) -> None:
    response = client.get("/api/v1/projects/unfiled-sessions/sessions")

    assert response.status_code == 200
    assert response.json()["items"] == []
    assert all(
        project["project_id"] != "unfiled-sessions"
        for project in client.get("/api/v1/projects").json()
    )


def test_create_existing_project_is_idempotent(client: TestClient) -> None:
    """Same id twice returns the live project at 200 — never 409, and never a
    rename of the project the caller already has."""
    response = client.post("/api/v1/projects", json={"project_id": "demo", "name": "Renamed"})
    assert response.status_code == 200
    assert response.json() == {"project_id": "demo", "name": "Demo", "session_count": 3}


def test_project_order_is_persisted_and_new_projects_append(client: TestClient) -> None:
    assert client.post("/api/v1/projects", json={"project_id": "alpha"}).status_code == 201
    assert client.post("/api/v1/projects", json={"project_id": "beta"}).status_code == 201

    response = client.put(
        "/api/v1/projects/order",
        json={"project_ids": ["beta", "demo", "alpha"]},
    )

    assert response.status_code == 200
    assert [project["project_id"] for project in response.json()] == ["beta", "demo", "alpha"]
    assert [project["project_id"] for project in client.get("/api/v1/projects").json()] == [
        "beta",
        "demo",
        "alpha",
    ]

    assert client.post("/api/v1/projects", json={"project_id": "later"}).status_code == 201
    assert [project["project_id"] for project in client.get("/api/v1/projects").json()] == [
        "beta",
        "demo",
        "alpha",
        "later",
    ]


def test_project_order_rejects_stale_or_incomplete_lists(client: TestClient) -> None:
    response = client.put("/api/v1/projects/order", json={"project_ids": []})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "project_invalid"


def test_create_project_rejects_case_only_difference(client: TestClient) -> None:
    response = client.post("/api/v1/projects", json={"project_id": "DEMO"})
    assert response.status_code == 409
    body = response.json()["error"]
    assert body["code"] == "project_conflict"
    assert "'demo'" in body["message"]
    # The near-miss id must not have been registered.
    assert [p["project_id"] for p in client.get("/api/v1/projects").json()] == ["demo"]


@pytest.mark.parametrize(
    "project_id",
    ["", "   ", "..", ".", "../escape", "a/b", "a\\b", "-leading", "x" * 65, "weiß"],
)
def test_create_project_rejects_invalid_ids(client: TestClient, project_id: str) -> None:
    response = client.post("/api/v1/projects", json={"project_id": project_id})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "project_invalid"


def test_create_project_traversal_creates_nothing_outside_workspace(
    client: TestClient, workspace: Path
) -> None:
    before = sorted(p.name for p in (workspace / "projects").iterdir())
    assert client.post("/api/v1/projects", json={"project_id": "../evil"}).status_code == 422
    assert not (workspace.parent / "evil").exists()
    assert sorted(p.name for p in (workspace / "projects").iterdir()) == before


def test_api_modules_do_not_import_streamlit() -> None:
    src_root = Path(eda_platform.__file__).resolve().parents[1]
    code = (
        "import sys\n"
        "import eda_platform.api.main\n"
        "import eda_platform.application.services.session_service\n"
        "loaded = [m for m in sys.modules if m == 'streamlit' or m.startswith('streamlit.')]\n"
        "assert not loaded, f'streamlit imported: {loaded}'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(src_root), "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, result.stderr


def test_reorder_refuses_a_project_deleted_after_the_precondition_read(tmp_path: Path) -> None:
    """The service validates on its own connection, so the store re-checks
    inside the fence. Without that, the deleted project is silently skipped by
    the update and still returned from the caller's stale snapshot."""
    store = ArtifactStore(tmp_path)
    for project_id in ("alpha", "beta", "gamma"):
        store.ensure_project(project_id, name=project_id)
    store.reorder_projects(["gamma", "beta", "alpha"])

    with pytest.raises(ProjectOrderConflictError):
        store.reorder_projects(["gamma", "beta", "alpha", "deleted-in-between"])

    order = [row["project_id"] for row in store.project_index_rows()]
    assert order == ["gamma", "beta", "alpha"], "a refused reorder must not partially apply"


def _index_lineage(store: ArtifactStore, session_id: str) -> str | None:
    row = store.get_session_index_row(session_id)
    return None if row is None else row["source_session_id"]


def test_a_corrected_manifest_can_clear_a_stale_lineage(tmp_path: Path) -> None:
    """`coalesce` could only ever add a value, so a run re-pointed at a root
    manifest kept its old parent and stayed in the wrong family forever."""
    store = ArtifactStore(tmp_path)
    store.ensure_project("demo", name="Demo")
    store.start_session("demo", "child")
    manifest = store.session_dir("demo", "child") / "manifest.json"

    manifest.write_text(json.dumps({"source_session_id": "parent"}), encoding="utf-8")
    store.refresh_session_index("demo", "child")
    assert _index_lineage(store, "child") == "parent"

    manifest.write_text(json.dumps({"source_session_id": None}), encoding="utf-8")
    store.refresh_session_index("demo", "child")
    assert _index_lineage(store, "child") is None


def test_an_unreadable_manifest_still_cannot_wipe_a_known_lineage(tmp_path: Path) -> None:
    """The paired control: clearing must come from a manifest that parsed, not
    from a torn file or a transient IO error."""
    store = ArtifactStore(tmp_path)
    store.ensure_project("demo", name="Demo")
    store.start_session("demo", "child")
    manifest = store.session_dir("demo", "child") / "manifest.json"
    manifest.write_text(json.dumps({"source_session_id": "parent"}), encoding="utf-8")
    store.refresh_session_index("demo", "child")

    manifest.write_text("{ this is not json", encoding="utf-8")
    store.refresh_session_index("demo", "child")

    assert _index_lineage(store, "child") == "parent"
