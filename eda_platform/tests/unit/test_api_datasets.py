"""Dataset + upload endpoints: happy paths, error envelope, validation."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eda_platform.api.main import create_app
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.sessions import SessionManifest

PROJECT = "demo"
RUN = "run_1"
DATASET = "ds_api00001"
CSV_NAME = "orders.csv"
CSV_BODY = "id,amount\n" + "".join(f"{i},{i}.5\n" for i in range(1, 8))


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    store = ArtifactStore(tmp_path)
    store.ensure_project(PROJECT, name="Demo")
    store.start_session(PROJECT, RUN)
    store.write_manifest(
        SessionManifest(
            session_id=RUN,
            project_id=PROJECT,
            input_hashes={CSV_NAME: "hash12345678"},
            code_version="v1",
        )
    )
    store.save_artifact(
        Artifact(
            id="prof_api",
            type=ArtifactType.DATASET_PROFILE,
            project_id=PROJECT,
            session_id=RUN,
            payload={
                "dataset_id": DATASET,
                "name": CSV_NAME,
                "rows": 7,
                "columns": 2,
                "column_names": ["id", "amount"],
                "dtypes": {"id": "int64", "amount": "float64"},
            },
        )
    )
    source = store.project_dir(PROJECT) / "uploads" / DATASET / "v1" / CSV_NAME
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(CSV_BODY)
    return tmp_path


@pytest.fixture
def client(workspace: Path) -> TestClient:
    return TestClient(create_app(workspace))


def test_list_run_datasets(client: TestClient) -> None:
    response = client.get(f"/api/v1/sessions/{RUN}/datasets")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    handle = body[0]
    assert handle["dataset_id"] == DATASET
    assert handle["display_name"] == CSV_NAME
    assert handle["row_count"] == 7
    assert handle["content_hash"] == "hash12345678"
    # Serialized under the public name "schema", relative URI only.
    assert handle["schema"] == [
        {"name": "id", "dtype": "int64"},
        {"name": "amount", "dtype": "float64"},
    ]
    assert not handle["original_uri"].startswith("/")


def test_list_datasets_unknown_run_404(client: TestClient) -> None:
    response = client.get("/api/v1/sessions/nope/datasets")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "session_not_found"


def test_get_schema(client: TestClient) -> None:
    response = client.get(f"/api/v1/sessions/{RUN}/datasets/{DATASET}/schema")
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "profile"
    assert [column["name"] for column in body["columns"]] == ["id", "amount"]


def test_get_schema_unknown_dataset_404(client: TestClient) -> None:
    response = client.get(f"/api/v1/sessions/{RUN}/datasets/ds_missing/schema")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "dataset_not_found"


def test_get_preview_pages(client: TestClient) -> None:
    response = client.get(
        f"/api/v1/sessions/{RUN}/datasets/{DATASET}/preview",
        params={"limit": 3, "offset": 6},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["columns"] == ["id", "amount"]
    assert body["rows"] == [[7, 7.5]]
    assert body["has_more"] is False
    assert body["source_format"] == "csv"


def test_preview_limit_validation_422(client: TestClient) -> None:
    response = client.get(
        f"/api/v1/sessions/{RUN}/datasets/{DATASET}/preview", params={"limit": 5000}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_upload_roundtrip_returns_completed_status(client: TestClient) -> None:
    response = client.post(
        f"/api/v1/projects/{PROJECT}/uploads",
        files={"file": ("new.csv", b"a,b\n1,2\n", "text/csv")},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "completed"
    assert body["dataset"]["display_name"] == "new.csv"
    assert body["dataset"]["schema"] == [
        {"name": "a", "dtype": "BIGINT"},
        {"name": "b", "dtype": "BIGINT"},
    ]


def test_upload_rejects_non_csv_422(client: TestClient) -> None:
    response = client.post(
        f"/api/v1/projects/{PROJECT}/uploads",
        files={"file": ("data.txt", b"x", "text/plain")},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "upload_invalid"


def test_upload_unknown_project_404(client: TestClient) -> None:
    response = client.post(
        "/api/v1/projects/nope/uploads",
        files={"file": ("a.csv", b"a\n1\n", "text/csv")},
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "project_not_found"


def test_upload_status_endpoint_is_not_exposed(client: TestClient) -> None:
    assert "/api/v1/uploads/{upload_id}" not in cast(FastAPI, client.app).openapi()["paths"]
    response = client.get("/api/v1/uploads/up_missing")
    assert response.status_code == 404
