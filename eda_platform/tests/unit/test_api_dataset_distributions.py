"""Per-column mini distribution endpoint: numeric histograms, categorical
top-k, the sampling footnote, and the empty/404 paths."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from data_operation_helpers import await_data_operation, operation_result_response
from fastapi.testclient import TestClient

from eda_platform.api.main import create_app
from eda_platform.application.distribution_view import DIST_SAMPLE_CAP, DIST_TOP_K
from eda_platform.application.services.dataset_service import DatasetService
from eda_platform.core.query import TrustedFileQueryEngine
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType

PROJECT = "demo"
RUN = "run_dist"
OTHER_RUN = "run_other"
DATASET = "ds_dist"
CSV_NAME = "orders.csv"

_CATEGORIES = ["a", "a", "a", "b", "b", "c", "d", "e", "f"]
# Fractional amounts keep this column on the continuous path: a short run of
# whole numbers is a level set, and is summarised as value counts instead.
CSV_BODY = "amount,label\n" + "".join(
    f"{index + 0.5},{_CATEGORIES[index]}\n" for index in range(len(_CATEGORIES))
)


def _profile(session_id: str) -> Artifact:
    return Artifact(
        id=f"prof_{session_id}",
        type=ArtifactType.DATASET_PROFILE,
        project_id=PROJECT,
        session_id=session_id,
        payload={
            "dataset_id": DATASET,
            "name": CSV_NAME,
            "rows": len(_CATEGORIES),
            "columns": 2,
            "column_names": ["amount", "label"],
            "dtypes": {"amount": "int64", "label": "object"},
        },
    )


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    store = ArtifactStore(tmp_path)
    store.ensure_project(PROJECT, name="Demo")
    store.start_session(PROJECT, RUN)
    store.start_session(PROJECT, OTHER_RUN)
    store.save_artifact(_profile(RUN))
    source = store.project_dir(PROJECT) / "uploads" / DATASET / "v1" / CSV_NAME
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(CSV_BODY, encoding="utf-8")
    return store


@pytest.fixture
def client(store: ArtifactStore) -> TestClient:
    return TestClient(create_app(store.root))


def _get(client: TestClient, session_id: str, dataset_id: str):
    started = client.post(
        f"/api/v1/sessions/{session_id}/datasets/{dataset_id}/distributions",
        headers={"Idempotency-Key": f"distribution-{uuid.uuid4()}"},
    )
    if started.status_code != 202:
        return started
    return operation_result_response(
        *await_data_operation(client, started, "dataset-distributions-result")
    )


def test_distributions_numeric_and_categorical(client: TestClient) -> None:
    response = _get(client, RUN, DATASET)
    assert response.status_code == 200
    body = response.json()
    assert (body["dataset_id"], body["session_id"]) == (DATASET, RUN)
    assert body["row_count"] == len(_CATEGORIES)
    assert body["sampled"] is False
    assert body["sample_rows"] == len(_CATEGORIES)
    assert body["sample_cap"] == DIST_SAMPLE_CAP
    assert body["top_k"] == DIST_TOP_K

    columns = {column["name"]: column for column in body["columns"]}
    assert set(columns) == {"amount", "label"}

    numeric = columns["amount"]
    assert numeric["kind"] == "numeric"
    assert numeric["min"] == 0.5
    assert numeric["max"] == float(len(_CATEGORIES) - 1) + 0.5
    assert sum(numeric["counts"]) == len(_CATEGORIES)
    assert len(numeric["bin_edges"]) == len(numeric["counts"]) + 1
    assert numeric["missing_percent"] == 0.0
    assert numeric["top"] is None

    categorical = columns["label"]
    assert categorical["kind"] == "categorical"
    assert categorical["unique_count"] == 6
    # Top-5 by count, then everything else rolled into other_count.
    assert categorical["top"][0] == {"value": "a", "count": 3}
    assert len(categorical["top"]) == DIST_TOP_K
    assert categorical["other_count"] == 1
    assert (categorical["len_min"], categorical["len_max"]) == (1, 1)
    assert categorical["counts"] is None


def test_distributions_reports_sampling(
    store: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The response carries the footnote inputs: sampled + how many rows."""
    import eda_platform.application.services.dataset_service as dataset_service_module

    monkeypatch.setattr(dataset_service_module, "DIST_SAMPLE_CAP", 4)
    body = DatasetService(
        store,
        TrustedFileQueryEngine([store.root / "projects"]),
    ).get_distributions(DATASET, RUN).model_dump()
    assert body["row_count"] == len(_CATEGORIES)
    assert body["sampled"] is True
    assert body["sample_rows"] == 4
    assert body["sample_cap"] == 4


def test_distributions_unknown_dataset_404(client: TestClient) -> None:
    response = _get(client, RUN, "ds_missing")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "dataset_not_found"


def test_distributions_unknown_run_404(client: TestClient) -> None:
    response = _get(client, "nope", DATASET)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "session_not_found"


def test_distributions_dataset_id_is_run_scoped(client: TestClient) -> None:
    """dataset_id is content-derived and repeats across runs; a run that never
    profiled it must 404 instead of borrowing another run's partition."""
    response = _get(client, OTHER_RUN, DATASET)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "dataset_not_found"


def test_distributions_missing_source_404(store: ArtifactStore) -> None:
    source = store.project_dir(PROJECT) / "uploads" / DATASET / "v1" / CSV_NAME
    source.unlink()
    client = TestClient(create_app(store.root))
    response = _get(client, RUN, DATASET)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "dataset_source_missing"
