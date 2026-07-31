"""Cleaning transparency read endpoints: the four-table cleaning log and the
raw before-cleaning view (profiles / charts / previews)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from eda_platform.api.main import create_app
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType

PROJECT = "demo"
RUN = "run_clean"
BARE_RUN = "run_bare"

RAW_PROFILE_PAYLOAD = {
    "dataset_id": "ds_raw",
    "name": "orders.csv",
    "rows": 12,
    "columns": 2,
    "column_names": ["id", "amount"],
    "dtypes": {"id": "int64", "amount": "float64"},
    "missing_values": {"id": 0, "amount": 2},
    "missing_percent": {"id": 0.0, "amount": 16.7},
    "numeric_columns": ["id", "amount"],
    "categorical_columns": [],
    "columns_detail": [
        {
            "name": "id",
            "dtype": "int64",
            "semantic_type": "id",
            "missing_count": 0,
            "missing_percent": 0.0,
            "unique_count": 12,
            "unique_percent": 100.0,
            "sample_values": ["1", "2", "3"],
        },
    ],
    "semantic_type_counts": {"id": 1},
}

RAW_CHART_PAYLOAD = {
    "dataset_id": "ds_raw",
    "title": "Raw amount by id",
    "description": "Before cleaning.",
    "mark": "bar",
    "encoding": {
        "x": {"field": "id", "type": "nominal"},
        "y": {"field": "amount", "type": "quantitative"},
    },
    "data": {"values": [{"id": "1", "amount": 2.5}]},
}

RAW_PREVIEW_PAYLOAD = {
    "dataset_id": "ds_raw",
    "name": "orders.csv",
    "rows": 12,
    "columns": 2,
    "column_names": ["id", "amount"],
    "rows_preview": [{"id": 1, "amount": 2.5}, {"id": 2, "amount": None}],
    "preview_row_limit": 100,
}

RECIPE_PAYLOAD = {
    "dataset_id": "ds_raw",
    "source_version": 1,
    "recipe_id": "recipe_demo",
    "created_by": "precleaning",
    "transforms": [
        {
            "transform_id": "drop_col_1",
            "type": "drop_column",
            "target_column": "notes",
            "params": {"reason": "high missingness"},
            "expected_impact_rows": 0,
            "description": "Drop notes (98% missing).",
        },
        {
            "transform_id": "drop_missing_1",
            "type": "drop_missing_rows",
            "params": {"method": "listwise"},
            "expected_impact_rows": 3,
            "description": "Drop rows with missing values.",
        },
    ],
    "guardrails": [
        {
            "code": "missing_row_drop_below_min_rows",
            "message": "Dropping missing rows would leave too few rows.",
            "params": {"min_rows": 10, "would_remain": 4},
        }
    ],
    "lineage": {
        "source_dataset_id": "ds_raw",
        "source_name": "orders.csv",
        "rows_before": 12,
        "rows_after": 9,
        "columns_before": 3,
        "columns_after": 2,
    },
}


def _artifact(artifact_id: str, artifact_type: ArtifactType, payload: dict) -> Artifact:
    return Artifact(
        id=artifact_id,
        type=artifact_type,
        project_id=PROJECT,
        session_id=RUN,
        payload=payload,
        plain_language="Raw chart words." if artifact_type is ArtifactType.RAW_CHART_SPEC else None,
    )


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    store = ArtifactStore(tmp_path)
    store.ensure_project(PROJECT, name="Demo")
    store.start_session(PROJECT, RUN)
    store.start_session(PROJECT, BARE_RUN)
    store.save_artifact(
        _artifact("raw_prof_1", ArtifactType.RAW_DATASET_PROFILE, RAW_PROFILE_PAYLOAD)
    )
    store.save_artifact(_artifact("raw_chart_1", ArtifactType.RAW_CHART_SPEC, RAW_CHART_PAYLOAD))
    store.save_artifact(
        _artifact("raw_prev_1", ArtifactType.RAW_DATA_PREVIEW, RAW_PREVIEW_PAYLOAD)
    )
    store.save_artifact(_artifact("recipe_1", ArtifactType.CLEANING_RECIPE, RECIPE_PAYLOAD))
    return store


@pytest.fixture
def client(store: ArtifactStore) -> TestClient:
    return TestClient(create_app(store.root))


# -- cleaning log ---------------------------------------------------------


def test_cleaning_log_four_tables(client: TestClient) -> None:
    response = client.get(f"/api/v1/sessions/{RUN}/cleaning/log")
    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == RUN
    assert body["recipe_count"] == 1

    assert body["summary"] == [
        {
            "dataset": "orders.csv",
            "recipe_id": "recipe_demo",
            "rows_before": 12,
            "rows_after": 9,
            "rows_removed": 3,
            "columns_before": 3,
            "columns_after": 2,
            "columns_removed": 1,
            "delete_steps": 2,
            "protection_triggers": 1,
            "requires_approval": True,
        }
    ]
    assert body["deleted_data"] == [
        {
            "dataset": "orders.csv",
            "operation": "drop_column",
            "column": "notes",
            "rows_deleted": 0,
            "columns_deleted": 1,
            "reason": "high missingness",
            "details": "Drop notes (98% missing).",
        },
        {
            "dataset": "orders.csv",
            "operation": "drop_missing_rows",
            "column": "",
            "rows_deleted": 3,
            "columns_deleted": 0,
            "reason": "listwise",
            "details": "Drop rows with missing values.",
        },
    ]
    assert body["protection_triggers"] == [
        {
            "dataset": "orders.csv",
            "code": "missing_row_drop_below_min_rows",
            "reason": "Dropping missing rows would leave too few rows.",
            "thresholds": "min_rows=10, would_remain=4",
        }
    ]
    # drop_column + drop_missing_rows + a fired guardrail -> three suggestions.
    assert [row["suggestion"] for row in body["suggestions"]] == [
        "Review dropped high-missing columns before modeling; some may be useful "
        "as missingness flags.",
        "Check whether removed missing rows are random or concentrated in a key group.",
        "When protection triggers, consider a less aggressive threshold, "
        "imputation, or manual review.",
    ]


def test_cleaning_log_empty_run(client: TestClient) -> None:
    """A run without a CleaningRecipe answers 200 with recipe_count 0."""
    response = client.get(f"/api/v1/sessions/{BARE_RUN}/cleaning/log")
    assert response.status_code == 200
    body = response.json()
    assert body["recipe_count"] == 0
    assert body["summary"] == []
    assert body["deleted_data"] == []
    assert body["protection_triggers"] == []
    assert body["suggestions"] == []


def test_cleaning_log_unknown_run_404(client: TestClient) -> None:
    response = client.get("/api/v1/sessions/nope/cleaning/log")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "session_not_found"


# -- raw before-cleaning view ---------------------------------------------


def test_cleaning_raw_view(client: TestClient) -> None:
    response = client.get(f"/api/v1/sessions/{RUN}/cleaning/raw")
    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == RUN
    assert body["precleaning_recorded"] is True

    profile = body["profiles"][0]
    assert (profile["dataset_id"], profile["name"]) == ("ds_raw", "orders.csv")
    assert (profile["rows"], profile["columns"]) == (12, 2)
    assert profile["semantic_type_counts"] == {"id": 1}
    assert profile["fields"][0]["column"] == "id"

    chart = body["charts"][0]
    assert chart["artifact_id"] == "raw_chart_1"
    assert chart["dataset_name"] == "orders.csv"
    assert chart["plain_language"] == "Raw chart words."
    assert chart["spec"]["mark"] == "bar"
    assert chart["spec"]["$schema"].startswith("https://vega.github.io/schema/vega-lite/")

    preview = body["previews"][0]
    assert preview["artifact_id"] == "raw_prev_1"
    assert preview["column_names"] == ["id", "amount"]
    assert preview["rows_preview"] == [
        {"id": 1, "amount": 2.5},
        {"id": 2, "amount": None},
    ]
    assert (preview["rows"], preview["columns"]) == (12, 2)


def test_cleaning_raw_distinguishes_no_precleaning_from_empty(client: TestClient) -> None:
    """precleaning_recorded is the flag that separates "the run never recorded a
    raw snapshot" from "it did, but this category is empty"."""
    response = client.get(f"/api/v1/sessions/{BARE_RUN}/cleaning/raw")
    assert response.status_code == 200
    body = response.json()
    assert body["precleaning_recorded"] is False
    assert (body["profiles"], body["charts"], body["previews"]) == ([], [], [])


def test_cleaning_raw_recipe_only_run_is_recorded(store: ArtifactStore) -> None:
    """A recipe with no raw snapshots still counts as pre-cleaning having run."""
    run = "run_recipe_only"
    store.start_session(PROJECT, run)
    store.save_artifact(
        Artifact(
            id="recipe_only",
            type=ArtifactType.CLEANING_RECIPE,
            project_id=PROJECT,
            session_id=run,
            payload=RECIPE_PAYLOAD,
        )
    )
    client = TestClient(create_app(store.root))
    body = client.get(f"/api/v1/sessions/{run}/cleaning/raw").json()
    assert body["precleaning_recorded"] is True
    assert body["profiles"] == []


def test_cleaning_raw_unknown_run_404(client: TestClient) -> None:
    response = client.get("/api/v1/sessions/nope/cleaning/raw")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "session_not_found"


def test_cleaning_raw_skips_expression_chart(store: ArtifactStore) -> None:
    """Same boundary as the chart detail endpoint: a spec carrying a vega
    expression is not served to the client renderer."""
    store.save_artifact(
        _artifact(
            "raw_chart_expr",
            ArtifactType.RAW_CHART_SPEC,
            {
                **RAW_CHART_PAYLOAD,
                "encoding": {
                    "x": {
                        "field": "id",
                        "type": "nominal",
                        "axis": {"labelExpr": "datum.label + '!'"},
                    }
                },
            },
        )
    )
    client = TestClient(create_app(store.root))
    body = client.get(f"/api/v1/sessions/{RUN}/cleaning/raw").json()
    assert [chart["artifact_id"] for chart in body["charts"]] == ["raw_chart_1"]


def test_cleaning_raw_relativizes_workspace_paths(store: ArtifactStore) -> None:
    absolute = str((store.root / "projects" / PROJECT / "uploads" / "a.csv").resolve())
    store.save_artifact(
        _artifact(
            "raw_prev_abs",
            ArtifactType.RAW_DATA_PREVIEW,
            {**RAW_PREVIEW_PAYLOAD, "name": absolute},
        )
    )
    client = TestClient(create_app(store.root))
    body = client.get(f"/api/v1/sessions/{RUN}/cleaning/raw").json()
    names = [preview["name"] for preview in body["previews"]]
    assert f"projects/{PROJECT}/uploads/a.csv" in names
    assert not any(name.startswith("/") for name in names)
