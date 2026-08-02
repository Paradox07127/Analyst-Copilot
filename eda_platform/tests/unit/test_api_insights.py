"""Insight endpoints (§10.2 P1): quality aggregation shape, profile shaping
edges (legacy payloads, empty runs), chart pagination/404s, spec-on-demand."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from eda_platform.api.main import create_app
from eda_platform.application.services.insight_service import (
    ChartNotFoundError,
    InsightService,
)
from eda_platform.core.ids import INTERNAL_SESSION_MARKER
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType

PROJECT = "demo"
RUN = "run_1"
BARE_RUN = "run_2"

PROFILE_PAYLOAD = {
    "dataset_id": "ds_1",
    "name": "orders.csv",
    "rows": 100,
    "columns": 2,
    "column_names": ["id", "amount"],
    "dtypes": {"id": "int64", "amount": "float64"},
    "missing_values": {"id": 0, "amount": 5},
    "missing_percent": {"id": 0.0, "amount": 5.0},
    "numeric_columns": ["id", "amount"],
    "categorical_columns": [],
    "columns_detail": [
        {
            "name": "id",
            "dtype": "int64",
            "semantic_type": "id",
            "missing_count": 0,
            "missing_percent": 0.0,
            "unique_count": 100,
            "unique_percent": 100.0,
            "sample_values": ["1", "2", "3", "4"],
        },
        {
            "name": "amount",
            "dtype": "float64",
            "semantic_type": "numeric",
            "missing_count": 5,
            "missing_percent": 5.0,
            "unique_count": 90,
            "unique_percent": 90.0,
            "sample_values": ["1.5"],
        },
    ],
    "semantic_type_counts": {"id": 1, "numeric": 1},
}

# Pre-columns_detail payload: exercises the legacy fallback shaping.
LEGACY_PROFILE_PAYLOAD = {
    "dataset_id": "ds_2",
    "name": "legacy.csv",
    "rows": 10,
    "columns": 2,
    "column_names": ["a", "b"],
    "dtypes": {"a": "int64"},
    "missing_values": {"a": 0},
    "missing_percent": {"a": 0.0},
    "numeric_columns": ["a"],
    "categorical_columns": [],
}

QUALITY_PAYLOAD = {
    "dataset_id": "ds_1",
    "issues": [
        {
            "severity": "critical",
            "code": "empty_column",
            "column": "amount",
            "message": "Column amount is empty.",
            "recommendation": "Drop it.",
        },
        {
            "severity": "warn",
            "code": "high_missing",
            "column": "amount",
            "message": "Column amount has 5.0% missing values.",
            "recommendation": "Review missingness.",
        },
        {
            "severity": "warn",
            "code": "high_missing",
            "column": "id",
            "message": "Column id has missing values.",
            "recommendation": "Review missingness.",
        },
    ],
}

CHART_PAYLOAD = {
    "dataset_id": "ds_1",
    "title": "Amount by id",
    "description": "Demo chart.",
    "mark": "bar",
    "encoding": {
        "x": {"field": "id", "type": "nominal"},
        "y": {"field": "amount", "type": "quantitative"},
    },
    "data": {"values": [{"id": "1", "amount": 2.5}]},
}


def _artifact(artifact_id: str, artifact_type: ArtifactType, payload: dict) -> Artifact:
    return Artifact(
        id=artifact_id,
        type=artifact_type,
        project_id=PROJECT,
        session_id=RUN,
        payload=payload,
        plain_language="Plain words." if artifact_type is ArtifactType.CHART_SPEC else None,
    )


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    store = ArtifactStore(tmp_path)
    store.ensure_project(PROJECT, name="Demo")
    store.start_session(PROJECT, RUN)
    store.start_session(PROJECT, BARE_RUN)
    store.save_artifact(_artifact("prof_1", ArtifactType.DATASET_PROFILE, PROFILE_PAYLOAD))
    store.save_artifact(
        _artifact("prof_2", ArtifactType.DATASET_PROFILE, LEGACY_PROFILE_PAYLOAD)
    )
    store.save_artifact(_artifact("quality_1", ArtifactType.QUALITY_ISSUE_SET, QUALITY_PAYLOAD))
    for index in range(3):
        store.save_artifact(
            _artifact(
                f"chart_{index}",
                ArtifactType.CHART_SPEC,
                {**CHART_PAYLOAD, "title": f"Chart {index}"},
            )
        )
    return store


@pytest.fixture
def client(store: ArtifactStore) -> TestClient:
    return TestClient(create_app(store.root))


def test_quality_aggregation_snapshot(client: TestClient) -> None:
    response = client.get(f"/api/v1/sessions/{RUN}/quality")
    assert response.status_code == 200
    assert response.json() == {
        "session_id": RUN,
        "critical": 1,
        "warn": 2,
        "info": 0,
        "datasets": [
            {
                "dataset_name": "orders.csv",
                "dataset_id": "ds_1",
                "critical": 1,
                "warn": 2,
                "info": 0,
            }
        ],
        "issues": [
            {
                "severity": "critical",
                "dataset_name": "orders.csv",
                "dataset_id": "ds_1",
                "code": "empty_column",
                "column": "amount",
                "message": "Column amount is empty.",
                "recommendation": "Drop it.",
            },
            {
                "severity": "warn",
                "dataset_name": "orders.csv",
                "dataset_id": "ds_1",
                "code": "high_missing",
                "column": "amount",
                "message": "Column amount has 5.0% missing values.",
                "recommendation": "Review missingness.",
            },
            {
                "severity": "warn",
                "dataset_name": "orders.csv",
                "dataset_id": "ds_1",
                "code": "high_missing",
                "column": "id",
                "message": "Column id has missing values.",
                "recommendation": "Review missingness.",
            },
        ],
    }


def test_quality_empty_run(client: TestClient) -> None:
    response = client.get(f"/api/v1/sessions/{BARE_RUN}/quality")
    assert response.status_code == 200
    body = response.json()
    assert (body["critical"], body["warn"], body["info"]) == (0, 0, 0)
    assert body["datasets"] == []
    assert body["issues"] == []


def test_quality_unknown_run_404(client: TestClient) -> None:
    response = client.get("/api/v1/sessions/missing/quality")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "session_not_found"


def test_profiles_shaping(client: TestClient) -> None:
    response = client.get(f"/api/v1/sessions/{RUN}/profiles")
    assert response.status_code == 200
    datasets = response.json()["datasets"]
    assert [dataset["name"] for dataset in datasets] == ["orders.csv", "legacy.csv"]

    modern = datasets[0]
    assert (modern["rows"], modern["columns"]) == (100, 2)
    assert modern["semantic_type_counts"] == {"id": 1, "numeric": 1}
    assert modern["fields"][0] == {
        "column": "id",
        "dtype": "int64",
        "semantic_type": "id",
        "missing_percent": 0.0,
        "unique_percent": 100.0,
        "sample_values": "1, 2, 3",
    }

    # Legacy payload without columns_detail: derived semantic types, no uniques.
    legacy = datasets[1]
    assert legacy["semantic_type_counts"] == {"numeric": 1, "unknown": 1}
    legacy_fields = {field["column"]: field for field in legacy["fields"]}
    assert legacy_fields["a"]["semantic_type"] == "numeric"
    assert legacy_fields["b"]["semantic_type"] == "unknown"
    assert legacy_fields["b"]["unique_percent"] is None


def test_profiles_empty_run(client: TestClient) -> None:
    response = client.get(f"/api/v1/sessions/{BARE_RUN}/profiles")
    assert response.status_code == 200
    assert response.json() == {"session_id": BARE_RUN, "datasets": []}


def test_list_charts_paginates_without_spec(client: TestClient) -> None:
    first = client.get(f"/api/v1/sessions/{RUN}/charts", params={"limit": 2})
    assert first.status_code == 200
    body = first.json()
    assert [item["title"] for item in body["items"]] == ["Chart 0", "Chart 1"]
    assert body["items"][0] == {
        "artifact_id": "chart_0",
        "title": "Chart 0",
        "dataset_id": "ds_1",
        "dataset_name": "orders.csv",
        "mark": "bar",
        "fields": ["id", "amount"],
        "description": "Demo chart.",
    }
    assert body["next_cursor"]

    second = client.get(
        f"/api/v1/sessions/{RUN}/charts",
        params={"limit": 2, "cursor": body["next_cursor"]},
    )
    assert [item["artifact_id"] for item in second.json()["items"]] == ["chart_2"]
    assert second.json()["next_cursor"] is None


def test_list_charts_bad_cursor_400(client: TestClient) -> None:
    response = client.get(f"/api/v1/sessions/{RUN}/charts", params={"cursor": "%%%"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_cursor"


def test_get_chart_returns_vegalite_spec(client: TestClient) -> None:
    response = client.get(f"/api/v1/sessions/{RUN}/charts/chart_1")
    assert response.status_code == 200
    body = response.json()
    assert body["artifact_id"] == "chart_1"
    assert body["dataset_name"] == "orders.csv"
    assert body["plain_language"] == "Plain words."
    spec = body["spec"]
    assert spec["$schema"].startswith("https://vega.github.io/schema/vega-lite/")
    assert spec["mark"] == "bar"
    assert spec["data"] == {"values": [{"id": "1", "amount": 2.5}]}


def test_get_chart_is_bound_to_run_partition(store: ArtifactStore) -> None:
    other_run = "run_other"
    store.start_session(PROJECT, other_run)
    store.save_artifact(
        Artifact(
            id="chart_1",
            type=ArtifactType.CHART_SPEC,
            project_id=PROJECT,
            session_id=other_run,
            payload={**CHART_PAYLOAD, "title": "Other chart"},
        )
    )
    response = TestClient(create_app(store.root)).get(f"/api/v1/sessions/{RUN}/charts/chart_1")
    assert response.status_code == 200
    assert response.json()["session_id"] == RUN
    assert response.json()["title"] == "Chart 1"


def test_get_chart_missing_404(client: TestClient) -> None:
    response = client.get(f"/api/v1/sessions/{RUN}/charts/nope")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "chart_not_found"


def test_get_chart_wrong_type_404(client: TestClient) -> None:
    """A non-chart artifact id must not serve through the chart endpoint."""
    response = client.get(f"/api/v1/sessions/{RUN}/charts/prof_1")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "chart_not_found"


def test_get_chart_external_data_404_but_listed(store: ArtifactStore) -> None:
    """A spec with non-inline data (url/format/name) must not serve as a detail;
    the metadata-only listing still shows it."""
    store.save_artifact(
        _artifact(
            "chart_url",
            ArtifactType.CHART_SPEC,
            {**CHART_PAYLOAD, "data": {"url": "https://evil.example/data.json"}},
        )
    )
    client = TestClient(create_app(store.root))
    detail = client.get(f"/api/v1/sessions/{RUN}/charts/chart_url")
    assert detail.status_code == 404
    assert detail.json()["error"]["code"] == "chart_not_found"
    listing = client.get(f"/api/v1/sessions/{RUN}/charts")
    assert "chart_url" in [item["artifact_id"] for item in listing.json()["items"]]


def test_get_chart_internal_run_404(store: ArtifactStore) -> None:
    store.start_session(PROJECT, f"{RUN}{INTERNAL_SESSION_MARKER}_probe")
    store.save_artifact(
        Artifact(
            id="chart_internal",
            type=ArtifactType.CHART_SPEC,
            project_id=PROJECT,
            session_id=f"{RUN}{INTERNAL_SESSION_MARKER}_probe",
            payload=CHART_PAYLOAD,
        )
    )
    client = TestClient(create_app(store.root))
    response = client.get(f"/api/v1/sessions/{RUN}/charts/chart_internal")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "chart_not_found"


def test_quality_same_named_datasets_get_distinct_cards(store: ArtifactStore) -> None:
    run = "run_dup"
    store.start_session(PROJECT, run)
    for dataset_id, issue_count in (("ds_a", 1), ("ds_b", 2)):
        store.save_artifact(
            Artifact(
                id=f"prof_{dataset_id}",
                type=ArtifactType.DATASET_PROFILE,
                project_id=PROJECT,
                session_id=run,
                payload={**PROFILE_PAYLOAD, "dataset_id": dataset_id, "name": "dup.csv"},
            )
        )
        store.save_artifact(
            Artifact(
                id=f"quality_{dataset_id}",
                type=ArtifactType.QUALITY_ISSUE_SET,
                project_id=PROJECT,
                session_id=run,
                payload={
                    "dataset_id": dataset_id,
                    "issues": QUALITY_PAYLOAD["issues"][:issue_count],
                },
            )
        )
    view = InsightService(store).get_quality(run)
    cards = {card.dataset_id: card for card in view.datasets}
    assert set(cards) == {"ds_a", "ds_b"}
    assert cards["ds_a"].dataset_name == "dup.csv (ds_a)"
    assert cards["ds_b"].dataset_name == "dup.csv (ds_b)"
    assert (cards["ds_a"].critical, cards["ds_a"].warn) == (1, 0)
    assert (cards["ds_b"].critical, cards["ds_b"].warn) == (1, 1)
    # Issue rows carry the disambiguated name so the dataset filter still works.
    assert {issue.dataset_name for issue in view.issues} == {
        "dup.csv (ds_a)",
        "dup.csv (ds_b)",
    }
    # And the raw id, so the client filter can compare ids instead of names.
    assert {issue.dataset_id for issue in view.issues} == {"ds_a", "ds_b"}


def test_get_chart_expression_spec_404(store: ArtifactStore) -> None:
    """A spec carrying vega expression constructs (expr/labelExpr/signal or a
    string condition test) must not serve as a detail."""
    store.save_artifact(
        _artifact(
            "chart_expr",
            ArtifactType.CHART_SPEC,
            {
                **CHART_PAYLOAD,
                "encoding": {
                    "x": {
                        "field": "id",
                        "type": "nominal",
                        "axis": {"labelExpr": "datum.label + '!'"},
                    },
                    "y": {"field": "amount", "type": "quantitative"},
                },
            },
        )
    )
    client = TestClient(create_app(store.root))
    response = client.get(f"/api/v1/sessions/{RUN}/charts/chart_expr")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "chart_not_found"


def test_get_chart_condition_test_string_404(store: ArtifactStore) -> None:
    store.save_artifact(
        _artifact(
            "chart_cond",
            ArtifactType.CHART_SPEC,
            {
                **CHART_PAYLOAD,
                "encoding": {
                    "color": {
                        "condition": {"test": "datum.amount > 1", "value": "red"},
                        "value": "blue",
                    },
                },
            },
        )
    )
    client = TestClient(create_app(store.root))
    response = client.get(f"/api/v1/sessions/{RUN}/charts/chart_cond")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "chart_not_found"


def test_expression_gate_covers_transform_constructs() -> None:
    """`ChartSpec.to_vegalite` drops a top-level transform, so this shape cannot
    reach get_chart today — but the same gate guards the custom-chart builder
    and the client mirror, where arbitrary spec dicts do flow."""
    from eda_platform.application.services.insight_service import (
        _contains_vega_expression,
    )

    assert _contains_vega_expression(
        {"transform": [{"calculate": "datum.amount * 2", "as": "doubled"}]}
    )
    assert _contains_vega_expression({"transform": [{"filter": "datum.amount > 1"}]})
    # A field predicate is data, not an expression, and must stay allowed.
    assert not _contains_vega_expression(
        {"transform": [{"filter": {"field": "amount", "gt": 1}}]}
    )


def test_get_chart_swapped_envelope_404(store: ArtifactStore) -> None:
    """The file behind the index row must be the artifact the row points at."""
    other_path = store.artifact_path(PROJECT, RUN, "chart_2")
    with closing(sqlite3.connect(store.db_path)) as conn, conn:
        conn.execute(
            "update artifacts set path = ? where artifact_id = ?",
            (str(other_path), "chart_1"),
        )
    client = TestClient(create_app(store.root))
    response = client.get(f"/api/v1/sessions/{RUN}/charts/chart_1")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "chart_not_found"
    # The listing skips the swapped row but still serves the intact ones.
    listing = client.get(f"/api/v1/sessions/{RUN}/charts")
    assert [item["artifact_id"] for item in listing.json()["items"]] == ["chart_0", "chart_2"]


def test_get_chart_oversized_payload_404(
    store: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    import eda_platform.application.services.insight_service as insight_service_module

    monkeypatch.setattr(insight_service_module, "MAX_ARTIFACT_PAYLOAD_BYTES", 16)
    service = InsightService(store)
    with pytest.raises(ChartNotFoundError):
        service.get_chart(RUN, "chart_1")
    assert service.list_charts(RUN).items == []


def test_chart_workspace_paths_relativized(store: ArtifactStore) -> None:
    absolute = str((store.root / "projects" / PROJECT / "uploads" / "a.csv").resolve())
    store.save_artifact(
        _artifact(
            "chart_path",
            ArtifactType.CHART_SPEC,
            {**CHART_PAYLOAD, "description": absolute},
        )
    )
    client = TestClient(create_app(store.root))
    body = client.get(f"/api/v1/sessions/{RUN}/charts/chart_path").json()
    assert body["description"] == f"projects/{PROJECT}/uploads/a.csv"
    assert body["spec"]["description"] == f"projects/{PROJECT}/uploads/a.csv"


def test_charts_cursor_bound_to_run_400(client: TestClient) -> None:
    first = client.get(f"/api/v1/sessions/{RUN}/charts", params={"limit": 2})
    cursor = first.json()["next_cursor"]
    assert cursor
    replay = client.get(
        f"/api/v1/sessions/{BARE_RUN}/charts", params={"limit": 2, "cursor": cursor}
    )
    assert replay.status_code == 400
    assert replay.json()["error"]["code"] == "invalid_cursor"


def test_list_charts_first_rows_invalid_still_fills_page(store: ArtifactStore) -> None:
    """Leading invalid rows must not surface as an empty page with a cursor."""
    run = "run_invalid_head"
    store.start_session(PROJECT, run)
    for index in range(3):
        store.save_artifact(
            Artifact(
                id=f"bad_{index}",
                type=ArtifactType.CHART_SPEC,
                project_id=PROJECT,
                session_id=run,
                payload={"dataset_id": "ds_1"},
            )
        )
    store.save_artifact(
        Artifact(
            id="good_chart",
            type=ArtifactType.CHART_SPEC,
            project_id=PROJECT,
            session_id=run,
            payload=CHART_PAYLOAD,
        )
    )
    page = InsightService(store).list_charts(run, limit=2)
    assert [item.artifact_id for item in page.items] == ["good_chart"]
    assert page.next_cursor is None


def test_get_chart_invalid_payload_404(store: ArtifactStore) -> None:
    store.save_artifact(
        _artifact("chart_bad", ArtifactType.CHART_SPEC, {"dataset_id": "ds_1"})
    )
    service = InsightService(store)
    with pytest.raises(ChartNotFoundError):
        service.get_chart(RUN, "chart_bad")


def test_list_charts_skips_invalid_payloads(store: ArtifactStore) -> None:
    store.save_artifact(
        _artifact("chart_bad", ArtifactType.CHART_SPEC, {"dataset_id": "ds_1"})
    )
    page = InsightService(store).list_charts(RUN)
    assert [item.artifact_id for item in page.items] == ["chart_0", "chart_1", "chart_2"]
