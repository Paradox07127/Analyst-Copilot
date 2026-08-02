"""Custom chart builder endpoint: spec shape per chart type, column validation,
cleaning switches, the row cap, and the vega-expression boundary."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pytest
from data_operation_helpers import await_data_operation, operation_result_response
from fastapi.testclient import TestClient

from eda_platform.api.main import create_app
from eda_platform.application.chart_builder import (
    CUSTOM_CHART_GROUP_LIMIT,
    CUSTOM_CHART_ROW_LIMIT,
)
from eda_platform.application.dto import CustomChartRequest
from eda_platform.application.services.dataset_service import DatasetService
from eda_platform.application.services.insight_service import (
    CustomChartValidationError,
    InsightService,
)
from eda_platform.core.query import TrustedFileQueryEngine
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType

PROJECT = "demo"
RUN = "run_charts"
OTHER_RUN = "run_other"
DATASET = "ds_charts"
CSV_NAME = "orders.csv"

# amount carries one extreme value (100) so the IQR fence has something to cut:
# quantiles of [1, 2, 3, 4, 100] give bounds (-1, 7).
CSV_BODY = (
    "amount,label,day\n"
    "1,a,2024-01-01\n"
    "2,a,2024-01-02\n"
    "3,b,2024-01-03\n"
    "4,b,2024-01-04\n"
    "100,c,2024-01-05\n"
)
CSV_ROWS = 5


def _profile(session_id: str) -> Artifact:
    return Artifact(
        id=f"prof_{session_id}",
        type=ArtifactType.DATASET_PROFILE,
        project_id=PROJECT,
        session_id=session_id,
        payload={
            "dataset_id": DATASET,
            "name": CSV_NAME,
            "rows": CSV_ROWS,
            "columns": 3,
            "column_names": ["amount", "label", "day"],
            "dtypes": {"amount": "int64", "label": "object", "day": "object"},
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


def _post(client: TestClient, **overrides: Any) -> Any:
    body: dict[str, Any] = {
        "dataset_id": DATASET,
        "chart_type": "bar",
        "x_column": "label",
        "y_column": "amount",
    }
    body.update(overrides)
    started = client.post(
        f"/api/v1/sessions/{RUN}/charts/custom",
        json=body,
        headers={"Idempotency-Key": f"chart-{uuid.uuid4()}"},
    )
    if started.status_code != 202:
        return started
    return operation_result_response(
        *await_data_operation(client, started, "custom-chart-result")
    )


@pytest.mark.parametrize("chart_type", ["bar", "line", "point", "area"])
def test_spec_shape_per_chart_type(client: TestClient, chart_type: str) -> None:
    response = _post(client, chart_type=chart_type, aggregate="sum")
    assert response.status_code == 200, response.text
    body = response.json()
    spec = body["spec"]
    assert spec["mark"] == chart_type
    assert spec["encoding"]["x"] == {"field": "label", "type": "nominal"}
    assert spec["encoding"]["y"] == {
        "field": "amount",
        "type": "quantitative",
        "title": "sum(amount)",
    }
    assert "bin" not in spec["encoding"]["x"]
    assert spec["data"]["values"] == [
        {"label": "a", "amount": 3.0},
        {"label": "b", "amount": 7.0},
        {"label": "c", "amount": 100.0},
    ]
    assert body["row_count"] == CSV_ROWS
    assert body["truncated"] is False


def test_histogram_bins_x_and_counts_y(client: TestClient) -> None:
    response = _post(client, chart_type="histogram", x_column="amount", y_column=None)
    assert response.status_code == 200, response.text
    spec = response.json()["spec"]
    # A histogram renders as a binned bar, never as mark "histogram".
    assert spec["mark"] == "bar"
    assert spec["encoding"]["x"] == {
        "field": "bin_start",
        "type": "quantitative",
        "bin": {"binned": True},
        "title": "amount",
    }
    assert spec["encoding"]["x2"] == {"field": "bin_end"}
    assert spec["encoding"]["y"] == {"field": "count", "type": "quantitative"}
    assert sum(row["count"] for row in spec["data"]["values"]) == CSV_ROWS


def test_histogram_rejects_non_numeric_x(client: TestClient) -> None:
    response = _post(client, chart_type="histogram", x_column="label", y_column=None)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "custom_chart_invalid"


def test_row_count_y_becomes_count_aggregate(client: TestClient) -> None:
    response = _post(client, y_column=None)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["spec"]["encoding"]["y"] == {
        "field": "Row count",
        "type": "quantitative",
        "title": "Row count",
    }
    assert body["aggregate"] == "count"
    assert body["spec"]["data"]["values"] == [
        {"label": "a", "Row count": 2},
        {"label": "b", "Row count": 2},
        {"label": "c", "Row count": 1},
    ]


def test_color_column_encoding_and_temporal_inference(client: TestClient) -> None:
    response = _post(client, x_column="day", color_column="label", aggregate="mean")
    assert response.status_code == 200, response.text
    spec = response.json()["spec"]
    # ISO date strings parse for >=80% of the sampled head, so x is temporal.
    assert spec["encoding"]["x"] == {"field": "day", "type": "temporal"}
    assert spec["encoding"]["color"] == {"field": "label", "type": "nominal"}


def test_default_aggregate_for_numeric_and_non_numeric(client: TestClient) -> None:
    """Omitted aggregate: sum for a numeric Y, count for a non-numeric one."""
    numeric = _post(client, aggregate=None).json()
    assert numeric["aggregate"] == "sum"
    assert numeric["spec"]["encoding"]["y"] == {
        "field": "amount",
        "type": "quantitative",
        "title": "sum(amount)",
    }
    categorical = _post(client, y_column="label", aggregate=None).json()
    assert categorical["aggregate"] == "count"
    assert categorical["spec"]["encoding"]["y"] == {
        "field": "Aggregated value",
        "type": "quantitative",
        "title": "count(label)",
    }


def test_aggregate_none_keeps_raw_y_field(client: TestClient) -> None:
    spec = _post(client, aggregate="none").json()["spec"]
    assert spec["encoding"]["y"] == {"field": "amount", "type": "quantitative"}


def test_unknown_column_is_422(client: TestClient) -> None:
    for field in ("x_column", "y_column", "color_column"):
        response = _post(client, **{field: "no_such_column"})
        assert response.status_code == 422, field
        assert response.json()["error"]["code"] == "custom_chart_invalid"
        assert "no_such_column" in response.json()["error"]["message"]


def test_unknown_chart_type_is_422(client: TestClient) -> None:
    response = _post(client, chart_type="pie")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_drop_outliers_removes_iqr_outliers(client: TestClient) -> None:
    kept = _post(client, drop_outliers=True).json()
    assert kept["row_count"] == CSV_ROWS - 1
    assert 100 not in [row["amount"] for row in kept["spec"]["data"]["values"]]
    unfiltered = _post(client, drop_outliers=False).json()
    assert unfiltered["row_count"] == CSV_ROWS


def test_drop_outliers_rejected_for_row_count_y(client: TestClient) -> None:
    response = _post(client, y_column=None, drop_outliers=True)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "custom_chart_invalid"


def test_histogram_drop_outliers_fences_x_without_a_y_column(client: TestClient) -> None:
    """A histogram's fence is a statement about X; requiring a Y column made the
    builder ask for a column it then ignored."""
    response = _post(
        client,
        chart_type="histogram",
        x_column="amount",
        y_column=None,
        drop_outliers=True,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    values = body["spec"]["data"]["values"]
    assert sum(row["count"] for row in values) == CSV_ROWS - 1
    assert max(row["bin_end"] for row in values) < 100
    assert body["row_count"] == CSV_ROWS - 1


def test_numeric_aggregate_over_a_text_y_is_422(client: TestClient) -> None:
    """`to_numeric(errors="coerce")` used to turn a text Y into all-NaN and
    return a 200 chart of zeros."""
    response = _post(client, x_column="amount", y_column="label", aggregate="sum")
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "custom_chart_invalid"
    assert "label" in body["error"]["message"]


def test_numeric_aggregate_over_numeric_strings_still_works(store: ArtifactStore) -> None:
    """The gate mirrors what the aggregation can coerce, so a column stored as
    text but holding plain numbers still charts."""
    source = store.project_dir(PROJECT) / "uploads" / DATASET / "v1" / CSV_NAME
    # The stray "unknown" is what makes pandas read the column as object.
    source.write_text(
        "amount,label,day\n"
        "1000,a,2024-01-01\n"
        "2000,a,2024-01-02\n"
        "unknown,a,2024-01-03\n",
        encoding="utf-8",
    )
    client = TestClient(create_app(store.root))

    response = _post(client, x_column="label", y_column="amount", aggregate="sum")

    assert response.status_code == 200, response.text
    assert response.json()["spec"]["data"]["values"] == [{"label": "a", "amount": 3000.0}]


def test_group_cap_rejects_an_unrenderable_number_of_groups(store: ArtifactStore) -> None:
    """An unbounded group dict is the OOM path; a chart of that many marks is
    unreadable anyway, so it is refused rather than silently sampled."""
    source = store.project_dir(PROJECT) / "uploads" / DATASET / "v1" / CSV_NAME
    rows = ["amount,label,day"] + [
        f"{index},label_{index},2024-01-01" for index in range(CUSTOM_CHART_GROUP_LIMIT + 5)
    ]
    source.write_text("\n".join(rows) + "\n", encoding="utf-8")
    client = TestClient(create_app(store.root))

    response = _post(client, x_column="label", y_column=None, aggregate="count")

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "custom_chart_invalid"
    assert str(CUSTOM_CHART_GROUP_LIMIT) in body["error"]["message"]


def test_drop_missing_removes_null_rows(store: ArtifactStore) -> None:
    source = store.project_dir(PROJECT) / "uploads" / DATASET / "v1" / CSV_NAME
    source.write_text(CSV_BODY + ",d,2024-01-06\n", encoding="utf-8")
    client = TestClient(create_app(store.root))
    dropped = _post(client, drop_missing=True).json()
    assert dropped["row_count"] == CSV_ROWS
    kept = _post(client, drop_missing=False).json()
    assert kept["row_count"] == CSV_ROWS + 1


def test_empty_result_is_200_with_zero_rows(store: ArtifactStore) -> None:
    """No rows left after filtering is an empty chart, not an error: the client
    tells the empty state from row_count, not from a status code."""
    source = store.project_dir(PROJECT) / "uploads" / DATASET / "v1" / CSV_NAME
    source.write_text("amount,label,day\n,,\n", encoding="utf-8")
    client = TestClient(create_app(store.root))
    body = _post(client, drop_missing=True).json()
    assert body["row_count"] == 0
    assert body["spec"]["data"]["values"] == []


def test_row_limit_truncates_and_flags(store: ArtifactStore) -> None:
    over_cap = CUSTOM_CHART_ROW_LIMIT + 1
    source = store.project_dir(PROJECT) / "uploads" / DATASET / "v1" / CSV_NAME
    source.write_text(
        "amount,label,day\n"
        + "".join(f"{index},a,2024-01-01\n" for index in range(over_cap)),
        encoding="utf-8",
    )
    client = TestClient(create_app(store.root))
    body = _post(client, aggregate="none").json()
    assert body["source_row_count"] == over_cap
    assert body["row_count"] == CUSTOM_CHART_ROW_LIMIT
    assert body["row_limit"] == CUSTOM_CHART_ROW_LIMIT
    assert body["truncated"] is True
    assert len(body["spec"]["data"]["values"]) == CUSTOM_CHART_ROW_LIMIT


def test_reported_aggregate_is_the_one_the_spec_uses(client: TestClient) -> None:
    """A row-count Y counts whatever was requested; the response must not claim
    an aggregate the spec never applied."""
    body = _post(client, y_column=None, aggregate="mean").json()
    assert body["aggregate"] == "count"
    assert body["spec"]["encoding"]["y"] == {
        "field": "Row count",
        "type": "quantitative",
        "title": "Row count",
    }


def test_inline_data_is_capped_by_byte_budget(
    store: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Row count alone does not bound the payload — a few wide text cells can."""
    import eda_platform.application.services.insight_service as insight_service_module

    monkeypatch.setattr(insight_service_module, "MAX_INLINE_DATA_BYTES", 60)
    datasets = DatasetService(
        store,
        TrustedFileQueryEngine([store.root / "projects"]),
    )
    body = InsightService(store).build_custom_chart(
        RUN,
        CustomChartRequest(
            dataset_id=DATASET,
            chart_type="bar",
            x_column="label",
            y_column="amount",
            aggregate="none",
        ),
        datasets=datasets,
    ).model_dump()
    assert body["truncated"] is True
    assert body["series_truncated"] is False
    assert 0 < body["row_count"] < CSV_ROWS
    assert body["source_row_count"] == CSV_ROWS


def test_byte_capped_aggregate_reports_dropped_groups_not_dropped_rows(
    store: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An aggregated chart keeps row_count at the observation count, so a byte
    cap left the client saying "showing 5 of 5 rows"."""
    import eda_platform.application.services.insight_service as insight_service_module

    monkeypatch.setattr(insight_service_module, "MAX_INLINE_DATA_BYTES", 30)
    datasets = DatasetService(
        store,
        TrustedFileQueryEngine([store.root / "projects"]),
    )
    body = InsightService(store).build_custom_chart(
        RUN,
        CustomChartRequest(
            dataset_id=DATASET,
            chart_type="bar",
            x_column="label",
            y_column="amount",
            aggregate="sum",
        ),
        datasets=datasets,
    ).model_dump()
    assert body["truncated"] is True
    assert body["series_truncated"] is True
    assert body["row_count"] == body["source_row_count"] == CSV_ROWS
    assert len(body["spec"]["data"]["values"]) < 3  # three label groups exist


def test_vega_expression_in_spec_is_rejected(
    store: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The expression boundary is the last gate before the spec reaches the
    client's vega runtime: a builder that ever emitted one must not be served."""
    import eda_platform.application.services.insight_service as insight_service_module

    def _poisoned(**_: Any) -> dict[str, Any]:
        return {"mark": "bar", "encoding": {"x": {"expr": "alert(1)"}}}

    monkeypatch.setattr(insight_service_module, "custom_chart_spec", _poisoned)
    datasets = DatasetService(
        store,
        TrustedFileQueryEngine([store.root / "projects"]),
    )
    with pytest.raises(CustomChartValidationError):
        InsightService(store).build_custom_chart(
            RUN,
            CustomChartRequest(
                dataset_id=DATASET,
                chart_type="bar",
                x_column="label",
                y_column="amount",
            ),
            datasets=datasets,
        )


def test_unknown_dataset_404(client: TestClient) -> None:
    response = _post(client, dataset_id="ds_missing")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "dataset_not_found"


def test_unknown_run_404(client: TestClient) -> None:
    response = client.post(
        "/api/v1/sessions/nope/charts/custom",
        json={"dataset_id": DATASET, "chart_type": "bar", "x_column": "label"},
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "session_not_found"


def test_dataset_id_is_run_scoped(client: TestClient) -> None:
    """dataset_id is content-derived and repeats across runs; a run that never
    profiled it must 404 instead of borrowing another run's partition."""
    started = client.post(
        f"/api/v1/sessions/{OTHER_RUN}/charts/custom",
        json={"dataset_id": DATASET, "chart_type": "bar", "x_column": "label"},
        headers={"Idempotency-Key": "run-scoped-chart"},
    )
    response = operation_result_response(
        *await_data_operation(client, started, "custom-chart-result")
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "dataset_not_found"
