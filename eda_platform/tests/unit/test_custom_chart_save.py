"""Saving a custom chart: `save_to_charts` persists the built spec as a
CHART_SPEC artifact in the source session, so it survives navigation and shows
up in the session's Charts listing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from eda_platform.application.dto import CustomChartRequest
from eda_platform.application.services.dataset_service import DatasetService
from eda_platform.application.services.insight_service import InsightService
from eda_platform.core.query import TrustedFileQueryEngine
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType

PROJECT = "demo"
RUN = "run_chart_save"
DATASET = "ds_save"
CSV_NAME = "orders.csv"
CSV_BODY = "amount,label\n1,a\n2,a\n3,b\n4,b\n"


def _profile() -> Artifact:
    return Artifact(
        id=f"prof_{RUN}",
        type=ArtifactType.DATASET_PROFILE,
        project_id=PROJECT,
        session_id=RUN,
        payload={
            "dataset_id": DATASET,
            "name": CSV_NAME,
            "rows": 4,
            "columns": 2,
            "column_names": ["amount", "label"],
            "dtypes": {"amount": "int64", "label": "object"},
            "missing_values": {"amount": 0, "label": 0},
            "missing_percent": {"amount": 0.0, "label": 0.0},
            "numeric_columns": ["amount"],
            "categorical_columns": ["label"],
        },
    )


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    store = ArtifactStore(tmp_path)
    store.ensure_project(PROJECT, name="Demo")
    store.start_session(PROJECT, RUN)
    store.save_artifact(_profile())
    source = store.project_dir(PROJECT) / "uploads" / DATASET / "v1" / CSV_NAME
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(CSV_BODY, encoding="utf-8")
    return store


def _build(store: ArtifactStore, **overrides: Any) -> Any:
    body: dict[str, Any] = {
        "dataset_id": DATASET,
        "chart_type": "bar",
        "x_column": "label",
        "y_column": "amount",
        "aggregate": "sum",
    }
    body.update(overrides)
    datasets = DatasetService(
        store,
        TrustedFileQueryEngine([store.root / "projects"]),
    )
    return InsightService(store).build_custom_chart(
        RUN,
        CustomChartRequest.model_validate(body),
        datasets=datasets,
    )


def test_save_to_charts_persists_a_listed_servable_chart(store: ArtifactStore) -> None:
    view = _build(store, save_to_charts=True)
    assert view.saved_chart_id

    service = InsightService(store)
    page = service.list_charts(RUN)
    assert [item.artifact_id for item in page.items] == [view.saved_chart_id]
    assert page.items[0].dataset_id == DATASET
    assert "amount" in page.items[0].title

    served = service.get_chart(RUN, view.saved_chart_id)
    assert served.spec["mark"] == "bar"
    assert served.spec["data"]["values"] == view.spec["data"]["values"]


def test_saving_the_same_chart_twice_stores_one_artifact(store: ArtifactStore) -> None:
    first = _build(store, save_to_charts=True)
    second = _build(store, save_to_charts=True)
    assert first.saved_chart_id == second.saved_chart_id
    assert len(InsightService(store).list_charts(RUN).items) == 1


def test_without_save_flag_nothing_lands_in_charts(store: ArtifactStore) -> None:
    view = _build(store)
    assert view.saved_chart_id is None
    assert InsightService(store).list_charts(RUN).items == []


def test_histogram_saves_as_servable_binned_bar(store: ArtifactStore) -> None:
    view = _build(
        store,
        chart_type="histogram",
        x_column="amount",
        y_column=None,
        aggregate=None,
    )
    assert view.saved_chart_id is None
    view = _build(
        store,
        chart_type="histogram",
        x_column="amount",
        y_column=None,
        aggregate=None,
        save_to_charts=True,
    )
    assert view.saved_chart_id
    served = InsightService(store).get_chart(RUN, view.saved_chart_id)
    assert served.spec["mark"] == "bar"
