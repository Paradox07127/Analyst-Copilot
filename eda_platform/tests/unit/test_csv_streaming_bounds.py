"""The two endpoints that shape a CSV in pandas must read it in bounded chunks,
and must keep the sampling/head semantics the workbench relies on."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

import eda_platform.application.services.dataset_service as dataset_service_module
import eda_platform.tools.loader as loader_module
from eda_platform.application.chart_builder import (
    CUSTOM_CHART_ROW_LIMIT,
    apply_outlier_bounds,
    select_chart_columns,
)
from eda_platform.application.dto import CustomChartRequest
from eda_platform.application.services.dataset_service import DatasetService
from eda_platform.application.services.insight_service import InsightService
from eda_platform.core.query import TrustedFileQueryEngine
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.tools.frame_stats import iqr_bounds
from eda_platform.tools.loader import stream_csv_chunks

PROJECT = "demo"
RUN = "run_stream"
DATASET = "ds_stream"
CSV_NAME = "big.csv"

CHUNK_ROWS = 1_000


class FakeReader:
    """Stands in for pandas' TextFileReader: iterable *and* a context manager,
    which is how the loader consumes it."""

    def __init__(self, chunks: Iterator[pd.DataFrame]) -> None:
        self._chunks = chunks

    def __iter__(self) -> Iterator[pd.DataFrame]:
        return self._chunks

    def __enter__(self) -> FakeReader:
        return self

    def __exit__(self, *_: object) -> bool:
        return False


class ReadSpy:
    """Records every DataFrame pandas hands back, so a test can assert that no
    single materialised frame exceeds the chunk size."""

    def __init__(self) -> None:
        self.frame_rows: list[int] = []
        self.chunksizes: list[int] = []

    @property
    def max_frame_rows(self) -> int:
        return max(self.frame_rows, default=0)


@pytest.fixture
def read_spy(monkeypatch: pytest.MonkeyPatch) -> ReadSpy:
    spy = ReadSpy()
    real_read_csv = pd.read_csv

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        chunksize = kwargs.get("chunksize")
        result = real_read_csv(*args, **kwargs)
        if chunksize is None:
            if isinstance(result, pd.DataFrame):
                spy.frame_rows.append(len(result))
            return result
        spy.chunksizes.append(int(chunksize))

        def _iter() -> Iterator[pd.DataFrame]:
            with result as reader:
                for chunk in reader:
                    spy.frame_rows.append(len(chunk))
                    yield chunk

        return FakeReader(_iter())

    monkeypatch.setattr(pd, "read_csv", wrapper)
    monkeypatch.setattr(loader_module, "CSV_CHUNK_ROWS", CHUNK_ROWS, raising=False)
    return spy


def _profile(rows: int, column_names: list[str]) -> Artifact:
    return Artifact(
        id=f"prof_{RUN}",
        type=ArtifactType.DATASET_PROFILE,
        project_id=PROJECT,
        session_id=RUN,
        payload={
            "dataset_id": DATASET,
            "name": CSV_NAME,
            "rows": rows,
            "columns": len(column_names),
            "column_names": column_names,
        },
    )


def _store_with(tmp_path: Path, frame: pd.DataFrame) -> ArtifactStore:
    store = ArtifactStore(tmp_path)
    store.ensure_project(PROJECT, name="Demo")
    store.start_session(PROJECT, RUN)
    store.save_artifact(_profile(len(frame), [str(c) for c in frame.columns]))
    source = store.project_dir(PROJECT) / "uploads" / DATASET / "v1" / CSV_NAME
    source.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(source, index=False)
    return store


# -- distributions --------------------------------------------------------

DIST_ROWS = 30_000
DIST_CAP = 1_000


def _dist_frame() -> pd.DataFrame:
    # ``idx`` is strictly increasing: a head-only sample would cap its max near
    # DIST_CAP, a whole-file sample lands near DIST_ROWS.
    return pd.DataFrame(
        {"idx": range(DIST_ROWS), "label": [f"g{index % 7}" for index in range(DIST_ROWS)]}
    )


@pytest.fixture
def dist_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> ArtifactStore:
    store = _store_with(tmp_path, _dist_frame())
    monkeypatch.setattr(dataset_service_module, "DIST_SAMPLE_CAP", DIST_CAP)
    return store


def _distributions(store: ArtifactStore) -> dict[str, Any]:
    return DatasetService(
        store,
        TrustedFileQueryEngine([store.root / "projects"]),
    ).get_distributions(DATASET, RUN).model_dump()


def test_distributions_reads_in_bounded_chunks(
    dist_store: ArtifactStore, read_spy: ReadSpy
) -> None:
    _distributions(dist_store)
    assert read_spy.chunksizes, "distributions must read the CSV with a chunksize"
    assert read_spy.max_frame_rows <= CHUNK_ROWS


def test_distributions_sample_spans_the_whole_file(
    dist_store: ArtifactStore, read_spy: ReadSpy
) -> None:
    """The cap is a random sample of the whole table, not its first rows."""
    body = _distributions(dist_store)
    assert body["row_count"] == DIST_ROWS
    assert body["sampled"] is True
    assert body["sample_rows"] == DIST_CAP

    idx = next(column for column in body["columns"] if column["name"] == "idx")
    assert idx["kind"] == "numeric"
    assert sum(idx["counts"]) == DIST_CAP
    # A first-N-rows shortcut would put max at ~DIST_CAP instead of ~DIST_ROWS.
    assert idx["max"] > 0.9 * (DIST_ROWS - 1)
    assert idx["min"] < 0.1 * DIST_ROWS
    # Uniform sampling keeps the sample mean near the population mean; the bound
    # is >5 sigma for a 1k sample of 0..29999.
    midpoint = (DIST_ROWS - 1) / 2
    edges = idx["bin_edges"]
    weighted = sum(
        count * (edges[position] + edges[position + 1]) / 2
        for position, count in enumerate(idx["counts"])
    )
    assert abs(weighted / DIST_CAP - midpoint) < 0.05 * DIST_ROWS


# -- custom charts --------------------------------------------------------

CHART_ROWS = 12_000


def _chart_frame() -> pd.DataFrame:
    # The first two thirds sit in a narrow band and the last third is an order of
    # magnitude higher, so the IQR fence over the whole column is far wider than
    # the fence any single leading chunk would produce.
    amounts = [index % 10 if index < 8_000 else 1_000 + index for index in range(CHART_ROWS)]
    return pd.DataFrame(
        {
            "idx": range(CHART_ROWS),
            "amount": amounts,
            "label": [f"g{index % 3}" for index in range(CHART_ROWS)],
        }
    )


@pytest.fixture
def chart_store(tmp_path: Path) -> ArtifactStore:
    return _store_with(tmp_path, _chart_frame())


def _build(store: ArtifactStore, **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "dataset_id": DATASET,
        "chart_type": "point",
        "x_column": "idx",
        "y_column": "amount",
        "aggregate": "none",
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
    ).model_dump()


def test_custom_chart_reads_in_bounded_chunks(
    chart_store: ArtifactStore, read_spy: ReadSpy
) -> None:
    _build(chart_store)
    assert read_spy.chunksizes, "custom charts must read the CSV with a chunksize"
    assert read_spy.max_frame_rows <= CHUNK_ROWS


def test_custom_chart_outliers_read_in_bounded_chunks(
    chart_store: ArtifactStore, read_spy: ReadSpy
) -> None:
    _build(chart_store, drop_outliers=True)
    assert read_spy.chunksizes
    assert read_spy.max_frame_rows <= CHUNK_ROWS


def test_custom_chart_keeps_head_semantics(
    chart_store: ArtifactStore, read_spy: ReadSpy
) -> None:
    body = _build(chart_store)
    assert body["source_row_count"] == CHART_ROWS
    assert body["row_count"] == CUSTOM_CHART_ROW_LIMIT
    assert body["truncated"] is True
    values = body["spec"]["data"]["values"]
    assert [row["idx"] for row in values[:3]] == [0, 1, 2]
    assert values[-1]["idx"] == CUSTOM_CHART_ROW_LIMIT - 1


def test_parser_failure_restarts_the_consumer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lenient-reader fallback happens mid-stream, so the consumer is called a
    second time and must not carry over the rows the failed attempt already saw."""
    source = tmp_path / "ragged.csv"
    source.write_text(
        "a,b\n" + "".join(f"{index},{index * 2}\n" for index in range(10)), encoding="utf-8"
    )
    real_read_csv = pd.read_csv
    attempts: list[str] = []

    def flaky(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("engine") == "python":
            attempts.append("lenient")
            return real_read_csv(*args, **kwargs)
        attempts.append("strict")

        def _fail_midway() -> Iterator[pd.DataFrame]:
            yield real_read_csv(source, nrows=4)
            raise pd.errors.ParserError("tokenizing failed past the first chunk")

        return FakeReader(_fail_midway())

    monkeypatch.setattr(pd, "read_csv", flaky)
    frame = stream_csv_chunks(
        source, lambda chunks: pd.concat(list(chunks), ignore_index=True), chunksize=4
    )
    assert attempts == ["strict", "lenient"]
    assert list(frame["a"]) == list(range(10))


def test_custom_chart_outlier_fence_matches_full_frame(
    chart_store: ArtifactStore, read_spy: ReadSpy
) -> None:
    """The IQR fence is a whole-column statistic; a fence derived from the
    leading chunk would keep a different number of rows."""
    # In-memory oracle: whole-frame column selection, then one IQR fence
    # computed over the complete Y column.
    reference = select_chart_columns(
        _chart_frame(), ["idx", "amount"], drop_missing=True
    )
    numeric_amount = cast(pd.Series, pd.to_numeric(reference["amount"], errors="coerce"))
    bounds = iqr_bounds(numeric_amount)
    assert bounds is not None
    reference = apply_outlier_bounds(reference, "amount", bounds)
    body = _build(chart_store, drop_outliers=True)
    assert body["source_row_count"] == len(reference)
    assert [row["idx"] for row in body["spec"]["data"]["values"][:5]] == [
        int(value) for value in reference["idx"].head(5)
    ]
