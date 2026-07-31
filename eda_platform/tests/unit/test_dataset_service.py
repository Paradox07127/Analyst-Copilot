"""DatasetService: profile-index extraction (no CSV reads), schema fallback,
preview paging, Parquet preference."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from eda_platform.application.services.dataset_service import (
    DatasetNotFoundError,
    DatasetService,
    DatasetSourceMissingError,
)
from eda_platform.application.services.session_service import SessionNotFoundError
from eda_platform.core.query import TrustedFileQueryEngine
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.sessions import SessionManifest

PROJECT = "demo"
RUN = "run_1"
DATASET = "ds_test0001"
CSV_NAME = "orders.csv"
CSV_BODY = "id,amount,when\n" + "".join(f"{i},{i}.5,2026-01-{i:02d}\n" for i in range(1, 11))


def _profile_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "dataset_id": DATASET,
        "name": CSV_NAME,
        "rows": 10,
        "columns": 3,
        "column_names": ["id", "amount", "when"],
        "dtypes": {"id": "int64", "amount": "float64", "when": "object"},
        "missing_values": {},
        "missing_percent": {},
        "numeric_columns": ["id", "amount"],
        "categorical_columns": [],
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
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
    return store


@pytest.fixture
def service(store: ArtifactStore) -> DatasetService:
    return DatasetService(store, TrustedFileQueryEngine([store.root / "projects"]))


def _save_profile(store: ArtifactStore, payload: dict[str, object]) -> None:
    store.save_artifact(
        Artifact(
            id=f"prof_{payload.get('dataset_id')}",
            type=ArtifactType.DATASET_PROFILE,
            project_id=PROJECT,
            session_id=RUN,
            payload=payload,
        )
    )


def _write_source(store: ArtifactStore, body: str = CSV_BODY) -> Path:
    source = store.project_dir(PROJECT) / "uploads" / DATASET / "v1" / CSV_NAME
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(body)
    return source


def test_list_datasets_from_profile_index(service: DatasetService, store: ArtifactStore) -> None:
    _save_profile(store, _profile_payload())
    _write_source(store)
    handles = service.list_datasets(RUN)
    assert len(handles) == 1
    handle = handles[0]
    assert handle.dataset_id == DATASET
    assert handle.project_id == PROJECT
    assert handle.display_name == CSV_NAME
    assert handle.original_uri == f"projects/{PROJECT}/uploads/{DATASET}/v1/{CSV_NAME}"
    assert not Path(handle.original_uri).is_absolute()
    assert handle.format == "csv"
    assert handle.content_hash == "hash12345678"
    assert handle.byte_size > 0
    assert handle.row_count == 10
    assert [column.name for column in handle.schema_] == ["id", "amount", "when"]
    assert handle.ingest_status == "ready"


def test_list_datasets_never_needs_the_csv(service: DatasetService, store: ArtifactStore) -> None:
    """Metadata comes from the artifact index alone: no source file on disk."""
    _save_profile(store, _profile_payload())
    handles = service.list_datasets(RUN)
    assert len(handles) == 1
    assert handles[0].ingest_status == "source_missing"
    assert handles[0].original_uri == ""
    assert handles[0].row_count == 10


def test_list_datasets_falls_back_to_disk_when_index_row_lost(
    service: DatasetService, store: ArtifactStore
) -> None:
    """Slice-E F4 stopgap: the artifacts table PK is a global artifact_id, so a
    same-id save from another project's run steals this run's index row. With
    the index empty, listing must fall back to the on-disk artifact files."""
    _save_profile(store, _profile_payload())
    _write_source(store)
    with sqlite3.connect(store.db_path) as conn:
        conn.execute("delete from artifacts where session_id = ?", (RUN,))
    handles = service.list_datasets(RUN)
    assert len(handles) == 1
    assert handles[0].dataset_id == DATASET
    assert handles[0].ingest_status == "ready"


def test_list_datasets_survives_same_id_save_in_another_project(
    service: DatasetService, store: ArtifactStore
) -> None:
    """Regression (slice-E review F4, now fixed at the index): saving the same
    content-derived profile id under another project keeps this run's index
    row — the listing is served by the index, not the disk fallback."""
    _save_profile(store, _profile_payload())
    _write_source(store)
    store.ensure_project("other_project", name="Other")
    store.start_session("other_project", "run_other")
    store.save_artifact(
        Artifact(
            id=f"prof_{DATASET}",
            type=ArtifactType.DATASET_PROFILE,
            project_id="other_project",
            session_id="run_other",
            payload=_profile_payload(),
        )
    )
    with sqlite3.connect(store.db_path) as conn:
        kept = conn.execute(
            "select count(*) from artifacts where project_id = ? and session_id = ?",
            (PROJECT, RUN),
        ).fetchone()[0]
    assert kept == 1
    handles = service.list_datasets(RUN)
    assert [handle.dataset_id for handle in handles] == [DATASET]
    assert handles[0].project_id == PROJECT


def test_list_datasets_unknown_run(service: DatasetService) -> None:
    with pytest.raises(SessionNotFoundError):
        service.list_datasets("missing_run")


def test_internal_runs_hidden(service: DatasetService) -> None:
    with pytest.raises(SessionNotFoundError):
        service.list_datasets("run__internal_x")


def test_schema_prefers_profile_payload(service: DatasetService, store: ArtifactStore) -> None:
    _save_profile(store, _profile_payload())
    # No source file at all: profile payload must be sufficient.
    schema = service.get_schema(DATASET, RUN)
    assert schema.source == "profile"
    assert [(column.name, column.dtype) for column in schema.columns] == [
        ("id", "int64"),
        ("amount", "float64"),
        ("when", "object"),
    ]


def test_schema_falls_back_to_header_inference(
    service: DatasetService, store: ArtifactStore
) -> None:
    _save_profile(store, _profile_payload(column_names=[], dtypes={}))
    _write_source(store)
    schema = service.get_schema(DATASET, RUN)
    assert schema.source == "inferred"
    assert [column.name for column in schema.columns] == ["id", "amount", "when"]


def test_schema_unknown_dataset(service: DatasetService, store: ArtifactStore) -> None:
    _save_profile(store, _profile_payload())
    with pytest.raises(DatasetNotFoundError):
        service.get_schema("ds_other", RUN)


def test_schema_source_missing_without_columns(
    service: DatasetService, store: ArtifactStore
) -> None:
    _save_profile(store, _profile_payload(column_names=[], dtypes={}))
    with pytest.raises(DatasetSourceMissingError):
        service.get_schema(DATASET, RUN)


def test_preview_pages_and_json_safe_rows(service: DatasetService, store: ArtifactStore) -> None:
    _save_profile(store, _profile_payload())
    _write_source(store)
    first = service.get_preview(DATASET, RUN, limit=4, offset=0)
    assert first.columns == ["id", "amount", "when"]
    assert len(first.rows) == 4
    assert first.has_more is True
    assert first.source_format == "csv"
    assert first.rows[0] == [1, 1.5, "2026-01-01"]
    last = service.get_preview(DATASET, RUN, limit=4, offset=8)
    assert len(last.rows) == 2
    assert last.has_more is False


def test_preview_clamps_limit(service: DatasetService, store: ArtifactStore) -> None:
    _save_profile(store, _profile_payload())
    _write_source(store)
    preview = service.get_preview(DATASET, RUN, limit=5_000)
    assert preview.limit == 200


def test_preview_missing_source(service: DatasetService, store: ArtifactStore) -> None:
    _save_profile(store, _profile_payload())
    with pytest.raises(DatasetSourceMissingError):
        service.get_preview(DATASET, RUN)


def test_preview_prefers_parquet_copy(service: DatasetService, store: ArtifactStore) -> None:
    _save_profile(store, _profile_payload())
    source = _write_source(store)
    engine = TrustedFileQueryEngine([store.root / "projects"])
    engine.copy_csv_to_parquet(source, source.parent / "parquet" / "orders.parquet")
    preview = service.get_preview(DATASET, RUN, limit=3)
    assert preview.source_format == "parquet"
    assert preview.rows[0][0] == 1
