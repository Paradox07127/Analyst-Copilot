"""UploadService: streaming staging, hashing, validation, atomic promote,
per-upload isolation, TTL sweep, Parquet flag."""

from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path

import pytest

import eda_platform.application.services.upload_service as upload_service_module
from eda_platform.application.services.session_service import ProjectNotFoundError
from eda_platform.application.services.upload_service import (
    STAGING_DIRNAME,
    UploadInUseError,
    UploadNotFoundError,
    UploadService,
    UploadTooLargeError,
    UploadValidationError,
    sanitize_upload_name,
    sweep_staging,
)
from eda_platform.core.ids import make_dataset_id
from eda_platform.core.query import TrustedFileQueryEngine
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType

PROJECT = "demo"
CSV_BODY = b"id,amount\n1,2.5\n2,3.5\n"


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    store = ArtifactStore(tmp_path)
    store.ensure_project(PROJECT, name="Demo")
    return store


def _service(store: ArtifactStore, **kwargs: object) -> UploadService:
    engine = TrustedFileQueryEngine([store.root / "projects"])
    return UploadService(store, engine, **kwargs)  # type: ignore[arg-type]


def test_upload_roundtrip(store: ArtifactStore) -> None:
    service = _service(store, parquet_enabled=False)
    status = service.create_upload(PROJECT, "orders.csv", io.BytesIO(CSV_BODY))
    assert status.status == "completed"
    handle = status.dataset
    assert handle is not None
    expected_hash = hashlib.sha256(CSV_BODY).hexdigest()[:12]
    assert handle.content_hash == expected_hash
    assert handle.dataset_id == make_dataset_id("orders.csv", expected_hash)
    final = store.root / handle.original_uri
    assert final.is_file()
    assert final.read_bytes() == CSV_BODY
    assert final.parent == store.project_dir(PROJECT) / "uploads" / handle.dataset_id / "v1"
    assert [column.name for column in handle.schema_] == ["id", "amount"]
    # Staging dir is gone after promotion.
    assert list((store.root / STAGING_DIRNAME).iterdir()) == []


def test_upload_filename_sanitized(store: ArtifactStore) -> None:
    service = _service(store, parquet_enabled=False)
    status = service.create_upload(PROJECT, "../../evil.csv", io.BytesIO(CSV_BODY))
    handle = status.dataset
    assert handle is not None
    assert handle.display_name == "evil.csv"
    resolved = (store.root / handle.original_uri).resolve()
    assert resolved.is_relative_to((store.project_dir(PROJECT) / "uploads").resolve())


def test_sanitize_upload_name_edge_cases() -> None:
    assert sanitize_upload_name("a\\b\\c.csv") == "c.csv"
    assert sanitize_upload_name("..") == "upload.csv"
    assert sanitize_upload_name("") == "upload.csv"


def test_upload_rejects_non_csv(store: ArtifactStore) -> None:
    service = _service(store, parquet_enabled=False)
    with pytest.raises(UploadValidationError):
        service.create_upload(PROJECT, "data.txt", io.BytesIO(b"x"))
    assert not (store.root / STAGING_DIRNAME).exists()


def test_upload_rejects_oversize_and_cleans_staging(store: ArtifactStore) -> None:
    service = _service(store, max_bytes=10, parquet_enabled=False)
    with pytest.raises(UploadTooLargeError):
        service.create_upload(PROJECT, "big.csv", io.BytesIO(b"a" * 100))
    assert list((store.root / STAGING_DIRNAME).iterdir()) == []


def test_upload_rejects_empty_file(store: ArtifactStore) -> None:
    service = _service(store, parquet_enabled=False)
    with pytest.raises(UploadValidationError):
        service.create_upload(PROJECT, "empty.csv", io.BytesIO(b"   \n  "))
    assert list((store.root / STAGING_DIRNAME).iterdir()) == []


def test_failed_upload_preserves_synchronous_error_contract(store: ArtifactStore) -> None:
    service = _service(store, max_bytes=10, parquet_enabled=False)
    with pytest.raises(UploadTooLargeError):
        service.create_upload(PROJECT, "big.csv", io.BytesIO(b"a" * 100))
    assert not hasattr(service, "_statuses")


def test_upload_unknown_project(store: ArtifactStore) -> None:
    service = _service(store, parquet_enabled=False)
    with pytest.raises(ProjectNotFoundError):
        service.create_upload("nope", "orders.csv", io.BytesIO(CSV_BODY))


def test_upload_service_has_no_process_local_status_api(store: ArtifactStore) -> None:
    service = _service(store)
    assert not hasattr(service, "_statuses")
    assert not hasattr(service, "get_upload")


def test_same_name_uploads_do_not_collide(store: ArtifactStore) -> None:
    """The legacy shared _incoming/<name> pattern broke under concurrency;
    per-upload staging must keep same-named files fully isolated."""
    service = _service(store, parquet_enabled=False)
    first = service.create_upload(PROJECT, "orders.csv", io.BytesIO(b"a,b\n1,2\n"))
    second = service.create_upload(PROJECT, "orders.csv", io.BytesIO(b"a,b\n3,4\n"))
    assert first.dataset is not None and second.dataset is not None
    assert first.dataset.dataset_id != second.dataset.dataset_id
    assert (store.root / first.dataset.original_uri).read_bytes() == b"a,b\n1,2\n"
    assert (store.root / second.dataset.original_uri).read_bytes() == b"a,b\n3,4\n"


def test_parquet_flag_generates_copy(store: ArtifactStore) -> None:
    service = _service(store, parquet_enabled=True)
    status = service.create_upload(PROJECT, "orders.csv", io.BytesIO(CSV_BODY))
    assert status.dataset is not None
    final = store.root / status.dataset.original_uri
    parquet = final.parent / "parquet" / "orders.parquet"
    assert parquet.is_file()


def test_parquet_flag_off_by_default(store: ArtifactStore, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EDA_PARQUET_INGEST_ENABLED", raising=False)
    service = _service(store)
    status = service.create_upload(PROJECT, "orders.csv", io.BytesIO(CSV_BODY))
    assert status.dataset is not None
    final = store.root / status.dataset.original_uri
    assert not (final.parent / "parquet").exists()


def test_sweep_staging_removes_only_expired(store: ArtifactStore) -> None:
    staging = store.root / STAGING_DIRNAME
    old_dir = staging / "up_old"
    new_dir = staging / "up_new"
    old_dir.mkdir(parents=True)
    new_dir.mkdir(parents=True)
    (old_dir / "a.csv").write_text("x")
    stale = old_dir.stat().st_mtime - 100_000
    os.utime(old_dir, (stale, stale))
    removed = sweep_staging(store.root, ttl_seconds=3_600)
    assert removed == 1
    assert not old_dir.exists()
    assert new_dir.exists()


def test_sweep_staging_no_dir(tmp_path: Path) -> None:
    assert sweep_staging(tmp_path) == 0


# --- deletion ---------------------------------------------------------------
def _upload(store: ArtifactStore, name: str = "sales.csv") -> str:
    status = _service(store).create_upload(PROJECT, name, io.BytesIO(CSV_BODY))
    assert status.dataset is not None
    return status.dataset.dataset_id


def _usage_rows(store: ArtifactStore) -> list[tuple[str, str]]:
    import sqlite3
    from contextlib import closing

    with closing(sqlite3.connect(store.root / "state.sqlite")) as conn:
        return [
            (str(r[0]), str(r[1]))
            for r in conn.execute("select project_id, dataset_id from upload_usage")
        ]


def test_delete_removes_the_file_and_frees_the_quota(store: ArtifactStore) -> None:
    dataset_id = _upload(store)
    dataset_dir = store.project_dir(PROJECT) / "uploads" / dataset_id
    assert dataset_dir.is_dir()
    assert (PROJECT, dataset_id) in _usage_rows(store)

    _service(store).delete_upload(PROJECT, dataset_id)

    assert not dataset_dir.exists()
    assert (PROJECT, dataset_id) not in _usage_rows(store)


def test_delete_refuses_while_a_session_still_reads_the_file(store: ArtifactStore) -> None:
    """Sessions resolve their source from uploads/<id>/v1; removing it under a
    finished analysis turns its table preview and cleaning pages into 404s."""
    dataset_id = _upload(store)
    store.start_session(PROJECT, "sess_holder")
    store.save_artifact(
        Artifact(
            id="prof_1",
            type=ArtifactType.DATASET_PROFILE,
            project_id=PROJECT,
            session_id="sess_holder",
            payload={"name": "sales", "dataset_id": dataset_id},
        )
    )
    store.mark_session_status(PROJECT, "sess_holder", "completed")

    with pytest.raises(UploadInUseError) as excinfo:
        _service(store).delete_upload(PROJECT, dataset_id)

    assert excinfo.value.session_ids == ["sess_holder"]
    assert (store.project_dir(PROJECT) / "uploads" / dataset_id).is_dir()


def test_delete_of_an_unknown_upload_is_a_404_not_a_silent_success(
    store: ArtifactStore,
) -> None:
    with pytest.raises(UploadNotFoundError):
        _service(store).delete_upload(PROJECT, "ds_nothing_here")


def test_delete_rejects_a_dataset_id_that_is_not_one_path_segment(
    store: ArtifactStore,
) -> None:
    with pytest.raises(UploadValidationError):
        _service(store).delete_upload(PROJECT, "../../etc")


def test_delete_on_a_missing_project_is_not_a_path_probe(store: ArtifactStore) -> None:
    with pytest.raises(ProjectNotFoundError):
        _service(store).delete_upload("no-such-project", "ds_x")


# --- listing ----------------------------------------------------------------
def test_list_uploads_returns_what_the_project_already_holds(store: ArtifactStore) -> None:
    """Files live under the project, not the session, so a second session can
    reuse them — but only if the UI can find out they exist."""
    first = _upload(store, "sales.csv")
    second = _upload(store, "inventory.csv")

    listed = _service(store).list_uploads(PROJECT)

    assert {handle.dataset_id for handle in listed} == {first, second}
    handle = next(item for item in listed if item.dataset_id == first)
    assert handle.display_name == "sales.csv"
    assert handle.byte_size == len(CSV_BODY)
    assert [column.name for column in handle.schema_] == ["id", "amount"]


def test_list_uploads_is_empty_before_anything_is_uploaded(store: ArtifactStore) -> None:
    assert _service(store).list_uploads(PROJECT) == []


def test_list_uploads_drops_a_deleted_dataset(store: ArtifactStore) -> None:
    dataset_id = _upload(store)
    service = _service(store)
    service.delete_upload(PROJECT, dataset_id)

    assert service.list_uploads(PROJECT) == []


def test_list_uploads_on_a_missing_project_is_not_a_path_probe(store: ArtifactStore) -> None:
    with pytest.raises(ProjectNotFoundError):
        _service(store).list_uploads("no-such-project")


def test_list_uploads_applies_limit_after_newest_first_ordering(
    store: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_ids = [
        _upload(store, "old.csv"),
        _upload(store, "middle.csv"),
        _upload(store, "new.csv"),
    ]
    uploads = store.project_dir(PROJECT) / "uploads"
    for timestamp, dataset_id in enumerate(dataset_ids, start=1):
        source = next((uploads / dataset_id / "v1").iterdir())
        os.utime(source, (timestamp, timestamp))
    monkeypatch.setattr(upload_service_module, "_LIST_UPLOADS_LIMIT", 2)

    listed = _service(store).list_uploads(PROJECT)

    assert [handle.dataset_id for handle in listed] == dataset_ids[:0:-1]
