"""Streaming CSV upload (§7.2): per-upload staging isolation, sha256 while
writing, atomic move into the canonical uploads layout, optional Parquet copy."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import BinaryIO

import duckdb

from eda_platform.application.dto import DatasetColumn, DatasetHandle, UploadStatus
from eda_platform.application.services.session_service import ProjectNotFoundError
from eda_platform.core.config import require_absolute_workspace
from eda_platform.core.fs import remove_tree
from eda_platform.core.ids import (
    DERIVED_SESSION_PREFIXES,
    INTERNAL_SESSION_MARKER,
    is_safe_session_id,
    make_dataset_id,
)
from eda_platform.core.query import TrustedFileQueryEngine
from eda_platform.core.store import ArtifactStore

MAX_UPLOAD_BYTES = 1 << 30  # 1 GiB
HEADER_SNIFF_BYTES = 8_192
STAGING_TTL_SECONDS = 24 * 3600
STAGING_DIRNAME = "_staging"
PARQUET_FLAG_ENV = "EDA_PARQUET_INGEST_ENABLED"
_CHUNK_SIZE = 1 << 20
_ALLOWED_EXTENSIONS = frozenset({".csv"})
# hash_file() truncates sha256 to 12 hex chars; keep dataset ids consistent.
_HASH_LENGTH = 12
# Bound on the in-use lookup; a project past this is far outside local-first use.
_USED_BY_SCAN_LIMIT = 2000
# A project past this is far outside local-first use; the list is a picker.
_LIST_UPLOADS_LIMIT = 500


class UploadServiceError(Exception):
    pass


class UploadValidationError(UploadServiceError):
    pass


class UploadTooLargeError(UploadServiceError):
    def __init__(self, max_bytes: int) -> None:
        super().__init__(f"Upload exceeds the {max_bytes} byte limit.")
        self.max_bytes = max_bytes


class UploadProjectByteQuotaError(UploadServiceError):
    def __init__(self, max_bytes: int) -> None:
        super().__init__(f"Project upload storage quota is {max_bytes} bytes.")
        self.max_bytes = max_bytes


class UploadFileQuotaError(UploadServiceError):
    def __init__(self, max_files: int) -> None:
        super().__init__(f"Project upload file quota is {max_files} files.")
        self.max_files = max_files


class UploadNotFoundError(UploadServiceError):
    def __init__(self, dataset_id: str) -> None:
        super().__init__(f"No upload {dataset_id} in this project.")
        self.dataset_id = dataset_id


class UploadInUseError(UploadServiceError):
    def __init__(self, dataset_id: str, session_ids: list[str]) -> None:
        super().__init__(
            f"Upload {dataset_id} is the source of "
            f"{len(session_ids)} session(s); delete those first."
        )
        self.dataset_id = dataset_id
        self.session_ids = session_ids


class UploadConcurrentQuotaError(UploadServiceError):
    def __init__(self, max_concurrent: int) -> None:
        super().__init__(f"Project allows {max_concurrent} concurrent uploads.")
        self.max_concurrent = max_concurrent


def sanitize_upload_name(raw: str) -> str:
    """Basename-only, matching pipeline_ui.persist_uploads: a browser-supplied
    name with path separators must not escape the staging directory. Glob
    metacharacters are stripped too — the name later reaches read_csv('...'),
    which glob-expands its argument."""
    safe = Path(str(raw).replace("\\", "/")).name
    # Glob metacharacters reach read_csv('...') which glob-expands; control
    # chars (NUL) make open() raise ValueError mid-request.
    safe = re.sub(r"[*?\[\]\x00-\x1f\x7f]", "_", safe)
    if not safe or safe in {".", ".."}:
        safe = "upload.csv"
    return safe


def parquet_ingest_enabled() -> bool:
    return os.environ.get(PARQUET_FLAG_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


class UploadService:
    def __init__(
        self,
        store: ArtifactStore,
        engine: TrustedFileQueryEngine,
        *,
        max_bytes: int = MAX_UPLOAD_BYTES,
        project_byte_quota: int = 1 << 40,
        project_file_quota: int = 10_000,
        concurrent_upload_quota: int = 8,
        parquet_enabled: bool | None = None,
    ) -> None:
        self._store = store
        self._engine = engine
        self._max_bytes = max_bytes
        self._project_byte_quota = project_byte_quota
        self._project_file_quota = project_file_quota
        self._concurrent_upload_quota = concurrent_upload_quota
        self._parquet_enabled = parquet_enabled

    def create_upload(self, project_id: str, filename: str, stream: BinaryIO) -> UploadStatus:
        # A registered project_id is still validated as a single path segment:
        # a poisoned DB row like ".." must not steer writes out of projects/.
        if Path(project_id).name != project_id or project_id in {".", ".."}:
            raise ProjectNotFoundError(project_id)
        if not self._store.project_exists(project_id):
            raise ProjectNotFoundError(project_id)
        safe_name = sanitize_upload_name(filename)
        if Path(safe_name).suffix.lower() not in _ALLOWED_EXTENSIONS:
            raise UploadValidationError("Only .csv uploads are accepted.")
        upload_id = f"up_{uuid.uuid4().hex[:12]}"
        rejected = self._store.reserve_upload(
            upload_id,
            project_id,
            file_quota=self._project_file_quota,
            concurrent_quota=self._concurrent_upload_quota,
            now=time.time(),
        )
        if rejected == "files":
            raise UploadFileQuotaError(self._project_file_quota)
        if rejected == "concurrent":
            raise UploadConcurrentQuotaError(self._concurrent_upload_quota)
        staging_dir = self._store.root / STAGING_DIRNAME / upload_id
        staging_path = staging_dir / safe_name
        final_path: Path | None = None
        try:
            staging_dir.mkdir(parents=True, exist_ok=False)
            content_hash = self._stream_to_staging(upload_id, stream, staging_path)
            dataset_id = make_dataset_id(safe_name, content_hash)
            final_path = self._promote(project_id, dataset_id, staging_path)
            handle = self._build_handle(project_id, dataset_id, final_path, content_hash)
            self._store.complete_upload_reservation(
                upload_id, dataset_id, handle.byte_size
            )
        except duckdb.Error as exc:
            # describe/COPY rejected the file: undo the promotion so an
            # unparsable upload never lingers in the canonical uploads dir.
            remove_tree(staging_dir, ignore_errors=True)
            self._rollback_promotion(final_path)
            message = "Uploaded file is not parseable as CSV."
            raise UploadValidationError(message) from exc
        except Exception:
            remove_tree(staging_dir, ignore_errors=True)
            self._rollback_promotion(final_path)
            raise
        finally:
            # No-op after completion; essential on validation, quota, DuckDB,
            # filesystem, or unexpected failures.
            self._store.release_upload_reservation(upload_id)
        remove_tree(staging_dir, ignore_errors=True)
        status = UploadStatus(
            upload_id=upload_id, project_id=project_id, status="completed", dataset=handle
        )
        return status

    def list_uploads(self, project_id: str) -> list[DatasetHandle]:
        """Datasets already stored in this project, newest first.

        Disk is the source of truth rather than the quota index: the index is
        rebuilt from these directories at startup, so anything it disagrees
        with is stale. Schema comes from a header-only DESCRIBE, which is why
        listing a 300 MB table costs the same as listing a 3 KB one.
        """
        if Path(project_id).name != project_id or project_id in {".", ".."}:
            raise ProjectNotFoundError(project_id)
        if not self._store.project_exists(project_id):
            raise ProjectNotFoundError(project_id)
        uploads = self._store.project_dir(project_id) / "uploads"
        if uploads.is_symlink() or not uploads.is_dir():
            return []
        candidates: list[tuple[float, str, Path]] = []
        for dataset_dir in uploads.iterdir():
            if dataset_dir.is_symlink() or not dataset_dir.is_dir():
                continue
            source = self._upload_source_path(dataset_dir)
            if source is None:
                continue
            candidates.append((_safe_mtime(source), dataset_dir.name, source))

        # Limit after ordering. Directory iteration order is filesystem-defined,
        # so stopping during the scan can omit the project's newest uploads.
        candidates.sort(key=lambda item: item[0], reverse=True)
        handles: list[DatasetHandle] = []
        for _, dataset_id, source in candidates[:_LIST_UPLOADS_LIMIT]:
            try:
                handles.append(self._listed_handle(project_id, dataset_id, source))
            except (OSError, duckdb.Error):
                # A file the engine cannot describe is still real and still
                # occupies quota; list it without a schema rather than hide it.
                handles.append(
                    DatasetHandle(
                        dataset_id=dataset_id,
                        project_id=project_id,
                        display_name=source.name,
                        original_uri=self._workspace_relative(source),
                        byte_size=_safe_size(source),
                        ingest_status="unreadable",
                    )
                )
        return handles

    def _upload_source_path(self, dataset_dir: Path) -> Path | None:
        version_dir = dataset_dir / "v1"
        if version_dir.is_symlink() or not version_dir.is_dir():
            return None
        for path in sorted(version_dir.iterdir()):
            if path.is_file() and not path.is_symlink():
                return path
        return None

    def _listed_handle(self, project_id: str, dataset_id: str, source: Path) -> DatasetHandle:
        described = self._engine.describe_file(source)
        return DatasetHandle(
            dataset_id=dataset_id,
            project_id=project_id,
            display_name=source.name,
            original_uri=self._workspace_relative(source),
            format=source.suffix.lstrip(".").lower() or "csv",
            byte_size=_safe_size(source),
            schema=[DatasetColumn(name=name, dtype=dtype) for name, dtype in described],
            ingest_status="ready",
        )

    def _workspace_relative(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self._store.root.resolve()))
        except ValueError:
            return path.name

    def delete_upload(self, project_id: str, dataset_id: str) -> str:
        """Remove one uploaded file, its Parquet copy and its quota row.

        Refuses while a session still lists the file: sessions read their source
        from this directory, and removing it silently turns a finished analysis
        into one whose table preview and cleaning pages 404.
        """
        if Path(project_id).name != project_id or project_id in {".", ".."}:
            raise ProjectNotFoundError(project_id)
        if not self._store.project_exists(project_id):
            raise ProjectNotFoundError(project_id)
        # Same containment rule the write path uses; a dataset id reaches this
        # from the URL and is about to become a directory name.
        if not is_safe_session_id(dataset_id):
            raise UploadValidationError("Dataset id is not a safe path segment.")
        dataset_dir = self._store.project_dir(project_id) / "uploads" / dataset_id
        uploads_root = (self._store.project_dir(project_id) / "uploads").resolve()
        if dataset_dir.is_symlink() or not dataset_dir.resolve().is_relative_to(uploads_root):
            raise UploadValidationError("Upload path escapes the project uploads directory.")
        display_name = self._upload_display_name(dataset_dir)
        if not dataset_dir.is_dir():
            if not self._store.forget_upload_usage(project_id, dataset_id):
                raise UploadNotFoundError(dataset_id)
            return dataset_id
        holders = self._sessions_using(project_id, display_name)
        if holders:
            raise UploadInUseError(dataset_id, holders)
        remove_tree(dataset_dir)
        self._store.forget_upload_usage(project_id, dataset_id)
        return dataset_id

    def _upload_display_name(self, dataset_dir: Path) -> str:
        version_dir = dataset_dir / "v1"
        if version_dir.is_symlink() or not version_dir.is_dir():
            return ""
        for path in sorted(version_dir.iterdir()):
            if path.is_file() and not path.is_symlink():
                return path.name
        return ""

    def _sessions_using(self, project_id: str, display_name: str) -> list[str]:
        """Session-index lookup by dataset name.

        Name, not dataset id: the index stores the names a session loaded, and
        re-uploading the same filename with different content mints a different
        id. Matching on name therefore over-refuses rather than letting a delete
        strand a session's source.
        """
        if not display_name:
            return []
        # The index stores the DatasetProfile's `name`, which is the stem
        # ("sales"), while the file on disk keeps its extension ("sales.csv").
        wanted = {display_name, Path(display_name).stem}
        rows = self._store.query_session_index_rows(
            project_id,
            limit=_USED_BY_SCAN_LIMIT,
            exclude_session_id_containing=INTERNAL_SESSION_MARKER,
            exclude_session_id_prefixes=DERIVED_SESSION_PREFIXES,
        )
        holders = []
        for row in rows:
            try:
                names = json.loads(row["dataset_names_json"] or "[]")
            except (TypeError, ValueError):
                continue
            if isinstance(names, list) and wanted.intersection(map(str, names)):
                holders.append(str(row["session_id"]))
        return holders

    def _stream_to_staging(
        self, upload_id: str, stream: BinaryIO, staging_path: Path
    ) -> str:
        digest = hashlib.sha256()
        size = 0
        header = b""
        with staging_path.open("wb") as sink:
            while True:
                chunk = stream.read(_CHUNK_SIZE)
                if not chunk:
                    break
                size += len(chunk)
                if size > self._max_bytes:
                    raise UploadTooLargeError(self._max_bytes)
                if not self._store.update_upload_reservation(
                    upload_id,
                    byte_size=size,
                    byte_quota=self._project_byte_quota,
                    now=time.time(),
                ):
                    raise UploadProjectByteQuotaError(self._project_byte_quota)
                if len(header) < HEADER_SNIFF_BYTES:
                    header += chunk[: HEADER_SNIFF_BYTES - len(header)]
                digest.update(chunk)
                sink.write(chunk)
        if not header.strip():
            raise UploadValidationError("Uploaded file is empty.")
        return digest.hexdigest()[:_HASH_LENGTH]

    def _promote(self, project_id: str, dataset_id: str, staging_path: Path) -> Path:
        final_dir = self._store.project_dir(project_id) / "uploads" / dataset_id / "v1"
        projects_root = (self._store.root / "projects").resolve()
        if not final_dir.resolve().is_relative_to(projects_root):
            raise UploadValidationError("Upload destination escapes the projects directory.")
        final_dir.mkdir(parents=True, exist_ok=True)
        final_path = final_dir / staging_path.name
        # Same filesystem (both under the workspace root), so this is atomic.
        os.replace(staging_path, final_path)
        if self._parquet_active():
            parquet_path = final_dir / "parquet" / f"{final_path.stem}.parquet"
            self._engine.copy_csv_to_parquet(final_path, parquet_path)
        return final_path

    def _rollback_promotion(self, final_path: Path | None) -> None:
        """Undo a canonical promotion after post-move validation failed."""
        if final_path is None:
            return
        # Remove the whole <dataset_id>/ dir: v1/ file plus any parquet output.
        dataset_dir = final_path.parent.parent
        remove_tree(dataset_dir, ignore_errors=True)

    def _parquet_active(self) -> bool:
        if self._parquet_enabled is not None:
            return self._parquet_enabled
        return parquet_ingest_enabled()

    def _build_handle(
        self, project_id: str, dataset_id: str, final_path: Path, content_hash: str
    ) -> DatasetHandle:
        described = self._engine.describe_file(final_path)
        try:
            original_uri = str(final_path.resolve().relative_to(self._store.root.resolve()))
        except ValueError:
            original_uri = final_path.name
        return DatasetHandle(
            dataset_id=dataset_id,
            project_id=project_id,
            display_name=final_path.name,
            original_uri=original_uri,
            format="csv",
            content_hash=content_hash,
            byte_size=final_path.stat().st_size,
            row_count=None,
            schema=[DatasetColumn(name=name, dtype=dtype) for name, dtype in described],
            ingest_status="ready",
        )


def _safe_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def sweep_staging(
    workspace: Path | str,
    *,
    ttl_seconds: int = STAGING_TTL_SECONDS,
    now: float | None = None,
) -> int:
    """Delete per-upload staging dirs older than the TTL; returns removed count.

    Staging lifetime is decoupled from the request (§7.2): nothing unlinks in a
    request finally-block, so orphans from crashed uploads are reaped here.
    """
    staging_root = require_absolute_workspace(workspace) / STAGING_DIRNAME
    # A symlinked staging root (or entry) would let rmtree chew through
    # whatever it points at; refuse to sweep anything that isn't a real dir.
    if staging_root.is_symlink() or not staging_root.is_dir():
        return 0
    reference = time.time() if now is None else now
    removed = 0
    for entry in staging_root.iterdir():
        if entry.is_symlink() or not entry.is_dir():
            continue
        try:
            age = reference - entry.stat().st_mtime
        except OSError:
            continue
        if age >= ttl_seconds:
            remove_tree(entry, ignore_errors=True)
            removed += 1
    return removed
