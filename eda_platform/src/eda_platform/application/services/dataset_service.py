"""Lazy dataset read use cases (§8.2/§8.3): metadata from the DatasetProfile
artifact index, schema/preview via the trusted DuckDB file engine — neither
materialises a DataFrame or reads a complete CSV.

`get_distributions` is the one exception: per-column histograms and top-k counts
are computed in pandas over a DIST_SAMPLE_CAP-row random sample of the whole
table, so the numbers match the table preview. The sample is
drawn while streaming the CSV, so a large upload never lands in memory whole."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from eda_platform.application.distribution_view import (
    DIST_NUMERIC_BINS,
    DIST_SAMPLE_CAP,
    DIST_TOP_K,
    column_distributions,
    reservoir_sample,
)
from eda_platform.application.dto import (
    ColumnDistribution,
    ColumnDistributionsView,
    DatasetColumn,
    DatasetHandle,
    DatasetPreview,
    DatasetSchema,
    DistributionCategory,
)
from eda_platform.application.services.session_service import SessionNotFoundError
from eda_platform.core.ids import INTERNAL_SESSION_MARKER
from eda_platform.core.query import TrustedFileQueryEngine
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.tools.loader import stream_csv_chunks

DEFAULT_PREVIEW_LIMIT = 100
MAX_PREVIEW_LIMIT = 200


class DatasetServiceError(Exception):
    pass


class DatasetNotFoundError(DatasetServiceError):
    error_code = "dataset_not_found"

    def __init__(self, dataset_id: str, session_id: str) -> None:
        super().__init__(f"Dataset not found in run {session_id}: {dataset_id}")
        self.dataset_id = dataset_id
        self.session_id = session_id


class DatasetSourceMissingError(DatasetServiceError):
    error_code = "dataset_source_missing"

    def __init__(self, dataset_id: str) -> None:
        super().__init__(f"Source data for dataset {dataset_id} is unavailable.")
        self.dataset_id = dataset_id


class DatasetService:
    def __init__(self, store: ArtifactStore, engine: TrustedFileQueryEngine) -> None:
        self._store = store
        self._engine = engine

    def list_datasets(self, session_id: str) -> list[DatasetHandle]:
        project_id = self._project_for_run(session_id)
        input_hashes = self._manifest_input_hashes(project_id, session_id)
        handles: list[DatasetHandle] = []
        seen: set[str] = set()
        for artifact in self._profiles(project_id, session_id):
            dataset_id = artifact.payload.get("dataset_id")
            if not isinstance(dataset_id, str) or dataset_id in seen:
                continue
            seen.add(dataset_id)
            handles.append(self._handle_from_profile(project_id, artifact, input_hashes))
        return handles

    def get_schema(self, dataset_id: str, session_id: str) -> DatasetSchema:
        project_id = self._project_for_run(session_id)
        artifact = self._profile_for(project_id, session_id, dataset_id)
        columns = _columns_from_payload(artifact.payload)
        if columns:
            return DatasetSchema(
                dataset_id=dataset_id, session_id=session_id, columns=columns, source="profile"
            )
        source = self._source_path(project_id, artifact)
        if source is None:
            raise DatasetSourceMissingError(dataset_id)
        described = self._engine.describe_file(source)
        return DatasetSchema(
            dataset_id=dataset_id,
            session_id=session_id,
            columns=[DatasetColumn(name=name, dtype=dtype) for name, dtype in described],
            source="inferred",
        )

    def get_preview(
        self,
        dataset_id: str,
        session_id: str,
        *,
        limit: int = DEFAULT_PREVIEW_LIMIT,
        offset: int = 0,
    ) -> DatasetPreview:
        limit = max(1, min(limit, MAX_PREVIEW_LIMIT))
        offset = max(0, offset)
        project_id = self._project_for_run(session_id)
        artifact = self._profile_for(project_id, session_id, dataset_id)
        source = self._preview_source(project_id, artifact)
        if source is None:
            raise DatasetSourceMissingError(dataset_id)
        # Fetch one extra row to derive has_more without a COUNT scan.
        columns, rows = self._engine.preview_file(source, limit=limit + 1, offset=offset)
        has_more = len(rows) > limit
        return DatasetPreview(
            dataset_id=dataset_id,
            session_id=session_id,
            columns=columns,
            rows=rows[:limit],
            offset=offset,
            limit=limit,
            has_more=has_more,
            source_format="parquet" if source.suffix.lower() == ".parquet" else "csv",
        )

    def get_distributions(
        self,
        dataset_id: str,
        session_id: str,
        *,
        cancel_check: Callable[[], object] | None = None,
    ) -> ColumnDistributionsView:
        """Per-column mini distributions for the table preview header strip."""
        project_id = self._project_for_run(session_id)
        artifact = self._profile_for(project_id, session_id, dataset_id)
        # The CSV, not _preview_source's Parquet: no Parquet engine is declared
        # as a dependency, and the loader brings the encoding/delimiter sniffing.
        source = self._source_path(project_id, artifact)
        if source is None:
            raise DatasetSourceMissingError(dataset_id)
        cap = DIST_SAMPLE_CAP
        sample, row_count = stream_csv_chunks(
            source,
            lambda chunks: reservoir_sample(
                chunks,
                cap=cap,
                cancel_check=cancel_check,
            ),
        )
        return ColumnDistributionsView(
            dataset_id=dataset_id,
            session_id=session_id,
            row_count=row_count,
            sampled=row_count > cap,
            sample_rows=min(row_count, cap),
            sample_cap=cap,
            bins=DIST_NUMERIC_BINS,
            top_k=DIST_TOP_K,
            columns=[
                _to_column_distribution(dist)
                for dist in column_distributions(
                    sample, sample_cap=cap, bins=DIST_NUMERIC_BINS, top_k=DIST_TOP_K
                )
            ],
        )

    def _project_for_run(self, session_id: str) -> str:
        if INTERNAL_SESSION_MARKER in session_id:
            raise SessionNotFoundError(session_id)
        row = self._store.get_session_index_row(session_id)
        if row is None:
            raise SessionNotFoundError(session_id)
        return str(row["project_id"])

    def _profiles(self, project_id: str, session_id: str) -> list[Artifact]:
        indexed = self._store.list_indexed_artifacts(
            project_id=project_id,
            session_id=session_id,
            artifact_types=(ArtifactType.DATASET_PROFILE,),
        )
        if indexed:
            return indexed
        # Defense layer: the artifacts index now keys on (artifact_id,
        # project_id, session_id), so cross-partition saves no longer steal rows.
        # Keep the on-disk fallback for index rows lost any other way
        # (pre-backfill DBs, manual deletion).
        on_disk, _warnings = self._store.list_artifacts_safe(
            project_id=project_id, session_id=session_id
        )
        return [
            artifact for artifact in on_disk if artifact.type is ArtifactType.DATASET_PROFILE
        ]

    def _profile_for(self, project_id: str, session_id: str, dataset_id: str) -> Artifact:
        for artifact in self._profiles(project_id, session_id):
            if artifact.payload.get("dataset_id") == dataset_id:
                return artifact
        raise DatasetNotFoundError(dataset_id, session_id)

    def _handle_from_profile(
        self,
        project_id: str,
        artifact: Artifact,
        input_hashes: dict[str, str],
    ) -> DatasetHandle:
        payload = artifact.payload
        dataset_id = str(payload.get("dataset_id"))
        name = payload.get("name")
        display_name = name if isinstance(name, str) else dataset_id
        rows = payload.get("rows")
        source = self._source_path(project_id, artifact)
        return DatasetHandle(
            dataset_id=dataset_id,
            project_id=project_id,
            display_name=display_name,
            original_uri=self._workspace_relative(source),
            format=source.suffix.lstrip(".").lower() if source is not None else "csv",
            content_hash=input_hashes.get(display_name, ""),
            byte_size=source.stat().st_size if source is not None else 0,
            row_count=rows if isinstance(rows, int) else None,
            schema=_columns_from_payload(payload),
            ingest_status="ready" if source is not None else "source_missing",
        )

    def _version_dir(self, project_id: str, dataset_id: str) -> Path:
        return self._store.project_dir(project_id) / "uploads" / dataset_id / "v1"

    def _source_path(self, project_id: str, artifact: Artifact) -> Path | None:
        dataset_id = artifact.payload.get("dataset_id")
        name = artifact.payload.get("name")
        if not isinstance(dataset_id, str):
            return None
        version_dir = self._version_dir(project_id, dataset_id)
        located = _locate_source(version_dir, name if isinstance(name, str) else "")
        if located is None:
            return None
        # A symlink escaping projects/ must read as source_missing, not as a
        # handle whose uri/size describe a file elsewhere (state.sqlite included).
        try:
            projects_root = (self._store.root / "projects").resolve()
            if not located.resolve().is_relative_to(projects_root):
                return None
        except OSError:
            return None
        return located

    def _preview_source(self, project_id: str, artifact: Artifact) -> Path | None:
        """Prefer the Parquet copy (§8.3) when the ingest step produced one."""
        dataset_id = artifact.payload.get("dataset_id")
        if isinstance(dataset_id, str):
            parquet_dir = self._version_dir(project_id, dataset_id) / "parquet"
            if parquet_dir.is_dir():
                parquet_files = sorted(parquet_dir.glob("*.parquet"))
                if len(parquet_files) == 1:
                    return parquet_files[0]
        return self._source_path(project_id, artifact)

    def _workspace_relative(self, path: Path | None) -> str:
        if path is None:
            return ""
        try:
            return str(path.resolve().relative_to(self._store.root.resolve()))
        except ValueError:
            return path.name

    def _manifest_input_hashes(self, project_id: str, session_id: str) -> dict[str, str]:
        try:
            manifest = self._store.read_manifest(project_id, session_id)
        except (OSError, ValueError):
            return {}
        if manifest is None:
            return {}
        return {
            str(name): value
            for name, value in manifest.input_hashes.items()
            if isinstance(value, str) and value
        }


def _to_column_distribution(dist: dict[str, Any]) -> ColumnDistribution:
    """One row of column_distributions() as a typed DTO; the numeric and
    categorical branches fill disjoint field groups."""
    top = dist.get("top")
    return ColumnDistribution(
        name=str(dist.get("name", "")),
        dtype=str(dist.get("dtype", "")),
        kind=str(dist.get("kind", "empty")),
        missing_percent=float(dist.get("missing_percent") or 0.0),
        counts=[int(count) for count in dist["counts"]] if "counts" in dist else None,
        bin_edges=[float(edge) for edge in dist["bin_edges"]] if "bin_edges" in dist else None,
        min=float(dist["min"]) if "min" in dist else None,
        max=float(dist["max"]) if "max" in dist else None,
        top=[DistributionCategory(value=str(label), count=int(count)) for label, count in top]
        if isinstance(top, list)
        else None,
        other_count=int(dist["other_count"]) if "other_count" in dist else None,
        unique_count=int(dist["unique_count"]) if "unique_count" in dist else None,
        len_min=int(dist["len_min"]) if "len_min" in dist else None,
        len_max=int(dist["len_max"]) if "len_max" in dist else None,
    )


def _columns_from_payload(payload: dict[str, object]) -> list[DatasetColumn]:
    column_names = payload.get("column_names")
    if not isinstance(column_names, list) or not column_names:
        return []
    dtypes = payload.get("dtypes")
    dtype_map = dtypes if isinstance(dtypes, dict) else {}
    return [
        DatasetColumn(name=str(name), dtype=str(dtype_map.get(name, "unknown")))
        for name in column_names
    ]


def _locate_source(version_dir: Path, name: str) -> Path | None:
    """Same tolerance as core.session_loader: exact recorded name, else lone CSV."""
    # A poisoned profile could carry an absolute or path-shaped name; joining
    # it below would stat outside version_dir. Treat such names as unrecorded.
    if name and (Path(name).is_absolute() or Path(name).name != name):
        name = ""
    if name:
        exact = version_dir / name
        if exact.is_file():
            return exact
    if not version_dir.is_dir():
        return None
    csvs = sorted(version_dir.glob("*.csv"))
    if len(csvs) == 1:
        return csvs[0]
    return None
