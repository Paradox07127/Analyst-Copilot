"""Quality/Profiles/Charts read use cases (§10.2 P1): aggregation reuses the
pure display-shaping helpers in application.workbench; chart listings are
summaries only — the full vega-lite spec is served per chart on demand."""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, cast

import pandas as pd

from eda_platform.application.chart_builder import (
    CUSTOM_CHART_ROW_LIMIT,
    ROW_COUNT_Y,
    apply_outlier_bounds,
    custom_chart_spec,
    default_custom_agg,
    select_chart_columns,
)
from eda_platform.application.dto import (
    ChartSummary,
    ChartView,
    CustomChartRequest,
    CustomChartView,
    DatasetProfileSummary,
    FieldProfileRow,
    Page,
    ProfilesView,
    QualityDatasetCard,
    QualityIssueRow,
    QualityView,
)
from eda_platform.application.services.artifact_service import (
    MAX_ARTIFACT_PAYLOAD_BYTES,
    _decode_cursor,
    _encode_cursor,
)
from eda_platform.application.services.dataset_service import (
    DatasetNotFoundError,
    DatasetService,
    DatasetSourceMissingError,
)
from eda_platform.application.services.session_service import SessionNotFoundError
from eda_platform.application.workbench import (
    dataset_display_rows,
    dataset_names_by_id,
    group_quality_issues,
    semantic_type_counts,
)
from eda_platform.application.workspace_paths import (
    relativize_warnings,
    relativize_workspace_paths,
)
from eda_platform.core.ids import INTERNAL_SESSION_MARKER
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType, DatasetProfile
from eda_platform.schemas.charts import ChartSpec
from eda_platform.tools.frame_stats import iqr_bounds
from eda_platform.tools.loader import read_csv_columns, stream_csv_chunks

DEFAULT_CHART_LIMIT = 50
MAX_CHART_LIMIT = 100
DATASET_NAMES_TTL_SECONDS = 30.0
VEGALITE_SCHEMA_URL = "https://vega.github.io/schema/vega-lite/v6.json"
# Rows are capped first; this bounds a response whose few columns hold free text.
MAX_INLINE_DATA_BYTES = 2_000_000

_SEVERITIES = ("critical", "warn", "info")


class ChartNotFoundError(Exception):
    def __init__(self, chart_id: str) -> None:
        super().__init__(f"Chart not found: {chart_id}")
        self.chart_id = chart_id


class CustomChartValidationError(Exception):
    """Rejected chart options. Raised with a message safe to echo: it only ever
    names columns and options the caller already sent."""

    error_code = "custom_chart_invalid"


class InsightService:
    def __init__(self, store: ArtifactStore) -> None:
        self._store = store
        self._dataset_names_cache: dict[tuple[str, str], tuple[float, dict[str, str]]] = {}

    def get_quality(self, session_id: str) -> QualityView:
        project_id = self._project_for_run(session_id)
        artifacts = self._store.list_indexed_artifacts(
            project_id=project_id,
            session_id=session_id,
            artifact_types=(ArtifactType.QUALITY_ISSUE_SET, ArtifactType.DATASET_PROFILE),
        )
        id_to_name = dataset_names_by_id(artifacts)
        name_counts = Counter(id_to_name.values())

        def display_name(dataset_id: str) -> str:
            name = id_to_name.get(dataset_id, dataset_id)
            # Same-named datasets keep distinct cards by appending the id.
            return f"{name} ({dataset_id})" if name_counts[name] > 1 else name

        # Grouped per issue-set artifact: the workbench helper drops the
        # dataset_id, which is needed to attribute counts when names collide.
        rows_by_severity: dict[str, list[tuple[str, dict[str, str | None]]]] = {
            severity: [] for severity in _SEVERITIES
        }
        for artifact in artifacts:
            if artifact.type is not ArtifactType.QUALITY_ISSUE_SET:
                continue
            dataset_id = str(artifact.payload.get("dataset_id") or "")
            grouped = group_quality_issues([artifact])
            for severity in _SEVERITIES:
                for row in grouped[severity]:
                    rows_by_severity[severity].append((dataset_id, row))
        issues: list[QualityIssueRow] = []
        cards: dict[str, QualityDatasetCard] = {}
        for severity in _SEVERITIES:
            for dataset_id, row in rows_by_severity[severity]:
                dataset_name = display_name(dataset_id)
                issues.append(
                    QualityIssueRow(
                        severity=severity,
                        dataset_name=dataset_name,
                        dataset_id=dataset_id,
                        code=str(row.get("code") or ""),
                        column=row.get("column"),
                        message=str(row.get("message") or ""),
                        recommendation=str(row.get("recommendation") or ""),
                    )
                )
                card = cards.setdefault(
                    dataset_id,
                    QualityDatasetCard(dataset_name=dataset_name, dataset_id=dataset_id),
                )
                if severity == "critical":
                    card.critical += 1
                elif severity == "warn":
                    card.warn += 1
                else:
                    card.info += 1
        return QualityView(
            session_id=session_id,
            critical=len(rows_by_severity["critical"]),
            warn=len(rows_by_severity["warn"]),
            info=len(rows_by_severity["info"]),
            datasets=list(cards.values()),
            issues=issues,
        )

    def get_profiles(self, session_id: str) -> ProfilesView:
        project_id = self._project_for_run(session_id)
        artifacts = self._store.list_indexed_artifacts(
            project_id=project_id,
            session_id=session_id,
            artifact_types=(ArtifactType.DATASET_PROFILE,),
        )
        datasets = []
        for artifact in artifacts:
            profile = DatasetProfile.model_validate(artifact.payload)
            datasets.append(
                DatasetProfileSummary(
                    dataset_id=profile.dataset_id,
                    name=profile.name,
                    rows=profile.rows,
                    columns=profile.columns,
                    semantic_type_counts=semantic_type_counts(artifact),
                    fields=[
                        FieldProfileRow.model_validate(row)
                        for row in dataset_display_rows(artifact)
                    ],
                )
            )
        return ProfilesView(session_id=session_id, datasets=datasets)

    def list_charts(
        self,
        session_id: str,
        *,
        limit: int = DEFAULT_CHART_LIMIT,
        cursor: str | None = None,
    ) -> Page[ChartSummary]:
        limit = max(1, min(limit, MAX_CHART_LIMIT))
        project_id = self._project_for_run(session_id)
        chart_type = ArtifactType.CHART_SPEC.value
        last_rowid = _decode_cursor(cursor, chart_type, session_id) if cursor else None
        dataset_names = self._dataset_names(project_id, session_id)
        # Invalid rows are skipped, so keep scanning batches until the page
        # holds `limit` valid summaries or the index runs out; the cursor always
        # points at the last row consumed, never a row that was skipped over.
        items: list[ChartSummary] = []
        next_cursor: str | None = None
        batch_size = limit + 1
        while True:
            rows = self._store.query_artifact_index_rows(
                project_id,
                session_id,
                artifact_type=chart_type,
                limit=batch_size,
                after_rowid=last_rowid,
            )
            exhausted = len(rows) < batch_size
            filled = False
            for index, row in enumerate(rows):
                last_rowid = row["rowid"]
                summary = self._chart_summary(row, project_id, session_id, dataset_names)
                if summary is None:
                    continue
                items.append(summary)
                if len(items) == limit:
                    if index + 1 < len(rows) or not exhausted:
                        next_cursor = _encode_cursor(last_rowid, chart_type, session_id)
                    filled = True
                    break
            if filled or exhausted:
                break
        return Page[ChartSummary](items=items, next_cursor=next_cursor)

    def _chart_summary(
        self,
        row: dict,
        project_id: str,
        session_id: str,
        dataset_names: dict[str, str],
    ) -> ChartSummary | None:
        artifact = self._read_artifact_row(row, project_id=project_id, session_id=session_id)
        if artifact is None:
            return None
        try:
            spec = ChartSpec.model_validate(artifact.payload)
        except ValueError:
            return None
        return ChartSummary(
            artifact_id=artifact.id,
            title=spec.title,
            dataset_id=spec.dataset_id,
            dataset_name=dataset_names.get(spec.dataset_id, spec.dataset_id),
            mark=spec.mark,
            fields=_encoding_fields(spec.encoding),
            description=spec.description,
        )

    def get_chart(self, session_id: str, chart_id: str) -> ChartView:
        project_id = self._project_for_run(session_id)
        row = self._store.artifact_index_row(
            chart_id, project_id=project_id, session_id=session_id
        )
        if (
            row is None
            or row["artifact_type"] != ArtifactType.CHART_SPEC.value
            or INTERNAL_SESSION_MARKER in str(row["session_id"])
        ):
            raise ChartNotFoundError(chart_id)
        artifact = self._read_artifact_row(
            row, project_id=str(row["project_id"]), session_id=str(row["session_id"])
        )
        if artifact is None:
            raise ChartNotFoundError(chart_id)
        try:
            spec = ChartSpec.model_validate(artifact.payload)
        except ValueError:
            raise ChartNotFoundError(chart_id) from None
        # Defensive boundary: artifact payloads are untrusted, and any data key
        # besides inline "values" (url/format/name/...) would make the client
        # renderer fetch external resources. Production chart producers only
        # ever emit inline values, so a non-conforming spec is unservable.
        if spec.data and set(spec.data) - {"values"}:
            raise ChartNotFoundError(chart_id)
        vegalite = spec.to_vegalite()
        # Same boundary for vega expression strings (expr/labelExpr/signal,
        # condition tests): they execute in the client's vega runtime, and no
        # production chart producer emits them.
        if _contains_vega_expression(vegalite):
            raise ChartNotFoundError(chart_id)
        dataset_names = self._dataset_names(artifact.project_id, artifact.session_id)
        return ChartView(
            artifact_id=artifact.id,
            session_id=artifact.session_id,
            title=spec.title,
            dataset_id=spec.dataset_id,
            dataset_name=dataset_names.get(spec.dataset_id, spec.dataset_id),
            description=spec.description,
            plain_language=artifact.plain_language,
            spec=vegalite,
        )

    def build_custom_chart(
        self,
        session_id: str,
        options: CustomChartRequest,
        *,
        datasets: DatasetService,
        cancel_check: Callable[[], object] | None = None,
    ) -> CustomChartView:
        """Ad-hoc chart over one of the run's datasets: the same shaping the
        custom chart builder does, returned as a self-contained spec."""
        source = self._custom_chart_source(session_id, options.dataset_id, datasets=datasets)
        self._validate_custom_options(options, read_csv_columns(source))

        y_column = options.y_column
        # De-duplicate while preserving order: one column may be X, Y and Color.
        selected_columns = list(
            dict.fromkeys(
                [options.x_column]
                + ([] if y_column is None else [y_column])
                + ([] if options.color_column is None else [options.color_column])
            )
        )
        capped, source_row_count = _stream_chart_frame(
            source,
            selected_columns=selected_columns,
            y_column=y_column,
            drop_missing=options.drop_missing,
            # A histogram's fence belongs to X and is applied while binning; a Y
            # fence here would filter on a column the chart never draws.
            drop_outliers=options.drop_outliers and options.chart_type != "histogram",
            limit=CUSTOM_CHART_ROW_LIMIT,
            cancel_check=cancel_check,
        )
        # Dtype-dependent checks run on the rows that will actually be charted:
        # a chunked read has no whole-file dtype to consult before reading.
        if options.chart_type == "histogram" and not pd.api.types.is_numeric_dtype(
            capped[options.x_column]
        ):
            raise CustomChartValidationError(
                f"A histogram needs a numeric X column; {options.x_column} is not numeric."
            )
        aggregate = options.aggregate or default_custom_agg(capped, y_column or ROW_COUNT_Y)
        # A histogram and a row-count Y always count, whatever was requested;
        # report what the spec does rather than what was asked for.
        if y_column is None or options.chart_type == "histogram":
            aggregate = "count"
        full_aggregate = options.chart_type == "histogram" or aggregate != "none"
        if options.chart_type == "histogram":
            chart_frame, spec = _full_histogram_chart(
                source,
                x_column=options.x_column,
                drop_missing=options.drop_missing,
                drop_outliers=options.drop_outliers,
                cancel_check=cancel_check,
            )
            # Bins are the whole population the chart describes, so report what
            # was binned rather than the pre-fence scan count.
            source_row_count = (
                int(cast("Any", chart_frame["count"]).sum()) if len(chart_frame) else 0
            )
        elif full_aggregate:
            output_y = y_column or ROW_COUNT_Y
            if output_y in {
                options.x_column,
                *([] if options.color_column is None else [options.color_column]),
            }:
                output_y = "Aggregated value"
            chart_frame = _stream_grouped_chart(
                source,
                selected_columns=selected_columns,
                x_column=options.x_column,
                y_column=y_column,
                color_column=options.color_column,
                output_y=output_y,
                aggregate=aggregate,
                drop_missing=options.drop_missing,
                drop_outliers=options.drop_outliers,
                cancel_check=cancel_check,
            )
            spec = custom_chart_spec(
                chart_type=options.chart_type,
                x_column=options.x_column,
                y_column=output_y,
                color_column=options.color_column,
                aggregate="none",
                frame=chart_frame,
            )
            encoding = spec.get("encoding")
            y_encoding = encoding.get("y") if isinstance(encoding, dict) else None
            if isinstance(y_encoding, dict):
                y_encoding["title"] = (
                    f"{aggregate}({y_column})" if y_column else ROW_COUNT_Y
                )
        else:
            chart_frame = capped
            spec = custom_chart_spec(
                chart_type=options.chart_type,
                x_column=options.x_column,
                y_column=y_column,
                color_column=options.color_column,
                aggregate=aggregate,
                frame=chart_frame,
            )
        # Scanned before the rows go in: data values are keyed by column name, so
        # scanning them would reject a dataset with a column named "expr" without
        # adding safety — vega evaluates expressions in the spec, not in the data.
        if _contains_vega_expression(spec):
            raise CustomChartValidationError(
                "The generated chart spec was rejected by the expression safety check."
            )
        values, size_capped = _inline_values(chart_frame)
        spec["$schema"] = VEGALITE_SCHEMA_URL
        spec["data"] = {"values": values}
        return CustomChartView(
            session_id=session_id,
            dataset_id=options.dataset_id,
            chart_type=options.chart_type,
            aggregate=aggregate,
            # Keep row_count as the number of filtered source observations.
            # Aggregated charts intentionally inline one row per group/bin, so
            # len(values) would make the API claim that observations vanished.
            row_count=source_row_count if full_aggregate else len(values),
            source_row_count=source_row_count,
            truncated=(not full_aggregate and source_row_count > CUSTOM_CHART_ROW_LIMIT)
            or size_capped,
            row_limit=CUSTOM_CHART_ROW_LIMIT,
            spec=spec,
        )

    def _custom_chart_source(
        self, session_id: str, dataset_id: str, *, datasets: DatasetService
    ) -> Path:
        handle = next(
            (item for item in datasets.list_datasets(session_id) if item.dataset_id == dataset_id),
            None,
        )
        if handle is None:
            raise DatasetNotFoundError(dataset_id, session_id)
        source = self._store.root / handle.original_uri if handle.original_uri else None
        # The uri arrives as a string through a DTO; re-check containment rather
        # than trust that it still resolves inside the workspace.
        if source is None or not _inside_projects(source, self._store.root):
            raise DatasetSourceMissingError(dataset_id)
        return source

    def _validate_custom_options(
        self, options: CustomChartRequest, column_names: list[str]
    ) -> None:
        columns = set(column_names)
        for column in (options.x_column, options.y_column, options.color_column):
            if column is not None and column not in columns:
                raise CustomChartValidationError(
                    f"Column is not part of this dataset: {column}"
                )
        # A histogram fences its own X distribution, so it needs no Y at all.
        if (
            options.drop_outliers
            and options.y_column is None
            and options.chart_type != "histogram"
        ):
            raise CustomChartValidationError(
                "drop_outliers needs a Y column; it cannot apply to a row count."
            )

    def _dataset_names(self, project_id: str, session_id: str) -> dict[str, str]:
        """The profile scan reads O(datasets) files; a short TTL absorbs the burst
        of per-chart detail requests. No lock: under FastAPI's thread pool the
        worst race is a harmless duplicate recompute."""
        key = (project_id, session_id)
        now = time.monotonic()
        cached = self._dataset_names_cache.get(key)
        if cached is not None and now < cached[0]:
            return cached[1]
        profiles = self._store.list_indexed_artifacts(
            project_id=project_id,
            session_id=session_id,
            artifact_types=(ArtifactType.DATASET_PROFILE,),
        )
        names = dataset_names_by_id(profiles)
        self._dataset_names_cache[key] = (now + DATASET_NAMES_TTL_SECONDS, names)
        return names

    def _project_for_run(self, session_id: str) -> str:
        if INTERNAL_SESSION_MARKER in session_id:
            raise SessionNotFoundError(session_id)
        row = self._store.get_session_index_row(session_id)
        if row is None:
            raise SessionNotFoundError(session_id)
        return str(row["project_id"])

    def _read_artifact_row(self, row: dict, *, project_id: str, session_id: str) -> Artifact | None:
        """Read an indexed artifact with the same guards as ArtifactService.get_artifact:
        workspace containment, size cap, envelope identity, path relativization."""
        path: Path = row["path"]
        root = self._store.root
        try:
            if not path.resolve().is_relative_to(root.resolve()):
                return None
            if path.stat().st_size > MAX_ARTIFACT_PAYLOAD_BYTES:
                return None
            artifact = Artifact.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        # A swapped or aliased file must not serve under this index row.
        if (
            artifact.id != str(row["artifact_id"])
            or artifact.project_id != project_id
            or artifact.session_id != session_id
        ):
            return None
        return artifact.model_copy(
            update={
                "payload": relativize_workspace_paths(artifact.payload, root),
                "warnings": relativize_warnings(list(artifact.warnings), root),
            }
        )


def _inside_projects(path: Path, root: Path) -> bool:
    try:
        return path.is_file() and path.resolve().is_relative_to((root / "projects").resolve())
    except OSError:
        return False


def _stream_chart_frame(
    source: Path,
    *,
    selected_columns: list[str],
    y_column: str | None,
    drop_missing: bool,
    drop_outliers: bool,
    limit: int,
    cancel_check: Callable[[], object] | None = None,
) -> tuple[pd.DataFrame, int]:
    """Table shaping: clean the whole table, then head(limit), done over
    row chunks, returning the capped frame and the pre-cap row count.

    The IQR fence is a whole-column statistic, so ``drop_outliers`` costs a second
    pass rather than a fence derived from whatever rows had been read so far.
    """
    bounds: tuple[float, float] | None = None
    if drop_outliers and y_column is not None:
        fence_column = y_column
        fence_input = stream_csv_chunks(
            source,
            lambda chunks: _fence_input(
                chunks,
                selected_columns=selected_columns,
                y_column=fence_column,
                drop_missing=drop_missing,
                cancel_check=cancel_check,
            ),
            usecols=selected_columns,
        )
        bounds = iqr_bounds(fence_input)
    return stream_csv_chunks(
        source,
        lambda chunks: _collect_chart_head(
            chunks,
            selected_columns=selected_columns,
            y_column=y_column,
            bounds=bounds,
            drop_missing=drop_missing,
            limit=limit,
            cancel_check=cancel_check,
        ),
        usecols=selected_columns,
    )


def _stream_grouped_chart(
    source: Path,
    *,
    selected_columns: list[str],
    x_column: str,
    y_column: str | None,
    color_column: str | None,
    output_y: str,
    aggregate: str,
    drop_missing: bool,
    drop_outliers: bool,
    cancel_check: Callable[[], object] | None,
) -> pd.DataFrame:
    bounds: tuple[float, float] | None = None
    if drop_outliers and y_column is not None:
        values = stream_csv_chunks(
            source,
            lambda chunks: _fence_input(
                chunks,
                selected_columns=selected_columns,
                y_column=y_column,
                drop_missing=drop_missing,
                cancel_check=cancel_check,
            ),
            usecols=selected_columns,
        )
        bounds = iqr_bounds(values)
    key_columns = [x_column, *([color_column] if color_column else [])]

    def consume(chunks: Iterator[pd.DataFrame]) -> pd.DataFrame:
        states: dict[tuple[object, ...], dict[str, Any]] = {}
        for chunk in chunks:
            if cancel_check is not None:
                cancel_check()
            working = select_chart_columns(
                chunk,
                selected_columns,
                drop_missing=drop_missing,
            )
            if bounds is not None and y_column is not None:
                working = apply_outlier_bounds(working, y_column, bounds)
            if y_column is not None:
                working[y_column] = pd.to_numeric(working[y_column], errors="coerce")
            group_key: str | list[str] = (
                key_columns[0] if len(key_columns) == 1 else key_columns
            )
            for raw_key, group in working.groupby(group_key, dropna=False, sort=False):
                key_values = raw_key if isinstance(raw_key, tuple) else (raw_key,)
                key = tuple(
                    None if cast("bool", pd.isna(value)) else value
                    for value in key_values
                )
                state = states.setdefault(key, {"rows": 0, "sum": 0.0, "count": 0, "values": []})
                state["rows"] += len(group)
                if y_column is None:
                    continue
                numeric = cast("pd.Series", group[y_column]).dropna()
                state["sum"] += float(numeric.sum())
                state["count"] += len(numeric)
                if aggregate == "median":
                    state["values"].extend(float(value) for value in numeric)
        rows: list[dict[str, Any]] = []
        for key, state in states.items():
            row = dict(zip(key_columns, key, strict=True))
            if aggregate == "count":
                row[output_y] = int(state["rows"])
            elif aggregate == "sum":
                row[output_y] = float(state["sum"])
            elif aggregate == "mean":
                row[output_y] = (
                    float(state["sum"]) / int(state["count"])
                    if state["count"]
                    else None
                )
            elif aggregate == "median":
                row[output_y] = (
                    float(pd.Series(state["values"]).median())
                    if state["values"]
                    else None
                )
            rows.append(row)
        return pd.DataFrame(rows, columns=[*key_columns, output_y])

    return stream_csv_chunks(source, consume, usecols=selected_columns)


def _full_histogram_chart(
    source: Path,
    *,
    x_column: str,
    drop_missing: bool,
    drop_outliers: bool,
    cancel_check: Callable[[], object] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    numeric = cast(
        "pd.Series",
        stream_csv_chunks(
            source,
            lambda chunks: _fence_input(
                chunks,
                selected_columns=[x_column],
                y_column=x_column,
                drop_missing=drop_missing,
                cancel_check=cancel_check,
            ),
            usecols=[x_column],
        ),
    ).dropna()
    if drop_outliers:
        bounds = iqr_bounds(numeric)
        if bounds is not None:
            numeric = cast("pd.Series", numeric[numeric.between(*bounds)])
    if numeric.empty:
        frame = pd.DataFrame(columns=["bin_start", "bin_end", "count"])
    else:
        bins = min(30, max(1, int(numeric.nunique())))
        buckets = cast("pd.Series", pd.cut(numeric, bins=bins))
        counts = buckets.value_counts(sort=False)
        frame = pd.DataFrame(
            [
                {
                    "bin_start": float(cast("pd.Interval", interval).left),
                    "bin_end": float(cast("pd.Interval", interval).right),
                    "count": int(count),
                }
                for interval, count in counts.items()
            ]
        )
    spec = {
        "mark": "bar",
        "encoding": {
            "x": {
                "field": "bin_start",
                "type": "quantitative",
                "bin": {"binned": True},
                "title": x_column,
            },
            "x2": {"field": "bin_end"},
            "y": {"field": "count", "type": "quantitative"},
        },
    }
    return frame, spec


def _fence_input(
    chunks: Iterator[pd.DataFrame],
    *,
    selected_columns: list[str],
    y_column: str,
    drop_missing: bool,
    cancel_check: Callable[[], object] | None = None,
) -> pd.Series:
    parts: list[pd.Series] = []
    for chunk in chunks:
        if cancel_check is not None:
            cancel_check()
        parts.append(
            cast(
                "pd.Series",
                pd.to_numeric(
                    select_chart_columns(
                        chunk, selected_columns, drop_missing=drop_missing
                    )[y_column],
                    errors="coerce",
                ),
            )
        )
    return pd.concat(parts, ignore_index=True) if parts else pd.Series(dtype="float64")


def _collect_chart_head(
    chunks: Iterator[pd.DataFrame],
    *,
    selected_columns: list[str],
    y_column: str | None,
    bounds: tuple[float, float] | None,
    drop_missing: bool,
    limit: int,
    cancel_check: Callable[[], object] | None = None,
) -> tuple[pd.DataFrame, int]:
    kept: list[pd.DataFrame] = []
    kept_rows = 0
    total = 0
    for chunk in chunks:
        if cancel_check is not None:
            cancel_check()
        cleaned = select_chart_columns(chunk, selected_columns, drop_missing=drop_missing)
        if bounds is not None and y_column is not None:
            cleaned = apply_outlier_bounds(cleaned, y_column, bounds)
        total += len(cleaned)
        if kept_rows < limit:
            head = cleaned.head(limit - kept_rows)
            kept.append(head)
            kept_rows += len(head)
    frame = (
        pd.concat(kept, ignore_index=True)
        if kept
        else pd.DataFrame(columns=pd.Index(selected_columns))
    )
    return frame, total


def _inline_values(frame: pd.DataFrame) -> tuple[list[dict[str, Any]], bool]:
    """Rows for `spec.data.values`, stopped at the byte budget. The JSON
    round-trip is what keeps NaN/Timestamp cells vega-safe."""
    encoded = cast("str", frame.to_json(orient="records", date_format="iso"))
    records = cast("list[dict[str, Any]]", json.loads(encoded))
    values: list[dict[str, Any]] = []
    budget = MAX_INLINE_DATA_BYTES
    for record in records:
        budget -= len(json.dumps(record, ensure_ascii=False).encode("utf-8"))
        if budget < 0:
            return values, True
        values.append(record)
    return values, False


_EXPRESSION_KEYS = frozenset({"expr", "labelExpr", "signal"})


def _contains_vega_expression(node: object) -> bool:
    """True if any level of the spec carries a vega expression construct."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key in _EXPRESSION_KEYS:
                return True
            if key == "condition":
                conditions = value if isinstance(value, list) else [value]
                for condition in conditions:
                    if isinstance(condition, dict) and isinstance(condition.get("test"), str):
                        return True
            if _contains_vega_expression(value):
                return True
        return False
    if isinstance(node, list):
        return any(_contains_vega_expression(item) for item in node)
    return False


def _encoding_fields(encoding: dict[str, object]) -> list[str]:
    """Field names referenced by the encoding channels, in channel order."""
    fields: list[str] = []
    for channel in encoding.values():
        if isinstance(channel, dict):
            field = channel.get("field")
            if isinstance(field, str) and field not in fields:
                fields.append(field)
    return fields
