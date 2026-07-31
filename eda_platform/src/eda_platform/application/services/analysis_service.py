"""Deep analysis read use case (§10.2 P1): reshapes the AnalysisTable,
StatTestResult and ModelCard artifacts a run already persisted. Nothing is
recomputed — statistics, effect sizes and metrics are served at stored
precision, and only display-level derivations (p-value formatting,
significance at alpha=0.05, effect magnitude) are added."""

from __future__ import annotations

from collections.abc import Callable

from eda_platform.application.dto import (
    AnalysisTableView,
    AnalysisView,
    ModelCardFeature,
    ModelCardLeakageCheck,
    ModelCardView,
    StatTestRow,
)
from eda_platform.application.services.session_service import SessionNotFoundError
from eda_platform.application.workbench import (
    SMALL_SAMPLE_THRESHOLD,
    dataset_names_by_id,
    effect_size_magnitude_badge,
    format_p_value,
    headline_metric,
    leakage_verdict_badge,
    min_sample_size,
    split_trivial_correlation_rows,
)
from eda_platform.application.workspace_paths import relativize_workspace_paths
from eda_platform.core.ids import INTERNAL_SESSION_MARKER
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import AnalysisTable, Artifact, ArtifactType
from eda_platform.schemas.model_card import ModelCard
from eda_platform.schemas.stats import StatTestResult

SIGNIFICANCE_ALPHA = 0.05

BASELINE_QUESTION_LABEL = "Baseline EDA (not tied to a selected question card)"

_ANALYSIS_TYPES = (
    ArtifactType.TABLE,
    ArtifactType.STAT_TEST_RESULT,
    ArtifactType.MODEL_CARD,
    ArtifactType.DATASET_PROFILE,
)

# Lineage walks are bounded: a malformed parent chain must not turn one page
# request into an unbounded artifact scan.
_MAX_LINEAGE_VISITS = 64


class AnalysisService:
    def __init__(self, store: ArtifactStore) -> None:
        self._store = store

    def get_analysis(self, session_id: str) -> AnalysisView:
        project_id = self._project_for_run(session_id)
        artifacts = [
            self._relativized(artifact)
            for artifact in self._store.list_indexed_artifacts(
                project_id=project_id, session_id=session_id, artifact_types=_ANALYSIS_TYPES
            )
        ]
        dataset_names = dataset_names_by_id(artifacts)

        def name_of(dataset_id: str) -> str:
            return dataset_names.get(dataset_id, dataset_id)

        lineage = _LineageResolver(self._store, project_id=project_id, session_id=session_id)
        tables: list[AnalysisTableView] = []
        stat_tests: list[StatTestRow] = []
        model_cards: list[ModelCardView] = []
        for artifact in artifacts:
            # Malformed payloads are skipped rather than failing the page: one
            # bad artifact must not hide the rest of the run's analysis.
            if artifact.type is ArtifactType.TABLE:
                table = _table_view(artifact, name_of, lineage)
                if table is not None:
                    tables.append(table)
            elif artifact.type is ArtifactType.STAT_TEST_RESULT:
                stat_test = _stat_test_row(artifact, name_of)
                if stat_test is not None:
                    stat_tests.append(stat_test)
            elif artifact.type is ArtifactType.MODEL_CARD:
                model_card = _model_card_view(artifact, name_of)
                if model_card is not None:
                    model_cards.append(model_card)
        tables.sort(key=lambda item: (item.dataset_name, item.title))
        return AnalysisView(
            session_id=session_id, tables=tables, stat_tests=stat_tests, model_cards=model_cards
        )

    def _relativized(self, artifact: Artifact) -> Artifact:
        return artifact.model_copy(
            update={"payload": relativize_workspace_paths(artifact.payload, self._store.root)}
        )

    def _project_for_run(self, session_id: str) -> str:
        if INTERNAL_SESSION_MARKER in session_id:
            raise SessionNotFoundError(session_id)
        row = self._store.get_session_index_row(session_id)
        if row is None:
            raise SessionNotFoundError(session_id)
        return str(row["project_id"])


class _LineageResolver:
    """Resolves an artifact's source question by walking parents on demand.

    The former UI held every session artifact in memory to do this; here only
    the parents actually visited are read, and each read is cached."""

    def __init__(self, store: ArtifactStore, *, project_id: str, session_id: str) -> None:
        self._store = store
        self._project_id = project_id
        self._session_id = session_id
        self._cache: dict[str, Artifact | None] = {}

    def question_label(self, artifact: Artifact) -> str:
        pending = list(artifact.parents)
        visited: set[str] = set()
        while pending and len(visited) < _MAX_LINEAGE_VISITS:
            artifact_id = pending.pop(0)
            if artifact_id in visited:
                continue
            visited.add(artifact_id)
            parent = self._get(artifact_id)
            if parent is None:
                continue
            if parent.type is ArtifactType.QUESTION_EXECUTION_RESULT:
                question = parent.payload.get("question")
                if isinstance(question, str) and question.strip():
                    return question.strip()
            pending.extend(parent.parents)
        return BASELINE_QUESTION_LABEL

    def _get(self, artifact_id: str) -> Artifact | None:
        if artifact_id not in self._cache:
            try:
                self._cache[artifact_id] = self._store.get_artifact(
                    artifact_id, project_id=self._project_id, session_id=self._session_id
                )
            except (KeyError, OSError, ValueError):
                self._cache[artifact_id] = None
        return self._cache[artifact_id]


def _table_view(
    artifact: Artifact, name_of: Callable[[str], str], lineage: _LineageResolver
) -> AnalysisTableView | None:
    try:
        table = AnalysisTable.model_validate(artifact.payload)
    except ValueError:
        return None
    if table.kind == "correlation":
        rows, trivial_rows = split_trivial_correlation_rows(table.rows)
    else:
        rows, trivial_rows = [dict(row) for row in table.rows], []
    sample = min_sample_size(table.rows)
    return AnalysisTableView(
        artifact_id=artifact.id,
        dataset_id=table.dataset_id,
        dataset_name=name_of(table.dataset_id),
        title=table.title,
        kind=table.kind,
        description=table.description,
        question=lineage.question_label(artifact),
        columns=_column_order(table.rows),
        rows=rows,
        trivial_rows=trivial_rows,
        min_sample_size=sample,
        small_sample=sample is not None and sample < SMALL_SAMPLE_THRESHOLD,
    )


def _column_order(rows: list[dict[str, object]]) -> list[str]:
    """Union of row keys in first-seen order — rows are heterogeneous dicts."""
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    return columns


def _stat_test_row(artifact: Artifact, name_of: Callable[[str], str]) -> StatTestRow | None:
    try:
        result = StatTestResult.model_validate(artifact.payload)
    except ValueError:
        return None
    magnitude = effect_size_magnitude_badge(result.test_type, result.effect_size)
    significant = None if result.p_value is None else result.p_value < SIGNIFICANCE_ALPHA
    return StatTestRow(
        artifact_id=artifact.id,
        dataset_id=result.dataset_id,
        dataset_name=name_of(result.dataset_id),
        test_type=result.test_type,
        group_column=result.group_column,
        value_column=result.value_column or result.category_column,
        statistic=result.statistic,
        p_value=result.p_value,
        p_value_display=format_p_value(result.p_value),
        effect_size=result.effect_size,
        effect_size_magnitude=magnitude or None,
        degrees_of_freedom=result.degrees_of_freedom,
        sample_size=result.sample_size,
        significant=significant,
        conclusion=_conclusion(significant, magnitude),
        groups=dict(result.groups),
        warnings=[warning.code for warning in result.warnings],
        small_sample=result.sample_size < SMALL_SAMPLE_THRESHOLD,
    )


def _conclusion(significant: bool | None, magnitude: str) -> str:
    if significant is None:
        return "No p-value reported — not interpretable."
    verdict = (
        f"Significant at alpha={SIGNIFICANCE_ALPHA}"
        if significant
        else f"Not significant at alpha={SIGNIFICANCE_ALPHA}"
    )
    return f"{verdict} ({magnitude} effect)" if magnitude else verdict


def _model_card_view(
    artifact: Artifact, name_of: Callable[[str], str]
) -> ModelCardView | None:
    try:
        card = ModelCard.model_validate(artifact.payload)
    except ValueError:
        return None
    verdict = leakage_verdict_badge(card.leakage_checks)
    headline = headline_metric(card)
    return ModelCardView(
        artifact_id=artifact.id,
        dataset_id=card.dataset_id,
        dataset_name=name_of(card.dataset_id),
        task_type=card.task_type,
        target_column=card.target_column,
        model_type=card.model_type,
        split_strategy=card.split_strategy,
        train_rows=card.train_rows,
        test_rows=card.test_rows,
        feature_columns=list(card.feature_columns),
        excluded_features=list(card.excluded_features),
        metrics=dict(card.metrics),
        headline_metric=None if headline is None else headline[0],
        headline_metric_value=None if headline is None else headline[1],
        baseline_accuracy=card.baseline_accuracy,
        leakage_verdict=verdict,
        leakage_checks=[
            ModelCardLeakageCheck(
                code=check.code,
                severity=check.severity,
                column=check.column,
                action=check.action,
                message=check.message,
            )
            for check in card.leakage_checks
        ],
        feature_importance=[
            ModelCardFeature(feature=item.feature, importance=item.importance)
            for item in card.feature_importance
        ],
        limitations=list(card.limitations),
    )
