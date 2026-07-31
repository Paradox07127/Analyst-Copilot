from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from eda_platform.schemas.artifacts import (
    AnalysisTable,
    Artifact,
    ArtifactType,
    DatasetProfile,
    QualityIssueSet,
)
from eda_platform.schemas.charts import ChartSpec
from eda_platform.schemas.model_card import ModelCard
from eda_platform.schemas.stats import StatTestResult

PayloadPolicy = Literal["schema_only", "schema+aggregates", "schema+aggregates+sample"]


class EvidenceArtifactSummary(BaseModel):
    artifact_id: str
    artifact_type: str
    title: str
    dataset_id: str | None = None
    summary: str = ""


class EvidenceDataset(BaseModel):
    artifact_id: str
    dataset_id: str
    name: str
    row_count: int
    column_count: int
    columns: list[str]
    dtypes: dict[str, str]
    semantic_type_counts: dict[str, int] = Field(default_factory=dict)
    primary_key_candidates: list[str] = Field(default_factory=list)
    missing_percent: dict[str, float] = Field(default_factory=dict)
    sample_rows: list[dict[str, Any]] = Field(default_factory=list)


class EvidenceQualityIssue(BaseModel):
    artifact_id: str
    dataset_id: str
    severity: str
    code: str
    column: str | None
    message: str
    recommendation: str
    # Structured mirrors of the message figures (analysis-v3 §11.3); None on
    # issues from legacy artifacts, which therefore resolve no numbers.
    metric_value: float | None = None
    metric_unit: Literal["raw", "percent"] = "raw"
    affected_count: int | None = None


class EvidenceAnalysisTable(BaseModel):
    artifact_id: str
    dataset_id: str
    title: str
    kind: str
    description: str
    rows: list[dict[str, Any]] = Field(default_factory=list)


class EvidenceChart(BaseModel):
    artifact_id: str
    dataset_id: str
    title: str
    description: str
    mark: str
    encoding_fields: list[str] = Field(default_factory=list)


class EvidenceStatTest(BaseModel):
    artifact_id: str
    dataset_id: str
    test_type: str
    group_column: str | None = None
    value_column: str | None = None
    category_column: str | None = None
    # Mirror StatTestResult: only computable tests are persisted, but keep these
    # optional so the models stay in sync and evidence assembly never crashes.
    statistic: float | None = None
    p_value: float | None = None
    effect_size: float | None = None
    sample_size: int
    warnings: list[str] = Field(default_factory=list)


class EvidenceModelCard(BaseModel):
    artifact_id: str
    dataset_id: str
    task_type: str
    target_column: str
    split_strategy: str
    model_type: str
    metrics: dict[str, float] = Field(default_factory=dict)
    leakage_checks: list[str] = Field(default_factory=list)
    excluded_features: list[str] = Field(default_factory=list)


class EvidencePack(BaseModel):
    payload_policy: PayloadPolicy
    artifact_index: dict[str, EvidenceArtifactSummary] = Field(default_factory=dict)
    datasets: list[EvidenceDataset] = Field(default_factory=list)
    quality_issues: list[EvidenceQualityIssue] = Field(default_factory=list)
    analysis_tables: list[EvidenceAnalysisTable] = Field(default_factory=list)
    charts: list[EvidenceChart] = Field(default_factory=list)
    stat_tests: list[EvidenceStatTest] = Field(default_factory=list)
    model_cards: list[EvidenceModelCard] = Field(default_factory=list)

    @property
    def quality_issue_count(self) -> int:
        return len(self.quality_issues)

    @property
    def analysis_table_count(self) -> int:
        return len(self.analysis_tables)

    @property
    def chart_count(self) -> int:
        return len(self.charts)

    @property
    def stat_test_count(self) -> int:
        return len(self.stat_tests)

    @property
    def model_card_count(self) -> int:
        return len(self.model_cards)


def build_evidence_pack(
    artifacts: list[Artifact],
    *,
    payload_policy: PayloadPolicy = "schema+aggregates",
    sample_limit: int = 50,
) -> EvidencePack:
    pack = EvidencePack(payload_policy=payload_policy)
    include_samples = payload_policy == "schema+aggregates+sample"
    bounded_sample_limit = max(0, min(sample_limit, 50))

    for artifact in artifacts:
        summary = _summarize_artifact(artifact)
        pack.artifact_index[artifact.id] = summary

        if artifact.type is ArtifactType.DATASET_PROFILE:
            profile = DatasetProfile.model_validate(artifact.payload)
            pack.datasets.append(
                EvidenceDataset(
                    artifact_id=artifact.id,
                    dataset_id=profile.dataset_id,
                    name=profile.name,
                    row_count=profile.rows,
                    column_count=profile.columns,
                    columns=profile.column_names,
                    dtypes=profile.dtypes,
                    semantic_type_counts=profile.semantic_type_counts,
                    primary_key_candidates=profile.primary_key_candidates,
                    missing_percent=(
                        {} if payload_policy == "schema_only" else profile.missing_percent
                    ),
                    sample_rows=(
                        profile.sample_rows[:bounded_sample_limit] if include_samples else []
                    ),
                )
            )
        elif artifact.type is ArtifactType.QUALITY_ISSUE_SET and payload_policy != "schema_only":
            issue_set = QualityIssueSet.model_validate(artifact.payload)
            for issue in issue_set.issues:
                pack.quality_issues.append(
                    EvidenceQualityIssue(
                        artifact_id=artifact.id,
                        dataset_id=issue_set.dataset_id,
                        severity=issue.severity,
                        code=issue.code,
                        column=issue.column,
                        message=issue.message,
                        recommendation=issue.recommendation,
                        metric_value=issue.metric_value,
                        metric_unit=issue.metric_unit,
                        affected_count=issue.affected_count,
                    )
                )
        elif artifact.type is ArtifactType.TABLE and payload_policy != "schema_only":
            table = AnalysisTable.model_validate(artifact.payload)
            pack.analysis_tables.append(
                EvidenceAnalysisTable(
                    artifact_id=artifact.id,
                    dataset_id=table.dataset_id,
                    title=table.title,
                    kind=table.kind,
                    description=table.description,
                    rows=table.rows[:20],
                )
            )
        elif artifact.type is ArtifactType.CHART_SPEC and payload_policy != "schema_only":
            chart = ChartSpec.model_validate(artifact.payload)
            pack.charts.append(
                EvidenceChart(
                    artifact_id=artifact.id,
                    dataset_id=chart.dataset_id,
                    title=chart.title,
                    description=chart.description,
                    mark=chart.mark,
                    encoding_fields=_encoding_fields(chart.encoding),
                )
            )
        elif artifact.type is ArtifactType.STAT_TEST_RESULT and payload_policy != "schema_only":
            result = StatTestResult.model_validate(artifact.payload)
            pack.stat_tests.append(
                EvidenceStatTest(
                    artifact_id=artifact.id,
                    dataset_id=result.dataset_id,
                    test_type=result.test_type,
                    group_column=result.group_column,
                    value_column=result.value_column,
                    category_column=result.category_column,
                    statistic=result.statistic,
                    p_value=result.p_value,
                    effect_size=result.effect_size,
                    sample_size=result.sample_size,
                    warnings=[warning.code for warning in result.warnings],
                )
            )
        elif artifact.type is ArtifactType.MODEL_CARD and payload_policy != "schema_only":
            card = ModelCard.model_validate(artifact.payload)
            pack.model_cards.append(
                EvidenceModelCard(
                    artifact_id=artifact.id,
                    dataset_id=card.dataset_id,
                    task_type=card.task_type,
                    target_column=card.target_column,
                    split_strategy=card.split_strategy,
                    model_type=card.model_type,
                    metrics=card.metrics,
                    leakage_checks=[check.code for check in card.leakage_checks],
                    excluded_features=card.excluded_features,
                )
            )
    return pack


def _summarize_artifact(artifact: Artifact) -> EvidenceArtifactSummary:
    dataset_id: str | None = None
    title = artifact.type.value
    summary = ""
    if artifact.type is ArtifactType.DATASET_PROFILE:
        profile = DatasetProfile.model_validate(artifact.payload)
        dataset_id = profile.dataset_id
        title = profile.name
        summary = f"{profile.rows} rows, {profile.columns} columns"
    elif artifact.type is ArtifactType.QUALITY_ISSUE_SET:
        issue_set = QualityIssueSet.model_validate(artifact.payload)
        dataset_id = issue_set.dataset_id
        title = "Quality issues"
        summary = f"{len(issue_set.issues)} issues"
    elif artifact.type is ArtifactType.TABLE:
        table = AnalysisTable.model_validate(artifact.payload)
        dataset_id = table.dataset_id
        title = table.title
        summary = table.description
    elif artifact.type is ArtifactType.CHART_SPEC:
        chart = ChartSpec.model_validate(artifact.payload)
        dataset_id = chart.dataset_id
        title = chart.title
        summary = chart.description
    elif artifact.type is ArtifactType.STAT_TEST_RESULT:
        result = StatTestResult.model_validate(artifact.payload)
        dataset_id = result.dataset_id
        title = f"Statistical test: {result.test_type}"
        summary = f"p={result.p_value:g}, n={result.sample_size}"
    elif artifact.type is ArtifactType.MODEL_CARD:
        card = ModelCard.model_validate(artifact.payload)
        dataset_id = card.dataset_id
        title = f"Model card: {card.target_column}"
        summary = f"{card.task_type} baseline with {card.split_strategy} split"
    return EvidenceArtifactSummary(
        artifact_id=artifact.id,
        artifact_type=artifact.type.value,
        title=title,
        dataset_id=dataset_id,
        summary=summary,
    )


def _encoding_fields(encoding: dict[str, Any]) -> list[str]:
    fields: list[str] = []
    for value in encoding.values():
        if isinstance(value, dict) and isinstance(value.get("field"), str):
            fields.append(value["field"])
    return fields
