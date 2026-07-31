from datetime import UTC, datetime

from eda_platform.core.ids import make_artifact_id, stable_hash
from eda_platform.schemas.artifacts import (
    Artifact,
    ArtifactType,
    DatasetProfile,
    EvidenceRef,
    QualityIssue,
    QualityIssueSet,
)
from eda_platform.schemas.charts import ChartSpec
from eda_platform.schemas.sessions import SessionManifest, TraceEvent


def test_artifact_round_trips_with_evidence() -> None:
    evidence = EvidenceRef(
        kind="stat",
        artifact_id="prof_abc12345",
        locator="columns.amount.missing_percent",
        value=12.5,
    )
    artifact = Artifact(
        id="quality_def45678",
        type=ArtifactType.QUALITY_ISSUE_SET,
        project_id="project_demo",
        session_id="run_demo",
        created_at=datetime(2026, 7, 3, tzinfo=UTC),
        parents=["prof_abc12345"],
        payload={"issues": []},
        evidence=[evidence],
    )

    restored = Artifact.model_validate_json(artifact.model_dump_json())

    assert restored.type is ArtifactType.QUALITY_ISSUE_SET
    assert restored.evidence[0].locator == "columns.amount.missing_percent"
    assert restored.parents == ["prof_abc12345"]


def test_domain_payload_models_are_serializable() -> None:
    profile = DatasetProfile(
        dataset_id="ds_orders",
        name="orders.csv",
        rows=3,
        columns=2,
        column_names=["order_id", "amount"],
        dtypes={"order_id": "int64", "amount": "float64"},
        missing_values={"order_id": 0, "amount": 1},
        missing_percent={"order_id": 0.0, "amount": 33.33},
        numeric_columns=["order_id", "amount"],
        categorical_columns=[],
    )
    issues = QualityIssueSet(
        dataset_id="ds_orders",
        issues=[
            QualityIssue(
                severity="warn",
                code="high_missing",
                column="amount",
                message="Column amount has 33.33% missing values.",
                recommendation="Review missingness before analysis.",
            )
        ],
    )
    chart = ChartSpec(
        dataset_id="ds_orders",
        title="Distribution of amount",
        mark="bar",
        encoding={
            "x": {"field": "amount", "type": "quantitative", "bin": True},
            "y": {"aggregate": "count", "type": "quantitative"},
        },
    )

    assert profile.model_dump()["rows"] == 3
    assert issues.issues[0].severity == "warn"
    assert chart.to_vegalite()["mark"] == "bar"


def test_run_manifest_and_trace_event_round_trip() -> None:
    manifest = SessionManifest(
        session_id="run_123",
        project_id="project_demo",
        input_hashes={"orders.csv": "abc"},
        code_version="unknown",
        model_versions={},
        seed=42,
    )
    trace = TraceEvent(
        session_id="run_123",
        event_type="step_completed",
        name="profile_dataset",
        started_at=datetime(2026, 7, 3, tzinfo=UTC),
        finished_at=datetime(2026, 7, 3, tzinfo=UTC),
        summary={"artifact_count": 1},
    )

    assert SessionManifest.model_validate(manifest.model_dump()).seed == 42
    assert TraceEvent.model_validate(trace.model_dump()).summary["artifact_count"] == 1


def test_trace_event_correlation_fields_are_versioned_and_legacy_compatible() -> None:
    correlated = TraceEvent(
        session_id="run_123",
        event_type="llm_usage",
        name="probe",
        trial_id="trial_1",
        investigation_id="inv_1",
        span_id="span_1",
        parent_span_id="span_root",
        call_id="call_1",
        attempt_id="attempt_1",
    )
    legacy = TraceEvent.model_validate(
        {"session_id": "run_old", "event_type": "step_completed", "name": "profile"}
    )

    assert correlated.schema_version == 2
    assert correlated.call_id == "call_1"
    assert correlated.parent_span_id == "span_root"
    assert legacy.schema_version == 2
    assert legacy.call_id is None


def test_stable_ids_keep_prefix() -> None:
    digest = stable_hash({"name": "orders.csv", "rows": 3})

    assert len(digest) == 12
    assert make_artifact_id("prof", {"name": "orders.csv", "rows": 3}).startswith("prof_")
