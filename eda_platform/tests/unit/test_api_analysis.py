"""Deep analysis endpoint (§10.2 P1): artifact shaping, correlation splitting,
question lineage, malformed-payload tolerance, and path relativization."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from eda_platform.api.main import create_app
from eda_platform.application.services.analysis_service import BASELINE_QUESTION_LABEL
from eda_platform.core.ids import INTERNAL_SESSION_MARKER
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType

PROJECT = "demo"
RUN = "run_1"
BARE_RUN = "run_2"

PROFILE_PAYLOAD = {
    "dataset_id": "ds_1",
    "name": "orders.csv",
    "rows": 100,
    "columns": 2,
    "column_names": ["id", "amount"],
    "dtypes": {"id": "int64", "amount": "float64"},
    "missing_values": {"id": 0, "amount": 0},
    "missing_percent": {"id": 0.0, "amount": 0.0},
    "numeric_columns": ["id", "amount"],
    "categorical_columns": [],
}

SUMMARY_TABLE = {
    "dataset_id": "ds_1",
    "title": "Numeric summary",
    "kind": "numeric_summary",
    "description": "Per-column descriptive statistics.",
    "rows": [{"column": "amount", "mean": 12.5, "sample_size": 100}],
}

CORRELATION_TABLE = {
    "dataset_id": "ds_1",
    "title": "Correlations",
    "kind": "correlation",
    "description": "Pairwise Pearson correlations.",
    "rows": [
        {
            "left": "amount",
            "right": "total",
            "pearson": 0.62,
            "abs_pearson": 0.62,
            "sample_size": 100,
            "is_trivial_pair": False,
        },
        {
            "left": "amount",
            "right": "amount_cents",
            "pearson": 1.0,
            "abs_pearson": 1.0,
            "sample_size": 100,
            "is_trivial_pair": True,
        },
    ],
}

STAT_TEST = {
    "dataset_id": "ds_1",
    "test_type": "independent_t_test",
    "group_column": "segment",
    "value_column": "amount",
    "statistic": 5.82,
    "p_value": 1.3e-08,
    "effect_size": 0.44,
    "sample_size": 1470,
    "groups": {"a": 1233, "b": 237},
    "warnings": [{"code": "unequal_variance", "severity": "warn", "message": "Variances differ."}],
}

MODEL_CARD = {
    "dataset_id": "ds_1",
    "task_type": "classification",
    "target_column": "churn",
    "feature_columns": ["amount", "tenure"],
    "excluded_features": ["customer_id"],
    "split_strategy": "random_stratified",
    "train_rows": 800,
    "test_rows": 200,
    "model_type": "logistic_regression",
    "metrics": {"accuracy": 0.83, "f1_weighted": 0.81},
    "baseline_accuracy": 0.7,
    "leakage_checks": [
        {
            "code": "target_leak",
            "severity": "critical",
            "column": "customer_id",
            "action": "excluded",
            "message": "Identifier dropped.",
        }
    ],
    "feature_importance": [{"feature": "tenure", "importance": 0.6}],
    "limitations": ["Single split; no cross-validation."],
}


def _artifact(
    artifact_id: str,
    artifact_type: ArtifactType,
    payload: dict,
    *,
    session_id: str = RUN,
    parents: list[str] | None = None,
) -> Artifact:
    return Artifact(
        id=artifact_id,
        type=artifact_type,
        project_id=PROJECT,
        session_id=session_id,
        parents=parents or [],
        payload=payload,
    )


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    store = ArtifactStore(tmp_path)
    store.ensure_project(PROJECT, name="Demo")
    store.start_session(PROJECT, RUN)
    store.start_session(PROJECT, BARE_RUN)
    store.save_artifact(_artifact("prof_1", ArtifactType.DATASET_PROFILE, PROFILE_PAYLOAD))
    store.save_artifact(_artifact("table_summary", ArtifactType.TABLE, SUMMARY_TABLE))
    store.save_artifact(_artifact("table_corr", ArtifactType.TABLE, CORRELATION_TABLE))
    store.save_artifact(_artifact("stat_1", ArtifactType.STAT_TEST_RESULT, STAT_TEST))
    store.save_artifact(_artifact("card_1", ArtifactType.MODEL_CARD, MODEL_CARD))
    return store


@pytest.fixture
def client(store: ArtifactStore) -> TestClient:
    return TestClient(create_app(store.root))


def test_analysis_shapes_tables_stats_and_cards(client: TestClient) -> None:
    body = client.get(f"/api/v1/sessions/{RUN}/analysis").json()

    titles = [table["title"] for table in body["tables"]]
    assert titles == ["Correlations", "Numeric summary"]
    summary = next(table for table in body["tables"] if table["kind"] == "numeric_summary")
    assert summary["dataset_name"] == "orders.csv"
    assert summary["columns"] == ["column", "mean", "sample_size"]
    assert summary["rows"] == [{"column": "amount", "mean": 12.5, "sample_size": 100}]
    assert summary["min_sample_size"] == 100
    assert summary["small_sample"] is False

    assert body["stat_tests"] == [
        {
            "artifact_id": "stat_1",
            "dataset_id": "ds_1",
            "dataset_name": "orders.csv",
            "test_type": "independent_t_test",
            "group_column": "segment",
            "value_column": "amount",
            "statistic": 5.82,
            "p_value": 1.3e-08,
            "p_value_display": "<0.001",
            "effect_size": 0.44,
            "effect_size_magnitude": "small",
            "degrees_of_freedom": None,
            "sample_size": 1470,
            "significant": True,
            "conclusion": "Significant at alpha=0.05 (small effect)",
            "groups": {"a": 1233, "b": 237},
            "warnings": ["unequal_variance"],
            "small_sample": False,
        }
    ]

    card = body["model_cards"][0]
    assert (card["target_column"], card["task_type"]) == ("churn", "classification")
    assert (card["headline_metric"], card["headline_metric_value"]) == ("accuracy", 0.83)
    assert card["baseline_accuracy"] == 0.7
    assert card["leakage_verdict"] == "mitigated"
    assert card["feature_importance"] == [{"feature": "tenure", "importance": 0.6}]
    assert card["limitations"] == ["Single split; no cross-validation."]


def test_correlation_table_splits_trivial_pairs(client: TestClient) -> None:
    body = client.get(f"/api/v1/sessions/{RUN}/analysis").json()
    correlation = next(table for table in body["tables"] if table["kind"] == "correlation")
    assert [row["right"] for row in correlation["rows"]] == ["total"]
    assert [row["right"] for row in correlation["trivial_rows"]] == ["amount_cents"]


def test_question_label_walks_artifact_lineage(store: ArtifactStore) -> None:
    """A table two hops from its question execution still shows the question;
    an unparented table discloses that it is baseline EDA."""
    store.save_artifact(
        _artifact(
            "qexec_1",
            ArtifactType.QUESTION_EXECUTION_RESULT,
            {"question": "Which segment churns most?", "status": "succeeded"},
        )
    )
    store.save_artifact(
        _artifact("sql_1", ArtifactType.SQL_RESULT, {"sql": "select 1"}, parents=["qexec_1"])
    )
    store.save_artifact(
        _artifact(
            "table_q",
            ArtifactType.TABLE,
            {**SUMMARY_TABLE, "title": "Question table"},
            parents=["sql_1"],
        )
    )
    client = TestClient(create_app(store.root))
    tables = {
        table["title"]: table["question"]
        for table in client.get(f"/api/v1/sessions/{RUN}/analysis").json()["tables"]
    }
    assert tables["Question table"] == "Which segment churns most?"
    assert tables["Numeric summary"] == BASELINE_QUESTION_LABEL


def test_stat_test_without_p_value_is_not_interpretable(store: ArtifactStore) -> None:
    store.save_artifact(
        _artifact(
            "stat_np",
            ArtifactType.STAT_TEST_RESULT,
            {**STAT_TEST, "p_value": None, "effect_size": None, "sample_size": 12},
        )
    )
    client = TestClient(create_app(store.root))
    rows = {
        row["artifact_id"]: row
        for row in client.get(f"/api/v1/sessions/{RUN}/analysis").json()["stat_tests"]
    }
    assert rows["stat_np"]["significant"] is None
    assert rows["stat_np"]["conclusion"] == "No p-value reported — not interpretable."
    assert rows["stat_np"]["p_value_display"] == ""
    assert rows["stat_np"]["small_sample"] is True


def test_analysis_skips_invalid_payloads(store: ArtifactStore) -> None:
    store.save_artifact(_artifact("table_bad", ArtifactType.TABLE, {"dataset_id": "ds_1"}))
    store.save_artifact(_artifact("stat_bad", ArtifactType.STAT_TEST_RESULT, {"x": 1}))
    store.save_artifact(_artifact("card_bad", ArtifactType.MODEL_CARD, {"x": 1}))
    client = TestClient(create_app(store.root))
    body = client.get(f"/api/v1/sessions/{RUN}/analysis").json()
    assert [table["artifact_id"] for table in body["tables"]] == ["table_corr", "table_summary"]
    assert [row["artifact_id"] for row in body["stat_tests"]] == ["stat_1"]
    assert [card["artifact_id"] for card in body["model_cards"]] == ["card_1"]


def test_analysis_empty_run(client: TestClient) -> None:
    body = client.get(f"/api/v1/sessions/{BARE_RUN}/analysis").json()
    assert body == {"session_id": BARE_RUN, "tables": [], "stat_tests": [], "model_cards": []}


def test_analysis_unknown_run_404(client: TestClient) -> None:
    response = client.get("/api/v1/sessions/missing/analysis")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "session_not_found"


def test_analysis_internal_run_404(client: TestClient) -> None:
    response = client.get(f"/api/v1/sessions/{RUN}{INTERNAL_SESSION_MARKER}_probe/analysis")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "session_not_found"


def test_analysis_relativizes_workspace_paths(store: ArtifactStore) -> None:
    absolute = str((store.root / "projects" / PROJECT / "uploads" / "orders.csv").resolve())
    store.save_artifact(
        _artifact(
            "table_path",
            ArtifactType.TABLE,
            {**SUMMARY_TABLE, "title": "Path table", "description": absolute},
        )
    )
    client = TestClient(create_app(store.root))
    body = client.get(f"/api/v1/sessions/{RUN}/analysis").json()
    table = next(item for item in body["tables"] if item["artifact_id"] == "table_path")
    assert table["description"] == f"projects/{PROJECT}/uploads/orders.csv"
    assert str(store.root) not in client.get(f"/api/v1/sessions/{RUN}/analysis").text
