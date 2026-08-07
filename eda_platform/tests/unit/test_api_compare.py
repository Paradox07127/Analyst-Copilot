"""Compare API vertical slice: two runs of one project shaped side by side,
deltas derived from artifacts, and a cross-project pair refused with 422."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eda_platform.api.main import create_app
from eda_platform.core.ids import make_artifact_id
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.sessions import SessionManifest


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("compare_api")
    store = ArtifactStore(root)
    store.ensure_project("demo", name="Demo")
    store.ensure_project("other", name="Other")
    _seed_run(
        store,
        "demo",
        "run_left",
        datasets=[("orders.csv", 100, 4)],
        critical=2,
        warn=1,
        charts=2,
        stat_tests=1,
        cost={"llm_calls": 3, "total_tokens": 1000, "est_cost_usd": 0.01},
    )
    _seed_run(
        store,
        "demo",
        "run_right",
        datasets=[("orders.csv", 120, 4), ("customers.csv", 30, 3)],
        critical=1,
        warn=1,
        charts=3,
        stat_tests=1,
        cost={"llm_calls": 5, "total_tokens": 2500, "est_cost_usd": 0.04},
        model_card={"target_column": "amount", "metrics": {"r2": 0.812345}},
    )
    _seed_run(store, "other", "run_foreign", datasets=[("orders.csv", 10, 2)])
    return root


@pytest.fixture(scope="module")
def app(workspace: Path) -> FastAPI:
    return create_app(workspace)


@pytest.fixture(scope="module")
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


def _save(store: ArtifactStore, project_id: str, session_id: str, artifact_type, payload) -> None:
    store.save_artifact(
        Artifact(
            id=make_artifact_id("cmp", {"run": session_id, "type": str(artifact_type), **payload}),
            type=artifact_type,
            project_id=project_id,
            session_id=session_id,
            payload=payload,
        )
    )


def _seed_run(
    store: ArtifactStore,
    project_id: str,
    session_id: str,
    *,
    datasets: list[tuple[str, int, int]],
    critical: int = 0,
    warn: int = 0,
    charts: int = 0,
    stat_tests: int = 0,
    cost: dict | None = None,
    model_card: dict | None = None,
) -> None:
    store.start_session(project_id, session_id)
    for name, rows, columns in datasets:
        _save(
            store,
            project_id,
            session_id,
            ArtifactType.DATASET_PROFILE,
            {"dataset_id": f"ds_{name}", "name": name, "rows": rows, "columns": columns},
        )
    issues = [
        {"severity": "critical", "code": f"c{i}", "message": "bad", "recommendation": "fix"}
        for i in range(critical)
    ] + [
        {"severity": "warn", "code": f"w{i}", "message": "meh", "recommendation": "review"}
        for i in range(warn)
    ]
    _save(
        store,
        project_id,
        session_id,
        ArtifactType.QUALITY_ISSUE_SET,
        {"dataset_id": f"ds_{datasets[0][0]}", "issues": issues},
    )
    for index in range(charts):
        _save(
            store,
            project_id,
            session_id,
            ArtifactType.CHART_SPEC,
            {"title": f"chart {index}", "dataset": datasets[0][0], "mark": "bar"},
        )
    for index in range(stat_tests):
        _save(
            store,
            project_id,
            session_id,
            ArtifactType.STAT_TEST_RESULT,
            {"test": f"t{index}", "p_value": 0.01},
        )
    if model_card is not None:
        _save(store, project_id, session_id, ArtifactType.MODEL_CARD, model_card)
    if cost is not None:
        _save(store, project_id, session_id, ArtifactType.SESSION_METRICS, cost)
    store.mark_session_status(project_id, session_id, "completed")


def _compare(client: TestClient, left: str, right: str):
    return client.get("/api/v1/compare", params={"left": left, "right": right})


def _metric(body: dict, key: str) -> dict:
    return next(row for row in body["metrics"] if row["key"] == key)


def _text(body: dict, key: str) -> dict:
    return next(row for row in body["text_rows"] if row["key"] == key)


def _value(envelope: dict) -> object:
    assert envelope["state"] == "value"
    return envelope["value"]


def test_compare_shapes_metrics_side_by_side(client: TestClient) -> None:
    response = _compare(client, "run_left", "run_right")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["project_id"] == "demo"
    assert body["left"]["session_id"] == "run_left"
    assert body["right"]["session_id"] == "run_right"

    datasets = _metric(body, "datasets")
    assert (_value(datasets["left"]), _value(datasets["right"]), datasets["delta"]) == (
        1.0,
        2.0,
        1.0,
    )
    rows = _metric(body, "rows")
    assert (_value(rows["left"]), _value(rows["right"]), rows["delta"]) == (
        100.0,
        150.0,
        50.0,
    )
    critical = _metric(body, "critical")
    assert (_value(critical["left"]), _value(critical["right"]), critical["delta"]) == (
        2.0,
        1.0,
        -1.0,
    )
    # Fewer critical issues is an improvement, so the client can colour it.
    assert critical["higher_is_better"] is False
    assert critical["optimization_direction"] == "minimize"
    assert critical["verdict"] == "improved"
    charts = _metric(body, "charts")
    assert (_value(charts["left"]), _value(charts["right"]), charts["delta"]) == (
        2.0,
        3.0,
        1.0,
    )
    assert charts["higher_is_better"] is None

    cost = _metric(body, "llm_cost_usd")
    assert _value(cost["left"]) == pytest.approx(0.01)
    assert _value(cost["right"]) == pytest.approx(0.04)
    assert cost["delta"] == pytest.approx(0.03)
    assert _metric(body, "llm_tokens")["delta"] == pytest.approx(1500.0)


def test_compare_marks_unchanged_metric_delta_as_none(client: TestClient) -> None:
    """Control probe: an identical metric must carry no delta at all, so the
    client never paints a 0 as a change."""
    body = _compare(client, "run_left", "run_right").json()
    stat_tests = _metric(body, "stat_tests")
    assert (_value(stat_tests["left"]), _value(stat_tests["right"])) == (1.0, 1.0)
    assert stat_tests["delta"] is None
    assert stat_tests["verdict"] == "unchanged"


def test_compare_reports_text_rows_and_ml_headline(client: TestClient) -> None:
    body = _compare(client, "run_left", "run_right").json()
    ml_target = _text(body, "ml_target")
    assert ml_target["left"]["state"] == "not_applicable"
    assert _value(ml_target["right"]) == "amount"
    assert ml_target["changed"] is None
    assert _value(_text(body, "ml_metric")["right"]) == "r2=0.812"


def test_compare_reports_artifact_and_dataset_diffs(client: TestClient) -> None:
    body = _compare(client, "run_left", "run_right").json()
    chart_delta = next(
        row for row in body["artifact_deltas"] if row["type"] == ArtifactType.CHART_SPEC.value
    )
    assert (
        _value(chart_delta["left"]),
        _value(chart_delta["right"]),
        chart_delta["delta"],
    ) == (2, 3, 1)
    model_delta = next(
        row for row in body["artifact_deltas"] if row["type"] == ArtifactType.MODEL_CARD.value
    )
    assert (
        _value(model_delta["left"]),
        _value(model_delta["right"]),
        model_delta["delta"],
    ) == (0, 1, 1)

    assert body["datasets"]["shared"] == ["orders.csv"]
    assert body["datasets"]["only_left"] == []
    assert body["datasets"]["only_right"] == ["customers.csv"]


def test_compare_is_symmetric_on_swap(client: TestClient) -> None:
    forward = _compare(client, "run_left", "run_right").json()
    backward = _compare(client, "run_right", "run_left").json()
    assert _metric(backward, "rows")["delta"] == -_metric(forward, "rows")["delta"]
    assert backward["datasets"]["only_left"] == forward["datasets"]["only_right"]


def test_compare_across_projects_is_422(client: TestClient) -> None:
    response = _compare(client, "run_left", "run_foreign")
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "compare_project_mismatch"


def test_compare_run_against_itself_is_422(client: TestClient) -> None:
    """A deep link can ask for it even though the picker cannot; an all-zero
    diff is not an answer worth rendering."""
    response = _compare(client, "run_left", "run_left")
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "compare_same_session"


def test_compare_unknown_run_is_404(client: TestClient) -> None:
    response = _compare(client, "run_left", "run_missing")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "session_not_found"


def test_compare_requires_both_sides(client: TestClient) -> None:
    assert client.get("/api/v1/compare", params={"left": "run_left"}).status_code == 422


def test_compare_scope_endpoint_filters_and_paginates(client: TestClient) -> None:
    response = client.get(
        "/api/v1/compare/artifacts",
        params={
            "left": "run_left",
            "right": "run_right",
            "filter": "differences",
            "limit": 1,
        },
    )
    assert response.status_code == 200, response.text
    first = response.json()
    assert first["scope"] == "artifacts"
    assert len(first["items"]) == 1
    assert first["items"][0]["change"] != "same"
    assert first["next_cursor"]

    next_response = client.get(
        "/api/v1/compare/artifacts",
        params={
            "left": "run_left",
            "right": "run_right",
            "filter": "differences",
            "limit": 1,
            "cursor": first["next_cursor"],
        },
    )
    assert next_response.status_code == 200, next_response.text
    second = next_response.json()
    assert second["items"]
    assert second["items"][0]["match_key"] != first["items"][0]["match_key"]

    invalid = client.get(
        "/api/v1/compare/artifacts",
        params={
            "left": "run_right",
            "right": "run_left",
            "filter": "differences",
            "cursor": first["next_cursor"],
        },
    )
    assert invalid.status_code == 400
    assert invalid.json()["error"]["code"] == "invalid_cursor"


def test_compare_scope_name_is_typed(client: TestClient) -> None:
    response = client.get(
        "/api/v1/compare/not-a-scope",
        params={"left": "run_left", "right": "run_right"},
    )
    assert response.status_code == 422


def test_failed_session_absence_is_unavailable_not_zero(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    store.ensure_project("demo", name="Demo")
    store.start_session("demo", "completed")
    store.mark_session_status("demo", "completed", "completed")
    store.start_session("demo", "failed")
    store.mark_session_status("demo", "failed", "failed")

    body = _compare(TestClient(create_app(tmp_path)), "completed", "failed").json()
    charts = _metric(body, "charts")
    assert charts["left"] == {"state": "value", "value": 0.0, "reason": None}
    assert charts["right"]["state"] == "unavailable"
    assert charts["delta"] is None
    assert charts["verdict"] == "unknown"
    assert _metric(body, "llm_tokens")["left"]["state"] == "missing"
    assert _metric(body, "llm_tokens")["right"]["state"] == "unavailable"


def test_a_profile_missing_a_field_makes_the_total_unavailable_not_smaller(
    tmp_path: Path,
) -> None:
    """Counting an unreadable field as 0 produced a plausible-looking total that
    silently under-reported. Only a total every profile contributed to is a
    value; anything else has to say so."""
    store = ArtifactStore(tmp_path)
    store.ensure_project("demo", name="Demo")
    _seed_run(store, "demo", "left", datasets=[("a", 100, 4), ("b", 50, 3)])
    _seed_run(store, "demo", "right", datasets=[("a", 100, 4), ("b", 50, 3)])
    # One profile on the right reports no row count at all.
    _save(
        store,
        "demo",
        "right",
        ArtifactType.DATASET_PROFILE,
        {"dataset_id": "ds_c", "name": "c", "columns": 2},
    )

    body = _compare(TestClient(create_app(tmp_path)), "left", "right").json()
    rows = _metric(body, "rows")

    assert rows["left"]["state"] == "value"
    assert rows["right"]["state"] == "unavailable"
    assert "did not report this field" in rows["right"]["reason"]
    assert rows["delta"] is None, "a delta needs a value on both sides"
    # The readable field on that same profile still totals normally.
    assert _metric(body, "columns")["right"]["state"] == "value"


def _derived(store: ArtifactStore, project_id: str, session_id: str, source: str) -> None:
    """A machinery session that carries part of the root's results."""
    store.start_session(project_id, session_id)
    store.write_manifest(
        SessionManifest(
            session_id=session_id,
            project_id=project_id,
            input_hashes={"a": "hash"},
            code_version="test",
            source_session_id=source,
        )
    )
    store.mark_session_status(project_id, session_id, "completed")


def test_results_produced_in_a_derived_session_count_for_its_root(tmp_path: Path) -> None:
    """Compare read the root session alone, so a run whose questions executed in
    a derived session reported zero for work it had actually done."""
    store = ArtifactStore(tmp_path)
    store.ensure_project("demo", name="Demo")
    _seed_run(store, "demo", "left", datasets=[("a", 100, 4)], charts=1)
    _seed_run(store, "demo", "right", datasets=[("a", 100, 4)], charts=1)
    # The right side ran two more charts inside a question session.
    _derived(store, "demo", "qsess_right", "right")
    for index in range(2):
        _save(
            store,
            "demo",
            "qsess_right",
            ArtifactType.CHART_SPEC,
            {"chart_id": f"extra_{index}", "spec": {}},
        )
    store.refresh_session_index("demo", "qsess_right")

    body = _compare(TestClient(create_app(tmp_path)), "left", "right").json()
    charts = _metric(body, "charts")

    assert (charts["left"]["value"], charts["right"]["value"]) == (1.0, 3.0)
    assert charts["delta"] == 2.0
    chart_delta = next(
        row for row in body["artifact_deltas"] if row["type"] == ArtifactType.CHART_SPEC.value
    )
    assert (chart_delta["left"]["value"], chart_delta["right"]["value"]) == (1, 3)


def test_an_unfinished_family_member_makes_an_absent_count_unavailable(
    tmp_path: Path,
) -> None:
    """Producer coverage is a property of the family, not of the root: a root
    that completed while its question session failed cannot assert a real 0."""
    store = ArtifactStore(tmp_path)
    store.ensure_project("demo", name="Demo")
    _seed_run(store, "demo", "left", datasets=[("a", 100, 4)])
    _seed_run(store, "demo", "right", datasets=[("a", 100, 4)])
    store.start_session("demo", "qsess_right")
    store.write_manifest(
        SessionManifest(
            session_id="qsess_right",
            project_id="demo",
            input_hashes={"a": "hash"},
            code_version="test",
            source_session_id="right",
        )
    )
    store.mark_session_status("demo", "qsess_right", "failed")
    store.refresh_session_index("demo", "qsess_right")

    body = _compare(TestClient(create_app(tmp_path)), "left", "right").json()
    charts = _metric(body, "charts")

    assert charts["left"]["state"] == "value"
    assert charts["right"]["state"] == "unavailable"
    assert charts["delta"] is None


def test_a_producer_label_is_not_a_comparability_dimension(tmp_path: Path) -> None:
    """`code_version` never moved with the code, so Compare could not judge on it.

    Every writer was a constant: "local" from auto-EDA and question batches,
    "<name>-orchestrator-v2" from the two orchestrators, and derived runs
    inherit whichever of those their source held. Two runs built from entirely
    different code therefore always matched on this dimension, and the verdict
    read "controlled" for something nobody had checked. The field still names
    the producer and is still reported per side; it no longer votes.
    """
    store = ArtifactStore(tmp_path)
    store.ensure_project("demo", name="Demo")
    labels = {"left": "local", "right": "investigation-orchestrator-v2"}
    for session_id, label in labels.items():
        _seed_run(store, "demo", session_id, datasets=[("a", 100, 4)])
        store.write_manifest(
            SessionManifest(
                session_id=session_id,
                project_id="demo",
                input_hashes={"a": "hash"},
                code_version=label,
            )
        )
        store.refresh_session_index("demo", session_id)

    comparability = _compare(TestClient(create_app(tmp_path)), "left", "right").json()[
        "comparability"
    ]

    assert "code_version" not in comparability["changed_dimensions"]
    assert "code_version" not in comparability["unknown_dimensions"]
