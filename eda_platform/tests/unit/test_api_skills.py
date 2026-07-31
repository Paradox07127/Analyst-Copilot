"""Skills API vertical slice: the library plus builtin seeds are listed with
their parameter signature, prepare validates the bindings and registers a
server-side approval over the concrete skill, and execute consumes it once into
a `skill_replay` job that runs the existing driver on a derived ssess_* run.
Spawned-worker pattern mirrors test_api_questions."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from job_test_helpers import run_claimed_job

from eda_platform.api.main import create_app
from eda_platform.application.ports import JobRef
from eda_platform.application.services import skill_service as skill_service_module
from eda_platform.application.services.approval_service import ApprovalService
from eda_platform.application.services.skill_service import SkillService
from eda_platform.core.skills_store import add_skill, load_skills
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.analysis_skill import skill_from_plan
from eda_platform.schemas.artifacts import ArtifactType
from eda_platform.schemas.plans import AnalysisPlan
from eda_platform.schemas.skills import AnalysisSkill

_JOB_TIMEOUT_SECONDS = 120.0
_SEED_ID = "group_value_comparison"
_LIBRARY_SKILL_ID = "skill_library_probe"
# Skills exercised on the two-dataset run: relation names collide after
# cleaning, so the second target gets the `_2` suffix.
_LITERAL_SKILL_ID = "skill_literal_probe"
_JOIN_SKILL_ID = "skill_join_probe"
_BLOCKED_SKILL_ID = "skill_blocked_probe"
_EXTERNAL_READ_SKILL_ID = "skill_external_read_probe"


def _seed_csv() -> str:
    rows = ["order_date,region,amount,quantity"]
    regions = ["north", "south", "east", "west"]
    for index in range(120):
        day = 1 + (index % 28)
        month = 1 + (index // 28) % 6
        amount = round(50 + (index % 4) * 40 + (index * 7 % 25), 2)
        rows.append(f"2024-{month:02d}-{day:02d},{regions[index % 4]},{amount},{1 + index % 5}")
    return "\n".join(rows) + "\n"


def _sales_csv(offset: int) -> str:
    """Header carries a space so one column is unbindable (review J4)."""
    rows = ["region,amount,unit price"]
    regions = ["north", "south"]
    for index in range(20):
        rows.append(f"{regions[index % 2]},{offset + index},{index}")
    return "\n".join(rows) + "\n"


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("skills_api")
    store = ArtifactStore(root)
    store.ensure_project("demo", name="Demo")
    seed = root / "seed" / "orders.csv"
    seed.parent.mkdir(parents=True, exist_ok=True)
    seed.write_text(_seed_csv(), encoding="utf-8")
    # Two file names that clean down to the same relation base name.
    (root / "seed" / "sales report.csv").write_text(_sales_csv(100), encoding="utf-8")
    (root / "seed" / "sales_report.csv").write_text(_sales_csv(500), encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def app(workspace: Path) -> FastAPI:
    return create_app(workspace)


@pytest.fixture(scope="module")
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


def _wait_terminal(client: TestClient, job_id: str) -> dict:
    deadline = time.monotonic() + _JOB_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        body = client.get(f"/api/v1/jobs/{job_id}").json()
        if body["status"] in {"completed", "failed", "cancelled"}:
            return body
        time.sleep(0.2)
    raise AssertionError(f"Job {job_id} did not reach a terminal status in time.")


@pytest.fixture(scope="module")
def source_run(client: TestClient, workspace: Path) -> str:
    """A completed offline run whose loaded datasets replays target."""
    session_id = "run_skills_src"
    response = client.post(
        f"/api/v1/sessions/{session_id}/jobs",
        json={
            "kind": "auto_eda",
            "project_id": "demo",
            "datasets": ["seed/orders.csv"],
            "llm": "offline",
            "generate_report": False,
        },
    )
    assert response.status_code == 201, response.text
    final = _wait_terminal(client, response.json()["job_id"])
    assert final["status"] == "completed", final
    add_skill(
        ArtifactStore(workspace).project_dir("demo"),
        skill_from_plan(
            AnalysisPlan(
                question="How does amount vary by region?",
                dataset_names=["orders"],
                columns=["region", "amount"],
                filters=[],
                sql="SELECT region, SUM(amount) AS total FROM orders GROUP BY 1",
                method="aggregation",
                rationale="Totals per region.",
                estimated_scan="small",
            ),
            "Revenue by region",
            "Saved from a validated chat plan.",
            source_session_id=session_id,
        ).model_copy(update={"skill_id": _LIBRARY_SKILL_ID}),
    )
    return session_id


def _library_skill(
    skill_id: str,
    *,
    question: str,
    dataset_names: list[str],
    columns: list[str],
    sql: str,
    source_session_id: str,
) -> AnalysisSkill:
    return skill_from_plan(
        AnalysisPlan(
            question=question,
            dataset_names=dataset_names,
            columns=columns,
            filters=[],
            sql=sql,
            method="aggregation",
            rationale="Probe.",
            estimated_scan="small",
        ),
        skill_id,
        "Saved for the multi-target replay probes.",
        source_session_id=source_session_id,
    ).model_copy(update={"skill_id": skill_id})


@pytest.fixture(scope="module")
def multi_run(client: TestClient, workspace: Path) -> str:
    """A completed run over two datasets whose relation names collide."""
    session_id = "run_skills_multi"
    response = client.post(
        f"/api/v1/sessions/{session_id}/jobs",
        json={
            "kind": "auto_eda",
            "project_id": "demo",
            "datasets": ["seed/sales report.csv", "seed/sales_report.csv"],
            "llm": "offline",
            "generate_report": False,
        },
    )
    assert response.status_code == 201, response.text
    final = _wait_terminal(client, response.json()["job_id"])
    assert final["status"] == "completed", final
    project_dir = ArtifactStore(workspace).project_dir("demo")
    add_skill(
        project_dir,
        _library_skill(
            _LITERAL_SKILL_ID,
            question="Totals per region, labelled with their source table.",
            dataset_names=["orders"],
            columns=["region", "amount"],
            sql=(
                "SELECT 'orders' AS source_label, region, SUM(amount) AS total "
                "FROM orders WHERE region <> 'orders' GROUP BY 1, 2"
            ),
            source_session_id=session_id,
        ),
    )
    add_skill(
        project_dir,
        _library_skill(
            _JOIN_SKILL_ID,
            question="How do the two extracts line up per region?",
            dataset_names=["left_tbl", "right_tbl"],
            columns=["region"],
            sql=(
                "SELECT l.region, COUNT(*) AS pairs FROM left_tbl l "
                "JOIN right_tbl r ON l.region = r.region GROUP BY 1 ORDER BY 1"
            ),
            source_session_id=session_id,
        ),
    )
    add_skill(
        project_dir,
        _library_skill(
            _BLOCKED_SKILL_ID,
            question="Which regions are not the gift-set bundle?",
            dataset_names=["orders"],
            columns=["region"],
            sql="SELECT region FROM orders WHERE region <> 'Gift Set'",
            source_session_id=session_id,
        ),
    )
    add_skill(
        project_dir,
        _library_skill(
            _EXTERNAL_READ_SKILL_ID,
            question="Read an arbitrary host file.",
            dataset_names=["orders"],
            columns=["region"],
            sql="SELECT * FROM read_csv('/etc/passwd')",
            source_session_id=session_id,
        ),
    )
    return session_id


def _skills(client: TestClient, session_id: str) -> dict:
    response = client.get(f"/api/v1/sessions/{session_id}/skills")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["session_id"] == session_id
    return body


def _dataset_id(client: TestClient, session_id: str) -> str:
    return _skills(client, session_id)["datasets"][0]["dataset_id"]


def _prepare(
    client: TestClient,
    session_id: str,
    skill_id: str,
    *,
    dataset_ids: list[str],
    bindings: dict[str, str] | None = None,
):
    return client.post(
        f"/api/v1/sessions/{session_id}/skills/{skill_id}/prepare",
        json={"dataset_ids": dataset_ids, "bindings": bindings or {}},
    )


def _prepared_ok(client: TestClient, session_id: str, skill_id: str, **kwargs) -> dict:
    response = _prepare(client, session_id, skill_id, **kwargs)
    assert response.status_code == 200, response.text
    return response.json()


def _execute(
    client: TestClient,
    session_id: str,
    skill_id: str,
    prepared: dict,
    idempotency_key: str | None = None,
):
    headers = {"Idempotency-Key": idempotency_key} if idempotency_key else {}
    return client.post(
        f"/api/v1/sessions/{session_id}/skills/{skill_id}/execute",
        json={
            "action_hash": prepared["action_hash"],
            "approval_token": prepared["approval_token"],
        },
        headers=headers,
    )


def test_list_serves_library_and_seeds_with_targets(client: TestClient, source_run: str) -> None:
    body = _skills(client, source_run)
    by_id = {item["skill_id"]: item for item in body["skills"]}

    library = by_id[_LIBRARY_SKILL_ID]
    assert library["source"] == "library"
    assert library["name"] == "Revenue by region"
    assert library["param_columns"] == ["region", "amount"]
    assert library["params"] == []
    assert library["source_session_id"] == source_run

    seed = by_id[_SEED_ID]
    assert seed["source"] == "seed"
    assert [param["name"] for param in seed["params"]] == ["group_col", "value_col"]
    assert {param["role"] for param in seed["params"]} == {"dimension", "measure"}

    dataset = body["datasets"][0]
    assert dataset["name"] == "orders.csv"
    assert dataset["relation"] == "orders"
    assert {"region", "amount"} <= {column["name"] for column in dataset["columns"]}
    assert all(column["bindable"] for column in dataset["columns"])


def test_list_skills_is_paginated(client: TestClient, source_run: str) -> None:
    first = client.get(
        f"/api/v1/sessions/{source_run}/skills", params={"limit": 1}
    ).json()
    assert len(first["skills"]) == 1
    assert len(first["datasets"]) <= 1
    assert first["next_cursor"]
    second = client.get(
        f"/api/v1/sessions/{source_run}/skills",
        params={"limit": 1, "cursor": first["next_cursor"]},
    ).json()
    assert second["skills"][0]["skill_id"] != first["skills"][0]["skill_id"]


def test_indexed_skill_page_does_not_reload_monolithic_sources(
    client: TestClient,
    source_run: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = client.get(
        f"/api/v1/sessions/{source_run}/skills", params={"limit": 1}
    ).json()
    assert first["next_cursor"]

    def forbidden(*_args: object, **_kwargs: object) -> list:
        raise AssertionError("an indexed page must not reload a full source")

    monkeypatch.setattr(skill_service_module, "load_skills", forbidden)
    monkeypatch.setattr(skill_service_module, "load_builtin_seeds", forbidden)
    second = client.get(
        f"/api/v1/sessions/{source_run}/skills",
        params={"limit": 1, "cursor": first["next_cursor"]},
    )
    assert second.status_code == 200


def test_skill_cursor_rejects_changed_library_before_rebuild(
    client: TestClient,
    workspace: Path,
    source_run: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = client.get(
        f"/api/v1/sessions/{source_run}/skills", params={"limit": 1}
    ).json()
    path = ArtifactStore(workspace).project_dir("demo") / "skills" / "skills.json"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    def forbidden(*_args: object, **_kwargs: object) -> list:
        raise AssertionError("a stale cursor must fail before a full rebuild")

    monkeypatch.setattr(skill_service_module, "load_skills", forbidden)
    stale = client.get(
        f"/api/v1/sessions/{source_run}/skills",
        params={"limit": 1, "cursor": first["next_cursor"]},
    )
    assert stale.status_code == 400
    assert stale.json()["error"]["code"] == "invalid_cursor"


def test_skill_cursor_is_bound_to_run(
    client: TestClient, workspace: Path, source_run: str
) -> None:
    cursor = client.get(
        f"/api/v1/sessions/{source_run}/skills", params={"limit": 1}
    ).json()["next_cursor"]
    ArtifactStore(workspace).start_session("demo", "run_skill_other")
    replay = client.get(
        "/api/v1/sessions/run_skill_other/skills",
        params={"limit": 1, "cursor": cursor},
    )
    assert replay.status_code == 400
    assert replay.json()["error"]["code"] == "invalid_cursor"


def test_list_unknown_run_is_404(client: TestClient) -> None:
    response = client.get("/api/v1/sessions/run_missing/skills")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "session_not_found"


def test_prepare_seed_binds_columns_into_the_sql_preview(
    client: TestClient, source_run: str
) -> None:
    dataset_id = _dataset_id(client, source_run)
    prepared = _prepared_ok(
        client,
        source_run,
        _SEED_ID,
        dataset_ids=[dataset_id],
        bindings={"group_col": "region", "value_col": "amount"},
    )
    assert prepared["session_id"] == source_run
    assert prepared["skill_id"] == _SEED_ID
    assert len(prepared["action_hash"]) == 64
    assert len(prepared["approval_token"]) == 32
    assert prepared["uses_llm"] is False
    assert prepared["dataset_names"] == ["orders.csv"]
    sql = prepared["sql_preview"]
    assert "{group_col}" not in sql and "{dataset}" not in sql
    assert "region" in sql and "amount" in sql and "FROM orders" in sql


def test_prepare_unknown_skill_is_404(client: TestClient, source_run: str) -> None:
    response = _prepare(
        client, source_run, "skill_nope", dataset_ids=[_dataset_id(client, source_run)]
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "skill_not_found"


def test_prepare_unknown_column_binding_is_422(client: TestClient, source_run: str) -> None:
    response = _prepare(
        client,
        source_run,
        _SEED_ID,
        dataset_ids=[_dataset_id(client, source_run)],
        bindings={"group_col": "region", "value_col": "not_a_column"},
    )
    assert response.status_code == 422, response.text
    body = response.json()["error"]
    assert body["code"] == "binding_invalid"
    assert "not_a_column" in body["message"]


def test_prepare_unbound_placeholder_is_422(client: TestClient, source_run: str) -> None:
    response = _prepare(
        client,
        source_run,
        _SEED_ID,
        dataset_ids=[_dataset_id(client, source_run)],
        bindings={"group_col": "region"},
    )
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "binding_invalid"


def test_prepare_unknown_dataset_is_422(client: TestClient, source_run: str) -> None:
    response = _prepare(
        client,
        source_run,
        _SEED_ID,
        dataset_ids=["ds_not_in_this_run"],
        bindings={"group_col": "region", "value_col": "amount"},
    )
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "binding_invalid"


def test_prepare_library_skill_rejects_bindings(client: TestClient, source_run: str) -> None:
    response = _prepare(
        client,
        source_run,
        _LIBRARY_SKILL_ID,
        dataset_ids=[_dataset_id(client, source_run)],
        bindings={"group_col": "region"},
    )
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "binding_invalid"


def test_prepare_library_skill_previews_saved_sql(client: TestClient, source_run: str) -> None:
    prepared = _prepared_ok(
        client,
        source_run,
        _LIBRARY_SKILL_ID,
        dataset_ids=[_dataset_id(client, source_run)],
    )
    assert prepared["name"] == "Revenue by region"
    assert prepared["sql_preview"].startswith("SELECT region, SUM(amount)")
    assert prepared["bindings"] == {}


def test_execute_full_chain_offline_writes_sql_result(
    client: TestClient, workspace: Path, source_run: str
) -> None:
    dataset_id = _dataset_id(client, source_run)
    prepared = _prepared_ok(
        client,
        source_run,
        _SEED_ID,
        dataset_ids=[dataset_id],
        bindings={"group_col": "region", "value_col": "amount"},
    )
    response = _execute(client, source_run, _SEED_ID, prepared, uuid.uuid4().hex)
    assert response.status_code == 201, response.text
    started = response.json()
    assert started["skill_id"] == _SEED_ID
    assert started["execution_session_id"].startswith("ssess_")
    assert started["job"]["session_id"] == started["execution_session_id"]

    final = _wait_terminal(client, started["job"]["job_id"])
    assert final["status"] == "completed", final
    assert final["kind"] == "skill_replay"

    store = ArtifactStore(workspace)
    results = store.list_indexed_artifacts(
        project_id="demo",
        session_id=started["execution_session_id"],
        artifact_types=(ArtifactType.SQL_RESULT,),
    )
    assert len(results) == 1
    payload = results[0].payload
    assert payload["row_count"] == 4  # one row per region
    assert "group_value" in payload["columns"]

    # The derived run stands on its own: completed status and a manifest that
    # points back at the run the replay targeted.
    detail = client.get(f"/api/v1/sessions/{started['execution_session_id']}").json()
    assert detail["status"] == "completed"
    assert detail["source_session_id"] == source_run
    # The source run's own status is untouched by the replay.
    assert client.get(f"/api/v1/sessions/{source_run}").json()["status"] == "completed"


def test_execute_replayed_token_is_409_consumed(client: TestClient, source_run: str) -> None:
    prepared = _prepared_ok(
        client,
        source_run,
        _LIBRARY_SKILL_ID,
        dataset_ids=[_dataset_id(client, source_run)],
    )
    first = _execute(client, source_run, _LIBRARY_SKILL_ID, prepared, uuid.uuid4().hex)
    assert first.status_code == 201, first.text
    _wait_terminal(client, first.json()["job"]["job_id"])

    replay = _execute(client, source_run, _LIBRARY_SKILL_ID, prepared, uuid.uuid4().hex)
    assert replay.status_code == 409, replay.text
    assert replay.json()["error"]["code"] == "approval_consumed"


def test_execute_idempotency_key_replays_same_job(
    app: FastAPI, client: TestClient, source_run: str
) -> None:
    prepared = _prepared_ok(
        client,
        source_run,
        _LIBRARY_SKILL_ID,
        dataset_ids=[_dataset_id(client, source_run)],
    )
    backend = app.state.job_service._backend
    original = backend.enqueue
    backend.enqueue = lambda command: JobRef(job_id=command.job_id, pid=None)
    key = uuid.uuid4().hex
    try:
        first = _execute(client, source_run, _LIBRARY_SKILL_ID, prepared, key)
        assert first.status_code == 201, first.text
        replay = _execute(client, source_run, _LIBRARY_SKILL_ID, prepared, key)
    finally:
        backend.enqueue = original
    assert replay.status_code == 201, replay.text
    assert replay.json()["job"]["job_id"] == first.json()["job"]["job_id"]
    assert replay.json()["execution_session_id"] == first.json()["execution_session_id"]


def test_execute_unknown_token_is_404(client: TestClient, source_run: str) -> None:
    prepared = _prepared_ok(
        client,
        source_run,
        _LIBRARY_SKILL_ID,
        dataset_ids=[_dataset_id(client, source_run)],
    )
    response = _execute(
        client, source_run, _LIBRARY_SKILL_ID, {**prepared, "approval_token": "f" * 32}
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "approval_not_found"


def test_execute_expired_approval_is_410(
    app: FastAPI, client: TestClient, workspace: Path, source_run: str
) -> None:
    store = ArtifactStore(workspace)
    original = app.state.skill_service
    app.state.skill_service = SkillService(
        store,
        app.state.dataset_service,
        ApprovalService(store, ttl_seconds=-1),
        app.state.job_service,
    )
    try:
        prepared = _prepared_ok(
            client,
            source_run,
            _LIBRARY_SKILL_ID,
            dataset_ids=[_dataset_id(client, source_run)],
        )
        response = _execute(client, source_run, _LIBRARY_SKILL_ID, prepared)
        assert response.status_code == 410, response.text
        assert response.json()["error"]["code"] == "approval_expired"
    finally:
        app.state.skill_service = original


def test_cleaning_kind_approval_rejected_on_skills(
    client: TestClient, workspace: Path, source_run: str
) -> None:
    """A different approval kind on the same run must read as not-found, never
    cross over into a replay."""
    digest, token, _expires = ApprovalService(ArtifactStore(workspace)).register(
        kind="cleaning_apply",
        session_id=source_run,
        project_id="demo",
        action={"type": "cleaning_probe"},
        payload={"skill_id": _LIBRARY_SKILL_ID},
    )
    response = client.post(
        f"/api/v1/sessions/{source_run}/skills/{_LIBRARY_SKILL_ID}/execute",
        json={"action_hash": digest, "approval_token": token},
    )
    assert response.status_code == 404, response.text
    assert response.json()["error"]["code"] == "approval_not_found"


def test_execute_approval_bound_to_another_skill_is_422(
    app: FastAPI,
    client: TestClient,
    workspace: Path,
    source_run: str,
) -> None:
    """The path must not be able to swap the skill an approval reviewed."""
    prepared = _prepared_ok(
        client,
        source_run,
        _LIBRARY_SKILL_ID,
        dataset_ids=[_dataset_id(client, source_run)],
    )
    response = _execute(client, source_run, _SEED_ID, prepared)
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "skill_invalid"
    row = ArtifactStore(workspace).get_pending_action(
        prepared["action_hash"], session_id=source_run
    )
    assert row is not None
    assert row["status"] == "pending"
    assert row["generation"] == prepared["approval_token"]

    backend = app.state.job_service._backend
    original = backend.enqueue
    backend.enqueue = lambda command: JobRef(job_id=command.job_id, pid=None)
    try:
        retried = _execute(client, source_run, _LIBRARY_SKILL_ID, prepared)
    finally:
        backend.enqueue = original
    assert retried.status_code == 201, retried.text
    ArtifactStore(workspace).mark_job_status(
        retried.json()["job"]["job_id"],
        "completed",
    )


def test_runner_fails_when_source_datasets_are_unavailable(workspace: Path) -> None:
    """Defence in depth: a job whose target dataset is not loadable must fail
    before the driver runs, and only the derived run is marked failed."""
    store = ArtifactStore(workspace)
    job = store.create_job(
        job_id=f"job_{uuid.uuid4().hex[:12]}",
        session_id="ssess_missing_probe",
        project_id="demo",
        kind="skill_replay",
        idempotency_key=None,
    )
    params = {
        "source_session_id": "run_skills_src",
        "skill_id": _LIBRARY_SKILL_ID,
        "skill": {
            "skill_id": _LIBRARY_SKILL_ID,
            "name": "Revenue by region",
            "plan": {
                "question": "How does amount vary by region?",
                "dataset_names": ["orders"],
                "columns": ["region", "amount"],
                "filters": [],
                "sql": "SELECT region, SUM(amount) AS total FROM orders GROUP BY 1",
                "method": "aggregation",
                "rationale": "Totals per region.",
                "estimated_scan": "small",
            },
            "param_columns": ["region", "amount"],
            "expected_datasets": ["orders"],
        },
        "dataset_ids": ["ds_not_loadable"],
    }
    run_claimed_job(workspace, str(job["job_id"]), json.dumps(params))
    final = store.get_job(str(job["job_id"]))
    assert final is not None
    assert final["status"] == "failed"
    assert final["error_code"] == "ValueError"
    assert "source data for the selected dataset(s) is unavailable" in str(final["error_message"])


def test_generic_jobs_route_rejects_skill_replay_kind(client: TestClient, source_run: str) -> None:
    response = client.post(
        f"/api/v1/sessions/{source_run}/jobs",
        json={"kind": "skill_replay", "datasets": ["seed/orders.csv"]},
    )
    assert response.status_code == 422


# --- rebinding: preview == approved payload == executed SQL (review I1/J1) ---


def _dataset_ids_by_name(client: TestClient, session_id: str) -> dict[str, str]:
    return {item["name"]: item["dataset_id"] for item in _skills(client, session_id)["datasets"]}


def _approved_sql(workspace: Path, session_id: str, action_hash: str) -> str:
    pending = ArtifactStore(workspace).get_pending_action(action_hash, session_id=session_id)
    assert pending is not None
    return str(json.loads(str(pending["payload_json"]))["skill"]["plan"]["sql"])


def _landed_sql(workspace: Path, execution_session_id: str) -> str:
    results = ArtifactStore(workspace).list_indexed_artifacts(
        project_id="demo",
        session_id=execution_session_id,
        artifact_types=(ArtifactType.SQL_RESULT,),
    )
    assert len(results) == 1, results
    return str(results[0].payload["sql"])


def test_prepare_rebinds_relations_without_touching_string_literals(
    client: TestClient, workspace: Path, multi_run: str
) -> None:
    """The preview, the approval payload and the SQL that lands on disk are one
    string, and a literal that happens to spell the old relation name survives."""
    dataset_id = _dataset_ids_by_name(client, multi_run)["sales report.csv"]
    prepared = _prepared_ok(client, multi_run, _LITERAL_SKILL_ID, dataset_ids=[dataset_id])

    preview = prepared["sql_preview"]
    assert "FROM sales_report " in preview
    assert "SELECT 'orders' AS source_label" in preview
    assert "region <> 'orders'" in preview
    assert _approved_sql(workspace, multi_run, prepared["action_hash"]) == preview

    response = _execute(client, multi_run, _LITERAL_SKILL_ID, prepared, uuid.uuid4().hex)
    assert response.status_code == 201, response.text
    started = response.json()
    final = _wait_terminal(client, started["job"]["job_id"])
    assert final["status"] == "completed", final
    assert _landed_sql(workspace, started["execution_session_id"]) == preview


def test_prepare_maps_two_targets_onto_the_suffixed_relation_names(
    client: TestClient, workspace: Path, multi_run: str
) -> None:
    """Two datasets whose names clean to the same base: the second is `_2`, and
    the mapping the preview shows is the mapping the replay executes."""
    by_name = _dataset_ids_by_name(client, multi_run)
    prepared = _prepared_ok(
        client,
        multi_run,
        _JOIN_SKILL_ID,
        dataset_ids=[by_name["sales report.csv"], by_name["sales_report.csv"]],
    )
    preview = prepared["sql_preview"]
    assert "FROM sales_report l" in preview
    assert "JOIN sales_report_2 r" in preview
    assert "left_tbl" not in preview and "right_tbl" not in preview

    response = _execute(client, multi_run, _JOIN_SKILL_ID, prepared, uuid.uuid4().hex)
    assert response.status_code == 201, response.text
    started = response.json()
    final = _wait_terminal(client, started["job"]["job_id"])
    assert final["status"] == "completed", final
    assert _landed_sql(workspace, started["execution_session_id"]) == preview


def test_prepare_allows_blocked_word_inside_string_literal(
    client: TestClient, multi_run: str
) -> None:
    """A data value is not executable SQL, even when it contains ``Set``."""
    dataset_id = _dataset_ids_by_name(client, multi_run)["sales report.csv"]
    prepared = _prepared_ok(
        client,
        multi_run,
        _BLOCKED_SKILL_ID,
        dataset_ids=[dataset_id],
    )
    assert "region <> 'Gift Set'" in prepared["sql_preview"]


def test_prepare_rejects_external_read_before_registering_approval(
    client: TestClient, multi_run: str
) -> None:
    dataset_id = _dataset_ids_by_name(client, multi_run)["sales report.csv"]
    response = _prepare(
        client,
        multi_run,
        _EXTERNAL_READ_SKILL_ID,
        dataset_ids=[dataset_id],
    )
    assert response.status_code == 422, response.text
    body = response.json()["error"]
    assert body["code"] == "skill_sql_rejected"
    assert "read_csv" in body["message"].lower()


def test_execute_replays_the_approved_content_after_the_library_changed(
    client: TestClient, workspace: Path, multi_run: str
) -> None:
    """Invariant probe: editing the library entry under the same skill_id
    between prepare and execute must not change what sessions."""
    dataset_id = _dataset_ids_by_name(client, multi_run)["sales report.csv"]
    prepared = _prepared_ok(client, multi_run, _LITERAL_SKILL_ID, dataset_ids=[dataset_id])
    approved = prepared["sql_preview"]

    project_dir = ArtifactStore(workspace).project_dir("demo")
    tampered = _library_skill(
        _LITERAL_SKILL_ID,
        question="Tampered after approval.",
        dataset_names=["orders"],
        columns=["region", "amount"],
        sql="SELECT region, COUNT(*) AS tampered FROM orders GROUP BY 1",
        source_session_id=multi_run,
    )
    add_skill(project_dir, tampered)
    try:
        response = _execute(client, multi_run, _LITERAL_SKILL_ID, prepared, uuid.uuid4().hex)
        assert response.status_code == 201, response.text
        started = response.json()
        assert _wait_terminal(client, started["job"]["job_id"])["status"] == "completed"
        landed = _landed_sql(workspace, started["execution_session_id"])
    finally:
        add_skill(
            project_dir,
            _library_skill(
                _LITERAL_SKILL_ID,
                question="Totals per region, labelled with their source table.",
                dataset_names=["orders"],
                columns=["region", "amount"],
                sql=(
                    "SELECT 'orders' AS source_label, region, SUM(amount) AS total "
                    "FROM orders WHERE region <> 'orders' GROUP BY 1, 2"
                ),
                source_session_id=multi_run,
            ),
        )
    assert landed == approved
    assert "tampered" not in landed


def test_list_marks_columns_that_cannot_be_bound(client: TestClient, multi_run: str) -> None:
    """A header with a space would be interpolated into SQL, so the form must
    not offer it (review J4)."""
    datasets = _skills(client, multi_run)["datasets"]
    dataset = next(item for item in datasets if item["name"] == "sales report.csv")
    by_name = {column["name"]: column["bindable"] for column in dataset["columns"]}
    assert by_name["region"] is True
    assert by_name["unit price"] is False


def test_replay_run_is_hidden_from_the_run_list_but_still_reachable(
    client: TestClient, multi_run: str
) -> None:
    """Derived runs are noise in the project list; deep links keep working."""
    dataset_id = _dataset_ids_by_name(client, multi_run)["sales report.csv"]
    prepared = _prepared_ok(client, multi_run, _LITERAL_SKILL_ID, dataset_ids=[dataset_id])
    started = _execute(client, multi_run, _LITERAL_SKILL_ID, prepared, uuid.uuid4().hex).json()
    _wait_terminal(client, started["job"]["job_id"])
    derived = started["execution_session_id"]

    listed = client.get("/api/v1/projects/demo/sessions", params={"limit": 100}).json()
    assert derived not in [run["session_id"] for run in listed["items"]]
    with_derived = client.get(
        "/api/v1/projects/demo/sessions", params={"limit": 100, "include_derived": "true"}
    ).json()
    assert derived in [run["session_id"] for run in with_derived["items"]]
    assert client.get(f"/api/v1/sessions/{derived}").status_code == 200


# --- save / delete: freezing a run's plan artifact into the project library ---


def _chat_plan_artifact(store: ArtifactStore, session_id: str, question: str) -> str:
    """A ChatTurnPlan artifact of the run — what the Chat page leaves behind
    once a plan is approved, and the only thing save accepts as a source."""
    from eda_platform.schemas.artifacts import Artifact

    payload = {
        "question": question,
        "dataset_names": ["orders"],
        "columns": ["region", "amount"],
        "filters": [],
        "sql": "SELECT region, AVG(amount) AS mean_amount FROM orders GROUP BY 1",
        "method": "aggregation",
        "rationale": "Mean per region.",
        "estimated_scan": "small",
        "raw_message": "the model's raw reply, which must not reach the plan",
    }
    artifact = Artifact(
        id=f"plan_{uuid.uuid4().hex[:12]}",
        type=ArtifactType.CHAT_TURN_PLAN,
        project_id="demo",
        session_id=session_id,
        payload=payload,
    )
    store.save_artifact(artifact)
    return artifact.id


def test_list_offers_this_runs_plan_artifacts_as_save_sources(
    client: TestClient, workspace: Path, source_run: str
) -> None:
    artifact_id = _chat_plan_artifact(
        ArtifactStore(workspace), source_run, "What is the mean amount per region?"
    )
    plans = {item["artifact_id"]: item for item in _skills(client, source_run)["savable_plans"]}
    assert artifact_id in plans
    plan = plans[artifact_id]
    assert plan["question"] == "What is the mean amount per region?"
    assert plan["sql"].startswith("SELECT region, AVG(amount)")
    assert plan["columns"] == ["region", "amount"]


def test_save_freezes_the_plan_into_the_library_and_delete_removes_it(
    client: TestClient, workspace: Path, source_run: str
) -> None:
    artifact_id = _chat_plan_artifact(
        ArtifactStore(workspace), source_run, "Which region spends most?"
    )
    response = client.post(
        f"/api/v1/sessions/{source_run}/skills",
        json={
            "source_artifact_id": artifact_id,
            "name": "Spend by region",
            "description": "Saved from a validated chat plan.",
        },
    )
    assert response.status_code == 201, response.text
    saved = response.json()
    assert saved["source"] == "library"
    assert saved["name"] == "Spend by region"
    assert saved["source_session_id"] == source_run
    assert saved["param_columns"] == ["region", "amount"]
    # raw_message is a model artefact, not part of the plan the skill freezes.
    assert "raw_message" not in saved["sql"]

    listed = {item["skill_id"] for item in _skills(client, source_run)["skills"]}
    assert saved["skill_id"] in listed

    deleted = client.delete(f"/api/v1/projects/demo/skills/{saved['skill_id']}")
    assert deleted.status_code == 204, deleted.text
    assert saved["skill_id"] not in {
        item["skill_id"] for item in _skills(client, source_run)["skills"]
    }


def test_save_rejects_an_artifact_from_another_run(
    client: TestClient, workspace: Path, source_run: str, multi_run: str
) -> None:
    """Project scope is not enough: the plan must belong to the run in the path."""
    artifact_id = _chat_plan_artifact(
        ArtifactStore(workspace), multi_run, "Plan that lives in the other run."
    )
    response = client.post(
        f"/api/v1/sessions/{source_run}/skills",
        json={"source_artifact_id": artifact_id, "name": "Cross-run steal", "description": ""},
    )
    assert response.status_code == 404, response.text
    assert response.json()["error"]["code"] == "skill_plan_not_found"


def test_save_unknown_artifact_is_404(client: TestClient, source_run: str) -> None:
    response = client.post(
        f"/api/v1/sessions/{source_run}/skills",
        json={"source_artifact_id": "plan_missing", "name": "Nope", "description": ""},
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "skill_plan_not_found"


def test_save_blank_name_is_422(client: TestClient, workspace: Path, source_run: str) -> None:
    artifact_id = _chat_plan_artifact(ArtifactStore(workspace), source_run, "Needs a name.")
    response = client.post(
        f"/api/v1/sessions/{source_run}/skills",
        json={"source_artifact_id": artifact_id, "name": "   ", "description": ""},
    )
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "skill_invalid"


def test_delete_seed_template_is_409(client: TestClient) -> None:
    response = client.delete(f"/api/v1/projects/demo/skills/{_SEED_ID}")
    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "skill_not_deletable"


def test_delete_unknown_skill_is_404(client: TestClient) -> None:
    response = client.delete("/api/v1/projects/demo/skills/skill_never_saved")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "skill_not_found"


def test_delete_in_unknown_project_is_404(client: TestClient) -> None:
    response = client.delete(f"/api/v1/projects/other/skills/{_LIBRARY_SKILL_ID}")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "project_not_found"


def test_failed_replay_still_leaves_a_manifest_on_the_derived_run(workspace: Path) -> None:
    """A refused replay must not orphan its derived run: the manifest describes
    the intent, so it is written before the driver runs (review I3)."""
    store = ArtifactStore(workspace)
    derived_session_id = f"ssess_manifest_probe_{uuid.uuid4().hex[:6]}"
    job = store.create_job(
        job_id=f"job_{uuid.uuid4().hex[:12]}",
        session_id=derived_session_id,
        project_id="demo",
        kind="skill_replay",
        idempotency_key=None,
    )
    dataset_id = next(
        artifact.payload["dataset_id"]
        for artifact in store.list_artifacts(project_id="demo", session_id="run_skills_src")
        if artifact.type is ArtifactType.DATASET_PROFILE
    )
    params = {
        "source_session_id": "run_skills_src",
        "skill_id": _LIBRARY_SKILL_ID,
        "skill": {
            "skill_id": _LIBRARY_SKILL_ID,
            "name": "Broken replay",
            "plan": {
                "question": "Does a bad column fail loudly?",
                "dataset_names": ["orders"],
                "columns": ["region"],
                "filters": [],
                # Passes the guards (region exists) but cannot bind in DuckDB.
                "sql": "SELECT no_such_column FROM orders",
                "method": "aggregation",
                "rationale": "Probe.",
                "estimated_scan": "small",
            },
            "param_columns": ["region"],
            "expected_datasets": ["orders"],
        },
        "dataset_ids": [dataset_id],
    }
    run_claimed_job(workspace, str(job["job_id"]), json.dumps(params))

    final = store.get_job(str(job["job_id"]))
    assert final is not None and final["status"] == "failed", final
    manifest = store.read_manifest("demo", derived_session_id)
    assert manifest is not None
    assert manifest.source_session_id == "run_skills_src"


# --------------------------------------------------------------------------- #
# Seed import: a bound seed becomes a saved library skill
# --------------------------------------------------------------------------- #
def _import_seed(
    client: TestClient,
    session_id: str,
    seed_id: str,
    *,
    dataset_ids: list[str],
    bindings: dict[str, str] | None = None,
    name: str = "",
):
    return client.post(
        f"/api/v1/sessions/{session_id}/skills/{seed_id}/import",
        json={
            "dataset_ids": dataset_ids,
            "bindings": bindings or {},
            "name": name,
        },
    )


def test_import_seed_lands_in_the_library_and_stays_replayable(
    client: TestClient, workspace: Path, source_run: str
) -> None:
    dataset_id = _dataset_id(client, source_run)
    bindings = {"group_col": "region", "value_col": "amount"}
    response = _import_seed(
        client, source_run, _SEED_ID, dataset_ids=[dataset_id], bindings=bindings
    )
    assert response.status_code == 201, response.text
    imported = response.json()
    assert imported["source"] == "library"
    assert imported["skill_id"] != _SEED_ID
    assert "{group_col}" not in imported["sql"] and "{dataset}" not in imported["sql"]
    assert imported["param_columns"] == ["region", "amount"]

    # Actually on disk, not just in the response.
    persisted = load_skills(ArtifactStore(workspace).project_dir("demo"))
    assert imported["skill_id"] in {skill.skill_id for skill in persisted}
    assert imported["skill_id"] in {
        item["skill_id"] for item in _skills(client, source_run)["skills"]
    }

    # A library skill takes no bindings: preparing it proves it is replayable.
    prepared = _prepared_ok(
        client, source_run, imported["skill_id"], dataset_ids=[dataset_id]
    )
    assert "FROM orders" in prepared["sql_preview"]

    # Idempotent on (seed, relation, bindings): no second row appears.
    replay = _import_seed(
        client, source_run, _SEED_ID, dataset_ids=[dataset_id], bindings=bindings
    )
    assert replay.status_code == 201, replay.text
    assert replay.json()["skill_id"] == imported["skill_id"]
    after = load_skills(ArtifactStore(workspace).project_dir("demo"))
    assert len(after) == len(persisted)

    deleted = client.delete(f"/api/v1/projects/demo/skills/{imported['skill_id']}")
    assert deleted.status_code == 204, deleted.text


def test_import_seed_honours_a_custom_name(
    client: TestClient, workspace: Path, source_run: str
) -> None:
    dataset_id = _dataset_id(client, source_run)
    response = _import_seed(
        client,
        source_run,
        _SEED_ID,
        dataset_ids=[dataset_id],
        bindings={"group_col": "region", "value_col": "amount"},
        name="Amount by region (imported)",
    )
    assert response.status_code == 201, response.text
    skill_id = response.json()["skill_id"]
    assert response.json()["name"] == "Amount by region (imported)"
    persisted = {
        skill.skill_id: skill.name
        for skill in load_skills(ArtifactStore(workspace).project_dir("demo"))
    }
    # The rename replaced the row import_seed wrote rather than duplicating it.
    assert persisted[skill_id] == "Amount by region (imported)"
    assert client.delete(f"/api/v1/projects/demo/skills/{skill_id}").status_code == 204


def test_import_unknown_seed_is_404(client: TestClient, source_run: str) -> None:
    response = _import_seed(
        client, source_run, "seed_never_shipped", dataset_ids=[_dataset_id(client, source_run)]
    )
    assert response.status_code == 404, response.text
    assert response.json()["error"]["code"] == "skill_not_found"


def test_import_seed_with_an_unknown_column_is_422(
    client: TestClient, workspace: Path, source_run: str
) -> None:
    before = len(load_skills(ArtifactStore(workspace).project_dir("demo")))
    response = _import_seed(
        client,
        source_run,
        _SEED_ID,
        dataset_ids=[_dataset_id(client, source_run)],
        bindings={"group_col": "region", "value_col": "not_a_column"},
    )
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "binding_invalid"
    # A refused import never reaches the library.
    assert len(load_skills(ArtifactStore(workspace).project_dir("demo"))) == before


def test_import_seed_onto_two_datasets_is_422(client: TestClient, multi_run: str) -> None:
    body = _skills(client, multi_run)
    dataset_ids = [dataset["dataset_id"] for dataset in body["datasets"]]
    assert len(dataset_ids) == 2
    response = _import_seed(
        client,
        multi_run,
        _SEED_ID,
        dataset_ids=dataset_ids,
        bindings={"group_col": "region", "value_col": "amount"},
    )
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "binding_invalid"


def test_import_seed_unknown_run_is_404(client: TestClient) -> None:
    response = _import_seed(client, "run_missing", _SEED_ID, dataset_ids=["ds_nope"])
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "session_not_found"
