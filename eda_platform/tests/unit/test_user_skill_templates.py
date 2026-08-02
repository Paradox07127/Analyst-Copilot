"""User-defined SQL skill templates: project-level CRUD with server-generated
content-addressed ids, three-level provenance on AnalysisSkill, and a real
trial execution when a user template is bound into a skill."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eda_platform.api.main import create_app
from eda_platform.core.skills_store import load_skills, load_user_templates
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.skills import AnalysisSkill

_JOB_TIMEOUT_SECONDS = 120.0
_BUILTIN_SEED_ID = "group_value_comparison"


def _orders_csv() -> str:
    rows = ["order_date,region,amount,quantity"]
    regions = ["north", "south", "east", "west"]
    for index in range(80):
        day = 1 + (index % 28)
        amount = round(50 + (index % 4) * 40, 2)
        rows.append(f"2024-01-{day:02d},{regions[index % 4]},{amount},{1 + index % 5}")
    return "\n".join(rows) + "\n"


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("user_templates_api")
    store = ArtifactStore(root)
    store.ensure_project("demo", name="Demo")
    seed = root / "seed" / "orders.csv"
    seed.parent.mkdir(parents=True, exist_ok=True)
    seed.write_text(_orders_csv(), encoding="utf-8")
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
def source_run(client: TestClient) -> str:
    session_id = "run_templates_src"
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
    return session_id


def _dataset_id(client: TestClient, session_id: str) -> str:
    body = client.get(f"/api/v1/sessions/{session_id}/skills").json()
    return body["datasets"][0]["dataset_id"]


def _template_body(**overrides: object) -> dict:
    body: dict = {
        "name": "Average value by group",
        "question": "What is the average {value_col} per {group_col}?",
        "sql": (
            "SELECT {group_col} AS group_value, AVG({value_col}) AS mean_value "
            "FROM {dataset} GROUP BY 1 ORDER BY 1"
        ),
        "method": "aggregation",
        "rationale": "Mean of a measure per category.",
        "params": [
            {"name": "group_col", "role": "dimension", "description": "Category column."},
            {"name": "value_col", "role": "measure", "description": "Numeric column."},
        ],
        "when_to_use": "Comparing a numeric measure across categories.",
        "when_not_to_use": "When the measure column is not numeric.",
    }
    body.update(overrides)
    return body


def _post_template(client: TestClient, body: dict, project_id: str = "demo"):
    return client.post(f"/api/v1/projects/{project_id}/skill-templates", json=body)


def _list_templates(client: TestClient, project_id: str = "demo") -> dict:
    response = client.get(f"/api/v1/projects/{project_id}/skill-templates")
    assert response.status_code == 200, response.text
    return response.json()


def _import_template(
    client: TestClient,
    session_id: str,
    template_id: str,
    *,
    dataset_ids: list[str],
    bindings: dict[str, str] | None = None,
    name: str = "",
):
    return client.post(
        f"/api/v1/sessions/{session_id}/skill-templates/{template_id}/import",
        json={"dataset_ids": dataset_ids, "bindings": bindings or {}, "name": name},
    )


# --- create / list / delete ---------------------------------------------------


def test_create_template_generates_id_and_lists_with_source_markers(
    client: TestClient, workspace: Path
) -> None:
    response = _post_template(client, _template_body())
    assert response.status_code == 201, response.text
    created = response.json()
    assert created["template_id"].startswith("tpl_")
    assert created["source"] == "user"
    assert created["when_to_use"] == "Comparing a numeric measure across categories."

    body = _list_templates(client)
    by_id = {item["template_id"]: item for item in body["templates"]}
    assert by_id[_BUILTIN_SEED_ID]["source"] == "builtin"
    assert by_id[created["template_id"]]["source"] == "user"
    assert by_id[created["template_id"]]["params"][0]["name"] == "group_col"

    persisted = load_user_templates(ArtifactStore(workspace).project_dir("demo"))
    assert created["template_id"] in {template.seed_id for template in persisted}


def test_create_template_is_idempotent_on_content(client: TestClient, workspace: Path) -> None:
    body = _template_body(name="Idempotency probe")
    first = _post_template(client, body)
    second = _post_template(client, body)
    assert first.status_code == 201 and second.status_code == 201
    assert first.json()["template_id"] == second.json()["template_id"]
    persisted = load_user_templates(ArtifactStore(workspace).project_dir("demo"))
    matches = [t for t in persisted if t.seed_id == first.json()["template_id"]]
    assert len(matches) == 1


def test_create_template_placeholder_mismatch_is_422(client: TestClient) -> None:
    body = _template_body(
        name="Mismatch probe",
        sql="SELECT {group_col} FROM {dataset} GROUP BY 1",
    )
    response = _post_template(client, body)
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "skill_invalid"


def test_create_template_dataset_param_name_is_422(client: TestClient) -> None:
    body = _template_body(
        name="Reserved name probe",
        sql="SELECT {dataset} FROM {dataset}",
        params=[{"name": "dataset", "role": "any"}],
    )
    response = _post_template(client, body)
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "skill_invalid"


def test_create_template_without_dataset_placeholder_is_422(client: TestClient) -> None:
    body = _template_body(
        name="No dataset probe",
        sql="SELECT {group_col}, AVG({value_col}) FROM orders GROUP BY 1",
    )
    response = _post_template(client, body)
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "skill_invalid"


def test_create_template_unknown_project_is_404(client: TestClient) -> None:
    response = _post_template(client, _template_body(), project_id="nope")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "project_not_found"


def test_delete_user_template_removes_it(client: TestClient) -> None:
    created = _post_template(client, _template_body(name="Delete me")).json()
    response = client.delete(
        f"/api/v1/projects/demo/skill-templates/{created['template_id']}"
    )
    assert response.status_code == 204, response.text
    listed = {item["template_id"] for item in _list_templates(client)["templates"]}
    assert created["template_id"] not in listed


def test_delete_builtin_template_is_409(client: TestClient) -> None:
    response = client.delete(f"/api/v1/projects/demo/skill-templates/{_BUILTIN_SEED_ID}")
    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "skill_not_deletable"


def test_delete_unknown_template_is_404(client: TestClient) -> None:
    response = client.delete("/api/v1/projects/demo/skill-templates/tpl_never_created")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "skill_template_not_found"


# --- binding: trial execution + provenance -------------------------------------


def test_bind_user_template_runs_trial_and_persists_user_template_origin(
    client: TestClient, workspace: Path, source_run: str
) -> None:
    created = _post_template(client, _template_body(name="Bind probe")).json()
    dataset_id = _dataset_id(client, source_run)
    response = _import_template(
        client,
        source_run,
        created["template_id"],
        dataset_ids=[dataset_id],
        bindings={"group_col": "region", "value_col": "amount"},
    )
    assert response.status_code == 201, response.text
    bound = response.json()
    assert bound["row_count"] == 4  # one row per region
    assert len(bound["rows_preview"]) == 4
    assert "group_value" in bound["columns"]
    assert bound["skill"]["source"] == "library"
    assert "{group_col}" not in bound["skill"]["sql"]

    persisted = {
        skill.skill_id: skill
        for skill in load_skills(ArtifactStore(workspace).project_dir("demo"))
    }
    skill = persisted[bound["skill"]["skill_id"]]
    assert skill.origin == "user_template"
    assert skill.when_to_use == "Comparing a numeric measure across categories."
    assert skill.when_not_to_use == "When the measure column is not numeric."

    # Idempotent on (template, relation, bindings): re-binding replaces in place.
    replay = _import_template(
        client,
        source_run,
        created["template_id"],
        dataset_ids=[dataset_id],
        bindings={"group_col": "region", "value_col": "amount"},
    )
    assert replay.status_code == 201, replay.text
    assert replay.json()["skill"]["skill_id"] == bound["skill"]["skill_id"]
    after = load_skills(ArtifactStore(workspace).project_dir("demo"))
    assert len(after) == len(persisted)


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO {dataset} SELECT {group_col} FROM {dataset}",
        "UPDATE {dataset} SET {group_col} = 'x'",
        "CREATE TABLE evil AS SELECT {group_col} FROM {dataset}",
    ],
)
def test_bind_write_sql_template_is_rejected_and_not_persisted(
    client: TestClient, workspace: Path, source_run: str, sql: str
) -> None:
    created = _post_template(
        client,
        _template_body(
            name=f"Write probe {sql.split(' ', 1)[0]}",
            question="What are the {group_col} values?",
            sql=sql,
            params=[{"name": "group_col", "role": "dimension"}],
        ),
    )
    assert created.status_code == 201, created.text
    before = load_skills(ArtifactStore(workspace).project_dir("demo"))
    response = _import_template(
        client,
        source_run,
        created.json()["template_id"],
        dataset_ids=[_dataset_id(client, source_run)],
        bindings={"group_col": "region"},
    )
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "skill_sql_rejected"
    assert len(load_skills(ArtifactStore(workspace).project_dir("demo"))) == len(before)


def test_bind_template_failing_at_execution_is_rejected_and_not_persisted(
    client: TestClient, workspace: Path, source_run: str
) -> None:
    created = _post_template(
        client,
        _template_body(
            name="Runtime failure probe",
            question="What are the {group_col} values?",
            sql="SELECT {group_col}, no_such_column FROM {dataset} GROUP BY 1, 2",
            params=[{"name": "group_col", "role": "dimension"}],
        ),
    ).json()
    before = load_skills(ArtifactStore(workspace).project_dir("demo"))
    response = _import_template(
        client,
        source_run,
        created["template_id"],
        dataset_ids=[_dataset_id(client, source_run)],
        bindings={"group_col": "region"},
    )
    assert response.status_code == 422, response.text
    body = response.json()["error"]
    assert body["code"] == "skill_trial_run_failed"
    assert "no_such_column" in body["message"]
    assert len(load_skills(ArtifactStore(workspace).project_dir("demo"))) == len(before)


def test_bind_with_unknown_column_is_422(client: TestClient, source_run: str) -> None:
    created = _post_template(client, _template_body(name="Bad binding probe")).json()
    response = _import_template(
        client,
        source_run,
        created["template_id"],
        dataset_ids=[_dataset_id(client, source_run)],
        bindings={"group_col": "region", "value_col": "not_a_column"},
    )
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "binding_invalid"


def test_bind_unknown_template_is_404(client: TestClient, source_run: str) -> None:
    response = _import_template(
        client,
        source_run,
        "tpl_missing",
        dataset_ids=[_dataset_id(client, source_run)],
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "skill_template_not_found"


def test_bind_builtin_seed_via_template_route_is_404(
    client: TestClient, source_run: str
) -> None:
    """Builtin seeds keep their own import route (and its unchanged behavior)."""
    response = _import_template(
        client,
        source_run,
        _BUILTIN_SEED_ID,
        dataset_ids=[_dataset_id(client, source_run)],
        bindings={"group_col": "region", "value_col": "amount"},
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "skill_template_not_found"


# --- provenance ----------------------------------------------------------------


def test_builtin_seed_import_sets_builtin_seed_origin(
    client: TestClient, workspace: Path, source_run: str
) -> None:
    response = client.post(
        f"/api/v1/sessions/{source_run}/skills/{_BUILTIN_SEED_ID}/import",
        json={
            "dataset_ids": [_dataset_id(client, source_run)],
            "bindings": {"group_col": "region", "value_col": "amount"},
            "name": "",
        },
    )
    assert response.status_code == 201, response.text
    skill_id = response.json()["skill_id"]
    persisted = {
        skill.skill_id: skill
        for skill in load_skills(ArtifactStore(workspace).project_dir("demo"))
    }
    assert persisted[skill_id].origin == "builtin_seed"


def test_old_skills_json_without_new_fields_loads_as_frozen_plan(tmp_path: Path) -> None:
    """Backward compatibility: a pre-provenance skills.json still loads."""
    legacy = {
        "version": 1,
        "skills": [
            {
                "skill_id": "skill_legacy",
                "name": "Legacy skill",
                "description": "",
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
                "created_at": "2026-01-01T00:00:00+00:00",
                "source_session_id": None,
            }
        ],
    }
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir(parents=True)
    (skills_dir / "skills.json").write_text(json.dumps(legacy), encoding="utf-8")
    loaded = load_skills(tmp_path)
    assert len(loaded) == 1
    skill = loaded[0]
    assert isinstance(skill, AnalysisSkill)
    assert skill.origin == "frozen_plan"
    assert skill.when_to_use == ""
    assert skill.when_not_to_use == ""
