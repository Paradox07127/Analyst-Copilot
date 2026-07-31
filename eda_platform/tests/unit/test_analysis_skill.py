"""M6.2 P2 — AnalysisSkill schema, persistence, and deterministic replay.

The store tests are the persistence DoD gate: a skill round-trips to
``<project_dir>/skills/skills.json`` and a missing/corrupt/legacy file loads as
empty (never raises). The driver tests are the replay DoD gate: replaying a
frozen plan against a dataset missing a referenced column is rejected
teaching-style (``ToolGuardError``), while a dataset that has the columns runs
through the read-only permission gate and produces a ``SqlResult`` — offline,
with no LLM.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from eda_platform.core.skills_store import add_skill, load_skills, save_skills
from eda_platform.core.store import ArtifactStore
from eda_platform.core.tool_guard import ToolGuardError
from eda_platform.drivers.analysis_skill import replay_skill, skill_from_plan
from eda_platform.schemas.artifacts import ArtifactType, SqlResult
from eda_platform.schemas.plans import AnalysisPlan
from eda_platform.schemas.skills import AnalysisSkill
from eda_platform.tools.loader import LoadedDataset, load_csv

GOLDEN_DATA = Path(__file__).parents[1] / "golden" / "data"


def _orders_plan() -> AnalysisPlan:
    """A validated plan over the golden orders relation (``ecommerce_orders``)."""
    return AnalysisPlan(
        question="Total amount by customer",
        dataset_names=["ecommerce_orders"],
        columns=["customer_id", "amount"],
        filters=[],
        sql=(
            "SELECT customer_id, SUM(amount) AS total_amount "
            "FROM ecommerce_orders GROUP BY customer_id"
        ),
        method="aggregation",
        rationale="Sum order amount per customer.",
    )


def _orders() -> LoadedDataset:
    return load_csv(GOLDEN_DATA / "ecommerce_orders.csv")


def _customers() -> LoadedDataset:
    return load_csv(GOLDEN_DATA / "ecommerce_customers.csv")


def _store(tmp_path: Path) -> ArtifactStore:
    store = ArtifactStore(tmp_path / "ws")
    store.ensure_project("proj_skill", name="Skill")
    store.start_session("proj_skill", "run_skill")
    return store


# --- schema + skill_from_plan ----------------------------------------------


def test_skill_from_plan_lifts_columns_and_datasets() -> None:
    skill = skill_from_plan(
        _orders_plan(), "Revenue by customer", "Sum of amount per customer.", "run_src"
    )

    assert skill.name == "Revenue by customer"
    assert skill.description == "Sum of amount per customer."
    assert skill.param_columns == ["customer_id", "amount"]
    assert skill.expected_datasets == ["ecommerce_orders"]
    assert skill.source_session_id == "run_src"
    assert skill.skill_id  # uuid default is populated
    assert skill.plan.sql == _orders_plan().sql


def test_skill_name_must_be_non_empty() -> None:
    with pytest.raises(ValidationError):
        AnalysisSkill(name="   ", plan=_orders_plan())


def test_skill_defaults_are_independent_per_instance() -> None:
    first = AnalysisSkill(name="a", plan=_orders_plan())
    second = AnalysisSkill(name="b", plan=_orders_plan())

    assert first.skill_id != second.skill_id


# --- store round-trip + backward compatibility -----------------------------


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    skill = skill_from_plan(_orders_plan(), "Revenue by customer")
    save_skills(tmp_path, [skill])

    loaded = load_skills(tmp_path)

    assert len(loaded) == 1
    assert loaded[0].skill_id == skill.skill_id
    assert loaded[0].name == "Revenue by customer"
    assert loaded[0].param_columns == ["customer_id", "amount"]
    assert loaded[0].plan.sql == skill.plan.sql


def test_missing_file_loads_empty_without_writing(tmp_path: Path) -> None:
    assert load_skills(tmp_path) == []
    assert not (tmp_path / "skills" / "skills.json").exists()


def test_corrupt_file_loads_empty(tmp_path: Path) -> None:
    path = tmp_path / "skills" / "skills.json"
    path.parent.mkdir(parents=True)
    path.write_text("{ this is not valid json", encoding="utf-8")

    assert load_skills(tmp_path) == []


def test_legacy_bare_array_file_loads(tmp_path: Path) -> None:
    """A hand-written top-level JSON array (no envelope) still loads."""
    skill = skill_from_plan(_orders_plan(), "Revenue by customer")
    path = tmp_path / "skills" / "skills.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps([skill.model_dump(mode="json")]), encoding="utf-8")

    loaded = load_skills(tmp_path)

    assert len(loaded) == 1
    assert loaded[0].name == "Revenue by customer"


def test_add_skill_is_idempotent_on_id(tmp_path: Path) -> None:
    skill = skill_from_plan(_orders_plan(), "Revenue by customer")
    add_skill(tmp_path, skill)
    renamed = skill.model_copy(update={"name": "Revenue by customer v2"})
    add_skill(tmp_path, renamed)

    loaded = load_skills(tmp_path)

    assert len(loaded) == 1
    assert loaded[0].name == "Revenue by customer v2"


# --- replay: column gate ----------------------------------------------------


def test_replay_missing_column_is_rejected_teaching_style(tmp_path: Path) -> None:
    skill = skill_from_plan(_orders_plan(), "Revenue by customer")
    store = _store(tmp_path)

    with pytest.raises(ToolGuardError) as excinfo:
        replay_skill(
            skill,
            [_customers()],  # has customer_id but not amount
            store=store,
            project_id="proj_skill",
            session_id="run_skill",
        )

    feedback = str(excinfo.value)
    assert "amount" in feedback  # names the missing column
    assert "How to fix" in feedback  # teaching-style guidance block


def test_replay_dataset_arity_mismatch_is_rejected(tmp_path: Path) -> None:
    plan = _orders_plan().model_copy(
        update={
            "dataset_names": ["ecommerce_orders", "ecommerce_customers"],
            "columns": ["customer_id"],
            "sql": "SELECT customer_id FROM ecommerce_orders",
        }
    )
    skill = skill_from_plan(plan, "Two-dataset skill")
    store = _store(tmp_path)
    products = load_csv(GOLDEN_DATA / "ecommerce_products.csv")

    # 2 referenced datasets cannot map onto 3 targets (and it is not single-target).
    with pytest.raises(ToolGuardError) as excinfo:
        replay_skill(
            skill,
            [_orders(), _customers(), products],
            store=store,
            project_id="proj_skill",
            session_id="run_skill",
        )

    assert "map" in str(excinfo.value).lower()


# --- replay: execution ------------------------------------------------------


def test_replay_same_dataset_produces_sql_result(tmp_path: Path) -> None:
    skill = skill_from_plan(_orders_plan(), "Revenue by customer")
    store = _store(tmp_path)

    result = replay_skill(
        skill,
        [_orders()],
        store=store,
        project_id="proj_skill",
        session_id="run_skill",
    )

    assert result.status == "answer"
    sql_artifacts = [a for a in result.artifacts if a.type is ArtifactType.SQL_RESULT]
    assert sql_artifacts
    sql_result = SqlResult.model_validate(sql_artifacts[0].payload)
    assert sql_result.row_count > 0


def test_replay_renamed_dataset_rewrites_relation(tmp_path: Path) -> None:
    """Replaying onto a differently-named file rebinds the FROM clause to it."""
    skill = skill_from_plan(_orders_plan(), "Revenue by customer")
    store = _store(tmp_path)
    renamed_path = tmp_path / "orders_2024.csv"
    shutil.copyfile(GOLDEN_DATA / "ecommerce_orders.csv", renamed_path)
    target = load_csv(renamed_path)

    result = replay_skill(
        skill,
        [target],
        store=store,
        project_id="proj_skill",
        session_id="run_skill",
    )

    assert result.status == "answer"
    assert result.sql is not None
    assert "orders_2024" in result.sql
    assert "ecommerce_orders" not in result.sql
    sql_artifacts = [a for a in result.artifacts if a.type is ArtifactType.SQL_RESULT]
    assert sql_artifacts
    assert SqlResult.model_validate(sql_artifacts[0].payload).row_count > 0
