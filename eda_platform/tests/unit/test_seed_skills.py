"""Seed skill library: builtin templates, instantiation, import, planner catalog.

Red/green gate for the seed-library feature: every builtin seed template must
validate, instantiate onto a synthetic dataset, and replay through the existing
deterministic read-only gate; imports are idempotent; ``catalog_block`` renders
a bounded planner context block; the question agent injects it only when
non-empty.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import pandas as pd
import pytest
from pydantic import BaseModel, ValidationError

from eda_platform.agents.question_agent import propose_llm_question_candidates
from eda_platform.core.skills_store import (
    add_skill,
    catalog_block,
    import_seed,
    instantiate_seed,
    load_builtin_seeds,
    load_skills,
)
from eda_platform.core.store import ArtifactStore
from eda_platform.core.tool_guard import ToolGuardError
from eda_platform.drivers.analysis_skill import replay_skill, skill_from_plan
from eda_platform.schemas.artifacts import ArtifactType
from eda_platform.schemas.plans import AnalysisPlan
from eda_platform.schemas.skills import SeedParam, SeedSkillTemplate
from eda_platform.tools.loader import LoadedDataset, load_csv
from eda_platform.tools.sql_runner import build_catalog

T = TypeVar("T", bound=BaseModel)


# --- synthetic dataset -------------------------------------------------------


def _synthetic_dataset(tmp_path: Path) -> LoadedDataset:
    frame = pd.DataFrame(
        {
            "order_id": [1, 2, 3, 4, 5, 6, 7, 7],
            "region": ["East", "East", "West", "West", "North", "North", "East", "West"],
            "category": ["A", "B", "A", "B", "A", "B", "A", "A"],
            "amount": [10.0, 20.0, 30.0, None, 50.0, 60.0, 500.0, 15.0],
            "order_date": [
                "2026-01-05",
                "2026-01-20",
                "2026-02-03",
                "2026-02-18",
                "2026-03-02",
                "2026-03-15",
                "2026-03-30",
                "2026-04-02",
            ],
        }
    )
    csv_path = tmp_path / "sales_data.csv"
    frame.to_csv(csv_path, index=False)
    return load_csv(csv_path)


# Role -> columns of the synthetic dataset, in preference order.
_ROLE_POOLS: dict[str, list[str]] = {
    "measure": ["amount"],
    "dimension": ["region", "category"],
    "timestamp": ["order_date"],
    "identifier": ["order_id"],
    "any": ["amount", "region"],
}


def _bindings_for(template: SeedSkillTemplate) -> dict[str, str]:
    """Bind each param to a distinct synthetic column matching its role."""
    used: set[str] = set()
    bindings: dict[str, str] = {}
    for param in template.params:
        pool = _ROLE_POOLS[param.role]
        column = next((name for name in pool if name not in used), pool[0])
        used.add(column)
        bindings[param.name] = column
    return bindings


def _relation_for(dataset: LoadedDataset) -> str:
    return build_catalog([dataset]).relations[dataset.record.name]


def _store(tmp_path: Path) -> ArtifactStore:
    store = ArtifactStore(tmp_path / "ws")
    store.ensure_project("proj_seed", name="Seed")
    store.start_session("proj_seed", "run_seed")
    return store


# --- builtin seed library ----------------------------------------------------


def test_builtin_seeds_load_and_validate() -> None:
    seeds = load_builtin_seeds()

    assert len(seeds) >= 8, "seed library must ship at least 8 valid templates"
    seed_ids = [seed.seed_id for seed in seeds]
    assert len(seed_ids) == len(set(seed_ids)), "seed ids must be unique"
    for seed in seeds:
        assert seed.params, f"seed {seed.seed_id} has no params"
        assert "{dataset}" in seed.sql


def test_every_seed_instantiates_and_replays_through_gate(tmp_path: Path) -> None:
    dataset = _synthetic_dataset(tmp_path)
    relation = _relation_for(dataset)
    store = _store(tmp_path)
    project_dir = tmp_path / "proj"

    for template in load_builtin_seeds():
        skill = import_seed(
            project_dir,
            template,
            relation_name=relation,
            bindings=_bindings_for(template),
        )
        assert "{" not in skill.plan.sql, f"unsubstituted placeholder in {template.seed_id}"
        assert skill.expected_datasets == [relation]

        result = replay_skill(
            skill,
            [dataset],
            store=store,
            project_id="proj_seed",
            session_id="run_seed",
        )
        assert result.status == "answer", (
            f"seed {template.seed_id} failed replay: {result.message}"
        )
        sql_artifacts = [a for a in result.artifacts if a.type is ArtifactType.SQL_RESULT]
        assert sql_artifacts, f"seed {template.seed_id} produced no SqlResult"


def test_import_seed_is_idempotent(tmp_path: Path) -> None:
    dataset = _synthetic_dataset(tmp_path)
    relation = _relation_for(dataset)
    template = load_builtin_seeds()[0]
    bindings = _bindings_for(template)

    first = import_seed(tmp_path, template, relation_name=relation, bindings=bindings)
    second = import_seed(tmp_path, template, relation_name=relation, bindings=bindings)

    assert first.skill_id == second.skill_id
    assert len(load_skills(tmp_path)) == 1

    other = dict(bindings)
    other[template.params[0].name] = "order_id"
    import_seed(tmp_path, template, relation_name=relation, bindings=other)
    assert len(load_skills(tmp_path)) == 2


# --- instantiation guards ----------------------------------------------------


def _template() -> SeedSkillTemplate:
    return SeedSkillTemplate(
        seed_id="demo_seed",
        name="Demo seed",
        question="How does {value_col} vary?",
        sql="SELECT AVG({value_col}) AS avg_value FROM {dataset}",
        method="aggregation",
        rationale="Average of one measure.",
        params=[SeedParam(name="value_col", role="measure")],
    )


def test_instantiate_rejects_unbound_param() -> None:
    with pytest.raises(ToolGuardError) as excinfo:
        instantiate_seed(_template(), relation_name="sales", bindings={})
    assert "value_col" in str(excinfo.value)


def test_instantiate_rejects_unsafe_column_name() -> None:
    for evil in ("amount; DROP TABLE x", 'amount"', "amount--", "a b"):
        with pytest.raises(ToolGuardError):
            instantiate_seed(
                _template(), relation_name="sales", bindings={"value_col": evil}
            )


def test_instantiate_rejects_unknown_binding() -> None:
    with pytest.raises(ToolGuardError):
        instantiate_seed(
            _template(),
            relation_name="sales",
            bindings={"value_col": "amount", "extra_col": "region"},
        )


def test_dataset_param_name_is_rejected() -> None:
    # {dataset} is bound to the target relation at instantiation time; a param
    # named "dataset" would let the binding silently hijack (or be hijacked by)
    # that substitution.
    with pytest.raises(ValidationError, match="reserved"):
        SeedSkillTemplate(
            seed_id="hijack",
            name="Hijack",
            question="q",
            sql="SELECT {dataset} FROM {dataset}",
            method="m",
            rationale="r",
            params=[SeedParam(name="dataset", role="any")],
        )


def test_instantiate_rejects_overlong_identifiers() -> None:
    long_column = "c" * 5000
    with pytest.raises(ToolGuardError) as excinfo:
        instantiate_seed(_template(), relation_name="sales", bindings={"value_col": long_column})
    # The guard message must not echo the 5000-char value back verbatim.
    assert long_column not in str(excinfo.value)

    long_relation = "r" * 5000
    with pytest.raises(ToolGuardError) as excinfo:
        instantiate_seed(
            _template(), relation_name=long_relation, bindings={"value_col": "amount"}
        )
    assert long_relation not in str(excinfo.value)

    # Control: exactly 128 chars is still a legal identifier.
    boundary = "c" * 128
    skill = instantiate_seed(
        _template(), relation_name="sales", bindings={"value_col": boundary}
    )
    assert boundary in skill.plan.sql


def test_import_seed_revalidates_template_instantiability(tmp_path: Path) -> None:
    # A template built around model validation (e.g. hand-crafted JSON loaded
    # with model_construct) could smuggle an undeclared placeholder into the
    # frozen SQL; import must re-validate before persisting.
    broken = SeedSkillTemplate.model_construct(
        seed_id="broken",
        name="Broken",
        question="q",
        sql="SELECT {value_col}, {smuggled} FROM {dataset}",
        method="m",
        rationale="r",
        params=[SeedParam(name="value_col", role="measure")],
        source_url="",
    )

    with pytest.raises(ValueError):
        import_seed(tmp_path, broken, relation_name="sales", bindings={"value_col": "amount"})
    assert load_skills(tmp_path) == []


def test_seed_template_placeholder_mismatch_rejected() -> None:
    with pytest.raises(ValidationError):
        SeedSkillTemplate(
            seed_id="bad",
            name="Bad",
            question="q",
            sql="SELECT {other_col} FROM {dataset}",
            method="m",
            rationale="r",
            params=[SeedParam(name="value_col", role="measure")],
        )
    with pytest.raises(ValidationError):
        SeedSkillTemplate(
            seed_id="bad2",
            name="Bad2",
            question="q",
            sql="SELECT {value_col} FROM fixed_table",
            method="m",
            rationale="r",
            params=[SeedParam(name="value_col", role="measure")],
        )


# --- catalog block -----------------------------------------------------------


def test_catalog_block_empty_library_is_empty_string(tmp_path: Path) -> None:
    assert catalog_block(tmp_path) == ""


def test_catalog_block_lists_skill_names_and_columns(tmp_path: Path) -> None:
    dataset = _synthetic_dataset(tmp_path)
    template = load_builtin_seeds()[0]
    skill = import_seed(
        tmp_path,
        template,
        relation_name=_relation_for(dataset),
        bindings=_bindings_for(template),
    )

    block = catalog_block(tmp_path)

    assert skill.name in block
    assert skill.param_columns[0] in block
    assert len(block) <= 600


def test_catalog_block_stays_bounded_and_notes_omissions(tmp_path: Path) -> None:
    plan = AnalysisPlan(
        question="A deliberately long question about revenue drivers by segment",
        dataset_names=["sales"],
        columns=["region", "amount"],
        filters=[],
        sql="SELECT region, SUM(amount) FROM sales GROUP BY region",
        method="aggregation",
        rationale="r",
    )
    for index in range(30):
        add_skill(tmp_path, skill_from_plan(plan, f"Skill number {index} with a long name"))

    block = catalog_block(tmp_path)

    assert len(block) <= 600
    assert "omitted" in block


# --- question agent injection ------------------------------------------------


class SpyQuestionLLM:
    """Records every ``structured`` payload; returns one valid proposal."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        self.calls.append({"task": task, "payload": payload})
        return schema.model_validate(
            {
                "questions": [
                    {
                        "question_en": "Which region has the most revenue?",
                        "target_datasets": ["sales.csv"],
                        "llm_business_relevance": 0.8,
                        "llm_actionability": 0.7,
                    }
                ]
            }
        )

    def text(self, *, task: str, payload: dict) -> str:
        return "fake"

    def last_usage(self) -> None:
        return None


def _profile_artifact(tmp_path: Path) -> Any:
    from eda_platform.tools.profiler import profile_dataset

    csv_path = tmp_path / "sales.csv"
    csv_path.write_text("region,revenue\nEast,10\nWest,20\n", encoding="utf-8")
    loaded = load_csv(csv_path, dataset_id="ds_sales")
    return profile_dataset(loaded, project_id="project_demo", session_id="run_demo")


def test_question_agent_injects_skills_catalog(tmp_path: Path) -> None:
    block = "Saved analysis skills (validated, replayable SQL):\n- Demo: q (columns: a)"
    spy = SpyQuestionLLM()

    result = propose_llm_question_candidates(
        [_profile_artifact(tmp_path)],
        llm=spy,
        skills_catalog=block,
    )

    payload = spy.calls[0]["payload"]
    assert "reusable_skills" in payload
    assert "Demo" in payload["reusable_skills"]
    assert result.skills_catalog_chars == len(block)


def test_question_agent_empty_catalog_not_injected(tmp_path: Path) -> None:
    spy = SpyQuestionLLM()

    result = propose_llm_question_candidates(
        [_profile_artifact(tmp_path)],
        llm=spy,
        skills_catalog="",
    )

    assert "reusable_skills" not in spy.calls[0]["payload"]
    assert result.skills_catalog_chars == 0


