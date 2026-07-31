"""Project-level persistence for :class:`AnalysisSkill` and the builtin seed library."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from importlib import resources
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from eda_platform.core.ids import make_artifact_id
from eda_platform.core.tool_guard import GuardViolation, raise_for_violations
from eda_platform.schemas.plans import AnalysisPlan
from eda_platform.schemas.skills import AnalysisSkill, SeedSkillTemplate


class SkillLibrary(BaseModel):
    """Versioned envelope for a project's saved skills."""

    version: int = 1
    skills: list[AnalysisSkill] = Field(default_factory=list)


def _skills_path(project_dir: Path | str) -> Path:
    return Path(project_dir) / "skills" / "skills.json"


def load_skills(project_dir: Path | str) -> list[AnalysisSkill]:
    """Load a project's skills; a missing/corrupt file yields an empty list."""
    path = _skills_path(project_dir)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    try:
        if isinstance(data, list):
            return [AnalysisSkill.model_validate(item) for item in data]
        return list(SkillLibrary.model_validate(data).skills)
    except (ValidationError, ValueError, TypeError):
        return []


def save_skills(project_dir: Path | str, skills: list[AnalysisSkill]) -> Path:
    """Atomically persist the skill list, creating ``skills/`` if needed."""
    path = _skills_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    library = SkillLibrary(skills=list(skills))
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(library.model_dump_json(indent=2), encoding="utf-8")
    os.replace(tmp_path, path)
    return path


def add_skill(project_dir: Path | str, skill: AnalysisSkill) -> list[AnalysisSkill]:
    """Append a skill, idempotent on ``skill_id`` (re-adding replaces in place)."""
    skills = load_skills(project_dir)
    for index, existing in enumerate(skills):
        if existing.skill_id == skill.skill_id:
            skills[index] = skill
            break
    else:
        skills.append(skill)
    save_skills(project_dir, skills)
    return skills


# --- builtin seed library ----------------------------------------------------

_SEED_TOOL = "import_seed_skill"
_SEED_RESOURCE = "seed_skills.json"
# Bound values are interpolated into SQL, so they must be plain identifiers.
_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# Real column/relation names never approach this; oversized values would ride
# guard error messages verbatim into the UI.
_MAX_IDENTIFIER_CHARS = 128


def shown_identifier(value: str) -> str:
    """Violation display value: never echo an oversized identifier verbatim."""
    if len(value) <= 64:
        return value
    return f"{value[:64]}… ({len(value)} chars)"


def is_bindable_identifier(value: str) -> bool:
    """Whether :func:`_guard_bindings` would accept this value as a bound column.

    Lets the API mark unbindable columns up front instead of letting the form
    offer them and refusing at prepare time (review J4).
    """
    return len(value) <= _MAX_IDENTIFIER_CHARS and bool(_SAFE_IDENTIFIER_RE.match(value))


def load_builtin_seeds() -> list[SeedSkillTemplate]:
    """Load the packaged seed templates; a missing/corrupt resource yields []."""
    try:
        text = (
            resources.files("eda_platform.resources").joinpath(_SEED_RESOURCE).read_text("utf-8")
        )
        data = json.loads(text)
        return [SeedSkillTemplate.model_validate(item) for item in data]
    except (OSError, ValueError, TypeError, ValidationError):
        return []


def instantiate_seed(
    template: SeedSkillTemplate,
    *,
    relation_name: str,
    bindings: Mapping[str, str],
) -> AnalysisSkill:
    """Bind a seed's placeholders to concrete columns, yielding a replayable skill.

    The skill_id is deterministic over (seed, relation, bindings) so re-importing
    the same instantiation replaces in place instead of duplicating.
    """
    _guard_bindings(template, relation_name=relation_name, bindings=bindings)
    substitutions = {**dict(bindings), "dataset": relation_name}
    sql = _substitute(template.sql, substitutions)
    question = _substitute(template.question, substitutions)
    columns: list[str] = []
    for param in template.params:
        column = bindings[param.name]
        if column not in columns:
            columns.append(column)
    plan = AnalysisPlan(
        question=question,
        dataset_names=[relation_name],
        columns=columns,
        filters=[],
        sql=sql,
        method=template.method,
        rationale=template.rationale,
        estimated_scan="small",
    )
    bound = ", ".join(f"{param.name}={bindings[param.name]}" for param in template.params)
    return AnalysisSkill(
        skill_id=make_artifact_id(
            "skillseed",
            {
                "seed_id": template.seed_id,
                "relation": relation_name,
                "bindings": dict(sorted(bindings.items())),
            },
        ),
        name=template.name,
        description=f"From seed '{template.seed_id}' on {relation_name} ({bound}).",
        plan=plan,
        param_columns=columns,
        expected_datasets=[relation_name],
    )


def import_seed(
    project_dir: Path | str,
    template: SeedSkillTemplate,
    *,
    relation_name: str,
    bindings: Mapping[str, str],
) -> AnalysisSkill:
    """Instantiate a seed and persist it into the project skill library (idempotent).

    Re-validates the template first: an instance built around model validation
    (e.g. model_construct on hand-crafted JSON) could otherwise freeze an
    undeclared placeholder into the persisted SQL.
    """
    try:
        template = SeedSkillTemplate.model_validate(template.model_dump())
    except ValidationError as exc:
        raise ValueError(
            f"seed template '{template.seed_id}' is not instantiable: {exc}"
        ) from exc
    skill = instantiate_seed(template, relation_name=relation_name, bindings=bindings)
    add_skill(project_dir, skill)
    return skill


def _guard_bindings(
    template: SeedSkillTemplate,
    *,
    relation_name: str,
    bindings: Mapping[str, str],
) -> None:
    violations: list[GuardViolation] = []
    if len(relation_name) > _MAX_IDENTIFIER_CHARS:
        violations.append(
            GuardViolation(
                field="relation_name",
                got=shown_identifier(relation_name),
                allowed=f"an identifier of at most {_MAX_IDENTIFIER_CHARS} characters",
                fix_hint="Pass the catalog relation name of the target dataset.",
                problem="relation name exceeds the identifier length limit.",
            )
        )
    elif not _SAFE_IDENTIFIER_RE.match(relation_name):
        violations.append(
            GuardViolation(
                field="relation_name",
                got=relation_name,
                allowed="a plain SQL identifier (letters, digits, underscore)",
                fix_hint="Pass the catalog relation name of the target dataset.",
                problem="relation name is not a safe identifier.",
            )
        )
    param_names = {param.name for param in template.params}
    for param in template.params:
        value = bindings.get(param.name)
        if value is None:
            violations.append(
                GuardViolation(
                    field=f"bindings[{param.name}]",
                    got=None,
                    allowed="a column name of the target dataset",
                    fix_hint=f"Bind the {{{param.name}}} placeholder to a column.",
                    problem="placeholder is unbound.",
                )
            )
        elif len(value) > _MAX_IDENTIFIER_CHARS:
            violations.append(
                GuardViolation(
                    field=f"bindings[{param.name}]",
                    got=shown_identifier(value),
                    allowed=f"an identifier of at most {_MAX_IDENTIFIER_CHARS} characters",
                    fix_hint="Pick a real column of the target dataset.",
                    problem="bound value exceeds the identifier length limit.",
                )
            )
        elif not _SAFE_IDENTIFIER_RE.match(value):
            violations.append(
                GuardViolation(
                    field=f"bindings[{param.name}]",
                    got=value,
                    allowed="a plain column identifier (letters, digits, underscore)",
                    fix_hint=(
                        "Pick a column whose name is a plain identifier; rename the "
                        "column upstream if it contains spaces or punctuation."
                    ),
                    problem="bound value is not a safe SQL identifier.",
                )
            )
    for extra in sorted(set(bindings) - param_names):
        violations.append(
            GuardViolation(
                field=f"bindings[{extra}]",
                got=bindings[extra],
                allowed=", ".join(sorted(param_names)),
                fix_hint="Bind only the placeholders this seed declares.",
                problem="binding does not match any declared placeholder.",
            )
        )
    raise_for_violations(_SEED_TOOL, violations)


def _substitute(text: str, substitutions: Mapping[str, str]) -> str:
    for name, value in substitutions.items():
        text = text.replace("{" + name + "}", value)
    return text


# --- planner-facing catalog --------------------------------------------------

_CATALOG_HEADER = "Saved analysis skills (validated, replayable SQL):"
_CATALOG_QUESTION_CHARS = 120


def catalog_block(project_dir: Path | str, *, max_chars: int = 600) -> str:
    """Render the project's skills as a compact context block; empty library -> ''."""
    skills = load_skills(project_dir)
    if not skills:
        return ""
    entries = [
        f"- {skill.name}: {_flat(skill.plan.question)}"
        f" (columns: {', '.join(skill.param_columns)})"
        for skill in skills
    ]
    total = len(entries)
    note_reserve = len(f"\n... ({total} more skills omitted)")
    out = [_CATALOG_HEADER]
    included = 0
    for line in entries:
        candidate = [*out, line]
        will_omit_more = (included + 1) < total
        projected = len("\n".join(candidate)) + (note_reserve if will_omit_more else 0)
        if included and projected > max_chars:
            break
        out = candidate
        included += 1
    if included < total:
        out.append(f"... ({total - included} more skills omitted)")
    block = "\n".join(out)
    return block if len(block) <= max_chars else block[:max_chars]


def _flat(text: str, limit: int = _CATALOG_QUESTION_CHARS) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else f"{flat[: limit - 1]}…"
