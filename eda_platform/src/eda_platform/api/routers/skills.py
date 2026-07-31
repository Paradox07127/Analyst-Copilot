"""Skills endpoints (§10.3): browse the project skill library plus the builtin
seed templates, prepare a bound replay for approval, and execute it as a
`skill_replay` job on a derived run."""

from __future__ import annotations

from fastapi import APIRouter, Header, Query, Request
from pydantic import BaseModel, Field

from eda_platform.api.errors import ApiErrorEnvelope
from eda_platform.application.dto import (
    SkillReplayPrepared,
    SkillReplayStarted,
    SkillSummary,
    SkillsView,
)
from eda_platform.application.services.skill_service import (
    MAX_SKILL_DESCRIPTION_CHARS,
    MAX_SKILL_NAME_CHARS,
    SkillService,
)

_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    404: {"model": ApiErrorEnvelope},
    409: {"model": ApiErrorEnvelope},
    410: {"model": ApiErrorEnvelope},
    422: {"model": ApiErrorEnvelope},
    500: {"model": ApiErrorEnvelope},
}

router = APIRouter(tags=["skills"], responses=_ERROR_RESPONSES)


class SkillReplayPrepareRequest(BaseModel):
    dataset_ids: list[str] = Field(min_length=1, max_length=20)
    bindings: dict[str, str] = Field(default_factory=dict, max_length=20)
    """Seed placeholder → column of the target dataset; empty for saved skills."""


class SkillReplayExecuteRequest(BaseModel):
    """No skill content on purpose: execute replays what the approval froze."""

    action_hash: str = Field(min_length=8)
    approval_token: str = Field(min_length=8)


class SeedImportRequest(BaseModel):
    dataset_ids: list[str] = Field(min_length=1, max_length=20)
    """Exactly one dataset of this run: a seed is instantiated on one table."""

    bindings: dict[str, str] = Field(default_factory=dict, max_length=20)
    """Seed placeholder → column of the target dataset."""

    name: str = Field(default="", max_length=MAX_SKILL_NAME_CHARS)
    """Optional override; empty keeps the seed template's own name."""


class SkillSaveRequest(BaseModel):
    source_artifact_id: str = Field(min_length=1, max_length=200)
    """A plan artifact of this run, from SkillsView.savable_plans."""

    name: str = Field(min_length=1, max_length=MAX_SKILL_NAME_CHARS)
    description: str = Field(default="", max_length=MAX_SKILL_DESCRIPTION_CHARS)


def _service(request: Request) -> SkillService:
    return request.app.state.skill_service


@router.get("/sessions/{session_id}/skills", response_model=SkillsView)
def list_skills(
    session_id: str,
    request: Request,
    limit: int = Query(50, ge=1, le=200),
    cursor: str | None = Query(None),
) -> SkillsView:
    return _service(request).list_skills(session_id, limit=limit, cursor=cursor)


@router.post("/sessions/{session_id}/skills", status_code=201, response_model=SkillSummary)
def save_skill(session_id: str, body: SkillSaveRequest, request: Request) -> SkillSummary:
    """Freeze one of this run's validated plan artifacts into a named skill."""
    return _service(request).save_skill(
        session_id,
        source_artifact_id=body.source_artifact_id,
        name=body.name,
        description=body.description,
    )


@router.post(
    "/sessions/{session_id}/skills/{seed_id}/import",
    status_code=201,
    response_model=SkillSummary,
    description=(
        "Bind a builtin seed template to one dataset of this run and save the "
        "result into the project skill library, where it outlives the run. "
        "Idempotent on the (seed, relation, bindings) triple: re-importing the "
        "same binding replaces that skill in place instead of duplicating it."
    ),
)
def import_seed_skill(
    session_id: str, seed_id: str, body: SeedImportRequest, request: Request
) -> SkillSummary:
    return _service(request).import_seed_skill(
        session_id,
        seed_id,
        dataset_ids=body.dataset_ids,
        bindings=body.bindings,
        name=body.name,
    )


@router.delete("/projects/{project_id}/skills/{skill_id}", status_code=204)
def delete_skill(project_id: str, skill_id: str, request: Request) -> None:
    """Remove a saved skill from the project library. Builtin seed templates
    are not deletable (409)."""
    _service(request).delete_skill(project_id, skill_id)


@router.post(
    "/sessions/{session_id}/skills/{skill_id}/prepare",
    response_model=SkillReplayPrepared,
)
def prepare_skill_replay(
    session_id: str,
    skill_id: str,
    body: SkillReplayPrepareRequest,
    request: Request,
) -> SkillReplayPrepared:
    return _service(request).prepare_replay(
        session_id,
        skill_id,
        dataset_ids=body.dataset_ids,
        bindings=body.bindings,
    )


@router.post(
    "/sessions/{session_id}/skills/{skill_id}/execute",
    status_code=201,
    response_model=SkillReplayStarted,
)
def execute_skill_replay(
    session_id: str,
    skill_id: str,
    body: SkillReplayExecuteRequest,
    request: Request,
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
) -> SkillReplayStarted:
    return _service(request).execute_replay(
        session_id,
        skill_id,
        action_hash=body.action_hash,
        approval_token=body.approval_token,
        idempotency_key=idempotency_key,
    )
