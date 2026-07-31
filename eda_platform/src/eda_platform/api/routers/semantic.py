"""Semantic layer endpoints (§7.5): read the project's semantic knowledge from
one run, edit field-meaning seeds under optimistic locking (`expected_version`
→ 409 `version_conflict`), and review join-whitelist entries and machine
meaning proposals with idempotent POSTs."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, Field

from eda_platform.api.errors import ApiErrorEnvelope
from eda_platform.application.dto import (
    EntityNoteView,
    FieldMeaningView,
    JoinWhitelistEntryView,
    MetricDefinitionView,
    ProposalReviewed,
    SemanticSeedsUpdated,
    SemanticView,
    VerifiedAnswerView,
    VerifiedRelationsUpdated,
)
from eda_platform.application.services.semantic_service import SemanticService

_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    404: {"model": ApiErrorEnvelope},
    409: {"model": ApiErrorEnvelope},
    422: {"model": ApiErrorEnvelope},
    500: {"model": ApiErrorEnvelope},
}

router = APIRouter(tags=["semantic"], responses=_ERROR_RESPONSES)


class SemanticSeedsUpdateRequest(BaseModel):
    expected_version: int = Field(ge=0)
    # Required on purpose: omitting the list must be a 422, never "clear all".
    field_meanings: list[FieldMeaningView] = Field(max_length=500)
    # Optional on purpose: null/omitted leaves that class untouched, so a client
    # editing one class cannot clear the others. Send [] to empty a class.
    metric_definitions: list[MetricDefinitionView] | None = Field(
        default=None, max_length=500
    )
    entity_notes: list[EntityNoteView] | None = Field(default=None, max_length=500)
    verified_answers: list[VerifiedAnswerView] | None = Field(
        default=None, max_length=500
    )


class JoinReviewRequest(BaseModel):
    label: str = Field(min_length=1)
    expected_version: int = Field(ge=0)


class VerifiedRelationDeleteRequest(BaseModel):
    """Identity of the row to drop, not its list position."""

    left: str = Field(min_length=1)
    right: str = Field(min_length=1)
    expected_version: int = Field(ge=0)


class ProposalAcceptRequest(BaseModel):
    dataset: str = Field(min_length=1)
    column: str = Field(min_length=1)
    expected_version: int = Field(ge=0)
    meaning: str | None = None
    unit: str | None = None


class ProposalRejectRequest(BaseModel):
    dataset: str = Field(min_length=1)
    column: str = Field(min_length=1)
    expected_version: int = Field(ge=0)


def _service(request: Request) -> SemanticService:
    return request.app.state.semantic_service


@router.get("/sessions/{session_id}/semantic", response_model=SemanticView)
def get_semantic(
    session_id: str,
    request: Request,
    limit: int = Query(50, ge=1, le=200),
    cursor: str | None = Query(None),
) -> SemanticView:
    return _service(request).get_view(session_id, limit=limit, cursor=cursor)


@router.put(
    "/sessions/{session_id}/semantic/seeds",
    response_model=SemanticSeedsUpdated,
    description=(
        "Replace the project's hand-editable seed classes under optimistic "
        "locking (expected_version). All classes live in one seeds.json and "
        "share one version counter, so one PUT may carry several of them and "
        "any accepted PUT bumps the version for all.\n\n"
        "`field_meanings` is REQUIRED: omitting it is a 422, never 'clear "
        "all'. `metric_definitions`, `entity_notes` and `verified_answers` are "
        "OPTIONAL: omitted (or null) leaves that class exactly as stored, so "
        "an editor that only knows about one class cannot wipe the others; "
        "send an empty list to clear a class deliberately.\n\n"
        "Every supplied list fully replaces its class — there is no per-row "
        "patch. Seed classes this endpoint does not accept (verified "
        "relations, column role seeds) are always untouched."
    ),
)
def update_semantic_seeds(
    session_id: str, body: SemanticSeedsUpdateRequest, request: Request
) -> SemanticSeedsUpdated:
    return _service(request).update_seeds(
        session_id,
        expected_version=body.expected_version,
        field_meanings=body.field_meanings,
        metric_definitions=body.metric_definitions,
        entity_notes=body.entity_notes,
        verified_answers=body.verified_answers,
    )


@router.post(
    "/sessions/{session_id}/semantic/joins/confirm",
    response_model=JoinWhitelistEntryView,
)
def confirm_join(
    session_id: str, body: JoinReviewRequest, request: Request
) -> JoinWhitelistEntryView:
    return _service(request).confirm_whitelist_join(
        session_id, body.label, expected_version=body.expected_version
    )


@router.post(
    "/sessions/{session_id}/semantic/joins/revoke",
    response_model=JoinWhitelistEntryView,
)
def revoke_join(
    session_id: str, body: JoinReviewRequest, request: Request
) -> JoinWhitelistEntryView:
    return _service(request).revoke_whitelist_join(
        session_id, body.label, expected_version=body.expected_version
    )


@router.post(
    "/sessions/{session_id}/semantic/verified-relations/delete",
    response_model=VerifiedRelationsUpdated,
    description=(
        "Remove one verified relation from the project seeds, identified by its "
        "(left, right) seed keys. Relations are written by confirming a "
        "relationship on the Relationships page; this endpoint only removes "
        "them.\n\n"
        "Deleting a relation that is already absent succeeds unchanged, so a "
        "double submit is idempotent and never deletes a neighbouring row."
    ),
)
def delete_verified_relation(
    session_id: str, body: VerifiedRelationDeleteRequest, request: Request
) -> VerifiedRelationsUpdated:
    return _service(request).delete_verified_relation(
        session_id,
        left=body.left,
        right=body.right,
        expected_version=body.expected_version,
    )


@router.post(
    "/sessions/{session_id}/semantic/proposals/accept",
    response_model=ProposalReviewed,
)
def accept_proposal(
    session_id: str, body: ProposalAcceptRequest, request: Request
) -> ProposalReviewed:
    return _service(request).accept_meaning_proposal(
        session_id,
        dataset=body.dataset,
        column=body.column,
        meaning=body.meaning,
        unit=body.unit,
        expected_version=body.expected_version,
    )


@router.post(
    "/sessions/{session_id}/semantic/proposals/reject",
    response_model=ProposalReviewed,
)
def reject_proposal(
    session_id: str, body: ProposalRejectRequest, request: Request
) -> ProposalReviewed:
    return _service(request).reject_meaning_proposal(
        session_id,
        dataset=body.dataset,
        column=body.column,
        expected_version=body.expected_version,
    )
