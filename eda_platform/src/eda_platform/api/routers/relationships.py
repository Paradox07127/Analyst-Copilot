"""Relationship endpoints (§7.5): read the run's relationship graph from
existing artifacts, prepare and execute a full validation as a background job,
and confirm/revoke the corresponding join (forwarded to the semantic layer,
which owns the whitelist rules)."""

from __future__ import annotations

from fastapi import APIRouter, Header, Request
from pydantic import BaseModel, Field

from eda_platform.api.errors import ApiErrorEnvelope
from eda_platform.application.dto import (
    RelationshipDiscoveryStarted,
    RelationshipEdge,
    RelationshipGraphView,
    RelationshipValidationPrepared,
    RelationshipValidationStarted,
)
from eda_platform.application.services.relationship_service import RelationshipService

_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    404: {"model": ApiErrorEnvelope},
    409: {"model": ApiErrorEnvelope},
    410: {"model": ApiErrorEnvelope},
    422: {"model": ApiErrorEnvelope},
    500: {"model": ApiErrorEnvelope},
}

router = APIRouter(tags=["relationships"], responses=_ERROR_RESPONSES)


class RelationshipValidateRequest(BaseModel):
    """No candidate content on purpose: validation runs what the approval froze."""

    action_hash: str = Field(min_length=8)
    approval_token: str = Field(min_length=8)


class RelationshipMutationRequest(BaseModel):
    expected_version: int = Field(ge=0)


def _service(request: Request) -> RelationshipService:
    return request.app.state.relationship_service


@router.get("/sessions/{session_id}/relationships", response_model=RelationshipGraphView)
def get_relationships(session_id: str, request: Request) -> RelationshipGraphView:
    return _service(request).get_graph(session_id)


@router.post(
    "/sessions/{session_id}/relationships/discover",
    status_code=201,
    response_model=RelationshipDiscoveryStarted,
    description=(
        "Discover relationship candidates for a run whose analysis deferred "
        "them. Reads every source CSV, so it runs as a background job on a "
        "derived rdsess_* run; the candidate artifacts land on this run."
    ),
)
def discover_relationships(
    session_id: str,
    request: Request,
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
) -> RelationshipDiscoveryStarted:
    return _service(request).discover(session_id, idempotency_key=idempotency_key)


@router.post(
    "/sessions/{session_id}/relationships/{relationship_id}/prepare-validate",
    response_model=RelationshipValidationPrepared,
)
def prepare_relationship_validation(
    session_id: str, relationship_id: str, request: Request
) -> RelationshipValidationPrepared:
    return _service(request).prepare_validate(session_id, relationship_id)


@router.post(
    "/sessions/{session_id}/relationships/{relationship_id}/validate",
    status_code=201,
    response_model=RelationshipValidationStarted,
)
def validate_relationship(
    session_id: str,
    relationship_id: str,
    body: RelationshipValidateRequest,
    request: Request,
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
) -> RelationshipValidationStarted:
    return _service(request).validate(
        session_id,
        relationship_id,
        action_hash=body.action_hash,
        approval_token=body.approval_token,
        idempotency_key=idempotency_key,
    )


@router.post(
    "/sessions/{session_id}/relationships/{relationship_id}/confirm",
    response_model=RelationshipEdge,
    description=(
        "Confirm the join behind this relationship. Delegates to the semantic "
        "layer: only a fresh, non-many-to-many validation may be confirmed."
    ),
)
def confirm_relationship(
    session_id: str,
    relationship_id: str,
    body: RelationshipMutationRequest,
    request: Request,
) -> RelationshipEdge:
    return _service(request).confirm(
        session_id, relationship_id, expected_version=body.expected_version
    )


@router.post(
    "/sessions/{session_id}/relationships/{relationship_id}/revoke",
    response_model=RelationshipEdge,
    description=(
        "Revoke an auto-confirmed join. A user-confirmed join is never "
        "downgraded (409 join_state_invalid)."
    ),
)
def revoke_relationship(
    session_id: str,
    relationship_id: str,
    body: RelationshipMutationRequest,
    request: Request,
) -> RelationshipEdge:
    return _service(request).revoke(
        session_id, relationship_id, expected_version=body.expected_version
    )
