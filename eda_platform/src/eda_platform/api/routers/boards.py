"""Investigation board endpoints (§7.5).

Writes retain their `expected_version` optimistic lock and also participate in
the shared, durable content-bound Idempotency-Key replay contract.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from eda_platform.api.errors import ApiErrorEnvelope
from eda_platform.application.dto import BoardCard, BoardColumn, BoardView
from eda_platform.application.services.board_service import (
    MAX_CARDS,
    MAX_COLUMNS,
    BoardService,
)

_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    404: {"model": ApiErrorEnvelope},
    409: {"model": ApiErrorEnvelope},
    422: {"model": ApiErrorEnvelope},
    500: {"model": ApiErrorEnvelope},
}

router = APIRouter(tags=["boards"], responses=_ERROR_RESPONSES)


class BoardUpdateRequest(BaseModel):
    expected_version: int = Field(ge=0)
    columns: list[BoardColumn] = Field(default_factory=list, max_length=MAX_COLUMNS)
    cards: list[BoardCard] = Field(default_factory=list, max_length=MAX_CARDS)


def _service(request: Request) -> BoardService:
    return request.app.state.board_service


@router.get("/projects/{project_id}/boards/{board_id}", response_model=BoardView)
def get_board(project_id: str, board_id: str, request: Request) -> BoardView:
    return _service(request).get_board(project_id, board_id)


@router.put("/projects/{project_id}/boards/{board_id}", response_model=BoardView)
def put_board(
    project_id: str, board_id: str, body: BoardUpdateRequest, request: Request
) -> BoardView:
    return _service(request).put_board(
        project_id,
        board_id,
        expected_version=body.expected_version,
        columns=body.columns,
        cards=body.cards,
    )
