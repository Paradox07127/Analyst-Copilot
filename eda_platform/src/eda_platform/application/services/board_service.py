"""Investigation board persistence (§7.5 / 阶段5 slice J).

One JSON document per board at ``projects/<pid>/boards/<board_id>.json``. The
edit counter lives inside that document (not in a sidecar like semantic seeds)
because the board format is service-owned end to end, so there is no core
writer whose file shape must stay untouched.

Concurrency: `self._lock` serializes read-modify-write within ONE process.
Running the API with multiple workers needs an external lock around boards/.
"""

from __future__ import annotations

import json
import os
import re
import secrets
from pathlib import Path
from threading import Lock

from eda_platform.application.dto import BoardCard, BoardColumn, BoardView
from eda_platform.application.services.session_service import ProjectNotFoundError
from eda_platform.core.store import ArtifactStore

MAX_COLUMNS = 20
MAX_CARDS = 500
VALID_REF_TYPES = frozenset({"none", "finding", "question", "artifact"})

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")


class BoardServiceError(Exception):
    pass


class BoardValidationError(BoardServiceError):
    pass


class BoardVersionConflictError(BoardServiceError):
    def __init__(self, expected_version: int, current_version: int) -> None:
        super().__init__(
            f"Board changed since it was loaded: expected version {expected_version}, "
            f"current version is {current_version}. Reload and retry."
        )
        self.expected_version = expected_version
        self.current_version = current_version


class BoardStateError(BoardServiceError):
    """The stored board document is unreadable or malformed."""


class BoardService:
    def __init__(self, store: ArtifactStore) -> None:
        self._store = store
        self._lock = Lock()

    def get_board(self, project_id: str, board_id: str) -> BoardView:
        self._require_project(project_id)
        board_id = self._require_board_id(board_id)
        path = self._board_path(project_id, board_id)
        if not path.exists():
            return BoardView(project_id=project_id, board_id=board_id, version=0)
        return self._read(path, project_id, board_id)

    def put_board(
        self,
        project_id: str,
        board_id: str,
        *,
        expected_version: int,
        columns: list[BoardColumn],
        cards: list[BoardCard],
    ) -> BoardView:
        self._require_project(project_id)
        board_id = self._require_board_id(board_id)
        if expected_version < 0:
            raise BoardValidationError("expected_version must not be negative.")
        self._validate(columns, cards)
        path = self._board_path(project_id, board_id)
        with self._lock:
            current = (
                self._read(path, project_id, board_id).version if path.exists() else 0
            )
            if expected_version != current:
                raise BoardVersionConflictError(expected_version, current)
            board = BoardView(
                project_id=project_id,
                board_id=board_id,
                version=current + 1,
                columns=columns,
                cards=cards,
            )
            self._write(path, board)
        return board

    def _validate(self, columns: list[BoardColumn], cards: list[BoardCard]) -> None:
        if len(columns) > MAX_COLUMNS:
            raise BoardValidationError(f"A board may hold at most {MAX_COLUMNS} columns.")
        if len(cards) > MAX_CARDS:
            raise BoardValidationError(f"A board may hold at most {MAX_CARDS} cards.")

        column_ids: set[str] = set()
        for column in columns:
            if not _ID_RE.match(column.id):
                raise BoardValidationError(f"Invalid column id: {column.id!r}")
            if column.id in column_ids:
                raise BoardValidationError(f"Duplicate column id: {column.id}")
            column_ids.add(column.id)
            if not column.title.strip():
                raise BoardValidationError(f"Column {column.id} needs a title.")

        card_ids: set[str] = set()
        for card in cards:
            if not _ID_RE.match(card.id):
                raise BoardValidationError(f"Invalid card id: {card.id!r}")
            if card.id in card_ids:
                raise BoardValidationError(f"Duplicate card id: {card.id}")
            card_ids.add(card.id)
            if not card.title.strip():
                raise BoardValidationError(f"Card {card.id} needs a title.")
            if card.ref_type not in VALID_REF_TYPES:
                raise BoardValidationError(
                    f"Card {card.id} has unknown ref_type {card.ref_type!r}; "
                    f"expected one of {', '.join(sorted(VALID_REF_TYPES))}."
                )
            if card.ref_type != "none" and not card.ref_id.strip():
                raise BoardValidationError(
                    f"Card {card.id} references {card.ref_type} but carries no ref_id."
                )

        # Every card sits in exactly one column: a card listed twice (or in no
        # column) would render duplicated or silently vanish after a drag.
        placed: set[str] = set()
        for column in columns:
            for card_id in column.card_ids:
                if card_id not in card_ids:
                    raise BoardValidationError(
                        f"Column {column.id} references unknown card {card_id}."
                    )
                if card_id in placed:
                    raise BoardValidationError(
                        f"Card {card_id} appears in more than one column."
                    )
                placed.add(card_id)
        orphans = sorted(card_ids - placed)
        if orphans:
            raise BoardValidationError(
                "Cards are not placed in any column: " + ", ".join(orphans)
            )

    def _read(self, path: Path, project_id: str, board_id: str) -> BoardView:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise BoardStateError(
                f"Board {board_id} is unreadable; restore or delete the file to recover."
            ) from exc
        if not isinstance(payload, dict):
            raise BoardStateError(f"Board {board_id} is malformed.")
        version = payload.get("version")
        if not (isinstance(version, int) and version >= 0):
            raise BoardStateError(f"Board {board_id} has a malformed version counter.")
        try:
            return BoardView(
                project_id=project_id,
                board_id=board_id,
                version=version,
                columns=[BoardColumn.model_validate(item) for item in payload.get("columns", [])],
                cards=[BoardCard.model_validate(item) for item in payload.get("cards", [])],
            )
        except ValueError as exc:
            raise BoardStateError(f"Board {board_id} is malformed.") from exc

    def _write(self, path: Path, board: BoardView) -> None:
        """tmp + os.replace, same atomic-write convention as semantic seeds."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": board.version,
            "columns": [column.model_dump(mode="json") for column in board.columns],
            "cards": [card.model_dump(mode="json") for card in board.cards],
        }
        tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_path, path)

    def _board_path(self, project_id: str, board_id: str) -> Path:
        return self._store.project_dir(project_id) / "boards" / f"{board_id}.json"

    def _require_board_id(self, board_id: str) -> str:
        if not _ID_RE.match(board_id):
            raise BoardValidationError(f"Invalid board id: {board_id!r}")
        return board_id

    def _require_project(self, project_id: str) -> None:
        if not self._store.project_exists(project_id):
            raise ProjectNotFoundError(project_id)
