"""Investigation board API: empty read for a board that was never written,
version-incrementing PUTs under the optimistic lock, 409 on a stale version,
and 422 for board structures the UI must never be able to persist."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eda_platform.api.main import create_app
from eda_platform.core.store import ArtifactStore

PROJECT = "proj_board"
BOARD = "investigation"


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    store = ArtifactStore(tmp_path)
    store.ensure_project(PROJECT, "Board project")
    return tmp_path


@pytest.fixture()
def client(workspace: Path) -> TestClient:
    app: FastAPI = create_app(workspace)
    return TestClient(app)


def _board(*, columns: list[dict], cards: list[dict], expected_version: int) -> dict:
    return {"expected_version": expected_version, "columns": columns, "cards": cards}


def _simple(expected_version: int, *, second_column_cards: list[str] | None = None) -> dict:
    return _board(
        expected_version=expected_version,
        columns=[
            {"id": "todo", "title": "To check", "card_ids": ["c1"]},
            {"id": "done", "title": "Confirmed", "card_ids": second_column_cards or []},
        ],
        cards=[
            {
                "id": "c1",
                "title": "Revenue dip in March",
                "ref_type": "finding",
                "ref_id": "find_1",
                "note": "",
            }
        ],
    )


def test_missing_board_reads_as_empty_version_zero(client: TestClient) -> None:
    response = client.get(f"/api/v1/projects/{PROJECT}/boards/{BOARD}")
    assert response.status_code == 200
    body = response.json()
    assert body == {
        "project_id": PROJECT,
        "board_id": BOARD,
        "version": 0,
        "columns": [],
        "cards": [],
    }


def test_put_increments_version_and_persists(client: TestClient, workspace: Path) -> None:
    first = client.put(f"/api/v1/projects/{PROJECT}/boards/{BOARD}", json=_simple(0))
    assert first.status_code == 200
    assert first.json()["version"] == 1

    second = client.put(
        f"/api/v1/projects/{PROJECT}/boards/{BOARD}",
        json=_simple(1, second_column_cards=[]),
    )
    assert second.status_code == 200
    assert second.json()["version"] == 2

    stored = json.loads(
        (workspace / "projects" / PROJECT / "boards" / f"{BOARD}.json").read_text(
            encoding="utf-8"
        )
    )
    assert stored["version"] == 2
    assert stored["cards"][0]["ref_id"] == "find_1"

    reread = client.get(f"/api/v1/projects/{PROJECT}/boards/{BOARD}").json()
    assert reread["version"] == 2
    assert [column["id"] for column in reread["columns"]] == ["todo", "done"]


def test_stale_expected_version_conflicts(client: TestClient) -> None:
    created = client.put(f"/api/v1/projects/{PROJECT}/boards/{BOARD}", json=_simple(0))
    assert created.status_code == 200

    stale = client.put(f"/api/v1/projects/{PROJECT}/boards/{BOARD}", json=_simple(0))
    assert stale.status_code == 409
    assert stale.json()["error"]["code"] == "version_conflict"

    # The conflicting write must not have landed.
    assert client.get(f"/api/v1/projects/{PROJECT}/boards/{BOARD}").json()["version"] == 1


def test_put_idempotency_replays_across_app_restart(
    client: TestClient, workspace: Path
) -> None:
    headers = {"Idempotency-Key": "board-response-loss-retry"}
    first = client.put(
        f"/api/v1/projects/{PROJECT}/boards/{BOARD}",
        json=_simple(0),
        headers=headers,
    )
    assert first.status_code == 200
    assert first.json()["version"] == 1

    # Recreate the application to prove replay lives in state.sqlite rather
    # than process memory. The stale expected_version must never reach the
    # board service on this retry.
    restarted = TestClient(create_app(workspace))
    replay = restarted.put(
        f"/api/v1/projects/{PROJECT}/boards/{BOARD}",
        json=_simple(0),
        headers=headers,
    )
    assert replay.status_code == 200
    assert replay.content == first.content
    assert restarted.get(
        f"/api/v1/projects/{PROJECT}/boards/{BOARD}"
    ).json()["version"] == 1


def test_put_idempotency_key_rejects_different_board_content(
    client: TestClient,
) -> None:
    headers = {"Idempotency-Key": "board-content-bound-key"}
    first = client.put(
        f"/api/v1/projects/{PROJECT}/boards/{BOARD}",
        json=_simple(0),
        headers=headers,
    )
    changed = client.put(
        f"/api/v1/projects/{PROJECT}/boards/{BOARD}",
        json=_simple(0, second_column_cards=["c1"]),
        headers=headers,
    )

    assert first.status_code == 200
    assert changed.status_code == 422
    assert changed.json()["error"]["code"] == "idempotency_key_reused"
    assert client.get(
        f"/api/v1/projects/{PROJECT}/boards/{BOARD}"
    ).json()["version"] == 1


def test_card_in_two_columns_is_rejected(client: TestClient) -> None:
    response = client.put(
        f"/api/v1/projects/{PROJECT}/boards/{BOARD}",
        json=_board(
            expected_version=0,
            columns=[
                {"id": "todo", "title": "To check", "card_ids": ["c1"]},
                {"id": "done", "title": "Confirmed", "card_ids": ["c1"]},
            ],
            cards=[{"id": "c1", "title": "Dup", "ref_type": "none", "ref_id": ""}],
        ),
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "board_invalid"


def test_orphan_card_is_rejected(client: TestClient) -> None:
    response = client.put(
        f"/api/v1/projects/{PROJECT}/boards/{BOARD}",
        json=_board(
            expected_version=0,
            columns=[{"id": "todo", "title": "To check", "card_ids": []}],
            cards=[{"id": "c1", "title": "Nowhere", "ref_type": "none", "ref_id": ""}],
        ),
    )
    assert response.status_code == 422
    assert "not placed in any column" in response.json()["error"]["message"]


def test_unknown_card_reference_is_rejected(client: TestClient) -> None:
    response = client.put(
        f"/api/v1/projects/{PROJECT}/boards/{BOARD}",
        json=_board(
            expected_version=0,
            columns=[{"id": "todo", "title": "To check", "card_ids": ["ghost"]}],
            cards=[],
        ),
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "board_invalid"


def test_unknown_ref_type_is_rejected(client: TestClient) -> None:
    response = client.put(
        f"/api/v1/projects/{PROJECT}/boards/{BOARD}",
        json=_board(
            expected_version=0,
            columns=[{"id": "todo", "title": "To check", "card_ids": ["c1"]}],
            cards=[{"id": "c1", "title": "Bad ref", "ref_type": "sql", "ref_id": "x"}],
        ),
    )
    assert response.status_code == 422
    assert "ref_type" in response.json()["error"]["message"]


def test_duplicate_card_id_in_the_cards_array_is_rejected(client: TestClient) -> None:
    """Two cards sharing an id would make every lookup by id ambiguous, even
    though each id is placed in exactly one column."""
    response = client.put(
        f"/api/v1/projects/{PROJECT}/boards/{BOARD}",
        json=_board(
            expected_version=0,
            columns=[{"id": "todo", "title": "To check", "card_ids": ["c1"]}],
            cards=[
                {"id": "c1", "title": "First", "ref_type": "none", "ref_id": ""},
                {"id": "c1", "title": "Second", "ref_type": "none", "ref_id": ""},
            ],
        ),
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "board_invalid"
    assert "Duplicate card id" in response.json()["error"]["message"]

    # Nothing landed: the board is still unwritten.
    assert client.get(f"/api/v1/projects/{PROJECT}/boards/{BOARD}").json()["version"] == 0


def test_board_id_outside_the_id_charset_is_rejected(client: TestClient) -> None:
    response = client.get(f"/api/v1/projects/{PROJECT}/boards/bad%20id")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "board_invalid"


def test_unknown_project_is_404(client: TestClient) -> None:
    response = client.get(f"/api/v1/projects/nope/boards/{BOARD}")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "project_not_found"
