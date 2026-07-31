"""Developer-inspector read endpoints: the run debug rollup, the debug.jsonl
download, and the captured LLM payload forensics feed."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from eda_platform.api.main import create_app
from eda_platform.core.bounded_pagination import JsonlPageIndex
from eda_platform.core.dev_log import LLM_DEBUG_FILENAME
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.sessions import SessionManifest, TraceEvent

PROJECT = "demo"
RUN = "run_debug"
BARE_RUN = "run_bare"

_START = datetime(2026, 7, 25, 10, 0, 0, tzinfo=UTC)


def _event(
    event_type: str,
    name: str,
    summary: dict,
    *,
    offset_s: int = 0,
    duration_s: float = 0.5,
) -> TraceEvent:
    started = _START + timedelta(seconds=offset_s)
    return TraceEvent(
        session_id=RUN,
        event_type=event_type,
        name=name,
        started_at=started,
        finished_at=started + timedelta(seconds=duration_s),
        summary=summary,
    )


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    store = ArtifactStore(tmp_path)
    store.ensure_project(PROJECT, name="Demo")
    store.start_session(PROJECT, RUN)
    store.start_session(PROJECT, BARE_RUN)
    store.write_manifest(
        SessionManifest(session_id=RUN, project_id=PROJECT, input_hashes={}, code_version="abc123")
    )
    store.save_artifact(
        Artifact(
            id="prof_1",
            type=ArtifactType.DATASET_PROFILE,
            project_id=PROJECT,
            session_id=RUN,
            payload={"dataset_id": "ds_1", "name": "orders.csv", "rows": 5, "columns": 1},
            warnings=["small sample"],
        )
    )
    for event in (
        _event(
            "llm_call",
            "m2_report_claim_plan",
            {
                "provider": "openai",
                "model": "gpt-4.1-mini",
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "total_tokens": 150,
                "estimated_cost_usd": 0.0012,
                "schema": "ClaimPlan",
                "attempt": 1,
                "status": "success",
            },
        ),
        _event(
            "tool_completed",
            "profile_dataset",
            {"row_count": 5, "truncated": False, "artifact_id": "prof_1"},
            offset_s=1,
        ),
        _event(
            "step_failed",
            "create_charts",
            {"error_type": "ValueError", "error": "no numeric columns"},
            offset_s=2,
        ),
        _event(
            "report_validation",
            "report",
            {
                "section_coverage": 0.8,
                "claim_section_coverage": 0.5,
                "claim_survival_rate": 0.75,
                "deterministic_repair_count": 2,
                "critical_count": 0,
            },
            offset_s=3,
        ),
    ):
        store.append_trace(PROJECT, event)
    return store


@pytest.fixture
def client(store: ArtifactStore) -> TestClient:
    return TestClient(create_app(store.root))


# -- run debug rollup -----------------------------------------------------


def test_debug_rollup(client: TestClient) -> None:
    response = client.get(f"/api/v1/sessions/{RUN}/debug")
    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == RUN
    assert body["code_version"] == "abc123"

    summary = body["summary"]
    assert summary["events"] == 4
    assert summary["artifacts"] == 1
    assert summary["llm_calls"] == 1
    assert summary["tool_calls"] == 1
    assert summary["errors"] == 1
    assert summary["total_tokens"] == 150
    assert summary["estimated_cost_usd"] == pytest.approx(0.0012)

    quality = body["report_quality"]
    assert quality["section_coverage"] == pytest.approx(0.8)
    assert quality["claim_survival_rate"] == pytest.approx(0.75)
    assert quality["deterministic_repair_count"] == 2
    assert quality["prompt_tokens_by_attempt"] == "1: 120"

    assert [row["event_type"] for row in body["timeline"]] == [
        "llm_call",
        "tool_completed",
        "step_failed",
        "report_validation",
    ]
    assert body["timeline"][0]["duration_ms"] == 500

    llm_row = body["llm_calls"][0]
    assert llm_row["task"] == "m2_report_claim_plan"
    assert llm_row["model"] == "gpt-4.1-mini"
    assert llm_row["total_tokens"] == 150
    assert llm_row["schema"] == "ClaimPlan"

    tool_row = body["tool_calls"][0]
    assert tool_row["tool"] == "profile_dataset"
    assert tool_row["row_count"] == 5
    assert tool_row["truncated"] is False

    assert body["errors"][0]["error"] == "no numeric columns"
    assert body["artifacts"] == [
        {"artifact_id": "prof_1", "type": "DatasetProfile", "parents": 0, "warnings": 1}
    ]


def test_debug_empty_run(client: TestClient) -> None:
    response = client.get(f"/api/v1/sessions/{BARE_RUN}/debug")
    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["events"] == 0
    assert body["summary"]["artifacts"] == 0
    assert body["code_version"] is None
    assert body["timeline"] == []
    assert body["llm_calls"] == []
    assert body["tool_calls"] == []
    assert body["errors"] == []
    assert body["artifacts"] == []
    assert body["report_quality"]["prompt_tokens_by_attempt"] == ""


def test_debug_rollup_strips_workspace_paths(store: ArtifactStore) -> None:
    """Summaries and exception text are producer-controlled prose; the server's
    absolute workspace path must not ride out inside them."""
    absolute = str((store.root / "projects" / PROJECT / "uploads" / "a.csv").resolve())
    store.append_trace(
        PROJECT,
        _event(
            "tool_failed",
            "load_csv",
            {"error": f"could not parse {absolute}", "path": absolute},
            offset_s=9,
        ),
    )
    client = TestClient(create_app(store.root))
    body = client.get(f"/api/v1/sessions/{RUN}/debug").json()
    blob = json.dumps(body)
    assert str(store.root.resolve()) not in blob
    assert f"projects/{PROJECT}/uploads/a.csv" in blob


def test_debug_unknown_run_404(client: TestClient) -> None:
    response = client.get("/api/v1/sessions/nope/debug")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "session_not_found"


def test_debug_uses_sql_keyset_pages_not_full_store_lists(
    store: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        ArtifactStore,
        "list_trace_events",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("debug must not materialize all trace events")
        ),
    )
    monkeypatch.setattr(
        ArtifactStore,
        "list_artifacts_safe",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("debug must not materialize all artifacts")
        ),
    )
    client = TestClient(create_app(store.root))
    first = client.get(f"/api/v1/sessions/{RUN}/debug", params={"limit": 2})
    assert first.status_code == 200, first.text
    assert len(first.json()["timeline"]) <= 2
    assert len(first.json()["artifacts"]) <= 2
    cursor = first.json()["next_cursor"]
    assert cursor
    second = client.get(
        f"/api/v1/sessions/{RUN}/debug", params={"limit": 2, "cursor": cursor}
    )
    assert second.status_code == 200, second.text
    assert second.json()["timeline"] != first.json()["timeline"]


# -- debug.jsonl download -------------------------------------------------


def test_debug_log_download_streams_file(store: ArtifactStore) -> None:
    lines = "".join(
        json.dumps({"session_id": RUN, "event_type": "step_started", "name": f"s{i}"}) + "\n"
        for i in range(3)
    )
    (store.session_dir(PROJECT, RUN) / "debug.jsonl").write_text(lines, encoding="utf-8")
    client = TestClient(create_app(store.root))
    response = client.get(f"/api/v1/sessions/{RUN}/debug/log")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert 'filename="debug.jsonl"' in response.headers["content-disposition"]
    assert "content-length" not in response.headers
    assert response.text == lines


def test_debug_log_download_redacts_workspace_and_api_key(
    store: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "sk-sensitive-test-key"
    monkeypatch.setenv("EDA_LLM_API_KEY", secret)
    absolute = str((store.root / "projects" / PROJECT / "uploads" / "a.csv").resolve())
    line = (
        json.dumps(
            {
                "summary": {
                    "detail": f"{'x' * (70 * 1024)}; path={absolute}; key={secret}"
                }
            }
        )
        + "\n"
    )
    (store.session_dir(PROJECT, RUN) / "debug.jsonl").write_text(line, encoding="utf-8")
    response = TestClient(create_app(store.root)).get(f"/api/v1/sessions/{RUN}/debug/log")
    assert response.status_code == 200
    assert str(store.root.resolve()) not in response.text
    assert secret not in response.text
    assert "projects/" in response.text
    assert "***" in response.text


def test_debug_log_missing_404(client: TestClient) -> None:
    response = client.get(f"/api/v1/sessions/{RUN}/debug/log")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "debug_log_not_found"
    # The envelope must not disclose where the workspace lives on disk.
    assert "/" not in response.json()["error"]["message"]


def test_debug_log_unknown_run_404(client: TestClient) -> None:
    response = client.get("/api/v1/sessions/nope/debug/log")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "session_not_found"


def test_debug_log_symlink_escape_404(store: ArtifactStore, tmp_path: Path) -> None:
    """A debug.jsonl symlinked outside the run directory must not be served."""
    outside = tmp_path / "outside.jsonl"
    outside.write_text('{"secret": true}\n', encoding="utf-8")
    link = store.session_dir(PROJECT, RUN) / "debug.jsonl"
    link.symlink_to(outside)
    client = TestClient(create_app(store.root))
    response = client.get(f"/api/v1/sessions/{RUN}/debug/log")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "debug_log_not_found"


# -- captured LLM payloads ------------------------------------------------


def _write_llm_debug(store: ArtifactStore, count: int) -> None:
    path = store.session_dir(PROJECT, RUN) / LLM_DEBUG_FILENAME
    path.write_text(
        "".join(
            json.dumps(
                {
                    "ts": f"2026-07-25T10:00:0{index}+00:00",
                    "kind": "structured",
                    "task": f"task_{index}",
                    "status": "success",
                    "duration_s": 1.25,
                    "model": "gpt-4.1-mini",
                    "prompt_tokens": 10 * index,
                    "completion_tokens": index,
                    "estimated_cost_usd": 0.0001,
                    "payload_preview": f"payload {index}",
                    "response_preview": f"response {index}",
                }
            )
            + "\n"
            for index in range(count)
        ),
        encoding="utf-8",
    )


def test_llm_calls_feed(store: ArtifactStore) -> None:
    _write_llm_debug(store, 3)
    client = TestClient(create_app(store.root))
    response = client.get(f"/api/v1/sessions/{RUN}/debug/llm-calls")
    assert response.status_code == 200
    body = response.json()
    assert body["next_cursor"] is None
    assert [item["task"] for item in body["items"]] == ["task_0", "task_1", "task_2"]
    first = body["items"][0]
    assert first["index"] == 1
    assert first["ts"] == "2026-07-25T10:00:00+00:00"
    assert first["kind"] == "structured"
    assert first["model"] == "gpt-4.1-mini"
    assert first["status"] == "success"
    assert first["payload_preview"] == "payload 0"
    assert first["response_preview"] == "response 0"
    assert first["duration_s"] == pytest.approx(1.25)


def test_llm_calls_pagination(store: ArtifactStore) -> None:
    _write_llm_debug(store, 5)
    client = TestClient(create_app(store.root))
    first = client.get(f"/api/v1/sessions/{RUN}/debug/llm-calls", params={"limit": 2})
    body = first.json()
    assert [item["index"] for item in body["items"]] == [1, 2]
    assert body["next_cursor"]

    second = client.get(
        f"/api/v1/sessions/{RUN}/debug/llm-calls",
        params={"limit": 2, "cursor": body["next_cursor"]},
    )
    assert [item["index"] for item in second.json()["items"]] == [3, 4]

    third = client.get(
        f"/api/v1/sessions/{RUN}/debug/llm-calls",
        params={"limit": 2, "cursor": second.json()["next_cursor"]},
    )
    assert [item["index"] for item in third.json()["items"]] == [5]
    assert third.json()["next_cursor"] is None


def test_large_llm_debug_page_never_uses_read_text(
    store: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_llm_debug(store, 5_000)
    capture = (store.session_dir(PROJECT, RUN) / LLM_DEBUG_FILENAME).resolve()
    original = Path.read_text

    def guarded(
        path: Path, encoding: str | None = None, errors: str | None = None
    ) -> str:
        if path.resolve() == capture:
            raise AssertionError("LLM JSONL must be streamed, not materialized")
        return original(path, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", guarded)
    client = TestClient(create_app(store.root))
    body = client.get(
        f"/api/v1/sessions/{RUN}/debug/llm-calls", params={"limit": 3}
    ).json()
    assert [item["index"] for item in body["items"]] == [1, 2, 3]
    assert body["next_cursor"]

    def no_rescan(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("an unchanged indexed LLM capture must not be rescanned")

    monkeypatch.setattr(JsonlPageIndex, "_extend_index", no_rescan)
    second = client.get(
        f"/api/v1/sessions/{RUN}/debug/llm-calls",
        params={"limit": 3, "cursor": body["next_cursor"]},
    )
    assert second.status_code == 200


def test_llm_cursor_rejects_changed_capture(store: ArtifactStore) -> None:
    _write_llm_debug(store, 10)
    client = TestClient(create_app(store.root))
    first = client.get(
        f"/api/v1/sessions/{RUN}/debug/llm-calls", params={"limit": 2}
    ).json()
    path = store.session_dir(PROJECT, RUN) / LLM_DEBUG_FILENAME
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"task": "new"}) + "\n")
    stale = client.get(
        f"/api/v1/sessions/{RUN}/debug/llm-calls",
        params={"limit": 2, "cursor": first["next_cursor"]},
    )
    assert stale.status_code == 400
    assert stale.json()["error"]["code"] == "invalid_cursor"


def test_llm_calls_missing_file_is_empty_page(client: TestClient) -> None:
    """No capture file is a normal empty page, not a 404."""
    response = client.get(f"/api/v1/sessions/{RUN}/debug/llm-calls")
    assert response.status_code == 200
    assert response.json() == {"items": [], "next_cursor": None}


def test_llm_calls_bad_cursor_400(store: ArtifactStore) -> None:
    _write_llm_debug(store, 2)
    client = TestClient(create_app(store.root))
    response = client.get(f"/api/v1/sessions/{RUN}/debug/llm-calls", params={"cursor": "abc"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_cursor"


def test_llm_calls_unknown_run_404(client: TestClient) -> None:
    response = client.get("/api/v1/sessions/nope/debug/llm-calls")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "session_not_found"


def test_llm_calls_redact_api_key_and_workspace_path(
    store: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Captured payloads echo whatever was sent; a configured key and the
    server's absolute workspace path must not reach the response."""
    monkeypatch.setenv("EDA_LLM_PROVIDER", "openai")
    monkeypatch.setenv("EDA_LLM_API_KEY", "sk-supersecret-value-1234")
    absolute = str((store.root / "projects" / PROJECT / "uploads" / "a.csv").resolve())
    path = store.session_dir(PROJECT, RUN) / LLM_DEBUG_FILENAME
    path.write_text(
        json.dumps(
            {
                "ts": "2026-07-25T10:00:00+00:00",
                "kind": "text",
                "task": "t",
                "status": "error: AuthError: bad key sk-supersecret-value-1234",
                "payload_preview": f"file={absolute}",
                "response_preview": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = TestClient(create_app(store.root))
    item = client.get(f"/api/v1/sessions/{RUN}/debug/llm-calls").json()["items"][0]
    assert "sk-supersecret-value-1234" not in item["status"]
    assert "***" in item["status"]
    assert item["payload_preview"] == f"file=projects/{PROJECT}/uploads/a.csv"
