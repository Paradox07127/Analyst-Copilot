"""Report/Artifact endpoints: happy paths, none-status 200, typed 404s,
pagination + type filter over HTTP."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import eda_platform.application.services.artifact_service as artifact_service_module
import eda_platform.application.services.report_service as report_service_module
from eda_platform.api.main import create_app
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.tools.agent_handoff import create_agent_handoff_artifact

PROJECT = "demo"
RUN = "run_1"
BARE_RUN = "run_2"


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    store = ArtifactStore(tmp_path)
    store.ensure_project(PROJECT, name="Demo")
    store.start_session(PROJECT, RUN)
    store.start_session(PROJECT, BARE_RUN)
    store.save_artifact(
        Artifact(
            id="md_1",
            type=ArtifactType.MARKDOWN_REPORT,
            project_id=PROJECT,
            session_id=RUN,
            payload={"markdown": "# Demo report\n\n| a | b |\n|---|---|\n| 1 | 2 |"},
        )
    )
    for index in range(3):
        store.save_artifact(
            Artifact(
                id=f"chart_{index}",
                type=ArtifactType.CHART_SPEC,
                project_id=PROJECT,
                session_id=RUN,
                payload={"title": f"Chart {index}"},
            )
        )
    return tmp_path


@pytest.fixture
def client(workspace: Path) -> TestClient:
    return TestClient(create_app(workspace))


def test_get_report(client: TestClient) -> None:
    response = client.get(f"/api/v1/sessions/{RUN}/report")
    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == RUN
    assert body["markdown"].startswith("# Demo report")
    assert body["status"] != "none"
    assert body["generated_at"]


def test_get_report_none_is_200(client: TestClient) -> None:
    response = client.get(f"/api/v1/sessions/{BARE_RUN}/report")
    assert response.status_code == 200
    assert response.json() == {
        "session_id": BARE_RUN,
        "status": "none",
        "markdown": "",
        "generated_at": None,
    }


def test_report_file_over_read_cap_is_413(
    client: TestClient, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(report_service_module, "MAX_REPORT_READ_BYTES", 16)
    report = workspace / "projects" / PROJECT / "sessions" / BARE_RUN / "report" / "report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("x" * 17, encoding="utf-8")
    response = client.get(f"/api/v1/sessions/{BARE_RUN}/report")
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "report_too_large"


def test_get_report_unknown_run_404(client: TestClient) -> None:
    response = client.get("/api/v1/sessions/missing/report")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "session_not_found"


def test_list_artifacts_paginates(client: TestClient) -> None:
    first = client.get(f"/api/v1/sessions/{RUN}/artifacts", params={"limit": 2})
    assert first.status_code == 200
    body = first.json()
    assert [item["artifact_id"] for item in body["items"]] == ["md_1", "chart_0"]
    assert all("payload" not in item for item in body["items"])
    assert body["next_cursor"]

    second = client.get(
        f"/api/v1/sessions/{RUN}/artifacts",
        params={"limit": 2, "cursor": body["next_cursor"]},
    )
    assert [item["artifact_id"] for item in second.json()["items"]] == ["chart_1", "chart_2"]
    assert second.json()["next_cursor"] is None


def test_list_artifacts_type_filter(client: TestClient) -> None:
    response = client.get(f"/api/v1/sessions/{RUN}/artifacts", params={"type": "ChartSpec"})
    assert response.status_code == 200
    items = response.json()["items"]
    assert [item["type"] for item in items] == ["ChartSpec"] * 3


def test_list_artifacts_limit_validated(client: TestClient) -> None:
    response = client.get(f"/api/v1/sessions/{RUN}/artifacts", params={"limit": 101})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_list_artifacts_bad_cursor_400(client: TestClient) -> None:
    response = client.get(f"/api/v1/sessions/{RUN}/artifacts", params={"cursor": "%%%"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_cursor"


def test_list_artifacts_cursor_type_mismatch_400(client: TestClient) -> None:
    first = client.get(f"/api/v1/sessions/{RUN}/artifacts", params={"limit": 2})
    cursor = first.json()["next_cursor"]
    assert cursor

    response = client.get(
        f"/api/v1/sessions/{RUN}/artifacts",
        params={"cursor": cursor, "type": "ChartSpec"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_cursor"


def test_get_artifact_too_large_413(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(artifact_service_module, "MAX_ARTIFACT_PAYLOAD_BYTES", 16)
    response = client.get(f"/api/v1/sessions/{RUN}/artifacts/chart_1")
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "artifact_too_large"


def test_get_artifact_detail(client: TestClient) -> None:
    response = client.get(f"/api/v1/sessions/{RUN}/artifacts/chart_1")
    assert response.status_code == 200
    body = response.json()
    assert body["artifact_id"] == "chart_1"
    assert body["type"] == "ChartSpec"
    assert body["payload"] == {"title": "Chart 1"}
    assert body["parents"] == []
    assert body["evidence"] == []
    assert body["env_digest"]


def test_agent_handoff_publish_barrier_and_typed_endpoint(
    client: TestClient, workspace: Path
) -> None:
    not_ready = client.get(f"/api/v1/sessions/{BARE_RUN}/agent-handoff")
    assert not_ready.status_code == 409
    assert not_ready.headers["retry-after"] == "2"
    assert not_ready.json()["error"]["code"] == "agent_handoff_not_ready"

    store = ArtifactStore(workspace)
    source = store.list_artifacts(project_id=PROJECT, session_id=RUN)
    handoff = create_agent_handoff_artifact(
        source,
        project_id=PROJECT,
        session_id=RUN,
        producer_version="test",
        execution_fingerprint="fingerprint",
        input_hashes={},
    )
    store.save_artifact(handoff)
    store.mark_session_status(PROJECT, RUN, "completed")
    response = client.get(f"/api/v1/sessions/{RUN}/agent-handoff")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body["artifact_id"] == handoff.id
    assert body["type"] == "AgentHandoff"
    assert body["payload"]["contract_version"] == "3.0"

    store.mark_session_status(PROJECT, BARE_RUN, "completed")
    missing = client.get(f"/api/v1/sessions/{BARE_RUN}/agent-handoff")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "agent_handoff_not_found"


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_agent_handoff_terminal_session_is_non_retryable_409(
    client: TestClient, workspace: Path, status: str
) -> None:
    ArtifactStore(workspace).mark_session_status(PROJECT, BARE_RUN, status)

    response = client.get(f"/api/v1/sessions/{BARE_RUN}/agent-handoff")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "agent_handoff_terminal"
    assert "retry-after" not in response.headers


def test_get_artifact_missing_404(client: TestClient) -> None:
    response = client.get(f"/api/v1/sessions/{RUN}/artifacts/nope")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "artifact_not_found"
