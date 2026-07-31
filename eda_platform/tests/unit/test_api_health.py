"""Health check API contract."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from eda_platform.api.main import create_app


def test_health_check_reports_ok(tmp_path: Path) -> None:
    response = TestClient(create_app(tmp_path)).get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_check_is_documented_in_openapi(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path))

    operation = client.get("/openapi.json").json()["paths"]["/api/v1/health"]["get"]
    response_schema = operation["responses"]["200"]["content"]["application/json"][
        "schema"
    ]

    assert operation["tags"] == ["system"]
    assert response_schema == {
        "$ref": "#/components/schemas/HealthStatusView"
    }
