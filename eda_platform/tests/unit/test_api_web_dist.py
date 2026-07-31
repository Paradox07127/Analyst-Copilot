"""create_app(serve_web_dist=...): SPA fallback, /api priority, static assets."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from eda_platform.api.main import create_app

INDEX_MARKER = "<title>spa-index</title>"


@pytest.fixture
def dist(tmp_path: Path) -> Path:
    root = tmp_path / "dist"
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text(f"<html><head>{INDEX_MARKER}</head></html>")
    (root / "assets" / "app.js").write_text("console.log('app')")
    (root / "favicon.svg").write_text("<svg/>")
    return root


@pytest.fixture
def client(tmp_path: Path, dist: Path) -> TestClient:
    app = create_app(workspace=tmp_path / "ws", serve_web_dist=dist)
    return TestClient(app)


def test_root_serves_index(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert INDEX_MARKER in response.text
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["cache-control"] == "no-cache"


def test_deep_link_falls_back_to_index(client: TestClient) -> None:
    for path in ("/projects", "/projects/p1/sessions/r1/data-map", "/no/such/route"):
        response = client.get(path)
        assert response.status_code == 200, path
        assert INDEX_MARKER in response.text, path


def test_api_routes_keep_priority(client: TestClient) -> None:
    response = client.get("/api/v1/projects")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert INDEX_MARKER not in response.text


def test_unknown_api_path_is_404_not_index(client: TestClient) -> None:
    for path in ("/api/v1/nonexistent", "/api", "/api/"):
        response = client.get(path)
        assert response.status_code == 404, path
        assert INDEX_MARKER not in response.text, path


def test_static_asset_served(client: TestClient) -> None:
    response = client.get("/assets/app.js")
    assert response.status_code == 200
    assert response.text == "console.log('app')"


def test_root_level_file_served(client: TestClient) -> None:
    response = client.get("/favicon.svg")
    assert response.status_code == 200
    assert response.text == "<svg/>"


def test_openapi_and_docs_not_hijacked(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert response.json()["info"]["title"] == "EDA Agent Platform API"


def test_file_outside_dist_not_leaked(client: TestClient, tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("top-secret")
    for path in ("/secret.txt", "/../secret.txt", "/%2e%2e/secret.txt"):
        response = client.get(path)
        assert "top-secret" not in response.text, path


def test_without_dist_root_stays_404(tmp_path: Path) -> None:
    app = create_app(workspace=tmp_path / "ws")
    client = TestClient(app)
    assert client.get("/").status_code == 404
    assert client.get("/api/v1/projects").status_code == 200


def test_missing_index_html_fails_fast(tmp_path: Path) -> None:
    empty = tmp_path / "empty-dist"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        create_app(workspace=tmp_path / "ws", serve_web_dist=empty)
