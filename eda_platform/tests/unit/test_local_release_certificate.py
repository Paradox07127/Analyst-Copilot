"""The operator-installed local certificate must open E4b through create_app.

These tests exercise the unmodified composition root: nothing replaces
``app.state.exploration_service``, so a regression that leaves the gate shut --
or one that opens it without a pinned issuer key -- fails here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from eda_platform.api.main import create_app
from eda_platform.application.services.exploration_service import (
    EXPLORATION_RELEASE_CERTIFICATE_ENV,
    EXPLORATION_RELEASE_TRUSTED_KEYS_ENV,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from issue_local_exploration_certificate import (  # noqa: E402, I001
    hard_caps_covering_every_tier,
    live_tool_capability_digest,
)
from issue_local_exploration_certificate import main as issue_local_certificate  # noqa: E402


@pytest.fixture
def installed_certificate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> tuple[Path, str]:
    out = tmp_path / "local.json"
    assert issue_local_certificate(["--out", str(out), "--providers", "openai"]) == 0
    exported = [
        line.split("=", 1)[1].strip().strip("\"'")
        for line in capsys.readouterr().out.splitlines()
        if line.startswith(f"export {EXPLORATION_RELEASE_TRUSTED_KEYS_ENV}=")
    ]
    assert len(exported) == 1
    monkeypatch.setenv(EXPLORATION_RELEASE_CERTIFICATE_ENV, str(out))
    return out, exported[0]


def test_capabilities_stay_closed_until_the_issuer_key_is_pinned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    installed_certificate: tuple[Path, str],
) -> None:
    _out, trusted = installed_certificate
    monkeypatch.delenv(EXPLORATION_RELEASE_TRUSTED_KEYS_ENV, raising=False)
    with TestClient(create_app(tmp_path / "closed")) as client:
        body = client.get("/api/v1/system/capabilities").json()
    assert body["exploration_available"] is False
    assert body["exploration_hint"]

    monkeypatch.setenv(EXPLORATION_RELEASE_TRUSTED_KEYS_ENV, trusted)
    with TestClient(create_app(tmp_path / "open")) as client:
        body = client.get("/api/v1/system/capabilities").json()
    assert body["exploration_available"] is True
    assert body["exploration_hint"] == ""


def test_exploration_router_answers_once_the_certificate_is_installed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    installed_certificate: tuple[Path, str],
) -> None:
    _out, trusted = installed_certificate
    monkeypatch.setenv(EXPLORATION_RELEASE_TRUSTED_KEYS_ENV, trusted)
    with TestClient(create_app(tmp_path / "workspace")) as client:
        response = client.get(
            "/api/v1/sessions/missing_session/explorations/expl_missing"
        )
    # The release gate no longer answers; the session lookup does.
    assert response.status_code == 404, response.text
    assert response.json()["error"]["code"] != "exploration_release_unavailable"


def test_a_tampered_certificate_body_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    installed_certificate: tuple[Path, str],
) -> None:
    out, trusted = installed_certificate
    monkeypatch.setenv(EXPLORATION_RELEASE_TRUSTED_KEYS_ENV, trusted)
    payload = json.loads(out.read_text(encoding="utf-8"))
    payload["hard_caps"]["max_cost_usd"] = payload["hard_caps"]["max_cost_usd"] * 10
    out.write_text(json.dumps(payload), encoding="utf-8")

    with TestClient(create_app(tmp_path / "tampered")) as client:
        body = client.get("/api/v1/system/capabilities").json()
    assert body["exploration_available"] is False


def test_issued_caps_cover_every_tier_and_bind_the_live_tool_inventory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "local.json"
    assert issue_local_certificate(["--out", str(out), "--providers", "openai"]) == 0
    capsys.readouterr()
    payload = json.loads(out.read_text(encoding="utf-8"))

    caps = hard_caps_covering_every_tier()
    assert payload["hard_caps"] == json.loads(caps.model_dump_json())
    assert (
        payload["bindings"]["tool_capability_digest"] == live_tool_capability_digest()
    )
    assert payload["providers"] == ["openai"]
