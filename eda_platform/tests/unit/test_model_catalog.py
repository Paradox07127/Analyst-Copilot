"""Live model discovery and the snapshot it falls back to."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from eda_platform.api.main import create_app
from eda_platform.application.dto import SettingsPatch
from eda_platform.application.services.settings_service import SettingsService
from eda_platform.core.llm import LLMSettings
from eda_platform.core.model_capabilities import (
    AGENT_MODEL_REGISTRY,
    agent_model_ids,
    agent_model_profiles,
    is_verified_agent_model,
)
from eda_platform.core.model_catalog import ModelCatalogError, fetch_model_catalog
from eda_platform.core.provider_registry import LLMProvider


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self, size: int | None = None) -> bytes:
        return self._body

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _patch_open(monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]) -> list[Any]:
    seen: list[Any] = []

    class _Opener:
        def open(self, req: Any, timeout: float | None = None) -> _Response:
            seen.append(req)
            return _Response(payload)

    monkeypatch.setattr(
        "eda_platform.core.model_catalog.request.build_opener",
        lambda *_a, **_k: _Opener(),
    )
    return seen


def test_openai_shape_is_read_from_the_data_array(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _patch_open(
        monkeypatch,
        {"data": [{"id": "gpt-4.1-mini", "owned_by": "openai", "created": 1}]},
    )
    settings = LLMSettings(provider=LLMProvider.OPENAI, model="gpt-4.1-mini", api_key="k")

    catalog = fetch_model_catalog(settings)

    assert [model.id for model in catalog.models] == ["gpt-4.1-mini"]
    assert catalog.endpoint.endswith("/models")
    assert seen[0].headers["Authorization"] == "Bearer k"


def test_anthropic_uses_its_own_path_and_auth_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _patch_open(monkeypatch, {"data": [{"id": "claude-opus-5"}]})
    settings = LLMSettings(
        provider=LLMProvider.ANTHROPIC, model="claude-opus-5", api_key="secret"
    )

    catalog = fetch_model_catalog(settings)

    assert catalog.endpoint.endswith("/v1/models")
    assert seen[0].headers["X-api-key"] == "secret"
    assert "Authorization" not in seen[0].headers


def test_openrouter_per_token_pricing_is_scaled_to_per_million(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenRouter documents prompt/completion as USD per token; showing those
    raw would put $0.0000006 on the settings page."""
    _patch_open(
        monkeypatch,
        {
            "data": [
                {
                    "id": "openai/gpt-5.6-terra",
                    "pricing": {"prompt": "0.0000006", "completion": "0.0000024"},
                }
            ]
        },
    )
    settings = LLMSettings(
        provider=LLMProvider.OPENROUTER,
        model="openai/gpt-5.6-terra",
        api_key="k",
    )

    model = fetch_model_catalog(settings).models[0]

    assert model.input_usd_per_1m == pytest.approx(0.60)
    assert model.output_usd_per_1m == pytest.approx(2.40)
    assert model.pricing_source == "provider_models_api"


def test_live_catalog_lists_everything_and_marks_what_is_verified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Filtering to the verified catalog let a hardcoded snapshot override the
    provider's own live answer, so a model newer than this repo was unreachable."""
    _patch_open(
        monkeypatch,
        {
            "data": [
                {"id": "gpt-5.6-sol"},
                {"id": "text-embedding-3-large"},
                {"id": "unknown-chat-model"},
            ]
        },
    )
    settings = LLMSettings(provider=LLMProvider.OPENAI, model="gpt-5.6-sol", api_key="k")

    catalog = fetch_model_catalog(settings)

    verified = {item.id: item.verified for item in catalog.models}
    assert set(verified) == {"gpt-5.6-sol", "text-embedding-3-large", "unknown-chat-model"}
    assert verified["gpt-5.6-sol"] is True
    assert verified["unknown-chat-model"] is False


def test_a_provider_with_no_list_endpoint_is_refused_before_any_request() -> None:
    settings = LLMSettings(provider=LLMProvider.OFFLINE, model="offline-deterministic")
    with pytest.raises(ModelCatalogError):
        fetch_model_catalog(settings)


def test_missing_credential_is_refused_before_any_request() -> None:
    settings = LLMSettings(provider=LLMProvider.OPENAI, model="gpt-4.1-mini", api_key="")
    with pytest.raises(ModelCatalogError):
        fetch_model_catalog(settings)


# --- service level ----------------------------------------------------------
def test_a_failed_fetch_falls_back_to_the_snapshot_and_says_so(tmp_path: Path) -> None:
    service = SettingsService(
        workspace=tmp_path.resolve(), defaults=LLMSettings(provider=LLMProvider.OFFLINE)
    )
    service.update_settings(
        SettingsPatch(provider="openai", api_key="k", model="gpt-4.1")
    )

    catalog = service.list_models()

    assert catalog.source == "snapshot"
    assert catalog.warning  # why it fell back, scrubbed
    assert "gpt-4.1" in {model.id for model in catalog.models}
    # The shipped table still prices the fallback list.
    priced = {model.id: model.input_usd_per_1m for model in catalog.models}
    assert priced["gpt-4.1"] == 2.0


def test_the_saved_key_never_appears_in_the_warning(tmp_path: Path) -> None:
    service = SettingsService(
        workspace=tmp_path.resolve(), defaults=LLMSettings(provider=LLMProvider.OFFLINE)
    )
    service.update_settings(
        SettingsPatch(provider="openai", api_key="sk-supersecret", model="gpt-4.1")
    )

    catalog = service.list_models()

    assert "sk-supersecret" not in catalog.warning


def test_endpoints_expose_the_catalog_and_a_forced_refresh(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path))

    listed = client.get("/api/v1/settings/models")
    refreshed = client.post("/api/v1/settings/models/refresh")

    assert listed.status_code == 200
    assert refreshed.status_code == 200
    body = listed.json()
    assert body["source"] in {"live", "snapshot"}
    assert body["pricing_catalog_version"]
    assert "invoice" in body["pricing_notice"]


def test_tool_calling_flag_actually_withdraws_a_model() -> None:
    """`tool_calling` used to be decoration: verification answered from registry
    membership alone, so a profile could declare False and stay verified. The
    flag has to be the thing being asked about."""
    provider = LLMProvider.OPENAI
    live = agent_model_profiles(provider)[0]

    assert is_verified_agent_model(provider, live.model_id)
    assert live.model_id in agent_model_ids(provider)

    withdrawn = replace(live, tool_calling=False)
    patched = {**AGENT_MODEL_REGISTRY, provider: (withdrawn, *agent_model_profiles(provider)[1:])}
    with patch.dict(AGENT_MODEL_REGISTRY, patched, clear=True):
        assert not is_verified_agent_model(provider, withdrawn.model_id)
        assert withdrawn.model_id not in agent_model_ids(provider)
