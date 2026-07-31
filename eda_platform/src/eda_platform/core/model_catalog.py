"""Runtime model discovery with bounded, credential-safe HTTP reads."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from urllib import error, request

from eda_platform.core.llm import LLMSettings
from eda_platform.core.model_capabilities import is_verified_agent_model
from eda_platform.core.provider_registry import auth_style, provider_spec

MODEL_LIST_TIMEOUT_SECONDS = 15.0
MAX_MODEL_LIST_BYTES = 2 * 1024 * 1024
MAX_MODELS = 1000


class ModelCatalogError(RuntimeError):
    """A deliberately scrubbed model-list failure safe to show in Settings."""


@dataclass(frozen=True)
class DiscoveredModel:
    id: str
    owned_by: str = ""
    created: int | None = None
    input_usd_per_1m: float | None = None
    output_usd_per_1m: float | None = None
    pricing_source: str = ""
    verified: bool = False


@dataclass(frozen=True)
class DiscoveredCatalog:
    models: tuple[DiscoveredModel, ...]
    endpoint: str
    truncated: bool = False


class _NoRedirect(request.HTTPRedirectHandler):
    """Never forward a provider credential to another origin via redirects."""

    def redirect_request(self, *args: object, **kwargs: object) -> None:
        return None


def fetch_model_catalog(settings: LLMSettings) -> DiscoveredCatalog:
    spec = provider_spec(settings.provider)
    if spec.model_discovery == "none":
        raise ModelCatalogError("This provider has no live model-list endpoint.")
    if spec.requires_api_key and not settings.api_key:
        raise ModelCatalogError("Save an API key before refreshing models.")
    base_url = settings.resolved_base_url.rstrip("/")
    if not base_url:
        raise ModelCatalogError("Save a base URL before refreshing models.")

    path = "/v1/models" if spec.model_discovery == "anthropic" else "/models"
    endpoint = f"{base_url}{path}"
    headers = _headers(settings)
    req = request.Request(endpoint, headers=headers, method="GET")
    opener = request.build_opener(_NoRedirect)
    try:
        with opener.open(
            req,
            timeout=min(settings.timeout_seconds, MODEL_LIST_TIMEOUT_SECONDS),
        ) as response:
            raw = response.read(MAX_MODEL_LIST_BYTES + 1)
    except error.HTTPError as exc:
        raise ModelCatalogError(
            f"Provider model list returned HTTP {exc.code}."
        ) from exc
    except (error.URLError, TimeoutError, OSError) as exc:
        raise ModelCatalogError(
            f"Provider model list request failed ({type(exc).__name__})."
        ) from exc

    if len(raw) > MAX_MODEL_LIST_BYTES:
        raise ModelCatalogError("Provider model list exceeded the 2 MiB safety limit.")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelCatalogError("Provider model list was not valid JSON.") from exc

    items = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        # A few native-compatible catalogs use `models`; accept it without
        # rewriting their model identifiers.
        items = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise ModelCatalogError("Provider model list did not contain a model array.")

    by_id: dict[str, DiscoveredModel] = {}
    for raw_item in items:
        item = _parse_item(raw_item, provider=settings.provider.value)
        if item is not None:
            # Everything the provider serves is listed. Filtering to the
            # verified catalog meant a hardcoded snapshot silently overrode the
            # provider's own live answer, so a model released after this repo
            # was written could not be chosen at all.
            by_id.setdefault(
                item.id,
                replace(
                    item,
                    verified=is_verified_agent_model(settings.provider, item.id),
                ),
            )
        if len(by_id) >= MAX_MODELS:
            break
    if not by_id:
        raise ModelCatalogError("Provider model list contained no usable entries.")
    return DiscoveredCatalog(
        models=tuple(sorted(by_id.values(), key=lambda item: item.id.casefold())),
        endpoint=endpoint,
        truncated=(
            (isinstance(payload, dict) and payload.get("has_more") is True)
            or (len(items) > len(by_id) and len(by_id) >= MAX_MODELS)
        ),
    )


def _headers(settings: LLMSettings) -> dict[str, str]:
    headers = {"Accept": "application/json", **settings.headers}
    style = auth_style(settings.provider)
    if style == "anthropic":
        headers["x-api-key"] = settings.api_key
        headers["anthropic-version"] = "2023-06-01"
    elif style == "api_key_header" and settings.api_key:
        headers["api-key"] = settings.api_key
    elif settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"
    if settings.organization:
        headers["OpenAI-Organization"] = settings.organization
    return headers


def _parse_item(value: object, *, provider: str) -> DiscoveredModel | None:
    if not isinstance(value, dict):
        return None
    raw_id = value.get("id") or value.get("name")
    if not isinstance(raw_id, str) or not raw_id.strip():
        return None
    raw_owner = value.get("owned_by")
    owned_by = raw_owner if isinstance(raw_owner, str) else ""
    pricing = value.get("pricing")
    input_price = output_price = None
    pricing_source = ""
    if isinstance(pricing, dict):
        if provider == "openrouter":
            # OpenRouter documents prompt/completion as USD per token.
            input_price = _per_token_to_per_1m(pricing.get("prompt"))
            output_price = _per_token_to_per_1m(pricing.get("completion"))
        elif provider == "together":
            # Together documents input/output as USD per 1M tokens.
            input_price = _nonnegative_float(pricing.get("input"))
            output_price = _nonnegative_float(pricing.get("output"))
        if input_price is not None or output_price is not None:
            pricing_source = "provider_models_api"
    created = value.get("created")
    return DiscoveredModel(
        id=raw_id.strip(),
        owned_by=owned_by,
        created=created if isinstance(created, int) and not isinstance(created, bool) else None,
        input_usd_per_1m=input_price,
        output_usd_per_1m=output_price,
        pricing_source=pricing_source,
    )


def _per_token_to_per_1m(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        price = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    if price < 0:
        return None
    return float(price * Decimal(1_000_000))


def _nonnegative_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        price = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return float(price) if price >= 0 else None
