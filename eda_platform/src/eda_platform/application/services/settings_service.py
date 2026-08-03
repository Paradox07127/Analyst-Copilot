"""Runtime LLM + analysis settings, scoped to a caller session (§6.0).

Session overrides live in this process's memory only. The API key never leaves
it: it is not persisted, not logged, not written to a trace event, and has no
field on any response DTO — `SettingsView` carries only a boolean and the last
4 characters.

Multi-worker limit: the store is a plain dict on one process. Behind two or
more uvicorn workers a session that configured worker A reads env defaults on
worker B, so single-worker is the supported deployment until this moves to a
shared store (Redis/SQLite). Not a correctness hazard — the fallback is the env
default, never another session's settings.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlsplit

from eda_platform import __version__
from eda_platform.application.dto import (
    AboutInfo,
    ConnectionTestResult,
    ModelCatalog,
    ModelCatalogItem,
    ProviderInfo,
    SettingsPatch,
    SettingsView,
)
from eda_platform.core.config import default_workspace, require_absolute_workspace
from eda_platform.core.env import (
    load_llm_settings_from_env_file,
    load_provider_api_keys_from_env_file,
)
from eda_platform.core.exploration_tiers import (
    ANALYSIS_DEPTH_TO_EXPLORATION_TIER,
    exploration_tier_for_analysis_depth,
)
from eda_platform.core.llm import (
    LLMConfigurationStatus,
    LLMProvider,
    LLMSettings,
    create_llm_client,
    llm_configuration_status,
)
from eda_platform.core.model_capabilities import (
    CAPABILITY_CATALOG_VERSION,
    agent_model_ids,
    agent_model_profile,
    is_verified_agent_model,
    selectable_model_ids,
)
from eda_platform.core.model_catalog import (
    DiscoveredModel,
    ModelCatalogError,
    fetch_model_catalog,
)
from eda_platform.core.provider_registry import (
    PRICING_CATALOG_VERSION,
    PRICING_CHECKED_AT,
    PROVIDER_REGISTRY,
    provider_spec,
)
from eda_platform.core.request_dialect import forget_learned_repairs
from eda_platform.core.tool_calling_probe import forget_probe_results

DEFAULT_SESSION_ID = "default"
DEFAULT_SESSION_TTL_SECONDS = 12 * 60 * 60
# Bounds the dict when many browser sessions hit one process; the oldest
# session is evicted and simply falls back to env defaults on its next call.
MAX_SESSIONS = 200

DEFAULT_PAYLOAD_POLICY = "schema+aggregates"
PAYLOAD_POLICIES = ("schema_only", "schema+aggregates", "schema+aggregates+sample")
STRUCTURED_MODES = ("auto", "json_schema", "json_object")

# The exploration meaning of this legacy integer has one authority in
# core.exploration_tiers: 0 quick, 1 standard, 2/3 deep. Existing investigation
# thresholds remain below until that older workflow is retired.
DEFAULT_ANALYSIS_DEPTH = min(ANALYSIS_DEPTH_TO_EXPLORATION_TIER)
MIN_ANALYSIS_DEPTH = min(ANALYSIS_DEPTH_TO_EXPLORATION_TIER)
MAX_ANALYSIS_DEPTH = max(ANALYSIS_DEPTH_TO_EXPLORATION_TIER)
DEEP_INVESTIGATION_DEPTH = 1
MACRO_LOOP_DEPTH = 2

MIN_TEMPERATURE, MAX_TEMPERATURE = 0.0, 2.0
MIN_MAX_TOKENS, MAX_MAX_TOKENS = 256, 200_000
MIN_TIMEOUT_SECONDS, MAX_TIMEOUT_SECONDS = 10.0, 600.0
# Bounds for Model & API pricing inputs.
MIN_USD_PER_1K, MAX_USD_PER_1K = 0.0, 1.0

# Kept short on purpose: a connectivity probe must fail fast, not inherit the
# generous per-call timeout tuned for report-sized structured output.
CONNECTION_TEST_TIMEOUT_SECONDS = 30.0
CONNECTION_TEST_MAX_TOKENS = 16
_MAX_ERROR_MESSAGE_CHARS = 300
# Long enough that switching between Settings tabs does not re-hit the provider,
# short enough that a key or base-URL change is reflected without a restart.
MODEL_CATALOG_TTL_SECONDS = 5 * 60


class SettingsValidationError(Exception):
    pass


class SettingsVersionConflictError(SettingsValidationError):
    def __init__(self, expected_version: int, current_version: int) -> None:
        super().__init__(
            f"Settings changed since they were loaded: expected version "
            f"{expected_version}, current version is {current_version}."
        )
        self.expected_version = expected_version
        self.current_version = current_version


@dataclass
class _Session:
    settings: LLMSettings
    payload_policy: str
    expires_at: float
    provider_settings: dict[LLMProvider, LLMSettings] = field(default_factory=dict)
    touched: bool = False
    """True once anything was overridden, so the view can report its source."""
    dev_mode: bool = False
    """Developer-view toggle. Presentation-only: no service branches on it."""
    analysis_depth: int = DEFAULT_ANALYSIS_DEPTH
    version: int = 0


@dataclass
class EffectiveSettings:
    """What a run should execute with. Internal — never serialized."""

    llm: LLMSettings
    payload_policy: str
    overridden: bool = False
    env_overlay: dict[str, str] = field(default_factory=dict)
    analysis_depth: int = DEFAULT_ANALYSIS_DEPTH


@dataclass
class _ModelCatalogCacheEntry:
    catalog: ModelCatalog
    expires_at: float


class SettingsService:
    def __init__(
        self,
        *,
        workspace: Path | None = None,
        defaults: LLMSettings | None = None,
        provider_api_keys: Mapping[LLMProvider, str] | None = None,
        ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS,
    ) -> None:
        self._workspace = (
            require_absolute_workspace(workspace) if workspace is not None else None
        )
        self._defaults = defaults if defaults is not None else load_llm_settings_from_env_file()
        self._provider_api_keys = dict(
            provider_api_keys
            if provider_api_keys is not None
            else (load_provider_api_keys_from_env_file() if defaults is None else {})
        )
        self._ttl_seconds = ttl_seconds
        self._sessions: dict[str, _Session] = {}
        self._model_catalog_cache: dict[str, _ModelCatalogCacheEntry] = {}
        self._lock = threading.RLock()

    # Read paths

    def get_settings(self, session_id: str = DEFAULT_SESSION_ID) -> SettingsView:
        return self._view(self._session(session_id))

    def list_providers(self) -> list[ProviderInfo]:
        return [
            ProviderInfo(
                provider=provider.value,
                display_name=spec.display_name,
                requires_api_key=spec.requires_api_key,
                requires_base_url=spec.requires_base_url,
                default_base_url=spec.default_base_url,
                preset_models=list(selectable_model_ids(provider)),
                structured_mode=spec.structured_mode,
                native=spec.native,
                text_transport="messages",
                structured_transport=(
                    "provider_tool"
                    if provider is LLMProvider.ANTHROPIC
                    else f"response_format:{spec.structured_mode}"
                ),
                tool_execution="host_orchestrated",
                pricing_catalog_version=(
                    PRICING_CATALOG_VERSION if spec.pricing_per_1m else ""
                ),
                pricing_source_url=spec.pricing_source_url,
                pricing_checked_at=PRICING_CHECKED_AT if spec.pricing_per_1m else "",
                capability_catalog_version=CAPABILITY_CATALOG_VERSION,
                agent_model_count=len(agent_model_ids(provider)),
            )
            for provider, spec in PROVIDER_REGISTRY.items()
        ]

    def list_models(
        self,
        session_id: str = DEFAULT_SESSION_ID,
        *,
        refresh: bool = False,
    ) -> ModelCatalog:
        """The provider's own model list, with the fallback stated not implied.

        This is a capability read, not inference, so it is deliberately absent
        from LLM spend. Settings used to present the built-in preset list as if
        it were current, which is how a retired model stayed selectable.
        """
        settings = self._session(session_id).settings.model_copy(deep=True)
        cache_key = _model_catalog_cache_key(settings)
        now = time.monotonic()
        with self._lock:
            cached = self._model_catalog_cache.get(cache_key)
            if not refresh and cached is not None and cached.expires_at > now:
                return cached.catalog

        fetched_at = datetime.now(UTC)
        try:
            discovered = fetch_model_catalog(settings)
            catalog = ModelCatalog(
                provider=settings.provider.value,
                models=[
                    _model_catalog_item(settings.provider, item)
                    for item in discovered.models
                ],
                source="live",
                fetched_at=fetched_at,
                endpoint=_display_endpoint(discovered.endpoint),
                truncated=discovered.truncated,
                pricing_catalog_version=PRICING_CATALOG_VERSION,
            )
        except ModelCatalogError as exc:
            catalog = ModelCatalog(
                provider=settings.provider.value,
                models=[
                    _fallback_model_item(settings.provider, model)
                    for model in selectable_model_ids(settings.provider)
                ],
                source="snapshot",
                fetched_at=fetched_at,
                warning=_redact(str(exc), settings.api_key),
                pricing_catalog_version=PRICING_CATALOG_VERSION,
            )
        with self._lock:
            self._model_catalog_cache[cache_key] = _ModelCatalogCacheEntry(
                catalog=catalog,
                expires_at=now + MODEL_CATALOG_TTL_SECONDS,
            )
        return catalog

    def resolve(self, session_id: str = DEFAULT_SESSION_ID) -> EffectiveSettings:
        """Settings a new run should execute with, plus the env overlay a worker
        subprocess needs. The key travels through the child's environment rather
        than its argv, which any user on the box can read via `ps`."""
        session = self._session(session_id)
        return EffectiveSettings(
            llm=session.settings,
            payload_policy=session.payload_policy,
            overridden=session.touched,
            env_overlay=_env_overlay(session.settings),
            analysis_depth=session.analysis_depth,
        )

    # Write paths

    def update_settings(
        self,
        patch: SettingsPatch,
        session_id: str = DEFAULT_SESSION_ID,
        *,
        expected_version: int | None = None,
    ) -> SettingsView:
        with self._lock:
            session = self._session(session_id)
            self._check_version(session, expected_version)
            current = session.settings
            target_provider = (
                current.provider
                if patch.provider is None
                else _parse_provider(patch.provider)
            )
            if target_provider is not current.provider:
                saved = session.provider_settings.get(target_provider)
                current = (
                    saved
                    if saved is not None
                    else self._new_provider_settings(target_provider, current)
                )
            settings, payload_policy = _apply_patch(
                current, session.payload_policy, patch
            )
            _forget_endpoint_learning(session.settings, settings)
            session.settings = settings
            session.provider_settings[settings.provider] = settings
            session.payload_policy = payload_policy
            if patch.dev_mode is not None:
                session.dev_mode = patch.dev_mode
            if patch.analysis_depth is not None:
                session.analysis_depth = _parse_analysis_depth(patch.analysis_depth)
            session.touched = True
            session.version += 1
            return self._view(session)

    def reset(
        self, session_id: str = DEFAULT_SESSION_ID, *, expected_version: int | None = None
    ) -> SettingsView:
        """Reset in place so the optimistic version remains monotonic."""
        with self._lock:
            session = self._session(session_id)
            self._check_version(session, expected_version)
            session.settings = self._defaults.model_copy(deep=True)
            session.provider_settings = {
                session.settings.provider: session.settings.model_copy(deep=True)
            }
            session.payload_policy = DEFAULT_PAYLOAD_POLICY
            session.touched = False
            session.dev_mode = False
            session.analysis_depth = DEFAULT_ANALYSIS_DEPTH
            session.version += 1
            return self._view(session)

    def test_connection(self, session_id: str = DEFAULT_SESSION_ID) -> ConnectionTestResult:
        """Send the smallest possible live request with the session's config.

        Client construction matches the offline check used by Settings; the
        extra round trip is what turns "configuration looks complete" into
        "the endpoint actually answers". Any failure text is scrubbed of the
        key before it leaves this method.
        """
        session = self._session(session_id)
        settings = session.settings
        if settings.provider is LLMProvider.OFFLINE:
            return ConnectionTestResult(
                ok=True,
                provider=settings.provider.value,
                model=settings.model,
                message="Offline mode: no request sent, deterministic fallback is used.",
            )
        probe = settings.model_copy(
            update={
                "timeout_seconds": min(settings.timeout_seconds, CONNECTION_TEST_TIMEOUT_SECONDS),
                "max_tokens": CONNECTION_TEST_MAX_TOKENS,
            }
        )
        started = time.monotonic()
        try:
            client = create_llm_client(probe)
            client.text(task="connection_check", payload={"ping": "ok"})
        except Exception as exc:  # noqa: BLE001 — any transport/config failure is a failed test
            return ConnectionTestResult(
                ok=False,
                provider=settings.provider.value,
                model=settings.model,
                elapsed_ms=_elapsed_ms(started),
                message=_redact(str(exc), settings.api_key),
                error_code=type(exc).__name__,
            )
        usage = client.last_usage()
        return ConnectionTestResult(
            ok=True,
            provider=settings.provider.value,
            # What the provider actually served, which an alias can make differ.
            model=usage.model if usage is not None else settings.model,
            elapsed_ms=_elapsed_ms(started),
            message=(
                "Provider responded. This probe is a real billed call and "
                "belongs to no session, so it is reported here only."
            ),
            prompt_tokens=usage.usage.prompt_tokens if usage is not None else None,
            completion_tokens=usage.usage.completion_tokens if usage is not None else None,
            estimated_cost_usd=usage.estimated_cost_usd if usage is not None else None,
            cost_basis=usage.cost_basis if usage is not None else "",
            request_id=usage.request_id if usage is not None else "",
            usage_reported=usage.usage_reported if usage is not None else None,
        )

    # Internals

    def _session(self, session_id: str) -> _Session:
        now = time.monotonic()
        self._evict_expired(now)
        session = self._sessions.get(session_id)
        if session is None:
            session = _Session(
                settings=self._defaults.model_copy(deep=True),
                payload_policy=DEFAULT_PAYLOAD_POLICY,
                expires_at=now + self._ttl_seconds,
            )
            session.provider_settings[session.settings.provider] = (
                session.settings.model_copy(deep=True)
            )
            if len(self._sessions) >= MAX_SESSIONS:
                oldest = min(self._sessions, key=lambda key: self._sessions[key].expires_at)
                del self._sessions[oldest]
            self._sessions[session_id] = session
        session.expires_at = now + self._ttl_seconds
        return session

    def _new_provider_settings(
        self, provider: LLMProvider, current: LLMSettings
    ) -> LLMSettings:
        models = selectable_model_ids(provider)
        return current.model_copy(
            deep=True,
            update={
                "provider": provider,
                "model": (
                    "offline-deterministic"
                    if provider is LLMProvider.OFFLINE
                    else (models[0] if models else "")
                ),
                "base_url": "",
                "api_key": self._provider_api_keys.get(provider, ""),
                "organization": "",
                "headers": {},
            },
        )

    def _evict_expired(self, now: float) -> None:
        for key in [key for key, item in self._sessions.items() if item.expires_at <= now]:
            del self._sessions[key]

    def _view(self, session: _Session) -> SettingsView:
        settings = session.settings
        status = _status(settings)
        verified = is_verified_agent_model(settings.provider, settings.model)
        return SettingsView(
            version=session.version,
            provider=settings.provider.value,
            model=settings.model,
            base_url=settings.base_url,
            resolved_base_url=settings.resolved_base_url,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            timeout_seconds=settings.timeout_seconds,
            structured_output_mode=settings.structured_output_mode,
            payload_policy=session.payload_policy,
            usd_per_1k_prompt=settings.usd_per_1k_prompt,
            usd_per_1k_completion=settings.usd_per_1k_completion,
            analysis_depth=session.analysis_depth,
            dev_mode=session.dev_mode,
            api_key_set=bool(settings.api_key),
            api_key_last4=_last4(settings.api_key),
            is_ready_for_live_calls=status.is_ready_for_live_calls,
            status_state=status.state,
            status_message=status.message,
            missing_fields=list(status.missing_fields),
            model_verified=verified,
            warnings=(
                []
                if verified or not settings.model
                else [
                    f"{settings.model} is not in the verified catalog. "
                    "The run will probe it for tool calling before spending."
                ]
            ),
            source="session" if session.touched else "env",
            about=self._about(),
        )

    @staticmethod
    def _check_version(session: _Session, expected_version: int | None) -> None:
        if expected_version is not None and expected_version != session.version:
            raise SettingsVersionConflictError(expected_version, session.version)

    def _about(self) -> AboutInfo:
        workspace = self._workspace
        if workspace is None:
            return AboutInfo(
                app_version=__version__,
                workspace_is_default=True,
                workspace_label="default workspace",
            )
        is_default = _same_path(workspace, default_workspace())
        return AboutInfo(
            app_version=__version__,
            workspace_is_default=is_default,
            workspace_label="default workspace" if is_default else f"…/{Path(workspace).name}",
        )


def _apply_patch(
    current: LLMSettings, current_policy: str, patch: SettingsPatch
) -> tuple[LLMSettings, str]:
    provider = current.provider if patch.provider is None else _parse_provider(patch.provider)
    model = current.model if patch.model is None else patch.model.strip()
    base_url = current.base_url if patch.base_url is None else patch.base_url.strip()
    api_key = current.api_key
    if patch.clear_api_key:
        api_key = ""
    elif patch.api_key is not None and patch.api_key.strip():
        api_key = patch.api_key.strip()
    structured = (
        current.structured_output_mode
        if patch.structured_output_mode is None
        else patch.structured_output_mode
    )
    if structured not in STRUCTURED_MODES:
        raise SettingsValidationError(
            f"structured_output_mode must be one of {', '.join(STRUCTURED_MODES)}."
        )
    policy = current_policy if patch.payload_policy is None else patch.payload_policy
    if policy not in PAYLOAD_POLICIES:
        raise SettingsValidationError(
            f"payload_policy must be one of {', '.join(PAYLOAD_POLICIES)}."
        )

    # Switching provider re-seeds model/base_url from the registry, otherwise a
    # DeepSeek model id would silently be posted at an OpenAI endpoint.
    if provider is not current.provider:
        if patch.model is None:
            models = selectable_model_ids(provider)
            model = models[0] if models else ""
        if patch.base_url is None:
            base_url = ""
        # Credentials are provider-scoped: never forward the previous
        # provider's secret to a newly selected endpoint unless this same patch
        # supplies a replacement.
        if patch.api_key is None:
            api_key = ""

    temperature = _bounded_float(
        "temperature", current.temperature, patch.temperature, MIN_TEMPERATURE, MAX_TEMPERATURE
    )
    max_tokens = int(
        _bounded_float(
            "max_tokens", current.max_tokens, patch.max_tokens, MIN_MAX_TOKENS, MAX_MAX_TOKENS
        )
    )
    timeout_seconds = _bounded_float(
        "timeout_seconds",
        current.timeout_seconds,
        patch.timeout_seconds,
        MIN_TIMEOUT_SECONDS,
        MAX_TIMEOUT_SECONDS,
    )
    usd_per_1k_prompt = _bounded_float(
        "usd_per_1k_prompt",
        current.usd_per_1k_prompt,
        patch.usd_per_1k_prompt,
        MIN_USD_PER_1K,
        MAX_USD_PER_1K,
    )
    usd_per_1k_completion = _bounded_float(
        "usd_per_1k_completion",
        current.usd_per_1k_completion,
        patch.usd_per_1k_completion,
        MIN_USD_PER_1K,
        MAX_USD_PER_1K,
    )

    # A half-filled config is a legal intermediate state, not an error: picking
    # a preset-less provider (Azure, LM Studio, custom) has to be possible
    # before its model and base URL are known. `_status` reports the gap as
    # "incomplete" and `create_llm_client` refuses at use time. A malformed
    # value, unlike a missing one, is still rejected here.
    if base_url and not base_url.startswith(("http://", "https://")):
        raise SettingsValidationError("base_url must start with http:// or https://.")
    # An unverified model is deliberately NOT rejected here. It cannot be: a
    # self-hosted model's id is whatever the operator named it, so no catalog
    # can enumerate the legal values. `_status` reports it as unverified and
    # the tool-calling probe settles it before the run spends anything.

    updated = current.model_copy(
        update={
            "provider": provider,
            "model": "offline-deterministic" if provider is LLMProvider.OFFLINE else model,
            "base_url": "" if provider is LLMProvider.OFFLINE else base_url,
            "api_key": api_key,
            "structured_output_mode": structured,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout_seconds": timeout_seconds,
            "usd_per_1k_prompt": usd_per_1k_prompt,
            "usd_per_1k_completion": usd_per_1k_completion,
        }
    )
    return updated, policy


def _model_catalog_item(
    provider: LLMProvider, discovered: DiscoveredModel
) -> ModelCatalogItem:
    """Provider-reported prices win over the shipped table; the table fills the
    gaps for the providers whose model list carries no pricing."""
    spec = provider_spec(provider)
    capability = agent_model_profile(provider, discovered.id)
    rates = spec.pricing_per_1m.get(discovered.id)
    return ModelCatalogItem(
        id=discovered.id,
        owned_by=discovered.owned_by,
        created=discovered.created,
        input_usd_per_1m=(
            discovered.input_usd_per_1m
            if discovered.input_usd_per_1m is not None
            else (rates[0] if rates is not None else None)
        ),
        output_usd_per_1m=(
            discovered.output_usd_per_1m
            if discovered.output_usd_per_1m is not None
            else (rates[1] if rates is not None else None)
        ),
        cache_read_usd_per_1m=spec.cache_read_per_1m.get(discovered.id),
        cache_write_usd_per_1m=spec.cache_write_per_1m.get(discovered.id),
        pricing_source=(
            discovered.pricing_source
            or ("public_list_price_snapshot" if rates is not None else "")
        ),
        capabilities=["tool_calling"] if capability is not None and capability.tool_calling else [],
        verified=is_verified_agent_model(provider, discovered.id),
        parallel_tool_calling=bool(
            capability is not None and capability.parallel_tool_calling
        ),
        structured_output=capability.structured_output if capability is not None else "",
        temperature_policy=(
            capability.temperature_policy if capability is not None else ""
        ),
    )


def _fallback_model_item(provider: LLMProvider, model: str) -> ModelCatalogItem:
    spec = provider_spec(provider)
    capability = agent_model_profile(provider, model)
    rates = spec.pricing_per_1m.get(model)
    return ModelCatalogItem(
        id=model,
        input_usd_per_1m=rates[0] if rates is not None else None,
        output_usd_per_1m=rates[1] if rates is not None else None,
        cache_read_usd_per_1m=spec.cache_read_per_1m.get(model),
        cache_write_usd_per_1m=spec.cache_write_per_1m.get(model),
        pricing_source="public_list_price_snapshot" if rates is not None else "",
        capabilities=["tool_calling"] if capability is not None and capability.tool_calling else [],
        verified=is_verified_agent_model(provider, model),
        parallel_tool_calling=bool(
            capability is not None and capability.parallel_tool_calling
        ),
        structured_output=capability.structured_output if capability is not None else "",
        temperature_policy=(
            capability.temperature_policy if capability is not None else ""
        ),
    )


def _model_catalog_cache_key(settings: LLMSettings) -> str:
    """Keyed on the credential too: two sessions pointed at the same base URL
    with different keys can legitimately see different model lists."""
    return "|".join(
        (
            settings.provider.value,
            settings.resolved_base_url,
            settings.organization,
            sha256(settings.api_key.encode("utf-8")).hexdigest(),
        )
    )


def _forget_endpoint_learning(before: LLMSettings, after: LLMSettings) -> None:
    """Both caches are keyed by (provider, model), which is not enough to
    identify a server: the same model id behind a new base URL is a different
    endpoint, and its dialect and tool support have to be rediscovered."""
    identity = (before.provider, before.resolved_base_url, before.model)
    if identity == (after.provider, after.resolved_base_url, after.model):
        return
    forget_learned_repairs()
    forget_probe_results()


def _display_endpoint(endpoint: str) -> str:
    """Host and path only — the query string can carry a key on some gateways."""
    parsed = urlsplit(endpoint)
    return f"{parsed.netloc}{parsed.path}"


def _status(settings: LLMSettings) -> LLMConfigurationStatus:
    """Core readiness check plus the two gaps it does not cover — an empty model
    and a base URL the registry marks required (Azure, custom endpoints)."""
    status = llm_configuration_status(settings)
    if settings.provider is LLMProvider.OFFLINE:
        return status
    missing = list(status.missing_fields)
    if not settings.model:
        missing.append("model")
    if provider_spec(settings.provider).requires_base_url and not settings.base_url:
        if "base_url" not in missing:
            missing.append("base_url")
    if missing == status.missing_fields:
        return status
    return status.model_copy(
        update={
            "state": "incomplete",
            "message": f"Missing required LLM setting: {', '.join(missing)}.",
            "is_ready_for_live_calls": False,
            "missing_fields": missing,
        }
    )


def _parse_analysis_depth(value: int) -> int:
    """Reject instead of silently changing the centrally mapped product tier."""
    try:
        exploration_tier_for_analysis_depth(value)
    except ValueError as exc:
        raise SettingsValidationError(str(exc)) from exc
    return int(value)


def _parse_provider(value: str) -> LLMProvider:
    try:
        return LLMProvider(value.strip())
    except ValueError:
        raise SettingsValidationError(f"Unknown provider: {value!r}") from None


def _bounded_float(
    name: str, current: float, value: float | None, low: float, high: float
) -> float:
    if value is None:
        return float(current)
    if not low <= value <= high:
        raise SettingsValidationError(f"{name} must be between {low:g} and {high:g}.")
    return float(value)


def _env_overlay(settings: LLMSettings) -> dict[str, str]:
    """Session settings as the env vars `core.env` already understands, so a
    worker subprocess resolves them without a new config channel."""
    overlay = {
        "EDA_LLM_PROVIDER": settings.provider.value,
        "EDA_LLM_MODEL": settings.model,
        "EDA_LLM_BASE_URL": settings.base_url,
        "EDA_LLM_TEMPERATURE": str(settings.temperature),
        "EDA_LLM_MAX_TOKENS": str(settings.max_tokens),
        "EDA_LLM_TIMEOUT_SECONDS": str(settings.timeout_seconds),
        "EDA_LLM_STRUCTURED_OUTPUT_MODE": settings.structured_output_mode,
        "EDA_LLM_USD_PER_1K_PROMPT": str(settings.usd_per_1k_prompt),
        "EDA_LLM_USD_PER_1K_COMPLETION": str(settings.usd_per_1k_completion),
    }
    if settings.api_key:
        overlay["EDA_LLM_API_KEY"] = settings.api_key
    return overlay


def _last4(api_key: str) -> str:
    # Short keys reveal too much of themselves in 4 characters.
    return api_key[-4:] if len(api_key) >= 8 else ""


def _redact(message: str, api_key: str) -> str:
    """Providers sometimes echo the request back in an error body; strip the key
    before the text reaches a response or a log."""
    if api_key:
        message = message.replace(api_key, "***")
    return message[:_MAX_ERROR_MESSAGE_CHARS]


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _same_path(left: Path, right: Path) -> bool:
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:
        return False
