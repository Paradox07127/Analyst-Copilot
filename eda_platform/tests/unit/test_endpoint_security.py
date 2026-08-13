from __future__ import annotations

import pytest

from eda_platform.core.endpoint_security import (
    ENDPOINT_ALLOWLIST_ENV,
    EndpointPolicyError,
    NoCredentialRedirect,
    validate_llm_base_url,
)
from eda_platform.core.provider_registry import LLMProvider


def test_managed_provider_cannot_be_repointed_with_its_credential() -> None:
    with pytest.raises(EndpointPolicyError, match="pinned"):
        validate_llm_base_url(
            LLMProvider.DEEPSEEK, "http://169.254.169.254/latest/meta-data"
        )


def test_custom_provider_is_loopback_only_without_operator_grant() -> None:
    with pytest.raises(EndpointPolicyError, match="loopback-only"):
        validate_llm_base_url(
            LLMProvider.OPENAI_COMPATIBLE,
            "https://models.internal.example/v1",
            environ={},
        )


def test_custom_remote_origin_requires_exact_operator_allowlist() -> None:
    env = {ENDPOINT_ALLOWLIST_ENV: "https://models.internal.example"}

    accepted = validate_llm_base_url(
        LLMProvider.OPENAI_COMPATIBLE,
        "https://models.internal.example/v1",
        environ=env,
    )

    assert accepted == "https://models.internal.example/v1"


@pytest.mark.parametrize(
    "url",
    [
        "https://user:secret@models.example/v1",
        "https://models.example/v1?redirect=localhost",
        "https://models.example/v1#credential",
    ],
)
def test_endpoint_rejects_ambiguous_authority_or_suffix(url: str) -> None:
    with pytest.raises(EndpointPolicyError):
        validate_llm_base_url(
            LLMProvider.OPENAI_COMPATIBLE,
            url,
            environ={ENDPOINT_ALLOWLIST_ENV: "https://models.example"},
        )


def test_authenticated_http_redirects_are_disabled() -> None:
    assert NoCredentialRedirect().redirect_request(None, None, None, None) is None
