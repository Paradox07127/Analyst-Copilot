"""Build reproducible environment fingerprints and tool references."""

from __future__ import annotations

import sys
from collections.abc import Callable
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from eda_platform.core.ids import stable_hash

# Runtime dependencies whose versions can change numeric results. A digest over
# these plus the interpreter version pins the environment tightly enough that a
# reproduction on a different digest is a genuine "different environment" signal.
_KEY_DEPENDENCIES: tuple[str, ...] = (
    "pandas",
    "numpy",
    "scipy",
    "scikit-learn",
    "duckdb",
)

_MISSING = "<missing>"


@lru_cache(maxsize=1)
def env_components() -> tuple[tuple[str, str], ...]:
    """Ordered ``(name, version)`` pairs that feed :func:`env_digest`."""
    components: list[tuple[str, str]] = [("python", _python_version())]
    components.extend((name, _safe_version(name)) for name in _KEY_DEPENDENCIES)
    return tuple(components)


@lru_cache(maxsize=1)
def env_digest() -> str:
    """Return a short, stable fingerprint of the analysis runtime."""
    return f"env_{stable_hash(env_components())}"


def code_ref(func: Callable[..., Any]) -> str:
    """Return a deterministic ``module.qualname`` reference to ``func``."""
    module = getattr(func, "__module__", "") or ""
    qualname = getattr(func, "__qualname__", None) or getattr(func, "__name__", "") or ""
    prefix = "eda_platform."
    if module.startswith(prefix):
        module = module[len(prefix) :]
    return f"{module}.{qualname}" if module else qualname


def _python_version() -> str:
    info = sys.version_info
    return f"{info.major}.{info.minor}.{info.micro}"


def _safe_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return _MISSING
