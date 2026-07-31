from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from enum import Enum

CANONICAL_KEY_VERSION = "v1"


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def normalize_values(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({normalize_text(value) for value in values if value.strip()}))


def canonical_evidence_signature(
    evidence: Iterable[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            {
                (normalize_text(kind), normalize_text(locator))
                for kind, locator in evidence
                if kind.strip() or locator.strip()
            }
        )
    )


def canonical_json(value: object) -> str:
    return json.dumps(
        _canonicalize(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def stable_digest(value: object, *, length: int = 24) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:length]


def versioned_key(namespace: str, value: object) -> str:
    return f"{namespace}:{CANONICAL_KEY_VERSION}:{stable_digest(value)}"


def artifact_parent_logical_key(parent_match_keys: Iterable[str]) -> str:
    """Build parentage from matched logical keys, never raw artifact ids."""
    return versioned_key(
        "artifact-parents",
        tuple(sorted({key for key in parent_match_keys if key})),
    )


def artifact_logical_key(
    *,
    artifact_type: str,
    stable_identity: str,
    parent_match_keys: Iterable[str] = (),
    occurrence_index: int = 0,
) -> str:
    return versioned_key(
        "artifact",
        {
            "artifact_type": normalize_text(artifact_type),
            "stable_identity": normalize_text(stable_identity),
            "parent_key": artifact_parent_logical_key(parent_match_keys),
            "occurrence_index": occurrence_index,
        },
    )


def execution_logical_span_key(
    *,
    parent_match_key: str,
    span_kind: str,
    operation_name: str,
    originating_key: str = "",
    tool_name: str = "",
    occurrence_index: int = 0,
) -> str:
    """Model and model configuration are deliberately absent from span identity."""
    return versioned_key(
        "execution-span",
        {
            "parent_match_key": parent_match_key,
            "span_kind": normalize_text(span_kind),
            "operation_name": normalize_text(operation_name),
            "originating_key": originating_key,
            "tool_name": normalize_text(tool_name),
            "occurrence_index": occurrence_index,
        },
    )


def _canonicalize(value: object) -> object:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return repr(value)
        return value
    if isinstance(value, Enum):
        return _canonicalize(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return _canonicalize(asdict(value))
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _canonicalize(model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        items = [_canonicalize(item) for item in value]
        return sorted(items, key=canonical_json)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, bytes):
        return value.hex()
    return str(value)
