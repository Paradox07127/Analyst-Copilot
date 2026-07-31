"""Shared SQL identifier helpers used by template SQL builders."""

from __future__ import annotations

import re


def safe_alias(value: str) -> str:
    """Turn a column/dataset name into a safe unquoted SQL alias fragment."""
    cleaned = re.sub(r"[^0-9A-Za-z_]", "_", value.strip())
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"col_{cleaned}"
    return cleaned


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'
