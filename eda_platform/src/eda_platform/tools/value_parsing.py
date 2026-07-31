from __future__ import annotations

import math
import re
from collections.abc import Iterable

_AGE_DURATION_PATTERN = re.compile(r"^(?P<years>\d{1,3})-(?P<days>\d{3})$")
_CURRENCY_PATTERN = re.compile(r"^[+$£€¥]")
# Add-only vs cb4fb1f: trailing (e/E exponent) branch lets scientific notation
# ("1.5e3", "1e10") parse instead of coercing to None. Plain decimals unchanged.
_NUMERIC_PATTERN = re.compile(r"^[+-]?(\d+(\.\d+)?|\.\d+)([eE][+-]?\d+)?$")
_NULL_STRINGS = {"", "-", "--", "na", "n/a", "nan", "none", "null"}
# Reject absurd age-duration years (e.g. "999-366") rather than emitting age=1000.
_MAX_AGE_DURATION_YEARS = 150


def parse_numeric_like(value: object, *, column_name: str = "") -> float | None:
    if _is_missing(value):
        return None
    text = str(value).strip()
    if text.lower() in _NULL_STRINGS:
        return None

    if _looks_like_age_column(column_name):
        age_value = _parse_age_duration(text)
        if age_value is not None:
            return age_value

    normalized = _normalize_numeric_text(text)
    if not _NUMERIC_PATTERN.fullmatch(normalized):
        return None
    try:
        return float(normalized)
    except ValueError:
        return None


def numeric_parse_success_percent(
    values: Iterable[object],
    *,
    column_name: str = "",
) -> float | None:
    total = 0
    parsed = 0
    for value in values:
        if _is_missing(value):
            continue
        text = str(value).strip()
        if text.lower() in _NULL_STRINGS:
            continue
        total += 1
        if parse_numeric_like(value, column_name=column_name) is not None:
            parsed += 1
    if total == 0:
        return None
    return round(parsed / total * 100, 2)


def _parse_age_duration(text: str) -> float | None:
    match = _AGE_DURATION_PATTERN.fullmatch(text.strip())
    if match is None:
        return None
    years = int(match.group("years"))
    days = int(match.group("days"))
    if days > 366 or years > _MAX_AGE_DURATION_YEARS:
        return None
    return round(years + days / 365, 2)


def _normalize_numeric_text(text: str) -> str:
    normalized = text.strip().replace(",", "").replace("_", "")
    normalized = normalized.replace("\u00a0", "")
    normalized = _CURRENCY_PATTERN.sub("", normalized)
    if normalized.endswith("%"):
        normalized = normalized[:-1]
    if normalized.startswith("(") and normalized.endswith(")"):
        normalized = f"-{normalized[1:-1]}"
    return normalized.strip()


def _looks_like_age_column(column_name: str) -> bool:
    normalized = column_name.lower().strip().replace(" ", "_")
    return normalized == "age" or normalized.endswith("_age") or normalized.endswith("age")


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False
