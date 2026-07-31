from __future__ import annotations

import re
from collections.abc import Callable
from typing import Literal

from eda_platform.core.ids import make_artifact_id
from eda_platform.schemas.artifacts import (
    Artifact,
    ArtifactType,
    DatasetProfile,
    PiiColumn,
    PiiReport,
)

PiiLabel = Literal["email", "phone", "name", "id", "unknown"]

_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PHONE_PATTERN = re.compile(r"^\+?[\d\s().-]+$")
_ID_NAME_HINTS = ("ssn", "passport", "id_card", "national_id")
_EMAIL_NAME_HINTS = ("email", "e_mail", "mail")
_PHONE_NAME_HINTS = ("phone", "mobile", "tel")
_PERSON_NAME_EXACT_HINTS = frozenset({"name"})
_PERSON_NAME_CONTEXT_TOKENS = frozenset({"customer", "user", "contact", "person", "client"})
_PHONE_SEPARATOR_PATTERN = re.compile(r"[\s().-]")
_DATE_LIKE_PATTERN = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$")


def tag_pii_columns(profile_artifact: Artifact, *, project_id: str, session_id: str) -> Artifact:
    profile = DatasetProfile.model_validate(profile_artifact.payload)
    report = PiiReport(
        dataset_id=profile.dataset_id,
        columns=[
            PiiColumn(column=column.name, label=label, reason=reason)
            for column in profile.columns_detail
            for label, reason in [
                _detect_pii(
                    column.name,
                    column.sample_values,
                    semantic_type=column.semantic_type,
                )
            ]
            if label is not None
        ],
    )
    payload = report.model_dump(mode="json")
    return Artifact(
        id=make_artifact_id("pii", payload),
        type=ArtifactType.PII_REPORT,
        project_id=project_id,
        session_id=session_id,
        parents=[profile_artifact.id],
        payload=payload,
    )


def mask_profile_artifact(
    profile_artifact: Artifact,
    pii_artifact: Artifact,
) -> Artifact:
    """Remove detected PII values from persisted profile samples."""
    profile = DatasetProfile.model_validate(profile_artifact.payload)
    labels = pii_labels(pii_artifact)
    if not labels:
        return profile_artifact
    masked_columns = [
        column.model_copy(
            update={
                "sample_values": (
                    [f"[PII:{labels[column.name]}]"]
                    if column.name in labels
                    else column.sample_values
                ),
                # Level lists carry every distinct value, so a PII column's
                # levels are the whole column in plain text. Counts survive;
                # the values do not.
                "category_levels": (
                    [
                        {"value": f"[PII:{labels[column.name]}]", "count": level["count"]}
                        for level in column.category_levels
                    ]
                    if column.name in labels
                    else column.category_levels
                ),
            }
        )
        for column in profile.columns_detail
    ]
    masked_rows = [
        {
            key: (f"[PII:{labels[key]}]" if key in labels and value is not None else value)
            for key, value in row.items()
        }
        for row in profile.sample_rows
    ]
    masked = profile.model_copy(
        update={
            "columns_detail": masked_columns,
            "sample_rows": masked_rows,
            "pii_columns": labels,
        }
    )
    payload = masked.model_dump(mode="json")
    return profile_artifact.model_copy(
        update={
            "id": make_artifact_id("prof", payload),
            "payload": payload,
            "warnings": [*profile_artifact.warnings, "pii_samples_masked"],
        }
    )
def mask_value(column: str, value: object, pii_artifact: Artifact) -> object:
    labels = pii_labels(pii_artifact)
    label = labels.get(column)
    if label is None:
        return value
    return f"[PII:{label}]"


def pii_labels(pii_artifact: Artifact) -> dict[str, str]:
    report = PiiReport.model_validate(pii_artifact.payload)
    return {pii.column: pii.label for pii in report.columns}


def _detect_pii(
    column_name: str,
    sample_values: list[str],
    *,
    semantic_type: str,
) -> tuple[PiiLabel | None, str]:
    normalized = column_name.strip().lower()
    if _matches_hint(normalized, _EMAIL_NAME_HINTS):
        return "email", "column-name"
    if _matches_hint(normalized, _PHONE_NAME_HINTS):
        return "phone", "column-name"
    if _matches_person_name_hint(normalized):
        return "name", "column-name"
    if _matches_hint(normalized, _ID_NAME_HINTS):
        return "id", "column-name"
    if _sample_match_rate(sample_values, _matches_email_value) >= 0.5:
        return "email", "sample-values"
    if semantic_type != "datetime" and _sample_match_rate(
        sample_values, _matches_phone_value
    ) >= 0.5:
        return "phone", "sample-values"
    return None, ""


def _matches_hint(value: str, hints: tuple[str, ...]) -> bool:
    return any(hint in value for hint in hints)


def _matches_person_name_hint(value: str) -> bool:
    if value in _PERSON_NAME_EXACT_HINTS:
        return True
    tokens = {token for token in re.split(r"[_\s]+", value) if token}
    return "name" in tokens and bool(tokens & _PERSON_NAME_CONTEXT_TOKENS)


def _matches_email_value(value: str) -> bool:
    return _EMAIL_PATTERN.match(value) is not None


def _matches_phone_value(value: str) -> bool:
    if _DATE_LIKE_PATTERN.fullmatch(value.strip()):
        return False
    if _PHONE_PATTERN.match(value) is None:
        return False
    digits = re.sub(r"\D", "", value)
    if not 7 <= len(digits) <= 15:
        return False
    return value.startswith("+") or _PHONE_SEPARATOR_PATTERN.search(value) is not None


def _sample_match_rate(values: list[str], matcher: Callable[[str], bool]) -> float:
    checked = [str(value).strip() for value in values[:10] if str(value).strip()]
    if not checked:
        return 0.0
    matches = sum(1 for value in checked if matcher(value))
    return matches / len(checked)
