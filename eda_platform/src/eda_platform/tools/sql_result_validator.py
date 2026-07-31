from __future__ import annotations

from pydantic import ValidationError

from eda_platform.schemas.artifacts import Artifact, ArtifactType, SqlResult
from eda_platform.schemas.chat import SqlResultValidation
from eda_platform.schemas.plans import AnalysisPlan


def validate_sql_result(artifact: Artifact, plan: AnalysisPlan) -> SqlResultValidation:
    if artifact.type is not ArtifactType.SQL_RESULT:
        return SqlResultValidation(
            status="fail",
            findings=[f"Expected SqlResult artifact, got {artifact.type.value}."],
        )

    try:
        result = SqlResult.model_validate(artifact.payload)
    except ValidationError as exc:
        return SqlResultValidation(
            status="fail",
            findings=[f"Invalid SqlResult payload: {exc.errors()[0]['msg']}"],
        )

    failures: list[str] = []
    warnings: list[str] = []
    if not result.columns:
        failures.append("Query returned no columns.")
    if result.row_count < 0:
        failures.append("Query returned an invalid negative row count.")
    if result.row_count == 0:
        warnings.append("Query returned zero rows.")
    if result.truncated:
        warnings.append("Preview is truncated; inspect or export the full result before deciding.")
    if plan.needs_approval:
        warnings.append("Plan was marked as requiring approval.")

    if failures:
        return SqlResultValidation(status="fail", findings=failures + warnings)
    if warnings:
        return SqlResultValidation(status="warn", findings=warnings)
    return SqlResultValidation(status="pass", findings=[])
