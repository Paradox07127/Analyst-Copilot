"""Compare use cases (§10.3 P2): two runs of one project side by side.

Read-only and index-first: artifact type counts come from the SQLite index, and
only the few artifact types a headline metric actually needs are read from disk
(§13.2). Metric extraction is deterministic — artifacts, never the LLM.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal

from eda_platform.application.dto import (
    ComparabilityView,
    CompareArtifactDelta,
    CompareDatasetDiff,
    CompareLineageView,
    CompareMetricRow,
    CompareRuntimeView,
    CompareSessionSide,
    CompareTextRow,
    CompareValue,
    CompareView,
    SessionDetail,
)
from eda_platform.application.services.session_family_service import (
    SessionFamilyService,
    SessionResultFamily,
)
from eda_platform.application.services.session_service import SessionService
from eda_platform.application.workbench import run_cost_summary, summarize_session
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType

# (key, label, higher_is_better). None means neutral — the client must not
# colour that delta, because more charts or rows is neither good nor bad.
_METRIC_ROWS: tuple[tuple[str, str, bool | None], ...] = (
    ("datasets", "Datasets", None),
    ("rows", "Total rows", None),
    ("columns", "Total columns", None),
    ("critical", "Critical issues", False),
    ("warn", "Warnings", False),
    ("info", "Info notes", None),
    ("charts", "Charts", None),
    ("stat_tests", "Stat tests", None),
    ("ml_models", "ML models", None),
    ("llm_tokens", "LLM tokens", None),
    ("llm_cost_usd", "Est. cost (USD)", False),
)

_TEXT_ROWS: tuple[tuple[str, str], ...] = (
    ("report_status", "Report status"),
    ("ml_target", "ML target"),
    ("ml_metric", "ML metric"),
)

# Which ML metric to surface as the headline, most-informative first.
_PREFERRED_METRICS: tuple[str, ...] = ("f1_weighted", "accuracy", "r2", "rmse", "mae")

# Payload-reading is limited to these; everything else is counted from the index.
_METRIC_ARTIFACT_TYPES: tuple[ArtifactType, ...] = (
    ArtifactType.DATASET_PROFILE,
    ArtifactType.QUALITY_ISSUE_SET,
    ArtifactType.MODEL_CARD,
    ArtifactType.SESSION_METRICS,
    ArtifactType.SESSION_SUMMARY,
)


class CompareServiceError(Exception):
    pass


class CompareSameRunError(CompareServiceError):
    def __init__(self, session_id: str) -> None:
        super().__init__(f"Run {session_id} cannot be compared against itself.")
        self.session_id = session_id


class CompareProjectMismatchError(CompareServiceError):
    def __init__(self, left_session_id: str, right_session_id: str) -> None:
        super().__init__(
            f"Sessions {left_session_id} and {right_session_id} belong to different projects; "
            "only runs of the same project can be compared."
        )
        self.left_session_id = left_session_id
        self.right_session_id = right_session_id


@dataclass(frozen=True)
class _FamilyFacts:
    """One side of a comparison, aggregated over its whole result family.

    Reading the root session alone under-reported every run whose questions,
    investigations or report were executed in a derived session — which is most
    of them. A root with no charts of its own reported zero while its question
    sessions held two.
    """

    project_id: str
    status: str
    """`completed` only when every member completed. A single unfinished member
    has to keep an absent result from being asserted as a real zero."""
    artifacts: list[Artifact] = field(default_factory=list)
    artifact_type_counts: dict[str, int] = field(default_factory=dict)
    dataset_names: list[str] = field(default_factory=list)
    report_status: str | None = None
    """The report can be generated in a derived session, so the root's own
    column is often empty while the family does have one."""


class CompareService:
    def __init__(
        self,
        store: ArtifactStore,
        runs: SessionService,
        families: SessionFamilyService | None = None,
    ) -> None:
        self._store = store
        self._runs = runs
        self._families = families or SessionFamilyService(store)

    def compare_runs(self, left_session_id: str, right_session_id: str) -> CompareView:
        if left_session_id == right_session_id:
            raise CompareSameRunError(left_session_id)
        left = self._runs.get_session_detail(left_session_id)
        right = self._runs.get_session_detail(right_session_id)
        if left.project_id != right.project_id:
            raise CompareProjectMismatchError(left_session_id, right_session_id)
        left_family = self._families.collect(left.project_id, left_session_id)
        right_family = self._families.collect(right.project_id, right_session_id)
        lineage = self._families.lineage(left.project_id, left_session_id, right_session_id)
        family_warnings = [
            *(f"left family: {warning}" for warning in left_family.warnings),
            *(f"right family: {warning}" for warning in right_family.warnings),
        ]
        if family_warnings:
            lineage = lineage.model_copy(
                update={"warnings": list(dict.fromkeys([*lineage.warnings, *family_warnings]))}
            )
        left_facts = self._family_facts(left, left_family)
        right_facts = self._family_facts(right, right_family)
        left_metrics = self._headline(left_facts)
        right_metrics = self._headline(right_facts)
        return CompareView(
            project_id=left.project_id,
            left=_to_side(left),
            right=_to_side(right),
            comparability=self._comparability(left, right, lineage),
            lineage=lineage,
            metrics=[
                _metric_row(key, label, better, left_metrics, right_metrics)
                for key, label, better in _METRIC_ROWS
            ],
            text_rows=[
                _text_row(key, label, left_metrics, right_metrics) for key, label in _TEXT_ROWS
            ],
            artifact_deltas=_artifact_deltas(left_facts, right_facts),
            datasets=_dataset_diff(left_facts, right_facts),
        )

    def _family_facts(
        self, detail: SessionDetail, family: SessionResultFamily
    ) -> _FamilyFacts:
        artifacts: list[Artifact] = []
        counts: Counter[str] = Counter()
        names: list[str] = []
        statuses: list[str] = []
        report_status: str | None = None
        for session_id in family.session_ids:
            artifacts.extend(
                self._store.list_indexed_artifacts(
                    project_id=detail.project_id,
                    session_id=session_id,
                    artifact_types=_METRIC_ARTIFACT_TYPES,
                )
            )
            counts.update(self._store.artifact_type_counts(detail.project_id, session_id))
            row = self._store.get_session_index_row(session_id)
            if row is None:
                # In the family walk but no longer in the index: the member's
                # results cannot be read, so the side is not fully covered.
                statuses.append("unknown")
                continue
            statuses.append(str(row.get("status") or "unknown"))
            names.extend(_index_dataset_names(row))
            if report_status is None and row.get("report_status"):
                report_status = str(row["report_status"])
        return _FamilyFacts(
            project_id=detail.project_id,
            status=_family_status(statuses),
            artifacts=artifacts,
            artifact_type_counts=dict(counts),
            # Members share the root's inputs; order is kept for display.
            dataset_names=list(dict.fromkeys(names)),
            report_status=report_status,
        )

    def _headline(self, facts: _FamilyFacts) -> dict[str, CompareValue[Any]]:
        """One side's comparable numbers, across its whole result family."""
        artifacts = facts.artifacts
        counts = facts.artifact_type_counts
        summary = summarize_session(artifacts)
        rows = _ProfileTotal()
        columns = _ProfileTotal()
        profile_count = 0
        for artifact in artifacts:
            if artifact.type is ArtifactType.DATASET_PROFILE:
                profile_count += 1
                rows.add(artifact.payload.get("rows"))
                columns.add(artifact.payload.get("columns"))
        quality_count = sum(
            artifact.type is ArtifactType.QUALITY_ISSUE_SET for artifact in artifacts
        )
        ml_target, ml_metric = _ml_headline(artifacts)
        cost = run_cost_summary(artifacts) or {}
        model_count = counts.get(ArtifactType.MODEL_CARD.value, 0)
        return {
            "datasets": _producer_count(profile_count, facts.status, "dataset profiling"),
            "rows": rows.value(profile_count, facts.status),
            "columns": columns.value(profile_count, facts.status),
            "critical": (
                _value(float(summary["critical"]))
                if quality_count
                else _absent(facts.status, "quality assessment was not produced")
            ),
            "warn": (
                _value(float(summary["warn"]))
                if quality_count
                else _absent(facts.status, "quality assessment was not produced")
            ),
            "info": (
                _value(float(summary["info"]))
                if quality_count
                else _absent(facts.status, "quality assessment was not produced")
            ),
            "charts": _producer_count(
                counts.get(ArtifactType.CHART_SPEC.value, 0),
                facts.status,
                "chart production",
            ),
            "stat_tests": _producer_count(
                counts.get(ArtifactType.STAT_TEST_RESULT.value, 0),
                facts.status,
                "statistical testing",
            ),
            "ml_models": _producer_count(model_count, facts.status, "model training"),
            "llm_tokens": _payload_number(
                cost, "total_tokens", facts.status, "execution metrics"
            ),
            "llm_cost_usd": _payload_number(
                cost, "estimated_cost_usd", facts.status, "execution metrics"
            ),
            "report_status": (
                _value(facts.report_status)
                if facts.report_status
                else _absent(facts.status, "a report status was not produced")
            ),
            "ml_target": _ml_value(
                ml_target, model_count=model_count, status=facts.status, label="ML target"
            ),
            "ml_metric": _ml_value(
                ml_metric, model_count=model_count, status=facts.status, label="ML metric"
            ),
        }

    def _comparability(
        self,
        left: SessionDetail,
        right: SessionDetail,
        lineage: CompareLineageView,
    ) -> ComparabilityView:
        left_row = self._store.get_session_index_row(left.session_id) or {}
        right_row = self._store.get_session_index_row(right.session_id) or {}
        left_runtime, left_warnings = _runtime_view(left_row)
        right_runtime, right_warnings = _runtime_view(right_row)
        dimensions = {
            "input_hashes": (left_runtime.input_hashes, right_runtime.input_hashes),
            "code_version": (left_runtime.code_version, right_runtime.code_version),
            "seed": (left_runtime.seed, right_runtime.seed),
            "model_versions": (left_runtime.model_versions, right_runtime.model_versions),
            "prompt_template_version": (
                left_runtime.prompt_template_version,
                right_runtime.prompt_template_version,
            ),
        }
        changed = [
            name
            for name, (left_value, right_value) in dimensions.items()
            if left_value.state == right_value.state == "value"
            and left_value.value != right_value.value
        ]
        unknown = [
            name
            for name, (left_value, right_value) in dimensions.items()
            if left_value.state != "value" or right_value.state != "value"
        ]
        if "input_hashes" in unknown:
            verdict = "unknown"
        elif "input_hashes" in changed:
            verdict = "not_directly_comparable"
        elif changed or unknown:
            verdict = "partially_controlled"
        else:
            verdict = "controlled"
        return ComparabilityView(
            verdict=verdict,
            left=left_runtime,
            right=right_runtime,
            changed_dimensions=changed,
            unknown_dimensions=unknown,
            warnings=list(
                dict.fromkeys([*left_warnings, *right_warnings, *lineage.warnings])
            ),
        )


def _to_side(detail: SessionDetail) -> CompareSessionSide:
    return CompareSessionSide(
        session_id=detail.session_id,
        project_id=detail.project_id,
        title=detail.title,
        status=detail.status,
        created_at=detail.created_at,
        dataset_names=list(detail.dataset_names),
        artifact_count=detail.artifact_count,
        report_status=detail.report_status,
    )


def _metric_row(
    key: str,
    label: str,
    higher_is_better: bool | None,
    left: dict[str, CompareValue[Any]],
    right: dict[str, CompareValue[Any]],
) -> CompareMetricRow:
    left_value = left[key]
    right_value = right[key]
    delta = None
    if (
        left_value.state == right_value.state == "value"
        and isinstance(left_value.value, int | float)
        and isinstance(right_value.value, int | float)
        and left_value.value != right_value.value
    ):
        delta = float(right_value.value) - float(left_value.value)
    direction: Literal["maximize", "minimize", "none"] = (
        "maximize"
        if higher_is_better is True
        else "minimize"
        if higher_is_better is False
        else "none"
    )
    return CompareMetricRow(
        key=key,
        label=label,
        left=left_value,
        right=right_value,
        delta=delta,
        optimization_direction=direction,
        verdict=_metric_verdict(left_value, right_value, delta, direction),
        higher_is_better=higher_is_better,
    )


def _text_row(
    key: str,
    label: str,
    left: dict[str, CompareValue[Any]],
    right: dict[str, CompareValue[Any]],
) -> CompareTextRow:
    left_value = left[key]
    right_value = right[key]
    comparable = left_value.state == right_value.state == "value"
    return CompareTextRow(
        key=key,
        label=label,
        left=left_value,
        right=right_value,
        changed=left_value.value != right_value.value if comparable else None,
    )


def _family_status(statuses: list[str]) -> str:
    """`completed` only when every member is. Otherwise the first member that
    is not, so `_producer_count`'s reason still names a real status rather than
    a synthetic one."""
    if statuses and all(status == "completed" for status in statuses):
        return "completed"
    return next((status for status in statuses if status != "completed"), "unknown")


def _index_dataset_names(row: dict[str, Any]) -> list[str]:
    try:
        parsed = json.loads(row.get("dataset_names_json") or "[]")
    except (TypeError, ValueError):
        return []
    return [str(name) for name in parsed] if isinstance(parsed, list) else []


def _artifact_deltas(
    left: _FamilyFacts, right: _FamilyFacts
) -> list[CompareArtifactDelta]:
    artifact_types = sorted(set(left.artifact_type_counts) | set(right.artifact_type_counts))
    return [
        _artifact_delta(artifact_type, left, right) for artifact_type in artifact_types
    ]


def _artifact_delta(
    artifact_type: str, left: _FamilyFacts, right: _FamilyFacts
) -> CompareArtifactDelta:
    left_value = _producer_count(
        left.artifact_type_counts.get(artifact_type, 0),
        left.status,
        f"{artifact_type} production",
    )
    right_value = _producer_count(
        right.artifact_type_counts.get(artifact_type, 0),
        right.status,
        f"{artifact_type} production",
    )
    delta = None
    if left_value.state == right_value.state == "value":
        delta = int(right_value.value or 0) - int(left_value.value or 0)
        if delta == 0:
            delta = None
    return CompareArtifactDelta(
        type=artifact_type,
        left=left_value,
        right=right_value,
        delta=delta,
    )


def _dataset_diff(left: _FamilyFacts, right: _FamilyFacts) -> CompareDatasetDiff:
    left_value = _dataset_names(left)
    right_value = _dataset_names(right)
    if left_value.state != "value" or right_value.state != "value":
        return CompareDatasetDiff(left=left_value, right=right_value)
    left_set = set(left_value.value or [])
    right_set = set(right_value.value or [])
    return CompareDatasetDiff(
        left=left_value,
        right=right_value,
        shared=sorted(left_set & right_set),
        only_left=sorted(left_set - right_set),
        only_right=sorted(right_set - left_set),
    )


def _ml_headline(artifacts: list[Artifact]) -> tuple[str | None, str | None]:
    for artifact in artifacts:
        if artifact.type is not ArtifactType.MODEL_CARD:
            continue
        target = str(artifact.payload.get("target_column") or "") or None
        return target, _primary_metric(artifact.payload.get("metrics"))
    return None, None


def _primary_metric(metrics: object) -> str | None:
    if not isinstance(metrics, dict) or not metrics:
        return None
    for key in _PREFERRED_METRICS:
        if key in metrics:
            return _format_metric(key, metrics[key])
    name, value = next(iter(metrics.items()))
    return _format_metric(str(name), value)


def _format_metric(name: str, value: object) -> str | None:
    number = _optional_number(value)
    return None if number is None else f"{name}={number:.3g}"


class _ProfileTotal:
    """Sums one numeric field across dataset profiles.

    A profile whose field will not parse is counted, not skipped: treating it
    as 0 produced a smaller-but-plausible total, which is exactly the
    "missing silently becomes a number" the compare contract forbids.
    """

    def __init__(self) -> None:
        self.total = 0.0
        self.unreadable = 0

    def add(self, raw: object) -> None:
        number = _optional_number(raw)
        if number is None:
            self.unreadable += 1
        else:
            self.total += number

    def value(self, profile_count: int, status: str) -> CompareValue[Any]:
        if not profile_count:
            return _absent(status, "dataset profiles were not produced")
        if self.unreadable:
            return _unavailable(
                f"{self.unreadable} of {profile_count} dataset profiles "
                "did not report this field"
            )
        return _value(self.total)


def _optional_number(value: object) -> float | None:
    """Payloads are untyped JSON; anything not numeric reads as absent."""
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _value[T](value: T) -> CompareValue[T]:
    return CompareValue[T](state="value", value=value)


def _missing(reason: str) -> CompareValue[Any]:
    return CompareValue[Any](state="missing", reason=reason)


def _unavailable(reason: str) -> CompareValue[Any]:
    return CompareValue[Any](state="unavailable", reason=reason)


def _not_applicable(reason: str) -> CompareValue[Any]:
    return CompareValue[Any](state="not_applicable", reason=reason)


def _absent(status: str, reason: str) -> CompareValue[Any]:
    if status == "completed":
        return _missing(reason)
    return _unavailable(f"{reason}; session status is {status}")


def _producer_count(count: int, status: str, producer: str) -> CompareValue[int]:
    """Observed output wins; absent output needs producer-coverage evidence."""
    if count > 0 or status == "completed":
        return _value(count)
    return _unavailable(
        f"{producer} has no observed output and producer-stage completion is unknown "
        f"(session status: {status})"
    )


def _payload_number(
    payload: dict[str, Any],
    key: str,
    status: str,
    producer: str,
) -> CompareValue[float]:
    number = _optional_number(payload.get(key))
    if number is not None:
        return _value(number)
    if payload:
        return _missing(f"{producer} did not record {key}")
    return _absent(status, f"{producer} was not produced")


def _ml_value(
    value: str | None,
    *,
    model_count: int,
    status: str,
    label: str,
) -> CompareValue[str]:
    if value is not None:
        return _value(value)
    if model_count > 0:
        return _missing(f"the model card did not record {label.lower()}")
    if status == "completed":
        return _not_applicable("this session produced no ML model")
    return _unavailable(
        f"{label} cannot be determined because model production did not complete"
    )


def _dataset_names(facts: _FamilyFacts) -> CompareValue[list[str]]:
    if facts.dataset_names or facts.status == "completed":
        return _value(list(facts.dataset_names))
    return _unavailable(
        "dataset identity is unavailable because input/profile production did not complete"
    )


def _metric_verdict(
    left: CompareValue[Any],
    right: CompareValue[Any],
    delta: float | None,
    direction: Literal["maximize", "minimize", "none"],
) -> Literal["improved", "regressed", "unchanged", "tradeoff", "unknown"]:
    """Vocabulary is fixed by the compare contract (plan section 17.4).
    `unknown` covers both "a side is not a value" and "no optimization
    direction, so a difference carries no quality meaning"."""
    if left.state != "value" or right.state != "value":
        return "unknown"
    if left.value == right.value:
        return "unchanged"
    if direction == "none":
        return "unknown"
    improved = delta is not None and (
        (direction == "maximize" and delta > 0)
        or (direction == "minimize" and delta < 0)
    )
    return "improved" if improved else "regressed"


def _runtime_view(row: dict[str, Any]) -> tuple[CompareRuntimeView, list[str]]:
    warnings: list[str] = []
    input_hashes = _json_mapping_value(row.get("input_hashes_json"), "input_hashes", warnings)
    model_versions = _json_mapping_value(
        row.get("model_versions_json"), "model_versions", warnings
    )
    code_version = _indexed_scalar_value(row.get("code_version"), "code_version")
    seed = _indexed_scalar_value(row.get("seed"), "seed")
    source_session_id = (
        _value(str(row["source_session_id"]))
        if row.get("source_session_id")
        else _not_applicable("the session is indexed as a root session")
    )
    return (
        CompareRuntimeView(
            input_hashes=input_hashes,
            code_version=code_version,
            seed=seed,
            model_versions=model_versions,
            source_session_id=source_session_id,
            prompt_template_version=_missing(
                "prompt/template version is not persisted in the current run manifest"
            ),
        ),
        warnings,
    )


def _json_mapping_value(
    raw: object,
    name: str,
    warnings: list[str],
) -> CompareValue[dict[str, str]]:
    if raw is None:
        return _missing(f"{name} is not persisted for this session")
    try:
        parsed = json.loads(str(raw))
    except (TypeError, ValueError):
        warnings.append(f"indexed {name} is unreadable")
        return _unavailable(f"indexed {name} is unreadable")
    if not isinstance(parsed, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in parsed.items()
    ):
        warnings.append(f"indexed {name} has an invalid shape")
        return _unavailable(f"indexed {name} has an invalid shape")
    return _value(parsed)


def _indexed_scalar_value(raw: object, name: str) -> CompareValue[Any]:
    if raw is None:
        return _missing(f"{name} is not persisted for this session")
    return _value(raw)  # type: ignore[arg-type]
