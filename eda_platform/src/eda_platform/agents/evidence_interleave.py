"""DI9 H9-A: write-time evidence interleaving (EvidFuse, arXiv:2601.05487).

The root cause of flat reports is "compute everything, then write": the writer
only ever sees a compressed evidence summary. This layer lets the writer LLM
*request* evidence right before writing a section — a typed
:class:`~eda_platform.schemas.reports.EvidenceRequest` pointing at an
already-persisted artifact — and a deterministic resolver reads the stored
payload and returns the values. The claim being written is then constrained by
the evidence it just fetched.

Red lines (sprint 9 plan §3):

- The resolver ONLY reads persisted ``SqlResult`` / ``StatTestResult`` /
  ``DatasetProfile`` / ``ValidatedFinding`` artifacts. It never runs new SQL
  and never computes new numbers — every granted value already exists in a
  deterministic reducer's output.
- The loop is bounded: at most ``per_section_limit`` requests per section and
  ``total_limit`` for the whole report. Once exhausted, the writer must write
  with the evidence it already holds.
- Resolution failures return a typed, teaching-style rejection (with a summary
  of the usable artifact catalog) instead of raising, so a confused writer can
  self-correct inside the same bounded loop.
- All LLM output produced with interleaved evidence still passes the existing
  L1 number gate / causal gate / three-tier gate downstream — nothing here
  exempts anything.

Locator grammar deliberately reuses the report validator's EvidenceRef
resolution semantics (``tools.report_validator``): ``rows`` / ``rows[i]`` /
``rows[i].col`` / bare column name for SQL results; ``statistic`` / ``p_value``
/ ``effect_size`` / ``sample_size`` for stat tests; ``rows`` / ``columns`` /
``missing_percent.<col>`` for dataset profiles; ``findings[i]`` for validated
findings.

Retirement clause: once long-context models are reliably grounded and feeding
the full evidence pack up front is affordable, this bounded request loop can be
simplified to a one-shot full-evidence injection (the typed grant models stay
as the injection format).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from pydantic import ValidationError

from eda_platform.core.store import ArtifactStore
from eda_platform.core.trace import (
    EVIDENCE_INTERLEAVE_GRANTED,
    EVIDENCE_INTERLEAVE_REJECTED,
    EVIDENCE_INTERLEAVE_REQUEST,
)
from eda_platform.schemas.artifacts import (
    Artifact,
    ArtifactType,
    DatasetProfile,
    EvidenceRef,
    SqlResult,
)
from eda_platform.schemas.investigations import ValidatedFinding
from eda_platform.schemas.reports import (
    EvidenceGrant,
    EvidenceRejection,
    EvidenceRequest,
    GrantValue,
    InterleaveExchange,
    InterleaveRejectionCode,
    InterleaveTranscript,
)
from eda_platform.schemas.stats import StatTestResult
from eda_platform.tools.report_validator import _sql_result_numbers

# The artifact types the deterministic resolver can read. Everything else is
# rejected: charts and reports carry no reducer numbers, and code/SQL execution
# artifacts must be requested through their persisted result forms.
_RESOLVABLE_TYPES = {
    ArtifactType.SQL_RESULT,
    ArtifactType.STAT_TEST_RESULT,
    ArtifactType.DATASET_PROFILE,
    ArtifactType.VALIDATED_FINDING,
}
_STAT_LOCATORS = ("statistic", "p_value", "effect_size", "sample_size")
_CATALOG_LIMIT = 15
_FINDING_LOCATOR_PATTERN = re.compile(r"^findings\[(?P<index>\d+)\]$")
_DEFAULT_SECTION = "default"

TraceSink = Callable[[str, dict[str, Any]], None]


class EvidenceResolver(Protocol):
    """Read-only lookup over already-persisted artifacts."""

    def lookup(self, artifact_id: str) -> Artifact | None: ...

    def catalog(self) -> list[str]: ...


class InMemoryEvidenceResolver:
    """Resolver over a run's in-memory artifact list (agentic report path).

    The artifacts handed to ``generate_agentic_report`` are already persisted
    by the upstream drivers; this resolver only indexes them for lookup.
    """

    def __init__(self, artifacts: Sequence[Artifact]) -> None:
        self._by_id = {artifact.id: artifact for artifact in artifacts}

    def lookup(self, artifact_id: str) -> Artifact | None:
        return self._by_id.get(artifact_id)

    def catalog(self) -> list[str]:
        summaries = [
            f"{artifact.id} ({artifact.type.value})"
            for artifact in self._by_id.values()
            if artifact.type in _RESOLVABLE_TYPES
        ]
        return summaries[:_CATALOG_LIMIT]


class StoreEvidenceResolver:
    """Resolver over the artifact store (decision report path).

    ``catalog_artifact_ids`` names the artifacts the writer is invited to use
    (typically the selected findings and their source artifacts); lookups are
    still allowed for any persisted artifact of the same project, but never for
    another project's data.
    """

    def __init__(
        self,
        store: ArtifactStore,
        *,
        project_id: str,
        artifact_session_ids: dict[str, str] | None = None,
        catalog_artifact_ids: Sequence[str] = (),
    ) -> None:
        self._store = store
        self._project_id = project_id
        self._artifact_session_ids = artifact_session_ids or {}
        self._catalog_ids = list(dict.fromkeys(catalog_artifact_ids))

    def lookup(self, artifact_id: str) -> Artifact | None:
        try:
            artifact = self._store.get_artifact(
                artifact_id,
                project_id=self._project_id,
                session_id=self._artifact_session_ids.get(artifact_id),
            )
        except (KeyError, OSError, ValueError):
            return None
        if artifact.project_id != self._project_id:
            return None
        return artifact

    def catalog(self) -> list[str]:
        summaries: list[str] = []
        for artifact_id in self._catalog_ids[:_CATALOG_LIMIT]:
            artifact = self.lookup(artifact_id)
            if artifact is None:
                continue
            if artifact.type in _RESOLVABLE_TYPES:
                summaries.append(f"{artifact.id} ({artifact.type.value})")
        return summaries


class EvidenceInterleaveSession:
    """The bounded write-time request loop.

    Enforces the per-section and whole-report budgets, resolves each request
    deterministically, records every exchange into the transcript, and emits
    ``evidence_interleave_request`` / ``_granted`` / ``_rejected`` trace events
    through the optional sink. ``request`` never raises: infrastructure errors
    degrade to a typed rejection so the report path cannot be broken by the
    interleave layer.
    """

    def __init__(
        self,
        resolver: EvidenceResolver,
        *,
        per_section_limit: int = 2,
        total_limit: int = 8,
        trace_sink: TraceSink | None = None,
    ) -> None:
        self._resolver = resolver
        self._per_section_limit = per_section_limit
        self._total_limit = total_limit
        self._trace_sink = trace_sink
        self._section_counts: dict[str, int] = {}
        self._total_count = 0
        self._exchanges: list[InterleaveExchange] = []

    @property
    def remaining_total(self) -> int:
        return max(self._total_limit - self._total_count, 0)

    def remaining_for_section(self, section: str) -> int:
        used = self._section_counts.get(section or _DEFAULT_SECTION, 0)
        return max(self._per_section_limit - used, 0)

    def catalog(self) -> list[str]:
        try:
            return self._resolver.catalog()
        except Exception:  # noqa: BLE001 - catalog is advisory, never fatal
            return []

    def granted_values(self) -> list[tuple[float, str, str | None]]:
        """Every granted number with coarse and exact gate-side units."""
        return [
            (item.value, item.unit, item.unit_label)
            for exchange in self._exchanges
            if exchange.grant is not None
            for item in exchange.grant.values
        ]

    def granted_artifact_ids(self) -> list[str]:
        return list(
            dict.fromkeys(
                exchange.grant.artifact_id
                for exchange in self._exchanges
                if exchange.grant is not None
            )
        )

    @property
    def transcript(self) -> InterleaveTranscript:
        return InterleaveTranscript(
            exchanges=list(self._exchanges),
            per_section_limit=self._per_section_limit,
            total_limit=self._total_limit,
            granted_count=sum(1 for item in self._exchanges if item.grant is not None),
            rejected_count=sum(1 for item in self._exchanges if item.rejection is not None),
        )

    def request(
        self, request: EvidenceRequest, *, section: str | None = None
    ) -> EvidenceGrant | EvidenceRejection:
        bucket = (section or request.section or _DEFAULT_SECTION).strip() or _DEFAULT_SECTION
        self._emit(
            EVIDENCE_INTERLEAVE_REQUEST,
            {
                "artifact_id": request.artifact_id,
                "locator": request.locator,
                "section": bucket,
            },
        )
        if self._total_count >= self._total_limit:
            return self._reject(
                request,
                bucket,
                code="budget_exhausted",
                message=(
                    f"The whole-report evidence budget ({self._total_limit} requests) is "
                    "exhausted. Write the remaining text using only evidence already granted."
                ),
            )
        if self._section_counts.get(bucket, 0) >= self._per_section_limit:
            return self._reject(
                request,
                bucket,
                code="budget_exhausted",
                message=(
                    f"Section '{bucket}' already used its {self._per_section_limit} evidence "
                    "requests. Write this section with the evidence already granted."
                ),
            )
        # Budget is consumed by every resolved attempt (granted or not) so a
        # writer cannot probe indefinitely; budget refusals above are free.
        self._total_count += 1
        self._section_counts[bucket] = self._section_counts.get(bucket, 0) + 1
        try:
            outcome = self._resolve(request)
        except Exception:  # noqa: BLE001 - the loop must never break the report
            outcome = EvidenceRejection(
                artifact_id=request.artifact_id,
                locator=request.locator,
                reason_code="invalid_payload",
                message="Evidence resolution failed unexpectedly; pick another artifact.",
                available_artifacts=self.catalog(),
            )
        if isinstance(outcome, EvidenceGrant):
            self._exchanges.append(
                InterleaveExchange(section=bucket, request=request, grant=outcome)
            )
            self._emit(
                EVIDENCE_INTERLEAVE_GRANTED,
                {
                    "artifact_id": outcome.artifact_id,
                    "locator": outcome.locator,
                    "section": bucket,
                    "value_count": len(outcome.values),
                    "text_count": len(outcome.texts),
                },
            )
            return outcome
        self._exchanges.append(
            InterleaveExchange(section=bucket, request=request, rejection=outcome)
        )
        self._emit(
            EVIDENCE_INTERLEAVE_REJECTED,
            {
                "artifact_id": outcome.artifact_id,
                "locator": outcome.locator,
                "section": bucket,
                "reason_code": outcome.reason_code,
            },
        )
        return outcome

    def _reject(
        self,
        request: EvidenceRequest,
        section: str,
        *,
        code: InterleaveRejectionCode,
        message: str,
        available: list[str] | None = None,
    ) -> EvidenceRejection:
        rejection = EvidenceRejection(
            artifact_id=request.artifact_id,
            locator=request.locator,
            reason_code=code,
            message=message,
            available_artifacts=self.catalog() if available is None else available,
        )
        self._exchanges.append(
            InterleaveExchange(section=section, request=request, rejection=rejection)
        )
        self._emit(
            EVIDENCE_INTERLEAVE_REJECTED,
            {
                "artifact_id": rejection.artifact_id,
                "locator": rejection.locator,
                "section": section,
                "reason_code": rejection.reason_code,
            },
        )
        return rejection

    def _resolve(self, request: EvidenceRequest) -> EvidenceGrant | EvidenceRejection:
        artifact = self._resolver.lookup(request.artifact_id)
        if artifact is None:
            return EvidenceRejection(
                artifact_id=request.artifact_id,
                locator=request.locator,
                reason_code="unknown_artifact",
                message=(
                    f"No persisted artifact '{request.artifact_id}' is available to this "
                    "report. Request one of the listed artifacts instead."
                ),
                available_artifacts=self.catalog(),
            )
        if artifact.type not in _RESOLVABLE_TYPES:
            return EvidenceRejection(
                artifact_id=request.artifact_id,
                locator=request.locator,
                reason_code="unsupported_type",
                message=(
                    f"Artifact type {artifact.type.value} carries no resolvable reducer "
                    "numbers. Only SqlResult, StatTestResult, DatasetProfile, and "
                    "ValidatedFinding artifacts can be requested."
                ),
                available_artifacts=self.catalog(),
            )
        try:
            return _resolve_typed(request, artifact, catalog=self.catalog())
        except ValidationError:
            return EvidenceRejection(
                artifact_id=request.artifact_id,
                locator=request.locator,
                reason_code="invalid_payload",
                message=(
                    f"The persisted payload of {artifact.id} does not parse as "
                    f"{artifact.type.value}; pick another artifact."
                ),
                available_artifacts=self.catalog(),
            )

    def _emit(self, event_type: str, summary: dict[str, Any]) -> None:
        if self._trace_sink is None:
            return
        try:
            self._trace_sink(event_type, summary)
        except Exception:  # noqa: BLE001 - observability must never break writes
            return


def _resolve_typed(
    request: EvidenceRequest, artifact: Artifact, *, catalog: list[str]
) -> EvidenceGrant | EvidenceRejection:
    if artifact.type is ArtifactType.SQL_RESULT:
        return _resolve_sql_result(request, artifact, catalog=catalog)
    if artifact.type is ArtifactType.STAT_TEST_RESULT:
        return _resolve_stat_test(request, artifact, catalog=catalog)
    if artifact.type is ArtifactType.DATASET_PROFILE:
        return _resolve_dataset_profile(request, artifact, catalog=catalog)
    return _resolve_validated_finding(request, artifact, catalog=catalog)


def _resolve_sql_result(
    request: EvidenceRequest, artifact: Artifact, *, catalog: list[str]
) -> EvidenceGrant | EvidenceRejection:
    sql_result = SqlResult.model_validate(artifact.payload)
    # Reuse the validator's exact locator resolution so a granted locator is by
    # construction one the downstream number gate can also resolve.
    reference = EvidenceRef(kind="sql", artifact_id=artifact.id, locator=request.locator)
    values = _sql_result_numbers(reference, sql_result)
    if not values:
        return EvidenceRejection(
            artifact_id=artifact.id,
            locator=request.locator,
            reason_code="unresolvable_locator",
            message=(
                f"Locator '{request.locator}' resolves to no numeric cell in this "
                "SqlResult. Use 'rows', 'rows[i]', 'rows[i].col', or a column name "
                f"from: {', '.join(sql_result.columns)}."
            ),
            available_artifacts=catalog,
        )
    # ``_sql_result_numbers`` only ever yields raw-unit values.
    return EvidenceGrant(
        artifact_id=artifact.id,
        artifact_type=artifact.type.value,
        locator=request.locator,
        values=[GrantValue(value=value, unit="raw") for value, _unit in values],
    )


def _resolve_stat_test(
    request: EvidenceRequest, artifact: Artifact, *, catalog: list[str]
) -> EvidenceGrant | EvidenceRejection:
    stat = StatTestResult.model_validate(artifact.payload)
    fields = {
        "statistic": stat.statistic,
        "p_value": stat.p_value,
        "effect_size": stat.effect_size,
        "sample_size": float(stat.sample_size),
    }
    locator = request.locator.strip()
    if not locator:
        values = [
            GrantValue(value=float(value))
            for name in _STAT_LOCATORS
            if (value := fields[name]) is not None
        ]
        return EvidenceGrant(
            artifact_id=artifact.id,
            artifact_type=artifact.type.value,
            locator=locator,
            values=values,
            texts=[f"{stat.test_type} on {stat.dataset_id}"],
        )
    value = fields.get(locator)
    if value is None:
        return EvidenceRejection(
            artifact_id=artifact.id,
            locator=locator,
            reason_code="unresolvable_locator",
            message=(
                f"Locator '{locator}' is not a resolvable StatTestResult field. "
                f"Use one of: {', '.join(_STAT_LOCATORS)} (or empty for all)."
            ),
            available_artifacts=catalog,
        )
    return EvidenceGrant(
        artifact_id=artifact.id,
        artifact_type=artifact.type.value,
        locator=locator,
        values=[GrantValue(value=float(value))],
        texts=[f"{stat.test_type} on {stat.dataset_id}"],
    )


def _resolve_dataset_profile(
    request: EvidenceRequest, artifact: Artifact, *, catalog: list[str]
) -> EvidenceGrant | EvidenceRejection:
    profile = DatasetProfile.model_validate(artifact.payload)
    locator = request.locator.strip()
    values: list[GrantValue] | None = None
    if locator in {"rows", "row_count"}:
        values = [GrantValue(value=float(profile.rows))]
    elif locator in {"columns", "column_count"}:
        values = [GrantValue(value=float(profile.columns))]
    elif locator in {"missing_percent", "missing_percent.*"}:
        values = [
            GrantValue(value=float(value), unit="percent")
            for value in profile.missing_percent.values()
        ]
    elif locator.startswith("missing_percent."):
        column = locator.removeprefix("missing_percent.")
        found = profile.missing_percent.get(column)
        if found is not None:
            values = [GrantValue(value=float(found), unit="percent")]
    if values is None:
        return EvidenceRejection(
            artifact_id=artifact.id,
            locator=locator,
            reason_code="unresolvable_locator",
            message=(
                f"Locator '{locator}' is not a resolvable DatasetProfile field. Use "
                "'rows', 'columns', 'missing_percent', or 'missing_percent.<column>' "
                f"with a column from: {', '.join(profile.column_names[:20])}."
            ),
            available_artifacts=catalog,
        )
    return EvidenceGrant(
        artifact_id=artifact.id,
        artifact_type=artifact.type.value,
        locator=locator,
        values=values,
        texts=[f"Profile of {profile.name}"],
    )


def _resolve_validated_finding(
    request: EvidenceRequest, artifact: Artifact, *, catalog: list[str]
) -> EvidenceGrant | EvidenceRejection:
    finding = ValidatedFinding.model_validate(artifact.payload)
    locator = request.locator.strip()
    claims = finding.findings
    if locator:
        match = _FINDING_LOCATOR_PATTERN.fullmatch(locator)
        index = int(match.group("index")) if match else -1
        if match is None or not 0 <= index < len(claims):
            return EvidenceRejection(
                artifact_id=artifact.id,
                locator=locator,
                reason_code="unresolvable_locator",
                message=(
                    f"Locator '{locator}' does not select a claim of this "
                    f"ValidatedFinding. Use '' for all claims or 'findings[i]' with "
                    f"i in 0..{max(len(claims) - 1, 0)}."
                ),
                available_artifacts=catalog,
            )
        claims = [claims[index]]
    values = [
        GrantValue(
            value=float(reference.value),
            unit=reference.unit,
            unit_label=reference.unit_label,
            unit_reference=reference.unit_reference,
        )
        for claim in claims
        for reference in claim.evidence
        if isinstance(reference.value, int | float) and not isinstance(reference.value, bool)
    ]
    return EvidenceGrant(
        artifact_id=artifact.id,
        artifact_type=artifact.type.value,
        locator=locator,
        values=values,
        texts=[claim.text for claim in claims],
    )
