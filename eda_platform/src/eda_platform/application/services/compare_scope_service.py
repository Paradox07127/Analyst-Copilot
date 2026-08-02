"""Lazy semantic Compare scopes built from persisted session read models.

The overview endpoint remains index-first. This service is invoked only after a
scope tab is opened, bounds every response, keeps both result families isolated,
and never returns arbitrary artifact/debug payloads.
"""

from __future__ import annotations

import base64
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from eda_platform.application.dto import (
    AnalysisView,
    CompareScopeCounts,
    CompareScopeField,
    CompareScopeItem,
    CompareScopeName,
    CompareScopeRecord,
    CompareScopeSideState,
    CompareScopeView,
    CompareSessionSide,
    FindingSummary,
    QuestionSummary,
    ReportView,
    SessionDetail,
    TraceEventRow,
)
from eda_platform.application.services.analysis_service import AnalysisService
from eda_platform.application.services.compare_matchers import (
    ArtifactComparable,
    ExecutionComparable,
    QuestionComparable,
    ReportSectionComparable,
    ScopeComparable,
    match_artifacts,
    match_execution_records,
    match_questions,
    match_report_sections,
    match_scope_records,
    normalize_text,
)
from eda_platform.application.services.compare_matchers.canonical import (
    canonical_json,
    stable_digest,
)
from eda_platform.application.services.compare_matchers.generic import MatchResult
from eda_platform.application.services.compare_service import (
    CompareProjectMismatchError,
    CompareSameRunError,
)
from eda_platform.application.services.finding_service import FindingService
from eda_platform.application.services.question_service import QuestionService
from eda_platform.application.services.report_service import ReportService
from eda_platform.application.services.session_family_service import (
    SessionFamilyService,
    SessionResultFamily,
)
from eda_platform.application.services.session_service import (
    InvalidCursorError,
    SessionNotFoundError,
    SessionService,
)
from eda_platform.application.services.trace_service import TraceService
from eda_platform.core.ids import INTERNAL_SESSION_MARKER
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType

CompareFilter = Literal["all", "differences"]

DEFAULT_SCOPE_LIMIT = 50
MAX_SCOPE_LIMIT = 100
MAX_SCOPE_SOURCE_RECORDS = 2000
MAX_FIELD_CHARS = 500
MAX_RECORD_FIELDS = 40
_VOLATILE_PAYLOAD_KEYS = frozenset(
    {
        "id",
        "artifact_id",
        "session_id",
        "project_id",
        "created_at",
        "updated_at",
        "generated_at",
        "timestamp",
        "path",
    }
)
_TRACE_SUMMARY_KEYS = (
    "status",
    "task",
    "kind",
    "model",
    "provider",
    "tool",
    "tool_name",
    "question_id",
    "investigation_id",
    "attempt",
    "retry",
    "total_tokens",
    "prompt_tokens",
    "completion_tokens",
    "estimated_cost_usd",
    "row_count",
    "truncated",
    "error_type",
)


@dataclass(frozen=True)
class _Prepared:
    record: CompareScopeRecord
    comparable: Any


@dataclass(frozen=True)
class _Extracted:
    prepared: list[_Prepared]
    state: CompareScopeSideState
    warnings: list[str]


@dataclass(frozen=True)
class _Pair:
    project_id: str
    left: SessionDetail
    right: SessionDetail
    left_family: SessionResultFamily
    right_family: SessionResultFamily


class CompareScopeService:
    def __init__(
        self,
        store: ArtifactStore,
        runs: SessionService,
        questions: QuestionService,
        analysis: AnalysisService,
        findings: FindingService,
        reports: ReportService,
        trace: TraceService,
        families: SessionFamilyService | None = None,
    ) -> None:
        self._store = store
        self._runs = runs
        self._questions = questions
        self._analysis = analysis
        self._findings = findings
        self._reports = reports
        self._trace = trace
        self._families = families or SessionFamilyService(store)

    def compare_scope(
        self,
        scope: CompareScopeName,
        left_session_id: str,
        right_session_id: str,
        *,
        filter_mode: CompareFilter = "all",
        limit: int = DEFAULT_SCOPE_LIMIT,
        cursor: str | None = None,
    ) -> CompareScopeView:
        limit = max(1, min(limit, MAX_SCOPE_LIMIT))
        pair = self._pair(left_session_id, right_session_id)
        left = self._extract(scope, pair.left, pair.left_family)
        right = self._extract(scope, pair.right, pair.right_family)
        matched = self._match(scope, left.prepared, right.prepared)
        items = self._items(matched, left.prepared, right.prepared)
        counts = _counts(items)
        if filter_mode == "differences":
            items = [item for item in items if item.change != "same"]
        offset = _decode_cursor(
            cursor,
            scope=scope,
            left=left_session_id,
            right=right_session_id,
            filter_mode=filter_mode,
        )
        page = items[offset : offset + limit]
        next_cursor = (
            _encode_cursor(
                offset + limit,
                scope=scope,
                left=left_session_id,
                right=right_session_id,
                filter_mode=filter_mode,
            )
            if offset + limit < len(items)
            else None
        )
        return CompareScopeView(
            project_id=pair.project_id,
            scope=scope,
            left=_to_side(pair.left),
            right=_to_side(pair.right),
            left_state=left.state,
            right_state=right.state,
            counts=counts,
            items=page,
            next_cursor=next_cursor,
            warnings=list(
                dict.fromkeys(
                    [
                        *pair.left_family.warnings,
                        *pair.right_family.warnings,
                        *left.warnings,
                        *right.warnings,
                    ]
                )
            ),
        )

    def _pair(self, left_session_id: str, right_session_id: str) -> _Pair:
        if left_session_id == right_session_id:
            raise CompareSameRunError(left_session_id)
        left = self._runs.get_session_detail(left_session_id)
        right = self._runs.get_session_detail(right_session_id)
        if left.project_id != right.project_id:
            raise CompareProjectMismatchError(left_session_id, right_session_id)
        return _Pair(
            project_id=left.project_id,
            left=left,
            right=right,
            left_family=self._families.collect(left.project_id, left_session_id),
            right_family=self._families.collect(right.project_id, right_session_id),
        )

    def _extract(
        self,
        scope: CompareScopeName,
        detail: SessionDetail,
        family: SessionResultFamily,
    ) -> _Extracted:
        extractors = {
            "questions": self._questions_scope,
            "analysis": self._analysis_scope,
            "findings": self._findings_scope,
            "report": self._report_scope,
            "artifacts": self._artifacts_scope,
            "execution": self._execution_scope,
        }
        try:
            prepared, warnings = extractors[scope](detail, family)
        except (OSError, ValueError, KeyError, SessionNotFoundError) as exc:
            return _Extracted(
                prepared=[],
                state=CompareScopeSideState(state="unavailable", reason=str(exc)),
                warnings=[f"{detail.session_id}: {scope} unavailable: {exc}"],
            )
        return _Extracted(
            prepared=prepared,
            state=_side_state(detail, prepared, scope),
            warnings=warnings,
        )

    def _questions_scope(
        self, detail: SessionDetail, family: SessionResultFamily
    ) -> tuple[list[_Prepared], list[str]]:
        del family
        view = self._questions.list_questions(detail.session_id)
        prepared: list[_Prepared] = []
        for question in view.questions[:MAX_SCOPE_SOURCE_RECORDS]:
            payload = _question_payload(question)
            record_id = f"{detail.session_id}:{question.question_id}"
            comparable = QuestionComparable(
                record_id=record_id,
                question_text=question.question,
                target_datasets=tuple(question.target_datasets),
                lineage_identity=question.question_id,
                comparison_payload=payload,
            )
            prepared.append(
                _Prepared(
                    record=_question_record(detail.session_id, record_id, question),
                    comparable=comparable,
                )
            )
        warnings = (
            [f"questions truncated at {MAX_SCOPE_SOURCE_RECORDS} records"]
            if len(view.questions) > MAX_SCOPE_SOURCE_RECORDS
            else []
        )
        return prepared, warnings

    def _analysis_scope(
        self, detail: SessionDetail, family: SessionResultFamily
    ) -> tuple[list[_Prepared], list[str]]:
        prepared: list[_Prepared] = []
        warnings: list[str] = []
        occurrences: Counter[str] = Counter()
        for session_id in _visible_family_ids(family):
            try:
                view = self._analysis.get_analysis(session_id)
            except SessionNotFoundError:
                continue
            prepared.extend(self._analysis_records(session_id, view, occurrences))
            if len(prepared) >= MAX_SCOPE_SOURCE_RECORDS:
                warnings.append(f"analysis truncated at {MAX_SCOPE_SOURCE_RECORDS} records")
                return prepared[:MAX_SCOPE_SOURCE_RECORDS], warnings
        return prepared, warnings

    def _analysis_records(
        self,
        session_id: str,
        view: AnalysisView,
        occurrences: Counter[str],
    ) -> list[_Prepared]:
        result: list[_Prepared] = []
        for table in view.tables:
            base = "|".join(
                (
                    "table",
                    normalize_text(table.question),
                    normalize_text(table.dataset_name),
                    normalize_text(table.kind),
                    normalize_text(table.title),
                )
            )
            stable = _with_occurrence(base, occurrences)
            payload = {
                "title": table.title,
                "description": table.description,
                "columns": table.columns,
                "rows": len(table.rows),
                "row_digest": stable_digest(table.rows),
                "min_sample_size": table.min_sample_size,
                "small_sample": table.small_sample,
            }
            record_id = f"{session_id}:{table.artifact_id}"
            result.append(
                _Prepared(
                    record=CompareScopeRecord(
                        record_id=record_id,
                        title=table.title,
                        kind=f"analysis table · {table.kind}",
                        status="small sample" if table.small_sample else "ready",
                        summary=_clip(table.description),
                        source_session_id=session_id,
                        artifact_id=table.artifact_id,
                        tags=[table.dataset_name],
                        fields=[
                            _field("question", "Question", table.question),
                            _field("dataset", "Dataset", table.dataset_name),
                            _field("rows", "Rows", len(table.rows), "number"),
                            _field("columns", "Columns", table.columns, "list"),
                            _field(
                                "min_sample_size",
                                "Minimum sample",
                                table.min_sample_size,
                                "number",
                            ),
                        ],
                    ),
                    comparable=ScopeComparable(record_id, stable, payload),
                )
            )
        for test in view.stat_tests:
            base = "|".join(
                (
                    "stat",
                    normalize_text(test.dataset_name),
                    normalize_text(test.test_type),
                    normalize_text(test.group_column or ""),
                    normalize_text(test.value_column or ""),
                )
            )
            stable = _with_occurrence(base, occurrences)
            payload = {
                "statistic": test.statistic,
                "p_value": test.p_value,
                "effect_size": test.effect_size,
                "sample_size": test.sample_size,
                "significant": test.significant,
                "conclusion": test.conclusion,
                "warnings": test.warnings,
            }
            record_id = f"{session_id}:{test.artifact_id}"
            result.append(
                _Prepared(
                    record=CompareScopeRecord(
                        record_id=record_id,
                        title=test.conclusion or test.test_type,
                        kind=f"statistical test · {test.test_type}",
                        status=(
                            "significant"
                            if test.significant is True
                            else "not significant"
                            if test.significant is False
                            else "unknown"
                        ),
                        source_session_id=session_id,
                        artifact_id=test.artifact_id,
                        tags=[test.dataset_name],
                        fields=[
                            _field("dataset", "Dataset", test.dataset_name),
                            _field("statistic", "Statistic", test.statistic, "number"),
                            _field("p_value", "P-value", test.p_value_display, "number"),
                            _field("effect_size", "Effect size", test.effect_size, "number"),
                            _field("sample_size", "Sample size", test.sample_size, "number"),
                        ],
                    ),
                    comparable=ScopeComparable(record_id, stable, payload),
                )
            )
        for model in view.model_cards:
            base = "|".join(
                (
                    "model",
                    normalize_text(model.dataset_name),
                    normalize_text(model.task_type),
                    normalize_text(model.target_column),
                )
            )
            stable = _with_occurrence(base, occurrences)
            payload = {
                "model_type": model.model_type,
                "split_strategy": model.split_strategy,
                "train_rows": model.train_rows,
                "test_rows": model.test_rows,
                "metrics": model.metrics,
                "leakage_verdict": model.leakage_verdict,
                "limitations": model.limitations,
            }
            record_id = f"{session_id}:{model.artifact_id}"
            result.append(
                _Prepared(
                    record=CompareScopeRecord(
                        record_id=record_id,
                        title=f"{model.model_type} · {model.target_column}",
                        kind=f"model · {model.task_type}",
                        status=model.leakage_verdict,
                        source_session_id=session_id,
                        artifact_id=model.artifact_id,
                        tags=[model.dataset_name],
                        fields=[
                            _field("dataset", "Dataset", model.dataset_name),
                            _field("target", "Target", model.target_column),
                            _field("split", "Split", model.split_strategy),
                            _field("train_rows", "Train rows", model.train_rows, "number"),
                            _field("test_rows", "Test rows", model.test_rows, "number"),
                            _field(
                                "headline_metric",
                                "Headline metric",
                                (
                                    f"{model.headline_metric}={model.headline_metric_value}"
                                    if model.headline_metric
                                    else "—"
                                ),
                            ),
                        ],
                    ),
                    comparable=ScopeComparable(record_id, stable, payload),
                )
            )
        return result

    def _findings_scope(
        self, detail: SessionDetail, family: SessionResultFamily
    ) -> tuple[list[_Prepared], list[str]]:
        view = self._findings.list_findings(detail.session_id)
        allowed = set(family.session_ids)
        findings = [finding for finding in view.findings if finding.source_session_id in allowed]
        occurrences: Counter[str] = Counter()
        prepared: list[_Prepared] = []
        for finding in findings[:MAX_SCOPE_SOURCE_RECORDS]:
            base = f"finding|{normalize_text(finding.question)}"
            stable = _with_occurrence(base, occurrences)
            payload = _finding_payload(finding)
            record_id = f"{finding.source_session_id}:{finding.artifact_id}"
            prepared.append(
                _Prepared(
                    record=_finding_record(record_id, finding),
                    comparable=ScopeComparable(record_id, stable, payload),
                )
            )
        warnings = list(view.warnings)
        if len(findings) > MAX_SCOPE_SOURCE_RECORDS:
            warnings.append(f"findings truncated at {MAX_SCOPE_SOURCE_RECORDS} records")
        return prepared, warnings

    def _report_scope(
        self, detail: SessionDetail, family: SessionResultFamily
    ) -> tuple[list[_Prepared], list[str]]:
        report = self._latest_family_report(family)
        if report is None or not report.markdown.strip():
            return [], []
        prepared: list[_Prepared] = []
        occurrences: Counter[str] = Counter()
        for index, (title, content) in enumerate(_markdown_sections(report.markdown)):
            base = f"report|{normalize_text(title)}"
            occurrence = occurrences[base]
            occurrences[base] += 1
            record_id = f"{report.session_id}:section:{index}"
            payload = {
                "content_digest": stable_digest(content),
                "word_count": len(content.split()),
                "status": report.status,
            }
            comparable = ReportSectionComparable(
                record_id=record_id,
                title=title,
                required_key=f"{normalize_text(title)}:{occurrence}",
                comparison_payload=payload,
            )
            prepared.append(
                _Prepared(
                    record=CompareScopeRecord(
                        record_id=record_id,
                        title=title,
                        kind="report section",
                        status=report.status,
                        summary=_clip(_plain_markdown(content)),
                        source_session_id=report.session_id,
                        fields=[
                            _field("word_count", "Words", len(content.split()), "number"),
                            _field("status", "Report status", report.status, "status"),
                            _field("preview", "Preview", _plain_markdown(content)),
                        ],
                    ),
                    comparable=comparable,
                )
            )
        return prepared, []

    def _latest_family_report(self, family: SessionResultFamily) -> ReportView | None:
        candidates: list[tuple[datetime | None, int, ReportView]] = []
        for index, session_id in enumerate(_visible_family_ids(family)):
            try:
                report = self._reports.get_report(session_id)
            except SessionNotFoundError:
                continue
            if report.markdown.strip():
                candidates.append((report.generated_at, index, report))
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda item: (
                item[0] is not None,
                item[0] or datetime.min,
                item[1],
            ),
        )[2]

    def _artifacts_scope(
        self, detail: SessionDetail, family: SessionResultFamily
    ) -> tuple[list[_Prepared], list[str]]:
        artifacts: list[Artifact] = []
        warnings: list[str] = []
        for session_id in family.session_ids:
            remaining = MAX_SCOPE_SOURCE_RECORDS - len(artifacts)
            if remaining <= 0:
                warnings.append(f"artifacts truncated at {MAX_SCOPE_SOURCE_RECORDS} records")
                break
            session_artifacts = self._store.list_indexed_artifacts(
                project_id=detail.project_id,
                session_id=session_id,
                artifact_types=tuple(ArtifactType),
            )
            artifacts.extend(session_artifacts[:remaining])
            if len(session_artifacts) > remaining:
                warnings.append(f"artifacts truncated at {MAX_SCOPE_SOURCE_RECORDS} records")
                break
        occurrences: Counter[str] = Counter()
        prepared: list[_Prepared] = []
        for artifact in artifacts:
            identity = _artifact_identity(artifact)
            occurrence = occurrences[identity]
            occurrences[identity] += 1
            payload = _artifact_comparison_payload(artifact)
            comparable = ArtifactComparable(
                record_id=f"{artifact.session_id}:{artifact.id}",
                artifact_type=artifact.type.value,
                stable_identity=identity,
                occurrence_index=occurrence,
                comparison_payload=payload,
            )
            keys = [str(key) for key in artifact.payload if str(key) not in _VOLATILE_PAYLOAD_KEYS][
                :MAX_RECORD_FIELDS
            ]
            prepared.append(
                _Prepared(
                    record=CompareScopeRecord(
                        record_id=comparable.record_id,
                        title=_artifact_title(artifact),
                        kind=artifact.type.value,
                        status="warning" if artifact.warnings else "produced",
                        summary=(
                            f"{len(artifact.parents)} parent(s) · "
                            f"{len(artifact.warnings)} warning(s)"
                        ),
                        source_session_id=artifact.session_id,
                        artifact_id=artifact.id,
                        tags=keys[:6],
                        evidence_ids=list(artifact.parents)[:20],
                        fields=[
                            _field("type", "Type", artifact.type.value),
                            _field("parents", "Parents", len(artifact.parents), "number"),
                            _field("warnings", "Warnings", len(artifact.warnings), "number"),
                            _field("payload_fields", "Payload fields", keys, "list"),
                        ],
                    ),
                    comparable=comparable,
                )
            )
        return prepared, warnings

    def _execution_scope(
        self, detail: SessionDetail, family: SessionResultFamily
    ) -> tuple[list[_Prepared], list[str]]:
        prepared: list[_Prepared] = []
        warnings: list[str] = []
        metrics = self._family_metrics(family)
        for key, label in _EXECUTION_METRICS:
            value = metrics.get(key, 0)
            record_id = f"{detail.session_id}:metric:{key}"
            comparable = ExecutionComparable(
                record_id=record_id,
                parent_match_key="session",
                span_kind="metric",
                operation_name=key,
                comparison_payload={"value": value},
            )
            prepared.append(
                _Prepared(
                    record=CompareScopeRecord(
                        record_id=record_id,
                        title=label,
                        kind="execution metric",
                        status="measured",
                        source_session_id=detail.session_id,
                        fields=[_field("value", "Value", value, "number")],
                    ),
                    comparable=comparable,
                )
            )

        occurrences: Counter[str] = Counter()
        for session_id in _visible_family_ids(family):
            try:
                events, truncated = self._trace_events(session_id)
            except SessionNotFoundError:
                continue
            if truncated:
                warnings.append(f"execution events truncated at {MAX_SCOPE_SOURCE_RECORDS} records")
            for event in events:
                if len(prepared) >= MAX_SCOPE_SOURCE_RECORDS:
                    break
                summary = _safe_trace_summary(event.summary)
                originating = str(
                    summary.get("question_id")
                    or summary.get("investigation_id")
                    or summary.get("task")
                    or ""
                )
                tool = str(summary.get("tool_name") or summary.get("tool") or "")
                base = "|".join(
                    (
                        event.event_type,
                        event.name,
                        originating,
                        tool,
                    )
                )
                occurrence = occurrences[base]
                occurrences[base] += 1
                record_id = f"{session_id}:event:{event.event_id}"
                payload = {
                    "duration_seconds": event.duration_seconds,
                    **summary,
                }
                comparable = ExecutionComparable(
                    record_id=record_id,
                    parent_match_key="session",
                    span_kind=event.event_type,
                    operation_name=event.name,
                    originating_key=originating,
                    tool_name=tool,
                    occurrence_index=occurrence,
                    model=str(summary.get("model") or ""),
                    comparison_payload=payload,
                )
                prepared.append(
                    _Prepared(
                        record=CompareScopeRecord(
                            record_id=record_id,
                            title=event.name or event.event_type,
                            kind=event.event_type,
                            status=str(summary.get("status") or "recorded"),
                            source_session_id=session_id,
                            tags=[
                                value
                                for value in (
                                    str(summary.get("model") or ""),
                                    tool,
                                    originating,
                                )
                                if value
                            ],
                            fields=[
                                _field(
                                    "duration_seconds",
                                    "Duration (s)",
                                    event.duration_seconds,
                                    "number",
                                ),
                                *[
                                    _field(key, key.replace("_", " ").title(), value)
                                    for key, value in summary.items()
                                    if key not in {"question_id", "investigation_id"}
                                ][:8],
                            ],
                        ),
                        comparable=comparable,
                    )
                )
        return prepared, warnings

    def _family_metrics(self, family: SessionResultFamily) -> dict[str, float]:
        totals: dict[str, float] = {key: 0.0 for key, _label in _EXECUTION_METRICS}
        for session_id in _visible_family_ids(family):
            try:
                metrics = self._trace.get_metrics(session_id)
            except SessionNotFoundError:
                continue
            for key in totals:
                value = getattr(metrics, key, 0)
                if isinstance(value, int | float):
                    totals[key] += float(value)
        return totals

    def _trace_events(self, session_id: str) -> tuple[list[TraceEventRow], bool]:
        events: list[TraceEventRow] = []
        cursor: str | None = None
        while len(events) < MAX_SCOPE_SOURCE_RECORDS:
            page = self._trace.list_events(session_id, limit=100, cursor=cursor)
            events.extend(page.items)
            cursor = page.next_cursor
            if not cursor:
                return events, False
        return events[:MAX_SCOPE_SOURCE_RECORDS], cursor is not None

    def _match(
        self,
        scope: CompareScopeName,
        left: list[_Prepared],
        right: list[_Prepared],
    ) -> list[MatchResult[Any]]:
        left_values = [item.comparable for item in left]
        right_values = [item.comparable for item in right]
        if scope == "questions":
            return match_questions(left_values, right_values)
        if scope in {"analysis", "findings"}:
            return match_scope_records(left_values, right_values)
        if scope == "report":
            return match_report_sections(left_values, right_values)
        if scope == "artifacts":
            return match_artifacts(left_values, right_values)
        return match_execution_records(left_values, right_values)

    def _items(
        self,
        matches: list[MatchResult[Any]],
        left: list[_Prepared],
        right: list[_Prepared],
    ) -> list[CompareScopeItem]:
        left_records = {item.comparable.record_id: item.record for item in left}
        right_records = {item.comparable.record_id: item.record for item in right}
        items: list[CompareScopeItem] = []
        for match in matches:
            left_value = match.left
            right_value = match.right
            items.append(
                CompareScopeItem(
                    match_key=match.match_key,
                    matcher_version=match.version,
                    reason=match.reason,
                    confidence=match.confidence.value,
                    match_status=match.match_status.value,
                    change=match.change.value,
                    left=(
                        left_records.get(left_value.record_id) if left_value is not None else None
                    ),
                    right=(
                        right_records.get(right_value.record_id)
                        if right_value is not None
                        else None
                    ),
                    changed_fields=_changed_fields(left_value, right_value),
                )
            )
        order = {"changed": 0, "added": 1, "removed": 2, "same": 3, "unavailable": 4}
        def sort_key(item: CompareScopeItem) -> tuple[int, str, str]:
            record = item.left if item.left is not None else item.right
            return (
                order[item.change],
                record.title if record is not None else "",
                item.match_key,
            )

        return sorted(items, key=sort_key)


_EXECUTION_METRICS = (
    ("llm_calls", "LLM calls"),
    ("tool_calls", "Tool calls"),
    ("total_tokens", "Total tokens"),
    ("est_cost_usd", "Estimated cost (USD)"),
    ("duration_seconds", "Duration (seconds)"),
    ("event_count", "Trace events"),
    ("failures_count", "Failures"),
    ("findings_count", "Findings"),
    ("question_answered", "Questions answered"),
    ("question_abstained", "Questions abstained"),
    ("question_failed", "Questions failed"),
)


def _question_payload(question: QuestionSummary) -> dict[str, Any]:
    return {
        "question": question.question,
        "origin": question.origin,
        "analysis_mode": question.analysis_mode,
        "value_category": question.value_category,
        "feasibility": question.feasibility_status,
        "proposed_action": question.proposed_action,
        "priority": question.priority,
        "target_datasets": sorted(question.target_datasets),
        "business_decision": question.business_decision,
        "executable": question.executable,
        "execution_outcome": question.execution.outcome if question.execution else None,
        "findings_count": question.execution.findings_count if question.execution else 0,
        "card_version": question.card_version,
        "value_hypothesis": question.value_hypothesis,
        "success_criterion": question.success_criterion,
        "risks": question.risks,
        "data_requirements": question.data_requirements,
    }


def _question_record(
    source_session_id: str,
    record_id: str,
    question: QuestionSummary,
) -> CompareScopeRecord:
    execution = question.execution.outcome if question.execution else "not executed"
    return CompareScopeRecord(
        record_id=record_id,
        title=question.question,
        kind=f"question · {question.origin}",
        status=execution,
        summary=_clip(question.business_decision or question.value_hypothesis),
        source_session_id=source_session_id,
        artifact_id=(question.execution.qexec_artifact_id if question.execution else None),
        tags=[
            value
            for value in (
                question.analysis_mode,
                question.feasibility_status,
                question.proposed_action,
            )
            if value
        ],
        evidence_ids=(
            [question.execution.qexec_artifact_id]
            if question.execution and question.execution.qexec_artifact_id
            else []
        ),
        fields=[
            _field("priority", "Priority", question.priority, "number"),
            _field("feasibility", "Feasibility", question.feasibility_status, "status"),
            _field("execution_outcome", "Execution", execution, "status"),
            _field("target_datasets", "Datasets", question.target_datasets, "list"),
            _field("business_decision", "Business decision", question.business_decision),
            _field("value_hypothesis", "Value hypothesis", question.value_hypothesis),
            _field("success_criterion", "Success criterion", question.success_criterion),
            _field("risks", "Risks", question.risks, "list"),
        ],
    )


def _finding_payload(finding: FindingSummary) -> dict[str, Any]:
    return {
        "question": finding.question,
        "claim_class": finding.claim_class,
        "evidence_support": finding.evidence_support,
        "analytical_reliability": finding.analytical_reliability,
        "decision_readiness": finding.decision_readiness,
        "report_readiness": finding.report_readiness,
        "statements": [statement.text for statement in finding.statements],
        "evidence": sorted(
            (
                evidence.kind,
                evidence.locator,
            )
            for statement in finding.statements
            for evidence in statement.evidence
        ),
        "limitations": finding.limitations,
        "interpretation": finding.interpretation,
        "freshness": finding.freshness.status,
    }


def _finding_record(record_id: str, finding: FindingSummary) -> CompareScopeRecord:
    statements = [statement.text for statement in finding.statements]
    evidence_ids = [
        evidence.artifact_id
        for statement in finding.statements
        for evidence in statement.evidence
        if evidence.artifact_id
    ]
    return CompareScopeRecord(
        record_id=record_id,
        title=finding.question,
        kind=f"finding · {finding.claim_class}",
        status=finding.decision_readiness,
        summary=_clip(finding.interpretation or (statements[0] if statements else "")),
        source_session_id=finding.source_session_id,
        artifact_id=finding.artifact_id,
        tags=[finding.analytical_reliability, finding.freshness.status],
        evidence_ids=list(dict.fromkeys(evidence_ids))[:20],
        fields=[
            _field("statements", "Statements", statements, "list"),
            _field("evidence_support", "Evidence", finding.evidence_support, "status"),
            _field(
                "analytical_reliability",
                "Reliability",
                finding.analytical_reliability,
                "status",
            ),
            _field(
                "decision_readiness",
                "Decision readiness",
                finding.decision_readiness,
                "status",
            ),
            _field("report_readiness", "Report readiness", finding.report_readiness, "status"),
            _field("freshness", "Freshness", finding.freshness.status, "status"),
            _field("limitations", "Limitations", finding.limitations, "list"),
        ],
    )


def _artifact_identity(artifact: Artifact) -> str:
    payload = artifact.payload
    parts = [artifact.type.value]
    for key in (
        "question_id",
        "dataset_id",
        "name",
        "title",
        "kind",
        "test_type",
        "target_column",
        "metric_id",
    ):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(f"{key}:{normalize_text(value)}")
    return "|".join(parts)


def _artifact_title(artifact: Artifact) -> str:
    for key in ("title", "question", "name", "dataset_id", "target_column"):
        value = artifact.payload.get(key)
        if isinstance(value, str) and value.strip():
            return _clip(value, 160)
    return artifact.type.value


def _artifact_comparison_payload(artifact: Artifact) -> dict[str, Any]:
    digests: dict[str, str] = {}
    for key in sorted(str(item) for item in artifact.payload):
        if key in _VOLATILE_PAYLOAD_KEYS or len(digests) >= MAX_RECORD_FIELDS:
            continue
        digests[key] = stable_digest(artifact.payload.get(key))
    return {
        "field_digests": digests,
        "warnings": [normalize_text(warning) for warning in artifact.warnings],
        "parent_count": len(artifact.parents),
    }


def _safe_trace_summary(summary: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in _TRACE_SUMMARY_KEYS:
        value = summary.get(key)
        if isinstance(value, str | int | float | bool) or value is None:
            result[key] = value
    return result


def _changed_fields(left: Any | None, right: Any | None) -> list[str]:
    if left is None or right is None:
        return []
    left_payload = getattr(left, "comparison_payload", {})
    right_payload = getattr(right, "comparison_payload", {})
    keys = sorted(set(left_payload) | set(right_payload))
    changed: list[str] = []
    for key in keys:
        if canonical_json(left_payload.get(key)) != canonical_json(right_payload.get(key)):
            if key == "field_digests":
                before = left_payload.get(key) or {}
                after = right_payload.get(key) or {}
                changed.extend(
                    f"payload.{field}"
                    for field in sorted(set(before) | set(after))
                    if before.get(field) != after.get(field)
                )
            else:
                changed.append(key)
    return changed[:MAX_RECORD_FIELDS]


def _counts(items: list[CompareScopeItem]) -> CompareScopeCounts:
    values = Counter(item.change for item in items)
    return CompareScopeCounts(
        added=values["added"],
        removed=values["removed"],
        changed=values["changed"],
        same=values["same"],
        unavailable=values["unavailable"],
    )


def _side_state(
    detail: SessionDetail,
    prepared: list[_Prepared],
    scope: CompareScopeName,
) -> CompareScopeSideState:
    if prepared:
        return CompareScopeSideState(state="value")
    if detail.status in {"completed", "complete"}:
        return CompareScopeSideState(
            state="missing",
            reason=f"The completed session produced no {scope} records.",
        )
    return CompareScopeSideState(
        state="unavailable",
        reason=f"Session status is {detail.status}; absence cannot be treated as zero.",
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


def _visible_family_ids(family: SessionResultFamily) -> list[str]:
    return [
        session_id for session_id in family.session_ids if INTERNAL_SESSION_MARKER not in session_id
    ]


def _with_occurrence(base: str, occurrences: Counter[str]) -> str:
    index = occurrences[base]
    occurrences[base] += 1
    return f"{base}|occurrence:{index}"


def _field(
    key: str,
    label: str,
    value: object,
    value_kind: Literal["text", "number", "status", "list", "code"] = "text",
) -> CompareScopeField:
    return CompareScopeField(
        key=key,
        label=label,
        value=_display(value),
        value_kind=value_kind,
    )


def _display(value: object) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list | tuple | set):
        return _clip(", ".join(str(item) for item in list(value)[:20])) or "—"
    return _clip(str(value))


def _clip(value: str, limit: int = MAX_FIELD_CHARS) -> str:
    normalized = " ".join(value.split())
    return normalized if len(normalized) <= limit else f"{normalized[: limit - 1]}…"


def _markdown_sections(markdown: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, list[str]]] = []
    current_title = "Report"
    current_lines: list[str] = []
    for line in markdown.splitlines():
        match = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if match:
            if current_lines or sections:
                sections.append((current_title, current_lines))
            current_title = match.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines or not sections:
        sections.append((current_title, current_lines))
    return [(title, "\n".join(lines).strip()) for title, lines in sections]


def _plain_markdown(markdown: str) -> str:
    return re.sub(r"[`*_>#\[\]()]", " ", markdown)


def _cursor_fingerprint(
    *,
    scope: CompareScopeName,
    left: str,
    right: str,
    filter_mode: CompareFilter,
) -> str:
    return stable_digest((scope, left, right, filter_mode), length=16)


def _encode_cursor(
    offset: int,
    *,
    scope: CompareScopeName,
    left: str,
    right: str,
    filter_mode: CompareFilter,
) -> str:
    payload = json.dumps(
        {
            "offset": offset,
            "fingerprint": _cursor_fingerprint(
                scope=scope,
                left=left,
                right=right,
                filter_mode=filter_mode,
            ),
        },
        separators=(",", ":"),
    ).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


def _decode_cursor(
    cursor: str | None,
    *,
    scope: CompareScopeName,
    left: str,
    right: str,
    filter_mode: CompareFilter,
) -> int:
    if not cursor:
        return 0
    try:
        padding = "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(cursor + padding))
        offset = int(payload["offset"])
        fingerprint = str(payload["fingerprint"])
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        raise InvalidCursorError() from None
    expected = _cursor_fingerprint(
        scope=scope,
        left=left,
        right=right,
        filter_mode=filter_mode,
    )
    if offset < 0 or fingerprint != expected:
        raise InvalidCursorError()
    return offset
