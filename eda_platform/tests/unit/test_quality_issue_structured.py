"""QualityIssue structured numeric fields (analysis-v3 §11.3).

R3: every quality.py generation point mirrors its message figures into
metric_value/affected_count. R1: structured artifacts resolve through the
validator's QUALITY_ISSUE_SET dispatch. R2: legacy artifacts (fields None)
resolve nothing, pinning frozen-corpus behavior.
"""

from __future__ import annotations

from typing import Literal

from eda_platform.core.ids import make_artifact_id
from eda_platform.schemas.artifacts import (
    Artifact,
    ArtifactType,
    ColumnProfile,
    DatasetProfile,
    EvidenceRef,
    QualityIssue,
    QualityIssueSet,
)
from eda_platform.schemas.reports import ReportClaim
from eda_platform.tools import report_validator as rv
from eda_platform.tools.evidence import build_evidence_pack
from eda_platform.tools.quality import scan_quality

_SemanticType = Literal[
    "numeric", "categorical", "datetime", "id", "boolean", "text", "unknown"
]


def _column(
    name: str,
    *,
    semantic_type: _SemanticType = "numeric",
    missing_percent: float = 0.0,
    unique_count: int = 50,
    unique_percent: float = 50.0,
    outlier_count: int = 0,
    warnings: list[str] | None = None,
) -> ColumnProfile:
    return ColumnProfile(
        name=name,
        dtype="object",
        semantic_type=semantic_type,
        missing_count=int(missing_percent),
        missing_percent=missing_percent,
        unique_count=unique_count,
        unique_percent=unique_percent,
        outlier_count=outlier_count,
        warnings=warnings or [],
    )


def _dirty_profile() -> DatasetProfile:
    columns = [
        _column("partial", missing_percent=45.0),
        _column("full", missing_percent=100.0),
        _column("amount", outlier_count=3),
        _column("constant", unique_count=1, unique_percent=1.0),
        _column("cust", semantic_type="id", unique_percent=100.0),
        _column("tag", semantic_type="categorical", unique_percent=87.5),
        _column("when", semantic_type="datetime", warnings=["date_parse_failure"]),
        _column("mixed", semantic_type="text", warnings=["mixed_type_string"]),
    ]
    return DatasetProfile(
        dataset_id="ds_structured",
        name="Structured",
        rows=100,
        columns=len(columns),
        column_names=[column.name for column in columns],
        dtypes={column.name: "object" for column in columns},
        missing_values={"partial": 45, "full": 100},
        missing_percent={"partial": 45.0, "full": 100.0},
        numeric_columns=["partial", "amount"],
        categorical_columns=["tag"],
        duplicate_rows=7,
        columns_detail=columns,
    )


def _profile_artifact(profile: DatasetProfile) -> Artifact:
    payload = profile.model_dump(mode="json")
    return Artifact(
        id=make_artifact_id("profile", payload),
        type=ArtifactType.DATASET_PROFILE,
        project_id="project_demo",
        session_id="run_demo",
        payload=payload,
    )


def _scan(profile: DatasetProfile) -> Artifact:
    return scan_quality(
        _profile_artifact(profile), project_id="project_demo", session_id="run_demo"
    )


def _issues_by_code(artifact: Artifact) -> dict[tuple[str, str | None], QualityIssue]:
    issue_set = QualityIssueSet.model_validate(artifact.payload)
    return {(issue.code, issue.column): issue for issue in issue_set.issues}


# --- R3: field mapping at every generation point ---


def test_high_missing_mirrors_percent() -> None:
    issue = _issues_by_code(_scan(_dirty_profile()))[("high_missing", "partial")]
    assert issue.metric_value == 45.0
    assert issue.metric_unit == "percent"
    assert issue.affected_count is None
    assert "45.00%" in issue.message


def test_duplicate_rows_mirrors_count() -> None:
    issue = _issues_by_code(_scan(_dirty_profile()))[("duplicate_rows", None)]
    assert issue.affected_count == 7
    assert issue.metric_value is None
    assert "7" in issue.message


def test_empty_column_mirrors_percent() -> None:
    issue = _issues_by_code(_scan(_dirty_profile()))[("empty_column", "full")]
    assert issue.metric_value == 100.0
    assert issue.metric_unit == "percent"
    assert issue.affected_count is None


def test_outlier_detected_mirrors_count() -> None:
    issue = _issues_by_code(_scan(_dirty_profile()))[("outlier_detected", "amount")]
    assert issue.affected_count == 3
    assert issue.metric_value is None


def test_high_cardinality_mirrors_percent() -> None:
    issue = _issues_by_code(_scan(_dirty_profile()))[
        ("high_cardinality_category", "tag")
    ]
    assert issue.metric_value == 87.5
    assert issue.metric_unit == "percent"
    assert issue.affected_count is None


def test_non_numeric_issues_keep_defaults() -> None:
    issues = _issues_by_code(_scan(_dirty_profile()))
    for key in [
        ("constant_column", "constant"),
        ("likely_id_column", "cust"),
        ("date_parse_failure", "when"),
        ("mixed_type_string", "mixed"),
    ]:
        issue = issues[key]
        assert issue.metric_value is None
        assert issue.metric_unit == "raw"
        assert issue.affected_count is None


def test_no_issue_placeholder_keeps_defaults() -> None:
    clean = DatasetProfile(
        dataset_id="ds_clean",
        name="Clean",
        rows=10,
        columns=1,
        column_names=["ok"],
        dtypes={"ok": "int64"},
        missing_values={"ok": 0},
        missing_percent={"ok": 0.0},
        numeric_columns=["ok"],
        categorical_columns=[],
        columns_detail=[_column("ok", unique_count=10, unique_percent=100.0)],
    )
    issue = _issues_by_code(_scan(clean))[("no_high_missing", None)]
    assert issue.metric_value is None
    assert issue.affected_count is None


def test_every_message_number_has_a_structured_mirror() -> None:
    issue_set = QualityIssueSet.model_validate(_scan(_dirty_profile()).payload)
    for issue in issue_set.issues:
        for value, is_percent in rv.extract_numbers(issue.message):
            if is_percent:
                assert issue.metric_unit == "percent"
                assert issue.metric_value == value, issue.code
            else:
                assert issue.affected_count == value, issue.code


# --- R1: structured artifacts resolve through the validator dispatch ---


def _structured_pack():
    profile = _dirty_profile()
    profile_artifact = _profile_artifact(profile)
    quality = scan_quality(
        profile_artifact, project_id="project_demo", session_id="run_demo"
    )
    return build_evidence_pack([profile_artifact, quality]), quality


def test_structured_missing_rate_claim_verifies() -> None:
    pack, quality = _structured_pack()
    claim = ReportClaim(
        text="Column partial has 45.00% missing values.",
        evidence=[
            EvidenceRef(
                kind="artifact",
                artifact_id=quality.id,
                locator="high_missing:partial",
            )
        ],
    )
    statuses = rv._numeric_token_statuses(
        claim, evidence_pack=pack, numeric_tolerance=0.01, sql_results={}
    )
    assert [(s.number, s.is_percent, s.status) for s in statuses] == [
        (45.0, True, "number_verified")
    ]


def test_structured_affected_count_is_exact() -> None:
    pack, quality = _structured_pack()
    evidence = EvidenceRef(
        kind="artifact", artifact_id=quality.id, locator="duplicate_rows:"
    )
    values = rv._resolve_evidence_numbers(evidence, pack, {})
    assert values == [(7.0, "raw", "exact", False)]


def test_structured_count_rejects_off_by_one() -> None:
    pack, quality = _structured_pack()
    claim = ReportClaim(
        text="Dataset contains 8 duplicate rows.",
        evidence=[
            EvidenceRef(
                kind="artifact", artifact_id=quality.id, locator="duplicate_rows:"
            )
        ],
    )
    statuses = rv._numeric_token_statuses(
        claim, evidence_pack=pack, numeric_tolerance=0.01, sql_results={}
    )
    assert [(s.number, s.status) for s in statuses] == [(8.0, "failed")]


def test_structured_issue_count_claim_verifies_on_whole_set() -> None:
    # F3 hand-off: the deterministic fallback's issue_count claim cites
    # locator="issues"; on structured artifacts the set cardinality resolves,
    # so the claim converts to number_verified without touching the exemption.
    pack, quality = _structured_pack()
    issue_count = len(QualityIssueSet.model_validate(quality.payload).issues)
    claim = ReportClaim(
        text=f"Structured quality scan found {issue_count} issues.",
        evidence=[
            EvidenceRef(kind="artifact", artifact_id=quality.id, locator="issues")
        ],
    )
    statuses = rv._numeric_token_statuses(
        claim, evidence_pack=pack, numeric_tolerance=0.01, sql_results={}
    )
    assert [(s.number, s.status) for s in statuses] == [
        (float(issue_count), "number_verified")
    ]


def test_unknown_locator_resolves_nothing() -> None:
    pack, quality = _structured_pack()
    evidence = EvidenceRef(
        kind="artifact", artifact_id=quality.id, locator="no_such_code:partial"
    )
    assert rv._resolve_evidence_numbers(evidence, pack, {}) == []


# --- Produced locator grammar + whole-set pool tightening ---


_STRUCTURED_PROBE_PAYLOAD = {
    "dataset_id": "ds_probe",
    "issues": [
        {
            "severity": "warn",
            "code": "high_missing",
            "column": "score",
            "message": "Column score has 88.34% missing values.",
            "recommendation": "Review missingness before use.",
            "metric_value": 88.34,
            "metric_unit": "percent",
        },
        {
            "severity": "warn",
            "code": "duplicate_rows",
            "column": None,
            "message": "Dataset contains 42 duplicate rows.",
            "recommendation": "Review duplicate records.",
            "affected_count": 42,
        },
    ],
}


def _probe_pack():
    artifact = Artifact(
        id="qual_probe",
        type=ArtifactType.QUALITY_ISSUE_SET,
        project_id="project_demo",
        session_id="run_demo",
        payload=_STRUCTURED_PROBE_PAYLOAD,
    )
    return build_evidence_pack([artifact])


def test_produced_column_locator_grammar_resolves() -> None:
    # agents/reporting.py emits "quality_issue:{code}:{column}"; the bare
    # "{code}:{column}" form is the same grammar without the prefix.
    pack = _probe_pack()
    for locator in ["quality_issue:high_missing:score", "high_missing:score"]:
        evidence = EvidenceRef(
            kind="artifact", artifact_id="qual_probe", locator=locator
        )
        assert rv._resolve_evidence_numbers(evidence, pack, {}) == [
            (88.34, "percent", "rounded", False)
        ], locator


def test_produced_dataset_level_locator_grammar_resolves() -> None:
    pack = _probe_pack()
    for locator in ["quality_issue:duplicate_rows:", "duplicate_rows:"]:
        evidence = EvidenceRef(
            kind="artifact", artifact_id="qual_probe", locator=locator
        )
        assert rv._resolve_evidence_numbers(evidence, pack, {}) == [
            (42.0, "raw", "exact", False)
        ], locator


def test_retired_column_first_form_resolves_nothing() -> None:
    # "{column}:{code}" was never produced by the platform; keeping it parsable
    # would let a swapped-field locator alias a different issue.
    pack = _probe_pack()
    evidence = EvidenceRef(
        kind="artifact", artifact_id="qual_probe", locator="score:high_missing"
    )
    assert rv._resolve_evidence_numbers(evidence, pack, {}) == []


def test_whole_set_locator_resolves_cardinality_only() -> None:
    pack = _probe_pack()
    for locator in ["issues", ""]:
        evidence = EvidenceRef(
            kind="artifact", artifact_id="qual_probe", locator=locator
        )
        assert rv._resolve_evidence_numbers(evidence, pack, {}) == [
            (2.0, "raw", "exact", False)
        ], locator


def test_whole_set_locator_cannot_wash_cross_column_percent() -> None:
    # A percent figure attributed to the wrong column must not verify against
    # the pooled metrics of every issue in the set: the whole-set pool carries
    # only the cardinality (raw), so the percent token has no percent pool.
    pack = _probe_pack()
    claim = ReportClaim(
        text="Column other_col has 88.34% missing values.",
        evidence=[
            EvidenceRef(kind="artifact", artifact_id="qual_probe", locator="issues")
        ],
    )
    statuses, details = rv._numeric_gate_outcome(
        claim, evidence_pack=pack, numeric_tolerance=0.01, sql_results={}
    )
    assert [(s.number, s.is_percent, s.status) for s in statuses] == [
        (88.34, True, "failed")
    ]
    assert [detail.reason for detail in details] == ["no_evidence_values"]


# --- R2: legacy artifacts (fields None) pin frozen-corpus behavior ---


_LEGACY_PAYLOAD = {
    "dataset_id": "ds_legacy",
    "issues": [
        {
            "severity": "warn",
            "code": "high_missing",
            "column": "review_comment",
            "message": "Column review_comment has 88.34% missing values.",
            "recommendation": "Review missingness before use.",
        },
        {
            "severity": "warn",
            "code": "duplicate_rows",
            "column": None,
            "message": "Dataset contains 261831 duplicate rows.",
            "recommendation": "Review duplicate records.",
        },
    ],
}


def _legacy_pack():
    artifact = Artifact(
        id="qual_legacy",
        type=ArtifactType.QUALITY_ISSUE_SET,
        project_id="project_demo",
        session_id="run_demo",
        payload=_LEGACY_PAYLOAD,
    )
    return build_evidence_pack([artifact])


def test_legacy_schema_defaults_are_none() -> None:
    issue_set = QualityIssueSet.model_validate(_LEGACY_PAYLOAD)
    for issue in issue_set.issues:
        assert issue.metric_value is None
        assert issue.metric_unit == "raw"
        assert issue.affected_count is None


def test_legacy_issue_resolves_nothing() -> None:
    pack = _legacy_pack()
    for locator in [
        "quality_issue:high_missing:review_comment",
        "high_missing:review_comment",
        "duplicate_rows:",
        "issues",
        "",
    ]:
        evidence = EvidenceRef(
            kind="artifact", artifact_id="qual_legacy", locator=locator
        )
        assert rv._resolve_evidence_numbers(evidence, pack, {}) == []


def test_legacy_prose_number_stays_unverified() -> None:
    pack = _legacy_pack()
    claim = ReportClaim(
        text="Column review_comment has 88.34% missing values.",
        evidence=[
            EvidenceRef(
                kind="artifact",
                artifact_id="qual_legacy",
                locator="review_comment:high_missing",
            )
        ],
    )
    statuses = rv._numeric_token_statuses(
        claim, evidence_pack=pack, numeric_tolerance=0.01, sql_results={}
    )
    assert [(s.number, s.status) for s in statuses] == [(88.34, "unverified")]
