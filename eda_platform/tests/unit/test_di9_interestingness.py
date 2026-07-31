"""H9-B interestingness scoring + finding-cluster deduplication (zero LLM).

Covers the sprint-9 red lines:

- Every score component comes from structured fields; sentence text is never
  parsed for numbers.
- Degenerate/identity patterns and unvalidated exploratory findings are
  penalised, never deleted.
- Deduplication labels supporting findings and selects representatives; no
  finding is removed anywhere.
- ``FindingScore`` stays backward compatible: legacy payloads validate and
  legacy scores serialize byte-identically (no ``interestingness`` key).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eda_platform.core.store import ArtifactStore
from eda_platform.core.trace import FINDINGS_DEDUPLICATED
from eda_platform.drivers.synthesis_orchestrator import create_synthesis_brief
from eda_platform.schemas.anomaly import AnomalyScreenResult
from eda_platform.schemas.artifacts import Artifact, ArtifactType, EvidenceRef
from eda_platform.schemas.investigations import InvestigationRecord, ValidatedFinding
from eda_platform.schemas.model_card import ModelCard
from eda_platform.schemas.questions import FindingScore, QuestionFinding
from eda_platform.schemas.stats import StatTestResult
from eda_platform.schemas.synthesis import SynthesisBrief
from eda_platform.tools.interestingness import (
    DEFAULT_COVERAGE,
    DEFAULT_DEVIATION,
    DEGENERATE_NONTRIVIALITY,
    EXPLORATORY_PENALTY,
    DedupFinding,
    deduplicate_findings,
    finding_interestingness,
    interestingness,
)
from eda_platform.tools.method_findings import (
    anomaly_findings,
    model_findings,
    stat_findings,
)


# --------------------------------------------------------------------------- #
# Component behaviour: positive and negative cases for all three components.
# --------------------------------------------------------------------------- #
def test_deviation_component_moves_the_value() -> None:
    strong = interestingness(deviation=0.9)
    weak = interestingness(deviation=0.1)
    absent = interestingness()

    assert strong.deviation == 0.9
    assert weak.deviation == 0.1
    assert strong.value > weak.value
    assert absent.deviation == DEFAULT_DEVIATION


def test_coverage_component_uses_row_share_and_defaults_when_unknown() -> None:
    broad = interestingness(rows_involved=90, dataset_row_count=100)
    narrow = interestingness(rows_involved=5, dataset_row_count=100)
    unknown = interestingness(rows_involved=None, dataset_row_count=None)

    assert broad.coverage == 0.9
    assert narrow.coverage == 0.05
    assert broad.value > narrow.value
    assert unknown.coverage == DEFAULT_COVERAGE


def test_nontriviality_penalizes_degenerate_patterns_without_zeroing() -> None:
    honest = interestingness(deviation=0.8)
    degenerate = interestingness(deviation=0.8, degenerate=True)

    assert honest.nontriviality == 1.0
    assert degenerate.nontriviality == DEGENERATE_NONTRIVIALITY
    assert degenerate.value < honest.value
    # Penalised, not deleted: the value stays observable and positive.
    assert degenerate.value > 0.0


def test_exploratory_without_stat_validation_is_penalized() -> None:
    validated = interestingness(deviation=0.8, exploratory=True, stat_validated=True)
    unvalidated = interestingness(deviation=0.8, exploratory=True, stat_validated=False)
    non_exploratory = interestingness(deviation=0.8, exploratory=False, stat_validated=False)

    assert unvalidated.value == pytest.approx(validated.value * EXPLORATORY_PENALTY, abs=1e-5)
    assert non_exploratory.value == validated.value


# --------------------------------------------------------------------------- #
# Finding-level extraction reads structured evidence only, never the text.
# --------------------------------------------------------------------------- #
def _stat_evidence(effect_size: float, p_value: float = 0.001) -> list[EvidenceRef]:
    return [
        EvidenceRef(kind="stat", artifact_id="artifact", locator="statistic", value=8.0),
        EvidenceRef(kind="stat", artifact_id="artifact", locator="p_value", value=p_value),
        EvidenceRef(kind="stat", artifact_id="artifact", locator="effect_size", value=effect_size),
        EvidenceRef(kind="stat", artifact_id="artifact", locator="sample_size", value=100),
    ]


def test_finding_interestingness_ignores_numbers_in_the_sentence() -> None:
    quiet = QuestionFinding(text="An observed difference.", evidence=_stat_evidence(0.4))
    loud = QuestionFinding(
        text="An observed difference of 999999 units across 888888 rows.",
        evidence=_stat_evidence(0.4),
    )

    assert finding_interestingness(quiet) == finding_interestingness(loud)


def test_finding_interestingness_penalizes_exploratory_without_stat_backing() -> None:
    evidence = [
        EvidenceRef(kind="sql", artifact_id="artifact", locator="rows[0].total", value=42)
    ]
    exploratory = QuestionFinding(text="A pattern.", evidence=evidence, exploratory=True)
    validated = QuestionFinding(text="A pattern.", evidence=evidence, exploratory=False)

    penalized = finding_interestingness(exploratory)
    baseline = finding_interestingness(validated)
    assert penalized.value == pytest.approx(baseline.value * EXPLORATORY_PENALTY, abs=1e-5)


def test_finding_interestingness_with_stat_backing_is_not_penalized() -> None:
    exploratory = QuestionFinding(
        text="A pattern.", evidence=_stat_evidence(0.4), exploratory=True
    )
    validated = QuestionFinding(
        text="A pattern.", evidence=_stat_evidence(0.4), exploratory=False
    )
    assert finding_interestingness(exploratory) == finding_interestingness(validated)


# --------------------------------------------------------------------------- #
# Reducer wiring: interestingness multiplies into final only when computed.
# --------------------------------------------------------------------------- #
def _stat_result(
    group_column: str = "region", value_column: str = "revenue"
) -> StatTestResult:
    return StatTestResult(
        dataset_id="sales.csv",
        test_type="one_way_anova",
        group_column=group_column,
        value_column=value_column,
        statistic=8.1,
        p_value=0.002,
        effect_size=0.2,
        sample_size=120,
    )


def test_stat_reducer_without_row_count_keeps_legacy_score() -> None:
    finding = stat_findings(_stat_result(), "artifact")[0]
    assert finding.score is not None
    assert finding.score.interestingness is None
    assert finding.score.final == 0.998  # impact x significance, DI8-D contract


def test_stat_reducer_multiplies_interestingness_into_final() -> None:
    finding = stat_findings(_stat_result(), "artifact", dataset_row_count=240)[0]
    score = finding.score
    assert score is not None
    assert score.interestingness is not None
    assert 0.0 < score.interestingness < 1.0
    assert score.final == pytest.approx(
        score.impact * score.significance * score.interestingness, abs=1e-5
    )


def test_stat_reducer_penalizes_identity_column_pair() -> None:
    honest = stat_findings(_stat_result(), "artifact", dataset_row_count=240)[0]
    identity = stat_findings(
        _stat_result(group_column="revenue", value_column="revenue"),
        "artifact",
        dataset_row_count=240,
    )[0]
    assert honest.score is not None and identity.score is not None
    assert identity.score.interestingness is not None
    assert honest.score.interestingness is not None
    assert identity.score.interestingness < honest.score.interestingness
    assert identity.score.final < honest.score.final


def _model_card(feature_columns: list[str]) -> ModelCard:
    return ModelCard(
        dataset_id="orders",
        task_type="regression",
        target_column="amount",
        feature_columns=feature_columns,
        split_strategy="random",
        train_rows=80,
        test_rows=20,
        model_type="baseline",
        metrics={"r2": 0.6},
    )


def test_model_reducer_penalizes_target_predicting_itself() -> None:
    honest = model_findings(_model_card(["quantity"]), "artifact", dataset_row_count=200)[0]
    identity = model_findings(
        _model_card(["quantity", "amount"]), "artifact", dataset_row_count=200
    )[0]
    assert honest.score is not None and identity.score is not None
    assert honest.score.interestingness is not None
    assert identity.score.interestingness is not None
    assert identity.score.interestingness < honest.score.interestingness


def _anomaly_result(outlier_count: int) -> AnomalyScreenResult:
    return AnomalyScreenResult(
        dataset_name="orders",
        column="amount",
        method="robust_zscore",
        threshold=3.5,
        total_rows=100,
        non_null_rows=80,
        outlier_count=outlier_count,
        outlier_percent=outlier_count / 80 * 100,
        median=10,
        mad=2,
        q1=8,
        q3=12,
    )


def test_anomaly_reducer_penalizes_count_equals_rows_identity() -> None:
    partial = anomaly_findings(_anomaly_result(20), "artifact")[0]
    everything = anomaly_findings(_anomaly_result(80), "artifact")[0]
    assert partial.score is not None and everything.score is not None
    # The anomaly result carries its own coverage anchor -> always computed.
    assert partial.score.interestingness is not None
    assert everything.score.interestingness is not None
    assert everything.score.interestingness < partial.score.interestingness


# --------------------------------------------------------------------------- #
# Cluster deduplication: key behaviour, representative choice, no deletion.
# --------------------------------------------------------------------------- #
def _dedup_item(
    ref: str,
    *,
    effect_size: float,
    final: float,
    columns: tuple[str, ...] = ("region", "revenue"),
) -> DedupFinding:
    return DedupFinding(
        ref=ref,
        finding=QuestionFinding(
            text=f"Finding {ref}.",
            evidence=_stat_evidence(effect_size),
            score=FindingScore(impact=1.0, significance=final, final=final),
        ),
        columns=columns,
    )


def test_dedup_merges_same_columns_same_direction_same_magnitude() -> None:
    high = _dedup_item("vf_high#0", effect_size=0.35, final=0.9)
    low = _dedup_item("vf_low#0", effect_size=0.31, final=0.4)

    clusters = deduplicate_findings([low, high])

    assert len(clusters) == 1
    cluster = clusters[0]
    assert cluster.representative.ref == "vf_high#0"
    assert [item.ref for item in cluster.supporting] == ["vf_low#0"]
    # Labelled in place, never deleted.
    assert high.finding.dedup_role == "representative"
    assert low.finding.dedup_role == "supporting"
    assert high.finding.dedup_cluster_key == low.finding.dedup_cluster_key
    assert {item.ref for item in cluster.members} == {"vf_high#0", "vf_low#0"}


def test_dedup_keeps_opposite_directions_apart() -> None:
    up = _dedup_item("vf_up#0", effect_size=0.35, final=0.9)
    down = _dedup_item("vf_down#0", effect_size=-0.35, final=0.4)

    clusters = deduplicate_findings([up, down])

    assert len(clusters) == 2
    assert up.finding.dedup_role == "representative"
    assert down.finding.dedup_role == "representative"


def test_dedup_keeps_different_magnitudes_and_columns_apart() -> None:
    base = _dedup_item("vf_a#0", effect_size=0.35, final=0.9)
    smaller = _dedup_item("vf_b#0", effect_size=0.035, final=0.9)
    other_pair = _dedup_item(
        "vf_c#0", effect_size=0.35, final=0.9, columns=("region", "freight")
    )

    clusters = deduplicate_findings([base, smaller, other_pair])

    assert len(clusters) == 3
    assert all(cluster.supporting == [] for cluster in clusters)


def test_dedup_representative_uses_interestingness_times_final() -> None:
    # Same cluster; the higher final wins because interestingness is equal.
    winner = _dedup_item("vf_w#0", effect_size=0.35, final=0.8)
    loser = _dedup_item("vf_l#0", effect_size=0.35, final=0.7)

    clusters = deduplicate_findings([loser, winner])

    assert clusters[0].representative.ref == "vf_w#0"


# --------------------------------------------------------------------------- #
# FindingScore backward compatibility.
# --------------------------------------------------------------------------- #
def test_finding_score_defaults_keep_legacy_shape() -> None:
    legacy = FindingScore(impact=1.0, significance=0.9, final=0.9)
    assert legacy.interestingness is None
    # Serialized payloads stay byte-compatible with pre-H9 artifacts.
    assert set(legacy.model_dump()) == {"impact", "significance", "final"}
    assert "interestingness" not in legacy.model_dump_json()


def test_finding_score_validates_legacy_payload_and_round_trips_new_one() -> None:
    legacy = FindingScore.model_validate(
        {"impact": 1.0, "significance": 0.9, "final": 0.9}
    )
    assert legacy.interestingness is None

    scored = FindingScore(impact=1.0, significance=0.9, interestingness=0.5, final=0.45)
    dumped = scored.model_dump()
    assert dumped["interestingness"] == 0.5
    assert FindingScore.model_validate(dumped) == scored


# --------------------------------------------------------------------------- #
# Synthesis selection quotes representatives only and discloses the merge.
# --------------------------------------------------------------------------- #
def _validated_finding(
    *,
    suffix: str,
    question: str,
    text: str,
    effect_size: float,
    final: float,
    source_artifact_id: str,
) -> ValidatedFinding:
    return ValidatedFinding(
        finding_id=f"finding_{suffix}",
        investigation_id=f"inv_{suffix}",
        question_id=f"q_{suffix}",
        question=question,
        claim_class="observed",
        findings=[
            QuestionFinding(
                text=text,
                evidence=_stat_evidence(effect_size),
                score=FindingScore(impact=1.0, significance=final, final=final),
            )
        ],
        evidence_support="high",
        analytical_reliability="high",
        decision_readiness="medium",
        report_eligible=True,
        report_readiness="eligible",
        report_readiness_reason="Validated with disclosed data conditions.",
        source_artifact_ids=[source_artifact_id],
    )


def _seed_near_duplicate_findings(tmp_path: Path) -> tuple[Path, str, list[str]]:
    workspace = tmp_path / "workspace"
    store = ArtifactStore(workspace)
    project_id = "project_di9"
    store.ensure_project(project_id, "DI9")
    store.start_session(project_id, "finding_run")

    specs = [
        (
            "rep",
            "How does revenue vary by region?",
            "The observed revenue difference across regions has eta squared 0.35.",
            0.35,
            0.9,
        ),
        (
            "sup",
            "Does revenue differ between regions?",
            "The observed revenue gap between regions has eta squared 0.31.",
            0.31,
            0.4,
        ),
    ]
    finding_ids: list[str] = []
    for suffix, question, text, effect_size, final in specs:
        stat_artifact = Artifact(
            id=f"stat_{suffix}",
            type=ArtifactType.STAT_TEST_RESULT,
            project_id=project_id,
            session_id="finding_run",
            payload={"group_column": "region", "value_column": "revenue"},
        )
        store.save_artifact(stat_artifact)
        finding = _validated_finding(
            suffix=suffix,
            question=question,
            text=text,
            effect_size=effect_size,
            final=final,
            source_artifact_id=stat_artifact.id,
        )
        finding_artifact = Artifact(
            id=f"vf_{suffix}",
            type=ArtifactType.VALIDATED_FINDING,
            project_id=project_id,
            session_id="finding_run",
            payload=finding.model_dump(mode="json"),
        )
        store.save_artifact(finding_artifact)
        record = InvestigationRecord(
            record_id=f"rec_{suffix}",
            investigation_id=f"inv_{suffix}",
            question_id=f"q_{suffix}",
            status="validated",
            reason_code="validated",
            reason="Deterministic gates passed.",
            next_action="None required.",
            finding_artifact_id=finding_artifact.id,
        )
        store.save_artifact(
            Artifact(
                id=f"rec_{suffix}",
                type=ArtifactType.INVESTIGATION_RECORD,
                project_id=project_id,
                session_id="finding_run",
                payload=record.model_dump(mode="json"),
            )
        )
        finding_ids.append(finding_artifact.id)
    return workspace, project_id, finding_ids


def test_synthesis_quotes_cluster_representatives_and_discloses_merges(
    tmp_path: Path,
) -> None:
    workspace, project_id, finding_ids = _seed_near_duplicate_findings(tmp_path)
    # The supporting finding is selected FIRST; representation must not depend
    # on selection order, and cross-question findings must cluster together.
    synthesis = create_synthesis_brief(
        project_id=project_id,
        finding_artifact_ids=[finding_ids[1], finding_ids[0]],
        workspace=workspace,
        session_id="synthesis_run",
    )
    brief = SynthesisBrief.model_validate(synthesis.artifact.payload)

    # Nothing is deleted: both findings stay selected and referenced.
    assert set(brief.selected_finding_artifact_ids) == set(finding_ids)
    evidence_beat = next(
        beat for beat in brief.storyline if beat.title == "Validated evidence"
    )
    # Only the cluster representative is quoted as evidence.
    assert "eta squared 0.35" in evidence_beat.body
    assert "eta squared 0.31" not in evidence_beat.body
    assert brief.headline == (
        "The observed revenue difference across regions has eta squared 0.35."
    )
    # The merge is disclosed in the limitations.
    assert any("merged into cluster representatives" in item for item in brief.limitations)

    store = ArtifactStore(workspace)
    events = store.list_trace_events(project_id=project_id, session_id="synthesis_run")
    dedup_events = [
        event for event in events if event.event_type == FINDINGS_DEDUPLICATED
    ]
    assert len(dedup_events) == 1
    assert dedup_events[0].summary["clusters"] == 1
    assert dedup_events[0].summary["merged_supporting"] == 1


def test_synthesis_without_duplicates_adds_no_merge_disclosure(tmp_path: Path) -> None:
    workspace, project_id, finding_ids = _seed_near_duplicate_findings(tmp_path)
    synthesis = create_synthesis_brief(
        project_id=project_id,
        finding_artifact_ids=[finding_ids[0]],
        workspace=workspace,
        session_id="synthesis_solo",
    )
    brief = SynthesisBrief.model_validate(synthesis.artifact.payload)
    assert not any(
        "merged into cluster representatives" in item for item in brief.limitations
    )
    evidence_beat = next(
        beat for beat in brief.storyline if beat.title == "Validated evidence"
    )
    assert "eta squared 0.35" in evidence_beat.body
