from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from semantic_test_helpers import add_verified_relation, load_seeds, save_seeds

from eda_platform.core.semantic import (
    EntityNote,
    SemanticSeeds,
    VerifiedRelation,
)
from eda_platform.schemas.artifacts import (
    Artifact,
    ArtifactType,
    DatasetProfile,
    EvidenceRef,
    QualityIssue,
    QualityIssueSet,
)
from eda_platform.schemas.reports import ReportBundle, ReportClaim, ReportSection
from eda_platform.tools.chart_specs import create_chart_specs
from eda_platform.tools.exporter import narrative_markdown, report_bundle_to_markdown
from eda_platform.tools.loader import load_csv
from eda_platform.tools.profiler import profile_dataset
from eda_platform.tools.quality import scan_quality

# --- semantic seeds -------------------------------------------------------


def test_load_seeds_missing_file_returns_empty(tmp_path: Path) -> None:
    seeds = load_seeds(tmp_path)

    assert seeds.version == 1
    assert seeds.verified_relations == []
    assert seeds.entity_notes == []
    assert not (tmp_path / "semantic" / "seeds.json").exists()


def test_save_and_load_seeds_roundtrip(tmp_path: Path) -> None:
    seeds = SemanticSeeds(
        verified_relations=[
            VerifiedRelation(
                left="players.team_id",
                right="teams.team_id",
                cardinality="many_to_one",
                confirmed_at=datetime(2026, 7, 3, tzinfo=UTC),
                source_session_id="run_a",
            )
        ],
        entity_notes=[EntityNote(name="team", note="A club roster.")],
    )

    path = save_seeds(tmp_path, seeds)
    assert path == tmp_path / "semantic" / "seeds.json"

    loaded = load_seeds(tmp_path)
    assert loaded.verified_relations[0].right == "teams.team_id"
    assert loaded.verified_relations[0].source_session_id == "run_a"
    assert loaded.entity_notes[0].note == "A club roster."


def test_add_verified_relation_is_idempotent_on_pair(tmp_path: Path) -> None:
    first = VerifiedRelation(
        left="players.team_id",
        right="teams.team_id",
        cardinality="many_to_one",
        confirmed_at=datetime(2026, 7, 3, tzinfo=UTC),
    )
    add_verified_relation(tmp_path, first)

    updated = VerifiedRelation(
        left="players.team_id",
        right="teams.team_id",
        cardinality="one_to_one",
        confirmed_at=datetime(2026, 7, 4, tzinfo=UTC),
        source_session_id="run_b",
    )
    seeds = add_verified_relation(tmp_path, updated)

    assert len(seeds.verified_relations) == 1
    relation = seeds.verified_relations[0]
    assert relation.cardinality == "one_to_one"
    assert relation.confirmed_at == datetime(2026, 7, 4, tzinfo=UTC)
    assert relation.source_session_id == "run_b"

    other = VerifiedRelation(
        left="matches.home_team_id",
        right="teams.team_id",
        cardinality="many_to_one",
    )
    seeds = add_verified_relation(tmp_path, other)
    assert len(seeds.verified_relations) == 2


# --- exporter Day-0 patches ----------------------------------------------


def _quality_and_chart_artifacts(tmp_path: Path) -> tuple[list, str]:
    csv_path = tmp_path / "teams.csv"
    # region is constant, notes is 100% empty -> constant_column + empty_column;
    # score has a high-missing rate -> high_missing high-risk column.
    csv_path.write_text(
        "team_id,region,score,notes\n"
        "1,East,10,\n"
        "2,East,,\n"
        "3,East,,\n"
        "4,East,,\n"
        "5,East,,\n",
        encoding="utf-8",
    )
    loaded = load_csv(csv_path, dataset_id="ds_teams")
    profile = profile_dataset(loaded, project_id="project_demo", session_id="run_demo")
    quality = scan_quality(profile, project_id="project_demo", session_id="run_demo")
    charts = create_chart_specs(
        loaded, profile, project_id="project_demo", session_id="run_demo"
    )
    return [profile, quality, *charts], quality.id


def _bundle_with_empty_sections() -> ReportBundle:
    bundle = ReportBundle.empty(project_id="project_demo", session_id="run_demo")
    for section in bundle.sections:
        section.body = "No validated conclusion is available for this section."
    return bundle


def test_limitations_section_synthesized_from_quality_artifact(tmp_path: Path) -> None:
    artifacts, _ = _quality_and_chart_artifacts(tmp_path)
    bundle = _bundle_with_empty_sections()

    markdown = report_bundle_to_markdown(bundle, artifacts=artifacts)
    limitations = _section_text(markdown, "## Limitations and Risks")

    assert "No validated conclusion is available" not in limitations
    # Each line reads as human prose, prefixed with the dataset name, and says
    # what the flag costs the analysis rather than naming the scanner code.
    assert "teams.csv: `notes` is entirely missing" in limitations
    assert "teams.csv: `region`, `score` hold one value in every row" in limitations
    assert "teams.csv: high-risk columns for analysis:" in limitations
    # One fact, one line: the grouped footer owns these codes, so the same
    # column must not also get a per-column sentence above it.
    assert limitations.count("`notes`") == 2  # empty-column footer + high-risk footer
    assert limitations.count("`region`") == 1
    # Raw evidence ids are kept out of the human-readable limitations prose (P0-3).
    assert "Evidence:" not in limitations


def test_appendix_section_renders_chart_inventory(tmp_path: Path) -> None:
    artifacts, _ = _quality_and_chart_artifacts(tmp_path)
    charts = [a for a in artifacts if a.type.value == "ChartSpec"]
    assert charts, "fixture must produce at least one chart"
    bundle = _bundle_with_empty_sections()

    markdown = report_bundle_to_markdown(bundle, artifacts=artifacts)
    appendix = _section_text(markdown, "## Appendix: Charts and Technical Summary")

    assert "No validated conclusion is available" not in appendix
    assert "Chart inventory:" in appendix
    for chart_artifact in charts:
        assert chart_artifact.id in appendix


def test_empty_sections_fall_back_without_artifacts() -> None:
    bundle = _bundle_with_empty_sections()

    markdown = report_bundle_to_markdown(bundle)
    limitations = _section_text(markdown, "## Limitations and Risks")

    assert "No validated conclusion is available for this section." in limitations


def test_claim_ledger_dedupes_evidence_and_narrative_drops_inline_ids(tmp_path: Path) -> None:
    artifacts, _ = _quality_and_chart_artifacts(tmp_path)
    profile = artifacts[0]
    bundle = _bundle_with_empty_sections()
    dup_ref = EvidenceRef(kind="stat", artifact_id=profile.id, locator="rows", value=5)
    claim = ReportClaim(
        id="c1",
        text="The dataset has 5 rows.",
        evidence=[dup_ref, dup_ref, dup_ref],
        referenced_datasets=["teams.csv"],
    )
    dataset_overview = next(
        section for section in bundle.sections if section.title == "Dataset Overview"
    )
    dataset_overview.claims.append(claim)

    markdown = report_bundle_to_markdown(bundle, artifacts=artifacts)

    # Narrative shows human claim text with no id prefix and no inline evidence line (P0-1).
    assert "- The dataset has 5 rows." in markdown
    assert "**c1**" not in markdown
    assert "- Evidence:" not in markdown
    # The claim payload keeps all three refs (lossless evidence).
    assert len(claim.evidence) == 3
    # The ledger cell still collapses the repeated id to one.
    assert f"{profile.id}, {profile.id}" not in markdown
    ledger_row = next(line for line in markdown.splitlines() if "| Dataset Overview | c1 |" in line)
    assert ledger_row.count(profile.id) == 1


def test_narrative_drops_claim_id_prefixes_and_inline_evidence() -> None:
    bundle = ReportBundle.empty(project_id="project_demo", session_id="run_demo")
    exec_section = next(s for s in bundle.sections if s.title == "Executive Summary")
    exec_section.claims.append(
        ReportClaim(
            id="exec_summary_1",
            text="Summary: Revenue is 120.",
            evidence=[
                EvidenceRef(kind="stat", artifact_id="table_1", locator="rows[0]", value=120)
            ],
        )
    )
    overview = next(s for s in bundle.sections if s.title == "Dataset Overview")
    overview.claims.append(
        ReportClaim(
            id="claim_1",
            text="The dataset has 3 rows.",
            evidence=[EvidenceRef(kind="stat", artifact_id="prof_1", locator="rows", value=3)],
        )
    )

    narrative = narrative_markdown(report_bundle_to_markdown(bundle))

    assert "**claim_" not in narrative
    assert "**exec_summary_" not in narrative
    assert "- Evidence:" not in narrative
    # The render-only "Summary: " prefix is stripped, but the bundle text is untouched.
    assert "- Revenue is 120." in narrative
    assert exec_section.claims[0].text == "Summary: Revenue is 120."


def test_duplicate_titled_sections_render_once() -> None:
    bundle = ReportBundle(
        project_id="project_demo",
        session_id="run_demo",
        sections=[
            ReportSection(
                title="Selected Analysis Focus",
                claims=[
                    ReportClaim(
                        id="a",
                        text="First focus.",
                        evidence=[EvidenceRef(kind="artifact", artifact_id="x", locator="l")],
                    )
                ],
            ),
            ReportSection(
                title="Selected Analysis Focus",
                claims=[
                    ReportClaim(
                        id="b",
                        text="Second focus.",
                        evidence=[EvidenceRef(kind="artifact", artifact_id="y", locator="l")],
                    )
                ],
            ),
        ],
    )

    markdown = report_bundle_to_markdown(bundle)

    assert markdown.count("## Selected Analysis Focus") == 1
    # Claims from both duplicate-titled sections survive the merge.
    assert "- First focus." in markdown
    assert "- Second focus." in markdown


def test_narrative_markdown_truncates_at_claim_ledger_and_is_noop_without_it() -> None:
    bundle = ReportBundle.empty(project_id="project_demo", session_id="run_demo")
    overview = next(s for s in bundle.sections if s.title == "Dataset Overview")
    overview.claims.append(
        ReportClaim(
            id="c1",
            text="The dataset has 3 rows.",
            evidence=[EvidenceRef(kind="stat", artifact_id="prof_1", locator="rows", value=3)],
        )
    )
    markdown = report_bundle_to_markdown(bundle)
    narrative = narrative_markdown(markdown)

    assert "### Claim Ledger" in markdown
    assert "### Claim Ledger" not in narrative
    assert "- The dataset has 3 rows." in narrative

    # Markdown stored by older runs without a ledger heading is returned unchanged.
    legacy = "# Old report\n\nSome prose without a ledger.\n"
    assert narrative_markdown(legacy) == legacy


def test_limitations_caps_columns_and_prefixes_dataset_name() -> None:
    columns = [f"col_{index}" for index in range(12)]
    profile_artifact = Artifact(
        id="prof_wide",
        type=ArtifactType.DATASET_PROFILE,
        project_id="project_demo",
        session_id="run_demo",
        payload=DatasetProfile(
            dataset_id="ds_wide",
            name="wide.csv",
            rows=10,
            columns=len(columns),
            column_names=columns,
            dtypes={column: "float64" for column in columns},
            missing_values={},
            missing_percent={},
            numeric_columns=columns,
            categorical_columns=[],
        ).model_dump(mode="json"),
    )
    quality_artifact = Artifact(
        id="qual_wide",
        type=ArtifactType.QUALITY_ISSUE_SET,
        project_id="project_demo",
        session_id="run_demo",
        payload=QualityIssueSet(
            dataset_id="ds_wide",
            issues=[
                QualityIssue(
                    severity="warn",
                    code="high_missing",
                    column=column,
                    message=f"{column} is mostly missing.",
                    recommendation="Review before analysis.",
                )
                for column in columns
            ],
        ).model_dump(mode="json"),
    )
    bundle = _bundle_with_empty_sections()

    markdown = report_bundle_to_markdown(
        bundle, artifacts=[profile_artifact, quality_artifact]
    )
    limitations = _section_text(markdown, "## Limitations and Risks")

    assert "wide.csv: high-risk columns for analysis:" in limitations
    # 12 columns -> 8 enumerated + a counted tail, keeping the total visible.
    assert "… and 4 more (see the Quality page)" in limitations
    assert limitations.count("`col_") == 8


def _section_text(markdown: str, heading: str) -> str:
    lines = markdown.splitlines()
    start = lines.index(heading)
    body: list[str] = []
    for line in lines[start + 1 :]:
        if line.startswith("## ") or line.startswith("### "):
            break
        body.append(line)
    return "\n".join(body)
