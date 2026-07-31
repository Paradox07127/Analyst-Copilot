import warnings
from pathlib import Path

import pandas as pd
import pytest

from eda_platform.core.llm import OfflineLLMClient
from eda_platform.drivers.auto_eda import run_auto_eda
from eda_platform.schemas.artifacts import Artifact, ArtifactType, QualityIssueSet
from eda_platform.tools.analysis import create_analysis_tables
from eda_platform.tools.anomaly import screen_anomalies
from eda_platform.tools.chart_specs import create_chart_specs
from eda_platform.tools.handoff import create_eda_handoff_artifact
from eda_platform.tools.loader import load_csv
from eda_platform.tools.profiler import profile_dataset
from eda_platform.tools.quality import scan_quality
from eda_platform.tools.stat_tests import run_stat_test


def test_loader_preserves_identifier_lexemes_and_large_nullable_integer(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ids.csv"
    path.write_text(
        "customer_id,postal,value\n"
        "00123,00456,9007199254740993\n"
        "00124,00457,\n",
        encoding="utf-8",
    )

    loaded = load_csv(path)

    assert loaded.frame["customer_id"].tolist() == ["00123", "00124"]
    assert loaded.frame["postal"].tolist() == ["00456", "00457"]
    assert loaded.frame["value"].iloc[0] == 9007199254740993


def test_correlation_uses_pairwise_complete_sample_size(tmp_path: Path) -> None:
    path = tmp_path / "pairwise.csv"
    pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [2.0, 4.0, 6.0, None],
            "c": [1.0, None, 3.0, 4.0],
        }
    ).to_csv(path, index=False)
    loaded = load_csv(path)
    profile = profile_dataset(loaded, project_id="p", session_id="s")

    tables = create_analysis_tables(loaded, profile, project_id="p", session_id="s")
    correlation = next(table for table in tables if table.payload["kind"] == "correlation")
    rows = {
        frozenset((row["column_a"], row["column_b"])): row
        for row in correlation.payload["rows"]
    }

    assert rows[frozenset(("a", "b"))]["sample_size"] == 3
    assert rows[frozenset(("a", "b"))]["excluded_pair_n"] == 1
    assert rows[frozenset(("a", "c"))]["sample_size"] == 3
    assert all(row["missing_policy"] == "pairwise_complete" for row in rows.values())


def test_correlation_ignores_constant_column_but_keeps_varying_pairs(
    tmp_path: Path,
) -> None:
    # StandardHours-style constant columns must neither emit numpy divide
    # warnings nor suppress correlations between the remaining varying pairs.
    path = tmp_path / "hr_like.csv"
    pd.DataFrame(
        {
            "standard_hours": [80.0, 80.0, 80.0, 80.0, 80.0],
            "salary": [1.5, 2.5, 3.5, 4.5, 6.0],
            "bonus": [3.1, 5.2, 7.0, 8.9, 12.1],
        }
    ).to_csv(path, index=False)
    loaded = load_csv(path)
    profile = profile_dataset(loaded, project_id="p", session_id="s")

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        tables = create_analysis_tables(loaded, profile, project_id="p", session_id="s")

    correlation = next(table for table in tables if table.payload["kind"] == "correlation")
    columns_in_rows = {
        column
        for row in correlation.payload["rows"]
        for column in (row["column_a"], row["column_b"])
    }
    assert columns_in_rows == {"salary", "bonus"}


def test_missing_group_is_excluded_from_statistical_comparison() -> None:
    frame = pd.DataFrame(
        {
            "group": ["a", "a", "a", "b", "b", "b", None, None],
            "value": [1, 2, 3, 4, 5, 6, 1000, 2000],
        }
    )

    result = run_stat_test(
        frame,
        dataset_id="d",
        test_type="independent_t_test",
        group_column="group",
        value_column="value",
    )

    assert result.sample_size == 6
    assert set(result.groups) == {"a", "b"}


def test_paired_test_requires_key_and_is_row_order_invariant() -> None:
    frame = pd.DataFrame(
        {
            "subject": [1, 2, 3, 4, 1, 2, 3, 4],
            "phase": ["before"] * 4 + ["after"] * 4,
            "value": [10, 20, 30, 40, 11, 18, 35, 39],
        }
    )
    with pytest.raises(ValueError, match="pair_column"):
        run_stat_test(
            frame,
            dataset_id="d",
            test_type="paired_t_test",
            group_column="phase",
            value_column="value",
        )

    original = run_stat_test(
        frame,
        dataset_id="d",
        test_type="paired_t_test",
        group_column="phase",
        value_column="value",
        pair_column="subject",
    )
    shuffled = run_stat_test(
        frame.sample(frac=1.0, random_state=7),
        dataset_id="d",
        test_type="paired_t_test",
        group_column="phase",
        value_column="value",
        pair_column="subject",
    )

    assert original.statistic == shuffled.statistic
    assert original.p_value == shuffled.p_value
    assert original.sample_size == 4


def test_empty_dataset_is_critical_and_sparse_spike_is_detected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("amount,group\n", encoding="utf-8")
    loaded = load_csv(path)
    profile = profile_dataset(loaded, project_id="p", session_id="s")
    quality = scan_quality(profile, project_id="p", session_id="s")
    issues = QualityIssueSet.model_validate(quality.payload)

    assert any(
        issue.code == "empty_dataset" and issue.severity == "critical"
        for issue in issues.issues
    )
    spike = screen_anomalies(
        pd.DataFrame({"value": [0.0] * 20 + [100.0]}),
        dataset_name="spike",
        column="value",
    )
    assert spike.outlier_count == 1
    assert spike.top_outliers[0].value == 100.0


def test_fully_empty_columns_do_not_consume_distribution_chart_quota(
    tmp_path: Path,
) -> None:
    # World Cup teams/players regression: 100%-empty numeric columns must not
    # occupy the distribution-chart slots and leave a table with zero histograms.
    path = tmp_path / "sparse.csv"
    rows = ["empty_0,empty_1,empty_2,empty_3,empty_4,good_a,good_b"]
    for index in range(30):
        rows.append(f",,,,,{index * 1.5 + 0.25},{100 - index * 2.5}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    loaded = load_csv(path)
    profile_artifact = profile_dataset(loaded, project_id="p", session_id="s")

    charts = create_chart_specs(loaded, profile_artifact, project_id="p", session_id="s")
    distribution_titles = [
        artifact.payload["title"]
        for artifact in charts
        if artifact.payload.get("category") == "distribution"
    ]

    assert any("good_a" in title for title in distribution_titles)
    assert any("good_b" in title for title in distribution_titles)


def test_numeric_pii_columns_do_not_leak_into_scatter_charts(tmp_path: Path) -> None:
    # A numeric phone column correlated with spend must not put raw row values
    # into persisted scatter ChartSpecs (analysis tables keep aggregates only).
    # "tel" is a PII name hint but not a loader string-preservation token, so
    # the column stays numeric and reaches the correlation/scatter path. The
    # correlation must stay below the trivial-pair ceiling so a scatter chart
    # is actually emitted.
    path = tmp_path / "phones.csv"
    rows = ["tel,spend,revenue"]
    jitter = [7, -13, 22, -4, 15, -19, 3, 11, -8, 18]
    for index in range(40):
        tel = 13800000000 + index * 1_000_003 + jitter[index % 10] * 400_000
        revenue = index * 5.0 + 3.0 + jitter[(index + 3) % 10] * 2.0
        rows.append(f"{tel},{index * 2.5 + 1.0},{revenue}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    result = run_auto_eda(
        [path],
        workspace=tmp_path / "workspace",
        project_id="p",
        session_id="s",
        llm=OfflineLLMClient(),
        generate_report=False,
    )
    serialized_charts = "\n".join(
        artifact.model_dump_json()
        for artifact in result.artifacts
        if artifact.type is ArtifactType.CHART_SPEC
    )

    scatter_titles = [
        artifact.payload["title"]
        for artifact in result.artifacts
        if artifact.type is ArtifactType.CHART_SPEC
        and artifact.payload["mark"] == "point"
    ]
    assert scatter_titles, "expected at least one scatter chart in this setup"
    assert "13802800000" not in serialized_charts


def test_bonferroni_family_is_not_the_upload_batch(tmp_path: Path) -> None:
    # Adding an unrelated file to the same run must not change another
    # dataset's adjusted p-value: the auto family is per dataset, not per batch.
    group_rows = ["group,value"]
    for index in range(12):
        group_rows.append(f"a,{index * 1.0}")
        group_rows.append(f"b,{index * 1.0 + 40.0}")
    first = tmp_path / "first.csv"
    first.write_text("\n".join(group_rows) + "\n", encoding="utf-8")
    second = tmp_path / "second.csv"
    second.write_text("\n".join(group_rows) + "\n", encoding="utf-8")

    result = run_auto_eda(
        [first, second],
        workspace=tmp_path / "workspace",
        project_id="p",
        session_id="s",
        llm=OfflineLLMClient(),
        generate_report=False,
    )
    stat_results = [
        artifact.payload
        for artifact in result.artifacts
        if artifact.type is ArtifactType.STAT_TEST_RESULT
    ]

    assert stat_results
    for payload in stat_results:
        assert payload["correction_method"] is None
        assert payload["adjusted_p_value"] is None
        assert any(
            warning["code"] == "exploratory_auto_selection"
            for warning in payload["warnings"]
        )


def test_auto_stat_test_never_selects_pii_columns(tmp_path: Path) -> None:
    # A numeric phone column must not become the boxplot measure: the ANOVA
    # boxplot artifact inlines raw per-row values of the selected measure.
    from eda_platform.drivers.auto_eda import _select_stat_test
    from eda_platform.schemas.artifacts import DatasetProfile
    from eda_platform.tools.pii import mask_profile_artifact, tag_pii_columns

    # Two distinct phone values with >=5 rows each make tel an *eligible*
    # grouping dimension; without the PII filter it wins over segment by
    # column order and its raw values become boxplot group labels.
    path = tmp_path / "grouped_phones.csv"
    rows = ["tel,score_value,segment"]
    values = [3.0, 7.5, 5.25, 9.0, 4.5, 8.25]
    for index in range(24):
        rows.append(
            f"{13800000000 + (index % 2) * 55_5001},"
            f"{values[index % 6] + index * 0.25},"
            f"{'a' if index % 2 == 0 else 'b'}"
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    loaded = load_csv(path)
    profile_artifact = profile_dataset(loaded, project_id="p", session_id="s")
    pii = tag_pii_columns(profile_artifact, project_id="p", session_id="s")
    masked = mask_profile_artifact(profile_artifact, pii)
    profile = DatasetProfile.model_validate(masked.payload)
    assert "tel" in profile.pii_columns

    spec = _select_stat_test(loaded, profile)

    assert spec is not None
    assert spec.value_column != "tel"
    assert spec.group_column != "tel"


def test_handoff_links_precleaned_dataset_to_raw_and_recipe(tmp_path: Path) -> None:
    # Under launch pre-clean the recipe/raw artifacts carry the *raw* dataset
    # id; the handoff must attach them to the analysed dataset instead of
    # dumping unlabeled ids into the global bucket.
    path = tmp_path / "clean.csv"
    path.write_text("amount,segment\n1.5,a\n2.5,b\n4.0,a\n", encoding="utf-8")
    loaded = load_csv(path)
    profile_artifact = profile_dataset(loaded, project_id="p", session_id="s")
    clean_id = profile_artifact.payload["dataset_id"]
    raw_id = "ds_raw_source"

    def _artifact(artifact_id: str, artifact_type: ArtifactType) -> Artifact:
        return Artifact(
            id=artifact_id,
            type=artifact_type,
            project_id="p",
            session_id="s",
            payload={"dataset_id": raw_id},
        )

    handoff = create_eda_handoff_artifact(
        [
            profile_artifact,
            _artifact("recipe_1", ArtifactType.CLEANING_RECIPE),
            _artifact("raw_prof_1", ArtifactType.RAW_DATASET_PROFILE),
            _artifact("raw_prev_1", ArtifactType.RAW_DATA_PREVIEW),
        ],
        project_id="p",
        session_id="s",
        raw_dataset_lineage={raw_id: clean_id},
    )

    bucket = handoff.payload["artifact_index"]["by_dataset"][clean_id]
    assert bucket["cleaning_recipe"] == "recipe_1"
    assert bucket["raw_profile"] == "raw_prof_1"
    assert bucket["raw_preview"] == "raw_prev_1"
    dataset_entry = handoff.payload["datasets"][0]
    assert dataset_entry["raw_dataset_id"] == raw_id
    assert handoff.payload["artifact_index"]["global"] == []


def test_pipeline_masks_pii_and_emits_compact_handoff(tmp_path: Path) -> None:
    path = tmp_path / "contacts.csv"
    path.write_text(
        "customer_email,segment,amount\n"
        "alice@example.com,A,10\n"
        "bob@example.com,B,20\n"
        "carol@example.com,A,30\n",
        encoding="utf-8",
    )

    result = run_auto_eda(
        [path],
        workspace=tmp_path / "workspace",
        project_id="p",
        session_id="s",
        llm=OfflineLLMClient(),
        generate_report=False,
    )
    serialized = "\n".join(
        artifact.model_dump_json()
        for artifact in result.artifacts
        if artifact.type
        in {
            ArtifactType.DATASET_PROFILE,
            ArtifactType.CHART_SPEC,
            ArtifactType.PII_REPORT,
            ArtifactType.EDA_HANDOFF,
        }
    )

    assert "alice@example.com" not in serialized
    assert "[PII:email]" in serialized
    handoff = next(
        artifact for artifact in result.artifacts if artifact.type is ArtifactType.EDA_HANDOFF
    )
    assert handoff.payload["datasets"][0]["analysis_ready"] is True
    assert "by_dataset" in handoff.payload["artifact_index"]
    assert handoff.payload["schema_version"] == 2
    assert handoff.payload["cross_dataset_relationships"]["status"] == "not_applicable"
