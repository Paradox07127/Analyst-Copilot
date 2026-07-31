from __future__ import annotations

from pathlib import Path

from eda_platform.tools.analysis import create_analysis_tables
from eda_platform.tools.chart_specs import create_chart_specs
from eda_platform.tools.evidence import build_evidence_pack
from eda_platform.tools.loader import load_csv
from eda_platform.tools.ml_baseline import create_model_card_artifact, run_baseline_model
from eda_platform.tools.profiler import profile_dataset
from eda_platform.tools.quality import scan_quality
from eda_platform.tools.stat_tests import create_stat_test_artifact, run_stat_test


def test_evidence_pack_indexes_profiles_quality_tables_and_charts(tmp_path: Path) -> None:
    csv_path = tmp_path / "sales.csv"
    csv_path.write_text(
        "order_id,order_date,revenue,region\n"
        "1,2026-01-01,10,East\n"
        "2,2026-01-02,,West\n"
        "3,2026-01-03,30,East\n",
        encoding="utf-8",
    )
    loaded = load_csv(csv_path, dataset_id="ds_sales")
    profile = profile_dataset(loaded, project_id="project_demo", session_id="run_demo")
    quality = scan_quality(profile, project_id="project_demo", session_id="run_demo")
    charts = create_chart_specs(
        loaded, profile, project_id="project_demo", session_id="run_demo"
    )
    tables = create_analysis_tables(
        loaded,
        profile,
        project_id="project_demo",
        session_id="run_demo",
    )

    pack = build_evidence_pack(
        [profile, quality, *charts, *tables],
        payload_policy="schema+aggregates",
    )

    assert pack.payload_policy == "schema+aggregates"
    assert profile.id in pack.artifact_index
    assert quality.id in pack.artifact_index
    assert charts[0].id in pack.artifact_index
    assert tables[0].id in pack.artifact_index
    assert pack.datasets[0].name == "sales.csv"
    assert pack.datasets[0].row_count == 3
    assert pack.datasets[0].column_count == 4
    assert "numeric" in pack.datasets[0].semantic_type_counts
    assert pack.datasets[0].sample_rows == []
    assert pack.quality_issue_count >= 1
    assert pack.analysis_table_count >= 1
    assert pack.chart_count >= 1


def test_evidence_pack_can_include_limited_samples_only_when_enabled(tmp_path: Path) -> None:
    csv_path = tmp_path / "sales.csv"
    csv_path.write_text(
        "order_id,revenue\n"
        "1,10\n"
        "2,20\n",
        encoding="utf-8",
    )
    loaded = load_csv(csv_path, dataset_id="ds_sales")
    profile = profile_dataset(loaded, project_id="project_demo", session_id="run_demo")

    pack = build_evidence_pack(
        [profile],
        payload_policy="schema+aggregates+sample",
        sample_limit=1,
    )

    assert len(pack.datasets[0].sample_rows) == 1
    assert pack.datasets[0].sample_rows[0]["order_id"] == 1


def test_evidence_pack_indexes_m5_stat_tests_and_model_cards(tmp_path: Path) -> None:
    csv_path = tmp_path / "customers.csv"
    csv_path.write_text(
        "segment,spend,visits,churned\n"
        + "\n".join(
            f"{'A' if i < 40 else 'B'},{float(i % 20)},{i % 6 + 1},{1 if i % 5 in {0, 1} else 0}"
            for i in range(80)
        )
        + "\n",
        encoding="utf-8",
    )
    loaded = load_csv(csv_path, dataset_id="ds_customers")
    stat_result = run_stat_test(
        loaded.frame,
        dataset_id="ds_customers",
        test_type="independent_t_test",
        group_column="segment",
        value_column="spend",
    )
    stat_artifact = create_stat_test_artifact(
        stat_result,
        project_id="project_demo",
        session_id="run_demo",
    )
    model_artifact = create_model_card_artifact(
        run_baseline_model(loaded.frame, dataset_id="ds_customers", target_column="churned"),
        project_id="project_demo",
        session_id="run_demo",
    )

    pack = build_evidence_pack(
        [stat_artifact, model_artifact],
        payload_policy="schema+aggregates",
    )

    assert stat_artifact.id in pack.artifact_index
    assert model_artifact.id in pack.artifact_index
    assert pack.stat_tests[0].test_type == "independent_t_test"
    assert pack.model_cards[0].target_column == "churned"
