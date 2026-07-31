from pathlib import Path

from eda_platform.tools.chart_specs import create_chart_specs
from eda_platform.tools.exporter import export_markdown_report
from eda_platform.tools.loader import load_csv
from eda_platform.tools.profiler import profile_dataset
from eda_platform.tools.quality import scan_quality


def test_report_contains_m1_sections_and_artifact_references(tmp_path: Path) -> None:
    csv_path = tmp_path / "sales.csv"
    csv_path.write_text(
        "order_id,order_date,amount,region\n"
        "1,2026-01-01,10,East\n"
        "2,2026-01-02,,West\n",
        encoding="utf-8",
    )
    loaded = load_csv(csv_path, dataset_id="ds_sales")
    profile = profile_dataset(loaded, project_id="project_demo", session_id="run_demo")
    quality = scan_quality(profile, project_id="project_demo", session_id="run_demo")
    charts = create_chart_specs(
        loaded, profile, project_id="project_demo", session_id="run_demo"
    )

    report = export_markdown_report(
        [profile, quality, *charts],
        project_id="project_demo",
        session_id="run_demo",
    )
    markdown = report.payload["markdown"]

    assert "## Data Map" in markdown
    assert "## Quality Risks" in markdown
    assert "## Generated Charts" in markdown
    assert "## Suggested Next Analyses" in markdown
    assert "## Limitations" in markdown
    assert profile.id in markdown
    assert quality.id in markdown
    assert "sales.csv" in markdown
    assert "PK candidates: order_id" in markdown
