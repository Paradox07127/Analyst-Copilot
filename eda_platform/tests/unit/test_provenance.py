"""M6.2 P1 — provenance-complete artifact.

Covers the three optional ``Artifact`` provenance fields end to end:
* ``env_digest`` — a stable, run-global environment fingerprint injected centrally
  at save time.
* ``code_ref`` + ``plain_language`` — set at each computation artifact's build site.
* surfacing of both in the report exporters.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from eda_platform.core.provenance import code_ref, env_components, env_digest
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType, DatasetProfile
from eda_platform.schemas.charts import ChartSpec
from eda_platform.schemas.reports import ReportBundle, ReportStatus
from eda_platform.tools.chart_specs import create_chart_specs
from eda_platform.tools.exporter import export_markdown_report, report_bundle_to_markdown
from eda_platform.tools.html_exporter import export_report_html
from eda_platform.tools.loader import load_csv
from eda_platform.tools.ml_baseline import create_model_card_artifact, run_baseline_model
from eda_platform.tools.profiler import profile_dataset
from eda_platform.tools.stat_tests import create_stat_test_artifact, run_stat_test


# --------------------------------------------------------------------------- #
# env_digest: stable, reproducible, run-global
# --------------------------------------------------------------------------- #
def test_env_digest_is_stable_across_calls() -> None:
    first = env_digest()
    second = env_digest()

    assert first == second
    assert first.startswith("env_")
    # Short, opaque token — hash only, no environment detail leaked.
    assert len(first) > len("env_")


def test_env_components_include_python_and_key_dependencies() -> None:
    components = dict(env_components())

    assert "python" in components
    for dependency in ("pandas", "numpy", "scipy", "scikit-learn", "duckdb"):
        assert dependency in components
        # Installed in this environment — never the missing sentinel here.
        assert components[dependency] != "<missing>"


def test_code_ref_strips_package_prefix() -> None:
    ref = code_ref(create_chart_specs)

    assert ref == "tools.chart_specs.create_chart_specs"
    assert not ref.startswith("eda_platform.")


# --------------------------------------------------------------------------- #
# Central env_digest injection at save time
# --------------------------------------------------------------------------- #
def _bare_artifact(artifact_id: str) -> Artifact:
    return Artifact(
        id=artifact_id,
        type=ArtifactType.MARKDOWN_REPORT,
        project_id="project_demo",
        session_id="run_demo",
        payload={"markdown": "# report"},
    )


def test_save_injects_env_digest_when_absent(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    store.ensure_project("project_demo", name="Demo")
    store.start_session("project_demo", "run_demo")

    artifact = _bare_artifact("report_abc12345")
    assert artifact.env_digest is None

    store.save_artifact(artifact)
    loaded = store.get_artifact("report_abc12345")

    assert loaded.env_digest == env_digest()


def test_env_digest_is_identical_for_all_artifacts_in_a_run(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    store.ensure_project("project_demo", name="Demo")
    store.start_session("project_demo", "run_demo")

    for index in range(3):
        store.save_artifact(_bare_artifact(f"report_{index:08d}"))

    digests = {
        artifact.env_digest
        for artifact in store.list_artifacts(project_id="project_demo", session_id="run_demo")
    }

    assert digests == {env_digest()}


def test_save_preserves_a_preset_env_digest(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    store.ensure_project("project_demo", name="Demo")
    store.start_session("project_demo", "run_demo")

    artifact = _bare_artifact("report_preset01")
    artifact.env_digest = "env_preset"

    store.save_artifact(artifact)
    loaded = store.get_artifact("report_preset01")

    assert loaded.env_digest == "env_preset"


# --------------------------------------------------------------------------- #
# Computation artifacts carry code_ref + plain_language
# --------------------------------------------------------------------------- #
def _chart_artifacts(tmp_path: Path) -> list[Artifact]:
    # Enough distinct amounts to be charted as a histogram: the assertions
    # below name _histogram_rows as the provenance under test.
    csv_path = tmp_path / "sales.csv"
    rows = ["order_date,amount,region"]
    for index in range(40):
        rows.append(
            f"2026-01-{(index % 28) + 1:02d},{10 + index * 3.5:.2f},"
            f"{'East' if index % 2 == 0 else 'West'}"
        )
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    loaded = load_csv(csv_path, dataset_id="ds_sales")
    profile = profile_dataset(loaded, project_id="project_demo", session_id="run_demo")
    return create_chart_specs(loaded, profile, project_id="project_demo", session_id="run_demo")


def test_chart_artifacts_have_code_ref_and_plain_language(tmp_path: Path) -> None:
    artifacts = _chart_artifacts(tmp_path)

    assert artifacts
    for artifact in artifacts:
        assert artifact.type is ArtifactType.CHART_SPEC
        assert artifact.code_ref and artifact.code_ref.startswith("tools.chart_specs.")
        assert artifact.plain_language

    histogram = next(a for a in artifacts if a.code_ref == "tools.chart_specs._histogram_rows")
    assert histogram.plain_language is not None
    assert "Histogram of amount" in histogram.plain_language
    assert "bins" in histogram.plain_language


def test_saved_chart_artifact_is_provenance_complete(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    store.ensure_project("project_demo", name="Demo")
    store.start_session("project_demo", "run_demo")

    for artifact in _chart_artifacts(tmp_path):
        store.save_artifact(artifact)

    saved = store.list_artifacts(project_id="project_demo", session_id="run_demo")
    assert saved
    for artifact in saved:
        assert artifact.env_digest
        assert artifact.code_ref
        assert artifact.plain_language


def test_stat_artifact_has_code_ref_and_plain_language() -> None:
    frame = pd.DataFrame(
        {
            "segment": ["A"] * 10 + ["B"] * 10,
            "revenue": [9, 10, 11, 10, 12, 11, 9, 10, 11, 12]
            + [21, 20, 22, 23, 19, 21, 22, 20, 23, 21],
        }
    )
    result = run_stat_test(
        frame,
        dataset_id="ds_sales",
        test_type="independent_t_test",
        group_column="segment",
        value_column="revenue",
    )
    artifact = create_stat_test_artifact(result, project_id="project_demo", session_id="run_demo")

    assert artifact.code_ref == "tools.stat_tests.run_stat_test"
    assert artifact.plain_language
    assert "independent t test" in artifact.plain_language
    assert "revenue" in artifact.plain_language


def test_model_card_artifact_has_code_ref_and_plain_language() -> None:
    frame = pd.DataFrame(
        {
            "customer_id": [f"C{i:03d}" for i in range(80)],
            "spend": [float(i % 20) for i in range(80)],
            "visits": [i % 7 + 1 for i in range(80)],
            "churned": [1 if i % 5 in {0, 1} else 0 for i in range(80)],
        }
    )
    card = run_baseline_model(frame, dataset_id="ds_customers", target_column="churned")
    artifact = create_model_card_artifact(card, project_id="project_demo", session_id="run_demo")

    assert artifact.code_ref == "tools.ml_baseline.run_baseline_model"
    assert artifact.plain_language
    assert "classification baseline" in artifact.plain_language
    assert "churned" in artifact.plain_language


# --------------------------------------------------------------------------- #
# Surfacing: reports show env_digest and plain_language
# --------------------------------------------------------------------------- #
def test_html_report_footer_shows_env_digest_and_chart_description() -> None:
    bundle = ReportBundle.empty(project_id="project_demo", session_id="run_demo")
    bundle.status = ReportStatus.VALIDATED
    chart = ChartSpec(
        dataset_id="ds_sales",
        title="Distribution of amount",
        description="Histogram of amount over all non-null rows.",
        mark="bar",
        data={"values": [{"bin": "0–10", "count": 2}]},
        encoding={
            "x": {"field": "bin", "type": "nominal"},
            "y": {"field": "count", "type": "quantitative"},
        },
    )

    html = export_report_html(bundle, charts=[chart])

    assert env_digest() in html
    assert "Environment digest" in html
    assert "Histogram of amount over all non-null rows." in html


def test_markdown_reports_surface_env_digest_and_plain_language(tmp_path: Path) -> None:
    chart_artifacts = _chart_artifacts(tmp_path)
    profile_artifact = Artifact(
        id="prof_demo01",
        type=ArtifactType.DATASET_PROFILE,
        project_id="project_demo",
        session_id="run_demo",
        payload=DatasetProfile(
            dataset_id="ds_sales",
            name="sales.csv",
            rows=3,
            columns=3,
            column_names=["order_date", "amount", "region"],
            dtypes={"order_date": "object", "amount": "int64", "region": "object"},
            missing_values={},
            missing_percent={},
            numeric_columns=["amount"],
            categorical_columns=["region"],
        ).model_dump(mode="json"),
    )
    artifacts = [profile_artifact, *chart_artifacts]
    plain = next(a.plain_language for a in chart_artifacts if a.plain_language)

    m1_report = export_markdown_report(
        artifacts, project_id="project_demo", session_id="run_demo"
    )
    assert f"`{env_digest()}`" in m1_report.payload["markdown"]
    assert plain in m1_report.payload["markdown"]

    bundle = ReportBundle.empty(project_id="project_demo", session_id="run_demo")
    agentic_markdown = report_bundle_to_markdown(bundle, artifacts=artifacts)
    assert f"`{env_digest()}`" in agentic_markdown
