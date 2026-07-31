"""Distribution-shape classification and the chart forms it selects."""

from pathlib import Path

import numpy as np
import pandas as pd

from eda_platform.application.distribution_view import column_distributions
from eda_platform.schemas.artifacts import DatasetProfile
from eda_platform.tools.chart_specs import create_chart_specs
from eda_platform.tools.frame_stats import distribution_kind, histogram_bin_count
from eda_platform.tools.loader import load_csv
from eda_platform.tools.profiler import profile_dataset


def test_distribution_kind_sees_through_dtype_to_actual_content() -> None:
    # dtype is not the question: a float column holding only 0.0/1.0 is a flag,
    # and an integer column holding 200 distinct values is continuous.
    assert distribution_kind(pd.Series([0, 1, 1, 0, 1] * 20)) == "binary"
    assert distribution_kind(pd.Series([0.0, 1.0, 1.0, 0.0] * 25)) == "binary"
    assert distribution_kind(pd.Series([1, 0, None, 1, 0] * 20)) == "binary"
    assert distribution_kind(pd.Series([True, False] * 50)) == "binary"
    assert distribution_kind(pd.Series([7.5] * 40)) == "constant"
    assert distribution_kind(pd.Series([(index % 5) + 1 for index in range(200)])) == "discrete"
    assert distribution_kind(pd.Series(np.linspace(0.0, 100.0, 500))) == "continuous"


def test_histogram_bin_count_adapts_instead_of_always_thirty() -> None:
    rng = np.random.default_rng(11)
    tight = pd.Series(rng.normal(size=60))
    wide = pd.Series(rng.normal(size=20_000))

    # Freedman-Diaconis scales with sample size; a fixed 30 does not.
    assert histogram_bin_count(tight) < histogram_bin_count(wide)
    assert 5 <= histogram_bin_count(tight) <= 60
    assert 5 <= histogram_bin_count(wide) <= 60
    # A degenerate spread must not produce a zero or negative bin count.
    assert histogram_bin_count(pd.Series([3.0] * 50)) >= 1


def test_preview_strip_gives_flag_columns_value_counts_not_ten_bins() -> None:
    # The reported defect: an Int64 0/1 column rendered as two tall end bars
    # with eight empty bins between them in the Table preview header.
    frame = pd.DataFrame(
        {
            "churned": pd.array(
                [1 if index % 4 == 0 else 0 for index in range(200)], dtype="Int64"
            ),
            "amount": [float(index) for index in range(200)],
        }
    )

    dists = {dist["name"]: dist for dist in column_distributions(frame)}

    assert dists["churned"]["kind"] == "categorical"
    assert {label for label, _ in dists["churned"]["top"]} == {"0", "1"}
    assert dists["amount"]["kind"] == "numeric"


def test_flag_and_low_cardinality_columns_get_bars_not_histograms(tmp_path: Path) -> None:
    path = tmp_path / "shapes.csv"
    rows = ["flag_float,rating,amount"]
    for index in range(200):
        flag = 1.0 if index % 4 == 0 else 0.0
        rows.append(f"{flag},{(index % 5) + 1},{float(index) * 1.5}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    loaded = load_csv(path)
    profile_artifact = profile_dataset(loaded, project_id="p", session_id="s")
    profile = DatasetProfile.model_validate(profile_artifact.payload)

    kinds = {column.name: column.distribution_kind for column in profile.columns_detail}
    assert kinds["flag_float"] == "binary"
    assert kinds["rating"] == "discrete"
    assert kinds["amount"] == "continuous"

    charts = {
        artifact.payload["title"]: artifact.payload
        for artifact in create_chart_specs(
            loaded, profile_artifact, project_id="p", session_id="s"
        )
    }
    # A flag renders as one bar per observed value, never as binned ranges.
    flag_chart = charts["Top values in flag_float"]
    assert len(flag_chart["data"]["values"]) == 2
    assert "bin_start" not in flag_chart["encoding"]["x"]["field"]
    assert "Distribution of flag_float" not in charts

    rating_chart = charts["Top values in rating"]
    assert len(rating_chart["data"]["values"]) == 5

    amount_chart = charts["Distribution of amount"]
    assert amount_chart["encoding"]["x"]["field"] == "bin_start"


def test_heavy_right_tail_histogram_bins_on_a_log_scale(tmp_path: Path) -> None:
    # Equal-width bins put 99% of a power-law column in the first bar, which
    # renders correctly and shows nothing.
    path = tmp_path / "skewed.csv"
    rows = ["spend"]
    for index in range(400):
        rows.append(f"{float(2 ** (index % 20)) + 1.0}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    loaded = load_csv(path)
    profile_artifact = profile_dataset(loaded, project_id="p", session_id="s")

    charts = {
        artifact.payload["title"]: artifact.payload
        for artifact in create_chart_specs(
            loaded, profile_artifact, project_id="p", session_id="s"
        )
    }
    spend = charts["Distribution of spend"]

    assert spend["encoding"]["x"].get("scale", {}).get("type") == "log"
    counts = [row["count"] for row in spend["data"]["values"]]
    # No single bin may swallow the column once it is binned in log space.
    assert max(counts) < sum(counts) * 0.5
