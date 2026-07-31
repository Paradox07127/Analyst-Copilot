from __future__ import annotations

from pathlib import Path

from eda_platform.schemas.artifacts import ArtifactType, PiiReport
from eda_platform.tools.loader import load_csv
from eda_platform.tools.pii import mask_value, tag_pii_columns
from eda_platform.tools.profiler import profile_dataset
from eda_platform.tools.value_profile import top_n_values


def test_tag_pii_columns_detects_names_email_and_phone(tmp_path: Path) -> None:
    csv_path = tmp_path / "customers.csv"
    csv_path.write_text(
            "customer_name,email,phone,region\n"
            "Alice,a@example.com,415-555-0100,East\n"
            "Bob,b@example.com,415-555-0101,West\n",
        encoding="utf-8",
    )
    loaded = load_csv(csv_path, dataset_id="ds_customers")
    profile = profile_dataset(loaded, project_id="project_demo", session_id="run_demo")

    artifact = tag_pii_columns(profile, project_id="project_demo", session_id="run_demo")
    report = PiiReport.model_validate(artifact.payload)
    pii_by_column = {column.column: column.label for column in report.columns}

    assert artifact.type is ArtifactType.PII_REPORT
    assert pii_by_column["customer_name"] == "name"
    assert pii_by_column["email"] == "email"
    assert pii_by_column["phone"] == "phone"
    assert "地区" not in pii_by_column


def test_mask_value_uses_pii_label_from_report(tmp_path: Path) -> None:
    csv_path = tmp_path / "customers.csv"
    csv_path.write_text("email,region\na@example.com,East\n", encoding="utf-8")
    loaded = load_csv(csv_path, dataset_id="ds_customers")
    profile = profile_dataset(loaded, project_id="project_demo", session_id="run_demo")
    pii = tag_pii_columns(profile, project_id="project_demo", session_id="run_demo")

    assert mask_value("email", "a@example.com", pii) == "[PII:email]"
    assert mask_value("region", "East", pii) == "East"


def test_top_n_values_masks_pii_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "customers.csv"
    csv_path.write_text(
        "email,region\n"
        "a@example.com,East\n"
        "b@example.com,East\n"
        "c@example.com,West\n",
        encoding="utf-8",
    )
    loaded = load_csv(csv_path, dataset_id="ds_customers")
    profile = profile_dataset(loaded, project_id="project_demo", session_id="run_demo")
    pii = tag_pii_columns(profile, project_id="project_demo", session_id="run_demo")

    value_profile = top_n_values(loaded, profile, pii, top_n=2)

    assert value_profile.dataset_id == "ds_customers"
    assert value_profile.values["email"] == [{"value": "[PII:email]", "count": 3}]
    assert value_profile.values["region"] == [
        {"value": "East", "count": 2},
        {"value": "West", "count": 1},
    ]
