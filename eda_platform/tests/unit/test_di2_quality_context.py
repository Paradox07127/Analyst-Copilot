from pathlib import Path

from eda_platform.schemas.artifacts import QualityIssueSet
from eda_platform.schemas.quality_context import QualityContext, QualityContextSet
from eda_platform.tools.loader import load_csv
from eda_platform.tools.profiler import profile_dataset
from eda_platform.tools.quality import scan_quality
from eda_platform.tools.quality_context import build_quality_context


def test_quality_context_contains_observations_without_business_cause_fields(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "orders.csv"
    csv_path.write_text(
        "region,amount\nNorth,10\nNorth,\nSouth,30\nSouth,\n",
        encoding="utf-8",
    )
    loaded = load_csv(csv_path, dataset_id="ds_orders")
    profile = profile_dataset(loaded, project_id="project", session_id="run")
    quality = scan_quality(profile, project_id="project", session_id="run")
    issue_set = QualityIssueSet.model_validate(quality.payload)
    assert issue_set.issues

    artifact = build_quality_context(
        loaded,
        profile,
        quality,
        project_id="project",
        session_id="run",
    )
    context_set = QualityContextSet.model_validate(artifact.payload)

    assert context_set.contexts
    assert artifact.parents == [profile.id, quality.id]
    assert set(context_set.contexts[0].model_dump()) == set(QualityContext.model_fields)
    forbidden = {"business_cause", "root_cause", "cause", "caused_by"}
    assert forbidden.isdisjoint(context_set.contexts[0].model_dump())
    assert all(
        "business cause remains unconfirmed" in context.report_limitation
        or context.requires_data
        for context in context_set.contexts
    )


def test_quality_context_rejects_mismatched_profile_and_issue_set(tmp_path: Path) -> None:
    csv_path = tmp_path / "orders.csv"
    csv_path.write_text("amount\n1\n2\n", encoding="utf-8")
    loaded = load_csv(csv_path, dataset_id="ds_orders")
    profile = profile_dataset(loaded, project_id="project", session_id="run")
    quality = scan_quality(profile, project_id="project", session_id="run")
    quality.payload["dataset_id"] = "ds_other"

    try:
        build_quality_context(
            loaded,
            profile,
            quality,
            project_id="project",
            session_id="run",
        )
    except ValueError as exc:
        assert "same dataset" in str(exc)
    else:
        raise AssertionError("mismatched artifacts must be rejected")
