from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from eda_platform.core.config import require_absolute_workspace
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.auto_eda import AutoEDAResult
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.tools.loader import LoadedDataset, load_csv


@dataclass
class LoadedRun:
    """Outcome of rebuilding an :class:`AutoEDAResult` from a past run on disk."""

    result: AutoEDAResult | None
    warnings: list[str]
    ok: bool
    """True when a usable result was reconstructed; False only when the run dir
    is missing/unreadable entirely."""
    datasets_available: bool
    """True when every profiled dataset's source CSV was reloaded (chat can
    work); False means the run is view-only."""


def load_run(project_id: str, session_id: str, *, workspace: Path | str) -> LoadedRun:
    """Rebuild an ``AutoEDAResult`` for a past run without ever raising."""
    warnings: list[str] = []
    workspace_path = require_absolute_workspace(workspace)

    try:
        store = ArtifactStore(workspace_path)
    except (OSError, ValueError) as exc:
        return LoadedRun(
            result=None,
            warnings=[f"workspace unreadable: {exc}"],
            ok=False,
            datasets_available=False,
        )

    # (a) Robust artifact list. A missing run dir => not loadable at all.
    session_dir = store.session_dir(project_id, session_id)
    if not session_dir.exists():
        return LoadedRun(
            result=None,
            warnings=[f"run not found: {session_id}"],
            ok=False,
            datasets_available=False,
        )
    artifacts, artifact_warnings = store.list_artifacts_safe(project_id=project_id, session_id=session_id)
    warnings.extend(artifact_warnings)
    manifest_data = _safe_manifest(store, project_id, session_id)

    # (b) Report markdown: prefer the MarkdownReport artifact, else the file.
    report_markdown = _report_markdown(artifacts, session_dir, warnings)

    # (c) Business context: best-effort, default "".
    business_context = _business_context(artifacts, manifest_data)

    # (d) Reconstruct loaded datasets from DatasetProfile artifacts.
    loaded_datasets, datasets_available = _reload_datasets(
        artifacts,
        workspace_path,
        project_id,
        warnings,
        input_hashes=_manifest_input_hashes(manifest_data),
    )

    # (e) Assemble. Usable when there are artifacts or a report to show.
    ok = bool(artifacts) or bool(report_markdown)
    result = AutoEDAResult(
        project_id=project_id,
        session_id=session_id,
        business_context=business_context,
        artifacts=artifacts,
        report_markdown=report_markdown,
        workspace=workspace_path,
        loaded_datasets=loaded_datasets,
        load_warnings=list(warnings),
    )
    return LoadedRun(
        result=result,
        warnings=warnings,
        ok=ok,
        datasets_available=datasets_available,
    )


def _report_markdown(artifacts: list[Artifact], session_dir: Path, warnings: list[str]) -> str:
    for artifact in artifacts:
        if artifact.type is ArtifactType.MARKDOWN_REPORT:
            markdown = artifact.payload.get("markdown")
            if isinstance(markdown, str):
                return markdown
    report_file = session_dir / "report" / "report.md"
    if report_file.is_file():
        try:
            return report_file.read_text(encoding="utf-8")
        except OSError:
            warnings.append("report/report.md unreadable")
    return ""


def _business_context(
    artifacts: list[Artifact],
    manifest_data: dict[str, object],
) -> str:
    for artifact in artifacts:
        if artifact.type is ArtifactType.SESSION_SUMMARY:
            context = artifact.payload.get("business_context")
            if isinstance(context, str):
                return context
    context = manifest_data.get("business_context")
    return context if isinstance(context, str) else ""


def _safe_manifest(store: ArtifactStore, project_id: str, session_id: str) -> dict[str, object]:
    try:
        manifest = store.read_manifest(project_id, session_id)
    except (OSError, ValueError):
        return {}
    return manifest.model_dump(mode="json") if manifest is not None else {}


def _manifest_input_hashes(manifest_data: dict[str, object]) -> dict[str, str]:
    raw_hashes = manifest_data.get("input_hashes")
    if not isinstance(raw_hashes, dict):
        return {}
    return {
        str(name): value
        for name, value in raw_hashes.items()
        if isinstance(value, str) and value
    }


def _reload_datasets(
    artifacts: list[Artifact],
    workspace: Path,
    project_id: str,
    warnings: list[str],
    *,
    input_hashes: dict[str, str],
) -> tuple[list[LoadedDataset], bool]:
    """Reload each profiled dataset from its upload; skip unavailable sources."""
    uploads_root = workspace / "projects" / project_id / "uploads"
    loaded: list[LoadedDataset] = []
    all_reloaded = True
    profiled = 0
    for artifact in artifacts:
        if artifact.type is not ArtifactType.DATASET_PROFILE:
            continue
        profiled += 1
        dataset_id = artifact.payload.get("dataset_id")
        name = artifact.payload.get("name")
        if not isinstance(dataset_id, str) or not isinstance(name, str):
            all_reloaded = False
            warnings.append("profile missing dataset_id/name; source not reloaded")
            continue
        source = _locate_source(uploads_root / dataset_id / "v1", name)
        if source is None:
            all_reloaded = False
            warnings.append(
                f"source data for {name} unavailable; chat/relationships disabled for it"
            )
            continue
        try:
            loaded.append(
                load_csv(
                    source,
                    dataset_id=dataset_id,
                    content_hash=input_hashes.get(name),
                )
            )
        except (OSError, ValueError):
            all_reloaded = False
            warnings.append(
                f"source data for {name} unavailable; chat/relationships disabled for it"
            )
    # No profiled datasets => nothing to reconstruct; treat as unavailable.
    datasets_available = all_reloaded and profiled > 0
    return loaded, datasets_available


def _locate_source(version_dir: Path, name: str) -> Path | None:
    exact = version_dir / name
    if exact.is_file():
        return exact
    if not version_dir.is_dir():
        return None
    # The recorded name may differ from what landed on disk; accept a lone csv.
    csvs = sorted(version_dir.glob("*.csv"))
    if len(csvs) == 1:
        return csvs[0]
    return None
