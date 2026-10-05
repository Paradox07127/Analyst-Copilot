"""Validate content identity before publishing repeated artifact envelopes."""

from collections.abc import Iterable

from eda_platform.core.ids import stable_hash
from eda_platform.core.provenance import env_digest
from eda_platform.schemas.artifacts import Artifact


def artifact_content_digest(artifact: Artifact) -> str:
    """Match persisted semantics, including provenance stamped by ArtifactStore."""
    value = artifact.model_dump(mode="json", exclude={"created_at"})
    if value["env_digest"] is None:
        value["env_digest"] = env_digest()
    return stable_hash(value, length=32)


def unique_artifacts(artifacts: Iterable[Artifact]) -> list[Artifact]:
    """Keep first equivalent output; never silently replace conflicting evidence."""
    unique: dict[str, Artifact] = {}
    digests: dict[str, str] = {}
    for artifact in artifacts:
        digest = artifact_content_digest(artifact)
        if artifact.id in digests and digests[artifact.id] != digest:
            raise ValueError(f"Conflicting content for artifact {artifact.id!r}.")
        if artifact.id not in unique:
            unique[artifact.id] = artifact
            digests[artifact.id] = digest
    return list(unique.values())
