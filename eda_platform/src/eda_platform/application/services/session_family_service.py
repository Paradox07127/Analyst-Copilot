"""Bounded session-family and ancestry reads backed by the sessions index."""

from __future__ import annotations

from dataclasses import dataclass

from eda_platform.application.dto import CompareLineageView
from eda_platform.application.services.session_service import SessionNotFoundError
from eda_platform.core.ids import DERIVED_SESSION_PREFIXES, INTERNAL_SESSION_MARKER
from eda_platform.core.store import ArtifactStore

MAX_FAMILY_DEPTH = 32
MAX_FAMILY_SESSIONS = 512
_FORK_LIFECYCLE_PREFIX = "fksess_"


@dataclass(frozen=True)
class SessionResultFamily:
    root_session_id: str
    session_ids: tuple[str, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Ancestry:
    path: tuple[str, ...]
    complete: bool
    warnings: tuple[str, ...]


class SessionFamilyService:
    def __init__(self, store: ArtifactStore) -> None:
        self._store = store

    def collect(self, project_id: str, root_session_id: str) -> SessionResultFamily:
        self._require_project_session(project_id, root_session_id)
        visited = {root_session_id}
        ordered = [root_session_id]
        warnings: list[str] = []
        frontier = [(root_session_id, 0)]
        while frontier:
            source_session_id, depth = frontier.pop(0)
            children = self._store.query_session_children(project_id, source_session_id)
            for child in children:
                child_id = str(child["session_id"])
                # Cycle first, family membership second. The other order was
                # silent whenever a cycle closed through an ordinary root id:
                # the prefix filter skipped it before anything noticed it had
                # already been walked, so the walk stopped without saying why.
                if child_id in visited:
                    warnings.append(f"lineage cycle stopped at {child_id}")
                    continue
                if not _belongs_to_result_family(child_id):
                    continue
                if depth + 1 > MAX_FAMILY_DEPTH:
                    warnings.append(
                        f"session family exceeded depth {MAX_FAMILY_DEPTH} at {child_id}"
                    )
                    continue
                if len(visited) >= MAX_FAMILY_SESSIONS:
                    warnings.append(
                        f"session family truncated at {MAX_FAMILY_SESSIONS} sessions"
                    )
                    frontier.clear()
                    break
                visited.add(child_id)
                ordered.append(child_id)
                frontier.append((child_id, depth + 1))
        return SessionResultFamily(
            root_session_id=root_session_id,
            session_ids=tuple(ordered),
            warnings=tuple(dict.fromkeys(warnings)),
        )

    def lineage(
        self,
        project_id: str,
        left_session_id: str,
        right_session_id: str,
    ) -> CompareLineageView:
        left = self._ancestry(project_id, left_session_id)
        right = self._ancestry(project_id, right_session_id)
        warnings = list(dict.fromkeys((*left.warnings, *right.warnings)))
        common = _nearest_common_ancestor(left.path, right.path)
        if common is None:
            relation = "unrelated" if left.complete and right.complete else "unknown"
            return CompareLineageView(
                relation=relation,
                left_path=list(left.path),
                right_path=list(right.path),
                warnings=warnings,
            )

        left_to_common = left.path[: left.path.index(common) + 1]
        right_to_common = right.path[: right.path.index(common) + 1]
        if len(left_to_common) == 1 or len(right_to_common) == 1:
            relation = (
                "direct_parent"
                if max(len(left_to_common), len(right_to_common)) == 2
                else "ancestor_descendant"
            )
        else:
            relation = "siblings"
        return CompareLineageView(
            relation=relation,
            common_ancestor_session_id=common,
            left_path=list(left_to_common),
            right_path=list(right_to_common),
            warnings=warnings,
        )

    def _ancestry(self, project_id: str, session_id: str) -> _Ancestry:
        self._require_project_session(project_id, session_id)
        path = [session_id]
        visited = {session_id}
        warnings: list[str] = []
        complete = True
        current_id = session_id
        for depth in range(MAX_FAMILY_DEPTH + 1):
            row = self._store.get_session_index_row(current_id)
            if row is None or str(row["project_id"]) != project_id:
                warnings.append(f"broken lineage: session {current_id} is unavailable")
                complete = False
                break
            source_id = row.get("source_session_id")
            if not isinstance(source_id, str) or not source_id:
                break
            if depth == MAX_FAMILY_DEPTH:
                warnings.append(f"lineage exceeded depth {MAX_FAMILY_DEPTH} at {current_id}")
                complete = False
                break
            if source_id in visited:
                warnings.append(f"lineage cycle stopped at {source_id}")
                complete = False
                break
            source_row = self._store.get_session_index_row(source_id)
            if source_row is None or str(source_row["project_id"]) != project_id:
                warnings.append(
                    f"broken lineage: source {source_id} referenced by {current_id} is unavailable"
                )
                complete = False
                break
            visited.add(source_id)
            path.append(source_id)
            current_id = source_id
        return _Ancestry(tuple(path), complete, tuple(warnings))

    def _require_project_session(self, project_id: str, session_id: str) -> dict:
        row = self._store.get_session_index_row(session_id)
        if row is None or str(row["project_id"]) != project_id:
            raise SessionNotFoundError(session_id)
        return row


def _belongs_to_result_family(session_id: str) -> bool:
    if INTERNAL_SESSION_MARKER in session_id:
        return True
    return session_id.startswith(
        tuple(prefix for prefix in DERIVED_SESSION_PREFIXES if prefix != _FORK_LIFECYCLE_PREFIX)
    )


def _nearest_common_ancestor(left: tuple[str, ...], right: tuple[str, ...]) -> str | None:
    right_distance = {session_id: index for index, session_id in enumerate(right)}
    candidates = [
        (left_index + right_distance[session_id], left_index, session_id)
        for left_index, session_id in enumerate(left)
        if session_id in right_distance
    ]
    return min(candidates)[2] if candidates else None
