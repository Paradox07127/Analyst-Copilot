from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

# Project-level semantic seeds (decision D6). User-confirmed joins and entity
# notes persist outside the run tree so chat planning and question discovery can
# reuse them across sessions. Stored at <project_dir>/semantic/seeds.json.


class VerifiedRelation(BaseModel):
    """One user-confirmed join, keyed on the (left, right) column pair."""

    left: str  # "dataset.column"
    right: str  # "dataset.column"
    cardinality: str
    confirmed_by: str = "user"
    confirmed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_session_id: str | None = None


class EntityNote(BaseModel):
    name: str
    note: str


class FieldMeaning(BaseModel):
    """A business meaning pinned to one column of one dataset (FR-15)."""

    dataset: str
    column: str
    meaning: str
    unit: str | None = None
    aliases: list[str] = Field(default_factory=list)


class MetricDefinition(BaseModel):
    """A named metric the user vouches for, so analyses agree on its meaning."""

    name: str
    definition: str
    formula: str | None = None
    caveats: str | None = None


class VerifiedAnswer(BaseModel):
    """A question the user has answered and blessed, for reuse as ground truth."""

    question: str
    answer: str
    evidence_note: str | None = None
    verified_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CategoryLevelMeaning(BaseModel):
    """What one observed value of a categorical column stands for.

    Column-level meaning is not enough for coded categories: knowing that
    ``status`` is an order state does not say whether ``C`` is Cancelled or
    Completed, and every downstream reading of that column depends on it.
    """

    dataset: str
    column: str
    value: str
    meaning: str


class ColumnRoleSeed(BaseModel):
    """Store a human-pinned role for one column."""

    dataset: str
    column: str
    role: str
    note: str = ""


class SemanticSeeds(BaseModel):
    version: int = 1
    verified_relations: list[VerifiedRelation] = Field(default_factory=list)
    entity_notes: list[EntityNote] = Field(default_factory=list)
    # Defaults keep older seed files loadable when these knowledge classes are absent.
    field_meanings: list[FieldMeaning] = Field(default_factory=list)
    category_level_meanings: list[CategoryLevelMeaning] = Field(default_factory=list)
    metric_definitions: list[MetricDefinition] = Field(default_factory=list)
    verified_answers: list[VerifiedAnswer] = Field(default_factory=list)
    # Human-pinned roles take precedence over inference.
    column_role_seeds: list[ColumnRoleSeed] = Field(default_factory=list)


def _metric_pin_line(metric: MetricDefinition) -> str:
    parts = [f"- {metric.name}: {metric.definition}"]
    if metric.formula:
        parts.append(f" Formula: {metric.formula}.")
    if metric.caveats:
        parts.append(f" Caveats: {metric.caveats}.")
    return "".join(parts)


def _field_pin_line(field: FieldMeaning) -> str:
    line = f"- {field.dataset}.{field.column}: {field.meaning}"
    extras: list[str] = []
    if field.unit:
        extras.append(f"unit: {field.unit}")
    if field.aliases:
        extras.append("aka " + ", ".join(field.aliases))
    if extras:
        line += f" ({'; '.join(extras)})"
    return line


def pinned_context_block(seeds: SemanticSeeds, *, max_chars: int = 800) -> str:
    """Render user-pinned metric/field definitions as a compact, deterministic block."""
    metrics = sorted(
        seeds.metric_definitions,
        key=lambda metric: (metric.name.casefold(), metric.name),
    )
    fields = sorted(
        seeds.field_meanings,
        key=lambda field: (
            field.dataset.casefold(),
            field.column.casefold(),
            field.dataset,
            field.column,
        ),
    )
    levels = sorted(
        seeds.category_level_meanings,
        key=lambda level: (
            level.dataset.casefold(),
            level.column.casefold(),
            level.value.casefold(),
        ),
    )
    entries: list[tuple[str, str]] = [("Metrics", _metric_pin_line(metric)) for metric in metrics]
    entries += [("Field meanings", _field_pin_line(field)) for field in fields]
    entries += [
        (
            "Category values",
            f"- {level.dataset}.{level.column} = {level.value}: {level.meaning}",
        )
        for level in levels
    ]
    total = len(entries)
    if total == 0:
        return ""

    # Reserve worst-case room for the omission note so a truncated block never
    # exceeds max_chars (the digit count of N is at most that of ``total``).
    note_reserve = len(f"\n… ({total} more definitions omitted)")
    out: list[str] = []
    current_section: str | None = None
    included = 0
    for section, line in entries:
        candidate = list(out)
        new_section = section != current_section
        if new_section:
            candidate.append(f"{section}:")
        candidate.append(line)
        will_omit_more = (included + 1) < total
        projected = len("\n".join(candidate)) + (note_reserve if will_omit_more else 0)
        if projected > max_chars:
            if out:
                break
            # Even the first entry overflows: truncate its text into the budget
            # so the block never exceeds max_chars (regression: a single huge
            # seed used to be included unconditionally).
            overflow = projected - max_chars
            candidate[-1] = line[: max(0, len(line) - overflow - 1)] + "…"
            out = candidate
            included += 1
            break
        out = candidate
        included += 1
        if new_section:
            current_section = section
    if included < total:
        out.append(f"… ({total - included} more definitions omitted)")
    return "\n".join(out)


# Join authorization is persisted as structured state and enforced by the SQL guard.

JoinStatus = Literal["proposed", "confirmed", "auto_confirmed"]

# Statuses that templates and execution may consume.
_USABLE_STATUSES: frozenset[str] = frozenset({"confirmed", "auto_confirmed"})


class JoinWhitelistEntry(BaseModel):
    """One directed join hypothesis with its lifecycle state."""

    left_dataset: str
    left_dataset_id: str | None = None
    left_columns: list[str]
    right_dataset: str
    right_dataset_id: str | None = None
    right_columns: list[str]
    # Cardinality prevents unsafe many-to-many templates.
    cardinality: str = "unknown"
    join_row_multiplier: float | None = None
    confidence_source: str = "relationship_discovery"
    # New confirmations require a persisted referential/cardinality audit.
    validation_verified: bool = False
    status: JoinStatus = "proposed"
    # Low-quality proposals sort last and cannot be auto-confirmed.
    quality: Literal["normal", "low"] = "normal"
    proposed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    confirmed_at: datetime | None = None
    confirmed_by: str = ""
    usage_count: int = 0

    def label(self) -> str:
        left = f"{self.left_dataset}.{'+'.join(self.left_columns)}"
        right = f"{self.right_dataset}.{'+'.join(self.right_columns)}"
        return f"{left} -> {right}"

    def validation_freshness(
        self,
        dataset_ids_by_name: Mapping[str, str],
    ) -> Literal["fresh", "stale", "unverifiable"]:
        """Compare persisted validation inputs with the current run's data."""
        if (
            not self.validation_verified
            or self.left_dataset_id is None
            or self.right_dataset_id is None
        ):
            return "unverifiable"
        current_left = dataset_ids_by_name.get(self.left_dataset)
        current_right = dataset_ids_by_name.get(self.right_dataset)
        if current_left is None or current_right is None:
            return "unverifiable"
        if current_left != self.left_dataset_id or current_right != self.right_dataset_id:
            return "stale"
        return "fresh"

    def is_usable(
        self,
        dataset_ids_by_name: Mapping[str, str] | None = None,
    ) -> bool:
        """True when templates/execution may consume this entry."""
        if self.status not in _USABLE_STATUSES:
            return False
        if dataset_ids_by_name is None:
            return True
        return (
            self.validation_freshness(dataset_ids_by_name) == "fresh"
            and self.cardinality != "many_to_many"
        )


class JoinWhitelist(BaseModel):
    """The project's structured join whitelist (artifact-shaped JSON store)."""

    version: int = 2
    entries: list[JoinWhitelistEntry] = Field(default_factory=list)

    def entry(self, label: str) -> JoinWhitelistEntry | None:
        for entry in self.entries:
            if entry.label() == label:
                return entry
        return None

    def usable_labels(
        self,
        dataset_ids_by_name: Mapping[str, str] | None = None,
    ) -> set[str]:
        """Labels templates may consume and the execution guard may pass."""
        return {entry.label() for entry in self.entries if entry.is_usable(dataset_ids_by_name)}

    def confirmed_labels(
        self,
        dataset_ids_by_name: Mapping[str, str] | None = None,
    ) -> set[str]:
        """Alias of :meth:`usable_labels` (auto_confirmed counts as confirmed)."""
        return self.usable_labels(dataset_ids_by_name)

    def auto_confirmed_labels(self) -> set[str]:
        """Labels currently in the machine-confirmed state (for UI/disclosure)."""
        return {entry.label() for entry in self.entries if entry.status == "auto_confirmed"}

    def entries_for(self, datasets: set[str]) -> list[JoinWhitelistEntry]:
        """Return entries whose two datasets are both in the requested scope."""
        return [
            entry
            for entry in self.entries
            if entry.left_dataset in datasets and entry.right_dataset in datasets
        ]

    def validation_freshness_counts(
        self,
        dataset_ids_by_name: Mapping[str, str],
    ) -> dict[str, int]:
        """Low-cardinality runtime health summary for relevant entries."""
        counts = {"fresh": 0, "stale": 0, "unverifiable": 0}
        for entry in self.entries_for(set(dataset_ids_by_name)):
            counts[entry.validation_freshness(dataset_ids_by_name)] += 1
        return counts

    def disclosure_notes(self, labels: Iterable[str] | None = None) -> list[str]:
        """Return disclosure text for machine-confirmed joins."""
        wanted = None if labels is None else set(labels)
        notes = [
            (
                f"Join {entry.label()} was auto-confirmed (high confidence); "
                "review it on the Knowledge page."
            )
            for entry in self.entries
            if entry.status == "auto_confirmed" and (wanted is None or entry.label() in wanted)
        ]
        return sorted(notes)

    def merge_proposals(self, proposals: Iterable[JoinWhitelistEntry]) -> int:
        """Add new proposals, idempotent on label; returns how many were new."""
        self.version = 2
        known = {entry.label(): entry for entry in self.entries}
        added = 0
        for proposal in proposals:
            existing = known.get(proposal.label())
            if existing is not None:
                # A fresh full validation supersedes machine-derived facts. A
                # confirmation is bound to the validated dataset identities:
                # it must never be silently transferred to changed uploads.
                if proposal.validation_verified:
                    identity_changed = (
                        existing.left_dataset_id != proposal.left_dataset_id
                        or existing.right_dataset_id != proposal.right_dataset_id
                    )
                    existing.left_dataset_id = proposal.left_dataset_id
                    existing.right_dataset_id = proposal.right_dataset_id
                    existing.cardinality = proposal.cardinality
                    existing.join_row_multiplier = proposal.join_row_multiplier
                    existing.validation_verified = True
                    existing.quality = proposal.quality
                    existing.confidence_source = proposal.confidence_source
                    if identity_changed:
                        existing.status = proposal.status
                        existing.confirmed_at = proposal.confirmed_at
                        existing.confirmed_by = proposal.confirmed_by
                        existing.proposed_at = proposal.proposed_at
                        existing.usage_count = 0
                    elif (
                        existing.status == "auto_confirmed" and proposal.status != "auto_confirmed"
                    ):
                        existing.status = "proposed"
                        existing.confirmed_at = None
                        existing.confirmed_by = ""
                continue
            self.entries.append(proposal)
            known[proposal.label()] = proposal
            added += 1
        return added


def join_whitelist_path(project_dir: Path | str) -> Path:
    """Where a project's join whitelist lives; callers must not rebuild this path."""
    return Path(project_dir) / "semantic" / "join_whitelist.json"


def load_join_whitelist(project_dir: Path | str) -> JoinWhitelist:
    """Load the project join whitelist; a missing file yields an empty one."""
    path = join_whitelist_path(project_dir)
    if not path.exists():
        return JoinWhitelist()
    return JoinWhitelist.model_validate_json(path.read_text(encoding="utf-8"))


def save_join_whitelist(project_dir: Path | str, whitelist: JoinWhitelist) -> Path:
    """Atomically persist the whitelist (same pattern as ``save_seeds``)."""
    path = join_whitelist_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(whitelist.model_dump_json(indent=2), encoding="utf-8")
    os.replace(tmp_path, path)
    return path


def confirm_join(
    project_dir: Path | str,
    label: str,
    *,
    confirmed_by: str = "user",
) -> JoinWhitelist:
    """Confirm a proposed or machine-confirmed join."""
    whitelist = load_join_whitelist(project_dir)
    entry = whitelist.entry(label)
    if entry is None:
        raise ValueError(f"Unknown join whitelist label: {label}")
    if entry.status == "proposed" and not entry.validation_verified:
        raise ValueError(
            f"Join whitelist entry requires full validation before confirmation: {label}"
        )
    entry.status = "confirmed"
    entry.confirmed_at = datetime.now(UTC)
    entry.confirmed_by = confirmed_by
    save_join_whitelist(project_dir, whitelist)
    return whitelist


def revoke_auto_confirmation(project_dir: Path | str, label: str) -> JoinWhitelist:
    """Return a machine-confirmed join to the proposed state."""
    whitelist = load_join_whitelist(project_dir)
    entry = whitelist.entry(label)
    if entry is None:
        raise ValueError(f"Unknown join whitelist label: {label}")
    if entry.status != "auto_confirmed":
        raise ValueError(
            f"Only auto_confirmed entries can be revoked; {label!r} is {entry.status!r}"
        )
    entry.status = "proposed"
    entry.confirmed_at = None
    entry.confirmed_by = ""
    save_join_whitelist(project_dir, whitelist)
    return whitelist


def record_join_usage(project_dir: Path | str, label: str) -> JoinWhitelist:
    """Record a successful execution over a whitelisted join."""
    whitelist = load_join_whitelist(project_dir)
    entry = whitelist.entry(label)
    if entry is not None:
        entry.usage_count += 1
        save_join_whitelist(project_dir, whitelist)
    return whitelist
