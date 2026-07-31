from __future__ import annotations

from collections.abc import Sequence

from eda_platform.schemas.relations import (
    ErDiagram,
    ErRelationRow,
    RelationshipCandidate,
    RelationshipCandidateSet,
    RelationshipValidation,
    RelationshipValidationSet,
)


def build_er_diagram(
    candidates: RelationshipCandidateSet | Sequence[RelationshipCandidate],
    validations: RelationshipValidationSet | Sequence[RelationshipValidation],
) -> ErDiagram:
    candidate_list = (
        candidates.candidates
        if isinstance(candidates, RelationshipCandidateSet)
        else list(candidates)
    )
    validation_list = (
        validations.validations
        if isinstance(validations, RelationshipValidationSet)
        else list(validations)
    )
    validation_by_label = {validation.pair.label(): validation for validation in validation_list}
    involved_columns: dict[str, tuple[str, set[str]]] = {}
    rows: list[ErRelationRow] = []

    for candidate in sorted(candidate_list, key=lambda item: item.pair.label()):
        pair = candidate.pair
        validation = validation_by_label.get(pair.label())
        cardinality = validation.cardinality if validation is not None else "unknown"
        note = "; ".join(validation.warnings) if validation is not None else ""
        rows.append(
            ErRelationRow(
                left=f"{pair.left_dataset_name}.{'+'.join(pair.left_columns)}",
                right=f"{pair.right_dataset_name}.{'+'.join(pair.right_columns)}",
                cardinality=cardinality,
                confidence=candidate.confidence,
                note=note,
            )
        )
        # Only columns on visible edges belong in graph nodes.
        if candidate.confidence == "low":
            continue
        _add_columns(
            involved_columns,
            pair.left_dataset_id,
            pair.left_dataset_name,
            pair.left_columns,
        )
        _add_columns(
            involved_columns,
            pair.right_dataset_id,
            pair.right_dataset_name,
            pair.right_columns,
        )

    # Stable port id per involved column so an edge can attach to the exact
    # referenced key row (an HTML-label cell port) instead of the whole table.
    port_ids: dict[str, dict[str, str]] = {
        dataset_id: {column: f"f{index}" for index, column in enumerate(sorted(columns))}
        for dataset_id, (_name, columns) in involved_columns.items()
    }

    # Stack each key column on its own table-label port row.
    lines = [
        "digraph er {",
        "  graph [rankdir=LR, nodesep=0.5, ranksep=1.3, splines=spline];",
        "  node [shape=plaintext];",
    ]
    for dataset_id, (dataset_name, columns) in sorted(
        involved_columns.items(),
        key=lambda item: (item[1][0], item[0]),
    ):
        title = _html_escape(dataset_name)
        cells = [f'<tr><td bgcolor="#d9e2ec" align="center"><b>{title}</b></td></tr>']
        for column in sorted(columns):
            port = port_ids[dataset_id][column]
            cells.append(
                f'<tr><td port="{port}" align="left">{_html_escape(column)}</td></tr>'
            )
        table = (
            '<table border="0" cellborder="1" cellspacing="0" cellpadding="6">'
            + "".join(cells)
            + "</table>"
        )
        lines.append(f'  "{_dot_id(dataset_id)}" [label=<{table}>];')

    for candidate in sorted(candidate_list, key=lambda item: item.pair.label()):
        if candidate.confidence == "low":
            continue
        pair = candidate.pair
        validation = validation_by_label.get(pair.label())
        cardinality = validation.cardinality if validation is not None else "unknown"
        attrs = [f'label="{_edge_escape(f"{cardinality} / {candidate.confidence}")}"']
        if candidate.confidence == "medium":
            attrs.append('style="dashed"')
        # FK side (left) exits east, PK side (right) enters west, so in the LR
        # layout the arrow runs cleanly key-row -> key-row.
        tail = _endpoint(pair.left_dataset_id, pair.left_columns, port_ids, compass="e")
        head = _endpoint(pair.right_dataset_id, pair.right_columns, port_ids, compass="w")
        lines.append(f"  {tail} -> {head} [{', '.join(attrs)}];")
    lines.append("}")
    return ErDiagram(dot_source="\n".join(lines), relations=rows)


def _endpoint(
    dataset_id: str,
    columns: Sequence[str],
    port_ids: dict[str, dict[str, str]],
    *,
    compass: str,
) -> str:
    """Attach to a single key column's port row; composite keys fall back to the box."""
    node = f'"{_dot_id(dataset_id)}"'
    ports = port_ids.get(dataset_id, {})
    if len(columns) == 1 and columns[0] in ports:
        return f"{node}:{ports[columns[0]]}:{compass}"
    return node


def _add_columns(
    involved_columns: dict[str, tuple[str, set[str]]],
    dataset_id: str,
    dataset_name: str,
    columns: Sequence[str],
) -> None:
    if dataset_id not in involved_columns:
        involved_columns[dataset_id] = (dataset_name, set())
    involved_columns[dataset_id][1].update(columns)


def _html_escape(value: str) -> str:
    """Escape text placed inside an HTML-like label cell."""
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _dot_id(value: str) -> str:
    """Escape a node id used inside a double-quoted DOT string."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _edge_escape(value: str) -> str:
    """Escape a plain double-quoted DOT string (edge label)."""
    return value.replace("\\", "\\\\").replace('"', '\\"')
