from __future__ import annotations

import json
import re
from statistics import mean, median
from typing import Any

from .models import CellState, IssueState, LogicalTable, PipelineSnapshot, RowProfile
from .token_counting import TokenCounter


def is_numeric(value: str) -> bool:
    cleaned = re.sub(r"[,$%()\s]", "", value)
    try:
        float(cleaned)
        return bool(cleaned)
    except ValueError:
        return False


def profile_row(snapshot: PipelineSnapshot, table: LogicalTable, row_id: str) -> RowProfile:
    cells = {cell.column_id: cell for cell in snapshot.row_cells(row_id)}
    values = [cells.get(column_id).text.strip() if cells.get(column_id) else "" for column_id in table.column_ids]
    nonempty = [value for value in values if value]
    text_values = [value for value in nonempty if not is_numeric(value)]
    numeric_values = [value for value in nonempty if is_numeric(value)]
    boxes = [cells[column_id].bbox for column_id in table.column_ids if column_id in cells and cells[column_id].bbox]
    page_width = snapshot.page_dimensions.get(table.page_number, (1.0, 1.0))[0] or 1.0
    bbox_width_ratio = (
        (max(box[2] for box in boxes) - min(box[0] for box in boxes)) / page_width if boxes else 0.0
    )
    starts = [box[0] for box in boxes]
    alignment_consistency = 1.0 if len(starts) <= 1 else max(0.0, 1.0 - ((max(starts) - min(starts)) / page_width))
    fill_ratio = len(nonempty) / max(1, len(table.column_ids))
    text_ratio = len(text_values) / max(1, len(nonempty))
    return RowProfile(
        fill_ratio=round(fill_ratio, 4),
        text_ratio=round(text_ratio, 4),
        numeric_ratio=round(len(numeric_values) / max(1, len(nonempty)), 4),
        average_text_length=round(mean(len(value) for value in nonempty), 2) if nonempty else 0.0,
        occupancy_pattern="".join("1" if value else "0" for value in values),
        following_similarity=0.0,
        bbox_width_ratio=round(bbox_width_ratio, 4),
        alignment_consistency=round(alignment_consistency, 4),
        header_candidate=fill_ratio >= 0.5 and len(nonempty) >= 2 and len(numeric_values) == 0,
    )


def ensure_profiles(snapshot: PipelineSnapshot) -> None:
    for table in snapshot.tables.values():
        profiles = [profile_row(snapshot, table, row_id) for row_id in table.row_ids]
        for index, row_id in enumerate(table.row_ids):
            profile = profiles[index]
            if index + 1 < len(profiles):
                next_profile = profiles[index + 1]
                matches = sum(
                    left == right
                    for left, right in zip(profile.occupancy_pattern, next_profile.occupancy_pattern)
                )
                profile.following_similarity = round(
                    matches / max(1, len(profile.occupancy_pattern)),
                    4,
                )
            snapshot.rows[row_id].profile = profile


def compact_row(snapshot: PipelineSnapshot, table: LogicalTable, row_id: str) -> dict[str, Any]:
    row = snapshot.rows[row_id]
    profile = row.profile or profile_row(snapshot, table, row_id)
    values = {
        cell.column_id: cell.text
        for cell in snapshot.row_cells(row_id)
        if cell.text.strip()
    }
    return {"row_id": row_id, "values": values, "profile": profile.__dict__}


def ordered_tables(snapshot: PipelineSnapshot) -> list[LogicalTable]:
    return sorted(
        snapshot.tables.values(),
        key=lambda table: (
            table.page_number,
            min((snapshot.rows[row_id].source_row_number for row_id in table.row_ids), default=10**9),
            table.table_id,
        ),
    )


def adjacent_table_pairs(snapshot: PipelineSnapshot) -> list[tuple[str, str]]:
    tables = ordered_tables(snapshot)
    return [
        (left.table_id, right.table_id)
        for left, right in zip(tables, tables[1:])
        if left.page_number == right.page_number
    ]


def _column_type_signature(snapshot: PipelineSnapshot, table: LogicalTable) -> list[str]:
    signatures = []
    for column_id in table.column_ids:
        values = [
            snapshot.cells[cell_id].text.strip()
            for row_id in table.row_ids[:8]
            if (cell_id := f"{row_id}::{column_id}") in snapshot.cells
            and snapshot.cells[cell_id].text.strip()
        ]
        if not values:
            signatures.append("empty")
            continue
        numeric_ratio = sum(is_numeric(value) for value in values) / len(values)
        signatures.append("numeric" if numeric_ratio >= 0.7 else "text" if numeric_ratio <= 0.3 else "mixed")
    return signatures


def _column_x_positions(snapshot: PipelineSnapshot, table: LogicalTable) -> list[float | None]:
    page_width = snapshot.page_dimensions.get(table.page_number, (1.0, 1.0))[0] or 1.0
    positions = []
    for column_id in table.column_ids:
        centers = [
            (cell.bbox[0] + cell.bbox[2]) / (2 * page_width)
            for cell in snapshot.table_cells(table.table_id)
            if cell.column_id == column_id and cell.bbox
        ]
        positions.append(round(mean(centers), 4) if centers else None)
    return positions


def table_pair_evidence(snapshot: PipelineSnapshot, left_id: str, right_id: str) -> dict[str, Any]:
    ensure_profiles(snapshot)
    left = snapshot.tables[left_id]
    right = snapshot.tables[right_id]
    same_count = len(left.column_ids) == len(right.column_ids)
    left_types = _column_type_signature(snapshot, left)
    right_types = _column_type_signature(snapshot, right)
    type_score = (
        sum(a == b or "empty" in {a, b} for a, b in zip(left_types, right_types)) / max(1, len(left_types))
        if same_count else 0.0
    )
    left_x = _column_x_positions(snapshot, left)
    right_x = _column_x_positions(snapshot, right)
    comparable_x = [(a, b) for a, b in zip(left_x, right_x) if a is not None and b is not None]
    geometry_score = (
        max(0.0, 1.0 - mean(abs(a - b) for a, b in comparable_x) / 0.08)
        if same_count and comparable_x else 0.0
    )
    left_patterns = [snapshot.rows[row_id].profile.occupancy_pattern for row_id in left.row_ids[-3:]]
    right_patterns = [snapshot.rows[row_id].profile.occupancy_pattern for row_id in right.row_ids[:3]]
    occupancy_score = (
        max(
            (
                sum(a == b for a, b in zip(left_pattern, right_pattern)) / max(1, len(left_pattern))
                for left_pattern in left_patterns
                for right_pattern in right_patterns
            ),
            default=0.0,
        )
        if same_count else 0.0
    )
    first_values = [
        cell.text.strip()
        for cell in snapshot.row_cells(right.row_ids[0])
        if cell.text.strip()
    ] if right.row_ids else []
    distinct_title = len(first_values) == 1 and first_values[0].lower().startswith(
        ("table ", "figure ", "appendix ")
    )
    left_boxes = [cell.bbox for cell in snapshot.table_cells(left_id) if cell.bbox]
    right_boxes = [cell.bbox for cell in snapshot.table_cells(right_id) if cell.bbox]
    heights = [box[3] - box[1] for box in left_boxes + right_boxes if box[3] > box[1]]
    boundary_gap_rows = (
        max(0.0, min(box[1] for box in right_boxes) - max(box[3] for box in left_boxes))
        / max(1.0, median(heights))
        if left_boxes and right_boxes else None
    )
    vertical_continuity_score = (
        1.0 if boundary_gap_rows is not None and boundary_gap_rows <= 2.0
        else 0.5 if boundary_gap_rows is not None and boundary_gap_rows <= 4.0
        else 0.0
    )
    score = (
        0.30 * float(same_count)
        + 0.25 * geometry_score
        + 0.20 * type_score
        + 0.10 * occupancy_score
        + 0.15 * vertical_continuity_score
        - (0.35 if distinct_title else 0.0)
    )
    return {
        "left_table_id": left_id,
        "right_table_id": right_id,
        "same_column_count": same_count,
        "left_column_count": len(left.column_ids),
        "right_column_count": len(right.column_ids),
        "left_column_types": left_types,
        "right_column_types": right_types,
        "left_column_x": left_x,
        "right_column_x": right_x,
        "geometry_score": round(geometry_score, 4),
        "type_score": round(type_score, 4),
        "occupancy_score": round(occupancy_score, 4),
        "boundary_gap_rows": round(boundary_gap_rows, 4) if boundary_gap_rows is not None else None,
        "vertical_continuity_score": vertical_continuity_score,
        "distinct_title_at_right_start": distinct_title,
        "deterministic_score": round(max(0.0, min(1.0, score)), 4),
    }


def table_reconciliation_context(snapshot: PipelineSnapshot, left_id: str, right_id: str) -> dict[str, Any]:
    left = snapshot.tables[left_id]
    right = snapshot.tables[right_id]
    return {
        "instruction": (
            "Decide whether the right table is an accidental continuation split from the left table. "
            "Keep separate when the right side begins a distinct titled table."
        ),
        "evidence": table_pair_evidence(snapshot, left_id, right_id),
        "left_tail": [compact_row(snapshot, left, row_id) for row_id in left.row_ids[-3:]],
        "right_head": [compact_row(snapshot, right, row_id) for row_id in right.row_ids[:3]],
    }


def metadata_context(snapshot: PipelineSnapshot, table_id: str) -> dict[str, Any]:
    table = snapshot.tables[table_id]
    ensure_profiles(snapshot)
    allowed_prefix = metadata_candidate_prefix(snapshot, table_id)
    limit = min(12, len(table.row_ids))
    while limit < min(32, len(table.row_ids)):
        if any(snapshot.rows[row_id].profile.header_candidate for row_id in table.row_ids[:limit]):
            break
        limit = min(limit + 8, 32, len(table.row_ids))
    return {
        "table_id": table_id,
        "instruction": "Identify only contiguous top-prefix metadata rows. Header continuations are not metadata.",
        "allowed_metadata_row_ids": allowed_prefix,
        "rows": [compact_row(snapshot, table, row_id) for row_id in table.row_ids[:limit]],
    }


def metadata_candidate_prefix(snapshot: PipelineSnapshot, table_id: str) -> list[str]:
    """Bound metadata decisions before the first stable table-like row."""
    ensure_profiles(snapshot)
    table = snapshot.tables[table_id]
    if not table.row_ids:
        return []

    profiles = [snapshot.rows[row_id].profile for row_id in table.row_ids]
    for index, profile in enumerate(profiles):
        next_profiles = profiles[index:index + 3]
        stable_structure = (
            profile.fill_ratio >= 0.66
            and len(next_profiles) >= 2
            and sum(candidate.fill_ratio >= 0.66 for candidate in next_profiles) >= 2
        )
        occupancy_transition = index > 0 and profile.occupancy_pattern != profiles[index - 1].occupancy_pattern
        if stable_structure or (occupancy_transition and profile.fill_ratio > profiles[index - 1].fill_ratio):
            return table.row_ids[:index]

    first_text = " ".join(cell.text for cell in snapshot.row_cells(table.row_ids[0]) if cell.text.strip()).lower()
    if first_text.startswith("table ") and len(table.row_ids) > 1:
        return table.row_ids[:1]
    return table.row_ids[: min(3, len(table.row_ids))]


def header_candidates(snapshot: PipelineSnapshot, table_id: str) -> list[str]:
    ensure_profiles(snapshot)
    table = snapshot.tables[table_id]
    candidates = []
    for index, row_id in enumerate(table.row_ids):
        profile = snapshot.rows[row_id].profile
        if index == 0:
            next_profile = snapshot.rows[table.row_ids[1]].profile if len(table.row_ids) > 1 else None
            first_is_plausible = (
                profile.fill_ratio >= 0.5
                or (
                    next_profile is not None
                    and (
                        profile.occupancy_pattern != next_profile.occupancy_pattern
                        or next_profile.numeric_ratio > profile.numeric_ratio
                    )
                )
            )
            if first_is_plausible:
                candidates.append(row_id)
            continue
        previous_profile = snapshot.rows[table.row_ids[index - 1]].profile
        next_profile = snapshot.rows[table.row_ids[index + 1]].profile if index + 1 < len(table.row_ids) else None
        transition = (
            next_profile is not None
            and profile.text_ratio >= 0.75
            and previous_profile.numeric_ratio > profile.numeric_ratio
            and next_profile.numeric_ratio > profile.numeric_ratio
        )
        occupancy_transition = (
            next_profile is not None
            and profile.header_candidate
            and profile.occupancy_pattern != previous_profile.occupancy_pattern
            and profile.occupancy_pattern != next_profile.occupancy_pattern
        )
        if transition or occupancy_transition:
            candidates.append(row_id)
    return candidates


def allowed_header_continuations(
    snapshot: PipelineSnapshot,
    table_id: str,
    candidate_row_id: str,
) -> list[str]:
    """Allow only an immediately following row with header-like, non-data geometry."""
    ensure_profiles(snapshot)
    table = snapshot.tables[table_id]
    index = table.row_ids.index(candidate_row_id)
    if index + 1 >= len(table.row_ids):
        return []
    next_row_id = table.row_ids[index + 1]
    profile = snapshot.rows[next_row_id].profile
    following = snapshot.rows[table.row_ids[index + 2]].profile if index + 2 < len(table.row_ids) else None
    if profile.numeric_ratio > 0:
        return []
    if following and profile.occupancy_pattern == following.occupancy_pattern:
        return []
    return [next_row_id]


def header_context(snapshot: PipelineSnapshot, table_id: str, candidate_row_id: str) -> dict[str, Any]:
    table = snapshot.tables[table_id]
    index = table.row_ids.index(candidate_row_id)
    nearby = table.row_ids[max(0, index - 2): min(len(table.row_ids), index + 4)]
    ordered = ordered_tables(snapshot)
    table_position = next(index for index, candidate in enumerate(ordered) if candidate.table_id == table_id)
    predecessor = ordered[table_position - 1] if table_position > 0 and ordered[table_position - 1].page_number == table.page_number else None
    return {
        "table_id": table_id,
        "candidate_row_id": candidate_row_id,
        "rows": [compact_row(snapshot, table, row_id) for row_id in nearby],
        "predecessor_tail_context_only": (
            [compact_row(snapshot, predecessor, row_id) for row_id in predecessor.row_ids[-2:]]
            if predecessor and index <= 1 else []
        ),
    }


def normalized_bbox(snapshot: PipelineSnapshot, table: LogicalTable, cell: CellState) -> list[float] | None:
    if not cell.bbox:
        return None
    width, height = snapshot.page_dimensions.get(table.page_number, (1.0, 1.0))
    return [
        round(cell.bbox[0] / width, 5),
        round(cell.bbox[1] / height, 5),
        round(cell.bbox[2] / width, 5),
        round(cell.bbox[3] / height, 5),
    ]


def column_split_context(snapshot: PipelineSnapshot, issue_id: str, sample_limit: int = 10) -> dict[str, Any]:
    issue = snapshot.issues[issue_id]
    if not issue.table_id or issue.table_id not in snapshot.tables:
        raise ValueError(f"Issue {issue_id} does not reference an active logical table.")
    table = snapshot.tables[issue.table_id]
    column_id = next(
        (cell.column_id for cell_id in issue.affected_cell_ids if (cell := snapshot.cells.get(cell_id))),
        issue.evidence.get("column_id"),
    )
    values = [
        snapshot.cells[cell_id]
        for cell_id in issue.affected_cell_ids
        if cell_id in snapshot.cells and snapshot.cells[cell_id].text.strip()
    ]
    if not values:
        values = [
            cell for cell in snapshot.table_cells(table.table_id)
            if cell.column_id == column_id and cell.text.strip()
        ]
    distinct = []
    seen_patterns = set()
    for cell in values:
        pattern = re.sub(r"\d", "N", re.sub(r"[A-Za-z]", "A", cell.text))
        if pattern not in seen_patterns or len(distinct) < 3:
            distinct.append(cell)
            seen_patterns.add(pattern)
        if len(distinct) >= sample_limit:
            break
    return {
        "task": (
            "Decide whether this single column consistently contains multiple fields. "
            "If it does, return a Python full-match regex with 2-4 named groups. "
            "The group names should be short field names and every sample must match."
        ),
        "current_header": table.column_names.get(column_id, column_id),
        "nearby_headers": [
            table.column_names[column]
            for column in table.column_ids
            if abs(table.column_ids.index(column) - table.column_ids.index(column_id)) <= 1
        ],
        "why_flagged": issue.explanation,
        "sample_values": [cell.text for cell in distinct],
        "_table_id": table.table_id,
        "_column_id": column_id,
    }


def placeholder_column_context(
    snapshot: PipelineSnapshot,
    table_id: str,
    column_id: str,
    sample_limit: int = 10,
) -> dict[str, Any]:
    table = snapshot.tables[table_id]
    source_position = table.column_ids.index(column_id)
    source_values = [
        snapshot.cells[cell_id].text
        for row_id in table.row_ids
        if (cell_id := f"{row_id}::{column_id}") in snapshot.cells
        and snapshot.cells[cell_id].text.strip()
    ]
    candidates = []
    for position, candidate_id in enumerate(table.column_ids):
        header = table.column_names.get(candidate_id, candidate_id)
        if candidate_id == column_id or re.fullmatch(r"col_\d+", header, re.IGNORECASE):
            continue
        relevant_rows = [
            row_id for row_id in table.row_ids
            if snapshot.cells.get(f"{row_id}::{column_id}")
            and snapshot.cells[f"{row_id}::{column_id}"].text.strip()
        ]
        occupied = sum(
            bool(snapshot.cells.get(f"{row_id}::{candidate_id}")
                 and snapshot.cells[f"{row_id}::{candidate_id}"].text.strip())
            for row_id in relevant_rows
        )
        examples = [
            snapshot.cells[cell_id].text
            for row_id in table.row_ids
            if (cell_id := f"{row_id}::{candidate_id}") in snapshot.cells
            and snapshot.cells[cell_id].text.strip()
        ][:3]
        candidates.append({
            "header": header,
            "position": "left" if position < source_position else "right",
            "distance": abs(position - source_position),
            "empty_where_placeholder_has_value": occupied == 0,
            "examples": examples,
        })
    candidates.sort(key=lambda candidate: (candidate["distance"], candidate["header"]))
    return {
        "task": (
            "Remove this placeholder column without losing data. Choose one reusable rule for the whole column. "
            "First decide whether each value is one semantic field or contains multiple semantic fields described "
            "by the candidate headers. Move only when each complete value is one field matching one named column; "
            "do not move a merged multi-field value into one column merely because that column is empty. "
            "Split when every value has the same semantic parts matching multiple empty named columns. "
            "Rename it when it is a genuine distinct field. Otherwise return unresolved."
        ),
        "placeholder_header": table.column_names.get(column_id, column_id),
        "sample_values": source_values[:sample_limit],
        "candidate_named_columns": candidates[:6],
        "examples_only_do_not_copy": [
            {
                "placeholder_values": ["Alice Engineering", "Bob Finance"],
                "empty_candidate_headers": ["Employee", "Department"],
                "decision": "split into Employee and Department",
                "why": "Each value contains a person and a department, matching two headers.",
            },
            {
                "placeholder_values": ["Face Wash", "Night Cream"],
                "empty_candidate_headers": ["Item Name", "Qty"],
                "decision": "move complete values into Item Name",
                "why": "Each value is one multiword item name; no part is a quantity.",
            },
        ],
        "response_rules": {
            "move": (
                "Use only for one semantic field. Return exactly one target_headers entry; leave regex and "
                "new_header empty."
            ),
            "split": (
                "Use when repeated values contain fields matching multiple candidate headers. Return 2-3 "
                "target_headers in capture order and a Python full-match regex with the same number of named "
                "groups; leave new_header empty."
            ),
            "rename": "Return a meaningful new_header; leave target_headers and regex empty.",
            "unresolved": "Use only when no lossless whole-column rule fits.",
        },
        "_table_id": table_id,
        "_column_id": column_id,
    }


def warning_context(snapshot: PipelineSnapshot, issue_id: str) -> dict[str, Any]:
    issue = snapshot.issues[issue_id]
    cell_id = next((value for value in issue.affected_cell_ids if value in snapshot.cells), "")
    return cell_warning_context(snapshot, cell_id, [issue])


def cell_warning_context(
    snapshot: PipelineSnapshot,
    cell_id: str,
    issues: list[IssueState],
) -> dict[str, Any]:
    source = snapshot.cells[cell_id]
    table_id = snapshot.rows[source.row_id].table_id
    table = snapshot.tables[table_id]
    source_index = table.column_ids.index(source.column_id)
    row_index = table.row_ids.index(source.row_id)

    def header(column_id: str) -> str:
        return table.column_names.get(column_id, column_id)

    relevant_column_ids = [
        column_id
        for index, column_id in enumerate(table.column_ids)
        if abs(index - source_index) <= 1
    ]
    return {
        "cell_id": cell_id,
        "issue_ids": [issue.issue_id for issue in issues],
        "source": {
            "cell_id": cell_id,
            "text": source.text,
            "header": header(source.column_id),
            "header_is_placeholder": header(source.column_id).startswith("col_"),
            "source_item_count": len(source.source_item_indexes),
        },
        "issues": [
            {
                "issue_id": issue.issue_id,
                "type": issue.issue_type,
                "severity": issue.severity,
                "explanation": issue.explanation,
            }
            for issue in issues
        ],
        "relevant_headers": [
            {
                "column_id": column_id,
                "name": header(column_id),
                "is_placeholder": header(column_id).startswith("col_"),
            }
            for column_id in relevant_column_ids
        ],
        "row_destinations": [
            {
                "cell_id": destination_id,
                "header": header(column_id),
                "header_is_placeholder": header(column_id).startswith("col_"),
                "current_text": snapshot.cells[destination_id].text,
                "distance_from_source": abs(index - source_index),
                "nearby_values": [
                    snapshot.cells[example_id].text
                    for nearby_row_id in table.row_ids[max(0, row_index - 2):row_index + 3]
                    if nearby_row_id != source.row_id
                    and (example_id := f"{nearby_row_id}::{column_id}") in snapshot.cells
                    and snapshot.cells[example_id].text.strip()
                ][:3],
            }
            for index, column_id in enumerate(table.column_ids)
            if (destination_id := f"{source.row_id}::{column_id}") in snapshot.cells
        ],
        "assignment_alternatives": [
            {
                "header": header(table.column_ids[int(alternative["col_index"]) - 1]),
                "score": alternative.get("score"),
            }
            for alternative in source.alternatives[:3]
            if isinstance(alternative, dict)
            and isinstance(alternative.get("col_index"), int)
            and 0 < alternative["col_index"] <= len(table.column_ids)
        ],
    }


def row_repair_context(
    snapshot: PipelineSnapshot,
    row_id: str,
    issues: list[IssueState],
) -> dict[str, Any]:
    table = snapshot.tables[snapshot.rows[row_id].table_id]
    named_headers: set[str] = set()
    affected_column_ids: set[str] = set()
    for issue in issues:
        expected = issue.evidence.get("expected_header")
        if isinstance(expected, str):
            named_headers.add(expected)
        named_headers.update(
            value for value in issue.evidence.get("expected_headers", [])
            if isinstance(value, str)
        )
        for cell_id in issue.affected_cell_ids:
            if cell_id in snapshot.cells and snapshot.cells[cell_id].row_id == row_id:
                affected_column_ids.add(snapshot.cells[cell_id].column_id)

    repair_column_ids = [
        column_id
        for column_id in table.column_ids
        if column_id in affected_column_ids
        or table.column_names.get(column_id, column_id) in named_headers
    ]
    if not named_headers:
        affected_indexes = [
            table.column_ids.index(column_id)
            for column_id in affected_column_ids
        ]
        repair_column_ids = [
            column_id
            for index, column_id in enumerate(table.column_ids)
            if any(abs(index - affected_index) <= 1 for affected_index in affected_indexes)
            or (
                any(abs(index - affected_index) <= 2 for affected_index in affected_indexes)
                and not snapshot.cells[f"{row_id}::{column_id}"].text.strip()
            )
        ]

    columns = []
    row_index = table.row_ids.index(row_id)
    for column_id in repair_column_ids:
        header = table.column_names.get(column_id, column_id)
        cell_id = f"{row_id}::{column_id}"
        columns.append({
            "header": header,
            "current_text": snapshot.cells[cell_id].text if cell_id in snapshot.cells else "",
            "examples_only_do_not_copy": [
                snapshot.cells[example_id].text
                for nearby_row_id in table.row_ids[max(0, row_index - 2):row_index + 3]
                if nearby_row_id != row_id
                and (example_id := f"{nearby_row_id}::{column_id}") in snapshot.cells
                and snapshot.cells[example_id].text.strip()
            ][:3],
        })
    return {
        "task": (
            "Repair only the listed columns in this one row. Arrange the available_parts under the correct headers. "
            "You may combine adjacent parts into one value. Nearby examples show column meaning only and must never "
            "be copied. Return final_values for every non-empty final value. Use every available part exactly once. "
            "Do not calculate or invent values."
        ),
        "problem": " ".join(dict.fromkeys(issue.explanation for issue in issues)),
        "available_parts": [
            part
            for column in columns
            for part in column["current_text"].split()
        ],
        "columns": columns,
        "_row_id": row_id,
        "_issue_ids": [issue.issue_id for issue in issues],
        "_column_ids_by_header": {
            table.column_names.get(column_id, column_id): column_id
            for column_id in repair_column_ids
        },
    }


def fit_context_budget(
    context: dict[str, Any],
    token_counter: TokenCounter,
    max_tokens: int,
    shrink_key: str,
) -> dict[str, Any]:
    fitted = dict(context)
    values = list(fitted.get(shrink_key, []))
    while len(values) > 1 and token_counter.count(json.dumps(fitted, separators=(",", ":"))) > max_tokens:
        values.pop()
        fitted[shrink_key] = values
    return fitted
