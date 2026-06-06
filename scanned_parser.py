from __future__ import annotations

from math import atan2, degrees
from statistics import median

from text_pipeline import (
    BBoxItem,
    build_column_names,
    cluster_rows,
    detect_column_boundary_candidates,
    detect_columns,
    score_item_for_columns,
)


def estimate_slant(items: list[BBoxItem]) -> float:
    angles: list[float] = []
    for item in items:
        if len(item.quad) != 4:
            continue
        top_left, top_right, bottom_right, bottom_left = item.quad
        angles.append(degrees(atan2(top_right[1] - top_left[1], top_right[0] - top_left[0])))
        angles.append(degrees(atan2(bottom_right[1] - bottom_left[1], bottom_right[0] - bottom_left[0])))

    if not angles:
        return 0.0
    return float(median(angles))


def detect_table_band(rows: list[list[BBoxItem]], page_width: float) -> tuple[int, int]:
    if not rows:
        return (0, -1)

    substantial_indices: list[int] = []
    width_floor = page_width * 0.35

    for index, row in enumerate(rows):
        row_width = max(item.x1 for item in row) - min(item.x0 for item in row)
        has_structure = len(row) >= 2 or row_width >= width_floor
        if has_structure:
            substantial_indices.append(index)

    if not substantial_indices:
        return (0, len(rows) - 1)

    best_start = substantial_indices[0]
    best_end = substantial_indices[0]
    run_start = substantial_indices[0]
    prev = substantial_indices[0]

    for index in substantial_indices[1:]:
        if index - prev > 1:
            if prev - run_start > best_end - best_start:
                best_start, best_end = run_start, prev
            run_start = index
        prev = index

    if prev - run_start > best_end - best_start:
        best_start, best_end = run_start, prev

    return (best_start, best_end)


def _row_top(row: list[BBoxItem]) -> float:
    return min(item.y0 for item in row)


def _row_bottom(row: list[BBoxItem]) -> float:
    return max(item.y1 for item in row)


def _segment_rows(rows: list[list[BBoxItem]]) -> list[list[list[BBoxItem]]]:
    if not rows:
        return []

    row_heights = [_row_bottom(row) - _row_top(row) for row in rows]
    median_height = median(row_heights) if row_heights else 10.0
    gap_threshold = max(14.0, median_height * 1.35)

    segments: list[list[list[BBoxItem]]] = []
    current_segment = [rows[0]]
    previous_bottom = _row_bottom(rows[0])

    for row in rows[1:]:
        row_top = _row_top(row)
        if row_top - previous_bottom > gap_threshold:
            segments.append(current_segment)
            current_segment = [row]
        else:
            current_segment.append(row)
        previous_bottom = _row_bottom(row)

    if current_segment:
        segments.append(current_segment)

    return segments


def _choose_alignment_rows(segment: list[list[BBoxItem]]) -> list[list[BBoxItem]]:
    multi_item_rows = [row for row in segment if len(row) >= 2]
    return multi_item_rows or segment


def _boundaries_to_centers(boundaries: list[float], rows: list[list[BBoxItem]]) -> list[float]:
    if not boundaries:
        return detect_columns(rows)

    if not rows:
        return []

    left_edge = min(item.x0 for row in rows for item in row)
    right_edge = max(item.x1 for row in rows for item in row)
    edges = [left_edge] + list(boundaries) + [right_edge]
    return [(edges[index] + edges[index + 1]) / 2 for index in range(len(edges) - 1)]


def _layouts_compatible(previous: list[float], current: list[float], tolerance: float) -> tuple[bool, float]:
    if len(previous) != len(current):
        return False, float("inf")
    if not previous and not current:
        return True, 0.0
    differences = [abs(left - right) for left, right in zip(previous, current)]
    distance = float(median(differences)) if differences else 0.0
    return distance <= tolerance, distance


def _issue(
    issues: list[dict],
    issue_type: str,
    severity: str,
    page_number: int,
    table_index: int,
    layout_region_index: int,
    explanation: str,
    suggested_action: str,
    **details,
) -> str:
    issue_id = f"p{page_number}_issue_{len(issues) + 1}"
    issues.append(
        {
            "issue_id": issue_id,
            "issue_type": issue_type,
            "severity": severity,
            "page_number": page_number,
            "table_index": table_index,
            "layout_region_index": layout_region_index,
            "explanation": explanation,
            "suggested_action": suggested_action,
            **details,
        }
    )
    return issue_id


def _add_merged_column_issues(issues: list[dict], cells: list[dict], page_number: int) -> None:
    """Promote repeated row-level merge evidence into a column-level warning."""
    evidence_by_column: dict[tuple[int, int, int], list[dict]] = {}
    issue_lookup = {issue["issue_id"]: issue for issue in issues}

    for cell in cells:
        related = [
            issue_lookup[issue_id]
            for issue_id in cell["issue_ids"]
            if issue_id in issue_lookup
            and issue_lookup[issue_id]["issue_type"] in {"possible_cell_collision", "item_crosses_boundary"}
        ]
        if not related:
            continue

        key = (cell["table_index"], cell["layout_region_index"], cell["col_index"])
        for issue in related:
            split_positions = []
            evidence = issue.get("evidence", {})
            if evidence.get("estimated_split_x") is not None:
                split_positions.append(evidence["estimated_split_x"])
            split_positions.extend(evidence.get("crossed_boundaries", []))
            evidence_by_column.setdefault(key, []).append(
                {
                    "row_index": cell["row_index"],
                    "source_item_indexes": cell["source_item_indexes"],
                    "issue_id": issue["issue_id"],
                    "split_positions": split_positions,
                }
            )

    for (table_index, layout_region_index, col_index), entries in evidence_by_column.items():
        affected_rows = sorted({entry["row_index"] for entry in entries})
        if len(affected_rows) < 2:
            continue

        split_positions = [
            position
            for entry in entries
            for position in entry["split_positions"]
        ]
        estimated_split = float(median(split_positions)) if split_positions else None
        source_item_indexes = sorted({
            source_index
            for entry in entries
            for source_index in entry["source_item_indexes"]
        })
        supporting_issue_ids = sorted({entry["issue_id"] for entry in entries})
        issue_id = _issue(
            issues,
            "possible_merged_column",
            "warning",
            page_number,
            table_index,
            layout_region_index,
            (
                f"col_{col_index} repeatedly contains merge-like geometry across "
                f"{len(affected_rows)} rows and may represent multiple physical columns."
            ),
            "Review the affected rows and consider splitting this entire column near the estimated x-position.",
            row_index=None,
            col_index=col_index,
            source_item_indexes=source_item_indexes,
            chosen_placement={"col_index": col_index},
            alternatives=[],
            evidence={
                "affected_rows": affected_rows,
                "support_count": len(affected_rows),
                "estimated_split_x": estimated_split,
                "supporting_issue_ids": supporting_issue_ids,
            },
        )
        for cell in cells:
            if (
                cell["table_index"] == table_index
                and cell["layout_region_index"] == layout_region_index
                and cell["col_index"] == col_index
                and cell["row_index"] in affected_rows
            ):
                cell["issue_ids"].append(issue_id)


def extract_table_scanned(
    items: list[BBoxItem],
    page_width: float,
) -> dict:
    slant_angle = estimate_slant(items)
    rows = cluster_rows(items, slant_angle=slant_angle)
    band_start, band_end = detect_table_band(rows, page_width)
    segmented_rows = _segment_rows(rows)
    item_indexes = {id(item): index for index, item in enumerate(items)}

    aligned_rows: list[list[str]] = []
    cells: list[dict] = []
    segment_column_centers: list[list[float]] = []
    boundary_metadata: list[dict] = []
    assignments: list[dict] = []
    issues: list[dict] = []
    row_table_indexes: list[int] = []
    row_layout_region_indexes: list[int] = []
    row_index = 1
    table_index = 1
    previous_centers: list[float] | None = None
    item_heights = [item.height for item in items if item.height > 0]
    layout_tolerance = max(18.0, (median(item_heights) * 2.0) if item_heights else 20.0)
    page_number = items[0].page_number if items else 1

    row_centers = [median([item.cy for item in row]) for row in rows]
    row_candidate_scores: dict[int, list[dict]] = {}
    for item in items:
        candidates = [
            {"row_index": index + 1, "score": max(0.0, 1.0 - abs(item.cy - center) / max(item.height * 2.0, 12.0))}
            for index, center in enumerate(row_centers)
        ]
        row_candidate_scores[id(item)] = sorted(candidates, key=lambda entry: -entry["score"])

    for layout_region_index, segment in enumerate(segmented_rows, start=1):
        alignment_rows = _choose_alignment_rows(segment)
        candidates = detect_column_boundary_candidates(alignment_rows)
        boundaries = [candidate["position"] for candidate in candidates if candidate["accepted"]]
        column_centers = _boundaries_to_centers(boundaries, alignment_rows)
        segment_column_centers.append(column_centers)

        if previous_centers is not None:
            compatible, layout_distance = _layouts_compatible(previous_centers, column_centers, layout_tolerance)
            if not compatible:
                _issue(
                    issues,
                    "incompatible_layout_regions",
                    "warning",
                    page_number,
                    table_index,
                    layout_region_index,
                    (
                        f"Layout region {layout_region_index} has column geometry that differs from the "
                        f"preceding region, but remains in the same source table."
                    ),
                    "Use metadata/header decisions to split logical tables only when the content supports it.",
                    row_index=row_index,
                    col_index=None,
                    source_item_indexes=[],
                    chosen_placement=None,
                    alternatives=[],
                    evidence={
                        "layout_distance": None if layout_distance == float("inf") else layout_distance,
                        "column_count_changed": layout_distance == float("inf"),
                        "tolerance": layout_tolerance,
                    },
                )
        previous_centers = column_centers

        for candidate in candidates:
            metadata = {
                **candidate,
                "page_number": page_number,
                "table_index": table_index,
                "layout_region_index": layout_region_index,
            }
            boundary_metadata.append(metadata)
            if not candidate["accepted"]:
                _issue(
                    issues,
                    "weak_column_boundary",
                    "info",
                    page_number,
                    table_index,
                    layout_region_index,
                    f"Ignored a weak column boundary near x={candidate['position']:.1f}.",
                    "Review the region if a column appears merged.",
                    row_index=None,
                    col_index=None,
                    source_item_indexes=[],
                    chosen_placement=None,
                    alternatives=[],
                    evidence=metadata,
                )

        left_edge = min(item.x0 for row in alignment_rows for item in row)
        right_edge = max(item.x1 for row in alignment_rows for item in row)
        for row in segment:
            grouped_items: list[list[BBoxItem]] = [[] for _ in range(len(boundaries) + 1)]
            item_assignment_data: dict[int, dict] = {}
            for item in sorted(row, key=lambda entry: entry.x0):
                column_scores = score_item_for_columns(item, boundaries, left_edge, right_edge)
                chosen = column_scores[0]
                alternatives = [
                    {
                        **candidate,
                        "table_index": table_index,
                        "layout_region_index": layout_region_index,
                        "row_index": row_index,
                    }
                    for candidate in column_scores[1:3]
                ]
                chosen_col = int(chosen["col_index"])
                grouped_items[chosen_col - 1].append(item)
                issue_ids: list[str] = []
                margin = float(chosen["score"]) - (float(alternatives[0]["score"]) if alternatives else 0.0)
                crossing = [boundary for boundary in boundaries if item.x0 < boundary < item.x1]

                if alternatives and margin < 0.18:
                    issue_ids.append(
                        _issue(
                            issues,
                            "ambiguous_column_assignment",
                            "warning",
                            page_number,
                            table_index,
                            layout_region_index,
                            (
                                f"Placed '{item.text}' in col_{chosen_col}, but col_"
                                f"{alternatives[0]['col_index']} is also plausible; score margin is {margin:.2f}."
                            ),
                            "Review the chosen cell and its ranked alternative.",
                            row_index=row_index,
                            col_index=chosen_col,
                            source_item_indexes=[item_indexes[id(item)]],
                            chosen_placement={"row_index": row_index, "col_index": chosen_col},
                            alternatives=alternatives,
                            evidence={"score_margin": margin, "ocr_confidence": item.confidence},
                        )
                    )
                if crossing:
                    issue_ids.append(
                        _issue(
                            issues,
                            "item_crosses_boundary",
                            "warning",
                            page_number,
                            table_index,
                            layout_region_index,
                            f"'{item.text}' crosses an accepted column boundary and may be a spanning or merged cell.",
                            "Inspect the source box before splitting or moving this value.",
                            row_index=row_index,
                            col_index=chosen_col,
                            source_item_indexes=[item_indexes[id(item)]],
                            chosen_placement={"row_index": row_index, "col_index": chosen_col},
                            alternatives=alternatives,
                            evidence={"crossed_boundaries": crossing, "score_margin": margin},
                        )
                    )
                    issue_ids.append(
                        _issue(
                            issues,
                            "possible_merged_cell",
                            "info",
                            page_number,
                            table_index,
                            layout_region_index,
                            f"'{item.text}' may visually span more than one column.",
                            "Preserve as one value unless downstream evidence confirms a split.",
                            row_index=row_index,
                            col_index=chosen_col,
                            source_item_indexes=[item_indexes[id(item)]],
                            chosen_placement={"row_index": row_index, "col_index": chosen_col},
                            alternatives=alternatives,
                            evidence={"crossed_boundaries": crossing},
                        )
                    )
                if item.confidence < 0.75:
                    issue_ids.append(
                        _issue(
                            issues,
                            "low_ocr_confidence",
                            "info",
                            page_number,
                            table_index,
                            layout_region_index,
                            f"OCR confidence for '{item.text}' is {item.confidence:.2f}.",
                            "Validate the text against the source image.",
                            row_index=row_index,
                            col_index=chosen_col,
                            source_item_indexes=[item_indexes[id(item)]],
                            chosen_placement={"row_index": row_index, "col_index": chosen_col},
                            alternatives=alternatives,
                            evidence={"ocr_confidence": item.confidence},
                        )
                    )

                row_scores = row_candidate_scores[id(item)]
                row_margin = row_scores[0]["score"] - (row_scores[1]["score"] if len(row_scores) > 1 else 0.0)
                if len(row_scores) > 1 and row_margin < 0.18:
                    issue_ids.append(
                        _issue(
                            issues,
                            "ambiguous_row_assignment",
                            "warning",
                            page_number,
                            table_index,
                            layout_region_index,
                            f"'{item.text}' is close to two detected rows.",
                            "Review whether the neighboring rows should be merged or the value moved.",
                            row_index=row_index,
                            col_index=chosen_col,
                            source_item_indexes=[item_indexes[id(item)]],
                            chosen_placement={"row_index": row_index, "col_index": chosen_col},
                            alternatives=row_scores[1:3],
                            evidence={"row_score_margin": row_margin},
                        )
                    )

                assignment_score = min(float(chosen["score"]), float(row_scores[0]["score"]))
                item_assignment_data[id(item)] = {
                    "assignment_score": assignment_score,
                    "alternatives": alternatives,
                    "issue_ids": issue_ids,
                }
                assignments.append(
                    {
                        "source_item_index": item_indexes[id(item)],
                        "text": item.text,
                        "page_number": page_number,
                        "table_index": table_index,
                        "layout_region_index": layout_region_index,
                        "row_index": row_index,
                        "col_index": chosen_col,
                        "assignment_score": assignment_score,
                        "column_assignment_score": float(chosen["score"]),
                        "row_assignment_score": float(row_scores[0]["score"]),
                        "alternatives": alternatives,
                        "issue_ids": issue_ids,
                    }
                )

            aligned_row = [" ".join(item.text for item in cell_items).strip() for cell_items in grouped_items]
            aligned_rows.append(aligned_row)
            row_table_indexes.append(table_index)
            row_layout_region_indexes.append(layout_region_index)

            for col_index, cell_items in enumerate(grouped_items, start=1):
                if cell_items:
                    cell_issue_ids = sorted({
                        issue_id
                        for item in cell_items
                        for issue_id in item_assignment_data[id(item)]["issue_ids"]
                    })
                    if len(cell_items) > 1:
                        ordered_cell_items = sorted(cell_items, key=lambda entry: entry.x0)
                        item_pairs = list(zip(ordered_cell_items, ordered_cell_items[1:]))
                        gaps = [
                            right.x0 - left.x1
                            for left, right in item_pairs
                        ]
                        if gaps and max(gaps) > max(18.0, median(item_heights) * 2.0 if item_heights else 18.0):
                            largest_gap_index = gaps.index(max(gaps))
                            gap_left, gap_right = item_pairs[largest_gap_index]
                            cell_issue_ids.append(
                                _issue(
                                    issues,
                                    "possible_cell_collision",
                                    "warning",
                                    page_number,
                                    table_index,
                                    layout_region_index,
                                    "Multiple distant OCR items were concatenated into one canonical cell.",
                                    "Review whether a rejected boundary should separate these values.",
                                    row_index=row_index,
                                    col_index=col_index,
                                    source_item_indexes=[item_indexes[id(item)] for item in cell_items],
                                    chosen_placement={"row_index": row_index, "col_index": col_index},
                                    alternatives=[],
                                    evidence={
                                        "largest_internal_gap": max(gaps),
                                        "estimated_split_x": (gap_left.x1 + gap_right.x0) / 2,
                                    },
                                )
                            )
                    cells.append(
                        {
                            "row_index": row_index,
                            "col_index": col_index,
                            "table_index": table_index,
                            "layout_region_index": layout_region_index,
                            "text": " ".join(item.text for item in cell_items).strip(),
                            "x0": min(item.x0 for item in cell_items),
                            "y0": min(item.y0 for item in cell_items),
                            "x1": max(item.x1 for item in cell_items),
                            "y1": max(item.y1 for item in cell_items),
                            "source_item_indexes": [item_indexes[id(item)] for item in cell_items],
                            "assignment_score": min(item_assignment_data[id(item)]["assignment_score"] for item in cell_items),
                            "alternatives": [
                                alternative
                                for item in cell_items
                                for alternative in item_assignment_data[id(item)]["alternatives"]
                            ],
                            "issue_ids": cell_issue_ids,
                        }
                    )
                else:
                    cells.append(
                        {
                            "row_index": row_index,
                            "col_index": col_index,
                            "table_index": table_index,
                            "layout_region_index": layout_region_index,
                            "text": "",
                            "x0": None,
                            "y0": None,
                            "x1": None,
                            "y1": None,
                            "source_item_indexes": [],
                            "assignment_score": 1.0,
                            "alternatives": [],
                            "issue_ids": [],
                        }
                    )
            row_index += 1

    max_columns = max((len(row) for row in aligned_rows), default=1)
    normalized_rows = [row + ([""] * (max_columns - len(row))) for row in aligned_rows]
    column_names = build_column_names(max_columns)
    for row_number in range(1, len(normalized_rows) + 1):
        existing_cols = {
            cell["col_index"]
            for cell in cells
            if cell["row_index"] == row_number
        }
        for col_index in range(1, max_columns + 1):
            if col_index in existing_cols:
                continue
            cells.append(
                {
                    "row_index": row_number,
                    "col_index": col_index,
                    "table_index": row_table_indexes[row_number - 1],
                    "layout_region_index": row_layout_region_indexes[row_number - 1],
                    "text": "",
                    "x0": None,
                    "y0": None,
                    "x1": None,
                    "y1": None,
                    "source_item_indexes": [],
                    "assignment_score": 1.0,
                    "alternatives": [],
                    "issue_ids": [],
                }
            )

    _add_merged_column_issues(issues, cells, page_number)

    return {
        "slant_angle": slant_angle,
        "rows": rows,
        "table_rows": rows,
        "column_centers": segment_column_centers,
        "boundary_metadata": boundary_metadata,
        "column_names": column_names,
        "aligned_rows": normalized_rows,
        "row_table_indexes": row_table_indexes,
        "row_layout_region_indexes": row_layout_region_indexes,
        "band": (band_start, band_end),
        "cells": sorted(cells, key=lambda cell: (cell["row_index"], cell["col_index"])),
        "assignments": assignments,
        "issues": issues,
    }
