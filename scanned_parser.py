from __future__ import annotations

from math import atan2, degrees
from statistics import median

from text_pipeline import (
    BBoxItem,
    assign_row_to_columns,
    assign_items_to_boundaries,
    build_column_names,
    cluster_rows,
    detect_column_boundaries,
    detect_columns,
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
    row_index = 1

    for segment in segmented_rows:
        alignment_rows = _choose_alignment_rows(segment)
        boundaries = detect_column_boundaries(alignment_rows)
        column_centers = _boundaries_to_centers(boundaries, alignment_rows)
        segment_column_centers.append(column_centers)
        for row in segment:
            aligned_row = assign_row_to_columns(row, column_centers, boundaries=boundaries)
            grouped_items = assign_items_to_boundaries(row, boundaries)
            aligned_rows.append(aligned_row)

            for col_index, cell_items in enumerate(grouped_items, start=1):
                if cell_items:
                    cells.append(
                        {
                            "row_index": row_index,
                            "col_index": col_index,
                            "text": " ".join(item.text for item in cell_items).strip(),
                            "x0": min(item.x0 for item in cell_items),
                            "y0": min(item.y0 for item in cell_items),
                            "x1": max(item.x1 for item in cell_items),
                            "y1": max(item.y1 for item in cell_items),
                            "source_item_indexes": [item_indexes[id(item)] for item in cell_items],
                        }
                    )
                else:
                    cells.append(
                        {
                            "row_index": row_index,
                            "col_index": col_index,
                            "text": "",
                            "x0": None,
                            "y0": None,
                            "x1": None,
                            "y1": None,
                            "source_item_indexes": [],
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
                    "text": "",
                    "x0": None,
                    "y0": None,
                    "x1": None,
                    "y1": None,
                    "source_item_indexes": [],
                }
            )

    return {
        "slant_angle": slant_angle,
        "rows": rows,
        "table_rows": rows,
        "column_centers": segment_column_centers,
        "column_names": column_names,
        "aligned_rows": normalized_rows,
        "band": (band_start, band_end),
        "cells": sorted(cells, key=lambda cell: (cell["row_index"], cell["col_index"])),
    }
