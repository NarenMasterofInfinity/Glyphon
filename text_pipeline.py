from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, radians, sin
from statistics import median
from typing import Any


@dataclass
class BBoxItem:
    text: str
    confidence: float
    x0: float
    y0: float
    x1: float
    y1: float
    page_number: int
    quad: list[tuple[float, float]] = field(default_factory=list)

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


def rotate_point(x: float, y: float, angle_deg: float) -> tuple[float, float]:
    theta = radians(angle_deg)
    return (
        (x * cos(theta)) - (y * sin(theta)),
        (x * sin(theta)) + (y * cos(theta)),
    )


def cluster_rows(
    items: list[BBoxItem],
    slant_angle: float = 0.0,
    y_tolerance: float | None = None,
) -> list[list[BBoxItem]]:
    if not items:
        return []

    if y_tolerance is None:
        heights = [item.height for item in items if item.height > 0]
        y_tolerance = max(6.0, median(heights) * 0.75 if heights else 8.0)

    projected = []
    for item in items:
        rx, ry = rotate_point(item.cx, item.cy, -slant_angle)
        projected.append((item, rx, ry))

    projected.sort(key=lambda row: row[2])
    rows: list[list[tuple[BBoxItem, float, float]]] = []
    current: list[tuple[BBoxItem, float, float]] = []
    anchor_y: float | None = None

    for item, rx, ry in projected:
        if anchor_y is None or abs(ry - anchor_y) <= y_tolerance:
            current.append((item, rx, ry))
            anchor_y = sum(entry[2] for entry in current) / len(current)
            continue

        rows.append(sorted(current, key=lambda entry: entry[1]))
        current = [(item, rx, ry)]
        anchor_y = ry

    if current:
        rows.append(sorted(current, key=lambda entry: entry[1]))

    return [[entry[0] for entry in row] for row in rows]


def detect_columns(rows: list[list[BBoxItem]], min_col_gap: float | None = None) -> list[float]:
    centers = sorted(item.cx for row in rows for item in row)
    if not centers:
        return []

    if min_col_gap is None:
        widths = [item.width for row in rows for item in row if item.width > 0]
        min_col_gap = max(18.0, median(widths) * 0.6 if widths else 24.0)

    clusters: list[list[float]] = [[centers[0]]]
    for center in centers[1:]:
        if center - clusters[-1][-1] <= min_col_gap:
            clusters[-1].append(center)
        else:
            clusters.append([center])

    return [sum(cluster) / len(cluster) for cluster in clusters]


def detect_column_boundaries(
    rows: list[list[BBoxItem]],
    min_gap: float | None = None,
    merge_tolerance: float | None = None,
) -> list[float]:
    if not rows:
        return []

    widths = [item.width for row in rows for item in row if item.width > 0]
    median_width = median(widths) if widths else 24.0

    if min_gap is None:
        min_gap = max(10.0, median_width * 0.22)

    if merge_tolerance is None:
        merge_tolerance = max(12.0, median_width * 0.35)

    separators: list[float] = []
    for row in rows:
        if len(row) < 2:
            continue
        ordered = sorted(row, key=lambda item: item.x0)
        for left_item, right_item in zip(ordered, ordered[1:]):
            gap = right_item.x0 - left_item.x1
            if gap >= min_gap:
                separators.append((left_item.x1 + right_item.x0) / 2)

    if not separators:
        return []

    separators.sort()
    clusters: list[list[float]] = [[separators[0]]]
    for separator in separators[1:]:
        if separator - clusters[-1][-1] <= merge_tolerance:
            clusters[-1].append(separator)
        else:
            clusters.append([separator])

    return [sum(cluster) / len(cluster) for cluster in clusters]


def detect_column_boundary_candidates(rows: list[list[BBoxItem]]) -> list[dict[str, Any]]:
    """Return accepted and rejected boundary candidates with supporting evidence."""
    if not rows:
        return []

    widths = [item.width for row in rows for item in row if item.width > 0]
    heights = [item.height for row in rows for item in row if item.height > 0]
    median_width = median(widths) if widths else 24.0
    median_height = median(heights) if heights else 10.0
    min_gap = max(10.0, median_width * 0.22)
    merge_tolerance = max(12.0, median_width * 0.35)

    separators: list[tuple[float, int, float]] = []
    for row_index, row in enumerate(rows, start=1):
        ordered = sorted(row, key=lambda item: item.x0)
        for left_item, right_item in zip(ordered, ordered[1:]):
            gap = right_item.x0 - left_item.x1
            if gap >= min_gap:
                separators.append(((left_item.x1 + right_item.x0) / 2, row_index, gap))

    separators.sort(key=lambda entry: entry[0])
    clusters: list[list[tuple[float, int, float]]] = []
    for separator in separators:
        if not clusters or separator[0] - clusters[-1][-1][0] > merge_tolerance:
            clusters.append([separator])
        else:
            clusters[-1].append(separator)

    candidates: list[dict[str, Any]] = []
    row_count = max(1, len(rows))
    for cluster in clusters:
        supporting_rows = sorted({entry[1] for entry in cluster})
        average_gap = sum(entry[2] for entry in cluster) / len(cluster)
        support_score = len(supporting_rows) / row_count
        strong_gap = average_gap >= max(median_height * 2.5, median_width * 0.8, 24.0)
        accepted = len(supporting_rows) >= 2 or strong_gap
        candidates.append(
            {
                "position": sum(entry[0] for entry in cluster) / len(cluster),
                "supporting_rows": supporting_rows,
                "support_count": len(supporting_rows),
                "support_score": support_score,
                "average_gap": average_gap,
                "accepted": accepted,
                "reason": "repeated_row_support" if len(supporting_rows) >= 2 else (
                    "strong_gap" if strong_gap else "weak_single_row_gap"
                ),
            }
        )
    return candidates


def column_intervals(boundaries: list[float], left_edge: float, right_edge: float) -> list[tuple[float, float]]:
    edges = [left_edge] + sorted(boundaries) + [right_edge]
    return list(zip(edges, edges[1:]))


def score_item_for_columns(
    item: BBoxItem,
    boundaries: list[float],
    left_edge: float,
    right_edge: float,
) -> list[dict[str, float | int]]:
    """Score every candidate column using horizontal bbox overlap and center distance."""
    intervals = column_intervals(boundaries, left_edge, right_edge)
    item_width = max(item.width, 1e-6)
    scored: list[dict[str, float | int]] = []
    for index, (start, end) in enumerate(intervals, start=1):
        overlap = max(0.0, min(item.x1, end) - max(item.x0, start))
        overlap_ratio = overlap / item_width
        interval_width = max(end - start, 1e-6)
        center = (start + end) / 2
        distance_score = max(0.0, 1.0 - abs(item.cx - center) / interval_width)
        score = (overlap_ratio * 0.8) + (distance_score * 0.2)
        scored.append(
            {
                "col_index": index,
                "score": round(score, 6),
                "overlap_ratio": round(overlap_ratio, 6),
                "center_distance": round(abs(item.cx - center), 6),
            }
        )
    return sorted(scored, key=lambda candidate: (-float(candidate["score"]), int(candidate["col_index"])))


def assign_row_to_boundaries(row: list[BBoxItem], boundaries: list[float]) -> list[str]:
    if not boundaries:
        return [" ".join(item.text for item in row)]

    cells = assign_items_to_boundaries(row, boundaries)
    return [" ".join(item.text for item in cell_items).strip() for cell_items in cells]


def assign_items_to_boundaries(row: list[BBoxItem], boundaries: list[float]) -> list[list[BBoxItem]]:
    ordered = sorted(row, key=lambda item: item.x0)
    grouped_items: list[list[BBoxItem]] = [[] for _ in range(len(boundaries) + 1)]
    for item in ordered:
        col_index = 0
        while col_index < len(boundaries) and item.cx > boundaries[col_index]:
            col_index += 1
        grouped_items[col_index].append(item)
    return grouped_items


def assign_row_to_columns(
    row: list[BBoxItem],
    column_centers: list[float],
    boundaries: list[float] | None = None,
) -> list[str]:
    if boundaries:
        return assign_row_to_boundaries(row, boundaries)

    if not column_centers:
        return [" ".join(item.text for item in row)]

    cells = [""] * len(column_centers)
    for item in sorted(row, key=lambda entry: entry.x0):
        col_index = min(
            range(len(column_centers)),
            key=lambda idx: abs(item.cx - column_centers[idx]),
        )
        if cells[col_index]:
            cells[col_index] = f"{cells[col_index]} {item.text}".strip()
        else:
            cells[col_index] = item.text.strip()
    return cells


def build_column_names(column_count: int) -> list[str]:
    count = max(1, column_count)
    return [f"col_{index}" for index in range(1, count + 1)]
