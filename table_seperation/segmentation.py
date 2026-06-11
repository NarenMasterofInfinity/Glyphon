from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
import math
import re
from statistics import median
from typing import Any, Callable, Iterable, Mapping, Sequence


class BoundaryDecision(str, Enum):
    SAME_TABLE = "SAME_TABLE"
    DIFFERENT_TABLE = "DIFFERENT_TABLE"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True)
class Cell:
    text: str
    page: int
    x1: float
    y1: float
    x2: float
    y2: float
    source_index: int = -1

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    def output(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "page": self.page,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
        }


@dataclass
class SegmentationConfig:
    remove_empty_cells: bool = True
    min_cell_width: float = 0.1
    min_cell_height: float = 0.1
    row_tolerance_factor: float = 0.65
    column_tolerance_factor: float = 0.8
    max_horizontal_gap_factor: float = 2.5
    max_vertical_gap_factor: float = 2.4
    cell_edge_threshold: float = 0.58
    nearby_vertical_gap_factor: float = 12.0
    nearby_horizontal_gap_factor: float = 8.0
    grid_match_tolerance_factor: float = 1.5
    same_table_score: float = 2.25
    different_table_score: float = -2.25
    high_confidence_llm: float = 0.8
    cross_page_gap_factor: float = 8.0
    max_sample_rows: int = 5


@dataclass
class RowGroup:
    page: int
    y: float
    cells: list[Cell]

    @property
    def values(self) -> list[str]:
        return [cell.text for cell in sorted(self.cells, key=lambda cell: cell.x1)]


@dataclass
class Component:
    component_id: int
    cells: list[Cell]
    rows: list[RowGroup] = field(default_factory=list)
    columns: list[float] = field(default_factory=list)
    bbox: tuple[float, float, float, float] = (0, 0, 0, 0)
    type_pattern: tuple[str, ...] = ()
    header_guess: list[str] = field(default_factory=list)
    sparse_row_indexes: list[int] = field(default_factory=list)
    section_row_indexes: list[int] = field(default_factory=list)
    total_row_indexes: list[int] = field(default_factory=list)
    text_density: float = 0.0

    @property
    def pages(self) -> tuple[int, ...]:
        return tuple(sorted({cell.page for cell in self.cells}))

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


@dataclass
class PairDecision:
    component_a: int
    component_b: int
    decision: BoundaryDecision
    confidence: float
    score: float
    relationship: str
    gap: float
    reasons: list[str]


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _value(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _normalize_cell(raw: Any, source_index: int) -> Cell | None:
    text = str(_value(raw, "text", "") or "").strip()
    page = int(_value(raw, "page", _value(raw, "page_number", 1)))
    if _value(raw, "x2") is not None:
        left, top = _value(raw, "x1"), _value(raw, "y1")
        right, bottom = _value(raw, "x2"), _value(raw, "y2")
    else:
        left, top = _value(raw, "x0"), _value(raw, "y0")
        right, bottom = _value(raw, "x1"), _value(raw, "y1")
    if None in (left, top, right, bottom):
        return None
    x1, x2 = sorted((float(left), float(right)))
    y1, y2 = sorted((float(top), float(bottom)))
    return Cell(text, page, x1, y1, x2, y2, source_index)


def normalize_cells(raw_cells: Iterable[Any], config: SegmentationConfig) -> list[Cell]:
    cells: list[Cell] = []
    for index, raw in enumerate(raw_cells):
        cell = _normalize_cell(raw, index)
        if cell is None:
            continue
        if config.remove_empty_cells and not cell.text:
            continue
        if cell.width < config.min_cell_width or cell.height < config.min_cell_height:
            continue
        cells.append(cell)
    return sorted(cells, key=lambda cell: (cell.page, cell.y1, cell.x1, cell.y2, cell.x2))


def parser_pages_to_cells(pages: Iterable[Any]) -> list[dict[str, Any]]:
    """Adapt the existing parser output to the segmenter's public cell schema.

    Raw parser items are preferred because they retain every positioned token.
    Parser cells are used as a fallback for callers that omit raw_items.
    """
    result: list[dict[str, Any]] = []
    for page in pages:
        page_number = int(_value(page, "page_number", 1))
        raw_items = list(_value(page, "raw_items", []) or [])
        sources = raw_items if raw_items else list(_value(page, "cells", []) or [])
        for source in sources:
            x0, y0 = _value(source, "x0"), _value(source, "y0")
            x1, y1 = _value(source, "x1"), _value(source, "y1")
            if None in (x0, y0, x1, y1):
                continue
            result.append(
                {
                    "text": str(_value(source, "text", "") or ""),
                    "page": page_number,
                    "x1": float(x0),
                    "y1": float(y0),
                    "x2": float(x1),
                    "y2": float(y1),
                }
            )
    return result


def _overlap(a1: float, a2: float, b1: float, b2: float) -> float:
    return max(0.0, min(a2, b2) - max(a1, b1))


def _ratio_overlap(a1: float, a2: float, b1: float, b2: float) -> float:
    return _overlap(a1, a2, b1, b2) / max(0.1, min(a2 - a1, b2 - b1))


def _axis_gap(a1: float, a2: float, b1: float, b2: float) -> float:
    return max(0.0, max(a1, b1) - min(a2, b2))


def _cell_edge_score(a: Cell, b: Cell, config: SegmentationConfig, scale: float) -> float:
    if a.page != b.page:
        return 0.0
    x_overlap = _ratio_overlap(a.x1, a.x2, b.x1, b.x2)
    y_overlap = _ratio_overlap(a.y1, a.y2, b.y1, b.y2)
    x_gap = _axis_gap(a.x1, a.x2, b.x1, b.x2)
    y_gap = _axis_gap(a.y1, a.y2, b.y1, b.y2)
    row_aligned = y_overlap >= 0.35 or abs(a.cy - b.cy) <= config.row_tolerance_factor * scale
    col_aligned = x_overlap >= 0.25 or abs(a.cx - b.cx) <= config.column_tolerance_factor * scale

    horizontal = row_aligned and x_gap <= config.max_horizontal_gap_factor * scale
    vertical = col_aligned and y_gap <= config.max_vertical_gap_factor * scale
    if not horizontal and not vertical:
        return 0.0
    if horizontal:
        return 0.48 + 0.32 * max(y_overlap, 0.5) + 0.2 * (1 - x_gap / (config.max_horizontal_gap_factor * scale))
    return 0.46 + 0.34 * max(x_overlap, 0.35) + 0.2 * (1 - y_gap / (config.max_vertical_gap_factor * scale))


def _initial_components(cells: list[Cell], config: SegmentationConfig) -> list[list[Cell]]:
    if not cells:
        return []
    heights = [cell.height for cell in cells if cell.height > 0]
    scale = max(1.0, median(heights))
    graph = UnionFind(len(cells))
    by_page: dict[int, list[int]] = defaultdict(list)
    for index, cell in enumerate(cells):
        by_page[cell.page].append(index)
    for indexes in by_page.values():
        for offset, left_index in enumerate(indexes):
            left = cells[left_index]
            for right_index in indexes[offset + 1 :]:
                right = cells[right_index]
                if right.y1 - left.y2 > config.max_vertical_gap_factor * scale and abs(right.cy - left.cy) > scale:
                    break
                if _cell_edge_score(left, right, config, scale) >= config.cell_edge_threshold:
                    graph.union(left_index, right_index)
    grouped: dict[int, list[Cell]] = defaultdict(list)
    for index, cell in enumerate(cells):
        grouped[graph.find(index)].append(cell)
    return list(grouped.values())


def _cluster_rows(cells: Sequence[Cell], config: SegmentationConfig) -> list[RowGroup]:
    scale = max(1.0, median([cell.height for cell in cells]))
    rows: list[RowGroup] = []
    for cell in sorted(cells, key=lambda item: (item.page, item.cy, item.x1)):
        candidates = [
            row
            for row in rows
            if row.page == cell.page and abs(row.y - cell.cy) <= config.row_tolerance_factor * scale
        ]
        if candidates:
            row = min(candidates, key=lambda candidate: abs(candidate.y - cell.cy))
            row.cells.append(cell)
            row.y = sum(item.cy for item in row.cells) / len(row.cells)
        else:
            rows.append(RowGroup(cell.page, cell.cy, [cell]))
    return sorted(rows, key=lambda row: (row.page, row.y))


def _cluster_positions(values: Sequence[float], tolerance: float) -> list[float]:
    groups: list[list[float]] = []
    for value in sorted(values):
        if groups and abs(value - sum(groups[-1]) / len(groups[-1])) <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [round(sum(group) / len(group), 3) for group in groups]


_NUMBER = re.compile(r"^[($+-]?\d[\d,]*(?:\.\d+)?%?\)?$")
_DATE = re.compile(r"^(?:\d{1,4}[-/.]){1,2}\d{1,4}$")
_TOTAL = re.compile(r"\b(total|subtotal|balance|net|grand total)\b", re.I)


def _text_type(text: str) -> str:
    compact = text.strip()
    if not compact:
        return "empty"
    if _NUMBER.match(compact):
        return "number"
    if _DATE.match(compact):
        return "date"
    return "string"


def _infer_component(component_id: int, cells: list[Cell], config: SegmentationConfig) -> Component:
    rows = _cluster_rows(cells, config)
    median_width = max(1.0, median([cell.width for cell in cells]))
    columns = _cluster_positions([cell.x1 for cell in cells], config.column_tolerance_factor * median_width)
    x1, y1 = min(cell.x1 for cell in cells), min(cell.y1 for cell in cells)
    x2, y2 = max(cell.x2 for cell in cells), max(cell.y2 for cell in cells)
    row_types = [tuple(_text_type(value) for value in row.values) for row in rows]
    pattern_length = max((len(pattern) for pattern in row_types), default=0)
    type_pattern = tuple(
        Counter(pattern[index] for pattern in row_types if index < len(pattern)).most_common(1)[0][0]
        for index in range(pattern_length)
    )
    sparse = [index for index, row in enumerate(rows) if len(row.cells) <= max(1, len(columns) // 2)]
    sections = [
        index
        for index in sparse
        if len(rows[index].cells) == 1 and _text_type(rows[index].cells[0].text) == "string"
    ]
    totals = [index for index, row in enumerate(rows) if _TOTAL.search(" ".join(row.values))]
    header_guess: list[str] = []
    for row in rows[:2]:
        values = row.values
        string_ratio = sum(_text_type(value) == "string" for value in values) / max(1, len(values))
        if string_ratio >= 0.75 and not _TOTAL.search(" ".join(values)):
            header_guess = values
            break
    occupied = sum(cell.width * cell.height for cell in cells)
    density = min(1.0, occupied / max(1.0, (x2 - x1) * (y2 - y1)))
    return Component(
        component_id=component_id,
        cells=cells,
        rows=rows,
        columns=columns,
        bbox=(x1, y1, x2, y2),
        type_pattern=type_pattern,
        header_guess=header_guess,
        sparse_row_indexes=sparse,
        section_row_indexes=sections,
        total_row_indexes=totals,
        text_density=round(density, 4),
    )


def _position_similarity(left: Sequence[float], right: Sequence[float], tolerance: float) -> float:
    if not left or not right:
        return 0.0
    matches = sum(any(abs(value - candidate) <= tolerance for candidate in right) for value in left)
    return matches / max(len(left), len(right))


def _type_similarity(left: Sequence[str], right: Sequence[str]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    return sum(left[index] == right[index] for index in range(size)) / max(len(left), len(right))


def _relationship(a: Component, b: Component) -> tuple[str, float]:
    if max(a.pages) < min(b.pages):
        return "next_page", float(min(b.pages) - max(a.pages))
    x_overlap = _ratio_overlap(a.bbox[0], a.bbox[2], b.bbox[0], b.bbox[2])
    y_overlap = _ratio_overlap(a.bbox[1], a.bbox[3], b.bbox[1], b.bbox[3])
    if y_overlap > 0.25:
        return "side_by_side", _axis_gap(a.bbox[0], a.bbox[2], b.bbox[0], b.bbox[2])
    if x_overlap > 0.15:
        return "vertical", _axis_gap(a.bbox[1], a.bbox[3], b.bbox[1], b.bbox[3])
    return "diagonal", math.hypot(
        _axis_gap(a.bbox[0], a.bbox[2], b.bbox[0], b.bbox[2]),
        _axis_gap(a.bbox[1], a.bbox[3], b.bbox[1], b.bbox[3]),
    )


def _score_pair(a: Component, b: Component, config: SegmentationConfig, scale: float) -> PairDecision:
    relationship, gap = _relationship(a, b)
    tolerance = config.grid_match_tolerance_factor * scale
    grid_similarity = _position_similarity(a.columns, b.columns, tolerance)
    width_similarity = min(a.width, b.width) / max(1.0, max(a.width, b.width))
    type_similarity = _type_similarity(a.type_pattern, b.type_pattern)
    reasons: list[str] = []
    score = 0.0

    if relationship == "side_by_side":
        y_alignment = _ratio_overlap(a.bbox[1], a.bbox[3], b.bbox[1], b.bbox[3])
        row_count_similarity = min(len(a.rows), len(b.rows)) / max(1, max(len(a.rows), len(b.rows)))
        strong_side_gap = gap >= 6 * scale
        edge_alignment = (
            abs(a.bbox[1] - b.bbox[1]) <= 1.5 * scale
            and abs(a.bbox[3] - b.bbox[3]) <= 1.5 * scale
        )
        column_fragments = (
            len(a.columns) == len(b.columns) == 1
            and y_alignment >= 0.75
            and row_count_similarity >= 0.75
            and edge_alignment
            and not strong_side_gap
        )
        if column_fragments:
            score += 2.8
            reasons.append(
                f"aligned single-column fragments: y={y_alignment:.2f}, rows={row_count_similarity:.2f}"
            )
        else:
            score -= 3.0
            if strong_side_gap:
                reasons.append("strong side-by-side whitespace gap")
            else:
                reasons.append("side-by-side structured blocks preserve separate 2D regions")
    elif relationship == "diagonal":
        score -= 1.25
        reasons.append("diagonal displacement")
    elif relationship == "next_page":
        score += 0.5
        reasons.append("adjacent-page continuation candidate")
    else:
        proximity = max(0.0, 1 - gap / (config.nearby_vertical_gap_factor * scale))
        score += 0.8 * proximity
        reasons.append(f"vertical proximity={proximity:.2f}")

    if grid_similarity >= 0.75:
        score += 1.8
        reasons.append(f"strong column-grid match={grid_similarity:.2f}")
    elif grid_similarity <= 0.35:
        score -= 1.7
        reasons.append(f"column-grid change={grid_similarity:.2f}")
    if width_similarity >= 0.8:
        score += 0.8
        reasons.append(f"similar width={width_similarity:.2f}")
    elif width_similarity <= 0.5:
        score -= 0.8
        reasons.append(f"width change={width_similarity:.2f}")
    if type_similarity >= 0.7:
        score += 0.8
        reasons.append(f"type continuity={type_similarity:.2f}")
    elif type_similarity <= 0.3:
        score -= 0.6
        reasons.append(f"type-pattern change={type_similarity:.2f}")

    b_has_header = bool(b.header_guess) and len(b.rows) > 1
    repeated_header = b_has_header and a.header_guess == b.header_guess
    distinct_header = b_has_header and bool(a.header_guess) and a.header_guess != b.header_guess
    meaningful_vertical_gap = relationship == "vertical" and gap >= 4 * scale
    large_vertical_gap = relationship == "vertical" and gap >= 6 * scale
    if meaningful_vertical_gap and distinct_header:
        score -= 1.5
        reasons.append("meaningful whitespace and distinct fresh header")
    if large_vertical_gap:
        score -= 1.5
        reasons.append("large whitespace gap")
    if repeated_header and relationship == "next_page":
        score += 0.7
        reasons.append("repeated header supports page continuation")
    elif b_has_header and relationship != "next_page":
        score -= 1.1
        reasons.append("next block starts with header evidence")
    if a.total_row_indexes and a.total_row_indexes[-1] >= len(a.rows) - 1:
        score -= 1.0
        reasons.append("previous block ends with total/footer evidence")
    if len(b.columns) == 1 and len(b.rows) <= 2:
        score += 0.2
        reasons.append("small one-column block treated as possible note/continuation")

    if meaningful_vertical_gap and distinct_header:
        decision = BoundaryDecision.DIFFERENT_TABLE
        reasons.append("distinct header and whitespace jointly establish a boundary")
    elif large_vertical_gap and b_has_header:
        decision = BoundaryDecision.DIFFERENT_TABLE
        reasons.append("large whitespace and new header jointly establish a boundary")
    elif score >= config.same_table_score:
        decision = BoundaryDecision.SAME_TABLE
    elif score <= config.different_table_score:
        decision = BoundaryDecision.DIFFERENT_TABLE
    else:
        decision = BoundaryDecision.UNCERTAIN
    confidence = min(0.99, 0.5 + abs(score) / 7)
    return PairDecision(a.component_id, b.component_id, decision, confidence, score, relationship, gap, reasons)


def _component_summary(component: Component, decision: PairDecision) -> dict[str, Any]:
    samples = [row.values for row in component.rows[:5]]
    return {
        "component_id": component.component_id,
        "header_guess": component.header_guess,
        "sample_rows": samples,
        "row_count": len(component.rows),
        "column_count": len(component.columns),
        "column_x_positions": component.columns,
        "type_pattern": component.type_pattern,
        "pages": component.pages,
        "bbox": component.bbox,
        "spatial_relationship": decision.relationship,
        "gap_size": decision.gap,
        "sparse_row_indexes": component.sparse_row_indexes,
        "section_label_row_indexes": component.section_row_indexes,
        "total_row_indexes": component.total_row_indexes,
    }


def judge_uncertain_boundary(component_a_summary: dict, component_b_summary: dict) -> dict:
    """Default optional-hook contract.

    Replace or wrap this function with an LLM client. The deterministic pipeline
    does not call it unless it is explicitly passed as ``llm_judge``.
    """
    del component_a_summary, component_b_summary
    return {
        "decision": BoundaryDecision.SAME_TABLE.value,
        "confidence": 0.0,
        "reason": "No LLM boundary judge configured; conservative merge preferred.",
    }


def _candidate_pairs(components: Sequence[Component], config: SegmentationConfig, scale: float) -> list[tuple[Component, Component]]:
    pairs: list[tuple[Component, Component]] = []
    for index, left in enumerate(components):
        for right in components[index + 1 :]:
            relationship, gap = _relationship(left, right)
            adjacent_pages = max(left.pages) + 1 == min(right.pages)
            if relationship == "next_page" and not adjacent_pages:
                continue
            if relationship == "vertical" and gap <= config.nearby_vertical_gap_factor * scale:
                pairs.append((left, right))
            elif relationship == "side_by_side" and gap <= config.nearby_horizontal_gap_factor * scale:
                pairs.append((left, right))
            elif relationship == "diagonal" and gap <= config.nearby_vertical_gap_factor * scale:
                pairs.append((left, right))
            elif adjacent_pages:
                pairs.append((left, right))
    return pairs


def _table_output(
    table_id: int,
    component_ids: list[int],
    cells: list[Cell],
    decisions: Sequence[PairDecision],
    config: SegmentationConfig,
) -> dict[str, Any]:
    inferred = _infer_component(table_id, cells, config)
    relevant = [
        decision
        for decision in decisions
        if decision.component_a in component_ids or decision.component_b in component_ids
    ]
    merges = [
        f"{decision.component_a}<->{decision.component_b}: " + "; ".join(decision.reasons)
        for decision in relevant
        if decision.decision == BoundaryDecision.SAME_TABLE
    ]
    splits = [
        f"{decision.component_a}<->{decision.component_b}: " + "; ".join(decision.reasons)
        for decision in relevant
        if decision.decision == BoundaryDecision.DIFFERENT_TABLE
    ]
    confidence_values = [decision.confidence for decision in relevant] or [0.75]
    boundary_decisions = [
        {
            "component_a": decision.component_a,
            "component_b": decision.component_b,
            "decision": decision.decision.value,
            "confidence": round(decision.confidence, 3),
            "score": round(decision.score, 3),
            "relationship": decision.relationship,
            "gap": round(decision.gap, 3),
            "reasons": decision.reasons,
        }
        for decision in relevant
    ]
    return {
        "table_id": table_id,
        "page_start": min(cell.page for cell in cells),
        "page_end": max(cell.page for cell in cells),
        "bbox": [round(value, 3) for value in inferred.bbox],
        "cells": [cell.output() for cell in sorted(cells, key=lambda item: (item.page, item.y1, item.x1))],
        "rows": [
            {"page": row.page, "y": round(row.y, 3), "values": row.values}
            for row in inferred.rows
        ],
        "columns": inferred.columns,
        "confidence": round(sum(confidence_values) / len(confidence_values), 3),
        "debug": {
            "components_merged": component_ids,
            "split_reasons": splits,
            "merge_reasons": merges,
            "boundary_decisions": boundary_decisions,
            "structure": {
                "row_count": len(inferred.rows),
                "column_count": len(inferred.columns),
                "width": round(inferred.width, 3),
                "height": round(inferred.height, 3),
                "text_density": inferred.text_density,
                "type_pattern": list(inferred.type_pattern),
                "header_guess": inferred.header_guess,
                "sparse_rows": inferred.sparse_row_indexes,
                "section_label_rows": inferred.section_row_indexes,
                "total_rows": inferred.total_row_indexes,
            },
        },
    }


def segment_tables(
    raw_cells: Iterable[Any],
    config: SegmentationConfig | None = None,
    llm_judge: Callable[[dict, dict], dict] | None = None,
) -> list[dict[str, Any]]:
    config = config or SegmentationConfig()
    cells = normalize_cells(raw_cells, config)
    if not cells:
        return []
    scale = max(1.0, median([cell.height for cell in cells]))
    components = [
        _infer_component(index, group, config)
        for index, group in enumerate(_initial_components(cells, config))
    ]
    decisions: list[PairDecision] = []

    for left, right in _candidate_pairs(components, config, scale):
        decision = _score_pair(left, right, config, scale)
        if decision.decision == BoundaryDecision.UNCERTAIN and llm_judge is not None:
            response = llm_judge(_component_summary(left, decision), _component_summary(right, decision))
            llm_decision = str(response.get("decision", "")).upper()
            llm_confidence = float(response.get("confidence", 0.0))
            decision.reasons.append(f"LLM: {response.get('reason', 'no reason')}")
            if llm_confidence >= config.high_confidence_llm and llm_decision in BoundaryDecision.__members__:
                decision.decision = BoundaryDecision(llm_decision)
                decision.confidence = llm_confidence
        if decision.decision == BoundaryDecision.UNCERTAIN:
            decision.decision = BoundaryDecision.SAME_TABLE
            decision.reasons.append("low-confidence uncertainty conservatively merged")
            decision.confidence = min(decision.confidence, 0.65)
        decisions.append(decision)

    merge_graph = UnionFind(len(components))
    same_decisions = [decision for decision in decisions if decision.decision == BoundaryDecision.SAME_TABLE]
    relationship_priority = {"side_by_side": 0, "next_page": 1, "vertical": 2, "diagonal": 3}
    same_decisions.sort(
        key=lambda decision: (relationship_priority.get(decision.relationship, 4), -decision.score)
    )
    for decision in same_decisions:
        left_root = merge_graph.find(decision.component_a)
        right_root = merge_graph.find(decision.component_b)
        if left_root == right_root:
            continue
        contradictory = [
            split
            for split in decisions
            if split.decision == BoundaryDecision.DIFFERENT_TABLE
            and {
                merge_graph.find(split.component_a),
                merge_graph.find(split.component_b),
            }
            == {left_root, right_root}
        ]
        if contradictory and decision.relationship in {"vertical", "diagonal"}:
            decision.decision = BoundaryDecision.DIFFERENT_TABLE
            decision.reasons.append(
                f"merge blocked by {len(contradictory)} contradictory split edge(s) between assembled blocks"
            )
            decision.confidence = max(decision.confidence, 0.75)
            continue
        merge_graph.union(decision.component_a, decision.component_b)

    grouped: dict[int, list[Component]] = defaultdict(list)
    for component in components:
        grouped[merge_graph.find(component.component_id)].append(component)
    outputs = []
    ordered_groups = sorted(
        grouped.values(),
        key=lambda group: min((cell.page, cell.y1, cell.x1) for component in group for cell in component.cells),
    )
    for table_id, group in enumerate(ordered_groups, start=1):
        group_cells = [cell for component in group for cell in component.cells]
        outputs.append(
            _table_output(table_id, [component.component_id for component in group], group_cells, decisions, config)
        )
    return outputs


def segment_parser_pages(
    pages: Iterable[Any],
    config: SegmentationConfig | None = None,
    llm_judge: Callable[[dict, dict], dict] | None = None,
) -> list[dict[str, Any]]:
    return segment_tables(parser_pages_to_cells(pages), config=config, llm_judge=llm_judge)
