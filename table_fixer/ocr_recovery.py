from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import ceil
import re
from statistics import median
from typing import Any, Callable
from uuid import uuid4

import numpy as np
from PIL import Image

from .context import ensure_profiles
from .models import CellState, DecisionRecord, IssueState, PipelineSnapshot


RECOVERY_DPI = 400
MIN_RECOVERY_CONFIDENCE = 0.50
GOLDEN_COVERAGE_THRESHOLD = 0.0
LENIENT_OVERLAP_THRESHOLD = 0.35


@dataclass
class LenientOCRItem:
    page_number: int
    text: str
    confidence: float
    bbox: tuple[float, float, float, float]

    @property
    def cx(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2

    @property
    def cy(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2


@dataclass
class RecoveryCandidate:
    candidate_id: str
    page_number: int
    table_id: str | None
    row_id: str | None
    text: str
    confidence: float
    bbox: tuple[float, float, float, float]
    target_cell_id: str | None = None
    proposed_column_id: str | None = None
    accepted: bool = False
    rejection_reason: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryResult:
    snapshot: PipelineSnapshot
    candidates: list[RecoveryCandidate]
    recovered_cell_count: int
    recovered_column_count: int
    rejected_candidate_count: int


LenientDetector = Callable[[Image.Image, int], list[LenientOCRItem]]


def _area(bbox: tuple[float, float, float, float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _intersection_area(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    return (
        max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
        * max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    )


def _coverage(
    candidate: tuple[float, float, float, float],
    other: tuple[float, float, float, float],
) -> float:
    return _intersection_area(candidate, other) / max(_area(candidate), 1e-6)


def _overlap_strength(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    return _intersection_area(left, right) / max(min(_area(left), _area(right)), 1e-6)


def _render_pages(pdf_bytes: bytes, page_numbers: set[int], dpi: int) -> dict[int, Image.Image]:
    import fitz

    scale = dpi / 72.0
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images: dict[int, Image.Image] = {}
    try:
        for page_number in sorted(page_numbers):
            page = doc[page_number - 1]
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            images[page_number] = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    finally:
        doc.close()
    return images


def _rapidocr_lenient_detector(dpi: int) -> LenientDetector:
    from rapidocr_onnxruntime import RapidOCR

    scale = dpi / 72.0
    engine = RapidOCR(det_db_thresh=0.10, det_db_box_thresh=0.15, text_score=0.0)

    def detect(image: Image.Image, page_number: int) -> list[LenientOCRItem]:
        result, _ = engine(np.asarray(image), use_det=True, use_cls=True, use_rec=True)
        items: list[LenientOCRItem] = []
        for entry in result or []:
            if not isinstance(entry, (list, tuple)) or len(entry) < 3:
                continue
            quad, text, confidence = entry[:3]
            cleaned = str(text).strip()
            if not cleaned:
                continue
            xs = [float(point[0]) / scale for point in quad]
            ys = [float(point[1]) / scale for point in quad]
            items.append(LenientOCRItem(
                page_number=page_number,
                text=cleaned,
                confidence=float(confidence),
                bbox=(min(xs), min(ys), max(xs), max(ys)),
            ))
        return items

    return detect


def _strongest_non_overlapping(items: list[LenientOCRItem]) -> tuple[list[LenientOCRItem], list[LenientOCRItem]]:
    """Confidence-ranked NMS: retain only the strongest item in each overlap group."""
    kept: list[LenientOCRItem] = []
    suppressed: list[LenientOCRItem] = []
    for item in sorted(items, key=lambda value: (-value.confidence, _area(value.bbox))):
        if any(_overlap_strength(item.bbox, winner.bbox) >= LENIENT_OVERLAP_THRESHOLD for winner in kept):
            suppressed.append(item)
        else:
            kept.append(item)
    return kept, suppressed


def _row_bands(snapshot: PipelineSnapshot, table_id: str) -> dict[str, tuple[float, float]]:
    table = snapshot.tables[table_id]
    centers: list[tuple[str, float, float]] = []
    for row_id in table.row_ids:
        boxes = [cell.bbox for cell in snapshot.row_cells(row_id) if cell.bbox]
        if boxes:
            centers.append((row_id, median((box[1] + box[3]) / 2 for box in boxes), median(box[3] - box[1] for box in boxes)))
    bands: dict[str, tuple[float, float]] = {}
    for index, (row_id, center, height) in enumerate(centers):
        top = (centers[index - 1][1] + center) / 2 if index else center - height
        bottom = (center + centers[index + 1][1]) / 2 if index + 1 < len(centers) else center + height
        bands[row_id] = (top, bottom)
    return bands


def _column_centers(snapshot: PipelineSnapshot, table_id: str) -> dict[str, float]:
    table = snapshot.tables[table_id]
    return {
        column_id: median(values)
        for column_id in table.column_ids
        if (values := [
            (cell.bbox[0] + cell.bbox[2]) / 2
            for cell in snapshot.table_cells(table_id)
            if cell.column_id == column_id and cell.bbox
        ])
    }


def _column_lanes(
    ordered_columns: list[str],
    centers: dict[str, float],
    table_left: float,
    table_right: float,
) -> dict[str, tuple[float, float]]:
    available = [(column_id, centers[column_id]) for column_id in ordered_columns if column_id in centers]
    lanes: dict[str, tuple[float, float]] = {}
    for index, (column_id, center) in enumerate(available):
        left = (available[index - 1][1] + center) / 2 if index else table_left
        right = (center + available[index + 1][1]) / 2 if index + 1 < len(available) else table_right
        lanes[column_id] = (left, right)
    return lanes


def _cluster_by_x(items: list[tuple[str, LenientOCRItem]], tolerance: float) -> list[list[tuple[str, LenientOCRItem]]]:
    clusters: list[list[tuple[str, LenientOCRItem]]] = []
    for entry in sorted(items, key=lambda value: value[1].cx):
        if not clusters or abs(entry[1].cx - median(item[1].cx for item in clusters[-1])) > tolerance:
            clusters.append([entry])
        else:
            clusters[-1].append(entry)
    return clusters


def recover_missed_glyphs(
    pdf_bytes: bytes,
    page_results: list[Any],
    source_snapshot: PipelineSnapshot,
    **kwargs: Any,
) -> PipelineSnapshot:
    return recover_missed_glyphs_with_result(pdf_bytes, page_results, source_snapshot, **kwargs).snapshot


def recover_missed_glyphs_with_result(
    pdf_bytes: bytes,
    page_results: list[Any],
    source_snapshot: PipelineSnapshot,
    *,
    lenient_detector: LenientDetector | None = None,
    page_images: dict[int, Image.Image] | None = None,
    dpi: int = RECOVERY_DPI,
) -> RecoveryResult:
    result = source_snapshot.clone(f"snapshot_ocr_recovery_{uuid4().hex[:8]}", "ocr_recovery")
    original_rows = {table_id: list(table.row_ids) for table_id, table in source_snapshot.tables.items()}
    original_columns = {table_id: list(table.column_ids) for table_id, table in source_snapshot.tables.items()}
    original_text = {cell_id: cell.text for cell_id, cell in source_snapshot.cells.items()}
    pages = {page.page_number: page for page in page_results}
    images = page_images or _render_pages(pdf_bytes, set(pages), dpi)
    detector = lenient_detector or _rapidocr_lenient_detector(dpi)
    candidates: list[RecoveryCandidate] = []
    recovered_cells = 0
    recovered_columns = 0

    recovered_by_page: dict[int, list[LenientOCRItem]] = {}
    for page_number, image in images.items():
        page = pages[page_number]
        golden_boxes = [(item.x0, item.y0, item.x1, item.y1) for item in page.raw_items if str(item.text).strip()]
        lenient_items = [
            item for item in detector(image, page_number)
            if item.text.strip()
            and item.confidence >= MIN_RECOVERY_CONFIDENCE
            and not any(_coverage(item.bbox, golden_box) > GOLDEN_COVERAGE_THRESHOLD for golden_box in golden_boxes)
        ]
        recovered_by_page[page_number], _ = _strongest_non_overlapping(lenient_items)

    for table_id, table in result.tables.items():
        bands = _row_bands(result, table_id)
        centers = _column_centers(result, table_id)
        boxes = [cell.bbox for cell in result.table_cells(table_id) if cell.bbox]
        if len(bands) < 2 or not centers or not boxes:
            continue
        table_left = min(box[0] for box in boxes)
        table_right = max(box[2] for box in boxes)
        lanes = _column_lanes(table.column_ids, centers, table_left, table_right)
        localized: list[tuple[str, LenientOCRItem]] = []
        for item in recovered_by_page.get(table.page_number, []):
            row_id = next((row_id for row_id, band in bands.items() if band[0] <= item.cy <= band[1]), None)
            if row_id and table_left - 30 <= item.cx <= table_right + 30:
                localized.append((row_id, item))

        consumed: set[int] = set()
        by_empty_cell: dict[str, list[tuple[int, LenientOCRItem]]] = {}
        for index, (row_id, item) in enumerate(localized):
            column_id = next((column_id for column_id, lane in lanes.items() if lane[0] <= item.cx <= lane[1]), None)
            target_id = f"{row_id}::{column_id}" if column_id else None
            target = result.cells.get(target_id) if target_id else None
            if target and not target.text.strip():
                by_empty_cell.setdefault(target_id, []).append((index, item))

        for target_id, entries in by_empty_cell.items():
            winner_index, winner = max(entries, key=lambda entry: entry[1].confidence)
            candidate = _candidate(winner, table_id, result.cells[target_id].row_id, target_id, "existing_empty_cell")
            target = result.cells[target_id]
            target.text = winner.text
            target.bbox = winner.bbox
            target.assignment_score = winner.confidence
            candidate.accepted = True
            consumed.update(index for index, _ in entries)
            candidates.append(candidate)
            recovered_cells += 1
            _record_recovery(result, candidate, "recover_empty_cell", [target_id])

        remaining = [entry for index, entry in enumerate(localized) if index not in consumed]
        row_heights = [bottom - top for top, bottom in bands.values()]
        tolerance = max(5.0, median(row_heights) * 0.45)
        for cluster in _cluster_by_x(remaining, tolerance):
            strongest_by_row: dict[str, LenientOCRItem] = {}
            for row_id, item in cluster:
                if row_id not in strongest_by_row or item.confidence > strongest_by_row[row_id].confidence:
                    strongest_by_row[row_id] = item
            row_count = len(bands)
            required = row_count if row_count == 2 else max(3, ceil(row_count * 0.6))
            cluster_x = median(item.cx for item in strongest_by_row.values())
            distance = min(abs(cluster_x - center) for center in centers.values())
            if len(strongest_by_row) < required or distance <= max(7.0, median(row_heights) * 0.7):
                continue
            accepted = [
                (_candidate(item, table_id, row_id, None, "new_column"), item)
                for row_id, item in strongest_by_row.items()
            ]
            new_column_id = _insert_recovered_column(result, table_id, cluster_x, accepted)
            recovered_columns += 1
            recovered_cells += len(accepted)
            for candidate, _ in accepted:
                candidate.proposed_column_id = new_column_id
                candidate.target_cell_id = f"{candidate.row_id}::{new_column_id}"
                candidate.accepted = True
                candidates.append(candidate)
            _record_recovery(
                result,
                accepted[0][0],
                "insert_recovered_column",
                [new_column_id, *[candidate.target_cell_id for candidate, _ in accepted]],
                extra={"support_rows": sorted(strongest_by_row), "recovered_values": len(accepted)},
            )

    for table_id, row_ids in original_rows.items():
        if result.tables[table_id].row_ids != row_ids:
            raise ValueError("OCR recovery changed existing row order.")
        existing_order = [column_id for column_id in result.tables[table_id].column_ids if column_id in original_columns[table_id]]
        if existing_order != original_columns[table_id]:
            raise ValueError("OCR recovery changed existing column order.")
    for cell_id, text in original_text.items():
        if text.strip() and result.cells[cell_id].text != text:
            raise ValueError("OCR recovery changed existing non-empty text.")

    result.recovery_audits = [asdict(candidate) for candidate in candidates]
    ensure_profiles(result)
    return RecoveryResult(
        snapshot=result,
        candidates=candidates,
        recovered_cell_count=recovered_cells,
        recovered_column_count=recovered_columns,
        rejected_candidate_count=sum(not candidate.accepted for candidate in candidates),
    )


def _candidate(
    item: LenientOCRItem,
    table_id: str,
    row_id: str,
    target_cell_id: str | None,
    kind: str,
) -> RecoveryCandidate:
    return RecoveryCandidate(
        candidate_id=f"recovery_candidate_{uuid4().hex[:12]}",
        page_number=item.page_number,
        table_id=table_id,
        row_id=row_id,
        text=item.text,
        confidence=item.confidence,
        bbox=item.bbox,
        target_cell_id=target_cell_id,
        evidence={"kind": kind},
    )


def _insert_recovered_column(
    snapshot: PipelineSnapshot,
    table_id: str,
    x_position: float,
    accepted: list[tuple[RecoveryCandidate, LenientOCRItem]],
) -> str:
    table = snapshot.tables[table_id]
    centers = _column_centers(snapshot, table_id)
    position = sum(centers.get(column_id, float("inf")) < x_position for column_id in table.column_ids)
    new_column_id = f"{table_id}_ocr_c{uuid4().hex[:8]}"
    table.column_ids.insert(position, new_column_id)
    used_names = set(table.column_names.values())
    placeholder_index = 1
    while f"col_{placeholder_index}" in used_names:
        placeholder_index += 1
    table.column_names[new_column_id] = f"col_{placeholder_index}"
    snapshot.column_lineage[new_column_id] = [new_column_id]
    values = {candidate.row_id: (candidate, item) for candidate, item in accepted}
    for row_id in table.row_ids:
        candidate_item = values.get(row_id)
        candidate, item = candidate_item if candidate_item else (None, None)
        cell_id = f"{row_id}::{new_column_id}"
        snapshot.cells[cell_id] = CellState(
            cell_id=cell_id,
            row_id=row_id,
            column_id=new_column_id,
            text=candidate.text if candidate else "",
            bbox=item.bbox if item else None,
            assignment_score=candidate.confidence if candidate else 1.0,
            ancestor_cell_ids=[cell_id],
        )
    return new_column_id


def _record_recovery(
    snapshot: PipelineSnapshot,
    candidate: RecoveryCandidate,
    action: str,
    affected_ids: list[str],
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    issue_id = f"ocr_recovery_issue_{uuid4().hex[:12]}"
    snapshot.issues[issue_id] = IssueState(
        issue_id=issue_id,
        source_issue_id=issue_id,
        issue_type="recovered_ocr_item" if action == "recover_empty_cell" else "recovered_missing_column",
        severity="info",
        table_id=candidate.table_id,
        affected_cell_ids=[value for value in affected_ids if "::" in value],
        status="resolved",
        explanation="A text item found only by the lenient OCR pass was added without changing golden OCR text.",
        suggested_action="No action required.",
        evidence={"candidate_id": candidate.candidate_id, **(extra or {})},
    )
    snapshot.decisions.append(DecisionRecord(
        decision_id=f"decision_{uuid4().hex[:12]}",
        phase="ocr_recovery",
        target_id=candidate.target_cell_id or str(candidate.table_id),
        action=action,
        confidence=candidate.confidence,
        reason="Strongest non-overlapping text item from the lenient OCR pass.",
        payload={"candidate_id": candidate.candidate_id, "text": candidate.text, **(extra or {})},
        valid=True,
        applied=True,
        affected_ids=affected_ids,
    ))
