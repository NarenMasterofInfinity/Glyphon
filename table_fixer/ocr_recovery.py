from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import ceil, floor
import re
from statistics import median
from typing import Any, Callable
from uuid import uuid4

import numpy as np
from PIL import Image

from .context import ensure_profiles
from .models import CellState, DecisionRecord, IssueState, PipelineSnapshot


MIN_RECOGNITION_CONFIDENCE = 0.85
RECOVERY_DPI = 400


@dataclass
class GlyphComponent:
    page_number: int
    bbox: tuple[float, float, float, float]
    area: float

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
    table_id: str
    row_id: str
    proposed_x: float
    crop_bbox: tuple[float, float, float, float]
    target_cell_id: str | None = None
    proposed_column_id: str | None = None
    recognition_variants: list[dict[str, Any]] = field(default_factory=list)
    recognized_text: str | None = None
    confidence: float = 0.0
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


Recognizer = Callable[[Image.Image], tuple[str, float]]
ComponentDetector = Callable[[Image.Image, int], list[GlyphComponent]]


def _overlap_ratio(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = width * height
    left_area = max((left[2] - left[0]) * (left[3] - left[1]), 1e-6)
    return intersection / left_area


def _expand_bbox(
    bbox: tuple[float, float, float, float],
    margin: float,
) -> tuple[float, float, float, float]:
    return (bbox[0] - margin, bbox[1] - margin, bbox[2] + margin, bbox[3] + margin)


def _overlaps_presented_item(
    component: GlyphComponent,
    presented_boxes: list[tuple[float, float, float, float]],
) -> bool:
    """Fail closed: any contact with an existing presented item excludes recovery."""
    return any(
        _overlap_ratio(component.bbox, _expand_bbox(box, 1.5)) > 0
        for box in presented_boxes
    )


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


def detect_glyph_components(image: Image.Image, page_number: int, dpi: int = RECOVERY_DPI) -> list[GlyphComponent]:
    """Find small ink components after suppressing long horizontal and vertical rules."""
    import cv2

    scale = dpi / 72.0
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, image.width // 25), 1)),
    )
    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, image.height // 25))),
    )
    ink = cv2.subtract(binary, cv2.bitwise_or(horizontal, vertical))
    count, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    components: list[GlyphComponent] = []
    min_height = max(3, int(scale * 1.2))
    max_height = max(min_height + 1, int(scale * 24))
    max_width = max(min_height + 1, int(scale * 28))
    for label in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[label])
        if area < max(4, int(scale)) or height < min_height or height > max_height or width > max_width:
            continue
        components.append(
            GlyphComponent(
                page_number=page_number,
                bbox=(x / scale, y / scale, (x + width) / scale, (y + height) / scale),
                area=area / (scale * scale),
            )
        )
    return components


def _rapidocr_recognizer() -> Recognizer:
    from rapidocr_onnxruntime import RapidOCR

    engine = RapidOCR()

    def recognize(image: Image.Image) -> tuple[str, float]:
        result, _ = engine(np.asarray(image), use_det=False, use_cls=False, use_rec=True)
        if not result:
            return "", 0.0
        first = result[0] if isinstance(result, list) else result
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            if isinstance(first[0], str):
                return str(first[0]).strip(), float(first[1])
            if len(first) >= 3 and isinstance(first[1], str):
                return str(first[1]).strip(), float(first[2])
        return "", 0.0

    return recognize


def _crop(image: Image.Image, bbox: tuple[float, float, float, float], dpi: int) -> Image.Image:
    scale = dpi / 72.0
    pixel_box = (
        max(0, floor(bbox[0] * scale)),
        max(0, floor(bbox[1] * scale)),
        min(image.width, ceil(bbox[2] * scale)),
        min(image.height, ceil(bbox[3] * scale)),
    )
    return image.crop(pixel_box)


def _recognize_candidate(
    candidate: RecoveryCandidate,
    image: Image.Image,
    recognizer: Recognizer,
    dpi: int,
) -> None:
    crop = _crop(image, candidate.crop_bbox, dpi)
    gray = crop.convert("L")
    array = np.asarray(gray)
    threshold = int(np.median(array))
    binary = Image.fromarray(np.where(array < threshold, 0, 255).astype(np.uint8), mode="L")
    variants = []
    for name, variant in (("grayscale", gray), ("binary", binary)):
        text, confidence = recognizer(variant)
        variants.append({"variant": name, "text": text.strip(), "confidence": float(confidence)})
    candidate.recognition_variants = variants
    texts = [entry["text"] for entry in variants]
    confidences = [entry["confidence"] for entry in variants]
    if texts[0] != texts[1]:
        candidate.rejection_reason = "recognition_variants_disagree"
        return
    if not re.fullmatch(r"[A-Za-z0-9]", texts[0]):
        candidate.rejection_reason = "not_one_alphanumeric_character"
        return
    if min(confidences) < MIN_RECOGNITION_CONFIDENCE:
        candidate.rejection_reason = "recognition_confidence_below_threshold"
        return
    candidate.recognized_text = texts[0]
    candidate.confidence = min(confidences)


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
    centers: dict[str, float] = {}
    for column_id in table.column_ids:
        values = [
            (cell.bbox[0] + cell.bbox[2]) / 2
            for cell in snapshot.table_cells(table_id)
            if cell.column_id == column_id and cell.bbox
        ]
        if values:
            centers[column_id] = median(values)
    return centers


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


def _candidate_crop(component: GlyphComponent, row_band: tuple[float, float]) -> tuple[float, float, float, float]:
    width = component.bbox[2] - component.bbox[0]
    padding = max(2.0, width * 0.8)
    return (
        component.bbox[0] - padding,
        max(row_band[0], component.bbox[1] - padding),
        component.bbox[2] + padding,
        min(row_band[1], component.bbox[3] + padding),
    )


def _cluster_components(components: list[tuple[str, GlyphComponent]], tolerance: float) -> list[list[tuple[str, GlyphComponent]]]:
    clusters: list[list[tuple[str, GlyphComponent]]] = []
    for entry in sorted(components, key=lambda value: value[1].cx):
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
    recognizer: Recognizer | None = None,
    component_detector: ComponentDetector | None = None,
    page_images: dict[int, Image.Image] | None = None,
    dpi: int = RECOVERY_DPI,
) -> RecoveryResult:
    result = source_snapshot.clone(f"snapshot_ocr_recovery_{uuid4().hex[:8]}", "ocr_recovery")
    original_rows = {table_id: list(table.row_ids) for table_id, table in source_snapshot.tables.items()}
    original_columns = {table_id: list(table.column_ids) for table_id, table in source_snapshot.tables.items()}
    original_text = {cell_id: cell.text for cell_id, cell in source_snapshot.cells.items()}
    recognizer = recognizer or _rapidocr_recognizer()
    detector = component_detector or (lambda image, page_number: detect_glyph_components(image, page_number, dpi))
    pages = {page.page_number: page for page in page_results}
    images = page_images or _render_pages(pdf_bytes, set(pages), dpi)
    components_by_page = {page_number: detector(image, page_number) for page_number, image in images.items()}
    candidates: list[RecoveryCandidate] = []
    recovered_cells = 0
    recovered_columns = 0

    for table_id, table in result.tables.items():
        page = pages.get(table.page_number)
        image = images.get(table.page_number)
        if page is None or image is None:
            continue
        bands = _row_bands(result, table_id)
        centers = _column_centers(result, table_id)
        if len(bands) < 2 or not centers:
            continue
        boxes = [cell.bbox for cell in result.table_cells(table_id) if cell.bbox]
        table_left = min(box[0] for box in boxes)
        table_right = max(box[2] for box in boxes)
        lanes = _column_lanes(table.column_ids, centers, table_left, table_right)
        raw_boxes = [(item.x0, item.y0, item.x1, item.y1) for item in page.raw_items]
        presented_cell_boxes = [
            cell.bbox
            for cell in result.table_cells(table_id)
            if cell.text.strip() and cell.bbox
        ]
        presented_boxes = list(dict.fromkeys([*raw_boxes, *presented_cell_boxes]))
        unmatched: list[tuple[str, GlyphComponent]] = []
        for component in components_by_page.get(table.page_number, []):
            row_id = next((row_id for row_id, band in bands.items() if band[0] <= component.cy <= band[1]), None)
            if not row_id or component.cx < table_left - 30 or component.cx > table_right + 30:
                continue
            if _overlaps_presented_item(component, presented_boxes):
                continue
            unmatched.append((row_id, component))

        consumed: set[int] = set()
        for index, (row_id, component) in enumerate(unmatched):
            column_id = next((column_id for column_id, lane in lanes.items() if lane[0] <= component.cx <= lane[1]), None)
            target_id = f"{row_id}::{column_id}" if column_id else None
            target = result.cells.get(target_id) if target_id else None
            if not target or target.text.strip():
                continue
            candidate = RecoveryCandidate(
                candidate_id=f"recovery_candidate_{uuid4().hex[:12]}",
                page_number=table.page_number,
                table_id=table_id,
                row_id=row_id,
                proposed_x=component.cx,
                crop_bbox=_candidate_crop(component, bands[row_id]),
                target_cell_id=target_id,
                evidence={"kind": "existing_empty_cell", "component_bbox": component.bbox},
            )
            _recognize_candidate(candidate, image, recognizer, dpi)
            if candidate.recognized_text:
                target.text = candidate.recognized_text
                target.bbox = component.bbox
                target.assignment_score = candidate.confidence
                candidate.accepted = True
                consumed.add(index)
                recovered_cells += 1
                _record_recovery(result, candidate, "recover_empty_cell", [target_id])
            candidates.append(candidate)

        remaining = [entry for index, entry in enumerate(unmatched) if index not in consumed]
        row_heights = [bottom - top for top, bottom in bands.values()]
        tolerance = max(5.0, median(row_heights) * 0.45)
        for cluster in _cluster_components(remaining, tolerance):
            distinct_rows = {row_id for row_id, _ in cluster}
            row_count = len(bands)
            required = row_count if row_count == 2 else max(3, ceil(row_count * 0.6))
            cluster_x = median(component.cx for _, component in cluster)
            distance = min(abs(cluster_x - center) for center in centers.values())
            if len(distinct_rows) < required or distance <= max(7.0, median(row_heights) * 0.7):
                for row_id, component in cluster:
                    candidates.append(RecoveryCandidate(
                        candidate_id=f"recovery_candidate_{uuid4().hex[:12]}",
                        page_number=table.page_number,
                        table_id=table_id,
                        row_id=row_id,
                        proposed_x=component.cx,
                        crop_bbox=component.bbox,
                        rejection_reason="insufficient_new_column_support",
                        evidence={"support_rows": sorted(distinct_rows), "required_rows": required},
                    ))
                continue
            accepted: list[tuple[RecoveryCandidate, GlyphComponent]] = []
            for row_id, component in cluster:
                candidate = RecoveryCandidate(
                    candidate_id=f"recovery_candidate_{uuid4().hex[:12]}",
                    page_number=table.page_number,
                    table_id=table_id,
                    row_id=row_id,
                    proposed_x=cluster_x,
                    crop_bbox=_candidate_crop(component, bands[row_id]),
                    evidence={"kind": "new_column", "support_rows": sorted(distinct_rows), "required_rows": required},
                )
                _recognize_candidate(candidate, image, recognizer, dpi)
                candidates.append(candidate)
                if candidate.recognized_text:
                    accepted.append((candidate, component))
            accepted_by_row: dict[str, tuple[RecoveryCandidate, GlyphComponent]] = {}
            for entry in accepted:
                candidate = entry[0]
                previous = accepted_by_row.get(candidate.row_id)
                if previous is None or candidate.confidence > previous[0].confidence:
                    if previous is not None:
                        previous[0].recognized_text = None
                        previous[0].rejection_reason = "duplicate_component_in_row"
                    accepted_by_row[candidate.row_id] = entry
                else:
                    candidate.recognized_text = None
                    candidate.rejection_reason = "duplicate_component_in_row"
            accepted = list(accepted_by_row.values())
            accepted_rows = {candidate.row_id for candidate, _ in accepted}
            if len(accepted_rows) < required:
                for candidate, _ in accepted:
                    candidate.recognized_text = None
                    candidate.accepted = False
                    candidate.rejection_reason = "insufficient_recognized_column_support"
                continue
            new_column_id = _insert_recovered_column(result, table_id, cluster_x, accepted)
            recovered_columns += 1
            recovered_cells += len(accepted)
            for candidate, _ in accepted:
                candidate.proposed_column_id = new_column_id
                candidate.target_cell_id = f"{candidate.row_id}::{new_column_id}"
                candidate.accepted = True
            _record_recovery(
                result,
                accepted[0][0],
                "insert_recovered_column",
                [new_column_id, *[candidate.target_cell_id for candidate, _ in accepted]],
                extra={"support_rows": sorted(accepted_rows), "recovered_values": len(accepted_rows)},
            )

    for table_id, row_ids in original_rows.items():
        if result.tables[table_id].row_ids != row_ids:
            raise ValueError("OCR recovery changed existing row order.")
        if [column_id for column_id in result.tables[table_id].column_ids if column_id in original_columns[table_id]] != original_columns[table_id]:
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


def _insert_recovered_column(
    snapshot: PipelineSnapshot,
    table_id: str,
    x_position: float,
    accepted: list[tuple[RecoveryCandidate, GlyphComponent]],
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
    values = {candidate.row_id: (candidate, component) for candidate, component in accepted}
    for row_id in table.row_ids:
        candidate_component = values.get(row_id)
        candidate, component = candidate_component if candidate_component else (None, None)
        cell_id = f"{row_id}::{new_column_id}"
        snapshot.cells[cell_id] = CellState(
            cell_id=cell_id,
            row_id=row_id,
            column_id=new_column_id,
            text=candidate.recognized_text if candidate else "",
            bbox=component.bbox if component else None,
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
        issue_type="recovered_missed_glyph" if action == "recover_empty_cell" else "recovered_missing_column",
        severity="info",
        table_id=candidate.table_id,
        affected_cell_ids=[value for value in affected_ids if "::" in value],
        status="resolved",
        explanation="A missed single-character OCR value was recovered from a targeted page-image crop.",
        suggested_action="No action required.",
        evidence={"candidate_id": candidate.candidate_id, **(extra or {})},
    )
    snapshot.decisions.append(DecisionRecord(
        decision_id=f"decision_{uuid4().hex[:12]}",
        phase="ocr_recovery",
        target_id=candidate.target_cell_id or candidate.table_id,
        action=action,
        confidence=candidate.confidence,
        reason="Two targeted recognition variants agreed on one alphanumeric character.",
        payload={"candidate_id": candidate.candidate_id, "text": candidate.recognized_text, **(extra or {})},
        valid=True,
        applied=True,
        affected_ids=affected_ids,
    ))
