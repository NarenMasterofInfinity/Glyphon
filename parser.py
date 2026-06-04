from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any

import fitz
import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

from scanned_parser import extract_table_scanned
from text_pipeline import BBoxItem


@dataclass
class CellExtraction:
    page_number: int
    row_index: int
    col_index: int
    text: str
    x0: float | None
    y0: float | None
    x1: float | None
    y1: float | None
    source_item_indexes: list[int]


@dataclass
class PageExtractionResult:
    page_number: int
    page_width: float
    page_height: float
    column_names: list[str]
    rows: list[list[str]]
    raw_items: list[BBoxItem]
    slant_angle: float
    column_centers: list[list[float]]
    table_band: tuple[int, int]
    cells: list[CellExtraction]


if hasattr(fitz, "TOOLS") and hasattr(fitz.TOOLS, "set_small_glyph_heights"):
    fitz.TOOLS.set_small_glyph_heights(True)

OCR_CONFIG_CANDIDATES: list[dict[str, Any]] = [
    {
        "text_score": 0.25,
        "min_height": 8,
        "min_side_len": 16,
        "max_side_len": 4000,
        "det_limit_side_len": 1216,
        "det_limit_type": "min",
        "det_thresh": 0.2,
        "det_box_thresh": 0.35,
        "det_unclip_ratio": 1.8,
        "return_word_box": True,
    },
    {},
]


def _build_ocr_engine() -> RapidOCR:
    last_error: Exception | None = None
    for config in OCR_CONFIG_CANDIDATES:
        try:
            return RapidOCR(**config)
        except TypeError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    return RapidOCR()


def _render_page_array(page: fitz.Page, dpi: int) -> tuple[np.ndarray, float]:
    scale = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return np.array(image), scale


def _build_bbox_item(
    text: str,
    confidence: float,
    quad: list[tuple[float, float]],
    page_number: int,
) -> BBoxItem | None:
    cleaned_text = str(text).strip()
    if not cleaned_text:
        return None

    xs = [point[0] for point in quad]
    ys = [point[1] for point in quad]
    return BBoxItem(
        text=cleaned_text,
        confidence=float(confidence),
        x0=min(xs),
        y0=min(ys),
        x1=max(xs),
        y1=max(ys),
        page_number=page_number,
        quad=quad,
    )


def _extract_native_page_items(page: fitz.Page, page_number: int) -> list[BBoxItem]:
    items: list[BBoxItem] = []
    for x0, y0, x1, y1, text, *_ in page.get_text("words", sort=True):
        item = _build_bbox_item(
            text=text,
            confidence=1.0,
            quad=[
                (float(x0), float(y0)),
                (float(x1), float(y0)),
                (float(x1), float(y1)),
                (float(x0), float(y1)),
            ],
            page_number=page_number,
        )
        if item is not None:
            items.append(item)
    return items


def _ocr_result_to_items(
    result: list[list[Any]] | None,
    page_number: int,
    scale: float,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> list[BBoxItem]:
    items: list[BBoxItem] = []
    if not result:
        return items

    for entry in result:
        if len(entry) < 3:
            continue
        quad, text, score = entry[:3]
        scaled_quad = [
            ((float(x) + offset_x) / scale, (float(y) + offset_y) / scale)
            for x, y in quad
        ]
        item = _build_bbox_item(text=text, confidence=float(score), quad=scaled_quad, page_number=page_number)
        if item is not None:
            items.append(item)
    return items


def _ocr_page_image(
    image_np: np.ndarray,
    page_number: int,
    ocr: RapidOCR,
    scale: float,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> list[BBoxItem]:
    result, _ = ocr(image_np)
    return _ocr_result_to_items(result, page_number, scale, offset_x=offset_x, offset_y=offset_y)


def _tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]

    step = max(1, tile_size - overlap)
    starts = list(range(0, max(1, length - tile_size + 1), step))
    last_start = length - tile_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _ocr_page_tiles(
    image_np: np.ndarray,
    page_number: int,
    ocr: RapidOCR,
    scale: float,
    tile_size: int = 1400,
    overlap: int = 220,
) -> list[BBoxItem]:
    height, width = image_np.shape[:2]
    items: list[BBoxItem] = []
    for top in _tile_starts(height, tile_size, overlap):
        for left in _tile_starts(width, tile_size, overlap):
            tile = image_np[top : min(top + tile_size, height), left : min(left + tile_size, width)]
            if tile.size == 0:
                continue
            items.extend(_ocr_page_image(tile, page_number, ocr, scale, offset_x=left, offset_y=top))
    return items


def _same_item(left: BBoxItem, right: BBoxItem) -> bool:
    if left.text != right.text:
        return False

    horizontal_gap = max(2.0, min(left.width, right.width) * 0.35)
    vertical_gap = max(2.0, min(left.height, right.height) * 0.35)
    return (
        abs(left.x0 - right.x0) <= horizontal_gap
        and abs(left.x1 - right.x1) <= horizontal_gap
        and abs(left.y0 - right.y0) <= vertical_gap
        and abs(left.y1 - right.y1) <= vertical_gap
    )


def _merge_items(*groups: list[BBoxItem]) -> list[BBoxItem]:
    merged: list[BBoxItem] = []
    for group in groups:
        for item in sorted(group, key=lambda entry: (-entry.confidence, entry.y0, entry.x0)):
            if any(_same_item(item, existing) for existing in merged):
                continue
            merged.append(item)
    return sorted(merged, key=lambda entry: (entry.y0, entry.x0))


def parse_pdf_pages(
    pdf_bytes: bytes,
    dpi: int = 300,
    page_numbers: list[int] | None = None,
) -> list[PageExtractionResult]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    ocr = _build_ocr_engine()
    results: list[PageExtractionResult] = []

    if page_numbers:
        target_pages = [page_number for page_number in page_numbers if 1 <= page_number <= len(doc)]
    else:
        target_pages = list(range(1, len(doc) + 1))

    for page_number in target_pages:
        page_index = page_number - 1
        page = doc[page_index]
        image_np, scale = _render_page_array(page, dpi)
        native_items = _extract_native_page_items(page, page_number)
        page_ocr_items = _ocr_page_image(image_np, page_number, ocr, scale)
        tile_ocr_items = _ocr_page_tiles(image_np, page_number, ocr, scale)
        raw_items = _merge_items(native_items, page_ocr_items, tile_ocr_items)
        extracted = extract_table_scanned(raw_items, page.rect.width)
        results.append(
            PageExtractionResult(
                page_number=page_number,
                page_width=page.rect.width,
                page_height=page.rect.height,
                column_names=extracted["column_names"],
                rows=extracted["aligned_rows"],
                raw_items=raw_items,
                slant_angle=extracted["slant_angle"],
                column_centers=extracted["column_centers"],
                table_band=extracted["band"],
                cells=[
                    CellExtraction(page_number=page_index + 1, **cell)
                    for cell in extracted["cells"]
                ],
            )
        )

    doc.close()
    return results


def render_page_image(pdf_bytes: bytes, page_number: int, zoom: float = 2.0) -> Image.Image:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_number - 1]
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return image


def _get_column_names(page: PageExtractionResult) -> list[str]:
    column_names = getattr(page, "column_names", None)
    if column_names:
        return list(column_names)

    legacy_headers = getattr(page, "headers", None)
    if legacy_headers:
        return [f"col_{index}" for index in range(1, len(legacy_headers) + 1)]

    max_cols = max((len(row) for row in getattr(page, "rows", [])), default=1)
    return [f"col_{index}" for index in range(1, max_cols + 1)]


def build_table_records(pages: list[PageExtractionResult]) -> tuple[list[str], list[dict]]:
    max_cols = max((len(_get_column_names(page)) for page in pages), default=0)
    headers = ["page_number", "row_number"] + [f"col_{index}" for index in range(1, max_cols + 1)]
    records: list[dict] = []

    for page in pages:
        for row_index, row in enumerate(page.rows, start=1):
            record = {"page_number": page.page_number, "row_number": row_index}
            for col_index in range(max_cols):
                header = headers[col_index + 2]
                record[header] = row[col_index] if col_index < len(row) else ""
            records.append(record)

    return headers, records


def export_combined_table(pages: list[PageExtractionResult]) -> bytes:
    import pandas as pd

    headers, records = build_table_records(pages)
    buffer = BytesIO()
    df = pd.DataFrame(records, columns=headers)
    with pd.ExcelWriter(buffer) as writer:
        df.to_excel(writer, sheet_name="tables", index=False)
    return buffer.getvalue()
