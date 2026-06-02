from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

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


def _ocr_page(page: fitz.Page, page_number: int, ocr: RapidOCR, dpi: int) -> list[BBoxItem]:
    scale = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    image_np = np.array(image)
    result, _ = ocr(image_np)

    items: list[BBoxItem] = []
    if not result:
        return items

    for quad, text, score in result:
        scaled_quad = [(float(x) / scale, float(y) / scale) for x, y in quad]
        xs = [point[0] for point in scaled_quad]
        ys = [point[1] for point in scaled_quad]
        cleaned_text = str(text).strip()
        if not cleaned_text:
            continue
        items.append(
            BBoxItem(
                text=cleaned_text,
                confidence=float(score),
                x0=min(xs),
                y0=min(ys),
                x1=max(xs),
                y1=max(ys),
                page_number=page_number,
                quad=scaled_quad,
            )
        )
    return items


def parse_pdf_pages(pdf_bytes: bytes, dpi: int = 200) -> list[PageExtractionResult]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    ocr = RapidOCR()
    results: list[PageExtractionResult] = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        raw_items = _ocr_page(page, page_index + 1, ocr, dpi)
        extracted = extract_table_scanned(raw_items, page.rect.width)
        results.append(
            PageExtractionResult(
                page_number=page_index + 1,
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
