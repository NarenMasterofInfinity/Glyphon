from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
import json

import fitz

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
    table_index: int
    layout_region_index: int
    assignment_score: float
    alternatives: list[dict]
    issue_ids: list[str]


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
    row_table_indexes: list[int]
    row_layout_region_indexes: list[int]
    boundary_metadata: list[dict]
    assignments: list[dict]
    issues: list[dict]


def _extract_page_words(page: fitz.Page, page_number: int) -> list[BBoxItem]:
    """Extract native PDF words as the same bbox items consumed by the OCR parser."""
    items: list[BBoxItem] = []
    for word in page.get_text("words", sort=True):
        x0, y0, x1, y1, text = word[:5]
        cleaned_text = str(text).strip()
        if not cleaned_text:
            continue
        items.append(
            BBoxItem(
                text=cleaned_text,
                confidence=1.0,
                x0=float(x0),
                y0=float(y0),
                x1=float(x1),
                y1=float(y1),
                page_number=page_number,
                quad=[
                    (float(x0), float(y0)),
                    (float(x1), float(y0)),
                    (float(x1), float(y1)),
                    (float(x0), float(y1)),
                ],
            )
        )
    return items


def parse_pdf_pages(
    pdf_bytes: bytes,
    dpi: int = 200,
    page_numbers: list[int] | None = None,
) -> list[PageExtractionResult]:
    """Parse searchable PDF text while preserving the existing parser contract.

    ``dpi`` is accepted for drop-in compatibility with the OCR parser and is
    intentionally unused because native PDF coordinates are resolution independent.
    """
    del dpi
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    results: list[PageExtractionResult] = []

    if page_numbers:
        target_pages = [page_number for page_number in page_numbers if 1 <= page_number <= len(doc)]
    else:
        target_pages = list(range(1, len(doc) + 1))

    try:
        for page_number in target_pages:
            page = doc[page_number - 1]
            raw_items = _extract_page_words(page, page_number)
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
                        CellExtraction(page_number=page_number, **cell)
                        for cell in extracted["cells"]
                    ],
                    row_table_indexes=extracted["row_table_indexes"],
                    row_layout_region_indexes=extracted["row_layout_region_indexes"],
                    boundary_metadata=extracted["boundary_metadata"],
                    assignments=extracted["assignments"],
                    issues=extracted["issues"],
                )
            )
    finally:
        doc.close()

    return results


def _get_column_names(page: PageExtractionResult) -> list[str]:
    if page.column_names:
        return list(page.column_names)
    max_cols = max((len(row) for row in page.rows), default=1)
    return [f"col_{index}" for index in range(1, max_cols + 1)]


def build_table_records(pages: list[PageExtractionResult]) -> tuple[list[str], list[dict]]:
    max_cols = max((len(_get_column_names(page)) for page in pages), default=0)
    headers = ["page_number", "table_index", "row_number"] + [
        f"col_{index}" for index in range(1, max_cols + 1)
    ]
    records: list[dict] = []

    for page in pages:
        for row_index, row in enumerate(page.rows, start=1):
            record = {
                "page_number": page.page_number,
                "table_index": page.row_table_indexes[row_index - 1],
                "row_number": row_index,
            }
            for col_index in range(max_cols):
                record[headers[col_index + 3]] = row[col_index] if col_index < len(row) else ""
            records.append(record)
    return headers, records


def all_issues(pages: list[PageExtractionResult]) -> list[dict]:
    return [issue for page in pages for issue in page.issues]


def all_assignments(pages: list[PageExtractionResult]) -> list[dict]:
    return [assignment for page in pages for assignment in page.assignments]


def _excel_safe_records(records: list[dict]) -> list[dict]:
    return [
        {
            key: json.dumps(value, ensure_ascii=True) if isinstance(value, (dict, list, tuple)) else value
            for key, value in record.items()
        }
        for record in records
    ]


def export_combined_table(pages: list[PageExtractionResult]) -> bytes:
    import pandas as pd

    headers, records = build_table_records(pages)
    buffer = BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        pd.DataFrame(records, columns=headers).to_excel(writer, sheet_name="tables", index=False)
        pd.DataFrame(_excel_safe_records(all_issues(pages))).to_excel(
            writer, sheet_name="extraction_issues", index=False
        )
        pd.DataFrame(_excel_safe_records(all_assignments(pages))).to_excel(
            writer, sheet_name="assignment_candidates", index=False
        )
    return buffer.getvalue()


def diagnostic_sidecar(pages: list[PageExtractionResult], operation_history: list[dict] | None = None) -> dict:
    return {
        "schema_version": "glyphon-extraction-diagnostics-v1",
        "parser_type": "native_pdf_text",
        "pages": [
            {
                "page_number": page.page_number,
                "page_width": page.page_width,
                "page_height": page.page_height,
                "slant_angle": page.slant_angle,
                "table_band": page.table_band,
                "column_centers": page.column_centers,
                "boundaries": page.boundary_metadata,
                "cells": [asdict(cell) for cell in page.cells],
                "assignments": page.assignments,
                "issues": page.issues,
                "source_items": [asdict(item) for item in page.raw_items],
            }
            for page in pages
        ],
        "operation_history": operation_history or [],
    }
