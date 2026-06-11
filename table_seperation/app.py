from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fitz
import pandas as pd
from PIL import Image, ImageDraw
import streamlit as st

from table_seperation import SegmentationConfig, segment_parser_pages


st.set_page_config(page_title="Glyphon Table Separation", layout="wide")
st.title("Glyphon 2D Table Separation")
st.caption("Upload a PDF, select pages, and split positioned parser cells into logical tables.")

COLORS = [
    "#dc2626",
    "#2563eb",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#be123c",
    "#4f46e5",
]


def _page_count(pdf_bytes: bytes) -> int:
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return len(document)
    finally:
        document.close()


@st.cache_data(show_spinner=False)
def _native_text_pages(pdf_bytes: bytes) -> set[int]:
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return {
            page_number
            for page_number in range(1, len(document) + 1)
            if document[page_number - 1].get_text("text").strip()
        }
    finally:
        document.close()


@st.cache_data(show_spinner=False)
def _parse_pages(pdf_bytes: bytes, pages: tuple[int, ...], parser_mode: str) -> list[Any]:
    if parser_mode == "Native PDF text":
        from text_parser import parse_pdf_pages
    else:
        from parser import parse_pdf_pages
    return parse_pdf_pages(pdf_bytes, page_numbers=list(pages))


@st.cache_data(show_spinner=False)
def _render_page(pdf_bytes: bytes, page_number: int, zoom: float) -> Image.Image:
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = document[page_number - 1]
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        return Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
    finally:
        document.close()


def _draw_tables(image: Image.Image, tables: list[dict], page_number: int, zoom: float) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output, "RGBA")
    for table in tables:
        color = COLORS[(table["table_id"] - 1) % len(COLORS)]
        rgb = tuple(int(color[index : index + 2], 16) for index in (1, 3, 5))
        page_cells = [cell for cell in table["cells"] if cell["page"] == page_number]
        if not page_cells:
            continue
        for cell in page_cells:
            box = tuple(round(cell[key] * zoom) for key in ("x1", "y1", "x2", "y2"))
            draw.rectangle(box, outline=rgb + (230,), fill=rgb + (28,), width=max(2, round(zoom)))
        x1 = min(cell["x1"] for cell in page_cells) * zoom
        y1 = min(cell["y1"] for cell in page_cells) * zoom
        x2 = max(cell["x2"] for cell in page_cells) * zoom
        y2 = max(cell["y2"] for cell in page_cells) * zoom
        draw.rectangle((x1, y1, x2, y2), outline=rgb + (255,), width=max(3, round(2 * zoom)))
        draw.rectangle((x1, max(0, y1 - 18 * zoom), x1 + 76 * zoom, y1), fill=rgb + (235,))
        draw.text((x1 + 4 * zoom, max(0, y1 - 16 * zoom)), f"Table {table['table_id']}", fill="white")
    return output


def _rows_dataframe(table: dict) -> pd.DataFrame:
    rows = table.get("rows", [])
    column_count = max((len(row["values"]) for row in rows), default=0)
    records = []
    for row in rows:
        record = {"page": row["page"], "y": row["y"]}
        record.update(
            {
                f"column_{index + 1}": row["values"][index] if index < len(row["values"]) else ""
                for index in range(column_count)
            }
        )
        records.append(record)
    return pd.DataFrame(records)


def _tables_csv(tables: list[dict]) -> bytes:
    frames = []
    for table in tables:
        frame = _rows_dataframe(table)
        frame.insert(0, "table_id", table["table_id"])
        frames.append(frame)
    if not frames:
        return b""
    return pd.concat(frames, ignore_index=True).to_csv(index=False).encode("utf-8")


def _config_controls() -> SegmentationConfig:
    with st.sidebar.expander("Segmentation thresholds"):
        edge = st.slider("Cell edge threshold", 0.30, 0.95, 0.58, 0.01)
        horizontal = st.slider("Maximum horizontal gap", 1.0, 8.0, 2.5, 0.1)
        vertical = st.slider("Maximum vertical gap", 1.0, 8.0, 2.4, 0.1)
        nearby = st.slider("Nearby component gap", 4.0, 30.0, 12.0, 0.5)
        same_score = st.slider("Same-table score", 1.0, 5.0, 2.25, 0.05)
        different_score = st.slider("Different-table score", -5.0, -1.0, -2.25, 0.05)
    return SegmentationConfig(
        cell_edge_threshold=edge,
        max_horizontal_gap_factor=horizontal,
        max_vertical_gap_factor=vertical,
        nearby_vertical_gap_factor=nearby,
        same_table_score=same_score,
        different_table_score=different_score,
    )


def _render_table(table: dict) -> None:
    structure = table["debug"]["structure"]
    title = (
        f"Table {table['table_id']} | pages {table['page_start']}-{table['page_end']} | "
        f"{structure['row_count']} rows x {structure['column_count']} columns"
    )
    with st.expander(title, expanded=True):
        metrics = st.columns(4)
        metrics[0].metric("Confidence", f"{table['confidence']:.1%}")
        metrics[1].metric("Cells", len(table["cells"]))
        metrics[2].metric("Components merged", len(table["debug"]["components_merged"]))
        metrics[3].metric("Sparse rows", len(structure["sparse_rows"]))
        st.dataframe(_rows_dataframe(table), use_container_width=True, hide_index=True)
        debug_tab, cells_tab = st.tabs(["Boundary decisions", "Raw cells"])
        with debug_tab:
            decisions = table["debug"].get("boundary_decisions", [])
            if decisions:
                st.dataframe(pd.DataFrame(decisions), use_container_width=True, hide_index=True)
            else:
                st.info("This table was produced from one connected component.")
            st.json(structure, expanded=False)
        with cells_tab:
            st.dataframe(pd.DataFrame(table["cells"]), use_container_width=True, hide_index=True)


uploaded = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

if uploaded is None:
    st.info("Upload a PDF to begin.")
    st.stop()

pdf_bytes = uploaded.getvalue()
file_key = sha256(pdf_bytes).hexdigest()
try:
    total_pages = _page_count(pdf_bytes)
except Exception as exc:
    st.error(f"Could not open PDF: {exc}")
    st.stop()

st.sidebar.success(f"{uploaded.name}: {total_pages} page(s)")
parser_mode = st.sidebar.radio("Existing parser", ["Native PDF text", "OCR"], horizontal=True)
page_options = list(range(1, total_pages + 1))
default_page = st.session_state.get(f"page_{file_key}", 1)
selected_page = st.sidebar.selectbox("Preview page", page_options, index=max(0, default_page - 1))
st.session_state[f"page_{file_key}"] = selected_page

selection_mode = st.sidebar.radio("Pages to split", ["Selected page", "Page range", "All pages"])
if selection_mode == "Selected page":
    pages = [selected_page]
elif selection_mode == "Page range":
    start_page, end_page = st.sidebar.select_slider(
        "Range", options=page_options, value=(selected_page, selected_page)
    )
    pages = list(range(start_page, end_page + 1))
else:
    pages = page_options

zoom = st.sidebar.slider("Preview zoom", 1.0, 3.0, 1.5, 0.25)
config = _config_controls()

if parser_mode == "Native PDF text":
    native_pages = _native_text_pages(pdf_bytes)
    missing = [page for page in pages if page not in native_pages]
    if missing:
        st.warning(
            f"Selected page(s) {missing} contain no native PDF text. Choose OCR for scanned pages."
        )

run = st.sidebar.button("Split tables", type="primary", use_container_width=True)
run_key = (file_key, tuple(pages), parser_mode, json.dumps(asdict(config), sort_keys=True))

if run:
    try:
        with st.spinner(f"Parsing and splitting {len(pages)} page(s)..."):
            parsed_pages = _parse_pages(pdf_bytes, tuple(pages), parser_mode)
            tables = segment_parser_pages(parsed_pages, config=config)
        st.session_state.separation_result = {"key": run_key, "tables": tables}
    except Exception as exc:
        st.exception(exc)
        st.stop()

result = st.session_state.get("separation_result")
if not result or result["key"] != run_key:
    st.info("Choose the parser and pages, then click **Split tables**.")
    st.stop()

tables = result["tables"]
page_tables = [
    table for table in tables if any(cell["page"] == selected_page for cell in table["cells"])
]
st.subheader(f"Page {selected_page}")
left, right = st.columns([1.1, 0.9])
with left:
    image = _render_page(pdf_bytes, selected_page, zoom)
    st.image(_draw_tables(image, page_tables, selected_page, zoom), use_container_width=True)
with right:
    st.metric("Logical tables", len(tables))
    st.metric("Tables visible on preview page", len(page_tables))
    st.metric("Pages segmented", len(pages))
    st.download_button(
        "Download segmentation JSON",
        data=json.dumps(tables, indent=2),
        file_name=f"{Path(uploaded.name).stem}_table_separation.json",
        mime="application/json",
        use_container_width=True,
    )
    st.download_button(
        "Download combined CSV",
        data=_tables_csv(tables),
        file_name=f"{Path(uploaded.name).stem}_table_separation.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.subheader("Split Results")
if not tables:
    st.warning("No positioned cells were found on the selected pages.")
else:
    for table in tables:
        _render_table(table)
