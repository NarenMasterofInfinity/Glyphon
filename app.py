from __future__ import annotations

import base64
import json
from io import BytesIO
from hashlib import sha256
from typing import Any

import fitz
import pandas as pd
import streamlit as st
from PIL import ImageDraw

from correction import (
    PendingChange,
    clone_df,
    data_columns,
    dataframe_to_csv,
    dataframe_to_excel,
    delete_columns,
    delete_rows,
    edit_cell,
    insert_column,
    insert_row,
    merge_columns,
    merge_rows,
    move_cell,
    operation_record,
    pending_to_metadata,
    rename_column,
    normalize_row_numbers,
    split_column_by_delimiter,
    split_row_by_delimiter,
)
from parser import CellExtraction, PageExtractionResult, build_table_records, parse_pdf_pages, render_page_image
from scanned_parser import extract_table_scanned


st.set_page_config(page_title="Glyphon Table Fixer", layout="wide")
st.title("Glyphon Table Fixer")

PARSER_CACHE_VERSION = "2026-06-02-correction-workspace-v1"
SYSTEM_COLUMNS = ["page_number", "row_number"]

VISUAL_WORKSPACE_HTML = """
<div id="glyphon-root" class="workspace">
  <section class="pdf-pane">
    <div class="visual-toolbar">
      <label>
        Mode
        <select id="guide-mode">
          <option value="inspect">Inspect</option>
          <option value="move_cell">Move cell</option>
          <option value="vertical_split">Vertical split guide</option>
          <option value="horizontal_split">Horizontal split guide</option>
        </select>
      </label>
      <button id="clear-guide" type="button">Clear guide</button>
      <button id="submit-guide" type="button" disabled>Preview split</button>
    </div>
    <div class="page-wrap">
      <img id="page-image" />
      <div id="overlay"></div>
      <div id="row-guide" class="guide row-guide"></div>
      <div id="col-guide" class="guide col-guide"></div>
      <div id="drawn-guide" class="drawn-guide"></div>
      <div id="guide-point-a" class="guide-point"></div>
      <div id="guide-point-b" class="guide-point"></div>
    </div>
  </section>
  <section class="table-pane">
    <div id="table-scroll">
      <table id="visual-table"></table>
    </div>
  </section>
  <dialog id="merge-dialog">
    <form id="merge-form">
      <h3 id="merge-title"></h3>
      <label>
        New header / label
        <input id="merge-name" autocomplete="off" />
      </label>
      <menu>
        <button id="merge-cancel" type="button">Cancel</button>
        <button type="submit">Preview</button>
      </menu>
    </form>
  </dialog>
  <dialog id="split-dialog">
    <form id="split-form">
      <h3 id="split-title"></h3>
      <div id="split-preview"></div>
      <div class="split-names">
        <label>
          Left / top suffix
          <input id="split-left-name" value="left" autocomplete="off" />
        </label>
        <label>
          Right / bottom suffix
          <input id="split-right-name" value="right" autocomplete="off" />
        </label>
      </div>
      <menu>
        <button id="split-cancel" type="button">Cancel</button>
        <button type="submit">Apply change</button>
      </menu>
    </form>
  </dialog>
  <dialog id="cell-dialog">
    <form id="cell-form">
      <h3 id="cell-title"></h3>
      <label>
        Value
        <textarea id="cell-value" rows="4"></textarea>
      </label>
      <menu>
        <button id="cell-cancel" type="button">Cancel</button>
        <button type="submit">Preview edit</button>
      </menu>
    </form>
  </dialog>
</div>
"""

VISUAL_WORKSPACE_CSS = """
.workspace {
  display: grid;
  grid-template-columns: minmax(380px, 1.08fr) minmax(420px, 0.92fr);
  gap: 14px;
  height: 760px;
  font-family: "Aptos", "Segoe UI", sans-serif;
}
.pdf-pane,
.table-pane {
  min-width: 0;
  overflow: auto;
  border: 1px solid #d7dce2;
  border-radius: 6px;
  background: #fff;
}
.visual-toolbar {
  position: sticky;
  top: 0;
  z-index: 8;
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 10px;
  border-bottom: 1px solid #d7dce2;
  background: #f8fafc;
}
.visual-toolbar label {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: #3b4654;
}
.visual-toolbar select {
  padding: 6px 8px;
  border: 1px solid #b9c3d0;
  border-radius: 4px;
  background: #fff;
}
.visual-toolbar button:disabled {
  cursor: not-allowed;
  opacity: 0.5;
}
.page-wrap {
  position: relative;
  width: max-content;
  max-width: 100%;
  margin: 0 auto;
}
#page-image {
  display: block;
  max-width: 100%;
  height: auto;
}
#overlay {
  position: absolute;
  inset: 0;
  pointer-events: none;
}
.bbox {
  position: absolute;
  box-sizing: border-box;
  border: 1.5px solid rgba(31, 99, 199, 0.85);
  background: transparent;
  pointer-events: none;
}
.bbox.row-hit {
  border-color: rgba(224, 143, 0, 0.95);
  background: rgba(255, 193, 7, 0.16);
}
.bbox.col-hit {
  border-color: rgba(27, 105, 220, 0.95);
  background: rgba(30, 136, 229, 0.13);
}
.bbox.cell-hit {
  border-color: rgba(0, 130, 75, 1);
  background: rgba(0, 184, 107, 0.26);
}
.guide {
  position: absolute;
  display: none;
  pointer-events: none;
  z-index: 4;
}
.row-guide {
  left: 0;
  right: 0;
  height: 0;
  border-top: 2px solid rgba(224, 143, 0, 0.95);
}
.col-guide {
  top: 0;
  bottom: 0;
  width: 0;
  border-left: 2px solid rgba(27, 105, 220, 0.95);
}
.drawn-guide {
  position: absolute;
  display: none;
  pointer-events: none;
  z-index: 6;
}
.drawn-guide.vertical {
  top: 0;
  bottom: 0;
  width: 0;
  border-left: 3px solid rgba(213, 72, 33, 0.95);
}
.drawn-guide.horizontal {
  left: 0;
  right: 0;
  height: 0;
  border-top: 3px solid rgba(213, 72, 33, 0.95);
}
.guide-point {
  position: absolute;
  z-index: 7;
  display: none;
  width: 15px;
  height: 15px;
  border: 2px solid #d54821;
  border-radius: 50%;
  background: #fff;
  box-shadow: 0 1px 5px rgba(20, 28, 38, 0.28);
  transform: translate(-50%, -50%);
  cursor: grab;
}
.guide-point:active {
  cursor: grabbing;
}
#table-scroll {
  overflow: auto;
  height: 100%;
}
#visual-table {
  width: 100%;
  border-collapse: collapse;
  table-layout: fixed;
  font-size: 13px;
}
#visual-table th,
#visual-table td {
  position: relative;
  min-width: 110px;
  border: 1px solid #d9dee5;
  padding: 6px 8px;
  vertical-align: top;
  background: #fff;
  overflow-wrap: anywhere;
}
#visual-table th {
  position: sticky;
  top: 0;
  z-index: 2;
  background: #f7f8fa;
  font-weight: 650;
}
#visual-table td:hover,
#visual-table td.active-cell {
  outline: 2px solid #00a86b;
  outline-offset: -2px;
  background: #eefbf5;
}
#visual-table td.move-source {
  outline: 2px solid #c96b00;
  outline-offset: -2px;
  background: #fff6df;
}
.merge-handle {
  position: absolute;
  z-index: 3;
  display: grid;
  place-items: center;
  width: 18px;
  height: 18px;
  border: 1px solid #305b9f;
  border-radius: 50%;
  background: #fff;
  color: #174ea6;
  font-size: 14px;
  line-height: 1;
  opacity: 0;
  cursor: pointer;
  transition: opacity 80ms ease;
}
th:hover .merge-col,
td:hover .merge-row {
  opacity: 1;
}
.merge-col {
  top: 50%;
  right: -9px;
  transform: translateY(-50%);
}
.merge-row {
  left: 50%;
  bottom: -9px;
  transform: translateX(-50%);
}
#merge-dialog {
  border: 1px solid #ccd3dc;
  border-radius: 8px;
  padding: 16px;
  width: min(360px, 90vw);
}
#merge-dialog::backdrop,
#split-dialog::backdrop {
  background: rgba(23, 32, 42, 0.22);
}
#merge-form,
#split-form {
  display: grid;
  gap: 12px;
}
#merge-title,
#split-title {
  margin: 0;
  font-size: 16px;
}
#merge-name,
#split-left-name,
#split-right-name {
  width: 100%;
  box-sizing: border-box;
  margin-top: 6px;
  padding: 8px;
  border: 1px solid #c8d0da;
  border-radius: 4px;
}
#split-dialog {
  border: 1px solid #ccd3dc;
  border-radius: 8px;
  padding: 16px;
  width: min(920px, 94vw);
}
#cell-dialog {
  border: 1px solid #ccd3dc;
  border-radius: 8px;
  padding: 16px;
  width: min(520px, 92vw);
}
#cell-dialog::backdrop {
  background: rgba(23, 32, 42, 0.22);
}
#cell-form {
  display: grid;
  gap: 12px;
}
#cell-title {
  margin: 0;
  font-size: 16px;
}
#cell-value {
  width: 100%;
  box-sizing: border-box;
  margin-top: 6px;
  padding: 8px;
  border: 1px solid #c8d0da;
  border-radius: 4px;
  resize: vertical;
  font: inherit;
}
#split-preview {
  max-height: 220px;
  overflow: auto;
  border: 1px solid #e2e7ee;
  border-radius: 4px;
  padding: 8px;
  background: #fafbfc;
  font-size: 13px;
}
.split-preview-table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 8px;
}
.split-preview-table th,
.split-preview-table td {
  border: 1px solid #d8dee8;
  padding: 6px 8px;
  vertical-align: top;
  text-align: left;
}
.split-preview-table th {
  background: #eef2f7;
  font-weight: 650;
}
.empty-preview {
  color: #6b7280;
}
.split-names {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
}
menu {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  padding: 0;
  margin: 0;
}
button {
  border: 1px solid #b9c3d0;
  border-radius: 4px;
  background: #fff;
  padding: 7px 12px;
  cursor: pointer;
}
button[type="submit"] {
  border-color: #174ea6;
  background: #174ea6;
  color: #fff;
}
@media (max-width: 900px) {
  .workspace {
    grid-template-columns: 1fr;
    height: auto;
  }
  .pdf-pane,
  .table-pane {
    height: 560px;
  }
}
"""

VISUAL_WORKSPACE_JS = """
export default function(component) {
  const { data, parentElement, setTriggerValue } = component;
  const root = parentElement.querySelector("#glyphon-root");
  if (!root || !data) return;

  const pageImage = root.querySelector("#page-image");
  const overlay = root.querySelector("#overlay");
  const table = root.querySelector("#visual-table");
  const rowGuide = root.querySelector("#row-guide");
  const colGuide = root.querySelector("#col-guide");
  const guideMode = root.querySelector("#guide-mode");
  const clearGuide = root.querySelector("#clear-guide");
  const submitGuide = root.querySelector("#submit-guide");
  const drawnGuide = root.querySelector("#drawn-guide");
  const pointA = root.querySelector("#guide-point-a");
  const pointB = root.querySelector("#guide-point-b");
  const dialog = root.querySelector("#merge-dialog");
  const mergeTitle = root.querySelector("#merge-title");
  const mergeName = root.querySelector("#merge-name");
  const mergeCancel = root.querySelector("#merge-cancel");
  const mergeForm = root.querySelector("#merge-form");
  const splitDialog = root.querySelector("#split-dialog");
  const splitTitle = root.querySelector("#split-title");
  const splitPreview = root.querySelector("#split-preview");
  const splitLeftName = root.querySelector("#split-left-name");
  const splitRightName = root.querySelector("#split-right-name");
  const splitCancel = root.querySelector("#split-cancel");
  const splitForm = root.querySelector("#split-form");
  const cellDialog = root.querySelector("#cell-dialog");
  const cellTitle = root.querySelector("#cell-title");
  const cellValue = root.querySelector("#cell-value");
  const cellCancel = root.querySelector("#cell-cancel");
  const cellForm = root.querySelector("#cell-form");
  let pendingMerge = null;
  let pendingSplit = null;
  let pendingEdit = null;
  let guidePoints = [];
  let draggingPoint = null;
  let drawingGuide = false;
  let moveSource = null;

  pageImage.src = data.image;
  overlay.innerHTML = "";
  table.innerHTML = "";
  drawnGuide.style.display = "none";
  pointA.style.display = "none";
  pointB.style.display = "none";
  submitGuide.disabled = true;

  const itemMap = new Map((data.items || []).map((item) => [item.index, item]));
  const rowMap = new Map((data.rows || []).map((row) => [Number(row.row_number), row]));
  const cellMap = new Map((data.cells || []).map((cell) => [`${cell.row}:${cell.col}`, cell]));
  const rowItems = new Map();
  const colItems = new Map();
  const cellItems = new Map();

  for (const cell of data.cells || []) {
    const rowKey = String(cell.row);
    const colKey = String(cell.col);
    const cellKey = `${cell.row}:${cell.col}`;
    for (const itemIndex of cell.source || []) {
      if (!rowItems.has(rowKey)) rowItems.set(rowKey, new Set());
      if (!colItems.has(colKey)) colItems.set(colKey, new Set());
      if (!cellItems.has(cellKey)) cellItems.set(cellKey, new Set());
      rowItems.get(rowKey).add(itemIndex);
      colItems.get(colKey).add(itemIndex);
      cellItems.get(cellKey).add(itemIndex);
    }
  }

  for (const item of data.items || []) {
    const box = document.createElement("div");
    box.className = "bbox";
    box.dataset.index = String(item.index);
    box.style.left = `${(item.x0 / data.pageWidth) * 100}%`;
    box.style.top = `${(item.y0 / data.pageHeight) * 100}%`;
    box.style.width = `${((item.x1 - item.x0) / data.pageWidth) * 100}%`;
    box.style.height = `${((item.y1 - item.y0) / data.pageHeight) * 100}%`;
    overlay.appendChild(box);
  }

  function setGuide(items, kind) {
    const selected = [...items].map((idx) => itemMap.get(idx)).filter(Boolean);
    if (!selected.length) return;
    const x0 = Math.min(...selected.map((item) => item.x0));
    const y0 = Math.min(...selected.map((item) => item.y0));
    const x1 = Math.max(...selected.map((item) => item.x1));
    const y1 = Math.max(...selected.map((item) => item.y1));
    if (kind === "row") {
      rowGuide.style.top = `${(((y0 + y1) / 2) / data.pageHeight) * 100}%`;
      rowGuide.style.display = "block";
    } else {
      colGuide.style.left = `${(((x0 + x1) / 2) / data.pageWidth) * 100}%`;
      colGuide.style.display = "block";
    }
  }

  function clearHighlights() {
    overlay.querySelectorAll(".bbox").forEach((box) => {
      box.classList.remove("row-hit", "col-hit", "cell-hit");
    });
    rowGuide.style.display = "none";
    colGuide.style.display = "none";
    table.querySelectorAll(".active-cell").forEach((cell) => cell.classList.remove("active-cell"));
    table.querySelectorAll(".move-source").forEach((cell) => cell.classList.remove("move-source"));
  }

  function addClass(items, className) {
    for (const itemIndex of items || []) {
      const box = overlay.querySelector(`.bbox[data-index="${itemIndex}"]`);
      if (box) box.classList.add(className);
    }
  }

  function activateCell(cellElement) {
    clearHighlights();
    const row = cellElement.dataset.row;
    const col = cellElement.dataset.col;
    const rowSet = rowItems.get(row) || new Set();
    const colSet = colItems.get(col) || new Set();
    const cellSet = cellItems.get(`${row}:${col}`) || new Set();
    addClass(rowSet, "row-hit");
    addClass(colSet, "col-hit");
    addClass(cellSet, "cell-hit");
    setGuide(rowSet, "row");
    setGuide(colSet, "col");
    cellElement.classList.add("active-cell");
    if (moveSource && moveSource.row === cellElement.dataset.row && moveSource.col === cellElement.dataset.col) {
      cellElement.classList.add("move-source");
    }
  }

  function clearMoveSource() {
    moveSource = null;
    table.querySelectorAll(".move-source").forEach((cell) => cell.classList.remove("move-source"));
  }

  function pagePointFromEvent(event) {
    const rect = pageImage.getBoundingClientRect();
    const x = Math.max(0, Math.min(data.pageWidth, ((event.clientX - rect.left) / rect.width) * data.pageWidth));
    const y = Math.max(0, Math.min(data.pageHeight, ((event.clientY - rect.top) / rect.height) * data.pageHeight));
    return { x, y };
  }

  function placeDot(dot, point) {
    dot.style.left = `${(point.x / data.pageWidth) * 100}%`;
    dot.style.top = `${(point.y / data.pageHeight) * 100}%`;
    dot.style.display = "block";
  }

  function guideValue() {
    if (guidePoints.length < 2) return null;
    if (guideMode.value === "vertical_split") {
      return (guidePoints[0].x + guidePoints[1].x) / 2;
    }
    if (guideMode.value === "horizontal_split") {
      return (guidePoints[0].y + guidePoints[1].y) / 2;
    }
    return null;
  }

  function renderDrawnGuide() {
    placeDot(pointA, guidePoints[0]);
    if (guidePoints[1]) {
      placeDot(pointB, guidePoints[1]);
    } else {
      pointB.style.display = "none";
    }
    const value = guideValue();
    submitGuide.disabled = value === null;
    drawnGuide.className = "drawn-guide";
    if (value === null) {
      drawnGuide.style.display = "none";
      return;
    }
    if (guideMode.value === "vertical_split") {
      drawnGuide.classList.add("vertical");
      drawnGuide.style.left = `${(value / data.pageWidth) * 100}%`;
      drawnGuide.style.top = "";
      drawnGuide.style.display = "block";
    } else if (guideMode.value === "horizontal_split") {
      drawnGuide.classList.add("horizontal");
      drawnGuide.style.top = `${(value / data.pageHeight) * 100}%`;
      drawnGuide.style.left = "";
      drawnGuide.style.display = "block";
    }
  }

  function resetGuide() {
    guidePoints = [];
    draggingPoint = null;
    drawingGuide = false;
    clearMoveSource();
    pointA.style.display = "none";
    pointB.style.display = "none";
    drawnGuide.style.display = "none";
    submitGuide.disabled = true;
  }

  function itemPartition(source, axis, value) {
    let before = 0;
    let after = 0;
    for (const itemIndex of source || []) {
      const item = itemMap.get(itemIndex);
      if (!item) continue;
      const itemValue = axis === "x" ? (item.x0 + item.x1) / 2 : (item.y0 + item.y1) / 2;
      if (itemValue <= value) before += 1;
      else after += 1;
    }
    return { before, after };
  }

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function partitionText(source, axis, value) {
    const before = [];
    const after = [];
    for (const itemIndex of source || []) {
      const item = itemMap.get(itemIndex);
      if (!item) continue;
      const itemValue = axis === "x" ? (item.x0 + item.x1) / 2 : (item.y0 + item.y1) / 2;
      if (itemValue <= value) before.push(item.text);
      else after.push(item.text);
    }
    return {
      before: before.join(" ").trim(),
      after: after.join(" ").trim(),
    };
  }

  function affectedColumns(value) {
    const affected = new Map();
    for (const cell of data.cells || []) {
      const split = itemPartition(cell.source, "x", value);
      if (split.before > 0 && split.after > 0) {
        const columnName = data.columns[cell.col - 1];
        if (columnName) affected.set(columnName, columnName);
      }
    }
    return [...affected.keys()];
  }

  function affectedRows(value) {
    const affected = new Map();
    for (const cell of data.cells || []) {
      const split = itemPartition(cell.source, "y", value);
      if (split.before > 0 && split.after > 0) {
        affected.set(cell.row, cell.row);
      }
    }
    return [...affected.keys()].sort((a, b) => a - b);
  }

  function openSplitDialog() {
    const value = guideValue();
    if (value === null) return;
    if (guideMode.value === "vertical_split") {
      const columns = affectedColumns(value);
      splitTitle.textContent = "Preview column split";
      if (columns.length) {
        const columnIndexes = new Map(columns.map((column) => [data.columns.indexOf(column) + 1, column]));
        const previewRows = [];
        for (const cell of data.cells || []) {
          const column = columnIndexes.get(cell.col);
          if (!column) continue;
          const split = itemPartition(cell.source, "x", value);
          if (!(split.before > 0 && split.after > 0)) continue;
          const parts = partitionText(cell.source, "x", value);
          const sourceRow = rowMap.get(Number(cell.row)) || {};
          previewRows.push(`
            <tr>
              <td>${escapeHtml(cell.row)}</td>
              <td>${escapeHtml(column)}</td>
              <td>${escapeHtml(sourceRow[column] ?? cell.text)}</td>
              <td>${escapeHtml(parts.before)}</td>
              <td>${escapeHtml(parts.after)}</td>
            </tr>
          `);
        }
        splitPreview.innerHTML = `
          <div><b>${escapeHtml(columns.join(", "))}</b> will be split by the vertical guide.</div>
          <table class="split-preview-table">
            <thead>
              <tr><th>Row</th><th>Column</th><th>Before</th><th>Left</th><th>Right</th></tr>
            </thead>
            <tbody>${previewRows.join("")}</tbody>
          </table>
        `;
      } else {
        splitPreview.innerHTML = `<div class="empty-preview">No existing column has source boxes on both sides of this guide.</div>`;
      }
      pendingSplit = { kind: "split_columns_visual", guideX: value, columns };
    } else if (guideMode.value === "horizontal_split") {
      const rows = affectedRows(value);
      splitTitle.textContent = "Preview row split";
      if (rows.length) {
        const previewRows = [];
        for (const rowNumber of rows) {
          const sourceRow = rowMap.get(Number(rowNumber)) || {};
          const before = [];
          const top = [];
          const bottom = [];
          for (const [idx, column] of (data.columns || []).entries()) {
            const cell = cellMap.get(`${rowNumber}:${idx + 1}`);
            before.push(`${column}: ${sourceRow[column] ?? ""}`);
            if (!cell) {
              top.push(`${column}: ${sourceRow[column] ?? ""}`);
              bottom.push(`${column}:`);
              continue;
            }
            const parts = partitionText(cell.source, "y", value);
            top.push(`${column}: ${parts.before}`);
            bottom.push(`${column}: ${parts.after}`);
          }
          previewRows.push(`
            <tr>
              <td>${escapeHtml(rowNumber)}</td>
              <td>${escapeHtml(before.join(" | "))}</td>
              <td>${escapeHtml(top.join(" | "))}</td>
              <td>${escapeHtml(bottom.join(" | "))}</td>
            </tr>
          `);
        }
        splitPreview.innerHTML = `
          <div><b>Rows ${escapeHtml(rows.join(", "))}</b> will be split by the horizontal guide.</div>
          <table class="split-preview-table">
            <thead>
              <tr><th>Row</th><th>Before</th><th>Top row</th><th>Bottom row</th></tr>
            </thead>
            <tbody>${previewRows.join("")}</tbody>
          </table>
        `;
      } else {
        splitPreview.innerHTML = `<div class="empty-preview">No existing row has source boxes on both sides of this guide.</div>`;
      }
      pendingSplit = { kind: "split_rows_visual", guideY: value, rows };
    }
    splitForm.querySelector('button[type="submit"]').disabled =
      (pendingSplit.columns && pendingSplit.columns.length === 0) ||
      (pendingSplit.rows && pendingSplit.rows.length === 0);
    splitDialog.showModal();
  }

  function isSplitMode() {
    return guideMode.value === "vertical_split" || guideMode.value === "horizontal_split";
  }

  function openEditDialog(cellElement) {
    pendingEdit = {
      rowNumber: Number(cellElement.dataset.row),
      colIndex: Number(cellElement.dataset.col),
      column: data.columns[Number(cellElement.dataset.col) - 1],
    };
    cellTitle.textContent = `Edit ${pendingEdit.column} at row ${pendingEdit.rowNumber}`;
    cellValue.value = cellElement.dataset.value || "";
    cellDialog.showModal();
  }

  pageImage.onpointerdown = (event) => {
    if (!isSplitMode()) return;
    drawingGuide = true;
    const point = pagePointFromEvent(event);
    guidePoints = [point];
    pageImage.setPointerCapture(event.pointerId);
    renderDrawnGuide();
    event.preventDefault();
  };
  pageImage.onpointermove = (event) => {
    if (!drawingGuide || !isSplitMode()) return;
    guidePoints[1] = pagePointFromEvent(event);
    renderDrawnGuide();
    event.preventDefault();
  };
  pageImage.onpointerup = (event) => {
    if (!drawingGuide || !isSplitMode()) return;
    drawingGuide = false;
    guidePoints[1] = pagePointFromEvent(event);
    pageImage.releasePointerCapture(event.pointerId);
    renderDrawnGuide();
    event.preventDefault();
  };

  for (const [idx, dot] of [pointA, pointB].entries()) {
    dot.onpointerdown = (event) => {
      draggingPoint = idx;
      dot.setPointerCapture(event.pointerId);
      event.stopPropagation();
    };
    dot.onpointermove = (event) => {
      if (draggingPoint !== idx || !guidePoints[idx]) return;
      guidePoints[idx] = pagePointFromEvent(event);
      renderDrawnGuide();
      event.stopPropagation();
    };
    dot.onpointerup = (event) => {
      draggingPoint = null;
      dot.releasePointerCapture(event.pointerId);
      event.stopPropagation();
    };
  }

  guideMode.onchange = resetGuide;
  clearGuide.onclick = resetGuide;
  submitGuide.onclick = openSplitDialog;

  const headerRow = document.createElement("tr");
  for (const [idx, column] of (data.columns || []).entries()) {
    const th = document.createElement("th");
    th.textContent = column;
    if (idx < data.columns.length - 1) {
      const handle = document.createElement("span");
      handle.className = "merge-handle merge-col";
      handle.textContent = "+";
      handle.title = `Merge ${column} and ${data.columns[idx + 1]}`;
      handle.onclick = (event) => {
        event.stopPropagation();
        pendingMerge = {
          kind: "merge_columns",
          columns: [column, data.columns[idx + 1]],
          defaultName: column,
        };
        mergeTitle.textContent = `Merge ${column} + ${data.columns[idx + 1]}`;
        mergeName.value = column;
        dialog.showModal();
      };
      th.appendChild(handle);
    }
    headerRow.appendChild(th);
  }
  table.appendChild(headerRow);

  for (const row of data.rows || []) {
    const tr = document.createElement("tr");
    for (const [idx, column] of (data.columns || []).entries()) {
      const td = document.createElement("td");
      td.textContent = row[column] ?? "";
      td.dataset.value = row[column] ?? "";
      td.dataset.row = String(row.row_number);
      td.dataset.col = String(idx + 1);
      td.onmouseenter = () => activateCell(td);
      td.onclick = () => {
        activateCell(td);
        if (guideMode.value !== "move_cell") return;
        if (!moveSource) {
          moveSource = {
            row: td.dataset.row,
            col: td.dataset.col,
            column: data.columns[idx],
          };
          td.classList.add("move-source");
          return;
        }
        if (moveSource.row === td.dataset.row && moveSource.col === td.dataset.col) {
          clearMoveSource();
          return;
        }
        setTriggerValue("move", {
          sourceRow: Number(moveSource.row),
          sourceCol: Number(moveSource.col),
          sourceColumn: moveSource.column,
          targetRow: Number(td.dataset.row),
          targetCol: Number(td.dataset.col),
          targetColumn: data.columns[idx],
        });
        clearMoveSource();
      };
      td.ondblclick = () => openEditDialog(td);
      if (idx === 0 && row.row_number < data.maxRow) {
        const handle = document.createElement("span");
        handle.className = "merge-handle merge-row";
        handle.textContent = "+";
        handle.title = `Merge row ${row.row_number} and ${row.row_number + 1}`;
        handle.onclick = (event) => {
          event.stopPropagation();
          pendingMerge = {
            kind: "merge_rows",
            rowNumbers: [row.row_number, row.row_number + 1],
            defaultName: `row_${row.row_number}`,
          };
          mergeTitle.textContent = `Merge row ${row.row_number} + ${row.row_number + 1}`;
          mergeName.value = `row_${row.row_number}`;
          dialog.showModal();
        };
        td.appendChild(handle);
      }
      tr.appendChild(td);
    }
    table.appendChild(tr);
  }

  table.onmouseleave = clearHighlights;
  mergeCancel.onclick = () => dialog.close();
  mergeForm.onsubmit = (event) => {
    event.preventDefault();
    if (!pendingMerge) return;
    const payload = { ...pendingMerge, name: mergeName.value };
    setTriggerValue("merge", payload);
    dialog.close();
  };
  splitCancel.onclick = () => splitDialog.close();
  splitForm.onsubmit = (event) => {
    event.preventDefault();
    if (!pendingSplit) return;
    setTriggerValue("split", {
      ...pendingSplit,
      leftName: splitLeftName.value,
      rightName: splitRightName.value,
    });
    splitDialog.close();
  };
  cellCancel.onclick = () => cellDialog.close();
  cellForm.onsubmit = (event) => {
    event.preventDefault();
    if (!pendingEdit) return;
    setTriggerValue("edit_cell", {
      ...pendingEdit,
      value: cellValue.value,
    });
    cellDialog.close();
  };
}
"""

VISUAL_WORKSPACE = st.components.v2.component(
    "glyphon_visual_workspace",
    html=VISUAL_WORKSPACE_HTML,
    css=VISUAL_WORKSPACE_CSS,
    js=VISUAL_WORKSPACE_JS,
)

POLYGON_SELECTOR_HTML = """
<div id="polygon-root" class="polygon-shell">
  <div class="polygon-toolbar">
    <button id="clear-polygon" type="button">Clear</button>
    <button id="submit-polygon" type="button" disabled>Extract selected table</button>
  </div>
  <div class="polygon-page">
    <img id="polygon-image" />
    <svg id="polygon-svg"></svg>
  </div>
</div>
"""

POLYGON_SELECTOR_CSS = """
.polygon-shell {
  height: 760px;
  overflow: auto;
  border: 1px solid #d7dce2;
  border-radius: 6px;
  background: #fff;
  font-family: "Aptos", "Segoe UI", sans-serif;
}
.polygon-toolbar {
  position: sticky;
  top: 0;
  z-index: 4;
  display: flex;
  gap: 8px;
  padding: 8px 10px;
  border-bottom: 1px solid #d7dce2;
  background: #f8fafc;
}
.polygon-toolbar button {
  border: 1px solid #b9c3d0;
  border-radius: 4px;
  background: #fff;
  padding: 7px 12px;
  cursor: pointer;
}
.polygon-toolbar button:disabled {
  cursor: not-allowed;
  opacity: 0.5;
}
#submit-polygon {
  border-color: #174ea6;
  background: #174ea6;
  color: #fff;
}
.polygon-page {
  position: relative;
  width: max-content;
  max-width: 100%;
  margin: 0 auto;
}
#polygon-image {
  display: block;
  max-width: 100%;
  height: auto;
}
#polygon-svg {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  overflow: visible;
}
.source-box {
  fill: transparent;
  stroke: rgba(31, 99, 199, 0.78);
  stroke-width: 1.5;
  vector-effect: non-scaling-stroke;
}
.source-box.inside {
  fill: rgba(0, 184, 107, 0.18);
  stroke: rgba(0, 130, 75, 1);
}
.selection-polygon {
  fill: rgba(213, 72, 33, 0.12);
  stroke: rgba(213, 72, 33, 0.95);
  stroke-width: 2.5;
  vector-effect: non-scaling-stroke;
}
.selection-line {
  fill: none;
  stroke: rgba(213, 72, 33, 0.95);
  stroke-width: 2;
  stroke-dasharray: 6 4;
  vector-effect: non-scaling-stroke;
}
.polygon-point {
  fill: #fff;
  stroke: #d54821;
  stroke-width: 2.5;
  cursor: grab;
  vector-effect: non-scaling-stroke;
}
.polygon-point:active {
  cursor: grabbing;
}
"""

POLYGON_SELECTOR_JS = """
export default function(component) {
  const { data, parentElement, setTriggerValue } = component;
  const root = parentElement.querySelector("#polygon-root");
  if (!root || !data) return;

  const image = root.querySelector("#polygon-image");
  const svg = root.querySelector("#polygon-svg");
  const clearButton = root.querySelector("#clear-polygon");
  const submitButton = root.querySelector("#submit-polygon");
  let points = [];
  let draggingIndex = null;

  image.src = data.image;

  function toSvgPoint(event) {
    const rect = image.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(data.pageWidth, ((event.clientX - rect.left) / rect.width) * data.pageWidth)),
      y: Math.max(0, Math.min(data.pageHeight, ((event.clientY - rect.top) / rect.height) * data.pageHeight)),
    };
  }

  function pointInPolygon(point, polygon) {
    let inside = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
      const xi = polygon[i].x, yi = polygon[i].y;
      const xj = polygon[j].x, yj = polygon[j].y;
      const intersects = ((yi > point.y) !== (yj > point.y)) &&
        (point.x < ((xj - xi) * (point.y - yi)) / (yj - yi + Number.EPSILON) + xi);
      if (intersects) inside = !inside;
    }
    return inside;
  }

  function render() {
    svg.setAttribute("viewBox", `0 0 ${data.pageWidth} ${data.pageHeight}`);
    svg.innerHTML = "";

    for (const item of data.items || []) {
      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("x", item.x0);
      rect.setAttribute("y", item.y0);
      rect.setAttribute("width", item.x1 - item.x0);
      rect.setAttribute("height", item.y1 - item.y0);
      rect.setAttribute("class", "source-box");
      if (points.length >= 3 && pointInPolygon({ x: (item.x0 + item.x1) / 2, y: (item.y0 + item.y1) / 2 }, points)) {
        rect.classList.add("inside");
      }
      svg.appendChild(rect);
    }

    if (points.length >= 2) {
      const path = document.createElementNS("http://www.w3.org/2000/svg", points.length >= 3 ? "polygon" : "polyline");
      path.setAttribute("points", points.map((point) => `${point.x},${point.y}`).join(" "));
      path.setAttribute("class", points.length >= 3 ? "selection-polygon" : "selection-line");
      svg.appendChild(path);
    }

    points.forEach((point, index) => {
      const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute("cx", point.x);
      circle.setAttribute("cy", point.y);
      circle.setAttribute("r", 6);
      circle.setAttribute("class", "polygon-point");
      circle.onpointerdown = (event) => {
        draggingIndex = index;
        circle.setPointerCapture(event.pointerId);
        event.stopPropagation();
      };
      circle.onpointermove = (event) => {
        if (draggingIndex !== index) return;
        points[index] = toSvgPoint(event);
        render();
        event.stopPropagation();
      };
      circle.onpointerup = (event) => {
        draggingIndex = null;
        circle.releasePointerCapture(event.pointerId);
        event.stopPropagation();
      };
      svg.appendChild(circle);
    });

    submitButton.disabled = points.length < 3;
  }

  svg.onclick = (event) => {
    if (event.target.classList.contains("polygon-point")) return;
    points.push(toSvgPoint(event));
    render();
  };
  clearButton.onclick = () => {
    points = [];
    render();
  };
  submitButton.onclick = () => {
    if (points.length < 3) return;
    setTriggerValue("polygon", { points });
  };

  render();
}
"""

POLYGON_SELECTOR = st.components.v2.component(
    "glyphon_polygon_selector",
    html=POLYGON_SELECTOR_HTML,
    css=POLYGON_SELECTOR_CSS,
    js=POLYGON_SELECTOR_JS,
)


@st.cache_data(show_spinner=False)
def run_parser(pdf_bytes: bytes, cache_version: str, page_numbers: tuple[int, ...]):
    del cache_version
    return parse_pdf_pages(pdf_bytes, page_numbers=list(page_numbers))


@st.cache_data(show_spinner=False)
def get_pdf_page_numbers(pdf_bytes: bytes) -> list[int]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_numbers = list(range(1, len(doc) + 1))
    doc.close()
    return page_numbers


def build_initial_table(page_results) -> pd.DataFrame:
    headers, records = build_table_records(page_results)
    return pd.DataFrame(records, columns=headers)


def get_file_key(pdf_bytes: bytes) -> str:
    return sha256(pdf_bytes).hexdigest()


def initialize_state(file_key: str, initial_df: pd.DataFrame, page_results) -> None:
    if st.session_state.get("file_key") == file_key:
        return

    st.session_state.file_key = file_key
    st.session_state.original_df = clone_df(initial_df)
    st.session_state.corrected_df = clone_df(initial_df)
    st.session_state.history = []
    st.session_state.redo_stack = []
    st.session_state.operation_history = []
    st.session_state.pending_change = None
    st.session_state.page_results = page_results
    st.session_state.page_drafts = {}


def initialize_selective_state(file_key: str, initial_df: pd.DataFrame, page_results, operation: dict[str, Any]) -> None:
    if st.session_state.get("file_key") == file_key:
        return

    st.session_state.file_key = file_key
    st.session_state.original_df = clone_df(initial_df)
    st.session_state.corrected_df = clone_df(initial_df)
    st.session_state.history = []
    st.session_state.redo_stack = []
    st.session_state.operation_history = [operation]
    st.session_state.pending_change = None
    st.session_state.page_results = page_results
    st.session_state.page_drafts = {}


def clear_page_drafts() -> None:
    st.session_state.page_drafts = {}


def extraction_state_key(file_key: str, selected_pages: list[int]) -> str:
    if not selected_pages:
        return f"{file_key}:all-pages"
    page_token = ",".join(str(page_number) for page_number in selected_pages)
    return f"{file_key}:pages:{page_token}"


def store_page_draft(state_key: str, page_number: int, page_df: pd.DataFrame) -> None:
    page_drafts = st.session_state.setdefault("page_drafts", {})
    page_drafts[(state_key, page_number)] = clone_df(page_df)


def page_draft_or_default(state_key: str, page_number: int, base_df: pd.DataFrame) -> pd.DataFrame:
    draft = st.session_state.get("page_drafts", {}).get((state_key, page_number))
    if draft is None:
        return clone_df(base_df)
    return clone_df(draft)


def set_pending(action: str, description: str, preview_df: pd.DataFrame, metadata: dict[str, Any]) -> None:
    st.session_state.pending_change = PendingChange(
        action=action,
        description=description,
        preview_df=preview_df,
        metadata=metadata,
    )


def accept_pending() -> None:
    pending = st.session_state.pending_change
    if pending is None:
        return

    st.session_state.history.append(clone_df(st.session_state.corrected_df))
    st.session_state.corrected_df = clone_df(pending.preview_df)
    st.session_state.redo_stack = []
    st.session_state.operation_history.append(operation_record(pending.action, pending.metadata))
    st.session_state.pending_change = None
    clear_page_drafts()


def reject_pending() -> None:
    st.session_state.pending_change = None


def undo() -> None:
    if not st.session_state.history:
        return
    st.session_state.redo_stack.append(clone_df(st.session_state.corrected_df))
    st.session_state.corrected_df = st.session_state.history.pop()
    st.session_state.operation_history.append(operation_record("undo", {}))
    st.session_state.pending_change = None
    clear_page_drafts()


def redo() -> None:
    if not st.session_state.redo_stack:
        return
    st.session_state.history.append(clone_df(st.session_state.corrected_df))
    st.session_state.corrected_df = st.session_state.redo_stack.pop()
    st.session_state.operation_history.append(operation_record("redo", {}))
    st.session_state.pending_change = None
    clear_page_drafts()


def commit_page_edits(state_key: str, page_number: int, edited_page_df: pd.DataFrame) -> None:
    current = st.session_state.corrected_df
    page_mask = current["page_number"] == page_number
    page_indexes = current.index[page_mask].tolist()

    if len(page_indexes) != len(edited_page_df):
        st.warning("Page row count changed in the editor. Use row operations for structural row changes.")
        return

    updated = current.copy()
    edited = edited_page_df.copy()
    edited.insert(0, "page_number", page_number)

    for source_index, (_, row) in zip(page_indexes, edited.iterrows()):
        for column in updated.columns:
            if column in row.index:
                updated.at[source_index, column] = row[column]

    if updated.equals(current):
        return

    st.session_state.history.append(clone_df(current))
    st.session_state.corrected_df = updated
    st.session_state.redo_stack = []
    st.session_state.operation_history.append(
        operation_record("edit_cells", {"page_number": page_number})
    )
    store_page_draft(state_key, page_number, page_dataframe(updated, page_number))


def page_dataframe(df: pd.DataFrame, page_number: int) -> pd.DataFrame:
    page_df = df.loc[df["page_number"] == page_number].copy()
    if page_df.empty:
        return page_df
    return page_df.drop(columns=["page_number"]).reset_index(drop=True)


def image_to_data_uri(image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def visual_workspace_payload(
    pdf_bytes: bytes,
    page_number: int,
    zoom: float,
    page_result,
    corrected_df: pd.DataFrame,
) -> dict[str, Any]:
    page_image = render_page_image(pdf_bytes, page_number, zoom=zoom)
    page_df = page_dataframe(corrected_df, page_number)
    columns = data_columns(corrected_df)

    rows = []
    for _, row in page_df.iterrows():
        record = {"row_number": int(row["row_number"])}
        for column in columns:
            record[column] = "" if pd.isna(row.get(column, "")) else str(row.get(column, ""))
        rows.append(record)

    cells = [
        {
            "row": cell.row_index,
            "col": cell.col_index,
            "text": cell.text,
            "source": cell.source_item_indexes,
        }
        for cell in getattr(page_result, "cells", [])
    ]

    items = [
        {
            "index": index,
            "text": item.text,
            "x0": item.x0,
            "y0": item.y0,
            "x1": item.x1,
            "y1": item.y1,
        }
        for index, item in enumerate(page_result.raw_items)
    ]

    return {
        "image": image_to_data_uri(page_image),
        "pageWidth": page_result.page_width,
        "pageHeight": page_result.page_height,
        "columns": columns,
        "rows": rows,
        "cells": cells,
        "items": items,
        "maxRow": max((row["row_number"] for row in rows), default=0),
    }


def polygon_selector_payload(
    pdf_bytes: bytes,
    page_number: int,
    zoom: float,
    page_result,
) -> dict[str, Any]:
    page_image = render_page_image(pdf_bytes, page_number, zoom=zoom)
    return {
        "image": image_to_data_uri(page_image),
        "pageWidth": page_result.page_width,
        "pageHeight": page_result.page_height,
        "items": [
            {
                "index": index,
                "text": item.text,
                "x0": item.x0,
                "y0": item.y0,
                "x1": item.x1,
                "y1": item.y1,
            }
            for index, item in enumerate(page_result.raw_items)
        ],
    }


def point_in_polygon(x: float, y: float, polygon: list[dict[str, float]]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i, point in enumerate(polygon):
        xi = float(point["x"])
        yi = float(point["y"])
        xj = float(polygon[j]["x"])
        yj = float(polygon[j]["y"])
        intersects = ((yi > y) != (yj > y)) and (
            x < ((xj - xi) * (y - yi)) / ((yj - yi) or 1e-9) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def build_selective_page_result(page_result, polygon: list[dict[str, float]]) -> PageExtractionResult:
    selected_items = [
        item
        for item in page_result.raw_items
        if point_in_polygon(item.cx, item.cy, polygon)
    ]
    extracted = extract_table_scanned(selected_items, page_result.page_width)
    return PageExtractionResult(
        page_number=page_result.page_number,
        page_width=page_result.page_width,
        page_height=page_result.page_height,
        column_names=extracted["column_names"],
        rows=extracted["aligned_rows"],
        raw_items=selected_items,
        slant_angle=extracted["slant_angle"],
        column_centers=extracted["column_centers"],
        table_band=extracted["band"],
        cells=[
            CellExtraction(page_number=page_result.page_number, **cell)
            for cell in extracted["cells"]
        ],
    )


def load_selective_extraction(file_key: str, selective_result: PageExtractionResult) -> None:
    version = st.session_state.get("selective_override_version", 0) + 1
    st.session_state.selective_override_version = version
    st.session_state.selective_override = {
        "file_key": file_key,
        "page_results": [selective_result],
        "state_key": f"{file_key}:selective:{selective_result.page_number}:{version}",
        "operation": operation_record(
            "selective_extraction",
            {
                "page_number": selective_result.page_number,
                "source_item_count": len(selective_result.raw_items),
                "version": version,
            },
        ),
    }


def handle_visual_merge_event(event: dict[str, Any] | None, corrected_df: pd.DataFrame, selected_page: int) -> None:
    if not event or st.session_state.pending_change is not None:
        return

    if event.get("kind") == "merge_columns":
        columns = event.get("columns") or []
        output_name = event.get("name") or columns[0]
        preview_df = merge_columns(corrected_df, columns, output_name)
        set_pending(
            "merge_columns",
            f"Merge columns {columns} into {output_name}.",
            preview_df,
            {"columns": columns, "output_name": output_name, "source": "visual_workspace"},
        )
        st.rerun()

    if event.get("kind") == "merge_rows":
        row_numbers = event.get("rowNumbers") or []
        page_indexes = corrected_df.index[corrected_df["page_number"] == selected_page].tolist()
        row_labels = {
            int(corrected_df.at[index, "row_number"]): index
            for index in page_indexes
        }
        preview_df = merge_rows(corrected_df, [row_labels[row_number] for row_number in row_numbers])
        set_pending(
            "merge_rows",
            f"Merge rows {row_numbers} on page {selected_page}.",
            preview_df,
            {"page_number": selected_page, "rows": row_numbers, "source": "visual_workspace"},
        )
        st.rerun()


def handle_visual_split_event(
    event: dict[str, Any] | None,
    corrected_df: pd.DataFrame,
    page_result,
    selected_page: int,
) -> None:
    if not event or st.session_state.pending_change is not None:
        return

    columns = data_columns(corrected_df)
    if event.get("kind") == "split_columns_visual":
        guide_x = float(event["guideX"])
        affected_columns = columns_crossing_visual_guide(page_result, columns, guide_x)
        if not affected_columns:
            st.warning("No merged column crosses the submitted vertical guide.")
            return

        updated_df = split_columns_by_visual_guide(
            corrected_df,
            page_result,
            selected_page,
            affected_columns,
            guide_x,
            event.get("leftName", "left"),
            event.get("rightName", "right"),
        )
        st.session_state.history.append(clone_df(corrected_df))
        st.session_state.corrected_df = updated_df
        st.session_state.redo_stack = []
        st.session_state.operation_history.append(
            operation_record(
                "split_columns_visual",
                {
                    "page_number": selected_page,
                    "columns": affected_columns,
                    "guide_x": guide_x,
                    "left_name": event.get("leftName", "left"),
                    "right_name": event.get("rightName", "right"),
                    "source": "visual_workspace",
                },
            )
        )
        clear_page_drafts()
        st.success(f"Applied split to columns {affected_columns} on page {selected_page}.")
        st.rerun()

    if event.get("kind") == "split_rows_visual":
        guide_y = float(event["guideY"])
        affected_rows = rows_crossing_visual_guide(page_result, guide_y)
        if not affected_rows:
            st.warning("No merged row crosses the submitted horizontal guide.")
            return

        updated_df = split_rows_by_visual_guide(
            corrected_df,
            page_result,
            selected_page,
            affected_rows,
            guide_y,
        )
        st.session_state.history.append(clone_df(corrected_df))
        st.session_state.corrected_df = updated_df
        st.session_state.redo_stack = []
        st.session_state.operation_history.append(
            operation_record(
                "split_rows_visual",
                {
                    "page_number": selected_page,
                    "rows": affected_rows,
                    "guide_y": guide_y,
                    "source": "visual_workspace",
                },
            )
        )
        clear_page_drafts()
        st.success(f"Applied split to rows {affected_rows} on page {selected_page}.")
        st.rerun()

def handle_visual_edit_event(
    event: dict[str, Any] | None,
    corrected_df: pd.DataFrame,
    selected_page: int,
) -> None:
    if not event or st.session_state.pending_change is not None:
        return

    column = event.get("column")
    row_number = int(event["rowNumber"])
    value = event.get("value", "")
    preview_df = edit_cell(corrected_df, selected_page, row_number, column, value)
    set_pending(
        "edit_cell",
        f"Edit {column} at row {row_number} on page {selected_page}.",
        preview_df,
        {"page_number": selected_page, "row_number": row_number, "column": column, "source": "visual_workspace"},
    )
    st.rerun()


def handle_visual_move_event(
    event: dict[str, Any] | None,
    corrected_df: pd.DataFrame,
    selected_page: int,
) -> None:
    if not event or st.session_state.pending_change is not None:
        return

    preview_df = move_cell(
        corrected_df,
        selected_page,
        int(event["sourceRow"]),
        event["sourceColumn"],
        selected_page,
        int(event["targetRow"]),
        event["targetColumn"],
    )
    set_pending(
        "move_cell",
        (
            f"Move {event['sourceColumn']} row {int(event['sourceRow'])} "
            f"to {event['targetColumn']} row {int(event['targetRow'])} on page {selected_page}."
        ),
        preview_df,
        {
            "page_number": selected_page,
            "source_row": int(event["sourceRow"]),
            "source_column": event["sourceColumn"],
            "target_row": int(event["targetRow"]),
            "target_column": event["targetColumn"],
            "source": "visual_workspace",
        },
    )
    st.rerun()


def cell_lookup(page_result) -> dict[tuple[int, int], Any]:
    return {(cell.row_index, cell.col_index): cell for cell in getattr(page_result, "cells", [])}


def source_indexes_for_selection(page_result, row_number: int | None, col_index: int | None) -> set[int]:
    indexes: set[int] = set()
    for cell in getattr(page_result, "cells", []):
        if row_number is not None and cell.row_index != row_number:
            continue
        if col_index is not None and cell.col_index != col_index:
            continue
        indexes.update(cell.source_item_indexes)
    return indexes


def draw_pdf_overlay(
    pdf_bytes: bytes,
    page_number: int,
    zoom: float,
    page_result,
    selected_row: int | None,
    selected_col: int | None,
    selected_cell: tuple[int, int] | None,
    vertical_guide: float | None,
    horizontal_guide: float | None,
):
    image = render_page_image(pdf_bytes, page_number, zoom=zoom).convert("RGBA")
    overlay = ImageDraw.Draw(image, "RGBA")
    raw_items = page_result.raw_items

    if raw_items:
        x0 = min(item.x0 for item in raw_items) * zoom
        y0 = min(item.y0 for item in raw_items) * zoom
        x1 = max(item.x1 for item in raw_items) * zoom
        y1 = max(item.y1 for item in raw_items) * zoom
        overlay.rectangle([x0, y0, x1, y1], outline=(40, 40, 40, 180), width=2)

    row_indexes = source_indexes_for_selection(page_result, selected_row, None)
    col_indexes = source_indexes_for_selection(page_result, None, selected_col)
    cell_indexes: set[int] = set()
    if selected_cell:
        cell = cell_lookup(page_result).get(selected_cell)
        if cell:
            cell_indexes = set(cell.source_item_indexes)

    for index, item in enumerate(raw_items):
        quad = [(x * zoom, y * zoom) for x, y in item.quad] if item.quad else [
            (item.x0 * zoom, item.y0 * zoom),
            (item.x1 * zoom, item.y0 * zoom),
            (item.x1 * zoom, item.y1 * zoom),
            (item.x0 * zoom, item.y1 * zoom),
        ]
        overlay.polygon(quad, outline=(200, 30, 30, 190))

        if index in row_indexes:
            overlay.polygon(quad, fill=(255, 214, 79, 90), outline=(230, 160, 20, 240))
        if index in col_indexes:
            overlay.polygon(quad, fill=(68, 138, 255, 80), outline=(20, 90, 210, 240))
        if index in cell_indexes:
            overlay.polygon(quad, fill=(50, 210, 130, 120), outline=(10, 140, 70, 255))

    if vertical_guide is not None:
        x = vertical_guide * zoom
        overlay.line([(x, 0), (x, image.height)], fill=(20, 100, 240, 255), width=3)
    if horizontal_guide is not None:
        y = horizontal_guide * zoom
        overlay.line([(0, y), (image.width, y)], fill=(240, 100, 20, 255), width=3)

    return image.convert("RGB")


def split_column_by_visual_guide(
    df: pd.DataFrame,
    page_result,
    page_number: int,
    column: str,
    guide_x: float,
    left_name: str,
    right_name: str,
) -> pd.DataFrame:
    split_df = df.copy()
    column_index = data_columns(split_df).index(column) + 1
    position = split_df.columns.get_loc(column)
    source_cells = cell_lookup(page_result)

    left_values: list[str] = []
    right_values: list[str] = []
    for _, row in split_df.iterrows():
        if int(row["page_number"]) != page_number:
            left_values.append(str(row[column]))
            right_values.append("")
            continue

        cell = source_cells.get((int(row["row_number"]), column_index))
        if not cell:
            left_values.append(str(row[column]))
            right_values.append("")
            continue

        left_parts = []
        right_parts = []
        for item_index in cell.source_item_indexes:
            item = page_result.raw_items[item_index]
            if item.cx <= guide_x:
                left_parts.append(item.text)
            else:
                right_parts.append(item.text)
        left_values.append(" ".join(left_parts).strip())
        right_values.append(" ".join(right_parts).strip())

    split_df = split_df.drop(columns=[column])
    split_df.insert(position, left_name.strip() or f"{column}_left", left_values)
    split_df.insert(position + 1, right_name.strip() or f"{column}_right", right_values)
    return split_df


def columns_crossing_visual_guide(page_result, columns: list[str], guide_x: float) -> list[str]:
    source_cells = cell_lookup(page_result)
    affected: list[str] = []
    for col_position, column in enumerate(columns, start=1):
        for cell in source_cells.values():
            if cell.col_index != col_position:
                continue
            left_count = 0
            right_count = 0
            for item_index in cell.source_item_indexes:
                item = page_result.raw_items[item_index]
                if item.cx <= guide_x:
                    left_count += 1
                else:
                    right_count += 1
            if left_count and right_count:
                affected.append(column)
                break
    return affected


def rows_crossing_visual_guide(page_result, guide_y: float) -> list[int]:
    source_cells = cell_lookup(page_result)
    affected: list[int] = []
    for row_number in sorted({cell.row_index for cell in source_cells.values()}):
        top_count = 0
        bottom_count = 0
        for cell in source_cells.values():
            if cell.row_index != row_number:
                continue
            for item_index in cell.source_item_indexes:
                item = page_result.raw_items[item_index]
                if item.cy <= guide_y:
                    top_count += 1
                else:
                    bottom_count += 1
        if top_count and bottom_count:
            affected.append(row_number)
    return affected


def split_columns_by_visual_guide(
    df: pd.DataFrame,
    page_result,
    page_number: int,
    columns: list[str],
    guide_x: float,
    left_suffix: str,
    right_suffix: str,
) -> pd.DataFrame:
    split_df = df.copy()
    left_suffix = left_suffix.strip() or "left"
    right_suffix = right_suffix.strip() or "right"

    for column in columns:
        if column not in split_df.columns:
            continue
        split_df = split_column_by_visual_guide(
            split_df,
            page_result,
            page_number,
            column,
            guide_x,
            f"{column}_{left_suffix}",
            f"{column}_{right_suffix}",
        )
    return split_df


def split_rows_by_visual_guide(
    df: pd.DataFrame,
    page_result,
    page_number: int,
    row_numbers: list[int],
    guide_y: float,
) -> pd.DataFrame:
    split_df = df.copy()
    for row_number in sorted(row_numbers, reverse=True):
        split_df = split_row_by_visual_guide(split_df, page_result, page_number, row_number, guide_y)
    return split_df


def split_row_by_visual_guide(
    df: pd.DataFrame,
    page_result,
    page_number: int,
    row_number: int,
    guide_y: float,
) -> pd.DataFrame:
    target_indexes = df.index[(df["page_number"] == page_number) & (df["row_number"] == row_number)].tolist()
    if not target_indexes:
        raise ValueError("Selected row is not present in the corrected table.")

    index = target_indexes[0]
    source_cells = cell_lookup(page_result)
    top_row = df.loc[index].copy()
    bottom_row = df.loc[index].copy()

    for col_position, column in enumerate(data_columns(df), start=1):
        cell = source_cells.get((row_number, col_position))
        if not cell:
            bottom_row[column] = ""
            continue

        top_parts = []
        bottom_parts = []
        for item_index in cell.source_item_indexes:
            item = page_result.raw_items[item_index]
            if item.cy <= guide_y:
                top_parts.append(item.text)
            else:
                bottom_parts.append(item.text)
        top_row[column] = " ".join(top_parts).strip()
        bottom_row[column] = " ".join(bottom_parts).strip()

    split_df = df.copy()
    split_df.loc[index] = top_row
    split_df = pd.concat(
        [split_df.iloc[: index + 1], pd.DataFrame([bottom_row]), split_df.iloc[index + 1 :]],
        ignore_index=True,
    )
    return normalize_row_numbers(split_df)


def render_page_picker(page_numbers: list[int], state_key: str) -> int:
    if not page_numbers:
        raise ValueError("At least one page is required.")

    widget_key = f"active_page_{state_key}"
    current_page = st.session_state.get(widget_key, page_numbers[0])
    if current_page not in page_numbers:
        current_page = page_numbers[0]
        st.session_state[widget_key] = current_page

    current_index = page_numbers.index(current_page)
    prev_col, picker_col, next_col = st.columns([0.9, 3.2, 0.9])
    with prev_col:
        if st.button("Previous", use_container_width=True, disabled=current_index == 0):
            st.session_state[widget_key] = page_numbers[current_index - 1]
            st.rerun()
    with picker_col:
        selected_page = st.selectbox(
            "Extracted page",
            options=page_numbers,
            index=page_numbers.index(st.session_state.get(widget_key, current_page)),
            key=f"{widget_key}_select",
        )
        st.session_state[widget_key] = selected_page
    with next_col:
        if st.button("Next", use_container_width=True, disabled=current_index == len(page_numbers) - 1):
            st.session_state[widget_key] = page_numbers[current_index + 1]
            st.rerun()

    active_page = st.session_state.get(widget_key, selected_page)
    st.caption(f"Working on page {active_page} of {len(page_numbers)} extracted page(s).")
    return active_page


def main() -> None:
    uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
    zoom = st.sidebar.slider("Preview zoom", min_value=1.0, max_value=4.0, value=2.0, step=0.25)

    if not uploaded_file:
        st.info("Upload a PDF to open the correction workspace.")
        return

    pdf_bytes = uploaded_file.read()
    file_key = get_file_key(pdf_bytes)
    all_page_numbers = get_pdf_page_numbers(pdf_bytes)
    extraction_request = st.session_state.get("extraction_request")

    if not extraction_request or extraction_request.get("file_key") != file_key:
        st.session_state.extraction_request = {
            "file_key": file_key,
            "selected_pages": [],
            "submitted": False,
        }
        extraction_request = st.session_state.extraction_request

    st.sidebar.subheader("Extraction scope")
    selected_pages = st.sidebar.multiselect(
        "Pages to extract",
        options=all_page_numbers,
        default=extraction_request.get("selected_pages", []),
        help="Leave this empty to extract all pages in the PDF.",
        key=f"page_scope_{file_key}",
    )
    st.sidebar.caption("Leave the page selector empty to run extraction on every page.")

    run_extraction = st.sidebar.button("Run extraction", type="primary", use_container_width=True)
    if run_extraction:
        st.session_state.extraction_request = {
            "file_key": file_key,
            "selected_pages": selected_pages,
            "submitted": True,
        }
        st.session_state.selective_override = None
        st.session_state.file_key = ""
        clear_page_drafts()
        st.rerun()

    if not st.session_state.extraction_request.get("submitted"):
        st.info("Choose pages in the sidebar, or leave the selection empty for all pages, then click `Run extraction`.")
        st.caption(f"This PDF has {len(all_page_numbers)} page(s).")
        return

    selected_pages = st.session_state.extraction_request.get("selected_pages", [])
    active_page_numbers = selected_pages or all_page_numbers
    active_state_key = extraction_state_key(file_key, active_page_numbers)

    base_page_results = run_parser(pdf_bytes, PARSER_CACHE_VERSION, tuple(active_page_numbers))

    if not base_page_results:
        st.warning("No pages were parsed.")
        return

    selective_override = st.session_state.get("selective_override")
    if selective_override and selective_override.get("file_key") == file_key:
        page_results = selective_override["page_results"]
        selective_state_key = selective_override["state_key"]
        initial_df = build_initial_table(page_results)
        initialize_selective_state(
            selective_state_key,
            initial_df,
            page_results,
            selective_override["operation"],
        )
        workspace_state_key = selective_state_key
    else:
        page_results = base_page_results
        initial_df = build_initial_table(page_results)
        initialize_state(active_state_key, initial_df, page_results)
        workspace_state_key = active_state_key

    corrected_df = st.session_state.corrected_df
    page_numbers = [result.page_number for result in page_results]
    selected_page = render_page_picker(page_numbers, workspace_state_key)
    page_result = next(result for result in page_results if result.page_number == selected_page)
    page_df = page_dataframe(corrected_df, selected_page)
    page_editor_df = page_draft_or_default(workspace_state_key, selected_page, page_df)
    columns = data_columns(corrected_df)

    toolbar_cols = st.columns([1, 1, 1, 1, 5])
    with toolbar_cols[0]:
        if st.button("Undo", disabled=not st.session_state.history, use_container_width=True):
            undo()
            st.rerun()
    with toolbar_cols[1]:
        if st.button("Redo", disabled=not st.session_state.redo_stack, use_container_width=True):
            redo()
            st.rerun()
    with toolbar_cols[2]:
        st.metric("Rows", len(corrected_df))
    with toolbar_cols[3]:
        st.metric("Columns", len(columns))

    pending = st.session_state.pending_change
    if pending is not None:
        st.warning(pending.description)
        preview = pending.preview_df
        st.dataframe(preview.head(20), use_container_width=True, height=220)
        accept_col, reject_col = st.columns([1, 1])
        with accept_col:
            if st.button("Accept change", type="primary", use_container_width=True):
                accept_pending()
                st.rerun()
        with reject_col:
            if st.button("Reject change", use_container_width=True):
                reject_pending()
                st.rerun()

    st.subheader(f"Page {selected_page} Visual Correction Workspace")
    visual_result = VISUAL_WORKSPACE(
        key=f"visual_workspace_{file_key}_{selected_page}",
        data=visual_workspace_payload(pdf_bytes, selected_page, zoom, page_result, corrected_df),
        height=780,
        on_merge_change=lambda: None,
        on_split_change=lambda: None,
        on_edit_cell_change=lambda: None,
        on_move_change=lambda: None,
    )
    handle_visual_merge_event(getattr(visual_result, "merge", None), corrected_df, selected_page)
    handle_visual_split_event(getattr(visual_result, "split", None), corrected_df, page_result, selected_page)
    handle_visual_edit_event(getattr(visual_result, "edit_cell", None), corrected_df, selected_page)
    handle_visual_move_event(getattr(visual_result, "move", None), corrected_df, selected_page)

    workspace_left, workspace_right = st.columns([1.1, 1])

    with workspace_left:
        st.subheader(f"Page {selected_page} Manual Cell Editor")
        st.caption("Draft edits on this page stay in place while you move between pages. Use `Save page edits` to commit them.")
        edited_page_df = st.data_editor(
            page_editor_df,
            width="stretch",
            height=430,
            disabled=["row_number"],
            num_rows="fixed",
            key=f"page_editor_{workspace_state_key}_{selected_page}",
        )
        store_page_draft(workspace_state_key, selected_page, edited_page_df)
        if st.button("Save page edits", use_container_width=True):
            commit_page_edits(workspace_state_key, selected_page, edited_page_df)
            st.rerun()

    with workspace_right:
        st.subheader("Advanced Fallback Controls")
        with st.expander("Manual structural controls", expanded=False):
            operation = st.selectbox(
                "Operation",
                [
                    "Merge columns",
                    "Split column by delimiter",
                    "Merge rows",
                    "Split row by delimiter",
                    "Insert row",
                    "Insert column",
                    "Rename column",
                    "Remove rows",
                    "Remove columns",
                ],
            )

        try:
            if operation == "Merge columns":
                merge_targets = st.multiselect("Columns", options=columns, default=columns[:2])
                output_name = st.text_input("Merged column name", value=merge_targets[0] if merge_targets else "merged_col")
                if st.button("Preview merge columns", use_container_width=True):
                    preview_df = merge_columns(corrected_df, merge_targets, output_name)
                    set_pending(
                        "merge_columns",
                        f"Merge columns {merge_targets} into {output_name}.",
                        preview_df,
                        {"columns": merge_targets, "output_name": output_name},
                    )
                    st.rerun()

            elif operation == "Split column by delimiter":
                target = st.selectbox("Column", options=columns)
                delimiter = st.text_input("Delimiter", value=" ")
                left_name = st.text_input("Left column name", value=f"{target}_left")
                right_name = st.text_input("Right column name", value=f"{target}_right")
                if st.button("Preview delimiter column split", use_container_width=True):
                    preview_df = split_column_by_delimiter(corrected_df, target, delimiter, left_name, right_name)
                    set_pending(
                        "split_column_delimiter",
                        f"Split {target} by delimiter.",
                        preview_df,
                        {"column": target, "delimiter": delimiter},
                    )
                    st.rerun()

            elif operation == "Merge rows":
                page_indexes = corrected_df.index[corrected_df["page_number"] == selected_page].tolist()
                row_labels = {
                    int(corrected_df.at[index, "row_number"]): index
                    for index in page_indexes
                }
                row_numbers = st.multiselect("Rows on selected page", options=list(row_labels.keys()))
                if st.button("Preview merge rows", use_container_width=True):
                    preview_df = merge_rows(corrected_df, [row_labels[row_number] for row_number in row_numbers])
                    set_pending(
                        "merge_rows",
                        f"Merge rows {row_numbers} on page {selected_page}.",
                        preview_df,
                        {"page_number": selected_page, "rows": row_numbers},
                    )
                    st.rerun()

            elif operation == "Split row by delimiter":
                page_indexes = corrected_df.index[corrected_df["page_number"] == selected_page].tolist()
                row_labels = {
                    int(corrected_df.at[index, "row_number"]): index
                    for index in page_indexes
                }
                row_number = st.selectbox("Row on selected page", options=list(row_labels.keys()))
                delimiter = st.text_input("Delimiter", value=" ")
                if st.button("Preview delimiter row split", use_container_width=True):
                    preview_df = split_row_by_delimiter(corrected_df, row_labels[row_number], delimiter)
                    set_pending(
                        "split_row_delimiter",
                        f"Split row {row_number} by delimiter.",
                        preview_df,
                        {"page_number": selected_page, "row_number": row_number, "delimiter": delimiter},
                    )
                    st.rerun()

            elif operation == "Insert row":
                after_row = st.selectbox(
                    "Insert after row",
                    options=[None] + page_df["row_number"].astype(int).tolist(),
                    format_func=lambda value: "Top of page" if value is None else str(value),
                )
                if st.button("Preview insert row", use_container_width=True):
                    if after_row is None:
                        after_index = corrected_df.index[corrected_df["page_number"] == selected_page].min() - 1
                    else:
                        after_index = corrected_df.index[
                            (corrected_df["page_number"] == selected_page) & (corrected_df["row_number"] == after_row)
                        ][0]
                    preview_df = insert_row(corrected_df, int(after_index), selected_page)
                    set_pending(
                        "insert_row",
                        f"Insert row on page {selected_page}.",
                        preview_df,
                        {"page_number": selected_page, "after_row": after_row},
                    )
                    st.rerun()

            elif operation == "Insert column":
                after_column = st.selectbox("Insert after column", options=[None] + columns, format_func=lambda value: "End" if value is None else value)
                column_name = st.text_input("New column name", value=f"col_{len(columns) + 1}")
                if st.button("Preview insert column", use_container_width=True):
                    preview_df = insert_column(corrected_df, after_column, column_name)
                    set_pending(
                        "insert_column",
                        f"Insert column {column_name}.",
                        preview_df,
                        {"after_column": after_column, "column_name": column_name},
                    )
                    st.rerun()

            elif operation == "Rename column":
                target = st.selectbox("Column", options=columns)
                new_name = st.text_input("New name", value=target)
                if st.button("Preview rename column", use_container_width=True):
                    preview_df = rename_column(corrected_df, target, new_name)
                    set_pending(
                        "rename_column",
                        f"Rename {target} to {new_name}.",
                        preview_df,
                        {"column": target, "new_name": new_name},
                    )
                    st.rerun()

            elif operation == "Remove rows":
                page_indexes = corrected_df.index[corrected_df["page_number"] == selected_page].tolist()
                row_labels = {
                    int(corrected_df.at[index, "row_number"]): index
                    for index in page_indexes
                }
                row_numbers = st.multiselect("Rows on selected page", options=list(row_labels.keys()))
                if st.button("Preview remove rows", use_container_width=True):
                    preview_df = delete_rows(corrected_df, [row_labels[row_number] for row_number in row_numbers])
                    set_pending(
                        "remove_rows",
                        f"Remove rows {row_numbers} on page {selected_page}.",
                        preview_df,
                        {"page_number": selected_page, "rows": row_numbers},
                    )
                    st.rerun()

            elif operation == "Remove columns":
                targets = st.multiselect("Columns", options=columns)
                if st.button("Preview remove columns", use_container_width=True):
                    preview_df = delete_columns(corrected_df, targets)
                    set_pending(
                        "remove_columns",
                        f"Remove columns {targets}.",
                        preview_df,
                        {"columns": targets},
                    )
                    st.rerun()

        except Exception as exc:
            st.error(str(exc))

    tabs = st.tabs(["Corrected Table", "Selective Extraction", "Lineage", "Export"])

    with tabs[0]:
        st.dataframe(st.session_state.corrected_df, use_container_width=True, height=420)

    with tabs[1]:
        if selective_override and selective_override.get("file_key") == file_key:
            st.caption("Selective extraction is active for this upload.")
            if st.button("Return to full-page extraction", use_container_width=True):
                st.session_state.selective_override = None
                st.session_state.file_key = ""
                clear_page_drafts()
                st.rerun()

        selective_page_number = st.selectbox(
            "Page",
            options=[result.page_number for result in base_page_results],
            key="selective_page_number",
        )
        selective_page_result = next(
            result for result in base_page_results if result.page_number == selective_page_number
        )
        polygon_result = POLYGON_SELECTOR(
            key=f"polygon_selector_{file_key}_{selective_page_number}",
            data=polygon_selector_payload(pdf_bytes, selective_page_number, zoom, selective_page_result),
            height=780,
            on_polygon_change=lambda: None,
        )
        polygon_event = getattr(polygon_result, "polygon", None)
        if polygon_event:
            polygon = polygon_event.get("points", [])
            if len(polygon) < 3:
                st.warning("Draw at least three polygon points around one table.")
            else:
                selective_result = build_selective_page_result(selective_page_result, polygon)
                if not selective_result.raw_items:
                    st.warning("No OCR boxes were found inside the selected polygon.")
                else:
                    load_selective_extraction(file_key, selective_result)
                    st.rerun()

    with tabs[2]:
        metadata = {
            "original_columns": st.session_state.original_df.columns.tolist(),
            "corrected_columns": st.session_state.corrected_df.columns.tolist(),
            "original_row_count": len(st.session_state.original_df),
            "corrected_row_count": len(st.session_state.corrected_df),
            "pending_change": pending_to_metadata(st.session_state.pending_change) if st.session_state.pending_change else None,
            "operation_history": st.session_state.operation_history,
        }
        st.json(metadata)

    with tabs[3]:
        st.download_button(
            "Download corrected Excel",
            data=dataframe_to_excel(st.session_state.corrected_df),
            file_name="glyphon_corrected_table.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
        st.download_button(
            "Download corrected CSV",
            data=dataframe_to_csv(st.session_state.corrected_df),
            file_name="glyphon_corrected_table.csv",
            mime="text/csv",
            use_container_width=True,
        )
        export_metadata = {
            "original_extraction": st.session_state.original_df.to_dict(orient="records"),
            "corrected_table": st.session_state.corrected_df.to_dict(orient="records"),
            "operation_history": st.session_state.operation_history,
        }
        st.download_button(
            "Download correction metadata",
            data=json.dumps(export_metadata, indent=2).encode("utf-8"),
            file_name="glyphon_correction_metadata.json",
            mime="application/json",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
