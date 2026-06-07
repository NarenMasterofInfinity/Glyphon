from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import streamlit as st


REVIEW_HTML = """
<div id="glyphon-review" class="review-shell">
  <section class="pdf-pane">
    <div class="review-toolbar">
      <label>
        Warning
        <select id="warning-filter"></select>
      </label>
      <button id="clear-selection" type="button" title="Clear selection">Clear</button>
    </div>
    <div class="page-frame">
      <img id="page-image" />
      <svg id="overlay"></svg>
    </div>
  </section>
  <section class="table-pane">
    <div id="table-scroll">
      <table id="review-table"></table>
    </div>
  </section>
  <aside class="detail-pane">
    <div id="details"></div>
  </aside>
</div>
"""

REVIEW_CSS = """
.review-shell {
  --panel-bg: #f6f7fb;
  --surface-bg: #ffffff;
  --surface-strong: #eef1f7;
  --text-color: #172033;
  --muted: #5f6b85;
  --border: rgba(23, 32, 51, 0.14);
  --cell: #2563eb;
  --warn: #d97706;
  --selected: #059669;
  display: grid;
  grid-template-columns: minmax(360px, 1.05fr) minmax(400px, 0.95fr) minmax(240px, 0.45fr);
  gap: 12px;
  height: 780px;
  color-scheme: light dark;
  color: var(--text-color);
  font-family: Inter, Aptos, "Segoe UI", sans-serif;
}
@media (prefers-color-scheme: dark) {
  .review-shell {
    --panel-bg: #171d28;
    --surface-bg: #111722;
    --surface-strong: #1c2432;
    --text-color: #edf2ff;
    --muted: #aeb8cf;
    --border: rgba(237, 242, 255, 0.14);
  }
}
.pdf-pane,
.table-pane,
.detail-pane {
  min-width: 0;
  overflow: auto;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--panel-bg);
}
.review-toolbar {
  position: sticky;
  top: 0;
  z-index: 5;
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 9px 10px;
  border-bottom: 1px solid var(--border);
  background: var(--panel-bg);
}
.review-toolbar label {
  display: flex;
  align-items: center;
  gap: 7px;
  font-size: 12px;
  color: var(--muted);
}
.review-toolbar select,
.review-toolbar button {
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 6px 8px;
  background: var(--surface-bg);
  color: var(--text-color);
}
.page-frame {
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
  width: 100%;
  height: 100%;
  overflow: visible;
}
.cell-box {
  fill: rgba(37, 99, 235, 0.06);
  stroke: var(--cell);
  stroke-width: 1.5;
  vector-effect: non-scaling-stroke;
  cursor: pointer;
}
.warning-box {
  fill: rgba(217, 119, 6, 0.08);
  stroke: var(--warn);
  stroke-width: 2;
  stroke-dasharray: 7 4;
  vector-effect: non-scaling-stroke;
  cursor: pointer;
}
.row-hit {
  fill: rgba(217, 119, 6, 0.13);
}
.col-hit {
  fill: rgba(37, 99, 235, 0.12);
}
.selected-box {
  fill: rgba(5, 150, 105, 0.23);
  stroke: var(--selected);
  stroke-width: 2.5;
}
.guide-line {
  stroke: var(--selected);
  stroke-width: 2;
  vector-effect: non-scaling-stroke;
}
#table-scroll {
  height: 100%;
  overflow: auto;
}
#review-table {
  width: 100%;
  min-width: 620px;
  border-collapse: collapse;
  table-layout: fixed;
  font-size: 13px;
}
#review-table th,
#review-table td {
  border: 1px solid var(--border);
  padding: 7px 8px;
  vertical-align: top;
  overflow-wrap: anywhere;
  color: var(--text-color);
}
#review-table th {
  position: sticky;
  top: 0;
  z-index: 2;
  background: var(--surface-strong);
  text-align: left;
  font-weight: 650;
}
#review-table td {
  background: var(--surface-bg);
  cursor: pointer;
}
#review-table td.selected-cell {
  outline: 2px solid var(--selected);
  outline-offset: -2px;
  background: color-mix(in srgb, var(--surface-bg) 78%, #059669 22%);
}
#review-table td.has-warning {
  box-shadow: inset 3px 0 0 var(--warn);
}
#details {
  padding: 12px;
  font-size: 13px;
}
#details h3 {
  margin: 0 0 8px;
  font-size: 15px;
}
#details dl {
  display: grid;
  grid-template-columns: minmax(88px, auto) 1fr;
  gap: 6px 9px;
}
#details dt {
  color: var(--muted);
}
#details dd {
  margin: 0;
  overflow-wrap: anywhere;
}
.warning-chip {
  display: inline-block;
  margin: 2px 4px 2px 0;
  padding: 2px 6px;
  border-radius: 999px;
  background: rgba(217, 119, 6, 0.15);
  color: var(--text-color);
}
@media (max-width: 1050px) {
  .review-shell {
    grid-template-columns: 1fr;
    height: auto;
  }
  .pdf-pane,
  .table-pane,
  .detail-pane {
    height: 560px;
  }
}
"""

REVIEW_JS = """
export default function(component) {
  const { data, parentElement, setTriggerValue } = component;
  const root = parentElement.querySelector("#glyphon-review");
  if (!root || !data) return;

  const image = root.querySelector("#page-image");
  const overlay = root.querySelector("#overlay");
  const table = root.querySelector("#review-table");
  const scroll = root.querySelector("#table-scroll");
  const details = root.querySelector("#details");
  const filter = root.querySelector("#warning-filter");
  const clearButton = root.querySelector("#clear-selection");

  image.src = data.imageUrl;
  overlay.innerHTML = "";
  table.innerHTML = "";
  details.innerHTML = "<h3>Selection</h3><p>Choose a PDF box or table cell.</p>";

  const cellById = new Map((data.cells || []).map((cell) => [cell.cell_id, cell]));
  const warningsById = new Map((data.warnings || []).map((warning) => [warning.issue_id, warning]));
  const cellsByRow = new Map();
  const cellsByCol = new Map();
  const tableCellById = new Map();
  let selectedCellId = null;

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function addGrouped(map, key, value) {
    if (!map.has(key)) map.set(key, []);
    map.get(key).push(value);
  }

  for (const cell of data.cells || []) {
    addGrouped(cellsByRow, cell.row_id, cell);
    addGrouped(cellsByCol, cell.column_id, cell);
  }

  function rectFromBox(box, className, dataset) {
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", box[0]);
    rect.setAttribute("y", box[1]);
    rect.setAttribute("width", Math.max(0, box[2] - box[0]));
    rect.setAttribute("height", Math.max(0, box[3] - box[1]));
    rect.setAttribute("class", className);
    for (const [key, value] of Object.entries(dataset || {})) {
      rect.dataset[key] = value;
    }
    return rect;
  }

  function unionBox(cells) {
    const boxes = cells.map((cell) => cell.bbox).filter(Boolean);
    if (!boxes.length) return null;
    return [
      Math.min(...boxes.map((box) => box[0])),
      Math.min(...boxes.map((box) => box[1])),
      Math.max(...boxes.map((box) => box[2])),
      Math.max(...boxes.map((box) => box[3])),
    ];
  }

  function renderDetails(cell) {
    if (!cell) {
      details.innerHTML = "<h3>Selection</h3><p>Choose a PDF box or table cell.</p>";
      return;
    }
    const chips = (cell.warning_details || [])
      .map((warning) => `<span class="warning-chip">${escapeHtml(warning.issue_type)}</span>`)
      .join("");
    details.innerHTML = `
      <h3>${escapeHtml(cell.header)}</h3>
      <dl>
        <dt>Text</dt><dd>${escapeHtml(cell.text)}</dd>
        <dt>Cell</dt><dd>${escapeHtml(cell.cell_id)}</dd>
        <dt>Row</dt><dd>${escapeHtml(cell.row_id)}</dd>
        <dt>Column</dt><dd>${escapeHtml(cell.column_id)}</dd>
        <dt>BBox</dt><dd>${escapeHtml(JSON.stringify(cell.bbox || []))}</dd>
        <dt>Score</dt><dd>${escapeHtml(cell.assignment_score)}</dd>
        <dt>Warnings</dt><dd>${chips || "None"}</dd>
      </dl>
    `;
  }

  function drawGuide(box, orientation) {
    if (!box) return;
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("class", "guide-line");
    if (orientation === "row") {
      const y = (box[1] + box[3]) / 2;
      line.setAttribute("x1", 0);
      line.setAttribute("x2", data.pageWidth);
      line.setAttribute("y1", y);
      line.setAttribute("y2", y);
    } else {
      const x = (box[0] + box[2]) / 2;
      line.setAttribute("x1", x);
      line.setAttribute("x2", x);
      line.setAttribute("y1", 0);
      line.setAttribute("y2", data.pageHeight);
    }
    overlay.appendChild(line);
  }

  function clearSelection() {
    selectedCellId = null;
    overlay.querySelectorAll(".selected-box,.row-hit,.col-hit").forEach((node) => {
      node.classList.remove("selected-box", "row-hit", "col-hit");
    });
    overlay.querySelectorAll(".guide-line").forEach((node) => node.remove());
    table.querySelectorAll(".selected-cell").forEach((node) => node.classList.remove("selected-cell"));
    renderDetails(null);
  }

  function selectCell(cellId, shouldScroll) {
    const cell = cellById.get(cellId);
    if (!cell) return;
    clearSelection();
    selectedCellId = cellId;
    const rowCells = cellsByRow.get(cell.row_id) || [];
    const colCells = cellsByCol.get(cell.column_id) || [];
    const rowIds = new Set(rowCells.map((item) => item.cell_id));
    const colIds = new Set(colCells.map((item) => item.cell_id));
    overlay.querySelectorAll(".cell-box").forEach((node) => {
      if (rowIds.has(node.dataset.cellId)) node.classList.add("row-hit");
      if (colIds.has(node.dataset.cellId)) node.classList.add("col-hit");
      if (node.dataset.cellId === cellId) node.classList.add("selected-box");
    });
    drawGuide(cell.bbox, "row");
    drawGuide(cell.bbox, "col");
    const td = tableCellById.get(cellId);
    if (td) {
      td.classList.add("selected-cell");
      if (shouldScroll) td.scrollIntoView({ block: "center", inline: "center", behavior: "smooth" });
    }
    renderDetails(cell);
  }

  function selectWarning(issueId) {
    const warning = warningsById.get(issueId);
    if (!warning) return;
    const affected = warning.affected_cells || [];
    if (affected.length) {
      selectCell(affected[0].cell_id, true);
    }
    details.innerHTML = `
      <h3>${escapeHtml(warning.issue_type)}</h3>
      <dl>
        <dt>Issue</dt><dd>${escapeHtml(warning.issue_id)}</dd>
        <dt>Status</dt><dd>${escapeHtml(warning.status)}</dd>
        <dt>Severity</dt><dd>${escapeHtml(warning.severity)}</dd>
        <dt>Explanation</dt><dd>${escapeHtml(warning.explanation)}</dd>
        <dt>Action</dt><dd>${escapeHtml(warning.suggested_action)}</dd>
      </dl>
    `;
  }

  function renderOverlay() {
    overlay.setAttribute("viewBox", `0 0 ${data.pageWidth} ${data.pageHeight}`);
    overlay.innerHTML = "";
    const selectedWarningType = filter.value;
    for (const cell of data.cells || []) {
      if (!cell.bbox) continue;
      const rect = rectFromBox(cell.bbox, "cell-box", { cellId: cell.cell_id });
      rect.onclick = () => selectCell(cell.cell_id, true);
      overlay.appendChild(rect);
    }
    for (const warning of data.warnings || []) {
      if (!warning.warning_box) continue;
      if (selectedWarningType !== "All warnings" && warning.issue_type !== selectedWarningType) continue;
      const rect = rectFromBox(warning.warning_box, "warning-box", { issueId: warning.issue_id });
      rect.onclick = () => selectWarning(warning.issue_id);
      overlay.appendChild(rect);
    }
    if (selectedCellId) selectCell(selectedCellId, false);
  }

  const warningTypes = ["All warnings", ...new Set((data.warnings || []).map((warning) => warning.issue_type))];
  filter.innerHTML = warningTypes.map((item) => `<option value="${escapeHtml(item)}">${escapeHtml(item)}</option>`).join("");
  filter.onchange = renderOverlay;
  clearButton.onclick = clearSelection;

  const headerRow = document.createElement("tr");
  for (const column of data.columns || []) {
    const th = document.createElement("th");
    th.textContent = column.name;
    headerRow.appendChild(th);
  }
  table.appendChild(headerRow);

  for (const row of data.rows || []) {
    const tr = document.createElement("tr");
    for (const column of data.columns || []) {
      const cell = (row.cells || []).find((item) => item.column_id === column.column_id);
      const td = document.createElement("td");
      td.textContent = cell ? cell.text : "";
      if (cell) {
        td.dataset.cellId = cell.cell_id;
        tableCellById.set(cell.cell_id, td);
        if ((cell.warning_ids || []).length) td.classList.add("has-warning");
        td.onclick = () => selectCell(cell.cell_id, false);
        td.ondblclick = () => {
          const value = window.prompt("Edit cell value", cell.text || "");
          if (value === null) return;
          setTriggerValue("edit_cell_text", { cell_id: cell.cell_id, text: value });
        };
      }
      tr.appendChild(td);
    }
    table.appendChild(tr);
  }

  renderOverlay();
}
"""

HAS_COMPONENT_V2 = hasattr(getattr(st, "components", None), "v2") and hasattr(st.components.v2, "component")

if HAS_COMPONENT_V2:
    REVIEW_COMPONENT = st.components.v2.component(
        "glyphon_api_review",
        html=REVIEW_HTML,
        css=REVIEW_CSS,
        js=REVIEW_JS,
    )
else:
    REVIEW_COMPONENT = None


def flatten_rows(table: dict[str, Any]) -> list[dict[str, Any]]:
    return table.get("metadata_rows", []) + table.get("header_rows", []) + table.get("data_rows", [])


def normalized_row_cells(row: dict[str, Any]) -> list[dict[str, Any]]:
    cells = []
    for cell in row.get("cells", []):
        normalized = dict(cell)
        normalized.setdefault("row_id", row.get("row_id"))
        normalized.setdefault("page_number", row.get("page_number"))
        normalized.setdefault("source_row_number", row.get("source_row_number"))
        cells.append(normalized)
    return cells


def review_payload(
    *,
    result: dict[str, Any],
    table_id: str,
    image_url: str,
) -> dict[str, Any]:
    table = next((item for item in result.get("tables", []) if item["table_id"] == table_id), None)
    if not table:
        return {
            "imageUrl": image_url,
            "pageWidth": 1,
            "pageHeight": 1,
            "columns": [],
            "rows": [],
            "cells": [],
            "warnings": [],
        }

    page_number = str(table.get("page_number", 1))
    dimensions = result.get("page_dimensions", {}).get(page_number, [1, 1])
    rows = flatten_rows(table)
    cells = [cell for row in rows for cell in normalized_row_cells(row)]
    return {
        "imageUrl": image_url,
        "pageWidth": dimensions[0] or 1,
        "pageHeight": dimensions[1] or 1,
        "columns": table.get("columns", []),
        "rows": rows,
        "cells": cells,
        "warnings": result.get("warnings", []),
    }


def render_review(payload: dict[str, Any], *, key: str, height: int = 800):
    if HAS_COMPONENT_V2:
        return REVIEW_COMPONENT(
            key=key,
            data=payload,
            height=height,
            on_edit_cell_text_change=lambda: None,
        )

    html = (
        REVIEW_HTML
        + "<style>"
        + REVIEW_CSS
        + "</style>"
        + "<script>"
        + "const __glyphonPayload = "
        + json.dumps(payload)
        + ";"
        + """
(() => {
  const root = document.getElementById("glyphon-review");
  if (!root) return;
  const image = root.querySelector("#page-image");
  const overlay = root.querySelector("#overlay");
  const table = root.querySelector("#review-table");
  const details = root.querySelector("#details");
  const filter = root.querySelector("#warning-filter");
  const clearButton = root.querySelector("#clear-selection");
  const data = __glyphonPayload;

  image.src = data.imageUrl;
  details.innerHTML = "<h3>Selection</h3><p>Choose a PDF box or table cell.</p>";
  const cellById = new Map((data.cells || []).map((cell) => [cell.cell_id, cell]));
  const warningsById = new Map((data.warnings || []).map((warning) => [warning.issue_id, warning]));
  const cellsByRow = new Map();
  const cellsByCol = new Map();
  const tableCellById = new Map();
  let selectedCellId = null;

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function addGrouped(map, key, value) {
    if (!map.has(key)) map.set(key, []);
    map.get(key).push(value);
  }

  for (const cell of data.cells || []) {
    addGrouped(cellsByRow, cell.row_id, cell);
    addGrouped(cellsByCol, cell.column_id, cell);
  }

  function rectFromBox(box, className, dataset) {
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", box[0]);
    rect.setAttribute("y", box[1]);
    rect.setAttribute("width", Math.max(0, box[2] - box[0]));
    rect.setAttribute("height", Math.max(0, box[3] - box[1]));
    rect.setAttribute("class", className);
    for (const [key, value] of Object.entries(dataset || {})) rect.dataset[key] = value;
    return rect;
  }

  function unionBox(cells) {
    const boxes = cells.map((cell) => cell.bbox).filter(Boolean);
    if (!boxes.length) return null;
    return [
      Math.min(...boxes.map((box) => box[0])),
      Math.min(...boxes.map((box) => box[1])),
      Math.max(...boxes.map((box) => box[2])),
      Math.max(...boxes.map((box) => box[3])),
    ];
  }

  function renderDetails(cell) {
    if (!cell) {
      details.innerHTML = "<h3>Selection</h3><p>Choose a PDF box or table cell.</p>";
      return;
    }
    const chips = (cell.warning_details || [])
      .map((warning) => `<span class="warning-chip">${escapeHtml(warning.issue_type)}</span>`)
      .join("");
    details.innerHTML = `
      <h3>${escapeHtml(cell.header)}</h3>
      <dl>
        <dt>Text</dt><dd>${escapeHtml(cell.text)}</dd>
        <dt>Cell</dt><dd>${escapeHtml(cell.cell_id)}</dd>
        <dt>Row</dt><dd>${escapeHtml(cell.row_id)}</dd>
        <dt>Column</dt><dd>${escapeHtml(cell.column_id)}</dd>
        <dt>BBox</dt><dd>${escapeHtml(JSON.stringify(cell.bbox || []))}</dd>
        <dt>Score</dt><dd>${escapeHtml(cell.assignment_score)}</dd>
        <dt>Warnings</dt><dd>${chips || "None"}</dd>
      </dl>
    `;
  }

  function drawGuide(box, orientation) {
    if (!box) return;
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("class", "guide-line");
    if (orientation === "row") {
      const y = (box[1] + box[3]) / 2;
      line.setAttribute("x1", 0);
      line.setAttribute("x2", data.pageWidth);
      line.setAttribute("y1", y);
      line.setAttribute("y2", y);
    } else {
      const x = (box[0] + box[2]) / 2;
      line.setAttribute("x1", x);
      line.setAttribute("x2", x);
      line.setAttribute("y1", 0);
      line.setAttribute("y2", data.pageHeight);
    }
    overlay.appendChild(line);
  }

  function clearSelection() {
    selectedCellId = null;
    overlay.querySelectorAll(".selected-box,.row-hit,.col-hit").forEach((node) => {
      node.classList.remove("selected-box", "row-hit", "col-hit");
    });
    overlay.querySelectorAll(".guide-line").forEach((node) => node.remove());
    table.querySelectorAll(".selected-cell").forEach((node) => node.classList.remove("selected-cell"));
    renderDetails(null);
  }

  function selectCell(cellId, shouldScroll) {
    const cell = cellById.get(cellId);
    if (!cell) return;
    clearSelection();
    selectedCellId = cellId;
    const rowCells = cellsByRow.get(cell.row_id) || [];
    const colCells = cellsByCol.get(cell.column_id) || [];
    const rowIds = new Set(rowCells.map((item) => item.cell_id));
    const colIds = new Set(colCells.map((item) => item.cell_id));
    overlay.querySelectorAll(".cell-box").forEach((node) => {
      if (rowIds.has(node.dataset.cellId)) node.classList.add("row-hit");
      if (colIds.has(node.dataset.cellId)) node.classList.add("col-hit");
      if (node.dataset.cellId === cellId) node.classList.add("selected-box");
    });
    drawGuide(cell.bbox, "row");
    drawGuide(cell.bbox, "col");
    const td = tableCellById.get(cellId);
    if (td) {
      td.classList.add("selected-cell");
      if (shouldScroll) td.scrollIntoView({ block: "center", inline: "center", behavior: "smooth" });
    }
    renderDetails(cell);
  }

  function selectWarning(issueId) {
    const warning = warningsById.get(issueId);
    if (!warning) return;
    const affected = warning.affected_cells || [];
    if (affected.length) selectCell(affected[0].cell_id, true);
    details.innerHTML = `
      <h3>${escapeHtml(warning.issue_type)}</h3>
      <dl>
        <dt>Issue</dt><dd>${escapeHtml(warning.issue_id)}</dd>
        <dt>Status</dt><dd>${escapeHtml(warning.status)}</dd>
        <dt>Severity</dt><dd>${escapeHtml(warning.severity)}</dd>
        <dt>Explanation</dt><dd>${escapeHtml(warning.explanation)}</dd>
        <dt>Action</dt><dd>${escapeHtml(warning.suggested_action)}</dd>
      </dl>
    `;
  }

  function renderOverlay() {
    overlay.setAttribute("viewBox", `0 0 ${data.pageWidth} ${data.pageHeight}`);
    overlay.innerHTML = "";
    const selectedWarningType = filter.value;
    for (const cell of data.cells || []) {
      if (!cell.bbox) continue;
      const rect = rectFromBox(cell.bbox, "cell-box", { cellId: cell.cell_id });
      rect.onclick = () => selectCell(cell.cell_id, true);
      overlay.appendChild(rect);
    }
    for (const warning of data.warnings || []) {
      if (!warning.warning_box) continue;
      if (selectedWarningType !== "All warnings" && warning.issue_type !== selectedWarningType) continue;
      const rect = rectFromBox(warning.warning_box, "warning-box", { issueId: warning.issue_id });
      rect.onclick = () => selectWarning(warning.issue_id);
      overlay.appendChild(rect);
    }
    if (selectedCellId) selectCell(selectedCellId, false);
  }

  const warningTypes = ["All warnings", ...new Set((data.warnings || []).map((warning) => warning.issue_type))];
  filter.innerHTML = warningTypes.map((item) => `<option value="${escapeHtml(item)}">${escapeHtml(item)}</option>`).join("");
  filter.onchange = renderOverlay;
  clearButton.onclick = clearSelection;

  const headerRow = document.createElement("tr");
  for (const column of data.columns || []) {
    const th = document.createElement("th");
    th.textContent = column.name;
    headerRow.appendChild(th);
  }
  table.appendChild(headerRow);

  for (const row of data.rows || []) {
    const tr = document.createElement("tr");
    for (const column of data.columns || []) {
      const cell = (row.cells || []).find((item) => item.column_id === column.column_id);
      const td = document.createElement("td");
      td.textContent = cell ? cell.text : "";
      if (cell) {
        td.dataset.cellId = cell.cell_id;
        tableCellById.set(cell.cell_id, td);
        if ((cell.warning_ids || []).length) td.classList.add("has-warning");
        td.onclick = () => selectCell(cell.cell_id, false);
      }
      tr.appendChild(td);
    }
    table.appendChild(tr);
  }

  renderOverlay();
})();
"""
        + "</script>"
    )
    st.components.v1.html(html, height=height, scrolling=False)
    return SimpleNamespace(edit_cell_text=None)
