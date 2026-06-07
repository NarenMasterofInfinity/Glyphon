from __future__ import annotations

from dataclasses import asdict
import json
from hashlib import sha256
from pathlib import Path
import sys
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fitz
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from parser import parse_pdf_pages as parse_ocr_pdf_pages
from text_parser import parse_pdf_pages as parse_text_pdf_pages
from table_fixer.context import ensure_profiles
from table_fixer.models import (
    CellState,
    DecisionRecord,
    IssueState,
    LogicalTable,
    PipelineSnapshot,
    PromptUsage,
    RowProfile,
    RowState,
)
from table_fixer.ollama_client import OllamaClientError, OllamaLLMClient
from table_fixer.pipeline import PhaseRun, TableFixerPipeline, snapshot_from_parser
from table_fixer.workspace import (
    WORKSPACES_ROOT,
    create_workspace,
    persist_event,
    persist_llm_trace,
    persist_parser_results,
    persist_source_pdf,
    persist_phase_run,
    persist_snapshot,
    write_json,
)


PHASES = ["reconciliation", "metadata", "headers", "columns", "warnings"]
EXTRACTION_MODES = ["text", "auto", "ocr"]
DEFAULT_MODEL = "gemma3:4b"
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_CONTEXT_TOKEN_BUDGET = 3500
DEFAULT_AUTO_APPLY_THRESHOLD = 0.80
DEFAULT_STRUCTURAL_THRESHOLD = 0.95
API_STATE_FILE = "api_state.json"
API_OPENAPI_PATH = Path(__file__).resolve().parent / "openapi.yaml"

app = FastAPI(
    title="Glyphon Table Fixer API",
    description=(
        "FastAPI wrapper around the existing Glyphon table-fixer workflow. "
        "Supports fresh extraction, continuation from workspace state, preview review, and auto-apply runs."
    ),
)


class ExecutionSettings(BaseModel):
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    context_token_budget: int = Field(default=DEFAULT_CONTEXT_TOKEN_BUDGET, ge=500, le=16000)
    auto_apply_threshold: float = Field(default=DEFAULT_AUTO_APPLY_THRESHOLD, ge=0.5, le=1.0)
    structural_auto_apply_threshold: float = Field(default=DEFAULT_STRUCTURAL_THRESHOLD, ge=0.8, le=1.0)


class ExecutePhasesRequest(ExecutionSettings):
    phases: list[str] = Field(default_factory=list)
    execution_mode: Literal["preview", "auto_apply"] = "auto_apply"


class ReviewDecisionRequest(BaseModel):
    decision: Literal["accept", "reject"]


class ManualActionRequest(BaseModel):
    base_phase: str = "source"
    actions: list[dict[str, Any]] = Field(min_length=1)
    note: str | None = None


class WorkspaceResponse(BaseModel):
    workspace_id: str
    status: dict[str, Any]
    result: dict[str, Any]
    pending_review: dict[str, Any] | None = None
    phase_runs: list[dict[str, Any]] = Field(default_factory=list)


def api_state_path(workspace: Path) -> Path:
    return workspace / API_STATE_FILE


def new_api_state() -> dict[str, Any]:
    return {
        "schema_version": "glyphon-table-fixer-api-state-v1",
        "accepted_labels": {"source": "source"},
        "pending_preview": None,
        "invalidated_phases": [],
        "manual_sequence": 0,
        "manual_history": [],
        "redo_stack": [],
    }


def load_api_state(workspace: Path) -> dict[str, Any]:
    path = api_state_path(workspace)
    if not path.exists():
        state = new_api_state()
        write_json(path, state)
        return state
    state = json.loads(path.read_text(encoding="utf-8"))
    defaults = new_api_state()
    changed = False
    for key, value in defaults.items():
        if key not in state:
            state[key] = value
            changed = True
    if changed:
        write_json(path, state)
    return state


def save_api_state(workspace: Path, state: dict[str, Any]) -> None:
    write_json(api_state_path(workspace), state)


def resolve_workspace(workspace_id: str) -> Path:
    workspace = WORKSPACES_ROOT / workspace_id
    if not workspace.exists() or not workspace.is_dir():
        raise HTTPException(status_code=404, detail=f"Workspace not found: {workspace_id}")
    return workspace


def snapshot_label_path(workspace: Path, label: str) -> Path:
    return workspace / label / "snapshot.json"


def load_snapshot(workspace: Path, label: str) -> PipelineSnapshot:
    path = snapshot_label_path(workspace, label)
    if not path.exists():
        raise HTTPException(
            status_code=409,
            detail=f"Snapshot label '{label}' is missing in workspace '{workspace.name}'.",
        )
    return snapshot_from_dict(json.loads(path.read_text(encoding="utf-8")))


def snapshot_from_dict(payload: dict[str, Any]) -> PipelineSnapshot:
    tables = {
        table_id: LogicalTable(
            table_id=value["table_id"],
            page_number=value["page_number"],
            source_table_index=value["source_table_index"],
            column_ids=list(value["column_ids"]),
            column_names=dict(value["column_names"]),
            row_ids=list(value["row_ids"]),
            metadata_row_ids=list(value.get("metadata_row_ids", [])),
            header_row_ids=list(value.get("header_row_ids", [])),
            ancestor_table_ids=list(value.get("ancestor_table_ids", [])),
        )
        for table_id, value in payload["tables"].items()
    }
    rows = {
        row_id: RowState(
            row_id=value["row_id"],
            page_number=value["page_number"],
            source_row_number=value["source_row_number"],
            table_id=value["table_id"],
            role=value.get("role", "data"),
            profile=RowProfile(**value["profile"]) if value.get("profile") else None,
            ancestor_row_ids=list(value.get("ancestor_row_ids", [])),
        )
        for row_id, value in payload["rows"].items()
    }
    cells = {
        cell_id: CellState(
            cell_id=value["cell_id"],
            row_id=value["row_id"],
            column_id=value["column_id"],
            text=value.get("text", ""),
            bbox=tuple(value["bbox"]) if value.get("bbox") else None,
            source_item_indexes=list(value.get("source_item_indexes", [])),
            warning_ids=list(value.get("warning_ids", [])),
            assignment_score=value.get("assignment_score", 1.0),
            alternatives=list(value.get("alternatives", [])),
            ancestor_cell_ids=list(value.get("ancestor_cell_ids", [])),
        )
        for cell_id, value in payload["cells"].items()
    }
    issues = {
        issue_id: IssueState(
            issue_id=value["issue_id"],
            source_issue_id=value["source_issue_id"],
            issue_type=value["issue_type"],
            severity=value["severity"],
            table_id=value.get("table_id"),
            affected_cell_ids=list(value.get("affected_cell_ids", [])),
            status=value["status"],
            explanation=value.get("explanation", ""),
            suggested_action=value.get("suggested_action", ""),
            evidence=dict(value.get("evidence", {})),
            ancestor_issue_ids=list(value.get("ancestor_issue_ids", [])),
        )
        for issue_id, value in payload["issues"].items()
    }
    decisions = [
        DecisionRecord(
            decision_id=value["decision_id"],
            phase=value["phase"],
            target_id=value["target_id"],
            action=value["action"],
            confidence=value["confidence"],
            reason=value.get("reason", ""),
            payload=dict(value.get("payload", {})),
            valid=value.get("valid", False),
            validation_errors=list(value.get("validation_errors", [])),
            applied=value.get("applied", False),
            affected_ids=list(value.get("affected_ids", [])),
            prompt_id=value.get("prompt_id"),
        )
        for value in payload.get("decisions", [])
    ]
    prompt_usage = [
        PromptUsage(
            prompt_id=value["prompt_id"],
            phase=value["phase"],
            purpose=value["purpose"],
            model=value["model"],
            system_tokens=value["system_tokens"],
            context_tokens=value["context_tokens"],
            input_tokens=value["input_tokens"],
            output_tokens=value["output_tokens"],
            duration_ms=value["duration_ms"],
            native_prompt_tokens=value.get("native_prompt_tokens"),
            native_output_tokens=value.get("native_output_tokens"),
            repair_parent_prompt_id=value.get("repair_parent_prompt_id"),
        )
        for value in payload.get("prompt_usage", [])
    ]
    page_dimensions = {
        int(page_number): tuple(dimensions)
        for page_number, dimensions in payload.get("page_dimensions", {}).items()
    }
    return PipelineSnapshot(
        snapshot_id=payload["snapshot_id"],
        phase=payload["phase"],
        tables=tables,
        rows=rows,
        cells=cells,
        issues=issues,
        page_dimensions=page_dimensions,
        decisions=decisions,
        prompt_usage=prompt_usage,
        prompt_audits=list(payload.get("prompt_audits", [])),
        row_lineage={key: list(value) for key, value in payload.get("row_lineage", {}).items()},
        column_lineage={key: list(value) for key, value in payload.get("column_lineage", {}).items()},
        cell_lineage={key: list(value) for key, value in payload.get("cell_lineage", {}).items()},
        invalidated_phases=list(payload.get("invalidated_phases", [])),
    )


def parse_optional_list(raw_value: str | None, *, item_type: type[int] | type[str]) -> list[int] | list[str]:
    if not raw_value:
        return []
    stripped = raw_value.strip()
    if not stripped:
        return []
    try:
        candidate = json.loads(stripped)
        if isinstance(candidate, list):
            return [item_type(item) for item in candidate]
    except json.JSONDecodeError:
        pass
    return [item_type(part.strip()) for part in stripped.split(",") if part.strip()]


def page_has_native_text(pdf_bytes: bytes, page_number: int) -> bool:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        if page_number < 1 or page_number > len(doc):
            return False
        page = doc[page_number - 1]
        return any(str(word[4]).strip() for word in page.get_text("words", sort=True))
    finally:
        doc.close()


def parse_pdf_pages_by_mode(
    pdf_bytes: bytes,
    *,
    extraction_mode: str,
    page_numbers: list[int] | None,
):
    if extraction_mode not in EXTRACTION_MODES:
        raise HTTPException(status_code=400, detail=f"Unknown extraction mode: {extraction_mode}")
    if extraction_mode == "text":
        return parse_text_pdf_pages(pdf_bytes, page_numbers=page_numbers)
    if extraction_mode == "ocr":
        return parse_ocr_pdf_pages(pdf_bytes, page_numbers=page_numbers)

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        target_pages = page_numbers or list(range(1, len(doc) + 1))
    finally:
        doc.close()

    text_pages = [page for page in target_pages if page_has_native_text(pdf_bytes, page)]
    ocr_pages = [page for page in target_pages if page not in text_pages]
    results = []
    if text_pages:
        results.extend(parse_text_pdf_pages(pdf_bytes, page_numbers=text_pages))
    if ocr_pages:
        results.extend(parse_ocr_pdf_pages(pdf_bytes, page_numbers=ocr_pages))
    return sorted(results, key=lambda page: page.page_number)


def validate_execute_request(request: ExecutePhasesRequest) -> None:
    if not request.phases:
        raise HTTPException(status_code=400, detail="At least one phase must be requested.")
    invalid = [phase for phase in request.phases if phase not in PHASES]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Unknown phases: {invalid}")
    if len(set(request.phases)) != len(request.phases):
        raise HTTPException(status_code=400, detail="Duplicate phases are not allowed.")
    if request.execution_mode == "preview" and len(request.phases) != 1:
        raise HTTPException(status_code=400, detail="Preview mode supports exactly one phase at a time.")


def validate_phase_or_source(phase: str) -> None:
    if phase != "source" and phase not in PHASES:
        raise HTTPException(status_code=404, detail=f"Unknown phase: {phase}")


def phase_and_later(phase: str) -> list[str]:
    if phase == "source":
        return list(PHASES)
    return PHASES[PHASES.index(phase):]


def phases_after(phase: str) -> list[str]:
    if phase == "source":
        return list(PHASES)
    return PHASES[PHASES.index(phase) + 1:]


def accepted_phases_from_state(state: dict[str, Any]) -> list[str]:
    return [phase for phase in PHASES if phase in state["accepted_labels"]]


def latest_accepted_phase(state: dict[str, Any]) -> str:
    accepted = accepted_phases_from_state(state)
    return accepted[-1] if accepted else "source"


def available_next_phases(state: dict[str, Any]) -> list[str]:
    if state.get("pending_preview"):
        return []
    latest = latest_accepted_phase(state)
    if latest == "source":
        return PHASES
    index = PHASES.index(latest) + 1
    return PHASES[index:]


def validate_requested_phases(state: dict[str, Any], phases: list[str]) -> None:
    if state.get("pending_preview"):
        pending_phase = state["pending_preview"]["phase"]
        raise HTTPException(
            status_code=409,
            detail=(
                f"Workspace has a pending preview for phase '{pending_phase}'. "
                "Accept or reject it before running more phases."
            ),
        )
    expected = available_next_phases(state)
    if not expected:
        raise HTTPException(status_code=409, detail="No further phases are available for this workspace.")
    if phases != expected[: len(phases)]:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Requested phases must be the next contiguous phase sequence from the accepted state. "
                f"Expected prefix: {expected}"
            ),
        )


def invalidate_from_phase(state: dict[str, Any], phase: str) -> None:
    for later in phase_and_later(phase):
        state["accepted_labels"].pop(later, None)
        pending = state.get("pending_preview")
        if pending and pending.get("phase") == later:
            state["pending_preview"] = None


def invalidate_after_base_phase(state: dict[str, Any], phase: str) -> list[str]:
    invalidated = phases_after(phase)
    for later in invalidated:
        state["accepted_labels"].pop(later, None)
        pending = state.get("pending_preview")
        if pending and pending.get("phase") == later:
            state["pending_preview"] = None
    state["invalidated_phases"] = invalidated
    return invalidated


def mark_phase_fresh(state: dict[str, Any], phase: str) -> None:
    state["invalidated_phases"] = [
        item for item in state.get("invalidated_phases", []) if item != phase
    ]


def build_pipeline(settings: ExecutionSettings) -> TableFixerPipeline:
    client = OllamaLLMClient(model=settings.model, base_url=settings.base_url)
    return TableFixerPipeline(
        client,
        auto_apply_threshold=settings.auto_apply_threshold,
        structural_auto_apply_threshold=settings.structural_auto_apply_threshold,
        context_token_budget=settings.context_token_budget,
    )


def run_phase(
    workspace: Path,
    state: dict[str, Any],
    phase: str,
    execution_mode: Literal["preview", "auto_apply"],
    settings: ExecutionSettings,
) -> tuple[PhaseRun, dict[str, Any]]:
    base_label = state["accepted_labels"]["source"]
    previous = "source"
    for candidate in PHASES:
        if candidate == phase:
            break
        if candidate in state["accepted_labels"]:
            previous = candidate
            base_label = state["accepted_labels"][candidate]
    source = load_snapshot(workspace, base_label)
    invalidate_from_phase(state, phase)
    pipeline = build_pipeline(settings)
    method = getattr(pipeline, f"run_{phase}")
    attempt_start = len(pipeline.client.prompt_attempts)
    try:
        run = method(source, auto_apply=execution_mode == "auto_apply")
    except OllamaClientError as exc:
        append_llm_trace(workspace, pipeline.client.prompt_attempts[attempt_start:])
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    append_llm_trace(workspace, pipeline.client.prompt_attempts[attempt_start:])
    if execution_mode == "preview":
        persist_phase_run(workspace, phase, run, "preview")
        state["pending_preview"] = {
            "phase": phase,
            "label": f"{phase}_preview",
            "source_phase": previous,
            "source_label": base_label,
        }
    else:
        persist_phase_run(workspace, phase, run, "auto_applied")
        state["accepted_labels"][phase] = f"{phase}_auto_applied"
        state["pending_preview"] = None
        mark_phase_fresh(state, phase)
    return run, summarize_phase_run(run, execution_mode)


def append_llm_trace(workspace: Path, attempts: list[dict[str, Any]]) -> None:
    trace_path = workspace / "llm_trace.json"
    trace = []
    if trace_path.exists():
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
    start_sequence = len(trace) + 1
    for offset, attempt in enumerate(attempts, start=0):
        trace.append({"sequence": start_sequence + offset, **attempt})
    persist_llm_trace(workspace, trace)


def summarize_phase_run(run: PhaseRun, execution_mode: str) -> dict[str, Any]:
    return {
        "phase": run.phase,
        "execution_mode": execution_mode,
        "source_snapshot_id": run.source_snapshot_id,
        "proposed_snapshot_id": run.proposed_snapshot.snapshot_id,
        "decision_count": len(run.decisions),
        "valid_decision_count": sum(record.valid for record in run.decisions),
        "applied_decision_count": sum(record.applied for record in run.decisions),
        "warning_count_after_phase": len(run.proposed_snapshot.active_warnings()),
    }


def issue_union_bbox(snapshot: PipelineSnapshot, issue: IssueState) -> list[float] | None:
    boxes = [
        snapshot.cells[cell_id].bbox
        for cell_id in issue.affected_cell_ids
        if cell_id in snapshot.cells and snapshot.cells[cell_id].bbox
    ]
    if not boxes:
        return None
    x0 = min(box[0] for box in boxes)
    y0 = min(box[1] for box in boxes)
    x1 = max(box[2] for box in boxes)
    y1 = max(box[3] for box in boxes)
    return [x0, y0, x1, y1]


def serialize_issue(snapshot: PipelineSnapshot, issue: IssueState) -> dict[str, Any]:
    affected_cells = []
    for cell_id in issue.affected_cell_ids:
        if cell_id not in snapshot.cells:
            continue
        cell = snapshot.cells[cell_id]
        row = snapshot.rows[cell.row_id]
        table = snapshot.tables[row.table_id]
        affected_cells.append(
            {
                "cell_id": cell.cell_id,
                "row_id": cell.row_id,
                "column_id": cell.column_id,
                "header": table.column_names.get(cell.column_id, cell.column_id),
                "text": cell.text,
                "bbox": list(cell.bbox) if cell.bbox else None,
            }
        )
    return {
        "issue_id": issue.issue_id,
        "source_issue_id": issue.source_issue_id,
        "issue_type": issue.issue_type,
        "severity": issue.severity,
        "status": issue.status,
        "table_id": issue.table_id,
        "explanation": issue.explanation,
        "suggested_action": issue.suggested_action,
        "evidence": issue.evidence,
        "ancestor_issue_ids": issue.ancestor_issue_ids,
        "warning_box": issue_union_bbox(snapshot, issue),
        "affected_cells": affected_cells,
    }


def serialize_row(snapshot: PipelineSnapshot, table: LogicalTable, row_id: str) -> dict[str, Any]:
    row = snapshot.rows[row_id]
    cells = []
    for column_id in table.column_ids:
        cell = snapshot.cells.get(f"{row_id}::{column_id}")
        if not cell:
            continue
        cells.append(
            {
                "cell_id": cell.cell_id,
                "column_id": column_id,
                "header": table.column_names.get(column_id, column_id),
                "text": cell.text,
                "bbox": list(cell.bbox) if cell.bbox else None,
                "source_item_indexes": cell.source_item_indexes,
                "warning_ids": cell.warning_ids,
                "warning_details": [
                    serialize_issue(snapshot, snapshot.issues[issue_id])
                    for issue_id in cell.warning_ids
                    if issue_id in snapshot.issues
                ],
                "assignment_score": cell.assignment_score,
                "alternatives": cell.alternatives,
                "ancestor_cell_ids": cell.ancestor_cell_ids,
            }
        )
    return {
        "row_id": row.row_id,
        "role": row.role,
        "page_number": row.page_number,
        "source_row_number": row.source_row_number,
        "ancestor_row_ids": row.ancestor_row_ids,
        "cells": cells,
    }


def summarize_snapshot(snapshot: PipelineSnapshot) -> dict[str, Any]:
    tables = []
    for table in snapshot.tables.values():
        tables.append(
            {
                "table_id": table.table_id,
                "page_number": table.page_number,
                "source_table_index": table.source_table_index,
                "ancestor_table_ids": table.ancestor_table_ids,
                "columns": [
                    {
                        "column_id": column_id,
                        "name": table.column_names.get(column_id, column_id),
                        "lineage": snapshot.column_lineage.get(column_id, []),
                    }
                    for column_id in table.column_ids
                ],
                "metadata_rows": [serialize_row(snapshot, table, row_id) for row_id in table.metadata_row_ids],
                "header_rows": [serialize_row(snapshot, table, row_id) for row_id in table.header_row_ids],
                "data_rows": [serialize_row(snapshot, table, row_id) for row_id in table.row_ids],
            }
        )
    issues = [serialize_issue(snapshot, issue) for issue in snapshot.issues.values()]
    return {
        "snapshot_id": snapshot.snapshot_id,
        "phase": snapshot.phase,
        "page_dimensions": {
            str(page_number): list(dimensions)
            for page_number, dimensions in snapshot.page_dimensions.items()
        },
        "tables": tables,
        "warnings": [issue for issue in issues if issue["severity"] == "warning"],
        "issues": issues,
        "decisions": [asdict(decision) for decision in snapshot.decisions],
        "prompt_usage": [asdict(usage) for usage in snapshot.prompt_usage],
        "invalidated_phases": snapshot.invalidated_phases,
    }


def find_cell(snapshot: PipelineSnapshot, cell_id: str) -> CellState:
    if cell_id not in snapshot.cells:
        raise HTTPException(status_code=400, detail=f"Cell not found: {cell_id}")
    return snapshot.cells[cell_id]


def apply_manual_action(snapshot: PipelineSnapshot, action: dict[str, Any]) -> dict[str, Any]:
    action_type = action.get("type") or action.get("action")
    if not isinstance(action_type, str):
        raise HTTPException(status_code=400, detail="Manual action requires a string type.")

    if action_type == "edit_cell_text":
        cell = find_cell(snapshot, str(action.get("cell_id", "")))
        before = cell.text
        cell.text = str(action.get("text", ""))
        return {"type": action_type, "cell_id": cell.cell_id, "before": before, "after": cell.text}

    if action_type == "set_warning_status":
        issue_id = str(action.get("issue_id", ""))
        if issue_id not in snapshot.issues:
            raise HTTPException(status_code=400, detail=f"Issue not found: {issue_id}")
        status = str(action.get("status", "active"))
        if status not in {"active", "review", "resolved", "dismissed"}:
            raise HTTPException(status_code=400, detail=f"Unsupported warning status: {status}")
        issue = snapshot.issues[issue_id]
        before = issue.status
        issue.status = status
        return {"type": action_type, "issue_id": issue_id, "before": before, "after": status}

    if action_type == "move_cell":
        source = find_cell(snapshot, str(action.get("source_cell_id", "")))
        target = find_cell(snapshot, str(action.get("target_cell_id", "")))
        source_before = source.text
        target_before = target.text
        mode = str(action.get("mode", "append"))
        if mode == "replace":
            target.text = source.text
        else:
            target.text = " ".join(value for value in [target.text.strip(), source.text.strip()] if value)
        source.text = ""
        return {
            "type": action_type,
            "source_cell_id": source.cell_id,
            "target_cell_id": target.cell_id,
            "source_before": source_before,
            "target_before": target_before,
            "source_after": source.text,
            "target_after": target.text,
        }

    if action_type == "split_cell_text":
        source = find_cell(snapshot, str(action.get("cell_id", "")))
        assignments = action.get("assignments", [])
        if not isinstance(assignments, list) or not assignments:
            raise HTTPException(status_code=400, detail="split_cell_text requires assignments.")
        changes = []
        for assignment in assignments:
            if not isinstance(assignment, dict):
                raise HTTPException(status_code=400, detail="Each split assignment must be an object.")
            target = find_cell(snapshot, str(assignment.get("cell_id", "")))
            before = target.text
            target.text = str(assignment.get("text", ""))
            changes.append({"cell_id": target.cell_id, "before": before, "after": target.text})
        return {"type": action_type, "source_cell_id": source.cell_id, "changes": changes}

    if action_type == "merge_adjacent_cells":
        cell_ids = [str(value) for value in action.get("cell_ids", [])]
        if len(cell_ids) < 2:
            raise HTTPException(status_code=400, detail="merge_adjacent_cells requires at least two cell_ids.")
        cells = [find_cell(snapshot, cell_id) for cell_id in cell_ids]
        target = find_cell(snapshot, str(action.get("target_cell_id") or cell_ids[0]))
        before = {cell.cell_id: cell.text for cell in cells}
        target.text = " ".join(cell.text.strip() for cell in cells if cell.text.strip())
        for cell in cells:
            if cell.cell_id != target.cell_id:
                cell.text = ""
        return {
            "type": action_type,
            "target_cell_id": target.cell_id,
            "before": before,
            "after": {cell.cell_id: cell.text for cell in cells},
        }

    if action_type == "set_decision_enabled":
        decision_id = str(action.get("decision_id", ""))
        enabled = bool(action.get("enabled", True))
        for record in snapshot.decisions:
            if record.decision_id == decision_id:
                before = record.payload.get("manual_enabled", True)
                record.payload["manual_enabled"] = enabled
                record.payload["manual_note"] = action.get("note", "")
                return {"type": action_type, "decision_id": decision_id, "before": before, "after": enabled}
        raise HTTPException(status_code=400, detail=f"Decision not found: {decision_id}")

    raise HTTPException(status_code=400, detail=f"Unsupported manual action type: {action_type}")


def apply_manual_actions(
    workspace: Path,
    state: dict[str, Any],
    request: ManualActionRequest,
) -> dict[str, Any]:
    validate_phase_or_source(request.base_phase)
    if request.base_phase not in state["accepted_labels"]:
        raise HTTPException(status_code=409, detail=f"Phase is not accepted: {request.base_phase}")

    before_labels = dict(state["accepted_labels"])
    source = load_snapshot(workspace, state["accepted_labels"][request.base_phase])
    manual = source.clone(
        f"snapshot_{request.base_phase}_manual_{state.get('manual_sequence', 0) + 1}",
        request.base_phase,
    )
    summaries = [apply_manual_action(manual, action) for action in request.actions]
    ensure_profiles(manual)
    state["manual_sequence"] = int(state.get("manual_sequence", 0)) + 1
    label = f"{request.base_phase}_manual_{state['manual_sequence']}"
    persist_snapshot(workspace, label, manual)
    state["accepted_labels"][request.base_phase] = label
    invalidated = invalidate_after_base_phase(state, request.base_phase)
    state["redo_stack"] = []
    history_entry = {
        "sequence": state["manual_sequence"],
        "base_phase": request.base_phase,
        "label": label,
        "note": request.note,
        "actions": request.actions,
        "summaries": summaries,
        "invalidated_phases": invalidated,
        "before_accepted_labels": before_labels,
        "after_accepted_labels": dict(state["accepted_labels"]),
    }
    state.setdefault("manual_history", []).append(history_entry)
    return history_entry


def restore_manual_history_state(state: dict[str, Any], entry: dict[str, Any], labels_key: str) -> None:
    state["accepted_labels"] = dict(entry[labels_key])
    state["pending_preview"] = None
    base_phase = entry["base_phase"]
    invalidated = phases_after(base_phase)
    for phase in invalidated:
        state["accepted_labels"].pop(phase, None)
    state["invalidated_phases"] = invalidated


def workspace_status(workspace: Path, state: dict[str, Any]) -> dict[str, Any]:
    manifest = json.loads((workspace / "manifest.json").read_text(encoding="utf-8"))
    accepted_phases = accepted_phases_from_state(state)
    return {
        "workspace_id": workspace.name,
        "created_at": manifest["created_at"],
        "source_file": manifest["source_file"],
        "file_key": manifest["file_key"],
        "page_numbers": manifest["page_numbers"],
        "extraction_mode": manifest.get("extraction_mode", "ocr"),
        "accepted_phases": accepted_phases,
        "latest_accepted_phase": latest_accepted_phase(state),
        "next_available_phases": available_next_phases(state),
        "pending_review_phase": state.get("pending_preview", {}).get("phase") if state.get("pending_preview") else None,
        "invalidated_phases": state.get("invalidated_phases", []),
        "has_manual_changes": bool(state.get("manual_history")),
        "manual_history_count": len(state.get("manual_history", [])),
        "redo_count": len(state.get("redo_stack", [])),
    }


def accepted_snapshot_from_state(workspace: Path, state: dict[str, Any]) -> PipelineSnapshot:
    latest = latest_accepted_phase(state)
    label = state["accepted_labels"].get(latest, "source")
    return load_snapshot(workspace, label)


def build_workspace_response(
    workspace: Path,
    state: dict[str, Any],
    *,
    phase_runs: list[dict[str, Any]] | None = None,
) -> WorkspaceResponse:
    accepted_snapshot = accepted_snapshot_from_state(workspace, state)
    pending_review = None
    pending = state.get("pending_preview")
    if pending:
        pending_snapshot = load_snapshot(workspace, pending["label"])
        pending_review = {
            "phase": pending["phase"],
            "source_phase": pending["source_phase"],
            "source_label": pending["source_label"],
            "proposed_result": summarize_snapshot(pending_snapshot),
        }
    return WorkspaceResponse(
        workspace_id=workspace.name,
        status=workspace_status(workspace, state),
        result=summarize_snapshot(accepted_snapshot),
        pending_review=pending_review,
        phase_runs=phase_runs or [],
    )


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "glyphon-table-fixer", "port": 8770}


@app.get("/openapi.yaml", include_in_schema=False)
def openapi_yaml() -> FileResponse:
    return FileResponse(API_OPENAPI_PATH, media_type="application/yaml", filename="openapi.yaml")


@app.post("/table-fixer/workspaces", response_model=WorkspaceResponse)
async def create_table_fixer_workspace(
    file: UploadFile = File(...),
    page_numbers: str | None = Form(default=None),
    phases: str | None = Form(default=None),
    execution_mode: Literal["preview", "auto_apply"] = Form(default="auto_apply"),
    extraction_mode: Literal["text", "auto", "ocr"] = Form(default="ocr"),
    model: str = Form(default=DEFAULT_MODEL),
    base_url: str = Form(default=DEFAULT_BASE_URL),
    context_token_budget: int = Form(default=DEFAULT_CONTEXT_TOKEN_BUDGET),
    auto_apply_threshold: float = Form(default=DEFAULT_AUTO_APPLY_THRESHOLD),
    structural_auto_apply_threshold: float = Form(default=DEFAULT_STRUCTURAL_THRESHOLD),
) -> WorkspaceResponse:
    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    requested_pages = parse_optional_list(page_numbers, item_type=int)
    requested_phases = parse_optional_list(phases, item_type=str)
    settings = ExecutePhasesRequest(
        phases=requested_phases,
        execution_mode=execution_mode,
        model=model,
        base_url=base_url,
        context_token_budget=context_token_budget,
        auto_apply_threshold=auto_apply_threshold,
        structural_auto_apply_threshold=structural_auto_apply_threshold,
    ) if requested_phases else None

    file_key = sha256(pdf_bytes).hexdigest()
    page_results = parse_pdf_pages_by_mode(
        pdf_bytes,
        extraction_mode=extraction_mode,
        page_numbers=requested_pages or None,
    )
    page_scope = requested_pages or [page.page_number for page in page_results]
    workspace = create_workspace(file.filename or "uploaded.pdf", file_key, page_scope, extraction_mode)
    persist_source_pdf(workspace, pdf_bytes)
    persist_parser_results(workspace, page_results)
    source = snapshot_from_parser(page_results)
    persist_snapshot(workspace, "source", source)
    state = new_api_state()
    save_api_state(workspace, state)

    phase_runs: list[dict[str, Any]] = []
    if settings:
        validate_execute_request(settings)
        validate_requested_phases(state, settings.phases)
        for phase in settings.phases:
            _, run_summary = run_phase(workspace, state, phase, settings.execution_mode, settings)
            phase_runs.append(run_summary)
            save_api_state(workspace, state)
            if settings.execution_mode == "preview":
                break
    return build_workspace_response(workspace, state, phase_runs=phase_runs)


@app.get("/table-fixer/workspaces/{workspace_id}", response_model=WorkspaceResponse)
def get_table_fixer_workspace(workspace_id: str) -> WorkspaceResponse:
    workspace = resolve_workspace(workspace_id)
    state = load_api_state(workspace)
    return build_workspace_response(workspace, state)


@app.get("/table-fixer/workspaces/{workspace_id}/snapshots/{phase}")
def get_table_fixer_snapshot(workspace_id: str, phase: str) -> dict[str, Any]:
    validate_phase_or_source(phase)
    workspace = resolve_workspace(workspace_id)
    state = load_api_state(workspace)
    label = state["accepted_labels"].get(phase)
    if not label:
        raise HTTPException(status_code=409, detail=f"Phase is not accepted: {phase}")
    return summarize_snapshot(load_snapshot(workspace, label))


@app.get("/table-fixer/workspaces/{workspace_id}/pages/{page_number}/image")
def get_table_fixer_page_image(
    workspace_id: str,
    page_number: int,
    zoom: float = 2.0,
) -> Response:
    workspace = resolve_workspace(workspace_id)
    pdf_path = workspace / "source.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Workspace source PDF is not available.")
    if zoom <= 0 or zoom > 6:
        raise HTTPException(status_code=400, detail="zoom must be > 0 and <= 6.")
    doc = fitz.open(str(pdf_path))
    try:
        if page_number < 1 or page_number > len(doc):
            raise HTTPException(status_code=404, detail=f"Page not found: {page_number}")
        page = doc[page_number - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        return Response(content=pix.tobytes("png"), media_type="image/png")
    finally:
        doc.close()


@app.post("/table-fixer/workspaces/{workspace_id}/execute", response_model=WorkspaceResponse)
def execute_table_fixer_phases(workspace_id: str, request: ExecutePhasesRequest) -> WorkspaceResponse:
    workspace = resolve_workspace(workspace_id)
    state = load_api_state(workspace)
    validate_execute_request(request)
    validate_requested_phases(state, request.phases)
    phase_runs = []
    for phase in request.phases:
        _, run_summary = run_phase(workspace, state, phase, request.execution_mode, request)
        phase_runs.append(run_summary)
        save_api_state(workspace, state)
        if request.execution_mode == "preview":
            break
    return build_workspace_response(workspace, state, phase_runs=phase_runs)


@app.post("/table-fixer/workspaces/{workspace_id}/reviews/{phase}", response_model=WorkspaceResponse)
def review_table_fixer_phase(workspace_id: str, phase: str, request: ReviewDecisionRequest) -> WorkspaceResponse:
    if phase not in PHASES:
        raise HTTPException(status_code=404, detail=f"Unknown phase: {phase}")
    workspace = resolve_workspace(workspace_id)
    state = load_api_state(workspace)
    pending = state.get("pending_preview")
    if not pending or pending.get("phase") != phase:
        raise HTTPException(status_code=409, detail=f"No pending preview exists for phase '{phase}'.")
    if request.decision == "accept":
        state["accepted_labels"][phase] = pending["label"]
        state["pending_preview"] = None
        mark_phase_fresh(state, phase)
        save_api_state(workspace, state)
    else:
        persist_event(workspace, phase, "rejected")
        state["pending_preview"] = None
        save_api_state(workspace, state)
    return build_workspace_response(workspace, state)


@app.post("/table-fixer/workspaces/{workspace_id}/manual-actions", response_model=WorkspaceResponse)
def apply_table_fixer_manual_actions(
    workspace_id: str,
    request: ManualActionRequest,
) -> WorkspaceResponse:
    workspace = resolve_workspace(workspace_id)
    state = load_api_state(workspace)
    apply_manual_actions(workspace, state, request)
    save_api_state(workspace, state)
    return build_workspace_response(workspace, state)


@app.post("/table-fixer/workspaces/{workspace_id}/history/undo", response_model=WorkspaceResponse)
def undo_table_fixer_manual_action(workspace_id: str) -> WorkspaceResponse:
    workspace = resolve_workspace(workspace_id)
    state = load_api_state(workspace)
    history = state.setdefault("manual_history", [])
    if not history:
        raise HTTPException(status_code=409, detail="No manual actions are available to undo.")
    entry = history.pop()
    state.setdefault("redo_stack", []).append(entry)
    restore_manual_history_state(state, entry, "before_accepted_labels")
    save_api_state(workspace, state)
    return build_workspace_response(workspace, state)


@app.post("/table-fixer/workspaces/{workspace_id}/history/redo", response_model=WorkspaceResponse)
def redo_table_fixer_manual_action(workspace_id: str) -> WorkspaceResponse:
    workspace = resolve_workspace(workspace_id)
    state = load_api_state(workspace)
    redo_stack = state.setdefault("redo_stack", [])
    if not redo_stack:
        raise HTTPException(status_code=409, detail="No manual actions are available to redo.")
    entry = redo_stack.pop()
    state.setdefault("manual_history", []).append(entry)
    restore_manual_history_state(state, entry, "after_accepted_labels")
    save_api_state(workspace, state)
    return build_workspace_response(workspace, state)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("table_fixer.api:app", host="0.0.0.0", port=8770, reload=False)
