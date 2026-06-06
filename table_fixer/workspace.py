from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .export import rows_by_role, table_records
from .models import PipelineSnapshot
from .pipeline import PhaseRun


WORKSPACES_ROOT = Path(__file__).resolve().parent / "workspaces"


def create_workspace(file_name: str, file_key: str, page_numbers: list[int]) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    workspace = WORKSPACES_ROOT / f"{timestamp}_{file_key[:10]}"
    suffix = 1
    while workspace.exists():
        workspace = WORKSPACES_ROOT / f"{timestamp}_{file_key[:10]}_{suffix}"
        suffix += 1
    workspace.mkdir(parents=True)
    write_json(
        workspace / "manifest.json",
        {
            "workspace_version": "glyphon-table-fixer-workspace-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_file": file_name,
            "file_key": file_key,
            "page_numbers": page_numbers,
        },
    )
    return workspace


def persist_snapshot(workspace: Path, label: str, snapshot: PipelineSnapshot) -> None:
    target = workspace / label
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "snapshot.json", snapshot.to_dict())
    write_json(target / "issues.json", [issue.__dict__ for issue in snapshot.issues.values()])
    write_json(target / "decisions.json", [decision.__dict__ for decision in snapshot.decisions])
    write_json(target / "prompt_usage.json", [usage.__dict__ for usage in snapshot.prompt_usage])
    write_json(target / "prompt_audits.json", snapshot.prompt_audits)
    write_json(target / "metadata_rows.json", rows_by_role(snapshot, "metadata"))
    write_json(target / "header_rows.json", rows_by_role(snapshot, "header"))
    for table_id in snapshot.tables:
        write_csv(target / "tables" / f"{safe_name(table_id)}.csv", table_records(snapshot, table_id))


def persist_parser_results(workspace: Path, page_results: list[Any]) -> None:
    from parser import all_assignments, all_issues, build_table_records, diagnostic_sidecar

    target = workspace / "parser"
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "diagnostics.json", diagnostic_sidecar(page_results))
    _, records = build_table_records(page_results)
    write_csv(target / "tables.csv", records)
    write_csv(target / "issues.csv", all_issues(page_results))
    write_csv(target / "assignments.csv", all_assignments(page_results))


def persist_phase_run(workspace: Path, phase: str, run: PhaseRun, status: str) -> None:
    label = f"{phase}_{status}"
    persist_snapshot(workspace, label, run.proposed_snapshot)
    write_json(workspace / label / "phase_decisions.json", [decision.__dict__ for decision in run.decisions])
    write_json(workspace / label / "phase_responses.json", [response.audit_record() for response in run.responses])


def persist_event(workspace: Path, phase: str, event: str) -> None:
    target = workspace / f"{phase}_{event}"
    target.mkdir(parents=True, exist_ok=True)
    write_json(
        target / "event.json",
        {"phase": phase, "event": event, "timestamp": datetime.now(timezone.utc).isoformat()},
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for record in records for key in record))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)
