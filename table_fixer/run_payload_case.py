from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from table_fixer.context import ensure_profiles
from table_fixer.models import CellState, IssueState, LogicalTable, PipelineSnapshot, RowState
from table_fixer.ollama_client import OllamaLLMClient
from table_fixer.pipeline import TableFixerPipeline


def build_snapshot(payload: dict) -> PipelineSnapshot:
    table_id = "p1_t1"
    column_names = payload["columns"]
    column_ids = [f"{table_id}_c{index}" for index in range(1, len(column_names) + 1)]
    table = LogicalTable(
        table_id=table_id,
        page_number=1,
        source_table_index=1,
        column_ids=column_ids,
        column_names=dict(zip(column_ids, column_names)),
        row_ids=[],
    )
    rows: dict[str, RowState] = {}
    cells: dict[str, CellState] = {}
    issues: dict[str, IssueState] = {}

    for row_number, values in enumerate(payload["rows"], start=1):
        row_id = f"{table_id}_r{row_number}"
        table.row_ids.append(row_id)
        rows[row_id] = RowState(
            row_id=row_id,
            page_number=1,
            source_row_number=row_number,
            table_id=table_id,
            ancestor_row_ids=[row_id],
        )
        for column_number, value in enumerate(values, start=1):
            column_id = column_ids[column_number - 1]
            cell_id = f"{row_id}::{column_id}"
            cells[cell_id] = CellState(
                cell_id=cell_id,
                row_id=row_id,
                column_id=column_id,
                text=value,
                bbox=(column_number * 10.0, row_number * 10.0, column_number * 10.0 + 8.0, row_number * 10.0 + 5.0),
                source_item_indexes=[row_number * 100 + column_number],
                ancestor_cell_ids=[cell_id],
            )

    def add_issue(spec: dict) -> None:
        row = spec.get("row")
        column = spec.get("column")
        affected = []
        if row and column:
            affected = [f"{table_id}_r{row}::{table_id}_c{column}"]
        else:
            for row_index in spec.get("row_indexes", []):
                affected.append(f"{table_id}_r{row_index}::{table_id}_c{spec['column']}")
        issues[spec["issue_id"]] = IssueState(
            issue_id=spec["issue_id"],
            source_issue_id=spec["issue_id"],
            issue_type=spec["issue_type"],
            severity=spec["severity"],
            table_id=table_id,
            affected_cell_ids=affected,
            status="active",
            explanation=spec["explanation"],
            suggested_action=spec["suggested_action"],
            evidence=spec.get("evidence", {}),
        )
        for cell_id in affected:
            cells[cell_id].warning_ids.append(spec["issue_id"])

    for issue in payload.get("column_issues", []):
        add_issue(issue)
    for issue in payload.get("cell_issues", []):
        add_issue(issue)

    snapshot = PipelineSnapshot(
        snapshot_id="invoice_payload_source",
        phase="source",
        tables={table_id: table},
        rows=rows,
        cells=cells,
        issues=issues,
        page_dimensions={1: (1000.0, 1000.0)},
    )
    ensure_profiles(snapshot)
    return snapshot


def row_values(snapshot: PipelineSnapshot, row_number: int) -> list[str]:
    table_id = "p1_t1"
    row_id = f"{table_id}_r{row_number}"
    table = snapshot.tables[table_id]
    return [
        snapshot.cells[f"{row_id}::{column_id}"].text
        for column_id in table.column_ids
    ]


def main() -> int:
    payload_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("table_fixer/mock_data/invoice_cell_column_payload.json")
    model = sys.argv[2] if len(sys.argv) > 2 else "gemma3:4b"
    compact = "--compact" in sys.argv[3:]
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    snapshot = build_snapshot(payload)
    client = OllamaLLMClient(model=model)
    pipeline = TableFixerPipeline(client, auto_apply_threshold=0.0, structural_auto_apply_threshold=0.0)

    columns = pipeline.run_columns(snapshot, auto_apply=True)
    warnings = pipeline.run_warnings(columns.proposed_snapshot, auto_apply=True)

    report = {
        "payload_path": str(payload_path),
        "model": model,
        "column_decisions": [
            {
                "target_id": record.target_id,
                "action": record.action,
                "valid": record.valid,
                "applied": record.applied,
                "reason": record.reason,
                "validation_errors": record.validation_errors,
            }
            for record in columns.decisions
        ],
        "warning_decisions": [
            {
                "target_id": record.target_id,
                "action": record.action,
                "valid": record.valid,
                "applied": record.applied,
                "reason": record.reason,
                "validation_errors": record.validation_errors,
                "payload": record.payload,
            }
            for record in warnings.decisions
        ],
        "rows_before": {
            str(row_number): payload["rows"][row_number - 1]
            for row_number in [2, 3, 4, 8, 9, 16, 17]
        },
        "rows_after_columns": {
            str(row_number): row_values(columns.proposed_snapshot, row_number)
            for row_number in [2, 3, 4, 8, 9, 16, 17]
        },
        "rows_after_warnings": {
            str(row_number): row_values(warnings.proposed_snapshot, row_number)
            for row_number in [2, 3, 4, 8, 9, 16, 17]
        },
        "active_issues_after_warnings": [
            {
                "issue_id": issue.issue_id,
                "issue_type": issue.issue_type,
                "status": issue.status,
                "affected_cell_ids": issue.affected_cell_ids,
                "explanation": issue.explanation,
            }
            for issue in warnings.proposed_snapshot.issues.values()
            if issue.status == "active"
        ],
        "prompt_audits": [] if compact else warnings.proposed_snapshot.prompt_audits,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
