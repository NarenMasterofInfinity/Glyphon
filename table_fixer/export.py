from __future__ import annotations

import json
from io import BytesIO
from typing import Any

import pandas as pd

from .models import PipelineSnapshot


def safe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: json.dumps(value, ensure_ascii=True) if isinstance(value, (dict, list, tuple)) else value
            for key, value in record.items()
        }
        for record in records
    ]


def table_records(snapshot: PipelineSnapshot, table_id: str) -> list[dict[str, Any]]:
    table = snapshot.tables[table_id]
    records = []
    for row_id in table.row_ids:
        record = {
            "table_id": table_id,
            "page_number": table.page_number,
            "row_id": row_id,
        }
        for column_id in table.column_ids:
            cell = snapshot.cells.get(f"{row_id}::{column_id}")
            record[table.column_names[column_id]] = cell.text if cell else ""
        records.append(record)
    return records


def rows_by_role(snapshot: PipelineSnapshot, role: str) -> list[dict[str, Any]]:
    records = []
    for row in snapshot.rows.values():
        if row.role != role:
            continue
        records.append(
            {
                "row_id": row.row_id,
                "page_number": row.page_number,
                "table_id": row.table_id,
                "role": role,
                "values": {
                    cell.column_id: cell.text
                    for cell in snapshot.row_cells(row.row_id)
                    if cell.text.strip()
                },
            }
        )
    return records


def audit_export(snapshot: PipelineSnapshot, phase_snapshots: dict[str, PipelineSnapshot]) -> dict[str, Any]:
    return {
        "schema_version": "glyphon-table-fixer-v1",
        "final_snapshot": snapshot.to_dict(),
        "metadata_rows": rows_by_role(snapshot, "metadata"),
        "header_rows": rows_by_role(snapshot, "header"),
        "logical_tables": {
            table_id: table_records(snapshot, table_id)
            for table_id in snapshot.tables
        },
        "phase_snapshots": {
            phase: phase_snapshot.to_dict()
            for phase, phase_snapshot in phase_snapshots.items()
        },
    }


def json_export(snapshot: PipelineSnapshot, phase_snapshots: dict[str, PipelineSnapshot]) -> bytes:
    return json.dumps(audit_export(snapshot, phase_snapshots), indent=2).encode("utf-8")


def excel_export(snapshot: PipelineSnapshot) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        for index, table_id in enumerate(snapshot.tables, start=1):
            pd.DataFrame(table_records(snapshot, table_id)).to_excel(
                writer,
                sheet_name=f"table_{index}"[:31],
                index=False,
            )
        pd.DataFrame(safe_records(rows_by_role(snapshot, "metadata"))).to_excel(writer, sheet_name="metadata_rows", index=False)
        pd.DataFrame(safe_records(rows_by_role(snapshot, "header"))).to_excel(writer, sheet_name="header_rows", index=False)
        pd.DataFrame(safe_records([issue.__dict__ for issue in snapshot.issues.values()])).to_excel(
            writer, sheet_name="issues", index=False
        )
        pd.DataFrame(safe_records([decision.__dict__ for decision in snapshot.decisions])).to_excel(
            writer, sheet_name="decisions", index=False
        )
        pd.DataFrame(safe_records([usage.__dict__ for usage in snapshot.prompt_usage])).to_excel(
            writer, sheet_name="token_usage", index=False
        )
        pd.DataFrame(safe_records(snapshot.prompt_audits)).to_excel(
            writer, sheet_name="prompt_audits", index=False
        )
    return buffer.getvalue()
