from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .api_client import PHASES


@dataclass
class PageJob:
    page_number: int
    workspace_id: str
    response: dict[str, Any]
    logs: list[dict[str, Any]] = field(default_factory=list)

    @property
    def status(self) -> dict[str, Any]:
        return self.response.get("status", {})

    @property
    def result(self) -> dict[str, Any]:
        pending = self.response.get("pending_review")
        if pending:
            return pending["proposed_result"]
        return self.response.get("result", {})

    @property
    def latest_phase(self) -> str:
        return self.status.get("latest_accepted_phase", "source")

    @property
    def next_phases(self) -> list[str]:
        return list(self.status.get("next_available_phases", []))

    @property
    def invalidated_phases(self) -> list[str]:
        return list(self.status.get("invalidated_phases", []))


def append_log(job: PageJob, phase: str, status: str, message: str, **extra: Any) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    job.logs.append(
        {
            "page": job.page_number,
            "workspace_id": job.workspace_id,
            "phase": phase,
            "status": status,
            "message": message,
            "timestamp": timestamp,
            **extra,
        }
    )


def phase_state(job: PageJob, phase: str) -> str:
    if phase in job.invalidated_phases:
        return "stale"
    if phase in job.status.get("accepted_phases", []):
        return "done"
    if job.status.get("pending_review_phase") == phase:
        return "review"
    if phase in job.next_phases:
        return "ready"
    return "pending"


def first_next_phase(jobs: list[PageJob]) -> str | None:
    indexes = []
    for job in jobs:
        if job.next_phases:
            indexes.append(PHASES.index(job.next_phases[0]))
    if not indexes:
        return None
    return PHASES[min(indexes)]


def remaining_phases(jobs: list[PageJob]) -> list[str]:
    first = first_next_phase(jobs)
    if not first:
        return []
    return PHASES[PHASES.index(first):]


def active_snapshot_phase(job: PageJob, requested_phase: str) -> str:
    if requested_phase == "source":
        return "source"
    accepted = set(job.status.get("accepted_phases", []))
    if requested_phase in accepted:
        return requested_phase
    pending = job.response.get("pending_review")
    if pending and pending.get("phase") == requested_phase:
        return requested_phase
    return job.latest_phase


def table_rows(table: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in table.get("metadata_rows", []) + table.get("header_rows", []) + table.get("data_rows", []):
        record = {
            "row_id": row["row_id"],
            "role": row.get("role", "data"),
            "page_number": row.get("page_number"),
            "source_row_number": row.get("source_row_number"),
        }
        for cell in row.get("cells", []):
            record[cell["header"]] = cell.get("text", "")
        rows.append(record)
    return rows


def merge_fragments(jobs: list[PageJob]) -> list[dict[str, Any]]:
    fragments = []
    for job in sorted(jobs, key=lambda item: item.page_number):
        result = job.response.get("result", {})
        tables = result.get("tables", [])
        indexes = [table.get("source_table_index", index + 1) for index, table in enumerate(tables)]
        first_index = min(indexes, default=1)
        last_index = max(indexes, default=1)
        for index, table in enumerate(tables, start=1):
            table_index = int(table.get("source_table_index", index))
            columns = [column["name"] for column in table.get("columns", [])]
            rows = []
            for row in table.get("data_rows", [])[:5]:
                values = {cell["header"]: cell.get("text", "") for cell in row.get("cells", [])}
                rows.append([values.get(column, "") for column in columns])
            fragments.append(
                {
                    "table_id": table["table_id"],
                    "page_number": int(table.get("page_number", job.page_number)),
                    "table_index": table_index,
                    "is_page_start": table_index == first_index,
                    "is_page_end": table_index == last_index,
                    "header": columns or ["col_1"],
                    "sample_rows": rows,
                }
            )
    return fragments
