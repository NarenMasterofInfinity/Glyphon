from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RowProfile:
    fill_ratio: float
    text_ratio: float
    numeric_ratio: float
    average_text_length: float
    occupancy_pattern: str
    following_similarity: float
    bbox_width_ratio: float
    alignment_consistency: float
    header_candidate: bool


@dataclass
class CellState:
    cell_id: str
    row_id: str
    column_id: str
    text: str
    bbox: tuple[float, float, float, float] | None
    source_item_indexes: list[int] = field(default_factory=list)
    warning_ids: list[str] = field(default_factory=list)
    assignment_score: float = 1.0
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    ancestor_cell_ids: list[str] = field(default_factory=list)


@dataclass
class RowState:
    row_id: str
    page_number: int
    source_row_number: int
    table_id: str
    role: str = "data"
    profile: RowProfile | None = None
    ancestor_row_ids: list[str] = field(default_factory=list)


@dataclass
class LogicalTable:
    table_id: str
    page_number: int
    source_table_index: int
    column_ids: list[str]
    column_names: dict[str, str]
    row_ids: list[str]
    metadata_row_ids: list[str] = field(default_factory=list)
    header_row_ids: list[str] = field(default_factory=list)
    ancestor_table_ids: list[str] = field(default_factory=list)


@dataclass
class IssueState:
    issue_id: str
    source_issue_id: str
    issue_type: str
    severity: str
    table_id: str | None
    affected_cell_ids: list[str]
    status: str
    explanation: str
    suggested_action: str
    evidence: dict[str, Any] = field(default_factory=dict)
    ancestor_issue_ids: list[str] = field(default_factory=list)


@dataclass
class DecisionRecord:
    decision_id: str
    phase: str
    target_id: str
    action: str
    confidence: float
    reason: str
    payload: dict[str, Any]
    valid: bool
    validation_errors: list[str] = field(default_factory=list)
    applied: bool = False
    affected_ids: list[str] = field(default_factory=list)
    prompt_id: str | None = None


@dataclass
class PromptUsage:
    prompt_id: str
    phase: str
    purpose: str
    model: str
    system_tokens: int
    context_tokens: int
    input_tokens: int
    output_tokens: int
    duration_ms: float
    native_prompt_tokens: int | None = None
    native_output_tokens: int | None = None
    repair_parent_prompt_id: str | None = None


@dataclass
class PipelineSnapshot:
    snapshot_id: str
    phase: str
    tables: dict[str, LogicalTable]
    rows: dict[str, RowState]
    cells: dict[str, CellState]
    issues: dict[str, IssueState]
    page_dimensions: dict[int, tuple[float, float]] = field(default_factory=dict)
    decisions: list[DecisionRecord] = field(default_factory=list)
    prompt_usage: list[PromptUsage] = field(default_factory=list)
    prompt_audits: list[dict[str, Any]] = field(default_factory=list)
    row_lineage: dict[str, list[str]] = field(default_factory=dict)
    column_lineage: dict[str, list[str]] = field(default_factory=dict)
    cell_lineage: dict[str, list[str]] = field(default_factory=dict)
    invalidated_phases: list[str] = field(default_factory=list)

    def clone(self, snapshot_id: str, phase: str) -> "PipelineSnapshot":
        cloned = deepcopy(self)
        cloned.snapshot_id = snapshot_id
        cloned.phase = phase
        cloned.invalidated_phases = []
        return cloned

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def table_cells(self, table_id: str) -> list[CellState]:
        table = self.tables[table_id]
        row_ids = set(table.row_ids + table.metadata_row_ids + table.header_row_ids)
        return [cell for cell in self.cells.values() if cell.row_id in row_ids]

    def row_cells(self, row_id: str) -> list[CellState]:
        return sorted(
            [cell for cell in self.cells.values() if cell.row_id == row_id],
            key=lambda cell: cell.column_id,
        )

    def active_warnings(self) -> list[IssueState]:
        return [
            issue
            for issue in self.issues.values()
            if issue.status == "active" and issue.severity == "warning"
        ]
