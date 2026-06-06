from __future__ import annotations

import re
from dataclasses import replace
from typing import Any
from uuid import uuid4

from .models import CellState, DecisionRecord, IssueState, LogicalTable, PipelineSnapshot


def decision(
    phase: str,
    target_id: str,
    action: str,
    confidence: float,
    reason: str,
    payload: dict[str, Any],
    valid: bool,
    errors: list[str],
    prompt_id: str | None,
) -> DecisionRecord:
    return DecisionRecord(
        decision_id=f"decision_{uuid4().hex[:12]}",
        phase=phase,
        target_id=target_id,
        action=action,
        confidence=float(confidence),
        reason=reason,
        payload=payload,
        valid=valid,
        validation_errors=errors,
        prompt_id=prompt_id,
    )


def apply_metadata(snapshot: PipelineSnapshot, decisions: list[DecisionRecord]) -> PipelineSnapshot:
    result = snapshot.clone(f"snapshot_metadata_{uuid4().hex[:8]}", "metadata")
    for record in decisions:
        result.decisions.append(record)
        if not record.valid or record.action != "remove_metadata":
            continue
        table = result.tables[record.target_id]
        requested = record.payload.get("metadata_row_ids", [])
        if requested and len(requested) >= len(table.row_ids):
            record.valid = False
            record.validation_errors.append("Metadata removal must leave at least one data row.")
            continue
        prefix: list[str] = []
        for row_id in table.row_ids:
            if row_id not in requested:
                break
            prefix.append(row_id)
        if prefix != requested:
            record.valid = False
            record.validation_errors.append("Metadata rows must be an exact contiguous top prefix.")
            continue
        table.metadata_row_ids.extend(prefix)
        table.row_ids = table.row_ids[len(prefix):]
        for row_id in prefix:
            result.rows[row_id].role = "metadata"
        record.applied = True
        record.affected_ids = prefix
    _rebase_issues_to_data_rows(result)
    return result


def apply_table_reconciliation(snapshot: PipelineSnapshot, decisions: list[DecisionRecord]) -> PipelineSnapshot:
    result = snapshot.clone(f"snapshot_reconciliation_{uuid4().hex[:8]}", "reconciliation")
    aliases: dict[str, str] = {}

    def current_table_id(table_id: str | None) -> str | None:
        while table_id in aliases:
            table_id = aliases[table_id]
        return table_id

    for record in decisions:
        result.decisions.append(record)
        if not record.valid or record.action != "merge":
            continue
        original_left_id = record.payload.get("left_table_id")
        original_right_id = record.payload.get("right_table_id")
        left_id = current_table_id(original_left_id)
        right_id = current_table_id(original_right_id)
        if left_id == right_id:
            record.applied = True
            record.affected_ids = [left_id] if left_id else []
            continue
        left = result.tables.get(left_id)
        right = result.tables.get(right_id)
        if not left or not right:
            record.valid = False
            record.validation_errors.append("Both merge tables must exist in the current snapshot.")
            continue
        if left.page_number != right.page_number or len(left.column_ids) != len(right.column_ids):
            record.valid = False
            record.validation_errors.append("Only same-page tables with equal column counts can merge.")
            continue

        column_map = dict(zip(right.column_ids, left.column_ids))
        cell_map: dict[str, str] = {}
        for row_id in right.metadata_row_ids + right.header_row_ids + right.row_ids:
            result.rows[row_id].table_id = left_id
            for old_column_id, new_column_id in column_map.items():
                old_cell_id = f"{row_id}::{old_column_id}"
                old_cell = result.cells.pop(old_cell_id, None)
                if not old_cell:
                    continue
                new_cell_id = f"{row_id}::{new_column_id}"
                old_cell.cell_id = new_cell_id
                old_cell.column_id = new_column_id
                old_cell.ancestor_cell_ids = [old_cell_id] + old_cell.ancestor_cell_ids
                result.cells[new_cell_id] = old_cell
                result.cell_lineage[old_cell_id] = [new_cell_id]
                cell_map[old_cell_id] = new_cell_id

        left.metadata_row_ids.extend(row_id for row_id in right.metadata_row_ids if row_id not in left.metadata_row_ids)
        left.header_row_ids.extend(row_id for row_id in right.header_row_ids if row_id not in left.header_row_ids)
        left.row_ids.extend(row_id for row_id in right.row_ids if row_id not in left.row_ids)
        left.ancestor_table_ids = list(dict.fromkeys(
            left.ancestor_table_ids + [right_id] + right.ancestor_table_ids
        ))
        del result.tables[right_id]
        aliases[right_id] = left_id
        if original_right_id != right_id:
            aliases[original_right_id] = left_id
        for issue in result.issues.values():
            issue.affected_cell_ids = [cell_map.get(cell_id, cell_id) for cell_id in issue.affected_cell_ids]
            if issue.table_id == right_id:
                issue.table_id = left_id
                issue.evidence = {**issue.evidence, "reconciled_from_table": right_id}
        record.applied = True
        record.affected_ids = [left_id, right_id] + list(cell_map.values())
    return result


def _header_name(snapshot: PipelineSnapshot, row_ids: list[str], column_id: str, fallback: str) -> str:
    parts = []
    for row_id in row_ids:
        cell = snapshot.cells.get(f"{row_id}::{column_id}")
        if cell and cell.text.strip() and cell.text.strip() not in parts:
            parts.append(cell.text.strip())
    return " ".join(parts).strip() or fallback


def apply_headers(snapshot: PipelineSnapshot, decisions: list[DecisionRecord]) -> PipelineSnapshot:
    result = snapshot.clone(f"snapshot_headers_{uuid4().hex[:8]}", "headers")
    for record in decisions:
        result.decisions.append(record)
        if not record.valid or record.action != "accept_header":
            continue
        table = result.tables.get(record.target_id)
        if not table:
            record.valid = False
            record.validation_errors.append("Target table does not exist.")
            continue
        groups = record.payload.get("header_groups", [])
        ordered_groups = sorted(
            groups,
            key=lambda group: table.row_ids.index(group[0])
            if group and group[0] in table.row_ids else 10**9,
        )
        for group in ordered_groups:
            if not group or any(row_id not in table.row_ids for row_id in group):
                record.valid = False
                record.validation_errors.append("Header group contains unknown rows.")
                break
            positions = [table.row_ids.index(row_id) for row_id in group]
            if positions != list(range(min(positions), max(positions) + 1)):
                record.valid = False
                record.validation_errors.append("Header continuation rows must be contiguous.")
                break
        if not record.valid:
            continue

        for group_index, group in enumerate(ordered_groups, start=1):
            positions = [table.row_ids.index(row_id) for row_id in group]
            start = min(positions)
            next_start = (
                table.row_ids.index(ordered_groups[group_index][0])
                if group_index < len(ordered_groups)
                else len(table.row_ids)
            )
            if not table.row_ids[start + len(group):next_start]:
                record.valid = False
                record.validation_errors.append("Header split must leave at least one data row.")
                break
        if not record.valid:
            continue

        consumed: set[str] = set()
        new_tables: list[LogicalTable] = []
        for group_index, group in enumerate(ordered_groups, start=1):
            positions = [table.row_ids.index(row_id) for row_id in group]
            start = min(positions)
            next_start = (
                table.row_ids.index(ordered_groups[group_index][0])
                if group_index < len(ordered_groups)
                else len(table.row_ids)
            )
            data_rows = [
                row_id for row_id in table.row_ids[start + len(group):next_start]
                if row_id not in consumed
            ]
            new_table_id = f"{table.table_id}_h{group_index}"
            new_table = LogicalTable(
                table_id=new_table_id,
                page_number=table.page_number,
                source_table_index=table.source_table_index,
                column_ids=list(table.column_ids),
                column_names={
                    column_id: _header_name(result, group, column_id, table.column_names[column_id])
                    for column_id in table.column_ids
                },
                row_ids=data_rows,
                metadata_row_ids=list(table.metadata_row_ids) if group_index == 1 else [],
                header_row_ids=list(group),
                ancestor_table_ids=[table.table_id] + table.ancestor_table_ids,
            )
            for row_id in group:
                result.rows[row_id].role = "header"
                consumed.add(row_id)
            for row_id in data_rows:
                result.rows[row_id].table_id = new_table_id
                consumed.add(row_id)
            new_tables.append(new_table)
        before_first = table.row_ids[:table.row_ids.index(ordered_groups[0][0])] if ordered_groups else table.row_ids
        if before_first:
            table.row_ids = before_first
        else:
            del result.tables[table.table_id]
        for new_table in new_tables:
            result.tables[new_table.table_id] = new_table
        _rebase_issue_tables(result)
        _rebase_issues_to_data_rows(result)
        record.applied = True
        record.affected_ids = [new_table.table_id for new_table in new_tables]
    return result


def _rebase_issue_tables(snapshot: PipelineSnapshot) -> None:
    for issue in list(snapshot.issues.values()):
        cells_by_table: dict[str, list[str]] = {}
        for cell_id in issue.affected_cell_ids:
            if cell_id not in snapshot.cells:
                continue
            row_id = snapshot.cells[cell_id].row_id
            table_id = snapshot.rows[row_id].table_id
            if table_id in snapshot.tables:
                cells_by_table.setdefault(table_id, []).append(cell_id)
        if len(cells_by_table) == 1:
            issue.table_id = next(iter(cells_by_table))
            continue
        if len(cells_by_table) > 1:
            issue.status = "superseded"
            for index, (table_id, cell_ids) in enumerate(sorted(cells_by_table.items()), start=1):
                child_id = f"{issue.issue_id}__table{index}"
                snapshot.issues[child_id] = IssueState(
                    issue_id=child_id,
                    source_issue_id=issue.source_issue_id,
                    issue_type=issue.issue_type,
                    severity=issue.severity,
                    table_id=table_id,
                    affected_cell_ids=cell_ids,
                    status="active",
                    explanation=f"{issue.explanation} Rebased after logical table split.",
                    suggested_action=issue.suggested_action,
                    evidence={**issue.evidence, "rebased_from_table": issue.table_id},
                    ancestor_issue_ids=[issue.issue_id] + issue.ancestor_issue_ids,
                )
                for cell_id in cell_ids:
                    cell = snapshot.cells[cell_id]
                    cell.warning_ids = [value for value in cell.warning_ids if value != issue.issue_id]
                    cell.warning_ids.append(child_id)
            continue
        if issue.table_id not in snapshot.tables:
            issue.table_id = None
            issue.status = "review"


def _rebase_issues_to_data_rows(snapshot: PipelineSnapshot) -> None:
    data_rows = {
        row_id
        for table in snapshot.tables.values()
        for row_id in table.row_ids
    }
    for issue in snapshot.issues.values():
        if issue.status != "active" or not issue.affected_cell_ids:
            continue
        issue.affected_cell_ids = [
            cell_id
            for cell_id in issue.affected_cell_ids
            if cell_id in snapshot.cells and snapshot.cells[cell_id].row_id in data_rows
        ]
        distinct_rows = {
            snapshot.cells[cell_id].row_id
            for cell_id in issue.affected_cell_ids
            if cell_id in snapshot.cells
        }
        minimum_support = 2 if issue.issue_type == "possible_merged_column" else 1
        if len(distinct_rows) < minimum_support:
            issue.status = "superseded"


UNSAFE_REGEX_PARTS = ("(?=", "(?!", "(?<=", "(?<!", "\\1", "\\2", "\\3", "(?P=")


def validate_split_regex(
    source_header: str,
    pattern: str,
    new_headers: list[str],
    values: list[str],
    expected: dict[str, Any],
) -> tuple[re.Pattern[str] | None, list[str], dict[str, list[str]]]:
    errors: list[str] = []
    actual: dict[str, list[str]] = {}
    if len(pattern) > 300 or any(part in pattern for part in UNSAFE_REGEX_PARTS):
        errors.append("Regex uses an unsafe or excessively long construct.")
        return None, errors, actual
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        return None, [f"Invalid regex: {exc}"], actual
    group_names = list(compiled.groupindex)
    if not 2 <= len(group_names) <= 4 or compiled.groups != len(group_names):
        errors.append("Regex must contain 2-4 named capture groups and no unnamed groups.")
    if len(new_headers) != len(group_names):
        errors.append("New header count must match named capture group count.")
    source_words = set(re.findall(r"[a-z0-9]+", source_header.lower()))
    if source_words and any(
        not set(re.findall(r"[a-z0-9]+", header.lower())).issubset(source_words)
        for header in new_headers
    ):
        errors.append("New headers must use words from the existing merged header.")
    for value in values:
        match = compiled.fullmatch(value)
        if not match:
            errors.append(f"Regex does not full-match value: {value}")
            continue
        groups = [str(match.group(name) or "").strip() for name in group_names]
        if any(not group for group in groups):
            errors.append(f"Regex produced an empty group for value: {value}")
        actual[value] = groups
    for source, expected_groups in expected.items():
        if source in actual and actual[source] != expected_groups:
            errors.append(f"Expected-output mismatch for value: {source}")
    return compiled, sorted(set(errors)), actual


def apply_column_split(
    snapshot: PipelineSnapshot,
    record: DecisionRecord,
    compiled: re.Pattern[str],
    actual: dict[str, list[str]],
) -> PipelineSnapshot:
    result = snapshot.clone(f"snapshot_columns_{uuid4().hex[:8]}", "columns")
    result.decisions.append(record)
    table = result.tables[record.payload["table_id"]]
    old_column_id = record.payload["column_id"]
    group_names = list(compiled.groupindex)
    new_headers = record.payload["new_headers"]
    old_position = table.column_ids.index(old_column_id)
    new_column_ids = [f"{old_column_id}__{name}" for name in group_names]
    table.column_ids[old_position:old_position + 1] = new_column_ids
    table.column_names.pop(old_column_id, None)
    table.column_names.update(dict(zip(new_column_ids, new_headers)))
    result.column_lineage[old_column_id] = new_column_ids
    affected: list[str] = []

    for row_id in table.row_ids + table.header_row_ids:
        old_cell_id = f"{row_id}::{old_column_id}"
        old_cell = result.cells.pop(old_cell_id, None)
        if not old_cell:
            continue
        if row_id in table.header_row_ids:
            values = new_headers if row_id == table.header_row_ids[0] else [""] * len(new_column_ids)
        else:
            values = actual.get(old_cell.text, [""] * len(new_column_ids))
        descendants = []
        for new_column_id, value in zip(new_column_ids, values):
            new_cell_id = f"{row_id}::{new_column_id}"
            result.cells[new_cell_id] = CellState(
                cell_id=new_cell_id,
                row_id=row_id,
                column_id=new_column_id,
                text=value,
                bbox=old_cell.bbox,
                source_item_indexes=list(old_cell.source_item_indexes),
                warning_ids=[],
                assignment_score=old_cell.assignment_score,
                alternatives=list(old_cell.alternatives),
                ancestor_cell_ids=[old_cell_id] + old_cell.ancestor_cell_ids,
            )
            descendants.append(new_cell_id)
            affected.append(new_cell_id)
        result.cell_lineage[old_cell_id] = descendants
        _rebase_cell_issues(result, old_cell, descendants)

    record.applied = True
    record.affected_ids = affected
    return result


def _rebase_cell_issues(snapshot: PipelineSnapshot, old_cell: CellState, descendants: list[str]) -> None:
    for issue_id in old_cell.warning_ids:
        issue = snapshot.issues.get(issue_id)
        if not issue:
            continue
        issue.status = "superseded"
        for index, descendant in enumerate(descendants, start=1):
            child_id = f"{issue.issue_id}__split{index}"
            snapshot.issues[child_id] = IssueState(
                issue_id=child_id,
                source_issue_id=issue.source_issue_id,
                issue_type=issue.issue_type,
                severity=issue.severity,
                table_id=issue.table_id,
                affected_cell_ids=[descendant],
                status="active",
                explanation=f"{issue.explanation} Rebased after column split.",
                suggested_action=issue.suggested_action,
                evidence={**issue.evidence, "rebased_from_cell": old_cell.cell_id},
                ancestor_issue_ids=[issue.issue_id] + issue.ancestor_issue_ids,
            )
            snapshot.cells[descendant].warning_ids.append(child_id)
        if len(old_cell.source_item_indexes) > 1:
            review_id = f"review_{old_cell.cell_id.replace('::', '_')}"
            snapshot.issues[review_id] = IssueState(
                issue_id=review_id,
                source_issue_id=review_id,
                issue_type="shared_source_after_split",
                severity="warning",
                table_id=issue.table_id,
                affected_cell_ids=list(descendants),
                status="active",
                explanation="Split descendants retain shared source geometry because ownership is ambiguous.",
                suggested_action="Review descendant source ownership.",
                evidence={"source_item_indexes": old_cell.source_item_indexes},
            )
            for descendant in descendants:
                snapshot.cells[descendant].warning_ids.append(review_id)


def validate_warning_action(snapshot: PipelineSnapshot, record: DecisionRecord) -> list[str]:
    errors: list[str] = []
    action = record.action
    payload = record.payload
    if action in {"no_issue", "mark_for_review"}:
        return errors
    source_id = payload.get("source_cell_id")
    target_id = payload.get("target_cell_id")
    if source_id not in snapshot.cells:
        errors.append("Source cell does not exist.")
        return errors
    if action in {"move_cell", "merge_adjacent_cells"}:
        if target_id not in snapshot.cells:
            errors.append("Target cell does not exist.")
            return errors
        if source_id == target_id:
            errors.append("Source and target cells must differ.")
            return errors
        source = snapshot.cells[source_id]
        target = snapshot.cells[target_id]
        source_table = snapshot.rows[source.row_id].table_id
        target_table = snapshot.rows[target.row_id].table_id
        if source_table != target_table:
            errors.append("Source and target must be in the same logical table.")
        if action == "move_cell" and target.text.strip():
            errors.append("Move target must be empty; use an explicit adjacent-cell merge for occupied targets.")
        if action == "merge_adjacent_cells":
            table = snapshot.tables[source_table]
            row_distance = abs(table.row_ids.index(source.row_id) - table.row_ids.index(target.row_id))
            col_distance = abs(table.column_ids.index(source.column_id) - table.column_ids.index(target.column_id))
            if row_distance + col_distance != 1:
                errors.append("Cells to merge must be adjacent.")
    if action == "split_cell_text" and not isinstance(payload.get("new_text"), str):
        errors.append("split_cell_text requires a replacement new_text and cannot create columns.")
    return errors


def apply_warning_decisions(snapshot: PipelineSnapshot, records: list[DecisionRecord]) -> PipelineSnapshot:
    result = snapshot.clone(f"snapshot_warnings_{uuid4().hex[:8]}", "warnings")
    for record in records:
        result.decisions.append(record)
        issue = result.issues.get(record.target_id)
        if not issue or not record.valid:
            continue
        if record.action == "no_issue":
            issue.status = "dismissed"
        elif record.action == "mark_for_review":
            issue.status = "review"
        elif record.action == "move_cell":
            source = result.cells[record.payload["source_cell_id"]]
            target = result.cells[record.payload["target_cell_id"]]
            target.text = f"{target.text} {source.text}".strip()
            source.text = ""
            issue.status = "resolved"
        elif record.action == "merge_adjacent_cells":
            source = result.cells[record.payload["source_cell_id"]]
            target = result.cells[record.payload["target_cell_id"]]
            target.text = f"{target.text} {source.text}".strip()
            target.ancestor_cell_ids.append(source.cell_id)
            source.text = ""
            issue.status = "resolved"
        elif record.action == "split_cell_text":
            result.cells[record.payload["source_cell_id"]].text = record.payload["new_text"]
            issue.status = "resolved"
        record.applied = True
        record.affected_ids = list(issue.affected_cell_ids)
    return result
