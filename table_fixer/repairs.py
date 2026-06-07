from __future__ import annotations

import re
from collections import Counter
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


def header_parts(value: str) -> list[str]:
    return [
        part.lower()
        for part in re.findall(r"[A-Z]+(?=[A-Z][a-z]|\b)|[A-Z]?[a-z]+|\d+", value)
    ]


def validate_header_split(source_header: str, new_headers: list[str]) -> list[str]:
    source_parts = header_parts(source_header)
    proposed_parts = [header_parts(header) for header in new_headers]
    if not 2 <= len(new_headers) <= 4:
        return ["Header-based split requires 2-4 new headers."]
    if any(not parts for parts in proposed_parts):
        return ["Every proposed header must contain meaningful header text."]
    flattened = [part for parts in proposed_parts for part in parts]
    if flattened != source_parts:
        return [
            "Column split rejected: proposed headers must exactly reconstruct the current header in order. "
            "Values cannot justify new fields that are absent from the header."
        ]
    return []


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


def infer_whitespace_split(
    source_header: str,
    values: list[str],
    *,
    require_header_parts: bool = False,
) -> tuple[str, list[str]] | None:
    parts = [value.split() for value in values]
    counts = {len(value_parts) for value_parts in parts}
    if len(counts) != 1:
        return None
    count = next(iter(counts), 0)
    if not 2 <= count <= 4:
        return None
    inferred_headers = re.findall(r"[A-Z]+(?=[A-Z][a-z]|\b)|[A-Z]?[a-z]+|\d+", source_header)
    if len(inferred_headers) != count:
        if require_header_parts:
            return None
        inferred_headers = [f"Field {index}" for index in range(1, count + 1)]
    group_names = [
        re.sub(r"[^a-z0-9]+", "_", header.lower()).strip("_") or f"field_{index}"
        for index, header in enumerate(inferred_headers, start=1)
    ]
    pattern = r"\s+".join(f"(?P<{name}>\\S+)" for name in group_names)
    return pattern, inferred_headers


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
    trigger_issue = result.issues.get(record.target_id)
    if trigger_issue:
        trigger_issue.status = "resolved"
        for cell_id in trigger_issue.affected_cell_ids:
            if cell_id in result.cells:
                result.cells[cell_id].warning_ids = [
                    issue_id for issue_id in result.cells[cell_id].warning_ids
                    if issue_id != trigger_issue.issue_id
                ]

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


def validate_placeholder_column_action(
    snapshot: PipelineSnapshot,
    table_id: str,
    column_id: str,
    action: str,
    target_headers: list[str],
    regex: str,
    new_header: str,
) -> tuple[list[str], dict[str, list[str]], list[str]]:
    errors: list[str] = []
    placements: dict[str, list[str]] = {}
    table = snapshot.tables.get(table_id)
    if not table or column_id not in table.column_ids:
        return ["Placeholder column does not exist."], placements, []

    header_to_ids: dict[str, list[str]] = {}
    for candidate_id in table.column_ids:
        header_to_ids.setdefault(table.column_names.get(candidate_id, candidate_id), []).append(candidate_id)
    target_ids: list[str] = []
    for header in target_headers:
        matches = header_to_ids.get(header, [])
        if len(matches) != 1:
            errors.append(f"Target header must identify exactly one column: {header}")
        else:
            target_ids.append(matches[0])
    if len(set(target_ids)) != len(target_ids):
        errors.append("Target headers must be distinct.")
    if column_id in target_ids:
        errors.append("Placeholder column cannot target itself.")

    source_rows = [
        row_id for row_id in table.row_ids
        if (cell := snapshot.cells.get(f"{row_id}::{column_id}")) and cell.text.strip()
    ]
    non_data_values = [
        snapshot.cells[cell_id].text
        for row_id in table.metadata_row_ids + table.header_row_ids
        if (cell_id := f"{row_id}::{column_id}") in snapshot.cells
        and snapshot.cells[cell_id].text.strip()
    ]
    if action in {"move", "split"} and non_data_values:
        errors.append("Cannot remove a placeholder column with non-empty metadata or header cells.")
    if action == "move":
        if len(target_headers) != 1 or regex or new_header:
            errors.append("Move requires one target header and no regex or new header.")
        for row_id in source_rows:
            if len(target_ids) != 1:
                break
            destination = snapshot.cells.get(f"{row_id}::{target_ids[0]}")
            if not destination or destination.text.strip():
                errors.append(f"Move destination is occupied or missing on row {snapshot.rows[row_id].source_row_number}.")
                continue
            placements[row_id] = [snapshot.cells[f"{row_id}::{column_id}"].text]
    elif action == "split":
        if not 2 <= len(target_headers) <= 3 or new_header:
            errors.append("Split requires 2-3 target headers and no new header.")
        compiled = None
        if len(regex) > 300 or any(part in regex for part in UNSAFE_REGEX_PARTS):
            errors.append("Regex uses an unsafe or excessively long construct.")
        else:
            try:
                compiled = re.compile(regex)
            except re.error as exc:
                errors.append(f"Invalid regex: {exc}")
        if compiled:
            group_names = list(compiled.groupindex)
            if compiled.groups != len(group_names) or len(group_names) != len(target_headers):
                errors.append("Split regex must have one named group per target header and no unnamed groups.")
            for row_id in source_rows:
                source = snapshot.cells[f"{row_id}::{column_id}"].text
                match = compiled.fullmatch(source)
                if not match:
                    errors.append(f"Split regex does not full-match row {snapshot.rows[row_id].source_row_number}.")
                    continue
                values = [str(match.group(name) or "").strip() for name in group_names]
                if any(not value for value in values):
                    errors.append(f"Split produced an empty value on row {snapshot.rows[row_id].source_row_number}.")
                if Counter(re.findall(r"\S+", source)) != Counter(
                    token for value in values for token in re.findall(r"\S+", value)
                ):
                    errors.append(f"Split must preserve every token on row {snapshot.rows[row_id].source_row_number}.")
                if len(target_ids) == len(target_headers):
                    for target_id in target_ids:
                        destination = snapshot.cells.get(f"{row_id}::{target_id}")
                        if not destination or destination.text.strip():
                            errors.append(
                                f"Split destination is occupied or missing on row "
                                f"{snapshot.rows[row_id].source_row_number}."
                            )
                    placements[row_id] = values
    elif action == "rename":
        if target_headers or regex or not new_header.strip():
            errors.append("Rename requires only a non-empty new header.")
        if re.fullmatch(r"col_\d+", new_header.strip(), re.IGNORECASE):
            errors.append("New header cannot be another placeholder.")
        if new_header.strip() in header_to_ids:
            errors.append("New header must be unique in the table.")
    elif action == "unresolved":
        if target_headers or regex or new_header:
            errors.append("Unresolved must not include targets, regex, or a new header.")
    else:
        errors.append(f"Unsupported placeholder action: {action}")
    return sorted(set(errors)), placements, target_ids


def apply_placeholder_column_action(
    snapshot: PipelineSnapshot,
    record: DecisionRecord,
    placements: dict[str, list[str]],
    target_ids: list[str],
) -> PipelineSnapshot:
    result = snapshot.clone(f"snapshot_columns_{uuid4().hex[:8]}", "columns")
    result.decisions.append(record)
    table = result.tables[record.payload["table_id"]]
    source_id = record.payload["column_id"]
    action = record.action
    if action == "rename":
        table.column_names[source_id] = record.payload["new_header"].strip()
        record.applied = True
        record.affected_ids = [source_id]
        return result
    if action == "unresolved":
        return result

    affected: list[str] = []
    all_rows = table.metadata_row_ids + table.header_row_ids + table.row_ids
    for row_id in table.row_ids:
        old_cell_id = f"{row_id}::{source_id}"
        old_cell = result.cells.get(old_cell_id)
        if not old_cell or row_id not in placements:
            continue
        descendants = []
        for target_id, value in zip(target_ids, placements[row_id]):
            target_cell = result.cells[f"{row_id}::{target_id}"]
            target_cell.text = value
            target_cell.source_item_indexes.extend(
                index for index in old_cell.source_item_indexes
                if index not in target_cell.source_item_indexes
            )
            target_cell.ancestor_cell_ids = list(dict.fromkeys(
                [old_cell_id] + old_cell.ancestor_cell_ids + target_cell.ancestor_cell_ids
            ))
            descendants.append(target_cell.cell_id)
            affected.append(target_cell.cell_id)
        result.cell_lineage[old_cell_id] = descendants
        for issue in result.issues.values():
            if old_cell_id in issue.affected_cell_ids:
                issue.affected_cell_ids = list(dict.fromkeys(
                    descendants if issue.affected_cell_ids == [old_cell_id]
                    else [item for item in issue.affected_cell_ids if item != old_cell_id] + descendants
                ))
        for descendant in descendants:
            result.cells[descendant].warning_ids = list(dict.fromkeys(
                result.cells[descendant].warning_ids + old_cell.warning_ids
            ))
    for row_id in all_rows:
        result.cells.pop(f"{row_id}::{source_id}", None)
    table.column_ids.remove(source_id)
    table.column_names.pop(source_id, None)
    result.column_lineage[source_id] = target_ids
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
    if action == "no_issue":
        return errors
    if action not in {"move_cell", "merge_adjacent_cells", "split_cell_text", "redistribute_cell", "redistribute_row"}:
        return ["Correction decision does not contain an allowed executable action."]
    if action == "redistribute_row":
        source_ids = payload.get("source_cell_ids")
        assignments = payload.get("assignments")
        if not isinstance(source_ids, list) or not source_ids:
            return ["redistribute_row requires source cells."]
        if not isinstance(assignments, list) or not assignments:
            return ["redistribute_row requires final assignments."]
        if any(source_id not in snapshot.cells for source_id in source_ids):
            return ["Every row-repair source must be an existing cell."]
        source_rows = {snapshot.cells[source_id].row_id for source_id in source_ids}
        if len(source_rows) != 1:
            errors.append("Row repair sources must belong to one row.")
        source_id_set = set(source_ids)
        target_ids: list[str] = []
        assigned_text: list[str] = []
        for assignment in assignments:
            target_id = assignment.get("target_cell_id") if isinstance(assignment, dict) else None
            text = assignment.get("text") if isinstance(assignment, dict) else None
            if target_id not in source_id_set:
                errors.append("Every row-repair target must be inside the supplied repair zone.")
                continue
            if not isinstance(text, str) or not text.strip():
                errors.append("Every row-repair assignment requires non-empty text.")
                continue
            target_ids.append(target_id)
            assigned_text.append(text)
        if len(target_ids) != len(set(target_ids)):
            errors.append("Row repair cannot assign multiple values to the same target.")
        tokens = lambda value: Counter(re.findall(r"\w+|[^\w\s]", value.lower()))
        source_text = " ".join(snapshot.cells[source_id].text for source_id in source_ids)
        if tokens(source_text) != tokens(" ".join(assigned_text)):
            errors.append("Row repair must preserve every source token exactly once.")
        current = {
            source_id: snapshot.cells[source_id].text.strip()
            for source_id in source_ids
            if snapshot.cells[source_id].text.strip()
        }
        proposed = {
            target_id: text.strip()
            for target_id, text in zip(target_ids, assigned_text)
            if text.strip()
        }
        if current == proposed:
            errors.append("Row repair must change at least one listed column.")
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
    if action == "split_cell_text":
        new_text = payload.get("new_text")
        if not isinstance(new_text, str):
            errors.append("split_cell_text requires a replacement new_text and cannot create columns.")
        elif new_text == snapshot.cells[source_id].text:
            errors.append("split_cell_text replacement must differ from the current cell text.")
    if action == "redistribute_cell":
        assignments = payload.get("assignments")
        if not isinstance(assignments, list) or not assignments:
            return errors + ["redistribute_cell requires at least one assignment."]
        source = snapshot.cells[source_id]
        source_table = snapshot.rows[source.row_id].table_id
        target_ids: list[str] = []
        assigned_text: list[str] = []
        for assignment in assignments:
            target_id = assignment.get("target_cell_id") if isinstance(assignment, dict) else None
            text = assignment.get("text") if isinstance(assignment, dict) else None
            if target_id not in snapshot.cells:
                errors.append("Every redistribution target must be an existing cell.")
                continue
            target = snapshot.cells[target_id]
            if snapshot.rows[target.row_id].table_id != source_table or target.row_id != source.row_id:
                errors.append("Redistribution targets must be in the source row and logical table.")
            if target_id != source_id and target.text.strip():
                errors.append("Redistribution targets must be empty unless the target is the source cell.")
            if not isinstance(text, str) or not text.strip():
                errors.append("Every redistribution assignment requires non-empty text.")
                continue
            target_ids.append(target_id)
            assigned_text.append(text)
        if len(target_ids) != len(set(target_ids)):
            errors.append("Redistribution cannot assign multiple values to the same target cell.")
        tokens = lambda value: Counter(re.findall(r"\w+|[^\w\s]", value.lower()))
        if tokens(source.text) != tokens(" ".join(assigned_text)):
            errors.append("Redistribution must preserve every source token exactly once.")
        if len(assignments) == 1 and target_ids == [source_id] and assigned_text == [source.text]:
            errors.append("Redistribution must change the cell placement or contents.")
    return errors


def apply_warning_decisions(snapshot: PipelineSnapshot, records: list[DecisionRecord]) -> PipelineSnapshot:
    result = snapshot.clone(f"snapshot_warnings_{uuid4().hex[:8]}", "warnings")
    for record in records:
        result.decisions.append(record)
        issue_ids = record.payload.get("issue_ids", [record.target_id])
        issues = [result.issues[issue_id] for issue_id in issue_ids if issue_id in result.issues]
        if not issues or not record.valid:
            continue
        if record.action == "no_issue":
            for issue in issues:
                issue.status = "dismissed"
        elif record.action == "move_cell":
            source = result.cells[record.payload["source_cell_id"]]
            target = result.cells[record.payload["target_cell_id"]]
            target.text = f"{target.text} {source.text}".strip()
            source.text = ""
            for issue in issues:
                issue.status = "resolved"
        elif record.action == "merge_adjacent_cells":
            source = result.cells[record.payload["source_cell_id"]]
            target = result.cells[record.payload["target_cell_id"]]
            target.text = f"{target.text} {source.text}".strip()
            target.ancestor_cell_ids.append(source.cell_id)
            source.text = ""
            for issue in issues:
                issue.status = "resolved"
        elif record.action == "split_cell_text":
            result.cells[record.payload["source_cell_id"]].text = record.payload["new_text"]
            for issue in issues:
                issue.status = "resolved"
        elif record.action == "redistribute_cell":
            source = result.cells[record.payload["source_cell_id"]]
            source.text = ""
            for assignment in record.payload["assignments"]:
                result.cells[assignment["target_cell_id"]].text = assignment["text"].strip()
            for issue in issues:
                issue.status = "resolved"
        elif record.action == "redistribute_row":
            for source_id in record.payload["source_cell_ids"]:
                result.cells[source_id].text = ""
            for assignment in record.payload["assignments"]:
                result.cells[assignment["target_cell_id"]].text = assignment["text"].strip()
            for issue in issues:
                issue.status = "resolved"
        record.applied = True
        record.affected_ids = list(dict.fromkeys(
            cell_id
            for issue in issues
            for cell_id in issue.affected_cell_ids
        ))
    return result
