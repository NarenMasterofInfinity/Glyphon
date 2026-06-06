from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any
from uuid import uuid4

from .context import (
    adjacent_table_pairs,
    allowed_header_continuations,
    cell_warning_context,
    column_split_context,
    ensure_profiles,
    fit_context_budget,
    header_candidates,
    header_context,
    metadata_candidate_prefix,
    metadata_context,
    placeholder_column_context,
    row_repair_context,
    table_pair_evidence,
    table_reconciliation_context,
)
from .models import (
    CellState,
    DecisionRecord,
    IssueState,
    LogicalTable,
    PipelineSnapshot,
    RowState,
)
from .ollama_client import OllamaLLMClient, StructuredResponse
from .repairs import (
    apply_column_split,
    apply_headers,
    apply_metadata,
    apply_placeholder_column_action,
    apply_table_reconciliation,
    apply_warning_decisions,
    decision,
    infer_whitespace_split,
    validate_split_regex,
    validate_placeholder_column_action,
    validate_warning_action,
)
from .schemas import (
    COLUMN_SPLIT_SCHEMA,
    HEADER_SCHEMA,
    METADATA_SCHEMA,
    PLACEHOLDER_COLUMN_SCHEMA,
    TABLE_RECONCILIATION_SCHEMA,
    ROW_REPAIR_SCHEMA,
)


SYSTEM_PROMPTS = {
    "reconciliation": (
        "You decide whether two adjacent extracted table segments are one accidentally split table. "
        "Use the compact structural evidence and boundary rows. Keep distinct titled tables separate. "
        "Return only schema JSON."
    ),
    "metadata": (
        "You classify top table rows. Return only schema JSON. Metadata must be a contiguous top prefix. "
        "Do not classify header continuations as metadata."
    ),
    "headers": (
        "You decide whether a candidate row is a table header and identify contiguous header continuations. "
        "Predecessor tail rows are context only and may reveal a preceding header fragment. Return only schema JSON."
    ),
    "columns": (
        "Work on the one supplied column only. Decide whether every sample contains the same multiple fields. "
        "Repeated whitespace-separated parts with different stable formats are separate fields, even when the "
        "current header combines their names. "
        "If yes, return a Python full-match regex with 2-4 named groups and clear field headers. "
        "If no, return no_split. Do not repeat the prompt. Return only schema JSON."
    ),
    "placeholder_column": (
        "Resolve one unnamed placeholder column using one lossless whole-column rule. A move is valid only when "
        "each complete value is one semantic field matching one empty named column. Never move a merged multi-field "
        "value wholly into one target merely because it is empty. Split when repeated values contain separate fields "
        "described by multiple empty named columns and every value follows the same pattern. Rename only when the "
        "values form a genuine distinct field. Return unresolved when none is reliable. Return only schema JSON."
    ),
    "warnings": (
        "Repair the one supplied row problem. Consider only the listed columns. Arrange available_parts under the "
        "correct headers. Nearby examples are context only and must not appear in output unless already present in "
        "available_parts. Return final_values mapping headers to final non-empty text. Use every available part "
        "exactly once. Do not calculate, invent, or request manual review. Return only schema JSON."
    ),
    "regex_repair": (
        "Repair only the supplied Python regex so actual output exactly matches expected output. "
        "Keep 2-4 named groups and return the column split schema JSON."
    ),
}


def normalize_header_groups(row_ids: list[str], groups: list[list[str]]) -> list[list[str]]:
    positions = []
    for group in groups:
        valid_positions = sorted({row_ids.index(row_id) for row_id in group if row_id in row_ids})
        if valid_positions:
            positions.append(valid_positions)
    positions.sort(key=lambda group: group[0])
    merged: list[list[int]] = []
    for group in positions:
        if merged and group[0] <= merged[-1][-1] + 1:
            merged[-1] = sorted(set(merged[-1] + group))
        else:
            merged.append(group)
    return [[row_ids[index] for index in group] for group in merged]


@dataclass
class PhaseRun:
    phase: str
    source_snapshot_id: str
    proposed_snapshot: PipelineSnapshot
    decisions: list[DecisionRecord]
    responses: list[StructuredResponse] = field(default_factory=list)


def snapshot_from_parser(page_results: list[Any]) -> PipelineSnapshot:
    tables: dict[str, LogicalTable] = {}
    rows: dict[str, RowState] = {}
    cells: dict[str, CellState] = {}
    issues: dict[str, IssueState] = {}
    page_dimensions: dict[int, tuple[float, float]] = {}

    for page in page_results:
        page_dimensions[page.page_number] = (page.page_width, page.page_height)
        table_rows: dict[int, list[int]] = {}
        for row_index, table_index in enumerate(page.row_table_indexes, start=1):
            table_rows.setdefault(table_index, []).append(row_index)
        max_columns = len(page.column_names)
        for table_index, source_rows in table_rows.items():
            table_id = f"p{page.page_number}_t{table_index}"
            column_ids = [f"{table_id}_c{index}" for index in range(1, max_columns + 1)]
            row_ids = []
            for source_row in source_rows:
                row_id = f"{table_id}_r{source_row}"
                row_ids.append(row_id)
                rows[row_id] = RowState(
                    row_id=row_id,
                    page_number=page.page_number,
                    source_row_number=source_row,
                    table_id=table_id,
                    ancestor_row_ids=[row_id],
                )
            tables[table_id] = LogicalTable(
                table_id=table_id,
                page_number=page.page_number,
                source_table_index=table_index,
                column_ids=column_ids,
                column_names=dict(zip(column_ids, page.column_names)),
                row_ids=row_ids,
                ancestor_table_ids=[table_id],
            )

        for source_cell in page.cells:
            table_id = f"p{page.page_number}_t{source_cell.table_index}"
            if table_id not in tables:
                continue
            row_id = f"{table_id}_r{source_cell.row_index}"
            column_id = f"{table_id}_c{source_cell.col_index}"
            cell_id = f"{row_id}::{column_id}"
            bbox = None
            if source_cell.x0 is not None:
                bbox = (source_cell.x0, source_cell.y0, source_cell.x1, source_cell.y1)
            cells[cell_id] = CellState(
                cell_id=cell_id,
                row_id=row_id,
                column_id=column_id,
                text=source_cell.text,
                bbox=bbox,
                source_item_indexes=list(source_cell.source_item_indexes),
                warning_ids=[],
                assignment_score=source_cell.assignment_score,
                alternatives=list(source_cell.alternatives),
                ancestor_cell_ids=[cell_id],
            )

        for source_issue in page.issues:
            table_id = f"p{page.page_number}_t{source_issue.get('table_index', 1)}"
            affected = []
            row_indexes = []
            if source_issue.get("row_index") is not None:
                row_indexes.append(source_issue["row_index"])
            row_indexes.extend(source_issue.get("evidence", {}).get("affected_rows", []))
            col_index = source_issue.get("col_index")
            for row_index in sorted(set(row_indexes)):
                if col_index is None:
                    continue
                cell_id = f"{table_id}_r{row_index}::{table_id}_c{col_index}"
                if cell_id in cells:
                    affected.append(cell_id)
            issue_id = source_issue["issue_id"]
            issues[issue_id] = IssueState(
                issue_id=issue_id,
                source_issue_id=issue_id,
                issue_type=source_issue["issue_type"],
                severity=source_issue["severity"],
                table_id=table_id if table_id in tables else None,
                affected_cell_ids=affected,
                status="active",
                explanation=source_issue["explanation"],
                suggested_action=source_issue["suggested_action"],
                evidence=dict(source_issue.get("evidence", {})),
            )
            for cell_id in affected:
                cells[cell_id].warning_ids.append(issue_id)

    snapshot = PipelineSnapshot(
        snapshot_id="snapshot_source",
        phase="source",
        tables=tables,
        rows=rows,
        cells=cells,
        issues=issues,
        page_dimensions=page_dimensions,
    )
    ensure_profiles(snapshot)
    return snapshot


class TableFixerPipeline:
    def __init__(
        self,
        client: OllamaLLMClient,
        *,
        auto_apply_threshold: float = 0.80,
        structural_auto_apply_threshold: float = 0.95,
        context_token_budget: int = 3500,
    ) -> None:
        self.client = client
        self.auto_apply_threshold = auto_apply_threshold
        self.structural_auto_apply_threshold = structural_auto_apply_threshold
        self.context_token_budget = context_token_budget

    def _call(
        self,
        phase: str,
        purpose: str,
        context: dict[str, Any],
        schema: dict[str, Any],
        repair_parent_prompt_id: str | None = None,
    ) -> StructuredResponse:
        return self.client.structured(
            phase=phase,
            purpose=purpose,
            system=SYSTEM_PROMPTS[phase if phase != "regex_repair" else "regex_repair"],
            context=context,
            schema=schema,
            repair_parent_prompt_id=repair_parent_prompt_id,
        )

    def _warning_batches(
        self,
        table_id: str | None,
        issue_type: str,
        contexts: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        contracts = {
            "none": {},
            "redistribute_cell": {
                "source_cell_id": "the supplied cell_id",
                "issue_ids": "all supplied issue_ids for the cell",
                "assignments": [{"target_cell_id": "supplied row destination", "text": "source text segment"}],
            },
        }
        batches: list[dict[str, Any]] = []
        current: list[dict[str, Any]] = []
        for context in contexts:
            candidate = {
                "table_id": table_id,
                "issue_type": issue_type,
                "correction_payload_contracts": contracts,
                "warnings": current + [context],
            }
            token_count = self.client.token_counter.count(json.dumps(candidate, separators=(",", ":")))
            if current and token_count > self.context_token_budget:
                batches.append({
                    "table_id": table_id,
                    "issue_type": issue_type,
                    "correction_payload_contracts": contracts,
                    "warnings": current,
                })
                current = [context]
            else:
                current.append(context)
        if current:
            batches.append({
                "table_id": table_id,
                "issue_type": issue_type,
                "correction_payload_contracts": contracts,
                "warnings": current,
            })
        return batches

    @staticmethod
    def _attach_responses(snapshot: PipelineSnapshot, responses: list[StructuredResponse]) -> None:
        snapshot.prompt_usage.extend(response.usage for response in responses)
        snapshot.prompt_audits.extend(response.audit_record() for response in responses)

    def run_metadata(self, snapshot: PipelineSnapshot, *, auto_apply: bool = False) -> PhaseRun:
        records: list[DecisionRecord] = []
        responses: list[StructuredResponse] = []
        for table_id in list(snapshot.tables):
            context = fit_context_budget(
                metadata_context(snapshot, table_id),
                self.client.token_counter,
                self.context_token_budget,
                "rows",
            )
            response = self._call("metadata", f"metadata:{table_id}", context, METADATA_SCHEMA)
            responses.append(response)
            parsed = response.parsed or {}
            allowed = metadata_candidate_prefix(snapshot, table_id)
            requested_raw = parsed.get("metadata_row_ids", [])
            requested = []
            for row_id in allowed:
                if row_id not in requested_raw:
                    break
                requested.append(row_id)
            table = snapshot.tables[table_id]
            prefix = table.row_ids[:len(requested)]
            errors = list(response.validation_errors)
            unknown = sorted({row_id for row_id in requested_raw if row_id not in table.row_ids})
            if requested != prefix or any(row_id not in allowed for row_id in requested):
                errors.append("Metadata rows must exactly match a contiguous allowed top prefix.")
            confidence = float(parsed.get("confidence", 0))
            if requested and len(requested) >= len(table.row_ids):
                errors.append("Metadata removal must leave at least one data row.")
            valid = not errors and (not auto_apply or confidence >= self.structural_auto_apply_threshold)
            records.append(
                decision(
                    "metadata",
                    table_id,
                    "remove_metadata",
                    confidence,
                    str(parsed.get("reasons", "")),
                    {
                        "metadata_row_ids": requested,
                        "header_continuation_row_ids": parsed.get("header_continuation_row_ids", []),
                        "ignored_response_row_ids": unknown,
                    },
                    valid,
                    errors,
                    response.prompt_id,
                )
            )
        proposed = apply_metadata(snapshot, records)
        self._attach_responses(proposed, responses)
        return PhaseRun("metadata", snapshot.snapshot_id, proposed, records, responses)

    def run_reconciliation(self, snapshot: PipelineSnapshot, *, auto_apply: bool = False) -> PhaseRun:
        records: list[DecisionRecord] = []
        responses: list[StructuredResponse] = []
        for left_id, right_id in adjacent_table_pairs(snapshot):
            evidence = table_pair_evidence(snapshot, left_id, right_id)
            score = float(evidence["deterministic_score"])
            errors: list[str] = []
            response: StructuredResponse | None = None
            if evidence["distinct_title_at_right_start"]:
                action = "keep_separate"
                confidence = max(0.90, round(1.0 - score, 4))
                reason = "The right segment begins with a distinct table-title signal."
            elif not evidence["same_column_count"]:
                action = "keep_separate"
                confidence = 1.0
                reason = "Column counts differ; deterministic merge is unsafe."
            elif score >= 0.90:
                action = "merge"
                confidence = score
                reason = "High deterministic compatibility across geometry, types, and occupancy."
            elif score <= 0.45:
                action = "keep_separate"
                confidence = round(1.0 - score, 4)
                reason = "Low deterministic compatibility or a distinct table-title signal."
            else:
                context = table_reconciliation_context(snapshot, left_id, right_id)
                response = self._call(
                    "reconciliation",
                    f"reconciliation:{left_id}:{right_id}",
                    context,
                    TABLE_RECONCILIATION_SCHEMA,
                )
                responses.append(response)
                parsed = response.parsed or {}
                action = parsed.get("action", "keep_separate")
                confidence = float(parsed.get("confidence", 0))
                reason = parsed.get("reason", "")
                errors.extend(response.validation_errors)
            valid = not errors and (
                not auto_apply
                or action == "keep_separate"
                or confidence >= self.structural_auto_apply_threshold
            )
            record = decision(
                "reconciliation",
                f"{left_id}::{right_id}",
                action,
                confidence,
                reason,
                {
                    "left_table_id": left_id,
                    "right_table_id": right_id,
                    "evidence": evidence,
                    "decision_source": "llm" if response else "deterministic",
                },
                valid,
                errors,
                response.prompt_id if response else None,
            )
            records.append(record)
        proposed = apply_table_reconciliation(snapshot, records)
        for record in records:
            if record.action == "keep_separate" and record.valid:
                record.applied = True
        self._attach_responses(proposed, responses)
        return PhaseRun("reconciliation", snapshot.snapshot_id, proposed, records, responses)

    def run_headers(self, snapshot: PipelineSnapshot, *, auto_apply: bool = False) -> PhaseRun:
        records: list[DecisionRecord] = []
        responses: list[StructuredResponse] = []
        for table_id in list(snapshot.tables):
            accepted_groups: list[list[str]] = []
            confidence_values = []
            reasons = []
            errors = []
            for candidate_id in header_candidates(snapshot, table_id):
                context = fit_context_budget(
                    header_context(snapshot, table_id, candidate_id),
                    self.client.token_counter,
                    self.context_token_budget,
                    "rows",
                )
                response = self._call("headers", f"header:{candidate_id}", context, HEADER_SCHEMA)
                responses.append(response)
                parsed = response.parsed or {}
                if parsed.get("is_header"):
                    allowed_continuations = allowed_header_continuations(snapshot, table_id, candidate_id)
                    group = [candidate_id] + [
                        row_id for row_id in allowed_continuations
                        if row_id in parsed.get("continuation_row_ids", [])
                    ]
                    accepted_groups.append(group)
                    confidence_values.append(float(parsed.get("confidence", 0)))
                    reasons.append(parsed.get("reason", ""))
                errors.extend(response.validation_errors)
            confidence = min(confidence_values, default=0.0)
            accepted_groups = normalize_header_groups(snapshot.tables[table_id].row_ids, accepted_groups)
            table_rows = snapshot.tables[table_id].row_ids
            for index, group in enumerate(accepted_groups):
                if len(group) > 2:
                    errors.append("Header groups may contain at most two rows.")
                    continue
                start = table_rows.index(group[0])
                next_start = table_rows.index(accepted_groups[index + 1][0]) if index + 1 < len(accepted_groups) else len(table_rows)
                if next_start - (start + len(group)) < 1:
                    errors.append(f"Header group starting at {group[0]} leaves no data rows.")
            valid = bool(accepted_groups) and not errors and (
                not auto_apply or confidence >= self.structural_auto_apply_threshold
            )
            records.append(
                decision(
                    "headers",
                    table_id,
                    "accept_header",
                    confidence,
                    " | ".join(reasons),
                    {"header_groups": accepted_groups},
                    valid,
                    errors,
                    responses[-1].prompt_id if responses else None,
                )
            )
        proposed = apply_headers(snapshot, records)
        self._attach_responses(proposed, responses)
        return PhaseRun("headers", snapshot.snapshot_id, proposed, records, responses)

    def run_columns(self, snapshot: PipelineSnapshot, *, auto_apply: bool = False) -> PhaseRun:
        current = snapshot.clone(f"snapshot_columns_{uuid4().hex[:8]}", "columns")
        records: list[DecisionRecord] = []
        responses: list[StructuredResponse] = []
        targets = [
            issue.issue_id for issue in snapshot.active_warnings()
            if issue.issue_type == "possible_merged_column"
        ]
        for issue_id in targets:
            if issue_id not in current.issues or current.issues[issue_id].status != "active":
                continue
            try:
                context = fit_context_budget(
                    column_split_context(current, issue_id),
                    self.client.token_counter,
                    self.context_token_budget,
                    "samples",
                )
            except ValueError as exc:
                record = decision(
                    "columns",
                    issue_id,
                    "mark_for_review",
                    0.0,
                    str(exc),
                    {},
                    False,
                    [str(exc)],
                    None,
                )
                records.append(record)
                current.decisions.append(record)
                current.issues[issue_id].status = "review"
                continue
            private_table_id = context.pop("_table_id")
            private_column_id = context.pop("_column_id")
            response = self._call("columns", f"column:{issue_id}", context, COLUMN_SPLIT_SCHEMA)
            responses.append(response)
            parsed = response.parsed or {}
            action = parsed.get("action", "no_split")
            confidence = float(parsed.get("confidence", 0))
            errors = list(response.validation_errors)
            compiled = None
            actual: dict[str, list[str]] = {}
            payload = {
                "table_id": private_table_id,
                "column_id": private_column_id,
                "regex": parsed.get("regex", ""),
                "new_headers": parsed.get("new_headers", []),
                "expected": parsed.get("expected", {}),
            }
            if action == "no_split":
                payload = {
                    "table_id": private_table_id,
                    "column_id": private_column_id,
                }
            if action == "split":
                table = current.tables[private_table_id]
                values = [
                    cell.text for row_id in table.row_ids
                    if (cell := current.cells.get(f"{row_id}::{private_column_id}")) and cell.text.strip()
                ]
                compiled, validation_errors, actual = validate_split_regex(
                    context["current_header"],
                    payload["regex"],
                    payload["new_headers"],
                    values,
                    payload["expected"],
                )
                errors.extend(validation_errors)
                repair_count = 0
                while errors and repair_count < 2:
                    repair_context = {
                        "source_header": context["current_header"],
                        "sample_values": values[:5],
                        "regex": payload["regex"],
                        "errors": errors,
                        "actual": actual,
                        "expected": payload["expected"],
                    }
                    repaired = self._call(
                        "regex_repair",
                        f"regex_repair:{issue_id}:{repair_count + 1}",
                        repair_context,
                        COLUMN_SPLIT_SCHEMA,
                        response.prompt_id,
                    )
                    responses.append(repaired)
                    repair_count += 1
                    repaired_parsed = repaired.parsed or {}
                    payload["regex"] = repaired_parsed.get("regex", payload["regex"])
                    payload["new_headers"] = repaired_parsed.get("new_headers", payload["new_headers"])
                    compiled, errors, actual = validate_split_regex(
                        context["current_header"],
                        payload["regex"],
                        payload["new_headers"],
                        values,
                        payload["expected"],
                    )
                if errors:
                    inferred = infer_whitespace_split(context["current_header"], values)
                    if inferred:
                        payload["regex"], payload["new_headers"] = inferred
                        payload["expected"] = {}
                        compiled, errors, actual = validate_split_regex(
                            context["current_header"],
                            payload["regex"],
                            payload["new_headers"],
                            values,
                            {},
                        )
            valid = not errors and (not auto_apply or confidence >= self.structural_auto_apply_threshold)
            record = decision(
                "columns",
                issue_id,
                action,
                confidence,
                parsed.get("reason", ""),
                payload,
                valid,
                errors,
                response.prompt_id,
            )
            records.append(record)
            if action == "split" and valid and compiled:
                current = apply_column_split(current, record, compiled, actual)
            else:
                current.decisions.append(record)
                if action == "no_split" and valid:
                    current.issues[issue_id].status = "dismissed"
                    record.applied = True

        for table_id in list(current.tables):
            table = current.tables[table_id]
            placeholder_ids = [
                column_id for column_id in table.column_ids
                if re.fullmatch(r"col_\d+", table.column_names.get(column_id, column_id), re.IGNORECASE)
            ]
            for column_id in placeholder_ids:
                if column_id not in current.tables[table_id].column_ids:
                    continue
                values = [
                    current.cells[cell_id].text
                    for row_id in current.tables[table_id].row_ids
                    if (cell_id := f"{row_id}::{column_id}") in current.cells
                    and current.cells[cell_id].text.strip()
                ]
                all_values = [
                    current.cells[cell_id].text
                    for row_id in (
                        current.tables[table_id].metadata_row_ids
                        + current.tables[table_id].header_row_ids
                        + current.tables[table_id].row_ids
                    )
                    if (cell_id := f"{row_id}::{column_id}") in current.cells
                    and current.cells[cell_id].text.strip()
                ]
                target_id = f"placeholder:{table_id}:{column_id}"
                if not all_values:
                    record = decision(
                        "columns",
                        target_id,
                        "remove_empty",
                        1.0,
                        "Placeholder column is empty in all data, metadata, and header rows.",
                        {"table_id": table_id, "column_id": column_id},
                        True,
                        [],
                        None,
                    )
                    records.append(record)
                    current = apply_placeholder_column_action(current, record, {}, [])
                    continue
                if not values:
                    record = decision(
                        "columns",
                        target_id,
                        "unresolved",
                        0.0,
                        "Placeholder has no data-row values but has preserved metadata or header text.",
                        {
                            "table_id": table_id,
                            "column_id": column_id,
                            "target_headers": [],
                            "regex": "",
                            "new_header": "",
                        },
                        True,
                        [],
                        None,
                    )
                    records.append(record)
                    current = apply_placeholder_column_action(current, record, {}, [])
                    continue

                context = fit_context_budget(
                    placeholder_column_context(current, table_id, column_id),
                    self.client.token_counter,
                    self.context_token_budget,
                    "sample_values",
                )
                private_table_id = context.pop("_table_id")
                private_column_id = context.pop("_column_id")
                response = self._call(
                    "placeholder_column",
                    target_id,
                    context,
                    PLACEHOLDER_COLUMN_SCHEMA,
                )
                responses.append(response)
                parsed = response.parsed or {}
                action = parsed.get("action", "unresolved")
                confidence = float(parsed.get("confidence", 0))
                payload = {
                    "table_id": private_table_id,
                    "column_id": private_column_id,
                    "target_headers": parsed.get("target_headers", []),
                    "regex": parsed.get("regex", ""),
                    "new_header": parsed.get("new_header", ""),
                }
                errors = list(response.validation_errors)
                validation_errors, placements, target_ids = validate_placeholder_column_action(
                    current,
                    private_table_id,
                    private_column_id,
                    action,
                    payload["target_headers"],
                    payload["regex"],
                    payload["new_header"],
                )
                errors.extend(validation_errors)
                if action == "split" and errors:
                    inferred = infer_whitespace_split(" ".join(payload["target_headers"]), values)
                    if inferred and len(inferred[1]) == len(payload["target_headers"]):
                        payload["regex"] = inferred[0]
                        errors, placements, target_ids = validate_placeholder_column_action(
                            current,
                            private_table_id,
                            private_column_id,
                            action,
                            payload["target_headers"],
                            payload["regex"],
                            payload["new_header"],
                        )
                valid = not errors and (
                    not auto_apply or confidence >= self.structural_auto_apply_threshold
                )
                record = decision(
                    "columns",
                    target_id,
                    action,
                    confidence,
                    parsed.get("reason", ""),
                    payload,
                    valid,
                    sorted(set(errors)),
                    response.prompt_id,
                )
                records.append(record)
                if valid and action in {"move", "split", "rename", "unresolved"}:
                    current = apply_placeholder_column_action(current, record, placements, target_ids)
                else:
                    current.decisions.append(record)
        self._attach_responses(current, responses)
        return PhaseRun("columns", snapshot.snapshot_id, current, records, responses)

    def run_warnings(self, snapshot: PipelineSnapshot, *, auto_apply: bool = False) -> PhaseRun:
        records: list[DecisionRecord] = []
        responses: list[StructuredResponse] = []
        row_issues: dict[str, list[IssueState]] = {}
        for issue in snapshot.issues.values():
            if issue.status != "active" or issue.issue_type == "possible_merged_column":
                continue
            for cell_id in issue.affected_cell_ids:
                if cell_id in snapshot.cells:
                    row_id = snapshot.cells[cell_id].row_id
                    if issue not in row_issues.setdefault(row_id, []):
                        row_issues[row_id].append(issue)
        for row_id, issues in row_issues.items():
            context = row_repair_context(snapshot, row_id, issues)
            issue_ids = context.pop("_issue_ids")
            column_ids_by_header = context.pop("_column_ids_by_header")
            context.pop("_row_id")
            def build_record(response: StructuredResponse) -> DecisionRecord:
                parsed = response.parsed or {}
                if isinstance(parsed.get("decisions"), list) and parsed["decisions"]:
                    legacy = parsed["decisions"][0]
                    legacy_payload = legacy.get("payload", {})
                    reverse_headers = {
                        f"{row_id}::{column_id}": header
                        for header, column_id in column_ids_by_header.items()
                    }
                    legacy_action = legacy.get("correction_action", legacy.get("action", "none"))
                    legacy_assignments = []
                    if legacy_action == "move_cell":
                        source_id = legacy_payload.get("source_cell_id")
                        target_id = legacy_payload.get("target_cell_id")
                        if source_id in snapshot.cells and target_id in reverse_headers:
                            legacy_assignments = [{
                                "target_header": reverse_headers[target_id],
                                "text": snapshot.cells[source_id].text,
                            }]
                    elif legacy_action == "redistribute_cell":
                        legacy_assignments = [
                            {
                                "target_header": reverse_headers.get(assignment.get("target_cell_id"), ""),
                                "text": assignment.get("text"),
                            }
                            for assignment in legacy_payload.get("assignments", [])
                            if isinstance(assignment, dict)
                        ]
                    parsed = {
                        "needs_correction": legacy.get("needs_correction", legacy_action != "none"),
                        "confidence": legacy.get("confidence", 0),
                        "reason": legacy.get("reason", ""),
                        "assignments": legacy_assignments,
                    }
                errors = list(response.validation_errors)
                needs_correction = parsed.get("needs_correction")
                action = "redistribute_row" if needs_correction is True else "no_issue"
                if needs_correction not in {True, False}:
                    action = "invalid_warning_decision"
                    errors.append("Row repair decision must provide a binary needs_correction value.")
                raw_assignments = parsed.get("assignments", [])
                if isinstance(parsed.get("final_values"), dict):
                    raw_assignments = [
                        {"target_header": header, "text": text}
                        for header, text in parsed["final_values"].items()
                    ]
                assignments = []
                for assignment in raw_assignments:
                    header = assignment.get("target_header") if isinstance(assignment, dict) else None
                    text = assignment.get("text") if isinstance(assignment, dict) else None
                    column_id = column_ids_by_header.get(header)
                    if not column_id:
                        errors.append(f"Unknown or unavailable target header: {header}")
                        continue
                    if not isinstance(text, str) or not text.strip():
                        errors.append("Assignments must contain non-empty final text only.")
                        continue
                    assignments.append({
                        "target_cell_id": f"{row_id}::{column_id}",
                        "text": text,
                    })
                if needs_correction is False and assignments:
                    errors.append("A no-correction decision must not include assignments.")
                payload = {
                    "source_cell_ids": [
                        f"{row_id}::{column_id}"
                        for column_id in column_ids_by_header.values()
                    ],
                    "assignments": assignments,
                    "issue_ids": issue_ids,
                }
                confidence = float(parsed.get("confidence", 0))
                record = decision(
                    "warnings", issue_ids[0], action, confidence, parsed.get("reason", ""),
                    payload, True, errors, response.prompt_id,
                )
                errors.extend(validate_warning_action(snapshot, record))
                required_confidence = max(self.auto_apply_threshold, 0.90) if action == "no_issue" else self.auto_apply_threshold
                record.valid = not errors and (not auto_apply or confidence >= required_confidence)
                return record

            response = self._call("warnings", f"row_repair:{row_id}", context, ROW_REPAIR_SCHEMA)
            responses.append(response)
            record = build_record(response)
            if not record.valid and record.action == "redistribute_row":
                retry_context = {
                    **context,
                    "retry_instruction": (
                        "Return corrected final_values only. Include each listed header at most once and omit empty "
                        "values. Split merged text instead of copying it unchanged. Never duplicate or invent text."
                    ),
                    "previous_final_values": (response.parsed or {}).get(
                        "final_values",
                        (response.parsed or {}).get("assignments", []),
                    ),
                    "validation_errors": record.validation_errors,
                }
                retry = self._call("warnings", f"row_repair_retry:{row_id}", retry_context, ROW_REPAIR_SCHEMA, response.prompt_id)
                responses.append(retry)
                record = build_record(retry)
            records.append(record)
        touched: set[str] = set()
        for record in records:
            action_cells = {
                value for key, value in record.payload.items()
                if key.endswith("_cell_id") and isinstance(value, str)
            } | {
                assignment.get("target_cell_id")
                for assignment in record.payload.get("assignments", [])
                if isinstance(assignment, dict) and isinstance(assignment.get("target_cell_id"), str)
            }
            if action_cells & touched and record.action != "no_issue":
                record.valid = False
                record.validation_errors.append("Conflicting action affects a cell already changed in this batch.")
            if record.valid:
                touched.update(action_cells)
        proposed = apply_warning_decisions(snapshot, records)
        self._attach_responses(proposed, responses)
        return PhaseRun("warnings", snapshot.snapshot_id, proposed, records, responses)

    @staticmethod
    def _sanitize_warning_payload(action: str, payload: dict[str, Any]) -> dict[str, Any]:
        if action in {"no_issue", "invalid_warning_decision"}:
            return {}
        if action in {"move_cell", "merge_adjacent_cells"}:
            return {
                "source_cell_id": payload.get("source_cell_id", payload.get("cell_id")),
                "target_cell_id": payload.get("target_cell_id"),
            }
        if action == "split_cell_text":
            return {
                "source_cell_id": payload.get("source_cell_id", payload.get("cell_id")),
                "new_text": payload.get("new_text", payload.get("new_cell_text")),
            }
        if action == "redistribute_cell":
            return {
                "source_cell_id": payload.get("source_cell_id", payload.get("cell_id")),
                "assignments": [
                    {
                        "target_cell_id": assignment.get("target_cell_id"),
                        "text": assignment.get("text"),
                    }
                    for assignment in payload.get("assignments", [])
                    if isinstance(assignment, dict)
                ],
            }
        return {}
