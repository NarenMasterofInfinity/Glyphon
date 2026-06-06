from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from table_fixer.context import ensure_profiles, header_context, metadata_context
from table_fixer.models import (
    CellState,
    IssueState,
    LogicalTable,
    PipelineSnapshot,
    PromptUsage,
    RowState,
)
from table_fixer.ollama_client import OllamaLLMClient, StructuredResponse
from table_fixer.pipeline import TableFixerPipeline, normalize_header_groups
from table_fixer.repairs import apply_headers, decision
from table_fixer.token_counting import TokenCounter
from table_fixer.workspace import persist_snapshot


class MockClient:
    model = "mock"
    token_counter = TokenCounter()

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def structured(self, **kwargs):
        self.calls.append(kwargs)
        parsed = self.responses.pop(0)
        prompt_id = f"prompt_{len(self.calls)}"
        usage = PromptUsage(
            prompt_id=prompt_id,
            phase=kwargs["phase"],
            purpose=kwargs["purpose"],
            model="mock",
            system_tokens=5,
            context_tokens=10,
            input_tokens=15,
            output_tokens=5,
            duration_ms=1.0,
        )
        return StructuredResponse(prompt_id, parsed, "{}", [], usage)


class HeaderAwareMockClient(MockClient):
    """Makes binary warning decisions using only the warning prompt context."""

    def __init__(self):
        super().__init__([])

    def structured(self, **kwargs):
        columns = kwargs["context"]["columns"]
        expected_header = kwargs["context"]["problem"].removeprefix("Expected header: ")
        source = next(column for column in columns if column["current_text"])
        if source["header"] == expected_header:
            response = {
                "needs_correction": False,
                "confidence": 0.99,
                "reason": f"Value is already under {expected_header}.",
                "assignments": [],
            }
        else:
            response = {
                "needs_correction": True,
                "confidence": 0.99,
                "reason": f"Value belongs under {expected_header}.",
                "assignments": [{"target_header": expected_header, "text": source["current_text"]}],
            }
        self.responses.append(response)
        return super().structured(**kwargs)


def make_snapshot(row_values, column_names=("col_1", "col_2"), issues=None):
    table_id = "p1_t1"
    column_ids = [f"{table_id}_c{index}" for index in range(1, len(column_names) + 1)]
    table = LogicalTable(
        table_id=table_id,
        page_number=1,
        source_table_index=1,
        column_ids=column_ids,
        column_names=dict(zip(column_ids, column_names)),
        row_ids=[],
    )
    rows = {}
    cells = {}
    for row_number, values in enumerate(row_values, start=1):
        row_id = f"{table_id}_r{row_number}"
        table.row_ids.append(row_id)
        rows[row_id] = RowState(row_id, 1, row_number, table_id, ancestor_row_ids=[row_id])
        for index, value in enumerate(values, start=1):
            column_id = column_ids[index - 1]
            cell_id = f"{row_id}::{column_id}"
            cells[cell_id] = CellState(
                cell_id=cell_id,
                row_id=row_id,
                column_id=column_id,
                text=value,
                bbox=(10.0 * index, 10.0 * row_number, 10.0 * index + 8, 10.0 * row_number + 5),
                ancestor_cell_ids=[cell_id],
            )
    snapshot = PipelineSnapshot(
        snapshot_id="source",
        phase="source",
        tables={table_id: table},
        rows=rows,
        cells=cells,
        issues=issues or {},
        page_dimensions={1: (200.0, 200.0)},
    )
    ensure_profiles(snapshot)
    return snapshot


def append_table(snapshot, table_index, row_values, column_names=("col_1", "col_2")):
    table_id = f"p1_t{table_index}"
    column_ids = [f"{table_id}_c{index}" for index in range(1, len(column_names) + 1)]
    table = LogicalTable(
        table_id=table_id,
        page_number=1,
        source_table_index=table_index,
        column_ids=column_ids,
        column_names=dict(zip(column_ids, column_names)),
        row_ids=[],
        ancestor_table_ids=[table_id],
    )
    start = max((row.source_row_number for row in snapshot.rows.values()), default=0) + 1
    for row_number, values in enumerate(row_values, start=start):
        row_id = f"{table_id}_r{row_number}"
        table.row_ids.append(row_id)
        snapshot.rows[row_id] = RowState(row_id, 1, row_number, table_id, ancestor_row_ids=[row_id])
        for index, value in enumerate(values, start=1):
            column_id = column_ids[index - 1]
            cell_id = f"{row_id}::{column_id}"
            snapshot.cells[cell_id] = CellState(
                cell_id=cell_id,
                row_id=row_id,
                column_id=column_id,
                text=value,
                bbox=(10.0 * index, 10.0 * row_number, 10.0 * index + 8, 10.0 * row_number + 5),
                ancestor_cell_ids=[cell_id],
            )
    snapshot.tables[table_id] = table
    ensure_profiles(snapshot)
    return table_id


def make_adjacent_tables(left_rows, right_rows, column_names=("col_1", "col_2")):
    snapshot = make_snapshot(left_rows, column_names)
    append_table(snapshot, 2, right_rows, column_names)
    return snapshot


class PipelineTests(unittest.TestCase):
    def test_structured_client_records_native_and_estimated_usage(self):
        client = OllamaLLMClient(model="mock")
        client._post = lambda path, payload: {
            "message": {"content": '{"answer":"yes"}'},
            "prompt_eval_count": 21,
            "eval_count": 4,
        }

        response = client.structured(
            phase="metadata",
            purpose="usage-test",
            system="Return JSON.",
            context={"row": "Title"},
            schema={"type": "object"},
        )

        self.assertTrue(response.valid)
        self.assertEqual(response.usage.native_prompt_tokens, 21)
        self.assertEqual(response.usage.native_output_tokens, 4)
        self.assertGreater(response.usage.input_tokens, 0)
        self.assertEqual(response.audit_record()["context"], {"row": "Title"})

    def test_workspace_snapshot_writes_json_and_table_csv(self):
        snapshot = make_snapshot([["Name", "Age"], ["Bob", "20"]])
        with TemporaryDirectory() as directory:
            persist_snapshot(Path(directory), "source", snapshot)

            self.assertTrue((Path(directory) / "source" / "snapshot.json").exists())
            self.assertTrue((Path(directory) / "source" / "issues.json").exists())
            self.assertTrue((Path(directory) / "source" / "tables" / "p1_t1.csv").exists())

    def test_metadata_context_is_adaptive(self):
        structured = make_snapshot([["A", "B"]] + [[str(index), str(index + 1)] for index in range(1, 20)])
        self.assertEqual(len(metadata_context(structured, "p1_t1")["rows"]), 12)

        unstructured = make_snapshot([[f"metadata {index}", ""] for index in range(20)])
        self.assertEqual(len(metadata_context(unstructured, "p1_t1")["rows"]), 20)

    def test_reconciliation_deterministically_merges_compatible_adjacent_tables(self):
        snapshot = make_adjacent_tables([["A", "1"], ["B", "2"]], [["C", "3"], ["D", "4"]])
        client = MockClient([])

        result = TableFixerPipeline(client).run_reconciliation(snapshot)

        self.assertEqual(client.calls, [])
        self.assertEqual(list(result.proposed_snapshot.tables), ["p1_t1"])
        self.assertEqual(len(result.proposed_snapshot.tables["p1_t1"].row_ids), 4)
        remapped = "p1_t2_r3::p1_t1_c1"
        self.assertIn(remapped, result.proposed_snapshot.cells)
        self.assertIn("p1_t2_r3::p1_t2_c1", result.proposed_snapshot.cell_lineage)

    def test_reconciliation_keeps_distinct_titled_table_separate(self):
        snapshot = make_adjacent_tables(
            [["A", "1"], ["B", "2"]],
            [["Table 2", ""], ["Name", "Value"], ["C", "3"]],
        )
        client = MockClient([])

        result = TableFixerPipeline(client).run_reconciliation(snapshot)

        self.assertEqual(client.calls, [])
        self.assertEqual(set(result.proposed_snapshot.tables), {"p1_t1", "p1_t2"})
        self.assertEqual(result.decisions[0].action, "keep_separate")

    def test_reconciliation_uses_compact_llm_call_for_ambiguous_pair(self):
        snapshot = make_adjacent_tables([["A", "1"], ["B", "2"]], [["C", "3"], ["D", "4"]])
        for cell in snapshot.table_cells("p1_t2"):
            x0, y0, x1, y1 = cell.bbox
            cell.bbox = (x0 + 8, y0, x1 + 8, y1)
        client = MockClient([{"action": "merge", "confidence": 0.96, "reason": "continuation"}])

        result = TableFixerPipeline(client).run_reconciliation(snapshot)

        self.assertEqual(len(client.calls), 1)
        self.assertLessEqual(len(client.calls[0]["context"]["left_tail"]), 3)
        self.assertLessEqual(len(client.calls[0]["context"]["right_head"]), 3)
        self.assertEqual(list(result.proposed_snapshot.tables), ["p1_t1"])

    def test_reconciliation_merges_compatible_chain_in_one_phase(self):
        snapshot = make_adjacent_tables([["A", "1"]], [["B", "2"]])
        append_table(snapshot, 3, [["C", "3"]])

        result = TableFixerPipeline(MockClient([])).run_reconciliation(snapshot)

        self.assertEqual(list(result.proposed_snapshot.tables), ["p1_t1"])
        self.assertEqual(len(result.proposed_snapshot.tables["p1_t1"].row_ids), 3)

    def test_header_context_includes_predecessor_tail_at_segment_start(self):
        snapshot = make_adjacent_tables([["Header part", ""]], [["Name", "Value"], ["C", "3"]])

        context = header_context(snapshot, "p1_t2", "p1_t2_r2")

        self.assertEqual(context["predecessor_tail_context_only"][0]["row_id"], "p1_t1_r1")

    def test_metadata_preserves_prefix_and_records_usage(self):
        snapshot = make_snapshot([["Title", ""], ["Subtitle", ""], ["Name", "Age"], ["Bob", "20"]])
        client = MockClient([{
            "metadata_row_ids": ["p1_t1_r1", "p1_t1_r2"],
            "header_continuation_row_ids": [],
            "confidence": 0.9,
            "reasons": {"p1_t1_r1": "title", "p1_t1_r2": "subtitle"},
        }])

        result = TableFixerPipeline(client).run_metadata(snapshot)

        table = result.proposed_snapshot.tables["p1_t1"]
        self.assertEqual(table.metadata_row_ids, ["p1_t1_r1", "p1_t1_r2"])
        self.assertEqual(table.row_ids[0], "p1_t1_r3")
        self.assertEqual(len(result.proposed_snapshot.prompt_usage), 1)
        self.assertEqual(len(result.proposed_snapshot.prompt_audits), 1)

    def test_metadata_clamps_runaway_response_to_allowed_prefix(self):
        snapshot = make_snapshot([["Table 1", ""], ["Name", "Age"], ["Bob", "20"]])
        client = MockClient([{
            "metadata_row_ids": ["p1_t1_r1", "p1_t1_r2", "p1_t1_r3"] + ["p1_t1_r3"] * 100,
            "header_continuation_row_ids": [],
            "confidence": 0.95,
            "reasons": {},
        }])

        result = TableFixerPipeline(client).run_metadata(snapshot)

        self.assertEqual(result.proposed_snapshot.tables["p1_t1"].metadata_row_ids, ["p1_t1_r1"])
        self.assertEqual(result.proposed_snapshot.tables["p1_t1"].row_ids, ["p1_t1_r2", "p1_t1_r3"])

    def test_auto_apply_rejects_low_confidence(self):
        snapshot = make_snapshot([["Title", ""], ["Name", "Age"]])
        client = MockClient([{
            "metadata_row_ids": ["p1_t1_r1"],
            "header_continuation_row_ids": [],
            "confidence": 0.7,
            "reasons": {},
        }])

        result = TableFixerPipeline(client).run_metadata(snapshot, auto_apply=True)

        self.assertEqual(result.proposed_snapshot.tables["p1_t1"].metadata_row_ids, [])
        self.assertFalse(result.decisions[0].valid)

    def test_headers_create_logical_table_and_preserve_header(self):
        snapshot = make_snapshot([["Report", ""], ["Name", "Age"], ["Bob", "20"], ["Sue", "30"]])
        record = decision(
            "headers",
            "p1_t1",
            "accept_header",
            0.95,
            "header",
            {"header_groups": [["p1_t1_r2"]]},
            True,
            [],
            "prompt_1",
        )

        result = apply_headers(snapshot, [record])

        self.assertIn("p1_t1_h1", result.tables)
        self.assertEqual(result.tables["p1_t1_h1"].header_row_ids, ["p1_t1_r2"])
        self.assertEqual(result.tables["p1_t1_h1"].column_names["p1_t1_c1"], "Name")
        self.assertEqual(
            normalize_header_groups(
                ["r1", "r2", "r3", "r4"],
                [["r2", "r3"], ["r3"], ["r4"]],
            ),
            [["r2", "r3", "r4"]],
        )

    def test_invalid_header_groups_do_not_partially_mutate_snapshot(self):
        snapshot = make_snapshot([["Name", "Age"], ["Bob", "20"]])
        record = decision(
            "headers",
            "p1_t1",
            "accept_header",
            1.0,
            "header",
            {"header_groups": [["p1_t1_r1"], ["unknown_row"]]},
            True,
            [],
            "prompt_1",
        )

        result = apply_headers(snapshot, [record])

        self.assertFalse(record.valid)
        self.assertEqual(result.tables["p1_t1"].row_ids, ["p1_t1_r1", "p1_t1_r2"])
        self.assertEqual(result.rows["p1_t1_r1"].role, "data")

    def test_column_split_validates_full_column_and_rebases_issues(self):
        snapshot = make_snapshot([["Bob 20"], ["Sue 30"]], column_names=("Name Age",))
        issue = IssueState(
            issue_id="merged",
            source_issue_id="merged",
            issue_type="possible_merged_column",
            severity="warning",
            table_id="p1_t1",
            affected_cell_ids=["p1_t1_r1::p1_t1_c1", "p1_t1_r2::p1_t1_c1"],
            status="active",
            explanation="merged",
            suggested_action="split",
        )
        snapshot.issues["merged"] = issue
        for cell_id in issue.affected_cell_ids:
            snapshot.cells[cell_id].warning_ids.append("merged")
        client = MockClient([{
            "action": "split",
            "confidence": 0.95,
            "reason": "two stable fields",
            "regex": r"(?P<name>[A-Za-z]+) (?P<age>\d+)",
            "new_headers": ["Name", "Age"],
            "expected": {"Bob 20": ["Bob", "20"], "Sue 30": ["Sue", "30"]},
        }])

        result = TableFixerPipeline(client).run_columns(snapshot)
        proposed = result.proposed_snapshot

        self.assertEqual(len(proposed.tables["p1_t1"].column_ids), 2)
        self.assertEqual(proposed.issues["merged"].status, "resolved")
        self.assertFalse(any(issue.ancestor_issue_ids == ["merged"] for issue in proposed.issues.values()))

    def test_column_regex_repair_falls_back_to_stable_whitespace_split(self):
        snapshot = make_snapshot([["Bob 20"], ["Sue 30"]], column_names=("Name Age",))
        affected = ["p1_t1_r1::p1_t1_c1", "p1_t1_r2::p1_t1_c1"]
        snapshot.issues["merged"] = IssueState(
            "merged", "merged", "possible_merged_column", "warning", "p1_t1",
            affected, "active", "merged", "split",
        )
        invalid = {
            "action": "split",
            "confidence": 0.95,
            "reason": "split",
            "regex": "(",
            "new_headers": ["Name", "Age"],
            "expected": {},
        }
        client = MockClient([invalid, invalid, invalid])

        result = TableFixerPipeline(client).run_columns(snapshot)

        self.assertEqual(len(client.calls), 3)
        self.assertTrue(result.decisions[0].valid)
        self.assertEqual(len(result.proposed_snapshot.tables["p1_t1"].column_ids), 2)

    def test_warning_phase_groups_and_resolves_cell_info_with_warning(self):
        snapshot = make_snapshot([["Bob", "20"]])
        warning_cell = "p1_t1_r1::p1_t1_c1"
        snapshot.issues["warn"] = IssueState(
            "warn", "warn", "ambiguous_column_assignment", "warning", "p1_t1",
            [warning_cell], "active", "warning", "review",
        )
        snapshot.issues["info"] = IssueState(
            "info", "info", "low_ocr_confidence", "info", "p1_t1",
            [warning_cell], "active", "info", "review",
        )
        client = MockClient([{
            "decisions": [{
                "issue_id": "warn",
                "needs_correction": False,
                "correction_action": "none",
                "confidence": 0.9,
                "reason": "placement is coherent",
                "payload": {},
            }]
        }])

        result = TableFixerPipeline(client).run_warnings(snapshot)

        self.assertEqual(len(client.calls), 1)
        self.assertEqual(result.proposed_snapshot.issues["warn"].status, "dismissed")
        self.assertEqual(result.proposed_snapshot.issues["info"].status, "dismissed")
        self.assertNotIn("issue_ids", client.calls[0]["context"])

    def test_warning_correction_rejects_invalid_occupied_target_move(self):
        snapshot = make_snapshot([["Bob", "20"]])
        source_cell = "p1_t1_r1::p1_t1_c1"
        target_cell = "p1_t1_r1::p1_t1_c2"
        snapshot.issues["warn"] = IssueState(
            "warn", "warn", "ambiguous_column_assignment", "warning", "p1_t1",
            [source_cell], "active", "warning", "review",
        )
        client = MockClient([{
            "decisions": [{
                "issue_id": "warn",
                "needs_correction": True,
                "correction_action": "move_cell",
                "confidence": 1.0,
                "reason": "move",
                "payload": {"source_cell_id": source_cell, "target_cell_id": target_cell},
            }]
        }])
        client.responses.append(client.responses[0])

        result = TableFixerPipeline(client).run_warnings(snapshot, auto_apply=True)

        self.assertFalse(result.decisions[0].valid)
        self.assertIn(
            "Row repair must preserve every source token exactly once.",
            result.decisions[0].validation_errors,
        )
        self.assertEqual(result.proposed_snapshot.cells[source_cell].text, "Bob")
        self.assertEqual(result.proposed_snapshot.cells[target_cell].text, "20")
        self.assertEqual(result.proposed_snapshot.issues["warn"].status, "active")

    def test_warning_context_includes_relevant_logical_headers(self):
        snapshot = make_snapshot([["Bob", "20"]], column_names=("Name", "Age"))
        warning_cell = "p1_t1_r1::p1_t1_c1"
        snapshot.issues["warn"] = IssueState(
            "warn", "warn", "ambiguous_column_assignment", "warning", "p1_t1",
            [warning_cell], "active", "warning", "review",
        )
        client = MockClient([{
            "decisions": [{
                "issue_id": "warn",
                "needs_correction": False,
                "correction_action": "none",
                "confidence": 0.95,
                "reason": "Name is correctly under Name",
                "payload": {},
            }]
        }])

        TableFixerPipeline(client).run_warnings(snapshot)

        context = client.calls[0]["context"]
        self.assertEqual([column["header"] for column in context["columns"]], ["Name", "Age"])
        self.assertNotIn("cell_id", json.dumps(context))

    def test_warning_binary_correction_applies_valid_code_action(self):
        snapshot = make_snapshot([["Bob", ""]])
        source_cell = "p1_t1_r1::p1_t1_c1"
        target_cell = "p1_t1_r1::p1_t1_c2"
        snapshot.issues["warn"] = IssueState(
            "warn", "warn", "ambiguous_column_assignment", "warning", "p1_t1",
            [source_cell], "active", "warning", "review",
        )
        client = MockClient([{
            "decisions": [{
                "issue_id": "warn",
                "needs_correction": True,
                "correction_action": "move_cell",
                "confidence": 0.99,
                "reason": "value belongs under Age",
                "payload": {"source_cell_id": source_cell, "target_cell_id": target_cell},
            }]
        }])
        client.responses.append(client.responses[0])

        result = TableFixerPipeline(client).run_warnings(snapshot, auto_apply=True)

        self.assertEqual(result.proposed_snapshot.cells[source_cell].text, "")
        self.assertEqual(result.proposed_snapshot.cells[target_cell].text, "Bob")
        self.assertEqual(result.proposed_snapshot.issues["warn"].status, "resolved")

    def test_mock_warning_data_uses_headers_for_binary_decisions_and_code_repairs(self):
        fixture_path = Path(__file__).parent / "mock_data" / "warning_header_context.json"
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        snapshot = make_snapshot(fixture["rows"], tuple(fixture["columns"]))
        for warning in fixture["warnings"]:
            cell_id = (
                f"p1_t1_r{warning['row']}::"
                f"p1_t1_c{warning['column']}"
            )
            snapshot.issues[warning["issue_id"]] = IssueState(
                warning["issue_id"],
                warning["issue_id"],
                warning["issue_type"],
                "warning",
                "p1_t1",
                [cell_id],
                "active",
                f"Expected header: {warning['expected_header']}",
                "repair",
                evidence={"expected_header": warning["expected_header"]},
            )

        result = TableFixerPipeline(HeaderAwareMockClient()).run_warnings(
            snapshot,
            auto_apply=True,
        )

        actions = {record.target_id: record.action for record in result.decisions}
        expected_actions = {
            warning["issue_id"]: (
                "redistribute_row"
                if warning["expected_action"] == "move_cell"
                else warning["expected_action"]
            )
            for warning in fixture["warnings"]
        }
        self.assertEqual(actions, expected_actions)
        self.assertNotIn("mark_for_review", actions.values())
        actual_rows = [
            [
                result.proposed_snapshot.cells[f"p1_t1_r{row}::p1_t1_c{column}"].text
                for column in range(1, len(fixture["columns"]) + 1)
            ]
            for row in range(1, len(fixture["rows"]) + 1)
        ]
        self.assertEqual(actual_rows, fixture["expected_rows"])
        self.assertEqual(result.proposed_snapshot.issues["misplaced_department"].status, "resolved")
        self.assertEqual(result.proposed_snapshot.issues["correct_age"].status, "dismissed")

    def test_redistribute_merged_cell_keeps_one_item_and_moves_one(self):
        snapshot = make_snapshot(
            [["Policy functions Financial", "", "22.5"]],
            column_names=("Policy functions", "Financial", "2009/10"),
        )
        source_id = "p1_t1_r1::p1_t1_c1"
        target_id = "p1_t1_r1::p1_t1_c2"
        for issue_id, issue_type, severity in (
            ("crosses", "item_crosses_boundary", "warning"),
            ("merged", "possible_merged_cell", "info"),
            ("assignment", "ambiguous_column_assignment", "warning"),
        ):
            snapshot.issues[issue_id] = IssueState(
                issue_id, issue_id, issue_type, severity, "p1_t1",
                [source_id], "active", issue_type, "repair",
            )
        client = MockClient([{
            "decisions": [{
                "cell_id": source_id,
                "needs_correction": True,
                "correction_action": "redistribute_cell",
                "confidence": 0.99,
                "reason": "Two fields were merged into the first cell.",
                "payload": {
                    "source_cell_id": source_id,
                    "assignments": [
                        {"target_cell_id": source_id, "text": "Policy functions"},
                        {"target_cell_id": target_id, "text": "Financial"},
                    ],
                },
            }]
        }])
        client.responses.append(client.responses[0])

        result = TableFixerPipeline(client).run_warnings(snapshot, auto_apply=True)

        self.assertEqual(len(result.decisions), 1)
        self.assertEqual(result.proposed_snapshot.cells[source_id].text, "Policy functions")
        self.assertEqual(result.proposed_snapshot.cells[target_id].text, "Financial")
        self.assertTrue(all(
            result.proposed_snapshot.issues[issue_id].status == "resolved"
            for issue_id in ("crosses", "merged", "assignment")
        ))
        context = client.calls[0]["context"]
        self.assertEqual(
            [column["header"] for column in context["columns"]],
            ["Policy functions", "Financial"],
        )

    def test_redistribute_merged_cell_can_move_all_items(self):
        snapshot = make_snapshot(
            [["Alice Engineering", "", ""]],
            column_names=("col_1", "Employee Name", "Department"),
        )
        source_id = "p1_t1_r1::p1_t1_c1"
        name_id = "p1_t1_r1::p1_t1_c2"
        department_id = "p1_t1_r1::p1_t1_c3"
        snapshot.issues["merged"] = IssueState(
            "merged", "merged", "item_crosses_boundary", "warning", "p1_t1",
            [source_id], "active", "Two values in placeholder column", "repair",
        )
        client = MockClient([{
            "decisions": [{
                "cell_id": source_id,
                "needs_correction": True,
                "correction_action": "redistribute_cell",
                "confidence": 0.99,
                "reason": "Placeholder column contains two merged values.",
                "payload": {
                    "assignments": [
                        {"target_cell_id": name_id, "text": "Alice"},
                        {"target_cell_id": department_id, "text": "Engineering"},
                    ],
                },
            }]
        }])

        result = TableFixerPipeline(client).run_warnings(snapshot, auto_apply=True)

        self.assertEqual(result.proposed_snapshot.cells[source_id].text, "")
        self.assertEqual(result.proposed_snapshot.cells[name_id].text, "Alice")
        self.assertEqual(result.proposed_snapshot.cells[department_id].text, "Engineering")
        self.assertEqual(client.calls[0]["context"]["columns"][0]["header"], "col_1")

    def test_redistribute_cell_rejects_token_loss_and_occupied_target_atomically(self):
        snapshot = make_snapshot(
            [["Alice Engineering", "occupied", ""]],
            column_names=("col_1", "Employee Name", "Department"),
        )
        source_id = "p1_t1_r1::p1_t1_c1"
        occupied_id = "p1_t1_r1::p1_t1_c2"
        snapshot.issues["warn"] = IssueState(
            "warn", "warn", "item_crosses_boundary", "warning", "p1_t1",
            [source_id], "active", "Merged and displaced", "repair",
        )
        client = MockClient([{
            "decisions": [{
                "cell_id": source_id,
                "needs_correction": True,
                "correction_action": "redistribute_cell",
                "confidence": 1.0,
                "reason": "Invalid plan",
                "payload": {
                    "assignments": [
                        {"target_cell_id": occupied_id, "text": "Alice"},
                    ],
                },
            }]
        }])
        client.responses.append(client.responses[0])

        result = TableFixerPipeline(client).run_warnings(snapshot, auto_apply=True)

        self.assertFalse(result.decisions[0].valid)
        self.assertIn(
            "Row repair must preserve every source token exactly once.",
            result.decisions[0].validation_errors,
        )
        self.assertEqual(result.proposed_snapshot.cells[source_id].text, "Alice Engineering")
        self.assertEqual(result.proposed_snapshot.cells[occupied_id].text, "occupied")
        self.assertEqual(result.proposed_snapshot.issues["warn"].status, "active")

    def test_warning_correction_normalizes_supported_payload_aliases(self):
        snapshot = make_snapshot([["Bad text", ""]])
        source_cell = "p1_t1_r1::p1_t1_c1"
        snapshot.issues["warn"] = IssueState(
            "warn", "warn", "item_crosses_boundary", "warning", "p1_t1",
            [source_cell], "active", "warning", "repair",
        )
        client = MockClient([{
            "decisions": [{
                "issue_id": "warn",
                "needs_correction": True,
                "correction_action": "split_cell_text",
                "confidence": 0.99,
                "reason": "replace text",
                "payload": {"cell_id": source_cell, "new_cell_text": "Correct text"},
            }]
        }])
        client.responses.append(client.responses[0])

        result = TableFixerPipeline(client).run_warnings(snapshot)

        self.assertEqual(result.proposed_snapshot.cells[source_cell].text, "Bad text")
        self.assertEqual(result.proposed_snapshot.issues["warn"].status, "active")
        self.assertFalse(result.decisions[0].valid)

    def test_omitted_warning_stays_active_without_manual_review(self):
        snapshot = make_snapshot([["Bob", "20"]])
        warning_cell = "p1_t1_r1::p1_t1_c1"
        snapshot.issues["warn"] = IssueState(
            "warn", "warn", "ambiguous_column_assignment", "warning", "p1_t1",
            [warning_cell], "active", "warning", "review",
        )

        result = TableFixerPipeline(MockClient([{"decisions": []}])).run_warnings(snapshot)

        self.assertEqual(result.proposed_snapshot.issues["warn"].status, "active")
        self.assertEqual(result.decisions[0].action, "invalid_warning_decision")
        self.assertFalse(result.decisions[0].valid)

    def test_warning_context_overflow_creates_multiple_batches(self):
        snapshot = make_snapshot([[f"row {index}", str(index)] for index in range(8)])
        for index in range(1, 9):
            issue_id = f"warn_{index}"
            cell_id = f"p1_t1_r{index}::p1_t1_c1"
            snapshot.issues[issue_id] = IssueState(
                issue_id, issue_id, "ambiguous_column_assignment", "warning", "p1_t1",
                [cell_id], "active", "warning " * 30, "review",
            )
        responses = [{"decisions": []} for _ in range(8)]
        client = MockClient(responses)

        TableFixerPipeline(client, context_token_budget=80).run_warnings(snapshot)

        self.assertEqual(len(client.calls), 8)
        self.assertTrue(all("columns" in call["context"] for call in client.calls))


if __name__ == "__main__":
    unittest.main()
