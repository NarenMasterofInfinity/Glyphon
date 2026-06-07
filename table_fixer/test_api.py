from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import fitz

from table_fixer import api
from table_fixer.context import ensure_profiles
from table_fixer.models import CellState, LogicalTable, PipelineSnapshot, RowState
from table_fixer.workspace import persist_snapshot


def make_snapshot(phase: str = "source") -> PipelineSnapshot:
    table_id = "p1_t1"
    column_ids = [f"{table_id}_c1", f"{table_id}_c2"]
    row_id = f"{table_id}_r1"
    table = LogicalTable(
        table_id=table_id,
        page_number=1,
        source_table_index=1,
        column_ids=column_ids,
        column_names={column_ids[0]: "Name", column_ids[1]: "Amount"},
        row_ids=[row_id],
        ancestor_table_ids=[table_id],
    )
    rows = {row_id: RowState(row_id, 1, 1, table_id, ancestor_row_ids=[row_id])}
    cells = {
        f"{row_id}::{column_ids[0]}": CellState(
            f"{row_id}::{column_ids[0]}",
            row_id,
            column_ids[0],
            "Alice",
            (10, 10, 50, 20),
            ancestor_cell_ids=[f"{row_id}::{column_ids[0]}"],
        ),
        f"{row_id}::{column_ids[1]}": CellState(
            f"{row_id}::{column_ids[1]}",
            row_id,
            column_ids[1],
            "100",
            (60, 10, 90, 20),
            ancestor_cell_ids=[f"{row_id}::{column_ids[1]}"],
        ),
    }
    snapshot = PipelineSnapshot(
        snapshot_id=f"snapshot_{phase}",
        phase=phase,
        tables={table_id: table},
        rows=rows,
        cells=cells,
        issues={},
        page_dimensions={1: (200, 200)},
    )
    ensure_profiles(snapshot)
    return snapshot


def minimal_pdf_bytes() -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "native text")
    doc.new_page()
    payload = doc.tobytes()
    doc.close()
    return payload


class TableFixerApiTests(TestCase):
    def test_manual_action_invalidates_downstream_accepted_phases(self):
        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            for phase in ["source", "reconciliation", "metadata", "headers", "columns", "warnings"]:
                persist_snapshot(workspace, phase, make_snapshot(phase))
            state = api.new_api_state()
            state["accepted_labels"] = {
                "source": "source",
                "reconciliation": "reconciliation",
                "metadata": "metadata",
                "headers": "headers",
                "columns": "columns",
                "warnings": "warnings",
            }

            request = api.ManualActionRequest(
                base_phase="metadata",
                actions=[
                    {
                        "type": "edit_cell_text",
                        "cell_id": "p1_t1_r1::p1_t1_c1",
                        "text": "Bob",
                    }
                ],
            )
            api.apply_manual_actions(workspace, state, request)

            self.assertIn("metadata", state["accepted_labels"])
            self.assertNotIn("headers", state["accepted_labels"])
            self.assertNotIn("columns", state["accepted_labels"])
            self.assertNotIn("warnings", state["accepted_labels"])
            self.assertEqual(state["invalidated_phases"], ["headers", "columns", "warnings"])
            edited = api.load_snapshot(workspace, state["accepted_labels"]["metadata"])
            self.assertEqual(edited.cells["p1_t1_r1::p1_t1_c1"].text, "Bob")

    def test_undo_and_redo_restore_manual_snapshot_labels_and_keep_downstream_stale(self):
        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            persist_snapshot(workspace, "source", make_snapshot("source"))
            persist_snapshot(workspace, "metadata", make_snapshot("metadata"))
            persist_snapshot(workspace, "headers", make_snapshot("headers"))
            state = api.new_api_state()
            state["accepted_labels"] = {"source": "source", "metadata": "metadata", "headers": "headers"}
            entry = api.apply_manual_actions(
                workspace,
                state,
                api.ManualActionRequest(
                    base_phase="metadata",
                    actions=[{"type": "edit_cell_text", "cell_id": "p1_t1_r1::p1_t1_c1", "text": "Bob"}],
                ),
            )

            api.restore_manual_history_state(state, entry, "before_accepted_labels")
            self.assertEqual(state["accepted_labels"]["metadata"], "metadata")
            self.assertNotIn("headers", state["accepted_labels"])
            self.assertEqual(state["invalidated_phases"], ["headers", "columns", "warnings"])

            api.restore_manual_history_state(state, entry, "after_accepted_labels")
            self.assertTrue(state["accepted_labels"]["metadata"].startswith("metadata_manual_"))
            self.assertNotIn("headers", state["accepted_labels"])

    def test_extraction_mode_dispatches_to_text_or_ocr_parser(self):
        with patch.object(api, "parse_text_pdf_pages", return_value=["text"]) as text_parser:
            result = api.parse_pdf_pages_by_mode(b"unused", extraction_mode="text", page_numbers=[1])
            self.assertEqual(result, ["text"])
            text_parser.assert_called_once()

        with patch.object(api, "parse_ocr_pdf_pages", return_value=["ocr"]) as ocr_parser:
            result = api.parse_pdf_pages_by_mode(b"unused", extraction_mode="ocr", page_numbers=[1])
            self.assertEqual(result, ["ocr"])
            ocr_parser.assert_called_once()

    def test_auto_extraction_splits_native_text_and_ocr_pages(self):
        pdf_bytes = minimal_pdf_bytes()
        text_page = SimpleNamespace(page_number=1)
        ocr_page = SimpleNamespace(page_number=2)
        with (
            patch.object(api, "page_has_native_text", side_effect=lambda _pdf, page: page == 1),
            patch.object(api, "parse_text_pdf_pages", return_value=[text_page]) as text_parser,
            patch.object(api, "parse_ocr_pdf_pages", return_value=[ocr_page]) as ocr_parser,
        ):
            result = api.parse_pdf_pages_by_mode(pdf_bytes, extraction_mode="auto", page_numbers=[1, 2])

        self.assertEqual([page.page_number for page in result], [1, 2])
        text_parser.assert_called_once_with(pdf_bytes, page_numbers=[1])
        ocr_parser.assert_called_once_with(pdf_bytes, page_numbers=[2])


if __name__ == "__main__":
    import unittest

    unittest.main()
