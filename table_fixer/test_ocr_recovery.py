from __future__ import annotations

from types import SimpleNamespace
import unittest

from PIL import Image

from table_fixer.models import CellState, LogicalTable, PipelineSnapshot, RowState
from table_fixer.ocr_recovery import LenientOCRItem, recover_missed_glyphs, recover_missed_glyphs_with_result


def make_source(rows: list[list[str]], centers: list[float]) -> tuple[PipelineSnapshot, list[SimpleNamespace]]:
    table_id = "p1_t1"
    column_ids = [f"{table_id}_c{index}" for index in range(1, len(centers) + 1)]
    table = LogicalTable(table_id, 1, 1, column_ids, dict(zip(column_ids, [f"col_{i}" for i in range(1, len(centers) + 1)])), [])
    row_states = {}
    cells = {}
    raw_items = []
    for row_number, values in enumerate(rows, start=1):
        row_id = f"{table_id}_r{row_number}"
        table.row_ids.append(row_id)
        row_states[row_id] = RowState(row_id, 1, row_number, table_id, ancestor_row_ids=[row_id])
        y0 = 10.0 + ((row_number - 1) * 20.0)
        for index, (value, center) in enumerate(zip(values, centers), start=1):
            column_id = column_ids[index - 1]
            cell_id = f"{row_id}::{column_id}"
            bbox = (center - 4, y0, center + 4, y0 + 8) if value else None
            cells[cell_id] = CellState(cell_id, row_id, column_id, value, bbox, ancestor_cell_ids=[cell_id])
            if bbox:
                raw_items.append(SimpleNamespace(text=value, x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3]))
    snapshot = PipelineSnapshot(
        "source", "source", {table_id: table}, row_states, cells, {},
        page_dimensions={1: (220.0, 100.0)},
    )
    return snapshot, [SimpleNamespace(page_number=1, raw_items=raw_items)]


def detector_for(items):
    return lambda _image, _page_number: items


def item(text, confidence, bbox):
    return LenientOCRItem(1, text, confidence, bbox)


class OCRRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.image = {1: Image.new("RGB", (1300, 600), "white")}

    def recover(self, source, pages, items):
        return recover_missed_glyphs_with_result(
            b"",
            pages,
            source,
            lenient_detector=detector_for(items),
            page_images=self.image,
        )

    def test_golden_detected_text_is_never_recovered_again(self):
        source, pages = make_source([["A", ""], ["B", "x"], ["C", "y"]], [20.0, 180.0])
        recovery = self.recover(source, pages, [
            item("A", 0.99, (16.0, 10.0, 24.0, 18.0)),
            item("q", 0.90, (176.0, 10.0, 184.0, 18.0)),
        ])
        self.assertEqual(recovery.snapshot.cells["p1_t1_r1::p1_t1_c1"].text, "A")
        self.assertEqual(recovery.snapshot.cells["p1_t1_r1::p1_t1_c2"].text, "q")
        self.assertEqual(len(recovery.candidates), 1)

    def test_lenient_box_partially_covered_by_golden_text_is_removed(self):
        source, pages = make_source([["A", ""], ["B", "x"], ["C", "y"]], [20.0, 180.0])
        recovery = self.recover(source, pages, [
            item("duplicate", 0.99, (22.0, 10.0, 60.0, 18.0)),
        ])
        self.assertEqual(recovery.candidates, [])
        self.assertEqual(recovery.snapshot.cells["p1_t1_r1::p1_t1_c1"].text, "A")

    def test_strongest_overlapping_lenient_item_wins(self):
        source, pages = make_source([["A", ""], ["B", "x"], ["C", "y"]], [20.0, 180.0])
        recovery = self.recover(source, pages, [
            item("l", 0.61, (176.0, 10.0, 184.0, 18.0)),
            item("i", 0.94, (175.5, 9.5, 184.5, 18.5)),
            item("1", 0.72, (176.2, 10.2, 183.8, 17.8)),
        ])
        self.assertEqual(recovery.snapshot.cells["p1_t1_r1::p1_t1_c2"].text, "i")
        self.assertEqual(len(recovery.candidates), 1)
        self.assertEqual(recovery.candidates[0].confidence, 0.94)

    def test_weak_lenient_items_are_not_recovered(self):
        source, pages = make_source([["A", ""], ["B", "x"], ["C", "y"]], [20.0, 180.0])
        recovery = self.recover(source, pages, [item("q", 0.40, (176.0, 10.0, 184.0, 18.0))])
        self.assertEqual(recovery.snapshot.cells["p1_t1_r1::p1_t1_c2"].text, "")
        self.assertEqual(recovery.candidates, [])

    def test_highest_confidence_item_wins_inside_empty_cell(self):
        source, pages = make_source([["A", ""], ["B", "x"], ["C", "y"]], [20.0, 180.0])
        recovery = self.recover(source, pages, [
            item("q", 0.80, (150.0, 10.0, 156.0, 18.0)),
            item("r", 0.91, (180.0, 10.0, 186.0, 18.0)),
        ])
        self.assertEqual(recovery.snapshot.cells["p1_t1_r1::p1_t1_c2"].text, "r")

    def test_inserts_supported_missing_column_without_changing_existing_text(self):
        source, pages = make_source([["A", "D"], ["B", "E"], ["C", "F"]], [20.0, 180.0])
        before = {cell_id: cell.text for cell_id, cell in source.cells.items()}
        recovery = self.recover(source, pages, [
            item("x", 0.90, (96.0, 10.0, 104.0, 18.0)),
            item("y", 0.91, (96.0, 30.0, 104.0, 38.0)),
            item("z", 0.92, (96.0, 50.0, 104.0, 58.0)),
        ])
        table = recovery.snapshot.tables["p1_t1"]
        new_column = next(column_id for column_id in table.column_ids if "_ocr_" in column_id)
        self.assertEqual(table.column_ids.index(new_column), 1)
        self.assertEqual(
            [recovery.snapshot.cells[f"p1_t1_r{i}::{new_column}"].text for i in range(1, 4)],
            ["x", "y", "z"],
        )
        self.assertEqual({cell_id: recovery.snapshot.cells[cell_id].text for cell_id in before}, before)

    def test_new_column_uses_strongest_overlapping_item_per_row(self):
        source, pages = make_source([["A", "D"], ["B", "E"], ["C", "F"]], [20.0, 180.0])
        recovery = self.recover(source, pages, [
            item("l", 0.60, (96.0, 10.0, 104.0, 18.0)),
            item("1", 0.93, (95.5, 9.5, 104.5, 18.5)),
            item("2", 0.94, (96.0, 30.0, 104.0, 38.0)),
            item("3", 0.95, (96.0, 50.0, 104.0, 58.0)),
        ])
        new_column = next(column_id for column_id in recovery.snapshot.tables["p1_t1"].column_ids if "_ocr_" in column_id)
        self.assertEqual(recovery.snapshot.cells[f"p1_t1_r1::{new_column}"].text, "1")

    def test_public_api_returns_snapshot(self):
        source, pages = make_source([["A", ""], ["B", "x"], ["C", "y"]], [20.0, 180.0])
        snapshot = recover_missed_glyphs(
            b"", pages, source,
            lenient_detector=detector_for([item("q", 0.90, (176.0, 10.0, 184.0, 18.0))]),
            page_images=self.image,
        )
        self.assertIsInstance(snapshot, PipelineSnapshot)


if __name__ == "__main__":
    unittest.main()
