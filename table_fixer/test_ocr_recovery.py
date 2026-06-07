from __future__ import annotations

from types import SimpleNamespace
import unittest

from PIL import Image

from table_fixer.models import CellState, LogicalTable, PipelineSnapshot, RowState
from table_fixer.ocr_recovery import GlyphComponent, recover_missed_glyphs, recover_missed_glyphs_with_result


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
                raw_items.append(SimpleNamespace(x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3]))
    snapshot = PipelineSnapshot(
        "source",
        "source",
        {table_id: table},
        row_states,
        cells,
        {},
        page_dimensions={1: (220.0, 100.0)},
    )
    page = SimpleNamespace(page_number=1, raw_items=raw_items)
    return snapshot, [page]


def detector_for(components):
    return lambda _image, _page_number: components


class OCRRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.image = {1: Image.new("RGB", (1300, 600), "white")}

    def test_recovers_single_letter_only_into_empty_existing_cell(self):
        source, pages = make_source([["A", ""], ["B", "x"], ["C", "y"]], [20.0, 180.0])
        component = GlyphComponent(1, (176.0, 10.0, 184.0, 18.0), 20.0)

        recovery = recover_missed_glyphs_with_result(
            b"",
            pages,
            source,
            recognizer=lambda _crop: ("q", 0.93),
            component_detector=detector_for([component]),
            page_images=self.image,
        )

        target = "p1_t1_r1::p1_t1_c2"
        self.assertEqual(recovery.snapshot.cells[target].text, "q")
        self.assertEqual(recovery.recovered_cell_count, 1)
        self.assertEqual(recovery.recovered_column_count, 0)
        self.assertEqual(source.cells[target].text, "")
        self.assertTrue(recovery.snapshot.decisions[0].applied)

    def test_inserts_supported_missing_column_without_disturbing_existing_cells(self):
        source, pages = make_source([["A", "D"], ["B", "E"], ["C", "F"]], [20.0, 180.0])
        components = [
            GlyphComponent(1, (96.0, 10.0 + offset, 104.0, 18.0 + offset), 20.0)
            for offset in (0.0, 20.0, 40.0)
        ]
        before_text = {cell_id: cell.text for cell_id, cell in source.cells.items()}
        before_columns = list(source.tables["p1_t1"].column_ids)

        recovery = recover_missed_glyphs_with_result(
            b"",
            pages,
            source,
            recognizer=lambda _crop: ("z", 0.95),
            component_detector=detector_for(components),
            page_images=self.image,
        )

        table = recovery.snapshot.tables["p1_t1"]
        new_column = next(column_id for column_id in table.column_ids if column_id not in before_columns)
        self.assertEqual(table.column_ids.index(new_column), 1)
        self.assertEqual([recovery.snapshot.cells[f"p1_t1_r{i}::{new_column}"].text for i in range(1, 4)], ["z", "z", "z"])
        self.assertEqual({cell_id: recovery.snapshot.cells[cell_id].text for cell_id in before_text}, before_text)
        self.assertEqual(recovery.recovered_column_count, 1)

    def test_new_column_leaves_unsupported_rows_blank(self):
        source, pages = make_source([["A", "E"], ["B", "F"], ["C", "G"], ["D", "H"]], [20.0, 180.0])
        components = [
            GlyphComponent(1, (96.0, 10.0 + offset, 104.0, 18.0 + offset), 20.0)
            for offset in (0.0, 20.0, 40.0)
        ]
        recovery = recover_missed_glyphs_with_result(
            b"",
            pages,
            source,
            recognizer=lambda _crop: ("m", 0.96),
            component_detector=detector_for(components),
            page_images=self.image,
        )
        new_column = next(column_id for column_id in recovery.snapshot.tables["p1_t1"].column_ids if "_ocr_" in column_id)
        self.assertEqual(recovery.snapshot.cells[f"p1_t1_r4::{new_column}"].text, "")

    def test_rejects_disagreement_and_low_confidence(self):
        source, pages = make_source([["A", ""], ["B", "x"], ["C", "y"]], [20.0, 180.0])
        component = GlyphComponent(1, (176.0, 10.0, 184.0, 18.0), 20.0)
        answers = iter([("a", 0.99), ("b", 0.99)])
        recovery = recover_missed_glyphs_with_result(
            b"",
            pages,
            source,
            recognizer=lambda _crop: next(answers),
            component_detector=detector_for([component]),
            page_images=self.image,
        )
        self.assertEqual(recovery.snapshot.cells["p1_t1_r1::p1_t1_c2"].text, "")
        self.assertEqual(recovery.candidates[0].rejection_reason, "recognition_variants_disagree")

        source, pages = make_source([["A", ""], ["B", "x"], ["C", "y"]], [20.0, 180.0])
        low_confidence = recover_missed_glyphs_with_result(
            b"",
            pages,
            source,
            recognizer=lambda _crop: ("a", 0.80),
            component_detector=detector_for([component]),
            page_images=self.image,
        )
        self.assertEqual(low_confidence.candidates[0].rejection_reason, "recognition_confidence_below_threshold")

    def test_rejects_component_overlapping_existing_ocr_item(self):
        source, pages = make_source([["A", ""], ["B", "x"], ["C", "y"]], [20.0, 180.0])
        component = GlyphComponent(1, (16.0, 10.0, 24.0, 18.0), 20.0)
        recovery = recover_missed_glyphs_with_result(
            b"",
            pages,
            source,
            recognizer=lambda _crop: ("a", 0.99),
            component_detector=detector_for([component]),
            page_images=self.image,
        )
        self.assertEqual(recovery.candidates, [])
        self.assertEqual(recovery.snapshot.recovery_audits, [])

    def test_excludes_tiny_partial_overlap_with_presented_item(self):
        source, pages = make_source([["A", "D"], ["B", "E"], ["C", "F"]], [20.0, 180.0])
        # Only a small edge touches the existing A bbox. It must not become new-column evidence.
        components = [
            GlyphComponent(1, (23.9, 10.0, 30.0, 18.0), 12.0),
            GlyphComponent(1, (23.9, 30.0, 30.0, 38.0), 12.0),
            GlyphComponent(1, (23.9, 50.0, 30.0, 58.0), 12.0),
        ]
        recovery = recover_missed_glyphs_with_result(
            b"",
            pages,
            source,
            recognizer=lambda _crop: ("i", 0.99),
            component_detector=detector_for(components),
            page_images=self.image,
        )
        self.assertEqual(recovery.recovered_column_count, 0)
        self.assertEqual(recovery.candidates, [])
        self.assertEqual(recovery.snapshot.tables["p1_t1"].column_ids, source.tables["p1_t1"].column_ids)

    def test_recovers_leftmost_and_rightmost_missing_columns(self):
        for x_position, expected_position in ((5.0, 0), (195.0, 2)):
            with self.subTest(x_position=x_position):
                source, pages = make_source([["A", "D"], ["B", "E"], ["C", "F"]], [20.0, 180.0])
                components = [
                    GlyphComponent(1, (x_position - 3, 10.0 + offset, x_position + 3, 18.0 + offset), 18.0)
                    for offset in (0.0, 20.0, 40.0)
                ]
                recovery = recover_missed_glyphs_with_result(
                    b"",
                    pages,
                    source,
                    recognizer=lambda _crop: ("r", 0.97),
                    component_detector=detector_for(components),
                    page_images=self.image,
                )
                new_column = next(column_id for column_id in recovery.snapshot.tables["p1_t1"].column_ids if "_ocr_" in column_id)
                self.assertEqual(recovery.snapshot.tables["p1_t1"].column_ids.index(new_column), expected_position)

    def test_workspace_persists_recovery_audit(self):
        import importlib.util

        if importlib.util.find_spec("pandas") is None:
            self.skipTest("Project persistence dependencies are not installed.")
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from table_fixer.workspace import persist_snapshot

        source, pages = make_source([["A", ""], ["B", "x"], ["C", "y"]], [20.0, 180.0])
        recovery = recover_missed_glyphs_with_result(
            b"",
            pages,
            source,
            recognizer=lambda _crop: ("q", 0.93),
            component_detector=detector_for([GlyphComponent(1, (176.0, 10.0, 184.0, 18.0), 20.0)]),
            page_images=self.image,
        )
        with TemporaryDirectory() as directory:
            persist_snapshot(Path(directory), "source", recovery.snapshot)
            self.assertTrue((Path(directory) / "source" / "ocr_recovery.json").exists())

    def test_public_api_returns_snapshot(self):
        source, pages = make_source([["A", ""], ["B", "x"], ["C", "y"]], [20.0, 180.0])
        snapshot = recover_missed_glyphs(
            b"",
            pages,
            source,
            recognizer=lambda _crop: ("q", 0.93),
            component_detector=detector_for([GlyphComponent(1, (176.0, 10.0, 184.0, 18.0), 20.0)]),
            page_images=self.image,
        )
        self.assertIsInstance(snapshot, PipelineSnapshot)


if __name__ == "__main__":
    unittest.main()
