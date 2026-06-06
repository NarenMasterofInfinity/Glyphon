import unittest

from scanned_parser import extract_table_scanned
from text_pipeline import BBoxItem


def box(text, x0, y0, x1, y1, confidence=0.99):
    return BBoxItem(text, confidence, x0, y0, x1, y1, 1, [])


class ExtractionDiagnosticsTests(unittest.TestCase):
    def test_shifted_layout_region_becomes_separate_table(self):
        items = [
            box("A1", 10, 10, 30, 20), box("B1", 110, 10, 130, 20),
            box("A2", 10, 25, 30, 35), box("B2", 110, 25, 130, 35),
            box("C1", 110, 100, 130, 110), box("D1", 210, 100, 230, 110),
            box("C2", 110, 115, 130, 125), box("D2", 210, 115, 230, 125),
        ]

        result = extract_table_scanned(items, 300)

        self.assertEqual(result["row_table_indexes"], [1, 1, 2, 2])
        self.assertIn("incompatible_layout_regions", {issue["issue_type"] for issue in result["issues"]})

    def test_weak_single_row_boundary_is_ignored_and_explained(self):
        result = extract_table_scanned(
            [box("left", 10, 10, 70, 20), box("right", 85, 10, 145, 20)],
            180,
        )

        self.assertEqual(result["aligned_rows"], [["left right"]])
        self.assertFalse(result["boundary_metadata"][0]["accepted"])
        self.assertIn("weak_column_boundary", {issue["issue_type"] for issue in result["issues"]})

    def test_boundary_crossing_value_is_canonical_once_with_alternative(self):
        items = [
            box("L1", 10, 10, 40, 20), box("R1", 160, 10, 190, 20),
            box("L2", 10, 25, 40, 35), box("R2", 160, 25, 190, 35),
            box("wide", 80, 40, 120, 50),
        ]

        result = extract_table_scanned(items, 220)

        self.assertEqual(sum(cell.count("wide") for row in result["aligned_rows"] for cell in row), 1)
        wide_assignment = next(item for item in result["assignments"] if item["text"] == "wide")
        self.assertTrue(wide_assignment["alternatives"])
        self.assertIn("item_crosses_boundary", {issue["issue_type"] for issue in result["issues"]})
        self.assertIn("possible_merged_cell", {issue["issue_type"] for issue in result["issues"]})

    def test_occupied_cell_does_not_create_column_and_reports_collision(self):
        result = extract_table_scanned(
            [box("left", 10, 10, 70, 20), box("right", 100, 10, 160, 20)],
            190,
        )

        self.assertEqual(len(result["aligned_rows"][0]), 1)
        self.assertEqual(result["aligned_rows"][0][0], "left right")
        self.assertIn("possible_cell_collision", {issue["issue_type"] for issue in result["issues"]})

    def test_ambiguous_row_and_low_ocr_are_explained(self):
        items = [
            box("top", 10, 8, 30, 12),
            box("tall", 50, 0, 70, 30, confidence=0.5),
            box("bottom", 10, 22, 30, 26),
        ]

        result = extract_table_scanned(items, 100)
        issue_types = {issue["issue_type"] for issue in result["issues"]}

        self.assertIn("ambiguous_row_assignment", issue_types)
        self.assertIn("low_ocr_confidence", issue_types)
        source_indexes = set(range(len(items)))
        for issue in result["issues"]:
            self.assertTrue(set(issue["source_item_indexes"]).issubset(source_indexes))
            self.assertIn("explanation", issue)
            self.assertIn("suggested_action", issue)
            self.assertIn("chosen_placement", issue)
            self.assertIn("alternatives", issue)
            self.assertIn("evidence", issue)

    def test_sparse_complex_table_keeps_rightmost_values_in_rightmost_column(self):
        items = [
            box("group", 10, 10, 50, 20), box("name", 100, 10, 140, 20), box("10", 190, 10, 210, 20),
            box("group2", 10, 25, 50, 35), box("name2", 100, 25, 140, 35), box("20", 190, 25, 210, 35),
            box("30", 190, 40, 210, 50),
        ]

        result = extract_table_scanned(items, 240)

        self.assertEqual(result["aligned_rows"][2], ["", "", "30"])
        self.assertEqual(result["row_table_indexes"], [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
