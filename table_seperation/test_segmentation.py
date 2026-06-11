from __future__ import annotations

from dataclasses import dataclass
import unittest

from table_seperation import parser_pages_to_cells, segment_tables
from table_seperation.sample_scenarios import scenarios


def cell(text, page, x, y, width=35, height=10):
    return {"text": text, "page": page, "x1": x, "y1": y, "x2": x + width, "y2": y + height}


def grid(page, x, y, rows, columns=3, dx=50, dy=15, prefix=""):
    return [
        cell(f"{prefix}{row}-{column}", page, x + column * dx, y + row * dy)
        for row in range(rows)
        for column in range(columns)
    ]


class SegmentationTests(unittest.TestCase):
    def test_realistic_spatial_scenarios(self):
        for name, scenario in scenarios().items():
            with self.subTest(name=name):
                self.assertEqual(len(segment_tables(scenario["cells"])), scenario["expected_tables"])

    def test_one_over_two_with_clear_lower_table_gap(self):
        top = grid(1, 10, 10, 3, columns=6, dx=55, prefix="T")
        lower_left = grid(1, 10, 100, 3, columns=2, dx=55, prefix="L")
        lower_right = grid(1, 210, 100, 3, columns=3, dx=55, prefix="R")
        self.assertEqual(len(segment_tables(top + lower_left + lower_right)), 3)

    def test_two_side_by_side_tables(self):
        cells = grid(1, 10, 10, 4, prefix="L") + grid(1, 230, 10, 4, prefix="R")
        self.assertEqual(len(segment_tables(cells)), 2)

    def test_two_tables_one_below_another_with_new_header_and_grid(self):
        top = grid(1, 10, 10, 4, prefix="T")
        bottom = grid(1, 35, 130, 4, columns=2, dx=70, prefix="B")
        self.assertEqual(len(segment_tables(top + bottom)), 2)

    def test_large_gap_and_fresh_header_split_matching_grids(self):
        header = [cell("Name", 1, 10, 10), cell("Amount", 1, 60, 10)]
        top = header + [cell("A", 1, 10, 25), cell("10", 1, 60, 25)]
        bottom = [cell("Item", 1, 10, 120), cell("Amount", 1, 60, 120)]
        bottom += [cell("B", 1, 10, 135), cell("20", 1, 60, 135)]
        self.assertEqual(len(segment_tables(top + bottom)), 2)

    def test_sparse_section_row_stays_in_table(self):
        cells = grid(1, 10, 10, 2) + [cell("Section A", 1, 10, 40, width=90)] + grid(1, 10, 55, 2)
        tables = segment_tables(cells)
        self.assertEqual(len(tables), 1)
        self.assertIn("Section A", [item["text"] for item in tables[0]["cells"]])

    def test_table_split_across_pages_merges(self):
        cells = grid(1, 10, 700, 3, prefix="A") + grid(2, 10, 20, 3, prefix="B")
        tables = segment_tables(cells)
        self.assertEqual(len(tables), 1)
        self.assertEqual((tables[0]["page_start"], tables[0]["page_end"]), (1, 2))

    def test_note_between_continued_rows_stays_with_table(self):
        cells = grid(1, 10, 10, 2) + [cell("Note: provisional", 1, 10, 50, width=120)] + grid(1, 10, 75, 2)
        tables = segment_tables(cells)
        self.assertEqual(len(tables), 1)

    def test_visually_close_logically_different_tables(self):
        left = grid(1, 10, 10, 3, columns=2, dx=45, prefix="L")
        right = grid(1, 125, 10, 3, columns=2, dx=45, prefix="R")
        self.assertEqual(len(segment_tables(left + right)), 2)

    def test_existing_parser_adapter_prefers_raw_items(self):
        @dataclass
        class Item:
            text: str
            x0: float
            y0: float
            x1: float
            y1: float

        @dataclass
        class Page:
            page_number: int
            raw_items: list[Item]
            cells: list

        adapted = parser_pages_to_cells([Page(3, [Item("A", 1, 2, 3, 4)], [])])
        self.assertEqual(adapted, [{"text": "A", "page": 3, "x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0}])


if __name__ == "__main__":
    unittest.main()
