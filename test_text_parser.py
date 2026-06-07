import unittest
from unittest.mock import MagicMock, patch

from text_parser import _extract_page_words, parse_pdf_pages


class TextParserTests(unittest.TestCase):
    def test_extract_page_words_preserves_native_bbox(self):
        page = MagicMock()
        page.get_text.return_value = [(10, 20, 40, 32, "value", 0, 0, 0)]

        items = _extract_page_words(page, 3)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].text, "value")
        self.assertEqual((items[0].x0, items[0].y0, items[0].x1, items[0].y1), (10, 20, 40, 32))
        self.assertEqual(items[0].page_number, 3)
        self.assertEqual(items[0].confidence, 1.0)

    @patch("text_parser.fitz.open")
    @patch("text_parser.extract_table_scanned")
    def test_parse_pdf_pages_returns_existing_parser_contract(self, extract_table, open_document):
        page = MagicMock()
        page.rect.width = 600
        page.rect.height = 800
        page.get_text.return_value = [(10, 20, 40, 32, "value", 0, 0, 0)]
        document = MagicMock()
        document.__len__.return_value = 1
        document.__getitem__.return_value = page
        open_document.return_value = document
        extract_table.return_value = {
            "column_names": ["col_1"],
            "aligned_rows": [["value"]],
            "slant_angle": 0.0,
            "column_centers": [[25.0]],
            "band": (0, 0),
            "cells": [{
                "row_index": 1, "col_index": 1, "text": "value",
                "x0": 10.0, "y0": 20.0, "x1": 40.0, "y1": 32.0,
                "source_item_indexes": [0], "table_index": 1,
                "layout_region_index": 1, "assignment_score": 1.0,
                "alternatives": [], "issue_ids": ["p1_issue_1"],
            }],
            "row_table_indexes": [1],
            "row_layout_region_indexes": [1],
            "boundary_metadata": [],
            "assignments": [{"text": "value"}],
            "issues": [{
                "issue_id": "p1_issue_1", "issue_type": "example",
                "severity": "info", "explanation": "example",
                "suggested_action": "review",
            }],
        }

        results = parse_pdf_pages(b"pdf", page_numbers=[1, 2])

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].cells[0].issue_ids, ["p1_issue_1"])
        self.assertEqual(results[0].issues[0]["severity"], "info")
        self.assertEqual(results[0].cells[0].x0, 10.0)
        document.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
