from __future__ import annotations

import unittest

from post_separation_merge.api import MergeRequest, process_merge_request


def fragment(
    table_id: str,
    page: int,
    index: int,
    header: list[str],
    rows: list[list[str]],
    *,
    start: bool,
    end: bool,
) -> dict:
    return {
        "table_id": table_id,
        "page_number": page,
        "table_index": index,
        "is_page_start": start,
        "is_page_end": end,
        "header": header,
        "sample_rows": rows,
    }


class MockClient:
    def __init__(self, result: dict):
        self.result = result
        self.calls: list[dict] = []

    def structured(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        return self.result


class MergeDecisionTests(unittest.TestCase):
    def test_identical_headers_merge_without_llm(self):
        request = MergeRequest(
            llm=True,
            tables=[
                fragment("p1_t2", 1, 2, ["Name", "Amount"], [["Alice", "10"]], start=False, end=True),
                fragment("p2_t1", 2, 1, [" name ", "AMOUNT"], [["Bob", "20"]], start=True, end=False),
            ],
        )
        client = MockClient({})

        response = process_merge_request(request, client)

        self.assertEqual(response.decisions[0].method, "deterministic")
        self.assertEqual(response.merge_groups[0].table_ids, ["p1_t2", "p2_t1"])
        self.assertEqual(client.calls, [])

    def test_llm_can_select_better_observed_header(self):
        request = MergeRequest(
            tables=[
                fragment("p1_t1", 1, 1, ["Tenant", "Am0unt"], [["Acme", "10"]], start=True, end=True),
                fragment("p2_t1", 2, 1, ["Tenant", "Amount"], [["Beta", "20"]], start=True, end=True),
            ]
        )
        client = MockClient(
            {
                "merge": True,
                "confidence": 0.95,
                "reason": "The same two columns and values continue.",
                "proposed_header": ["Tenant", "Amount"],
            }
        )

        response = process_merge_request(request, client)

        self.assertEqual(response.decisions[0].method, "llm")
        self.assertEqual(response.merge_groups[0].proposed_header, ["Tenant", "Amount"])
        self.assertIn("Previous-page header", client.calls[0]["prompt"])

    def test_llm_false_leaves_differing_headers_separate(self):
        request = MergeRequest(
            llm=False,
            tables=[
                fragment("p1_t1", 1, 1, ["Tenant", "Am0unt"], [], start=True, end=True),
                fragment("p2_t1", 2, 1, ["Tenant", "Amount"], [], start=True, end=True),
            ],
        )

        response = process_merge_request(request)

        self.assertFalse(response.decisions[0].merge)
        self.assertEqual(response.decisions[0].method, "llm_disabled")
        self.assertEqual(len(response.merge_groups), 2)

    def test_only_cross_page_end_to_start_is_considered(self):
        request = MergeRequest(
            llm=False,
            tables=[
                fragment("p1_t1", 1, 1, ["A"], [], start=True, end=False),
                fragment("p1_t2", 1, 2, ["A"], [], start=False, end=True),
                fragment("p2_t1", 2, 1, ["A"], [], start=True, end=False),
                fragment("p2_t2", 2, 2, ["A"], [], start=False, end=True),
            ],
        )

        response = process_merge_request(request)

        self.assertEqual(
            [(item.left_table_id, item.right_table_id) for item in response.decisions],
            [("p1_t2", "p2_t1")],
        )

    def test_invented_llm_header_is_rejected(self):
        request = MergeRequest(
            tables=[
                fragment("p1_t1", 1, 1, ["Tenant", "Am0unt"], [], start=True, end=True),
                fragment("p2_t1", 2, 1, ["Tenant", "Amount"], [], start=True, end=True),
            ]
        )
        client = MockClient(
            {
                "merge": True,
                "confidence": 0.9,
                "reason": "Continuation.",
                "proposed_header": ["Tenant", "Total"],
            }
        )

        with self.assertRaisesRegex(Exception, "corresponding input header"):
            process_merge_request(request, client)


if __name__ == "__main__":
    unittest.main()
