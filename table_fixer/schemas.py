from __future__ import annotations

CONFIDENCE = {"type": "number", "minimum": 0, "maximum": 1}

METADATA_SCHEMA = {
    "type": "object",
    "properties": {
        "metadata_row_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
        "header_continuation_row_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
        "confidence": CONFIDENCE,
        "reasons": {"type": "object"},
    },
    "required": ["metadata_row_ids", "header_continuation_row_ids", "confidence", "reasons"],
}

HEADER_SCHEMA = {
    "type": "object",
    "properties": {
        "is_header": {"type": "boolean"},
        "continuation_row_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
        "confidence": CONFIDENCE,
        "reason": {"type": "string"},
    },
    "required": ["is_header", "continuation_row_ids", "confidence", "reason"],
}

TABLE_RECONCILIATION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["merge", "keep_separate"]},
        "confidence": CONFIDENCE,
        "reason": {"type": "string"},
    },
    "required": ["action", "confidence", "reason"],
}

COLUMN_SPLIT_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["split", "no_split"]},
        "confidence": CONFIDENCE,
        "reason": {"type": "string"},
        "regex": {"type": "string"},
        "new_headers": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 4},
        "expected": {"type": "object"},
    },
    "required": ["action", "confidence", "reason", "regex", "new_headers", "expected"],
}

WARNING_BATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "decisions": {
            "type": "array",
            "maxItems": 12,
            "items": {
                "type": "object",
                "properties": {
                    "issue_id": {"type": "string"},
                    "action": {
                        "type": "string",
                        "enum": [
                            "no_issue",
                            "move_cell",
                            "merge_adjacent_cells",
                            "split_cell_text",
                            "mark_for_review",
                        ],
                    },
                    "confidence": CONFIDENCE,
                    "reason": {"type": "string"},
                    "payload": {"type": "object"},
                },
                "required": ["issue_id", "action", "confidence", "reason", "payload"],
            },
        }
    },
    "required": ["decisions"],
}
