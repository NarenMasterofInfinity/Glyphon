"""Deterministic 2D table segmentation for Glyphon parser output."""

from .segmentation import (
    BoundaryDecision,
    SegmentationConfig,
    judge_uncertain_boundary,
    parser_pages_to_cells,
    segment_parser_pages,
    segment_tables,
)

__all__ = [
    "BoundaryDecision",
    "SegmentationConfig",
    "judge_uncertain_boundary",
    "parser_pages_to_cells",
    "segment_parser_pages",
    "segment_tables",
]
