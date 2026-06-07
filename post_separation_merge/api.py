from __future__ import annotations

import json
import re
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from post_separation_merge.ollama_client import OllamaClientError, OllamaLLMClient


DEFAULT_MODEL = "gemma3:4b"
DEFAULT_BASE_URL = "http://localhost:11434"

app = FastAPI(
    title="Glyphon Post-Separation Merge API",
    description="Decides which page-level table fragments belong to the same logical table.",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


class TableFragment(BaseModel):
    table_id: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    table_index: int = Field(ge=1)
    is_page_start: bool
    is_page_end: bool
    header: list[str] = Field(min_length=1)
    sample_rows: list[list[str]] = Field(default_factory=list, max_length=10)


class MergeRequest(BaseModel):
    tables: list[TableFragment] = Field(min_length=1)
    llm: bool = True
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL


class PairDecision(BaseModel):
    left_table_id: str
    right_table_id: str
    merge: bool
    method: Literal["deterministic", "llm", "llm_disabled", "incompatible"]
    confidence: float = Field(ge=0, le=1)
    reason: str
    proposed_header: list[str] | None = None


class MergeGroup(BaseModel):
    table_ids: list[str]
    proposed_header: list[str]


class MergeResponse(BaseModel):
    decisions: list[PairDecision]
    merge_groups: list[MergeGroup]


LLM_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "merge": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
        "proposed_header": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["merge", "confidence", "reason", "proposed_header"],
    "additionalProperties": False,
}


def normalized_header(header: list[str]) -> list[str]:
    return [re.sub(r"\s+", " ", cell).strip().casefold() for cell in header]


def exact_header_match(left: TableFragment, right: TableFragment) -> bool:
    normalized = normalized_header(left.header)
    return bool(any(normalized)) and normalized == normalized_header(right.header)


def boundary_pairs(tables: list[TableFragment]) -> list[tuple[TableFragment, TableFragment]]:
    ordered = sorted(tables, key=lambda table: (table.page_number, table.table_index))
    return [
        (left, right)
        for left, right in zip(ordered, ordered[1:])
        if left.is_page_end
        and right.is_page_start
        and right.page_number == left.page_number + 1
    ]


def llm_prompt(left: TableFragment, right: TableFragment) -> str:
    return "\n".join(
        [
            "Decide whether these are two consecutive fragments of the same table.",
            "Merge only when the columns and data meaning continue across the page break.",
            "If merging, choose one header value per column. Every chosen value must be copied exactly from either header below. Do not invent values.",
            "Return only the requested decision object.",
            "",
            f"Previous-page header: {json.dumps(left.header, ensure_ascii=True)}",
            f"Previous-page last rows: {json.dumps(left.sample_rows, ensure_ascii=True)}",
            f"Next-page header: {json.dumps(right.header, ensure_ascii=True)}",
            f"Next-page first rows: {json.dumps(right.sample_rows, ensure_ascii=True)}",
        ]
    )


def validate_llm_decision(
    result: dict[str, Any], left: TableFragment, right: TableFragment
) -> PairDecision:
    merge = result.get("merge")
    confidence = result.get("confidence")
    reason = result.get("reason")
    proposed = result.get("proposed_header")
    if not isinstance(merge, bool) or not isinstance(reason, str):
        raise OllamaClientError("Ollama decision has invalid merge or reason fields.")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
        raise OllamaClientError("Ollama decision has invalid confidence.")
    if not isinstance(proposed, list) or not all(isinstance(cell, str) for cell in proposed):
        raise OllamaClientError("Ollama decision has an invalid proposed_header.")

    if merge:
        allowed_by_column = [set(values) for values in zip(left.header, right.header)]
        if len(proposed) != len(left.header) or any(
            cell not in allowed_by_column[index] for index, cell in enumerate(proposed)
        ):
            raise OllamaClientError(
                "Ollama proposed_header must select each cell from the corresponding input header cells."
            )
    else:
        proposed = None

    return PairDecision(
        left_table_id=left.table_id,
        right_table_id=right.table_id,
        merge=merge,
        method="llm",
        confidence=float(confidence),
        reason=reason.strip(),
        proposed_header=proposed,
    )


def decide_pair(
    left: TableFragment,
    right: TableFragment,
    *,
    use_llm: bool,
    client: OllamaLLMClient | None,
) -> PairDecision:
    if len(left.header) != len(right.header):
        return PairDecision(
            left_table_id=left.table_id,
            right_table_id=right.table_id,
            merge=False,
            method="incompatible",
            confidence=1,
            reason="The fragments have different column counts.",
        )
    if exact_header_match(left, right):
        return PairDecision(
            left_table_id=left.table_id,
            right_table_id=right.table_id,
            merge=True,
            method="deterministic",
            confidence=1,
            reason="The normalized headers are identical.",
            proposed_header=left.header,
        )
    if not use_llm:
        return PairDecision(
            left_table_id=left.table_id,
            right_table_id=right.table_id,
            merge=False,
            method="llm_disabled",
            confidence=0,
            reason="Headers differ and LLM decisions are disabled.",
        )
    assert client is not None
    return validate_llm_decision(
        client.structured(prompt=llm_prompt(left, right), schema=LLM_DECISION_SCHEMA),
        left,
        right,
    )


def make_merge_groups(
    tables: list[TableFragment], decisions: list[PairDecision]
) -> list[MergeGroup]:
    merged_right = {decision.right_table_id for decision in decisions if decision.merge}
    decision_by_left = {decision.left_table_id: decision for decision in decisions if decision.merge}
    groups: list[MergeGroup] = []

    for table in sorted(tables, key=lambda item: (item.page_number, item.table_index)):
        if table.table_id in merged_right:
            continue
        ids = [table.table_id]
        header = table.header
        while ids[-1] in decision_by_left:
            decision = decision_by_left[ids[-1]]
            ids.append(decision.right_table_id)
            header = decision.proposed_header or header
        groups.append(MergeGroup(table_ids=ids, proposed_header=header))
    return groups


def process_merge_request(
    request: MergeRequest, client: OllamaLLMClient | None = None
) -> MergeResponse:
    ids = [table.table_id for table in request.tables]
    if len(ids) != len(set(ids)):
        raise ValueError("table_id values must be unique.")
    positions = [(table.page_number, table.table_index) for table in request.tables]
    if len(positions) != len(set(positions)):
        raise ValueError("Each page_number and table_index pair must be unique.")
    if request.llm and client is None:
        client = OllamaLLMClient(model=request.model, base_url=request.base_url)
    decisions = [
        decide_pair(left, right, use_llm=request.llm, client=client)
        for left, right in boundary_pairs(request.tables)
    ]
    return MergeResponse(decisions=decisions, merge_groups=make_merge_groups(request.tables, decisions))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "glyphon-post-separation-merge"}


@app.post("/merge-decisions", response_model=MergeResponse)
def merge_decisions(request: MergeRequest) -> MergeResponse:
    try:
        return process_merge_request(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OllamaClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
