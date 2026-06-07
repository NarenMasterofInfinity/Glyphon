from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import requests


PHASES = ["reconciliation", "metadata", "headers", "columns", "warnings"]


class ApiError(RuntimeError):
    pass


@dataclass
class ApiConfig:
    table_fixer_url: str = "http://localhost:8770"
    merge_url: str = "http://localhost:8780"
    model: str = "gemma3:4b"
    base_url: str = "http://localhost:11434"
    context_token_budget: int = 3500
    auto_apply_threshold: float = 0.80
    structural_auto_apply_threshold: float = 0.95


def _base(url: str) -> str:
    return url.rstrip("/")


def _raise_for_response(response: requests.Response) -> None:
    if response.ok:
        return
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    raise ApiError(f"{response.status_code}: {detail}")


def _json(response: requests.Response) -> dict[str, Any]:
    _raise_for_response(response)
    return response.json()


class TableFixerClient:
    def __init__(self, config: ApiConfig):
        self.config = config

    @property
    def root(self) -> str:
        return _base(self.config.table_fixer_url)

    def health(self) -> dict[str, Any]:
        return _json(requests.get(f"{self.root}/health", timeout=10))

    def create_workspace(
        self,
        *,
        pdf_bytes: bytes,
        filename: str,
        page_numbers: list[int],
        extraction_mode: str,
        phases: list[str] | None = None,
        execution_mode: str = "auto_apply",
    ) -> dict[str, Any]:
        data: dict[str, Any] = {
            "page_numbers": json.dumps(page_numbers),
            "extraction_mode": extraction_mode,
            "execution_mode": execution_mode,
            "model": self.config.model,
            "base_url": self.config.base_url,
            "context_token_budget": str(self.config.context_token_budget),
            "auto_apply_threshold": str(self.config.auto_apply_threshold),
            "structural_auto_apply_threshold": str(self.config.structural_auto_apply_threshold),
        }
        if phases:
            data["phases"] = json.dumps(phases)
        files = {"file": (filename or "uploaded.pdf", pdf_bytes, "application/pdf")}
        response = requests.post(
            f"{self.root}/table-fixer/workspaces",
            data=data,
            files=files,
            timeout=None,
        )
        return _json(response)

    def get_workspace(self, workspace_id: str) -> dict[str, Any]:
        return _json(requests.get(f"{self.root}/table-fixer/workspaces/{workspace_id}", timeout=30))

    def get_snapshot(self, workspace_id: str, phase: str) -> dict[str, Any]:
        return _json(
            requests.get(
                f"{self.root}/table-fixer/workspaces/{workspace_id}/snapshots/{phase}",
                timeout=30,
            )
        )

    def page_image_url(self, workspace_id: str, page_number: int, zoom: float = 2.0) -> str:
        return f"{self.root}/table-fixer/workspaces/{workspace_id}/pages/{page_number}/image?zoom={zoom}"

    def execute(self, workspace_id: str, phases: list[str], execution_mode: str) -> dict[str, Any]:
        payload = {
            "phases": phases,
            "execution_mode": execution_mode,
            "model": self.config.model,
            "base_url": self.config.base_url,
            "context_token_budget": self.config.context_token_budget,
            "auto_apply_threshold": self.config.auto_apply_threshold,
            "structural_auto_apply_threshold": self.config.structural_auto_apply_threshold,
        }
        response = requests.post(
            f"{self.root}/table-fixer/workspaces/{workspace_id}/execute",
            json=payload,
            timeout=None,
        )
        return _json(response)

    def review(self, workspace_id: str, phase: str, decision: str) -> dict[str, Any]:
        response = requests.post(
            f"{self.root}/table-fixer/workspaces/{workspace_id}/reviews/{phase}",
            json={"decision": decision},
            timeout=60,
        )
        return _json(response)

    def manual_actions(
        self,
        workspace_id: str,
        *,
        base_phase: str,
        actions: list[dict[str, Any]],
        note: str | None = None,
    ) -> dict[str, Any]:
        response = requests.post(
            f"{self.root}/table-fixer/workspaces/{workspace_id}/manual-actions",
            json={"base_phase": base_phase, "actions": actions, "note": note},
            timeout=60,
        )
        return _json(response)

    def undo(self, workspace_id: str) -> dict[str, Any]:
        response = requests.post(
            f"{self.root}/table-fixer/workspaces/{workspace_id}/history/undo",
            timeout=60,
        )
        return _json(response)

    def redo(self, workspace_id: str) -> dict[str, Any]:
        response = requests.post(
            f"{self.root}/table-fixer/workspaces/{workspace_id}/history/redo",
            timeout=60,
        )
        return _json(response)


class MergeClient:
    def __init__(self, config: ApiConfig):
        self.config = config

    @property
    def root(self) -> str:
        return _base(self.config.merge_url)

    def merge_decisions(self, tables: list[dict[str, Any]], *, llm: bool = True) -> dict[str, Any]:
        response = requests.post(
            f"{self.root}/merge-decisions",
            json={
                "tables": tables,
                "llm": llm,
                "model": self.config.model,
                "base_url": self.config.base_url,
            },
            timeout=None,
        )
        return _json(response)
