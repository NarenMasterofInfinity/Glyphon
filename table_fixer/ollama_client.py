from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .models import PromptUsage
from .token_counting import TokenCounter


class OllamaClientError(RuntimeError):
    """Raised when Ollama cannot complete a request."""


@dataclass
class StructuredResponse:
    prompt_id: str
    parsed: dict[str, Any] | None
    raw_response: str
    validation_errors: list[str]
    usage: PromptUsage
    context: dict[str, Any] = field(default_factory=dict)
    system_prompt: str = ""
    schema: dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return self.parsed is not None and not self.validation_errors

    def audit_record(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "phase": self.usage.phase,
            "purpose": self.usage.purpose,
            "system_prompt": self.system_prompt,
            "context": self.context,
            "schema": self.schema,
            "raw_response": self.raw_response,
            "parsed": self.parsed,
            "validation_errors": self.validation_errors,
            "usage": self.usage.__dict__,
        }


@dataclass
class OllamaLLMClient:
    model: str = "gemma3:4b"
    base_url: str = "http://localhost:11434"
    timeout: float = 120.0
    token_counter: TokenCounter = field(default_factory=TokenCounter)
    prompt_attempts: list[dict[str, Any]] = field(default_factory=list)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}{path}"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise OllamaClientError(f"Failed to reach Ollama at {url}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise OllamaClientError(f"Invalid JSON response from Ollama at {url}") from exc

    def structured(
        self,
        *,
        phase: str,
        purpose: str,
        system: str,
        context: dict[str, Any],
        schema: dict[str, Any],
        repair_parent_prompt_id: str | None = None,
    ) -> StructuredResponse:
        prompt_id = f"prompt_{uuid4().hex[:12]}"
        context_text = json.dumps(context, ensure_ascii=True, separators=(",", ":"))
        attempt = {
            "prompt_id": prompt_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "purpose": purpose,
            "model": self.model,
            "system_prompt": system,
            "context": context,
            "schema": schema,
            "repair_parent_prompt_id": repair_parent_prompt_id,
            "status": "pending",
            "raw_response": None,
            "parsed": None,
            "validation_errors": [],
            "error": None,
            "usage": None,
        }
        self.prompt_attempts.append(attempt)
        started = time.perf_counter()
        try:
            response = self._post(
                "/api/chat",
                {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": context_text},
                    ],
                    "format": schema,
                    "stream": False,
                    "options": {"temperature": 0, "num_predict": 768},
                },
            )
        except Exception as exc:
            attempt["status"] = "failed"
            attempt["error"] = str(exc)
            attempt["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)
            raise
        duration_ms = (time.perf_counter() - started) * 1000
        raw = response.get("message", {}).get("content", "")
        parsed = None
        errors: list[str] = []
        try:
            candidate = json.loads(raw)
            if not isinstance(candidate, dict):
                errors.append("Structured response must be a JSON object.")
            else:
                parsed = candidate
        except json.JSONDecodeError as exc:
            errors.append(f"Invalid structured JSON: {exc}")

        system_tokens = self.token_counter.count(system)
        context_tokens = self.token_counter.count(context_text)
        usage = PromptUsage(
            prompt_id=prompt_id,
            phase=phase,
            purpose=purpose,
            model=self.model,
            system_tokens=system_tokens,
            context_tokens=context_tokens,
            input_tokens=system_tokens + context_tokens,
            output_tokens=self.token_counter.count(raw),
            duration_ms=duration_ms,
            native_prompt_tokens=response.get("prompt_eval_count"),
            native_output_tokens=response.get("eval_count"),
            repair_parent_prompt_id=repair_parent_prompt_id,
        )
        attempt.update({
            "status": "invalid_response" if errors else "completed",
            "raw_response": raw or None,
            "parsed": parsed,
            "validation_errors": list(errors),
            "usage": usage.__dict__,
            "duration_ms": round(duration_ms, 2),
        })
        return StructuredResponse(
            prompt_id,
            parsed,
            raw,
            errors,
            usage,
            context=context,
            system_prompt=system,
            schema=schema,
        )
