from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class OllamaClientError(RuntimeError):
    """Raised when Ollama cannot complete a structured request."""


@dataclass
class OllamaLLMClient:
    model: str = "gemma3:4b"
    base_url: str = "http://localhost:11434"
    timeout: float = 120.0

    def structured(self, *, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/api/chat"
        request = urllib.request.Request(
            url,
            data=json.dumps(
                {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "format": schema,
                    "stream": False,
                    "options": {"temperature": 0, "num_predict": 256},
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            result = json.loads(payload.get("message", {}).get("content", ""))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise OllamaClientError(f"Failed to reach Ollama at {url}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise OllamaClientError("Ollama returned invalid structured JSON.") from exc
        if not isinstance(result, dict):
            raise OllamaClientError("Ollama structured response must be an object.")
        return result
