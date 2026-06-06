from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TokenCounter:
    encoding_name: str = "cl100k_base"

    def count(self, text: str) -> int:
        try:
            import tiktoken

            return len(tiktoken.get_encoding(self.encoding_name).encode(text))
        except (ImportError, ValueError):
            # Keeps the app usable before optional dependencies are installed.
            return max(1, len(text) // 4) if text else 0
