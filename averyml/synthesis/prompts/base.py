"""Abstract base class for prompt sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class PromptSource(ABC):
    """Interface for loading coding problem prompts."""

    @abstractmethod
    def load(self, max_prompts: int | None = None) -> list[dict[str, Any]]:
        """Load prompts. Each item has: prompt_id, prompt_text, metadata."""
        ...

    @abstractmethod
    def format_for_model(self, prompt: dict[str, Any], tokenizer: Any) -> str:
        """Apply chat template and return the final prompt string."""
        ...
