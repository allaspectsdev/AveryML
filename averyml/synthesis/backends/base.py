"""Abstract base class for synthesis inference backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from averyml.config.synthesis import DecodingConfig


class SynthesisBackend(ABC):
    """Interface for model inference during data synthesis."""

    @abstractmethod
    def load_model(self, model_id: str, **kwargs: Any) -> None:
        """Load a model and tokenizer."""
        ...

    @abstractmethod
    def generate(
        self,
        prompts: list[str],
        decoding: DecodingConfig,
        max_tokens: int,
        seed: int,
    ) -> list[str]:
        """Generate completions for a batch of prompts."""
        ...

    @property
    @abstractmethod
    def tokenizer(self):
        """Return the loaded tokenizer."""
        ...

    def cleanup(self) -> None:
        """Release model resources."""
        pass
