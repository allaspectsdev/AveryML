"""Abstract base class for training backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from averyml.config.training import TrainingConfig


class TrainingBackend(ABC):
    """Interface for SFT training backends."""

    @abstractmethod
    def train(self, config: TrainingConfig, dataset, tokenizer=None) -> Path:
        """Run training and return the path to the final checkpoint."""
        ...
