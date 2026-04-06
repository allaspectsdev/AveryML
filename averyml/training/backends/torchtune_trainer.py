"""torchtune-based training backend (placeholder)."""

from __future__ import annotations

import logging
from pathlib import Path

from averyml.config.training import TrainingConfig
from averyml.training.backends.base import TrainingBackend
from averyml.utils.registry import training_backend_registry

logger = logging.getLogger(__name__)


@training_backend_registry.register("torchtune")
class TorchtuneTrainerBackend(TrainingBackend):
    """torchtune-based SFT backend for efficient long-context training.

    Generates a torchtune config and invokes `tune run`.
    Requires torchtune to be installed: pip install averyml[torchtune]
    """

    def train(self, config: TrainingConfig, dataset, tokenizer=None) -> Path:
        raise NotImplementedError(
            "torchtune backend is not yet implemented. "
            "Use 'hf_trainer' backend instead, or contribute an implementation."
        )
