"""SSD Trainer orchestrator: Step 2 of the SSD pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

from averyml.config.training import TrainingConfig
from averyml.training.data import SFTDataset
from averyml.utils.registry import training_backend_registry

logger = logging.getLogger(__name__)


class SSDTrainer:
    """Orchestrates fine-tuning (Step 2 of SSD).

    Pipeline:
    1. Load synthesis output as SFT dataset
    2. Tokenize with prompt masking (loss on completion only)
    3. Fine-tune with the configured backend (HF Trainer or torchtune)
    4. Save final checkpoint
    """

    def __init__(self, config: TrainingConfig):
        self.config = config

    def run(self) -> Path:
        """Execute the fine-tuning pipeline. Returns path to final checkpoint."""
        from transformers import AutoTokenizer

        logger.info(f"Preparing SFT dataset from {self.config.dataset_path}")
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dataset = SFTDataset(
            dataset_path=self.config.dataset_path,
            tokenizer=tokenizer,
            max_seq_length=self.config.max_seq_length,
            packing=self.config.packing,
        ).load()

        logger.info(f"Dataset ready: {len(dataset)} samples")

        backend_cls = training_backend_registry.get(self.config.backend)
        backend = backend_cls()

        checkpoint_path = backend.train(self.config, dataset, tokenizer=tokenizer)
        logger.info(f"Training complete. Checkpoint: {checkpoint_path}")
        return checkpoint_path
